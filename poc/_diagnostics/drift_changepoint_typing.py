"""Drift-vs-changepoint anomaly typing on the clean LatAD community pipeline.

Every window off the training distribution is an anomaly; this script TYPES the deviation:
  DRIFT-anomaly       = slow, on-trend elevation of a community's surprise (SWaT: RO fouling + analyser
                        electrode drift, a monotone ramp over days)
  CHANGEPOINT-anomaly = abrupt, off-trend excursion (attacks / faults)
Per community and seed, the per-window surprise series s_g(t) (experts_full/expert_<DS>.npz test_surprise)
is decomposed CAUSALLY into
  slow_g(t) = median of s_g over the previous K windows (K = 24 h of wall-clock, set A PRIORI: fouling is
              a days-scale ramp; sensitivity 6/12/48 h and a 6 h-lagged variant are REPORTED, not selected)
  fast_g(t) = s_g(t) - slow_g(t)                       (changepoint component)
The calibration slice (held-out 20% train-normal) is decomposed the same way and is the p-value reference
for the residuals, exactly as it is the reference for the raw surprises in the headline. The changepoint
detector CP = z(HC_coh over residual p-values) + z(LatAD global tail, residual-detrended the same way),
i.e. the headline fusion (ensemble_final.ensemble_scores 'HCcoh+LatAD') applied to the fast component.
The drift detector DR = the same HC_coh fusion applied to the slow baselines against the calib reference.
Typing (seed-mean scores, thresholds = 99th pct of the calib-vs-calib score, no test labels):
  flagged by headline & CP > thr_cp -> changepoint-type ; flagged & CP <= thr_cp -> drift-type.

Invariants (stated before any number was seen):
  I1  K=None (no decomposition) reproduces the headline 'HCcoh+LatAD' score EXACTLY.
  I2  HAI / WADI_clean (small drift): CP Difficult AUROC within +-0.02 of the headline.
  I3  No reported method below 0.5 on the All subset.
  I4  A synthetic slow ramp added to every surprise series (and to the global LatAD score) changes the CP
      AUROC by < 0.02 while degrading the headline (the decomposition removes exactly what it claims).
  I5  Masking: no attack episode's median percentile (vs test normals) drops by > 0.10 headline -> CP.
Wall-clock per window step: SWaT_canon / WADI_clean 300 s (10 s rows, stride 30), HAI 30 s (1 Hz, stride 30).
Outputs: _diagnostics/drift_changepoint_typing.json (incremental per dataset) + .log
"""
from __future__ import annotations
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ["EXPERTS_DIR"] = os.path.join(ROOT, "sota_bundle", "experts_full")
from sklearn.metrics import roc_auc_score
import ensemble_final as EF

HERE = os.path.dirname(os.path.abspath(__file__))
OUTJ = os.path.join(HERE, "drift_changepoint_typing.json")
LOG = open(os.path.join(HERE, "drift_changepoint_typing.log"), "a")
STEP_S = {"SWaT_canon": 300, "WADI_clean": 300, "HAI": 30}
HOURS = [6, 12, 24, 48]; APRIORI = 24; GAP_H = 6
REPS = int(os.environ.get("BOOT_REPS", "2000"))
DATASETS = sys.argv[1:] or ["SWaT_canon", "HAI", "WADI_clean"]
HEADKEY = "HCcoh+LatAD"


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def causal_median(x, K, init, gap=0):
    """median of x[t-gap-K : t-gap] (strictly past), min periods max(3, K//4); init (train level) before."""
    m = pd.Series(np.asarray(x, float)).rolling(K, min_periods=max(3, K // 4)).median().shift(1 + gap).values
    return np.where(np.isnan(m), init, m)


def decompose(X, K, init, gap, scale=False, cmad=None):
    """X (S,n) -> slow (S,n), fast (S,n). K=None -> slow=0 (identity, headline).
    scale=True (v2): the on-trend component also carries a slow SCALE, the causal rolling MAD over the same K
    windows (median of |x - slow|), floored at the calibration MAD so a quiet community is never more sensitive
    than at train level: fast = (x - slow) / max(rolling MAD, calib MAD). Added after the v1 masking check
    showed the drift swamping the fusion through the residual NOISE level (see .md, root cause)."""
    if K is None:
        return np.zeros_like(X), X.copy()
    slow = np.stack([causal_median(X[g], K, init[g], gap) for g in range(X.shape[0])])
    fast = X - slow
    if scale:
        rmad = np.stack([causal_median(np.abs(fast[g]), K, cmad[g] / 1.4826, gap) for g in range(X.shape[0])]) * 1.4826
        fast = fast / np.maximum(rmad, cmad[:, None])
    return slow, fast


def calib_mad(C):
    return 1.4826 * np.median(np.abs(C - np.median(C, 1, keepdims=True)), 1) + 1e-6


z = lambda s, r: (s - r.mean()) / (r.std() + 1e-9)


def hc_fuse(RC, RT, w):
    """cohesion-weighted HC over per-community p-values (reference RC), z-scored vs calib-vs-calib."""
    S = RC.shape[0]
    P = np.stack([EF.pval(RC[g], RT[g]) for g in range(S)]); Pc = np.stack([EF.pval(RC[g], RC[g]) for g in range(S)])
    hc, hcc = EF.HC(P, wt=w), EF.HC(Pc, wt=w)
    return z(hc, hcc), z(hcc, hcc)


def scores(Ex, d, K, gap=0, detrend_global=True, ramp=None, scale=False):
    """Return dict of (nseed, n) test scores and (nseed, ncal) calib-vs-calib scores: head, cp, dr."""
    Cal, Tst = Ex["calib_surprise"], Ex["test_surprise"]; nseed, S, n = Tst.shape
    w = Ex["comm_cohesion"].astype(float) * np.sqrt(Ex["comm_size"].astype(float))
    lat, lat_tr = d["LatAD"], d["LatAD_train"]; nseed = min(nseed, lat.shape[0])
    out = {k: [] for k in ("cp", "cp_cal", "dr", "dr_cal", "head", "head_cal", "slowT")}
    for sd in range(nseed):
        C, T, L, Ltr = Cal[sd].astype(float), Tst[sd].astype(float), lat[sd].astype(float), lat_tr[sd].astype(float)
        if ramp is not None:                                   # I4: synthetic slow ramp (test only)
            T = T + ramp * np.linspace(0, 1, n)[None, :] * C.std(1, keepdims=True)
            L = L + ramp * np.linspace(0, 1, n) * Ltr.std()
        init = np.median(C, 1); cmad = calib_mad(C)
        slowC, RC = decompose(C, K, init, gap, scale, cmad); slowT, RT = decompose(T, K, init, gap, scale, cmad)
        # headline (raw surprises, raw LatAD) for reference. The calib slice is the contiguous LAST 20% of the
        # train windows (experts: Xn[nfit:]), and LatAD_train covers all train windows, so [-ncal:] aligns.
        ncal = C.shape[1]; assert len(Ltr) >= ncal
        zh, zh_c = hc_fuse(C, T, w)
        zl_head = z(EF.surv(Ltr, L), EF.surv(Ltr, Ltr)); zl_head_c = z(EF.surv(Ltr, Ltr), EF.surv(Ltr, Ltr))[-ncal:]
        out["head"].append(zh + zl_head); out["head_cal"].append(zh_c + zl_head_c)
        # changepoint fusion on the fast residuals
        zc, zc_c = hc_fuse(RC, RT, w)
        if detrend_global and K is not None:
            li = np.median(Ltr[None, :], 1); lm = calib_mad(Ltr[-ncal:][None, :])
            Ltr_r = decompose(Ltr[None, :], K, li, gap, scale, lm)[1][0]; L_r = decompose(L[None, :], K, li, gap, scale, lm)[1][0]
            zl = z(EF.surv(Ltr_r, L_r), EF.surv(Ltr_r, Ltr_r)); zl_c = z(EF.surv(Ltr_r, Ltr_r), EF.surv(Ltr_r, Ltr_r))[-ncal:]
        else:
            zl, zl_c = zl_head, zl_head_c
        out["cp"].append(zc + zl); out["cp_cal"].append(zc_c + zl_c)
        # drift fusion on the slow baselines (against the raw calib surprise reference)
        if K is None:
            out["dr"].append(np.zeros(n)); out["dr_cal"].append(np.zeros(C.shape[1]))
        else:
            zd, _ = hc_fuse(C, slowT, w); zdc, _ = hc_fuse(C, slowC, w)
            out["dr"].append(zd); out["dr_cal"].append(zdc)
        out["slowT"].append(slowT)
    return {k: np.stack(v) for k, v in out.items()}


def au_ms(y, arr, m):
    k = (y == 0) | m
    if arr.ndim == 2:
        return float(np.mean([roc_auc_score(y[k], arr[i][k]) for i in range(arr.shape[0])]))
    return float(roc_auc_score(y[k], arr[k]))


def pct_vs_normals(s, nrm):
    return np.searchsorted(np.sort(s[nrm]), s, side="left") / nrm.sum()


ALL = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
for name in DATASETS:
    t0 = time.time(); log(f"\n================ {name} ================")
    d = np.load(f"{EF.OUT}/scores_{name}.npz"); y = d["label"].astype(int); n = len(y); nrm = y == 0; an = y == 1
    Ex = np.load(f"{os.environ['EXPERTS_DIR']}/expert_{name}.npz", allow_pickle=True)
    assert np.array_equal(Ex["y"].astype(int), y), "expert/scores label mismatch"
    if name == "SWaT_canon":
        assert "fit_surprise" in Ex.files and Ex["test_surprise"].shape[1] == 24, "SWaT_canon must be the official S=24 bundle with fit_surprise"
    pf = np.load(f"{HERE}/pca_filter_doublehard_{name}.npz")
    assert np.array_equal(pf["y"], y) and np.allclose(pf["maxz"], d["maxz"]) and np.isclose(float(pf["maxz_thr"]), float(d["maxz_thr"]))
    mthr = float(d["maxz_thr"]); diff = an & (d["maxz"] <= mthr); dh_lin = pf["dhard_lin"].astype(bool); dh_pca = pf["dhard_pca"].astype(bool)
    masks = {"All": an, "Difficult": diff, "DoubleHard_lin": dh_lin, "DoubleHard_pca": dh_pca}
    fn, W, stride = EF.E.RAW[name]; Lblk = int(np.ceil(W / stride)) + 1
    step = STEP_S[name]; K24 = int(APRIORI * 3600 / step); gapK = int(GAP_H * 3600 / step)
    eps = EF.episodes(y)
    log(f"windows {n}, anomalies {an.sum()}, episodes {len(eps)}, subsets: " + ", ".join(f"{k}={int(m.sum())}" for k, m in masks.items())
        + f"; step {step}s, K(24h)={K24}, longest episode {max(len(e) for e in eps)} win = {max(len(e) for e in eps) * step / 3600:.1f} h")

    # ---- I1: identity reproduces the headline exactly
    ens, y2, _, nseed = EF.ensemble_scores(name); head_ref = ens[HEADKEY]
    S0 = scores(Ex, d, None)
    assert np.allclose(S0["cp"], head_ref, atol=1e-6) and np.allclose(S0["head"], head_ref, atol=1e-6), "I1 FAILED: K=None != headline"
    log(f"I1 OK: K=None reproduces headline {HEADKEY} exactly (max abs diff {np.abs(S0['cp'] - head_ref).max():.2e})")

    # ---- baselines on the same subsets
    base = {"trivial max|z|": d["maxz"], "IF": d["IF"], "AE": d["AE"], "linres": d["linres"], "USAD": d["USAD"], "TranAD": d["TranAD"],
            "LatAD (global)": d["LatAD"], "HC_coh": ens["HC_coh"], "null+HC": ens["null+HC"], "HCcoh+LatAD (headline)": head_ref}
    if "GDN" in d.files:
        base["GDN"] = d["GDN"]
    tab = {k: {m: round(au_ms(y, v, mm), 3) for m, mm in masks.items()} for k, v in base.items()}
    log("\nbaselines (5-seed mean AUROC): " + " | ".join(masks))
    for k, r in tab.items():
        log(f"  {k:26s} " + " ".join(f"{r[m]:.3f}" for m in masks))

    # ---- sensitivity over the timescale (reported, not selected) + variants
    sens = {}; SP1 = SP = None
    for v, sc in (("v1_level", False), ("v2_level+scale", True)):
        for h in HOURS:
            K = int(h * 3600 / step); Sh = scores(Ex, d, K, scale=sc)
            sens[f"{v}_{h}h"] = {m: round(au_ms(y, Sh["cp"], mm), 3) for m, mm in masks.items()}
            if h == APRIORI:
                if sc: SP = Sh
                else: SP1 = Sh
        Sg = scores(Ex, d, K24, gap=gapK, scale=sc); sens[f"{v}_{APRIORI}h_lag{GAP_H}h"] = {m: round(au_ms(y, Sg["cp"], mm), 3) for m, mm in masks.items()}
        Sr = scores(Ex, d, K24, detrend_global=False, scale=sc); sens[f"{v}_{APRIORI}h_rawGlobal"] = {m: round(au_ms(y, Sr["cp"], mm), 3) for m, mm in masks.items()}
    log("\nCP fusion AUROC by variant and slow-baseline timescale (a priori = 24h; PRIMARY = v2_level+scale_24h):")
    for k, r in sens.items():
        log(f"  {k:26s} " + " ".join(f"{r[m]:.3f}" for m in masks))
    cp, head, dr = SP["cp"], SP["head"], SP["dr"]
    res_cp = sens[f"v2_level+scale_{APRIORI}h"]; res_head = tab["HCcoh+LatAD (headline)"]
    tab["CP v1 (24h level only)"] = sens[f"v1_level_{APRIORI}h"]
    tab["CP (24h level+scale, PRIMARY)"] = res_cp; tab["DR (24h, drift score)"] = {m: round(au_ms(y, dr, mm), 3) for m, mm in masks.items()}
    log(f"  {'DR drift score':26s} " + " ".join(f"{tab['DR (24h, drift score)'][m]:.3f}" for m in masks))

    # ---- I3
    below = [k for k, r in tab.items() if r["All"] < 0.5]
    log(f"I3 {'OK' if not below else 'VIOLATED'}: methods below chance on All: {below}")

    # ---- I4: synthetic ramp invariance
    # I4 ramps are specified in calib-SD PER DAY (a 24 h baseline can only track drift slower than its own window);
    # the criterion applies to the 1 and 2 sd/day ramps. A fixed per-record 5 sd ramp is reported as well for
    # reference (on a 2-day record that is 2.5 sd/day, outside the assumed regime).
    days = n * step / 86400.0; i4 = {}; ok4 = True
    for lab, amp, judged in ((f"1sd/day", 1.0 * days, True), (f"2sd/day", 2.0 * days, True), ("5sd/record", 5.0, False)):
        Sramp = scores(Ex, d, K24, ramp=amp, scale=True)
        r4 = dict(amp_record_sd=round(amp, 2), head_Difficult=round(au_ms(y, Sramp["head"], diff), 3), head_All=round(au_ms(y, Sramp["head"], an), 3),
                  cp_Difficult=round(au_ms(y, Sramp["cp"], diff), 3), cp_All=round(au_ms(y, Sramp["cp"], an), 3))
        r4["ok"] = bool(abs(r4["cp_Difficult"] - res_cp["Difficult"]) < 0.02 and abs(r4["cp_All"] - res_cp["All"]) < 0.02)
        if judged: ok4 &= r4["ok"]
        i4[lab] = r4
        log(f"I4 {'OK' if r4['ok'] else 'VIOLATED'}{'' if judged else ' (reference only)'}: +{lab} ramp ({amp:.1f} sd over {days:.1f} d) -> headline Diff {res_head['Difficult']:.3f}->{r4['head_Difficult']:.3f}, All {res_head['All']:.3f}->{r4['head_All']:.3f};"
            f" CP Diff {res_cp['Difficult']:.3f}->{r4['cp_Difficult']:.3f}, All {res_cp['All']:.3f}->{r4['cp_All']:.3f}")

    # ---- I5: masking per attack episode (seed-mean percentiles vs test normals)
    # Two percentiles per episode: GLOBAL (vs all test normals; what AUROC sees) and ERA-LOCAL (vs the test normals
    # within +-24 h; whether the attack still stands out from its contemporaneous normals). Under drift the
    # headline's global percentile of a late attack is a time-position artefact (every late window outranks every
    # early one), so the PRIMARY masking criterion is the era-local one; the global one is reported alongside.
    sm = {"head": head.mean(0), "v1": SP1["cp"].mean(0), "v2": cp.mean(0)}
    pg = {k: pct_vs_normals(v, nrm) for k, v in sm.items()}
    def local_pct(s, e):
        idx = np.arange(max(0, e[0] - K24), min(n, e[-1] + K24 + 1)); ref = np.sort(s[idx[nrm[idx]]])
        return float(np.median(np.searchsorted(ref, s[e], side="left") / len(ref)))
    ep_rows = []
    for e in eps:
        g = {k: float(np.median(pg[k][e])) for k in sm}; l = {k: local_pct(sm[k], e) for k in sm}
        ep_rows.append(dict(start=int(e[0]), n=int(len(e)), hours=round(len(e) * step / 3600, 2), difficult=int(diff[e].sum()),
                            global_head=round(g["head"], 3), global_v1=round(g["v1"], 3), global_v2=round(g["v2"], 3),
                            local_head=round(l["head"], 3), local_v1=round(l["v1"], 3), local_v2=round(l["v2"], 3),
                            masked_local=bool(l["v2"] < l["head"] - 0.10), masked_local_v1=bool(l["v1"] < l["head"] - 0.10),
                            masked_global=bool(g["v2"] < g["head"] - 0.10), masked_global_v1=bool(g["v1"] < g["head"] - 0.10)))
    masked = [r for r in ep_rows if r["masked_local"]]; masked1 = [r for r in ep_rows if r["masked_local_v1"]]
    mg = [r for r in ep_rows if r["masked_global"]]; mg1 = [r for r in ep_rows if r["masked_global_v1"]]
    d95 = lambda k, m: round(float((pg[k][m] >= 0.95).mean()), 3)
    det = dict(det95_all={k: d95(k, an) for k in sm}, det95_difficult={k: d95(k, diff) for k in sm},
               n_masked_local_v1=len(masked1), n_masked_local_v2=len(masked), n_masked_global_v1=len(mg1), n_masked_global_v2=len(mg),
               n_gain_local_gt_0_1_v2=int(sum(r["local_v2"] > r["local_head"] + 0.1 for r in ep_rows)), n_gain_global_gt_0_1_v2=int(sum(r["global_v2"] > r["global_head"] + 0.1 for r in ep_rows)))
    log(f"I5 (era-local, PRIMARY) v1 {'OK' if not masked1 else 'VIOLATED'} ({len(masked1)}/{len(eps)}); v2 {'OK' if not masked else 'VIOLATED'} ({len(masked)}/{len(eps)} masked, {det['n_gain_local_gt_0_1_v2']} gained >0.1)."
        f"  Global-rank: v1 {len(mg1)} masked, v2 {len(mg)} masked, {det['n_gain_global_gt_0_1_v2']} gained >0.1.  det@95pct(global) all {det['det95_all']}, difficult {det['det95_difficult']}")
    if name == "SWaT_canon" or masked or masked1 or mg:
        for r in (ep_rows if name == "SWaT_canon" else [r for r in ep_rows if r["masked_local"] or r["masked_local_v1"] or r["masked_global"]]):
            log(f"    ep@{r['start']:5d} n={r['n']:3d} ({r['hours']:5.2f} h, diff {r['difficult']:3d})  global head {r['global_head']:.3f} v1 {r['global_v1']:.3f} v2 {r['global_v2']:.3f} | local head {r['local_head']:.3f} v1 {r['local_v1']:.3f} v2 {r['local_v2']:.3f}"
                f"{'  MASKED(local)' if r['masked_local'] else ''}{'  masked(global)' if r['masked_global'] else ''}")

    # ---- bootstrap: CP vs headline, and CP vs strongest alternative, per subset
    boots = {}
    for m, mm in masks.items():
        EF.RNG = np.random.default_rng(0); boots[f"cp_minus_headline[{m}]"] = EF.boot(y, cp, head, mm, Lblk, reps=REPS)
        alt = max((k for k in tab if not k.startswith("CP") and not k.startswith("DR")), key=lambda k: tab[k][m])
        EF.RNG = np.random.default_rng(0); b = EF.boot(y, cp, base[alt], mm, Lblk, reps=REPS); b["alt"] = alt; b["alt_auroc"] = tab[alt][m]
        boots[f"cp_minus_strongest[{m}]"] = b
        log(f"  boot [{m:14s}] CP-headline: {boots[f'cp_minus_headline[{m}]']}")
        log(f"  boot [{m:14s}] CP-strongest ({alt} {tab[alt][m]:.3f}): diff {b['diff']:+.3f} CI {b['diff_ci']} P(<=0)={b['p_le_0']}")

    # ---- typing (seed-mean scores; thresholds from calib-vs-calib, no test labels)
    hm, cm, dm = head.mean(0), cp.mean(0), dr.mean(0)
    thr_h, thr_c, thr_d = (float(np.quantile(SP[k].mean(0), 0.99)) for k in ("head_cal", "cp_cal", "dr_cal"))
    slowT = SP["slowT"].mean(0); Cal = Ex["calib_surprise"].astype(float).mean(0)
    q99g = np.quantile(Cal, 0.99, axis=1)                       # per-community train-level 99th pct of the raw surprise
    fh, fc = hm > thr_h, cm > thr_c
    fd = (slowT > q99g[:, None]).any(0)                          # DRIFTED window: the slow on-trend baseline itself is off the train distribution in >=1 community
    fd_score = dm > thr_d
    third = np.minimum((np.arange(n) * 3) // n, 2)
    typing = dict(thr_head=round(thr_h, 3), thr_cp=round(thr_c, 3), thr_drift=round(thr_d, 3),
                  attacks=dict(n=int(an.sum()), flagged_head=round(float(fh[an].mean()), 3), flagged_cp=round(float(fc[an].mean()), 3),
                               typed_changepoint_of_head_flagged=round(float(fc[an & fh].mean()), 3) if (an & fh).any() else None,
                               difficult_flagged_head=round(float(fh[diff].mean()), 3), difficult_flagged_cp=round(float(fc[diff].mean()), 3),
                               difficult_typed_changepoint_of_head_flagged=round(float(fc[diff & fh].mean()), 3) if (diff & fh).any() else None),
                  normals=dict(n=int(nrm.sum()), false_alarm_head=round(float(fh[nrm].mean()), 3), false_alarm_cp=round(float(fc[nrm].mean()), 3),
                               drifted=round(float(fd[nrm].mean()), 3), drifted_by_DR_score=round(float(fd_score[nrm].mean()), 3), drifted_and_head_flagged=int((nrm & fd & fh).sum()),
                               drifted_head_flagged_typed_drift=round(float((~fc)[nrm & fd & fh].mean()), 3) if (nrm & fd & fh).any() else None,
                               nondrifted_head_flagged=int((nrm & ~fd & fh).sum()),
                               nondrifted_head_flagged_typed_drift=round(float((~fc)[nrm & ~fd & fh].mean()), 3) if (nrm & ~fd & fh).any() else None),
                  by_third=[dict(third=t + 1, n_normal=int((nrm & (third == t)).sum()), n_attack=int((an & (third == t)).sum()),
                                 head_FA=round(float(fh[nrm & (third == t)].mean()), 3), cp_FA=round(float(fc[nrm & (third == t)].mean()), 3),
                                 drifted=round(float(fd[nrm & (third == t)].mean()), 3),
                                 attack_flag_head=round(float(fh[an & (third == t)].mean()), 3) if (an & (third == t)).any() else None,
                                 attack_flag_cp=round(float(fc[an & (third == t)].mean()), 3) if (an & (third == t)).any() else None) for t in range(3)])
    log(f"\nTYPING (thr head {thr_h:.2f}, cp {thr_c:.2f}, drift {thr_d:.2f}):")
    log(f"  attacks: flagged head {typing['attacks']['flagged_head']:.3f}, cp {typing['attacks']['flagged_cp']:.3f}; of head-flagged attacks typed CHANGEPOINT {typing['attacks']['typed_changepoint_of_head_flagged']}"
        f"; difficult: head {typing['attacks']['difficult_flagged_head']:.3f}, cp {typing['attacks']['difficult_flagged_cp']:.3f}, typed CP {typing['attacks']['difficult_typed_changepoint_of_head_flagged']}")
    log(f"  normals: FA head {typing['normals']['false_alarm_head']:.3f}, cp {typing['normals']['false_alarm_cp']:.3f}; drifted (DR>thr) {typing['normals']['drifted']:.3f};"
        f" drifted&head-flagged n={typing['normals']['drifted_and_head_flagged']} typed DRIFT {typing['normals']['drifted_head_flagged_typed_drift']};"
        f" non-drifted&head-flagged n={typing['normals']['nondrifted_head_flagged']} typed DRIFT {typing['normals']['nondrifted_head_flagged_typed_drift']}")
    for r in typing["by_third"]:
        log(f"  third {r['third']}: normals {r['n_normal']} head_FA {r['head_FA']:.3f} cp_FA {r['cp_FA']:.3f} drifted {r['drifted']:.3f} | attacks {r['n_attack']} flag head {r['attack_flag_head']} cp {r['attack_flag_cp']}")

    # ---- per-community drift signature: slow baseline at end vs start of the test-normal stream (seed-mean, raw surprise units and calib-SD units)
    slowT = SP["slowT"].mean(0); Cal = Ex["calib_surprise"].astype(float).mean(0)
    idxn = np.where(nrm)[0]; first, last = idxn[: max(5, len(idxn) // 10)], idxn[-max(5, len(idxn) // 10):]
    comm = []
    for g in range(slowT.shape[0]):
        ch = [int(c) for c in Ex["comm_channels"][g] if c >= 0]
        comm.append(dict(g=g, size=len(ch), cohesion=round(float(Ex["comm_cohesion"][g]), 3), channels=ch,
                         calib_med=round(float(np.median(Cal[g])), 3), slow_start=round(float(np.median(slowT[g][first])), 3),
                         slow_end=round(float(np.median(slowT[g][last])), 3),
                         end_minus_calib_in_calibSD=round(float((np.median(slowT[g][last]) - np.median(Cal[g])) / (Cal[g].std() + 1e-9)), 2)))
    try:
        sens_names = list(fn()[3])
        for c in comm:
            c["names"] = [str(sens_names[i]) for i in c["channels"]]
    except Exception as ex:
        log(f"  [warn] channel names unavailable for {name}: {ex}")
    # drifting community = its slow baseline over the test-NORMAL stream exceeds the calib 99th pct of the raw surprise on >= 50% of normal windows
    for g, c in enumerate(comm):
        q99 = float(np.quantile(Cal[c["g"]], 0.99))
        c["frac_normal_windows_slow_above_calib_q99"] = round(float((slowT[c["g"]][idxn] > q99).mean()), 3)
        c["drifting"] = bool(c["frac_normal_windows_slow_above_calib_q99"] >= 0.5)
    comm.sort(key=lambda c: -c["end_minus_calib_in_calibSD"])
    n_drift_comm = sum(c["drifting"] for c in comm)
    log(f"\nper-community slow baseline (seed-mean): DRIFTING communities (slow baseline > calib q99 on >=50% of test-normal windows): {n_drift_comm}/{len(comm)}")
    for c in comm[:12]:
        log(f"  g{c['g']:2d} size {c['size']:2d} coh {c['cohesion']:.2f}  calib {c['calib_med']:6.2f}  start {c['slow_start']:6.2f}  end {c['slow_end']:6.2f}  ({c['end_minus_calib_in_calibSD']:+.1f} sd, above-q99 {c['frac_normal_windows_slow_above_calib_q99']:.2f}){' DRIFT' if c['drifting'] else ''}  {c.get('names', c['channels'])}")
    # composition of the headline-flagged windows (drift-type vs changepoint-type), overall / normals / attacks
    # three-way typing of every headline-flagged window: CHANGEPOINT (CP score above its calib-99th threshold),
    # DRIFT (not CP-flagged AND the window is drifted: its slow baseline is off-train in >=1 community),
    # SUBTHRESHOLD (neither: the headline flag is not reproduced by CP and no drift is present -> threshold noise)
    comp = {}
    for lab, mm in (("all_flagged", fh), ("flagged_normals", fh & nrm), ("flagged_attacks", fh & an), ("flagged_difficult_attacks", fh & diff)):
        comp[lab] = dict(n=int(mm.sum()), changepoint_type=round(float(fc[mm].mean()), 3) if mm.any() else None,
                         drift_type=round(float((~fc & fd)[mm].mean()), 3) if mm.any() else None,
                         subthreshold=round(float((~fc & ~fd)[mm].mean()), 3) if mm.any() else None)
    comp["test_normal_drifted_frac"] = typing["normals"]["drifted"]; comp["n_drifting_communities"] = n_drift_comm; comp["n_communities"] = len(comm)
    comp["drifting_communities"] = [dict(g=c["g"], names=c.get("names", c["channels"])) for c in comm if c["drifting"]]
    for lab in ("all_flagged", "flagged_normals", "flagged_attacks", "flagged_difficult_attacks"):
        c = comp[lab]; log(f"COMPOSITION {lab:26s} n={c['n']:5d}: changepoint-type {c['changepoint_type']}, drift-type {c['drift_type']}, subthreshold {c['subthreshold']}")
    log(f"COMPOSITION test-normal drifted {comp['test_normal_drifted_frac']}; drifting communities {n_drift_comm}/{len(comm)}")

    ALL[name] = dict(n=n, n_anom=int(an.sum()), n_episodes=len(eps), subsets={k: int(m.sum()) for k, m in masks.items()}, step_s=step, K24=K24,
                     table=tab, sensitivity=sens, invariants=dict(I1_identity=True, I3_below_chance_All=below, I4_ramp=i4, I4_ok=ok4, I5_masking=det, I5_ok=not masked),
                     episodes=ep_rows, bootstrap=boots, typing=typing, composition=comp, communities=comm, seconds=round(time.time() - t0, 1))
    json.dump(ALL, open(OUTJ, "w"), indent=1)
    log(f"[{name} done in {time.time() - t0:.0f}s] saved -> {OUTJ}")
log("\nALL DONE")
