"""Drift-GATED changepoint head inside the reported detector (HCcoh+LatAD), gate fixed from TRAIN-NORMAL only.

The changepoint (fast-residual) score of drift_changepoint_typing.py recovers SWaT as a SEPARATE score. Here it is
fused into the headline as a per-TERM gated head: each community g (and the global LatAD term) uses the causal
level+scale residual (v2, K = 24 h a priori, identical code) IF AND ONLY IF a train-normal stationarity test flags
that term as drifting; otherwise the term is bit-identical to the headline. No test data enter the gate.

Gate statistic (per term, seed-mean): FA_g = fraction of the SECOND half of the held-out calibration slice (last 20 %
of train windows, contiguous, time-ordered) that exceeds the 99th percentile of the FIRST half. Stationary -> ~0.01.
A-priori rule: ON iff FA_g > 0.05 (five times nominal). Sensitivity over the budget {0.02, 0.03, 0.10, 0.20}, plus the
two limits all-off (== headline) and all-on (== CP v2 of the typing study), is REPORTED, not selected.

Invariants (stated before any AUROC was read):
  I1  all-off reproduces the headline 'HCcoh+LatAD' exactly (asserted); all-on reproduces CP v2 (typing JSON, 1e-3).
  I2  for every OFF term the p-value inputs are bit-identical to the headline's (asserted per seed).
  I3  HAI / WADI_clean Difficult within +-0.02 of the headline at the primary budget.
  I5  masking: no SWaT attack episode loses > 0.10 era-local percentile headline -> gated.
Outputs: _diagnostics/drift_gated_head.json (incremental per dataset) + .log
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
OUTJ = os.path.join(HERE, "drift_gated_head.json"); TYPJ = os.path.join(HERE, "drift_changepoint_typing.json")
LOG = open(os.path.join(HERE, "drift_gated_head.log"), "a")
STEP_S = {"SWaT_canon": 300, "WADI_clean": 300, "HAI": 30}; APRIORI_H = 24
BUDGET = 0.05; BUDGETS = [0.02, 0.03, 0.05, 0.10, 0.20]
REPS = int(os.environ.get("BOOT_REPS", "2000"))
DATASETS = sys.argv[1:] or ["SWaT_canon", "WADI_clean", "HAI"]
HEADKEY = "HCcoh+LatAD"


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


# ---- decomposition, copied verbatim from drift_changepoint_typing.py (v2 level+scale) ----
def causal_median(x, K, init, gap=0):
    m = pd.Series(np.asarray(x, float)).rolling(K, min_periods=max(3, K // 4)).median().shift(1 + gap).values
    return np.where(np.isnan(m), init, m)


def decompose(X, K, init, cmad):
    slow = np.stack([causal_median(X[g], K, init[g]) for g in range(X.shape[0])])
    fast = X - slow
    rmad = np.stack([causal_median(np.abs(fast[g]), K, cmad[g] / 1.4826) for g in range(X.shape[0])]) * 1.4826
    return slow, fast / np.maximum(rmad, cmad[:, None])


def calib_mad(C):
    return 1.4826 * np.median(np.abs(C - np.median(C, 1, keepdims=True)), 1) + 1e-6


z = lambda s, r: (s - r.mean()) / (r.std() + 1e-9)


def gate_stats(Ex, d):
    """Train-normal only. Returns (fa_comm (S,), fa_glob float): seed-mean FA of calib 2nd half at 1st-half q99."""
    C = Ex["calib_surprise"].astype(float); nseed, S, ncal = C.shape; h = ncal // 2
    fa = np.array([[(C[sd, g, h:] > np.quantile(C[sd, g, :h], 0.99)).mean() for g in range(S)] for sd in range(nseed)]).mean(0)
    Ltr = d["LatAD_train"].astype(float)[:nseed, -ncal:]
    fg = float(np.mean([(Ltr[sd, h:] > np.quantile(Ltr[sd, :h], 0.99)).mean() for sd in range(nseed)]))
    return fa, fg


def gated_scores(Ex, d, K, on_comm, on_glob, head_P=None):
    """(nseed,n) fused score and (nseed,ncal) calib-vs-calib reference. on_comm (S,) bool, on_glob bool.
    head_P: optional list per seed of (P, Pc) headline p-values, used to assert I2 for OFF terms."""
    Cal, Tst = Ex["calib_surprise"], Ex["test_surprise"]; nseed, S, n = Tst.shape
    w = Ex["comm_cohesion"].astype(float) * np.sqrt(Ex["comm_size"].astype(float))
    lat, lat_tr = d["LatAD"], d["LatAD_train"]; nseed = min(nseed, lat.shape[0])
    sc, cal, Pkeep = [], [], []
    for sd in range(nseed):
        C, T, L, Ltr = Cal[sd].astype(float), Tst[sd].astype(float), lat[sd].astype(float), lat_tr[sd].astype(float)
        ncal = C.shape[1]; init = np.median(C, 1); cmad = calib_mad(C)
        if on_comm.any() or on_glob:
            _, RC = decompose(C, K, init, cmad); _, RT = decompose(T, K, init, cmad)
        P = np.stack([EF.pval(RC[g], RT[g]) if on_comm[g] else EF.pval(C[g], T[g]) for g in range(S)])
        Pc = np.stack([EF.pval(RC[g], RC[g]) if on_comm[g] else EF.pval(C[g], C[g]) for g in range(S)])
        if head_P is not None:                                    # I2: off terms bit-identical to headline inputs
            hP, hPc = head_P[sd]; off = ~on_comm
            assert np.array_equal(P[off], hP[off]) and np.array_equal(Pc[off], hPc[off]), "I2 FAILED"
        hc, hcc = EF.HC(P, wt=w), EF.HC(Pc, wt=w); zh, zh_c = z(hc, hcc), z(hcc, hcc)
        if on_glob:
            li = np.median(Ltr[None, :], 1); lm = calib_mad(Ltr[-ncal:][None, :])
            Ltr_r = decompose(Ltr[None, :], K, li, lm)[1][0]; L_r = decompose(L[None, :], K, li, lm)[1][0]
            zl = z(EF.surv(Ltr_r, L_r), EF.surv(Ltr_r, Ltr_r)); zl_c = z(EF.surv(Ltr_r, Ltr_r), EF.surv(Ltr_r, Ltr_r))[-ncal:]
        else:
            zl = z(EF.surv(Ltr, L), EF.surv(Ltr, Ltr)); zl_c = z(EF.surv(Ltr, Ltr), EF.surv(Ltr, Ltr))[-ncal:]
        sc.append(zh + zl); cal.append(zh_c + zl_c); Pkeep.append((P, Pc))
    return np.stack(sc), np.stack(cal), Pkeep


def au_ms(y, arr, m):
    k = (y == 0) | m
    if arr.ndim == 1:
        return float(roc_auc_score(y[k], arr[k]))
    return float(np.mean([roc_auc_score(y[k], arr[i][k]) for i in range(arr.shape[0])]))


def pct_vs_normals(s, nrm):
    return np.searchsorted(np.sort(s[nrm]), s, side="left") / nrm.sum()


ALL = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
TYP = json.load(open(TYPJ))
for name in DATASETS:
    t0 = time.time(); log(f"\n================ {name} ================")
    d = np.load(f"{EF.OUT}/scores_{name}.npz"); y = d["label"].astype(int); n = len(y); nrm = y == 0; an = y == 1
    Ex = np.load(f"{os.environ['EXPERTS_DIR']}/expert_{name}.npz", allow_pickle=True)
    assert np.array_equal(Ex["y"].astype(int), y)
    if name == "SWaT_canon":
        assert "fit_surprise" in Ex.files and Ex["test_surprise"].shape[1] == 24, "SWaT_canon must be the official S=24 bundle"
    pf = np.load(f"{HERE}/pca_filter_doublehard_{name}.npz")
    assert np.array_equal(pf["y"], y) and np.allclose(pf["maxz"], d["maxz"])
    mthr = float(d["maxz_thr"]); diff = an & (d["maxz"] <= mthr)
    masks = {"All": an, "Difficult": diff, "DoubleHard_lin": pf["dhard_lin"].astype(bool), "DoubleHard_pca": pf["dhard_pca"].astype(bool)}
    fn, W, stride = EF.E.RAW[name]; Lblk = int(np.ceil(W / stride)) + 1
    step = STEP_S[name]; K24 = int(APRIORI_H * 3600 / step); eps = EF.episodes(y); S = Ex["test_surprise"].shape[1]
    tab = lambda arr: {m: round(au_ms(y, arr, mm), 3) for m, mm in masks.items()}
    fmt = lambda r: " ".join(f"{r[m]:.3f}" for m in masks)

    # ---- gate statistics (train-normal only)
    fa, fg = gate_stats(Ex, d)
    log(f"gate stat FA99 (calib 2nd half above 1st-half q99; nominal 0.01): global LatAD {fg:.3f}; communities median {np.median(fa):.3f}, "
        f"top: " + ", ".join(f"g{g}={fa[g]:.3f}" for g in np.argsort(-fa)[:8]))

    # ---- I1: all-off == headline; all-on == CP v2
    ens, _, _, nseed = EF.ensemble_scores(name); head = ens[HEADKEY]
    off = np.zeros(S, bool); s_off, c_off, head_P = gated_scores(Ex, d, K24, off, False)
    assert np.allclose(s_off, head, atol=1e-6), "I1 FAILED: all-off != headline"
    s_on, c_on, _ = gated_scores(Ex, d, K24, ~off, True, head_P)
    r_head, r_on = tab(head), tab(s_on); r_typ = TYP[name]["table"]["CP (24h level+scale, PRIMARY)"]
    ok_on = all(abs(r_on[m] - r_typ[m]) < 1e-3 for m in masks)
    log(f"I1 all-off == headline OK (max abs diff {np.abs(s_off - head).max():.2e}); all-on == CP v2 {'OK' if ok_on else 'MISMATCH'} ({fmt(r_on)} vs typing {fmt(r_typ)})")
    log(f"  subsets: " + ", ".join(f"{k}={int(m.sum())}" for k, m in masks.items()) + f"; {len(eps)} episodes; K24={K24}")
    log(f"  {'headline (all off)':34s} {fmt(r_head)}")

    # ---- budget sweep (sensitivity), primary = 0.05
    sweep = {}; prim = None
    for b in BUDGETS:
        onc, ong = fa > b, fg > b
        s_b, c_b, _ = gated_scores(Ex, d, K24, onc, ong, head_P)
        r = tab(s_b); sweep[str(b)] = dict(budget=b, n_comm_on=int(onc.sum()), comm_on=[int(g) for g in np.where(onc)[0]], glob_on=bool(ong), auroc=r)
        log(f"  {'gated budget ' + f'{b:.2f}' + f' ({int(onc.sum())}/{S} comm, glob {int(ong)})':34s} {fmt(r)}" + ("   <- PRIMARY" if b == BUDGET else ""))
        if b == BUDGET:
            prim = (s_b, c_b, onc, ong)
    log(f"  {'all-on (CP v2)':34s} {fmt(r_on)}")
    # secondary: global term forced ON at the primary community gate (diagnostic only, reported not selected)
    s_g, _, _ = gated_scores(Ex, d, K24, prim[2], True, head_P); r_g = tab(s_g)
    log(f"  {'primary comm gate + global ON':34s} {fmt(r_g)}   (diagnostic)")
    s_p, c_p, onc, ong = prim; r_p = sweep[str(BUDGET)]["auroc"]

    # ---- I3 non-regression
    i3 = None
    if name != "SWaT_canon":
        i3 = bool(abs(r_p["Difficult"] - r_head["Difficult"]) <= 0.02)
        log(f"I3 {'OK' if i3 else 'VIOLATED'}: Difficult headline {r_head['Difficult']:.3f} -> gated {r_p['Difficult']:.3f}")

    # ---- bootstrap gated - headline (and vs strongest non-LatAD baseline on the subset)
    base = {"trivial max|z|": d["maxz"], "IF": d["IF"], "AE": d["AE"], "linres": d["linres"], "USAD": d["USAD"], "TranAD": d["TranAD"], "LatAD (global)": d["LatAD"]}
    boots = {}
    for m, mm in masks.items():
        EF.RNG = np.random.default_rng(0); b1 = EF.boot(y, s_p, head, mm, Lblk, reps=REPS)
        alt = max(base, key=lambda k: au_ms(y, base[k], mm)); EF.RNG = np.random.default_rng(0); b2 = EF.boot(y, s_p, base[alt], mm, Lblk, reps=REPS)
        b2["alt"] = alt; b2["alt_auroc"] = round(au_ms(y, base[alt], mm), 3); boots[m] = dict(vs_headline=b1, vs_strongest=b2)
        log(f"  boot [{m:14s}] gated-headline {b1['diff']:+.3f} CI {b1['diff_ci']} P(<=0)={b1['p_le_0']} | vs {alt} {b2['alt_auroc']:.3f}: {b2['diff']:+.3f} CI {b2['diff_ci']} P={b2['p_le_0']}")

    # ---- I5 masking + detection at the train-calibrated threshold (seed-mean scores, calib-vs-calib q99)
    hm, gm = head.mean(0), s_p.mean(0); thr_h, thr_g = float(np.quantile(c_off.mean(0), 0.99)), float(np.quantile(c_p.mean(0), 0.99))
    pg = {"head": pct_vs_normals(hm, nrm), "gated": pct_vs_normals(gm, nrm)}
    def local_pct(s, e):
        idx = np.arange(max(0, e[0] - K24), min(n, e[-1] + K24 + 1)); ref = np.sort(s[idx[nrm[idx]]])
        return float(np.median(np.searchsorted(ref, s[e], side="left") / len(ref)))
    rows = []
    for e in eps:
        lh, lg = local_pct(hm, e), local_pct(gm, e)
        rows.append(dict(start=int(e[0]), n=int(len(e)), difficult=int(diff[e].sum()), global_head=round(float(np.median(pg["head"][e])), 3), global_gated=round(float(np.median(pg["gated"][e])), 3),
                         local_head=round(lh, 3), local_gated=round(lg, 3), masked_local=bool(lg < lh - 0.10), masked_global=bool(np.median(pg["gated"][e]) < np.median(pg["head"][e]) - 0.10),
                         det_head=bool((hm[e] > thr_h).any()), det_gated=bool((gm[e] > thr_g).any())))
    masked = [r for r in rows if r["masked_local"]]; mg = [r for r in rows if r["masked_global"]]
    det = dict(thr_head=round(thr_h, 3), thr_gated=round(thr_g, 3), FA_normals_head=round(float((hm[nrm] > thr_h).mean()), 3), FA_normals_gated=round(float((gm[nrm] > thr_g).mean()), 3),
               attack_windows_flagged_head=round(float((hm[an] > thr_h).mean()), 3), attack_windows_flagged_gated=round(float((gm[an] > thr_g).mean()), 3),
               difficult_windows_flagged_head=round(float((hm[diff] > thr_h).mean()), 3), difficult_windows_flagged_gated=round(float((gm[diff] > thr_g).mean()), 3),
               episodes_detected_head=int(sum(r["det_head"] for r in rows)), episodes_detected_gated=int(sum(r["det_gated"] for r in rows)), n_episodes=len(eps),
               episodes_lost=[r["start"] for r in rows if r["det_head"] and not r["det_gated"]], episodes_gained=[r["start"] for r in rows if r["det_gated"] and not r["det_head"]],
               n_masked_local=len(masked), n_masked_global=len(mg), n_gain_local_gt_0_1=int(sum(r["local_gated"] > r["local_head"] + 0.1 for r in rows)))
    log(f"I5 era-local {'OK' if not masked else 'VIOLATED'} ({len(masked)}/{len(eps)} masked >0.10, {det['n_gain_local_gt_0_1']} gained >0.1); global-rank {len(mg)} masked. "
        f"Train-calibrated q99 threshold: normal FA {det['FA_normals_head']:.3f} -> {det['FA_normals_gated']:.3f}; attack windows flagged {det['attack_windows_flagged_head']:.3f} -> {det['attack_windows_flagged_gated']:.3f}; "
        f"episodes detected {det['episodes_detected_head']} -> {det['episodes_detected_gated']} / {len(eps)} (lost {det['episodes_lost']}, gained {det['episodes_gained']})")
    for r in (rows if name == "SWaT_canon" else masked + mg):
        log(f"    ep@{r['start']:5d} n={r['n']:3d} diff {r['difficult']:3d}  global {r['global_head']:.3f}->{r['global_gated']:.3f}  local {r['local_head']:.3f}->{r['local_gated']:.3f}"
            f"  det {int(r['det_head'])}->{int(r['det_gated'])}{'  MASKED(local)' if r['masked_local'] else ''}{'  masked(global)' if r['masked_global'] else ''}")

    ALL[name] = dict(n=n, n_anom=int(an.sum()), n_episodes=len(eps), subsets={k: int(m.sum()) for k, m in masks.items()}, K24=K24, S=S,
                     gate=dict(rule="FA99 of calib 2nd half at 1st-half q99 > budget; per community and global term", primary_budget=BUDGET,
                               fa_comm=[round(float(v), 4) for v in fa], fa_glob=round(fg, 4), primary_comm_on=[int(g) for g in np.where(onc)[0]], primary_glob_on=bool(ong)),
                     headline=r_head, all_on_cp_v2=r_on, primary_gated=r_p, primary_gated_globON=r_g, sweep=sweep,
                     invariants=dict(I1_alloff_headline=True, I1_allon_cpv2=ok_on, I2_off_terms_identical=True, I3_ok=i3, I5_ok=not masked),
                     bootstrap=boots, detection=det, episodes=rows, seconds=round(time.time() - t0, 1))
    json.dump(ALL, open(OUTJ, "w"), indent=1)
    log(f"[{name} done in {time.time() - t0:.0f}s] saved -> {OUTJ}")
log("\nALL DONE")
