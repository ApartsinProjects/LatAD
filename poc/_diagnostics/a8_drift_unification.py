"""A8 (between-regime valley) vs DRIFT on the clean LatAD pipeline: are they the same phenomenon?

Inputs (nothing retrained, nothing committed):
  - A8 geometry per window: a8_valley_windows_<DS>_seed<s>.csv (clean run of a8_valley_real.py; PCA-10 + GMM on
    winfeat 'stats', partition in_cluster / valley / beyond / off, axis position t, tube distance perp, envelope dh)
  - drift / changepoint decomposition per window: recomputed here bit-identically from drift_changepoint_typing.py
    (v2 level+scale, K = 24 h a priori) on sota_bundle/experts_full/expert_<DS>.npz (SWaT_canon = official S=24,
    fit_surprise asserted): head (HCcoh+LatAD), CP (changepoint), DR (drift) seed-mean scores, DRIFTED flag
    (slow baseline above the calib 99th pct in >= 1 community), per-community slow baseline in calib-SD units.
Questions (stated before any number was read):
  Q1 clean A8 presence: valley share of attacks and of test normals per dataset (the leaked SWaT numbers are void).
  Q2 A8 x drift: (a) cross-tab of the geometric partition with the DRIFTED flag on normals and by record third;
     (b) Spearman of the continuous between-ness min(t, 1-t) [and perp, dh] with the DR score, normals / attacks;
     (c) do the drifting communities carry the valley structure: the same partition refit on the channel subspace of
     the drifting communities vs on the complement; (d) drift DIRECTION vs valley AXIS: cosine between the drifted
     normals' displacement from their home regime and the axis to the second-nearest regime, and whether the drift
     path ends inside another train regime (A8) or outside every regime (departure).
  Q3 typing of valley attacks: head / CP / DR AUROC per partition, CHANGEPOINT / DRIFT / SUBTHRESHOLD typing of the
     head-flagged attacks per partition, era-local percentile change head -> CP per partition.
Invariants: (I1) recomputed head equals ensemble_final 'HCcoh+LatAD' (max abs diff < 1e-6); (I2) window counts agree
  between csv, expert and scores (1498 / 14819 / 575); (I3) the typing table reproduces drift_changepoint_typing.json
  (test-normal drifted fraction 0.666 / 0.297 / 0.000 and 7 / 0 / 0 drifting communities); (I4) a shuffled DRIFTED
  flag gives |Spearman| < 0.05 with between-ness (null).
Outputs: a8_drift_unification.json, a8_drift_unification.log. Usage (from poc/): python _diagnostics/a8_drift_unification.py [DS ...]
"""
from __future__ import annotations
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ["EXPERTS_DIR"] = os.path.join(ROOT, "sota_bundle", "experts_full")
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, fisher_exact
import ensemble_final as EF
from a8_valley_real import Geo, CACHE, PARTS, KS  # clean geometry code (same GMM recipe)

OUTJ = os.path.join(HERE, "a8_drift_unification.json")
LOG = open(os.path.join(HERE, "a8_drift_unification.log"), "a")
STEP_S = {"SWaT_canon": 300, "WADI_clean": 300, "HAI": 30}
APRIORI = 24; SEEDS_GEO = (0, 1, 2); NPC = 10
DATASETS = sys.argv[1:] or ["SWaT_canon", "HAI", "WADI_clean"]


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


# ------------------------------------------------------------------ decomposition (copied verbatim from drift_changepoint_typing.py)
def causal_median(x, K, init, gap=0):
    m = pd.Series(np.asarray(x, float)).rolling(K, min_periods=max(3, K // 4)).median().shift(1 + gap).values
    return np.where(np.isnan(m), init, m)


def decompose(X, K, init, gap, scale=False, cmad=None):
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
    S = RC.shape[0]
    P = np.stack([EF.pval(RC[g], RT[g]) for g in range(S)]); Pc = np.stack([EF.pval(RC[g], RC[g]) for g in range(S)])
    hc, hcc = EF.HC(P, wt=w), EF.HC(Pc, wt=w)
    return z(hc, hcc), z(hcc, hcc)


def scores(Ex, d, K, gap=0, detrend_global=True, scale=False):
    Cal, Tst = Ex["calib_surprise"], Ex["test_surprise"]; nseed, S, n = Tst.shape
    w = Ex["comm_cohesion"].astype(float) * np.sqrt(Ex["comm_size"].astype(float))
    lat, lat_tr = d["LatAD"], d["LatAD_train"]; nseed = min(nseed, lat.shape[0])
    out = {k: [] for k in ("cp", "cp_cal", "dr", "dr_cal", "head", "head_cal", "slowT", "fastT")}
    for sd in range(nseed):
        C, T, L, Ltr = Cal[sd].astype(float), Tst[sd].astype(float), lat[sd].astype(float), lat_tr[sd].astype(float)
        init = np.median(C, 1); cmad = calib_mad(C)
        slowC, RC = decompose(C, K, init, gap, scale, cmad); slowT, RT = decompose(T, K, init, gap, scale, cmad)
        ncal = C.shape[1]
        zh, zh_c = hc_fuse(C, T, w)
        zl_head = z(EF.surv(Ltr, L), EF.surv(Ltr, Ltr)); zl_head_c = z(EF.surv(Ltr, Ltr), EF.surv(Ltr, Ltr))[-ncal:]
        out["head"].append(zh + zl_head); out["head_cal"].append(zh_c + zl_head_c)
        zc, zc_c = hc_fuse(RC, RT, w)
        if detrend_global and K is not None:
            li = np.median(Ltr[None, :], 1); lm = calib_mad(Ltr[-ncal:][None, :])
            Ltr_r = decompose(Ltr[None, :], K, li, gap, scale, lm)[1][0]; L_r = decompose(L[None, :], K, li, gap, scale, lm)[1][0]
            zl = z(EF.surv(Ltr_r, L_r), EF.surv(Ltr_r, Ltr_r)); zl_c = z(EF.surv(Ltr_r, Ltr_r), EF.surv(Ltr_r, Ltr_r))[-ncal:]
        else:
            zl, zl_c = zl_head, zl_head_c
        out["cp"].append(zc + zl); out["cp_cal"].append(zc_c + zl_c)
        if K is None:
            out["dr"].append(np.zeros(n)); out["dr_cal"].append(np.zeros(C.shape[1]))
        else:
            zd, _ = hc_fuse(C, slowT, w); zdc, _ = hc_fuse(C, slowC, w)
            out["dr"].append(zd); out["dr_cal"].append(zdc)
        out["slowT"].append(slowT); out["fastT"].append(RT)
    return {k: np.stack(v) for k, v in out.items()}


# ------------------------------------------------------------------ helpers
def auroc(y, s):
    if y.sum() == 0 or y.sum() == len(y): return float("nan")
    return float(roc_auc_score(y, s))


def part_auroc(s, nrm, idx):
    if len(idx) == 0: return float("nan")
    return auroc(np.r_[np.zeros(nrm.sum()), np.ones(len(idx))], np.r_[s[nrm], s[idx]])


def pct_vs(s, nrm):
    return np.searchsorted(np.sort(s[nrm]), s, side="left") / nrm.sum()


def local_pct(s, idx, nrm, K):
    """era-local percentile (vs test normals within +-K windows) of each window in idx."""
    n = len(s); out = np.zeros(len(idx))
    for q, i in enumerate(idx):
        r = np.arange(max(0, i - K), min(n, i + K + 1)); ref = np.sort(s[r[nrm[r]]])
        out[q] = np.searchsorted(ref, s[i], side="left") / max(1, len(ref))
    return out


def sp(a, b):
    if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0: return (float("nan"), float("nan"))
    r = spearmanr(a, b); return (float(r.statistic), float(r.pvalue))


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def feat_cols(ch_idx, C):
    """winfeat 'stats' is stat-major: 6 blocks of C channels."""
    return np.concatenate([np.asarray(ch_idx) + k * C for k in range(6)])


def subspace_partition(Ftr, Fte, cols, seed):
    mu, sd = Ftr[:, cols].mean(0), Ftr[:, cols].std(0) + 1e-8
    Ztr, Zte = (Ftr[:, cols] - mu) / sd, (Fte[:, cols] - mu) / sd
    npc = min(NPC, len(cols)); pca = PCA(npc, random_state=0).fit(Ztr); Ztr, Zte = pca.transform(Ztr), pca.transform(Zte)
    G = Geo(Ztr, seed); R = float(np.quantile(G.dist(Ztr).min(1), 0.995)); P = G.partition(Zte, R)
    return G, R, P, Ztr, Zte


# ------------------------------------------------------------------ main
ALL = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
for name in DATASETS:
    t0 = time.time(); log(f"\n================ {name} ================")
    d = np.load(f"{EF.OUT}/scores_{name}.npz"); y = d["label"].astype(int); n = len(y); nrm = y == 0; an = y == 1
    Ex = np.load(f"{os.environ['EXPERTS_DIR']}/expert_{name}.npz", allow_pickle=True)
    assert np.array_equal(Ex["y"].astype(int), y)
    if name == "SWaT_canon":
        assert "fit_surprise" in Ex.files and Ex["test_surprise"].shape[1] == 24, "SWaT_canon must be the official S=24 bundle with fit_surprise"
    step = STEP_S[name]; K24 = int(APRIORI * 3600 / step); third = np.minimum((np.arange(n) * 3) // n, 2)
    mthr = float(d["maxz_thr"]); diff = an & (d["maxz"] <= mthr)
    # ---- drift decomposition (v2, 24 h)
    ens, y2, _, nseed = EF.ensemble_scores(name); head_ref = ens["HCcoh+LatAD"]
    SP = scores(Ex, d, K24, scale=True)
    assert np.allclose(SP["head"], head_ref, atol=1e-6), "I1 FAILED"
    log(f"I1 OK: recomputed head == ensemble_final HCcoh+LatAD (max abs diff {np.abs(SP['head'] - head_ref).max():.1e}); windows {n}, attacks {an.sum()}, difficult {diff.sum()}, K24={K24}")
    hm, cm, dm = SP["head"].mean(0), SP["cp"].mean(0), SP["dr"].mean(0)
    thr_h, thr_c, thr_d = (float(np.quantile(SP[k].mean(0), 0.99)) for k in ("head_cal", "cp_cal", "dr_cal"))
    slowT = SP["slowT"].mean(0); Cal = Ex["calib_surprise"].astype(float).mean(0); S = slowT.shape[0]
    q99g = np.quantile(Cal, 0.99, axis=1); cmed = np.median(Cal, 1); csd = Cal.std(1) + 1e-9
    fh, fc = hm > thr_h, cm > thr_c
    fd = (slowT > q99g[:, None]).any(0)                                  # DRIFTED window (typing definition)
    slow_sd = (slowT - cmed[:, None]) / csd[:, None]                     # per-community slow baseline, calib-SD units
    drift_mag = slow_sd.max(0)                                           # strongest community drift per window
    typ = np.where(fc, "changepoint", np.where(fd, "drift", "subthreshold"))
    idxn = np.where(nrm)[0]
    drifting = [g for g in range(S) if (slowT[g][idxn] > q99g[g]).mean() >= 0.5]
    # HAI has no community at the >=50% rule; its drift is the front-loaded level step in g1/g11 (typing .md), used for the subspace test
    SUB_OVERRIDE = {"HAI": [1, 11]}
    sub_comms = drifting if drifting else SUB_OVERRIDE.get(name, [])
    tn_drifted = float(fd[nrm].mean())
    log(f"I3: test-normal drifted {tn_drifted:.3f}, drifting communities {len(drifting)}/{S} {drifting}")
    names = None
    try:
        raw = np.load(os.path.join(CACHE, f"raw_{name}.npz"), allow_pickle=True); names = [str(c) for c in raw["ch"]]; C = len(names)
    except Exception as ex:
        log(f"  [warn] raw cache without channel names: {ex}")
    comm_ch = [[int(c) for c in Ex["comm_channels"][g] if c >= 0] for g in range(S)]
    drift_ch = sorted(set(c for g in sub_comms for c in comm_ch[g]))
    other_ch = sorted(set(range(C)) - set(drift_ch)) if names else []
    if names:
        log(f"  subspace test communities {sub_comms} -> channels ({len(drift_ch)}): {[names[c] for c in drift_ch]}")

    res = dict(n=n, n_attacks=int(an.sum()), n_difficult=int(diff.sum()), K24=K24, thr=dict(head=thr_h, cp=thr_c, drift=thr_d),
               test_normal_drifted=tn_drifted, drifting_communities=drifting, drift_channels=[names[c] for c in drift_ch] if names else drift_ch,
               seeds={})
    # ---- per geometry seed
    Fc = np.load(os.path.join(CACHE, f"a8_valley_feat_{name}.npz")); Ftr, Fte = Fc["Ftr"], Fc["Fte"]
    assert len(Fte) == n, "I2 FAILED: feature/window count"
    for seed in SEEDS_GEO:
        csv = os.path.join(HERE, f"a8_valley_windows_{name}_seed{seed}.csv")
        if not os.path.exists(csv): log(f"  seed {seed}: no csv, skipped"); continue
        V = pd.read_csv(csv); assert len(V) == n and np.array_equal(V["label"].values, y), "I2 FAILED: csv alignment"
        part = V["part"].values.astype(str); t = V["t"].values; perp = V["perp"].values; dh = V["dh"].values
        betw = np.where(part == "in_cluster", 0.0, np.clip(np.minimum(t, 1 - t), 0, None))   # between-ness, 0 inside a regime
        outenv = part != "in_cluster"
        R = dict()
        # Q1 shares
        R["share"] = {who: {p: r3((part[m] == p).mean()) for p in PARTS} for who, m in (("normals", nrm), ("attacks", an), ("difficult", diff))}
        R["share_normals_by_third"] = [{p: r3((part[nrm & (third == k)] == p).mean()) for p in PARTS} for k in range(3)]
        R["drifted_normals_by_third"] = [r3(fd[nrm & (third == k)].mean()) for k in range(3)]
        log(f"\n[seed {seed}] Q1 partition shares: normals {R['share']['normals']} | attacks {R['share']['attacks']} | difficult {R['share']['difficult']}")
        log(f"  normals by third: " + " | ".join(f"T{k+1} valley {R['share_normals_by_third'][k]['valley']} off {R['share_normals_by_third'][k]['off']} in {R['share_normals_by_third'][k]['in_cluster']} drifted {R['drifted_normals_by_third'][k]}" for k in range(3)))
        # Q2a cross-tab partition x drifted on normals
        ct = {}
        for p in PARTS:
            m = nrm & (part == p); ct[p] = dict(n=int(m.sum()), drifted=r3(fd[m].mean()) if m.any() else None,
                                                 drift_mag_med=r3(np.median(drift_mag[m])) if m.any() else None,
                                                 DR_pct_med=r3(np.median(pct_vs(dm, nrm)[m])) if m.any() else None)
        a, b = int((nrm & (part == "valley") & fd).sum()), int((nrm & (part == "valley") & ~fd).sum())
        c, dd = int((nrm & (part != "valley") & fd).sum()), int((nrm & (part != "valley") & ~fd).sum())
        R["normals_part_x_drifted"] = ct; R["valley_vs_drifted_fisher_p"] = float(fisher_exact([[a, b], [c, dd]])[1]) if tn_drifted > 0 else None
        log(f"  Q2a normals: partition x DRIFTED: " + " | ".join(f"{p} n={v['n']} drifted {v['drifted']} slow-max {v['drift_mag_med']} sd, DR pct {v['DR_pct_med']}" for p, v in ct.items()) + f"  (valley vs drifted Fisher p {R['valley_vs_drifted_fisher_p']})")
        # where are the drifted normals geometrically?
        md = nrm & fd
        R["drifted_normals_partition"] = {p: r3((part[md] == p).mean()) for p in PARTS} if md.any() else None
        R["nondrifted_normals_partition"] = {p: r3((part[nrm & ~fd] == p).mean()) for p in PARTS}
        log(f"  drifted normals fall in: {R['drifted_normals_partition']} ; non-drifted normals: {R['nondrifted_normals_partition']}")
        # Q2b correlations
        cor = {}
        for who, m in (("normals", nrm), ("attacks", an), ("all", np.ones(n, bool))):
            mo = m & outenv
            cor[who] = dict(n_out=int(mo.sum()),
                            betw_vs_DR=sp(betw[mo], dm[mo]), betw_vs_driftmag=sp(betw[mo], drift_mag[mo]), betw_vs_CP=sp(betw[mo], cm[mo]),
                            perp_vs_DR=sp(perp[mo], dm[mo]), dh_vs_DR=sp(dh[mo], dm[mo]), dh_vs_driftmag=sp(dh[mo], drift_mag[mo]),
                            valley_ind_vs_DR=sp((part[m] == "valley").astype(float), dm[m]), off_ind_vs_DR=sp((part[m] == "off").astype(float), dm[m]),
                            outenv_ind_vs_DR=sp(outenv[m].astype(float), dm[m]), dh_vs_DR_allwin=sp(dh[m], dm[m]))
            rng = np.random.default_rng(seed); fds = rng.permutation(fd[mo]); cor[who]["I4_null_betw_vs_shuffled_drifted"] = sp(betw[mo], fds.astype(float))
        R["correlations"] = cor
        for who in cor:
            c_ = cor[who]; log(f"  Q2b {who:8s} (n out-of-envelope {c_['n_out']}): rho(betweenness, DR) {c_['betw_vs_DR'][0]:+.3f} p={c_['betw_vs_DR'][1]:.2g}; rho(betw, slow-max sd) {c_['betw_vs_driftmag'][0]:+.3f};"
                f" rho(perp, DR) {c_['perp_vs_DR'][0]:+.3f}; rho(dh, DR) {c_['dh_vs_DR'][0]:+.3f}; rho(dh, slow-max) {c_['dh_vs_driftmag'][0]:+.3f};"
                f" rho(valley-ind, DR) {c_['valley_ind_vs_DR'][0]:+.3f}; rho(off-ind, DR) {c_['off_ind_vs_DR'][0]:+.3f}; rho(out-of-env, DR) {c_['outenv_ind_vs_DR'][0]:+.3f}; I4 null {c_['I4_null_betw_vs_shuffled_drifted'][0]:+.3f}")
        # Q2d direction: drifted normals' displacement from home vs axis to second-nearest regime (full-space geometry refit, same seed)
        mu0, sd0 = Ftr.mean(0), Ftr.std(0) + 1e-8
        pca = PCA(NPC, random_state=0).fit((Ftr - mu0) / sd0); Ztr, Zte = pca.transform((Ftr - mu0) / sd0), pca.transform((Fte - mu0) / sd0)
        G = Geo(Ztr, seed)
        h, j = V["h"].values.astype(int), V["j"].values.astype(int)
        Pchk = G.partition(Zte, float(np.quantile(G.dist(Ztr).min(1), 0.995)))
        agree_h = float((Pchk["h"] == h).mean()); agree_part = float((Pchk["part"].astype(str) == part).mean())
        log(f"  geometry refit check (seed {seed}, K={G.K}): home agreement with csv {agree_h:.3f}, partition agreement {agree_part:.3f}")
        R["geometry_refit_agreement"] = dict(home=r3(agree_h), part=r3(agree_part), K=int(G.K))
        # drift direction = drifted-normal centroid minus non-drifted-normal centroid (late-third minus train centroid when nothing is drifted)
        vdrift = (Zte[nrm & fd].mean(0) - Zte[nrm & ~fd].mean(0)) if fd[nrm].any() and (~fd & nrm).any() else (Zte[nrm & (third == 2)].mean(0) - Ztr.mean(0))
        cos_axis, cos_disp = [], []
        mo = nrm & outenv; io = np.where(mo)[0]
        for i in io:
            u = Zte[i] - G.mu[h[i]]; v = G.mu[j[i]] - G.mu[h[i]]
            cos_axis.append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))
            cos_disp.append(float(u @ vdrift / (np.linalg.norm(u) * np.linalg.norm(vdrift) + 1e-12)))
        cos_axis, cos_disp = np.array(cos_axis), np.array(cos_disp); dsel = fd[io]
        R["direction_drifted_out_normals"] = dict(n=int(dsel.sum()), cos_axis_med=r3(np.median(cos_axis[dsel])) if dsel.any() else None,
                                                  frac_cos_axis_gt_0_7=r3((cos_axis[dsel] > 0.7).mean()) if dsel.any() else None,
                                                  cos_drift_med=r3(np.median(cos_disp[dsel])) if dsel.any() else None,
                                                  frac_cos_drift_gt_0_7=r3((cos_disp[dsel] > 0.7).mean()) if dsel.any() else None,
                                                  t_med=r3(np.median(t[io][dsel])) if dsel.any() else None, perp_over_R_med=None)
        # does the drift path end inside another train regime? nearest-component distance of last-third normals vs R
        Rq = float(np.quantile(G.dist(Ztr).min(1), 0.995)); dist_te = G.dist(Zte)
        late = nrm & (third == 2); early = nrm & (third == 0)
        R["direction"] = dict(n_out_normals=int(mo.sum()),
                              cos_disp_vs_axis_hj_med=r3(np.median(cos_axis)), frac_cos_axis_gt_0_7=r3((cos_axis > 0.7).mean()),
                              cos_disp_vs_late_drift_dir_med=r3(np.median(cos_disp)), frac_cos_drift_gt_0_7=r3((cos_disp > 0.7).mean()),
                              late_normals_nearest_dist_med=r3(np.median(dist_te[late].min(1))) if late.any() else None,
                              early_normals_nearest_dist_med=r3(np.median(dist_te[early].min(1))) if early.any() else None, R=r3(Rq),
                              late_normals_inside_any_regime=r3((dist_te[late].min(1) <= Rq).mean()) if late.any() else None,
                              late_normals_second_nearest_over_nearest_med=r3(np.median(np.sort(dist_te[late], 1)[:, 1] / np.sort(dist_te[late], 1)[:, 0])) if late.any() else None,
                              late_normals_t_med=r3(np.median(t[late])) if late.any() else None, late_normals_perp_over_R_med=r3(np.median(perp[late]) / Rq) if late.any() else None,
                              home_regime_of_late_normals=[int(k) for k in np.bincount(h[late], minlength=G.K).argsort()[::-1][:3]] if late.any() else None,
                              home_regime_of_early_normals=[int(k) for k in np.bincount(h[early], minlength=G.K).argsort()[::-1][:3]] if early.any() else None)
        Dd = R["direction"]; R["direction_drifted_out_normals"]["perp_over_R_med"] = r3(np.median(perp[io][dsel]) / Rq) if dsel.any() else None; Dx = R["direction_drifted_out_normals"]
        log(f"  Q2d direction (out-of-envelope normals n={Dd['n_out_normals']}): cos(displacement, axis h->j) median {Dd['cos_disp_vs_axis_hj_med']} (>0.7: {Dd['frac_cos_axis_gt_0_7']}); cos(displacement, drift direction) median {Dd['cos_disp_vs_late_drift_dir_med']} (>0.7: {Dd['frac_cos_drift_gt_0_7']})")
        log(f"     DRIFTED out-of-envelope normals n={Dx['n']}: cos(axis h->j) median {Dx['cos_axis_med']} (>0.7: {Dx['frac_cos_axis_gt_0_7']}); cos(drift dir) median {Dx['cos_drift_med']} (>0.7: {Dx['frac_cos_drift_gt_0_7']}); t median {Dx['t_med']}, perp/R median {Dx['perp_over_R_med']}")
        log(f"     late-third normals: nearest-regime dist median {Dd['late_normals_nearest_dist_med']} vs R {Dd['R']} (early third {Dd['early_normals_nearest_dist_med']}); inside any regime {Dd['late_normals_inside_any_regime']}; d2/d1 median {Dd['late_normals_second_nearest_over_nearest_med']}; t median {Dd['late_normals_t_med']}, perp/R median {Dd['late_normals_perp_over_R_med']}; home regimes late {Dd['home_regime_of_late_normals']} early {Dd['home_regime_of_early_normals']}")
        # Q2c subspace geometry: drifting-community channels vs complement
        sub = {}
        if names and drift_ch and other_ch:
            for lab, chs in (("drifting_comm_channels", drift_ch), ("other_channels", other_ch)):
                cols = feat_cols(chs, C); Gs, Rs, Ps, Ztr_s, Zte_s = subspace_partition(Ftr, Fte, cols, seed); ps = Ps["part"].astype(str)
                bs = np.where(ps == "in_cluster", 0.0, np.clip(np.minimum(Ps["t"], 1 - Ps["t"]), 0, None)); os_ = ps != "in_cluster"
                sub[lab] = dict(n_channels=len(chs), K=int(Gs.K), R=r3(Rs),
                                normals={p: r3((ps[nrm] == p).mean()) for p in PARTS}, attacks={p: r3((ps[an] == p).mean()) for p in PARTS},
                                normals_by_third=[{p: r3((ps[nrm & (third == k)] == p).mean()) for p in PARTS} for k in range(3)],
                                drifted_normals={p: r3((ps[nrm & fd] == p).mean()) for p in PARTS} if fd[nrm].any() else None,
                                nondrifted_normals={p: r3((ps[nrm & ~fd] == p).mean()) for p in PARTS},
                                rho_betw_DR_normals=sp(bs[nrm & os_], dm[nrm & os_]), rho_dh_DR_normals=sp(Ps["dh"][nrm], dm[nrm]),
                                late_normals_inside_any_regime=r3((Ps["dh"][nrm & (third == 2)] <= Rs).mean()),
                                late_normals_perp_over_R_med=r3(np.median(Ps["perp"][nrm & (third == 2)]) / Rs),
                                n_close_pairs_D_lt_2R=int((Gs.D[np.triu_indices(Gs.K, 1)] < 2 * Rs).sum()))
                s_ = sub[lab]
                log(f"  Q2c subspace {lab} ({len(chs)} ch, K={s_['K']}, R={s_['R']}): normals {s_['normals']} attacks {s_['attacks']}")
                log(f"       normals by third: " + " | ".join(f"T{k+1} valley {s_['normals_by_third'][k]['valley']} off {s_['normals_by_third'][k]['off']} in {s_['normals_by_third'][k]['in_cluster']}" for k in range(3))
                    + f"; drifted normals {s_['drifted_normals']}; rho(betw, DR) normals {s_['rho_betw_DR_normals'][0]:+.3f} p={s_['rho_betw_DR_normals'][1]:.2g}; late inside-regime {s_['late_normals_inside_any_regime']}, perp/R {s_['late_normals_perp_over_R_med']}")
        R["subspace"] = sub
        # Q3 typing of attacks per partition
        ty = {}
        pg = {k: pct_vs(v, nrm) for k, v in (("head", hm), ("cp", cm), ("dr", dm))}
        for p in PARTS + ["all"]:
            m = an & ((part == p) if p != "all" else True); idx = np.where(m)[0]
            if len(idx) == 0: ty[p] = dict(n=0); continue
            mf = m & fh
            ty[p] = dict(n=len(idx), n_difficult=int((m & diff).sum()), flagged_head=r3(fh[m].mean()), flagged_cp=r3(fc[m].mean()),
                         typed_of_head_flagged={k: r3((typ[mf] == k).mean()) for k in ("changepoint", "drift", "subthreshold")} if mf.any() else None,
                         drifted_frac=r3(fd[m].mean()),
                         auroc_head=r3(part_auroc(hm, nrm, idx)), auroc_cp=r3(part_auroc(cm, nrm, idx)), auroc_dr=r3(part_auroc(dm, nrm, idx)),
                         global_pct_med=dict(head=r3(np.median(pg["head"][m])), cp=r3(np.median(pg["cp"][m])), dr=r3(np.median(pg["dr"][m]))),
                         local_pct_med=dict(head=r3(np.median(local_pct(hm, idx, nrm, K24))), cp=r3(np.median(local_pct(cm, idx, nrm, K24)))))
            t_ = ty[p]
            log(f"  Q3 attacks {p:11s} n={t_['n']:4d} (diff {t_['n_difficult']:3d}): flagged head {t_['flagged_head']} cp {t_['flagged_cp']}; typed {t_['typed_of_head_flagged']}; drifted {t_['drifted_frac']};"
                f" AUROC head/cp/dr {t_['auroc_head']}/{t_['auroc_cp']}/{t_['auroc_dr']}; global pct head/cp {t_['global_pct_med']['head']}/{t_['global_pct_med']['cp']}; era-local head/cp {t_['local_pct_med']['head']}/{t_['local_pct_med']['cp']}")
        R["attack_typing"] = ty
        res["seeds"][str(seed)] = R
    ALL = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}   # re-read: parallel per-dataset processes share the file
    ALL[name] = res; json.dump(ALL, open(OUTJ, "w"), indent=1, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o))
    log(f"[{name} done in {time.time() - t0:.0f}s] saved -> {OUTJ}")
log("\nALL DONE")
