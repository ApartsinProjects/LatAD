"""DOUBLE-HARD subset with a PCA linear-Gaussian second filter (replaces LinRes as filter B).

Copy of rev4_doublehard.py with ONLY filter B changed. A window is double-hard iff it is an anomaly AND
separated by NEITHER filter, both calibrated on TRAIN-normal at the 99th percentile:
  A. univariate max|z| over the mean stat-block (maxz / maxz_thr from scores_<DS>.npz, unchanged);
  B. PCA on the train-normal standardized windowed feature matrix (the identical Xtr the detectors see:
     build_scores_table FIX 2 standardization + eda_real CLIP; no one-hot indicators). Keep the components
     explaining 95% of train-normal variance (sensitivity at 90/99). Two statistics:
        T2  = sum_{retained i} score_i^2 / lambda_i           (Hotelling, within-subspace Mahalanobis)
        SPE = ||x_centered||^2 - sum_{retained i} score_i^2   (Q statistic, energy in dropped components)
     Filter B flags a window if T2 > q99(T2_train) OR SPE > q99(SPE_train).
NEW double-hard = anomaly & maxz<=maxz_thr & T2<=thrT2 & SPE<=thrSPE. The detector under test never
participates. Every method (including LinRes, now a fair detector) is re-scored on the new subset; the
episode-block bootstrap (identical to rev4_doublehard.boot) tests the regime-community headline
(HCcoh+LatAD, experts_full) against the strongest DETECTOR (trivial max|z| excluded: it is a filter).
Also reports the T2-only variant (filter B = T2 only). Outputs:
  _diagnostics/doublehard_pca_all3.json           all numbers
  _diagnostics/pca_filter_doublehard_<DS>.npz       per-window T2 / SPE / flags (intermediate results)
"""
from __future__ import annotations
import json, os, sys, time, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
POC = os.path.dirname(os.path.abspath(__file__)); os.chdir(POC); sys.path.insert(0, POC)
os.environ.setdefault("EXPERTS_DIR", "sota_bundle/experts_full")
import eda_real as E
from onehot_filter import build_feats, loco_residual
import ensemble_final as EF

OUT = os.path.join(POC, "_diagnostics")
DATASETS = sys.argv[1:] or ["WADI_clean", "HAI", "SWaT_canon"]
GDN_FILE = {"WADI_clean": "score_GDN_WADI_clean_s0.npy", "HAI": "gdn_fast_score_HAI_s0.npy",
            # SWaT_canon: use the clean official-normal GDN dump; the sibling
            # _diagnostics/score_GDN_SWaT_canon_s0.npy is the leaked/corrupted pre-fix dump (quarantined).
            "SWaT_canon": "sota_pull_official/score_GDN_SWaT_canon_s0.npy"}
HEADKEY = "HCcoh+LatAD"                       # the paper's "LatAD (regime-community)" headline
REPS = int(os.environ.get("BOOT_REPS", "2000"))
VAR_KEEP = 0.95; VAR_SENS = (0.90, 0.95, 0.99)
RNG = np.random.default_rng(0)


def log(msg):
    print(msg, flush=True)


def episodes(y):
    eps, i, n = [], 0, len(y)
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1:
                j += 1
            eps.append(np.arange(i, j)); i = j
        else:
            i += 1
    return eps


def mean_auroc(arr, yk, keep):
    if arr.ndim == 2:
        return float(np.nanmean([roc_auc_score(yk, arr[i][keep]) for i in range(arr.shape[0])]))
    return float(roc_auc_score(yk, arr[keep]))


def boot(y, method, compet, hard, L, reps=2000):
    """Identical to rev4_doublehard.boot: moving-block bootstrap of normals + episode bootstrap of the
    hard windows; one-sided P = fraction of resampled paired differences <= 0."""
    y = y.astype(int); norm = np.where(y == 0)[0]
    heps = [e[hard[e]] for e in episodes(y) if hard[e].any()]
    keep0 = np.where((y == 0) | hard)[0]
    a_pt = mean_auroc(method, y[keep0], keep0); c_pt = mean_auroc(compet, y[keep0], keep0)
    diffs = []
    for _ in range(reps):
        nb = int(np.ceil(len(norm) / L))
        st = RNG.integers(0, max(1, len(norm) - L + 1), size=nb)
        sn = np.concatenate([norm[s:s + L] for s in st])[:len(norm)]
        if not heps:
            break
        pick = RNG.integers(0, len(heps), size=len(heps))
        sh = np.concatenate([heps[k] for k in pick])
        if len(sh) < 2:
            continue
        keep = np.concatenate([sn, sh]); yk = y[keep]
        if yk.sum() < 2 or (yk == 0).sum() < 2:
            continue
        diffs.append(mean_auroc(method, yk, keep) - mean_auroc(compet, yk, keep))
    diffs = np.array(diffs)
    q = lambda v: [round(float(np.quantile(v, 0.025)), 3), round(float(np.quantile(v, 0.975)), 3)]
    return dict(diff=round(a_pt - c_pt, 3), diff_ci=q(diffs) if len(diffs) else None,
                p_le_0=round(float((diffs <= 0).mean()), 4) if len(diffs) else None,
                n_episodes=len(heps), n_boot=int(len(diffs)), method_auroc=round(a_pt, 3),
                compet_auroc=round(c_pt, 3))


def detector_inputs(name):
    """The standardized windowed feature matrices the detectors consume (build_scores_table FIX 2)."""
    D = E.load(name)
    Xtr0, Xte0 = D["Xn_w"].astype(np.float64), D["Xa_w"].astype(np.float64)
    clipv = E.CLIP.get(name)
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr, Xte = (Xtr0 - mu) / sig, (Xte0 - mu) / sig
    if clipv:
        Xtr, Xte = np.clip(Xtr, -clipv, clipv), np.clip(Xte, -clipv, clipv)
    return D, Xtr, Xte


def pca_stats(Xtr, Xte, var_keep):
    """Fit PCA on train (already train-centered); return T2/SPE for train and test, k, eigenvalues."""
    m = Xtr.mean(0); Ztr, Zte = Xtr - m, Xte - m            # re-centre (clip can shift the mean slightly)
    U, S, Vt = np.linalg.svd(Ztr, full_matrices=False)
    lam = S ** 2 / (len(Ztr) - 1)                          # eigenvalues of the train covariance
    cum = np.cumsum(lam) / lam.sum()
    k = int(min(np.searchsorted(cum, var_keep) + 1, len(lam)))
    lam_k = lam[:k]; P = Vt[:k].T                          # (d,k)
    def stats(Z):
        sc = Z @ P
        t2 = (sc ** 2 / lam_k).sum(1)
        spe = (Z ** 2).sum(1) - (sc ** 2).sum(1)
        return t2, np.maximum(spe, 0.0)
    return stats(Ztr), stats(Zte), k, lam, cum


def gdn_windows(name, Xa_raw, W, stride, n):
    f = os.path.join(OUT, GDN_FILE[name])
    if not os.path.exists(f):
        return None
    ts = np.load(f); ts = ts.mean(1) if ts.ndim > 1 else ts
    starts = list(range(0, len(Xa_raw) - W + 1, stride))
    if len(ts) != len(Xa_raw) or len(starts) != n:
        log(f"   [warn] GDN length mismatch on {name}: ts {len(ts)} raw {len(Xa_raw)} wins {len(starts)} vs {n}")
        return None
    return np.array([ts[i:i + W].mean() for i in starts], np.float32)


ALL = {}
for name in DATASETS:
    t0 = time.time()
    fn, W, stride = E.RAW[name]
    D, Xtr, Xte = detector_inputs(name)
    d = np.load(f"{OUT}/scores_{name}.npz"); y = d["label"].astype(int)
    assert np.array_equal(y, D["ya_w"].astype(int)), f"{name}: label mismatch scores vs eda_real"
    assert len(Xte) == len(y)
    mthr = float(d["maxz_thr"]); maxz = d["maxz"]
    Xn_raw, Xa_raw = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)

    # ---- filter B (old): one-hot LinRes, for the comparison + as a fair detector on the new subset
    Fn, Fa, grp = build_feats(Xn_raw, Xa_raw, W, stride, onehot=True)
    r_tr, r_te = loco_residual(Fn, Fa[:len(y)], grp); lin_thr = float(np.quantile(r_tr, 0.99))
    canon = (y == 1) & (maxz <= mthr)
    dhard_lin = canon & (r_te <= lin_thr)

    # ---- filter B (new): PCA T2 + SPE, 95% variance, thresholds on train-normal
    (t2_tr, spe_tr), (t2_te, spe_te), k, lam, cum = pca_stats(Xtr, Xte, VAR_KEEP)
    thr_t2 = float(np.quantile(t2_tr, 0.99)); thr_spe = float(np.quantile(spe_tr, 0.99))
    sepB = (t2_te > thr_t2) | (spe_te > thr_spe)
    dhard = canon & ~sepB
    dhard_t2only = canon & (t2_te <= thr_t2)
    # invariants
    assert not (dhard & ~canon).any(), "double-hard must be a subset of difficult"
    tr_flag = ((t2_tr > thr_t2) | (spe_tr > thr_spe)).mean()
    d_full = Xtr.shape[1]
    # PCA self-consistency: with ALL components, SPE must vanish and T2 = full Mahalanobis (rank permitting)
    (t2_full_tr, spe_full_tr), _, k_full, _, _ = pca_stats(Xtr, Xte[:2], 1.0 + 1e-12)
    log(f"\n=== {name}: d={d_full}, n_train={len(Xtr)}, n_test={len(y)}, anomalies={int(y.sum())}, "
        f"difficult={int(canon.sum())} ===")
    log(f"   PCA k@95%={k}/{d_full}  (k@90%={int(np.searchsorted(cum, .90) + 1)}, k@99%={int(np.searchsorted(cum, .99) + 1)}); "
        f"thrT2={thr_t2:.2f} thrSPE={thr_spe:.2f}; train flag rate {tr_flag*100:.2f}% (expect ~1-2%); "
        f"all-components SPE max={spe_full_tr.max():.2e} (expect ~0), k_full={k_full}")
    # per-window intermediate results, saved before anything else
    np.savez(f"{OUT}/pca_filter_doublehard_{name}.npz", y=y, maxz=maxz, maxz_thr=mthr, t2=t2_te, spe=spe_te,
             t2_train=t2_tr, spe_train=spe_tr, thr_t2=thr_t2, thr_spe=thr_spe, k=k, linres_te=r_te,
             linres_tr=r_tr, lin_thr=lin_thr, dhard_pca=dhard, dhard_lin=dhard_lin, dhard_t2only=dhard_t2only)

    # ---- how the flags relate: which statistic removes what; SPE vs LinRes agreement on difficult anomalies
    diff_idx = np.where(canon)[0]
    flag_t2 = t2_te[diff_idx] > thr_t2; flag_spe = spe_te[diff_idx] > thr_spe; flag_lin = r_te[diff_idx] > lin_thr
    rho_spe_lin = float(spearmanr(spe_te, r_te).correlation)
    rho_spe_lin_diff = float(spearmanr(spe_te[diff_idx], r_te[diff_idx]).correlation) if len(diff_idx) > 3 else None
    flags = dict(n_difficult=int(canon.sum()), flagged_T2=int(flag_t2.sum()), flagged_SPE=int(flag_spe.sum()),
                 flagged_T2_or_SPE=int((flag_t2 | flag_spe).sum()), flagged_LinRes=int(flag_lin.sum()),
                 SPE_and_LinRes=int((flag_spe & flag_lin).sum()), SPE_not_LinRes=int((flag_spe & ~flag_lin).sum()),
                 LinRes_not_SPE=int((flag_lin & ~flag_spe).sum()),
                 spearman_SPE_vs_LinRes_all_test=round(rho_spe_lin, 3),
                 spearman_SPE_vs_LinRes_difficult=round(rho_spe_lin_diff, 3) if rho_spe_lin_diff is not None else None,
                 test_normal_flag_rate_B=round(float(sepB[y == 0].mean()), 4),
                 test_normal_flag_rate_LinRes=round(float((r_te > lin_thr)[y == 0].mean()), 4))
    log(f"   difficult {flags['n_difficult']}: T2 flags {flags['flagged_T2']}, SPE flags {flags['flagged_SPE']}, "
        f"union {flags['flagged_T2_or_SPE']}, LinRes flags {flags['flagged_LinRes']} "
        f"(SPE&Lin {flags['SPE_and_LinRes']}, SPE-only {flags['SPE_not_LinRes']}, Lin-only {flags['LinRes_not_SPE']}); "
        f"rho(SPE,LinRes) all={rho_spe_lin:.3f} difficult={rho_spe_lin_diff}")

    # ---- sensitivity of the subset size to the variance cut
    sens = {}
    for v in VAR_SENS:
        (a_tr, b_tr), (a_te, b_te), kk, _, _ = pca_stats(Xtr, Xte, v)
        ta, tb = float(np.quantile(a_tr, 0.99)), float(np.quantile(b_tr, 0.99))
        dh = canon & (a_te <= ta) & (b_te <= tb)
        sens[str(v)] = dict(k=kk, n_double_hard=int(dh.sum()), n_episodes=len([e for e in episodes(y) if dh[e].any()]))
    log("   sensitivity: " + ", ".join(f"{v}: k={s['k']} n={s['n_double_hard']}/{s['n_episodes']}ep" for v, s in sens.items()))

    # ---- methods (same sources as the paper's Table 4 pipeline)
    methods = {"trivial max|z|": d["maxz"], "IF": d["IF"], "AE": d["AE"], "LinRes": d["linres"],
               "USAD": d["USAD"], "TranAD": d["TranAD"]}
    msf = f"{OUT}/scores_sota_ms_{name}.npz"
    if os.path.exists(msf):
        ms = np.load(msf, allow_pickle=True)
        assert np.array_equal(ms["label"].astype(int), y)
        for m in ("USAD", "TranAD"):
            if m in ms.files:
                methods[m] = ms[m]
    g = gdn_windows(name, Xa_raw, W, stride, len(y))
    if g is not None:
        methods["GDN"] = g
    # boosted_LOO: precomputed clean channel-wise boosted LOO residual (fair detector; PCA is the filter)
    bpath = f"{OUT}/boosted_loo_{name}.npz"
    if os.path.exists(bpath):
        b_te = np.load(bpath)["b_te"]
        assert len(b_te) == len(y), f"{name}: boosted_LOO length {len(b_te)} != y {len(y)}"
        methods["boosted_LOO"] = np.asarray(b_te, np.float64)
    else:
        log(f"   [warn] no boosted_loo_{name}.npz")
    methods["LatAD (global density)"] = d["LatAD"]
    ens, y2, _, nseed = EF.ensemble_scores(name)
    assert np.array_equal(y2, y)
    methods["LatAD (regime-community)"] = ens[HEADKEY]
    # sanity: the LinRes column in scores npz must be the same statistic as r_te (fair-detector re-score)
    rho_lin = float(spearmanr(d["linres"], r_te).correlation)
    log(f"   spearman(scores.linres, loco r_te) = {rho_lin:.3f}")

    def table(mask):
        keep = np.where((y == 0) | mask)[0]; rows = {}
        for m, arr in methods.items():
            if arr.ndim == 2:
                v = [roc_auc_score(y[keep], arr[i][keep]) for i in range(arr.shape[0])]
                rows[m] = dict(auroc=round(float(np.mean(v)), 3), sd=round(float(np.std(v)), 3),
                               per_seed=[round(float(a), 4) for a in v])
            else:
                rows[m] = dict(auroc=round(float(roc_auc_score(y[keep], arr[keep])), 3))
        return rows

    rows = table(dhard); rows_lin = table(dhard_lin); rows_t2 = table(dhard_t2only)
    eps_d = [e for e in episodes(y) if dhard[e].any()]
    eps_lin = [e for e in episodes(y) if dhard_lin[e].any()]
    eps_t2 = [e for e in episodes(y) if dhard_t2only[e].any()]
    log(f"   DOUBLE-HARD (PCA T2|SPE): {int(dhard.sum())} windows / {len(eps_d)} episodes   "
        f"[LinRes-based: {int(dhard_lin.sum())}/{len(eps_lin)};  T2-only: {int(dhard_t2only.sum())}/{len(eps_t2)}]")
    log(f"   {'method':26} {'PCA-dhard':>14} {'LinRes-dhard':>14} {'T2only-dhard':>14}")
    for m in rows:
        f = lambda r: f"{r['auroc']:.3f}" + (f"±{r['sd']:.3f}" if 'sd' in r else "      ")
        log(f"   {m:26} {f(rows[m]):>14} {f(rows_lin[m]):>14} {f(rows_t2[m]):>14}")

    # ---- significance: headline vs strongest EXTERNAL detector on the new subset (filters and our own
    #      global-density ablation excluded from the competitor pick; both reported separately)
    EXTERNAL = ["IF", "AE", "LinRes", "USAD", "TranAD", "GDN", "boosted_LOO"]
    HEADM = "LatAD (regime-community)"; L = int(np.ceil(W / stride)) + 1

    def sig_block(mask, rws):
        if mask.sum() < 3:
            return None, None, None, None
        ck_ = max([m for m in EXTERNAL if m in rws], key=lambda m: rws[m]["auroc"])
        s = boot(y, methods[HEADM], methods[ck_], mask, L, reps=REPS)
        s_lin = None if ck_ == "LinRes" else boot(y, methods[HEADM], methods["LinRes"], mask, L, reps=REPS)
        s_glob = boot(y, methods[HEADM], methods["LatAD (global density)"], mask, L, reps=REPS)
        return ck_, s, s_lin, s_glob

    ck, sig, sig_lin, sig_glob = sig_block(dhard, rows)
    if sig:
        log(f"   headline vs strongest external detector {ck}: diff {sig['diff']:+.3f} CI {sig['diff_ci']} "
            f"P(<=0)={sig['p_le_0']} ({sig['n_episodes']} episodes, {sig['n_boot']} boots)")
    if sig_lin:
        log(f"   headline vs LinRes: diff {sig_lin['diff']:+.3f} CI {sig_lin['diff_ci']} P(<=0)={sig_lin['p_le_0']}")
    if sig_glob:
        log(f"   headline vs LatAD global density: diff {sig_glob['diff']:+.3f} CI {sig_glob['diff_ci']} P(<=0)={sig_glob['p_le_0']}")

    # ---- 99%-variance variant (the cut at which SPE retains the small-eigenvalue directions): full table + significance
    (a_tr, b_tr), (a_te, b_te), k99, _, _ = pca_stats(Xtr, Xte, 0.99)
    dhard99 = canon & (a_te <= float(np.quantile(a_tr, 0.99))) & (b_te <= float(np.quantile(b_tr, 0.99)))
    rows99 = table(dhard99); eps99 = [e for e in episodes(y) if dhard99[e].any()]
    ck99, sig99, _, _ = sig_block(dhard99, rows99)
    log(f"   99%-variance variant (k={k99}): {int(dhard99.sum())} windows / {len(eps99)} episodes; "
        + ", ".join(f"{m}={r['auroc']:.3f}" for m, r in rows99.items()))
    if sig99:
        log(f"   99% variant: headline vs {ck99}: diff {sig99['diff']:+.3f} CI {sig99['diff_ci']} P(<=0)={sig99['p_le_0']}")

    ALL[name] = dict(
        d=int(d_full), n_train=int(len(Xtr)), n_test=int(len(y)), n_anom=int(y.sum()), n_difficult=int(canon.sum()),
        maxz_thr=round(mthr, 4), pca=dict(k95=k, var_keep=VAR_KEEP, thr_t2=round(thr_t2, 4), thr_spe=round(thr_spe, 4),
                                          train_flag_rate=round(float(tr_flag), 4), top_eig_frac=round(float(cum[0]), 4)),
        lin_thr=round(lin_thr, 4),
        subset=dict(pca=dict(n=int(dhard.sum()), episodes=len(eps_d)),
                    linres=dict(n=int(dhard_lin.sum()), episodes=len(eps_lin)),
                    t2only=dict(n=int(dhard_t2only.sum()), episodes=len(eps_t2)),
                    pca_subset_of_linres=int((dhard & dhard_lin).sum()),
                    pca_only=int((dhard & ~dhard_lin).sum()), linres_only=int((dhard_lin & ~dhard).sum())),
        flags=flags, sensitivity=sens, rows_pca=rows, rows_linres=rows_lin, rows_t2only=rows_t2,
        competitor=ck, significance=sig, significance_vs_linres=sig_lin, significance_vs_latad_global=sig_glob,
        pca99=dict(k=k99, n=int(dhard99.sum()), episodes=len(eps99), rows=rows99, competitor=ck99, significance=sig99),
        seconds=round(time.time() - t0, 1))
    json.dump(ALL, open(f"{OUT}/doublehard_pca_all3.json", "w"), indent=1)   # incremental persist
    log(f"   saved ({ALL[name]['seconds']}s)")

log(f"\nsaved -> {OUT}/doublehard_pca_all3.json")
