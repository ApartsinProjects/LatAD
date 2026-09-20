"""FIX B: robust re-centering of each community's TEST-time surprise before the p-value against the calibration
slice. Drift model: a sustained additive offset in a community's score level between train and test. Correction
(a-priori, no test labels, no tuning): shift each community's test scores so that their MEDIAN equals the median
of that community's calibration scores; the median is the standard robust location and is valid because
anomalies are a minority of test windows (SWaT_canon 15.6%, WADI_clean 9.7%, HAI 4.4%). Nothing else changes:
same experts, same HC_coh + LatAD null-term fusion, same weights.
Evaluated per experts artifact: Modal experts_full for all three datasets (FIX B alone) and the local SWaT
re-trainings (untrimmed cur_fit = FIX none; trimmed = FIX A; trimmed + re-centred = FIX A+B).
Masking check: per-window percentile of every ANOMALY window vs test normals before/after; detection rate at
percentile >= 0.95 (all anomalies, difficult) and per-episode drops. Output: drift_fix_eval.json"""
from __future__ import annotations
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from sklearn.metrics import roc_auc_score
import eda_real as E
from onehot_filter import build_feats, loco_residual
import ensemble_final as EF

HERE = os.path.dirname(os.path.abspath(__file__))
OUTJ = os.path.join(HERE, "drift_fix_eval.json"); ALL = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}


def ctx(name):
    d = np.load(f"{EF.OUT}/scores_{name}.npz"); y = d["label"].astype(int)
    D = E.load(name); fn, W, stride = E.RAW[name]
    Fn, Fa, grp = build_feats(np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float), W, stride, onehot=True); Fa = Fa[:len(y)]
    r_tr, r_te = loco_residual(Fn, Fa, grp)
    diff = (y == 1) & (d["maxz"] <= float(d["maxz_thr"])); dhard = diff & (r_te <= np.quantile(r_tr, 0.99))
    return d, y, diff, dhard, int(np.ceil(W / stride)) + 1


def fuse(Ex, d, recenter):
    Cal, Tst = Ex["calib_surprise"], Ex["test_surprise"]; nseed, S, _ = Tst.shape
    coh = Ex["comm_cohesion"].astype(float); size = Ex["comm_size"].astype(float); w = coh * np.sqrt(size)
    lat, lat_tr = d["LatAD"], d["LatAD_train"]; nseed = min(nseed, lat.shape[0]); z = lambda s, r: (s - r.mean()) / (r.std() + 1e-9)
    acc = []; shifts = np.zeros((nseed, S))
    for sd in range(nseed):
        T = Tst[sd].copy()
        if recenter:
            sh = np.median(T, 1) - np.median(Cal[sd], 1); T = T - sh[:, None]; shifts[sd] = sh
        P = np.stack([EF.pval(Cal[sd, g], T[g]) for g in range(S)]); Pc = np.stack([EF.pval(Cal[sd, g], Cal[sd, g]) for g in range(S)])
        zl = z(EF.surv(lat_tr[sd], lat[sd]), EF.surv(lat_tr[sd], lat_tr[sd]))
        acc.append(z(EF.HC(P, wt=w), EF.HC(Pc, wt=w)) + zl)
    return np.stack(acc), shifts


def au_ms(y, arr, m):
    k = (y == 0) | m
    return float(np.mean([roc_auc_score(y[k], arr[i][k]) for i in range(arr.shape[0])])) if arr.ndim == 2 else float(roc_auc_score(y[k], arr[k]))


def masking(y, before, after, diff):
    nrm = y == 0
    pr = lambda s: np.searchsorted(np.sort(s[nrm]), s, side="left") / nrm.sum()
    pb, pa = pr(before.mean(0)), pr(after.mean(0)); an = y == 1
    r = dict(det95_all_before=round(float((pb[an] >= 0.95).mean()), 3), det95_all_after=round(float((pa[an] >= 0.95).mean()), 3),
             det95_diff_before=round(float((pb[diff] >= 0.95).mean()), 3), det95_diff_after=round(float((pa[diff] >= 0.95).mean()), 3),
             n_anom_pct_drop_gt_0_1=int(((pb - pa)[an] > 0.1).sum()), n_anom_pct_gain_gt_0_1=int(((pa - pb)[an] > 0.1).sum()))
    eps = EF.episodes(y); drops = []
    for e in eps:
        b, a = float(np.median(pb[e])), float(np.median(pa[e]))
        if b - a > 0.1:
            drops.append(dict(start=int(e[0]), n=int(len(e)), before=round(b, 3), after=round(a, 3)))
    r["episodes_masked(median pct drop>0.1)"] = drops; r["n_episodes"] = len(eps)
    return r


jobs = [(a, b) for a in sys.argv[1:] for b in [None]] if len(sys.argv) > 1 else []
# (dataset, artifact dir, label)
if not jobs:
    jobs = [("SWaT_canon", "sota_bundle/experts_full"), ("WADI_clean", "sota_bundle/experts_full"), ("HAI", "sota_bundle/experts_full"),
            ("SWaT_canon", "sota_bundle/experts_variants/cur_fit"), ("SWaT_canon", "sota_bundle/experts_variants/trim6h")]
else:
    jobs = [(a.split(":")[0], a.split(":")[1]) for a in sys.argv[1:]]
for name, adir in jobs:
    p = os.path.join(ROOT, adir, f"expert_{name}.npz")
    if not os.path.exists(p):
        print("missing", p); continue
    Ex = np.load(p, allow_pickle=True); d, y, diff, dhard, L = ctx(name)
    base, _ = fuse(Ex, d, False); fixb, shifts = fuse(Ex, d, True)
    r = dict(artifact=adir, base=dict(Difficult=round(au_ms(y, base, diff), 3), DoubleHard=round(au_ms(y, base, dhard), 3)),
             fixB=dict(Difficult=round(au_ms(y, fixb, diff), 3), DoubleHard=round(au_ms(y, fixb, dhard), 3)),
             median_shift_abs_med=round(float(np.median(np.abs(shifts))), 3), median_shift_abs_max=round(float(np.abs(shifts).max()), 3),
             masking=masking(y, base, fixb, diff))
    EF.RNG = np.random.default_rng(0); r["boot_fixB_minus_base_Difficult"] = EF.boot(y, fixb, base, diff, L, reps=2000)
    EF.RNG = np.random.default_rng(0); r["boot_fixB_minus_base_DoubleHard"] = EF.boot(y, fixb, base, dhard, L, reps=2000)
    bp = os.path.join(HERE, f"boosted_loo_{name}.npz")
    if os.path.exists(bp):
        b_te = np.load(bp)["b_te"]
        EF.RNG = np.random.default_rng(0); r["boot_fixB_minus_boosted_Difficult"] = EF.boot(y, fixb, b_te, diff, L, reps=2000)
        EF.RNG = np.random.default_rng(0); r["boot_fixB_minus_boosted_DoubleHard"] = EF.boot(y, fixb, b_te, dhard, L, reps=2000)
        EF.RNG = np.random.default_rng(0); r["boot_base_minus_boosted_Difficult"] = EF.boot(y, base, b_te, diff, L, reps=2000)
    ALL[f"{name}@{adir}"] = r; json.dump(ALL, open(OUTJ, "w"), indent=1)
    print(f"\n=== {name} @ {adir} ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
