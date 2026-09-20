"""Abrupt-residual (detrended) scoring of the community nulls. For each community and seed: causal slow trend
= median of the previous K windows (initialised at the calibration-slice median = train-normal level); fast
residual = score - trend; p-value of the residual against the calibration slice's own residuals (same
procedure); HC_coh + LatAD null term unchanged. A-priori timescale 24 h (drift is between recordings, days;
the longest SWaT attack, #28, is 9.5 h and must be shorter than the trend window); sensitivity 6 h / 12 h / 48 h.
Window durations: SWaT_canon and WADI_clean 300 s (10x downsampled, stride 30), HAI 30 s (1 Hz, stride 30).
Masking check per attack episode (median percentile vs test normals before/after), long episodes flagged.
Output: detrend_eval.json"""
from __future__ import annotations
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ["EXPERTS_DIR"] = os.path.join(ROOT, "sota_bundle", "experts_full")
from sklearn.metrics import roc_auc_score
import eda_real as E
from onehot_filter import build_feats, loco_residual
import ensemble_final as EF

HERE = os.path.dirname(os.path.abspath(__file__))
WIN_S = {"SWaT_canon": 300, "WADI_clean": 300, "HAI": 30}
HOURS = [6, 12, 24, 48]; APRIORI = 24


def causal_trend(x, K, init):
    n = len(x); tr = np.empty(n); tr[0] = init
    for t in range(1, n):
        lo = max(0, t - K); tr[t] = np.median(x[lo:t]) if t - lo >= max(3, K // 4) else init
    return tr


def fuse(Ex, d, K):
    Cal, Tst = Ex["calib_surprise"], Ex["test_surprise"]; nseed, S, _ = Tst.shape
    coh = Ex["comm_cohesion"].astype(float); size = Ex["comm_size"].astype(float); w = coh * np.sqrt(size)
    lat, lat_tr = d["LatAD"], d["LatAD_train"]; nseed = min(nseed, lat.shape[0]); z = lambda s, r: (s - r.mean()) / (r.std() + 1e-9)
    acc = []
    for sd in range(nseed):
        if K is None:
            C, T = Cal[sd], Tst[sd]
        else:
            C = np.stack([Cal[sd, g] - causal_trend(Cal[sd, g], K, np.median(Cal[sd, g])) for g in range(S)])
            T = np.stack([Tst[sd, g] - causal_trend(Tst[sd, g], K, np.median(Cal[sd, g])) for g in range(S)])
        P = np.stack([EF.pval(C[g], T[g]) for g in range(S)]); Pc = np.stack([EF.pval(C[g], C[g]) for g in range(S)])
        zl = z(EF.surv(lat_tr[sd], lat[sd]), EF.surv(lat_tr[sd], lat_tr[sd]))
        acc.append(z(EF.HC(P, wt=w), EF.HC(Pc, wt=w)) + zl)
    return np.stack(acc)


def au_ms(y, arr, m):
    k = (y == 0) | m
    return float(np.mean([roc_auc_score(y[k], arr[i][k]) for i in range(arr.shape[0])])) if arr.ndim == 2 else float(roc_auc_score(y[k], arr[k]))


OUT = {}
for name in ["SWaT_canon", "WADI_clean", "HAI"]:
    d = np.load(f"{EF.OUT}/scores_{name}.npz"); y = d["label"].astype(int)
    D = E.load(name); fn, W, stride = E.RAW[name]
    Fn, Fa, grp = build_feats(np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float), W, stride, onehot=True); Fa = Fa[:len(y)]
    r_tr, r_te = loco_residual(Fn, Fa, grp); diff = (y == 1) & (d["maxz"] <= float(d["maxz_thr"])); dhard = diff & (r_te <= np.quantile(r_tr, 0.99)); nrm = y == 0
    Ex = np.load(f"{os.environ['EXPERTS_DIR']}/expert_{name}.npz", allow_pickle=True)
    base = fuse(Ex, d, None); res = dict(base=dict(Difficult=round(au_ms(y, base, diff), 3), DoubleHard=round(au_ms(y, base, dhard), 3)), hours={})
    L = int(np.ceil(W / stride)) + 1; pr = lambda s: np.searchsorted(np.sort(s[nrm]), s, side="left") / nrm.sum(); pb = pr(base.mean(0)); an = y == 1
    bp = os.path.join(HERE, f"boosted_loo_{name}.npz"); b_te = np.load(bp)["b_te"] if os.path.exists(bp) else None
    print(f"\n=== {name}: base diff {res['base']['Difficult']:.3f} dhard {res['base']['DoubleHard']:.3f} ===", flush=True)
    for h in HOURS:
        K = int(h * 3600 / WIN_S[name]); f = fuse(Ex, d, K); pa = pr(f.mean(0))
        eps = EF.episodes(y)
        masked = [dict(start=int(e[0]), n=int(len(e)), hours=round(len(e) * WIN_S[name] / 3600, 2), before=round(float(np.median(pb[e])), 3), after=round(float(np.median(pa[e])), 3))
                  for e in eps if np.median(pb[e]) - np.median(pa[e]) > 0.1]
        r = dict(K=K, Difficult=round(au_ms(y, f, diff), 3), DoubleHard=round(au_ms(y, f, dhard), 3),
                 det95_all_before=round(float((pb[an] >= 0.95).mean()), 3), det95_all_after=round(float((pa[an] >= 0.95).mean()), 3),
                 det95_diff_before=round(float((pb[diff] >= 0.95).mean()), 3), det95_diff_after=round(float((pa[diff] >= 0.95).mean()), 3),
                 n_anom_pct_drop_gt_0_1=int(((pb - pa)[an] > 0.1).sum()), n_anom_pct_gain_gt_0_1=int(((pa - pb)[an] > 0.1).sum()), episodes_masked=masked,
                 longest_episode=dict(n=int(max(len(e) for e in eps)), hours=round(max(len(e) for e in eps) * WIN_S[name] / 3600, 2)))
        if h == APRIORI:
            EF.RNG = np.random.default_rng(0); r["boot_detrended_minus_base_Difficult"] = EF.boot(y, f, base, diff, L, reps=2000)
            EF.RNG = np.random.default_rng(0); r["boot_detrended_minus_base_DoubleHard"] = EF.boot(y, f, base, dhard, L, reps=2000)
            if b_te is not None:
                EF.RNG = np.random.default_rng(0); r["boot_detrended_minus_boosted_Difficult"] = EF.boot(y, f, b_te, diff, L, reps=2000)
                EF.RNG = np.random.default_rng(0); r["boot_detrended_minus_boosted_DoubleHard"] = EF.boot(y, f, b_te, dhard, L, reps=2000)
        res["hours"][h] = r
        print(f"  {h:2d} h (K={K}): diff {r['Difficult']:.3f} dhard {r['DoubleHard']:.3f}  det95 all {r['det95_all_before']:.3f}->{r['det95_all_after']:.3f}  diff {r['det95_diff_before']:.3f}->{r['det95_diff_after']:.3f}  drops>0.1: {r['n_anom_pct_drop_gt_0_1']} gains: {r['n_anom_pct_gain_gt_0_1']}  masked episodes: {masked}", flush=True)
        for k in ("boot_detrended_minus_base_Difficult", "boot_detrended_minus_boosted_Difficult", "boot_detrended_minus_boosted_DoubleHard"):
            if k in r:
                print(f"      {k}: {r[k]}", flush=True)
    OUT[name] = res; json.dump(OUT, open(os.path.join(HERE, "detrend_eval.json"), "w"), indent=1)
print("saved", flush=True)
