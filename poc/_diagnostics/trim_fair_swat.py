"""Trim-consistent SWaT comparison: EVERY retrained method fit on the SAME trimmed train-normal.
Trim = the first 21,600 s (6 h) of the SWaT_canon train recording as delivered by eda_real (which has already
dropped the loader's 2% warm-up, 27,741 s); at 10x downsampling that is 2,160 raw rows = exactly 72 leading
windows (stride 30), so window grids stay aligned with sota_bundle/experts_variants/trim6h (TRIM=72).
Retrained on trimmed train: LinRes, boosted channel-wise LOO (5-fold cross-fitted train reference), Isolation
Forest (5 seeds), AutoEncoder (5 seeds), global LatAD null (5 seeds, build_scores_table config), community
experts (trim6h). NOT retrained (excluded from the trim comparison): USAD, TranAD, GDN (Modal, full data).
Difficulty (maxz_thr) and double-hard (lin_thr) thresholds recomputed on trimmed train-normal. Test untouched.
Masking check: per-episode median percentile of the trimmed headline vs the untrimmed Modal headline.
Output: trim_fair_swat.json"""
from __future__ import annotations
import os, sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ["EXPERTS_DIR"] = os.path.join(ROOT, "sota_bundle", "experts_full")
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.model_selection import KFold
import eda_real as E
from onehot_filter import build_feats, loco_residual
from models_vade import train_vade
from compare_baselines import ae_scores
import ensemble_final as EF

HERE = os.path.dirname(os.path.abspath(__file__)); name = "SWaT_canon"; TRIM_ROWS, TRIM_WIN = 2160, 72; SEEDS = [0, 1, 2, 3, 4]
t0 = time.time()
d = np.load(f"{EF.OUT}/scores_{name}.npz"); y = d["label"].astype(int)
D = E.load(name); Du = E.load(name, clip=None); fn, W, stride = E.RAW[name]
Xn_raw, Xa_raw = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
Xn_w, Xa_w = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32)
B = np.load(os.path.join(ROOT, "sota_bundle", "ens_bundle", f"bundle_{name}.npz"))
assert B["Xn_w"].shape[0] == Xn_w.shape[0] and np.allclose(B["Xn_w"][:5], Xn_w[:5], atol=1e-4), "bundle/eda window grid mismatch"
Xn_raw_t, Xn_w_t = Xn_raw[TRIM_ROWS:], Xn_w[TRIM_WIN:]
res = dict(trim_rows=TRIM_ROWS, trim_windows=TRIM_WIN, n_train_windows_before=int(len(Xn_w)), n_train_windows_after=int(len(Xn_w_t)),
           n_test=int(len(y)), n_anom=int(y.sum()), n_episodes=len(EF.episodes(y)))
# ---- difficulty / double-hard on trimmed train-normal ----
Xtr0u, Xte0u = Du["Xn_w"].astype(np.float32)[TRIM_WIN:], Du["Xa_w"].astype(np.float32); C6 = Xte0u.shape[1] // 6
maxz = np.abs(Xte0u[:, :C6]).max(1); mthr_t = float(np.quantile(np.abs(Xtr0u[:, :C6]).max(1), 0.99))
Fn, Fa, grp = build_feats(Xn_raw_t, Xa_raw, W, stride, onehot=True); Fa = Fa[:len(y)]
r_tr, r_te = loco_residual(Fn, Fa, grp); lthr_t = float(np.quantile(r_tr, 0.99))
diff_t = (y == 1) & (maxz <= mthr_t); dhard_t = diff_t & (r_te <= lthr_t)
diff_0 = (y == 1) & (d["maxz"] <= float(d["maxz_thr"]))
res.update(maxz_thr_full=round(float(d["maxz_thr"]), 3), maxz_thr_trim=round(mthr_t, 3), n_diff_full=int(diff_0.sum()), n_diff_trim=int(diff_t.sum()),
           n_dhard_trim=int(dhard_t.sum()), n_diff_episodes_trim=len([e for e in EF.episodes(y) if diff_t[e].any()]))
print(res, flush=True)
# ---- baselines on trimmed train ----
meth = {"linres_trim": r_te}
Rte = np.zeros_like(Fa); Rtr = np.zeros_like(Fn); kf = KFold(5, shuffle=True, random_state=0)
for j in range(Fn.shape[1]):
    cols = np.where(grp != grp[j])[0]
    m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=0).fit(Fn[:, cols], Fn[:, j]); Rte[:, j] = (m.predict(Fa[:, cols]) - Fa[:, j]) ** 2
    for tr, va in kf.split(Fn):
        mm = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=0).fit(Fn[tr][:, cols], Fn[tr, j]); Rtr[va, j] = (mm.predict(Fn[va][:, cols]) - Fn[va, j]) ** 2
meth["boosted_trim"] = Rte.mean(1); b_tr = Rtr.mean(1)
print(f"boosted done {time.time()-t0:.0f}s", flush=True)
mu, sig = Xn_w_t.mean(0), Xn_w_t.std(0) + 1e-8
Xtr = ((Xn_w_t - mu) / sig).astype(np.float32); Xte = ((Xa_w - mu) / sig).astype(np.float32)
kd = min(80, max(20, len(Xtr) // 10)); lat, lat_tr, s_if, s_ae = [], [], [], []
for sd in SEEDS:
    v = train_vade(Xtr, n_clusters=40, latent_dim=16, epochs=40, warmup=8, seed=sd, device="cpu")
    v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd); v.fit_resid_head(Xtr); v.fit_basin_head(Xtr)
    lat.append(np.asarray(v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto"))); lat_tr.append(np.asarray(v.anomaly_score_hard(Xtr, use_resid="auto", use_basin="auto")))
    s_if.append(-IsolationForest(n_estimators=200, random_state=sd).fit(Xtr).score_samples(Xte)); s_ae.append(ae_scores(Xtr, Xte, seed=sd))
    print(f"  seed {sd} global LatAD/IF/AE done {time.time()-t0:.0f}s", flush=True)
lat, lat_tr = np.stack(lat), np.stack(lat_tr); meth["LatAD_global_trim"] = lat; meth["IF_trim"] = np.stack(s_if); meth["AE_trim"] = np.stack(s_ae)
# ---- community fusion with trimmed experts + trimmed global null ----
Ex = np.load(os.path.join(ROOT, "sota_bundle", "experts_variants", "trim6h", f"expert_{name}.npz"), allow_pickle=True)
Cal, Tst = Ex["calib_surprise"], Ex["test_surprise"]; S = Tst.shape[1]; coh = Ex["comm_cohesion"].astype(float); size = Ex["comm_size"].astype(float); w = coh * np.sqrt(size)
z = lambda s, r: (s - r.mean()) / (r.std() + 1e-9); acc = {"HCcoh+LatAD_trim": [], "null+HC_trim": [], "cohmax+LatAD_trim": []}; wn = w / (w.max() + 1e-9)
for sd in range(min(5, Tst.shape[0])):
    P = np.stack([EF.pval(Cal[sd, g], Tst[sd, g]) for g in range(S)]); Pc = np.stack([EF.pval(Cal[sd, g], Cal[sd, g]) for g in range(S)])
    tails = np.stack([EF.surv(Cal[sd, g], Tst[sd, g]) for g in range(S)]); tails_c = np.stack([EF.surv(Cal[sd, g], Cal[sd, g]) for g in range(S)])
    zl = z(EF.surv(lat_tr[sd], lat[sd]), EF.surv(lat_tr[sd], lat_tr[sd])); hc, hc_c = EF.HC(P), EF.HC(Pc)
    acc["HCcoh+LatAD_trim"].append(z(EF.HC(P, wt=w), EF.HC(Pc, wt=w)) + zl); acc["null+HC_trim"].append(np.maximum(z(hc, hc_c), zl))
    acc["cohmax+LatAD_trim"].append(z((wn[:, None] * tails).max(0), (wn[:, None] * tails_c).max(0)) + zl)
for k, v in acc.items():
    meth[k] = np.stack(v)
# untrimmed references (Modal experts + full global) for the masking check and context
ens0, _, _, _ = EF.ensemble_scores(name); meth["HCcoh+LatAD_full(ref)"] = ens0["HCcoh+LatAD"]; meth["boosted_full(ref)"] = np.load(os.path.join(HERE, "boosted_loo_SWaT_canon.npz"))["b_te"]


def au_ms(arr, m):
    k = (y == 0) | m
    return float(np.mean([roc_auc_score(y[k], arr[i][k]) for i in range(arr.shape[0])])) if arr.ndim == 2 else float(roc_auc_score(y[k], arr[k]))


res["auroc_trim_subsets"] = {k: dict(Difficult=round(au_ms(v, diff_t), 3), DoubleHard=round(au_ms(v, dhard_t), 3), All=round(au_ms(v, y == 1), 3)) for k, v in meth.items()}
res["auroc_full_subsets"] = {k: dict(Difficult=round(au_ms(v, diff_0), 3)) for k, v in meth.items()}
print("\nAUROC on trim-consistent subsets:")
for k, v in res["auroc_trim_subsets"].items():
    print(f"  {k:26s} diff {v['Difficult']:.3f}  dhard {v['DoubleHard']:.3f}  all {v['All']:.3f}   (diff on full-threshold subset {res['auroc_full_subsets'][k]['Difficult']:.3f})")
L = int(np.ceil(W / stride)) + 1; res["boot"] = {}
for a, b in [("HCcoh+LatAD_trim", "boosted_trim"), ("null+HC_trim", "boosted_trim"), ("HCcoh+LatAD_trim", "linres_trim"), ("HCcoh+LatAD_trim", "HCcoh+LatAD_full(ref)"), ("boosted_trim", "boosted_full(ref)")]:
    for sub, mk in (("Difficult", diff_t), ("DoubleHard", dhard_t)):
        EF.RNG = np.random.default_rng(0); r = EF.boot(y, meth[a], meth[b], mk, L, reps=2000); res["boot"][f"{a} - {b} [{sub}]"] = r
        print(f"  boot {a} - {b} [{sub}]: {r}")
# masking check: trimmed headline vs untrimmed headline, per anomaly window / episode
nrm = y == 0; pr = lambda s: np.searchsorted(np.sort(s[nrm]), s, side="left") / nrm.sum()
pb, pa = pr(meth["HCcoh+LatAD_full(ref)"].mean(0)), pr(meth["HCcoh+LatAD_trim"].mean(0)); an = y == 1
mk = dict(det95_all_before=round(float((pb[an] >= 0.95).mean()), 3), det95_all_after=round(float((pa[an] >= 0.95).mean()), 3),
          n_anom_pct_drop_gt_0_1=int(((pb - pa)[an] > 0.1).sum()), n_anom_pct_gain_gt_0_1=int(((pa - pb)[an] > 0.1).sum()),
          episodes_masked=[dict(start=int(e[0]), n=int(len(e)), before=round(float(np.median(pb[e])), 3), after=round(float(np.median(pa[e])), 3)) for e in EF.episodes(y) if np.median(pb[e]) - np.median(pa[e]) > 0.1])
res["masking_trim_vs_full_headline"] = mk; print("masking:", mk)
res["not_retrained_excluded"] = ["USAD", "TranAD", "GDN"]; res["elapsed_s"] = round(time.time() - t0, 1)
json.dump(res, open(os.path.join(HERE, "trim_fair_swat.json"), "w"), indent=1); print("saved", flush=True)
