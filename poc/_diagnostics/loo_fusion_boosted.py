"""Option (c): add the boosted channel-wise LOO residual as an additional GLOBAL expert to the LatAD fusion,
calibrated on TRAIN-normal only, combined by the SAME fixed rule as the LatAD null term.

Rule (ensemble_final): HCcoh+LatAD = z(HC_coh | calib slice) + zl, with zl = z(surv(lat_tr, lat) | surv(lat_tr, lat_tr)).
New term: zb = z(surv(b_tr, b_te) | surv(b_tr, b_tr)), b_tr = 5-fold CROSS-FITTED train-normal boosted residuals
(held-out, so the reference is not the optimistic in-sample residual), b_te = test residuals from the full-train fit.
Augmented (z-sum family): X + zb for X in {HCcoh+LatAD, cohmax+LatAD}; global: zl + zb.
Augmented (max family):   max(null+HC, zb).
No weight, gate or threshold; the boosted expert is deterministic (HGB random_state=0), the fusion stays 5-seed.

Exploratory train-only gate: chronological 80/20 split of the TRAIN recording; fit the boosted LOO on the first
80%, FP rate of the last 20% at the fit-slice in-sample p99 and median ratio hold/fit. Label-free, test-free.

Per-dataset boosted scores are cached in _diagnostics/boosted_loo_<ds>.npz (resumable). Results ->
_diagnostics/loo_fusion_boosted.json. Invariants: I1 linear LOO == npz linres; I2 current headlines reproduce
headline_clean_results.md / paper; I3 boosted SWaT difficult == 0.881 (loo_boosted_swat)."""
from __future__ import annotations
import os, sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("EXPERTS_DIR", os.path.join(ROOT, "sota_bundle", "experts_full"))
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import eda_real as E
from onehot_filter import build_feats, loco_residual
import ensemble_final as EF

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = int(os.environ.get("BOOT_REPS", "2000"))


def hgb():
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=0)


def boosted_scores(name, Fn, Fa, grp):
    cache = os.path.join(HERE, f"boosted_loo_{name}.npz")
    if os.path.exists(cache):
        z = np.load(cache); return z["b_tr_cv"], z["b_te"], float(z["gate_fp"]), float(z["gate_ratio"])
    t0 = time.time(); nf = Fn.shape[1]
    Rte = np.zeros((len(Fa), nf)); Rtr = np.zeros_like(Fn)
    kf = KFold(5, shuffle=True, random_state=0)
    cut = int(0.8 * len(Fn)); Rfit = np.zeros((cut, nf)); Rhold = np.zeros((len(Fn) - cut, nf))
    for j in range(nf):
        cols = np.where(grp != grp[j])[0]
        m = hgb().fit(Fn[:, cols], Fn[:, j]); Rte[:, j] = (m.predict(Fa[:, cols]) - Fa[:, j]) ** 2
        for tr, va in kf.split(Fn):
            mm = hgb().fit(Fn[tr][:, cols], Fn[tr, j]); Rtr[va, j] = (mm.predict(Fn[va][:, cols]) - Fn[va, j]) ** 2
        mc = hgb().fit(Fn[:cut, cols], Fn[:cut, j])
        Rfit[:, j] = (mc.predict(Fn[:cut, cols]) - Fn[:cut, j]) ** 2
        Rhold[:, j] = (mc.predict(Fn[cut:, cols]) - Fn[cut:, j]) ** 2
        if j % 10 == 0:
            print(f"    {name} feature {j}/{nf} ({time.time()-t0:.0f}s)", flush=True)
    b_tr_cv, b_te = Rtr.mean(1), Rte.mean(1)
    sfit, shold = Rfit.mean(1), Rhold.mean(1)
    gate_fp = float((shold > np.quantile(sfit, 0.99)).mean()); gate_ratio = float(np.median(shold) / np.median(sfit))
    np.savez(cache, b_tr_cv=b_tr_cv, b_te=b_te, gate_fp=gate_fp, gate_ratio=gate_ratio, cut=cut)
    return b_tr_cv, b_te, gate_fp, gate_ratio


def au_ms(y, arr, m):
    k = (y == 0) | m
    if arr.ndim == 1:
        return float(roc_auc_score(y[k], arr[k]))
    return float(np.mean([roc_auc_score(y[k], arr[i][k]) for i in range(arr.shape[0])]))


def bt(y, a, b, m, L):
    EF.RNG = np.random.default_rng(0)
    return EF.boot(y, a, b, m, L, reps=REPS)


ALL = {}
OUTJ = os.path.join(HERE, "loo_fusion_boosted.json")
if os.path.exists(OUTJ):
    ALL = json.load(open(OUTJ))
for name in (sys.argv[1:] or ["SWaT_canon", "WADI_clean", "HAI"]):
    t0 = time.time()
    ens, y, d, nseed = EF.ensemble_scores(name)
    fn, W, stride = E.RAW[name]; D = E.load(name)
    Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
    Fn, Fa, grp = build_feats(Xn, Xa, W, stride, onehot=True); Fa = Fa[:len(y)]
    r_tr, r_te = loco_residual(Fn, Fa, grp); lin_thr = float(np.quantile(r_tr, 0.99))
    maxz, mthr = d["maxz"], float(d["maxz_thr"])
    diff = (y == 1) & (maxz <= mthr); dhard = diff & (r_te <= lin_thr); nrm = y == 0
    res = dict(n_diff=int(diff.sum()), n_dhard=int(dhard.sum()),
               I1_linres_relerr=float(np.max(np.abs(r_te - d["linres"]) / (np.abs(d["linres"]) + 1e-6))))
    b_tr, b_te, gate_fp, gate_ratio = boosted_scores(name, Fn, Fa, grp)
    res["gate_train_chrono_holdout_FP_at_fit_p99"] = gate_fp; res["gate_train_chrono_holdout_median_ratio"] = gate_ratio
    res["boosted_testnormal_FP_at_cv_train_p99"] = float((b_te[nrm] > np.quantile(b_tr, 0.99)).mean())
    # ---- the new global expert, same transform as the LatAD null term ----
    z = lambda s, r: (s - r.mean()) / (r.std() + 1e-9)
    zb = z(EF.surv(b_tr, b_te), EF.surv(b_tr, b_tr))                      # (n,)
    lat, lat_tr = d["LatAD"], d["LatAD_train"]
    zl = np.stack([z(EF.surv(lat_tr[s], lat[s]), EF.surv(lat_tr[s], lat_tr[s])) for s in range(nseed)])
    meth = {"boosted_LOO": b_te, "linres": r_te, "LatAD_global": lat,
            "HCcoh+LatAD": ens["HCcoh+LatAD"], "null+HC": ens["null+HC"], "cohmax+LatAD": ens["cohmax+LatAD"],
            "AUG HCcoh+LatAD+B": ens["HCcoh+LatAD"] + zb[None, :],
            "AUG cohmax+LatAD+B": ens["cohmax+LatAD"] + zb[None, :],
            "AUG null+HC+B (max)": np.maximum(ens["null+HC"], zb[None, :]),
            "AUG LatAD+B (global)": zl + zb[None, :]}
    res["auroc"] = {k: dict(Difficult=round(au_ms(y, v, diff), 3), DoubleHard=round(au_ms(y, v, dhard), 3),
                            All=round(au_ms(y, v, y == 1), 3)) for k, v in meth.items()}
    print(f"\n=== {name} (diff {diff.sum()}, dhard {dhard.sum()}; gate FP {gate_fp:.3f}, ratio {gate_ratio:.2f}; "
          f"I1 {res['I1_linres_relerr']:.1e}) ===", flush=True)
    for k, v in res["auroc"].items():
        print(f"  {k:24s} diff {v['Difficult']:.3f}  dhard {v['DoubleHard']:.3f}  all {v['All']:.3f}", flush=True)
    # ---- bootstraps ----
    L = int(np.ceil(W / stride)) + 1
    pairs = [("AUG HCcoh+LatAD+B", "boosted_LOO"), ("AUG HCcoh+LatAD+B", "HCcoh+LatAD"), ("AUG HCcoh+LatAD+B", "linres"),
             ("AUG null+HC+B (max)", "boosted_LOO"), ("AUG null+HC+B (max)", "null+HC"),
             ("HCcoh+LatAD", "AUG HCcoh+LatAD+B")]
    res["boot"] = {}
    for a, b in pairs:
        for sub, mk in (("Difficult", diff), ("DoubleHard", dhard)):
            r = bt(y, meth[a], meth[b], mk, L); res["boot"][f"{a} - {b} [{sub}]"] = r
            print(f"  boot {a} - {b} [{sub}]: diff {r['diff']} CI {r['diff_ci']} P(<=0)={r['p_le_0']} eps {r['n_episodes']}", flush=True)
    res["elapsed_s"] = round(time.time() - t0, 1)
    ALL[name] = res
    json.dump(ALL, open(OUTJ, "w"), indent=1)
print("saved", flush=True)
