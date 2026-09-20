"""Boosted channel-wise LOO baseline (Sarfraz/Garg-style) on CLEAN SWaT_canon.
Slim recompute of the difficult/doublehard AUROC only (no 2000-rep bootstraps); reuses the
verbatim boosted_loco from loo_boosted_swat.py. Reports linres (invariant I1: reproduces
npz linres) and the community fusion (HCcoh+LatAD) on the SAME clean difficult subset for context.

Invariants:
  I1  linear LOO residual reproduces scores_SWaT_canon.npz['linres'] (rel-err < 1e-4).
  I2  difficult subset == 91 windows.
Old-leaked boosted difficult AUROC was 0.881 (loo_boosted_swat.json).
"""
from __future__ import annotations
import os, sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("EXPERTS_DIR", os.path.join(ROOT, "sota_bundle", "experts_full"))
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
import eda_real as E
from onehot_filter import build_feats, loco_residual
import ensemble_final as EF

HERE = os.path.dirname(os.path.abspath(__file__)); NAME = "SWaT_canon"


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k]))


def au_ms(y, arr, m):
    k = (y == 0) | m
    return float(np.mean([roc_auc_score(y[k], arr[i][k]) for i in range(arr.shape[0])]))


def hgb():
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=0)


def boosted_loco(Fn, Fa, grp):
    Rte = np.zeros((len(Fa), Fn.shape[1]))
    for j in range(Fn.shape[1]):
        cols = np.where(grp != grp[j])[0]
        m = hgb().fit(Fn[:, cols], Fn[:, j])
        Rte[:, j] = (m.predict(Fa[:, cols]) - Fa[:, j]) ** 2
    return Rte


t0 = time.time()
ens, y, d, nseed = EF.ensemble_scores(NAME)
fn, W, stride = E.RAW[NAME]; D = E.load(NAME)
Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
maxz, mthr = d["maxz"], float(d["maxz_thr"])
Fn, Fa, grp = build_feats(Xn, Xa, W, stride, onehot=True); Fa = Fa[:len(y)]
r_tr, r_te = loco_residual(Fn, Fa, grp)
diff = (y == 1) & (maxz <= mthr)
lin_thr = float(np.quantile(r_tr, 0.99)); dhard = diff & (r_te <= lin_thr)
I1 = float(np.max(np.abs(r_te - d["linres"]) / (np.abs(d["linres"]) + 1e-6)))
assert int(diff.sum()) == 91, f"I2 FAILED: difficult={int(diff.sum())}"
print(f"subsets: diff {int(diff.sum())} dhard {int(dhard.sum())}  I1(linres rel-err)={I1:.1e}", flush=True)

Rte = boosted_loco(Fn, Fa, grp); sh = Rte.mean(1)
res = {"name": NAME, "n_diff": int(diff.sum()), "n_dhard": int(dhard.sum()), "I1_linres_relerr": I1,
       "boosted_diff": au(y, sh, diff), "boosted_dhard": au(y, sh, dhard), "boosted_all": au(y, sh, y == 1),
       "linres_diff": au(y, r_te, diff), "linres_dhard": au(y, r_te, dhard),
       "HCcoh+LatAD_diff": au_ms(y, ens["HCcoh+LatAD"], diff), "HCcoh+LatAD_dhard": au_ms(y, ens["HCcoh+LatAD"], dhard),
       "LatAD_global_diff": au(y, d["LatAD"], diff) if d["LatAD"].ndim == 1 else au_ms(y, d["LatAD"], diff),
       "elapsed_s": round(time.time() - t0, 1)}
json.dump(res, open(os.path.join(HERE, "clean_swat_boosted_loo.json"), "w"), indent=1)
print(json.dumps(res, indent=1), flush=True)
print(f"OLD-leaked boosted difficult was 0.881; NEW-clean = {res['boosted_diff']:.3f}", flush=True)
