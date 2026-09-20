"""Verify the kNN-in-latent lead (from the rare-regime exploration): does scoring a
window by its distance to the k nearest TRAIN-normal latent points beat / help the
parametric density head on the difficult subset? 5 seeds x 3 datasets, construct-matched
to E5 (same ens_bundle, same difficulty split: easy=maxz>thr, difficult=anom&~easy,
per-seed AUROC then mean).

Methods (difficult-subset AUROC):
  dens   = cross-channel latent density head only (E5 arm-b winner; use_near/resid/basin off)
  full   = the reported LatAD head (use_resid='auto', use_basin='auto')
  knn{k} = mean Euclidean distance to the k nearest train-normal latent points
  fuse   = zscore(dens) + zscore(knn10)   (both standardized against train)

Local CPU. Writes _diagnostics/e2_knn_latent.json. Does NOT touch the paper or models_vade.
"""
from __future__ import annotations
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from models_vade import train_vade

BUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sota_bundle", "ens_bundle")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics", "e2_knn_latent.json")
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}
SEEDS = [0, 1, 2, 3, 4]
KS = [5, 10, 20]


def au(y, s, keep):
    yy = y[keep]
    if yy.sum() < 2 or (yy == 0).sum() < 2:
        return float("nan")
    return float(roc_auc_score(yy, s[keep]))


def zt(a, ref):
    return (a - ref.mean()) / (ref.std() + 1e-9)


def run(name):
    B = np.load(f"{BUN}/bundle_{name}.npz")
    Xn = B["Xn_w"].astype(np.float32); Xa = B["Xa_w"].astype(np.float32)
    y = B["y"].astype(int); thr = float(B["maxz_thr"]); maxz = B["maxz"]
    easy = (y == 1) & (maxz > thr); hard = (y == 1) & ~easy
    keep = np.where((y == 0) | hard)[0]
    mu, sig = Xn.mean(0), Xn.std(0) + 1e-8
    Zn = ((Xn - mu) / sig).astype(np.float32); Za = ((Xa - mu) / sig).astype(np.float32)
    K, LD = CFG[name]
    kd = min(80, max(20, len(Zn) // 10))
    rows = []
    for seed in SEEDS:
        v = train_vade(Zn, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
        v.fit_residual_whitener(Zn); v.fit_latent_density(Zn, k_density=kd, seed=seed)
        v.fit_resid_head(Zn, seed=seed); v.fit_basin_head(Zn, seed=seed)
        Ltr = v._encode_mean(Zn); Lte = v._encode_mean(Za)
        dens = np.asarray(v.anomaly_score_hard(Za, use_near=False, use_resid=False, use_basin=False))
        full = np.asarray(v.anomaly_score_hard(Za, use_resid="auto", use_basin="auto"))
        res = {"dens": au(y, dens, keep), "full": au(y, full, keep)}
        knn10 = None
        for k in KS:
            nn = NearestNeighbors(n_neighbors=k).fit(Ltr)
            d_te, _ = nn.kneighbors(Lte); s = d_te.mean(1)
            res[f"knn{k}"] = au(y, s, keep)
            if k == 10:
                knn10 = s
        # fuse density + knn10, both standardized on train
        d_tr, _ = NearestNeighbors(n_neighbors=10).fit(Ltr).kneighbors(Ltr)
        knn_tr = d_tr.mean(1); dens_tr = np.asarray(v.anomaly_score_hard(Zn, use_near=False, use_resid=False, use_basin=False))
        fuse = zt(dens, dens_tr) + zt(knn10, knn_tr)
        res["fuse"] = au(y, fuse, keep)
        res.update(dataset=name, seed=seed, n_diff=int(hard.sum()))
        rows.append(res)
        print(f"[{name} s{seed}] dens={res['dens']:.3f} full={res['full']:.3f} "
              f"knn10={res['knn10']:.3f} fuse={res['fuse']:.3f}", flush=True)
    return rows


all_rows = []
for nm in ["WADI", "HAI", "SWaT"]:
    all_rows += run(nm)
json.dump(all_rows, open(OUT, "w"), indent=1)

print("\n==== difficult-subset AUROC, 5-seed mean +/- sd ====")
methods = ["dens", "full"] + [f"knn{k}" for k in KS] + ["fuse"]
print(f"{'dataset':6} " + " ".join(f"{m:>12}" for m in methods))
for nm in ["WADI", "HAI", "SWaT"]:
    r = [x for x in all_rows if x["dataset"] == nm]
    cells = []
    for m in methods:
        v = [x[m] for x in r]
        cells.append(f"{np.mean(v):.3f}±{np.std(v):.3f}")
    print(f"{nm:6} " + " ".join(f"{c:>12}" for c in cells))
print("\n---- kNN10 vs full-head, kNN10 vs dens (mean delta) ----")
for nm in ["WADI", "HAI", "SWaT"]:
    r = [x for x in all_rows if x["dataset"] == nm]
    dk = np.mean([x["knn10"] for x in r]); df = np.mean([x["full"] for x in r]); dd = np.mean([x["dens"] for x in r])
    fu = np.mean([x["fuse"] for x in r])
    print(f"{nm:6} knn10-full={dk-df:+.3f}  knn10-dens={dk-dd:+.3f}  fuse-full={fu-df:+.3f}  "
          f"WIN(knn>full)={dk>df}  WIN(fuse>full)={fu>df}")
print(f"\nsaved -> {OUT}")
