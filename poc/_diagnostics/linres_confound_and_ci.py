"""Two decision-relevant checks, read-only on cached data:
 (1) LinRes scaling confound: is linres's WADI advantage real discrete signal or an artifact of
     build_feats leaving one-hot fractions UNSCALED while numeric means are standardized? Recompute
     linres with ALL columns standardized and compare canonical difficult-AUROC.
 (2) Episode-block bootstrap CI for the pro-LatAD easy-filter result (LatAD - best deep baseline)
     on the '+T2+T3' harder difficult subset, HAI and SWaT.
"""
from __future__ import annotations
import os, sys, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eda_real as E
from onehot_filter import build_feats, loco_residual
HERE = os.path.dirname(os.path.abspath(__file__)); DMAX = 6


def canon_masks(name):
    d = dict(np.load(os.path.join(HERE, f"scores_{name}.npz")))
    y = d["label"].astype(int); triv = d["maxz"]; thr = float(d["maxz_thr"])
    hard = (y == 1) & ~((y == 1) & (triv > thr))
    return d, y, hard


def au(y, s, hard):
    k = (y == 0) | hard
    return float(roc_auc_score(y[k], s[k])) if 0 < y[k].sum() < k.sum() else float("nan")


def confound(name):
    d, y, hard = canon_masks(name); n = len(y)
    D = E.load(name); fn, W, stride = E.RAW[name]
    Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
    Fn, Fa, grp = build_feats(Xn, Xa, W, stride, onehot=True); Fa = Fa[:n]
    _, r_unscaled = loco_residual(Fn, Fa, grp)                       # current (one-hot unscaled)
    mu, sd = Fn.mean(0), Fn.std(0) + 1e-8
    _, r_scaled = loco_residual((Fn - mu) / sd, (Fa - mu) / sd, grp)  # all cols standardized
    # numeric-only (drop discrete channels entirely)
    keep = np.array([j for j in range(Fn.shape[1])
                     if len(np.unique(Xn[:, grp[j]])) > DMAX])
    if len(keep):
        _, r_numonly = loco_residual(Fn[:, keep], Fa[:, keep], grp[keep])
    else:
        r_numonly = np.full(n, np.nan)
    print(f"  {name}: linres difficult-AUROC  unscaled-onehot={au(y,r_unscaled,hard):.3f}  "
          f"all-standardized={au(y,r_scaled,hard):.3f}  numeric-only={au(y,r_numonly,hard):.3f}")


def episodes(y):
    out = []; i = 0
    while i < len(y):
        if y[i] == 1:
            j = i
            while j < len(y) and y[j] == 1: j += 1
            out.append(np.arange(i, j)); i = j
        else:
            i += 1
    return out


def ci_easyfilter(name, reps=2000):
    d, y, _ = canon_masks(name); n = len(y)
    D = E.load(name); fn, W, stride = E.RAW[name]
    Xtr0 = np.asarray(D["Xn_w"], np.float32); Xte0 = np.asarray(D["Xa_w"], np.float32)
    Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    T1 = (np.abs((Xte0 - mu) / sig).max(1) > np.quantile(np.abs((Xtr0 - mu) / sig).max(1), 0.99))
    p = PCA(min(Xtr0.shape[1], len(Xtr0) - 1), random_state=0).fit((Xtr0 - mu) / sig)
    Ptr = p.transform((Xtr0 - mu) / sig); Pte = p.transform((Xte0 - mu) / sig); ps = Ptr.std(0) + 1e-8
    T3 = (np.abs(Pte / ps).max(1) > np.quantile(np.abs(Ptr / ps).max(1), 0.99))
    Fn, Fa, grp = build_feats(Xn, Xa, W, stride, onehot=True); Fa = Fa[:n]
    dc = [j for j in range(Fn.shape[1]) if 1 < len(np.unique(Xn[:, grp[j]])) <= DMAX]
    if dc:
        Dn = Fn[:, dc]; dmu, dsd = Dn.mean(0), Dn.std(0) + 1e-8
        T2 = (np.abs((Fa[:, dc] - dmu) / dsd).max(1) > np.quantile(np.abs((Dn - dmu) / dsd).max(1), 0.99))
    else:
        T2 = np.zeros(n, bool)
    diff = (y == 1) & ~(T1 | T2 | T3)
    Lat = d["LatAD"].mean(0) if d["LatAD"].ndim > 1 else d["LatAD"]
    deep = {k: (d[k].mean(0) if d[k].ndim > 1 else d[k]) for k in ("AE", "USAD", "TranAD", "IF") if k in d}
    best = max(deep, key=lambda k: au(y, deep[k], diff))
    # episode-block bootstrap: resample anomaly episodes (within difficult) + normals
    eps = [e[diff[e]] for e in episodes(y) if diff[e].any()]
    normals = np.where(y == 0)[0]
    rng = np.random.default_rng(0); diffs = []
    for _ in range(reps):
        be = [eps[i] for i in rng.integers(0, len(eps), len(eps))]
        idx = np.concatenate(be + [rng.choice(normals, len(normals), replace=True)])
        yy = y[idx]
        if yy.sum() < 2: continue
        diffs.append(roc_auc_score(yy, Lat[idx]) - roc_auc_score(yy, deep[best][idx]))
    diffs = np.array(diffs)
    print(f"  {name}: +T2+T3 difficult n_anom={int(diff.sum())} episodes={len(eps)}  "
          f"LatAD={au(y,Lat,diff):.3f} vs best-deep({best})={au(y,deep[best],diff):.3f}  "
          f"Delta={diffs.mean():+.3f} CI[{np.quantile(diffs,.025):+.3f},{np.quantile(diffs,.975):+.3f}] "
          f"p(<=0)={ (diffs<=0).mean():.3f}")


if __name__ == "__main__":
    print("== (1) LinRes scaling confound (difficult-AUROC) ==")
    for nm in ("WADI_clean", "HAI", "SWaT"):
        try: confound(nm)
        except Exception as e: print(f"  {nm} ERR {type(e).__name__}: {e}")
    print("== (2) episode-block CI, LatAD - best-deep on +T2+T3 difficult ==")
    for nm in ("HAI", "SWaT", "WADI_clean"):
        try: ci_easyfilter(nm)
        except Exception as e: print(f"  {nm} ERR {type(e).__name__}: {e}")
