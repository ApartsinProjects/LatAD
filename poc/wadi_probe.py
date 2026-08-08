"""Probe simple hand-crafted signals on the 11 range-hard WADI windows (frozen/held sensors,
subtle multi-channel low shift). If any separates them from normal, that is the signal to build in.
No VaDE -- direct AUROC of each candidate detector on the range-hard subset AND the full anomaly set.
"""
from __future__ import annotations
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
import eda_real as E

D = E.load("WADI"); fn, W, stride = E.RAW["WADI"]
Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
y = D["ya_w"].astype(int)
starts = list(range(0, len(Xa) - W + 1, stride))[:len(y)]
nstarts = list(range(0, len(Xn) - W + 1, stride))
mu, sd = Xn.mean(0), Xn.std(0) + 1e-8
lo, hi = Xn.min(0), Xn.max(0)
oor = np.array([bool(((Xa[i:i + W] < lo) | (Xa[i:i + W] > hi)).any()) for i in starts])
hard = (y == 1) & ~oor
# normal per-channel within-window velocity (median over normal windows) as the 'alive' reference
nvel = np.median(np.stack([np.abs(np.diff(Xn[i:i + W], 0)).mean(0) for i in nstarts]), 0) + 1e-9


def feats(Xsrc, st):
    F = {"l2": [], "modcount": [], "frozen": [], "pin": [], "lowshift": []}
    for i in st:
        w = Xsrc[i:i + W]; z = (w.mean(0) - mu) / sd
        vel = np.abs(np.diff(w, 0)).mean(0)
        F["l2"].append(np.sqrt((z ** 2).mean()))                    # multi-channel magnitude
        F["modcount"].append((np.abs(z) > 1.5).sum())               # # channels moderately off
        F["frozen"].append((vel < 0.2 * nvel).sum())               # # channels velocity-collapsed
        F["pin"].append(((w.min(0) <= lo + 1e-6) | (w.max(0) >= hi - 1e-6)).sum())  # # at envelope edge
        F["lowshift"].append((z < -1.5).sum())                      # # channels held LOW
    return {k: np.array(v, float) for k, v in F.items()}


Fa = feats(Xa, starts); Fn = feats(Xn, nstarts)
yfull = np.r_[np.zeros(len(nstarts), int), y]


def au(yy, s, m):
    k = (yy == 0) | m
    return roc_auc_score(yy[k], s[k]) if m.sum() >= 3 else float("nan")


print(f"range-hard n={int(hard.sum())}")
print(f"  {'signal':10} {'AUROC@range-hard':>17} {'AUROC@all-anom':>15}")
for k in Fa:
    s_full = np.r_[Fn[k], Fa[k]]
    print(f"  {k:10} {au(y, Fa[k], hard):>17.3f} {au(yfull, s_full, yfull == 1):>15.3f}")
# combine the two most promising by rank-sum
combo_hard = None
