"""Do the range-hard WADI windows form temporal RUNS, and does integrating the anomaly score over
time separate them? Rationale: normal WADI noise is TRANSIENT (drift/spikes that come and go), so a
trailing average suppresses it; a SUSTAINED weak attack accumulates. Uses LatAD per-window scores
from the table (multiseed mean) -- no retraining. Also tests latent-trajectory 'jump' (score change).
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
lo, hi = Xn.min(0), Xn.max(0)
oor = np.array([bool(((Xa[i:i + W] < lo) | (Xa[i:i + W] > hi)).any()) for i in starts])
hard = (y == 1) & ~oor

d = np.load("_diagnostics/scores_WADI.npz")
assert len(d["label"]) == len(y)
s = d["LatAD"].mean(0)                                   # multiseed-mean per-window anomaly score


def au(sc, m):
    k = (y == 0) | m
    return roc_auc_score(y[k], sc[k]) if m.sum() >= 3 else float("nan")


# runs of range-hard windows
idx = np.where(hard)[0]
runs = []
if len(idx):
    st = idx[0]; prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((st, prev)); st = i; prev = i
    runs.append((st, prev))
print(f"range-hard n={len(idx)} indices={idx.tolist()}")
print(f"runs (contiguous): {[(a, b, b-a+1) for a, b in runs]}")


def trailing_mean(sc, P):
    return np.array([sc[max(0, i - P + 1):i + 1].mean() for i in range(len(sc))])


print(f"\n  {'transform':18} {'AUROC@range-hard':>17} {'AUROC@all-anom':>15}")
print(f"  {'raw LatAD':18} {au(s, hard):>17.3f} {au(s, y == 1):>15.3f}")
for P in (2, 4, 8, 16, 32):
    si = trailing_mean(s, P)
    print(f"  {'trailing-mean P=' + str(P):18} {au(si, hard):>17.3f} {au(si, y == 1):>15.3f}")
# trajectory: absolute jump in score between consecutive windows (entering/leaving a state)
jump = np.abs(np.r_[0, np.diff(s)])
print(f"  {'|score jump|':18} {au(jump, hard):>17.3f} {au(jump, y == 1):>15.3f}")
