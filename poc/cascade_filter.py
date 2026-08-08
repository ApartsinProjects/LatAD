"""Cascade difficulty filter (simple, linear). Second stage after max|z|:
  - average each signal over the window -> per-channel window means (n_win, C)
  - fit, on NORMAL, a leave-one-out linear regression predicting each channel's mean from the
    other N-1 channel means (captures normal LINEAR cross-channel structure)
  - residual MSE per window = mean_c (actual_c - predicted_c)^2  -> a linear correlation-break score
An anomaly is 'easy' if it trips max|z| OR the linear-residual threshold; what survives BOTH is the
genuinely hard set (needs the nonlinear joint density). We report the residual detector's own
AUROC on easy/difficult and how the cascade reshapes the hard subset.
"""
from __future__ import annotations
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import eda_real as E


def au(y, s, mask):
    k = (y == 0) | mask
    return float(roc_auc_score(y[k], s[k])) if mask.sum() >= 3 else float("nan")


def chan_means(X, W, stride):
    return np.stack([X[i:i + W].mean(0) for i in range(0, len(X) - W + 1, stride)])


def loo_residual(Mn, Mte):
    """residual MSE per row of Mte from N leave-one-out linear regressions fit on Mn."""
    C = Mn.shape[1]
    res_tr = np.zeros(len(Mn)); res_te = np.zeros(len(Mte))
    for c in range(C):
        cols = [j for j in range(C) if j != c]
        lr = LinearRegression().fit(Mn[:, cols], Mn[:, c])
        res_tr += (lr.predict(Mn[:, cols]) - Mn[:, c]) ** 2
        res_te += (lr.predict(Mte[:, cols]) - Mte[:, c]) ** 2
    return res_tr / C, res_te / C


def run(name):
    D = E.load(name); fn, W, stride = E.RAW[name]
    Xn, Xa, y = D["Xn_raw"], D["Xa_raw"], D["ya_w"]
    Mn = chan_means(np.asarray(Xn, float), W, stride)
    Ma = chan_means(np.asarray(Xa, float), W, stride)
    y = np.asarray(y, int)
    if len(Ma) != len(y):                                  # align (labels come from same windowing)
        n = min(len(Ma), len(y)); Ma, y = Ma[:n], y[:n]
    mu, sd = Mn.mean(0), Mn.std(0) + 1e-8
    Zn = (Mn - mu) / sd; Za = (Ma - mu) / sd

    zmax = np.abs(Za).max(1); zmax_tr = np.abs(Zn).max(1)   # filter A: per-channel magnitude
    res_tr, res_te = loo_residual(Zn, Za)                    # filter B: linear cross-channel residual

    yy = np.r_[np.zeros(len(Zn), int), y]
    res_all = np.r_[res_tr, res_te]; zmax_all = np.r_[zmax_tr, zmax]
    thrZ = np.quantile(zmax_tr, 0.99); thrR = np.quantile(res_tr, 0.99)

    easyA = (yy == 1) & (zmax_all > thrZ)                    # tripped by max|z|
    hardA = (yy == 1) & ~easyA                               # current 'difficult'
    easyB = (yy == 1) & (res_all > thrR)                     # tripped by linear residual
    hard_casc = (yy == 1) & ~easyA & ~easyB                  # survives BOTH -> genuinely hard

    print(f"\n=== {name}  (anom windows={int((yy==1).sum())}, thrZ={thrZ:.1f}, thrR={thrR:.3f}) ===")
    print(f"  linear-residual detector AUROC:  all={au(yy,res_all,(yy==1)):.3f}  "
          f"easy(max|z|)={au(yy,res_all,easyA):.3f}  difficult(max|z|)={au(yy,res_all,hardA):.3f}")
    print(f"  subset sizes:  current-difficult(maxz only)={int(hardA.sum())}  "
          f"-> survives linear-residual too = {int(hard_casc.sum())}  "
          f"(residual reclassifies {int(hardA.sum()-hard_casc.sum())} as easy)")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI", "HAI", "SWaT"]):
        run(nm)
