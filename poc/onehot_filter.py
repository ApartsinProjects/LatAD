"""Discrete-aware linear-residual filter. Actuator/state channels (e.g. SWaT MV valves, pumps) are
categorical: predicting their numeric value linearly is meaningless. We encode each window as:
  continuous channel  -> window mean (standardized on normal)
  discrete channel    -> per-state window FRACTION (soft one-hot; unseen states -> fractions sum <1)
  constant channel    -> dropped (its anomalies are the max|z| filter's job)
Then a leave-one-CHANNEL-out linear regression predicts each feature from features of OTHER channels;
residual MSE per window is the filter score. Compare numeric-LinRes vs onehot-LinRes on detection
AUROC (all + max|z|-difficult) and held-out-normal calibration.
"""
from __future__ import annotations
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import eda_real as E

DMAX = 6  # channel is 'discrete' if <= DMAX distinct normal values


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k])) if m.sum() >= 3 else float("nan")


def win_means(X, W, stride):
    return np.stack([X[i:i + W].mean(0) for i in range(0, len(X) - W + 1, stride)])


def build_feats(Xn, Xa, W, stride, onehot):
    """returns Fn, Fa (window feature matrices) and grp (original-channel id per feature col)."""
    starts_n = range(0, len(Xn) - W + 1, stride); starts_a = range(0, len(Xa) - W + 1, stride)
    cols_n, cols_a, grp = [], [], []
    for c in range(Xn.shape[1]):
        states = np.unique(Xn[:, c]); k = len(states)
        if k <= 1:
            continue                                        # constant -> drop
        if onehot and k <= DMAX:                            # discrete -> per-state window fraction
            for s in states:
                cols_n.append([np.mean(Xn[i:i + W, c] == s) for i in starts_n])
                cols_a.append([np.mean(Xa[i:i + W, c] == s) for i in starts_a])
                grp.append(c)
        else:                                               # continuous (or numeric if !onehot)
            mn = [Xn[i:i + W, c].mean() for i in starts_n]; ma = [Xa[i:i + W, c].mean() for i in starts_a]
            mu, sd = np.mean(mn), np.std(mn) + 1e-9
            cols_n.append(((np.array(mn) - mu) / sd).tolist()); cols_a.append(((np.array(ma) - mu) / sd).tolist())
            grp.append(c)
    return np.array(cols_n).T, np.array(cols_a).T, np.array(grp)


def loco_residual(Fn, Fa, grp):
    r_tr = np.zeros(len(Fn)); r_te = np.zeros(len(Fa)); ncol = Fn.shape[1]
    for j in range(ncol):
        pred_cols = np.where(grp != grp[j])[0]              # leave one CHANNEL out (all its features)
        lr = LinearRegression().fit(Fn[:, pred_cols], Fn[:, j])
        r_tr += (lr.predict(Fn[:, pred_cols]) - Fn[:, j]) ** 2
        r_te += (lr.predict(Fa[:, pred_cols]) - Fa[:, j]) ** 2
    return r_tr / ncol, r_te / ncol


def run(name):
    D = E.load(name); fn, W, stride = E.RAW[name]
    Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
    y = D["ya_w"].astype(int)
    Xte0, Xtr0 = D["Xa_w"].astype(np.float32), D["Xn_w"].astype(np.float32)
    C6 = Xte0.shape[1] // 6
    maxz = np.abs(Xte0[:, :C6]).max(1); thr = float(np.quantile(np.abs(Xtr0[:, :C6]).max(1), 0.99))
    hard = (y == 1) & ~(maxz > thr)
    print(f"\n=== {name}  (anom={int((y==1).sum())}, max|z|-difficult={int(hard.sum())}) ===")
    for tag, oh in [("numeric", False), ("onehot", True)]:
        Fn, Fa, grp = build_feats(Xn, Xa, W, stride, oh)
        if len(Fa) != len(y):
            m = min(len(Fa), len(y)); Fa, yy = Fa[:m], y[:m]
        else:
            yy = y
        rtr, rte = loco_residual(Fn, Fa, grp)
        yfull = np.r_[np.zeros(len(Fn), int), yy]; rfull = np.r_[rtr, rte]
        fp = (rte[yy == 0] > np.quantile(rtr, 0.99)).mean()
        h = hard[:len(yy)]
        print(f"  {tag:8} feats={Fn.shape[1]:3}  AUROC all={au(yfull,rfull,(yfull==1)):.3f} "
              f"difficult={au(yy,rte,h):.3f}  test-normal-FP={fp*100:.1f}%")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI", "HAI", "SWaT"]):
        run(nm)
