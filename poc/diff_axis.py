"""Why are SWaT 'difficult' anomalies not actually difficult? Characterize the max|z|-difficult
subset of each dataset with DATA-ONLY statistics (no AE / no learned model):
  maxz   : per-channel max|z|            (the CURRENT difficulty axis)
  l2     : ||z||_2 / sqrt(d)             (coherent multi-channel magnitude)
  maha   : Mahalanobis dist on train cov (joint deviation, on+off manifold)
  onman  : energy INSIDE normal PCA subspace (reconstructable coherent shift)
  offman : residual OUTSIDE normal PCA subspace (correlation-break signal)
For each stat we report the AUROC that stat alone gives on the difficult subset. A stat with
high AUROC on 'difficult' means those anomalies are trivially separable by that simple rule,
i.e. they are not truly hard. Correlation-break faults should be caught ONLY by offman.
"""
from __future__ import annotations
import sys, numpy as np
from numpy.linalg import pinv
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import eda_real as E

VARKEEP = 0.95


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k])) if m.sum() >= 3 else float("nan")


def run(name):
    D = E.load(name)
    Xtr, Xte, y = D["Xn_w"].astype(np.float64), D["Xa_w"].astype(np.float64), D["ya_w"].astype(int)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr = (Xtr - mu) / sd; Zte = (Xte - mu) / sd
    # current difficulty axis: per-channel max|z| on first sixth (raw-channel block)
    C6 = Xte.shape[1] // 6
    maxz = np.abs(Zte[:, :C6]).max(1); maxz_tr = np.abs(Ztr[:, :C6]).max(1)
    hard = (y == 1) & ~(maxz > np.quantile(maxz_tr, 0.99))
    easy = (y == 1) & (maxz > np.quantile(maxz_tr, 0.99))

    # normal PCA subspace (linear reconstructor, data-only)
    p = PCA(n_components=VARKEEP, svd_solver="full").fit(Ztr)
    kdim = p.n_components_
    proj = p.transform(Zte); recon = p.inverse_transform(proj)
    onman = np.sqrt((proj ** 2).sum(1) / max(kdim, 1))              # inside normal subspace
    offman = np.sqrt(((Zte - recon) ** 2).sum(1) / Zte.shape[1])    # residual (correlation-break)
    l2 = np.sqrt((Zte ** 2).sum(1) / Zte.shape[1])
    # Mahalanobis on a diagonal+shrunk train cov (cheap, stable)
    cov = np.cov(Ztr.T) + 1e-2 * np.eye(Ztr.shape[1]); IC = pinv(cov)
    maha = np.einsum("ij,jk,ik->i", Zte, IC, Zte) ** 0.5

    stats = {"maxz": maxz, "l2": l2, "maha": maha, "onman": onman, "offman": offman}
    print(f"\n=== {name}  (PCA dim {kdim}, {int(easy.sum())} easy / {int(hard.sum())} difficult anomalies) ===")
    print(f"  {'stat':7} {'AUROC@difficult':>16} {'AUROC@easy':>12} {'median(diff)/median(normal)':>28}")
    for k, s in stats.items():
        ad = au(y, s, hard); ae = au(y, s, easy)
        rel = np.median(s[hard]) / (np.median(s[y == 0]) + 1e-9)
        print(f"  {k:7} {ad:>16.3f} {ae:>12.3f} {rel:>28.2f}")
    # fraction of 'difficult' anomalies already flagged by a MULTIVARIATE rule (l2 or maha > 99th pct normal)
    for k in ["l2", "maha", "offman"]:
        thr = np.quantile(stats[k][y == 0], 0.99)
        frac = (stats[k][hard] > thr).mean()
        print(f"  -> {frac*100:5.1f}% of 'difficult' exceed the 99th-normal-pct of {k}")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI", "HAI", "SWaT"]):
        run(nm)
