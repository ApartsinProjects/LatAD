"""Local density-gradient score (refined C2). Per assigned mode k, the Gaussian log-density
gradient is g = -Sigma_k^-1 (x-mu_k). Decompose into:
  magnitude  = ||g||  (~ how strongly density pulls back; close to Mahalanobis)
  alignment  = cos(g, x-mu_k)  (isotropic 'just far' -> ~1; anisotropic correlation-break -> <1)
  off-frac   = ||off-manifold part of (x-mu)|| / ||x-mu||   (scale-free: fraction off the mode subspace)
Hard anomalies are moderate-magnitude but anisotropic, so 'misalignment' (1-cos) and off-frac
should add signal a pure-magnitude Mahalanobis misses. Test HARD-AUROC + fusions on WADI."""
from __future__ import annotations
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from winfeat import window_features
import eda_real as E

W, ST = 60, 30


def win(X, y=None):
    Xw, yl = [], []
    for i in range(0, len(X) - W + 1, ST):
        Xw.append(window_features(X[i:i + W], "stats"))
        if y is not None:
            yl.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(Xw, np.float32), (np.asarray(yl, int) if y is not None else None)


def hauc(y, s, hard):
    keep = (y == 0) | hard
    return roc_auc_score(y[keep], s[keep])


def main():
    D = E.load("WADI"); Xn, Xa, yp = D["Xn_raw"], D["Xa_raw"], D["ya_raw"]
    Xtr, _ = win(Xn); Xte, yw = win(Xa, yp)
    C6 = Xte.shape[1] // 6
    triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    hard = (yw == 1) & (triv <= np.quantile(trn, 0.99)); nh = int(hard.sum())

    # reduce to 20-d (the whitening sweet spot), assign modes, per-mode mean+precision+subspace
    P = PCA(20, random_state=0).fit(Xtr); Ztr = P.transform(Xtr); Zte = P.transform(Xte)
    g = GaussianMixture(20, covariance_type="full", random_state=0, reg_covar=1e-3).fit(Ztr)
    atr = g.predict(Ztr); ate = g.predict(Zte)
    mahal = np.zeros(len(Zte)); mag = np.zeros(len(Zte)); misalign = np.zeros(len(Zte)); offfrac = np.zeros(len(Zte))
    for k in np.unique(ate):
        idx = np.where(ate == k)[0]
        Zk = Ztr[atr == k]
        if len(Zk) < 40:
            mahal[idx] = 0; continue
        mu = Zk.mean(0); lw = LedoitWolf().fit(Zk); Prec = lw.precision_
        # per-mode on-manifold subspace = top eigenvectors of the mode covariance (95% var)
        ev, U = np.linalg.eigh(lw.covariance_); order = np.argsort(ev)[::-1]; ev = ev[order]; U = U[:, order]
        r = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95) + 1); Bk = U[:, :r]
        d = Zte[idx] - mu                                   # (n,20)
        gvec = d @ Prec                                     # -grad direction = Sigma^-1 d
        mahal[idx] = np.einsum("ij,ij->i", d, gvec)         # Mahalanobis^2
        nd = np.linalg.norm(d, axis=1) + 1e-9; ng = np.linalg.norm(gvec, axis=1) + 1e-9
        mag[idx] = ng
        cos = np.einsum("ij,ij->i", d, gvec) / (nd * ng)    # cos(d, Sigma^-1 d) in [0,1]
        misalign[idx] = 1.0 - cos                           # anisotropy of the pull
        on = d @ Bk @ Bk.T; off = d - on
        offfrac[idx] = np.linalg.norm(off, axis=1) / nd     # scale-free off-manifold fraction
    z = lambda a: (a - a.mean()) / (a.std() + 1e-9)
    ifs = -IsolationForest(n_estimators=200, random_state=0).fit(Xtr).decision_function(Xte)

    scores = {
        "IF-raw (ref)": ifs,
        "Mahalanobis (magnitude)": mahal,
        "grad misalignment (1-cos)": misalign,
        "off-manifold fraction": offfrac,
        "Mahal + misalignment": z(mahal) + z(misalign),
        "Mahal + off-fraction": z(mahal) + z(offfrac),
        "IF + Mahal + off-frac": z(ifs) + z(mahal) + z(offfrac),
    }
    print(f"WADI HARD n={nh}\n{'score':<28}{'HARD_AUROC':>11}")
    print("-" * 39)
    for n, s in scores.items():
        print(f"{n:<28}{hauc(yw, s, hard):>11.3f}")


if __name__ == "__main__":
    main()
