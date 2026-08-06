"""Per-mode Mahalanobis detector (the A+B+C recipe that beat IsolationForest on the WADI
HARD subset, 0.702 vs 0.695). No autoencoder, no reconstruction:

  A  discover modes (GMM) and give EACH its own covariance;
  B  reduce to the intrinsic dimension FIRST (global PCA to `pca_dim`), so the whitening
     is not dominated by high-dimensional noise directions;
  C  score = Mahalanobis distance to the assigned mode (a density/distance score, whose
     Sigma^-1 off-diagonals penalise correlation breaks that reconstruction MSE misses);
  F  LedoitWolf shrinkage per mode (handles small/rare modes); global-cov fallback.

fit(Xtr) on normal window features; score(X) returns per-window anomaly scores.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf


class PerModeMahal:
    def __init__(self, pca_dim=20, n_modes=20, min_mode=40, seed=0):
        self.pca_dim = pca_dim; self.n_modes = n_modes; self.min_mode = min_mode; self.seed = seed

    def fit(self, Xtr):
        self.pca = PCA(min(self.pca_dim, Xtr.shape[1]), random_state=self.seed).fit(Xtr)
        Z = self.pca.transform(Xtr).astype(np.float64)
        self.gmm = GaussianMixture(self.n_modes, covariance_type="full",
                                   reg_covar=1e-3, random_state=self.seed).fit(Z)
        a = self.gmm.predict(Z)
        self.glob = LedoitWolf().fit(Z)                    # fallback for tiny modes
        self.lw = {}
        for k in np.unique(a):
            Zk = Z[a == k]
            self.lw[k] = LedoitWolf().fit(Zk) if len(Zk) >= self.min_mode else self.glob
        return self

    def score(self, X):
        Z = self.pca.transform(X).astype(np.float64); a = self.gmm.predict(Z)
        s = np.zeros(len(Z))
        for k in np.unique(a):
            idx = a == k
            s[idx] = self.lw.get(k, self.glob).mahalanobis(Z[idx])
        return s


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    from winfeat import window_features
    import eda_real as E
    W, ST = 60, 30
    def win(Xx, yy=None):
        a, b = [], []
        for i in range(0, len(Xx) - W + 1, ST):
            a.append(window_features(Xx[i:i + W], "stats"))
            if yy is not None:
                b.append(int(yy[i:i + W].mean() > 0.05))
        return np.asarray(a, np.float32), (np.asarray(b, int) if yy is not None else None)
    D = E.load("WADI"); Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    hard = (yw == 1) & (triv <= np.quantile(trn, 0.99))
    s = PerModeMahal().fit(Xtr).score(Xte)
    keep = (yw == 0) | hard
    print(f"WADI: ALL AUROC {roc_auc_score(yw, s):.3f}   HARD AUROC {roc_auc_score(yw[keep], s[keep]):.3f}")
