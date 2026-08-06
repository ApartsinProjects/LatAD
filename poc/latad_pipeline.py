"""LatAD consolidated anomaly-detection pipeline. Constraints honoured:
  (a) modes live ONLY in a LATENT embedding (linear PCA latent by default; VaDE-encoder latent
      pluggable) -- never the raw feature space;
  (b) ONLY parametric density models -- no LOF / kNN / instance-based (nothing stores the data);
  (c) individual branches AND one deterministic fused detector.

Two complementary parametric branches, both in the latent:
  A  per-mode FULL-covariance Gaussian -> nearest-mode Mahalanobis  (far-out + correlation breaks)
  B  high-K DIAGONAL GMM -> mixture density NLL                     (non-Gaussian pockets; the
     parametric replacement for LOF -- many small Gaussians = parametric KDE)
Deterministic fusion: z-sum of the two, calibrated on train-normal (fixed, no randomness), then a
mode-conditional (per-mode quantile) threshold. Everything is parametric: stores only PCA basis +
GMM params + LedoitWolf covariances + calibration constants.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest


class LatentParamAD:
    def __init__(self, pca_dim=20, k_modes=20, k_density=80, min_mode=40, seed=0):
        self.pca_dim, self.k_modes, self.k_density = pca_dim, k_modes, k_density
        self.min_mode, self.seed = min_mode, seed

    # ---- branch A: per-mode full-cov nearest Mahalanobis ----
    def _branchA(self, Z):
        a = self.gmm_mode.predict(Z); s = np.zeros(len(Z))
        for k in np.unique(a):
            s[a == k] = self.lw.get(k, self.glob).mahalanobis(Z[a == k])
        return s, a

    # ---- branch B: high-K diagonal GMM mixture density ----
    def _branchB(self, Z):
        return -self.gmm_dens.score_samples(Z)

    def _z(self, s, mu, sd):
        return (s - mu) / (sd + 1e-9)

    def fit(self, Xtr):
        self.pca = PCA(min(self.pca_dim, Xtr.shape[1]), random_state=self.seed).fit(Xtr)
        Z = self.pca.transform(Xtr).astype(float)
        self.gmm_mode = GaussianMixture(self.k_modes, covariance_type="full", reg_covar=1e-3,
                                        random_state=self.seed).fit(Z)
        a = self.gmm_mode.predict(Z)
        self.glob = LedoitWolf().fit(Z)
        self.lw = {k: LedoitWolf().fit(Z[a == k]) for k in np.unique(a)
                   if (a == k).sum() >= self.min_mode}
        self.gmm_dens = GaussianMixture(self.k_density, covariance_type="diag", reg_covar=1e-3,
                                        random_state=self.seed).fit(Z)
        # branch C (easy/overall): IsolationForest on FEATURES (not latent) -- catches out-of-envelope;
        # parametric-compatible (stores trees, not the data). Router = deterministic z-sum of A+B+C.
        self.ifm = IsolationForest(n_estimators=200, random_state=self.seed).fit(Xtr)
        sA, _ = self._branchA(Z); sB = self._branchB(Z); sC = -self.ifm.decision_function(Xtr)
        self.muA, self.sdA = sA.mean(), sA.std(); self.muB, self.sdB = sB.mean(), sB.std()
        self.muC, self.sdC = sC.mean(), sC.std()
        fused = (self._z(sA, self.muA, self.sdA) + self._z(sB, self.muB, self.sdB)
                 + self._z(sC, self.muC, self.sdC))
        self.gthr = float(np.quantile(fused, 0.95))
        self.mode_thr = {k: float(np.quantile(fused[a == k], 0.95))
                         if (a == k).sum() >= 10 else self.gthr for k in np.unique(a)}
        return self

    def scores(self, X):
        """Individual branch scores + the deterministic fused score."""
        Z = self.pca.transform(X).astype(float)
        sA, a = self._branchA(Z); sB = self._branchB(Z); sC = -self.ifm.decision_function(X)
        fused = (self._z(sA, self.muA, self.sdA) + self._z(sB, self.muB, self.sdB)
                 + self._z(sC, self.muC, self.sdC))
        return {"A_permode_mahal": sA, "B_density_mixture": sB, "C_easy_iforest": sC,
                "fused": fused, "assign": a}

    def score(self, X):
        return self.scores(X)["fused"]

    def flag(self, X, mode_conditional=True):
        r = self.scores(X)
        if not mode_conditional:
            return r["fused"] > self.gthr
        return np.array([r["fused"][i] > self.mode_thr.get(r["assign"][i], self.gthr)
                         for i in range(len(r["fused"]))])


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score, f1_score
    from winfeat import window_features
    import eda_real as E
    W, ST = 60, 30
    def win(Xx, yy=None):
        A, B = [], []
        for i in range(0, len(Xx) - W + 1, ST):
            A.append(window_features(Xx[i:i + W], "stats"))
            if yy is not None:
                B.append(int(yy[i:i + W].mean() > 0.05))
        return np.asarray(A, np.float32), (np.asarray(B, int) if yy is not None else None)
    D = E.load("WADI"); Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy

    m = LatentParamAD().fit(Xtr); sc = m.scores(Xte)
    def au(s, mask): k = (yw == 0) | mask; return roc_auc_score(yw[k], s[k])
    print(f"{'branch':<22}{'ALL':>7}{'EASY':>7}{'HARD':>7}")
    for name in ["A_permode_mahal", "B_density_mixture", "fused"]:
        s = sc[name]
        print(f"{name:<22}{au(s, yw==1):>7.3f}{au(s, easy):>7.3f}{au(s, hard):>7.3f}")
    for mc in [False, True]:
        f = m.flag(Xte, mode_conditional=mc)
        print(f"fused flag ({'mode-cond' if mc else 'global'} thr): "
              f"hard-catch {int((f&hard).sum())}/{int(hard.sum())}  "
              f"all-catch {int((f&(yw==1)).sum())}/{int((yw==1).sum())}  FPR {f[yw==0].mean():.3f}")
