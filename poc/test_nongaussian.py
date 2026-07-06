"""Hypothesis: some misses are non-Gaussian WITHIN-mode pockets. Mahalanobis assumes each
mode is a Gaussian ellipsoid, so a point inside the ellipsoid but in a low-density pocket
(curved/hollow mode) gets low Mahalanobis = called normal. A NON-PARAMETRIC density (kNN
distance to normal, or LOF) measures actual local density and should catch such pockets.

Test on WADI HARD: per-mode Mahalanobis vs kNN-to-normal vs LOF vs fusion. Also: for the
anomalies Mahalanobis MISSES, is their kNN distance large (i.e., they ARE in pockets)?"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
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


def main():
    D = E.load("WADI"); Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    hard = (yw == 1) & (triv <= np.quantile(trn, 0.99))
    P = PCA(20, random_state=0).fit(Xtr); Ztr = P.transform(Xtr).astype(float); Zte = P.transform(Xte).astype(float)
    g = GaussianMixture(20, covariance_type="full", random_state=0, reg_covar=1e-3).fit(Ztr)
    atr = g.predict(Ztr); ate = g.predict(Zte)
    lw = {k: LedoitWolf().fit(Ztr[atr == k]) for k in np.unique(atr) if (atr == k).sum() >= 40}
    glob = LedoitWolf().fit(Ztr)
    s_mahal = np.zeros(len(Zte))
    for k in np.unique(ate):
        s_mahal[ate == k] = lw.get(k, glob).mahalanobis(Zte[ate == k])

    # kNN distance to NORMAL (non-parametric density), global in 20-d
    def knn_dist(kk):
        nn = NearestNeighbors(n_neighbors=kk).fit(Ztr)
        return nn.kneighbors(Zte)[0][:, -1]                 # dist to k-th nearest normal
    s_knn5 = knn_dist(5); s_knn20 = knn_dist(20)
    # per-mode kNN: k-th nearest normal WITHIN the assigned mode
    s_pmknn = np.zeros(len(Zte))
    for k in np.unique(ate):
        Zk = Ztr[atr == k]
        if len(Zk) < 10:
            s_pmknn[ate == k] = 0; continue
        nn = NearestNeighbors(n_neighbors=min(5, len(Zk))).fit(Zk)
        s_pmknn[ate == k] = nn.kneighbors(Zte[ate == k])[0][:, -1]
    s_lof = -LocalOutlierFactor(20, novelty=True).fit(Ztr).decision_function(Zte)
    z = lambda a: (a - a.mean()) / (a.std() + 1e-9)

    def hauc(s):
        keep = (yw == 0) | hard; return roc_auc_score(yw[keep], s[keep])
    scores = {"per-mode Mahalanobis": s_mahal, "kNN-5 to normal": s_knn5, "kNN-20 to normal": s_knn20,
              "per-mode kNN-5": s_pmknn, "LOF (20-d)": s_lof,
              "Mahal + kNN20": z(s_mahal) + z(s_knn20), "Mahal + per-mode-kNN": z(s_mahal) + z(s_pmknn)}
    print(f"WADI HARD n={int(hard.sum())}\n{'detector':<26}{'HARD_AUROC':>11}")
    print("-" * 37)
    for n, s in scores.items():
        print(f"{n:<26}{hauc(s):>11.3f}")

    # are the Mahalanobis-missed anomalies in kNN pockets?
    thr = np.quantile(s_mahal[yw == 0], 0.95); missed = hard & (s_mahal <= thr)
    knn_pct = np.array([(s_knn20[yw == 0] < s_knn20[i]).mean() for i in np.where(missed)[0]])
    print(f"\nMahalanobis-missed hard anomalies: {int(missed.sum())}")
    print(f"  their kNN-20 distance percentile vs normal: mean {knn_pct.mean()*100:.0f}%  "
          f"(>90%% => in a pocket kNN would catch). #above90%%: {int((knn_pct>0.9).sum())}/{int(missed.sum())}")


if __name__ == "__main__":
    main()
