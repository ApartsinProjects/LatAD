"""Replace LOF (non-parametric, stores all normal data) with a PARAMETRIC density that still
captures non-Gaussian modes/pockets: a rich full-covariance GMM scored by the MIXTURE density
(log-sum over all components, not nearest-mode). A high-K GMM is a universal density approximator
that stores only K*(mean+cov), not the data. Test whether it matches LOF-latent (0.732) on WADI HARD.

Storage: LOF keeps 2614x20 = ~52k floats; a K=80 full-cov GMM in 20-d = 80*(20 + 210) ~ 18k floats
and O(K) query vs O(n) neighbour search."""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
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
    def hauc(s): keep = (yw == 0) | hard; return roc_auc_score(yw[keep], s[keep])

    lof = -LocalOutlierFactor(20, novelty=True).fit(Ztr).decision_function(Zte)
    print(f"{'detector':<34}{'HARD':>7}{'ALL':>7}{'stores':>16}")
    print(f"{'LOF-latent (non-parametric)':<34}{hauc(lof):>7.3f}{roc_auc_score(yw,lof):>7.3f}{'~52k (all data)':>16}")
    for K in [20, 40, 80, 120]:
        for cov in ["full", "diag"]:
            g = GaussianMixture(K, covariance_type=cov, random_state=0, reg_covar=1e-3,
                                n_init=1, max_iter=100).fit(Ztr)
            s = -g.score_samples(Zte)                       # mixture NLL (all components)
            npar = K * (20 + (210 if cov == "full" else 20))
            print(f"{'GMM-' + cov + ' K=' + str(K) + ' mixtureNLL':<34}{hauc(s):>7.3f}"
                  f"{roc_auc_score(yw,s):>7.3f}{'~' + str(npar//1000) + 'k params':>16}")


if __name__ == "__main__":
    main()
