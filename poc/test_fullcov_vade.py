"""Full-covariance VaDE-family test. Our VaDE uses a DIAGONAL GMM prior (0.502 hard). Try
FULL covariance: a neural VAE latent + full-covariance per-mode GMM + Mahalanobis (sequential
VAE+GMM is the practical equivalent of joint full-cov VaDE; sequential ~ joint on real data,
sec 2.10). Compare vs (a) diagonal VaDE, (b) linear PCA-20 + full-cov Mahal (0.702), across
latent sizes, to see if the NEURAL encoder + full cov beats the LINEAR full-cov version."""
from __future__ import annotations
import numpy as np, torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from models_vade import train_plain_vae, train_vade
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


def fullcov_mahal(Ztr, Zte, K=20):
    """full-covariance GMM modes + per-mode LedoitWolf Mahalanobis (assigned mode)."""
    g = GaussianMixture(K, covariance_type="full", random_state=0, reg_covar=1e-3).fit(Ztr)
    atr, ate = g.predict(Ztr), g.predict(Zte)
    lw = {k: LedoitWolf().fit(Ztr[atr == k]) for k in np.unique(atr) if (atr == k).sum() >= 40}
    glob = LedoitWolf().fit(Ztr); s = np.zeros(len(Zte))
    for k in np.unique(ate):
        s[ate == k] = lw.get(k, glob).mahalanobis(Zte[ate == k])
    return s


def main():
    D = E.load("WADI"); Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    hard = (yw == 1) & (triv <= np.quantile(trn, 0.99))
    def hauc(s): keep = (yw == 0) | hard; return roc_auc_score(yw[keep], s[keep])

    # reference: linear PCA-20 + full-cov Mahal
    P = PCA(20, random_state=0).fit(Xtr)
    print(f"{'method':<34}{'ALL':>7}{'HARD':>7}")
    s = fullcov_mahal(P.transform(Xtr).astype(float), P.transform(Xte).astype(float))
    print(f"{'PCA-20 + full-cov Mahal (ref)':<34}{roc_auc_score(yw, s):>7.3f}{hauc(s):>7.3f}")
    # diagonal VaDE (reference)
    v = train_vade(Xtr, 20, 10, epochs=40, warmup=8, seed=0, device="cpu"); v.fit_residual_whitener(Xtr)
    sv = v.anomaly_score(Xte)
    print(f"{'diagonal VaDE (ref)':<34}{roc_auc_score(yw, sv):>7.3f}{hauc(sv):>7.3f}")

    # NEURAL VAE latent + full-cov Mahal, several latent sizes
    for ld in [10, 20, 40]:
        pv = train_plain_vae(Xtr, latent_dim=ld, epochs=60, seed=0, device="cpu")
        Ztr = pv.encode_mean(Xtr).astype(float); Zte = pv.encode_mean(Xte).astype(float)
        s = fullcov_mahal(Ztr, Zte)
        print(f"{'VAE-latent(' + str(ld) + ') + full-cov Mahal':<34}{roc_auc_score(yw, s):>7.3f}{hauc(s):>7.3f}")


if __name__ == "__main__":
    main()
