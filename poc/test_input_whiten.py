"""Test: whiten the INPUT (not the reconstruction residual). A correlation break in x
shows up in (x-mu)^T Sigma_x^-1 (x-mu) because Sigma_x^-1 has off-diagonal terms; the
reconstruction-residual whitener misses it because the AE reconstructs the anomaly (r~0).
Compare HARD-subset AUROC: input-Mahalanobis (global + per-mode) vs recon-residual-whitened
vs latent-NLL vs IF-raw."""
from __future__ import annotations
import numpy as np, torch
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from models_vade import train_vade, _recon_energy
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

    res = {}
    res["IF-raw"] = -IsolationForest(n_estimators=200, random_state=0).fit(Xtr).decision_function(Xte)

    # (1) INPUT Mahalanobis, GLOBAL: Sigma_x on all normal windows
    lw = LedoitWolf().fit(Xtr)
    res["input-Mahal global"] = lw.mahalanobis(Xte)

    # (2) INPUT Mahalanobis, PER-MODE: GMM modes, per-mode LedoitWolf on the raw features
    pca = PCA(min(40, Xtr.shape[1]), random_state=0).fit(Xtr)   # PCA only to assign modes stably
    g = GaussianMixture(20, covariance_type="diag", random_state=0, reg_covar=1e-2).fit(pca.transform(Xtr))
    atr = g.predict(pca.transform(Xtr)); ate = g.predict(pca.transform(Xte))
    lws = {}
    for k in np.unique(atr):
        Xk = Xtr[atr == k]
        if len(Xk) >= 40:
            lws[k] = LedoitWolf().fit(Xk)
    sm = np.zeros(len(Xte))
    for k in np.unique(ate):
        idx = np.where(ate == k)[0]
        lwk = lws.get(k, lw)
        sm[idx] = lwk.mahalanobis(Xte[idx])
    res["input-Mahal per-mode"] = sm

    # (2b) REDUCE-then-whiten: PCA to r dims (drops noise directions), THEN Mahalanobis
    for r in [10, 20, 40]:
        p = PCA(r, random_state=0).fit(Xtr)
        lwr = LedoitWolf().fit(p.transform(Xtr))
        res[f"PCA{r}->Mahal global"] = lwr.mahalanobis(p.transform(Xte))
    # (2c) per-mode PCA(r_k by 95% var) -> Mahalanobis (idea A+B)
    smp = np.zeros(len(Xte))
    for k in np.unique(atr):
        Xk = Xtr[atr == k]
        if len(Xk) < 40:
            smp[ate == k] = 0; continue
        pk = PCA(min(30, len(Xk) - 1, Xk.shape[1]), random_state=0).fit(Xk)
        cum = np.cumsum(pk.explained_variance_ratio_); rk = int(np.searchsorted(cum, 0.95) + 1)
        pk = PCA(rk, random_state=0).fit(Xk); lwk = LedoitWolf().fit(pk.transform(Xk))
        idx = np.where(ate == k)[0]
        if len(idx):
            smp[idx] = lwk.mahalanobis(pk.transform(Xte[idx]))
    res["per-mode PCA95->Mahal"] = smp

    # (3) RECON-residual whitened + (4) latent-NLL, from a VaDE (the current pipeline)
    v = train_vade(Xtr, 20, 10, epochs=40, warmup=8, seed=0, device="cpu"); v.fit_residual_whitener(Xtr)
    with torch.no_grad():
        xt = torch.as_tensor(Xte, dtype=torch.float32); mu = v.encode(xt)[0]; xh = v.decode(mu)
    res["recon-residual whiten"] = np.asarray(_recon_energy(xt, xh, v.res_whitener))
    with torch.no_grad():
        res["latent-NLL"] = -v._log_pz_given_c(mu).max(1).values.cpu().numpy()

    print(f"WADI HARD n={nh}\n{'score':<26}{'HARD_AUROC':>11}")
    print("-" * 37)
    for n, s in res.items():
        print(f"{n:<26}{hauc(yw, s, hard):>11.3f}")


if __name__ == "__main__":
    main()
