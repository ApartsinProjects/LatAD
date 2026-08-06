"""Fix VaDE on HARD. Diagnosis said: (a) recon residual is dead on hard (AUROC .49 --
correlation-break faults reconstruct fine); (b) latent NLL half-works (.66) but VaDE's
mode covariance is DIAGONAL, blind to correlation breaks between latent dims.

So keep VaDE's jointly-learned latent, but replace the scoring head:
  V0  VaDE as-is (recon + diagonal latent NLL)          [baseline]
  V1  latent NLL only (drop the dead recon term)
  V2  per-mode FULL-cov LedoitWolf Mahalanobis on z     (fixes the diagonal blindness)
  V3  high-K diagonal density GMM on z                  (non-Gaussian pockets)
  V4  z-sum(V2, V3)                                     (correlation break + pockets)
Reference: same per-mode full-cov on a PCA-20 latent (pipeline branch A).
"""
from __future__ import annotations
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from models_vade import train_vade, _recon_energy, _as_tensor
from winfeat import window_features
import eda_real as E

W, ST = 60, 30
def win(X, y=None):
    A, B = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], "stats"))
        if y is not None:
            B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)


@torch.no_grad()
def latent(v, Xs):
    return v.encode(_as_tensor(Xs, v))[0].cpu().numpy().astype(float)


def permode_mahal(Ztr, Zte, K, min_mode=30, seed=0):
    gm = GaussianMixture(K, covariance_type="full", reg_covar=1e-3, random_state=seed).fit(Ztr)
    a_tr, a_te = gm.predict(Ztr), gm.predict(Zte)
    glob = LedoitWolf().fit(Ztr)
    lw = {k: LedoitWolf().fit(Ztr[a_tr == k]) for k in np.unique(a_tr) if (a_tr == k).sum() >= min_mode}
    def sc(Z, a):
        s = np.zeros(len(Z))
        for k in np.unique(a):
            s[a == k] = lw.get(k, glob).mahalanobis(Z[a == k])
        return s
    return sc(Zte, a_te), sc(Ztr, a_tr)


def density_gmm(Ztr, Zte, K=80, seed=0):
    gm = GaussianMixture(K, covariance_type="diag", reg_covar=1e-3, random_state=seed).fit(Ztr)
    return -gm.score_samples(Zte), -gm.score_samples(Ztr)


def au(y, s, mask):
    k = (y == 0) | mask
    return roc_auc_score(y[k], s[k])


def zt(s, ref):
    return (s - ref.mean()) / (ref.std() + 1e-9)


def main(name="WADI", K=20, seed=0):
    D = E.load(name)
    Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = ((Xtr - m) / sd).astype(np.float32), ((Xte - m) / sd).astype(np.float32)
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy

    v = train_vade(Xtr_s, n_clusters=K, latent_dim=10, epochs=40, warmup=8, seed=seed)
    v.fit_residual_whitener(Xtr_s)
    v0_te = v.anomaly_score(Xte_s)                                     # recon + diag NLL
    # V1: latent NLL only (encode -> nearest-mode diagonal log-density)
    with torch.no_grad():
        nl_te = -v._log_pz_given_c(v.encode(_as_tensor(Xte_s, v))[0]).max(1).values.cpu().numpy()

    Ztr, Zte = latent(v, Xtr_s), latent(v, Xte_s)                     # VaDE latent
    v2_te, v2_tr = permode_mahal(Ztr, Zte, K, seed=seed)              # full-cov Mahal on z
    v3_te, v3_tr = density_gmm(Ztr, Zte, K=80, seed=seed)            # high-K density on z
    v4_te = zt(v2_te, v2_tr) + zt(v3_te, v3_tr)

    # reference: same full-cov Mahal but on a PCA-20 latent (pipeline branch A)
    pca = PCA(min(20, Xtr_s.shape[1]), random_state=seed).fit(Xtr_s)
    Ptr, Pte = pca.transform(Xtr_s).astype(float), pca.transform(Xte_s).astype(float)
    p2_te, _ = permode_mahal(Ptr, Pte, K, seed=seed)

    rows = [("V0 VaDE (recon+diagNLL)", v0_te), ("V1 latent NLL only", nl_te),
            ("V2 VaDE-z permode fullcov", v2_te), ("V3 VaDE-z density K80", v3_te),
            ("V4 z-sum(V2,V3)", v4_te), ("ref PCA-z permode fullcov", p2_te)]
    print(f"\n===== {name}  (easy={int(easy.sum())} hard={int(hard.sum())}) =====")
    print(f"{'method':<28}{'ALL':>7}{'EASY':>7}{'HARD':>7}")
    for nm, s in rows:
        print(f"{nm:<28}{au(yw, s, yw==1):>7.3f}{au(yw, s, easy):>7.3f}{au(yw, s, hard):>7.3f}")


if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI"]):
        K = {"WADI": 20, "HAI": 24, "SKAB": 16}.get(nm, 20)
        main(nm, K=K)
