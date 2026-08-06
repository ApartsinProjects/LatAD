"""Fused improved-VaDE. The mode-modeling sweep (test_vade_modes) showed the winning
hard mechanism is dataset-dependent and the families are complementary:
  WADI -> latent density ; HAI -> responsibility-weighted whitened residual ; SKAB -> closest-NLL.
So fuse all three, each z-calibrated on train-normal, and check we get the best-available
on every dataset (vs VaDE-as-is and the 2-head anomaly_score_hard).
"""
from __future__ import annotations
import numpy as np, torch
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from models_vade import train_vade, _as_tensor
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
def internals(v, Xs):
    x = _as_tensor(Xs, v); mu, _ = v.encode(x); xh = v.decode(mu)
    logN = v._log_pz_given_c(mu).cpu().numpy()
    logpi = torch.log_softmax(v.pi_logit, 0).cpu().numpy()
    lp = logpi[None] + logN; G = np.exp(lp - logsumexp(lp, 1, keepdims=True))
    return mu.cpu().numpy().astype(float), logN, G, (x - xh).cpu().numpy().astype(float)


def au(y, s, mask):
    k = (y == 0) | mask; return roc_auc_score(y[k], s[k])


def main(name, K, seed=0):
    D = E.load(name); Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = ((Xtr - m) / sd).astype(np.float32), ((Xte - m) / sd).astype(np.float32)
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy

    v = train_vade(Xtr_s, n_clusters=K, latent_dim=10, epochs=40, warmup=8, seed=seed)
    Ztr, lNtr, Gtr, Rtr = internals(v, Xtr_s); Zte, lNte, Gte, Rte = internals(v, Xte_s)
    a_tr = Gtr.argmax(1)

    # head 1: closest-mode diagonal NLL
    h1_tr, h1_te = -lNtr.max(1), -lNte.max(1)
    # head 2: density K80 on latent
    g = GaussianMixture(80, covariance_type="diag", reg_covar=1e-3, random_state=seed).fit(Ztr)
    h2_tr, h2_te = -g.score_samples(Ztr), -g.score_samples(Zte)
    # head 3: responsibility-weighted whitened residual (residual reduced to 30-dim)
    rp = PCA(min(30, Rtr.shape[1]), random_state=seed).fit(Rtr)
    Qtr, Qte = rp.transform(Rtr), rp.transform(Rte)
    rlw = {k: LedoitWolf().fit(Qtr[a_tr == k]) for k in np.unique(a_tr) if (a_tr == k).sum() >= 30}
    rglob = LedoitWolf().fit(Qtr)
    def rw(Q, G):
        mk = np.stack([rlw.get(k, rglob).mahalanobis(Q) for k in range(K)], 1)
        return (G * mk).sum(1)
    h3_tr, h3_te = rw(Qtr, Gtr), rw(Qte, Gte)

    def z(s, ref): return (s - ref.mean()) / (ref.std() + 1e-9)
    fused = z(h1_te, h1_tr) + z(h2_te, h2_tr) + z(h3_te, h3_tr)
    two = z(h1_te, h1_tr) + z(h2_te, h2_tr)                 # current anomaly_score_hard
    s0 = v.__class__.anomaly_score  # not used; keep VaDE-as-is via method
    v.fit_residual_whitener(Xtr_s); asis = v.anomaly_score(Xte_s)

    print(f"\n== {name} (easy={int(easy.sum())} hard={int(hard.sum())}) ==   ALL   EASY   HARD")
    for nm, s in [("VaDE as-is (recon+NLL)", asis), ("2-head (NLL+density)", two),
                  ("3-head (+resp-whiten)", fused),
                  ("  h1 closest-NLL", z(h1_te, h1_tr)), ("  h2 density", z(h2_te, h2_tr)),
                  ("  h3 resp-whiten resid", z(h3_te, h3_tr))]:
        print(f"{nm:<26}{au(yw,s,yw==1):>7.3f}{au(yw,s,easy):>7.3f}{au(yw,s,hard):>7.3f}")


if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI", "HAI", "SKAB"]):
        main(nm, {"WADI": 20, "HAI": 24, "SKAB": 16}.get(nm, 20))
