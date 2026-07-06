"""Deep dive on VaDE hard-anomaly scoring. Trains VaDE once per dataset, then:

  A. MODE STRUCTURE of hard anomalies -- do they live in one mode? are they
     'between modes' (low max-responsibility / high entropy)?  This decides
     whether closest-mode vs Bayesian-mixture NLL matters.
  B. LATENT-NLL variants:
       L1 diag NLL, closest mode           (VaDE's own head)
       L2 diag NLL, Bayesian mixture        (logsumexp over pi-weighted comps)
       L3 full-cov per-mode Mahal, closest  (fuller mode model)
       L4 full-cov Bayesian mixture NLL
       L5/6/7 density head K=40/80/160       (fine-tuned parametric KDE)
       L8 diag NLL closest, PER-MODE z-normalised (== per-mode threshold)
  C. RESIDUAL-WHITENING variants (residual reduced to 30-dim, per-mode cov):
       R1 global whitening                  (VaDE recon baseline)
       R2 per-mode whitening (assigned mode)
       R3 responsibility-WEIGHTED whitening (sum_k gamma_k * mahal_k(r))
       R4 per-mode whitening, PER-MODE z-normalised

NLL note: 'closest' uses max_k log N(z|c_k); 'Bayesian' uses logsumexp_k[log pi_k +
log N(z|c_k)] i.e. the true mixture density. The density head (L5-7) is Bayesian by
construction (score_samples = mixture log-likelihood).
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


def au(y, s, mask):
    k = (y == 0) | mask
    return roc_auc_score(y[k], s[k])


@torch.no_grad()
def vade_internals(v, Xs):
    """latent mu, per-comp diag log-density (N,K), responsibilities (N,K), residual."""
    x = _as_tensor(Xs, v)
    mu, _ = v.encode(x)
    x_hat = v.decode(mu)
    logN = v._log_pz_given_c(mu).cpu().numpy()                 # (N,K) log N_diag(z|c_k)
    logpi = torch.log_softmax(v.pi_logit, 0).cpu().numpy()
    logpcz = logpi[None] + logN
    gamma = np.exp(logpcz - logsumexp(logpcz, 1, keepdims=True))
    z = mu.cpu().numpy().astype(float)
    r = (x - x_hat).cpu().numpy().astype(float)
    return z, logN, logpi, gamma, r


def permode_z(score, a_tr, s_tr, a_te):
    """z-normalise a score per assigned mode using that mode's train stats."""
    mu = {k: s_tr[a_tr == k].mean() for k in np.unique(a_tr)}
    sd = {k: s_tr[a_tr == k].std() + 1e-9 for k in np.unique(a_tr)}
    gm, gs = s_tr.mean(), s_tr.std() + 1e-9
    return np.array([(score[i] - mu.get(a_te[i], gm)) / sd.get(a_te[i], gs)
                     for i in range(len(score))])


def main(name="WADI", K=20, seed=0):
    D = E.load(name)
    Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = ((Xtr - m) / sd).astype(np.float32), ((Xte - m) / sd).astype(np.float32)
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy

    v = train_vade(Xtr_s, n_clusters=K, latent_dim=10, epochs=40, warmup=8, seed=seed)
    v.fit_residual_whitener(Xtr_s)
    Ztr, lNtr, logpi, Gtr, Rtr = vade_internals(v, Xtr_s)
    Zte, lNte, _, Gte, Rte = vade_internals(v, Xte_s)
    a_tr, a_te = Gtr.argmax(1), Gte.argmax(1)

    # ---------- A. mode structure of hard anomalies ----------
    ent = lambda G: -(G * np.log(G + 1e-12)).sum(1)
    hmax, nmax = Gte[hard].max(1), Gtr.max(1)
    print(f"\n===== {name}  (easy={int(easy.sum())} hard={int(hard.sum())}, K={K}) =====")
    modes_hit, counts = np.unique(a_te[hard], return_counts=True)
    order = np.argsort(-counts)
    print("A. HARD mode structure:")
    print(f"   hard windows land in {len(modes_hit)}/{K} modes; top-3: "
          + ", ".join(f"m{modes_hit[i]}:{counts[i]}" for i in order[:3]))
    print(f"   max-responsibility  hard {hmax.mean():.2f}  vs normal {nmax.mean():.2f}  "
          f"(low = 'between modes')")
    print(f"   resp entropy        hard {ent(Gte[hard]).mean():.2f}  vs normal {ent(Gtr).mean():.2f}")
    print(f"   frac hard with max-resp<0.5: {(hmax < 0.5).mean():.2f}")

    # ---------- B. latent-NLL variants ----------
    # full-cov per-mode (LedoitWolf) + logdet for Bayesian full-cov mixture
    lw = {k: LedoitWolf().fit(Ztr[a_tr == k]) for k in np.unique(a_tr) if (a_tr == k).sum() >= 30}
    glob = LedoitWolf().fit(Ztr)
    def mahal_closest(Z, a):
        s = np.zeros(len(Z))
        for k in np.unique(a):
            s[a == k] = lw.get(k, glob).mahalanobis(Z[a == k])
        return s
    d = Ztr.shape[1]
    pik = np.array([ (a_tr == k).mean() for k in range(K) ])
    def fullcov_mix_nll(Z):
        comp = np.full((len(Z), K), -1e9)
        for k, est in lw.items():
            diff = Z - est.location_
            mah = np.einsum('ni,ij,nj->n', diff, est.precision_, diff)
            logdet = np.linalg.slogdet(est.covariance_)[1]
            comp[:, k] = np.log(pik[k] + 1e-12) - 0.5 * (d * np.log(2*np.pi) + logdet + mah)
        return -logsumexp(comp, 1)
    def dens(kk):
        g = GaussianMixture(kk, covariance_type="diag", reg_covar=1e-3, random_state=seed).fit(Ztr)
        return -g.score_samples(Zte)
    L1 = -lNte.max(1)                                              # diag closest
    L2 = -logsumexp(logpi[None] + lNte, 1)                        # diag Bayesian mixture
    L3 = mahal_closest(Zte, a_te)                                 # full-cov closest
    L4 = fullcov_mix_nll(Zte)                                     # full-cov Bayesian mixture
    L5, L6, L7 = dens(40), dens(80), dens(160)                    # density head sweep
    L8 = permode_z(L1, a_tr, -lNtr.max(1), a_te)                 # diag closest, per-mode z
    print("B. latent-NLL variants          ALL   EASY   HARD")
    for nm, s in [("L1 diag closest", L1), ("L2 diag Bayesian mix", L2),
                  ("L3 fullcov closest", L3), ("L4 fullcov Bayesian mix", L4),
                  ("L5 density K40", L5), ("L6 density K80", L6), ("L7 density K160", L7),
                  ("L8 diag closest permode-z", L8)]:
        print(f"   {nm:<26}{au(yw,s,yw==1):>6.3f}{au(yw,s,easy):>7.3f}{au(yw,s,hard):>7.3f}")

    # ---------- C. residual-whitening variants ----------
    rp = PCA(min(30, Rtr.shape[1]), random_state=seed).fit(Rtr)   # reduce residual
    Qtr, Qte = rp.transform(Rtr), rp.transform(Rte)
    rlw = {k: LedoitWolf().fit(Qtr[a_tr == k]) for k in np.unique(a_tr) if (a_tr == k).sum() >= 30}
    rglob = LedoitWolf().fit(Qtr)
    R1 = rglob.mahalanobis(Qte)                                   # global whitening
    def permode_res(Q, a):
        s = np.zeros(len(Q))
        for k in np.unique(a):
            s[a == k] = rlw.get(k, rglob).mahalanobis(Q[a == k])
        return s
    R2 = permode_res(Qte, a_te)                                   # assigned-mode whitening
    # R3 responsibility-weighted: sum_k gamma_k * mahal_k(r)
    mahk = np.zeros((len(Qte), K))
    for k in range(K):
        est = rlw.get(k, rglob); mahk[:, k] = est.mahalanobis(Qte)
    R3 = (Gte * mahk).sum(1)
    R2tr = permode_res(Qtr, a_tr)
    R4 = permode_z(R2, a_tr, R2tr, a_te)                          # per-mode z-normalised
    print("C. residual-whitening variants  ALL   EASY   HARD")
    for nm, s in [("R1 global whiten", R1), ("R2 per-mode whiten", R2),
                  ("R3 resp-weighted whiten", R3), ("R4 per-mode whiten permode-z", R4)]:
        print(f"   {nm:<26}{au(yw,s,yw==1):>6.3f}{au(yw,s,easy):>7.3f}{au(yw,s,hard):>7.3f}")


if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI"]):
        K = {"WADI": 20, "HAI": 24, "SKAB": 16}.get(nm, 20)
        main(nm, K=K)
