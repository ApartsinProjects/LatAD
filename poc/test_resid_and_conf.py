"""(1) resid-head on/off/auto in anomaly_score_hard; (2) confusion measures beyond entropy
-- 1st-to-2nd peak MARGIN and peak RATIO, per-mode calibrated; (3) K-selection (BIC) and
latent-dim selection (PCA participation ratio) on train-normal only.
"""
from __future__ import annotations
import numpy as np, torch
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from models_vade import train_vade, _as_tensor
from winfeat import window_features
import eda_real as E

W, ST = 60, 30
def win(X, y=None):
    A, B = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], "stats"))
        if y is not None: B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)

def au(y, s, mask): k = (y == 0) | mask; return roc_auc_score(y[k], s[k])

def permode_z(s_te, a_tr, s_tr, a_te):
    mu = {k: s_tr[a_tr == k].mean() for k in np.unique(a_tr)}
    sd = {k: s_tr[a_tr == k].std() + 1e-9 for k in np.unique(a_tr)}
    g0, g1 = s_tr.mean(), s_tr.std() + 1e-9
    return np.array([(s_te[i] - mu.get(a_te[i], g0)) / sd.get(a_te[i], g1) for i in range(len(s_te))])

@torch.no_grad()
def resp(v, Xs):
    mu = v.encode(_as_tensor(Xs, v))[0]
    lN = v._log_pz_given_c(mu).cpu().numpy()
    lp = torch.log_softmax(v.pi_logit, 0).cpu().numpy()[None] + lN
    return np.exp(lp - logsumexp(lp, 1, keepdims=True))

def select_K(Ztr, seed=0):
    ks = [5, 10, 15, 20, 30, 40]
    bic = [GaussianMixture(k, covariance_type="diag", reg_covar=1e-3, random_state=seed).fit(Ztr).bic(Ztr) for k in ks]
    return ks[int(np.argmin(bic))], dict(zip(ks, [round(b) for b in bic]))

def select_latentdim(Xtr):
    ev = PCA().fit(Xtr).explained_variance_
    pr = (ev.sum() ** 2) / (np.square(ev).sum())                # participation ratio
    cum = np.cumsum(ev) / ev.sum()
    d90 = int(np.searchsorted(cum, 0.90) + 1)
    return round(pr, 1), d90

def main(name, K, seed=0):
    D = E.load(name); Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = ((Xtr - m) / sd).astype(np.float32), ((Xte - m) / sd).astype(np.float32)
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy

    v = train_vade(Xtr_s, n_clusters=K, latent_dim=10, epochs=40, warmup=8, seed=seed)
    v.fit_residual_whitener(Xtr_s); v.fit_latent_density(Xtr_s, k_density=80); v.fit_resid_head(Xtr_s)

    print(f"\n########## {name} (easy={int(easy.sum())} hard={int(hard.sum())}) ##########")
    # (1) resid head on/off/auto
    print("(1) anomaly_score_hard        ALL   EASY   HARD")
    for lbl, kw in [("resid=False", dict(use_resid=False)), ("resid=True", dict(use_resid=True)),
                    (f"resid=auto[{'ON' if v._resid_auto else 'off'}]", dict(use_resid="auto"))]:
        s = v.anomaly_score_hard(Xte_s, **kw)
        print(f"   {lbl:<22}{au(yw,s,yw==1):>6.3f}{au(yw,s,easy):>7.3f}{au(yw,s,hard):>7.3f}")

    # (2) confusion measures: margin (g1-g2), peak ratio (g2/g1), entropy -- per-mode calibrated
    Gtr, Gte = resp(v, Xtr_s), resp(v, Xte_s); a_tr, a_te = Gtr.argmax(1), Gte.argmax(1)
    st = lambda G: np.sort(G, 1)
    marg = lambda G: st(G)[:, -1] - st(G)[:, -2]                 # high=confident (invert for anomaly)
    ratio = lambda G: st(G)[:, -2] / (st(G)[:, -1] + 1e-12)      # high=confused
    ent = lambda G: -(G * np.log(G + 1e-12)).sum(1)
    print("(2) confusion score           ALL   EASY   HARD")
    for lbl, f, inv in [("margin g1-g2", marg, True), ("peak ratio g2/g1", ratio, False),
                        ("entropy", ent, False)]:
        s_tr, s_te = f(Gtr), f(Gte)
        z = permode_z(s_te, a_tr, s_tr, a_te)
        if inv: z = -z
        print(f"   {lbl:<22}{au(yw,z,yw==1):>6.3f}{au(yw,z,easy):>7.3f}{au(yw,z,hard):>7.3f}")

    # (3) hyperparameter selection (train-normal only)
    Ztr = v._encode_mean(Xtr_s)
    kbest, bics = select_K(Ztr); pr, d90 = select_latentdim(Xtr_s)
    print(f"(3) K* by BIC on latent = {kbest}   (bics {bics})")
    print(f"    latent-dim: participation-ratio {pr}, 90%-var dim {d90}  (used latent_dim=10)")


if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI", "HAI", "SKAB"]):
        main(nm, {"WADI": 20, "HAI": 24, "SKAB": 16}.get(nm, 20))
