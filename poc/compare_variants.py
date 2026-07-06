"""All VaDE scoring variants on 3 datasets x {ALL,EASY,HARD} x {AUC, best-F1, FPR@best-F1}.
Trains VaDE once/dataset; every fit + calibration uses TRAIN-NORMAL only. The only place
test labels enter is the best-F1 threshold sweep (oracle best-F1, the field-standard TS-AD
metric); AUROC is threshold-free. Includes the new RESPONSIBILITY-ENTROPY score (per-mode
calibrated on normal).
"""
from __future__ import annotations
import numpy as np, torch
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
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
    return mu.cpu().numpy().astype(float), logN, logpi, G, (x - xh).cpu().numpy().astype(float)


def metr(y, s, mask):
    keep = (y == 0) | mask; yk, sk = y[keep], s[keep]
    if (yk == 1).sum() < 3:
        return np.nan, np.nan, np.nan
    au = roc_auc_score(yk, sk)
    qs = np.quantile(sk, np.linspace(0.80, 0.999, 60))
    f1, tb = max(((f1_score(yk, sk > t), t) for t in qs), key=lambda p: p[0])
    return au, f1, float((sk[yk == 0] > tb).mean())


def permode_z(score_te, a_tr, s_tr, a_te):
    mu = {k: s_tr[a_tr == k].mean() for k in np.unique(a_tr)}
    sd = {k: s_tr[a_tr == k].std() + 1e-9 for k in np.unique(a_tr)}
    gm, gs = s_tr.mean(), s_tr.std() + 1e-9
    return np.array([(score_te[i] - mu.get(a_te[i], gm)) / sd.get(a_te[i], gs) for i in range(len(score_te))])


def main(name, K, seed=0):
    D = E.load(name); Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = ((Xtr - m) / sd).astype(np.float32), ((Xte - m) / sd).astype(np.float32)
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy

    v = train_vade(Xtr_s, n_clusters=K, latent_dim=10, epochs=40, warmup=8, seed=seed)
    v.fit_residual_whitener(Xtr_s)
    Ztr, lNtr, logpi, Gtr, Rtr = internals(v, Xtr_s)
    Zte, lNte, _, Gte, Rte = internals(v, Xte_s)
    a_tr, a_te = Gtr.argmax(1), Gte.argmax(1)
    def z(s, r): return (s - r.mean()) / (r.std() + 1e-9)

    # --- heads ---
    asis = v.anomaly_score(Xte_s)
    nll_te, nll_tr = -lNte.max(1), -lNtr.max(1)                       # closest-NLL
    gd = GaussianMixture(80, covariance_type="diag", reg_covar=1e-3, random_state=seed).fit(Ztr)
    den_te, den_tr = -gd.score_samples(Zte), -gd.score_samples(Ztr)   # density K80
    # full-cov closest Mahal
    lw = {k: LedoitWolf().fit(Ztr[a_tr == k]) for k in np.unique(a_tr) if (a_tr == k).sum() >= 30}
    glob = LedoitWolf().fit(Ztr)
    def mahal(Z, a):
        s = np.zeros(len(Z))
        for k in np.unique(a): s[a == k] = lw.get(k, glob).mahalanobis(Z[a == k])
        return s
    fc_te = mahal(Zte, a_te)
    # resp-weighted whitened residual
    rp = PCA(min(30, Rtr.shape[1]), random_state=seed).fit(Rtr)
    Qtr, Qte = rp.transform(Rtr), rp.transform(Rte)
    rlw = {k: LedoitWolf().fit(Qtr[a_tr == k]) for k in np.unique(a_tr) if (a_tr == k).sum() >= 30}
    rglob = LedoitWolf().fit(Qtr)
    rw_te = (Gte * np.stack([rlw.get(k, rglob).mahalanobis(Qte) for k in range(K)], 1)).sum(1)
    # NEW: responsibility-entropy, per-mode calibrated on normal (high entropy = between modes)
    ent = lambda G: -(G * np.log(G + 1e-12)).sum(1)
    H_te, H_tr = ent(Gte), ent(Gtr)
    ent_te = permode_z(H_te, a_tr, H_tr, a_te)
    # IF-easy branch (features)
    ifm = IsolationForest(n_estimators=200, random_state=seed).fit(Xtr_s)
    if_te, if_tr = -ifm.decision_function(Xte_s), -ifm.decision_function(Xtr_s)

    # --- fusions ---
    two = z(nll_te, nll_tr) + z(den_te, den_tr)                       # anomaly_score_hard
    two_if = two + z(if_te, if_tr)                                    # + IF-easy (branch C)
    two_ent = two + np.maximum(0.0, z(H_te, H_tr))                    # + entropy (one-sided)
    three = two + z(rw_te, (Gtr * np.stack([rlw.get(k, rglob).mahalanobis(Qtr) for k in range(K)], 1)).sum(1))

    variants = [
        ("VaDE as-is (recon+NLL)", asis), ("closest-NLL only", nll_te),
        ("density K80", den_te), ("full-cov closest Mahal", fc_te),
        ("resp-weighted whiten resid", rw_te), ("resp-ENTROPY (per-mode cal)", ent_te),
        ("IF-easy (features)", if_te),
        ("2-head hard (NLL+dens)", two), ("2-head + IF-easy", two_if),
        ("2-head + entropy", two_ent), ("3-head (+resp-whiten)", three),
    ]
    print(f"\n########## {name}  (easy={int(easy.sum())} hard={int(hard.sum())}, K={K}) ##########")
    print(f"{'variant':<28}{'ALL au/f1/fpr':>18}{'EASY au/f1/fpr':>18}{'HARD au/f1/fpr':>18}")
    for nm, s in variants:
        cells = []
        for msk in (yw == 1, easy, hard):
            au, f1, fpr = metr(yw, s, msk); cells.append(f"{au:.2f}/{f1:.2f}/{fpr:.2f}")
        print(f"{nm:<28}" + "".join(f"{c:>18}" for c in cells))


if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI", "HAI", "SKAB"]):
        main(nm, {"WADI": 20, "HAI": 24, "SKAB": 16}.get(nm, 20))
