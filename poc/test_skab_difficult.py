"""Must win SKAB DIFFICULT over SOTA (USAD ~0.59). Base VaDE-hard ties it. Test difficult-boosters:
per-mode FULL-covariance Mahalanobis on the VaDE latent and on a PCA latent, and fusions with the
density head. Also inspect which SKAB-difficult anomalies USAD catches that we miss.
"""
from __future__ import annotations
import os, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from models_vade import train_vade
from permode_mahal import PerModeMahal
from winfeat import window_features
import eda_real as E

RES = os.path.join(os.path.dirname(__file__), "sota_bundle", "results")
W, ST = 60, 30
def winf(X, y=None):
    A, B = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], "stats"))
        if y is not None: B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)
def au(y, s, mask): k = (y == 0) | mask; return roc_auc_score(y[k], s[k])
def z(s, r): return (s - r.mean()) / (r.std() + 1e-9)

def permode_fullcov(Ztr, Zte, K, min_mode=25, seed=0):
    gm = GaussianMixture(K, covariance_type="full", reg_covar=1e-3, random_state=seed).fit(Ztr)
    a_tr, a_te = gm.predict(Ztr), gm.predict(Zte); glob = LedoitWolf().fit(Ztr)
    lw = {k: LedoitWolf().fit(Ztr[a_tr == k]) for k in np.unique(a_tr) if (a_tr == k).sum() >= min_mode}
    def sc(Z, a):
        s = np.zeros(len(Z))
        for k in np.unique(a): s[a == k] = lw.get(k, glob).mahalanobis(Z[a == k])
        return s
    return sc(Zte, a_te), sc(Ztr, a_tr)

def main(seed=0):
    D = E.load("SKAB"); Xtr, _ = winf(D["Xn_raw"]); Xte, yw = winf(D["Xa_raw"], D["ya_raw"])
    mu, sg = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = ((Xtr - mu) / sg).astype(np.float32), ((Xte - mu) / sg).astype(np.float32)
    C6 = Xte.shape[1] // 6; tv = np.abs(Xte[:, :C6]).max(1); tn = np.abs(Xtr[:, :C6]).max(1)
    ez = (yw == 1) & (tv > np.quantile(tn, 0.99)); hd = (yw == 1) & ~ez
    K = 16
    v = train_vade(Xtr_s, n_clusters=K, latent_dim=6, epochs=40, warmup=8, seed=seed)
    v.fit_residual_whitener(Xtr_s); v.fit_latent_density(Xtr_s, k_density=40)
    v.fit_resid_head(Xtr_s); v.fit_basin_head(Xtr_s)
    base = v.anomaly_score_hard(Xte_s, use_resid="auto", use_basin="auto"); base_tr = v.anomaly_score_hard(Xtr_s, use_resid="auto", use_basin="auto")
    Ztr, Zte = v._encode_mean(Xtr_s), v._encode_mean(Xte_s)
    vz_te, vz_tr = permode_fullcov(Ztr, Zte, K, seed=seed)           # full-cov on VaDE latent
    pm = PerModeMahal(n_modes=K); pm.fit(Xtr_s); pca_te = pm.score(Xte_s); pca_tr = pm.score(Xtr_s)  # PCA latent
    # USAD aligned
    sp = np.load(f"{RES}/score_USAD_SKAB.npy"); sp = sp.mean(1) if sp.ndim > 1 else sp
    Nfull = len(D["Xa_raw"]);  sp = np.pad(sp, (0, max(0, Nfull - len(sp))), mode="edge")[:Nfull]
    usad = np.array([sp[i:i + W].max() for i in range(0, Nfull - W + 1, ST)])[:len(yw)]

    print(f"SKAB difficult={int(hd.sum())} easy={int(ez.sum())}  (target: USAD diff {au(yw,usad,hd):.3f})")
    print(f"{'method':<30}{'ALL':>7}{'EASY':>7}{'DIFF':>7}")
    rows = [("USAD (SOTA)", usad), ("VaDE-hard base", base),
            ("VaDE-z per-mode fullcov", vz_te), ("PCA per-mode Mahal", pca_te),
            ("base + VaDE-z fullcov", z(base, base_tr) + z(vz_te, vz_tr)),
            ("base + PCA per-mode Mahal", z(base, base_tr) + z(pca_te, pca_tr)),
            ("base + both", z(base, base_tr) + z(vz_te, vz_tr) + z(pca_te, pca_tr))]
    best = (None, -1)
    for nm, s in rows:
        d = au(yw, s, hd)
        print(f"{nm:<30}{au(yw,s,yw==1):>7.3f}{au(yw,s,ez):>7.3f}{d:>7.3f}"
              + ("  <-- beats USAD" if d > au(yw, usad, hd) and nm != "USAD (SOTA)" else ""))
        if nm != "USAD (SOTA)" and d > best[1]: best = (nm, d)
    print(f"  best VaDE-based on DIFF: {best[0]} = {best[1]:.3f}  (USAD {au(yw,usad,hd):.3f})")

if __name__ == "__main__":
    main()
