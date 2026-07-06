"""Window / hyperparameter selection. Questions:
  - automatic W? propose a LABEL-FREE selector: W ~ c * tau, tau = autocorrelation decorrelation
    time of normal channels (lag where mean |ACF| first drops below 1/e). Physics-grounded.
  - do K, L depend on W? joint grid W x K x L, report all-AUROC, see if argmax K/L shift with W.
  - best joint config? report the sweep argmax and whether (tau-W, BIC-K, PR-L) predicts it.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from models_vade import train_vade
from winfeat import window_features
import eda_real as E

def acf_tau(Xn, maxlag=300):
    """mean over channels of the lag where |autocorrelation| first drops below 1/e."""
    taus = []
    for c in range(Xn.shape[1]):
        x = Xn[:, c] - Xn[:, c].mean(); v = (x * x).mean()
        if v < 1e-9: continue
        for lag in range(1, min(maxlag, len(x) // 2)):
            if abs((x[:-lag] * x[lag:]).mean() / v) < 0.3679:
                taus.append(lag); break
        else:
            taus.append(maxlag)
    return float(np.median(taus)) if taus else np.nan

def part_ratio(X):
    ev = PCA().fit(X).explained_variance_
    return (ev.sum() ** 2) / (np.square(ev).sum() + 1e-12)

def bic_K(Z, seed=0):
    ks = [8, 12, 16, 24, 40]
    return ks[int(np.argmin([GaussianMixture(k, covariance_type="diag", reg_covar=1e-3,
                                             random_state=seed).fit(Z).bic(Z) for k in ks]))]

def winf(X, W, ST, y=None):
    A, B = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], "stats"))
        if y is not None: B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)

def au(y, s, mask): k = (y == 0) | mask; return roc_auc_score(y[k], s[k])

def main(name, seed=0):
    D = E.load(name)
    tau = acf_tau(D["Xn_raw"])
    print(f"\n##### {name}  ACF tau={tau:.0f} -> suggested W~2tau={2*tau:.0f} #####")
    print(f"{'W':>4}{'K':>4}{'L':>4}{'ALL':>8}{'EASY':>8}{'DIFF':>8}{'PR':>6}{'BICk':>6}")
    best = (None, -1)
    for W in [30, 60, 120]:
        ST = W // 2
        Xtr, _ = winf(D["Xn_raw"], W, ST); Xte, yw = winf(D["Xa_raw"], W, ST, y=D["ya_raw"])
        mu, sg = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr_s, Xte_s = ((Xtr - mu) / sg).astype(np.float32), ((Xte - mu) / sg).astype(np.float32)
        C6 = Xte.shape[1] // 6; tv = np.abs(Xte[:, :C6]).max(1); tn = np.abs(Xtr[:, :C6]).max(1)
        ez = (yw == 1) & (tv > np.quantile(tn, 0.99)); hd = (yw == 1) & ~ez
        pr = part_ratio(Xtr_s)
        for K in [16, 40]:
            for L in [6, 16]:
                v = train_vade(Xtr_s, n_clusters=K, latent_dim=L, epochs=40, warmup=8, seed=seed)
                v.fit_residual_whitener(Xtr_s); v.fit_latent_density(Xtr_s, k_density=80)
                v.fit_resid_head(Xtr_s); v.fit_basin_head(Xtr_s)
                s = v.anomaly_score_hard(Xte_s, use_resid="auto", use_basin="auto")
                aa = au(yw, s, yw == 1)
                bk = bic_K(v._encode_mean(Xtr_s), seed) if (K == 16 and L == 6) else ""
                print(f"{W:>4}{K:>4}{L:>4}{aa:>8.3f}{au(yw,s,ez):>8.3f}{au(yw,s,hd):>8.3f}"
                      f"{pr:>6.1f}{str(bk):>6}")
                if aa > best[1]: best = ((W, K, L, aa), aa)
    b = best[0]
    print(f"  -> best joint config: W={b[0]} K={b[1]} L={b[2]}  ALL {b[3]:.3f}   (tau-W~{2*tau:.0f})")

if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI", "HAI", "SKAB"]):
        main(nm)
