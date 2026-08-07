"""Better DENSITY estimation in the latent (user ideas: non-diagonal / more params /
better mode fitting / hierarchical). Current head is a diagonal GMM with K=80. Test:
  diag80   : current (diagonal, K=80)
  full_Ks  : FULL covariance at K=K* (captures latent correlations)
  full_8   : full covariance, few modes
  bayes    : BayesianGaussianMixture (auto-prunes modes, full cov)
  agglo    : hierarchical (Ward) clustering -> per-cluster full Gaussian
Difficult-subset AUROC, seeds 0-2. Targets the CPS datasets where density is the signal.
"""
from __future__ import annotations
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score
from scipy.stats import multivariate_normal
from models_vade import train_vade
import eda_real as E

CFG = {"WADI": (20, 10), "HAI": (40, 16), "SKAB": (16, 6), "SWaT": (40, 16), "PSM": (40, 16), "SMD": (40, 16)}
KST = {"WADI": 22, "HAI": 24, "SKAB": 12, "SWaT": 25, "PSM": 25, "SMD": 15}
SEEDS = [0, 1, 2]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics"); os.makedirs(OUT, exist_ok=True)


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k])) if m.sum() >= 3 else float("nan")


def agglo_density(Ztr, Zte, k):
    lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Ztr)
    comps = []
    for c in range(k):
        pts = Ztr[lab == c]
        if len(pts) < Ztr.shape[1] + 2:
            continue
        comps.append((len(pts) / len(Ztr), multivariate_normal(pts.mean(0), np.cov(pts.T) + 1e-3 * np.eye(Ztr.shape[1]), allow_singular=True)))
    def nll(Z):
        L = np.stack([np.log(w + 1e-12) + mvn.logpdf(Z) for w, mvn in comps], 1)
        from scipy.special import logsumexp
        return -logsumexp(L, 1)
    return nll(Zte), nll(Ztr)


def run(name):
    D = E.load(name); Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"]
    C6 = Xte0.shape[1] // 6; triv = np.abs(Xte0[:, :C6]).max(1); trn = np.abs(Xtr0[:, :C6]).max(1)
    hard = (y == 1) & ~(triv > np.quantile(trn, 0.99))
    K, LD = CFG[name]; mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
    ks = KST[name]
    R = {k: [] for k in ["diag80", "full_Ks", "full_8", "bayes", "agglo"]}

    def score(gmm_te, gmm_tr):
        s = (gmm_te - gmm_tr.mean()) / (gmm_tr.std() + 1e-9)
        return au(y, s, hard)
    for seed in SEEDS:
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
        Ztr = v._encode_mean(Xtr); Zte = v._encode_mean(Xte)
        kd = min(80, max(20, len(Ztr) // 10))
        for tag, est in [("diag80", GaussianMixture(kd, covariance_type="diag", reg_covar=1e-3, random_state=seed)),
                         ("full_Ks", GaussianMixture(min(ks, len(Ztr)//(LD+2)), covariance_type="full", reg_covar=1e-2, random_state=seed)),
                         ("full_8", GaussianMixture(8, covariance_type="full", reg_covar=1e-2, random_state=seed)),
                         ("bayes", BayesianGaussianMixture(n_components=min(ks, len(Ztr)//(LD+2)), covariance_type="full", reg_covar=1e-2, random_state=seed, max_iter=200))]:
            try:
                est.fit(Ztr); R[tag].append(au(y, ((-est.score_samples(Zte)) - (-est.score_samples(Ztr)).mean())/((-est.score_samples(Ztr)).std()+1e-9), hard))
            except Exception:
                R[tag].append(float("nan"))
        try:
            nte, ntr = agglo_density(Ztr, Zte, min(ks, len(Ztr)//(LD+2)))
            R["agglo"].append(score(nte, ntr))
        except Exception:
            R["agglo"].append(float("nan"))
    r = {"dataset": name, **{k: round(float(np.nanmean(v)), 3) for k, v in R.items()}}
    json.dump(r, open(f"{OUT}/density_{name}.json", "w"), indent=1)
    print(f"{name:5} diag80={r['diag80']:.3f} full_Ks={r['full_Ks']:.3f} full_8={r['full_8']:.3f} "
          f"bayes={r['bayes']:.3f} agglo={r['agglo']:.3f}", flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI", "HAI", "SWaT", "PSM", "SMD", "SKAB"]):
        run(nm)
