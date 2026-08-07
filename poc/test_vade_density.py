"""User's ask: keep the ORIGINAL VaDE clustering (nearest mu_c assignment), but fit a
NON-DIAGONAL covariance per mode. Compare per-mode covariance models on the SAME modes:
  vade_diag : per-mode diagonal covariance
  vade_full : per-mode FULL covariance
  vade_lw   : per-mode Ledoit-Wolf shrinkage (partial/regularised full -- robust for small modes)
vs the current head (diag80 = fresh diagonal GMM, K=80). Difficult-subset AUROC, seeds 0-2.
"""
from __future__ import annotations
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
import eda_real as E

CFG = {"WADI": (20, 10), "HAI": (40, 16), "SKAB": (16, 6), "SWaT": (40, 16), "PSM": (40, 16), "SMD": (40, 16)}
SEEDS = [0, 1, 2]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics"); os.makedirs(OUT, exist_ok=True)


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k])) if m.sum() >= 3 else float("nan")


def zt(s, ref): return (s - ref.mean()) / (ref.std() + 1e-9)


def per_mode_density(Ztr, a, K, cov):
    """Build a mixture over the VaDE modes with per-mode covariance `cov` in {diag,full,lw}."""
    d = Ztr.shape[1]; comps = []
    gcov = np.cov(Ztr.T) + 1e-3 * np.eye(d)                     # fallback for tiny modes
    for k in range(K):
        pts = Ztr[a == k]
        w = len(pts) / len(Ztr)
        if w == 0:
            continue
        m = pts.mean(0)
        if len(pts) < d + 2:
            C = gcov
        elif cov == "diag":
            C = np.diag(pts.var(0) + 1e-3)
        elif cov == "full":
            C = np.cov(pts.T) + 1e-2 * np.eye(d)
        else:  # lw
            C = LedoitWolf().fit(pts).covariance_ + 1e-4 * np.eye(d)
        comps.append((w, multivariate_normal(m, C, allow_singular=True)))

    def nll(Z):
        L = np.stack([np.log(w + 1e-12) + mvn.logpdf(Z) for w, mvn in comps], 1)
        return -logsumexp(L, 1)
    return nll


def run(name):
    D = E.load(name); Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"]
    C6 = Xte0.shape[1] // 6; triv = np.abs(Xte0[:, :C6]).max(1); trn = np.abs(Xtr0[:, :C6]).max(1)
    hard = (y == 1) & ~(triv > np.quantile(trn, 0.99))
    K, LD = CFG[name]; mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
    R = {k: [] for k in ["diag80", "vade_diag", "vade_full", "vade_lw"]}
    for seed in SEEDS:
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
        Ztr = v._encode_mean(Xtr); Zte = v._encode_mean(Xte)
        muc = v.mu_c.detach().cpu().numpy()
        a_tr = ((Ztr[:, None] - muc[None]) ** 2).sum(-1).argmin(1)
        kd = min(80, max(20, len(Ztr) // 10))
        g = GaussianMixture(kd, covariance_type="diag", reg_covar=1e-3, random_state=seed).fit(Ztr)
        R["diag80"].append(au(y, zt(-g.score_samples(Zte), -g.score_samples(Ztr)), hard))
        for tag, cov in [("vade_diag", "diag"), ("vade_full", "full"), ("vade_lw", "lw")]:
            nll = per_mode_density(Ztr, a_tr, K, cov)
            R[tag].append(au(y, zt(nll(Zte), nll(Ztr)), hard))
    r = {"dataset": name, **{k: round(float(np.nanmean(v)), 3) for k, v in R.items()}}
    json.dump(r, open(f"{OUT}/vade_density_{name}.json", "w"), indent=1)
    print(f"{name:5} diag80={r['diag80']:.3f} | vade_diag={r['vade_diag']:.3f} "
          f"vade_full={r['vade_full']:.3f} vade_lw={r['vade_lw']:.3f}", flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI", "HAI", "SWaT", "PSM", "SMD", "SKAB"]):
        run(nm)
