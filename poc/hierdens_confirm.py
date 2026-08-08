"""5-seed confirmation: does the HIERARCHICAL (agglomerative Ward -> per-cluster full Gaussian)
density head beat the current DIAGONAL K=80 head on the difficult subset? Paired by seed (same
VaDE latents feed both heads), canonical max|z| difficulty split. Reports per-dataset mean+/-std for
each head AND the paired difference (agglo - diag) with its own std -- the paired test is what
decides significance. A win must have a positive paired mean and a paired std small enough that the
mean is clearly above zero.
"""
from __future__ import annotations
import sys, json, os, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
import eda_real as E

CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}
KST = {"WADI": 22, "HAI": 24, "SWaT": 25}
SEEDS = [0, 1, 2, 3, 4]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k])) if m.sum() >= 3 else float("nan")


def zt(s, ref): return (s - ref.mean()) / (ref.std() + 1e-9)


def agglo_density(Ztr, Zte, k):
    lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Ztr)
    comps = []
    for c in range(k):
        pts = Ztr[lab == c]
        if len(pts) < Ztr.shape[1] + 2:
            continue
        comps.append((len(pts) / len(Ztr),
                      multivariate_normal(pts.mean(0), np.cov(pts.T) + 1e-3 * np.eye(Ztr.shape[1]), allow_singular=True)))

    def nll(Z):
        L = np.stack([np.log(w + 1e-12) + mvn.logpdf(Z) for w, mvn in comps], 1)
        return -logsumexp(L, 1)
    return nll(Zte), nll(Ztr)


def run(name):
    D = E.load(name); Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"]
    C6 = Xte0.shape[1] // 6; triv = np.abs(Xte0[:, :C6]).max(1); trn = np.abs(Xtr0[:, :C6]).max(1)
    hard = (y == 1) & ~(triv > np.quantile(trn, 0.99))
    K, LD = CFG[name]; ks = KST[name]
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
    diag, agglo, diff = [], [], []
    for seed in SEEDS:
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
        Ztr = v._encode_mean(Xtr); Zte = v._encode_mean(Xte)
        kd = min(80, max(20, len(Ztr) // 10))
        g = GaussianMixture(kd, covariance_type="diag", reg_covar=1e-3, random_state=seed).fit(Ztr)
        a_diag = au(y, zt(-g.score_samples(Zte), -g.score_samples(Ztr)), hard)
        nte, ntr = agglo_density(Ztr, Zte, min(ks, len(Ztr) // (LD + 2)))
        a_agglo = au(y, zt(nte, ntr), hard)
        diag.append(a_diag); agglo.append(a_agglo); diff.append(a_agglo - a_diag)
        print(f"    {name} seed {seed}: diag={a_diag:.3f} agglo={a_agglo:.3f} diff={a_agglo-a_diag:+.3f}", flush=True)
    r = {"dataset": name, "n_hard": int(hard.sum()),
         "diag_mean": float(np.mean(diag)), "diag_std": float(np.std(diag)),
         "agglo_mean": float(np.mean(agglo)), "agglo_std": float(np.std(agglo)),
         "paired_diff_mean": float(np.mean(diff)), "paired_diff_std": float(np.std(diff)),
         "diag": diag, "agglo": agglo}
    json.dump(r, open(f"{OUT}/hierdens_{name}.json", "w"), indent=1)
    sig = "WIN" if np.mean(diff) > 2 * (np.std(diff) / np.sqrt(len(diff)) + 1e-9) else "n.s."
    print(f"  {name}: diag={r['diag_mean']:.3f}+/-{r['diag_std']:.3f}  agglo={r['agglo_mean']:.3f}+/-{r['agglo_std']:.3f}  "
          f"paired_diff={r['paired_diff_mean']:+.3f}+/-{r['paired_diff_std']:.3f}  [{sig}]", flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI", "HAI", "SWaT"]):
        run(nm)
