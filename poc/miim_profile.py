"""
Quantify the MIIM properties we assume of a dataset - so 'multimodal',
'imbalanced', 'massive' become measured numbers, not claims. Given normal data X:

  d_int      intrinsic dimension (PCA components for 95% variance)
  Khat       number of modes selected by BIC over a GMM (multimodality/massive)
  bic_gain   per-sample BIC improvement of Khat over K=1 (>0 => genuinely
             multimodal; ~0 => one blob)
  silhouette separation of the Khat clustering (how 'implicit'/hard the modes are)
  K_eff      effective number of modes exp(H(pi)) (participation ratio)
  imbalance  1 - K_eff/Khat  (0 balanced -> 1 extreme imbalance)
  gini       Gini of mode weights
  zipf_beta  fitted power-law exponent of sorted mode sizes (our Zipf assumption)

Run:  C:/Python314/python.exe miim_profile.py
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def _gini(w):
    w = np.sort(w); n = len(w)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(w) / (n * w.sum() + 1e-12))


def intrinsic_dim(X, var=0.95):
    ev = PCA().fit(X).explained_variance_ratio_
    return int(np.searchsorted(np.cumsum(ev), var) + 1)


def miim_profile(X, Ks=(1, 2, 4, 8, 16, 32, 64), sub=4000, seed=0):
    rng = np.random.default_rng(seed)
    if len(X) > sub:
        X = X[rng.choice(len(X), sub, replace=False)]
    d_int = intrinsic_dim(X)
    Xr = PCA(n_components=int(min(max(d_int, 2), X.shape[1], 20)),
             random_state=seed).fit_transform(X)          # stabilise GMM in intrinsic space
    bic = {}
    for K in Ks:
        if K < len(Xr):
            bic[K] = GaussianMixture(K, covariance_type="diag", reg_covar=1e-4,
                                     random_state=seed, n_init=1).fit(Xr).bic(Xr)
    Khat = min(bic, key=bic.get)
    g = GaussianMixture(Khat, covariance_type="diag", reg_covar=1e-4,
                        random_state=seed, n_init=2).fit(Xr)
    w = g.weights_
    K_eff = float(np.exp(-(w * np.log(w + 1e-12)).sum()))
    sizes = np.sort(w)[::-1]
    beta = float(-np.polyfit(np.log(np.arange(1, len(sizes) + 1)), np.log(sizes + 1e-12), 1)[0])
    lab = g.predict(Xr)
    sil = float(silhouette_score(Xr, lab)) if len(np.unique(lab)) > 1 else float("nan")
    return {"n": len(X), "d": X.shape[1], "d_int": d_int, "Khat": Khat,
            "bic_gain": float((bic[1] - bic[Khat]) / len(Xr)) if 1 in bic else float("nan"),
            "silhouette": sil, "K_eff": K_eff, "imbalance": 1 - K_eff / Khat,
            "gini": _gini(w), "zipf_beta": beta}


if __name__ == "__main__":
    import torch
    from data import make_miim
    from skab import load_skab
    from hai import load_hai

    ds_syn, _ = make_miim(n_modes=40, seed=0)
    datasets = {
        "synthetic-MIIM": ds_syn.x_train,
        "SKAB": load_skab(rep="stats").x_train,
        "HAI": load_hai(rep="stats").x_train,
    }
    keys = ["d", "d_int", "Khat", "bic_gain", "silhouette", "K_eff", "imbalance", "gini", "zipf_beta"]
    print(f"{'dataset':<16}" + "".join(f"{k:>11}" for k in keys))
    for name, X in datasets.items():
        p = miim_profile(np.asarray(X))
        print(f"{name:<16}" + "".join(f"{p[k]:>11.3f}" if isinstance(p[k], float)
                                       else f"{p[k]:>11}" for k in keys))
