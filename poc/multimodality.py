"""Measure multimodality of NORMAL CPS data per dataset, model-agnostically (on the
standardized window features the detector sees -- NOT the VaDE latent, which is trained
with a K-mode prior and would be circular). Quantifies the MIIM claim and tests whether
SKAB is genuinely near-unimodal relative to WADI/HAI.

Measures (train-normal only):
  * BIC-optimal number of modes K* (diagonal GMM sweep) -- the headline measure.
  * BIC improvement ratio (K=1 -> K=Kmax): how much structure beyond one blob.
  * Silhouette at K* (mode separation).
  * Sarle's bimodality coefficient (max over top PCs); >0.555 suggests non-unimodal.
Outputs _diagnostics/multimodality.json and a figure (PNG+SVG).
"""
from __future__ import annotations
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import skew, kurtosis
import eda_real as E

DATASETS = ["WADI", "HAI", "SWaT", "MetroPT"]
KMAX = 25
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics"); os.makedirs(OUT, exist_ok=True)
COL = {"WADI": "#1f4e79", "HAI": "#0f7d84", "SKAB": "#c0392b", "SWaT": "#9a6a12", "BATADAL": "#7d5ba6", "TEP": "#c1440e"}


def bimodality_coeff(X, npc=5):
    """Sarle's BC = (g1^2 + 1) / (g2 + 3) per PC; >0.555 => likely non-unimodal. Max over PCs."""
    bcs = []
    for j in range(min(npc, X.shape[1])):
        v = X[:, j]
        g1 = float(skew(v)); g2 = float(kurtosis(v))  # excess kurtosis
        n = len(v)
        corr = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)) if n > 3 else 0
        bcs.append((g1 ** 2 + 1) / (g2 + corr + 3 + 1e-9))
    return float(max(bcs)), [round(b, 3) for b in bcs]


def analyze(name):
    D = E.load(name)
    X = D["Xn_w"].astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    # PCA to a stable dimensionality for GMM BIC (retain 95% var, cap 20)
    p = PCA(n_components=min(20, X.shape[1], X.shape[0] - 1), random_state=0).fit(X)
    d95 = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.95) + 1)
    d = max(2, min(d95, 20))
    Z = p.transform(X)[:, :d]

    bics = []
    for K in range(1, KMAX + 1):
        g = GaussianMixture(K, covariance_type="diag", n_init=2, random_state=0, reg_covar=1e-4).fit(Z)
        bics.append(float(g.bic(Z)))
    bics = np.array(bics)
    kstar = int(np.argmin(bics) + 1)
    improve = float((bics[0] - bics.min()) / (abs(bics[0]) + 1e-9))   # relative BIC gain over 1 blob
    # silhouette at K* (subsample for speed)
    sil = None
    if kstar > 1:
        g = GaussianMixture(kstar, covariance_type="diag", n_init=2, random_state=0, reg_covar=1e-4).fit(Z)
        lab = g.predict(Z)
        if len(set(lab)) > 1:
            idx = np.random.default_rng(0).choice(len(Z), min(3000, len(Z)), replace=False)
            sil = float(silhouette_score(Z[idx], lab[idx]))
    bc, bc_pcs = bimodality_coeff(Z)
    return {"dataset": name, "n_normal": int(len(X)), "pca_dim": d,
            "Kstar_BIC": kstar, "bic_improve_ratio": round(improve, 4),
            "silhouette_at_Kstar": None if sil is None else round(sil, 3),
            "bimodality_coeff_max": round(bc, 3), "bimodality_per_pc": bc_pcs,
            "bic_curve": [round(b, 1) for b in bics]}


def main():
    res = {name: analyze(name) for name in DATASETS}
    json.dump(res, open(os.path.join(OUT, "multimodality.json"), "w"), indent=1)
    # figure: BIC curves (normalized) + K* bar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    for name in DATASETS:
        b = np.array(res[name]["bic_curve"])
        bn = (b - b.min()) / (b.max() - b.min() + 1e-9)  # normalized shape
        ax1.plot(range(1, KMAX + 1), bn, marker="o", ms=3, color=COL[name], label=name)
        ax1.scatter([res[name]["Kstar_BIC"]], [bn[res[name]["Kstar_BIC"] - 1]], color=COL[name], s=70, zorder=5,
                    edgecolor="white", linewidth=1)
    ax1.set_xlabel("number of GMM modes K"); ax1.set_ylabel("normalized BIC (lower = better fit)")
    ax1.set_title("Model selection on normal data: multimodal sets keep improving with K")
    ax1.legend(frameon=False); ax1.grid(alpha=.2)
    ks = [res[n]["Kstar_BIC"] for n in DATASETS]
    ax2.bar(DATASETS, ks, color=[COL[n] for n in DATASETS])
    for i, k in enumerate(ks): ax2.text(i, k + 0.3, str(k), ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("BIC-optimal modes $K^*$"); ax2.set_title("Effective number of operating modes")
    ax2.grid(alpha=.2, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "multimodality.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "multimodality.svg"), bbox_inches="tight")
    # console table
    print(f"{'dataset':6} {'K*':>4} {'BICgain':>8} {'silhouette':>11} {'bimodalCoeff':>13}")
    for n in DATASETS:
        r = res[n]
        print(f"{n:6} {r['Kstar_BIC']:>4} {r['bic_improve_ratio']:>8.3f} "
              f"{str(r['silhouette_at_Kstar']):>11} {r['bimodality_coeff_max']:>13.3f}")


if __name__ == "__main__":
    main()
