"""
Publication-quality figures for the PoC. One 2x2 panel:

  (A) VaDE latent (2-D PCA): normal points coloured by operating mode, with the
      two fault families overlaid - shows the multimodal normal + where faults sit.
  (B) Anomaly-score distributions by group (symlog x) - OOD separates far out,
      pockets are the hard case overlapping normal.
  (C) Component 2: basin-agreement distribution, rare-normal vs pocket - the
      mechanism, rare-but-valid modes sit in ONE stable basin, pockets split.
  (D) Method comparison from the 5-seed run: AUROC (up) and rare-mode FPR (down).

Run after run_seeds.py (needs results/results_multiseed.json).
  C:/Python314/python.exe make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import component2
from data import make_miim
from models_vade import train_vade

OUT = Path(__file__).parent / "results"
SEED = 0
# consistent group palette
C = {"common-normal": "#2f8f7f", "rare-normal": "#3b6fb0",
     "pocket": "#111111", "ood": "#e0872b"}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(SEED); torch.manual_seed(SEED)
    ds, meta = make_miim(n_modes=40, seed=SEED)
    vade = train_vade(ds.x_train, n_clusters=40, latent_dim=10, epochs=60,
                      seed=SEED, device=device)
    vade.fit_residual_whitener(ds.x_train)
    base = vade.anomaly_score(ds.x_test)
    agr, dist = component2.basin_features(vade, ds.x_test)

    rare = set(meta["rare_modes"])
    normal = ds.y_test == 0
    is_rare = np.array([m in rare for m in ds.mode_test]) & normal
    is_common = np.array([m not in rare for m in ds.mode_test]) & normal
    pkt = ds.atype_test == "pocket"
    ood = ds.atype_test == "ood"

    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("MIIM anomaly detection: normal structure, faults, and the basin test",
                 fontsize=15, fontweight="bold", y=0.995)

    # ---- (A) latent PCA ----
    with torch.no_grad():
        z = vade.encode(torch.as_tensor(ds.x_test, dtype=torch.float32,
                                        device=device))[0].cpu().numpy()
    z2 = z - z.mean(0)
    _, _, vt = np.linalg.svd(z2, full_matrices=False)
    p = z2 @ vt[:2].T
    a = ax[0, 0]
    # subsample normal for clarity, colour by mode
    idx = np.where(normal)[0]
    sub = np.random.default_rng(0).choice(idx, size=min(2500, len(idx)), replace=False)
    a.scatter(p[sub, 0], p[sub, 1], c=ds.mode_test[sub], cmap="tab20", s=9,
              alpha=0.55, linewidths=0, label="_normal modes")
    a.scatter(p[pkt, 0], p[pkt, 1], marker="x", c=C["pocket"], s=26,
              linewidths=1.1, label=f"pocket fault (n={pkt.sum()})")
    a.scatter(p[ood, 0], p[ood, 1], marker="+", c=C["ood"], s=26,
              linewidths=1.1, label=f"ood fault (n={ood.sum()})")
    a.set_title("(A) VaDE latent (PCA) — normal coloured by operating mode")
    a.set_xlabel("PC1"); a.set_ylabel("PC2")
    a.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # ---- (B) score distributions, symlog ----
    b = ax[0, 1]
    for lbl, key, mask in [("common-normal", "common-normal", is_common),
                           ("rare-normal", "rare-normal", is_rare),
                           ("pocket fault", "pocket", pkt),
                           ("ood fault", "ood", ood)]:
        b.hist(base[mask], bins=50, alpha=0.6, density=True, color=C[key], label=lbl)
    b.set_xscale("symlog")
    b.set_title("(B) VaDE anomaly score by group (symlog x)")
    b.set_xlabel("anomaly score  (recon + latent density)")
    b.set_ylabel("density"); b.legend(fontsize=8)

    # ---- (C) basin agreement: rare-normal vs pocket ----
    c = ax[1, 0]
    bins = np.linspace(0, 1, 26)
    c.hist(agr[is_rare], bins=bins, alpha=0.7, density=True, color=C["rare-normal"],
           label=f"rare-normal (valid)  mean={agr[is_rare].mean():.2f}")
    c.hist(agr[pkt], bins=bins, alpha=0.7, density=True, color=C["pocket"],
           label=f"pocket fault  mean={agr[pkt].mean():.2f}")
    c.set_title("(C) Component 2 — basin agreement separates rare-valid from pockets")
    c.set_xlabel("basin agreement  (1 = one stable mode, 0.5 = split between two)")
    c.set_ylabel("density"); c.legend(fontsize=8, loc="upper left")

    # ---- (D) method comparison bar chart ----
    d = ax[1, 1]
    js = OUT / "results_multiseed.json"
    if js.exists():
        res = json.loads(js.read_text())
        methods = list(res.keys())
        short = [m.replace(" (sequential)", "\n(seq)").replace(" (joint, ours)", "\n(joint)")
                 .replace(" + basin (full, ours)", "\n+basin").replace("Isolation", "I.")
                 for m in methods]
        auroc = [res[m]["AUROC"][0] for m in methods]
        auroc_e = [res[m]["AUROC"][1] for m in methods]
        rfpr = [res[m]["rare_mode_FPR"][0] for m in methods]
        x = np.arange(len(methods)); w = 0.38
        d.bar(x - w / 2, auroc, w, yerr=auroc_e, capsize=3, color="#356084", label="AUROC ↑")
        d.bar(x + w / 2, rfpr, w, color="#e0872b", label="rare-mode FPR ↓")
        d.set_xticks(x); d.set_xticklabels(short, fontsize=7.5)
        d.set_ylim(0, 1.05); d.axhline(0.5, ls=":", c="grey", lw=0.8)
        d.set_title("(D) 5-seed comparison — AUROC vs rare-mode FPR")
        d.legend(fontsize=8)
    else:
        d.text(0.5, 0.5, "run run_seeds.py first", ha="center")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT / "figure.png", dpi=140)
    print(f"[out] wrote {OUT/'figure.png'}")


if __name__ == "__main__":
    main()
