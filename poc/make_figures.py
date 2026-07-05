"""
Result figures, redesigned for intuition. One 2x2 panel:

  (A) Difficulty ladder: detection (TPR @ 5% FPR) versus how far the fault sits
      from the nearest normal mode (envelope units). A line per method - the
      single most intuitive view: global/reconstruction detectors are flat near
      zero, structure-aware detection rises steeply and detects even hard faults.
  (B) Per-mode false-alarm rate (C4): each bar is one operating mode's FPR under
      a global threshold (many rare modes alarm constantly) versus mode-conditional
      thresholds (all controlled). Shows the worst-case story the mean hides.
  (C) Basin agreement (C2): rare-but-valid modes sit in one stable basin
      (agreement -> 1); pocket faults split between two (agreement -> 0.5).
  (D) Latent map (VaDE, PCA): normal coloured by mode, faults overlaid - orients
      the reader to the multimodal-normal + fault geometry.

Run AFTER the experiment re-runs (needs a free GPU). C:/Python314/python.exe make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import baselines
import component2
import component4
from data import make_miim
from generate import (ModeDistance, TrueNormalOracle, control_difficulty_input,
                      in_range_mask, shuffle_candidates)
from models_vade import train_vade

OUT = Path(__file__).parent / "results"
SEED = 0
COL = {"IsolationForest": "#888", "LOF": "#c44", "AutoEncoder": "#c9a227",
       "VaDE (joint)": "#356084", "VaDE+basin": "#15212e"}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(SEED); torch.manual_seed(SEED)
    ds, meta = make_miim(n_modes=40, seed=SEED)
    md = ModeDistance(ds.x_train, ds.mode_train)
    oracle = TrueNormalOracle(meta["oracle_normal"])
    vade = train_vade(ds.x_train, n_clusters=40, latent_dim=10, epochs=60, seed=SEED, device=dev)
    vade.fit_residual_whitener(ds.x_train)

    # ---- mine C1 faults on a difficulty ladder ----
    rng = np.random.default_rng(SEED)
    cand = shuffle_candidates(ds.x_train, ds.mode_train, 8000, swap_frac=0.4, rng=rng)
    pool = np.concatenate([control_difficulty_input(cand, ds.x_train, ds.mode_train, a)
                           for a in (0.1, 0.2, 0.35, 0.5, 0.7, 1.0)])
    keep = in_range_mask(pool, ds.x_train) & oracle.is_anomaly(pool)
    Xf = pool[keep]
    ratio = md.min_sigma(Xf)[0] / float(np.percentile(md.min_sigma(ds.x_test[ds.y_test == 0])[0], 99))
    xtr, xnorm = ds.x_train, ds.x_test[ds.y_test == 0]

    # scores for each method on normal + faults
    scores = {}
    for name, fn in [("IsolationForest", baselines.run_iforest), ("LOF", baselines.run_lof),
                     ("AutoEncoder", baselines.run_autoencoder)]:
        tr, te = fn(xtr, np.concatenate([xnorm, Xf]), SEED)
        scores[name] = (tr, te[:len(xnorm)], te[len(xnorm):])
    vv, _ = component2.vade_scores(vade, xtr, np.concatenate([xnorm, Xf]))
    for nm, key in [("VaDE (joint, ours)", "VaDE (joint)"), ("VaDE + basin (full, ours)", "VaDE+basin")]:
        tr, te = vv[nm]; scores[key] = (tr, te[:len(xnorm)], te[len(xnorm):])

    edges = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]

    fig, ax = plt.subplots(2, 2, figsize=(13, 10.5))

    # ---- (A) difficulty ladder ----
    a = ax[0, 0]
    for name, (tr, sn, sf) in scores.items():
        thr = np.quantile(sn, 0.95)      # 5% FPR on normal
        tpr = [float((sf[(ratio >= edges[i]) & (ratio < edges[i + 1])] > thr).mean())
               if ((ratio >= edges[i]) & (ratio < edges[i + 1])).any() else np.nan
               for i in range(len(edges) - 1)]
        a.plot(centers, tpr, "-o", ms=4, color=COL[name], label=name,
               lw=2.2 if "VaDE" in name else 1.3)
    a.set_xlabel("fault distance from nearest mode  (× normal envelope)")
    a.set_ylabel("detection rate  (TPR @ 5% FPR)")
    a.set_title("(A) Difficulty ladder — detection vs how hard the fault is")
    a.set_ylim(-0.03, 1.03); a.grid(alpha=.25); a.legend(fontsize=8, loc="upper left")

    # ---- (B) per-mode FPR: global vs mode-conditional ----
    b = ax[0, 1]
    tr_v, te_v = vv["VaDE + basin (full, ours)"]
    normal = ds.y_test == 0
    # te_v is scored on [xnorm, Xf]; the first len(xnorm) entries are the normal
    # test points, in the same order as ds.x_test[normal].
    sn_v, modes_n = te_v[:len(xnorm)], ds.mode_test[normal]
    gthr = float(np.quantile(tr_v, 0.95))
    fpr_g = component4.per_mode_fpr(sn_v, modes_n, gthr, min_count=15)
    trc = component4.assign_clusters(vade, xtr); tec = component4.assign_clusters(vade, ds.x_test)
    thr_d = component4.mode_conditional_thresholds(tr_v, trc, 0.05, global_thr=gthr)
    flag_c = component4.apply_mode_conditional(sn_v, tec[normal], thr_d, gthr)
    fpr_c = {m: flag_c[modes_n == m].mean() for m in fpr_g}
    order = sorted(fpr_g, key=lambda m: -fpr_g[m])
    xs = np.arange(len(order))
    b.bar(xs - 0.2, [fpr_g[m] for m in order], 0.4, color="#c44", label="global threshold")
    b.bar(xs + 0.2, [fpr_c[m] for m in order], 0.4, color="#2f8f7f", label="mode-conditional")
    b.axhline(0.05, ls=":", c="k", lw=.8, label="5% target")
    b.set_xlabel("operating mode (sorted by global FPR)")
    b.set_ylabel("per-mode false-alarm rate")
    b.set_title("(B) C4 — a global threshold lets rare modes alarm constantly")
    b.legend(fontsize=8)

    # ---- (C) basin agreement ----
    c = ax[1, 0]
    agr, _ = component2.basin_features(vade, ds.x_test)
    rare = set(meta["rare_modes"])
    is_rare = np.array([m in rare for m in ds.mode_test]) & normal
    pkt = ds.atype_test == "pocket"
    bins = np.linspace(0, 1, 26)
    c.hist(agr[is_rare], bins, density=True, alpha=.7, color="#3b6fb0",
           label=f"rare-valid mode  (mean {agr[is_rare].mean():.2f})")
    c.hist(agr[pkt], bins, density=True, alpha=.7, color="#111",
           label=f"pocket fault  (mean {agr[pkt].mean():.2f})")
    c.set_xlabel("basin agreement  (1 = one stable mode, 0.5 = split)")
    c.set_ylabel("density"); c.set_title("(C) C2 — basin test tells rare-valid from pocket")
    c.legend(fontsize=8, loc="upper left")

    # ---- (D) latent map ----
    d = ax[1, 1]
    with torch.no_grad():
        z = vade.encode(torch.as_tensor(ds.x_test, dtype=torch.float32, device=dev))[0].cpu().numpy()
    z2 = z - z.mean(0); _, _, vt = np.linalg.svd(z2, full_matrices=False); p = z2 @ vt[:2].T
    idx = np.random.default_rng(0).choice(np.where(normal)[0], min(2500, normal.sum()), replace=False)
    d.scatter(p[idx, 0], p[idx, 1], c=ds.mode_test[idx], cmap="tab20", s=8, alpha=.5, linewidths=0)
    d.scatter(p[pkt, 0], p[pkt, 1], marker="x", c="#111", s=22, label="pocket")
    d.scatter(p[ds.atype_test == "ood", 0], p[ds.atype_test == "ood", 1], marker="+",
              c="#e0872b", s=22, label="ood")
    d.set_title("(D) VaDE latent (PCA) — normal modes + faults")
    d.set_xlabel("PC1"); d.set_ylabel("PC2"); d.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT / "figure.png", dpi=140)
    print(f"[out] wrote {OUT/'figure.png'}")


if __name__ == "__main__":
    main()
