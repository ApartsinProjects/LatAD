"""
Prior-research proof of concept for the LatAD proposal.

Validates the core claim: on CPS-style normal data with Massive, Implicit,
Imbalanced Multimodality (MIIM), a JOINT latent-encoding-and-clustering detector
(VaDE) reduces both characteristic errors named in the proposal - false alarms
on rare-but-valid modes, and missed faults hiding between modes - relative to
standard anomaly-detection baselines and to a sequential (encode-then-cluster)
ablation.

Run:  C:/Python314/python.exe poc/run.py
Switch to real data later by replacing make_miim(...) with load_wadi(root).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import baselines
from data import make_miim
from evaluate import evaluate, format_table
from models_vade import train_plain_vae, train_vade

OUT = Path(__file__).parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_modes", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--latent_dim", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="tiny run to smoke-test wiring")
    args = ap.parse_args()

    if args.quick:
        args.n_modes, args.epochs = 8, 8

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"device={device}  seed={args.seed}  n_modes={args.n_modes}")

    # ---------------- data ----------------
    ds, meta = make_miim(n_modes=args.n_modes, seed=args.seed)
    n_rare = len(meta["rare_modes"])
    print(f"[data] {ds.name}: train={len(ds.x_train)} test={len(ds.x_test)} "
          f"feats={ds.n_features} modes={args.n_modes} ({n_rare} rare)")
    print(f"[data] rarest train mode has {int(meta['train_counts'].min())} samples, "
          f"most common {int(meta['train_counts'].max())}")

    scores = {}          # method -> (train_scores, test_scores)
    t0 = time.time()

    # ---------------- baselines (PyOD) ----------------
    print("[fit] Isolation Forest ...")
    scores["IsolationForest"] = baselines.run_iforest(ds.x_train, ds.x_test, args.seed)
    print("[fit] LOF ...")
    scores["LOF"] = baselines.run_lof(ds.x_train, ds.x_test, args.seed)
    print("[fit] AutoEncoder ...")
    scores["AutoEncoder"] = baselines.run_autoencoder(ds.x_train, ds.x_test, args.seed)

    # ---------------- sequential ablation: VAE then GMM ----------------
    print("[fit] Plain VAE + post-hoc GMM (sequential) ...")
    pv = train_plain_vae(ds.x_train, latent_dim=args.latent_dim,
                         epochs=args.epochs, seed=args.seed, device=device)
    pv.fit_gmm(ds.x_train, n_clusters=args.n_modes, seed=args.seed)
    scores["VAE+GMM (sequential)"] = (
        pv.anomaly_score(torch.as_tensor(ds.x_train, device=device)),
        pv.anomaly_score(torch.as_tensor(ds.x_test, device=device)))

    # ---------------- our approach: joint VaDE ----------------
    print("[fit] VaDE (joint latent + clustering) ...")
    vade = train_vade(ds.x_train, n_clusters=args.n_modes, latent_dim=args.latent_dim,
                      epochs=args.epochs, seed=args.seed, device=device, verbose=True)
    scores["VaDE (joint, ours)"] = (
        vade.anomaly_score(torch.as_tensor(ds.x_train, device=device)),
        vade.anomaly_score(torch.as_tensor(ds.x_test, device=device)))

    print(f"[fit] all methods done in {time.time() - t0:.0f}s")

    # ---------------- evaluate ----------------
    results = {name: evaluate(tr, te, ds, meta) for name, (tr, te) in scores.items()}
    table = format_table(results)
    print("\n" + table + "\n")

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    (OUT / "results.txt").write_text(table)
    print(f"[out] wrote {OUT/'results.json'} and results.txt")

    # ---------------- figure (optional) ----------------
    try:
        make_figure(scores, ds, vade, device)
        print(f"[out] wrote {OUT/'figure.png'}")
    except Exception as e:  # pragma: no cover - plotting is a nicety
        print(f"[out] figure skipped: {e}")


def make_figure(scores, ds, vade, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) VaDE latent (2D via PCA) coloured by ground-truth mode, faults overlaid.
    with torch.no_grad():
        z = vade.encode(torch.as_tensor(ds.x_test, dtype=torch.float32, device=device))[0].cpu().numpy()
    z2 = z - z.mean(0)
    u, s, vt = np.linalg.svd(z2, full_matrices=False)
    p = z2 @ vt[:2].T
    normal = ds.y_test == 0
    ax = axes[0]
    ax.scatter(p[normal, 0], p[normal, 1], c=ds.mode_test[normal], s=5,
               cmap="tab20", alpha=0.5, linewidths=0)
    ax.scatter(p[ds.atype_test == "pocket", 0], p[ds.atype_test == "pocket", 1],
               marker="x", c="black", s=28, label="pocket fault")
    ax.scatter(p[ds.atype_test == "ood", 0], p[ds.atype_test == "ood", 1],
               marker="+", c="red", s=28, label="ood fault")
    ax.set_title("VaDE latent (PCA): normal modes + injected faults")
    ax.legend(loc="upper right", fontsize=8)

    # (b) score distributions for our method: normal vs pocket vs ood.
    _, te = scores["VaDE (joint, ours)"]
    ax = axes[1]
    for lbl, mask, c in [("normal", ds.atype_test == "none", "#2f8f7f"),
                         ("pocket fault", ds.atype_test == "pocket", "#356084"),
                         ("ood fault", ds.atype_test == "ood", "#e0872b")]:
        ax.hist(te[mask], bins=60, alpha=0.6, density=True, label=lbl, color=c)
    ax.set_title("VaDE anomaly score by group")
    ax.set_xlabel("anomaly score"); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "figure.png", dpi=130)


if __name__ == "__main__":
    main()
