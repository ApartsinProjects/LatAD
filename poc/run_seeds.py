"""
Multi-seed driver: repeats the full MIIM experiment over several seeds and
reports mean +/- std per method, so the method ranking rests on error bars
rather than one lucky draw. Everything is co-computed in a single pass per seed
(same data, same split) for a valid number-by-number comparison.

Run:  C:/Python314/python.exe poc/run_seeds.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import baselines
import component2
from data import make_miim
from evaluate import evaluate
from models_vade import train_plain_vae, train_vade

OUT = Path(__file__).parent / "results"
METHODS = ["IsolationForest", "LOF", "AutoEncoder",
           "VAE+GMM (sequential)", "VaDE (joint, ours)",
           "VaDE + basin (full, ours)"]
METRICS = ["AUROC", "AUPRC", "rare_mode_FPR", "common_mode_FPR",
           "pocket_recall", "ood_recall"]


def run_once(seed, n_modes, epochs, latent_dim, device, pretrain, joint_lr):
    ds, meta = make_miim(n_modes=n_modes, seed=seed)
    scores = {}
    scores["IsolationForest"] = baselines.run_iforest(ds.x_train, ds.x_test, seed)
    scores["LOF"] = baselines.run_lof(ds.x_train, ds.x_test, seed)
    scores["AutoEncoder"] = baselines.run_autoencoder(ds.x_train, ds.x_test, seed)

    pv = train_plain_vae(ds.x_train, latent_dim=latent_dim, epochs=max(epochs, pretrain),
                         seed=seed, device=device)
    pv.fit_gmm(ds.x_train, n_clusters=n_modes, seed=seed)
    pv.fit_residual_whitener(ds.x_train)          # whitened residual score
    scores["VAE+GMM (sequential)"] = (pv.anomaly_score(ds.x_train),
                                      pv.anomaly_score(ds.x_test))

    # Proper VaDE recipe: longer pretraining, gentler/shorter joint phase.
    vade = train_vade(ds.x_train, n_clusters=n_modes, latent_dim=latent_dim,
                      pretrain_epochs=pretrain, epochs=epochs, lr=joint_lr,
                      seed=seed, device=device)
    vade.fit_residual_whitener(ds.x_train)
    # Component 1+2: base (recon+density) and basin-augmented (recon+instability).
    vade_variants, _ = component2.vade_scores(vade, ds.x_train, ds.x_test)
    scores.update(vade_variants)

    return {m: evaluate(tr, te, ds, meta) for m, (tr, te) in scores.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n_modes", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=60, help="joint-phase epochs")
    ap.add_argument("--pretrain", type=int, default=30, help="VAE pretrain epochs")
    ap.add_argument("--joint_lr", type=float, default=1e-3)
    ap.add_argument("--latent_dim", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(exist_ok=True)
    agg = {m: defaultdict(list) for m in METHODS}

    for s in range(args.seeds):
        print(f"=== seed {s} ===")
        np.random.seed(s); torch.manual_seed(s)
        res = run_once(s, args.n_modes, args.epochs, args.latent_dim, device,
                       args.pretrain, args.joint_lr)
        for m in METHODS:
            for k in METRICS:
                agg[m][k].append(res[m][k])
        print(f"  VaDE AUROC={res['VaDE (joint, ours)']['AUROC']:.3f}  "
              f"LOF AUROC={res['LOF']['AUROC']:.3f}  "
              f"seq AUROC={res['VAE+GMM (sequential)']['AUROC']:.3f}")

    # ---- summary table: mean +/- std ----
    summary = {m: {k: [float(np.mean(v)), float(np.std(v))]
                   for k, v in agg[m].items()} for m in METHODS}
    (OUT / "results_multiseed.json").write_text(json.dumps(summary, indent=2))

    cols = ["AUROC", "AUPRC", "rare_mode_FPR", "pocket_recall", "ood_recall"]
    head = f"{'method':<26}" + "".join(f"{c:>18}" for c in cols)
    lines = [f"MIIM benchmark, {args.seeds} seeds, {args.n_modes} modes  (mean +/- std)",
             head, "-" * len(head)]
    for m in METHODS:
        row = f"{m:<26}"
        for c in cols:
            mu, sd = summary[m][c]
            row += f"{mu:>10.3f}+/-{sd:<5.3f}"
        lines.append(row)
    lines += ["", "rare_mode_FPR: lower is better.  pocket/ood_recall: higher is better."]
    table = "\n".join(lines)
    (OUT / "results_multiseed.txt").write_text(table)
    print("\n" + table)
    print(f"\n[out] wrote {OUT/'results_multiseed.json'} and .txt")


if __name__ == "__main__":
    main()
