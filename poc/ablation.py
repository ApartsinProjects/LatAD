"""
Ablation: does each component (and whitening) contribute, separately and
combined? All terms are co-computed in ONE pass per seed on ONE config (same
VaDE, same split), standardised on train-normal, then fused per configuration -
so the number-by-number comparison is valid.

Component -> term it contributes -> failure mode it targets:
  C1 foundation : nearest-component latent NLL (+ reconstruction)   multimodal normal
  whitening     : Ledoit-Wolf Mahalanobis residual (vs plain sumsq) correlated channels
  C2 basin      : basin instability (1 - agreement)                 rare-mode false POSITIVE
  C3 ensemble   : min per-mode reconstruction error (+ owner-gap)   pocket false NEGATIVE

Run:  C:/Python314/python.exe ablation.py --seeds 3
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

import component2
import component3
from data import make_miim
from evaluate import evaluate
from models_vade import train_vade

device = "cuda" if torch.cuda.is_available() else "cpu"


def _terms(vade, ens, X):
    """All standardisable anomaly terms for points X (higher = more anomalous)."""
    xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        mu = vade.encode(xt)[0]
        xhat = vade.decode(mu)
        r = (xt - xhat).cpu().numpy()
        recon_plain = (r ** 2).sum(1)
        recon_whit = 0.5 * vade.res_whitener.mahalanobis(r)
        density = -vade._log_pz_given_c(mu).max(1).values.cpu().numpy()
    agr, _ = component2.basin_features(vade, X)
    basin = 1.0 - agr
    ens_min, ens_gap = ens.scores(X)
    return {"recon_plain": recon_plain, "recon_whit": recon_whit,
            "density": density, "basin": basin,
            "ens_min": ens_min, "ens_gap": -ens_gap}   # small gap -> anomalous


# configurations: each is a list of term keys (fused, equal weight, z-scored)
CONFIGS = {
    "C1 foundation (plain recon)": ["recon_plain", "density"],
    "+ whitening":                 ["recon_whit", "density"],
    "+ C2 basin":                  ["recon_whit", "density", "basin"],
    "+ C3 ensemble":               ["recon_whit", "density", "ens_min", "ens_gap"],
    "+ C2 + C3 (full)":            ["recon_whit", "density", "basin", "ens_min", "ens_gap"],
}
METRICS = ["AUROC", "AUPRC", "rare_mode_FPR", "pocket_recall", "ood_recall"]


def fuse(keys, tr_terms, te_terms):
    """z-score each term on train-normal, sum -> (train_score, test_score)."""
    tr = np.zeros(len(next(iter(tr_terms.values()))))
    te = np.zeros(len(next(iter(te_terms.values()))))
    for k in keys:
        mu, sd = tr_terms[k].mean(), tr_terms[k].std() + 1e-9
        tr = tr + (tr_terms[k] - mu) / sd
        te = te + (te_terms[k] - mu) / sd
    return tr, te


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    agg = {c: defaultdict(list) for c in CONFIGS}

    for seed in range(args.seeds):
        print(f"=== seed {seed} ===")
        np.random.seed(seed); torch.manual_seed(seed)
        ds, meta = make_miim(n_modes=40, seed=seed)
        vade = train_vade(ds.x_train, n_clusters=40, latent_dim=10, epochs=60,
                          seed=seed, device=device)
        vade.fit_residual_whitener(ds.x_train)
        ens = component3.ClusterEnsemble(vade, ds.x_train)
        tr_terms = _terms(vade, ens, ds.x_train)
        te_terms = _terms(vade, ens, ds.x_test)
        for name, keys in CONFIGS.items():
            tr, te = fuse(keys, tr_terms, te_terms)
            res = evaluate(tr, te, ds, meta)
            for m in METRICS:
                agg[name][m].append(res[m])

    # ---- table ----
    cols = ["AUROC", "AUPRC", "rare_mode_FPR", "pocket_recall"]
    head = f"{'configuration':<30}" + "".join(f"{c:>16}" for c in cols)
    print("\n" + head); print("-" * len(head))
    for name in CONFIGS:
        row = f"{name:<30}"
        for c in cols:
            v = agg[name][c]
            row += f"{np.mean(v):>9.3f}+/-{np.std(v):<4.3f}"
        print(row)
    print("\nrare_mode_FPR lower=better (C2 target); pocket_recall higher=better (C3 target).")


if __name__ == "__main__":
    main()
