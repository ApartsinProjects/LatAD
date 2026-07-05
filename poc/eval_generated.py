"""
The 'money' evaluation: detect the Component-1 generated faults - channel-shuffle
dependency violations that keep every channel's marginal realistic and break only
the joint structure - and see which detectors catch them, as a function of how
far the fault sits from the nearest operating mode (Mahalanobis sigma).

Faults are mined into a difficulty ladder by their TRUE distance-to-nearest-mode
(model-agnostic), all physically in-range. The thesis prediction: marginal /
global detectors (Isolation Forest) stay blind because the marginals look normal,
while the structure-aware latent+clustering method detects them from a low sigma.

Run:  C:/Python314/python.exe eval_generated.py
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import baselines
import component2
from data import make_miim
from generate import (ModeDistance, TrueNormalOracle, control_difficulty_input,
                      in_range_mask, shuffle_candidates)
from models_vade import train_plain_vae, train_vade

SEED = 0
device = "cuda" if torch.cuda.is_available() else "cpu"


def mine_ladder(ds, md, oracle, n_base=8000, seed=0):
    """Generate shuffle candidates across an alpha sweep, keep only genuine
    (oracle-certified), in-range faults; return them with their sigma distance."""
    rng = np.random.default_rng(seed)
    cand = shuffle_candidates(ds.x_train, ds.mode_train, n_base, swap_frac=0.4, rng=rng)
    pool = [control_difficulty_input(cand, ds.x_train, ds.mode_train, a)
            for a in (0.1, 0.2, 0.35, 0.5, 0.7, 1.0)]
    X = np.concatenate(pool)
    sigma, _ = md.min_sigma(X)
    keep = in_range_mask(X, ds.x_train) & oracle.is_anomaly(X, q=0.99)  # physical + real anomaly
    return X[keep], sigma[keep]


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    ds, meta = make_miim(n_modes=40, seed=SEED)
    md = ModeDistance(ds.x_train, ds.mode_train)     # true modes + labels (not the detector)
    oracle = TrueNormalOracle(meta["oracle_normal"], k=5)

    # sigma scale of genuine normal -> the "normal envelope" for relative bands.
    nsig, _ = md.min_sigma(ds.x_test[ds.y_test == 0])
    env = float(np.percentile(nsig, 99))          # 1 "envelope" = normal p99 sigma
    print(f"normal min-sigma: p50={np.percentile(nsig,50):.2f} "
          f"p95={np.percentile(nsig,95):.2f} p99(env)={env:.2f}")

    Xf, sig = mine_ladder(ds, md, oracle, seed=SEED)
    ratio = sig / env                             # distance in envelope units
    print(f"mined faults: {len(Xf)}  sigma range {sig.min():.1f}-{sig.max():.1f} "
          f"(x envelope: {ratio.min():.2f}-{ratio.max():.2f})")

    # ---- train detectors (all on normal train only) ----
    xtr, xte_norm = ds.x_train, ds.x_test[ds.y_test == 0]
    scorers = {}
    for name, fn in [("IsolationForest", baselines.run_iforest),
                     ("LOF", baselines.run_lof),
                     ("AutoEncoder", baselines.run_autoencoder)]:
        tr, te = fn(xtr, np.concatenate([xte_norm, Xf]), SEED)
        scorers[name] = (tr, te[:len(xte_norm)], te[len(xte_norm):])

    pv = train_plain_vae(xtr, latent_dim=10, epochs=60, seed=SEED, device=device)
    pv.fit_gmm(xtr, 40, SEED); pv.fit_residual_whitener(xtr)
    scorers["VAE+GMM (seq)"] = (pv.anomaly_score(xtr), pv.anomaly_score(xte_norm),
                                pv.anomaly_score(Xf))

    vade = train_vade(xtr, n_clusters=40, latent_dim=10, epochs=60, seed=SEED, device=device)
    vade.fit_residual_whitener(xtr)
    vv, _ = component2.vade_scores(vade, xtr, np.concatenate([xte_norm, Xf]))
    for name, (tr, te) in vv.items():
        nm = name.replace(" (joint, ours)", " (joint)").replace(" + basin (full, ours)", "+basin")
        scorers[nm] = (tr, te[:len(xte_norm)], te[len(xte_norm):])

    # Difficulty bands in ENVELOPE units (x normal-p99 sigma). Drop faults inside
    # the normal envelope (<1x = accidental normals) and easy far tail (>=4x);
    # the headline is the BOUNDARY SHELL [1x, 4x). AUROC is prevalence-invariant
    # so no balancing needed; AUPRC is dropped (it would just track prevalence).
    bands = [(1, 1.5), (1.5, 2), (2, 4), (4, 999)]
    shell = (ratio >= 1) & (ratio < 4)
    rng = np.random.default_rng(SEED)
    if shell.sum() == 0:
        raise SystemExit("no faults in boundary shell [1-4x envelope]; widen the band")

    # Split held-out normal into a calibration half (sets the threshold) and an
    # evaluation half (measures realized FPR) - a random split, since xte_norm is
    # ordered common-then-rare. Threshold = 5% FPR on realistic normal (rare modes
    # INCLUDED), so TPR@5%FPR rewards separating rare-valid modes from faults.
    perm = rng.permutation(len(xte_norm))
    cal_i, evl_i = perm[:len(perm) // 2], perm[len(perm) // 2:]

    print(f"\nboundary shell [1-4x envelope]: {int(shell.sum())} faults; "
          f"threshold = 5% FPR on normal (rare included)")
    print(f"{'method':<20}{'AUROC':>8}{'realFPR':>9}{'TPR@5%FPR':>11}")
    detail = {}
    for name, (tr, sn, sf) in scorers.items():
        sn, sf = np.asarray(sn), np.asarray(sf)
        thr = np.quantile(sn[cal_i], 0.95)
        real_fpr = float((sn[evl_i] > thr).mean())
        y = np.r_[np.zeros(len(evl_i)), np.ones(int(shell.sum()))]
        s = np.r_[sn[evl_i], sf[shell]]
        auroc = roc_auc_score(y, s)
        tpr = float((sf[shell] > thr).mean())
        print(f"{name:<20}{auroc:>8.3f}{real_fpr:>9.3f}{tpr:>11.3f}")
        detail[name] = [float((sf[(ratio >= a) & (ratio < b)] > thr).mean())
                        if ((ratio >= a) & (ratio < b)).any() else float('nan')
                        for a, b in bands]

    # ---- TPR@5%FPR vs difficulty (envelope units) = the ranking curve ----
    band_hdr = "".join(f"{f'{a}-{b}x':>9}" for a, b in bands)
    print(f"\nTPR@5%FPR by distance band (x normal envelope):")
    print(f"{'method':<20}{band_hdr}")
    counts = [int(((ratio >= a) & (ratio < b)).sum()) for a, b in bands]
    print(f"{'(n faults)':<20}" + "".join(f"{c:>9}" for c in counts))
    for name, rates in detail.items():
        print(f"{name:<20}" + "".join(f"{r:>9.2f}" for r in rates))


if __name__ == "__main__":
    main()
