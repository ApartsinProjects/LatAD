"""
Framework vs baselines on real CPS data (SKAB, HAI).

Threshold calibrated on a held-out split of the TEST-normal windows (real CPS test
sessions drift from the training session, so a train-calibrated threshold is
invalid; calibrating on recent known-normal operation is the realistic protocol
and not label leakage). Reports AUROC, AUPRC (the meaningful metric under the low
anomaly base rate), realized FPR, and TPR@5%FPR, plus the C4 per-cluster
false-alarm check on the DISCOVERED clusters.

Run:  C:/Python314/python.exe run_real.py skab
      C:/Python314/python.exe run_real.py hai
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import baselines
import component2
import component4
from models_vade import train_vade

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(name):
    np.random.seed(0); torch.manual_seed(0)
    if name == "skab":
        from skab import load_skab
        ds = load_skab(W=20, stride=10, rep="stats")
    elif name == "hai":
        from hai import load_hai
        ds = load_hai(W=60, stride=60, rep="stats")
    else:
        raise SystemExit("usage: run_real.py [skab|hai]")

    y = ds.y_test
    print(f"{ds.name}: train={len(ds.x_train)} test={len(ds.x_test)} "
          f"feats={ds.n_features} anomaly-frac={y.mean():.3f}")
    rng = np.random.default_rng(0)
    nidx = rng.permutation(np.where(y == 0)[0])
    cal, evl = nidx[:len(nidx) // 2], nidx[len(nidx) // 2:]

    scores = {}
    for nm, fn in [("IsolationForest", baselines.run_iforest),
                   ("LOF", baselines.run_lof), ("AutoEncoder", baselines.run_autoencoder)]:
        scores[nm] = fn(ds.x_train, ds.x_test, 0)[1]
    vade = train_vade(ds.x_train, n_clusters=20, latent_dim=10, epochs=60, seed=0, device=device)
    vade.fit_residual_whitener(ds.x_train)
    vv, _ = component2.vade_scores(vade, ds.x_train, ds.x_test)
    scores["VaDE (joint)"] = vv["VaDE (joint, ours)"][1]
    scores["VaDE + basin"] = vv["VaDE + basin (full, ours)"][1]

    print(f"\n{'method':<18}{'AUROC':>8}{'AUPRC':>8}{'realFPR':>9}{'TPR@5%FPR':>11}")
    for nm, te in scores.items():
        thr = float(np.quantile(te[cal], 0.95))
        print(f"{nm:<18}{roc_auc_score(y, te):>8.3f}{average_precision_score(y, te):>8.3f}"
              f"{float((te[evl] > thr).mean()):>9.3f}{float((te[y == 1] > thr).mean()):>11.3f}")

    # ---- C4 on discovered clusters ----
    tr_v = vv["VaDE + basin (full, ours)"][0]; te_v = scores["VaDE + basin"]
    trc = component4.assign_clusters(vade, ds.x_train)
    tec = component4.assign_clusters(vade, ds.x_test)
    gthr = float(np.quantile(te_v[cal], 0.95))
    normal = y == 0
    r = component4.risk_summary(te_v[evl], tec[evl], gthr, min_count=20)
    print(f"\nC4 (discovered clusters): global 5% budget -> mean FPR {r['mean_fpr']:.3f}, "
          f"TFAR90 {r['tfar_90']:.3f}, clusters>50% {r['modes_over_50pct']}/{r['n_modes']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hai")
