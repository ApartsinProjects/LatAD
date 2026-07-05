"""
Framework vs baselines on real SKAB data (first real-CPS validation).

Calibration-free: thresholds set on training-normal scores only. Reports AUROC,
AUPRC, and TPR at a 5%-FPR operating point (realized FPR shown). Also runs the
Component-4 mode-conditional risk check on the DISCOVERED clusters (SKAB has no
ground-truth modes), to see whether real data also shows per-cluster false-alarm
heterogeneity.

Run:  C:/Python314/python.exe run_skab.py
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import baselines
import component2
import component4
from models_vade import train_vade
from skab import load_skab

device = "cuda" if torch.cuda.is_available() else "cpu"
N_CLUSTERS = 20


def main():
    np.random.seed(0); torch.manual_seed(0)
    ds = load_skab(W=20, stride=4)
    print(f"{ds.name}: train={len(ds.x_train)} test={len(ds.x_test)} "
          f"feats={ds.n_features} anomaly-frac={ds.y_test.mean():.2f}")
    y = ds.y_test
    normal = y == 0

    scorers = {}
    for name, fn in [("IsolationForest", baselines.run_iforest),
                     ("LOF", baselines.run_lof),
                     ("AutoEncoder", baselines.run_autoencoder)]:
        tr, te = fn(ds.x_train, ds.x_test, 0)
        scorers[name] = (tr, te)

    vade = train_vade(ds.x_train, n_clusters=N_CLUSTERS, latent_dim=10, epochs=60,
                      seed=0, device=device)
    vade.fit_residual_whitener(ds.x_train)
    vv, _ = component2.vade_scores(vade, ds.x_train, ds.x_test)
    for nm, key in [("VaDE (joint, ours)", "VaDE (joint)"),
                    ("VaDE + basin (full, ours)", "VaDE + basin")]:
        scorers[key] = vv[nm]

    print(f"\n{'method':<20}{'AUROC':>8}{'AUPRC':>8}{'realFPR':>9}{'TPR@5%FPR':>11}")
    for name, (tr, te) in scorers.items():
        thr = float(np.quantile(tr, 0.95))
        realfpr = float((te[normal] > thr).mean())
        tpr = float((te[y == 1] > thr).mean())
        print(f"{name:<20}{roc_auc_score(y, te):>8.3f}{average_precision_score(y, te):>8.3f}"
              f"{realfpr:>9.3f}{tpr:>11.3f}")

    # ---- C4 on discovered clusters ----
    tr_v, te_v = vv["VaDE + basin (full, ours)"]
    trc = component4.assign_clusters(vade, ds.x_train)
    tec = component4.assign_clusters(vade, ds.x_test)
    gthr = float(np.quantile(tr_v, 0.95))
    rg = component4.risk_summary(te_v[normal], tec[normal], gthr, min_count=20)
    thr_d = component4.mode_conditional_thresholds(tr_v, trc, 0.05, global_thr=gthr)
    flag = component4.apply_mode_conditional(te_v[normal], tec[normal], thr_d, gthr)
    # per-cluster FPR under conditional rule
    v = np.array([flag[tec[normal] == c].mean() for c in np.unique(tec[normal])
                  if (tec[normal] == c).sum() >= 20])
    print(f"\nC4 (discovered clusters, {rg['n_modes']} with >=20 pts):")
    print(f"  global 5% budget : mean {rg['mean_fpr']:.3f} | TFAR90 {rg['tfar_90']:.3f} | "
          f"clusters>50% {rg['modes_over_50pct']}/{rg['n_modes']}")
    print(f"  mode-conditional : mean {flag.mean():.3f} | TFAR90 {np.quantile(v,0.9):.3f} | "
          f"clusters>50% {(v>0.5).sum()}/{len(v)}")
    tpr_g = float((te_v[y == 1] > gthr).mean())
    tpr_c = float(component4.apply_mode_conditional(te_v[y == 1], tec[y == 1], thr_d, gthr).mean())
    print(f"  anomaly detection: global {tpr_g:.3f} -> mode-conditional {tpr_c:.3f}")


if __name__ == "__main__":
    main()
