"""
Component 4 - mode-conditional risk control (evaluation + decision rule).

Motivation: an average false-alarm rate hides SYSTEMATIC per-mode failure. A rare
but legitimate mode (e.g. a vehicle moving backwards) may be alarmed on EVERY time
it occurs; averaged over all operation this barely moves the mean FPR, yet in the
field it is a guaranteed nuisance alarm whenever that mode is entered. The
operationally meaningful quantity is therefore not the mean but the WORST-mode (or
a high-quantile) false-alarm rate.

Two parts:
  1. Evaluation:  per-mode FPR, and its worst-case / high-quantile summary.
  2. Decision rule: MODE-CONDITIONAL thresholds. Assign each point to its
     discovered cluster and threshold against that cluster's own normal-score
     distribution, so every mode is individually held at the target FPR - a
     per-mode guarantee, not just an average. Needs no true labels (uses the
     clusters discovered by the foundation).
"""

from __future__ import annotations

import numpy as np
import torch

from models_vade import _as_tensor


def assign_clusters(vade, X):
    with torch.no_grad():
        z = vade.encode(_as_tensor(X, vade))[0]
        return vade._log_pz_given_c(z).argmax(1).cpu().numpy()


def per_mode_fpr(scores_normal, modes_normal, threshold, min_count=10):
    """FPR within each mode that has >= min_count normal points."""
    out = {}
    for m in np.unique(modes_normal):
        s = scores_normal[modes_normal == m]
        if len(s) >= min_count:
            out[int(m)] = float((s > threshold).mean())
    return out


def threshold_at_tpr(fault_scores, target_tpr):
    """Threshold that achieves a target detection rate on known faults, so
    per-mode FPR is compared across detectors at MATCHED sensitivity (an
    insensitive detector cannot 'win' a low FPR by flagging nothing)."""
    return float(np.quantile(np.asarray(fault_scores), 1.0 - target_tpr))


def tail_fpr(scores_normal, modes_normal, threshold, p=0.90, min_count=20):
    """TFAR_p: p-quantile of per-mode FPR over modes with >= min_count normal
    points (drops extreme tiny clusters), plus the worst-decile mean (CVaR)."""
    v = np.array(list(per_mode_fpr(scores_normal, modes_normal, threshold, min_count).values()))
    if len(v) == 0:
        return {"tfar_p": float("nan"), "cvar_10": float("nan"), "n_modes": 0}
    q = float(np.quantile(v, p))
    tail = v[v >= np.quantile(v, 0.90)]
    return {"tfar_p": q, "cvar_10": float(tail.mean()) if len(tail) else q,
            "n_modes": int(len(v))}


def risk_summary(scores_normal, modes_normal, threshold, min_count=20):
    # min_count=20 keeps the per-mode FPR (and its worst-case order statistic)
    # from being dominated by tiny-sample rare modes; still a small-sample metric.
    fpr = per_mode_fpr(scores_normal, modes_normal, threshold, min_count)
    v = np.array(list(fpr.values()))
    if len(v) == 0:
        return {"mean_fpr": float((scores_normal > threshold).mean()),
                "mean_mode_fpr": float("nan"), "worst_mode_fpr": float("nan"),
                "q90_mode_fpr": float("nan"), "tfar_90": float("nan"),
                "cvar_10": float("nan"), "modes_over_50pct": 0, "n_modes": 0}
    tail = tail_fpr(scores_normal, modes_normal, threshold, 0.90, min_count)
    return {"tfar_90": tail["tfar_p"], "cvar_10": tail["cvar_10"],
        "mean_fpr": float((scores_normal > threshold).mean()),
        "mean_mode_fpr": float(v.mean()),
        "worst_mode_fpr": float(v.max()),
        "q90_mode_fpr": float(np.quantile(v, 0.90)),
        "modes_over_50pct": int((v > 0.5).sum()),
        "n_modes": len(v),
    }


def mode_conditional_thresholds(train_scores, train_clusters, target_fpr=0.05,
                                min_count=10, global_thr=None):
    """Per-cluster threshold = (1-target) quantile of that cluster's train-normal
    scores. Clusters with too few samples fall back to the global threshold."""
    thr = {}
    for c in np.unique(train_clusters):
        s = train_scores[train_clusters == c]
        thr[c] = (float(np.quantile(s, 1 - target_fpr))
                  if len(s) >= min_count else global_thr)
    return thr


def apply_mode_conditional(test_scores, test_clusters, thr_dict, global_thr):
    t = np.array([thr_dict.get(int(c), global_thr) for c in test_clusters])
    return test_scores > t


if __name__ == "__main__":
    from data import make_miim
    from models_vade import train_vade
    import component2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, meta = make_miim(n_modes=40, seed=0)
    vade = train_vade(ds.x_train, n_clusters=40, latent_dim=10, epochs=60,
                      seed=0, device=device)
    vade.fit_residual_whitener(ds.x_train)

    # anomaly score (recon + basin, the C2-emphasizing config)
    (variants, _) = component2.vade_scores(vade, ds.x_train, ds.x_test)
    tr, te = variants["VaDE + basin (full, ours)"]

    normal = ds.y_test == 0
    sn, modes_n = te[normal], ds.mode_test[normal]
    # Fixed 5% alarm-budget operating point: this is where the per-mode
    # HETEROGENEITY is the story (some modes consume the whole budget). Robust
    # metrics: TFAR90 (90th-pct per-mode FPR, tiny clusters dropped) and CVaR10.
    global_thr = float(np.quantile(tr, 0.95))
    # (also report at matched detection so an insensitive detector cannot win)
    matched_thr = threshold_at_tpr(te[ds.y_test == 1], 0.80)

    print("GLOBAL threshold (5% mean-FPR budget):")
    r = risk_summary(sn, modes_n, global_thr)
    print(f"  mean FPR {r['mean_fpr']:.3f} | TFAR90 {r['tfar_90']:.3f} | "
          f"CVaR10 {r['cvar_10']:.3f} | worst {r['worst_mode_fpr']:.3f} | "
          f"modes>50%FPR {r['modes_over_50pct']}/{r['n_modes']}")
    rm = risk_summary(sn, modes_n, matched_thr)
    print(f"  [matched TPR=0.8]  TFAR90 {rm['tfar_90']:.3f} | CVaR10 {rm['cvar_10']:.3f}")

    # mode-conditional decision rule using DISCOVERED clusters
    tr_clusters = assign_clusters(vade, ds.x_train)
    te_clusters = assign_clusters(vade, ds.x_test)
    thr_dict = mode_conditional_thresholds(tr, tr_clusters, 0.05, global_thr=global_thr)
    flagged_n = apply_mode_conditional(sn, te_clusters[normal], thr_dict, global_thr)

    # recompute per-mode FPR under the conditional rule (min_count=20)
    fpr_c = {}
    for m in np.unique(modes_n):
        mask = modes_n == m
        if mask.sum() >= 20:
            fpr_c[m] = flagged_n[mask].mean()
    v = np.array(list(fpr_c.values()))
    print("MODE-CONDITIONAL thresholds (per-cluster 5% FPR guarantee):")
    print(f"  mean FPR {flagged_n.mean():.3f} | TFAR90 {np.quantile(v,0.9):.3f} | "
          f"CVaR10 {v[v>=np.quantile(v,0.9)].mean():.3f} | worst {v.max():.3f} | "
          f"modes>50%FPR {(v>0.5).sum()}/{len(v)}")

    # detection cost of the guarantee
    for tag, mask in [("pocket", ds.atype_test == "pocket"), ("ood", ds.atype_test == "ood")]:
        g = float((te[mask] > global_thr).mean())
        c = float(apply_mode_conditional(te[mask], te_clusters[mask], thr_dict, global_thr).mean())
        print(f"  detection {tag}: global {g:.3f} -> mode-conditional {c:.3f}")
