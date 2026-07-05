"""
Calibration-free evaluation. Decision thresholds are fixed on NORMAL training
scores only (no labelled faults), matching the proposal's claim that detectors
must be calibrated without abnormal data.

Reported per method:
  - AUROC, AUPRC          : standard ranking quality (threshold-independent).
  - FPR at a 5% operating point set on normal data.
  - rare_mode_FPR         : false-positive rate on rare-but-valid normal modes
                            (the trust-eroding error). Its gap over common-mode
                            FPR is the false-positive story.
  - pocket_recall         : detection rate on 'pocket' faults between two modes
                            (the dangerous false negative).
  - ood_recall            : detection rate on far-outside faults (sanity check).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate(train_scores, test_scores, ds, meta, fpr_target=0.05):
    y = ds.y_test
    thr = np.quantile(train_scores, 1.0 - fpr_target)  # calibration-free op point
    flagged = test_scores > thr

    normal = y == 0
    rare_modes = meta["rare_modes"]
    is_rare = np.array([m in rare_modes for m in ds.mode_test]) & normal
    is_common = (~np.array([m in rare_modes for m in ds.mode_test])) & normal

    def recall(mask):
        return float(flagged[mask].mean()) if mask.any() else float("nan")

    return {
        "AUROC": float(roc_auc_score(y, test_scores)),
        "AUPRC": float(average_precision_score(y, test_scores)),
        "FPR": recall(normal),
        "rare_mode_FPR": recall(is_rare),
        "common_mode_FPR": recall(is_common),
        "pocket_recall": recall(ds.atype_test == "pocket"),
        "ood_recall": recall(ds.atype_test == "ood"),
    }


def format_table(results: dict) -> str:
    cols = ["AUROC", "AUPRC", "rare_mode_FPR", "pocket_recall", "ood_recall"]
    head = f"{'method':<22}" + "".join(f"{c:>15}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        row = f"{name:<22}" + "".join(f"{r[c]:>15.3f}" for c in cols)
        lines.append(row)
    lines.append("")
    lines.append("rare_mode_FPR: lower is better (valid rare modes wrongly alarmed).")
    lines.append("pocket_recall / ood_recall: higher is better (faults caught).")
    return "\n".join(lines)
