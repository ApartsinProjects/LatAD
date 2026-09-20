# Leak-free TPR@FPR and AUPRC, Difficult subset (train-calibrated thresholds)

Threshold = q(1-FPR) of the method's TRAIN-NORMAL score distribution (never test data). TPR = recall on Difficult-subset anomaly windows ((label==1) & (maxz<=maxz_thr)). Realized test-normal FPR is reported to show transfer. AUPRC (average_precision_score) is threshold-free, computed on the same Difficult-vs-all-test-normal set.


## WADI_clean

- n_test=575, anomalies=56, difficult=30, test_normals=519
- prevalence (difficult / (difficult+test_normal)) = 0.0546 (AUPRC chance baseline; report AUPRC / prevalence as lift)
- calibration source for LatAD (regime-community): calib_surprise ONLY (held-out 20% slice; expert bundle lacks fit_surprise)
- AUROC cross-check (must reproduce known values): {'regime_community_headline_Difficult': 0.771}
- degenerate all-flag sanity: {'tpr': 1.0, 'test_fpr': 1.0} (expect tpr=1.0, test_fpr=1.0)

| Method | AUPRC | AUPRC/prevalence | TPR@1%FPR | train FPR@1% | test FPR@1% | TPR@5%FPR | train FPR@5% | test FPR@5% |
|---|---|---|---|---|---|---|---|---|
| trivial max|z| | 0.0718 | 1.314 | 0.0 | 0.0103 | 0.0077 | 0.0 | 0.0501 | 0.0366 |
| IF | 0.1102 | 2.017 | 0.0533 | 0.0103 | 0.0104 | 0.1266 | 0.0501 | 0.0409 |
| AE | 0.0985 | 1.803 | 0.02 | 0.0103 | 0.0066 | 0.1 | 0.0501 | 0.0351 |
| LinRes | 0.4999 | 9.148 | 0.3667 | 0.0103 | 0.0058 | 0.5667 | 0.0501 | 0.0751 |
| LatAD (global density) | 0.0943 | 1.726 | 0.02 | 0.0103 | 0.0116 | 0.1333 | 0.0501 | 0.0528 |
| LatAD (regime-community) | 0.2925 | 5.353 | 0.26 | 0.0115 | 0.0174 | 0.4867 | 0.0516 | 0.059 |
| USAD | 0.0891 | 1.631 | NA | NA | NA | NA | NA | NA |
  - *USAD*: no leak-free train-normal score available (external SOTA harness; test-only dump)
| TranAD | 0.1016 | 1.859 | NA | NA | NA | NA | NA | NA |
  - *TranAD*: no leak-free train-normal score available (external SOTA harness; test-only dump)

## HAI

- n_test=14819, anomalies=652, difficult=167, test_normals=14167
- prevalence (difficult / (difficult+test_normal)) = 0.0117 (AUPRC chance baseline; report AUPRC / prevalence as lift)
- calibration source for LatAD (regime-community): calib_surprise ONLY (held-out 20% slice; expert bundle lacks fit_surprise)
- AUROC cross-check (must reproduce known values): {'regime_community_headline_Difficult': 0.845}
- degenerate all-flag sanity: {'tpr': 1.0, 'test_fpr': 1.0} (expect tpr=1.0, test_fpr=1.0)

| Method | AUPRC | AUPRC/prevalence | TPR@1%FPR | train FPR@1% | test FPR@1% | TPR@5%FPR | train FPR@5% | test FPR@5% |
|---|---|---|---|---|---|---|---|---|
| trivial max|z| | 0.0083 | 0.712 | 0.0 | 0.01 | 0.2797 | 0.018 | 0.05 | 0.2821 |
| IF | 0.0242 | 2.077 | 0.1485 | 0.01 | 0.0809 | 0.3246 | 0.05 | 0.1819 |
| AE | 0.2418 | 20.754 | 0.5533 | 0.01 | 0.1163 | 0.6563 | 0.05 | 0.1946 |
| LinRes | 0.0761 | 6.532 | 0.497 | 0.01 | 0.3623 | 0.7485 | 0.05 | 0.4834 |
| LatAD (global density) | 0.2507 | 21.518 | 0.6132 | 0.01 | 0.0959 | 0.7054 | 0.05 | 0.195 |
| LatAD (regime-community) | 0.2456 | 21.08 | 0.6814 | 0.0101 | 0.1027 | 0.8096 | 0.0501 | 0.2543 |
| USAD | 0.0112 | 0.961 | NA | NA | NA | NA | NA | NA |
  - *USAD*: no leak-free train-normal score available (external SOTA harness; test-only dump)
| TranAD | 0.0099 | 0.85 | NA | NA | NA | NA | NA | NA |
  - *TranAD*: no leak-free train-normal score available (external SOTA harness; test-only dump)

## SWaT_canon

- n_test=1498, anomalies=233, difficult=91, test_normals=1265
- prevalence (difficult / (difficult+test_normal)) = 0.0671 (AUPRC chance baseline; report AUPRC / prevalence as lift)
- calibration source for LatAD (regime-community): fit_surprise + calib_surprise (full train-normal)
- AUROC cross-check (must reproduce known values): {'regime_community_headline_Difficult': 0.524, 'regime_community_drift_aware_CP_Difficult': 0.723}
- degenerate all-flag sanity: {'tpr': 1.0, 'test_fpr': 1.0} (expect tpr=1.0, test_fpr=1.0)

| Method | AUPRC | AUPRC/prevalence | TPR@1%FPR | train FPR@1% | test FPR@1% | TPR@5%FPR | train FPR@5% | test FPR@5% |
|---|---|---|---|---|---|---|---|---|
| trivial max|z| | 0.106 | 1.58 | 0.0 | 0.0105 | 0.0364 | 0.1429 | 0.0501 | 0.064 |
| IF | 0.087 | 1.296 | 0.1297 | 0.0105 | 0.071 | 0.3187 | 0.0501 | 0.2349 |
| AE | 0.0882 | 1.314 | 0.6374 | 0.0105 | 0.6449 | 0.7231 | 0.0501 | 0.7221 |
| LinRes | 0.0966 | 1.439 | 0.6593 | 0.0105 | 0.7352 | 0.7473 | 0.0501 | 0.7494 |
| LatAD (global density) | 0.0699 | 1.042 | 0.7714 | 0.0105 | 0.7667 | 0.8242 | 0.0501 | 0.827 |
| LatAD (regime-community) | 0.1338 | 1.994 | 0.8066 | 0.0105 | 0.7964 | 0.8769 | 0.0501 | 0.8582 |
| USAD | 0.0718 | 1.07 | NA | NA | NA | NA | NA | NA |
  - *USAD*: no leak-free train-normal score available (external SOTA harness; test-only dump)
| TranAD | 0.0725 | 1.08 | NA | NA | NA | NA | NA | NA |
  - *TranAD*: no leak-free train-normal score available (external SOTA harness; test-only dump)
| GDN | 0.0727 | 1.083 | NA | NA | NA | NA | NA | NA |
  - *GDN*: no leak-free train-normal score available (external SOTA harness; test-only dump)
| LatAD (regime-community, drift-aware) | 0.2171 | 3.235 | 0.2901 | 0.0105 | 0.0577 | 0.4857 | 0.0501 | 0.1507 |
