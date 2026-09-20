# VUS-ROC / VUS-PR (Paparrizos et al., VLDB 2022) -- full (All) window-level set

Reimplemented directly from the VLDB'22 definition (pip `vus` / thedatumorg/VUS could not be installed under Python 3.14 -- `pip install vus` fails with `ModuleNotFoundError: pkg_resources` in its isolated build backend, and its `arch` build dependency has no Python-3.14 wheel). For buffer l=0..L, anomaly ranges are extended by l windows on each side with a linear label-weight decay (weight=1-k/(l+1) at the k-th window outside the range); ROC-AUC/PR-AUC are computed (sklearn, sample_weight=weight) on the extended labels, then averaged over l. l is in units of scoring windows (the same window grid as scores_<DS>.npz).


## WADI_clean

- n_windows=575, anomalies=56, n_ranges=11, mean_range_len=5.091 windows, prevalence=0.0974
- L_default=3 (round(mean_range_len/2)); also reporting L_list=[0, 3, 6]

| Method | VUS-ROC@L0 | VUS-PR@L0 | VUS-ROC@L3 | VUS-PR@L3 | VUS-ROC@L6 | VUS-PR@L6 |
|---|---|---|---|---|---|---|
| trivial max|z| | 0.7855 | 0.5646 | 0.7503 | 0.5408 | 0.7251 | 0.5271 |
| IF | 0.7249 | 0.3131 | 0.6920 | 0.3576 | 0.6657 | 0.3791 |
| AE | 0.7921 | 0.5364 | 0.7498 | 0.5316 | 0.7202 | 0.5235 |
| LinRes | 0.8345 | 0.6851 | 0.7994 | 0.6414 | 0.7811 | 0.6281 |
| USAD | 0.7570 | 0.4933 | 0.7236 | 0.4985 | 0.6951 | 0.5003 |
| TranAD | 0.7863 | 0.5321 | 0.7529 | 0.5369 | 0.7261 | 0.5369 |
| LatAD (global density) | 0.7171 | 0.2322 | 0.6843 | 0.2774 | 0.6553 | 0.2976 |
| LatAD (regime-community) | 0.8270 | 0.5046 | 0.7858 | 0.5351 | 0.7489 | 0.5303 |

- LatAD (regime-community) VUS-PR@L3 = 0.5351; next-best = LinRes (0.6414); LEADS = **False**, margin = -0.1063
- overall best VUS-PR@L3: LinRes = 0.6414

## HAI

- n_windows=14819, anomalies=652, n_ranges=38, mean_range_len=17.158 windows, prevalence=0.044
- L_default=9 (round(mean_range_len/2)); also reporting L_list=[0, 9, 18]

| Method | VUS-ROC@L0 | VUS-PR@L0 | VUS-ROC@L9 | VUS-PR@L9 | VUS-ROC@L18 | VUS-PR@L18 |
|---|---|---|---|---|---|---|
| trivial max|z| | 0.8059 | 0.5983 | 0.7629 | 0.5307 | 0.7325 | 0.4885 |
| IF | 0.8442 | 0.3759 | 0.7936 | 0.3598 | 0.7555 | 0.3416 |
| AE | 0.9230 | 0.7290 | 0.8632 | 0.6445 | 0.8212 | 0.5840 |
| LinRes | 0.7793 | 0.4178 | 0.7297 | 0.3667 | 0.6948 | 0.3381 |
| USAD | 0.8487 | 0.6343 | 0.7982 | 0.5607 | 0.7619 | 0.5100 |
| TranAD | 0.8340 | 0.6219 | 0.7782 | 0.5472 | 0.7383 | 0.4961 |
| LatAD (global density) | 0.9332 | 0.7155 | 0.8836 | 0.6403 | 0.8503 | 0.5873 |
| LatAD (regime-community) | 0.9482 | 0.7635 | 0.8893 | 0.6912 | 0.8470 | 0.6292 |

- LatAD (regime-community) VUS-PR@L9 = 0.6912; next-best = AE (0.6445); LEADS = **True**, margin = +0.0467
- overall best VUS-PR@L9: LatAD (regime-community) = 0.6912

## SWaT_canon

- n_windows=1498, anomalies=233, n_ranges=25, mean_range_len=9.32 windows, prevalence=0.1555
- L_default=5 (round(mean_range_len/2)); also reporting L_list=[0, 5, 10]

| Method | VUS-ROC@L0 | VUS-PR@L0 | VUS-ROC@L5 | VUS-PR@L5 | VUS-ROC@L10 | VUS-PR@L10 |
|---|---|---|---|---|---|---|
| trivial max|z| | 0.8526 | 0.7476 | 0.8193 | 0.7256 | 0.7941 | 0.7184 |
| IF | 0.8004 | 0.6493 | 0.7452 | 0.6231 | 0.7133 | 0.6125 |
| AE | 0.7864 | 0.6541 | 0.7331 | 0.6401 | 0.7018 | 0.6281 |
| LinRes | 0.7760 | 0.6917 | 0.7125 | 0.6362 | 0.6824 | 0.6223 |
| USAD | 0.7629 | 0.6676 | 0.7077 | 0.6282 | 0.6773 | 0.6150 |
| TranAD | 0.7612 | 0.6673 | 0.7062 | 0.6282 | 0.6762 | 0.6153 |
| LatAD (global density) | 0.7519 | 0.5920 | 0.7023 | 0.5745 | 0.6734 | 0.5713 |
| LatAD (regime-community) | 0.7876 | 0.6576 | 0.7349 | 0.6252 | 0.7039 | 0.6103 |
| GDN | 0.7608 | 0.6655 | 0.7057 | 0.6270 | 0.6754 | 0.6142 |

- LatAD (regime-community) VUS-PR@L5 = 0.6252; next-best = trivial max|z| (0.7256); LEADS = **False**, margin = -0.1004
- overall best VUS-PR@L5: trivial max|z| = 0.7256

## Invariant checks

| Dataset | L=0 crosscheck OK | L0 VUS-ROC | L0 direct AUROC | L0 VUS-PR | L0 direct AUPRC | random-score VUS-PR | prevalence | perfect-score VUS-PR |
|---|---|---|---|---|---|---|---|---|
| WADI_clean | True | 0.7118 | 0.7118 | 0.2142 | 0.2142 | 0.1354 | 0.0974 | 1.0 |
| HAI | True | 0.9409 | 0.9409 | 0.7434 | 0.7434 | 0.0603 | 0.044 | 1.0 |
| SWaT_canon | True | 0.7484 | 0.7484 | 0.6279 | 0.6279 | 0.1879 | 0.1555 | 1.0 |
