# LatAD paper — build reference (single source of truth for the rewrite)

Headline detector = **LatAD (regime-community density)** = `HCcoh+LatAD` on full-stack experts.
Realization on the 4 design axes:
1. **Representation:** VaDE (jointly-learned latent + regime GMM). [A1,A7]
2. **Density estimation = ensemble:**
   - **(2a) construction:** correlation-community factorization of the train-normal correlation graph (HAC communities, sizes 3-25 = physical subsystems); one small VaDE density per community + the global VaDE as the unfactorized "null" expert.
   - **(2b) aggregation:** cohesion-weighted Higher Criticism over per-community tail p-values, fused (z-sum) with the global LatAD score. [2a: A5,A8,locality; 2b: A6/A3,A8]
3. **Scoring heads (per community = full LatAD):** latent density (A2/A4) + rare-regime-safe nearest-component (A6) + whitened cross-channel residual (A8) + basin (A3), auto-gated.
4. **Calibration:** train-normal-only — per-head z-normalization, held-out-normal auto-gate, per-community tail/p-value calibration.

Base ablation detector = **LatAD (global density)** = single global VaDE (the previous headline).

Artifacts: `_diagnostics/ensemble_head.json` (headline + baselines + significance),
`_diagnostics/ensemble_full.json` (all aggregators, full-stack), `_diagnostics/ensemble_density.json` (all aggregators, density-only).
Regenerate: `EXPERTS_DIR=sota_bundle/experts_full HEAD="HCcoh+LatAD" BOOT_REPS=2000 python ensemble_final.py`.

## Table 3 — AUROC (5-seed mean±sd) / best raw F1, by method × subset
Anom counts: WADI 56 (37 easy + 19 diff), HAI 652 (485+167), SWaT 182 (144+38).

### WADI
| Method | All | Easy | Difficult |
|---|---|---|---|
| trivial max\|z\| | 0.558 / 0.000 | 0.687 / 0.000 | 0.307 / 0.000 |
| Isolation Forest | 0.726±0.007 / 0.378 | 0.751±0.010 / 0.378 | 0.677±0.011 / 0.150 |
| AutoEncoder | 0.743±0.001 / 0.611 | 0.907±0.002 / 0.724 | 0.425±0.005 / 0.223 |
| LinRes | 0.595 / 0.214 | 0.699 / 0.220 | 0.392 / 0.182 |
| USAD | 0.698 / 0.532 | 0.901 / 0.667 | 0.303 / 0.032 |
| TranAD | 0.721 / 0.556 | 0.921 / 0.694 | 0.333 / 0.041 |
| LatAD (global density) | 0.792±0.012 / 0.565 | 0.845±0.007 / 0.680 | 0.690±0.027 / 0.222 |
| **LatAD (regime-community)** | **0.862±0.017 / 0.692** | **0.896±0.012 / 0.762** | **0.796±0.041 / 0.367** |

### HAI
| Method | All | Easy | Difficult |
|---|---|---|---|
| trivial max\|z\| | 0.806 / 0.666 | 0.966 / 0.800 | 0.340 / 0.000 |
| Isolation Forest | 0.844±0.011 / 0.418 | 0.919±0.016 / 0.468 | 0.627±0.005 / 0.059 |
| AutoEncoder | 0.923±0.001 / 0.733 | 0.980 / 0.803 | 0.757±0.003 / 0.340 |
| LinRes | 0.779 / 0.499 | 0.846 / 0.605 | 0.586 / 0.121 |
| USAD | 0.849 / 0.695 | 0.970 / 0.827 | 0.497 / 0.022 |
| TranAD | 0.834 / 0.686 | 0.968 / 0.820 | 0.445 / 0.016 |
| LatAD (global density) | 0.933±0.007 / 0.714 | 0.975±0.004 / 0.780 | 0.811±0.016 / 0.365 |
| **LatAD (regime-community)** | **0.949±0.004 / 0.756** | **0.983±0.002 / 0.822** | **0.849±0.018 / 0.364** |

### SWaT
| Method | All | Easy | Difficult |
|---|---|---|---|
| trivial max\|z\| | 0.988 / 0.972 | 1.000 / 1.000 | 0.943 / 0.848 |
| Isolation Forest | 0.959±0.003 / 0.914 | 0.987±0.002 / 0.960 | 0.853±0.011 / 0.659 |
| AutoEncoder | 0.987±0.001 / 0.933 | 1.000 / 0.997 | 0.939±0.006 / 0.648 |
| LinRes | 0.991 / 0.948 | 1.000 / 0.997 | 0.959 / 0.710 |
| USAD | 0.973 / 0.948 | 1.000 / 0.997 | 0.871 / 0.742 |
| TranAD | 0.972 / 0.949 | 1.000 / 0.997 | 0.867 / 0.738 |
| GDN (SWaT only, single-run) | 0.973 / 0.946 | 1.000 / 0.990 | 0.871 / 0.698 |
| LatAD (global density) | 0.991±0.002 / 0.950 | 0.999±0.002 / 0.974 | 0.960±0.006 / 0.761 |
| **LatAD (regime-community)** | **0.993±0.001 / 0.966** | 1.000 / 0.985 | **0.969±0.006 / 0.855** |

## Table 4 — Double-hard AUROC (windows/episodes: WADI 11/5, HAI 84/19, SWaT 18/1)
| Method | WADI | HAI | SWaT |
|---|---|---|---|
| trivial max\|z\| | 0.283 | 0.349 | 0.880 |
| Isolation Forest | 0.599±0.013 | 0.635±0.012 | 0.789±0.011 |
| AutoEncoder | 0.336±0.007 | 0.730±0.003 | 0.890±0.010 |
| LinRes | 0.251 | 0.465 | 0.914 |
| USAD | 0.257 | 0.471 | 0.785 |
| TranAD | 0.276 | 0.418 | 0.779 |
| LatAD (global density) | 0.601±0.027 | 0.806±0.019 | 0.921±0.012 |
| **LatAD (regime-community)** | **0.728±0.077** | **0.819±0.020** | **0.936±0.013** |

## Significance (LatAD regime-community vs strongest baseline, episode-block bootstrap)
- WADI Difficult: vs IF, diff +0.12, 95% CI [-0.005, 0.257], P(<=0)=0.032 (5 episodes) -> near-significant.
- WADI Double-hard: vs IF, diff +0.129, CI [-0.022, 0.496], P=0.053 (5 ep) -> numerical.
- HAI Difficult: vs AE, diff +0.092, CI [0.046, 0.16], P=0.000 (26 ep) -> SIGNIFICANT.
- HAI Double-hard: vs AE, diff +0.089, CI [0.02, 0.194], P=0.000 (19 ep) -> SIGNIFICANT.
- SWaT Difficult/Double-hard: single attack episode -> numerical lead only, NO episode-level generalization claim.

## Hyperparameters (from code, fact-checked)
- VaDE K: WADI 20, HAI 40, SWaT 40. Latent dim: WADI 10, HAI 16, SWaT 16. Density-head K=80 (data-capped min(80,max(20,n/10))).
- Residual head: PCA->30 dims on normal residuals; per-regime Ledoit-Wolf precision (global fallback <30 samples); auto-gate ratio q95(heldout)/q95(fit) < 1.5 -> ON; A=first 80% train-normal, B=last 20%. Gate ratios: WADI 5.24 (off), HAI 1.17 (on), SWaT ~0.85 (on).
- Basin head: R=16 perturbations, lambda0=2.5, deadzone delta=0.15, amb=0.5, noise=0.5*latent-std; rho~0.05<delta -> lambda=0 (no-op on all three).
- Community construction: HAC (average linkage) on 1-|corr| of train-normal window means; nested subtrees size 3-25. S = WADI 45, SWaT 26, HAI 28 communities. cohesion = mean pairwise |rho|; weight w = cohesion*sqrt(size).
- Windows W=60 stride=30; 6 per-channel stats. Downsample: WADI 10x, SWaT 10x, HAI NATIVE (no downsample; SOTA HAI scores integer-ratio upsampled to our grid).

## Multimodality (dataset characterization, from _diagnostics/multimodality.json)
K*_BIC / silhouette: WADI 22 / 0.06, HAI 24 / 0.077, SWaT 25 / 0.291.
-> WADI+HAI regimes overlap (low silhouette) = where community factorization wins; SWaT crisp (0.291) = ceiling.

## Pending consistency fixes (science audit) — still to apply in rewrite
- Table 2 / mechanism probe (recon_why.py): seed-0 single-model -> label "illustrative single seed" or recompute 5-seed; probe recon is plain-AE recon (not whitened).
- ELBO +H(q) removed (done). Latent dim 6->10-16 (done). HAI downsampling wording (done). "provably"->"demonstrably". FPR promised-never-delivered -> drop. "best on every dataset" -> "best or tied". Head numbering (i)-(iii). "single trained model" vs 5-seed wording.
- Name unified to LatAD throughout; VaDE-hard+resid(auto) only as config id.
- Fig 1 rebuild from Table 3 difficult column (headline bars).
