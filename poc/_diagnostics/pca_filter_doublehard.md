# Double-hard subset with a PCA linear-Gaussian second filter (T2 + SPE)

Script `poc/rev4_doublehard_pca.py` (copy of `rev4_doublehard.py`; only filter B changed, episode-block
bootstrap identical, 2000 reps, seed 0). Numbers in `_diagnostics/pca_filter_doublehard.json`; per-window
T2 / SPE / flags in `_diagnostics/pca_filter_doublehard_<DS>.npz`; run log `pca_filter_doublehard.log`;
divergence diagnostics `pca_filter_why_spe_weak.py`, `pca_filter_hai_t2_why.py`, log `pca_filter_diagnostics.log`.
Python `C:\Python314\python`, CPU only. Agent wall-clock: about 25 min (the three 2000-rep bootstraps take
2 + 6 + 2 min; the rest is diagnostics).

## Filter definition

- Filter A (unchanged): max|z| over the mean stat-block, threshold = train-normal 99th percentile (`scores_<DS>.npz`).
- Filter B (new): PCA fitted on the TRAIN-normal standardized windowed feature matrix, the identical
  6-statistics-per-channel input the detectors consume (`build_scores_table` FIX-2 standardization, `eda_real`
  CLIP; no one-hot indicators). Components explaining 95% of train-normal variance retained.
  T2 = sum over retained PCs of score^2 / eigenvalue; SPE = energy in the dropped components.
  Thresholds: train-normal 99th percentile of T2 and of SPE, independently. Separable-B = T2 > thrT2 OR SPE > thrSPE.
- Double-hard = anomaly AND maxz <= thr AND T2 <= thrT2 AND SPE <= thrSPE. The detector under test plays no role.
- Headline = LatAD (regime-community) = `HCcoh+LatAD` on `sota_bundle/experts_full` (the paper's Table 4 headline).
  USAD/TranAD five-seed on HAI (`scores_sota_ms_HAI.npz`), single-run on WADI and canonical SWaT; GDN single-seed
  (window-averaged per-timestep scores, as in `gdn_doublehard.py`).

Mechanical checks (all pass): with all components retained SPE is 2e-12 to 3e-11 (zero); train flag rate of the
OR-rule 1.4 to 1.9% (two independent 1% tails); `scores.linres` is the LOCO residual re-used as a detector
(Spearman 1.000); new subset is a strict subset of difficult (asserted).

| | WADI (clean) | HAI | SWaT (canonical) |
|---|---|---|---|
| feature dim d / k at 95% (90 / 99) | 732 / 113 (75 / 194) | 354 / 60 (42 / 101) | 306 / 43 (30 / 75) |
| thrT2 / thrSPE (train 99th) | 406.8 / 125.3 | 173.0 / 45.6 | 535.0 / 85.3 |
| difficult (max|z| only) | 30 | 167 | 85 |
| flagged by T2 / SPE / either / LinRes | 0 / 1 / 1 / 11 | 90 / 110 / 112 / 83 | 4 / 27 / 27 / 26 |

## 1. Subset sizes and validity

| subset (windows / distinct attack episodes) | WADI | HAI | SWaT |
|---|---|---|---|
| current, LinRes-based (paper) | 19 / 7 | 84 / 19 | 59 / 18 |
| **new, PCA T2+SPE at 95%** | **29 / 8** | **55 / 13** | **58 / 18** |
| PCA at 90% / 99% variance | 30 / 8, 21 / 7 | 68 / 16, 11 / 7 | 63 / 18, 52 / 17 |
| T2-only (no SPE) | 30 / 8 | 77 / 17 | 81 / 21 |
| overlap: in both / PCA-only / LinRes-only | 19 / 10 / 0 | 33 / 22 / 51 | 51 / 7 / 8 |

Invariant 1 (strict subset of difficult): holds on all three (asserted in code).
Invariant 2 (trivial max|z| AUROC near chance on the new subset): WADI 0.591, HAI 0.336, SWaT 0.588. On WADI the
PCA filter removes only one of the 30 difficult windows, so the "double-hard" set is the difficult set minus one
window and the trivial rule keeps the 0.60 it has on the difficult subset (the max|z| filter caps this by
construction; below-chance on HAI). On SWaT the size matches the LinRes-based subset almost exactly (58 vs 59,
51 shared). On HAI the PCA filter is much more aggressive than LinRes (112 vs 83 of 167 flagged).

## 2. New Table 4: AUROC on the PCA double-hard subset (five-seed mean +- SD where applicable)

| Method | WADI (29 / 8) | HAI (55 / 13) | SWaT (58 / 18) |
|---|---|---|---|
| trivial max|z| (filter, floor) | 0.591 | 0.336 | 0.588 |
| Isolation Forest | 0.622 +- 0.006 | 0.435 +- 0.023 | 0.535 +- 0.008 |
| AutoEncoder | 0.616 +- 0.004 | 0.418 +- 0.008 | 0.623 +- 0.010 |
| LinRes (one-hot), now a fair detector | 0.742 | 0.547 | 0.705 |
| USAD | 0.565 | 0.351 +- 0.018 | 0.573 |
| TranAD | 0.601 | 0.296 +- 0.004 | 0.569 |
| GDN | 0.648 | 0.354 | 0.588 |
| LatAD (global density) | 0.623 +- 0.012 | 0.559 +- 0.040 | 0.731 +- 0.010 |
| **LatAD (regime-community)** | **0.763 +- 0.024** | **0.648 +- 0.039** | **0.767 +- 0.006** |

LatAD (regime-community) has the highest mean on all three datasets; the strongest external detector is LinRes on
all three. For reference, the same methods on the current LinRes-based subset reproduce the paper's Table 4
exactly (WADI 0.662 / HAI 0.814 / SWaT 0.773 for the headline; all baseline cells identical), so the pipeline is the
paper's pipeline and only the subset moved.

## 3. Significance: headline vs strongest external detector (episode-block bootstrap, 2000 reps)

| dataset | competitor | headline | competitor | paired diff | 95% CI | one-sided P | episodes |
|---|---|---|---|---|---|---|---|
| WADI | LinRes | 0.763 | 0.742 | +0.021 | [-0.165, 0.201] | 0.46 | 8 |
| HAI | LinRes | 0.648 | 0.547 | +0.101 | [-0.090, 0.337] | 0.17 | 13 |
| SWaT | LinRes | 0.767 | 0.705 | +0.061 | [0.004, 0.137] | **0.015** | 18 |

Verdicts: SWaT significant at alpha 0.05; HAI a numerical lead (+0.10) that the 13-episode bootstrap does not
support; WADI a numerical tie. Against the internal global-density ablation the regime-community headline is
significant on WADI (+0.140, CI [0.062, 0.243], P < 0.0005) and SWaT (+0.036, CI [-0.003, 0.077], P = 0.042), not
on HAI (+0.089, P = 0.11).

Compare the current (LinRes-based) verdicts: HAI +0.084 P = 0.0005 vs AE, SWaT +0.085 P = 0.0005 vs LinRes, WADI
P = 0.43. The HAI verdict flips from significant to not significant under the PCA filter (see section 4 for why:
the PCA filter removes the HAI windows on which every learned detector, ours included, does well).

## 4. LinRes-floor check: REFUTED for the specified filter

Expectation: SPE measures the same off-subspace linear-correlation break as LinRes, so LinRes should sit near the
floor on the new subset. Result: LinRes is the strongest external detector on all three datasets (0.742 / 0.547
/ 0.705), far above the trivial floor on WADI and SWaT. SPE and LinRes do NOT diverge as statistics: their
Spearman rank correlation on the difficult anomalies is 0.77 (WADI), 0.75 (SWaT); only on HAI is it 0.18. They
diverge in TAIL CALIBRATION, and on HAI in what the extra stat blocks expose. Root causes, from the per-window
diagnostics:

1. WADI (SPE too weak). Raw SPE sums unnormalized residual energy over the 619 dropped dimensions of a
   732-dim space; its train 99th percentile is set by the largest dropped components, and the difficult
   anomalies fall at train-percentiles 0.14 to 0.997 of SPE (one above 0.99) versus 0.99 to 1.00 of LinRes for 11
   of 30. Three controls confirm the cause is the statistic's tail, not the feature space: PCA on the LinRes
   one-hot feature matrix itself still flags only 1 at 95% variance; raising the cut to 99% variance (so SPE
   contains only the small-eigenvalue directions) flags 8 to 10 windows, 7 of them among LinRes's 11 (and 9 of
   30 on the full 732-dim space); and a per-column
   normalized SPE (residual scaled by its train std, a LinRes-like normalization) flags 8. The columns carrying
   50% of the train SPE energy are 130 of 732, spread over all six stat blocks, so no single block dominates.
2. SWaT (agreement). SPE flags 27, LinRes 26, 19 shared; the subsets coincide (58 vs 59 windows, 51 shared) and
   LinRes stays at 0.705 (0.688 on the current subset). Here the LinRes-defined subset never put LinRes at the
   floor either: 0.688 is the paper's own Table 4 value, above every baseline. The 99th-percentile OR-rule removes
   the extreme tail, but AUROC on the survivors is a ranking over the sub-threshold windows, where LinRes still
   orders anomalies above test-normals. A filter at the 99th percentile does not make its own statistic a
   chance-level ranker; it removes the top 1%-tail events.
3. HAI (SPE and T2 far stronger than LinRes). T2 alone flags 90 of 167, SPE 110, and as a difficult-subset
   detector T2 scores 0.807 and SPE 0.747 (LinRes 0.586). Decomposing T2 by stat block on the difficult
   anomalies: mean-block 5.3, std-block 932, range-block 951 (train: 6, 11, 11). PCA restricted to the mean
   block gives T2 AUROC 0.438 (flags 0) while the std block alone gives 0.802 (flags 96). The HAI "difficult"
   anomalies (difficult by max|z| of the window MEANS) carry 15 to 98 sigma excursions in the within-window std
   and range of single channels (P1_PCV01D/Z, P2_VYT02/03). Filter A never inspects those blocks, LinRes uses
   window means only, so both pass them; any linear-Gaussian model on the detectors' 6-stat input catches them
   trivially. The windows the PCA filter removes are precisely the ones on which AE (0.730 -> 0.418), IF, GDN,
   USAD, TranAD and LatAD-global (0.806 -> 0.559) had their signal, which is why every method drops on HAI.
   Note for the paper: this is a limitation of filter A as currently defined (mean block only); a max|z| over all
   six blocks would classify these HAI windows as easy.

Bottom line for the floor check: LinRes at 0.547 on HAI is low in absolute terms and near the other
detectors, but it is not suppressed on WADI/SWaT, and the reason is not a divergence between SPE and LinRes as
measures of linear structure; it is that raw 95%-variance SPE has a much heavier train tail than the per-channel
LOCO residual (WADI), and that on SWaT neither filter drives its own statistic to chance on the survivors.

## 5. T2-only variant (filter B = T2 > thrT2 only)

| | WADI | HAI | SWaT |
|---|---|---|---|
| T2-only subset (windows / episodes) | 30 / 8 | 77 / 17 | 81 / 21 |
| LinRes AUROC there (vs T2+SPE subset) | 0.750 (0.742) | 0.562 (0.547) | 0.772 (0.705) |
| headline there | 0.771 | 0.698 | 0.829 |

SPE-specific suppression of LinRes: WADI 0.008, HAI 0.015, SWaT 0.067 AUROC. Almost all of the LinRes drop
(where there is one) comes from SPE, and on SWaT the SPE flags overlap the LinRes flags 19/27, consistent with
the two measuring the same off-subspace structure; but the suppression is small in absolute terms because the
99th-percentile rule removes only the tail.

## 6. 99%-variance variant (SPE restricted to small-eigenvalue directions)

| | WADI (21 / 7) | HAI (11 / 7) | SWaT (52 / 17) |
|---|---|---|---|
| LinRes | 0.644 | 0.393 | 0.693 |
| LatAD (global density) | 0.584 | 0.620 | 0.766 |
| LatAD (regime-community) | 0.683 | 0.583 | 0.810 |
| headline vs strongest external | +0.039 vs LinRes, P = 0.44 | +0.150 vs IF, CI [0.004, 0.299], P = 0.021 | +0.044 vs LinRes, CI [-0.012, 0.102], P = 0.058 |

At 99% the WADI subset (21 / 7) approaches the LinRes-based one (19 / 7) and LinRes drops to 0.644, still the
strongest external detector. HAI shrinks to 11 windows / 7 episodes, too small to carry a table entry; there the
global-density ablation (0.620) edges the headline (0.583).

## Other observations

- Test-normal flag rate of filter B: WADI 2.1%, SWaT 1.1%, HAI 41% (LinRes: 0.6% / 2.0% / 36%). HAI test-normal
  windows are distribution-shifted relative to train-normal under every linear model; this does not affect the
  subset (only anomaly windows are filtered) but it is the same drift that depresses all HAI difficult-subset
  AUROCs.
- Retained-eigenvalue range at 95%: HAI lam_1 / lam_60 = 96; train T2 99th percentile 173 versus chi-square(60)
  99th percentile 88, so train-normal is heavy-tailed under the Gaussian model on every dataset (WADI thrT2 407
  at k = 113, SWaT 535 at k = 43).

## Recommendation

The specified PCA T2+SPE filter at 95% variance is a defensible, standard, train-normal-calibrated second filter,
and the headline stays first on all three datasets, but it changes what "double-hard" means per dataset: on WADI
it removes almost nothing (the difficult subset already is the double-hard subset), on SWaT it reproduces the
LinRes-based subset, and on HAI it removes the variance-magnitude anomalies that every learned detector was
detecting, which costs the HAI significance verdict. If the paper adopts it, Table 4 becomes the table in
section 2 with the section 3 verdicts (SWaT significant, HAI and WADI numerical leads), and the caption should
say that LinRes is scored as a detector. The HAI finding (max|z| on means passes 15 to 98 sigma std/range
excursions) is worth reporting regardless of which filter B is used.
