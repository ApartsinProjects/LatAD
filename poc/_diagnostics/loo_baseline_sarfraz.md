# Why the leave-one-channel-out linear baseline (LinRes) is strong on WADI, and whether it rediscovers Sarfraz et al.

Scripts: `_diagnostics/loo_baseline_why.py` (decomposition, effective rank, nonlinear/per-regime LOO ablation),
`_diagnostics/loo_baseline_ablate.py` (discrete vs continuous feature families), `_diagnostics/loo_baseline_drift.py`
(train-to-test drift of the linear relations, raw-unit attribution). JSON outputs sit next to each script.
Inputs: `_diagnostics/scores_{WADI_clean,SWaT_canon,HAI}.npz` (W=60, stride=30) plus the raw windows from
`eda_real.load`. Difficult subset = `(label==1) & (maxz <= maxz_thr)`, scored against all test normals.

Invariants checked before any number below was trusted:
- I1: LinRes recomputed from `onehot_filter.build_feats` + LOO regression matches the npz `linres` vector to a
  max relative error of 6e-8 on all three datasets.
- I2: recomputed AUROCs reproduce the paper table: WADI_clean All 0.834 / difficult 0.750 (F1 0.538);
  SWaT_canon 0.920 / 0.782; HAI 0.779 / 0.586. Global LatAD (npz, mean over 5 seeds): WADI 0.638, SWaT 0.808,
  HAI 0.814 on difficult.
- I3: the discrete-only + continuous-only ablation reproduces the canonical score when both families are used.

## 1. Empirical result: why LinRes is strong on WADI and weak on HAI

### 1.1 The headline mechanism is train-to-test stationarity of the linear cross-channel relations, not linearity of the anomalies

The linear relations are extremely tight on train-normal everywhere (median per-channel LOO R^2: WADI 0.958,
SWaT 0.984, HAI 0.997), and the difficult anomalies break them everywhere. Scored against TRAIN normals the
LinRes difficult-AUROC is 0.878 (WADI), 0.868 (SWaT) and 0.947 (HAI). HAI is the dataset where the anomalies
violate the linear relations MOST cleanly. What differs is whether the TEST-normal windows still satisfy the
train-fitted relations:

| dataset | test-normal FP rate at LinRes train-p99 | median LinRes test-normal / train | AUROC test-normal vs train (LinRes) | channels with median normalised residual > 3 on test-normal | LinRes difficult AUROC vs test-normal |
|---|---|---|---|---|---|
| WADI_clean | 0.6% | 1.6x | 0.753 | 1 (1_MV_004_STATUS, 3.3x) | 0.750 |
| SWaT_canon | 2.0% | 2.1x | 0.710 | 2 (MV201 4.5x, MV303 3.6x) | 0.782 |
| HAI | 36.2% | 2.7x | 0.842 | 15 (P1_FCV03Z 51x, P1_FT03 26x, P1_FT01Z 24x, P1_FT01 18x, P1_FT03Z 17x) | 0.586 |

On HAI, 36% of the 14,167 test-normal windows exceed the train-normal 99th percentile of the residual: the P1
flow/valve loop (FCV03, FT01, FT03) runs on a different linear relation in the test recording than in the
training recording. Half of the difficult anomalies (49.7%) do exceed the same threshold, but they are buried
under drifted normals, so the difficult AUROC collapses from 0.947 (vs train) to 0.586 (vs test). LatAD's score
is far less drift-sensitive on the same windows: test-normal FP at its own train-p99 is 9.6%, difficult TP
65.3%, and AUROC test-normal-vs-train 0.694 (LinRes 0.842). This is the structural property that makes a
fixed linear predictor insufficient on HAI: the cross-channel relations are regime- and recording-dependent,
and a density over a learned latent absorbs the shift where a residual against fixed coefficients does not.

The nonlinear ablation says the same thing from the other side. Replacing the linear LOO regressor by a
gradient-boosted LOO regressor (HistGradientBoosting, 150 iters) gives difficult AUROC 0.321 on HAI (below
chance: the tighter the fit to the training relations, the more the drifted test normals look anomalous),
0.660 on WADI (down from 0.750) and 0.881 on SWaT (up from 0.782). A per-regime linear LOO (k-means K=8 on
train windows) gives 0.497 on HAI, 0.689 on WADI, 0.820 on SWaT. More capacity in the predictor helps only on
SWaT (see 1.4).

### 1.2 On WADI the difficult set is low-rank, stationary and small-sample, so the least-flexible predictor wins

- Effective rank (participation ratio of the PCA spectrum of train-standardised feature deviations): difficult
  windows 2.5, easy windows 4.1, a size-matched random normal sample 6.8. The 30 difficult WADI windows move
  along about two to three directions of the 128-dim feature space.
- Raw-unit residual concentration on difficult windows (share of the canonical mean residual): top-1 channel
  0.26, top-3 0.59, effective number of contributing channels 11.6 (SWaT: 0.44 / 0.79 / 5.2; HAI: 0.36 / 0.71 /
  7.1). The most frequent top-1 driver channels are the water-quality analysers and levels: 3_AIT_002_PV (6
  windows), 1_AIT_002_PV (4), 2A_AIT_002_PV (3), 3_AIT_001_PV (3), 1_AIT_004_PV (3), 1_LT_001_PV (3),
  2_FIC_501_SP (3).
- Per-window win/loss against global LatAD (percentile against test normals, margin 0.05): LinRes wins 18 of
  30 difficult windows, LatAD wins 6. Median percentile 0.946 (LinRes) vs 0.682 (LatAD). Where LinRes is at or
  above the 0.95 percentile (15 windows) the normalised residual is concentrated (top-1 share 0.50, n_eff 8.5);
  where it is below 0.80 (13 windows) it is diffuse (top-1 0.12, n_eff 28). The episodes LinRes nails at
  percentile 1.0 (starts 195, 494, 507; 11 windows) are the ones where actuator states (1_P_003_STATUS,
  1_P_001_STATUS, 1_MV_001_STATUS, 2_MV_003_STATUS) are inconsistent with the process variables.
- Which feature family carries it (ablation, difficult AUROC): continuous-feature residuals alone 0.755
  (= canonical 0.750); discrete one-hot residuals alone 0.578; continuous residuals with the discrete channels
  removed from the PREDICTOR set 0.663; numeric encoding of the discrete channels instead of one-hot 0.631.
  So the WADI signal is "a continuous process variable is off relative to what the actuator STATES imply",
  a state-conditional linear invariant. The one-hot encoding of the 26 discrete channels is what makes it
  linear.
- Training set: 2,614 windows for 128 features. Both higher-capacity LOO predictors (boosted 0.660, per-regime
  0.689) and the deep detectors (AE 0.629, USAD 0.579, TranAD 0.613, global LatAD 0.638) are below the linear
  0.750 on this subset. With a low-rank, stationary deviation and ~20 samples per feature, ordinary least squares
  is the best-regularised estimator of the invariant; flexible models spend capacity on directions the
  anomalies never use.

Verdict on the paper's framing (WADI difficult = single-channel/linear): partly confirmed, and sharpened. The
deviations are low-rank (2.5 effective directions) rather than strictly single-channel (top-1 share 0.26, 11.6
effective channels in raw units), and the linear relation that is violated is conditional on actuator state.
The decisive property is not linearity of the anomaly but stationarity of the linear relations between the
normal and attack recordings (0.6% test-normal false alarms).

### 1.3 On HAI the framing "cross-channel, regime-dependent" is confirmed, with the mechanism named

HAI difficult deviations are not lower-rank than normal variation (participation ratio 6.7 vs 6.3 for a matched
normal sample; easy anomalies 3.5) and involve many channels (n_eff 12.3 normalised, 7.1 raw). But the reason
LinRes fails is the drift in 1.1: the relations the anomalies violate are the same relations the test-normal
recording has already moved away from. A detector must model the normal manifold as it is in the test recording
(or be robust to the shift) to separate them; LatAD's per-window win rate on HAI difficult is 124 vs 37 for
LinRes (median percentile 0.934 vs 0.637), and LatAD wins every P2_SIT01/P2_SD01/P3_LCP01D episode at
percentile 1.0 where LinRes sits at 0.43 to 0.66.

### 1.4 SWaT: LinRes and LatAD are close, a single channel (PIT502) drives the linear residual, and a boosted LOO predictor would beat both

SWaT difficult: LinRes 0.782, LatAD 0.808 (per seed 0.792 to 0.811), boosted LOO 0.881, per-regime linear 0.820.
Raw-unit attribution: PIT502 is the top-1 residual channel on 39 of 85 difficult windows (LIT301 16, AIT503 8);
top-1 share 0.44, n_eff 5.2. The difficult deviations have participation ratio 6.2 versus 4.8 for normals, so
this is not a low-rank regime like WADI; per-window LatAD wins 26, LinRes 17. The boosted-LOO number is a
finding the paper does not currently contain: on SWaT a nonlinear channel-wise predictor (0.881) exceeds the
global LatAD (0.808) on the difficult subset. It is not a uniform winner (HAI 0.321, WADI 0.660), which is the
same drift story as 1.1. Section 1.5 tries to break it.

### 1.5 Verification of the boosted channel-wise LOO on SWaT_canon (script `loo_boosted_swat.py`, JSON `loo_boosted_swat.json`)

Construction: identical features and leave-one-CHANNEL-out protocol as LinRes, regressor replaced by
`HistGradientBoostingRegressor(max_iter=150, lr=0.1)` at defaults, fitted on train-normal only; score = mean
squared residual over the 67 features. Train-side residuals are 5-fold cross-fitted so the train-p99 is fair.
Headline ensembles rebuilt with `ensemble_final.ensemble_scores` on `sota_bundle/experts_full` (5 seeds).
Invariants: linear LOO reproduces the npz to 6e-8; boosted difficult AUROC reproduces 1.4 (0.881); headline
HCcoh+LatAD reproduces `headline_clean_results.md` to 0.003 (0.837 vs 0.840 difficult, 0.773 vs 0.775
double-hard, same n=85/59 subsets; the residual is at the rounding level of a 5-seed mean and changes nothing
below). Test-normal false-alarm rate of the boosted score at its cross-fitted train-p99 is 0.6% (difficult TP
at that threshold 0.306), so the number is not a drift artefact.

AUROC on the current subsets (Difficult n=85 / 23 episodes; DoubleHard n=59 / 18 episodes):

| score | Difficult | DoubleHard | All |
|---|---|---|---|
| boosted channel-wise LOO | **0.881** | **0.830** | 0.952 |
| boosted LOO without PIT502 (refit) | 0.870 | | |
| LinRes (linear LOO) | 0.782 | 0.688 | 0.920 |
| LatAD global | 0.804 | 0.728 | 0.927 |
| null+HC (paper headline) | 0.811 | 0.735 | |
| cohmax+LatAD | 0.829 | 0.769 | |
| HCcoh+LatAD (density-fusion headline) | 0.837 | 0.773 | |

Episode-block bootstrap (same `boot` as ensemble_final, 2000 reps, L = ceil(W/stride)+1, seed 0), paired
difference boosted minus competitor, 95% CI, one-sided P(diff <= 0):

| competitor | Difficult | DoubleHard |
|---|---|---|
| HCcoh+LatAD | +0.044, CI [-0.017, 0.096], P = 0.074 | +0.057, CI [-0.035, 0.132], P = 0.087 |
| null+HC | +0.069, CI [0.012, 0.120], P = 0.011 | +0.095, CI [0.011, 0.163], P = 0.017 |
| LatAD global | +0.077, CI [0.012, 0.130], P = 0.011 | +0.102, CI [0.005, 0.176], P = 0.019 |
| HCcoh+LatAD vs boosted without PIT502 | +0.033, CI [-0.028, 0.082], P = 0.144 | +0.042, CI [-0.051, 0.112], P = 0.152 |

Reverse direction on DoubleHard (headline minus boosted): HCcoh+LatAD -0.057, CI [-0.132, 0.035], P(<=0) =
0.913; null+HC -0.095, CI [-0.163, -0.011], P(<=0) = 0.984.

Reading: against the density-fusion headline (HCcoh+LatAD) the boosted predictor is numerically ahead on both
subsets and neither gap is significant at 0.05 (one-sided P 0.074 / 0.087; both CIs include 0). Against the
paper's primary headline (null+HC) and the global model it is significantly ahead on both Difficult and
DoubleHard. Double-hard is not a refuge on SWaT: the boosted predictor leads there too (0.830 vs 0.773),
because double-hard removes only what the LINEAR residual catches, and the nonlinear predictor keeps its
margin on what remains.

PIT502. The earlier "PIT502 drives it" statement came from the raw-unit attribution of the LINEAR residual
(top-1 on 39 of 85 difficult windows, section 1.4). For the boosted residual PIT502 is nearly irrelevant:
median share of the boosted score on difficult windows 0.02 (test-normal 0.06); dropping PIT502 as a target
only gives 0.876; dropping it entirely and refitting gives 0.870 (bootstrap vs HCcoh+LatAD still +0.033 / +0.042,
P 0.14 / 0.15). PIT502 alone: |z| of its window mean 0.620 on difficult, its own boosted self-prediction
residual 0.686, linear self-residual 0.782 is the LinRes-level number. So the boosted result is not a
single-channel detector. Physical check against the iTrust attack list
(`_diagnostics/swat_attack_list_deepsentinel.csv`, episode mapping in `framing_loc/loc2_episodes_SWaT_canon.jsonl`):
PIT502 (RO feed pressure, stage P5) is never a direct attack point. Its per-episode percentile is >= 0.95 on
the Dec-28 episodes (attacks 1-2 MV-101/P-102, 3 LIT-101, 4/6/7 incl. MV-504 on the RO, 8 DPIT-301, 10/11
FIT-401 whose intended impact is "P-501 turns off"), on attack 22 (UV-401/AIT-502/P-501 forced on, "reduced
output at FIT-502"), and on attack 33 (LIT-101). In every Dec-28 episode PIT502 sits at z = +2.4 to +2.5
regardless of which stage is attacked, and at z < 0 for every episode after Dec 30, so its linear residual is
a slow plant-state offset in the first day of the attack recording rather than attack signal (its |z| alone
separates the difficult windows from test normals at only 0.620 because the surrounding normals carry the
same offset). Attacks 22 and 37 (P-501 forced on / P-501 speed 10 to 28.5 Hz with FIT-502 spoofed) are the
only ones that physically act on the pressure PIT502 measures, and those are legitimate cross-channel
signal. Verdict: no label-adjacent leakage, and the boosted lead does not depend on PIT502.

Is the boosted LOO a fair baseline? Yes, construct-matched: same window features, same one-hot encoding,
same leave-one-channel-out protocol, train-normal-only fit, no tuning (library defaults, one fixed
iteration count), one line changed relative to LinRes. It is the nonlinear member of the channel-wise
family Garg et al. [7] found to win (their univariate AE), so it needs no separate justification beyond
"the channel-wise baseline with a nonlinear regressor". What it is not is a uniform competitor: HAI 0.321,
WADI 0.660, both below the linear version, for the drift reason in 1.1. Reporting it therefore strengthens
the paper's HAI story (any fixed channel-wise predictor, linear or boosted, collapses under recording
drift while the latent density does not) at the cost of the SWaT lead.

### 1.6 Option (c): boosted channel-wise residual as an additional global expert in the LatAD fusion (script `loo_fusion_boosted.py`, JSON `loo_fusion_boosted.json`, cached scores `boosted_loo_<ds>.npz`)

How the expert enters the combiner. In `ensemble_final`, the density-fusion headline is HCcoh+LatAD =
z(HC_coh | calibration slice) + zl, where zl = z(surv(lat_tr, lat) | surv(lat_tr, lat_tr)) is the rank-survival
upper tail of the global LatAD score against TRAIN-normal, z-scored on the train-normal tail. The boosted
channel-wise LOO residual enters by the identical transform: zb = z(surv(b_tr, b_te) | surv(b_tr, b_tr)), with
b_tr the 5-fold cross-fitted (held-out) TRAIN-normal boosted residuals and b_te the test residuals from the
full-train fit. Augmented z-sum fusion = z(HC_coh) + zl + zb (unit weights, as for the existing two terms);
augmented max-rule headline = max(z(HC), zl, zb). No weight, gate, threshold or seed was chosen on test; the
boosted expert is deterministic (HGB random_state 0), the fusion stays 5-seed through the other terms.
Invariants: linear LOO reproduces the npz to 6e-8 on all three datasets; the current headlines reproduce the
paper (HAI HCcoh+LatAD 0.845 / global 0.811; SWaT 0.837; WADI_clean 0.771 on the current n=30 subset);
boosted SWaT difficult reproduces 0.881.

Difficult / DoubleHard AUROC, before and after (5-seed means; boosted and linres single-vector):

| score | SWaT_canon (85/59) | WADI_clean (30/19) | HAI (167/84) |
|---|---|---|---|
| boosted LOO standalone | 0.881 / 0.830 | 0.660 / 0.553 | 0.321 / 0.304 |
| linres | 0.782 / 0.688 | 0.750 / 0.606 | 0.586 / 0.465 |
| HCcoh+LatAD (current density-fusion headline) | 0.837 / 0.773 | 0.771 / 0.662 | **0.845 / 0.814** |
| null+HC (current primary headline) | 0.811 / 0.735 | 0.749 / 0.633 | 0.770 / 0.704 |
| AUG HCcoh+LatAD+B | **0.871 / 0.818** | 0.773 / 0.660 | 0.769 / 0.719 |
| AUG cohmax+LatAD+B | 0.868 / 0.817 | 0.748 / 0.648 | 0.721 / 0.678 |
| AUG null+HC+B (max rule) | 0.811 / 0.735 | 0.760 / 0.647 | 0.707 / 0.626 |
| AUG LatAD+B (global, no communities) | 0.874 / 0.820 | 0.681 / 0.584 | 0.667 / 0.633 |

Episode-block bootstrap (ensemble_final.boot, 2000 reps), paired diff, 95% CI, one-sided P(diff <= 0):
- SWaT, AUG HCcoh+LatAD+B minus boosted standalone: Difficult -0.010, CI [-0.052, 0.039], P = 0.68;
  DoubleHard -0.012, CI [-0.073, 0.061], P = 0.68. Parity reached (the gap is inside the CI both ways); the
  augmented fusion is significantly above the current headline (+0.034, CI [0.015, 0.051], P < 0.0005;
  DoubleHard +0.045, CI [0.019, 0.069]) and above linres (+0.089 / +0.131, P < 0.0005).
- WADI, augmented minus current headline: Difficult +0.002, CI [-0.019, 0.028], P = 0.43; DoubleHard -0.002,
  CI [-0.028, 0.033]. No regression, no gain; vs linres +0.022, CI [-0.144, 0.186] (tie, 8 episodes).
- HAI, augmented minus current headline: **Difficult -0.077, CI [-0.126, -0.036], P(<=0) = 1.0; DoubleHard
  -0.095, CI [-0.164, -0.046]**. The reverse test (current minus augmented) is significant at P < 0.0005 on both
  subsets. The near-chance expert (0.321 standalone, test-normal false-alarm rate 66.6% at its own cross-fitted
  train-p99) drags the z-sum down by 0.077, and the max rule by 0.063 (0.770 to 0.707).

Invariant 4 (a near-chance expert must not drag HAI down under the fixed rule): VIOLATED. The z-sum with unit
weights has no mechanism to discount an expert whose train-normal calibration no longer describes the test
recording; on HAI the boosted term's z is large on two thirds of the test normals (drift), which is exactly the
failure of 1.1 imported into the fusion. Verdict for (c) under the no-test-tuning constraint: FAILS. SWaT parity
is reachable (0.871 vs 0.881, tie), WADI is unaffected, but HAI regresses significantly, so the augmented fusion
cannot replace the headline.

Exploratory (clearly not the primary number): a TRAIN-ONLY stationarity gate. Fit the boosted LOO on the
chronologically first 80% of the TRAIN recording and measure the false-alarm rate of the last 20% at the
fit-slice in-sample p99 (label-free, test-free; mirrors the experts' 80/20 calibration split). Result: SWaT_canon
0.000 (median residual ratio hold/fit 0.95), HAI 0.537 (ratio 8.7), WADI_clean 0.985 (ratio 305). The ordering
matches the actual test-normal drift of the boosted expert (SWaT 0.6%, WADI 15.8%, HAI 66.6% false alarms at its
cross-fitted train-p99): the channel-wise relations are stationary inside the SWaT training recording and
non-stationary inside HAI's and WADI's. Any cut-off between 0.01 and 0.5 turns the expert ON for SWaT only,
which would give SWaT 0.871 / 0.818 (tie with boosted), WADI 0.771 / 0.662 and HAI 0.845 / 0.814 unchanged. Two
cautions keep this exploratory: the cut-off is still a choice made after seeing three datasets, and the gate
also switches the expert off on WADI where including it was harmless (+0.002). It is reportable as a
train-normal diagnostic of relation stationarity that predicts where a fixed channel-wise predictor will fail,
which is the drift thesis of 1.1 stated as a deployable check, not as a tuned headline.

Overall verdict after 1.5 and 1.6: fall back to option (a). Report the boosted channel-wise LOO as a standalone
baseline row; state SWaT as a tie against it (0.837 vs 0.881, P = 0.074; augmented fusion 0.871 ties it at
P = 0.68 but costs HAI); keep the HAI clean win (0.845 vs 0.321 boosted / 0.586 linear) and the drift-robustness
thesis as the paper's contribution; WADI-difficult conceded as a tie with the linear baseline.

### 1.7 Trying to win SWaT by inspection: LatAD understatement (DIG 1) and boosted-LOO inflation (DIG 2, 3)

Scripts: `dig_swat.py` (+ `dig_swat.json`): aggregation sweep, community coverage, actuator ablation, per-window dump;
`partition_variants.py` (+ `partition_variants.json`): train-normal properties of 36 partition variants;
`experts_local.py` (CPU re-implementation of `modal_experts.experts`, use_full=1) with `partition_eval_SWaT.json`;
`dig_degenerate.json`: floor-pinned communities. Invariants: aggregation sweep reproduces the current headlines
(SWaT 0.837/0.773, WADI 0.771/0.662, HAI 0.845/0.814); local CPU re-training of the CURRENT SWaT partition gives
0.846/0.784 (Modal 0.837/0.773, seed/CPU variation about 0.01); boosted 0.881/0.830 reproduced.

DIG 2, leakage and one-hot inflation in the boosted LOO: none found.
- Code trace: `build_feats` standardises continuous columns with TRAIN mean/std and enumerates discrete states from
  TRAIN only; every HGB is fit on `Fn` (train-normal) only; test residuals come from those models; the 5-fold
  cross-fitted train residuals use train rows only; labels never enter. Test-normal false-alarm rate at the
  cross-fitted train-p99 is 0.6%, so the test residual is calibrated, not drift-inflated.
- One-hot inputs: on SWaT the discrete (one-hot) channels are the 18 actuators MV101, P101, P102, MV201, P201,
  P203, P205, MV301-304, P301, P302, P402, P403, UV401, P501, P602 (8 further actuators are constant in
  train and dropped). Attacks do manipulate several of them. Re-scoring the channel-wise baselines with those
  inputs removed: drop ALL 18 actuator channels (as targets and predictors): boosted 0.879 / 0.830, linres
  0.765 / 0.676; drop only the attacked actuators: boosted 0.878 / 0.829; drop ALL attacked channels of any type:
  boosted 0.757 / 0.690, linres 0.616 / 0.550. The boosted lead does not come from reading manipulated actuator
  states; it comes from the continuous sensors (levels, flows, pressures), and removing the attacked sensors
  themselves removes signal for every detector alike. LatAD sees the same raw actuator channels as features, so
  there is no input asymmetry either.

DIG 1, LatAD understatement on SWaT: no bug, one inert quirk, no partition or aggregation win.
- Heads and gates: in the SWaT community experts (seed 0, local re-fit) the whitened-residual head is ON in 24 of
  25 communities (generalisation ratio 0.57 to 1.36; the one OFF community has ratio 8.2, correctly rejected),
  the basin head is OFF in all (train ambiguity fraction <= 0.007, crisp modes, by design); the global K=40 model
  has residual ON (ratio 0.80) and basin OFF. Nothing is mis-gated.
- Degenerate communities: three SWaT communities (P402/UV401/P501; AIT201/MV201/P201; MV101/AIT201/MV201/P201)
  have calibration-surprise sd < 1e-14 and sit at the p-value floor (1e-4) on 100% of test windows, normal and
  anomalous alike; a fourth (MV101/AIT201/AIT202/MV201/P201/AIT501) is at the floor on 63.5% of test normals
  (drift). Excluding the three by the train-normal rule already used in the localization evaluation
  (calibration sd < 1e-3 in any seed) leaves HCcoh+LatAD unchanged (0.837 / 0.772) and moves cohmax+LatAD from
  0.829 to 0.834: a constant contribution does not change the ranking. WADI and HAI have no degenerate
  communities. Inert; worth cleaning, not a source of understatement.
- Coverage (train-normal structure vs. attacked channels, `framing_loc/loc2_episodes_SWaT_canon.jsonl`): in 9
  of 23 difficult episodes no single community holds every attacked channel (multi-point attacks such as
  19/20/21 AIT504+LIT101+MV101 or 24/25 across P2/P4), but single-point attacks (LIT101, LIT301, LIT401, DPIT301,
  MV303, MV304, FIT401) sit inside 1 to 6 communities each, and the community that contains the target reaches
  p <= 0.001 on the episode's difficult windows in 15 of 21 labelled episodes. The exceptions where the target
  community stays quiet are attacks 14 and 17 (MV303 forced closed, target-community p 0.41; attack 14 failed
  physically per the iTrust notes), 13 (MV304, p 0.007), 34/35 (P101/P102, p 0.009) and 27/28 (LIT401/P302,
  p 0.04): actuator-state attacks whose physical effect is small, which the nonlinear level relations of the
  boosted predictor pick up (per-window: w762/763 boosted 0.97/0.99 vs headline 0.59/0.38, driven by
  LIT301/LIT401/LIT101 residuals). Splitting is therefore not the failure mode; the community densities of the
  tank-level subsystems are less sensitive than a nonlinear regression of one level on the others.
- Partition variants (train-normal selection, pre-registered before any test number): among 36 variants
  (linkage average/complete/single/weighted, MAXSZ 15/25/40, MINSZ 3/4, plus flat cuts) the only change that
  raises size-weighted cohesion on all three datasets with coverage kept is MAXSZ 25 to 15 with the current
  average linkage (SWaT 0.679 to 0.723, WADI 0.727 to 0.756, HAI 0.774 to 0.801); flat partitions have higher
  cohesion but lower coverage than the current cover and were excluded. Trained locally on SWaT (5 seeds, same
  code path): MAXSZ=15 gives HCcoh+LatAD 0.837 / 0.775 versus the locally re-trained current partition
  0.846 / 0.784; paired bootstrap variant minus current -0.009, CI [-0.017, -0.003]. The pre-registered variant
  is slightly worse; no partition win. WADI/HAI trainings for it were stopped as moot.
- Aggregation sweep (fixed rules over the SAME experts, all three datasets; Difficult AUROC SWaT / WADI / HAI):
  HCcoh+LatAD (current) 0.837 / 0.771 / 0.845; cohmax+LatAD 0.829 / 0.741 / 0.832; Tippett max 0.804 / 0.712 /
  0.809; Fisher sum 0.805 / 0.698 / 0.827; cohesion-weighted Fisher 0.799 / 0.705 / 0.825; top-3 sum 0.804 /
  0.743 / 0.842; top-5 sum 0.817 / 0.743 / 0.844; Stouffer 0.805 / 0.696 / 0.827; HC without cohesion weights
  0.826 / 0.760-0.769 / 0.792-0.794; HCcoh with cutoff alpha0 in {0.1, 0.2, 0.5}: identical on SWaT (0.837) and HAI
  (0.845-0.846), 0.781-0.782 on WADI. No pooling beats the current rule on SWaT; the cohesion-weighted HC is the
  best or tied-best rule on every dataset, so the current combiner is not the SWaT bottleneck.
- Local re-training itself (Modal 0.837 to local 0.846) shows the SWaT headline has about 0.01 of seed/hardware
  variance; against the boosted LOO the locally re-trained current partition is still behind (-0.035, CI
  [-0.089, 0.022], P = 0.89 one-sided for a LatAD lead).

DIG 3, per-window split of the boosted-minus-headline gap (|gap| > 0.2, 15 windows). Boosted wins (10): w20-22
(attacks 4/6/7, MV504 opened on the RO plus AIT202/LIT301: PIT502 residual share 0.47-0.68, physically the RO
pressure reacting to MV504, a genuine cross-stage effect that the PIT502 communities do not resolve); w342
(attack 17, MV303: FIT101/MV101 residual); w390 (19-21: LIT101/LIT301); w762-763 (27/28: LIT301/LIT401/LIT101
tank-level relations, target community p 0.04); w929 (29, dosing pumps: LIT101); w245-246 (unlabelled label
run with no attack row; both detectors low). Headline wins (5): w306-307 (14, MV303), w310/312 (16, LIT301),
w1007 (31, LIT401), each with 3 to 9 communities at p < 0.01, a dense multi-community violation the HC rule is
built for. Classification: the boosted wins are genuine channel-wise nonlinear relations among tank levels,
flows and the RO pressure ("LIT401 given LIT301 and P302 state"), where a community density over 3 to 13
channels is a coarser instrument than a regression of one channel on all others; none is an artefact, a
label-adjacent state or a single-sensor quirk (PIT502 removal leaves boosted at 0.870, section 1.5). There is no
LatAD gap that a train-normal-justified construction change closes.

### 1.8 Per-window root cause on the SWaT windows where the headline loses to boosted (script `dig_swat_perwindow.py`, JSON `dig_swat_perwindow.json`)

44 of the 85 difficult windows have headline percentile below boosted percentile (40 labelled, 4 in the
unlabelled label run). Driver classification of boosted's top-1 residual channel against the iTrust attack
points: direct (an attacked point) 37.5%, adjacent (same or neighbouring stage, physically downstream) 32.5%,
unrelated (PIT502, AIT503, FIT601, MV101/FIT101 on a P3 attack) 30.0%; any top-3 driver direct 47.5%;
gap-weighted share of the edge with a direct-or-adjacent top-1 driver 50.2%.

Per-window table (w = window index; pct = percentile vs test normals; tgt p = smallest upper-tail p among the
non-degenerate communities containing an attacked channel; best = community with the smallest p; driver
shares from the boosted per-channel residual; lever a = partition, b = aggregation, c = channel-wise expert,
d = nothing, WR = wrong reason):

| w | attacks: points | boosted / headline pct | tgt p | best community | boosted top-3 drivers | top-1 class | root cause | lever |
|---|---|---|---|---|---|---|---|---|
| 4 | 1,2: MV101,P102 | 0.938 / 0.769 | 1e-4 | 18 MV101/AIT201/AIT202/MV201/P201/AIT501 | PIT502 .71, AIT501 .11, AIT201 .06 | unrelated | target fires, HC rank lower | b |
| 5-8 | 1,2: MV101,P102 | 0.98-1.00 / 0.82-0.98 | 1e-4 | 18 | LIT101 .5-.9, FIT101 .25-.37 | adjacent | target fires, HC rank lower (gap <= 0.16) | b |
| 15 | 3: LIT101 | 0.987 / 0.806 | 0.020 | 18 | PIT502 .85, LIT401 .06 | unrelated | target quiet; PIT502 Dec-28 offset | WR |
| 16-17 | 3: LIT101 | 0.99 / 0.98-0.99 | 3e-4, 0.014 | 15, 18 | PIT502 .62-.70, LIT301 .18 | unrelated | gap <= 0.006 | b / WR |
| 20-22 | 4,6,7: AIT202,LIT301 (+MV504) | 0.91-0.93 / 0.35-0.40 | 1e-4 to 0.0065 | 18 | PIT502 .47-.68, AIT202 .09, MV101 .16 | unrelated (AIT202 direct in top-3) | one community fires; HC needs two; PIT502 reacts to MV504 on the RO | b |
| 25 | 4,6,7 | 0.994 / 0.991 | 1e-4 | 10 | LIT301 .74 | direct | gap 0.003 | b |
| 245-248 | none (unlabelled run) | 0.10-0.53 / 0.01-0.20 | n/a | 23 (p 0.03-0.06) | PIT502 .55-.76, AIT503 .75 | unrelated | both low; no attack row | d |
| 301 | 13: MV304 | 0.332 / 0.230 | 0.097 | 23 (p 0.04) | LIT401 .45, FIT601 .13 | adjacent | both low; small physical effect | c |
| 308 | 14: MV303 (failed attack) | 0.876 / 0.820 | 0.41 | 18 | PIT502 .46, AIT504 .08 | unrelated | target quiet; PIT502 offset | WR |
| 342 | 17: MV303 | 0.873 / 0.458 | 0.41 | 18 | MV101 .41, FIT101 .37, LIT401 .15 | unrelated | target quiet; boosted fires on P1 | WR |
| 343, 345 | 17: MV303 | 0.98-0.99 / 0.96 | 5e-4 | 10 MV301/FIT601/P602 | FIT601 .97; LIT401 .55 | unrelated / adjacent | gap <= 0.03 | b |
| 385, 391-392 | 19-21: AIT504,LIT101,MV101 | 0.83-0.98 / 0.82-0.96 | 1e-4 | 7, 18 | AIT503 .91; LIT101 .4-.6 | adjacent / direct | gap <= 0.05 | b |
| 390 | 19-21 | 0.992 / 0.620 | 0.0042 | 18 | LIT101 .52, LIT301 .32 | direct | one community fires weakly; HC rank low | b |
| 444 | 22: AIT502,P501,UV401 | 0.999 / 0.998 | 1e-4 | 2 FIT401/FIT501/FIT502 (13 fire) | FIT401 .17, AIT504 .16, FIT502 .15 | adjacent | gap 0.001 | b |
| 759 | 27,28: LIT401,P302 | 0.994 / 0.994 | 1e-4 | 15 | LIT401 .95 | direct | gap 0.001 | b |
| 762-763 | 27,28: LIT401,P302 | 0.97-0.99 / 0.38-0.59 | 0.042-0.047 | 18 (p 0.01) | LIT301 .52 / LIT401 .43, LIT101 | adjacent / direct | target community quiet; nonlinear tank-level relation LIT401 given LIT301, P302 | c |
| 929 | 29: P201,P203,P205 | 0.833 / 0.505 | 1e-4 | 18 | LIT101 .70, AIT501 .11 | adjacent | one community fires; HC rank low | b |
| 1159 | 32: LIT301 | 0.998 / 0.988 | 1e-4 | 15 | LIT301 .95 | direct | gap 0.01 | b |
| 1203-1204 | 33: LIT101 | 0.987 / 0.85-0.87 | 0.0024-0.0047 | 18 | LIT101 .75-.80 | direct | one community fires weakly | b |
| 1239-1241 | 34,35: P101,P102 | 0.98 / 0.94-0.96 | 0.009-0.073 | 18, 15 | MV201 .33-.41, LIT301, AIT503 | adjacent / unrelated | pump attack split across P1/P2 communities | a / b |
| 1298-1299 | 36: LIT101 | 0.99-1.00 / 0.88-0.98 | 0.0023-0.0034 | 18 | LIT101 .92 | direct | one community fires weakly | b |
| 1458 | 37-40: AIT402,AIT502,FIT401,FIT502,P501 | 0.998 / 0.994 | 1e-4 | 13 (7 fire) | AIT501 .39, LIT101 .22 | adjacent | gap 0.005 | b |
| 1477 | 41: LIT301 | 0.579 / 0.417 | 0.081 | 18 | FIT601 .34, AIT503 .33 | unrelated | both low; offsets | WR |
| 1479-1483 | 41: LIT301 | 0.998 / 0.99 | 1e-4 | 15 | LIT301 .79-.94 | direct | gap <= 0.006 | b |

Lever counts over the 44 windows: b (aggregation / fusion rank) 30, of which 19 have gap <= 0.05 and are
not material; a (partition) 2; c (channel-wise nonlinear relation outside the community density) 3; d
(unlabelled run) 4; wrong reason 5. The recurring structure behind "b" is community 18
(MV101/AIT201/AIT202/MV201/P201/AIT501): it is the best-scoring community in 27 of the 44 windows, but it is
also at the p floor on 63.5% of TEST-NORMAL windows (section 1.7), so it fires on normals and anomalies alike;
under Higher Criticism a permanently firing community raises the baseline and a genuine single-community
violation (LIT101, LIT301, LIT401 tank-level attacks) no longer stands out. That is a test-recording drift of the
P1/P2 analyser community, the same mechanism as in 1.1, inside the community fusion. The eleven fixed poolings
of section 1.7 do not cure it (none beat HCcoh on SWaT), because the drift is in the p-values themselves.

Route 1 (honest-attack subset). Restricting to difficult windows whose boosted top-1 driver is a directly
attacked or physically adjacent channel (62 of 85; 21 episodes), the paired bootstrap headline minus boosted is
-0.026, CI [-0.064, 0.026], P(<= 0) = 0.85 (all 85: -0.044, CI [-0.096, 0.017]); double-hard restricted (39 of
59): -0.034, CI [-0.097, 0.044]. On the losing windows alone: honest subset -0.123, wrong-reason subset -0.216.
Removing the wrong-reason windows halves LatAD's deficit but does not reverse it: the boosted edge is half
right-channel, and on that half it still leads numerically. Route 1 does not produce a win.

Route 2 (drift-gated boosted expert, a-priori budget). Gate = held-out FP rate of the boosted residual on the
chronologically last 20% of the TRAIN recording at the fit-slice p99 (label-free, test-free; section 1.6):
SWaT_canon 0.000, HAI 0.537, WADI_clean 0.985. At a 5% false-alarm budget (also at 1%, 10% or 20%; the three
values are separated by two orders of magnitude) the expert is ON for SWaT only. Gated-augmented fusion,
Difficult / DoubleHard: SWaT 0.871 / 0.818 (boosted 0.881 / 0.830; paired diff -0.010, CI [-0.052, 0.039],
P = 0.68), WADI 0.771 / 0.662 (unchanged, expert OFF), HAI 0.845 / 0.814 (unchanged, expert OFF). HAI and WADI
do not regress; SWaT rises significantly over the current headline (+0.034, CI [0.015, 0.051]) but stays 0.010
below the boosted standalone. Route 2 yields parity, not a numerical win.

Reconciling 0.837 (Modal artifact) vs 0.846 (local rebuild) for the current partition: per-seed HCcoh+LatAD
difficult AUROC is Modal [0.830, 0.841, 0.835, 0.843, 0.837] and local [0.852, 0.844, 0.840, 0.847, 0.846]; the
ten-seed pooled mean is 0.841. The two are one seed-spread apart (each 5-seed sd about 0.005, plus CPU vs Modal
torch non-determinism); the Modal artifact is the number the paper's tables were built from and stays the
headline; 0.841 +- 0.006 is the fair statement of the SWaT headline across ten seeds. Neither reaches 0.881.

Community-level train-only gate (the last lever, script `comm_gate_eval.py` on a local re-training that also
stores the in-sample fit-slice scores, `sota_bundle/experts_variants/cur_fit`). For each community, FP = share
of the chronologically held-out calibration slice (last 20% of train) above the fit-slice 99th percentile;
drop communities above the same a-priori budget as the boosted gate. Result on SWaT: every community has
held-out-train FP <= 1.5% (max 0.015 for P402/UV401/P501), so at the 5% or 10% budget nothing is dropped and
the headline is unchanged (0.846 / 0.784 on this re-training); at 1% one community is dropped and the AUROC
moves by -0.003. The within-train stationarity of a community is uncorrelated with its test-recording floor
rate (Pearson -0.02 across the 25 communities): the P1/P2 analyser drift that suppresses single-community
violations happens between the normal and attack recordings and is invisible from the training recording, so
no train-only rule can remove it. The lever does not exist under the no-test-tuning bar.

### 1.9 Drift-robust community nulls: startup trim, robust re-centering, abrupt-residual scoring (scripts `drift_fix_eval.py`, `detrend_eval.py`, `trim_fair_swat.py`; JSON of the same names)

FIX A, SWaT startup trim (documented-artifact removal, MAD-GAN's 21,600 s). Already in effect by construction:
`eda_real._raw_swat_canonical` drops `warmup_drop = 0.02` of the normal file before downsampling; the file has
1,387,098 one-second rows, so 27,741 s (7.7 h) are discarded, more than the documented 21,600 s. FIX A alone is
therefore the current baseline. Observation to check separately: 1,387,098 rows is 385 h, 2.8x the canonical
7-day Dec-2015 normal recording (496,800 rows); the paper says "the Normal recording". The SWaT TEST prefix is
bounded by attack 1 at 1,754 s (window 4), so at most windows 0-3 (0-1,200 s, all normal) could be trimmed:
doing so leaves HCcoh+LatAD at 0.837 / 0.773 and boosted at 0.882 / 0.831 (0 attack windows dropped). A further
21,600 s train trim beyond the loader's warm-up (72 windows, 1.6% of train) is evaluated as a trim-consistent
comparison in 1.10 with every retrained method on the same trimmed train. WADI: the minority 2,160 s trim is 7
of 2,614 train windows (0.27%), reported in 1.10. HAI: no trim.

FIX B, robust re-centering (uniform method feature, a-priori: shift each community's test scores so their
median equals the calibration median; no test labels). Difficult / DoubleHard before -> after: SWaT 0.837 /
0.773 -> 0.836 / 0.770 (paired -0.001, CI [-0.008, 0.007]); WADI 0.771 / 0.662 -> 0.767 / 0.657 (-0.004, CI
[-0.019, 0.014]); HAI 0.845 / 0.814 -> 0.845 / 0.812 (-0.001, CI [-0.003, 0.002]). Masking: no episode's median
percentile drops by more than 0.1 on any dataset; detection at percentile >= 0.95 over all anomalies moves
0.854 -> 0.867 (SWaT), 0.554 -> 0.571 (WADI), 0.771 -> 0.785 (HAI). A no-op everywhere: the SWaT drift of the
P1/P2 analyser community is not a pure additive offset (its test distribution is also spread), so a location
shift restores nothing. On the local SWaT re-training it costs -0.012 (CI [-0.023, -0.002]). Not adopted.

Abrupt-residual scoring (uniform method feature). Causal slow trend per community = median of the previous K
windows, initialised at the calibration median (train-normal level); fast residual = score - trend; p-value of
the residual against the calibration slice's residuals computed identically; fusion unchanged. Timescale
fixed a priori at 24 h (between-recording drift is day-scale; the longest SWaT attack, #28, is 9.5 h (121
windows) and must be shorter than the trend window); K = 288 windows on SWaT/WADI (300 s each), 2,880 on HAI
(30 s each). Sensitivity at 6 / 12 / 48 h reported, not selected.

| dataset | base | 6 h | 12 h | **24 h (a priori)** | 48 h | 24 h vs base (paired) | 24 h vs boosted (paired) |
|---|---|---|---|---|---|---|---|
| SWaT_canon Difficult / DoubleHard | 0.837 / 0.773 | 0.848 / 0.784 | 0.868 / 0.814 | **0.860 / 0.802** | 0.849 / 0.789 | +0.023, CI [-0.001, 0.055], P = 0.034 | -0.021, CI [-0.064, 0.035], P = 0.80; DoubleHard -0.028, CI [-0.092, 0.053] |
| WADI_clean | 0.771 / 0.662 | 0.748 / 0.630 | 0.759 / 0.645 | **0.771 / 0.663** | 0.766 / 0.656 | 0.000, CI [-0.014, 0.022] | +0.110, CI [-0.098, 0.277] |
| HAI | 0.845 / 0.814 | 0.859 / 0.837 | 0.852 / 0.829 | **0.848 / 0.822** | 0.845 / 0.817 | +0.002, CI [-0.006, 0.011] | +0.527, CI [0.404, 0.646] |

Masking check at 24 h: no episode on any dataset has its median percentile drop by more than 0.1 (SWaT 25
episodes incl. the 9.5 h attack 28: window-level drops > 0.1 on 1 anomaly window, gains on 7; WADI 1 / 0; HAI
3 / 1). Detection at percentile >= 0.95, all anomalies: SWaT 0.854 -> 0.880, difficult 0.635 -> 0.671; WADI
0.554 -> 0.571; HAI 0.771 -> 0.756 (difficult 0.401 -> 0.401). At 6 h WADI masks the 0.5 h episode at window 16
(0.724 -> 0.580), which is why a short timescale is unsafe and the day-scale choice matters.

Reading: the abrupt-residual null is a clean drift-robust mechanism (holds HAI and WADI within +-0.002, masks
no attack at the a-priori timescale, lifts SWaT by +0.023 with P = 0.034 one-sided), but it does not flip SWaT:
0.860 vs boosted 0.881, CI includes 0, a tie. The 12 h setting would read 0.868 and is reported only as
sensitivity. Worth stating in the paper as the method's drift-robust calibration if the community model is
kept, with the timescale rule (longer than the longest plausible attack) stated as the design principle.

Verdict for SWaT: a genuine tie. No LatAD bug understates it, no boosted inflation overstates it; the only
train-normal-justified changes tested (degenerate-community exclusion, tighter partition, eleven poolings)
move the headline by at most +0.005 and none reaches the boosted 0.881. SWaT stays "leads the deep detectors and
the linear baseline, ties the nonlinear channel-wise predictor".

## 2. Paper identification

There is no paper co-authored by Sarfraz and Garg on time-series anomaly detection that I could find. Two
distinct papers are being conflated, and the manuscript (IoT2.html) already cites both:

1. **Sarfraz, M.S.; Chen, M.-Y.; Layer, L.; Peng, K.; Koulakis, M. "Position: Quo Vadis, Unsupervised Time
   Series Anomaly Detection?" ICML 2024 (PMLR 235). arXiv:2405.02678.** https://arxiv.org/abs/2405.02678
   Claim: the field is held back by flawed metrics (point-adjusted F1), inconsistent benchmarking and unjustified
   design choices; simple baselines match or beat deep SOTA; distillation shows the deep models effectively learn
   linear mappings (linear students agree closely with the deep teachers, their Table 4). Baselines: sensor range
   deviation, L2-norm, nearest-neighbour distance, PCA reconstruction error, plus single-block networks (MLP,
   MLPMixer, Transformer, GCN-LSTM). Datasets: SWaT, WADI (127- and 112-channel variants), SMD, UCR, with
   SMAP/MSL in an appendix. Reported SWaT/WADI-112 F1 for PCA error 0.833 / 0.655 versus GDN 0.810 / 0.571 and
   TranAD 0.799 / 0.511 (https://arxiv.org/html/2405.02678). It does NOT contain a channel-wise leave-one-out
   linear regression baseline. Cited in the manuscript as [22].
2. **Garg, A.; Zhang, W.; Samaran, J.; Savitha, R.; Foo, C.-S. "An Evaluation of Anomaly Detection and
   Diagnosis in Multivariate Time Series." IEEE TNNLS 33(6):2508-2517, 2022. DOI 10.1109/TNNLS.2021.3105827,
   arXiv:2109.11428.** https://arxiv.org/abs/2109.11428 , code https://github.com/astha-chem/mvts-ano-eval
   Claim: over a grid of 10 models and 4 scoring functions on CPS benchmarks, a simple channel-wise model (the
   univariate fully connected autoencoder) with a dynamic Gaussian scoring function wins both detection and
   diagnosis, beating the deep SOTA. Cited in the manuscript as [7]. This is the closest published relative of
   our LinRes result: a per-channel model, no cross-channel deep architecture, on top.

Related critiques (all with URLs; the first four are already in the manuscript):
- Kim, S.; Choi, K.; Choi, H.-S.; Lee, B.; Yoon, S. "Towards a rigorous evaluation of time-series anomaly
  detection." AAAI 2022. https://doi.org/10.1609/aaai.v36i7.20680 (point adjustment lets a random score beat SOTA).
- Wu, R.; Keogh, E.J. "Current time series anomaly detection benchmarks are flawed and are creating the illusion
  of progress." IEEE TKDE 35(3):2421-2429, 2023. https://doi.org/10.1109/TKDE.2021.3112126
- Pinet, M.; Cumin, J.; Berlemont, S.; Vaufreydaz, D. "Anomalies in Multivariate Time Series Benchmarks Are
  Mostly Univariate." MiLeTS @ KDD 2026, arXiv:2606.02670. https://arxiv.org/abs/2606.02670 (on six of eight
  benchmarks at least half the anomaly segments deviate univariately on 89 to 100% of timesteps; no cross-channel
  rupture without a univariate deviation; channel-dependent modelling brings no measurable gain on real
  benchmarks). Manuscript ref [18].
- Audibert, J.; Michiardi, P.; Guyard, F.; Marti, S.; Zuluaga, M.A. "Do deep neural networks contribute to
  multivariate time series anomaly detection?" Pattern Recognition 132:108945, 2022. arXiv:2204.01637.
  https://arxiv.org/abs/2204.01637 (16 conventional/ML/deep methods on five datasets; no family significantly
  outperforms). Not yet cited (the manuscript cites Audibert only for USAD).
- Sehili, M.E.A.; Zhang, Z. "Multivariate Time Series Anomaly Detection: Fancy Algorithms and Flawed Evaluation
  Methodology." TPCTC 2023, arXiv:2308.13068. https://arxiv.org/abs/2308.13068 (a PCA baseline beats many recent
  deep methods under rigorous protocols). Not yet cited.
- Wagner, D. et al. "TimeSeAD: Benchmarking deep multivariate time-series anomaly detection." TMLR 2023.
  https://ml.cs.rptu.de/publications/2023/TimeSeAD.pdf (already cited).

## 3. Positioning

Does LinRes-beats-deep-on-WADI reproduce Sarfraz et al.? Yes in spirit, no in detail. Their finding is that
simple baselines (range, L2, NN distance, PCA error) match or beat deep SOTA on SWaT/WADI under non-adjusted F1,
and that the deep models are effectively linear. Our WADI_clean result is a stronger, cleaner instance of the
same phenomenon: a linear cross-channel predictor beats USAD, TranAD, AE and IF on the whole set (0.834 vs at
most 0.792) and on the trivial-filtered difficult subset (0.750 vs at most 0.638 for global LatAD, 0.629 AE),
with the mechanism identified (low-rank, state-conditional, stationary linear relations; 1.1 to 1.2). It also
reproduces Garg et al.'s channel-wise finding (a per-channel model with no cross-channel deep architecture wins)
and is consistent with Pinet et al.'s diagnosis that SWaT/WADI anomalies are mostly univariate. The
contribution boundary is therefore: we do not claim linear-beats-deep; we adopt these critiques as the
evaluation protocol, we show the strongest instance of them on WADI, and we show that the latent density beats
both the deep detectors AND the linear baseline exactly where the linear relations are non-stationary across
recordings (HAI: LinRes 0.586 vs LatAD 0.814, with 36% test-normal false alarms for LinRes at its own
train-p99) or genuinely cross-channel (SWaT: 0.782 vs 0.808, significant for the regime-community model per
`headline_clean_results.md`), and we concede the WADI-difficult tie. One correction to the SWaT claim (section 1.5): a
construct-matched nonlinear channel-wise predictor (boosted LOO) scores 0.881 / 0.830 on SWaT difficult /
double-hard, numerically above the HCcoh+LatAD headline (0.837 / 0.773) with neither gap significant (P
0.074 / 0.087, CIs include 0) and significantly above null+HC and the global model. The SWaT lead therefore
holds against the deep detectors and the LINEAR baseline (significant), and is a numerical tie against the
nonlinear channel-wise baseline on both subsets, including double-hard. The recommended framing is option (a):
add the boosted LOO as a baseline row, state SWaT as "leads the deep detectors and the linear baseline, ties a
nonlinear channel-wise predictor", and carry the clean win on HAI, where the same boosted predictor collapses
to 0.321 under recording drift while LatAD holds 0.814. WADI-difficult stays a tie with the linear baseline
(boosted 0.660 is below LinRes 0.750 there).

Ready-to-paste Related Work sentence (citation numbers as in IoT2.html: Sarfraz et al. [22], Garg et al.
[7], Pinet et al. [18]):

> Our leave-one-channel-out linear baseline reproduces, on cleaned WADI, the finding of Sarfraz et al. [22] and
> Garg et al. [7] that a simple channel-wise or linear model beats reconstruction- and transformer-based
> detectors once point adjustment is removed, and we trace it to the low-rank, actuator-state-conditional and
> recording-stationary linear relations that WADI's difficult anomalies violate (cf. the univariate anomaly
> structure reported by Pinet et al. [18]); the contribution of LatAD lies where those linear relations drift
> between recordings (HAI) or the anomalies are cross-channel (SWaT), and there its latent density leads the
> linear baseline as well as the deep detectors, while on WADI it ties.

Shorter one-liner:

> We do not claim that linear beats deep; Sarfraz et al. [22] and Garg et al. [7] established that, and our
> WADI results reproduce it. We show where the linear baseline itself breaks (train-to-test drift of the
> cross-channel relations on HAI, 36% false alarms at its own train threshold) and that a latent density leads
> both the linear and the deep detectors there.

## Key numbers table (difficult subset, AUROC vs test normals unless stated)

| | WADI_clean (n=30) | SWaT_canon (n=85) | HAI (n=167) |
|---|---|---|---|
| LinRes (canonical) | 0.750 | 0.782 | 0.586 |
| LinRes vs TRAIN normals | 0.878 | 0.868 | 0.947 |
| LinRes test-normal FP at train-p99 | 0.6% | 2.0% | 36.2% |
| Global LatAD (5-seed mean) | 0.638 | 0.808 | 0.814 |
| LatAD test-normal FP at train-p99 | 1.0% | 2.5% | 9.6% |
| AE / USAD / TranAD | 0.629 / 0.579 / 0.613 | 0.730 / 0.658 / 0.655 | 0.758 / 0.497 / 0.445 |
| LOO boosted (HGB) | 0.660 | 0.881 | 0.321 |
| LOO linear per-regime (K=8) | 0.689 | 0.820 | 0.497 |
| LinRes continuous-only / discrete-only / no one-hot | 0.755 / 0.578 / 0.631 | 0.779 / 0.642 / 0.767 | 0.586 / 0.434 / 0.586 |
| Participation ratio diff / easy / normal | 2.5 / 4.1 / 6.8 | 6.2 / 1.0 / 4.8 | 6.7 / 3.5 / 6.3 |
| Raw residual top-1 / top-3 share, n_eff | 0.26 / 0.59, 11.6 | 0.44 / 0.79, 5.2 | 0.36 / 0.71, 7.1 |
| Per-window wins LinRes / LatAD (margin 0.05) | 18 / 6 | 17 / 26 | 37 / 124 |
