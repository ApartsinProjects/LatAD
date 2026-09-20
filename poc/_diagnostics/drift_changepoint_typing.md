# Drift-vs-changepoint anomaly typing on the clean community pipeline (HAI, WADI_clean, SWaT_canon)

Script `drift_changepoint_typing.py`, numbers `drift_changepoint_typing.json`, log `drift_changepoint_typing.log`.
Experts: `sota_bundle/experts_full/expert_{HAI,WADI_clean,SWaT_canon}.npz` (SWaT_canon = official Dec-2015 normal,
S=24, `fit_surprise` present, asserted). Baselines and subsets from `scores_<DS>.npz` + `pca_filter_doublehard_<DS>.npz`.
5 seeds, AUROC = seed mean. Bootstrap = `ensemble_final.boot` (episode-block, 2000 reps). Nothing retrained, nothing committed.

## Verdict first

- **SWaT_canon is recovered.** Scoring the changepoint component lifts the community fusion from the drift-swamped
  0.524 Difficult / 0.178 PCA-double-hard to **0.723 / 0.671** (All 0.788 -> 0.880). Against the headline the gain is
  significant on every subset (Difficult +0.200, CI [0.018, 0.355], P=0.014; PCA-double-hard +0.493, CI [0.349, 0.619],
  P<0.001). Against the strongest alternative on the clean SWaT (the trivial max|z| rule) it leads on Difficult
  (+0.096, CI [-0.004, 0.209], P=0.031) and PCA-double-hard (+0.157, CI [-0.013, 0.329], P=0.036): a lead whose CI
  just touches zero, i.e. marginal, not a tie. The normal-window false-alarm rate at the train-calibrated threshold
  falls from 79% to 6.5%.
- **HAI and WADI_clean do not regress.** The same decomposition with the same a-priori 24 h timescale is a near-no-op:
  HAI Difficult 0.845 -> 0.835 (diff -0.010, CI [-0.026, 0.005], P(<=0)=0.88, inside the +-0.02 invariant),
  WADI_clean 0.771 -> 0.779 (+0.008, CI [-0.014, 0.034]).
- **Typing validates against the only ground truth (attack labels).** Of the headline-flagged attacks, 74.8% (SWaT),
  88.5% (HAI), 61.9% (WADI) type as changepoint; of the headline-flagged drifted normals, 97.4% (SWaT) and 92.5% (HAI)
  type as drift. WADI has no drifted normals at all.
- **Masking, checked per attack episode:** under the era-local criterion (rank against the test normals within +-24 h)
  the changepoint score loses two of SWaT's 25 episodes by >0.10 (ep@660: 0.967 -> 0.857; ep@1158: 0.949 -> 0.782,
  the latter inside the 24 h shadow of the 10 h attack #758) and gains five; HAI 0/38, WADI 0/11. The invariant "no
  episode masked" is therefore NOT met on SWaT; both losses are small and localized, and are reported, not hidden.
- **Two invariant results that must be read with the caveat attached:** (a) the synthetic-ramp invariance (I4) holds on
  HAI, is marginal on SWaT (Difficult -0.024 vs the 0.02 tolerance) and fails on WADI_clean (-0.055), whose 2-day record
  is only twice the baseline window; a causal one-sided median lags a ramp by half its window and leaves a residual bias.
  (b) The method reached its final form after one root-cause revision (v1 level-only -> v2 level+scale) prompted by the
  masking check, documented below with both variants' numbers.
- **Side finding (outside scope, must reach the parent):** the SWaT GDN column in `doublehard_pca_all3.md` (0.571 on
  PCA-double-hard, "GDN leads") comes from `score_GDN_SWaT_canon_s0.npy` (mtime 09-18 11:03, pre-official-normal, leaked
  train). The fresh official GDN (`scores_SWaT_canon.npz["GDN"]` = `scores_sota_ms_SWaT_canon.npz`, 09-19 21:17) is
  uncorrelated with it (r = -0.01) and scores 0.115 on that subset. `rev4_doublehard_pca_all3.py` `GDN_FILE["SWaT_canon"]`
  points at the stale dump.

## 1. Drift / changepoint composition (the drift spectrum made concrete)

Uniform decomposition, same a-priori timescale on all three datasets. A test-normal window is DRIFTED if the slow
on-trend baseline itself is off the train distribution (above the calibration 99th pct of the raw surprise) in at
least one community. A community is DRIFTING if that holds on >= 50% of the test-normal windows. Every window flagged
by the headline (seed-mean HCcoh+LatAD above its calibration 99th pct) is typed CHANGEPOINT (changepoint score above
its own calibration 99th pct), DRIFT (not changepoint AND drifted) or SUBTHRESHOLD (neither: threshold noise).

| dataset | test-normal windows drifted | drifting communities | headline-flagged windows: changepoint / drift / subthreshold | flagged NORMALS typed drift | flagged ATTACKS typed changepoint |
|---|---|---|---|---|---|
| WADI_clean | **0.0 %** | **0 / 44** | n=29: 65.5 / 0.0 / 34.5 % | 0 % (n=8; all changepoint or subthreshold) | 61.9 % (n=21; 38.1 % subthreshold, 0 drift) |
| HAI | **29.7 %** (front-loaded: 73 % of first-third normals, 0 % second third, 15 % third; half of all drifted normals lie in hours 0-24) | **0 / 28** by the >= 50 % rule; g1 (P1_PCV02Z, P2_SD01, P2_SIT01) and g11 (same plus P1_LCV01D/Z, P1_PIT01, P4_ST_PT01) carry a level offset that begins ~6 h into the test and decays over the record (slow baseline 1731 -> 219 vs calib -0.24), above the calib 99th pct on 30 % of the normal windows, 83 % of them in the first third | n=2074: 74.2 / 19.8 / 6.0 % | 25.9 % (n=1483) | 88.5 % (n=591) |
| SWaT_canon | **66.6 %** (4 % of first-third normals, 100 % of thirds 2-3) | **7 / 24**, all chemistry / RO / analyser | n=1216: 19.7 / 71.7 / 8.6 % | 81.9 % (n=1002) | 74.8 % (n=214) |

Reading: WADI is stationary (the decomposition is a no-op: no drifted normals, no drifting community, the "drift-type"
share is exactly zero and the non-changepoint share is threshold noise). HAI has a moderate, front-loaded drift (a decaying
level offset in two valve-position / P2-turbine communities, present from ~6 h into the test through the whole record,
plus a smaller late episode); SWaT has a large, monotone, community-localized drift. The expected ordering
WADI < HAI < SWaT is confirmed; the HAI fraction is 30 % (the coordinator's prior of 39 % used a different drift
definition), SWaT 67 % (prior 69 %).

### Drifting communities (SWaT_canon), slow baseline at the end of the record minus calibration median, in calib-SD

| g | size | cohesion | end level (sd) | channels |
|---|---|---|---|---|
| 18 | 3 | 0.64 | +66.4 | AIT202, AIT203, AIT401 |
| 15 | 3 | 0.74 | +43.7 | AIT201, AIT501, AIT503 |
| 19 | 13 | 0.67 | +20.3 | AIT201, AIT402, FIT401, AIT501-504, FIT501-504, PIT501, PIT503 |
| 14 | 10 | 0.87 | +15.2 | AIT402, FIT401, AIT502, AIT504, FIT501-504, PIT501, PIT503 |
| 8 | 8 | 0.96 | +11.0 | FIT401, AIT504, FIT501-504, PIT501, PIT503 |
| 21 | 14 | 0.61 | +10.3 | AIT201, AIT402, FIT401, AIT501-504, FIT501-504, PIT501-503 |
| 22 | 17 | 0.50 | +8.5 | AIT201-203, AIT401-402, FIT401, AIT501-504, FIT501-504, PIT501-503 |

Every drifting community is built from the P2/P4/P5 analysers (AIT2xx/4xx/5xx) and the RO train (FIT4xx/5xx, PIT5xx);
the next community down (g3, FIT503/PIT501/PIT503) sits at +2.5 sd and is not drifting; the mechanical communities
(pumps, valves, levels, UF: g11, g16, g17, g20, g23) stay within +1.1 sd. This matches the stated physical signature
(RO fouling + analyser electrode drift) channel for channel. WADI's top mover (g19, the 2A/2B AIT_003/004 analysers)
reaches only +1.1 sd; HAI's top movers are the g1/g11 step, not a ramp.

## 2. Method (fixed before any AUROC was read; one documented revision)

Per community g and seed, surprise s_g(t) (test_surprise) is decomposed causally:
- slow_g(t) = median of s_g over the previous K windows, K = 24 h of wall-clock (SWaT/WADI 288 windows of 300 s, HAI
  2880 windows of 30 s), initialised at the calibration-slice median, min 6 h of history; sensitivity 6/12/48 h and a
  6 h-lagged 24 h window are reported, not selected.
- fast_g(t) = (s_g(t) - slow_g(t)) / max(rolling MAD over the same K windows, calibration MAD).
- The calibration slice (held-out last 20 % of train windows) is decomposed identically and is the p-value reference
  for the fast residuals, exactly as it is for the raw surprises in the headline. The global LatAD term is decomposed
  the same way against LatAD_train. The changepoint detector CP = z(HC_coh over residual p-values) + z(residual LatAD
  tail) is `ensemble_final.ensemble_scores["HCcoh+LatAD"]` applied to the fast component; K=None reproduces the headline
  bit-for-bit (I1, asserted on all three datasets).

**Revision v1 -> v2 (root-cause driven, reported plainly).** v1 used the level only (fast = s - slow). It lifted SWaT
Difficult to 0.625 but masked 7/25 episodes by global rank. Inspection of the residuals showed why: in the analyser
communities the test-normal residual MAD in thirds 2-3 is 8-41x the calibration MAD (99th pct of late-normal residuals
29-33 vs 3-4 in calibration), so the drift swamped the fusion through the NOISE level once its mean level was removed.
The on-trend component therefore has to carry a slow scale as well as a slow level; v2 divides by the causal rolling
MAD over the same window, floored at the calibration MAD so no community is ever more sensitive than at train level.
No timescale or threshold was changed; v1 numbers are kept in every table.

## 3. AUROC (5-seed mean); subsets All / Difficult / DoubleHard_lin / DoubleHard_pca

Subset sizes: SWaT_canon 233 / 91 / 31 / 28 (25 episodes); HAI 652 / 167 / 84 / 55 (38); WADI_clean 56 / 30 / 19 / 29 (11).

| method | SWaT_canon | HAI | WADI_clean |
|---|---|---|---|
| trivial max\|z\| | 0.853 / 0.627 / 0.582 / 0.514 | 0.806 / 0.340 / 0.349 / 0.336 | 0.786 / 0.601 / 0.493 / 0.591 |
| IF | 0.800 / 0.550 / 0.387 / 0.326 | 0.844 / 0.627 / 0.635 / 0.435 | 0.725 / 0.634 / 0.530 / 0.622 |
| AE | 0.786 / 0.517 / 0.210 / 0.154 | 0.923 / 0.757 / 0.730 / 0.418 | 0.792 / 0.628 / 0.535 / 0.616 |
| linres | 0.776 / 0.464 / 0.156 / 0.153 | 0.779 / 0.586 / 0.465 / 0.547 | 0.834 / 0.750 / 0.606 / 0.742 |
| USAD | 0.763 / 0.477 / 0.152 / 0.125 | 0.849 / 0.497 / 0.471 / 0.368 | 0.757 / 0.579 / 0.466 / 0.565 |
| TranAD | 0.761 / 0.477 / 0.150 / 0.119 | 0.834 / 0.445 / 0.418 / 0.297 | 0.786 / 0.613 / 0.489 / 0.601 |
| GDN (official, fresh) | 0.761 / 0.473 / 0.148 / 0.115 | n/a | n/a |
| LatAD (global) | 0.752 / 0.472 / 0.219 / 0.184 | 0.933 / 0.811 / 0.806 / 0.559 | 0.717 / 0.634 / 0.571 / 0.623 |
| HC_coh | 0.787 / 0.523 / 0.190 / 0.181 | 0.932 / 0.801 / 0.737 / 0.625 | 0.843 / 0.795 / 0.696 / 0.788 |
| null+HC | 0.794 / 0.515 / 0.193 / 0.181 | 0.913 / 0.770 / 0.704 / 0.627 | 0.819 / 0.749 / 0.633 / 0.740 |
| **HCcoh+LatAD (headline, drift-swamped on SWaT)** | 0.788 / 0.524 / 0.193 / 0.178 | 0.948 / 0.845 / 0.814 / 0.648 | 0.827 / 0.771 / 0.662 / 0.763 |
| CP v1 (24 h, level only) | 0.836 / 0.625 / 0.503 / 0.456 | 0.944 / 0.836 / 0.805 / 0.615 | 0.829 / 0.776 / 0.671 / 0.768 |
| **CP v2 (24 h, level+scale), PRIMARY** | **0.880 / 0.723 / 0.719 / 0.671** | 0.943 / 0.835 / 0.800 / 0.636 | 0.830 / 0.779 / 0.676 / 0.771 |
| DR (drift score, not a detector) | 0.563 / 0.387 / 0.079 / 0.088 | 0.339 / 0.455 / 0.403 / 0.539 | 0.586 / 0.506 / 0.427 / 0.506 |

The DR row is the slow component fused; it is below chance on HAI All by construction (attacks are short and do not
raise a 24 h median, drifted normals do), and is listed only to show that the drift and changepoint components
separate the two populations in opposite directions. It is the only sub-0.5 entry (I3).

### Bootstrap (episode-block, 2000 reps), CP v2 minus comparator

| dataset, subset | vs headline: diff [95 % CI], P(<=0) | strongest alternative | vs strongest: diff [CI], P |
|---|---|---|---|
| SWaT All | +0.092 [0.019, 0.301], 0.006 | trivial max\|z\| 0.853 | +0.027 [-0.018, 0.126], 0.12 |
| SWaT Difficult | +0.200 [0.018, 0.355], 0.014 | trivial 0.627 | +0.096 [-0.004, 0.209], 0.031 |
| SWaT DoubleHard_lin | +0.526 [0.377, 0.640], <0.001 | trivial 0.582 | +0.137 [-0.014, 0.323], 0.044 |
| SWaT DoubleHard_pca | +0.493 [0.349, 0.619], <0.001 | trivial 0.514 | +0.157 [-0.013, 0.329], 0.036 |
| HAI Difficult | -0.010 [-0.026, 0.005], 0.88 | headline 0.845 | same |
| HAI DoubleHard_pca | -0.012 [-0.049, 0.025], 0.65 | headline 0.648 | same |
| WADI Difficult | +0.008 [-0.014, 0.034], 0.27 | HC_coh 0.795 | -0.016 [-0.048, 0.018], 0.84 |
| WADI DoubleHard_pca | +0.008 [-0.014, 0.034], 0.27 | HC_coh 0.788 | -0.017 [-0.049, 0.019], 0.84 |

### Timescale sensitivity (CP v2; Difficult / DoubleHard_pca), a priori = 24 h

| K | SWaT_canon | HAI | WADI_clean |
|---|---|---|---|
| 6 h | 0.731 / 0.666 (All 0.631: the 10 h attack is absorbed) | 0.868 / 0.663 | 0.707 / 0.697 |
| 12 h | 0.757 / 0.727 (All 0.777) | 0.856 / 0.671 | 0.754 / 0.745 |
| **24 h** | **0.723 / 0.671 (All 0.880)** | 0.835 / 0.636 | 0.779 / 0.771 |
| 48 h | 0.706 / 0.593 | 0.823 / 0.630 | 0.775 / 0.767 |
| 24 h, 6 h lag | 0.689 / 0.616 | 0.810 / 0.621 | 0.775 / 0.768 |
| 24 h, global term raw | 0.689 / 0.378 | 0.857 / 0.677 | 0.774 / 0.766 |

The result is not knife-edge: every timescale from 6 h to 48 h recovers SWaT to 0.70-0.76 Difficult and 0.59-0.73
double-hard, and none moves HAI or WADI by more than the bootstrap CI except HAI at 6 h (upward). Shorter windows
trade All for Difficult on SWaT because the 10 h attack #758 is longer than half the window and is absorbed into the
baseline (the masking mechanism the a-priori choice avoids). Detrending the global LatAD term costs HAI ~0.02 and is
kept for consistency (the headline treats the community and global terms identically).

## 4. Typing validation (seed-mean scores; thresholds = calibration 99th pct; attack labels used only to evaluate)

| dataset | attacks flagged: headline / CP | head-flagged attacks typed changepoint (all / difficult) | normal false alarms: headline / CP | drifted normals: fraction; of those head-flagged, typed drift | non-drifted head-flagged normals typed drift |
|---|---|---|---|---|---|
| SWaT_canon | 0.918 / 0.687 | 0.748 / 0.417 | **0.792 / 0.065** | 0.666; 0.974 (n=843) | 0.635 (n=159; spread over the record, 9 % in the first 6 h: headline flags that neither component reproduces, i.e. the sub-threshold class) |
| HAI | 0.906 / 0.802 | 0.885 / 0.723 | 0.105 / 0.072 | 0.297; 0.925 (n=415) | 0.078 (n=1068) |
| WADI_clean | 0.375 / 0.232 | 0.619 / 0.750 | 0.015 / 0.012 | 0.000; n/a | 0.25 (n=8) |

By record third (SWaT): headline false alarms 0.40 / 1.00 / 1.00, CP 0.14 / 0.045 / 0.009; attacks flagged by CP
0.63 / 0.80 / 0.30. The last-third attack flag rate at a FIXED train-calibrated threshold is low (10 of 33 windows)
even though these attacks rank at 0.94-0.98 against their contemporaneous normals: the rolling scale deflates the
whole late era, normals and attacks alike, so a fixed absolute threshold under-fires there. Ranking (AUROC) is
recovered; a deployable alarm threshold in a heavily drifted era needs a rolling calibration too, which this study
does not add.

Difficult SWaT attacks type as changepoint less often (0.417) than all attacks (0.748): 55.6 % of the head-flagged
difficult attacks are DRIFT-typed, i.e. under the headline they were "flagged" only by riding on the drifted era's
elevated baseline and carry little off-trend residual themselves. Their AUROC still improves from 0.524 to 0.723
because the CP score removes the far larger drift contribution from the normals.

## 5. Invariants (all stated before the numbers)

| | SWaT_canon | HAI | WADI_clean |
|---|---|---|---|
| I1 K=None == headline exactly | OK (max abs diff 0) | OK | OK |
| I2 HAI/WADI Difficult within +-0.02 of headline | n/a | OK (-0.010) | OK (+0.008) |
| I3 nothing below 0.5 on All | OK | only DR (drift score, by construction) | OK |
| I4 ramp invariance (1 and 2 sd/day) | MARGINAL: CP Diff 0.723 -> 0.699 / 0.684 (headline 0.524 -> 0.494 / 0.488, All 0.788 -> 0.680) | OK: CP 0.835 -> 0.822 / 0.824 (headline 0.845 -> 0.709 / 0.564) | VIOLATED: CP 0.779 -> 0.724 / 0.682 (headline 0.771 -> 0.677 / 0.616) |
| I5 no episode masked, era-local (+-24 h) | VIOLATED: 2/25 (ep@660 -0.11, ep@1158 -0.17), 5 gained >0.1 | OK 0/38 (v1: 1/38) | OK 0/11 |
| I5 global rank, for reference | 6 lost >0.1, 11 gained >0.1; det@95 all 0.639 -> 0.661, difficult 0.253 -> 0.275 | 1 lost (ep@3059, already at 0.36) | 0 |

I4 reading: the causal baseline removes most of an injected ramp's damage (SWaT All: headline -0.108 at 2 sd/day, CP
-0.023; HAI Difficult: headline -0.281, CP -0.011) but a one-sided median lags a ramp by K/2 = 12 h and leaves a
residual bias; on WADI's 2-day record (2x the window) with 30 difficult windows in 8 episodes that bias costs 0.05.
The tolerance was set at 0.02 before the run and the SWaT/WADI results are reported as misses.

I5 reading (why global rank is the wrong primary): under the drift-swamped headline every late window, normal or
attack, outranks every early one. Early SWaT attacks were locally at 0.87-0.95 percentile yet globally at 0.23-0.25;
late attacks were globally at 0.97 only by time position. The CP score restores a globally comparable score (early
attacks 0.95-0.98 globally). The two era-local losses are real: ep@660 (30 min, all 6 windows difficult, DPIT301/
FIT301/LIT301 and LIT101/P101/FIT201 communities) and ep@1158 (15 min), which lies within 24 h after the 10 h attack
#758 whose residuals inflate the rolling MAD of the mechanical communities (scale / calib-MAD 2.5). A long attack
shadows the following window; this is the price of the scale term (v1 loses the same episode more, 0.949 -> 0.694,
by the level term's median shift). The per-episode table for all 25 SWaT episodes is in the log and JSON.

## 6. Honest verdict

Drift-vs-changepoint typing does what was asked on the axis that matters: on the clean SWaT_canon the changepoint
score recovers the community fusion from chance to 0.723 Difficult / 0.671 double-hard, significant against the
drift-swamped baseline and marginally ahead of the strongest clean alternative (the trivial rule; the deep baselines
and the fresh GDN are far below), while cutting false alarms from 79 % to 6.5 %; HAI and WADI are unchanged within
their CIs; attacks type as changepoint and drifted normals as drift at 75-97 %. It does not meet two of its own
invariants cleanly: two short SWaT episodes lose era-local rank (one in the shadow of the 10 h attack), and the
synthetic-ramp invariance is marginal on SWaT and fails on the short WADI record. The composition result is the
cleaner headline: the three datasets sit at 0 % / 30 % / 67 % drifted normals with 0 / 0 / 7 drifting communities,
and SWaT's seven are exactly the analyser + RO communities.

Not done here: a rolling (era-local) alarm threshold for deployment; a fix to `rev4_doublehard_pca_all3.py`'s stale
SWaT GDN file; any retraining.
