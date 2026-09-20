# SWaT_canon drift-cleaning proof case (literature-documented drift normals removed from the test partition)

Script `drift_clean_swat.py`, results `drift_clean_swat.json`, log `drift_clean_swat.log`. All scores are the
existing per-window artifacts (`scores_SWaT_canon.npz`, `boosted_loo_SWaT_canon.npz`, `experts_full`), so every
method is re-scored on the identical cleaned window set; nothing is retrained. BOOT_REPS=2000, episode-block
bootstrap identical to `ensemble_final.boot`. Difficult subset = (label==1) & (maxz<=maxz_thr), n=85 / 23 episodes.

## Verdict first

- **Conservative (literature-only) cleaning does NOT bring the community headline to parity.** Removing the 919
  literature-drift normals: HCcoh+LatAD 0.837 -> 0.865, boosted LOO 0.881 -> 0.874. Paired diff -0.009,
  95% CI [-0.065, 0.048], one-sided P(<=0)=0.66. The gap shrinks from -0.044 to -0.009 but stays negative.
- **Broad EDA cleaning flips the sign** (HCcoh+LatAD 0.862 vs boosted 0.792, diff +0.070, CI [0.007, 0.141],
  P=0.016), but it removes 1136 of 1265 normals (90%), leaves 129 normal windows all from the first third of
  the recording, and rests on a train reference that this data cannot support (see the provenance finding).
  It is reported, not endorsed.
- **A data-provenance finding overrides the drift premise and must reach the paper before any SWaT number is
  reused**: the training set used for every SWaT_canon model contains the canonical test-normal rows verbatim
  (367,557 of 395,298; 93%), and seven channels including AIT201 are not recorded at all in the mirror's
  Normal_v0 segment (NaN for all 496,800 rows, forward-filled to a constant by the loader). Details below.

## Provenance finding (I3, verified row-for-row)

The Kaggle mirror `normal.csv` (1,387,098 rows) is not the Normal recording. Segmenting it by timestamp:

| rows | n | span | what it is |
|---|---|---|---|
| 0 - 395,298 | 395,298 | 28/12 10:00 - 02/01 14:59, with gaps exactly at the attack intervals | the canonical attack file's **normal** rows (row-wise equality with `swat_attack_canonical.npz[ya==0]`: 1.000) |
| 395,298 - 892,098 | 496,800 | 22/12 16:00 - 28/12 09:59 | SWaT_Dataset_Normal_v0 |
| 892,098 - 1,387,098 | 495,000 | 22/12 16:30 - 28/12 09:59 | Normal_v1 = v0 shifted by 1800 s (row-wise equality 1.000): a duplicate |

`eda_real._raw_swat_canonical` trains on all of `normal.csv` (2% warm-up dropped, downsample 10), so the SWaT_canon
TRAIN set holds 367,557 of the 395,298 canonical test-normal rows, plus Normal_v0 twice. In the Normal_v0/v1
segments the mirror has NaN for the whole length of MV101, AIT201, MV201, P201, P202, P204, MV303; the loader's
ffill sets them to the attack file's last value (AIT201 = 168.1 for all 7 days). Consequences:

1. The premise "drift normals are under-represented normal operation" is false for this pipeline: the test-normal
   rows are in train verbatim. Whatever the models do on them is not a coverage gap.
2. Every SWaT_canon number in the paper (all methods) is computed with test normals seen in training; the leak is
   uniform across methods but inflates absolute AUROC and undermines the drift-robustness narrative for SWaT.
3. The non-canonical `SWaT` split (test = last 20% of `normal.csv` = tail of Normal_v1) is leaked too, since v1
   duplicates v0 which is in train.
4. A clean SWaT_canon requires retraining every expert/baseline on Normal_v0 only (44 recorded channels, or the
   official xlsx for all 51). This is outside the present task; it is the required next step.

## Step 1: literature drift regions (DAICS)

- **P102**: proper train (Normal_v0) has the single state {1}; test gains state 2. In the test partition state 2
  occurs in 313 downsampled rows, of which 304 are attack-labelled (the P102 attacks themselves) and 9 normal.
  Windows containing state 2: 18, of which 17 attack, **1 normal**. The P102 rule is inert as a drift-normal cleaner.
- **AIT201**: literature train range [251, 272]; test-normal spans [168.0, 266.1] and declines monotonically over
  the four days (window medians 262 -> 169). Windows whose AIT201 window-mean lies outside [251, 272]: 1100, of
  which **919 normal** (of 1265) and 181 attack (kept). The mirror's own Normal_v0 cannot supply an AIT201
  train range (unrecorded); the leaked train's range [168, 263] would flag 0 normals.
- Literature drift-normal set = 919 windows (73% of test normals). All of them sit in thirds 2-3 of the recording
  plus 94 in the first third; the 346 kept normals are all first-third (day 1). Difficult attacks are spread
  43 / 18 / 24 across thirds, and 47 of the 85 have AIT201 outside [251, 272]. The cleaned comparison therefore
  pits day-1 normals against attacks from all five days.

## Step 2: EDA drift labeler (row-level KS, train vs test-normal), a-priori rule KS > 0.5

Proper train (Normal_v0, the 44 recorded channels), top 10 by KS:
AIT402 0.956 (test-normal mass outside train support 0.84), AIT502 0.794 (0.49), AIT501 0.691 (0.38),
AIT202 0.672, AIT504 0.667 (0.59), PIT503 0.554, FIT301 0.521, FIT503 0.508, PIT502 0.505, PIT501 0.487.
**P102 ranks 33** (constant in train, so KS is degenerate; its state gain is 9 rows out of 39,524 test-normal rows)
and **AIT201 cannot be ranked** (unrecorded in the mirror's Normal_v0). The validation the task asked for
(P102 and AIT201 among the top-shifted channels) is therefore not met on this data, for a data reason, not a
method reason. Against the leaked train, AIT201 ranks 1 (KS 0.729) only because the leaked train mixes the
declining attack-file values with 7 days of the ffilled constant 168.1; that ranking is an artifact.

The EDA-flagged channels (KS > 0.5) are AIT402, AIT502, AIT501, AIT202, AIT504, PIT503, FIT301, FIT503, PIT502.
A window is EDA-drift if any flagged channel's window-mean falls outside that channel's Normal_v0 [min, max].
Normal windows flagged per channel: AIT402 1075, AIT504 767, AIT502 617, AIT501 485, PIT503 46, PIT502 27,
FIT503 2, AIT202 0, FIT301 0. Union with the literature set: **1136 normals removed (90%), 129 kept, all
first-third**. This removes far more than the literature names, and the Normal_v0 reference it uses is the
mirror's (narrow analyzer ranges, e.g. AIT402 train [154, 208] vs test-normal [141, 328]).

## Steps 3-5: cleaned vs uncleaned Difficult AUROC (attacks intact: 233 windows, 85 difficult, 0 removed)

| method | uncleaned | literature-clean (919 removed) | delta | EDA-broad (1136 removed) | delta | removed-normal score pct (lit) |
|---|---|---|---|---|---|---|
| trivial max\|z\| (floor) | 0.646 | 0.475 | -0.171 | 0.506 | -0.140 | 0.445 |
| linres | 0.782 | 0.799 | +0.017 | 0.782 | 0.000 | 0.496 |
| IF | 0.627 | 0.659 | +0.032 | 0.690 | +0.063 | 0.516 |
| AE | 0.729 | 0.749 | +0.020 | 0.780 | +0.051 | 0.507 |
| USAD | 0.658 | 0.644 | -0.014 | 0.712 | +0.054 | 0.489 |
| TranAD | 0.655 | 0.660 | +0.005 | 0.724 | +0.069 | 0.499 |
| LatAD (global) | 0.804 | 0.820 | +0.016 | 0.829 | +0.025 | 0.517 |
| null+HC | 0.811 | 0.838 | +0.027 | 0.843 | +0.032 | 0.739 |
| **HCcoh+LatAD (community headline)** | **0.837** | **0.865** | +0.028 | **0.862** | +0.025 | 0.526 |
| **boosted LOO** | **0.881** | **0.874** | -0.007 | **0.792** | -0.089 | 0.521 |

Last column (I5): mean percentile of the removed normals' scores among all test normals. For HCcoh+LatAD it is
0.526 and for boosted 0.521: the literature-drift normals were NOT high-scoring false positives for either
method. The +0.028 for the headline comes from composition (day-1 normals only), not from removing errors.
null+HC is the one method for which the removed normals were genuinely high-scoring (0.739). The trivial floor
drops below 0.5 because the retained day-1 normals include start-up windows with larger max|z| than the
difficult attacks (which have maxz <= thr by definition).

Episode-block bootstrap, HCcoh+LatAD minus boosted LOO on Difficult (23 episodes, 2000 reps):

| test set | diff | 95% CI | one-sided P(<=0) |
|---|---|---|---|
| uncleaned | -0.044 | [-0.096, 0.017] | 0.926 |
| literature-clean | -0.009 | [-0.065, 0.048] | 0.662 |
| EDA-broad | +0.070 | [0.007, 0.141] | 0.016 |

Sub-variants: P102-only removes 1 normal and changes nothing (diff -0.043); AIT201-only is identical to the
literature variant (the AIT201 rule does all the work).

## Invariants

- I1 uncleaned Difficult AUROCs reproduce `loo_fusion_boosted.json` (0.837 / 0.881 / 0.782 / 0.804): pass.
- I2 zero attack windows removed in every variant; n_diff = 85 throughout: pass (asserted in code).
- I3 `normal.csv` head == canonical test-normal rows, row-wise 1.000; Normal_v1 == Normal_v0 shifted 1800 s,
  row-wise 1.000: pass, which establishes the leak.
- I4 EDA validation: predicted P102/AIT201 in the proper-train top-10: **fails** (P102 rank 33, AIT201
  unrecorded). Predicted no AIT201 range shift against the leaked train: holds (out-of-support 0.026, range
  [168, 263] covers the test range) even though KS ranks it first for the artifact reason above.
- I5 removed-normal score percentiles reported per method (table).

## Guardrails recap

Only label==0 windows removed; 233 attack windows and the 85-window difficult subset bit-identical before and
after; drift regions defined by DAICS (P102 state, AIT201 [251,272]) and by an a-priori KS>0.5 / out-of-train-
support rule, never by any method's errors; cleaned and uncleaned reported for all ten methods.

## Bottom line for the paper

The SWaT difficult-subset gap to the boosted LOO cannot be attributed to literature-documented drift normals:
after removing them the headline still trails (-0.009, P=0.66). More important, the SWaT_canon train/test
construction is leaked (test normals inside train, seven channels unrecorded in the mirror's Normal_v0), so the
SWaT drift-robustness claims and all SWaT_canon numbers need a rebuild on Normal_v0-only training before any
of them, cleaned or not, can be used.
