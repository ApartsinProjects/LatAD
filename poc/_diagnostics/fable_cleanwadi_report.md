# Cleaned-WADI difficult subset: can LatAD legitimately win, tie, or does it lose?

Scripts (all under `poc/_diagnostics/`): `fable_cleanwadi_rebuild.py` (one-pass co-computed score table),
`fable_cleanwadi_analyze.py` (error analysis), `fable_cleanwadi_sweep.py` (H5 sweep with train-only selection),
`fable_cleanwadi_e2_support.py` (E2 support partition). Artifacts: `fable_cleanwadi_scores_clip10_default.npz` (the
construct-matched table, K=20/LD=10), `fable_cleanwadi_scores_clip10_{K30LD16,K30LD8,dropAIT004}.npz`,
`fable_cleanwadi_analysis.json`, `fable_cleanwadi_windows.json`, `fable_cleanwadi_sweep.jsonl` + `_summary.json`,
`fable_cleanwadi_k30_boot.json`, `fable_cleanwadi_e2_support.json`, logs `fable_cleanwadi_*.log`.
Nothing in the paper or the exported checkpoints was touched; `eda_real.py` was not edited (the fix is one line, stated below).

## 1. Verdict

**Statistical TIE with the autoencoder on the canonical difficult subset. LatAD does not win; both trail the linear-residual
filter. The train-only criteria do not select the residual head, and the config they do select (K=30) stays inside the CI.**

The cited "AE 0.779 vs LatAD 0.731 on 30 windows" came from a mis-constructed subset AND an unclipped loader (section 2).
Corrected, co-computed in one pass on identical windows, seeds and split (`fable_cleanwadi_scores_clip10_default.npz`):

| method | difficult: 43 anomalies vs 519 normals (5 seeds) | easy (13) | all (56) |
|---|---|---|---|
| linres (leave-one-channel-out linear residual) | 0.787 | 0.993 | 0.834 |
| l2 | 0.751 | 0.999 | 0.809 |
| **AE** | **0.739 +- 0.003** | 0.999 | 0.799 |
| **LatAD (K=20, LD=10, reported config)** | **0.734 +- 0.008** | 0.991 | 0.793 |
| IF | 0.681 +- 0.006 | 0.871 | 0.725 |

Paired bootstrap over difficult-subset windows (3000 resamples, seed-mean AUROC):
LatAD - AE = **-0.005, 95% CI [-0.033, +0.026]**, P(<=0) = 0.64. LatAD - IF = +0.052 [-0.011, +0.125].
LatAD - linres = -0.053 [-0.142, +0.038]. LatAD - l2 = -0.018 [-0.046, +0.012].

Same construction on the paper's dirty WADI (`scores_WADI.npz`, 19 difficult anomalies): LatAD 0.690 +- 0.027, AE 0.425,
IF 0.677, linres 0.392, l2 0.445, TranAD 0.333, USAD 0.303; LatAD - AE = +0.266 [+0.178, +0.351]. On dirty data 288/519 test
normals carry the saturated `2B_AIT_002_PV` offset (+10 sigma after clipping); reconstruction and linear-residual methods flag those
normals and fall below chance on the difficult subset, LatAD's latent density does not. That is what the dirty-WADI margin measures.
Once the artifact channel is removed the asymmetry disappears and the remaining difficult anomalies are of a kind (section 5) that a
linear residual catches best.

## 2. Bugs found (H1), with corrected numbers

**Bug A: the cited difficult subset is the inverse of the paper's definition.** The number quoted to me used
`mask = maxz > maxz_thr` over ALL windows ("30 windows, 26 anomalous"). The paper (section 5.3) and `improve_multiseed.py:43` define
`easy = (y==1) & (maxz > thr)`, `difficult = (y==1) & ~easy`, each scored against ALL test normals. The quoted subset is the 26 EASY
anomalies plus the 4 normals above threshold, a 26-vs-4 AUROC. On the canonical subset built from the very same (unclipped) table,
LatAD and AE both score 0.620 (tie). Every conclusion drawn from the 0.731/0.779 pair is void.

**Bug B: `E.load("WADI_clean")` runs unclipped.** `eda_real.CLIP` has `"WADI": 10.0` but no `"WADI_clean"` key, so
`clip = CLIP.get(name)` is `None`. The `_raw_wadi_clean` docstring says "everything else matches `_raw_wadi`", and the Modal SOTA
export (`sota_bundle/prep_sota_general.py`) DID clip the clean data at 10 (`wadi_clean_test.npy` max |z| = 10.0), so the local
LatAD/AE/IF-clean scores and the Modal USAD/TranAD-clean scores were computed on different inputs. On the unclipped table: 38 channels
exceed |z| = 10 in test; five constant-in-train STATUS/CO channels (`2_MCV_007_CO`, `1_P_006_STATUS`, `1_MV_002_STATUS`,
`1_MV_003_STATUS`, `2_PIC_003_SP`; train std 1e-8) reach |z| = 1e8 to 1e10 when they flip in the attack recording; AE scores reach 3.7e36
and LatAD 7.8e31, so the affected rankings are set by overflow, not by the model (AE's exactly-zero seed variance was the symptom).
The difficulty stratum also changes: a STATUS flip in a few rows of a 60-row window yields a window-mean of ~1e8 unclipped but
10*k/60 clipped, so the canonical split is 30/26 unclipped and **43 difficult / 13 easy** with the clip. The 43 is the correct stratum (it
is what the clipped SOTA bundle sees; the 19 dirty-difficult anomalies are a strict subset of it).
**Fix**: add `"WADI_clean": 10.0` to `eda_real.CLIP` (one line; my rebuild passes `clip=10.0` explicitly) and regenerate
`scores_WADI_clean.npz`; `fable_cleanwadi_scores_clip10_default.npz` is that regeneration.

Invariants checked: (i) clipped test features max at 10.0, matching the SOTA bundle; (ii) with both auto gates off, `full(auto)` equals
`dens+near` exactly (`allclose` True); (iii) all 19 dirty-difficult windows are inside the 43 clean-difficult windows; (iv) every raw-feature
method saturates on the easy subset (0.99-1.0), as the split definition requires; (v) the 80/20 time-ordered held-out slice of train
is 2.7% out-of-support, so the OOS threshold is not drift-inflated.

## 3. Train-only config and head selection (H4, H5): does any legitimate choice turn the tie into a win?

**Residual head.** The auto gate (`resid_gen_ratio < 1.5`, held-out-normal q99 / fit q99 of the residual score) is OFF on every one of
the 45 sweep runs and all 5 seeds of every rebuilt table (per-config mean ratios 1.8 to 3.3; minimum single run 1.44 but no config
averages below 1.5; K=30 tables: 4.2 to 7.6). Forcing it on would give 0.760 (K=20) to 0.784 (K=30/LD=10) on the difficult subset, ahead
of AE, but the train-only criterion says the residual does not generalise on clean WADI. Adopting it would be a test-label choice.
**Not adopted; the head stays off.**

**K x latent_dim sweep** (K in {12,20,30} x LD in {8,10,16}, 5 seeds, `fable_cleanwadi_sweep.jsonl`). For each run a model is fit on
the first 80% of train windows and scored on the last 20%; test difficult-AUROC is computed from a separate full-train model and never used
for selection. Per-config means:

| K | LD | held-out density NLL | NLL / LD | held-out minus fit gap | q99 tail ratio | resid ratio | test difficult (5-seed) |
|---|---|---|---|---|---|---|---|
| 12 | 8 | 12.79 | 1.598 | 1.95 | 1.11 | 1.94 | 0.745 +- 0.011 |
| 12 | 10 | 15.26 | 1.526 | 2.45 | 1.15 | 1.81 | 0.740 +- 0.013 |
| 12 | 16 | 20.78 | 1.299 | 4.31 | 1.08 | 2.06 | 0.730 +- 0.006 |
| 20 | 8 | 12.61 | 1.577 | 2.01 | 1.27 | 2.35 | 0.747 +- 0.011 |
| 20 | 10 (reported) | 14.92 | 1.492 | 2.54 | 1.10 | 2.44 | 0.734 +- 0.008 |
| 20 | 16 | 20.17 | 1.261 | 4.21 | 1.18 | 2.44 | 0.736 +- 0.006 |
| 30 | 8 | 12.46 | 1.558 | 2.08 | 1.16 | 3.29 | 0.751 +- 0.006 |
| 30 | 10 | 14.45 | 1.445 | 2.47 | 1.18 | 2.77 | 0.748 +- 0.019 |
| 30 | 16 | 19.61 | 1.226 | 4.42 | 1.08 | 2.88 | 0.747 +- 0.011 |

Train-only rules, applied blind: lowest held-out NLL per latent dim -> K=30/LD=16; q99 tail ratio closest to 1 -> K=30/LD=16; lowest raw
held-out NLL -> K=30/LD=8 (this rule is dimension-confounded and always prefers the smallest LD; listed for completeness); smallest
held-out-minus-fit gap -> K=12/LD=8. Within every LD, held-out NLL picks K=30. The whole 9-config range of test difficult-AUROC is 0.730 to
0.751 (AE 0.739); Spearman(held-out NLL, test difficult) over 45 runs = -0.45, so the train signal is informative but the effect it can buy is
about 0.015.

Rebuilt 5-seed tables for the selected configs, bootstrapped against the same AE/IF/linres/l2 columns (identical windows and seeds):

| LatAD config | difficult | vs AE | vs IF | vs linres | vs l2 |
|---|---|---|---|---|---|
| K=20/LD=10 (reported) | 0.734 +- 0.008 | -0.005 [-0.033, +0.026] | +0.052 [-0.011, +0.125] | -0.053 [-0.142, +0.038] | -0.018 [-0.046, +0.012] |
| **K=30/LD=16 (train-selected)** | 0.747 +- 0.011 | **+0.008 [-0.021, +0.040]** | +0.066 [-0.002, +0.140] | -0.041 [-0.127, +0.052] | -0.004 [-0.038, +0.031] |
| K=30/LD=8 (raw-NLL rule) | 0.751 +- 0.006 | +0.013 [-0.017, +0.046] | +0.070 [+0.004, +0.143] | -0.034 [-0.126, +0.057] | -0.001 [-0.034, +0.032] |

**Outcome, stated plainly: the train criterion keeps the residual head OFF and picks K=30; that config edges AE by +0.008 to +0.013 with a
CI that includes 0. The tie stands.** Two further cautions: (a) a per-dataset held-out-NLL K rule must then also govern HAI and SWaT
(currently K=40 there, not swept), so K=30 cannot be reported for WADI alone without running that rule everywhere; (b) at K=30 the residual
gate ratio rises to 4-8, further from ever switching on.

## 4. Coverage, duplicates, mislabels (coordinator's data-quality hypothesis)

Out-of-support (OOS) = 1-NN distance to train-normal windows above the train leave-one-out 99th percentile (30.7), the `dq_audit.py`
definition, recomputed on the clean clip=10 loader.

- **Test normals OOS: 5 / 519 (1.0%)**, in 4 runs of length 1-2. The 19% OOS rate in `dq_audit.json` was the dirty loader and is the
  `2B_AIT_002_PV` offset. LatAD ranks these 5 at mean percentile 0.991, AE at 0.994: both flag them. Of LatAD's 6 false alarms at 1% FPR,
  4 are OOS (AE: 5 of 6); at 5% FPR, 5 of 26 for both methods. The other 21 LatAD false alarms at 5% FPR are in-support windows dominated
  by level-switch (`2_LS_*_AL/AH`, |z| 3-9) and `2_MCV_201/501_CO` actuator excursions, 15 of them within 1-13 windows of an attack.
- **Exclusion test (0 anomalies removed)**: without the 5 OOS normals, LatAD 0.738 +- 0.008, AE 0.743 +- 0.003, IF 0.686, linres 0.788,
  l2 0.756; LatAD - AE = -0.005 [-0.032, +0.026]. Control (drop 5 random in-support normals): -0.005 +- 0.000. **Coverage does not explain
  the tie; the HAI-style reframing does not transfer to clean WADI.**
- **Difficult anomalies OOS: 14 / 43**, every one a STATUS/CO flip window (`2_MCV_007_CO`, `1_MV_002/003_STATUS`, `1_P_006_STATUS`,
  `2_PIC_003_SP`; |z| 180-1070 in feature space) that every method ranks at percentile 0.998-1.000. They contribute nothing to the LatAD-AE
  difference; that difference lives entirely in the 29 in-support difficult anomalies.
- **Duplicates**: the 43 difficult windows form 14 contiguous episodes (lengths 6,4,3,4,1,1,1,1,3,2,5,4,4,4) in 13 of the 14 attack segments.
  Single-linkage at the train-LOO p10 / p50 radius gives 42 / 40 clusters; minimum pairwise distance 3.96; median within-difficult 1-NN 20.7
  vs median to-train 20.1. No exact or near duplicates. The overlap-driven effective N is 14 episodes, which is why the CI is +-0.03.
- **Mislabel candidates**: 6 anomaly windows (all difficult) lie within the train-LOO median distance (10.4) of some normal: w16 (48% attack
  rows), w18, w195 (7% attack rows, boundary), w360-362 (segment 7). Segment 7 (rows 10846-10866, 200 s) shifts no channel by more than 0.37
  sigma against its +-300-row context; all methods rank its 3 windows at percentile 0.02-0.06. These are boundary or undetectable windows,
  not mislabels one could remove without editing test labels; not removed.

## 5. Mechanism (H6): what AE catches that LatAD misses, and why neither matches linres

Seed-mean percentile against the 519 test normals, in-support difficult windows: 33 of 43 windows differ by < 10 points.
AE ahead by > 10 points (7): w16, w21 (seg 0), w196 (seg 1), w236 (seg 4), w508 (seg 12), w546, w547 (seg 13).
LatAD ahead by > 10 points (3): w20 (seg 0), w195 (seg 1), w496 (seg 9).
Per segment (LatAD / AE): seg 0 0.56/0.55, seg 1 0.77/0.72, seg 2 0.98/0.97, seg 4 0.59/0.61, seg 7 0.03/0.05, seg 9 0.92/0.89,
seg 12 0.65/0.71, seg 13 0.67/0.72. The two alternate segment by segment; no segment is caught by one and missed by the other.

Raw-signal reading of the segments where AE leads (window-mean z of top channels; channel shifts vs +-300-row context):
- **Segment 0 (rows 511-661)**: a P1 pump/valve state change (`1_FIT_001_PV` +2.0 sigma, `1_P_001/003_STATUS`, `1_MV_001_STATUS` +2.0),
  while the window features are dominated by the three AIT_004 analyzers at -2.1 to -3.1 sigma, an offset that 36-60% of test NORMALS also
  carry (section 6). The anomaly's own signature is a +2 sigma single-channel step on top of a global 2-sigma analyzer offset shared with
  normals; both LatAD and AE sit at percentile 0.3-0.7.
- **Segments 12 and 13 (rows 15217-15274, 16360-16423)**: `2_FIT_002_PV` -1.9 / `2_FIT_001_PV` +1.5 flow steps plus actuator STATUS flips,
  riding the deepest AIT_004 offset in the recording (-3.7 to -4.3 sigma). linres ranks them 0.94-1.0 (one channel departs from its linear
  prediction from the rest), AE 0.5-0.8, LatAD 0.4-0.8.
- **Segment 4 (rows 7078-7145)**: P2 flow/pressure setpoint change (`2_FQ_201_PV`, `2_FIC_201_PV` +1.8, `2_PIC_003_CO` +1.4); linres 0.94-1.0.
- **Segment 1 (rows 5906-5965)**: `2_FIC_401_SP` setpoint step (+2.3) and `2_LS_401_AL` level switch; LatAD leads on w195 (0.59 vs 0.20),
  AE on w196.

Reading: the in-support difficult anomalies of clean WADI are **single-channel setpoint or flow steps of 1.5-3 sigma with the rest of
the plant intact**, i.e. linear-relationship breaks that a leave-one-channel-out linear residual isolates (linres 0.787). They are not
rare-regime or between-regime events. LatAD's latent density and AE's reconstruction both blur a one-channel step against the analyzer
offset every test window carries. That is the same mechanism the paper describes for the easy/difficult split of SWaT (linres is the
strongest classical baseline there too), and it is why the WADI difficult lead does not survive the artifact removal.

## 6. AIT_004: a second unstable analyzer?

No. By the evidence bar that justified dropping `2B_AIT_002_PV` (train mean 9.09 / std 0.16 vs attack-recording-normal mean 4503 / std
4028: a 27,000-train-sd mean shift, 24,500x std ratio, non-overlapping ranges), the AIT_004 conductivity analyzers show a slow physical
drift, not a rescale:

| channel | train mean / std | test-normal mean / std | mean shift (train sd) | std ratio | ranges |
|---|---|---|---|---|---|
| 2B_AIT_002_PV (dropped) | 9.09 / 0.16 | 4503 / 4028 | 27359 | 24519 | (8.5, 38.8) vs (8.7, 8128) |
| 1_AIT_004_PV | 493.7 / 19.6 | 454.1 / 18.3 | -2.0 | 0.93 | (0, 527) vs (0, 485) |
| 2A_AIT_004_PV | 486.7 / 5.6 | 476.5 / 5.0 | -1.8 | 0.90 | (470, 502) vs (463, 488) |
| 2B_AIT_004_PV | 491.1 / 6.3 | 478.2 / 4.1 | -2.0 | 0.64 | (437, 571) vs (466, 490) |

The per-day train means of `1_AIT_004_PV` already fall monotonically over the 14 days (503, 509, 510, 502, 493, 490, 487, 482, 470, 464) and
the test blocks continue the same slope (457 to 439). Same units, same scale, overlapping ranges, a trend that starts inside train: this
is process drift that a deployed model must live with, and it is what the `_raw_wadi_clean` docstring already decided to keep.
Exploratory rebuild with the three analyzers dropped (`--drop`, 5 seeds, `fable_cleanwadi_scores_clip10_dropAIT004.npz`): identical
difficult set (43), LatAD 0.732 +- 0.005, AE 0.733 +- 0.001, IF 0.676, linres 0.773, l2 0.748; LatAD - AE = +0.000 [-0.031, +0.037].
Dropping them would not change the verdict and is not justified by the data. Not a fix.

## 7. E2 support-partition measurement (clean WADI, K=20, LD=10, 5 seeds; thresholds from train only)

Per test window: nearest-component NLL n(x) = min_c -log N(z | c); pi of that component; mixture NLL m(x) = -logsumexp(log pi_c + log N_c).
Test normals partitioned as out-of-support (n(x) > train q99), in-support-low-pi (pi_nearest < 0.5/K = 0.025), in-support-high-pi (rest).
Rare-regime FPR = fraction flagged at the train-99th-percentile threshold of each score.

| seed | pi min / median / max | low-pi components | train windows in them | test normals: OOS / low-pi / high-pi | FPR low-pi group: nearest / mixture |
|---|---|---|---|---|---|
| 0 | 0.005 / 0.037 / 0.248 | 4 | 148 | 3 / 29 / 487 | 0.00 / 0.00 |
| 1 | 0.011 / 0.045 / 0.084 | 3 | 138 | 4 / 28 / 487 | 0.00 / 0.00 |
| 2 | 0.022 / 0.039 / 0.160 | 2 | 116 | 4 / 28 / 487 | 0.00 / 0.00 |
| 3 | 0.008 / 0.041 / 0.115 | 4 | 175 | 4 / 21 / 494 | 0.00 / 0.00 |
| 4 | 0.016 / 0.040 / 0.132 | 4 | 217 | 1 / 52 / 466 | 0.00 / 0.00 |

Alternative low-pi definition (bottom quartile of pi): 37-78 test normals per seed, FPR 0.00 / 0.00 as well.

Counts refute the "VaDE collapses rare regimes so the group is empty" hypothesis: 2-4 components per seed carry pi < 0.025 (as low as
0.005), they hold 116-217 train windows (4-8% of train), and 21-52 test normals (4-10%) land in them in-support. But on that group the
nearest and mixture scores produce identical, zero false-alarm rates at the train-99th threshold: the mixture's log-pi penalty (about 2 nats
between pi = 0.005 and pi = 0.04) is small against the spread of the NLL itself, so on clean WADI the nearest-vs-mixture choice has no
measurable rare-regime FPR consequence. For the E2 rebuttal this means: the rare-regime group exists and is populated, the model does not
collapse it, and the nearest-mode rule is a no-cost safeguard here rather than a demonstrated win; any claimed rare-regime FPR advantage
must come from a benchmark where low-pi normals score near the threshold, which WADI-clean is not.

## 8. Honesty statement

Recommended framing: **"On cleaned WADI, LatAD ties the autoencoder on the difficult subset (0.734 vs 0.739 at the reported config;
0.747 vs 0.739 at the train-selected K=30; CI on the difference spans 0 either way), leads Isolation Forest, and trails the leave-one-channel-out
linear residual (0.787), whose target, single-channel setpoint steps, is what the cleaned difficult subset consists of."** This is a tie, not a
win, and it is defensible in a rebuttal: one co-computed table, identical windows and seeds, paired CIs, the two bugs behind the earlier
"AE beats LatAD by 0.05" named and fixed, the only score-lifting change (forcing the residual head) rejected by LatAD's own train-only gate,
and the K=30 sensitivity reported with its CI rather than swapped in. The dirty-WADI difficult lead (0.690 vs 0.425) should be described as
robustness to the rescaled `2B_AIT_002_PV` channel, which is exactly what it measures; if the paper keeps dirty WADI as a headline it must say so,
and if it moves to clean WADI the WADI row becomes a tie with AE.
