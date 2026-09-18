# A9 / A10 measurement: are "many clocks" and "path dependence" present in WADI, HAI, SWaT?

Script `_diagnostics/a9a10_measure.py` (rows appended to `a9a10_measure.jsonl`; run-1 archive `a9a10_measure_run1.jsonl`;
post-hoc `a9a10_posthoc.log`; per-window scores `a9a10_wsweep_<ds>.npz`, `a9a10_scatter_<ds>_<latent>.npz`; figure
`a9a10_scatter.png`). Nothing in the paper or the shipped model was touched. Datasets are the paper's configurations:
WADI_clean (575 test windows, 56 anomalies = 13 easy + 43 difficult), HAI (14 819 / 652 = 485 + 167), SWaT_canon
(1 498 / 233 = 148 + 85); grid W=60, stride 30, difficulty = the paper's max|u| rule at the train 99th percentile.
Detectors here are deliberately simple and fixed (PCA30 + diagonal GMM K=24, IsolationForest, trivial max|u|; single
seed), so every number is a property of the data, not of LatAD.

## Verdict (one paragraph)

**A10 (path dependence) is MEASURED ABSENT on all three testbeds.** Three conditional scorers (linear AR on the two
preceding non-overlapping windows, a kNN successor model, a K=24 regime-transition matrix), validated on a synthetic
positive control and guarded by an order-shuffle control, find no anomaly windows that are marginally normal yet
improbable given history beyond the shuffle floor, once windows whose history already contains an attack are excluded.
The 15 WADI residual-frontier windows sit between the 30th and 90th percentile of normal on every marginal and
conditional axis (none above the 95th); conditioning on history does not crack them. **A9 (many clocks) is MEASURED
PRESENT as a data property everywhere** (per-channel timescales span 3.4 to 3.7 decades in all three plants) **and is
exploitable on HAI only**: integrating a 600-sample window lifts difficult-subset AUROC 0.782 to 0.841 on the 145
difficult windows with no easy slot in their history (29 windows separable only at W=600 vs 10 only at W=60; multiscale
fusion 0.864 vs 0.800), while on WADI and SWaT longer windows hurt (0.673 to 0.596; 0.723 to 0.547). The paper's claim
"snapshot-detectable benchmarks" is therefore measured-true for A10 on all three and for A9 on WADI and SWaT; on HAI
part of the difficult subset lives at a longer timescale than W=60 (a slow, persistent fault that accumulates), which
is an A9 lever the paper already flags as "complementary" (Appendix A, +0.072 with temporal features). No WADI difficult
window is cracked by history or by scale.

## Instruments and their validation

Synthetic positive control (`SYNTH_cycle`): three regimes on a strict cycle A->B->C->A with dwell 5 windows, AR(1)
noise; anomaly = one backwards step (an illegal successor; the window itself is a perfectly normal regime). 3 599 train
/ 1 809 test windows, 225 anomaly windows, 221 of them "difficult" by the max|u| rule.

| scorer | AUROC (difficult) | cell (marg<.95 & cond>.99): anomalies / normals | shuffle control AUROC (3 seeds) |
|---|---|---|---|
| marginal GMM | 0.579 | (reference) | |
| conditional AR (linear) | 0.713 | 0/171 vs 4/1542 | 0.555 to 0.574 |
| conditional kNN successor | **0.764** | **65/171 vs 14/1542** | 0.527 to 0.549 |
| regime-transition matrix | **0.720** | **73/171 vs 0/1542** | 0.451 to 0.508 |

Never-seen transitions: 46% of anomaly windows vs 0% of normals. Shuffling the train order collapses every
conditional to chance and the AR R^2 from 0.39 to 0.005. So the kNN and transition instruments detect a
history-dependent fault that the marginal does not; the linear AR sees "a change" rather than "an illegal change" (its
cell is empty), so on real data the AR gain must be read as change-point sensitivity, not path legality. A null on real
data is therefore "not present", not "test broken".

Contamination guard (added after run 1): a longer window or an AR history that contains an earlier attack slot can flag
a later window through its history. Every A9/A10 gain is therefore reported on the full subset AND on the subset whose
extra history is attack-free ("strict") or free of easy-anomaly slots ("loose"). Cell counts use test-normal quantile
thresholds (marginal < q95, conditional > q99 of test normals) so unshuffled and shuffled fits are comparable
(run 1 used a contiguous-crossfit train calibration that is not comparable across the shuffle; its AUROCs, which are
calibration-free, agree with run 2).

## A9.1 Per-channel timescales (train-normal, ACF 1/e decay tau in grid samples; lmax 5000; Welch dominant period)

| dataset | active ch | tau p05 / p50 / p95 | decades (p95/p05) | log10 tau std | share tau<20 / 20-60 / 60-180 / 180-600 / >=600 | fastest | slowest |
|---|---|---|---|---|---|---|---|
| WADI_clean | 92/122 | 2 / 168 / 4743 | 3.37 | 0.92 | .21 / .04 / .29 / .26 / .20 | 3_AIT_004_PV, 3_AIT_003_PV (1) | 1_AIT_001_PV, 2B_AIT_001_PV (>=5000) |
| HAI | 50/59 | 1 / 84 / 5000 | 3.70 | 1.31 | .20 / .16 / .16 / .02 / .46 | P4_ST_TT01, P2_VT01e (1) | P1_FT03Z, P4_ST_PS (>=5000) |
| SWaT_canon | 43/51 | 2 / 128 / 5000 | 3.38 | 1.04 | .16 / .26 / .35 / .00 / .23 | MV301 (1), P602 (2) | AIT402, AIT501 (>=5000) |

(WADI and SWaT grids are 10 s per sample, HAI 1 s.) Literal "many clocks": in every plant a fifth of the channels decorrelate
within a third of the window and a fifth to a half take more than ten windows. Dominant Welch periods likewise span 1.6
to 3.6 decades. This is the property the paper's Table 1 calls "not observed"; it is observed. Whether it matters for
detection is the next item.

## A9.2 Window-size sweep (stride 30 fixed; label anchored on the paper's W=60 slot; difficult set fixed at W=60)

Difficult-subset AUROC, PCA+GMM detector (IF in the jsonl):

| dataset | W=20 | W=60 | W=180 | W=600 | fusion (max pct over W) | n difficult |
|---|---|---|---|---|---|---|
| WADI_clean | 0.647 | 0.684 | 0.555 | 0.677 | 0.691 (IF: 0.735 vs 0.689) | 43 |
| HAI | 0.752 | 0.801 | 0.830 | **0.859** | **0.864** | 167 |
| SWaT_canon | 0.676 | 0.753 | 0.751 | 0.669 | 0.783 | 85 |

Contamination-controlled, same subset scored at every W (`a9a10_posthoc.log`):

| dataset | subset (defined at W=600) | n | W=20 | W=60 | W=180 | W=600 | windows >p95 only at W=600 / only at W=60 |
|---|---|---|---|---|---|---|---|
| HAI | loose (no easy slot in history) | 145 | 0.735 | 0.782 | 0.809 | **0.841** | **29 / 10** |
| HAI | strict (history attack-free) | 42 | 0.779 | 0.772 | 0.671 | 0.705 | 2 / 7 |
| WADI_clean | loose | 31 | 0.665 | 0.673 | 0.488 | 0.596 | 2 / 4 |
| WADI_clean | strict | 10 | 0.550 | 0.536 | 0.411 | 0.602 | 2 / 0 |
| SWaT_canon | loose | 46 | 0.632 | 0.723 | 0.710 | 0.547 | 3 / 11 |
| SWaT_canon | strict | 19 | 0.660 | 0.755 | 0.623 | 0.490 | 1 / 7 |

Reading. On HAI the W=600 gain survives the loose control (the history contains only difficult, per-slot faint attack
slots, so the long window integrates a persistent faint fault) and disappears under the strict control (episode
onsets, where there is no fault history to integrate and the long window only dilutes). That is a scale effect in the
A9 sense: three entirely-difficult HAI episodes flip from unseparable to separable only at W=600 (windows 2953-2964:
0.45 -> 0.91 of windows above p95; 3059-3078: 0.37 -> 0.95; 3429-3436: 0.00 -> 1.00; per-episode table in the jsonl).
On WADI and SWaT every longer window loses on every subset; the two WADI episodes that gain at W=600 (283-286, n=3,
contains one easy window; 506-510, n=4) are 2 to 3 windows and do not move the subset AUROC. Caveat for deployment:
the GMM false-alarm rate at the train 95th percentile grows with W (WADI 3.7% -> 12.1%, SWaT 8.8% -> 21.1% at W=600),
so a long-window detector needs its own calibration; AUROC is unaffected.

## A10.1 Marginal vs conditional surprise (PCA-20 latent of the stats features; VaDE-10 latent as a check)

Difficult-subset AUROC (all difficult windows / difficult windows with attack-free history):

| dataset (latent) | marginal GMM | marginal Gauss | cond AR | cond kNN | cond transition | shuffle AR / kNN / trans (hard, seed-mean) |
|---|---|---|---|---|---|---|
| WADI (pca20) | 0.697 / 0.615 | 0.629 / 0.580 | 0.672 / 0.597 | 0.687 / 0.589 | 0.491 / 0.466 | 0.631 / 0.620 / 0.569 |
| WADI (vade10) | 0.723 / 0.698 | 0.705 / 0.691 | 0.735 / 0.685 | 0.712 / 0.721 | 0.540 / 0.575 | 0.732 / 0.693 / 0.619 |
| HAI (pca20) | 0.793 / 0.755 | 0.756 / 0.712 | 0.820 / 0.746 | 0.790 / 0.748 | 0.614 / 0.573 | 0.760 / 0.764 / 0.467 |
| HAI (vade10) | 0.733 / 0.662 | 0.703 / 0.623 | 0.794 / 0.686 | 0.775 / 0.721 | 0.674 / 0.586 | 0.704 / 0.703 / 0.655 |
| SWaT (pca20) | 0.777 / 0.757 | 0.675 / 0.747 | 0.717 / 0.621 | 0.774 / 0.706 | 0.653 / 0.522 | 0.676 / 0.675 / 0.590 |
| SWaT (vade10) | 0.719 / 0.692 | 0.634 / 0.648 | 0.736 / 0.644 | 0.712 / 0.616 | 0.648 / 0.531 | 0.644 / 0.620 / 0.582 |

Shuffle invariant holds: shuffled AR equals the Gaussian marginal (SWaT 0.676 vs 0.675; HAI 0.760 vs 0.756; WADI
0.631 vs 0.629) and the AR R^2 drops from 0.55 to 0.71 to under 0.02, so the conditional models do learn real temporal
structure of normal operation. It just does not separate anomalies: on WADI and SWaT every conditional is at or below
the marginal on every subset; on HAI the AR gain (+0.027 pca, +0.061 vade on all 167) vanishes on the 42 windows with
attack-free history (0.746 vs 0.755; 0.686 vs 0.662), i.e. it came from histories that already contained the attack.

The A10 cell (marginally normal AND conditionally improbable; test-normal thresholds; unshuffled vs shuffle mean):

| dataset | scorer | all: anomalies (difficult) / normals | attack-free history: anomalies (difficult) / normals | shuffle, attack-free history |
|---|---|---|---|---|
| WADI (pca20) | AR / kNN / trans | 4(2)/3, 0/4, 0/6 of 39 anom, 489 normals | 0/2, 0/4, 0/6 of 14 anom, 457 normals | 0/2, 0/4, 1/5 |
| HAI (pca20) | AR / kNN / trans | 109(24)/90, 34(19)/29, 39(15)/114 of 281 anom, 13 454 normals | 14(8)/42, **18(11)/27**, 7(2)/102 of 61 anom, 13 312 normals | 3/5, 3/7, 5/56 |
| SWaT (pca20) | AR / kNN / trans | 0/8, 5(4)/2, 7(6)/9 of 65 anom, 1 197 normals | 0/1, 0/2, 1/4 of 24 anom, 1 116 normals | 0/1, 0/2, 0/1 |

Never-seen regime transitions (K=24): anomaly vs normal windows: WADI 18% vs 4.8% (difficult 12%), HAI 28% vs 5.8%
(difficult 9%), SWaT 9% vs 1.0% (difficult 13%); transition AUROC on the difficult subset is 0.49 to 0.65, below the
marginal everywhere, so the enrichment is carried by easy anomalies that also change regime.

The one residual, HAI kNN cell with attack-free history (18 anomaly windows, 11 difficult, vs 27 of 13 312 normals;
shuffle 2-3 vs 4-9), was traced window by window (`a9a10_posthoc.log`): all 18 are episode-onset windows (position 0
or 1 in 17 of the 38 episodes), all 18 already exceed the train 95th percentile of the marginal GMM at W=60 (they fall
below the test-normal q95 only because 5.3% of HAI test normals exceed the train 99th percentile), and their conditional
percentiles are 0.997 to 1.0 because the state jumps away from the history (P1_FT01 +6.0 sigma, P1_B4002 -6.9 sigma at
onset). This is change-point sensitivity at fault onset, not a normal state made illegal by its path; it cracks no
window the marginal misses.

## A10.2 The WADI residual frontier (15 difficult windows no snapshot detector separates at 5% FPR)

Windows 16-21 (segment 0, 2B/2A_AIT_004_PV about -2.6 sigma), 237-238 (segment 4, AIT_004 about -3 sigma), 360-362
(segment 7, 3_LT_001_PV +1.6, 1_LT_001_PV -1.4), 544-547. Train-calibrated percentiles (pca20 / vade10):

| window | marginal GMM | cond AR | cond kNN | cond transition | transition count in train |
|---|---|---|---|---|---|
| 16 / 17 / 18 / 19 / 20 / 21 | .33 .84 .27 .70 .48 .58 | .36 .93 .40 .36 .40 .60 | .39 .98 .29 .55 .22 .36 | .70 .63 .61 .44 .38 .46 | 20, 31, 12, 26, 31, 34 |
| 237 / 238 | .40 .38 | .47 .29 | .71 .56 | .05 .67 | 196, 26 |
| 360 / 361 / 362 | .011 .026 .018 | .035 .040 .040 | .005 .008 .006 | .05 .05 .05 | 196 (x3) |
| 544 / 545 / 546 / 547 | .75 .51 .64 .64 | .92 .65 .66 .37 | .88 .40 .50 .41 | .72 .70 .42 .42 | 7, 12, 15, 15 |

Frontier-vs-normal AUROC: marginal 0.595 / 0.496, AR 0.491 / 0.435, kNN 0.592 / 0.464, transition 0.432 / 0.359;
windows above the 95th percentile on any axis: 0 (pca20 kNN: 1; vade10 kNN: 2), above the 99th: 0. All 15 sit on
well-travelled regime transitions (train counts 7 to 196). **A10 does not crack the WADI frontier**; these windows are
not history-dependent faults by any of the three instruments.

Speculative lead, flagged as such (n=3, one episode): windows 360-362 are hyper-typical, at the 1st to 3rd percentile
of normal on the marginal AND the kNN conditional, with 96-97 of 122 channels frozen within the window against a train
median of 73 (p05 58). A low-tail (two-sided) marginal test would flag them, but at the cost of 4.9% of test normals
(25/515) below the same 3% cut, for 3 of 56 anomaly windows: roughly a wash, not a win, unless a frozen-channel-count
feature separates them more sharply. Not measured further here.

## Per-dataset verdict

| | A9 clocks (data) | A9 exploitable (scale) | A10 path dependence | "snapshot-detectable" claim |
|---|---|---|---|---|
| WADI_clean | PRESENT (3.4 decades) | ABSENT: longer windows lose on every subset; 2 to 3-window flips only | ABSENT: no conditional beats the marginal; cells at shuffle level; frontier untouched | MEASURED TRUE; difficult windows are in the bulk of normal on every axis (small n: 43 difficult, 10-14 in strict subsets) |
| HAI | PRESENT (3.7 decades) | PRESENT: +0.06 difficult AUROC at W=600 on 145 windows, 29 windows only at W=600, 3 all-difficult episodes flip | ABSENT: AR gain vanishes with attack-free history; residual kNN cell = 18 onset windows already >p95 marginally | TRUE for A10; PARTLY FALSE for A9: a slow persistent fault accumulates over 600 s (multiscale lever, consistent with Appendix A +0.072) |
| SWaT_canon | PRESENT (3.4 decades) | ABSENT: W=600 loses (0.723 -> 0.547 loose) | ABSENT: all conditionals below the marginal; cells empty | MEASURED TRUE |

Suggested paper wording change (not applied): Table 1 says A9 is "not observed"; the timescale spread IS observed in
all three plants, what is absent is a scale-only fault on WADI and SWaT. A10 can now be stated as measured absent
(three conditional instruments, positive control, shuffle control) rather than asserted.

## Caveats

Single seed, simple fixed detectors (not the shipped LatAD); WADI difficult n=43 and strict subsets of 10 to 14 windows,
so WADI subset AUROCs move by 0.05 per window; HAI test normals drift relative to train (5.3% above the train 99th
percentile), which is why cells use test-normal thresholds; the history for the conditional models is the two preceding
non-overlapping windows (120 samples), so longer-memory path dependence (hours) is not tested; the regime-transition
model uses K=24 KMeans on the PCA latent, not the shipped VaDE mixture. Cranfield and CAN-bus were not run: both are
normal-only in this repo, so only A9.1 would be measurable and it adds nothing to the verdict.
