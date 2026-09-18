# A8 as proximity masking: is an extreme of one regime absorbed by an adjacent regime and missed?

Scripts: `a8_masking.py` (regime geometry, masking test, context recovery, synthetic control, WADI frontier;
one JSON line per item in `a8_masking.<ds>.jsonl`, per-window tables `a8_masking_<ds>_seed<s>.npz`) and
`a8_masking_obs_sweep.py` (robustness of the observation-space test over the regime partition: PCA 10/20 x
K 8/16/24 x 3 GMM seeds x 2 context rules; `kind=obs_sweep` rows in the same jsonl, plus LatAD seeds 0-2 on WADI).
Datasets are the paper's configurations (WADI_clean 2614 train / 575 test windows, 56 anomalies of which 43
difficult; HAI 18359 / 14819, 652 / 167; SWaT_canon 4530 / 1498, 233 / 85), W=60, stride 30, "stats" features.
The detector is the shipped recipe refit in place (K=20/40/40, latent 10/16/16, 40 epochs, k_density 80,
resid auto, basin auto): seed-0 scores correlate 0.99 to 1.00 with `scores_<ds>.npz` (AUROC 0.808 vs 0.798
shipped on WADI, 0.928 vs 0.927 SWaT, 0.935 vs 0.941 HAI). Three VaDE seeds per dataset. Paper and shipped
model untouched.

## Definitions used (the corrected A8)

- Regime k = a train-normal component; its empirical Gaussian (Ledoit-Wolf, full covariance) in the scoring
  space gives a Mahalanobis distance d_k to every window; q50_k and q99_k are the distances of k's own train
  members. "Close to k" = d_k < q50_k, "far from k" = d_k > q99_k. Everything is scale-free per regime.
- Nearest regime = argmin_k d_k (the s_near head). Expected regime = the regime of the last NORMAL window
  before the anomaly (up to 20 windows back); every WADI/SWaT anomaly window and 570 of 652 HAI windows have
  one. Transition matrix P(next | current) from the train sequence gives a transition-weighted context score.
- A8-masking pattern (strict): far from the expected regime AND nearest regime != expected AND close (< q50)
  to that nearest regime. Loose pattern: far from expected AND absorbed by a different regime (any depth).
- Missed = shipped LatAD score at or below the train-normal 95th percentile (5% FPR budget; test-normal q95
  also reported). The same pattern is counted on TEST-NORMAL windows as the base rate, since a legal regime
  switch produces exactly the same signature (far from the previous regime, inside another).
- Two scoring spaces: the model's own VaDE latent with its own components (what the shipped heads see), and
  a model-free PCA-10 of the standardised features with a full-covariance GMM-16 partition (the partition
  the earlier A8 screen measured valleys on), swept over K / dim / GMM seed in the robustness run.

## Bottom line

**A8-as-proximity-masking is not what causes the misses on any of the three testbeds.** In the model's own
latent, missed anomalies sit inside the median contour of the regime the plant was in (WADI: median distance
to the expected regime 0.50 to 0.71 of its q99, identical to test normals at 0.50 to 0.62; every missed
WADI episode onset at 0.4 to 1.1; HAI: 0.32 to 0.44, identical to normals; 96 to 100% of missed HAI windows
inside the nearest regime's q50). The strict pattern occurs in 0 to 3 of 25 to 30 missed WADI windows
(base rate on normals 4 to 6%), 5 to 7 of 33 to 35 missed SWaT windows (base rate 17 to 22%, i.e. at chance),
0 of 56 to 61 missed HAI windows. Nothing in the data looks like the synthetic positive control, where the
same test finds the pattern in 100% (loose) / 37 to 42% (strict) of the injected excursions against a 1 to
4% base rate.

**A weaker, partition-dependent signal exists on WADI only, and it is the A10 lead, not A8.** With a
model-free observation-space partition, 41% of missed WADI windows (range 22 to 70% over 18 partitions)
are beyond the q99 of the regime they came from and absorbed by a neighbouring regime, against a 12.5%
base rate on test normals (enrichment 3.5x, range 1.6 to 5.6). Scoring each window against its expected
regime instead of its nearest regime raises the frontier-vs-normal AUROC from 0.48 to 0.53 (LatAD, three
seeds) to 0.67 (range 0.45 to 0.83) and a robust fusion of LatAD with that context score raises the
difficult-subset AUROC from 0.715 / 0.733 / 0.753 (LatAD seeds 1 / 2 / 0) to 0.793 / 0.802 / 0.805 (mean over
18 partitions, better than LatAD in 16 to 17 of 18). The strict "absorbed inside the neighbour's core"
version stays at chance (0 to 6 windows, base rate 3 to 7%); the windows are beyond their own regime's q99
but only at the edge of the neighbour (median 1.0 to 1.4 of its q50), i.e. they are small departures that a
per-regime envelope test with the right regime catches, which the nearest-regime head lets through. On
SWaT and HAI the same fusion gives nothing (SWaT 0.805 to 0.787 to 0.818 depending on the fusion, HAI
0.818 to 0.796 to 0.811). n is small (WADI: 27 missed windows in 8 episodes, 12 to 19 pattern windows in
3 to 4 episodes) and the partition spread is wide, so this is a lead for the A10 extension, not a result.

So: A8 in the paper's Table 1 stays "not observed", with a sharper statement of what was tested (below). The
paper's A10 row gains a concrete, measured motivation on WADI. The WADI residual frontier is not A8-masked in
the model's latent and is partly explained (windows 544 to 547, 237 to 238) by a regime-context score in
observation space.

## 1. Synthetic positive control: the blind spot is real and the instrument sees it

Two regimes A and B, 3 factors mapped to 48 features (rank-3 plus 0.1 noise, like window statistics),
centroids 6 within-regime sd apart along factor 1 (LDA/Bayes error 0.13%: distinct, deep valley, no
overlap), AR(1) within regime, dwell 40 windows, 4000 train / 3000 test windows. Anomaly TOWARD-B: while in
A, a 3-window excursion of factor 1 to +5 (an extreme of A landing 1 sd from B's centre), 68 windows in 25
episodes. Control AWAY: factor 1 to -5 (empty space), 72 windows. Invariant stated in advance: the nearest /
density heads must catch AWAY and miss TOWARD-B; the expected-regime score must catch both.

| head (shipped recipe, VaDE K=2 / K=8) | TOWARD-B: TPR at 5% FPR / AUROC | AWAY: TPR / AUROC |
|---|---|---|
| LatAD `anomaly_score_hard` | 0.06 / 0.54 (K=2), 0.09 / 0.53 (K=8) | 1.00 / 1.000 |
| density head alone (K=80 GMM) | 0.13 / 0.12 | 1.00 |
| nearest-component head alone | 0.04 / 0.09 | 1.00 |
| expected-regime score (this test, VaDE-K2 regimes) | 1.00 / 0.980 | 1.00 / 0.979 |
| transition-weighted score | 0.81 / 0.964 | 1.00 / 1.000 |
| expected-regime score, true labels in PCA-10 | 1.00 / 0.978 | 1.00 / 0.976 |

Every TOWARD-B window is assigned to the B component by the model (100%); 94% are missed by LatAD. The
masking pattern fires on 100% of them (loose) and 37 to 42% (strict; the q50 core of B in 8 to 10 dimensions
is tighter than "1 sd from the centre along one factor"), against 1.3% (strict) / 3.6% (loose) on normals.
At the same 5% test-normal false-positive budget the expected-regime score recovers 65 of 65 missed
windows and loses none. Cost of context: the expected-regime score's false positives are the legal
switches (2.7% of normal windows are the first window after an A/B switch); with K=8 the VaDE
over-segments each regime (normal "switch" rate 49%) and the context score's TPR at 5% FPR drops to 0.53
while its AUROC stays 0.95, so partition granularity is the main practical risk of the method (seen again on
SWaT below). Conclusion: the shipped heads have exactly the A8 blind spot the corrected definition
describes, and the test built here detects it. A null on real data is therefore "absent", not "blind".

## 2. Regime geometry: are regimes close enough for masking to be possible?

Pairwise centroid separation in pooled Mahalanobis units (sqrt of d' S^-1 d, S the average covariance), and
the "reach" test: is B's centre inside A's q99 envelope (so an A point at its q99 edge sits on B's centre)?

| dataset, space (regimes >= 20 windows) | q50 / q99 of own members (chi ref) | separation min / p10 / median / max | share of pairs < 4 / < 6 | pairs where one q99 envelope reaches the other centre | train windows inside two q50 cores |
|---|---|---|---|---|---|
| WADI, VaDE latent 10-D, seed 0 (20 regimes) | 2.25 / 3.34 (3.06 / 4.82) | 1.9 / 3.8 / 7.4 / 16 | 13% / 33% | 43 of 190 | 10.9% |
| WADI, VaDE seeds 1, 2 | 2.3 / 3.4 | 2.1 / 3.9 / 7.6 / 30; 1.7 / 3.5 / 7.1 / 34 | 12 to 15% / 31 to 35% | 48 / 48 | 6.0 / 6.5% |
| WADI, PCA-10 GMM-16 | 2.53 / 4.51 | 2.4 / 3.5 / 6.4 / 21 | 15% / 46% | 38 of 120 | 9.0% |
| SWaT, VaDE latent 16-D, seed 0 (33 regimes) | 2.60 / 7.15 (3.92 / 5.66) | 4.1 / 9.4 / 24.6 / 126 | 0% / 3% | 71 of 528 | 0.0% |
| SWaT, PCA-10 GMM-16 (12 regimes) | 2.13 / 10.44 | 3.4 / 5.6 / 11.0 / 35 | 3% / 12% | 15 of 66 | 0.8% |
| HAI, VaDE latent 16-D (35 regimes; one holds 18348 of 18359 windows under re-assignment) | 3.60 / 6.39 | 1.8 / 4.6 / 9.1 / 42 | 7% / 21% | 154 | 52% |
| HAI, PCA-10 GMM-16 | 2.88 / 4.85 | 2.0 / 3.8 / 10.7 / 35 | 14% / 31% | 32 of 120 | 21% |

Reading, and reconciliation with the earlier finding that nearest pairs lack density valleys (64 to 92%)
while overlapping little (OVL <= 0.27):
- Adjacency is real and common. On WADI and HAI a third of regime pairs are within 6 pooled sd and the
  closest pairs are at 1.7 to 2.7, which is the geometry where an extreme of one regime (q99 = 3.3 to 4.5)
  lands inside the neighbour's core (q50 = 2.3 to 2.9). No valley plus small OVL plus separation 2 to 4 is
  one consistent picture: adjacent slices of one elongated mode, touching, not coincident. The precondition
  for A8 masking (close, distinct neighbours) holds on WADI and HAI; it does not on SWaT (median separation
  11 to 25, no pair under 3.4, valleys everywhere).
- The closest WADI pairs in the VaDE latent are tiny components (n = 11 to 56) next to the dominant one
  (n = 939 to 1574, 36 to 60% of train); the closest PCA pairs are balanced (n = 94 to 268). On HAI the VaDE
  mixture is effectively one regime for this test (its Ledoit-Wolf covariance swallows the others; normal
  switch rate 0.1%), so HAI's verdict rests on the observation-space partition.
- 6 to 11% of WADI train windows and 21% of HAI (PCA) sit inside two q50 cores at once: the space between
  adjacent regimes is populated, which is why "absorbed by the neighbour" cannot by itself mark an anomaly.

## 3. The masking test on the real anomalies

Per dataset and seed, in the model's own latent (VaDE regimes) and in PCA-10 GMM-16. "Loose" = beyond
q99 of the expected regime and nearest regime differs; "strict" adds inside the neighbour's q50. Base rate
= the same pattern on test-normal windows with context.

| dataset, space, seed | missed anomalies (of which difficult) | strict pattern: missed / base rate | loose pattern: missed / base rate | missed windows inside the expected regime's q50 core | median d_exp (q99 units): missed / normals |
|---|---|---|---|---|---|
| WADI VaDE s0 | 27 (26) | 1 / 4.1% | 2 / 7.5% | 13 | 0.50 / 0.50 |
| WADI VaDE s1 | 25 (25) | 3 / 5.8% | 7 / 13.5% | 4 | 0.71 / 0.62 |
| WADI VaDE s2 | 30 (29) | 0 / 3.7% | 4 / 8.7% | 15 | 0.51 / 0.54 |
| WADI PCA-10 GMM-16 | 27 (26) | 3 / 5.0% | 12 / 13.3% | 3 | 0.99 / 0.63 |
| WADI PCA, 18 partitions (mean [range]) | 27 | 1.4 [0, 6] / 4.9% | 41% [22, 70] / 12.5% [7, 17] | | |
| SWaT VaDE s0 / s1 / s2 | 33 / 35 / 34 (all difficult) | 7 / 5 / 5 vs 21.7 / 17.0 / 17.6% | 26 / 26 / 23 vs 53 / 49 / 45% | 2 / 1 / 3 | 2.4 to 3.4 / 0.7 to 1.5 |
| SWaT PCA-10 GMM-16 | 33 | 6 / 15.7% | 13 / 29.8% | 3 | 0.85 / 0.51 |
| SWaT PCA, 18 partitions | 33 | 3.8 [0, 10] / 10.3% | 48% [6, 73] / 30.6% [10, 50] | | |
| HAI VaDE s0 / s1 / s2 | 56 / 61 / 60 (47 / 51 / 49) | 0 / 0 / 0 vs 0.0% | 0 / 0 / 0 vs 0.0 to 0.1% | 54 / 59 / 52 | 0.40 / 0.44 / 0.32 vs 0.42 / 0.50 / 0.32 |
| HAI PCA-10 GMM-16 | 56 (47) | 4 / 0.7% | 7 / 3.0% | 12 | 0.65 / 0.67 |
| HAI PCA, 18 partitions | 56 | 0.6 [0, 6] / 0.4% | 4.7% [0, 16] / 2.3% [0, 5] | | |

- WADI in the model's latent: the missed anomalies are in the bulk of the regime the plant was in. Their
  distance to the expected regime has the same distribution as the normals' (median 0.50 vs 0.50 of q99,
  p90 0.71 vs 0.93). Episode view (`a8_episodes`, three seeds): 8 of 11 episodes have a missed window; of
  their onset windows (the first attack window, whose context is genuinely pre-attack) 5 to 6 are missed,
  and every one of those sits at 0.4 to 1.1 of its expected regime's q99, none inside another regime's core.
  The only recurring loose-pattern episode is 494 to 505 (seeds 0, 1, 2: 2, 1, 3 windows), where the
  missed windows come late in an 11-window episode whose earlier windows are caught.
- WADI in observation space: 12 of 27 missed windows (seed-0 GMM-16) fit the loose pattern, from episodes
  195 to 198 (expected regime 4, absorbed by 6 or 8; top channels 2_FIC_401_SP +2.3 sigma, 2B_AIT_004_PV
  -2.2), 235 to 236 (expected 14, absorbed by 8; 2A_AIT_004_PV -2.8 to -3.0, 3_LT_001_PV +2.4 to +2.6),
  544 to 547 (expected 12, absorbed by 8; 2B_AIT_004_PV -2.2, 2_FIC_501_SP +1.7, then 2_LT_002_PV +3.8 at
  547), 507 and 202. Distances to the expected regime 1.1 to 4.0 q99 units; distance to the absorbing regime
  0.7 to 1.7 of its q50 (three inside the core: 546, 547, 236). The absorbing regime 8 (n = 240) is 3.1 to
  4.1 pooled sd from regimes 12, 14 and 4, i.e. an adjacent slice, and the observed transition 12 -> 8 has
  train probability 0.22, 14 -> 8 has 0.03, 4 -> 8 has 0.07. This is the A8 signature in observation space,
  at 3.5x the normal base rate, on roughly 4 episodes.
- SWaT: the plant cycles through regimes every window or two (normal switch rate 43 to 58%, loose base rate
  30 to 53%); missed anomalies "absorbed by another regime" are at or slightly above that base rate (loose
  enrichment 1.5x, strict at chance). The three missed SWaT anomalies that are inside another regime's core
  with a huge distance to the expected one (windows 246 to 248, d_exp 9.4, latad percentile 0.20 to 0.22:
  AIT201 +3.2 sigma, P201 -1.8) are chemistry-channel offsets that put the window in a different regime's
  core for the whole episode; a context score would flag them, and it would equally flag the half of normal
  windows that switch. Not A8-masking in any usable sense.
- HAI: no masking in the latent (zero pattern windows, zero base rate: the model has one effective regime)
  and 7 of 56 (4 strict) in observation space, all but one from a single episode (3430 to 3435: P1_PCV01D
  +1.8 sigma, P1_B2016 +2.0; expected regime 1, absorbed by 14, 6.4 pooled sd apart, transition probability
  0.001). Base rate 3.0% / 0.7%, so the episode is genuine but it is one episode of 38.

## 4. Context recovery: nearest-regime score vs expected-regime score

AUROC of each score against test normals (LatAD = shipped; nearest = d to nearest regime / its q99;
expected = d to the expected regime / its q99; transition = NLL under the transition-weighted mixture from
the previous regime). Fused = LatAD + expected as a robust z-sum (median/IQR of test normals, clipped at
20; the plain z-sum is degenerate because a few clipped-channel WADI test normals give LatAD a std of 3e12).

| dataset, space | subset (n) | LatAD | nearest | expected | transition | fused LatAD+expected |
|---|---|---|---|---|---|---|
| WADI VaDE s0 / s1 / s2 | difficult (43) | 0.753 / 0.715 / 0.733 | 0.738 / 0.683 / 0.637 | 0.713 / 0.758 / 0.642 | 0.760 / 0.723 / 0.741 | |
| WADI VaDE s0 / s1 / s2 | frontier (15) | 0.527 / 0.477 / 0.507 | 0.585 / 0.499 / 0.391 | 0.500 / 0.506 / 0.346 | 0.542 / 0.515 / 0.501 | |
| WADI PCA, 18 partitions, LatAD s0 | difficult (43) | 0.753 | 0.710 [0.65, 0.78] | 0.729 [0.62, 0.82] | 0.718 [0.68, 0.75] | 0.805 [0.72, 0.87], > LatAD in 16/18 |
| WADI PCA, LatAD s1 / s2 | difficult (43) | 0.715 / 0.733 | | | | 0.793 [0.70, 0.86] / 0.802 [0.71, 0.87], > LatAD in 17/18 |
| WADI PCA, 18 partitions | missed by LatAD (27) | 0.611 | 0.649 | 0.713 [0.54, 0.85] | 0.639 | |
| WADI PCA, 18 partitions | frontier (15) | 0.527 | 0.598 [0.48, 0.71] | 0.673 [0.45, 0.83] | 0.512 | |
| WADI PCA GMM-16 | strict-masked missed (3) / loose (12) | 0.53 / 0.71 | 0.20 / 0.70 | 0.90 / 0.92 | 0.76 / 0.88 | |
| SWaT VaDE s0 / s1 / s2 | difficult (85) | 0.805 / 0.804 / 0.798 | 0.675 / 0.486 / 0.498 | 0.732 / 0.663 / 0.681 | 0.820 / 0.802 / 0.800 | |
| SWaT PCA, 18 partitions | difficult (85) | 0.805 | 0.758 | 0.735 [0.62, 0.81] | 0.808 | 0.787 [0.76, 0.83], > LatAD in 4/18 (transition-fused 0.818, 16/18) |
| HAI VaDE s0 / s1 / s2 | difficult (167) | 0.818 / 0.794 / 0.816 | 0.603 / 0.556 / 0.562 | 0.603 / 0.554 / 0.561 | 0.802 / 0.773 / 0.799 | |
| HAI PCA, 18 partitions | difficult (167) | 0.818 | 0.641 | 0.701 [0.62, 0.82] | 0.769 | 0.796 [0.77, 0.84], > LatAD in 2/18 |
| HAI PCA GMM-16 | strict-masked missed (4) / loose (7) | 0.13 / 0.25 | 0.10 / 0.26 | 0.99 / 0.99 | 0.37 / 0.37 | |

- In the model's latent the expected-regime score does not beat the nearest-regime score on any dataset's
  difficult subset beyond seed noise (WADI +0.06, -0.03, +0.005 over the three seeds), and it is far worse
  than LatAD on SWaT and HAI. The context is coarse there (WADI: one component holds 36 to 60% of train; HAI:
  one effective regime), so "expected regime" carries little information.
- In observation space on WADI, the context score is the best single score on the windows LatAD misses
  (0.71 vs 0.61) and on the frontier (0.67 vs 0.53), and the robust fusion lifts the difficult subset by
  +0.05 to +0.08 for every LatAD seed, in 16 to 17 of 18 partitions. The gain is partition-dependent (fused
  range 0.70 to 0.87; best at K = 24, PCA-10, "previous window" rule: 0.82 alone, 0.87 fused; worst at
  K = 16 seeds 1, 2) and it is bought with the legal-switch false positives (the expected score at its
  train-q95 threshold fires on 2 to 5% of WADI test normals, comparable to LatAD's 4 to 6%).
- On SWaT and HAI the fusion is neutral or harmful. On HAI the strict-masked episode 3430 to 3435 is
  recovered perfectly by the context score (AUROC 0.99 vs 0.13 for LatAD), but it is 4 windows.
- Recovery at a fixed 5% test-normal budget (seed 0, GMM-16): WADI expected recovers 4 of 29 missed and
  loses 21 caught (it is not a replacement); transition-weighted recovers 10, loses 9; HAI expected recovers
  75 of 246 (33 of 117 difficult) and loses 57, the rest of the gain being HAI's known test-normal drift.

## 5. The WADI residual frontier (windows 16 to 21, 237 to 238, 360 to 362, 544 to 547)

In the model's latent (three seeds): all 15 windows have expected regime = nearest regime = the dominant
component (seed 0: regime 10; only window 21 switches, to a regime with transition probability 0.017), d to
the expected regime 0.46 to 0.69 of q99 (windows 360 to 362 at 0.46 to 0.47, the "hyper-typical" trio),
expected-score percentile 0.20 to 0.64, strict and loose pattern 0 of 15 in seeds 0 and 2, 1 of 15 in seed
1. **Not A8-masked in the model's latent.**

In observation space (GMM-16, seed 0): 544 to 547 (expected regime 12, absorbed by 8, d_exp 1.1 to 2.5,
context percentile 0.88 to 0.96; LatAD 0.45 to 0.89) and 237 to 238 (expected 14, absorbed by 8, d_exp 0.9
to 1.0, context percentile 0.83 to 0.86) are beyond or at their previous regime's q99 while at the edge of
regime 8; 16 to 21 stay inside regime 15 (d_exp 0.6 to 1.2), 360 to 362 inside regime 14 (0.6). Over the 18
partitions, 1 to 10 frontier windows (mean 5.6) fit the loose pattern and 0 to 9 (mean 4.3) exceed the
context score's train 95th percentile; frontier AUROC 0.67 mean. So about a third of the frontier
(episode 544 to 547 consistently, 237 to 238 in most partitions) is a departure from the regime the plant was
in that lands at the edge of an adjacent regime, and a context-conditioned score sees it; the other two
thirds (16 to 21, 360 to 362) are inside their own regime on every axis and remain unexplained.

## 6. What this means for the paper (no edits made)

- A8 remains "specified, not observed", now under the correct reading of A8 (false negatives by absorption)
  and with an instrument shown to catch the effect when it exists. Recommended wording of the test: "an
  anomaly beyond the 99% envelope of the regime the plant was in, scored inside an adjacent regime by the
  nearest-component head: 0 to 3 of 25 to 30 missed WADI windows (base rate 4 to 6%), 0 of 56 to 61 HAI,
  at the 17 to 22% base rate on SWaT (which cycles regimes every window); a synthetic excursion of this
  kind is missed by the shipped heads (TPR 0.06) and caught by an expected-regime score (TPR 1.00)".
- The regime geometry precondition for A8 (adjacent, distinct regimes 2 to 4 sd apart, no valleys) IS
  present on WADI and HAI; what is absent is anomalies that exploit it. This resolves the earlier tension:
  "no overlap" and "no valleys" are both right, and neither one decides A8.
- A10 gets a measured motivation: on WADI, a context-conditioned regime score is complementary to LatAD
  (+0.05 to +0.08 difficult AUROC across three LatAD seeds, partition-dependent), and it partly explains the
  residual frontier. The earlier A10 measurement (AR / kNN / transition on the PCA-20 latent) did not see
  this because it conditioned on the previous window's position, not on the previous window's regime
  envelope; the per-regime q99 normalisation is what separates a 2-sigma analyzer offset from ordinary
  within-regime motion.
- The transit-window false-positive cost from the previous A8 note still stands and applies doubly to any
  context score: legal switches are its false positives. Normal windows change regime in 25 to 36% of WADI
  windows, 17% of HAI windows (observation space) and 43 to 58% of SWaT windows; the expected-regime score at
  its train 95th percentile fires on 2 to 5% of WADI test normals, 11 to 19% of HAI's (which drift) and 0.2
  to 5% of SWaT's. Its usefulness scales inversely with how often normal operation switches regime.

## 7. Caveats

- n: WADI 27 missed windows in 8 episodes, 12 to 19 loose-pattern windows in 3 to 4 episodes, frontier 15
  windows in 4 episodes; window-level binomial tests are quoted in `a8_episodes` output but windows within
  an episode are not independent, so the episode counts are the real evidence (WADI: 3 to 4 of 8 missed
  episodes show the pattern in observation space, 1 to 3 in the latent).
- The observation-space result depends on the partition (loose share 22 to 70%, fused AUROC 0.70 to 0.87
  over 18 partitions); no partition was selected on the test set, all 18 are reported, and 16 to 17 of 18
  beat LatAD, but the best partitions should not be quoted alone.
- Expected regime = previous normal window's regime (1-step). A majority-of-3 rule was also run (`rule =
  maj3`): base rates roughly double and the enrichment falls to 1.5 to 2.1x on WADI, so the signal is at the
  1-window scale. Longer memories were not tested.
- The strict q50 criterion is conservative in 10 to 16 dimensions (the synthetic excursion 1 sd from B's
  centre along one factor is inside B's q50 only 37 to 42% of the time); the loose pattern is the operative
  one, and its base rate on normals is the control that keeps it honest.
- HAI's VaDE latent is not usable for regime context (one effective regime after Ledoit-Wolf re-assignment;
  38 of 40 components populated by responsibility but all within the dominant one's envelope), and HAI test
  normals drift (LatAD fires on 19 to 20% of them at the train q95), which inflates every train-calibrated
  recovery count on HAI; AUROCs are unaffected.
- The three VaDE seeds are refits of the shipped recipe (correlation 0.98 to 1.00 with the shipped scores),
  not the shipped checkpoints themselves; `k_density` and all heads as in `build_scores_table.py`.
- Agent execution: about 70 minutes of tool calls; local CPU about 15 minutes total (three datasets x three
  seeds in parallel at 20 to 90 s per fit, six partition sweeps at 1 to 3 minutes each); no cloud cost.
