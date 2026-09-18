# A8-masking breadth sweep over every dataset on disk

Scripts: `a8_masking_allsets.py` (geometry, susceptibility, anomaly-masking rate, context-recovery gain; rows in
`a8_masking_allsets.jsonl`, one `main` row per (dataset, seed) plus `pairs` and `examples` rows, resumable) and
`a8_absorption_allsets.py` (the second failure mode, isolated anomaly absorbed by a wide component; rows in
`a8_absorption_allsets.jsonl`). Logs `a8_masking_allsets.g{1,2,3}.log`, `*.synth*.log`. Earlier versions of the
masking jsonl without the controls are kept as `a8_masking_allsets.v{1,2,3}_*.jsonl`. Paper and shipped model untouched.
Three GMM seeds per dataset (0, 1, 2); the PCA and the windows are deterministic, so seed spread is the partition spread only.

## Verdict in four lines

1. **A8 geometry (close-but-distinct regimes) is present in every real dataset except SKAB.** Label-free: 24 to 72 of the
   K(K-1)/2 discovered-regime pairs on WADI, HAI, SWaT, Cranfield, WindSCADA, AHU, CAN-bus and MetroPT are within twice the
   normal envelope of each other (D < 2R, R = 5 to 7) and separated by a density valley and by a linear classifier at
   under 5% error; 94 to 100% of the train mass sits in a component that has such a neighbour. Closest distinct pairs are 2.2
   to 3.9 Mahalanobis units apart. The masking corridor exists geometrically almost everywhere.
2. **No real anomaly set exploits it.** The anomaly-masking rate (true anomaly far from the pre-episode regime, inside another
   regime, missed by the nearest-component head) is 0.000 to 0.010 on WADI, HAI, SWaT, SKAB, Cranfield, MetroPT, AHU and the
   four SMD machines, and 0.20 on WindSCADA; but the same statistic on test-NORMAL windows over the same horizon (the
   regime-switch base rate) is 0.005 to 0.33, and WindSCADA's 0.20 sits below its own base rate of 0.33. Every real dataset
   has a negative excess. The two nearest things to a demonstrator are WindSCADA (inspected: the "masked" windows are the
   turbine dropping from generating to idle as the wind dies, absorbed by the idle regime, labelled anomalous only because
   they fall in a 72 h pre-failure window) and the paper's own MIIM synthetic, where 7 of 51 `drift` anomalies (14%) show
   the pattern and no other anomaly type does.
3. **The labelled metric is valid only where regime switching is rare, and this is a property of the failure mode, not of
   the metric.** The positive control (A drifting onto a close-distinct B) gives masking rate 0.12 against 0.000 for the
   negative control (same drift into empty space) and an oracle context-recovery gain of +0.16 against -0.05. But its
   lag-matched base rate is 0.11: at the window-snapshot level a masked drift is indistinguishable from a legitimate A-to-B
   switch over the same horizon; only the path (A10) separates them. A snapshot head therefore cannot have an A8 blind spot
   that a snapshot expectation fixes; A8 is real only as an A10 problem.
4. **The second failure mode (isolated anomaly absorbed by a wide component) is real and is a resolution effect.** With a
   two-component head, 28 to 80% of kNN-isolated anomalies are missed on WADI, AHU, WindSCADA, Cranfield and MIIM; the
   BIC-selected head (K = 13 to 16) brings that to 1.5 to 27%, and the paper's 80-component recipe to 0 to 13%. The ring
   control confirms the mechanism (wide head misses 99.7% of frozen-centre anomalies, narrow head 0.6%). None of these
   isolated anomalies is ever scored below the train median by any head (absorbed-at-q50 = 0.000 everywhere).

## 1. Inventory

| dataset | labels | channels | rate | W/stride (windows) | train / test windows (anomalous) | loader | notes |
|---|---|---|---|---|---|---|---|
| WADI (WADI_clean) | attack | 122 | 1 Hz, ds10 | 60/30 (10 min) | 2614 / 575 (32) | cached `raw_WADI_clean.npz` (eda_real) | 2B_AIT_002_PV dropped; clip 10 |
| HAI 20.07 | attack | 59 | 1 Hz | 60/30 (60 s) | 18359 / 14819 (580) | cached (eda_real) | train1+2, test1+2 concatenated |
| SWaT canonical | attack | 51 | 1 Hz, ds10 | 60/30 (10 min) | 4530 / 1498 (174) | cached (eda_real) | Dec-2015 interleaved attack file |
| SKAB | anomaly | 8 | 1 Hz | 20/10 | 1161 / 1874 (666) | new (per-file streams) | alternate files train/test; train = all-normal runs |
| Cranfield | fault (EvoFault>0) | 24 | 1 Hz | 20/10 | 3336 / 9273 (7747) | new (Training.mat T1-T3; FaultyCase1-6) | cases 1-3 faulty throughout, 4-6 have a clean prefix; clip 10 |
| MetroPT-3 | 3 failure windows | 15 | 1 min | 60/30 (1 h) | 1483 / 3571 (93) | `metropt_data.npz` | segment-level labels from maintenance reports |
| WindSCADA T06 | 72 h pre-failure | 79 | 10 min | 12/6 (2 h) | 10346 / 14230 (469) | `wind_t06.npz` | weak labels; clip 10 |
| AHU_office (was `_future`) | rule-derived FDD | 18 | 1 h | 24/12 (24 h) | 3258 / 7298 (2139) | new; 20 AHUs, even = train (normal runs), odd = test | seasonal NaN filled 0; 1774 ambiguous windows |
| SMD machine-1-1 / 1-4 / 2-1 / 3-7 | per-step | 38 | 1 min | 60/30 | 948/948 (89), 789/789 (22), 788/788 (34), 955/955 (14) | new (eda_real recipe, per machine) | clip 10 |
| CAN-bus (OBD-II) | none | 8 | 4 s | 6/3 (24 s) | 2581 / - | `canbus_a8_screen.load_canbus_normal` | 19 trips, normal only |
| Paderborn K001-K005 | none (4 labelled operating conditions) | 13 (band powers) | 64 kHz -> 2048-sample frames | 10/5 frames | 9603 / - | `eda_real._raw_paderborn` + `a3_screen_smd_pu._frame_bandpower` | healthy bearings only |
| MIIM_synth (paper PoC) | 6 anomaly types | 24 (pre-windowed 144-d) | - | - | 3119 / 1999 (151) | `miim_unified_seed0.npz` | test treated as one time-ordered stream |
| not evaluated | | | | | | | `_future/Scania` (per-vehicle tabular readouts, not a stream), `_future/ALFA_UAV`, `_future/ESA_ADB` (zipped, not wired), AHU hospital/auditorium (same source as office; office is the largest) |

PCA-10 explained variance: 0.48 (WADI) to 0.86 (CAN-bus); 0.40 to 0.42 on the synthetic controls.

## 2. Metric definitions (byte-identical across datasets; only the window grid is per dataset)

Features: winfeat `stats` (6 per channel) on train-standardised raw streams, feature-standardised on train, PCA-10 fit on
train, full-covariance GMM on train with K in {2, 3, 4, 6, 8, 12, 16} chosen by BIC (K = 16 on the large testbeds, 3 to 9 on
SKAB, SMD, the controls). Per-component NLL s_k(x) = 0.5 (d_k^2 + log det S_k + D log 2 pi), no mixing weight.
Nearest-component head s_NC = min_k s_k. Thresholds q50 and q99 of s_NC over train. home(x) = argmin_k s_k.
R = q99.5 of the train home-Mahalanobis distance (5.0 to 7.5; chi_10 q99.5 is 5.1).

(a) Regime proximity: pairwise D_ij = sqrt(dmu' ((S_i+S_j)/2)^-1 dmu); valley ratio = min mixture density on the segment
mu_i -> mu_j over the lower endpoint density; LDA 5-fold CV error between members. **Close-distinct pair := D_ij < 2R and
valley < 0.5 and LDA error < 0.05.** "Close" means B lies within twice A's normal envelope, i.e. an A-anomaly of modest size
reaches it. Reported: count, train mass in components with such a neighbour, smallest D among valley-separated pairs.

(b) Label-free susceptibility, two versions. Shell: 200 directions per component, probe at Mahalanobis radius R from the
home centre; valid if the home alone flags it (s_h > q99); masked if some valley-separated other component absorbs it
(s_k < q99). Directed corridor: fraction of the segment mu_h -> mu_k where the home flags but k absorbs. The shell version
is nearly blind by construction (a random direction in 10-D rarely points at the one close neighbour: 0.5% on the positive
control), and the corridor version saturates at 1.0 on every dataset (any valley-separated pair has a corridor near k
regardless of distance). Neither is used for ranking; the (a) close-distinct mass is the label-free headline.

(c) Anomaly-masking rate (labelled): episodes = maximal runs of non-normal windows containing a label-1 window (window label
1 if > 50% of rows anomalous, 0 if none, ambiguous otherwise). Expected regime E = majority home of the up-to-5 label-0
windows before the episode (oracle; needs >= 2). For each label-1 window: far := s_E > q99; close_other := min_{k != E}
s_k < q50; missed := s_NC <= q99. **masked := far and close_other** (implies missed). masking_rate = masked / evaluated
anomalous windows. Decomposition of NC misses into masked and subtle (not far from E at all). **Lag-matched control**: for
each test-normal window draw a lag from the anomalous windows' (t - episode start) distribution and use the majority home of
the 5 windows ending that lag earlier; the same far-and-close test gives the regime-switch base rate over the same horizon.
excess = masking_rate - base rate.

(d) Context-recovery gain (labelled): s_ctx(x_t) = s_{E_t}(x_t). Difficult subset = test windows with s_NC <= q99
(NC-normal-looking). gain = AUROC(s_ctx) - AUROC(s_NC) on that subset. Three expectation rules: label-free (majority home of
the previous 5 windows; deployable), oracle (pre-episode regime for anomalies, label-free for normals), lag-matched (oracle for
anomalies, frozen lag-matched expectation for normals, so both classes face an equally stale expectation).

(e) Absorption by a wide component (second failure mode): knn(x) = mean Euclidean distance in PCA-10 to the 10 nearest
train windows (train: leave-one-out); isolated := knn > q99 of train. Heads: NC_BIC (above), WIDE (K = 2 full-cov mixture
NLL), NARROW (diagonal mixture with kd = min(80, max(20, n/10)) components, the paper's recipe in PCA-10).
absorbed(head) := isolated and NLL < q50 train; absorbed99 := isolated and NLL <= q99 (isolated yet not flagged at 1% FPR);
iso-miss = absorbed99 among isolated anomalies; rank gap = Spearman(knn, NLL) on train minus on anomalies.

## 3. Controls (what a null means)

| control | design | (a) close-distinct | (c) masking rate / lag-matched base | (d) gain label-free / oracle / lag-matched | (e) iso-miss NC / WIDE / NARROW |
|---|---|---|---|---|---|
| SYNTH_pos | A, B at raw offset 1.8 sd along an all-channel direction (feature-space D = 6.3, R = 5.0), far C; anomalies = A segments drifting onto B (half) or by the same amount toward empty space (half); dwell 600 to 2400 s | 5 of 9 pairs, mass 1.00 | 0.123 +- 0.011 / 0.111 +- 0.002 | -0.012 / **+0.163** / +0.006 | 0.36 / 0.36 / 0.07 |
| SYNTH_neg | identical train stream; all anomalies drift into empty space | 5 of 9 | **0.000** / 0.111 | -0.052 / **-0.051** / -0.209 | 0.24 / 0.24 / 0.05 |
| SYNTH_pos_slow | as pos with dwell 6000 to 24000 s (switch every ~400 windows), K = 3 found | 1 of 3 (A-B, D = 7.9) | 0.053 / 0.102 | +0.004 / **+0.103** / -0.065 | - |
| SYNTH_ring | ring regime (8 channels oscillating at phases i pi/4, amplitude 3 sd, period 600 s) centred away from A; anomaly = oscillation frozen at the ring centre | - | - | - | **0.019 / 0.997 / 0.006** |

Reading. (c) and the oracle (d) separate positive from negative control cleanly (0.12 vs 0.000; +0.16 vs -0.05). The
positive-control masking rate is exactly what the geometry predicts: with D_AB = 6.3 and R = 5.0, only the last ~22% of an
absorbed drift is both beyond A's envelope and inside B's median contour, and half the episodes are absorbed, giving 0.11.
The lag-matched base rate (0.11) equals it because an absorbed drift, seen as a snapshot against a frozen expectation, is a
regime switch; slowing the switching (pos_slow) does not help because the anomaly episodes scale with the dwell. The
label-free 5-window rule follows the drift and recovers nothing (-0.012), which is a statement about short memory, not about
the data. The ring control passes with margin: 99.6% of frozen-centre windows are kNN-isolated, the wide head misses 99.7% of
them, the K = 9 BIC head 1.9%, the 80-component head 0.6%.

Three pipeline lessons found while building the controls (each is a bug that produced a plausible number first): a fixed
K = 16 over-splits a 3-regime stream into arbitrary slices with spurious valleys (45 of 120 sibling pairs), hence K by BIC;
a regime offset confined to one channel among 144 unit-variance features is invisible to PCA-10 (true A-B separation
collapsed to 2.0), hence the all-channel offsets; and windows straddling regime switches inflate within-regime covariance
along the between-regime axis (A-C separation 10.4 instead of 20 at dwell 300 s), hence the longer dwells.

## 4. Ranked table, all datasets

Geometry (label-free) and masking (labelled). Mean +- sd over 3 GMM seeds.

| dataset | labels | K | close-distinct pairs / all | mass in close-distinct comps | min D distinct | masking rate (oracle E) | lag-matched base rate | excess | NC miss @q99 | masked / subtle share of misses | ctx gain difficult: label-free / oracle / lag-matched |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SYNTH_pos (positive) | Y | 4.7 | 5.0 / 9 | 1.00 | 6.30 | 0.123 +- 0.011 | 0.111 | +0.012 | 0.650 | 0.19 / 0.43 | -0.012 / +0.163 / +0.006 |
| SYNTH_neg (negative) | Y | 4.7 | 5.0 / 9 | 1.00 | 6.30 | 0.000 | 0.111 | -0.111 | 0.328 | 0.00 / 0.93 | -0.052 / -0.051 / -0.209 |
| WindSCADA | Y | 16 | 57.0 / 120 | 1.00 | 2.70 | **0.201 +- 0.004** | **0.333** | -0.132 | 0.910 | 0.22 / 0.12 | -0.085 / +0.097 / -0.137 |
| MIIM_synth (paper PoC) | Y | 16 | 64.3 / 120 | 1.00 | 3.15 | 0.042 +- 0.006 | 0.272 | -0.230 | 0.779 | 0.05 / 0.21 | -0.196 / -0.113 / -0.200 |
| WADI | Y | 14.7 | 72.3 / 102 | 1.00 | 2.82 | 0.010 +- 0.015 | 0.100 | -0.089 | 0.656 | 0.02 / 0.41 | -0.120 / -0.058 / -0.160 |
| SWaT | Y | 16 | 33.7 / 120 | 0.99 | 3.89 | 0.008 +- 0.003 | 0.117 | -0.109 | 0.184 | 0.04 / 0.20 | -0.189 / -0.180 / -0.189 |
| SKAB | Y | 5.3 | 0.0 / 12 | 0.00 | 4.32 | 0.002 +- 0.001 | 0.010 | -0.008 | 0.758 | 0.00 / 0.94 | -0.007 / +0.007 / -0.015 |
| HAI | Y | 16 | 35.3 / 120 | 1.00 | 2.37 | 0.000 | 0.035 | -0.035 | 0.122 | 0.00 / 0.64 | +0.005 / +0.041 / -0.071 |
| Cranfield | Y | 16 | 45.7 / 120 | 1.00 | 2.19 | 0.000 | 0.005 | -0.005 | 0.284 | 0.00 / 0.62 | -0.051 / -0.024 / -0.093 |
| AHU_office | Y | 16 | 24.7 / 120 | 0.99 | 3.20 | 0.000 | 0.201 | -0.201 | 0.405 | 0.00 / 0.27 | -0.182 / -0.065 / -0.205 |
| MetroPT | Y | 13.3 | 21.0 / 84 | 0.97 | 2.54 | 0.000 | 0.037 | -0.037 | 0.000 | - | - (no NC misses) |
| SMD-1-1 | Y | 7.3 | 11.0 / 24 | 0.99 | 2.56 | 0.000 | 0.072 | -0.072 | 0.019 | 0.00 / 0.00 | +0.015 / +0.100 / -0.004 |
| SMD-1-4 | Y | 9.3 | 10.7 / 41 | 0.96 | 2.68 | 0.000 | 0.000 | 0.000 | 0.091 | 0.00 / 0.13 | -0.058 / +0.066 / +0.049 |
| SMD-2-1 | Y | 8.0 | 12.0 / 28 | 0.98 | 2.86 | 0.000 | 0.011 | -0.011 | 0.147 | 0.00 / 0.93 | -0.053 / -0.074 / -0.186 |
| SMD-3-7 | Y | 9.3 | 14.3 / 41 | 0.98 | 2.74 | 0.000 | 0.018 | -0.018 | 0.000 | - | - |
| CAN-bus | N | 9.3 | 24.3 / 41 | 1.00 | 2.56 | - | - | - | - | - | - |
| Paderborn | N | 10.7 | 10.7 / 53 | 0.94 | 4.31 | - | - | - | - | - | - |

Absorption by a wide component (second failure mode; kNN in the same PCA-10 space).

| dataset | isolated anomalies (knn > q99) | isolated test-normals | iso-miss @q99: NC_BIC / WIDE / NARROW | absorbed99 (share of all anomalies): NC / WIDE / NARROW | overall miss @q99: NC / WIDE / NARROW | rank gap NC / WIDE / NARROW |
|---|---|---|---|---|---|---|
| SYNTH_ring (positive) | 0.996 | 0.005 | 0.019 / **0.997** / 0.006 | 0.019 / 0.993 / 0.006 | 0.022 / 0.996 / 0.009 | 0.42 / 0.36 / 0.35 |
| WADI | 0.156 | 0.008 | **0.27 / 0.80 / 0.13** | 0.042 / 0.125 / 0.021 | 0.656 / 0.906 / 0.635 | 0.05 / -0.02 / 0.09 |
| AHU_office | 0.387 | 0.023 | 0.09 / **0.48** / 0.03 | 0.035 / 0.186 / 0.010 | 0.390 / 0.795 / 0.362 | 0.12 / 0.01 / 0.03 |
| MIIM_synth | 0.192 | 0.019 | 0.26 / **0.71** / 0.07 | 0.051 / 0.137 / 0.013 | 0.779 / 0.898 / 0.678 | 0.10 / 0.24 / 0.05 |
| WindSCADA | 0.066 | 0.010 | 0.23 / 0.39 / 0.08 | 0.015 / 0.026 / 0.005 | 0.910 / 0.934 / 0.901 | 0.02 / -0.16 / -0.01 |
| Cranfield | 0.322 | 0.231 | 0.015 / 0.28 / 0.000 | 0.005 / 0.092 / 0.000 | 0.409 / 0.764 / 0.397 | 0.00 / -0.21 / -0.01 |
| MetroPT | 0.086 | 0.194 | 0.00 / 0.25 +- 0.27 / 0.00 | 0.000 / 0.022 / 0.000 | 0.000 / 0.39 +- 0.42 / 0.000 | 0.55 +- 0.48 / 0.20 / 0.03 |
| SMD-2-1 | 0.500 | 0.035 | 0.00 / 0.12 / 0.00 | 0.000 / 0.059 / 0.000 | 0.147 / 0.412 / 0.010 | -0.16 / -0.10 / -0.37 |
| SMD-3-7 | 0.571 | 0.009 | 0.00 / 0.08 / 0.00 | 0.000 / 0.048 / 0.000 | 0.000 / 0.262 / 0.000 | -0.30 / -0.16 / -0.35 |
| SWaT | 0.753 | 0.001 | 0.000 / 0.06 / 0.005 | 0.000 / 0.046 / 0.004 | 0.184 / 0.284 / 0.188 | 0.31 / 0.12 / 0.08 |
| HAI | 0.883 | 0.118 | 0.03 / 0.01 / 0.03 | 0.026 / 0.010 / 0.025 | 0.122 / 0.117 / 0.132 | -0.20 / -0.35 / -0.12 |
| SMD-1-1 | 0.966 | 0.126 | 0.004 / 0.03 / 0.000 | 0.004 / 0.030 / 0.000 | 0.019 / 0.060 / 0.000 | 0.10 / -0.23 / -0.02 |
| SKAB | 0.182 | 0.003 | 0.014 / 0.017 / 0.000 | 0.003 / 0.003 / 0.000 | 0.758 / 0.775 / 0.710 | -0.05 / -0.01 / -0.11 |
| SMD-1-4 | 0.182 | 0.000 | 0.00 / 0.00 / 0.00 | 0.000 / 0.000 / 0.000 | 0.091 / 0.636 / 0.000 | -0.06 / -0.19 / -0.21 |

absorbed-at-q50 (isolated and scored below the train median) is 0.000 for every real dataset and every head; the only
non-zero values are the ring control under the wide head (0.30 +- 0.21). Isolated anomalies are never scored as
"typical"; when they are missed, they sit between q50 and q99.

## 5. What the numbers say

**Geometry.** Close-distinct pairs are the rule, not the exception: on WADI 72 of 102 pairs, Cranfield 46 of 120, WindSCADA
57 of 120. The nearest valley-separated pairs sit at D = 2.2 to 2.9 (WADI, HAI, Cranfield, WindSCADA, SMD, CAN-bus), well
inside 2R = 10 to 15. This agrees with `a8_remeasure.md` (nearest-pair LDA error 8 to 11% mean on discovered clusters, deep
valleys on Cranfield setpoints) and is the A8 precondition under the correct formulation. SKAB is the exception (BIC picks
K = 5, no pair passes; single-loop rig). Paderborn's four labelled operating conditions confirm the picture with physical
labels: the 900 rpm condition is 3.9 to 4.4 units from each 1500 rpm condition (LDA error 0.5 to 1.3%), while the three
1500 rpm conditions are 0.4 to 1.2 apart and not separable (LDA error 30 to 43%; load torque and radial force barely move
the band-power signature).

**Masking rate.** On every real dataset the oracle masking rate is at or below its lag-matched base rate; the excess is
negative everywhere (-0.005 on Cranfield to -0.23 on MIIM). Where the base rate is small enough for the metric to have
power (Cranfield 0.005, SKAB 0.010, SMD-1-4 0.000, SMD-2-1 0.011, SMD-3-7 0.018, HAI 0.035), the masking rate is 0.000 to
0.002 with 14 to 2284 anomalous windows evaluated. Nearest-component misses on those datasets are "subtle" (not far from the
expected regime at all): 62 to 94% of misses on Cranfield, HAI, SKAB, SMD-2-1. WADI's one masked window (t = 508, expected
component 5 at NLL 36.8, absorbed by component 8 at 20.4) is 1 of 32 and appears in one seed of three. SWaT's masked
window (t = 763, s_E = 220, absorbed at 4.4) is a plant-state jump of the kind SWaT's tank cycling produces 12% of the time
in normal data.

**WindSCADA inspected.** The 8 example windows (t = 2201 to 2212, stream 0, expected component 9, absorbers 0, 5, 2, 15)
show the turbine going from generating (Gen_RPM +0.5 z, ActPwr +0.4 z over the previous 5 windows) to stopped
(Gen_RPM -1.7 z, WindSpeed -1.2 z, ActPwr -1.1 z) with label fraction 0.83 to 1.00. That is the low-wind idle regime,
which the normal stream also enters 33% of the time over the same 34-window (68 h) horizon. The label is "within 72 h of a
failure"; nothing in the window distinguishes it from a calm day. WindSCADA is not a demonstrator.

**MIIM synthetic by anomaly type (seed 0).** drift: 7 of 51 masked (14%), NC missed 46 of 51; bad_transition 0 of 31
(missed 28); pocket 0 of 16; wrong_for_regime 0 of 15; near_boundary 0 of 18; ood 0 of 20. The masking pattern appears only
for drifts, the same mechanism as the positive control, and the MIIM test stream switches modes so often (base rate 0.27)
that it is not separable from switching at the snapshot level either.

**Context recovery.** The deployable label-free rule never helps (gain -0.20 to +0.02 on the difficult subset). The oracle
rule helps on the positive controls (+0.16, +0.10) and on WindSCADA (+0.10), SMD-1-1 (+0.10 +- 0.07), SMD-1-4 (+0.07),
HAI (+0.04); but once normal windows get an equally stale expectation (lag-matched), only SMD-1-4 stays positive (+0.05,
22 anomalous windows) and the controls drop to +0.006 / -0.065. The oracle gain on real data is the same switching artefact
as the oracle masking rate.

**Absorption.** The wide K = 2 head misses 80% of WADI's isolated anomalies, 71% of MIIM's, 48% of AHU's, 39% of
WindSCADA's, 28% of Cranfield's; the BIC head misses 27 / 26 / 9 / 23 / 1.5%; the 80-component diagonal head 13 / 7 / 3 /
7.5 / 0%. On the testbeds where anomalies are strongly isolated (HAI 88%, SWaT 75%, SMD-1-1 97%) every head catches them
(iso-miss 0 to 3%). The rank gap is positive under the wide head on MIIM (0.24), MetroPT (0.20), SWaT (0.12): the wide head
ranks anomalies less like a local-density measure would than it ranks train data. MetroPT's wide head is unstable across
seeds (miss 0.39 +- 0.42), a two-component fit on a plant with 13 BIC components.

## 6. Which metric captures A8-masking, and what to say in the paper

Recommended for the paper, in this order:

1. **Close-distinct pair mass** (label-free, (a)): the share of train mass in a discovered regime that has a valley-separated
   neighbour within twice its own q99.5 envelope. It states the precondition directly, is 0.94 to 1.00 on every multi-loop
   plant and 0.00 on SKAB, and needs no anomaly labels. Report with the smallest distinct-pair D.
2. **Masking rate with its lag-matched base rate** (labelled, (c)): keep both numbers side by side; the metric passes the
   positive/negative control pair on the rate (0.12 vs 0.000) and the base rate is what makes a null on real data mean
   "absent" rather than "unmeasurable". State plainly that where switching is rare the rate is 0.000 to 0.002.
3. Do not use the shell susceptibility, the directed corridor, or any expectation-based AUROC gain as A8 evidence: the first
   is blind (0.5% on the positive control), the second saturates (1.0 everywhere), and the gains are switching artefacts
   unless lag-matched, after which they are null on real data.
4. **Isolated-and-missed rate by head resolution** (e) is a separate, well-behaved metric for the over-smoothing failure
   mode and directly motivates the 80-component density head: it turns 28 to 80% wide-head misses of isolated anomalies
   into 0 to 13%.

Substance for the paper (no edits made): the A8 precondition, adjacent yet distinct regimes, holds on every multi-loop
dataset examined; the A8 failure mode, an anomaly of one regime absorbed by an adjacent regime, is not observed above the
regime-switch base rate on any of the 14 labelled real streams, and on the controls it is only identifiable through the
path, i.e. it belongs with A10. The over-smoothing failure mode is observed and is resolved by the density head's
resolution.

## 7. Cross-check with the parallel deep A8-masking agent (`a8_masking.{WADI_clean,HAI,SWaT_canon}.jsonl`)

Their comparable configuration (`pca10_gmm16`, previous-regime rule, q95 thresholds, window label > 5% attack) gives an
anomaly A8-pattern rate of 0.054 on WADI against a normal-window rate of 0.050, 0.009 vs 0.007 on HAI, 0.061 vs 0.157 on
SWaT; in the VaDE latent the anomaly rates are 0.000 to 0.054 (WADI), 0.000 (HAI), 0.038 to 0.053 (SWaT), never above the
normal rate. Mine: 0.010 / 0.000 / 0.008 against lag-matched base rates 0.100 / 0.035 / 0.117. The two sweeps use
different window labels (theirs 56 WADI anomalies, mine 32), thresholds (q95 vs q99) and expectation rules, and agree on the
conclusion: no excess over the base rate on WADI, HAI or SWaT. No disagreement to flag. Their obs_sweep rows also show the
K, seed and rule sensitivity of the WADI base rate (0.03 to 0.09), consistent with my seed spread (0.100 +- 0.013).

## 8. Caveats

- Three GMM seeds per dataset; PCA and windows are deterministic. Seed spread is visible where the partition matters
  (WADI close-distinct pairs 72 +- 17, MetroPT wide-head numbers).
- Small anomaly counts: WADI 32 windows / 8 episodes, SMD-1-4 22 / 8, SMD-3-7 14 / 3, MetroPT 93 / 3. A masking rate of
  0.000 on 14 windows bounds nothing below ~7%.
- Labels are weak on WindSCADA (72 h pre-failure), MetroPT (maintenance segments) and AHU (rule-derived FDD, 1774 ambiguous
  windows excluded); Cranfield cases 1 to 3 are faulty throughout, so their episodes start at the file start and have no
  pre-episode context (2284 of 7747 anomalous windows evaluated).
- Everything is in PCA-10 of the stats features with a GMM head, not the VaDE latent with the paper's head; the parallel
  agent's VaDE-latent numbers on WADI/HAI/SWaT agree in sign and magnitude.
- The labelled masking metric has no power when regime switching over the anomaly horizon is frequent (WindSCADA, AHU,
  SWaT, WADI, MIIM base rates 0.10 to 0.33); the null on those datasets is "not above base rate", the null on Cranfield,
  HAI, SKAB and SMD is a near-zero rate with a near-zero base rate.
- The AHU loader fills seasonal NaN channels with 0, which makes heating and cooling seasons separate regimes by
  construction; its close-distinct pairs partly reflect that.
- Agent execution: about 2 h 20 min of tool calls including four rounds of control debugging; local CPU compute about 12
  minutes total (each full sweep 3 to 5 min in three parallel processes); no cloud cost.
