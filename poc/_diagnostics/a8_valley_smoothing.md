# A8 valley smoothing: does a deep reconstructor miss anomalies that land BETWEEN two close regimes, and does latent density catch them?

Scripts (all in `_diagnostics/`, CPU only, re-runnable, resumable): `a8_valley_real.py` (partition + per-partition
detection + interaction + miss enrichment; rows in `a8_valley_real.jsonl`, per-window tables in
`a8_valley_windows_<ds>_seed<s>.csv`, logs `a8_valley_real.*.log`), `a8_valley_summ.py` (tables, `a8_valley_summ.log`),
`a8_valley_inspect.log` (per-window listing of every valley anomaly), `a8_valley_troughs.py` (valley depth of the real
regime geometry, `a8_valley_troughs.json/.log`), `a8_valley_synth.py` (linear-manifold two-regime control,
`a8_valley_synth.jsonl`, `.png`), `a8_valley_synth_curved.py` (curved-manifold control that separates interpolation
from extrapolation, `a8_valley_synth_curved.jsonl`, `.png`). Figures: `a8_valley_real.png`, `a8_valley_synth.png`,
`a8_valley_synth_curved.png`. Paper and shipped model untouched.

## Hypothesis under test

Two normal regimes A and B sit close in value space. A deep reconstructor (USAD, TranAD) learns a decision surface
that treats the whole span between the two regime centres as reconstructable. An anomaly that is an extreme value of A
landing in the inter-regime valley reconstructs like normal and is missed; a latent-density head scores the valley as
improbable and catches it. Prediction: the deep reconstructors underperform specifically on valley anomalies, and
LatAD's edge over them is concentrated there.

## Operational definition (label-free geometry, independent of every detector)

Window features (winfeat `stats`, 6 per channel, W = 60, stride 30: the detectors' own window grid), feature-standardised
on train, PCA-10 fit on train, full-covariance GMM on train with K by BIC in {2, 3, 4, 6, 8, 12, 16}, three GMM seeds.
For each test window: d_k = Mahalanobis distance to component k; h = nearest, j = second nearest; R = q99.5 of the
train nearest-component distance (the normal envelope; 7.3 to 7.6 on SWaT, 5.4 on WADI, 4.3 to 4.5 on HAI). In the
pooled-covariance whitened frame of (h, j), with axis v = mu_j - mu_h and u = x - mu_h: t = <u, v>/|v|^2 (0 at h, 1 at
j), perp = |u - t v|.

| partition | rule | meaning |
|---|---|---|
| in_cluster | d_h <= R | inside the envelope of some regime |
| valley | d_h > R, 0 < t < 1, perp <= R | between the two nearest centres, inside the inter-regime tube |
| beyond | d_h > R, (t <= 0 or t >= 1), perp <= R | on the axis but past a centre (an extreme away from the other regime) |
| off | d_h > R, perp > R | leaves the inter-regime tube: off-manifold |

`valley_distinct` additionally requires the (h, j) pair to be density-valley-separated (mixture density along the
segment dips below 0.5 of the lower endpoint); on these three datasets it is the same set as `valley` to within one
window. A looser variant (out of envelope and 0 < t < 1, no tube gate) and a continuous "betweenness" score
min(t, 1 - t) were also evaluated for sensitivity.

Invariants stated before running: (I1) in-cluster anomalies near chance for every detector; (I2) off-manifold anomalies
with large d_h caught by every detector; (I3) test normals mostly in_cluster; (I4) window count equals the scores npz
and window labels agree with the npz label vector; (I5) permuting partition labels among anomalies gives a null
interaction centred at 0.

Checks: (I4) 1498 / 575 / 14819 test windows, label agreement 0.995 / 0.995 / 1.000 (the npz label is authoritative).
(I3) normals in_cluster 0.99 (SWaT), 0.98 (WADI), 0.87 (HAI; HAI test drifts, 6.7% of its normals are off-manifold by
this geometry). (I1) in-cluster anomaly AUROC 0.55 to 0.81 with recall@1% 0.00 to 0.44. (I2) off anomalies recall@1%
0.97 (SWaT), 0.87 (WADI), 0.79 to 0.92 (HAI) for every deep head. (I5) permutation null sd 0.003 to 0.056, mean 0.

## 1. How many anomalies are valley anomalies?

Seed-mean share of labelled anomalous windows (3 GMM seeds; counts per seed in brackets):

| dataset | n anomalies | in_cluster | **valley** | beyond | off | normals in valley |
|---|---|---|---|---|---|---|
| SWaT | 233 | 0.31 | **0.049** [11, 8, 15] | 0.019 | 0.62 | 0.005 (2 to 9 windows) |
| WADI | 56 | 0.82 | **0.089** [8, 4, 3] | 0.024 | 0.066 | 0.008 (3 to 6) |
| HAI | 652 | 0.32 | **0.005** [2, 4, 3] | 0.018 | 0.66 | 0.014 (180 to 237) |

Valley anomalies are rare: 5% of SWaT attacks, 9% of WADI attacks (3 to 8 windows) and 0.5% of HAI attacks (2 to 4
windows). Two thirds of SWaT and HAI attacks are off-manifold; four fifths of WADI attacks are inside a regime envelope.
The loose definition (no tube gate) raises the valley share to 12 to 21% of out-of-envelope anomalies on SWaT and 18 to
25% on HAI, and does not change any conclusion below (median LatAD / USAD / TranAD percentiles 0.998 / 0.998 / 0.998 on
the loose valley set vs 0.998 / 0.999 / 0.999 on the rest, `a8_valley_inspect.log`).

Why so rare: the valleys of the real regime geometry are shallow (`a8_valley_troughs.log`). Over the close regime pairs
(D < 2R, density-valley-separated; 30 to 95 pairs per dataset), the floor of the valley along the centre-to-centre
segment sits at the train log-density percentile 0.6 to 0.8 (median) on SWaT, 11 to 12 on WADI and 6 to 36 on HAI, and
the floor lies within R of its nearest component on 96 to 100% of WADI and HAI pairs (median nearest-component
Mahalanobis distance at the floor 1.8 to 3.4). On WADI and HAI the space between two close regimes is inside the normal
envelope: no density head fitted to this geometry can flag it at 1% FPR, so an "extreme of A landing between A and B" is
an in-cluster point by construction. Only SWaT has valleys deep enough (26 to 30% of floors outside R) to host valley
anomalies, and it is the dataset with the most of them.

## 2. Per-partition detection (AUROC of the partition's anomalies vs all test normals / recall at 1% FPR; seed mean)

| dataset | partition | n | LatAD | USAD | TranAD | AE | linres |
|---|---|---|---|---|---|---|---|
| SWaT | in_cluster | 71-73 | 0.78 / 0.06 | 0.60 / 0.07 | 0.60 / 0.09 | 0.68 / 0.08 | 0.76 / 0.22 |
| SWaT | **valley** | 8-15 | 0.99 / 0.50 | 0.95 / 0.66 | 0.96 / 0.75 | 0.98 / 0.71 | 0.94 / 0.74 |
| SWaT | beyond | 4-5 | 0.99 / 0.33 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.93 | 0.99 / 0.83 |
| SWaT | off | 141-150 | 1.00 / 0.97 | 1.00 / 0.97 | 1.00 / 0.97 | 1.00 / 1.00 | 1.00 / 0.97 |
| WADI | in_cluster | 45-48 | 0.67 / 0.00 | 0.71 / 0.25 | 0.75 / 0.25 | 0.75 / 0.29 | 0.80 / 0.44 |
| WADI | **valley** | 3-8 | 0.95 / 0.28 | 0.96 / 0.85 | 0.97 / 0.85 | 0.99 / 0.85 | 1.00 / 0.83 |
| WADI | beyond | 1-2 | 0.92 / 0.17 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| WADI | off | 2-5 | 0.96 / 0.40 | 0.99 / 0.87 | 0.99 / 0.87 | 0.99 / 0.87 | 1.00 / 0.93 |
| HAI | in_cluster | 205-211 | 0.81 / 0.09 | 0.61 / 0.13 | 0.57 / 0.13 | 0.77 / 0.13 | 0.68 / 0.11 |
| HAI | **valley** | 2-4 | 0.91 / 0.25 | 0.75 / 0.25 | 0.77 / 0.25 | 0.91 / 0.25 | 0.74 / 0.11 |
| HAI | beyond | 7-15 | 0.93 / 0.53 | 0.66 / 0.14 | 0.68 / 0.14 | 0.95 / 0.53 | 0.69 / 0.07 |
| HAI | off | 426-432 | 0.99 / 0.89 | 0.97 / 0.81 | 0.96 / 0.79 | 1.00 / 0.92 | 0.83 / 0.48 |

Reading. On SWaT and WADI the valley anomalies are caught by every head: USAD and TranAD score them at AUROC 0.95 to
0.97 and their recall at 1% FPR on valley anomalies (0.66 to 0.85) is higher than LatAD's (0.28 to 0.50). Per window
(`a8_valley_inspect.log`), 10 of SWaT's 11 seed-0 valley windows sit above the 0.90 normal percentile under USAD and
TranAD; the one SWaT window a deep head misses (t = 759, USAD 0.34, TranAD 0.37, LatAD 0.94) appears in one seed of
three. On WADI the seed-0 valley set has USAD / TranAD percentiles 0.99 to 1.00 on 7 of 8 windows while LatAD sits at
0.79 to 0.99. On HAI the valley set is 2 to 4 windows; LatAD leads USAD / TranAD there (0.91 vs 0.75 / 0.77), but it
leads by the same or a larger margin on the `beyond` partition (0.93 vs 0.66 / 0.68, n = 7 to 15) and on the in-cluster
partition (0.81 vs 0.61 / 0.57, n = 205 to 211), and the plain AE baseline matches LatAD on every HAI partition (0.91 /
0.95 / 0.77 / 1.00). HAI's LatAD advantage is a near-manifold advantage over USAD and TranAD specifically, not a valley
advantage over reconstruction.

Interaction statistic (LatAD - deep)[valley] - (LatAD - deep)[off], bootstrap 95% CI, permutation p:

| dataset | seed | n valley / off | vs USAD | vs TranAD | vs AE |
|---|---|---|---|---|---|
| SWaT | 0 | 11 / 144 | +0.031 [+0.002, +0.065] p = 0.000 | +0.016 [-0.002, +0.036] p = 0.006 | +0.021 [-0.004, +0.059] p = 0.008 |
| SWaT | 1 | 8 / 150 | +0.072 [-0.006, +0.225] p = 0.056 | +0.068 [-0.008, +0.212] p = 0.042 | +0.007 [-0.008, +0.032] p = 0.060 |
| SWaT | 2 | 15 / 141 | +0.003 [-0.004, +0.014] p = 0.190 | +0.004 [-0.005, +0.019] p = 0.156 | -0.003 [-0.006, -0.000] p = 0.000 |
| WADI | 1 | 4 / 5 | -0.009 [-0.074, +0.059] p = 0.84 | -0.008 [-0.072, +0.059] p = 0.86 | -0.003 p = 0.96 |
| WADI | 2 | 3 / 4 | +0.058 [+0.004, +0.120] p = 0.27 | +0.057 [+0.002, +0.116] p = 0.28 | +0.055 p = 0.28 |
| HAI | 1 | 4 / 428 | +0.270 [+0.051, +0.483] p = 0.002 | +0.251 [+0.073, +0.388] p = 0.000 | +0.018 [+0.001, +0.036] p = 0.006 |
| HAI | 2 | 3 / 432 | +0.157 [-0.068, +0.372] p = 0.018 | +0.125 [-0.041, +0.395] p = 0.052 | +0.011 [-0.001, +0.032] p = 0.072 |

(WADI seed 0 and HAI seed 0 have too few off or valley windows.) The SWaT interactions are +0.003 to +0.07 with the
sign and significance changing across GMM seeds; WADI is null; HAI's +0.13 to +0.27 rests on 3 to 4 windows, and the
same three or four windows give an interaction of only +0.01 to +0.02 against the plain AE. The continuous version
(Spearman between betweenness min(t, 1 - t) and LatAD percentile minus deep percentile over out-of-envelope anomalies)
is +0.06 to +0.22 on SWaT (significant in one seed of three), -0.14 to +0.57 on WADI (n = 8 to 11, none significant), and
negative on HAI (-0.04 to -0.15, significant at p = 0.002 in seed 2: the MORE between two regimes an HAI anomaly sits,
the SMALLER LatAD's edge over the deep heads).

## 3. Miss enrichment: are the windows USAD/TranAD miss but LatAD catches valley anomalies?

G = anomalies with deep score <= q99 of test-normal and LatAD score > q99 of test-normal.

| dataset | deep | \|G\| | valley share in G / outside G / all anomalies | off share in G / outside | in_cluster share in G / outside |
|---|---|---|---|---|---|
| SWaT | USAD | 2 | 0.00 / 0.05 / 0.05 | 0.50 / 0.62 | 0.50 / 0.31 |
| SWaT | TranAD | 2 | 0.00 / 0.05 / 0.05 | 0.50 / 0.62 | 0.50 / 0.31 |
| WADI | USAD, TranAD | 0 | (empty; the reverse set, deep catches and LatAD misses, has 17 windows) | | |
| HAI | USAD | 76 | **0.000** / 0.003 to 0.007 / 0.003 to 0.006 | **0.79 to 0.84 / 0.64** (Fisher p <= 0.007) | 0.09 to 0.15 / 0.35 (p < 0.001) |
| HAI | TranAD | 83 | **0.000** / 0.004 to 0.007 | **0.81 to 0.86 / 0.63** (p <= 0.001) | 0.08 to 0.13 / 0.35 (p < 0.001) |
| HAI | AE | 2 | 0.00 | 0.00 / 0.66 | 1.00 / 0.32 |

The only dataset with a sizeable "deep misses, LatAD catches" set is HAI (76 to 83 windows against USAD / TranAD), and
that set contains zero valley anomalies in every seed. It is enriched in OFF-manifold anomalies (79 to 86% vs 63 to 64%
of the other anomalies) and depleted in in-cluster ones. Within the off-manifold class, the G windows are the
smaller-magnitude ones (nearest-component Mahalanobis median 9.2 vs 15.6 for the off-manifold anomalies USAD also
catches; perpendicular distance 10.4 vs 15.8): USAD and TranAD miss moderate off-manifold departures on HAI, which
LatAD and the plain AE both catch (AE's G set is 2 windows). On SWaT G is 2 windows and on WADI it is empty; on WADI the
reverse set (USAD / TranAD catch, LatAD misses) is 17 windows.

## 4. Synthetic controls

### 4a. Linear-manifold two-regime world (`a8_valley_synth.py`)

D = 20, 4-dim linear manifold, regimes A and B separated by `gap` (within-regime sd units) along one manifold axis,
isotropic noise 0.05. Anomalies from A points, all of size gap/2: `valley` toward B (lands at the midpoint), `beyond`
away from B, `off` orthogonal to the manifold. MLP autoencoder (latent 4) reconstruction error; GMM in the AE latent
(K by BIC); PCA-4 linear reconstruction. Seed-mean AUROC vs test normal (3 seeds, 1500 normals, 400 anomalies each):

| gap | recon valley / beyond / off | density valley / beyond / off | linear recon valley / beyond |
|---|---|---|---|
| single regime (size 2.5) | 0.54 / 0.52 / 1.00 | 0.85 / 0.84 / 0.61 | 0.51 / 0.50 |
| 3 | 0.51 / 0.50 / 1.00 | 0.47 / 0.68 / 0.50 | 0.51 / 0.50 |
| 4 | 0.51 / 0.50 / 1.00 | 0.57 / 0.75 / 0.52 | 0.50 / 0.50 |
| 5 | 0.52 / 0.50 / 1.00 | 0.71 / 0.83 / 0.56 | 0.50 / 0.49 |
| 6 | 0.53 / 0.50 / 1.00 | 0.83 / 0.90 / 0.62 | 0.50 / 0.49 |
| 8 | 0.60 / 0.50 / 1.00 | 0.96 / 0.98 / 0.76 | 0.50 / 0.49 |
| 12 | 0.81 / 0.53 / 1.00 | 1.00 / 1.00 / 0.93 | 0.50 / 0.49 |

Invariant S1 holds (single regime: valley = beyond for both heads). Invariant S2 holds (off caught at 1.00 everywhere).
The reconstructor misses valley anomalies (0.51 to 0.60 for gaps 3 to 8), but it misses `beyond` anomalies of the same
size just as completely (0.50), so in a linear world the miss is not a valley effect: any in-manifold displacement is
reconstructable by a decoder whose image is the subspace. The density head's valley score is BELOW its beyond score
at every gap (0.47 vs 0.68 at gap 3, 0.83 vs 0.90 at gap 6): the midpoint between two regimes is a denser place than
the same distance outward, and for gaps up to 5 (midpoint within 2.5 sd of each centre) the density head is at or
near chance on the valley. Only at gap 12 does the reconstructor start to see the valley (0.81), because the AE
contracts the empty middle toward the modes instead of interpolating; this is the opposite of smoothing.

### 4b. Curved-manifold world (`a8_valley_synth_curved.py`)

The linear world cannot separate interpolation (valley) from extrapolation (beyond) for a reconstructor. Here the
normal manifold is curved: x = U_in z + kappa U_q q(z)/gap_ref + noise, with q(z) the 10 quadratic monomials of the 4
manifold coordinates in a frame orthogonal to U_in (D = 24). Regimes A, B at z1 = 0 and z1 = gap on the surface.
Anomalies from A points, all of size gap/2 in z: `valley` ON the surface at z1 = gap/2 (an extreme of A drifting toward
B along the physical manifold: reachable by a decoder that interpolates between the regimes), `beyond` ON the surface
at z1 = -gap/2 (reachable only by extrapolating the curve), `chord` the straight-line observation-space midpoint between
the A point and its B image (off the surface by the sagitta: what a decoder that interpolates linearly in x would
produce), `off` orthogonal to both frames. kappa = 0 reproduces the linear world. Seed-mean AUROC (3 seeds):

| kappa | gap | recon valley / beyond / chord / off | density valley / beyond / chord / off | linear recon valley / beyond / chord |
|---|---|---|---|---|
| 0 | 4 | 0.51 / 0.51 / 0.50 / 1.00 | 0.60 / 0.75 / 0.57 / 0.53 | 0.51 / 0.51 / 0.50 |
| 0 | 8 | 0.59 / 0.53 / 0.58 / 1.00 | 0.96 / 0.98 / 0.96 / 0.70 | 0.50 / 0.51 / 0.49 |
| 0.5 | 4 | 0.52 / 0.58 / 1.00 / 1.00 | 0.51 / 0.77 / 0.50 / 0.51 | 0.49 / 0.87 / 0.23 |
| 0.5 | 6 | **0.58 / 0.70** / 1.00 / 1.00 | 0.75 / 0.92 / 0.71 / 0.55 | 0.66 / 0.97 / 0.14 |
| 0.5 | 8 | **0.70 / 0.85** / 1.00 / 1.00 | 0.91 / 0.98 / 0.87 / 0.63 | 0.83 / 1.00 / 0.10 |
| 0.5 | 12 | 0.97 / 0.99 / 1.00 / 1.00 | 0.99 / 1.00 / 0.98 / 0.79 | 0.97 / 1.00 / 0.07 |
| 1.0 | 4 | 0.55 / 0.63 / 1.00 / 1.00 | 0.49 / 0.76 / 0.48 / 0.47 | 0.47 / 0.89 / 0.27 |
| 1.0 | 6 | **0.68 / 0.76** / 1.00 / 1.00 | 0.69 / 0.91 / 0.63 / 0.52 | 0.60 / 0.97 / 0.18 |
| 1.0 | 8 | **0.84 / 0.90** / 1.00 / 1.00 | 0.89 / 0.98 / 0.80 / 0.61 | 0.73 / 0.99 / 0.15 |
| 1.0 | 12 | 0.99 / 1.00 / 1.00 / 1.00 | 0.98 / 1.00 / 0.94 / 0.79 | 0.75 / 1.00 / 0.22 |

Invariants: (C1) holds (kappa = 0 gives valley = beyond for recon); (C2) holds (off at 1.00 everywhere). (C3), the
mechanism itself, holds only in a weak form: with curvature the AE reconstructs the on-manifold valley point better than
the on-manifold beyond point, so recon AUROC(valley) is below recon AUROC(beyond) by 0.06 to 0.15 at gaps 6 and 8, and the
gap closes at gap 12 (0.97 vs 0.99). But three things break the story as a reconstruction-vs-density contrast:

1. The `chord` anomaly (the straight-line interpolation in observation space) is caught by the AE at 1.00 in every
   curved setting. The decoder does not smooth linearly across the valley; it follows the manifold. The reconstruction
   blind spot is "anomalies on the manifold", and the valley is only the part of the manifold that is easiest to reach.
2. The density head shows the same valley-vs-beyond asymmetry (0.75 vs 0.92, 0.91 vs 0.98, 0.69 vs 0.91, 0.89 vs 0.98):
   the space between two regimes is denser than the same distance outward, so the valley is harder for density too.
   The interaction (density - recon)[valley] - (density - recon)[beyond] is -0.05, +0.08, -0.13, -0.03 at (kappa, gap) =
   (0.5, 6), (0.5, 8), (1.0, 6), (1.0, 8): no consistent sign. Density's advantage over reconstruction is a manifold
   advantage, not a valley advantage.
3. At gap 4 (centres 4 sd apart, the anomaly 2 sd from A) neither head sees the valley (recon 0.52 to 0.55, density 0.49
   to 0.51); the midpoint is inside both envelopes, exactly as on WADI and HAI (section 1).

Figure `a8_valley_synth_curved.png` shows the three panels (kappa = 0, 0.5, 1.0).

## 5. Normals in the valley (the cost side)

At each detector's global 1%-FPR threshold, the false-alarm rate on test-normal windows that fall in the valley partition
is LatAD 0.40 (SWaT, 2 to 9 windows), 0.06 (WADI), 0.003 (HAI); USAD / TranAD 0.12 (SWaT), 0.11 (WADI), 0.002 to 0.006
(HAI). Where valleys are deep enough to hold anomalies (SWaT), LatAD also fires on the normal windows passing through
them three times as often as the deep heads do.

## Verdict

The valley-smoothing mechanism is not empirically supported on any of the three datasets, and the synthetic controls
explain why. (i) Valley anomalies barely exist: 0.5% of HAI attacks, 5% of SWaT attacks, 9% of WADI attacks (2 to 15
windows), because the inter-regime valleys of the real geometry are shallow (their floors sit at the train 6th to 36th
density percentile on WADI and HAI, within the normal envelope of the nearest regime on 96 to 100% of close pairs), so an
extreme of one regime moving toward its neighbour becomes an in-cluster point before it becomes a valley point. (ii) Where
valley anomalies do exist (SWaT, WADI) USAD and TranAD catch them (AUROC 0.95 to 0.97, recall at 1% FPR 0.66 to 0.85,
higher than LatAD's 0.28 to 0.50). (iii) The one dataset where LatAD beats USAD / TranAD by a wide margin, HAI, has a
"deep misses, LatAD catches" set of 76 to 83 windows that contains zero valley anomalies; it is enriched in
moderate-magnitude OFF-manifold anomalies (79 to 86%), which the plain AE baseline also catches, and the correlation
between betweenness and LatAD's edge on HAI is negative. (iv) The valley-vs-off interaction is +0.003 to +0.07 with
seed-dependent sign on SWaT, null on WADI, and rests on 3 to 4 windows on HAI. (v) In synthetic worlds, a valley effect for
the reconstructor appears only with a curved manifold and only as "on-manifold interpolation is easier than on-manifold
extrapolation" (deficit 0.06 to 0.15 AUROC at gaps 6 to 8, gone at gap 12); the straight-line chord is caught at 1.00, the
density head shows the same asymmetry, and at realistic gaps (4 sd) neither head sees the valley at all. What the data do
support, and what the paper already says in the "reconstructable but improbable" framing (`framing_brainstorm.md`), is a
manifold statement: USAD and TranAD miss moderate departures that a density head and a plain AE both flag on HAI. That
result should not be attributed to inter-regime valleys. For the paper: keep A8 as a geometric precondition
(close-but-distinct regimes exist) and do not claim a valley-specific detection advantage; nothing here yields a
wins-only sentence about the valley mechanism.
