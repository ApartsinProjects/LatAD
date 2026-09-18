# A3 ("thin between-regime pockets"): alternative measures, detectors, and a cross-validation pilot

Report only. Scripts: `_diagnostics/fable_a3_pilot.py`, `fable_a3_pilot2.py`, `fable_a3_pilot3.py`;
raw numbers: `_diagnostics/fable_a3_pilot.json`; logs `fable_a3_pilot*.log`. One seed unless stated,
CPU, 21 s + ~40 s + ~5 s. Nothing in paper/, models, or checkpoints was touched.

## 1. Verdict in three lines

1. **"A3 absent on WADI/HAI/SWaT" is robust across every measure tried** (five measures, two spaces, two
   clusterings): the three benchmarks are low in absolute terms on all of them. ρ is not hiding A3 there.
2. **"SKAB has A3" is measure-dependent, and the current ρ gate is fragile.** SKAB scores high on every
   *ambiguity/overlap* measure (ρ, responsibility entropy, kNN label purity) but *lowest of all datasets*
   on the *thin-pocket* measure (between-mode points that are also rare). SKAB's between-mode mass is
   0.6-0.8x as far from its neighbours as core mass, i.e. denser, not thinner. What SKAB has is heavy
   mode overlap, which is exactly what `fit_basin_head`'s own docstring says it needs ("only helps when
   modes OVERLAP"). The paper's "narrow valid-but-rare bands" wording describes a geometry no dataset in
   the study exhibits in train-normal mass.
3. **The ρ<0.5 gate is seed-fragile on SKAB at the reported config** (K16/LD6/ep40): ρ = 0.58, 0.55,
   0.45, 0.18, 0.08 over seeds 0-4, so `basin_lam` = 1.07, 1.01, 0.76, 0.07, 0.00. On two of five seeds
   the head that produces the reported SKAB lift is effectively off. Mean normalised responsibility
   entropy is stable over the same seeds (0.29, 0.32, 0.28, 0.23, 0.26) and separates SKAB from the
   benchmarks (0.053 / 0.066 / 0.002) by 3.5x at the worst seed. That is the recommended replacement.

## 2. Invariants stated before running, and outcome

| id | invariant | outcome |
|---|---|---|
| I0a | 2 well-separated Gaussians, K=2: between ≈ 0, gap ≈ 0 | PASS (between 0.004, gap 0.000) |
| I0b | same + 20 % uniform bridge mass, K=2: between ≈ 0.20 | PASS for `between` (0.147; bridge t∈[0.15,0.85] clipped to [0.2,0.8] gives 0.17 expected). **FAIL for `gap`** (0.000): my bridge had spread 0.7 vs cluster spread 1.0, so it was *denser* than the clusters. The control did not validate `gap`; see I0d. |
| I0c | over-clustering a single blob (K=8 on 2 Gaussians) should NOT create between-mass if the measure is sound | **FAIL**: between 0.33, beta_knn 0.36. Chord/purity measures inflate under over-segmentation. Caveat carried into every K-dependent number below. |
| I0d | the rare-and-between (`gap`) ratio must exceed the 10 % base rate somewhere it should | PASS on miim train-normal (gap/between 0.36-0.52 vs 0.10 base: rare mode-transition windows sit on chords). So `gap` does respond to genuine thin bridges. |
| I1 | if a measure is valid and agrees with ρ: SKAB high, benchmarks low | PASS for entropy and beta_knn(K=dataset K); **FAIL for `between` and `gap`** (details in §3) |
| I2 | ordering stable to K (dataset K vs 8) | FAIL for beta_knn (SKAB 0.18 → 0.04 drops below WADI 0.08 at K=8) |
| I3 | SKAB ρ reproduces 0.58 at LD6/ep40 and ~0.10 at LD8/ep30 | PASS (0.58/0.55 and 0.10/0.00 over seeds 0,1). The earlier screen-vs-witness discrepancy is real config-dependence, not a feature or code difference (both use the same `window_features("stats")`). |

## 3. Pilot numbers (train-normal, seed 0)

### 3a. VaDE-free, PCA-20 of standardised window features, KMeans(K), LOO kNN k=10

| dataset | K | beta_knn (10-NN same-label share < 0.5) | between (interior chord, within in-cluster radius) | gap (between AND kNN radius > q90 of core) | gap/between |
|---|---|---|---|---|---|
| SKAB | 16 (reported) | **0.182** | 0.245 | **0.005** | 0.02 |
| SKAB | 8 | 0.043 | 0.045 | 0.007 | 0.16 |
| WADI | 20 (reported) | 0.092 | 0.267 | 0.016 | 0.06 |
| WADI | 8 | 0.084 | 0.201 | 0.010 | 0.05 |
| HAI | 40 (reported) | 0.064 | 0.340 | 0.021 | 0.06 |
| HAI | 8 | 0.024 | 0.187 | 0.004 | 0.02 |
| SWaT | 40 (reported) | 0.025 | 0.154 | 0.017 | 0.11 |
| SWaT | 8 | 0.013 | 0.052 | 0.006 | 0.12 |
| miim normal | 8 / 16 | 0.034 / 0.046 | 0.129 / 0.122 | 0.047 / 0.064 | 0.36 / 0.52 |

miim rows are under-clustered (the unified generator has 63 ground-truth modes), so `between` there is
mostly whole modes lying between 8 centroids; the row is kept only as the I0d sanity reference for `gap`.

### 3b. VaDE latent space, VaDE argmax labels, `mu_c` centroids (cached `e2_fable_*.npz`; SKAB refit)

| dataset | interior | between | gap | gap/between | median kNN radius, between / core |
|---|---|---|---|---|---|
| SKAB s0..s4 | 0.41-0.59 | 0.35-0.46 | 0.000-0.018 | 0.00-0.05 | **0.60-0.77** (between mass is denser than core) |
| WADI | 0.40 | 0.17 | 0.007 | 0.04 | 1.44 |
| HAI | 0.65 | 0.38 | 0.018 | 0.05 | 0.99 |
| SWaT | 0.22 | 0.11 | **0.036** | **0.33** | **2.19** |

### 3c. VaDE-dependent, graded (responsibility entropy; ρ at three thresholds)

| dataset / config | ρ<0.5 | ρ<0.6 | ρ<0.7 | mean max-resp | H_norm mean | frac H_norm > 0.3 |
|---|---|---|---|---|---|---|
| SKAB K16/LD6/ep40 s0 | 0.580 | 0.613 | 0.630 | 0.62 | 0.291 | 0.61 |
| SKAB same, s1 | 0.553 | 0.628 | 0.690 | 0.59 | 0.322 | 0.61 |
| SKAB same, s2 / s3 / s4 | 0.453 / 0.177 / 0.075 | | | | 0.275 / 0.228 / 0.261 | |
| SKAB K16/LD8/ep30 s0 | 0.103 | 0.528 | 0.768 | 0.66 | 0.268 | 0.72 |
| SKAB K16/LD8/ep30 s1 | **0.000** | 0.310 | 0.540 | 0.76 | 0.161 | 0.01 |
| WADI K20/LD10 (cached) | 0.013 | 0.047 | 0.081 | 0.94 | 0.053 | 0.02 |
| HAI K40/LD16 (cached) | 0.023 | 0.064 | 0.115 | 0.91 | 0.066 | 0.02 |
| SWaT K40/LD16 (cached) | 0.000 | 0.001 | 0.002 | 1.00 | 0.002 | 0.00 |

ρ<0.5 on SKAB spans 0.00-0.58 across (LD, epochs, seed); the max-responsibility distribution on SKAB is
broad and centred near the 0.5 cut, so a hard threshold turns configuration noise into a gate flip.
H_norm spans 0.16-0.32 on SKAB and never exceeds 0.07 on a benchmark.

### 3d. Detector axis (axis ii)

SKAB difficult subset, witness split (185 hard windows), input-space PCA-20 detectors:
kNN-radius 0.466, chord-proximity 0.483, rank-sum gap 0.457, versus witness base 0.500 and basin
head forced-on 0.605 (λ=1) / 0.645 (λ=2). All three VaDE-free geometric detectors are at or below
chance on SKAB-hard; none rivals the basin head. Below-chance kNN means SKAB's hard anomalies sit in
*denser* regions than typical normals, consistent with "overlap core", not "thin pocket".

Benchmarks, hard subset, VaDE-latent detectors vs cached base head (seed 0, `heads_*.npz`, same split
asserted):

| dataset | base | kNN-radius(z) | chord | rank-sum gap | base + 0.5·gap | n_hard | hard interior / normal interior |
|---|---|---|---|---|---|---|---|
| WADI | 0.670 | 0.659 | 0.542 | 0.624 | 0.669 | 19 | 0.58 / 0.46 |
| HAI | 0.811 | 0.801 | 0.397 | 0.664 | 0.797 | 167 | 0.75 / 0.70 |
| SWaT | 0.945 | **0.960** | 0.701 | 0.918 | 0.931 | 38 | **0.55 / 0.14** |

The gap detector never beats base and adding it never helps. SWaT is the one place with a lead: hard
anomalies sit on between-mode chords 4x more often than normals, but a plain kNN radius in z already
scores them (0.960 > 0.945), so they are between-mode *and* improbable: density catches them, they are
not hiding. That is the opposite of the A3 threat model.

miim (unified, seed 0) pocket faults vs normal, input-space PCA-20: kNN-radius AUROC 0.831, chord
0.699, gap 0.812. The VaDE mixture head in `e2_fable_miim.jsonl` reports pocket TPR 0.375 at 3.1 % FPR,
which is not construct-matched to an AUROC, so no ranking is claimed; the point is that pocket faults in
miim are caught by plain density, again not the A3 threat model.

## 4. Ranked measures and detectors (i = cross-validates "A3 absent on benchmarks"? ii = detects A3 anomalies better than the basin head? iii = cost)

### Measures

1. **Responsibility entropy H_norm (graded ρ)**. i: yes; benchmarks ≤ 0.07 vs SKAB ≥ 0.16 across every
   config and seed, the only measure whose SKAB-vs-benchmark separation survives seeds. ii: n/a (a gate,
   not a detector). iii: free (one line on the existing responsibilities). Still VaDE-dependent, but no
   threshold cliff. **Recommended as the gate statistic.**
2. **kNN label-purity ambiguity beta_knn (VaDE-free)**. i: agrees at the reported K (SKAB 0.18 vs
   0.03-0.09) but the ordering flips at K=8 and I0c shows it inflates with over-segmentation. ii: n/a.
   iii: seconds. Useful as an independent sanity witness, not as a gate.
3. **Rare-and-between "gap" fraction (reachable but improbable)**. i: yes for the benchmarks in the sense
   that all are ≤ 0.02 in feature space and ≤ 0.036 in latent; but it also puts SKAB *lowest*, so it
   contradicts the thin-pocket reading of SKAB. Validated only by I0d (miim transition windows), not by a
   clean bridge control. ii: as a detector, at or below base everywhere. iii: seconds. Its real value is
   diagnostic: it is the one measure that distinguishes "overlap" from "thin pocket".
4. **Between-chord fraction alone**. i: no; it is dominated by over-segmentation (K=8 on one blob gives
   0.33) and ranks HAI > WADI > SKAB. Not usable as an A3 measure. iii: seconds.
5. **Reconstruction-error-low ∧ density-low gap** (the literal paper thesis with the decoder as
   "reachability"). Not run. Assessment: it would inherit the decoder's own generalisation failures (the
   resid head on WADI is disabled for exactly this reason: recon drifts on held-out normal), so low recon
   error is not a clean reachability witness. iii: cheap with cached models. Lower priority than 3.
6. **Persistent-homology necks / local intrinsic dimension between clusters**. Not run. Would give a
   principled thin-bridge count (H0 death times between clusters), but on 738-dim stats features with
   2.6k-18k windows it needs subsampling and a PCA choice that already decides the answer. iii: minutes
   with `ripser`, plus a new dependency. Only worth it if 3 had contradicted ρ.
7. **Silhouette-valley mass / gap statistic**. Not run. Global scalars that cannot say *where* the mass
   is; subsumed by 2 and 3.

### Detectors

1. **Basin-agreement head (current)**. Best on SKAB-hard among everything measured (0.605-0.645 vs
   ≤ 0.50). Its mechanism is mode overlap, which is what SKAB has. Its gate is the weak part.
2. **kNN radius in VaDE latent**. Matches or beats base on SWaT-hard (0.960 vs 0.945), ties on WADI/HAI,
   chance on SKAB. It is a density head, not an A3 head; it does not detect anything that hides from
   density.
3. **Rank-sum gap (kNN ∧ chord)**. Never beats base; on HAI the chord term is actively harmful (0.397).
4. **One-class boundary model on the inter-mode region**. Not run. On the pilot geometry the inter-mode
   region of SKAB is *denser* than core, so a one-class model would learn it as normal and gain nothing;
   on the benchmarks the region carries ≤ 2 % of mass, too little to fit. Not recommended.

## 5. Is "A3 absent on the benchmarks" robust or measure-dependent?

Robust. Five measures (ρ at three thresholds, entropy, beta_knn, between, gap), two spaces (feature PCA-20
and VaDE latent), two clusterings (KMeans at reported K and at 8, VaDE argmax) all put WADI/HAI/SWaT low
in absolute terms, and the latent-gap detector finds no exploitable hidden mass. The largest "hidden"
signal is SWaT's latent gap fraction 0.036 with a 2.2x rarity ratio, and §3d shows plain density already
scores those points.

What is measure-dependent is the *positive* case. SKAB is high on ambiguity measures and lowest on the
thin-pocket measure. The correct reading of the SKAB result is "the basin head helps when modes overlap
heavily (mean max-responsibility 0.62, entropy 0.29)", which is what the code says and what the head
does. The scope claim "A3 = narrow valid-but-rare bands" is not what the SKAB witness demonstrates, and no
dataset in the study has more than ~2-4 % of train-normal mass in rare between-mode bands.

## 6. One recommended next step

Replace the gate statistic in `fit_basin_head` from `frac(max-resp < 0.5)` with mean normalised
responsibility entropy (threshold ~0.15, sitting between the benchmark max 0.066 and the SKAB seed-min
0.228), then re-run `a3_witness.py` on SKAB over seeds 0-4 and confirm (a) the gate fires on all five
SKAB seeds, (b) the difficult-AUROC lift persists per seed, (c) it never fires on WADI/HAI/SWaT (already
implied by the cached entropies). Cost: SKAB has 400 train windows, so five witness runs are ~2 minutes
of agent time on CPU; the benchmark side needs no new fits. This is a change to the model file, so it is
outside this report's remit and is left to the caller. If the lift does not persist per seed once the
gate is stable, the SKAB witness itself is a seed artefact and the paper's A3 sentence should be reworded
to "mode overlap" regardless.

## 7. Things I tried to break

- Same-split assertion between `e2_fable_*.npz` labels and `heads_*.npz` labels passed for all three
  benchmarks (§3d numbers are construct-matched to the cached base head, seed 0 only).
- SKAB ρ=0.58 reproduced exactly at the witness config and seed, so the pilot's SKAB fits are the
  witness's fits, not a different model.
- `gap` was checked against the I0b control and failed there because the control's bridge was dense;
  it was then validated on miim transition windows (I0d). Treat `gap` as a diagnostic with a partially
  validated control, not as a calibrated estimator.
- All K-dependent numbers were run at two K's; the K=8 control (I0c) shows chord/purity measures inflate
  under over-segmentation, so no absolute `between` value is quoted as A3 mass anywhere above.
- miim train-normal rows are under-clustered (63 true modes vs K≤16) and are not used for any claim except
  the I0d sanity check; the miim pocket detector numbers are AUROC and the cached reference is TPR@FPR, so
  no comparison between them is claimed.
