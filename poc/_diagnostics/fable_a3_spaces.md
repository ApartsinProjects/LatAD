# Is "A3 absent on WADI/HAI/SWaT" an artifact of clustering K (H1) or latent-space coarseness (H2)?

Report only. Nothing in paper/, models, or checkpoints was touched. Scripts and raw numbers, all in
`_diagnostics/`: `fable_a3_spaces_lib.py` (measures), `fable_a3_spaces_real.{py,json,log}` (H2 on the
four datasets), `fable_a3_spaces_hai21.{py,json}` (H1b, HAI regime 21), `fable_a3_spaces_rare.{py,json,log}`
(H1 rare-regime loophole), `fable_a3_spaces_synth3.{py,json,log}` (the synthetic invariant; v1 and v2 are
kept as failed controls, see §4), `fable_a3_spaces_floor.{py,log}` (side finding, §6). The K-sweep numbers
are the pre-existing `a3_ksweep.json` (job `byefagy9t`). One seed throughout unless stated. CPU total
about 15 min (HAI VaDE fits dominate).

## 1. Verdict

**"A3 absent on the benchmarks" is robust to K and to the representation.** No K in 8..128, no PCA
dimension from 5 to the full 306-738 features, and no VaDE latent in LD 6..32 raises WADI/HAI/SWaT
above the synthetic no-overlap floor. The two hypothesised hiding mechanisms do not exist as mechanisms:

- **H1 (merging hides overlap): rejected.** Un-merging can only raise responsibility entropy, and the
  benchmarks do not rise: at K=64 they read 0.031 / 0.084 / 0.010 (WADI / HAI / SWaT), at K=128 on the
  cached latents 0.011 / 0.042 / 0.002. Points that land in components with < 1 % train mass (up to
  49-54 % of all mass at K=128) read *lower* entropy than the bulk (WADI 0.010 vs 0.012, HAI 0.041 vs
  0.043, SWaT 0.0007 vs 0.002). Rare regimes are separated clumps, not overlap partners. HAI's regime-21
  block is not merged at the paper's K (99.8 % in its own component of 0.40 % train mass, entropy
  0.0003), nor at K=8 (own component, 0.32 %); forcing a dedicated component changes train entropy by
  -0.0004 and makes 0.0 % of train windows ambiguous. The block sits 12.9 latent units from its nearest
  train window (train LOO 99th percentile: 2.8) and 59.6 feature units (99th pct 13.6): it is unseen,
  not overlapping.
- **H2 (coarse latent hides overlap): not a viable mechanism.** On the synthetic known-A3 control, a
  latent too coarse to separate the overlapping pair *raises* the measured entropy (PCA-1 0.286, PCA-2
  0.182, VaDE LD1 0.174 vs 0.12-0.13 in every rich space, truth 0.132); it never lowers it. Coarseness
  fabricates ambiguity (the mixture cuts through the merged blob); it cannot hide it. On the real
  benchmarks the reads fall or stay flat as the space gets richer (WADI VaDE 0.074 -> 0.034 -> 0.012 at
  LD 6/16/32; SWaT 0.007 -> 0.003 -> 0.002; HAI 0.091 -> 0.051 -> 0.085), and the dimension-fair
  kNN read is flat across every PCA-k and the full space (WADI 0.07-0.12, HAI 0.06-0.07, SWaT
  0.03-0.04). Nothing appears in a richer space.

Nothing here reveals A3 on a benchmark, so the paper's benchmark-side claim stands. **What the sweep
did overturn is the SKAB side** (§6): SKAB's high responsibility entropy (0.29-0.32) is produced by
VaDE's component variance floor on a 400-window dataset, not by point-level regime overlap. The same
SKAB latent with empirically estimated component variances reads 0.045-0.048, indistinguishable from
WADI (0.053-0.057). That changes what "A3 holds on SKAB" means. §7 runs the resolving experiment: the
ρ gate is a floor artifact on every seed, and the basin-head lift does not survive a floor at the
empirical variance level, so **the SKAB A3 witness is a floor artifact** and A3 is not demonstrated on
any dataset in the study.

## 2. H1: K-sweep interpretation (job `byefagy9t`, `a3_ksweep.json`; VaDE responsibility entropy H_norm)

| dataset (n train) | K=8 | K=16 | K=32 | K=64 | points / component at K=64 |
|---|---|---|---|---|---|
| SKAB (400) | 0.063 | 0.267 | 0.415 | 0.527 | 6 |
| WADI (2614) | 0.067 | 0.056 | 0.050 | 0.031 | 41 |
| HAI (18359) | 0.080 | 0.147 | 0.096 | 0.084 | 287 |
| SWaT (3623) | 0.016 | 0.010 | 0.008 | 0.010 | 57 |

Invariant stated before reading: if merging hid A3, benchmark entropy rises toward SKAB as K grows.
Outcome: WADI falls monotonically, SWaT is flat at ~0.01, HAI peaks at K=16 (0.147) and returns to
0.08. None approaches SKAB's K=16 value. The synthetic control (§4) shows what over-clustering does to a
genuinely clumpy dataset: one surplus component over four clean regimes reads 0.13-0.15, K=2x true
reads 0.16-0.29. The benchmarks at K=64 (about 3x their BIC K* of 22+) stay at or below the no-overlap
floor, so their mass is in clumps that a finer mixture keeps carving cleanly. WADI at K=64 has more
points per component (41) than SKAB at K=16 (25) and reads 0.031 vs 0.267, so the SKAB/benchmark gap
is not a points-per-component effect either.

SKAB's own rise (0.06 at K=8 to 0.53 at K=64, six windows per component) is the over-segmentation
signature from the control, not evidence of overlap; see §6 for what its K=16 value is made of.

### 2b. Rare-regime loophole (`fable_a3_spaces_rare.json`): regimes rarer than 1/K

Diag GMMs at K = paper / 64 / 128 on the cached paper-config latents and on PCA-20 of the features.
Entropy of points whose component carries < 1 % train mass, vs the rest:

| dataset | K | # components < 1 % | mass in them | H rare points | H bulk points | rare points with H > 0.3 |
|---|---|---|---|---|---|---|
| WADI latent | 128 | 101 | 0.49 | 0.010 | 0.012 | 0.0 % |
| WADI PCA-20 | 128 | 99 | 0.52 | 0.003 | 0.004 | 0.0 % |
| HAI latent | 128 | 89 | 0.50 | 0.041 | 0.043 | 0.1 % |
| HAI PCA-20 | 128 | 92 | 0.54 | 0.024 | 0.027 | 0.0 % |
| SWaT latent | 128 | 95 | 0.29 | 0.0007 | 0.002 | 0.0 % |
| SKAB latent (refit) | 64 | 29 | 0.11 | 0.002 | 0.052 | 0.0 % |

Invariant: rare regimes that overlap big ones read high entropy once they get their own component.
Outcome: rare-component points are the *crispest* points everywhere. The loophole is closed down to
components of 1-10 windows.

### 2c. HAI regime 21 (`fable_a3_spaces_hai21.json`; test windows 6436-7081, all test-normal)

| check | result |
|---|---|
| paper K=40: block's component | 99.8 % in component 21; train mass 0.40 % (73 windows); block H 0.0003, rho 0.0; test-normal outside the block H 0.071 |
| merged at coarser K? | GMM on the same latent, K=8: own component (train mass 0.32 %); K=16: own (0.56 %); K=40: own (0.05 %) |
| distance to train | latent: median NN 12.9 vs train LOO q99 2.82 (100 % of block beyond q99); features: 59.6 vs 13.6 (100 %) |
| forced dedicated component (K=41, weight 0.4 % or 6.3 %) | train H 0.0658 -> 0.0654; rho unchanged 0.0226; 0.0 % of train ambiguous toward it; 99.5 % of the block captured |

The block is a separated, unseen regime (the coverage story already in the paper), not a merged
overlap partner. Invariants B1-B4 of the script all hold.

## 3. H2: representation sweep on the real datasets (`fable_a3_spaces_real.json`, K = paper's K)

VaDE-free reads: `gmm_H` = mean normalised responsibility entropy of a diag GMM(K) fit in that space;
`knn` = fraction of train windows whose 10 nearest neighbours are < 50 % same KMeans(K) label (the
dimension-fair twin of rho; GMM-label kNN is not dimension-fair, see §4c). VaDE reads at LD 6/16/32.

| dataset | space | gmm_H | knn (KMeans labels) | | VaDE LD | vade_H | rho<0.5 | latent knn |
|---|---|---|---|---|---|---|---|---|
| SKAB K16 | PCA-5 (70 % var) | 0.043 | 0.053 | | 6 | **0.291** | 0.580 | 0.075 |
| | PCA-20 (99 %) | 0.020 | 0.183 | | 16 | 0.262 | 0.158 | 0.068 |
| | full 48 | 0.007 | 0.125 | | 32 | 0.273 | 0.138 | 0.123 |
| WADI K20 | PCA-5 (36 %) | 0.073 | 0.069 | | 6 | 0.074 | 0.019 | 0.099 |
| | PCA-20 (62 %) | 0.032 | 0.101 | | 16 | 0.034 | 0.009 | 0.121 |
| | PCA-50 (82 %) | 0.014 | 0.104 | | 32 | 0.012 | 0.001 | 0.092 |
| | PCA-200 (99 %) | 0.005 | 0.106 | | | | | |
| | full 738 | 0.002 | 0.115 | | | | | |
| HAI K40 | PCA-5 (44 %) | 0.120 | 0.061 | | 6 | 0.091 | 0.030 | 0.051 |
| | PCA-20 (76 %) | 0.029 | 0.064 | | 16 | 0.051 | 0.014 | 0.071 |
| | PCA-50 (93 %) | 0.016 | 0.064 | | 32 | 0.085 | 0.034 | 0.106 |
| | PCA-200 (100 %) | 0.009 | 0.068 | | | | | |
| | full 354 | 0.008 | 0.068 | | | | | |
| SWaT K40 | PCA-5 (51 %) | 0.020 | 0.026 | | 6 | 0.007 | 0.000 | 0.019 |
| | PCA-20 (84 %) | 0.003 | 0.025 | | 16 | 0.003 | 0.001 | 0.021 |
| | PCA-50 (97 %) | 0.001 | 0.036 | | 32 | 0.002 | 0.000 | 0.016 |
| | PCA-200 (100 %) | 0.0002 | 0.030 | | | | | |
| | full 306 | 0.0002 | 0.030 | | | | | |

Reading it against the synthetic scale (§4, same measures, K = true regime count): known overlap reads
gmm_H 0.12-0.13 and knn_true 0.08 in PCA-5..50; no overlap reads 0.03-0.04 and 0.02. The benchmarks at
PCA-20/50 sit at the no-overlap level (0.03 / 0.03 / 0.003 for WADI / HAI / SWaT) despite using K =
20-40 rather than 5, which the control shows can only inflate the read. The one benchmark cell above
0.1 is HAI PCA-5 at K=40 (0.120): 40 components in a 5-d space carrying 44 % of the variance, i.e.
over-clustering in an under-specified space; it is gone by PCA-20 (0.029). Full-space gmm_H values
(0.0002-0.008) are not comparable across dimensions (§4c) and are listed only for completeness.

The VaDE reads move the wrong way for H2: richer latents lower WADI and SWaT, and HAI's LD32 (0.085)
is within its seed noise of LD6 (0.091). SKAB's VaDE read is flat at 0.26-0.29 across LD while its
latent kNN read is 0.07-0.12, which is the discrepancy pursued in §6.

## 4. Synthetic invariant (`fable_a3_spaces_synth3.json`)

Design: 200 features from 10 correlated latent factors (sd 3 x 0.8^j) through random loadings plus
iid noise, per-feature standardised like the real pipeline. Five regimes: four far apart (+-12 on
factors 0/1), one overlapping pair R1/R2 separated by 2.5 sd along factor 2, whose variance stays below
factors 0 and 1 so PCA-1 and PCA-2 drop the overlap direction (the H2 scenario). Controls: no-A3 twin
(pair separated by 8 sd) and an over-segmentation twin (four regimes, no pair, still K=5). Truth is the
exact Bayes posterior entropy from the generating factors (normalised by log 5).

| case (truth H) | PCA-1 | PCA-2 | PCA-5 | PCA-20 | PCA-50 | PCA-200 | VaDE LD1 | LD2 | LD6 | LD16 |
|---|---|---|---|---|---|---|---|---|---|---|
| known-A3 (0.132), K=5 | 0.286 | 0.182 | **0.126** | **0.124** | **0.120** | 0.102 | 0.174 | 0.117 | 0.156 | 0.065 |
|  ARI vs truth | 0.29 | 0.46 | 0.75 | 0.75 | 0.74 | 0.74 | 0.29 | 0.71 | 0.72 | 0.73 |
|  knn ambiguity, TRUE labels | 0.50 | 0.27 | 0.08 | 0.08 | 0.08 | 0.08 | 0.23 | 0.09 | 0.09 | 0.08 |
| no-A3 (0.034), K=5 | 0.346 | 0.125 | **0.036** | **0.035** | **0.034** | 0.027 | 0.092 | 0.045 | 0.032 | 0.043 |
|  ARI vs truth | 0.26 | 0.65 | 0.95 | 0.95 | 0.95 | 0.94 | 0.82 | 0.94 | 0.93 | 0.70 |
| over-seg twin (0.050), K=5 on 4 regimes | 0.250 | 0.104 | 0.146 | 0.140 | 0.129 | 0.075 | 0.162 | 0.157 | 0.176 | 0.093 |
| known-A3, K=10 (2x true) | 0.215 | 0.327 | 0.271 | 0.229 | 0.220 | 0.107 | | 0.231 | 0.243 | |
| no-A3, K=10 | 0.242 | 0.248 | 0.214 | 0.192 | 0.157 | 0.080 | | 0.195 | 0.171 | |

Invariant outcomes:

- **S1 (machinery detects the effect): PASS.** In every rich space (PCA-5..50, VaDE LD2/6) the known-A3
  read is within 0.03 of the truth (0.132) and 3-4x the no-A3 twin, which itself matches its truth
  (0.034). The mixture recovers the overlapping pair (ARI 0.72-0.75; the overlap itself caps ARI below
  the no-A3 twin's 0.95).
- **S4 (H2 mechanism): FAIL for H2.** The coarse spaces mix the pair (kNN-true ambiguity 0.50 / 0.27 in
  PCA-1/2 vs 0.08 rich) and the measured entropy goes UP (0.286 / 0.182), not down. The no-A3 twin in
  PCA-1 reads 0.346, higher than genuine A3 in a rich space. A coarse latent fabricates overlap; it does
  not hide it. Therefore a low benchmark read cannot be a coarse-latent artifact. H2 is not viable.
- **S3/S5 (over-clustering): confirmed as the artifact direction.** One surplus component (over-seg twin)
  reads 0.13-0.15, the same as genuine A3; K=2x true reads 0.16-0.33 regardless of overlap. Entropy
  therefore cannot distinguish overlap from over-splitting when K exceeds the regime count, and it can
  only move UP with K. This is why the benchmarks' flat-or-falling K-sweep is decisive and SKAB's rising
  one is not.
- **VaDE LD1 on known-A3 reads 0.174 with ARI 0.29** (structure lost) and **LD16 reads 0.065** (below
  truth, the encoder pulls the pair apart): VaDE's own entropy is a biased overlap estimator at both
  extremes of LD, which is one more reason the LD-flatness of SKAB and the LD-fall of WADI/SWaT should
  be read with the kNN column alongside.

### 4c. Two failed controls, kept for the record

- v1 (`fable_a3_spaces_synth.json`): isotropic 200-d noise. After per-feature standardisation nothing
  recovers the regimes at any dimension (ARI <= 0.10 for known-A3, no-A3 and a single blob alike; all
  three read the same). Invalid as a control; replaced.
- v2 (`fable_a3_spaces_synth2.json`): correlated factors but only three regimes and the overlap on the
  6th factor: a K=3 diag GMM prefers cutting a high-variance factor over finding the 2.5 sd bimodality
  (ARI <= 0.29 everywhere), so its 0.36 read was wrong-cut ambiguity. Replaced by v3 (four anchors take
  four components, forcing the fifth onto the pair).
- Dimension effects seen in all versions: diag-GMM entropy shrinks mechanically with dimension
  (known-A3 0.126 at PCA-5 -> 0.102 at PCA-200; the single blob at K=3 reads 0.69 at PCA-5 and 0.48 at
  PCA-200), and kNN ambiguity with GMM labels inflates with dimension (blob: 0.04 at PCA-5, 0.57 at
  PCA-200) because diag-Mahalanobis labels stop matching Euclidean neighbourhoods. kNN with KMeans labels
  is the only read that is flat across dimension, so it is the one used for cross-space comparison in
  §3. Per-PC whitening beyond PCA-5 destroys the structure (kNN-true ambiguity 0.26-0.78) and is not
  used.

## 5. Bottom line on the question asked

"A3 absent on WADI/HAI/SWaT" is robust. It survives K from 8 to 128 (VaDE and post-hoc GMM), every PCA
dimension from 5 to full, three VaDE latent widths, a direct probe of the rarest components, and the one
named rare regime (HAI-21). The synthetic control shows the machinery detects a 2.5 sd overlap at truth
level in exactly the spaces where the benchmarks read at the no-overlap floor, and shows that both
proposed hiding mechanisms push the measure in the wrong direction (finer K and coarser latents can
only add ambiguity). No cell in any sweep needs the "suspect extreme" caveat on the benchmark side: the
highest benchmark value anywhere is HAI PCA-5 at K=40 (0.120), an over-clustering cell that vanishes at
PCA-20.

## 6. Side finding that changes the reference: SKAB's A3 signature is the variance floor

Surfaced by §2b: a diag GMM refit on SKAB's own VaDE latent (same 400 points) reads H = 0.051 while
VaDE's responsibilities on those points read 0.291. `fable_a3_spaces_floor.py`, two seeds, paper config
K16/LD6/ep40:

| | seed 0 | seed 1 | WADI (cached, K20/LD10) |
|---|---|---|---|
| VaDE own H_norm | 0.291 | 0.322 | 0.053 |
| component-dims sitting at the variance floor (sd 0.224) | 96 % | 89 % | 4 % |
| VaDE component sd vs empirical within-component sd | 0.229 vs 0.111 | 0.241 vs 0.127 | 0.781 vs 0.879 |
| VaDE means and pi, variances re-estimated from the data | **0.048** | **0.045** | 0.057 |
| refit GMM on the latent | 0.051 | 0.045 | 0.054 |
| refit GMM with the floor imposed | 0.316 | 0.189 | |

On SKAB, VaDE's components are pinned at the floor and twice as wide as the clumps they cover; replacing
the floored variances with the empirical ones collapses the entropy to the WADI level, and imposing the
floor on a clean refit recreates it. On WADI the floor is inactive and the read is unchanged. So the
quantity the paper calls A3 support on SKAB (rho 0.58, H 0.29) is responsibility broadening from the
`logvar_floor = log 0.05` guard acting on a small latent, not between-regime mass. The latent geometry
of SKAB (kNN ambiguity 0.07, refit-GMM H 0.05) is as crisp as the benchmarks'.

What this does and does not say: it does not touch the measured basin-head lift on SKAB-hard (AUROC
0.605-0.645 vs 0.50 in the earlier pilot); that head consumes the floored responsibilities, and whether
its gain depends on the floor is a separate one-line experiment (raise or drop `logvar_floor`, re-score
SKAB-hard). It does mean the ordering "SKAB has A3, benchmarks do not" is a statement about VaDE's
prior guard interacting with n=400, and the paper's A3 text should not be built on rho or H_norm as
currently computed. Recommended next step for the main thread: rerun the SKAB basin-head witness with
the floor at 0.01 and at 0.05 (two fits, about a minute), and rerun the K16 rho/H read with empirical
variances, before deciding how the A3 paragraph is worded.

## 7. Variance-floor resolution: does the SKAB basin-head lift survive a lowered floor?

Scripts `fable_a3_floor_resolution.{py,json,log}` (retrain at three floors, 3 seeds, full witness) and
`fable_a3_floor_2x2.{py,json,log}` (train-floor x score-floor isolation). The floor is changed by a local
monkeypatch of `VaDE.__init__` (`models_vade.py` untouched). Reported config throughout: SKAB, K=16,
LD=6, 40 epochs, heads fit exactly as `a3_seed_robustness.py` (185 hard / 69 easy test windows, 400 train).

**Invariant, stated before the lowered-floor runs and met exactly:** at the default floor 0.05, seed 0
reproduces ρ = 0.580, H = 0.291, OFF 0.500 / ON(1) 0.605 / ON(2) 0.645 (lift +0.105 / +0.145); seeds
1 and 2 reproduce `a3_seed_robustness.json` to the third decimal (0.552/0.322/+0.044, 0.453/0.275/+0.068).
WADI at floor 0.05 reproduces its cached read (H 0.053, ρ 0.013).

### 7a. Retrained at each floor (difficult-subset AUROC; "emp" = same model, empirical within-component variances)

| floor | seed | ρ | H | ρ emp | H emp | dims at floor | gate λ | OFF | ON(1) | lift(1) | ON(2) | lift(2) | all-anomaly OFF → ON(1) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 (paper) | 0 | 0.580 | 0.291 | 0.003 | 0.048 | 96 % | 1.075 | 0.500 | 0.605 | **+0.105** | 0.645 | +0.145 | 0.620 → 0.683 |
| | 1 | 0.552 | 0.322 | 0.003 | 0.045 | 89 % | 1.006 | 0.452 | 0.496 | +0.044 | 0.542 | +0.090 | 0.592 → 0.623 |
| | 2 | 0.453 | 0.275 | 0.005 | 0.076 | 84 % | 0.756 | 0.469 | 0.538 | +0.068 | 0.581 | +0.112 | 0.598 → 0.648 |
| 0.02 | 0 | 0.198 | 0.225 | 0.000 | 0.046 | 85 % | 0.119 | 0.496 | 0.601 | +0.105 | 0.650 | +0.154 | 0.619 → 0.685 |
| | 1 | 0.170 | 0.246 | 0.003 | 0.047 | 72 % | 0.050 | 0.461 | 0.512 | +0.051 | 0.561 | +0.100 | 0.600 → 0.637 |
| | 2 | 0.098 | 0.217 | 0.000 | 0.052 | 72 % | 0.000 | 0.457 | 0.504 | +0.047 | 0.549 | +0.091 | 0.589 → 0.623 |
| 0.01 | 0 | 0.000 | 0.011 | 0.003 | 0.003 | 71 % | 0.000 | 0.553 | 0.531 | **-0.021** | 0.513 | -0.040 | 0.661 → 0.646 |
| | 1 | 0.007 | 0.119 | 0.000 | 0.044 | 45 % | 0.000 | 0.454 | 0.473 | +0.019 | 0.519 | +0.065 | 0.592 → 0.607 |
| | 2 | 0.043 | 0.156 | 0.000 | 0.041 | 55 % | 0.000 | 0.407 | 0.429 | +0.022 | 0.502 | +0.095 | 0.557 → 0.569 |
| WADI 0.05 | 0 | 0.013 | 0.053 | 0.015 | 0.057 | 4 % | 0.0 | | | | | | |
| WADI 0.01 | 0 | 0.012 | 0.053 | 0.014 | 0.057 | 0 % | 0.0 | | | | | | |

Mean forced lift(1) over seeds: +0.072 (0.05), +0.068 (0.02), **+0.007 (0.01; one seed negative)**.
The auto gate λ is 0.76-1.08 at the paper floor, 0-0.12 at floor 0.02, 0 at 0.01: the pipeline as
shipped (`use_basin='auto'`) delivers zero basin contribution at any floor below the default. WADI is
untouched by the floor (4 % of dims at the floor, 0 % at 0.01; ρ/H identical), so the benchmark side of
the comparison does not depend on this parameter.

### 7b. Where the floor acts: train floor x score floor (same trained model, basin head refit under each scoring variance; OFF identical within a row)

| train floor | seed | score 0.05 (ρ / lift1 / lift2) | score 0.02 | score 0.01 | score empirical variances |
|---|---|---|---|---|---|
| 0.05 | 0 | 0.580 / +0.105 / +0.145 | 0.207 / +0.112 / +0.152 | 0.052 / +0.112 / +0.150 | 0.003 / **+0.042** / +0.078 |
| 0.05 | 1 | 0.552 / +0.044 / +0.090 | 0.140 / +0.041 / +0.089 | 0.060 / +0.039 / +0.096 | 0.003 / **-0.007** / -0.002 |
| 0.05 | 2 | 0.453 / +0.068 / +0.112 | 0.087 / +0.042 / +0.082 | 0.025 / +0.036 / +0.073 | 0.000 / **-0.006** / -0.017 |
| 0.01 | 0 | 0.033 / +0.035 / +0.045 | 0.000 / +0.014 / +0.017 | 0.000 / -0.021 / -0.040 | 0.000 / -0.004 / -0.011 |
| 0.01 | 1 | 0.530 / +0.025 / +0.068 | 0.085 / +0.023 / +0.064 | 0.007 / +0.019 / +0.065 | 0.003 / +0.020 / +0.058 |
| 0.01 | 2 | 0.512 / +0.014 / +0.065 | 0.130 / +0.018 / +0.088 | 0.043 / +0.022 / +0.095 | 0.000 / +0.036 / +0.136 |

Two separate facts fall out:

1. **ρ is a pure scoring-time floor artifact.** On the same trained model, ρ goes 0.58 → 0.21 → 0.05 →
   0.003 as the scoring floor is lowered to the empirical level, with no retraining. The gate statistic
   the paper uses to declare A3 on SKAB measures how far the guard `logvar_floor = log 0.05` sits above
   the empirical component spread (sd 0.22 vs 0.11-0.15 on 400 windows). It is not between-regime mass.
2. **The forced-on lift is carried by the uniform floored variances, not by the latent geometry.** With
   the paper's trained models it persists while any uniform floor is imposed at score time (score 0.01
   still gives +0.04 to +0.11: with all 16 components pinned to one width the agreement head is a
   nearest-centroid stability test), and it vanishes the moment the components are given their actual
   variances (+0.042 / -0.007 / -0.006, mean +0.010). Models trained at floor 0.01 give +0.007 mean at
   their own floor (one seed negative) and +0.025 mean even when scored with the wide floor put back.
   The OFF base on SKAB-hard is at chance at every floor (0.41-0.55); the head's job in the paper is to
   lift a chance-level base to 0.50-0.60, and it can only do that with the floor binding.

### 7c. Verdict

**Witness-is-artifact.** The SKAB A3 witness has two parts and both are the floor: the ρ = 0.58 gate
disappears on every seed once component variances are at their empirical level (ρ ≤ 0.005), and the
basin-head lift on SKAB-hard, +0.072 mean at the default floor, is +0.007 at floor 0.01 (2 of 3 seeds
positive, 1 negative, all within the ±0.05 seed spread of a chance-level base) and +0.010 with
empirical variances on the paper's own models. It does not survive as an effect independent of the
floor. Combined with §1-§5 (no benchmark shows overlap under any K or representation), A3 is not
demonstrated on any dataset in the study. The paper's A3 claim ("thin between-regime pockets", the ρ
gate, and the SKAB basin lift as its witness) cannot stand as written; the narrowed fallback in the
coordinator's question ("basin head improves SKAB-hard detection independent of ρ") is not supported
either, because the lift needs the floor as much as ρ does.

What remains true and usable: the basin head is a no-op on WADI/HAI/SWaT under every floor (λ = 0), so
removing the A3 component changes no benchmark headline number; only SKAB rows that used the gated
head (`basin_lam` > 0, seeds 0-3 at the reported config) need re-derivation with the head off.

Compute for §7: 15 SKAB fits + 2 WADI fits, about 4 min CPU, one process.
