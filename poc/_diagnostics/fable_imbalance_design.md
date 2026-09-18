# Induced-imbalance stability experiment: design review (Fable, 2026-09-17)

Report-only. Nothing in the paper, model, or checkpoints was touched. Smoke artifacts:
`fable_imbalance_smoke.py`, `fable_imbalance_smoke_SKAB.log`, `fable_imbalance_smoke_SKAB.json` (this dir).

## A. Verdict

**Yes, with changes.** The experiment is well-posed once (1) the knob is a *targeted* log-ratio starvation of one
pre-screened, isolated, populous regime rather than a global power tilt, (2) hold-N-by-upsampling is dropped in
favour of a quantity-matched uniform-subsample control arm, (3) the starved regime's valid windows come from a
held-out slice of *train* (not from the test set), and (4) the primary statistic is the area under the
degradation curve of the matched-threshold FPR, paired across seeds, with a rare-vs-anomaly AUROC non-inferiority
guard.

**Single most likely outcome: NULL against AE (parity or LatAD slightly worse), a slope gap only against IF.**
Three independent facts point the same way:

1. The prior HAI sweep (`e2_fable_sweep.jsonl`, 2 seeds, 3 populous target regimes starved from 9.7% of train
   down to 2 windows = 0.014%) shows LatAD's FPR on the starved regime's valid windows **flat at 0.000 at every
   ratio**, both at the train-99pct threshold and at matched 1% FPR. Purity of the components holding those
   windows fell 0.80 to 0.45 and the number of holding components rose 6 to 9: the starved windows were absorbed
   by neighbouring components and still scored under threshold. On HAI the starved regimes are in-support of
   their neighbours, so starvation moves nothing, for any method. A flat curve for everyone is an uninformative
   null, not a stability win.
2. Mechanism ranking of mass-sensitivity. IF (path length shortens wherever training mass is sparse) is the most
   mass-sensitive; LatAD's reported score is an absolute latent density (high-K diagonal GMM, `fit_latent_density`)
   plus the nearest-mode diagonal NLL, both of which lose a component once local mass falls below the head's
   resolution; AE reconstruction error is the least mass-sensitive (an MLP with a 16-d bottleneck interpolates a
   region it saw a few times, and with zero examples its error depends on distance to the learned manifold, which
   is the same coverage signal the density heads see). Predicted degradation order (fastest first): IF > LatAD >=
   AE. A LatAD-vs-IF gap is likely but not paper-grade; a LatAD-vs-AE gap is the one that would matter and the
   mechanism predicts it is absent or reversed.
3. The E2 diagnostics established that VaDE does not allocate a thin low-pi component to a rare regime; the
   pi-only counterfactual in the sweep gave an ideal-case gap of exactly 0. LatAD therefore carries no
   rarity-immune head; under starvation it is an absolute-density detector like the rest.

Under the wins-only rule, the expected deliverable of this experiment is a registry entry ("all methods degrade
at the same rate; regime coverage is a data property"), usable as one rebuttal sentence. The one place a real
positive is plausible is synthetic `miim_gen`, where regimes are isolated by construction and rarity already
costs LatAD 5x FPR (`e2_fable_miim.jsonl`: rare-gt-normal FPR 0.125 vs plain 0.026): there the curve has
dynamic range and the LatAD-vs-AE comparison is a live question, but any win there is scoped to data satisfying
A6, not to the three benchmarks.

## B. Rigorous definition

### B1. Regimes (external, fixed, method-independent)

- Features: the paper's windowed stats features `Xn_w` (W=60, stride 30), standardised with the *full* train
  mean/sd (frozen for all beta). PCA fitted on train to `d' = 8` (SKAB) or the number of components reaching 90%
  variance capped at 16 (HAI/WADI/SWaT/miim).
- Clustering: k-means (`n_init=10`, seeded by the split seed) on the PCA scores of train; K chosen by silhouette
  over 3..12 **on train only**. Test and validation windows are assigned by nearest centroid. One partition per
  (dataset, split seed), shared by every method and every beta. Ground-truth mode labels replace k-means on
  `miim_gen`.
- Target-regime eligibility, decided from geometry alone before any method is scored (this is what keeps it
  out of the test-set-tuning category):
  - share in [3%, 15%] of train (populous enough to leave >= 50 held-out valid windows, rare enough that removing
    it changes N by <= 15%);
  - isolation index `I = d(centroid_T, nearest other centroid) / mean within-T distance-to-centroid >= 2`;
  - support gap: median kNN (k=10) distance from T's windows to the *rest* of train exceeds the 95th percentile
    of the within-rest kNN distance. If no regime passes, the dataset is declared "no dynamic range" and is not
    run. On SKAB the only rare k-means regime has 5 windows (1.2%) and fails the share rule; the HAI regimes
    starved in the earlier sweep fail the support-gap rule (that is why their curves were flat).

### B2. The knob

Primary knob (targeted): `r = 10^-beta`, `n_T(beta) = round(r * n_T(0))`, other regimes untouched, beta in
{0, 0.5, 1, 1.5, 2, 3, inf} (inf = regime removed). Down-sampling of T is a seeded permutation prefix so that the
sets are nested across beta (monotone by construction, one fewer noise source).

Secondary knob (the proposed global tilt), kept only as a robustness arm: `s_c(beta) = p_c^(1+beta) / sum_k
p_k^(1+beta)`, `n_c = floor(s_c N)` with the rounding remainder given to the largest regime. The smoke shows its
two defects: on SKAB it drives the 5-window regime to 0 already at beta=0.5 (no resolution when the smallest
regime is tiny), and at beta=4 it also starves the second regime (95 to 1 window), so "which regime is rare"
changes with beta.

### B3. Data quantity: no upsampling; a quantity-matched control arm instead

Hold-N by upsample-with-replacement is rejected. Smoke evidence: at beta=4 the tilt produced 25% duplicate rows
and the train-99pct threshold collapsed for IF and AE (other-regime FPR 0.000 to 0.347 for both), because
duplicated points are memorised (lower reconstruction error, denser leaves), tightening the threshold. Duplicates
also spike every kNN/GMM density estimate and break the iid assumption of any bootstrap. Loss re-weighting is not
uniform across methods either (sklearn `GaussianMixture.fit` takes no `sample_weight`; the AE loss and IF do), so
it would introduce a method-dependent implementation difference.

Instead: total N is allowed to fall by at most `n_T(0)` (<= 15% by the eligibility rule), and every (beta, seed)
gets a **uniform-control arm**: a uniform random subsample of the beta=0 train set with the same N as the starved
set. The imbalance effect is `starved - uniform` at equal N. The uniform arm must be flat in beta (see invariants).

### B4. Splits

Before any starvation, hold out from train a validation slice: 30% of the target regime's windows (>= 50) and
15% of the rest, by the split seed. Val-target windows are valid normal windows of the starved regime by
construction, immune to test-set regime drift (HAI regime 21 is drifted; SKAB's test has 5 normals in its rare
regime). The test set is used only for difficult-subset AUROC and the rare-vs-anomaly AUROC.

### B5. Frozen quantities (must not move with beta)

Feature standardisation (full-train mean/sd); the difficulty mask (`max|z| > 99th pct of full train`); VaDE
K, latent dim, epochs, warm-up; `k_density` (freeze at its beta=0 value, do not let `min(80, N//10)` shrink);
AE architecture and epoch count; IF trees; the regime partition. Only the training row set changes.

### B6. Metrics per (dataset, beta, seed, arm, method)

1. `FPR_T^matched`: fraction of val-target windows above the threshold that gives 1% FPR on val-other. Primary.
2. `FPR_T^op`: same at the operational train-99pct threshold (reported, secondary; mixes threshold drift with
   score drift).
3. `dS_T`: median score of val-target, z-scored against val-other, minus its beta=0 value. Continuous, not
   granular, usable when val-target is small.
4. `AUROC_rare_vs_hard`: val-target (label 0) vs difficult test anomalies (label 1). The guard: any lever that
   lowers rare FPR by flattening the score also hides anomalies (E2 lesson), so a "graceful" FPR curve only counts
   if this stays non-inferior.
5. `AUROC_diff`: difficult-subset test AUROC (overall detection health).
6. Mechanistic covariates: number and purity of VaDE components holding val-target; median kNN distance from
   val-target to the starved train set (the support distance, the method-independent driver).

### B7. Degradation statistic

FPR curves saturate at 1, so a straight-line slope on beta is ill-defined. Primary statistic: **AUDC** = mean of
`FPR_T^matched` over the finite beta grid (trapezoid in beta), one number per (method, seed, arm). Secondary:
OLS slope of `dS_T` on beta over the pre-saturation range (beta <= 2), which is the continuous analogue.

Comparison: the same starved sets feed all methods, so differences are paired by (split seed, model seed).
`Delta = AUDC_LatAD - AUDC_AE` per seed; 95% CI by paired bootstrap over seeds (>= 5 split seeds x 2 model seeds)
and a sign-flip permutation test on the paired differences (exact for 10 pairs); within-seed uncertainty from a
bootstrap over val-target windows, reported separately. Direction is pre-registered as two-sided (the mechanism
predicts LatAD is not better than AE).

Win criterion (pre-registered): `Delta < 0` with the 95% CI excluding 0, AND `AUROC_rare_vs_hard` of LatAD
non-inferior to AE at every beta (margin 0.02), AND beta=0 `AUROC_diff` equal to the published number for that
dataset (construct check). Anything else is registry-only.

### B8. Invariant and sanity list (each with its expected outcome stated in advance)

- Sampler: beta=0 returns the identity multiset (asserted in the smoke: OK); `n_T(beta)` non-increasing and nested
  (OK); no duplicate rows at any beta (no upsampling); `sum n_c == N_beta` exactly.
- beta=0 reproduces each method's baseline **bitwise** (same seed, same rows, `max|score diff| == 0`), and the
  uniform-control arm at beta=0 is the same run.
- Uniform-control arm flat: `|FPR_T(beta) - FPR_T(0)| <= 2 x` its seed SD for all beta. If it moves, the
  quantity confound is real and the reported effect is `starved - uniform`, never `starved` alone.
- Monotonicity: seed-mean `FPR_T^matched` non-decreasing in beta for every method (Spearman rho > 0 with the
  grid); a decrease beyond seed noise is a bug (threshold or mask leaked from the starved set).
- beta=inf gives the maximum of each curve.
- Negative control: a pure coverage detector (kNN distance to the starved train set) must degrade the most;
  its `FPR_T(inf) - FPR_T(0)` defines the dataset's dynamic range and must exceed 0.3, else the dataset is
  reported as "no dynamic range" and no method comparison is made.
- Positive control: an oracle trained on the *full* beta=0 train (kNN distance to full train) must be flat in
  beta; any movement means the pipeline leaks starvation into the evaluation (threshold, mask, or scaling).
- A constant score has undefined matched FPR; guard against ties at the threshold by reporting the fraction of
  tied scores.
- A suspiciously clean result (all methods exactly 0 or exactly 1 at every beta, as in the HAI sweep and the SKAB
  smoke) is treated as absence of dynamic range, not as a finding, until the negative control confirms the range.

## C. Confounds and failure modes not in the original design

1. **Duplicate inflation** (demonstrated): 25% duplicates moved IF and AE other-regime FPR from 0.000 to 0.347
   through threshold collapse; duplicates also spike kNN/GMM densities and invalidate bootstraps. Fix: B3.
2. **Knob resolution**: a power tilt on a dataset whose smallest regime is tiny zeroes it in the first step and
   starves other regimes at larger beta (SKAB smoke). Fix: targeted log-ratio knob, B2.
3. **In-support targets give flat curves for everyone** (HAI sweep: 0.000 down to 2 windows). Fix: the isolation
   and support-gap pre-screen, B1, plus the coverage negative control that measures dynamic range.
4. **Ceiling at beta=0**: a regime that is already out of support (SKAB's 5-window regime: LatAD rare FPR 1.0 at
   beta=0, IF/AE 0.6) has no room to degrade. Fix: the share >= 3% rule.
5. **Test-set regime drift / absence**: SKAB has 5 test normals in its rare regime (FPR granularity 0.2); HAI's
   rare test regime is drifted. Fix: val-target from train, B4; the test set is used for AUROC only.
6. **Threshold drift**: the train-99pct threshold moves as train composition changes, so `FPR_T^op` conflates
   threshold and score movement. Fix: matched threshold on val-other as primary.
7. **Difficulty mask drift**: recomputing `max|z|` percentiles on the starved train shifts the hard subset.
   Fix: frozen at beta=0, B5.
8. **Standardisation drift**: recomputing mean/sd on the starved train moves every feature. Fix: frozen.
9. **beta x hyperparameter interaction**: `k_density = min(80, N//10)` and VaDE K would change resolution with N.
   Fix: frozen, B5. Report a K-sensitivity arm (K/2, 2K) once, at beta in {0, 2} only.
10. **Small-N FPR noise**: report the continuous `dS_T` alongside FPR; require >= 50 val-target windows.
11. **k-means clusters are not regimes**: a cluster can cut a continuous manifold; starving half a manifold
    leaves the other half in support. The support-gap rule catches this; the isolation index alone does not.
12. **Selection leakage**: choosing target regimes after seeing any method's scores is tuning. Selection is by
    geometry only (B1), logged before the first model is trained.
13. **Seed structure**: split seed and model seed are crossed; the paired test must pair on both.
14. **AE optimisation under changing N**: full-batch 40 epochs is 40 gradient steps regardless of N, so the AE
    is equally under-trained at every beta (fine), but with upsampling it would not be (another reason for B3).
15. **The "graceful FPR" trap**: a flatter FPR curve is worthless if it comes from a flatter score (E2: every
    rare-FPR lever hides anomalies). The rare-vs-hard AUROC guard is mandatory.
16. **Multiple datasets and metrics**: one pre-registered primary (`AUDC` of `FPR_T^matched`, LatAD vs AE);
    everything else descriptive.

## D. Can it rescue E2 or demonstrate A6?

**It cannot rescue E2.** E2's claim was head-specific: the nearest-component NLL ignores pi and therefore should
raise fewer alarms on rare valid windows than the pi-weighted mixture. The diagnostics showed both heads sit on
the same components and VaDE never allocates a thin component to the rare regime; the pi-only counterfactual gap
was 0 at every ratio. This experiment compares LatAD to other methods, not LatAD's heads to each other, and it
changes nothing about component allocation, so the E2 mechanism stays absent. Adding the nearest head as a fourth
"method" would reproduce the E2 parity, not overturn it.

**It can support a weak form of A6 only if the mechanism prediction is wrong.** Precise claim it would license:
"Under induced rarefaction of an isolated, valid operating regime (r from 1 to 10^-3), LatAD's matched-FPR
degradation area is lower than AE's by Delta (95% CI excluding 0, paired over 10 seeds) with no loss of
rare-vs-hard-anomaly ranking, on <datasets that passed the dynamic-range screen>." This is a stability property
of the latent density head, not evidence that LatAD models rare regimes (E2 shows it does not). If it holds only
on `miim_gen`, the scope is "data satisfying A6 by construction" and the claim goes in the MIIM validation
section, not the benchmark tables. If the outcome is parity (the expected case), the legitimate use is one
registry line and a rebuttal sentence: "an induced-rarefaction stress test shows LatAD, AE and IF degrade at
statistically indistinguishable rates; the rare-regime limit is training coverage, not the detector".

## E. Run plan (minimum compute)

Pilot first (agent time ~15 min; compute ~2 min CPU, single thread, will not contend with `adaptive_density.py`):

1. `miim_gen` seed 0 and SKAB: run the regime split and eligibility screen; print share, isolation index,
   support gap for every regime; pick the target by rule; abort the dataset if none passes.
2. beta in {0, inf} only, 1 seed: coverage negative control and full-train oracle. Check dynamic range > 0.3 and
   oracle flat. This decides whether the dataset can carry the experiment at all.
3. beta=0 bitwise-baseline invariant for LatAD (`anomaly_score_hard`, paper config), AE (`ae_scores`), IF.
4. The uniform-control arm at beta=inf: must be within seed noise of beta=0.

Full run only for datasets that pass the pilot:

- Datasets: `miim_gen` (isolated modes, real dynamic range; the live LatAD-vs-AE question), HAI (targeting
  the drifted regime-21 block if it passes the support-gap rule, otherwise "no dynamic range"), SKAB with a
  populous target only if one passes the isolation rule (the smoke suggests K=3 with one 75% mode; unlikely).
  WADI/SWaT: too few regimes (registry); run the screen, expect exclusion, report it.
- Grid: beta in {0, 0.5, 1, 1.5, 2, 3, inf}; 5 split seeds x 2 model seeds; arms: starved, uniform-control;
  methods: LatAD, AE, IF, kNN-coverage (negative control), full-train oracle (positive control).
- Compute (single core, from the smoke timings: SKAB 1-3 s per beta for all three methods; miim ~15 s per VaDE
  on 3k windows; HAI ~30 s per VaDE on 15k): SKAB ~3 min, miim ~35 min, HAI ~70 min. Under 2 CPU-hours total,
  no cloud. Agent execution ~45-60 min including analysis. Append one JSON line per (dataset, beta, seed, arm)
  as it completes, resumable by key.
- Analysis: AUDC per (method, seed); paired Delta LatAD-AE with bootstrap CI and sign-flip test; the
  non-inferiority guard; the invariant table; the mechanistic covariates plotted against FPR to show the
  support-distance driver.

## F. Feasibility smoke (SKAB, 1 seed, single thread, ~10 s total)

Pipeline behaviour only; the numbers are 1 seed on 5 windows and say nothing about methods.

- External regimes: PCA(8) + k-means, K=3 by silhouette (0.605); train shares 0.238 / 0.750 / 0.012; test-normal
  counts 129 / 243 / 5. The rarest regime has 5 train and 5 test-normal windows: fails the share rule, FPR
  granularity 0.2.
- Sampler invariants: beta=0 identity multiset OK; rare share monotone non-increasing OK; the global tilt zeroed
  the rare regime at beta=0.5 and reduced regime 0 from 95 to 1 window at beta=4 (knob defects, B2).
- Duplicate confound: dup fraction 0.10 / 0.16 / 0.22 / 0.25 at beta 0.5 / 1 / 2 / 4; at beta=4 the train-99pct
  threshold collapsed (IF other-FPR 0.000 to 0.347, AE 0.000 to 0.347, LatAD 0.035 to 0.140).
- Ceiling: at beta=0 the 5-window regime is already out of support (LatAD rare FPR 1.0, IF and AE 0.6 at the
  operational threshold; 0.8 / 1.0 / 1.0 matched), so SKAB's natural rare regime has no room to degrade
  (consistent with the registry: tiny regimes are out of support).
- Timing: 1-3 s per beta for IF + AE + VaDE(K=16, LD=6, 40 epochs) on 400 windows.

Bottom line for the pilot: the sampler and split code are sound; the knob, the hold-N mechanism and the
target-regime selection must change as in B1-B3 before any method comparison is meaningful.
