# Regime-imbalance knob: implementation audit (Fable, 2026-09-17)

Scope: `_diagnostics/regime_imbalance_knob.py` audited against `_diagnostics/fable_imbalance_design.md`.
Report-only for the model and paper; the diagnostic script was fixed in place and is runnable. Smoke evidence:
SKAB, 1 seed, full beta grid, single thread (`regime_imbalance_smoke_SKAB.log` for the bitwise check,
`regime_imbalance_smoke_SKAB3.log` for the grid, `regime_imbalance_SKAB.json` for the per-seed records).
Nothing below was run on the full grid.

## Summary of verdicts

| # | Check | Verdict on the ORIGINAL script | After fix |
|---|---|---|---|
| 1 | beta=0 == full-data baseline | PASS (bitwise) | PASS, now asserted in code |
| 2 | Uniform control fairness | PASS on N; FAIL on the evaluation subset (see 3) | PASS |
| 3 | Rarefied-regime definition and FPR metric | FAIL (three defects) | PASS structurally; one design-level caveat remains |
| 4 | Confounds from the design doc | PASS on mask, scaling, replacement, partition; FAIL on `k_density`; small-N noise unaddressed | PASS on `k_density`; seeds still 2 |
| 5 | Construct match across methods | PASS | PASS |
| 6 | Tests the hypothesis; slope/contrast computed | FAIL (nothing beyond per-cell means) | PASS: AUDC, imbalanced-minus-uniform, paired LatAD-minus-AE, guard, TPR |

## 1. Invariant: beta=0 reproduces the full-data baseline. PASS

Expected outcome, stated before running: `sample_idx(lab, 0, ...)` returns exactly `arange(N)` for both arms, and
`score_all(Xtr[idx0], ...)` equals `score_all(Xtr, ...)` with max |difference| == 0 for every scorer (same seed,
same rows in the same order, CPU torch).

Result (SKAB, N=400, seed 0):

```
beta0 identity: True          uniform beta0 identity: True
LatAD: max|full-sub| = 0.000e+00   IF: 0.000e+00   AE: 0.000e+00   COV: 0.000e+00
nested across beta: True  sizes [400, 201, 155, 143]
```

Construct check against the published number: beta=0 LatAD difficult-subset AUROC = 0.6103 (seed 0, auto basin
lambda 1.075). The A3 witness (`a3_skab_witness.json`, same K=16, LD=6, kd=40, 185 hard anomalies) gives 0.6054 at
forced lambda 1.0 and the paper reports 0.605. Same config, consistent within the lambda difference. AE 0.504 and IF
0.528 at beta=0 are the SKAB baselines of the same pipeline (`build_scores_table.py` uses the identical calls; the
only difference, `fit_residual_whitener`, feeds `use_recon`, which is off, so it is a no-op for the score).

The original code passed this invariant by construction (`(n/nmax)**0 == 1`, `round(len*1) == len`,
`choice(idx, len, replace=False)` is a permutation, then sorted). The fix adds an explicit
`assert np.array_equal(idx, arange(N))` at beta=0 so a future edit cannot silently break it.

## 2. Uniform-control fairness. PASS on N, FAIL on the evaluated subset (fixed)

- Target N: the control drew `len(sample_idx(lab, beta, default_rng(0)))` points. N depends only on the regime sizes
  and beta (the rng only chooses WHICH rows), so the target was deterministic and identical to the imbalanced arm.
  No off-by-seeding issue: the fresh `default_rng(0)` inside the call was a separate instance from the arm's rng,
  so the uniform draw consumed an unconsumed stream. Correct, if opaque.
- The real defect was downstream: the control's `rarefied` set was recomputed from ITS OWN draw (`lab[idx]`). A
  uniform subsample cuts no regime by >50% except by chance, so the control's rare set was empty (or a random tiny
  set) and its rare FPR was `nan`. The two arms were never measured on the same test windows, so the
  "imbalanced minus uniform at equal N" contrast did not exist in the output.
- Sampling seed was fixed at 0 for every model seed, so the two seeds shared one subsample: the seed SD would have
  understated the sampling noise.

Fix: one fixed rare-regime set R* per dataset (section 3) used by both arms at every beta; the sampling rng is
`default_rng(1000 + model_seed)` so subsample and model noise are crossed; the control asserts `len(idx) ==
len(idx_imbalanced)`. Verified: `kept` is equal across arms at every beta (400/201/155/143), and the COV negative
control on the uniform arm is flat (0.032, 0.032, 0.024, 0.044) while on the imbalanced arm it rises
(0.032, 0.084, 0.296, 0.636), which is the expected behaviour of a pure coverage detector.

## 3. Rarefied-regime definition and the FPR metric. FAIL (fixed), one design caveat remains

Three defects in the original:

1. **No baseline.** `rarefied = regimes cut >50%` is empty at beta=0, so `rare_norm` was empty and rare FPR was
   `nan` at beta=0. The curve had no anchor and the doc-string's own invariant ("beta=0 reproduces the baseline")
   was untestable for the primary metric.
2. **Drifting subset.** Retention `(n_i/nmax)^beta < 0.5` admits regimes with `n_i < nmax/2` at beta=1 but
   `n_i < 0.84 nmax` at beta=4, so the set of test windows being scored grew with beta. A rising FPR curve then
   mixes score movement with subset composition (the design's confound C2, "which regime is rare changes with beta").
3. **Circular, capped threshold.** The task brief describes the threshold as train-99; the code used
   `np.quantile(s[y == 0], 0.99)`, the 99th percentile of ALL test normals, rare windows included. That fixes the
   overall test FPR at 1% by construction, so rare FPR was bounded above by `0.01 * N_normal / n_rare`: with 250 rare
   test normals out of 377 on SKAB the cap is 0.015, below the dynamic range the experiment is meant to observe. The
   rare scores also entered their own threshold.

Fixes applied: R* = regimes whose retention at the largest beta in the grid is below 0.5, computed once from the
train regime sizes and frozen for every beta and both arms; `rare_norm` is a fixed test-window set with a beta=0
value; the threshold is the 99th percentile of the NON-rare test normals (the design's "matched on val-other"
analogue), so it is never set by the rare scores and has no cap. Assigning test windows to train k-means regimes by
nearest centroid is what the design specifies and is acceptable; the design's warning stands that test-set regime
drift (HAI regime 21) makes a test-derived rare set less clean than a train hold-out, which this script does not
implement.

Remaining design-level caveat (not a bug, but it changes what the knob measures): the global power tilt with
KREG=16 makes almost every regime "rare". On SKAB the sizes are `[39, 31, 2, 24, 29, 2, 3, 53, 2, 26, 25, 2, 19,
127, 11, 5]`; R* is 15 of 16 regimes, 273 of 400 train windows and 250 of 377 test normals. At beta=4 the kept
train set is 143 windows of which 127 are the one dominant regime. The experiment therefore measures "collapse the
training set to its dominant regime and score every other regime as rare", not "starve one isolated regime". The
matched threshold is then set on the 127 dominant-regime test normals (its 99th percentile is the second-highest
score, so it is a tail statistic of 127 points). This is exactly the knob-resolution defect the design doc
rejected in B2/C2 in favour of a targeted per-regime starvation; the implementation kept the global tilt.

## 4. Confounds from the design doc. Mostly PASS; `k_density` FAIL (fixed); seeds still thin

- Difficulty mask: computed once from the full `Xtr0`/`Xte0` before the loop, never touched by `idx`. PASS (frozen).
- Standardisation: `mu, sig` from the full train, frozen. PASS.
- Duplicate inflation: `rng.choice(..., replace=False)` everywhere; the fix adds `assert len(unique(idx)) == len(idx)`
  and per-regime permutation prefixes so subsamples are nested across beta (design B2). PASS, verified nested.
- Regime partition: k-means fitted once on the full train, labels and test assignment fixed across beta. PASS.
- `k_density = min(80, max(20, N//10))` was recomputed from the SUBSAMPLED N inside `score_all`, so LatAD's density
  head changed resolution with beta (design B5 says freeze; confound C9). FAIL, fixed: `kd` is computed once from
  the full N and passed in (SKAB: 40 at every beta).
- VaDE K, latent dim, epochs, warm-up, AE architecture and its 40 full-batch steps, IF trees: fixed. PASS. Note that
  IF's `max_samples = min(256, N)` and the AE's 40 full-batch steps both interact with N; that is precisely what the
  uniform arm absorbs, so the contrast (not the imbalanced curve alone) is the reportable quantity. The 1-seed
  smoke already shows IF's uniform arm moving (0.368, 0.444, 0.092, 0.616), so the quantity confound is real for IF.
- Small-N noise: 2 model seeds, previously 1 sampling seed. Now crossed, but still 2 pairs; a paired CI is not
  possible at 2 seeds. The design asks for 5 split x 2 model seeds; the script exposes `seeds=` for that.
- Rare-set ceiling: R* includes regimes with 2 to 5 windows that are already out of support at beta=0 (design C4);
  their contribution is present at beta=0 and cannot grow, so it dilutes rather than inflates the curve.

## 5. Construct match. PASS

Per (beta, arm, seed) one `idx` is drawn and `score_all(Xtr[idx], Xte, ...)` trains LatAD, IF, AE (and now the COV
control) on that same row set and scores the same `Xte`; `hard`, `rare_norm`, `other_norm` are computed once per
dataset and reused for every method and cell. All numbers in one cell come from one pass on one config.

## 6. Does it test the hypothesis, and is the contrast computed? FAIL (fixed)

The original output held only per-cell means of `diff` and `rarefpr`; no slope, no area, no imbalanced-minus-uniform
difference, no pairing by seed, no guard against the graceful-FPR trap. Added to the script:

- per-seed records for every cell (`per_seed` in the JSON);
- AUDC = trapezoid mean of the rare FPR over the beta grid, per (method, arm, seed), plus `rise_end_minus_0`;
- `contrast_audc = AUDC_imbalanced - AUDC_uniform` per method and seed (the design's reportable quantity);
- `delta_LatAD_minus_AE_contrast` and `delta_LatAD_minus_IF_contrast`, paired by seed (two-sided, per B7);
- `rare_vs_hard` AUROC (rare normals vs difficult anomalies), the design's mandatory guard;
- `hard_tpr` at the same matched threshold, so a low rare FPR can be read against what it costs in detection;
- COV (10-NN mean distance to the sampled train) as the negative control; `dynamic_range_COV = rise(COV)` must
  exceed 0.3 before any method comparison is read (design B8). SKAB smoke: 0.604, so SKAB has range under this knob.

What the 1-seed SKAB smoke shows, and why it must NOT be read as a win. Rare FPR at the matched threshold, imbalanced
arm, beta 0/1/2/4: LatAD 0.040/0.024/0.036/0.072; AE 0.032/0.100/0.504/0.564; IF 0.368/0.620/0.616/0.644; COV
0.032/0.084/0.296/0.636. Contrast AUDC: LatAD 0.005, AE 0.328, IF 0.247, COV 0.263; delta LatAD-AE = -0.323. Taken
alone this is the hypothesised direction. The guard and the TPR say otherwise: LatAD's `rare_vs_hard` falls from
0.620 to 0.453 (rare normals rank ABOVE difficult anomalies) and its `hard_tpr` at the same threshold is
0.070/0.049/0.032/0.087, i.e. at the operating point where it raises 7% false alarms on rare normals it also catches
7-9% of difficult anomalies: the score is flat outside the dominant regime, so both rates sit near the threshold's
own tail. AE's rare FPR climbs to 0.56 while its `hard_tpr` climbs to 0.18 and its `rare_vs_hard` drops to 0.22; it
degrades by scoring rare normals high, LatAD "degrades" by scoring nothing high. This is the E2 lesson the design
names as confound C15: a flatter FPR curve from a flatter score is not stability. With 1 seed on the dataset the
design already marked as unsuitable, the numbers are pipeline evidence only.

## Bugs fixed in `regime_imbalance_knob.py` (script left runnable)

1. Rare-regime set fixed across beta and shared by both arms (was per-cell, empty at beta=0 and for the uniform arm).
2. Threshold moved from all-test-normal 99th pct (circular, capped) to non-rare test-normal 99th pct.
3. `k_density` frozen at the full-N value.
4. Nested per-regime permutation prefixes; sampling rng crossed with the model seed; no-duplicate and beta=0-identity
   asserts; equal-N assert for the control.
5. Summary block: per-seed AUDC, contrast, paired deltas, `rare_vs_hard` guard, `hard_tpr`, COV negative control and
   dynamic range; regime sizes (train and test-normal) and R* logged in the JSON; JSON written after every beta.
6. `np.trapezoid` replaced by an explicit trapezoid (the installed numpy lacks it).

Not changed (design deviations that are choices, not bugs, flagged for the owner): the knob is still the global
power tilt (design rejects it, B2/C2); rare windows come from the test set rather than a train hold-out (B4); no
per-regime eligibility screen (B1); 2 seeds (B7 asks 10 pairs); no positive-control full-train oracle (COV at
beta=0 on the uniform arm plays that role only approximately).

## Verdict

As originally implemented the experiment could not give a trustworthy answer: the primary metric had no beta=0
baseline, its evaluated subset drifted with beta, the threshold was set partly by the rare scores themselves and
capped the observable range, the uniform control was scored on a different (usually empty) window set, and the LatAD
density head changed resolution with N. Those are fixed and the beta=0 invariant holds bitwise.

As now implemented it is internally consistent and will give an answer to the question it actually asks: "when the
training set collapses toward its dominant k-means cluster, how fast does each method's matched-threshold FPR on the
other clusters rise, relative to a uniform subsample of equal size". That question is broader and blunter than the
design's targeted isolated-regime starvation, and on small datasets (SKAB) "rare" is two thirds of the data. Before
the results are used for any claim: (a) read the `rare_vs_hard` guard and `hard_tpr` alongside rare FPR, since the
smoke shows LatAD's flat FPR coincides with a flat score; (b) run at least 5 sampling seeds x 2 model seeds so the
paired delta has a CI; (c) prefer the design's targeted knob on a pre-screened regime if the claim is about rare
valid regimes rather than dominant-regime collapse. Under the wins-only rule, the SKAB smoke as it stands supports
no positive statement.
