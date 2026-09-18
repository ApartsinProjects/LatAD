# Cranfield A8 (between-regime overlap) screen

## (a) Finer Kaggle dataset search

Searched Kaggle (`kaggle datasets list -s "three phase flow"`, `-s "multiphase flow"`,
`-s cranfield`, `-s "flow facility"`, `-s "air water flow"`, `-s "flow regime"`) using the
provided API token. Only one Cranfield-relevant hit: `afrniomelo/cranfield`
("Cranfield Multiphase Flow Facility", 16,955,277 bytes). Listed its files directly
(`kaggle datasets files afrniomelo/cranfield`):

```
FaultyCase1.mat  FaultyCase2.mat  FaultyCase3.mat  FaultyCase4.mat
FaultyCase5.mat  FaultyCase6.mat  Training.mat
```

Same 7 files, same sizes (within a few hundred bytes — packaging/checksum noise) as the
MATLAB-Central copy already on disk at
`poc/datasets/_new/Cranfield/CUcasestudy/CUcasestudy/`. No other dataset on Kaggle
matches the Ruiz-Carcel Cranfield rig or exposes the full 4x5=20-point air/water
flow-rate operating grid; `afrniomelo/pronto-benchmark` is a different (PRONTO)
benchmark, not Cranfield. Also checked `IEEE DataPort` / general terms above; nothing
finer surfaced.

**Verdict: no finer Kaggle dataset exists.** Nothing downloaded to `Cranfield_fine/`.
Proceeded with the existing data: `Training.mat` normal regimes T1 (10,372 x 24),
T2 (9,825 x 24), T3 (13,200 x 24), 1 Hz, 24 process variables.

## (b) K-sweep: responsibility entropy / max-responsibility on Cranfield

Method: pooled T1+T2+T3 as train-normal (3,336 windows total after windowing,
W=20 s / stride=10 s, `winfeat.window_features(..., "stats")`, per-channel
standardization on the pooled train-normal), VaDE (`models_vade.train_vade`,
latent_dim=8, epochs=40, warmup=8, seed=0, CPU), K swept 8/16/32/64/128. Script:
`poc/_diagnostics/cranfield_a8_screen.py`, results in
`poc/_diagnostics/cranfield_a8_screen.json`.

| K | mean max-resp | H_norm (entropy) | rho (<0.5) | rho (<0.6) |
|---|---|---|---|---|
| 8   | 0.937 | 0.080 | 0.011 | 0.038 |
| 16  | 0.952 | 0.048 | 0.011 | 0.037 |
| 32  | 0.957 | 0.033 | 0.006 | 0.032 |
| 64  | 0.959 | 0.027 | 0.007 | 0.027 |
| 128 | 0.967 | 0.018 | 0.003 | 0.017 |

Reference points (established, `a3_rho_screen.json` / `a3_ksweep.json`):
- SKAB: entropy 0.06 (K8) rising to 0.53 (K64) — the one dataset that showed high overlap
  (later traced to a variance-floor artifact, see below).
- WADI: ~0.03-0.07 across K — no overlap.
- HAI: ~0.08-0.15 across K — no overlap.
- SWaT: ~0.01-0.02 across K — no overlap.

Cranfield's entropy (0.018-0.080) and mean max-responsibility (0.94-0.97, i.e. >93%
of each window's probability mass sits on a single VaDE component at every K) sit
squarely inside the WADI/HAI/SWaT band, not anywhere near SKAB's. Windows are
crisply assigned to one operating regime; `rho` (fraction of windows in a thin
between-component pocket) never exceeds 1.1%, versus SKAB's double-digit-percent
pockets.

## (c) Variance-floor sanity check

Picked the highest-entropy K from the sweep (K=8, H_norm=0.080) and re-ran seeds 0-2,
each time also recomputing responsibilities at the *same* encoder latent z but with
per-component variance **empirically estimated** from the assigned points
(`var_hat`) instead of the model's `exp(logvar_floor)=0.05` floor — the exact
procedure that exposed the SKAB "A8 witness" as a floor artifact in
`fable_a3_floor_resolution.py`.

| K | seed | H (floored) | H (empirical) | rho (floored) | rho (empirical) | frac components at floor | collapse? |
|---|---|---|---|---|---|---|---|
| 8 | 0 | 0.080 | 0.065 | 0.011 | 0.003 | 0.00 | No |
| 8 | 1 | 0.054 | 0.044 | 0.003 | 0.002 | 0.06 | No |
| 8 | 2 | 0.054 | 0.041 | 0.004 | 0.002 | 0.09 | No |

There is nothing to collapse: entropy was already low under the floored fit (0.05-0.08,
same order as WADI/HAI/SWaT), and it stays low (in fact drops slightly further) under
the empirical-variance refit. Only 0-9% of components ever sit exactly at the variance
floor, versus the SKAB case where the floor artifact came from a large fraction of
components collapsing onto it. No artifact to rule out here because there was no high
entropy to begin with.

## (d) Verdict

**A8 (between-regime overlap) is NOT observed on Cranfield.** Its three normal
operating regimes (T1/T2/T3) are cleanly separable in the VaDE latent at every K
tested (entropy 0.018-0.080, mean max-responsibility 0.94-0.97), matching the
WADI/HAI/SWaT "no overlap" band rather than SKAB's high-overlap band, and the
variance-floor refit confirms this is a real (not floor-artifact) result — there
is no elevated entropy for the refit to unmask as spurious.

Caveat carried over from `cranfield_download.md`: this uses the 3 coarse normal
regimes (T1/T2/T3) available in this packaging, not the full 20-point 4x5 air/water
flow-rate grid described in the Ruiz-Carcel paper — no dataset exposing that finer
grid was found on Kaggle or elsewhere in this pass. If a finer regime grid ever
surfaces, it remains the more decisive test for A8 on this rig; on the data
available now, the result is a clean absence, consistent with every other
real-world dataset screened to date (WADI, HAI, SWaT). SKAB is the only outlier,
and it too resolved to a floor artifact under the same check.

Paper and shipped model untouched (registry/diagnostics-only finding, per instructions).
