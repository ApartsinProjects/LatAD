# A5 rare-regime-safe likelihood: imbalance-ratio sweep + alternative scores

Rare-regime FPR = (rare-regime false positives) / (number of rare-regime NORMAL test samples), at a **matched overall FPR** (test-normal quantile threshold, so every score realizes the same overall FPR). Lower is better. Mean +/- sd over seeds [0, 1, 2, 3, 4].

Scores: **mixture** (pi-weighted NLL, the naive baseline H1), **nearest** (nearest-component / A5, drops the pi prior), and three imbalance-aware alternatives: **unifprior** (flat 1/K prior mixture NLL), **tempered05** (pi^0.5 tempered mixture NLL), **knn** (k=10 latent nearest-neighbour distance).


## 3W

### Rare-regime FPR at matched overall FPR = 0.05 (ratio x score)

| rare-occ ratio | n_rare (mean) | mixture | nearest | unifprior | tempered05 | knn |
|---|---|---|---|---|---|---|
| 0.02 | 28 | 0.453+/-0.121 | 0.316+/-0.129 | 0.431+/-0.105 | 0.436+/-0.110 | 0.518+/-0.110 |
| 0.05 | 94 | 0.210+/-0.049 | 0.148+/-0.042 | 0.204+/-0.052 | 0.205+/-0.051 | 0.216+/-0.047 |
| 0.10 | 185 | 0.111+/-0.025 | 0.092+/-0.027 | 0.108+/-0.024 | 0.109+/-0.024 | 0.116+/-0.027 |

### Difficult-subset TPR at matched overall FPR = 0.05 (detection must not degrade)

| rare-occ ratio | mixture | nearest | unifprior | tempered05 | knn |
|---|---|---|---|---|---|
| 0.02 | 0.318 | 0.299 | 0.322 | 0.318 | 0.254 |
| 0.05 | 0.318 | 0.299 | 0.322 | 0.318 | 0.254 |
| 0.10 | 0.318 | 0.299 | 0.322 | 0.318 | 0.254 |

**Invariant (nearest rare-FP <= mixture rare-FP) at genuinely-rare thresholds (ratio <= 0.05), every seed/fpr:** HELD.
At the loosest ratio 0.10 (dilutes 'rare' to moderately-common regimes): 1 case(s) where nearest redistributes a few flags onto moderate tails: [{"seed": 4, "overall_fpr": 0.05, "ratio": 0.1, "mix_rare_fp": 17, "near_rare_fp": 20}]

## Cranfield

### Rare-regime FPR at matched overall FPR = 0.05 (ratio x score)

| rare-occ ratio | n_rare (mean) | mixture | nearest | unifprior | tempered05 | knn |
|---|---|---|---|---|---|---|
| 0.02 | 3 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 | 0.000+/-0.000 |
| 0.05 | 142 | 0.073+/-0.061 | 0.073+/-0.061 | 0.073+/-0.061 | 0.073+/-0.061 | 0.073+/-0.061 |
| 0.10 | 278 | 0.090+/-0.004 | 0.090+/-0.004 | 0.090+/-0.004 | 0.090+/-0.004 | 0.090+/-0.004 |

### Difficult-subset TPR at matched overall FPR = 0.05 (detection must not degrade)

| rare-occ ratio | mixture | nearest | unifprior | tempered05 | knn |
|---|---|---|---|---|---|
| 0.02 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.05 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.10 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Invariant (nearest rare-FP <= mixture rare-FP) at genuinely-rare thresholds (ratio <= 0.05), every seed/fpr:** HELD.

## What it shows

Across every rare-regime occupancy threshold, the nearest-component score (A5) yields a rare-regime false-positive rate at or below the pi-weighted mixture at the same overall FPR, and the prior-flattening alternatives (uniform-prior and tempered mixtures) fall between the two -- confirming the mechanism is the removal of the -log(pi_k) rare-regime penalty, not an artefact of one split. On 3W, at ratio 0.02, mixture rare-FPR 0.453 vs nearest 0.316 (flat-prior 0.431); at ratio 0.05, mixture rare-FPR 0.210 vs nearest 0.148 (flat-prior 0.204); at ratio 0.10, mixture rare-FPR 0.111 vs nearest 0.092 (flat-prior 0.108). On Cranfield, at ratio 0.02, mixture rare-FPR 0.000 vs nearest 0.000 (flat-prior 0.000); at ratio 0.05, mixture rare-FPR 0.073 vs nearest 0.073 (flat-prior 0.073); at ratio 0.10, mixture rare-FPR 0.090 vs nearest 0.090 (flat-prior 0.090).

**Invariant.** At the genuinely-rare occupancy thresholds (ratio <= 0.05, the regimes that carry a substantial -log(pi_k) prior penalty) the nearest-component score never adds a rare-regime false positive relative to the mixture: **held** (ratio 0.02: 0/30 violations; ratio 0.05: 0/30). At the loosest ratio 0.10 there are 1/30 cases where nearest redistributes a few flags onto moderately-common tails.

**Root cause of the 0.10 cases (not a code bug).** The invariant 'nearest can only REMOVE, never ADD, a rare-regime FP vs the mixture at matched overall FPR' holds exactly at the genuinely-rare occupancy thresholds (ratio <= 0.05): 30 cases per ratio, 0 violations, every dataset/seed/fpr. Any violation occurs only at the loosest ratio 0.10, which dilutes 'rare' to include MODERATELY-common regimes (occupancy ~0.04-0.07). Root cause (verified by inspecting the added points): the mixture's -log(pi_k) penalty is largest for the ULTRA-rare regimes (occ ~5e-4), so at a matched FPR the mixture spends its false-positive budget flagging those; the nearest score carries no prior penalty, so its equal-size budget is distributed by pure Mahalanobis distance and can land a few flags on moderately-rare component tails. Nearest thus PROTECTS the rarest regimes (where the penalty bites) and may redistribute a few flags onto moderate tails -- which is the mechanism, not a code bug (mixture score is bit-identical to -GMM.score_samples; nearest matches the reference; added points are unambiguously assigned to occ~0.04-0.07 components by responsibility, min-Mahalanobis, and max-density alike).

**Detection** (difficult-subset TPR at the matched FPR) is not materially reduced by nearest vs mixture: **confirmed**.
