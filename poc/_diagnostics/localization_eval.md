# Subsystem-localization evaluation (LatAD section 7 claim)

**Claim under test (section 7).** LatAD's regime-community model produces a per-community
surprise score, so an alarm can be localized to the "most-surprised" correlation-community
subsystem rather than only a plant-wide flag. The claim is asserted but never measured. Here
we measure whether the *most-surprised community actually contains the attacked channel(s)*,
on WADI_clean (44 communities), HAI (28), SWaT_canon (25).

Reproduce: `python _diagnostics/localization_eval.py` (writes `_diagnostics/loc_results/`).
Per-window rows in `loc_<ds>.jsonl`, per-dataset summary in `loc_summary.jsonl`, synthetic
control in `synthetic_control.json`.

---

## 1. Community -> channel recovery

Membership is stored directly in the expert artifact (`sota_bundle/experts_full/expert_<ds>.npz`,
key `comm_channels`, a padded `(S, mx)` array of raw-channel indices per community), so no
reconstruction was needed. The channel index space is the `eda_real.load(ds)['ch']` ordering;
`eda_real.load` reproduces each `ens_bundle/bundle_<ds>.npz` exactly (`Xn_w` allclose, `ya_w`
identical), so `comm_channels` indices map cleanly to named sensor tags.

| Dataset | Communities S | Paper | Match | Raw channels | Channels in >=1 community | Avg communities / covered channel |
|---|---|---|---|---|---|---|
| WADI_clean | 44 | 44 | yes | 122 | 76 (62%) | 3.8 |
| HAI | 28 | 28 | yes | 59 | 49 (83%) | 4.2 |
| SWaT_canon | 25 | 25 | yes | 51 | 41 (80%) | 4.2 |

Two structural facts drive everything below:

- **Communities overlap.** They are nested HAC subtrees (size 3-25), so a channel sits in ~4
  communities on average. The correct random baseline is therefore **coverage-aware**, not `1/S`
  (see section 3).
- **Coverage is incomplete.** 18-38% of raw channels are in *no* community (they are near-constant
  in train-normal, or fall only in subtrees outside the size-[3,25] band). An attack whose most
  deviating channel is uncovered cannot be localized to any community at all.

## 2. Per-community surprise

`test_surprise` has shape `(nseed=5, S, ntest)`: each community's standardized surprise on each
test window (already z-scored per community against its own calibration tail inside
`modal_experts.py`, so cross-community argmax is apples-to-apples). We average over seeds and rank
the S communities per window. Most-surprised community = argmax over S.

## 3. Attack-target source and baseline

**Source: PROXY (not the published attack tables).** For each attacked window we take the target
channel to be the one with the largest standardized deviation, `max` over the six per-channel
stat-blocks (level / variability / min / max / net-trend / range) of `|z|`, standardized on
train-normal window features. This is *construct-matched* to the detector: it lives in the exact
feature space the communities and surprises are built on, so an attack that perturbs dynamics
rather than level still registers. **This proxy is somewhat favorable to the detector** (surprise
and channel deviation share the deviation driver), so a *low* score under it is strong evidence
against the claim; a *high* score should be tempered. (A published ground-truth table would be
strictly better where the attacked point is named; the iTrust SWaT/WADI and HAI tables were being
retrieved in parallel. The proxy is the primary metric here per the stated fallback.)

**Coverage-aware random baseline.** For a target covered by `kc` communities out of S, a random
size-k community shortlist hits with probability `1 - C(S-kc, k) / C(S, k)`. We average this over
the evaluated windows/episodes and report it next to the observed accuracy. The naive `k/S` is
also shown; it *understates* chance 3-4x and should not be used as the reference.

## 4. Localization accuracy (window level)

Accuracy is over ALL attacked windows (a window whose proxy-target is uncovered counts as a miss).
`rand` = coverage-aware random baseline. A hit@k = any community containing the target channel is
in the top-k most-surprised.

| Dataset | n win | coverage | top-1 | rand-1 | top-3 | rand-3 | top-5 | rand-5 |
|---|---|---|---|---|---|---|---|---|
| WADI_clean | 56 | 0.48 | **0.036** | 0.026 | 0.179 | 0.074 | 0.286 | 0.117 |
| HAI | 652 | 0.84 | **0.436** | 0.137 | 0.683 | 0.348 | 0.759 | 0.494 |
| SWaT_canon | 233 | 0.91 | **0.009** | 0.105 | 0.129 | 0.284 | 0.777 | 0.426 |

Reading, per dataset:

- **HAI: genuine top-1 localization.** 0.436 vs 0.137 random = **3.2x chance**; top-3 0.683 vs
  0.348; median best-rank of the target community = **1**. The most-surprised community is the
  attacked subsystem more often than not.
- **WADI_clean: no top-1 localization.** 0.036 is at the 0.026 random baseline. There is weak
  aggregate signal (on the covered subset the target community's mean rank is 5.4 vs 16.3 random),
  but the #1 slot does not find it, and **only 48% of attacked windows are coverable at all**, so
  even a perfect ranker would top out near 0.5.
- **SWaT_canon: below chance at top-1 and top-3.** 0.009 vs 0.105 and 0.129 vs 0.284. The cause is
  not a bug (the synthetic control passes and the ranking direction is verified): **two small
  communities (comm 1 and comm 17) are the top-1 most-surprised in 100% of the 233 attacked
  windows.** The surprise maximum is a globally-loud community, essentially constant across attacks,
  so it lands on the attacked channel's community *less* often than a random community would. Only a
  top-5 shortlist recovers useful signal (0.777 vs 0.426).

The same "loud-community monopoly" is milder on HAI (two communities hold 57% of top-1 slots) and
weak on WADI (top community only 12%), which is exactly why HAI localizes at top-1 and WADI/SWaT do
not.

Concretely on SWaT, the two perpetually-top communities are `{P402, UV401, P501}` (stage 4/5) and
`{MV101, AIT201, MV201, P201}` (stage 1/2), while the single most common proxy-attack target is
`MV304` (109 of 233 attacked windows, stage 3), which is in neither. The surprise maximum points at
a different stage than the one being attacked, which is the direct cause of the below-chance top-1.

## 5. Difficult subset

Restricted to `hard` windows (attack windows NOT trivially separable by max|z|):

| Dataset | n hard | top-1 | rand-1 | top-3 | rand-3 | top-5 | rand-5 |
|---|---|---|---|---|---|---|---|
| WADI_clean | 43 | 0.047 | 0.033 | 0.186 | 0.092 | 0.302 | 0.145 |
| HAI | 167 | 0.443 | 0.158 | 0.587 | 0.399 | 0.754 | 0.566 |
| SWaT_canon | 85 | 0.012 | 0.144 | 0.341 | 0.378 | 0.741 | 0.552 |

The difficult subset tracks the full set: HAI keeps its ~2.8x top-1 lift on hard windows;
WADI stays at chance; SWaT stays below chance at top-1 and only reaches useful accuracy at top-5.

## 6. Episode level

Attacks grouped into contiguous runs of attacked windows; per-community surprise averaged over the
episode's windows, then ranked. Target = peak deviation channel over the episode.

| Dataset | n episodes | coverage | top-1 | rand-1 | top-3 | rand-3 | top-5 | rand-5 |
|---|---|---|---|---|---|---|---|---|
| WADI_clean | 11 | 0.27 | 0.000 | 0.029 | 0.091 | 0.078 | 0.091 | 0.118 |
| HAI | 38 | 0.66 | 0.079 | 0.116 | 0.447 | 0.288 | 0.579 | 0.403 |
| SWaT_canon | 25 | 0.76 | 0.000 | 0.101 | 0.080 | 0.267 | 0.480 | 0.395 |

**Episode-level localization does not hold on any dataset.** Even HAI, which localizes per-window,
drops to at/below chance at top-1 once surprise is averaged over an episode (the episode mean is
again dominated by the loud communities). Top-3/top-5 on HAI stay modestly above chance; WADI and
SWaT are at or below chance except SWaT top-5.

## 7. Synthetic control (pipeline sanity)

Placing an artificial surprise maximum on a community that *contains* the proxy target must fire
top-1; placing it on a community that does *not* contain the target must miss. Both hold
(`positive_localizes=True`, `negative_misses=True`, `PASS`), confirming the argmax-over-communities
and channel->community membership logic read the surprises correctly. So the poor top-1 numbers on
WADI/SWaT are a property of the surprise ranking, not of the evaluation code.

## 8. Verdict

**Does LatAD localize faults to the correct subsystem, and by how much over chance? Partly, and only
on one of three datasets at the operating point the claim actually states.**

- The literal claim, "localize to the **most-surprised** (top-1) community," holds on **HAI**
  (0.44, 3.2x the coverage-aware baseline, including on hard attacks) but **fails on WADI_clean**
  (at chance) and **fails on SWaT_canon** (below chance, because one or two globally-loud
  communities monopolize the top rank).
- A relaxed "**top-5 shortlist**" reading is defensible on HAI (0.76) and SWaT (0.78) but not WADI
  (0.29, and only 48% of attacks are even coverable).
- **Episode-level** localization does not hold on any dataset at top-1.
- These numbers come from a **detector-favorable proxy**; the true, published-target numbers are
  unlikely to be higher.

**Recommendation: soften the deployment claim.** Section 7 should not assert that the most-surprised
community identifies the attacked subsystem in general. Supported wording, if the point is kept:

> On HAI, the single most-surprised correlation community is the attacked subsystem in 44% of
> attack windows (3.2x a coverage-aware random-community baseline of 14%, and 44% on the difficult
> subset); on SWaT a top-5 community shortlist contains it 78% of the time. Localization is
> dataset-dependent: it does not hold at top-1 on WADI or SWaT, where the surprise maximum is
> dominated by a few globally-loud communities, and it weakens to about chance when surprise is
> pooled over a whole attack episode. We therefore present community surprise as a *triage shortlist*
> rather than a single-subsystem locator.

If instead the claim is meant to be a headline capability, it is **not supported** by the evidence
and should be removed or demoted to future work, since it holds at top-1 only on HAI and only per
window.

### Optional table for section 7 (if the softened wording is kept)

| Dataset | S | top-1 | random top-1 | top-5 | random top-5 |
|---|---|---|---|---|---|
| HAI | 28 | 0.44 | 0.14 | 0.76 | 0.49 |
| SWaT | 25 | 0.01 | 0.11 | 0.78 | 0.43 |
| WADI | 44 | 0.04 | 0.03 | 0.29 | 0.12 |

(Per attack window; target channel = max standardized deviation over 6 stat-blocks; random baseline
is coverage-aware for overlapping communities.)

## Notes / honesty

- Random baseline is coverage-aware (`1 - C(S-kc,k)/C(S,k)`), not `1/S`; the naive `1/S` would have
  inflated every "lift" 3-4x.
- Target proxy uses the 6-stat-block max deviation. A mean-block-only proxy gives similar WADI/SWaT
  conclusions but understates HAI (top-1 0.15 instead of 0.44), because HAI attacks perturb
  variability/range more than level. The 6-block proxy is the construct-matched choice.
- Coverage gaps (WADI 48%, SWaT 91%, HAI 84%) cap the achievable accuracy; WADI's low coverage
  alone bounds its top-k well below 1.
- The synthetic control validates the reading pipeline but not the VaDE surprises themselves; the
  surprises are taken as produced by the paper's expert build.
