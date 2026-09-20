# Negative/null-result triage (post SWaT leak-fix + PCA double-hard benchmark hardening)

Read-only synthesis. Scope: every exploratory direction in `review_round_1/RESEARCH_REGISTRY.md` and the
`_diagnostics/*.md` verdict files that returned a null/negative/inconclusive result, cross-checked against
two changes since they were written: (a) `SWaT_canon` now trains on the official Dec-2015 iTrust
`Normal_v1` (0.000 train/test-normal overlap, `_assert_swat_no_leak` guard) instead of the leaked Kaggle
`normal.csv` mirror (93-100% of test-normals were verbatim in train); (b) the "double-hard" second filter
is now PCA-based (Hotelling T2 + SPE at 95% train-normal variance) rather than the LinRes residual.
Manuscript not touched.

## How to read "root cause" and "changed by leak-fix / double-hard?"
- **bug**: a code defect (wrong mask, unclipped constant channel, stale cached artifact, non-idempotent patch).
- **leak**: the verdict was computed (wholly or partly) on the leaked SWaT_canon train set, so the number
  is provably stale regardless of its sign.
- **old-hard-def**: the verdict used the LinRes-based difficult/double-hard subset definitions that have
  since been replaced by the PCA T2+SPE filter, so the *anomaly population being measured* has changed.
- **genuinely null**: re-derivable on the current clean/hardened pipeline and not attributable to either.

---

## Triage table

| experiment | what it tried | result | root cause of the negative | changed by leak-fix / double-hard? | worth retrying now? | rough agent-time/compute |
|---|---|---|---|---|---|---|
| **SWaT nonlinear channel-wise LOO ("boosted") vs community headline** (`loo_baseline_sarfraz.md` §1.5-1.9) | Verdict "SWaT: genuine tie" between the density-community headline (0.837) and a per-channel gradient-boosted LOO baseline (0.881) on the (then) difficult subset (n=85, LinRes-based hard-def) | Numerically boosted-LOO led, CI included 0 (tie) on the **leaked** SWaT train | **leak** — the 0.881 boosted number matches the OLD-leaked column in `swat_leak_fix.md` exactly; on the official clean train boosted-LOO **collapses to 0.412** difficult (`drift_clean_swat.md` §"Full clean-official difficult-subset table") | yes, directly (leak) and indirectly (double-hard subset n changed 59→28/31) | **YES, but not a rerun of the old question** — the interesting unanswered question is whether boosted-LOO recovers under the SAME causal drift/changepoint decomposition that recovers the community headline (0.524→0.723 Difficult, `drift_changepoint_typing.md`). That fair fight has not been run. | ~20-40 min agent time (reuse existing `boosted_loo_SWaT_canon.npz` machinery + the `decompose()` function from `drift_changepoint_typing.py`, apply to the boosted score, no retraining) |
| **PCA T2+SPE double-hard filter, headline vs LinRes** (`pca_filter_doublehard.md`) | Defined the new double-hard filter and re-scored Table 4 on it | HAI verdict flips from significant to non-significant (the PCA filter removes exactly the HAI windows every learned detector, including LatAD, was doing well on); WADI/SWaT numerical leads, SWaT significant | **leak** (SWaT numbers, 85/59-window subsets, match OLD-leaked) + **old-hard-def is literally the subject** (this file defines the *new* hard-def, but ran it against the still-leaked SWaT pipeline) | yes (SWaT side) | **NO further retry needed** — `drift_changepoint_typing.md` already re-ran the PCA double-hard subset (`DoubleHard_pca`, n=28/31 clean) on the official pipeline as part of a larger, more complete study (see next row) and superseded this file's SWaT numbers | already done |
| **Drift-vs-changepoint typing** (`drift_changepoint_typing.md`) | Decompose each community's surprise into slow drift + fast changepoint residual; re-score SWaT/HAI/WADI on official-clean SWaT with the PCA-double-hard subset | **Positive, not negative** — recovers SWaT community fusion from drift-swamped 0.524→0.723 Difficult (significant, P=0.014) and 0.178→0.671 PCA-double-hard (P<0.001); HAI/WADI within their pre-registered no-regression band | n/a (this is the corrected/successor result to the two rows above) | n/a | **Already a validated win — recommend wiring into the paper** (supersedes the stale "SWaT ties the linear/nonlinear baseline" framing in `loo_baseline_sarfraz.md`). Two honest caveats it reports itself: 2/25 SWaT episodes lose era-local rank in the shadow of a 10 h attack, and the synthetic-ramp invariant is marginal on SWaT / fails on WADI's short record. | already done; ~0 further compute to adopt, some writing time |
| **Drift-gated changepoint head** (per-term train-only stationarity gate, `drift_gated_head.md`/`.json`) | Gate the changepoint decomposition per community/global term from train-only stats, instead of applying it everywhere | Primary budget (0.05) gives SWaT Difficult 0.549, far short of the un-gated CP v2's 0.723; effectively a null improvement over the plain headline (0.524→0.549) | **genuinely null** — already run on the official clean pipeline (subset sizes 91/167/30 match current clean numbers) | no (already clean) | **NO** — conclusively inferior to just always applying the changepoint decomposition (`drift_changepoint_typing.md`); the per-term gate under-fires because term-level train/test stationarity doesn't predict which terms carry test-era drift (comm-gate Pearson −0.02 vs test floor rate, same finding in `loo_baseline_sarfraz.md` §1.8) | none — already conclusively answered |
| **SWaT Table 5 (source-of-gain: cross-channel vs marginal density)** (`swat_clean_ablations.md` §1) | Cross-channel latent density beats channel-independent marginal product by +0.141 (leaked) | On official clean data the cross-channel density **collapses to chance** (0.472) and the gain **inverts** to −0.050 | **leak** | yes | **NO retry — already retried and correction needed in paper.** The paper's Table 5 SWaT row and its "source of gain" prose must be corrected or the SWaT column dropped/footnoted | already done (correction, not experiment) |
| **SWaT Table 6 (score-head ablation)** (`swat_clean_ablations.md` §2) | Reconstruction/density/nearest-component heads compared, leaked | Every head collapses toward chance on official clean data (best head, reconstruction, drops 0.835→0.567) | **leak** | yes | **NO retry — correction needed.** Table 6 SWaT column must be replaced or caveated | already done |
| **SWaT Table 7 (subsystem localization)** (`swat_clean_ablations.md` §3) | Paper claims Top-1 localization 0.46 (P=0.004), significant vs coverage-aware random | On official clean data Top-1 is 0.25-0.33, **not significant** (p=0.16-0.48) | **leak** | yes | **NO retry — correction needed.** This is the highest-severity correction in the whole set: a *positive, in-paper, significance-tested* claim collapses to null on clean data. Must be pulled from the manuscript or explicitly re-derived and reported as null | already done (a rerun exists; result is null) |
| **A8 valley smoothing (SWaT leg)** (`a8_valley_smoothing.md`, SWaT re-check inside `drift_clean_swat.md`) | Hypothesis: deep reconstructors miss anomalies landing in the inter-regime "valley"; density catches them | Null on all 3 real datasets in the original run; **re-verified null on official-clean SWaT specifically** (valley AUROC 0.28, indistinguishable from USAD/TranAD 0.25-0.26; the clean deep-baseline collapse is uniform across partitions, not valley-specific) | genuinely null (not a leak artifact — re-run confirms the same conclusion pre- and post-fix) | tested, confirmed unchanged | **NO** — two independent confirmations (real-data 3-dataset screen + clean-SWaT-specific re-check) plus a synthetic control that explains why (valleys are shallow / inside the normal envelope on WADI/HAI; SWaT valley anomalies are rare, 5% of attacks) | none — already conclusively answered twice |
| **A3 basin head / SKAB witness** (`fable_a3_spaces.md`, referenced in registry) | Hypothesized between-regime hard envelopes (A3), validated once on SKAB (ρ=0.58) | **WITHDRAWN**: the SKAB witness was a `logvar_floor` variance-floor artifact (ρ 0.58→0.00 when the floor is loosened); A3 absent on 8 real datasets screened including WADI/HAI/SWaT | **bug** (an unintended regularizer floor, not a leak or hard-def issue) | no — unrelated to SWaT/hard-def | **NO** — this is a decisive negative already fully investigated (H1+H2+floor-sensitivity+synthetic control); basin head is confirmed auto-off on all benchmarks. Correctly already removed from the paper (Table 6/A3 dropped) | none |
| **Rare-regime modeling fixes** (shrinkage/variance-floor/per-regime-threshold/kNN/merge) | Try to reduce HAI rare-regime false positives without hiding anomalies | No fix found — tiny regimes are out-of-support (a data-coverage limit); the real HAI "rare FPR" driver is one drifted regime (21), not small-sample noise | genuinely null | no — HAI-only, untouched by the SWaT leak; the hard-def change doesn't bear on this (it's about FPR on rare-regime normals, not the anomaly-difficulty axis) | **NO** — well-diagnosed structural limit (out-of-support coverage), and the productive follow-up (new-normal-regime discovery, folding regime 21 back) is already a separate, promising, non-null thread | none |
| **E2 nearest-component vs π-weighted mixture** (registry, `e2_nearest.py`) | Rare-regime-safety variant of the density head | NO WIN — parity on all 3 benchmarks; VaDE won't allocate a low-π component to a very rare regime even when forced to 0.01% imbalance | genuinely null | no — mechanism is about regime allocation at train time, orthogonal to the SWaT test-leak or the difficulty-axis definition | **NO** — already exhaustively probed (adversarial Fable audit, forced-imbalance stress test, synthetic control); registry correctly keeps it as rebuttal-only | none |
| **Regime-imbalance single-knob (β-downsampling)** (`fable_sampler_audit.md`) | Vary regime imbalance via a single knob β, measure rare-regime FPR | NULL — "graceful-FPR trap": the apparent LatAD win is the score going flat, not staying informative; rare-vs-hard AUROC drops 0.62→0.45 under the manipulation | **bug-adjacent** (a measurement artifact: the naive contrast metric doesn't detect score collapse) but the underlying finding (no real robustness win) is genuine, confirmed by the guard | mostly SKAB/synthetic-regime, not leak-affected | **NO** — the guard that caught the trap is exactly the "does the invariant hold" check the negative-result protocol asks for; a full grid was explicitly recommended against in the registry | none |
| **Locally-adaptive density head (LOF)** (registry entry, corrected once already) | Swap fixed high-K GMM for local-density-ratio scoring | First verdict ("hurts everywhere") was itself a subset-mask bug; corrected canonical rerun: helps SKAB (+0.093), ties WADI/HAI, **hurts SWaT** (−0.055) | the SWaT number is **pre-leak-fix** (this rerun predates the official-normal SWaT pipeline — dated to the same session as the density-only headline, before `swat_leak_fix.md`) | **yes, likely** — SWaT's density collapses under the clean pipeline (0.804→0.472 for global density; the whole "SWaT is the best-covered, easiest set" premise behind LOF hurting it no longer holds) | **YES** — cheap, mechanically well-understood re-run (`adaptive_density.py`, construct-matched, 5-seed) on `SWaT_canon` official-clean only; WADI/HAI/SKAB unaffected, don't re-run | ~15-20 min agent time (one dataset, one script, existing 5-seed harness) |
| **kNN-in-latent scoring** (registry, `knn_latent_verify.py`) | Swap the fixed-K density head for kNN-in-latent | No general win — construct-matched 5-seed: helps SWaT only (+0.016, "the easiest/least-discriminative set"), hurts WADI (−0.025), neutral HAI | explicitly attributed in the registry to SWaT being over-covered/easy — **that premise is falsified by the leak fix** (SWaT is no longer easy; global density is near chance on Difficult) | **yes, likely** | **YES** — same cheap re-run pattern as LOF, same script family, SWaT_canon official-clean only | ~15-20 min agent time |
| **Representation ablation (stats vs temporal window features)** (`repr_ablation.md`/`.log`) | Does the temporal (not just per-window-stat) representation drive the gain, vs the community/density architecture (§6 claim) | "temporal − stats ≤ +0.01 across datasets ⇒ gain is not from representation" — but HAI shows +0.0715 and SWaT +0.0260, both **above** the 0.01 threshold the interpretation line itself states | **ambiguous — needs a provenance check before trusting either the old verdict or a rerun.** The printed HAI/SWaT difficult-AUROC numbers (0.7426/0.8141, 0.7500/0.7760) do not match any of the current clean headline numbers (HAI 0.845, SWaT 0.837/0.524/0.723 depending on which pipeline), so this was very likely run on the leaked SWaT + a superseded HAI/WADI feature build. The stated conclusion also mischaracterizes its own HAI/SWaT deltas as "≤0.01" when they are 2-7x that. | probably (SWaT at minimum) | **YES, but fix the interpretation bug first** — before re-running, resolve why the write-up's own +0.0715/+0.026 deltas were called "≤+0.01"; this could be a copy-paste of a stale summary line, or the ≤0.01 threshold could have been checked against a different (WADI-only) row. Re-run on all 3 datasets on the current clean pipeline once the discrepancy is understood, since the §6 "gain isn't from representation" claim is currently unsupported by its own numbers | ~10 min to read `repr_ablation.py` and resolve the interpretation bug; ~20-30 min to re-run 3 datasets if a rerun is warranted |
| **Community detection-set complementarity** (`complementarity_fullhead.json`, HAI) | LatAD-exclusive vs TranAD/linres/AE detections on the difficult subset | n_difficult=167 (old LinRes-based hard-def); only_LatAD=8 of 67 caught | **old-hard-def** — computed against the pre-PCA-double-hard n=167 HAI difficult subset; HAI is not leak-affected, but the difficulty population it's scored against has changed (PCA filter takes HAI difficult from 167→55/PCA-double-hard) | yes (hard-def only, HAI not leaked) | **YES — cheap** — recompute the same catch-set/Jaccard table on the PCA double-hard HAI subset (`pca_filter_doublehard_HAI.npz` already exists) to see whether the "8 LatAD-exclusive catches" figure changes when scored against the harder population | ~10 min agent time (rerun one existing script against a different precomputed subset file) |
| **Per-term community train-only stationarity gate** (`comm_gate_eval.json`, used inside `loo_baseline_sarfraz.md` §1.8 Route 2) | Use train-only stationarity to decide which communities' boosted-LOO expert to trust at test time | Null — within-train community stationarity is uncorrelated (Pearson −0.02) with the community's actual test-recording drift rate; the lever "does not exist under the no-test-tuning bar" | genuinely null, by construction (a train-only statistic cannot see a test-recording-only shift) | not leak-sensitive as a mechanism, though its numeric inputs may be pre-leak-fix (same session as `loo_baseline_sarfraz.md`) | **NO** — the negative result is a structural, not a data, finding: no train-only statistic can detect a shift that only exists in the comparison between two different recordings. Re-running on clean data cannot fix this | none |
| **regime_imbalance_knob.py (SKAB grid)** | Follow-on run of the β-imbalance knob for SKAB | **Crashed**, `FileNotFoundError: scores_SKAB.npz` missing | **bug** (missing precomputed artifact, trivial) | no | **NO** — the underlying question (regime-imbalance robustness) is already answered NULL by `fable_sampler_audit.md`'s more careful, guarded version; this crashed run is a redundant, less-guarded attempt at the same null question, not worth resurrecting | none (would be ~5 min to fix the FileNotFoundError, but not worth it) |

---

## Prioritized retry list (highest expected value first)

### 1. Fair-fight boosted-LOO under the causal drift/changepoint decomposition (SWaT_canon)
**Why first:** this is the one place where the *old* (leaked) verdict was an explicit, citable claim in the
manuscript's positioning section ("SWaT ties a nonlinear channel-wise baseline", `loo_baseline_sarfraz.md`
§3) and where the clean-data collapse (0.881→0.412) is already known but the natural counter-question — does
the boosted-LOO baseline *also* recover once its own signal is drift-decomposed the same way the community
headline was? — has not been asked. If it does not recover, the paper gets a clean, strong "our drift
decomposition specifically rescues the community/density mechanism, not detectors in general" story. If it
does recover, the fair comparison must be reported instead of the current asymmetric one.
- **Invariant to state in advance:** applying the identical `decompose()` (causal median + causal MAD, K=24h,
  a-priori, no test tuning) to the boosted-LOO score on WADI/HAI must reproduce those datasets' headline
  numbers unchanged (I1/I2 style checks already used in `drift_changepoint_typing.py` and
  `drift_gated_head.py`) — a policy that is a no-op on stationary/non-drifting data (WADI: 0 drifting
  communities) should not change WADI's boosted-LOO score.
- **Degenerate-setting check:** K=None (no decomposition) must reproduce the raw boosted-LOO AUROC (0.412
  clean SWaT) exactly.
- Cost: ~20-40 min agent time, no retraining, reuses two existing scripts' machinery.

### 2. LOF / kNN-in-latent density-head variants, SWaT_canon official-clean only
**Why:** both were rejected specifically because "SWaT is the easiest, best-covered set" — a premise the leak
fix falsifies (global density on SWaT_canon Difficult is now 0.472-0.804 depending on decomposition, far from
"easiest"). Cheap, mechanically simple re-runs on an existing 5-seed harness.
- **Invariant:** the synthetic diffuse-valid-mode control that both variants passed before (LOF penalty ratio
  ≈1.0 on a diffuse-but-valid mode) is dataset-independent and should still pass; only the real-SWaT numbers
  are expected to move.
- **Degenerate check:** WADI/HAI/SKAB numbers, unaffected by the SWaT leak, must reproduce their prior values
  exactly (both variants are dataset-independent code paths; rerunning WADI/HAI/SKAB would be a regression
  test, not new science — skip unless the reproduction check fails).
- Cost: ~15-20 min agent time each, ~30-40 min total for both.

### 3. Community detection-set complementarity, HAI, under the PCA double-hard subset
**Why:** cheapest item on the list (existing precomputed subset file, one script rerun) and it directly
strengthens or revises a "near-disjoint detection" claim likely destined for the paper's discussion of why
the community/density model adds value beyond deep reconstruction baselines.
- **Invariant:** the new PCA-double-hard HAI population (n=55) is a strict subset of the old n=167 population
  (already asserted true in `pca_filter_doublehard.md`); the only-LatAD count on the smaller, harder subset
  should not exceed the only-LatAD count on the larger subset in absolute terms, only possibly in proportion.
- Cost: ~10 min agent time.

### 4. Resolve the repr_ablation.py interpretation bug, then decide whether to rerun
**Why:** lowest confidence item — the write-up's own printed deltas (+0.0715 HAI, +0.026 SWaT) contradict its
stated "≤0.01 ⇒ representation doesn't matter" conclusion supporting §6. This needs a fifteen-minute read of
the script and its data provenance before spending compute on a rerun; if the provenance is confirmed stale
(pre-leak-fix SWaT / superseded HAI build), rerun on the current clean pipeline for all three datasets.
- **Invariant:** re-derive the "temporal - stats" delta and check it against whatever numeric threshold the
  prose actually uses (not a threshold that contradicts its own printed numbers); if the real finding is
  "temporal helps on HAI/SWaT, not on WADI," that is a different (and reportable) claim than "gain isn't
  from representation," and the paper's §6 language should not overclaim invariance until this is resolved.
- Cost: ~10 min triage read; ~20-30 min rerun if warranted.

---

## Summary of corrections the paper needs regardless of any retry (found while triaging, not asked for but load-bearing)
These are not "retry" candidates — they are already-completed clean reruns whose results contradict numbers
currently reported as significant, in-paper findings:
1. **SWaT Table 7 (subsystem localization, Top-1 0.46, P=0.004)** — collapses to non-significant (p=0.16-0.48)
   on the official clean pipeline. Highest-severity item: a positive significance claim in the manuscript is
   not reproducible on leak-free data.
2. **SWaT "source of gain" (Table 5, cross-channel density +0.141)** — inverts to −0.050 (chance-level) clean.
3. **SWaT score-head ordering (Table 6)** — every head collapses toward chance; reconstruction is still
   relatively best but drops from 0.835 to 0.567.
4. **SWaT "ties the nonlinear channel-wise baseline" framing** (`loo_baseline_sarfraz.md` §3's recommended
   related-work sentence) — built on the leaked 0.837-vs-0.881 comparison; superseded by
   `drift_changepoint_typing.md`'s 0.723 (decomposed community headline) vs 0.412 (raw boosted-LOO, not yet
   decomposed — see retry #1).
5. **`rev4_doublehard_pca_all3.py`'s cached SWaT GDN column** (flagged inside `drift_changepoint_typing.md`
   as a "side finding, outside scope, must reach the parent") points at a stale pre-official-normal GDN dump
   (`score_GDN_SWaT_canon_s0.npy`, mtime predates the leak fix); the fresh official GDN score is uncorrelated
   with it (r=−0.01) and scores 0.115, not 0.571, on the PCA-double-hard subset. This is a live bug in a
   script that will silently reproduce a wrong "GDN leads" cell if rerun as-is.
