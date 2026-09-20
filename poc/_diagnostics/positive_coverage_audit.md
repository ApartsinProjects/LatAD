# Positive-result coverage audit of IoT2.html (revision2)

Read-only audit. Manuscript not edited. Scope: every established positive/significant result in the
named diagnostics, cross-checked against the current `poc/paper/IoT2.html` (as of 2026-09-20, branch
`revision2`, includes the "rev2-highlight"/"rev-del" edit markup from the recent ChatGPT-review pass).

**Headline finding: the manuscript is in unusually good shape.** Nearly every positive result in the
diagnostics is already correctly reflected, at the corrected/clean numbers, including several
non-trivial corrections (SWaT leak fix, SWaT localization collapse, A8 "specified not realized",
nearest-component honest caveat). One real gap (drift-recovery framing/fairness) and one real overclaim
(nearest-component FPR-protection sentence) are identified below.

---

## Coverage table

| Result | Evidence source (numbers) | In paper? Where | Verdict |
|---|---|---|---|
| HAI difficult-subset lead, significant (0.845 vs deep baselines 0.44–0.50) | `swat_clean_ablations.md`, `doublehard_pca_all3.md`, headline JSON | Abstract; §6 Table 2; §8 conclusion, CI [0.042,0.157] | **INCLUDED** |
| WADI leads every nonlinear baseline, ties linear (0.771) | same | Abstract; §6 Table 2 | **INCLUDED** |
| Measured drift spectrum WADI 1.5% / HAI 39% / SWaT 69% | `drift_changepoint_typing.md` Table 4, `a8_drift_lens_verdict.md` §3 | Abstract; §7 Table 4 (WADI 1.5%, HAI 39%, SWaT 67% in §7.1 text — see note) | **INCLUDED** (note: §7 prose gives SWaT drift as ~66.6%/67%, Table 4 elsewhere; abstract rounds to 69% vs diagnostic 66.6% — negligible, same order, not misleading) |
| SWaT drift-aware changepoint recovery: Difficult 0.524→0.723 (P=0.014), DoubleHard 0.178→0.671 (P<0.001), All 0.788→0.880 (P=0.006) | `drift_changepoint_typing.md` §3 bootstrap table | Abstract; §7.2 Table 3, full CI/P text | **INCLUDED**, with correct CIs and even the marginal-vs-trivial-rule caveat (+0.096 P=0.031, +0.157 P=0.036) reproduced verbatim | 
| Typing validation vs attack labels (74.8/88.5/61.9% changepoint; 97.4/92.5% drift) | `drift_changepoint_typing.md` §4 | §7.2 Table 4 text | **INCLUDED** |
| HAI/WADI near-no-op under the same decomposition (within CI) | `drift_changepoint_typing.md` §"Verdict first" | §7.2 Table 3 + text | **INCLUDED** |
| Localization: HAI significant (top-1 0.395 vs 0.246 random, P=0.024) | `localization_clean.md` | §8 "Subsystem localization" + Table 5 (0.39/0.25/0.68/0.56) | **INCLUDED**, exact clean numbers |
| Localization: WADI directional (top-1 0.333 vs 0.131, P=0.086) | `localization_clean.md` | §8 text + Table 5 (0.33/0.13/0.56/0.31) | **INCLUDED** |
| Localization: SWaT does NOT survive (top-3 0.458 < random 0.542) | `localization_clean.md` | §8 text: "SWaT is omitted: its train-normal ranking is defeated by the record's drift"; Table 5 caption explicitly drops SWaT | **CORRECTLY EXCLUDED** (this is a null result, correctly kept out per wins-only) |
| Source-of-gain (Table A5): HAI +0.200, WADI +0.017, SWaT −0.050 (drift-dominated, inverts) | `swat_clean_ablations.md` §1 (SWaT leg); companion clean HAI/WADI numbers | §6 "Where the gain comes from" text; Table A5 | **INCLUDED**, clean numbers, SWaT correctly shown as non-win with drift caveat |
| Score-head ablation (Table A6): SWaT collapses to ~0.47–0.57 on every head | `swat_clean_ablations.md` §2 | Table A6, §6 text ("On SWaT every raw head sits at the drift-suppressed difficult-subset floor 0.47–0.57...") | **INCLUDED**, correctly caveated as drift-suppressed, not claimed as a head-ordering win |
| GDN construct-matched baseline, fresh (not the stale leaked 0.571 dump) | `drift_changepoint_typing.md` side finding; `doublehard_pca_all3.md` (fresh GDN SWaT doublehard 0.115) | Table 2 (GDN SWaT difficult 0.481), Table A4 (GDN SWaT doublehard 0.115) | **INCLUDED correctly** — paper already uses the fresh GDN values, not the stale `score_GDN_SWaT_canon_s0.npy` dump the diagnostic flagged as a live bug in `rev4_doublehard_pca_all3.py`. No manuscript action needed; the stale-file bug is confined to that one script. |
| A8 precondition real (0.94–1.00 close-distinct mass) but failure mode absent (masking null on real data) | `a8_drift_lens_verdict.md` | §8 "Assumption A8" text, Appendix C, Table C1 | **INCLUDED**, worded exactly as the diagnostic's recommended "specified but not realized" resting state — no overclaim |
| Nearest-component NLL: costs 0.022 AUROC on WADI when folded into base score | `nearest_component_verdict.md` §2 | §4.3(ii) text: "it can leave the base score marginally below the density head alone (Table A6, WADI)"; Table A6 (base 0.634 vs density-alone 0.656) | **INCLUDED**, correctly caveated | 
| PCA double-hard leaderboard: LatAD leads WADI/HAI (0.763/0.648), not significant vs LinRes on either (P=0.17/0.46); IF significantly beats LatAD on SWaT doublehard | `doublehard_pca_all3.md` | Table A4; §6 text gives WADI +0.021/HAI +0.101 as "numerical lead," does not claim significance | **INCLUDED**, appropriately hedged ("numerical lead," not "significant") — matches diagnostic's own non-significant verdict |
| Boosted-LOO (undecomposed) fails to recover SWaT doublehard (0.117) while LatAD+drift-typing reaches 0.671 | `swat_clean_ablations.md`, `doublehard_pca_all3.md` | §6 text: "confirming that boosting the linear predictor does not recover the subset" | **PARTIALLY UNDER-PRESENTED — see Priority Flag 1 below** |

---

## MISSING wins worth adding

**None identified.** Every established, clean, significant positive result in the read diagnostics
(drift-changepoint recovery, HAI/WADI localization, source-of-gain, GDN construct-matched baseline, A8
precondition) is already in the manuscript at its corrected value. There is no clean, final, positive
result sitting only in `_diagnostics/` that the paper omits.

## NOT-A-WIN / overclaim currently in the paper

### Priority Flag 1 — SWaT drift-recovery framing is not fully fair to the boosted-LOO comparison (UNDER-PRESENTED, needs a caveat, not a retraction)

`poc/_diagnostics/retry_results.md` Experiment 1 (COMPLETE — both blocking invariants passed: I1 exact
raw-score reproduction, I2 no-op-on-stationary-WADI check) found that when the **same** causal
drift/changepoint decomposition used for the community score is applied to the raw boosted leave-one-out
linear baseline, it **also** recovers substantially: Difficult 0.412→0.567 (+0.155) and DoubleHard_pca
0.117→0.553 (+0.436) — 78–88% of the community method's own gain (+0.199 / +0.493). The community method
still ends up numerically ahead after decomposition (0.723 vs 0.567 Difficult; 0.671 vs 0.553 DoubleHard),
so the paper's numbers are not wrong, but the sentence at §6 ("A boosted leave-one-channel-out linear
predictor... reaches only 0.649/0.313/0.117 on the three subsets, confirming that boosting the linear
predictor does not recover the subset") only tests the **undecomposed** boosted baseline against the
**decomposed** community score's neighboring text. A reader could infer the community/density architecture
is what makes drift-decomposition work, when the diagnostic shows the decomposition benefit is
substantially generic across method families on SWaT, not community-specific.

This is not a null result to smuggle in (wins-only is respected: the community method is still ahead, both
before and after decomposition, by a comparable margin). It is a **precision/fairness caveat** the paper
should carry, per `retry_results.md`'s own recommended framing: *"drift-typing recovers signal for multiple
method families on SWaT_canon, including both the community/density method and a boosted linear LOO
baseline."* Recommend either (a) adding one clarifying sentence in §7.2 or §6 noting the decomposition also
lifts a decomposed boosted-LOO baseline to ~0.57/0.55 (still below LatAD's 0.723/0.671), or (b) leaving the
current sentence but scoping its claim explicitly to "the undecomposed boosting" so it isn't read as
covering the decomposed case that was never tested there.

### Priority Flag 2 — Nearest-component NLL "protects the false-alarm rate on rare regimes" is an overclaim (NOT-A-WIN-IN-PAPER)

Paper text (§4.3.ii, line ~549-552): *"This term is a rare-regime safety head: its role is to keep a valid
window in a sparsely populated regime from being flagged for rarity alone (A5)... it protects the
false-alarm rate on rare regimes rather than raising the aggregate score."*

`nearest_component_verdict.md` (verdict: **CUT**) directly tested this claim on all three real, leak-fixed
datasets (`nc_real_rare_fpr.jsonl`, 5 seeds, rare-regime subsets R1–R3) at the only operationally meaningful
threshold (FPR matched to 1% overall): **`nearest` and `mixture` give identical FPR to 4+ decimal places in
nearly every (dataset, seed) cell.** The one place a difference appears (an uncalibrated train-p99
threshold on a tiny R1-in-support subgroup, n=5–91 windows) vanishes once the threshold is recalibrated,
and the effect is confirmed real only in a closed-form synthetic oracle with the true mixture known exactly
— not once VaDE has to learn the mixture (the MIIM sweep shows `nearest` sometimes *worse* than `mixture`
on the target rare-valid group).

So the specific mechanism claim — that this head measurably protects rare-regime FPR on the paper's own
benchmarks — is not supported by the data; it is a "specified but not realized on these three datasets"
result, structurally identical to how the paper already (correctly) frames A8. The paper does correctly
report the AUROC cost (0.022 on WADI), but the compensating benefit it states in the same breath ("it
protects the false-alarm rate on rare regions") has no supporting evidence in the diagnostics — on the
contrary, the diagnostic recommends cutting the term from the fused score entirely. Recommend either
softening "it protects the false-alarm rate on rare regimes" to state this is the design motivation (A5)
rather than a measured effect, or adding a one-line caveat that at a matched-FPR operating point this
protection is not distinguishable from the plain mixture on the three benchmarks (mirroring how A8 and the
localization SWaT-null are already caveated elsewhere in the same manuscript).

---

## Retry-batch items: verify when finished (NOT established, correctly absent from paper)

Per `retry_results.md`, only **Experiment 1** (boosted-LOO under drift decomposition, flagged above) is
complete with both invariants passed. The other three retry jobs left uncommitted output in
`_diagnostics/` but have no corresponding write-up in `retry_results.md`, so they are **PENDING**, not
established findings, and correctly do not appear in the paper:

- **Experiment 2a (LOF density-head variant, SWaT_canon)** — `retry_exp2a_lof.log` shows numeric output
  (SWaT diff gmm=0.263 vs lof=0.266, CI includes 0, P(≤0)=0.235) but this is a **null/non-significant**
  result even on the raw numbers shown; not a positive result to add regardless of completion status.
- **Experiment 2b (kNN-in-latent, SWaT_canon)** — script present (`retry_exp2b_knn_latent.py`) but no log
  output reviewed / no result file found; status unknown, treat as not run or not finished.
- **Experiment 3 (HAI community-complementarity on PCA double-hard subset)** — `retry_exp3_build_experts_hai.log`
  shows the expert-build step in progress (S=28 communities, HAI) with no complementarity numbers yet
  written; **incomplete**.
- **Experiment 4 (representation ablation, repr_ablation ambiguity check)** — `retry_exp4_repr_ablation.log`
  shows only the WADI_clean row (temporal − stats = +0.0023, "stats sufficient"); HAI and SWaT rows, which
  are the ones the original `negative_result_triage.md` flagged as inconsistent with the paper's "gain
  isn't from representation" line, are **not yet present**. This is exactly the item flagged as needing
  resolution before any claim about representation is trusted; **do not cite `repr_ablation.json`'s current
  contents for anything** until the HAI/SWaT rows are produced and the interpretation-threshold discrepancy
  from `negative_result_triage.md` is resolved. The paper does not currently make a strong "representation
  doesn't matter" claim in its highlighted text (only the smaller Table A2 lever is mentioned), so there is
  no immediate manuscript risk, but this should not be strengthened until Experiment 4 finishes cleanly.

None of these four pending items currently appear in the paper as a claim, so there is no wins-only
violation; flagging them here only so they are not mistaken for established results in a future pass.

---

## Bottom line

- All significant, clean, positive results identified in the named diagnostics are present in the
  manuscript, generally at the exact corrected numbers (SWaT drift-changepoint recovery, HAI/WADI
  localization split, source-of-gain table, GDN fresh values, A8 "specified not realized").
- No missing wins were found.
- Two items need attention, both about **overclaiming/imprecision**, not about a null result being
  presented as positive: (1) the boosted-LOO "does not recover" sentence should be scoped to the
  undecomposed comparison it actually tested (`retry_results.md` Exp. 1), and (2) the nearest-component
  head's "protects the false-alarm rate on rare regimes" claim is not supported by the matched-FPR
  measurement in `nearest_component_verdict.md` and should be softened to a design-motivation statement or
  explicitly caveated.
- Four retry-batch experiments (2a, 2b, 3, 4) are incomplete/partial; none of them is currently cited in
  the paper, so no wins-only violation exists today, but they should not be treated as established until
  `retry_results.md` is updated with their full write-ups and invariant checks.
