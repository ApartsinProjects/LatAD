# Retry experiments (clean, leak-free LatAD pipeline) — E:\Projects\Backlog\LatAD\poc

Run under the auto-research skill discipline (invariant-before-report, negative-result triage).
`/c/Python314/python`, all scripts run from `poc/`. IoT2.html was NOT touched (read-only, for context only).

---

## Experiment 1 — Fair-fight boosted-LOO under causal drift/changepoint decomposition (SWaT_canon)

Script: `_diagnostics/retry_exp1_boostedloo_drift.py` (reuses the verbatim `causal_median()`/`decompose()`
functions copy-pasted from `_diagnostics/drift_changepoint_typing.py`, which cannot be imported directly
because it executes a full sweep at module scope). Output: `retry_exp1.json`, `retry_exp1.log`.

**Question**: the community/density headline recovers on SWaT under causal drift decomposition
(Difficult 0.524→0.723, DoubleHard_pca 0.178→0.671). Does the boosted leave-one-out (LOO) linear baseline
ALSO recover when its own per-window score is drift-decomposed the same way (K=24h causal median +
causal-MAD scale floor, a priori, no test-set tuning), or is the recovery specific to the
community/density mechanism?

### Invariant checks (both BLOCKING, both PASSED)

- **INVARIANT 1** (K=None reproduces the raw clean boosted-LOO figure exactly): raw boosted-LOO Difficult
  AUROC from `boosted_loo_SWaT_canon.npz`'s `b_te` = **0.411736**, vs the reference clean-pipeline figure
  in `clean_swat_boosted_loo.json` (`boosted_diff` = 0.411736). Exact match (abs diff 0.00e+00). **PASS.**
- **INVARIANT 2** (degenerate/no-op on WADI_clean, which has 0/44 drifting communities per the community
  analysis): raw Difficult AUROC 0.660 → decomposed (v2 level+scale) 0.669 (delta **+0.009**), decomposed
  (v1 level-only) 0.691 (delta +0.031). Both deltas are within the same small-magnitude band the
  community side shows on WADI (headline 0.771 → CP-primary 0.779, delta +0.008). **PASS** on the
  AUROC-based no-op criterion specified in the task. (Diagnostic-only, not gating: full-series Spearman
  rho(raw, decomposed) is 0.60–0.78, not ~1 — this is because the v2 scale-floor renormalizes by a
  rolling MAD even absent any level drift, which reorders scores locally without moving the
  Difficult-subset AUROC; this matches the community side's own behavior, where the scale variant also
  only produces a small AUROC move on stationary WADI despite touching every score.)

### Headline numbers (SWaT_canon, K=24h causal median + causal-MAD scale, PRIMARY variant)

| | raw (headline) | decomposed (CP, 24h level+scale) | gain |
|---|---|---|---|
| **boosted-LOO** Difficult AUROC | 0.412 | 0.567 | **+0.155** |
| **boosted-LOO** DoubleHard_pca AUROC | 0.117 | 0.553 | **+0.436** |
| **boosted-LOO** DoubleHard_lin AUROC | 0.124 | 0.625 | +0.501 |
| **community (HCcoh+LatAD)** Difficult AUROC | 0.524 | 0.723 | +0.199 |
| **community (HCcoh+LatAD)** DoubleHard_pca AUROC | 0.178 | 0.671 | +0.493 |
| **community (HCcoh+LatAD)** DoubleHard_lin AUROC | 0.193 | 0.719 | +0.526 |

### Verdict

**Boosted-LOO ALSO recovers substantially under the identical drift decomposition.** The gain is close in
magnitude to the community method's gain (+0.155 vs +0.199 on Difficult; +0.436 vs +0.493 on
DoubleHard_pca) — roughly 78–88% of the community method's absolute gain. This means the drift/changepoint
decomposition's benefit is **largely NOT specific to the community/density mechanism**: it is substantially
a generic property of applying causal drift-typing to ANY per-window anomaly score on SWaT_canon, not a
uniquely-community-structured effect.

**Correction to the paper's fairness framing**: the community headline's "0.524→0.723" recovery cannot, by
itself, be presented as evidence that the community method uniquely benefits from drift/changepoint typing.
A boosted linear LOO baseline recovers by a comparable amount when the same decomposition is applied to its
own score. The community method still ends up with the higher absolute post-decomposition AUROC (0.723 vs
0.567 Difficult; 0.671 vs 0.553 DoubleHard_pca) — so the community method's raw representational quality
carries some of the win — but that residual advantage (~0.10–0.16 AUROC) is much smaller than the raw-score
gap suggested by the undecomposed headline gap (also ~0.10, since the undecomposed gap was 0.524-0.412=0.112
and DH_pca 0.178-0.117=0.061). In other words: SWaT_canon's community method wins over boosted-LOO both
before and after decomposition by roughly the same margin; the decomposition helps both methods by a similar
absolute amount. If this recovery number is used in the manuscript, it should be presented as "drift-typing
recovers signal for multiple method families on SWaT_canon, including both the community/density method and
a boosted linear LOO baseline" rather than as a claim specific to the proposed method's mechanism.


---

## Experiment 4 — repr_ablation interpretation bug + clean-pipeline rerun

Script: `_diagnostics/repr_ablation.py` (rerun on the current clean pipeline, all 3 datasets).
Outputs: `repr_ablation.json` (overwritten with clean numbers), `retry_exp4_repr_ablation.log`.
Old log backed up to `repr_ablation.log.PRE_RETRY_BACKUP_20260918`.

**Question**: the write-up claims the difficult-subset gain is "not from the representation," but the
script's own printed deltas (+0.0715 HAI, +0.026 SWaT old) exceed the stated 0.01 threshold. Resolve
the provenance and the interpretation.

### Provenance check (the reason a rerun was warranted)

The old `repr_ablation.log` SWaT_canon absolute AUROCs were stats **0.7500** / temporal **0.7760**.
The clean-pipeline rerun gives stats **0.2469** / temporal **0.4883** — a collapse of ~0.50 AUROC.
WADI_clean (0.6728/0.6751) and HAI (0.7426/0.8141) reproduce the old log **bit-identically**. This
confirms the old SWaT numbers were computed on the pre-leak-fix (leaked) SWaT pipeline; HAI/WADI were
never leaked, so they were always clean. The rerun was therefore necessary and is now on the clean
pipeline for all three datasets.

### Invariant / re-derived deltas (clean pipeline, 5 seeds, fixed stats-maxz difficulty mask)

| dataset | stats-repr difficult AUROC | temporal-repr difficult AUROC | temporal − stats |
|---|---|---|---|
| WADI_clean (n_diff=43) | 0.6728 ± 0.0060 | 0.6751 ± 0.0146 | **+0.0023** (immaterial) |
| HAI (n_diff=167) | 0.7426 ± 0.0109 | 0.8141 ± 0.0072 | **+0.0715** (temporal helps) |
| SWaT_canon (n_diff=36) | 0.2469 ± 0.0261 | 0.4883 ± 0.0258 | **+0.2414** (temporal helps strongly) |

The manuscript's Table A2 (IoT2.html lines 1284-1286) already carries HAI **0.743/0.814 (+0.071)** and
SWaT **0.247/0.488 (+0.241)** — matching this clean rerun exactly. So the **numbers in the manuscript
table are current/clean**, not stale. (Manuscript's WADI Table-A2 value is 0.634/0.626; my rerun gives
0.6728/0.6751 — a small absolute offset likely from a slightly different WADI config, but the delta sign
and "immaterial on WADI" reading agree.)

### The actual interpretation bug (located)

The `"≤ +0.01 ⇒ gain is NOT from the representation"` claim is **in the SCRIPT, not the manuscript**:
- `repr_ablation.py` line 7 (docstring) and line 69 (final print) both state the blanket line
  "temporal−stats ≤ +0.01 across datasets ⇒ the gain is NOT from the representation (supports §6)."
  This blanket line is simply **FALSE** — it contradicts the script's own per-dataset deltas (+0.0715
  HAI, +0.2414 SWaT, both far above +0.01). The script's per-dataset prints ("temporal helps") are
  correct; only its summary line is wrong/stale. This is the "interpretation bug" — a stale summary
  line, not a manuscript claim.

The manuscript prose is mostly aligned with the true per-dataset result (it reports +0.071 and +0.241
explicitly and calls the representation a "complementary lever"), with ONE phrase that overreaches:
- **IoT2.html line 686**: "the latent-density mechanism, **not the choice of window statistics, carries
  the HAI and WADI result**." For **HAI** this is a mild overclaim: swapping stats→temporal features
  (holding the density mechanism fixed) adds **+0.0715** to the global-density difficult AUROC
  (0.743→0.814), which is not "immaterial." The window representation demonstrably matters on HAI. The
  WADI half of the claim holds (+0.002).
- IoT2.html lines 951, 1270-1272 handle this more carefully (they report +0.071 HAI / +0.241 SWaT and
  frame the representation as a "complementary lever aligned with A9, left to future work"), so those
  passages are defensible.

### Verdict (correction, not a paper-worthy win)

- The **temporal representation HELPS on HAI (+0.071) and SWaT (+0.241) but not WADI (+0.002)** — a
  DIFFERENT claim than "the gain isn't from representation." Per-dataset, representation is immaterial
  only on WADI.
- **Correction needed to the manuscript**: the line-686 summary "not the choice of window statistics
  carries the HAI ... result" overstates for HAI, where representation adds +0.071. The nuanced
  passages (§ "Where the gain comes from", Table A2 caption/prose) are fine and already report the true
  deltas. Recommended (for the user/a separate step, since IoT2.html must not be edited here): soften
  line 686 to scope the "not the representation" claim to WADI, or acknowledge that on HAI the
  representation is a real (+0.071) complementary lever rather than immaterial.
- **Also fix the script's stale interpretation line** (`repr_ablation.py` lines 7 and 69): its blanket
  "≤ +0.01 ⇒ not from representation" contradicts its own numbers and should read per-dataset
  ("immaterial on WADI; temporal helps on HAI +0.071 and SWaT +0.241").
- On SWaT the manuscript's own framing is correct: temporal/spectral features nearly double the
  difficult AUROC (0.247→0.488) because they capture the same §7 drift non-stationarity the changepoint
  typing models directly.


---

## Experiment 2 — LOF and kNN-in-latent density-head variants, SWaT_canon official-clean

Scripts: `_diagnostics/adaptive_density.py` (LOF variant; output `adaptive_density_retry2a.json`,
`retry_exp2a_lof.log`) and `_diagnostics/retry_exp2b_knn_latent.py` (kNN-in-latent variant, a
construct-matched copy of `knn_latent_verify.py` pointed at the official-clean `bundle_SWaT_canon.npz`;
output `retry_exp2b.json`, `retry_exp2b_knn.log`). 5 seeds each. Canonical `adaptive_density.json` and
`e2_knn_latent.json` were NOT overwritten.

**Question**: both variants were previously rejected on the rationale "SWaT is the easiest/best-covered
dataset" — a premise the leak fix falsifies. Re-run construct-matched on SWaT_canon (official-clean).

### Invariant checks (all PASSED)

- **Synthetic diffuse-valid-mode control** (LOF, dataset-independent): `INVARIANT_lof_less_rare_penalty
  = true` — LOF penalty ratio on a diffuse-but-valid mode = **1.0** < fixed-KDE **1.327** (LOF does not
  over-penalize a rare-but-valid regime). **PASS**, unchanged.
- **Regression check (dataset-independent prior number reproduced):**
  - LOF variant, HAI difficult AUROC: gmm **0.7891** / fixed_kde **0.7846** / lof **0.7974** — reproduces
    the canonical `adaptive_density.json` values exactly. **PASS.**
  - kNN variant, HAI difficult AUROC: dens **0.7891** / full **0.8101** / knn10 **0.7877** — reproduces
    the canonical `e2_knn_latent.json` values bit-identically. **PASS.**
  Both regression checks pass, so the harness/code has not drifted and the new SWaT_canon numbers are
  trustworthy.

### SWaT_canon official-clean difficult-subset AUROC (5-seed mean ± sd)

LOF variant (adaptive_density harness, winfeats-stats difficulty mask, n_diff=36):

| method | difficult AUROC |
|---|---|
| gmm (current fixed-K GMM density head) | 0.2628 ± 0.0137 |
| fixed_kde | 0.2450 ± 0.0148 |
| **lof** | 0.2661 ± 0.0163 |
| lof − gmm (paired bootstrap) | **+0.003**, 95% CI [−0.043, +0.051], p(≤0)=0.47 |

kNN-in-latent variant (ens_bundle harness, paper-canonical difficulty mask, n_diff=91):

| method | difficult AUROC |
|---|---|
| dens (density head only) | 0.5181 ± 0.012 |
| full (reported LatAD head) | 0.5180 ± 0.014 |
| knn5 / **knn10** / knn20 | 0.5273 / **0.5290 ± 0.018** / 0.5322 |
| fuse (dens ⊕ knn10) | 0.5190 ± 0.012 |
| knn10 − full | **+0.011** | knn10 − dens **+0.011** (both within seed noise sd≈0.02) |

(The two harnesses report different absolute difficult AUROC magnitudes — ~0.26 vs ~0.52 — because of
different feature construction and difficulty-mask definitions, n_diff 36 vs 91; the ens_bundle path,
n_diff=91, is the one construct-matched to the main results tables. Both agree qualitatively: clean
SWaT_canon difficult is near/below chance, not "easy.")

### Verdict (NULL — no paper-worthy win)

**Neither variant helps meaningfully on clean SWaT_canon.** LOF ties the fixed-K GMM density head
(+0.003, CI spans 0, p=0.47); kNN-in-latent gives a marginal +0.011 over the density head, well within
the ~0.02 seed noise. The old rejection rationale ("SWaT is the easiest/best-covered set") IS falsified
by the leak fix — clean SWaT_canon difficult AUROC is near chance (~0.26–0.52 across harnesses) for
every latent-density variant, not easy (the leaked pipeline gave the kNN harness 0.96 on SWaT). But the
practical conclusion is unchanged: swapping the fixed-K GMM density head for LOF or kNN-in-latent
produces no meaningful gain on clean SWaT_canon; the difficult-subset signal is drift-dominated and near
chance for all local/absolute density heads alike (this is what §7's drift-decomposition, not a
density-head swap, addresses). Report as a null; keep both variants in the diagnostics/registry, not the
paper.


---

## Experiment 3 — Community detection-set complementarity, HAI, PCA double-hard subset

Script: `_diagnostics/retry_exp3_complementarity_doublehard.py` (construct-matched to
`complementarity_fullhead.py`: same ensemble scores, equal-alarm budget = 5%·n_normal + n_diff, same
method set). Prerequisite: the HAI expert bundle was rebuilt with `fit_surprise`
(`experts_local.py HAI hai_full5 average 25 3 5`, 5 seeds, → `sota_bundle/experts_variants/hai_full5/`)
because the canonical `experts_full/expert_HAI.npz` is the old no-`fit_surprise` format the current
`ensemble_final.py` rejects. Outputs: `retry_exp3.json`, `retry_exp3.log`.

**Question**: recompute the LatAD-exclusive vs TranAD/linres/AE/IF catch-set and Jaccard table scored
against the PCA-double-hard HAI subset (n=55) instead of the old maxz-only "difficult" subset (n=167).

### Invariant checks (both PASSED)

- **Strict-subset containment** (index-level, not just count): PCA double-hard (n=55) is a strict subset
  of the old maxz-difficult population (n=167); **0** double-hard windows lie outside the old subset.
  **PASS.**
- **Monotonicity**: only-LatAD catches on the smaller/harder subset (**0**) must not exceed only-LatAD
  catches on the larger n=167 subset (**9**). 0 ≤ 9. **PASS.**

### Catch counts and Jaccard (equal-alarm budget)

| | old n=167 (maxz-difficult) | PCA double-hard n=55 |
|---|---|---|
| alarm budget | 875 | 763 |
| caught: HCcoh+LatAD(full) | 68 | **0** |
| caught: AE | 50 | 0 |
| caught: IF | 17 | 0 |
| caught: linres | 11 | 4 |
| caught: TranAD | 1 | 0 |
| **only-LatAD** | **9** | **0** |
| LatAD not linres / linres not LatAD | 62 / 5 | 0 / 4 |
| LatAD not TranAD / TranAD not LatAD | 67 / 0 | 0 / 0 |
| Jaccard(LatAD, TranAD) | 0.015 | 0.0 |
| Jaccard(LatAD, linres) | 0.082 | 0.0 |

(My old-n167 only-LatAD = 9 vs the published `complementarity_fullhead.json` = 8; caught HCcoh+LatAD 68
vs 67 — a ±1 difference from the rebuilt bundle + current `ensemble_final.py` calibration. The
old-n167 side reproduces the published figures within ±1, confirming the harness is intact.)

### Verdict (NULL on the double-hard subset — scope correction, not a win)

**The detection-set complementarity / LatAD-exclusivity story does NOT survive on the PCA double-hard
HAI subset.** At the equal-alarm budget, HCcoh+LatAD catches **0 of the 55** double-hard anomalies (as
do IF, AE, TranAD); only linres catches 4. The "8–9 LatAD-exclusive catches" figure that holds on the
broader maxz-difficult subset (n=167) drops to **0** on the double-hard subset, and all Jaccard overlaps
degenerate to 0 (LatAD catches nothing to overlap). The complementarity was carried by the ~112 windows
that are difficult-by-maxz but NOT double-hard-by-PCA; the hardest 55 are missed by every learned
detector at a 5% alarm budget. This is consistent with the drift_changepoint HAI DoubleHard_pca AUROC of
~0.763 (a ranking measure) coexisting with 0 catches at the tight 5%-FPR operating point — most
double-hard anomalies rank below the 95th normal percentile.

If the manuscript makes a "near-disjoint detection / LatAD catches faults others miss" complementarity
claim, it must be scoped to the maxz-difficult subset (n=167), NOT the PCA double-hard subset, where the
claim is unsupported (0 exclusive catches). Report as a null / scoping correction; keep in the registry.


---

## FINAL SUMMARY (all four experiments)

| Exp | invariant(s) | key numbers | verdict |
|---|---|---|---|
| **1** boosted-LOO drift decomposition (SWaT_canon) | I1 K=None reproduces raw 0.4117 EXACTLY (PASS); I2 WADI no-op AUROC delta +0.009 (PASS) | boosted-LOO Difficult 0.412→0.567 (+0.155), DoubleHard_pca 0.117→0.553 (+0.436); community 0.524→0.723 (+0.199), 0.178→0.671 (+0.493) | **Correction to fairness framing**: boosted-LOO ALSO recovers substantially (~78–88% of the community method's absolute gain). Drift decomposition is largely NOT specific to the community mechanism. |
| **2** LOF + kNN density-head variants (SWaT_canon) | synthetic diffuse-mode control PASS (LOF penalty 1.0<1.327); HAI regression reproduces prior numbers bit-identically for BOTH variants (PASS) | LOF: lof−gmm +0.003 (CI [−0.043,+0.051], p=0.47); kNN: knn10−full +0.011 (within seed noise). Clean SWaT_canon difficult ≈0.26–0.52 (near chance) | **NULL**: neither variant helps. Old "SWaT is easiest" premise falsified (near chance now), but no density-head-swap win. |
| **3** complementarity, HAI, PCA double-hard | strict-subset containment (0 outside, PASS); monotonicity only-LatAD 0≤9 (PASS) | old n=167: only-LatAD=9, LatAD catches 68; double-hard n=55: only-LatAD=**0**, LatAD catches **0**, only linres catches 4 | **NULL on double-hard / scope correction**: complementarity vanishes on the hardest subset; any "LatAD-exclusive catches" claim must be scoped to the maxz-difficult subset (n=167), not double-hard. |
| **4** repr_ablation interpretation bug + rerun | old SWaT (0.75/0.776) was stale/pre-leak-fix; clean rerun reproduces WADI+HAI bit-identically, SWaT collapses to 0.247/0.488 | temporal−stats: WADI +0.002, HAI +0.071, SWaT +0.241 | **Correction needed**: temporal representation HELPS on HAI (+0.071) and SWaT (+0.241), immaterial only on WADI. Script's blanket "≤0.01 ⇒ not from representation" line is wrong; manuscript line 686 ("not window statistics carries the HAI result") mildly overclaims for HAI. Manuscript Table A2 numbers are already clean/current. |

### Paper-worthy wins ready to bring to the user for a manuscript decision

**None of the four is a new positive win to wire in.** Three of the four are corrections/nulls that
bear on EXISTING manuscript claims and should be brought to the user for an editing decision (the user /
a separate step edits IoT2.html — this run did not touch it):

1. **Exp 1 (highest priority — fairness framing correction)**: the SWaT drift-decomposition recovery
   (0.524→0.723) is NOT specific to the community/density mechanism — a boosted linear LOO baseline
   recovers by a comparable absolute amount (0.412→0.567) under the identical decomposition. If the
   manuscript presents the drift-decomposition recovery as evidence for the community method's unique
   benefit, that framing needs softening to "drift-typing recovers signal for multiple method families
   on SWaT, including a boosted linear baseline." The community method still leads in absolute
   post-decomposition AUROC (0.723 vs 0.567), but by a margin similar to the pre-decomposition gap.

2. **Exp 4 (interpretation correction)**: IoT2.html line 686 ("the latent-density mechanism, not the
   choice of window statistics, carries the HAI and WADI result") mildly overclaims for HAI, where the
   temporal representation adds +0.071. Recommended: scope the "not the representation" claim to WADI.
   Also fix the stale blanket interpretation line in `repr_ablation.py` (lines 7, 69). Manuscript Table
   A2 numbers themselves are already clean/current.

3. **Exp 3 (scope correction)**: any detection-set complementarity / "LatAD-exclusive catches" claim
   must be scoped to the maxz-difficult subset (n=167), not the PCA double-hard subset (n=55), where
   LatAD's exclusive catches drop to 0.

4. **Exp 2 (null, no action)**: LOF and kNN-in-latent density-head swaps do not help on clean
   SWaT_canon; stays in the registry, not the paper.

All invariants across all four experiments PASSED; no invariant failure required root-causing a
retracted number. All artifacts saved beside this file: `retry_exp1.{json,log}`,
`adaptive_density_retry2a.json` + `retry_exp2a_lof.log`, `retry_exp2b.json` + `retry_exp2b_knn.log`,
`retry_exp3.{json,log}`, `repr_ablation.json` (clean) + `retry_exp4_repr_ablation.log`
(old log backed up to `repr_ablation.log.PRE_RETRY_BACKUP_20260918`).
