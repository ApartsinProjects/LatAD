# A8 reframe verdict: dataset property vs deep-AD diagnostic vs drift unification

Scope: read-only synthesis of `a8_valley_*`, `a8_masking_*`, `a8_absorption_allsets*`, `a8_method_leverage*`,
`a8_drift_unification*`, `reframe_plan.md`, and Appendix C of `poc/paper/IoT2.html`. No manuscript or diagnostic
files edited.

## 1. What the measurements show

The signal is clean and internally consistent, not noise: three independent measurement pipelines
(`a8_valley_real.py` geometric partitioning, `a8_masking.py`/`a8_masking_allsets.py` proximity-masking with
lag-matched controls, `a8_absorption_allsets.py` wide-head absorption) agree on the same two-part picture across
SWaT_canon, WADI_clean, HAI, plus a synthetic-controls suite (linear- and curved-manifold, positive/negative
masking controls, a frozen-centre absorption ring):

- **The A8 precondition (close-but-distinct normal regimes) is real and large.** Close-distinct pair mass is 0.99
  to 1.00 of train mass on WADI, HAI, SWaT (min pairwise Mahalanobis distance 2.2 to 3.9), and 0.94 to 1.00 across
  a broader 14-dataset screen (`a8_masking_allsets.md` §1, §4). Only SKAB (single-loop rig, BIC picks K=5) has
  zero close-distinct pairs. This is a measured, label-free dataset property with a defensible magnitude: it holds
  on every multi-loop plant tested and is absent on the one single-loop rig tested, which is exactly the
  qualitative behavior the geometry predicts.
- **The failure mode the paper originally hypothesized (an anomaly of one regime absorbed into an adjacent
  regime's core, both for reconstruction-smoothing and for a low-resolution density head) is absent on every real
  dataset at the labelled level, and present only on synthetic positive controls built to contain it.** Masking
  rate (labelled) is at or below its lag-matched regime-switch base rate everywhere: WADI 0.010 vs base 0.100,
  HAI 0.000 vs 0.035, SWaT 0.008 vs 0.117, Cranfield 0.000 vs 0.005, SKAB 0.002 vs 0.010 (`a8_masking_allsets.md`
  Table in §4; identical numbers appear in Table C1 of the paper). The synthetic positive control gives 0.123 vs
  base 0.111 (small but positive excess by construction) and the negative control gives 0.000 vs 0.111 (a large
  negative excess, as it must). Real data sits with the negative control, not the positive one.
- **Positive/negative controls behave as designed**, which is the strongest evidence the pipeline is sensitive and
  the real-data null is a real null rather than a blind instrument: the synthetic masking-positive control (A
  drifting onto a close, distinct B) is caught by the metric at TPR 1.00/AUROC 0.98 (context score) against
  TPR 0.06/AUROC 0.53-0.54 (shipped LatAD head), exactly the blind spot A8 predicts (`a8_masking.md` §1); the
  absorption ring control shows the wide K=2 head missing 99.7% of frozen-centre anomalies vs 0.6% for the
  80-component head (`a8_masking_allsets.md` §4, "e"); the valley-synthetic controls reproduce a reconstruction
  blind spot only under a curved manifold and only as an on-manifold interpolation/extrapolation asymmetry, never
  as a literal "linear smoothing across the valley" (chord anomalies are caught at AUROC 1.00 in every curved
  setting; `a8_valley_smoothing.md` §4b).
- **Where a masking-like signal does appear on real data, it is small, partition-dependent, and reclassifies as an
  A10 (path-dependent) effect rather than an A8 (snapshot) effect.** On WADI only, an observation-space partition
  finds 41% (range 22-70% over 18 partitions) of missed windows beyond their own regime's envelope and absorbed by
  a neighbor, vs a 12.5% base rate (enrichment ~3.5x); but the identical statistic computed at the window-snapshot
  level cannot be distinguished from a legitimate regime switch over the same horizon (lag-matched base rate 11%
  matches the positive control's 12% almost exactly) — only the path (A10) separates a masked drift from a normal
  A-to-B switch (`a8_masking_allsets.md` verdict line 3; `a8_masking.md` §3).

Magnitude and consistency verdict: the precondition is large and robust (mass 0.94-1.00, holds across 14
datasets); the hypothesized failure mode is small-to-absent and, on the one dataset (WADI) where anything positive
shows up, the lead does not survive being fused into the paper's actual community-aggregation headline
(`a8_method_leverage.md`, see §2 below). This is a clean, defensible negative result, not measurement noise.

## 2. Is "deep models smooth valleys and miss between-regime anomalies" empirically supported?

**No**, on all three real datasets, and the synthetic controls explain the mechanism well enough to say why not.

Effect sizes (`a8_valley_smoothing.md`):
- Valley anomalies are rare in the first place: 5% of SWaT attacks (8-15 windows), 9% of WADI attacks (3-8
  windows), 0.5% of HAI attacks (2-4 windows), because the real inter-regime valleys are shallow (floor at the
  train 0.6th-36th density percentile, inside the normal envelope of the nearest regime on 96-100% of WADI/HAI
  close pairs).
- Where valley anomalies exist (SWaT, WADI), the deep reconstructors do NOT under-detect them: USAD/TranAD AUROC
  0.95-0.97 on valley anomalies, recall@1%FPR 0.66-0.85, both *higher* than LatAD's own valley-partition recall
  (0.25-0.50). The predicted asymmetry runs backward on real data.
- The interaction statistic (LatAD-minus-deep advantage on valley vs off-manifold anomalies) is +0.003 to +0.07
  with seed-dependent sign on SWaT, null on WADI (both signs, CIs crossing 0, n=3-8), and +0.13 to +0.27 on HAI but
  resting on only 3-4 windows and shrinking to +0.01 to +0.02 against a plain AE baseline (i.e. it is a
  near-manifold advantage over USAD/TranAD specifically, not a valley-specific one).
- HAI's one sizeable "deep misses, LatAD catches" set (76-83 windows vs USAD/TranAD) contains **zero** valley
  anomalies in every seed; it is enriched in moderate-magnitude off-manifold anomalies (79-86% vs 63-64% baseline,
  Fisher p<=0.007), which the plain AE baseline also catches.

Positive control that validates pipeline sensitivity: yes, but it needed a curved manifold, not a linear one.
In the linear synthetic world (`a8_valley_synth.py`), the reconstructor misses "valley" anomalies (AUROC 0.51-0.60
at gaps 3-8) but misses "beyond" anomalies of identical size equally badly (AUROC ~0.50) — in a linear world there
is no valley-specific effect, only a general in-manifold-displacement blind spot. Only in the curved-manifold
world (`a8_valley_synth_curved.py`) does a genuine valley-vs-beyond asymmetry appear for the reconstructor
(recon AUROC valley 0.58-0.84 vs beyond 0.70-0.90 at gaps 6-8, a 0.06-0.15 AUROC deficit), and even there: (i) the
straight-line "chord" anomaly is caught at AUROC 1.00 in every curved setting (the decoder follows the manifold,
it does not linearly smooth across it); (ii) the density head shows the *same* valley-vs-beyond asymmetry
(no consistent-sign interaction, -0.13 to +0.08); (iii) at realistic gaps (4 within-regime sd, matching real
WADI/HAI geometry) neither head sees the valley at all (AUROC ~0.50-0.55 for both).

So the magnitude the pipeline can detect when the effect is deliberately built in (0.06-0.15 AUROC deficit, curved
manifold, gap>=6) simply does not appear in real data at any comparable scale, and the one place a large
LatAD-over-deep margin does appear (HAI) is traceable to something else (near-manifold/off-manifold departures,
not valleys). `a8_valley_smoothing.md`'s own verdict line states this plainly: "the valley-smoothing mechanism is
not empirically supported on any of the three datasets."

## 3. Does A8 unify with the Section 7 drift framing?

**Partially, and the unification is a correlation between partition position and drift status, not an identity of
mechanism.** `a8_drift_unification.py`'s Q2a tables give the cleanest evidence (per-seed, SWaT_canon):

| partition | normals drifted share | slow-max sd | drift-recovery pct |
|---|---|---|---|
| in_cluster | 0.052-0.068 | ~2.0 | 0.20-0.21 |
| valley | 0.45-0.61 | 3.0-6.3 | 0.30-0.39 |
| beyond | 0.67-0.82 | 6.5-32.6 | 0.39-0.50 |
| off | 0.92-0.95 | 58.9-59.2 | 0.50-0.53 |

(HAI shows the same ordering with in_cluster 0.28, valley 0.47-0.62, beyond 0.60-0.64; WADI has no drift at all,
consistent with `reframe_plan.md`'s measured drift spectrum of WADI 1.5% / HAI 30-39% / SWaT 67-69%.) Being
farther from the nearest train-normal regime (in_cluster -> valley -> beyond -> off) monotonically increases the
chance a test-normal window is classified as "drifted" by the causal slow/fast decomposition, and the Fisher test
for valley-vs-drifted is significant (p=1.8e-8 to 0.13 across seeds on SWaT, p=5e-19 to 1e-8 on HAI). This says
A8's geometric partition (how far/where a window sits relative to the discovered regimes) and C3's drift/
changepoint typing are picking up **overlapping but not identical** structure: most of the mass that drifts is
off-manifold (share 0.83-0.88 of drifted normals fall in "off", only 0.03-0.12 fall in "valley"), so drift is
mostly an off-manifold phenomenon and only secondarily a valley (near-boundary) phenomenon.

On the attack side, Q3 shows the same partitions differ sharply in how well they are caught and how they are
typed: SWaT "off" attacks are 76-85% changepoint-typed and recovered at AUROC 0.90-0.93 by the changepoint score;
"valley" attacks are more often typed drift (22-38%, vs 0-22% for off) and score lower on the changepoint head
(AUROC 0.67-0.87 vs 0.92-0.99 for off), i.e. valley attacks behave more like the paper's "normal-mode coverage
gap" drift sub-cause (a legitimate-looking excursion into under-sampled territory) than like a
gradual-degradation drift. This is consistent with, but does not prove, the intuition that A8's between-regime
space and Section 7's "normal-mode coverage gap" sub-cause of drift are related framings of the same
under-sampling problem: both are about test behavior landing in low-train-density territory relative to the
discovered regime structure. They are not shown to be numerically identical (no single number in these diagnostics
equates a "valley window" count with a "coverage-gap drift" count), and the connection is empirical/correlational,
not derived.

## 4. Recommendation

**Recommendation: keep A8 as "specified but not realized," essentially the paper's current Appendix C framing —
do not promote a valley-specific detection-advantage claim.** The diagnostics do support tightening and slightly
extending the existing Table C1 framing (below), but not overturning it.

Current paper state (Appendix C, `poc/paper/IoT2.html` around line 2940-2981, "Table C1"): the appendix already
states exactly what the diagnostics found — "The A8 precondition holds on every multi-loop plant (close-distinct
mass 0.99 to 1.00 in Table C1, 0.94 to 1.00 across the broader screen); the overlap, the masking excess over the
base rate, and the over-coverage miss of the 80-component head are all absent." Table C1 reports, per dataset,
close-distinct mass and min D, overlap coefficient, masking rate vs base rate, and isolated-missed wide/80-comp —
and those four numbers per dataset (WADI 1.00(2.8)/0.09(0.27)/0.010 per 0.100/0.80 per 0.13; HAI
1.00(2.4)/0.09(0.26)/0.000 per 0.035/0.01 per 0.03; SWaT 0.99(3.9)/0.02(0.05)/0.008 per 0.117/0.06 per 0.005) are
traced exactly to `a8_masking_allsets.jsonl`/`.md` §4 and `a8_absorption_allsets.jsonl`/`.md` §4. Section 8's
prose already says "A8 (between-regime overlap) is specified in Table 1, but the overlap it posits is absent from
the datasets studied, so it is deferred to future work" and A10 already carries "the masking form of A8 coupled to
A10." This is the correct, defensible resting state given the evidence: the precondition is a validated measured
dataset property (already in the paper, correctly framed), but the failure mode is not a validated
deep-AD-failure diagnostic (it fires only on synthetic positive controls, not on any real anomaly set at the
labelled level).

**What would need to change to justify promotion**, and why the diagnostics stop short of it:
- `a8_method_leverage.md` explicitly retracted the one number that looked like a method-level win: the earlier
  claim that a context/expected-regime head lifts WADI's difficult AUROC from 0.73 to 0.80 used a label-using
  oracle rule, the pre-leak-fix 43-window subset, and single-latent LatAD (0.634) rather than the paper's actual
  community-aggregation headline (0.795) as the base. Under label-free rules, the clean 30-window subset, and the
  correct headline, every context head and every kNN/local-density head either does nothing (P(diff<=0) 0.4-0.7 on
  HAI/SWaT) or actively hurts WADI (headline+context: -0.008 to -0.059 AUROC, P>=0.83 for every label-free rule;
  headline+kNN: -0.023 to -0.004, P 0.85-1.00). The one apparent positive (HAI selfgate context, +0.012 to +0.024
  difficult/double-hard AUROC) is reproduced by a regime-free temporal-persistence control (a rolling max of the
  detector's own alarm) at equal or greater magnitude (rmax5 +0.038/+0.042), so it is alarm smoothing, not regime
  geometry (`a8_method_leverage.md`, "The one positive" subsection).
- No sentence about a valley-specific or between-regime-specific detection edge is supportable: the interaction
  statistics are null or sign-unstable on SWaT/WADI and rest on 2-4 windows on HAI (§2 above).

**If the paper wants one incremental addition** (optional, not required — this is a tightening, not a
promotion), the following sentence could be added to Appendix C's closing paragraph, sourced as shown:

> "A weaker signal, restricted to WADI and dependent on the geometric partition chosen, reclassifies as a path
> effect: in observation space, 41% (range 22-70% over 18 partitions) of WADI's missed anomalies are beyond their
> pre-episode regime's envelope and absorbed by an adjacent one (vs a 12.5% regime-switch base rate on test
> normals), but the same statistic computed at the window-snapshot level is indistinguishable from a legitimate
> regime switch (lag-matched base rate 11%, matching the synthetic positive control's rate of 12% almost exactly);
> A8 is measurable only as an A10 (path-dependent) phenomenon, not as a snapshot one."
> — sourced to `a8_masking.md` §2-3 (WADI VaDE/PCA rows) and `a8_masking_allsets.md` verdict line 3.

This sentence is defensible (controlled, replicated across 18 partitions, explicitly null-checked against a
lag-matched base rate) but is explicitly framed as a null/negative result about A8-as-snapshot, reinforcing rather
than reversing "specified but not realized." It should not be paired with any AUROC-lift claim, because
`a8_method_leverage.md` shows the lift does not survive fusion into the real headline.

## 5. Risk flags

- **SWaT_canon leak-fix timing is mostly but not entirely clean.** The formal git fix commit is
  `1224d49` "revision2: fix SWaT_canon train/test data leak" at **2026-09-19T21:41:56+03:00**. Cross-referencing
  file mtimes:
  - `a8_valley_real.jsonl` (the file cited in Sections 1-2 above) is **confirmed post-fix**: it was regenerated at
    2026-09-19 21:49:59, seven minutes after the leaked pre-fix version was explicitly archived as
    `a8_valley_real.jsonl.leaked_bak` (21:48:29). This is the cleanest provenance in the whole diagnostic set — the
    leaked version is preserved as an artifact precisely so the clean rerun is auditable.
  - `a8_drift_unification.json`/`.log` (Section 3) are dated 2026-09-20 08:20, unambiguously **post-fix**.
  - `a8_masking_allsets.jsonl` (10:14), `a8_absorption_allsets.jsonl` (10:04), `a8_method_leverage.jsonl` (10:45)
    and `a8_method_leverage_tables.md` (11:08) are all dated **2026-09-18**, more than a day *before* the formal
    git commit. However, `clean_recompute.md` (also 2026-09-18, 10:48) documents FIX 1/2/3 (the same
    train-normal-calibration, clip, and subset fixes referenced by the leak-fix commit) as already applied at that
    point, and `a8_method_leverage.md` explicitly states "Everything is on the CLEAN pipeline of
    clean_recompute.md." This indicates the underlying data/code fix was already in place locally on 09-18, and
    the 09-19 commit formalized it in git history; the masking/absorption/leverage numbers are very likely already
    clean, but this is inferred from documentation cross-reference, not from an explicit before/after archived
    pair the way `a8_valley_real.jsonl` has one.
  - **`a8_masking_allsets.jsonl` in particular (10:14) predates `clean_recompute.md`'s own write time (10:48) by
    34 minutes**, i.e. it may have been generated before the documented fix was written up, though the fix code
    itself could have existed earlier that morning. This file is the most ambiguous one relative to the leak fix
    and should be **re-run rather than assumed clean** if it is to be cited for anything beyond the qualitative
    "no excess over base rate" conclusion (which is also independently confirmed by the unambiguously-clean
    `a8_masking.md` cross-check in its own §7, run against `a8_masking.{WADI_clean,HAI,SWaT_canon}.jsonl`, and
    ultimately by Table C1's already-published numbers, which match `a8_masking_allsets.md` §4 exactly, implying
    Table C1 was built from this same file without a re-run after the formal commit).
  - Practical read: the paper's own Table C1 already carries these possibly-pre-formal-fix numbers, so if a re-run
    is warranted for provenance hygiene, it should target `a8_masking_allsets.py`/`a8_absorption_allsets.py` for
    SWaT_canon specifically and diff the result against Table C1 before treating either as final. Given (i) the
    `clean_recompute.md` cross-reference and (ii) that the masking-rate conclusion for SWaT_canon is already deep
    in "null" territory (0.008 vs 0.117 base, a 15x gap), a residual leak would need to be very large to flip the
    qualitative verdict, but the exact numbers 0.99(3.9)/0.008/0.06 for SWaT in Table C1 should be treated as
    "very likely clean, not verified clean" until re-run against a file with the same before/after provenance
    `a8_valley_real.jsonl` has.
- **No bug found in the geometry/partition code itself.** All five stated invariants in `a8_valley_smoothing.md`
  (I1-I5: window-count/label agreement, in-cluster near-chance, off-manifold near-ceiling, npz-label consistency,
  permutation-null centered at 0) pass on every dataset, and the synthetic positive/negative controls in both
  `a8_masking.md` and `a8_valley_smoothing.md` behave exactly as pre-declared, which is the strongest evidence
  against a code-level artifact explaining the "absent" real-data result.
- **Small-n caveats already flagged in the source files and inherited here**: SWaT valley n=8-15, WADI valley
  n=3-8, HAI valley n=2-4 windows (Section 2); WADI's A10-flavoured masking lead rests on 3-4 episodes out of 8
  (Section 4's optional sentence). None of these are large enough to support a standalone claim beyond the
  qualified language used above.
- **Stale-number risk already self-corrected in-repo**: `a8_method_leverage.md` itself documents and retracts an
  earlier over-stated WADI context-head lift (0.73->0.80) that used a label-leaking oracle rule and the pre-fix
  43-window subset; that retraction is already accounted for in Section 4 above and should not be re-introduced
  into the paper from any older diagnostic snapshot.
