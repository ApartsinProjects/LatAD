# Gap data inventory (reviewer round, revision3)

Read-only audit of whether existing artifacts already contain the data to close the four
reviewer gaps, or whether new computation is required. Paths are exact; branch noted per file
(`wt` = current working tree / revision3, `r2` = revision2). "Portable" = liftable into the
paper with no new run.

---

## GAP 1 — Time-aware significance with proper serial-dependence handling

**VERDICT: data exists (portable) for the block-length-from-ACF + block-length sensitivity
asks; the training-seed hierarchical variant is NOT present.** The reviewer framed these as
"and/or", so the primary asks are covered.

**Artifacts**
- `poc/_diagnostics/rev4_timeaware.json` (wt / revision3; NOT on revision2)
- `poc/rev4_stats_timeaware.py` (r2; the generator, NOT in wt)
- `poc/_diagnostics/rev4_stats.json` (wt + r2; the base episode + moving-block bootstrap it strengthens)

**What is present.** `rev4_stats_timeaware.py` runs an episode + **moving-block** bootstrap and
reports each dataset under **three block lengths**:
- `L = 1` (i.i.d. resample, sanity/invariant),
- `L = overlap` (mechanical window-overlap length: WADI 3, HAI 4, SWaT 4),
- `L = acf` (block length chosen from the score autocorrelation: first lag where the normal-window
  score ACF drops below 0.1) — WADI `L_acf=1`, HAI `L_acf=300`, SWaT `L_acf=1`.

Per (dataset x block length) it stores `auroc`, `auroc_ci`, `compet_auroc`, `diff`, `diff_ci`,
`p_diff_le_0`, `win`, `n_boot`. Verdicts survive the longer ACF-derived blocks:
- **HAI** (vs AE): win at all three — diff +0.054, `p_diff_le_0` = 0.002 (iid) / 0.003 (overlap) /
  0.005 (acf, L=300). Win holds even at the 300-window block.
- **WADI** (vs IF): tie at all three — diff +0.013, p ~ 0.33-0.36.
- **SWaT** (vs linres): tie at all three — diff +0.001, p ~ 0.39-0.48.

So: (a) block length varies with reported CI/p, (b) block length is ACF-selected (HAI's ACF block
is 300, far longer than the overlap length of 4), (c) it is a moving-block bootstrap.

**What is absent.** No **hierarchical resampling over training seeds**. `boot()` resamples windows
(moving block) on the **seed-mean** score series and averages AUROC over the fixed seed set; seeds
are never resampled. If the reviewer insists specifically on folding training-seed uncertainty into
the CI, that variant needs a new run. (Seed spread is reported separately as `_sd` fields in
`rev4_stats.json` deployable metrics, but not as a hierarchical bootstrap CI.)

---

## GAP 2 — Rare-regime nearest-component validation beyond one dataset/one comparison

**VERDICT: partial.** Denominators, multi-dataset coverage, full 3W methodology and 3-seed
uncertainty all exist and are portable. Missing: an **alternative imbalance-aware score** and a
**controlled imbalance-ratio sweep**.

**Artifacts (all r2; NOT in wt)**
- `poc/_diagnostics/inv_a5_rarity.py`
- `poc/_diagnostics/inv_a5_rarity_rows.jsonl` (per dataset x seed rows)
- `poc/_diagnostics/inv_a5_rarity_summary.json` (aggregated)

**What is present.**
- **Multiple datasets, 3 seeds each:** DATASETS = 3W, SKAB, SMD, MetroPT, Cranfield; SEEDS = 0,1,2.
  (Only 3W has substantial rare-regime structure; SKAB / MetroPT rows are all-zero and SMD near-zero,
  so the substantive comparison effectively fires on 3W, with SMD marginal.)
- **Denominators for an FPR normalization (the reviewer's exact ask):** per row `n_teneg`
  (test-normal count) and `n_rare_teneg` (test-normal in rare regimes, occupancy < 0.05), plus
  rare-regime FP counts `fp05_mixture_rareFP` / `fp05_nearest_rareFP` at matched 5% FPR. For 3W:
  - seed 0: rareFP 17 -> 11, n_rare_teneg 57, n_teneg 333
  - seed 1: rareFP 17 -> 9,  n_rare_teneg 79, n_teneg 337
  - seed 2: rareFP 23 -> 16, n_rare_teneg 144, n_teneg 447
  (These are the 17/17/23 -> 11/9/16 Appendix-E counts; the rare-regime denominators to turn them
  into an FPR are the `n_rare_teneg` column.) `fp05_invariant_holds` = True all seeds
  (nearest rare-FP <= mixture rare-FP, the stated a-priori invariant).
- **Operating-point metrics with uncertainty:** matched and train-calibrated TPR@FPR at
  FPR = 0.01/0.05/0.10 for both `difficult` and `double_hard` positive sets, mixture vs nearest,
  with `_mean`/`_std` over 3 seeds in the summary; also `med_occ_defended`, `frac_defended_in_rare`,
  and up to 8 inspected defended-FP samples per row.

**What is absent.**
- **Alternative imbalance-aware score:** the only contrast is mixture-NLL (H1, includes the
  -log(pi_k) prior) vs nearest-component (H5, drops the prior). No third imbalance-aware baseline
  (e.g. a class-prior-corrected / balanced / reweighted score) is computed anywhere in these
  artifacts. Needs a new run to satisfy "at least one alternative imbalance-aware score".
- **Multiple imbalance RATIOS:** coverage is multiple datasets with different *natural* rare-regime
  occupancy (`min_occ` varies), not a controlled imbalance-ratio sweep on a fixed dataset. If the
  reviewer means a deliberate ratio sweep, that is not present.

---

## GAP 3 — Source-of-gain: raw-sequential input variant

**VERDICT: data exists (portable).** The raw-sequence variant the reviewer thought was not run
**was in fact run**, 5 seeds, on all three headline datasets.

**Artifacts (all r2; NOT in wt)**
- `poc/_diagnostics/raw_sequence_ablation.md`
- `poc/_diagnostics/raw_sequence_ablation.json`
- `poc/_diagnostics/raw_sequence_ablation.seeds.jsonl`
- `poc/_diagnostics/raw_sequence_ablation.py` (+ `.run.log`, `raw_seq_swat_only.run.log`)

**What is present.** Same global-density LatAD detector, same fixed stats-derived difficult mask,
5 seeds. STATS = current 6-statistic window representation; RAW = standardized windowed raw values
flattened over the W=60 window (60C dim), no statistical reduction. Difficult-subset AUROC
(5-seed mean +/- sd):

| dataset | STATS dim | RAW dim | STATS AUROC | RAW AUROC | RAW - STATS | verdict |
|---|---|---|---|---|---|---|
| WADI_clean | 732 | 7320 | 0.6728 +/- 0.006 | 0.6044 +/- 0.0251 | -0.0684 | RAW hurts |
| HAI | 354 | 3540 | 0.7426 +/- 0.0109 | 0.5123 +/- 0.0178 | -0.2303 | RAW hurts |
| SWaT_canon | 306 | 3060 | 0.2469 +/- 0.0261 | 0.1870 +/- 0.019 | -0.0599 | RAW hurts |

Invariants recorded and passing: STATS reproduces the clean repr-ablation reference, RAW dim ==
60C, zero-input degenerate AUROC = 0.5 (chance). Conclusion in the artifact: feeding the unreduced
raw window strictly degrades difficult-subset AUROC on all three; the gain comes from the density
mechanism on a compact representation, not from the stats representation withholding raw info.
(Separate `e42_temporal_rep_cranfield.py` (r2) covers the engineered temporal/spectral
representation — Table A2 territory — not the raw-sequence ask.)

---

## GAP 4 — Drift operating point: TPR/recall at the calibrated threshold (79% -> 6.5% FA)

**VERDICT: data exists (portable).** TPR/recall at the same train-calibrated threshold where the
SWaT normal-window false-alarm rate is 6.5% is recorded, not just AUROC.

**Artifacts (r2; NOT in wt)**
- `poc/_diagnostics/drift_changepoint_typing.json` (the `typing` block)
- `poc/_diagnostics/drift_changepoint_typing.md`, `.log`, `.py`

**What is present.** `typing` block, SWaT_canon, train-calibrated thresholds `thr_head` = 5.555,
`thr_cp` = 6.452:
- normals (n = 1265): `false_alarm_head` = 0.792 (79%), `false_alarm_cp` = **0.065 (6.5%)** — the
  claimed FPR drop, and the operating point.
- attacks (n = 233) **at that same threshold**: `flagged_head` = 0.918, `flagged_cp` = **0.687**
  (all-attack recall / TPR of the changepoint score), `flagged_head` difficult subset
  `difficult_flagged_head` = 0.791, `difficult_flagged_cp` = **0.330** (difficult-subset recall).
  `typed_changepoint_of_head_flagged` = 0.748.

So at the operating point where the changepoint-score false-alarm rate is 6.5% (down from the
head's 79%), attack detection sensitivity is 68.7% overall / 33.0% on the difficult subset. This is
exactly the TPR/recall-at-operating-point the reviewer asked for. (The `.md` states the 79% -> 6.5%
FA drop in prose; the log line 73-74 mirrors the numbers.)

---

## Summary table

| Gap | Portable? | Where |
|---|---|---|
| 1 Time-aware / block-length sensitivity | Yes for block-len-from-ACF + sensitivity; **seed-hierarchical needs new run** | `poc/_diagnostics/rev4_timeaware.json` (wt), `poc/rev4_stats_timeaware.py` (r2) |
| 2 Rare-regime nearest-component | **Partial** — denominators + multi-dataset + 3W methodology + uncertainty exist; **alternative imbalance-aware score + controlled ratio sweep absent** | `poc/_diagnostics/inv_a5_rarity_{summary.json,rows.jsonl,.py}` (r2) |
| 3 Raw-sequential input variant | **Yes** — already run, 5 seeds, all 3 datasets | `poc/_diagnostics/raw_sequence_ablation.{md,json,seeds.jsonl,py}` (r2) |
| 4 Drift operating-point TPR/recall | **Yes** — TPR at calibrated 6.5%-FA threshold recorded | `poc/_diagnostics/drift_changepoint_typing.json` `typing` block (r2) |
