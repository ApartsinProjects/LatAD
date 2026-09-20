# Nearest-component NLL head (Section 4.3.ii, A5): keep-or-cut verdict

Scope: read-only analysis of `e2_nearest.py`, `_diagnostics/nc_real_rare_fpr.{py,jsonl,*.log}`,
`_diagnostics/nc_miim_sweep.{py,jsonl}`, `_diagnostics/nc_oracle_sweep.json`, `fix_verify.py` +
`_diagnostics/fix_verify_*.json`, `models_vade.py` (`anomaly_score_hard`), `ensemble_final.py`, and
Table 6 in `paper/IoT2.html`. Manuscript not touched.

## 1. Does the head measurably prevent false alarms on valid rare-regime windows?

**On the three real, leak-fixed datasets (`nc_real_rare_fpr.jsonl`, 5 seeds each, current
SWaT_canon/WADI_clean/HAI loaders, PCA double-hard mask): no, not at any threshold a deployment
would actually use.**

At the *matched-1%-FPR* threshold (the score is recalibrated so the overall test-normal FPR is 1%
— the only operationally meaningful comparison, since both heads are recalibrated the same way),
`nearest` and `mixture` give **identical FPR** on the rare-regime subsets (R1 = VaDE components
with train occupancy < 2%, R2 = bottom-quartile π, R3 = rare density-head components) in
essentially every (dataset, seed) row, to 4+ decimal places. Examples:
- SWaT_canon seed 0: mixture R1 FPR = 0.013035, nearest R1 FPR = 0.013035 (identical).
- WADI_clean seed 0: mixture R1 FPR = 0.0, nearest R1 FPR = 0.0 (identical; R1 has only 11 windows).
- HAI seed 0: mixture R1 FPR = 0.045396, nearest R1 FPR = 0.045396 (identical).

At the *uncalibrated train-p99* threshold, a difference appears only in the smallest, most
purpose-built subgroup (`R1_rare_vade_insupport`: rare-VaDE-component AND still within the
nearest head's own train-normal support — the one slice where the π penalty is the *only* thing
that can differ). There `nearest` sometimes hits exactly 0 FPR against `mixture`'s 0.01–0.06
(e.g. SWaT seed 0: mixture 0.0385 vs nearest 0.0; HAI seed 0: mixture 0.0032 vs nearest 0.0). But
this group is tiny (n = 5–91 windows depending on dataset/seed) and the effect disappears once the
threshold is recalibrated to a fixed FPR budget — i.e. it is a threshold-choice artifact, not an
operational gain.

**In the fully synthetic closed-form oracle** (`nc_oracle_sweep.json`, true generative mixture
known exactly, no learning), the effect is real, large, and exactly as designed: as imbalance
grows (f: 0.1→0.002, imbalance ratio 1×→55×) `mixture` rare-regime FPR balloons from 0.0093 to
0.2407 (matches the closed-form prediction 0.2523), while `nearest` stays flat at ≈0.0094
throughout. This is the textbook A5 mechanism, cleanly demonstrated — but only when the mixture is
handed to the head, not learned.

**Once the mixture has to be *learned* by VaDE** (`nc_miim_sweep.jsonl`, MIIM synthetic with
ground-truth mode labels, imbalance sweep), the effect is inconsistent: `nearest` FPR is
frequently equal to or **higher** than `mixture` FPR on the target rare-valid group, e.g.
target=13/seed=1: mixture fpr_val_target = 0.0164 vs nearest = 0.0410 (nearest 2.5× worse). So the
protection that is clean in closed form does not reliably survive contact with a learned
component allocation.

## 2. Effect on headline difficult-subset AUROC (per-head ablation)

Table 6 (already in the manuscript, current clean pipeline) plus the independent
`nc_real_rare_fpr.jsonl` 5-seed re-run and the earlier 5-seed `fix_verify_*.json` (pre-leak-fix
loaders, corroborating direction/magnitude) agree:

| dataset | density alone | nearest alone | base (density+nearest) | Δ from dropping nearest |
|---|---|---|---|---|
| WADI | 0.656 | 0.603 | 0.634 | **+0.022** (Table 6) / +0.0196 (`fix_verify`) |
| HAI | 0.802 | 0.797 | 0.801 | +0.001 (Table 6, within seed σ) / +0.0056 (`fix_verify`) |
| SWaT | 0.472 | 0.479 | 0.472 | 0.000 (Table 6) / −0.0005 (`fix_verify`, noise) |
| SKAB | — | — | — | +0.0011 (`fix_verify`, noise) |

`nearest` alone is never the best head on any real dataset. Folding it into `base` never helps and
costs a real 0.02 AUROC on WADI — exactly the number the manuscript already flags ("it can leave
the base score marginally below the density head alone (Table 6, WADI)"). `fix_verify.py`
independently ran the exact ablation ("adopt the fix only if it is ≥ current on every dataset") on
all four datasets on the pre-leak-fix pipeline and it **passes on all four** (WADI +0.020, HAI
+0.006, SWaT −0.0005, SKAB +0.001) — i.e. dropping the nearest term from the fused score was
already found to be a strict-or-neutral win before the leak fix, and the post-fix Table 6 numbers
reproduce the same pattern.

## 3. Is the A5 role actually exercised by any real dataset, or only defensible in principle?

Only in principle / only in the closed-form synthetic oracle. On WADI_clean, HAI, and SWaT_canon,
the mixture-sum washing-out failure mode the head is meant to guard against does not show up at
any calibrated operating threshold: `mixture` and `nearest` produce the same false-alarm rate on
rare-regime normals to 4 decimal places in nearly every (dataset, seed, regime-definition) cell.
The one place a difference appears (`R1_rare_vade_insupport` at an uncalibrated train-p99
threshold) involves single-digit-to-low-double-digit window counts — not enough to support a
dataset-level claim, and it vanishes under the realistic matched-FPR calibration. The learned-MIIM
synthetic sweep, which is closer to the real deployment regime than the closed-form oracle, does
not reproduce the oracle's clean win either (nearest is sometimes worse than mixture there).

## 4. Bug / staleness risk

`models_vade.py::anomaly_score_hard` (lines ~276–303) carries a docstring that already states the
conclusion of this verdict — *"an ablation across four datasets shows the high-K density head
subsumes it (density-only ≥ density+nearest on WADI/HAI/SWaT, negligible on SKAB), so the reported
model uses density-only base"* — but the function's own default is `use_near=True`, and every
caller that builds the reported/shipped score (`build_scores_table.py`, `export_checkpoints.py`
line 42, `firmup.py`, `abs_split.py`) calls it without overriding `use_near`, so they all silently
get `use_near=True`. The comment is stale/aspirational: **the actually-shipped `LatAD`/`base` score
still includes the nearest term**, which is why Table 6's `base` row is 0.022 below `density` on
WADI instead of matching it. `use_near=False` is exercised only in three one-off diagnostic
scripts (`fix_verify.py`, `knn_latent_verify.py`, `v2_temporal.py`), never in the pipeline that
produces the numbers reported in the paper. This is a real code/doc mismatch independent of the
keep/cut decision below, and worth fixing regardless.

## 5. Verdict: CUT (drop the nearest term from the fused score; keep it only as a documented, rejected ablation)

Evidence converges from four independent angles — the current Table 6, the post-leak-fix
`nc_real_rare_fpr.jsonl` 5-seed rerun, the pre-fix `fix_verify.py` 5-seed 4-dataset sweep, and the
codebase's own (unenforced) docstring — that folding the nearest-component NLL into the fused
score never helps difficult-subset AUROC and costs **0.02 AUROC on WADI**, while its claimed
rare-regime false-alarm protection is not distinguishable from the plain mixture at any calibrated
operating threshold on WADI_clean, HAI, or SWaT_canon (identical FPR to 4 decimal places in nearly
every cell of `nc_real_rare_fpr.jsonl`). The protection is real only in the closed-form synthetic
oracle where the true mixture is known exactly, not once VaDE has to learn it (MIIM sweep, real
data).

**Recommended action:** set `anomaly_score_hard`'s effective default to `use_near=False`
(density-only base) for the reported/shipped `LatAD` score — this is a pure win (+0.022 WADI, ~0
elsewhere) and requires no retraining, since `fix_verify.py` already shows it holds per-seed on
every dataset. In the manuscript, keep `Section 4.3.ii` and the nearest-component row in Table 6
(moving to the appendix regardless) as a documented, explicitly-rejected ablation rather than a
live head in the fused score, with one clean sentence:

> *"A nearest-component substitute for the mixture-sum density was evaluated as a candidate
> safeguard against rare-regime false alarms (Appendix Table 6); at a matched false-positive-rate
> operating point it did not reduce the false-alarm rate on rare-regime normals below the
> mixture-sum density on any of the three benchmarks and cost 0.022 AUROC on WADI, so the reported
> score uses the density head alone."*

This removes a component whose benefit is unearned on real data, recovers the WADI gap the paper
already flags, and turns a reviewer-contested design choice (R1-2/E2) into a reported negative
result with numbers, rather than a principle-only justification.
