# Manuscript reconciliation: IoT2.html to clean-recompute numbers

Target: `poc/paper/IoT2.html` (change-marked; new/changed text in `rev2-highlight` green, deletions in
`rev-del`). Sources of truth: `_diagnostics/clean_recompute.md` (Tables 3/4/5/6/A1 + significance),
`gdn_rescore_clean.py` (GDN), `a8_remeasure.md`, `a8_masking.md`, `a8_masking_allsets.md`,
`a8_method_leverage.md` (A8), `a9a10_measure.md` (A9/A10), `scout_dataset_scale.md` (dataset scale).
DOCX left unbuilt; no pandoc run.

## Table-by-table diff (old -> new)

### Table 3 (per-subset AUROC/F1)
- WADI block fully replaced (FIX 1+2+3, difficult 43->30, anomaly split 13+43 -> 26+30 easy/difficult):
  trivial 0.769/1.000/0.699 -> 0.786/0.999/0.601; IF 0.725/0.871/0.681 -> 0.725/0.829/0.634;
  AE 0.799/0.999/0.739 -> 0.792/0.981/0.628; LinRes 0.834/0.993/0.787 -> 0.834/0.932/0.750;
  USAD 0.757/0.994/0.685 -> 0.757/0.962/0.579; TranAD 0.786/0.998/0.722 -> 0.786/0.986/0.613;
  LatAD-global 0.793/0.991/0.734 -> 0.717/0.813/0.634;
  headline (regime-community) 0.864/0.996/0.824 -> 0.827/0.892/0.771. F1s to clean 2-dp.
  WADI per-seed SDs dropped (clean_recompute Table 3 reports point AUROC only for WADI).
- HAI headline only: 0.949/0.983/0.849 -> 0.948/0.984/0.845 (F1 0.756/0.822/0.364 -> 0.76/0.82/0.36).
  HAI baselines unchanged (I2).
- SWaT headline only: 0.941/0.999/0.840 -> 0.938/0.995/0.837 (F1 0.850/0.963/0.562 -> 0.83/0.92/0.57).
  SWaT baselines unchanged (I2).
- GDN row ADDED (SOTA group) in all three blocks (construct-matched single-seed rescore, AUROC only):
  WADI 0.801/0.963/0.660, HAI 0.845/0.970/0.481, SWaT 0.876/0.997/0.665; F1 = "-".
- Win (bold) cells re-marked: WADI-All best now LinRes 0.834 (was headline); WADI-Easy trivial 0.999;
  WADI-Diff headline 0.771. HAI/SWaT All/Easy/Diff headline still leads (0.948/0.938 etc). Caption updated
  (GDN now included, not omitted).

### Table 4 (double-hard)
- WADI column fully replaced (double-hard 27->19): trivial 0.612->0.493, IF 0.581->0.530, AE 0.671->0.535,
  LinRes 0.660->0.606, USAD 0.595->0.466, TranAD 0.629->0.489, LatAD-global 0.682->0.571,
  headline 0.743->0.662.
- HAI headline 0.819->0.814; SWaT headline 0.775->0.773. HAI/SWaT baselines unchanged.
- GDN row ADDED as "-" (double-hard not re-aggregated for the single-seed GDN rescore; noted in caption).
- Caption significance: WADI 27/10 ep +0.083 P=0.22 -> 19/7 ep +0.056 [-0.244,0.305] P=0.43;
  HAI +0.089 [0.022,0.186] -> +0.084 [0.019,0.189] P=0.0005; SWaT +0.087 [0.029,0.151] -> +0.085 [0.030,0.146].

### Table 5 (source of gain)
- WADI cross-channel 0.743->0.656, marginal 0.747->0.639, gain -0.004->+0.017 (30 windows). HAI/SWaT
  (+0.200/+0.141) unchanged. Caption window count 43->30. Prose "-0.004" -> "+0.017".

### Table 6 (heads, WADI column)
- recon 0.813->0.740, latent density 0.758->0.656, nearest 0.719->0.603, base 0.743->0.634,
  base+resid(=LatAD-global) 0.743->0.634. Ordering (recon>density>base) preserved. HAI/SWaT unchanged.
- Prose citing these WADI head values updated (0.813->0.740, 0.758->0.656, 0.743->0.634).

### Table A1 (ladder)
- LatAD-global 0.734->0.634; density-only experts 0.816/0.814/0.827 -> HC_coh 0.795/0.801/0.822 (WADI 0.795
  now bold = column best); full per-community headline 0.824/0.849/0.840 -> 0.771/0.845/0.837.
- Aggregation: sum 0.790/0.828/0.812 -> 0.698/0.827/0.805; max 0.811/0.802/0.803 -> 0.712/0.809/0.804;
  HC-unweighted 0.812/0.828/0.829 -> 0.760/0.793/0.826; cohesion-weighted HC (headline) 0.824/0.849/0.840
  -> 0.771/0.845/0.837 (WADI win removed, HAI/SWaT win kept); HC+null max-fused 0.806/0.846/0.823 ->
  0.749/0.770/0.811.

### Significance prose (Abstract, Fig 3, §6, §7, Conclusion)
- HAI difficult +0.092 [0.047,0.157] -> +0.088 [0.042,0.157] P=0.000 (vs AE 0.757).
- SWaT difficult +0.058 [0.015,0.107] P=0.003 -> +0.055 [0.012,0.104] P=0.007 (vs LinRes 0.782).
- WADI difficult +0.037 [-0.088,0.181] P=0.32 (11 ep) -> +0.020 [-0.161,0.205] P=0.46 (8 ep, 30 windows).
- Difficulty sizes stated: WADI 30, HAI 167, SWaT 85.

## Prose reworks
- Abstract / §6 Overall / §6 Difficult / §7 What-the-results-show / Conclusion: "best overall AUROC on
  every dataset" -> "best overall AUROC on HAI and SWaT"; on WADI leads difficult subset and every LEARNED
  detector but ties the linear baseline (full-set LinRes 0.834 vs LatAD 0.827).
- WADI mechanism reworded everywhere: the false claim "factorization lifts single-latent global density
  (0.734) to the WADI lead (0.824)" replaced by: single-latent LatAD-global collapses to 0.634 (below
  LinRes 0.750) on the clean 30-window subset; the WADI lead is carried by the community density
  aggregation HC_coh 0.795; the fused headline is 0.771; WADI significant vs every learned detector, ties
  only the linear baseline. Noted the 12 constant-channel-flip windows were a difficulty-stratification
  artifact (inconsistent clip) now fixed, moving WADI difficult 43->30.
- GDN: prose in §2.4, §5.5 changed from "did not complete on SWaT / not reported" to "benchmarked as a
  construct-matched single-seed SOTA baseline". §7 mechanism strengthened with GDN HAI-difficult collapse
  (0.481) alongside USAD 0.477 / TranAD 0.444 (a graph detector also failing on
  reconstructable-but-improbable faults).
- A8 (§7 + Appendix C + Table 1 A8 row + §3 + §4): the invalid VaDE-responsibility screen replaced by the
  observation-space story: (a) precondition (close-distinct adjacent regimes) present on nearly every real
  CPS dataset; (b) heavy overlap absent under a valid metric (synthetic 50%-overlap positive control the old
  responsibility screen scored "crisp"); (c) A8-masking not triggered by these benchmarks' anomalies and
  detectable only WITH the path (A10) - a snapshot cannot distinguish a masked anomaly from a legal regime
  switch; (d) over-coverage defended by the M=80 density head (wide K=2 misses ~80% of isolated WADI
  anomalies, M=80 head ~13%); (e) kNN head redundant (Spearman 0.90-0.93 with the density head). Table C1
  replaced with observation-space measures (close-distinct pair mass + min D, overlap coefficient mean/max,
  masking rate vs lag-matched base rate, isolated-missed by head resolution) for WADI/HAI/SWaT/Cranfield/
  SKAB + a positive-control row. SKAB reframed (variance-floor artifact AND no close-distinct pairs; the
  metric cannot witness A8 either way). §3 "regimes overlap" softened to "least separated" (silhouette
  slices a continuous ridge, not heavy overlap).
- A9/A10 (Table 1 rows + §7): A9 IS present in the data (per-channel autocorrelation 3.4-3.7 decades on all
  three), exploitable on HAI (~+0.06 at W=600 on the attack-free-history difficult subset, corroborating
  Appendix A +0.072); WADI/SWaT longer windows do not help. A10 MEASURED absent (three conditional scorers,
  passing synthetic positive control, order-shuffle guard). "Snapshot-detectable" now stated as a measured
  finding. Table 1 A8-A10 group header "Specified, not observed" -> "Specified, not realized by the reported
  window-only model".
- Dataset-scale scholarship added: Intro scope-note sentence (benchmark sizes), new §2.5 "Benchmark scale"
  paragraph, and a §7 "Benchmark scale and the trajectory assumptions" paragraph. New refs appended:
  [54] MetroPT (Veloso, Sci Data 2022), [55] BATADAL (Taormina, JWRPM 2018), [56] N-CMAPSS (Arias Chao,
  Data 2021), [57] BDG2 (Miller, Sci Data 2020), [58] TSB-UAD (Paparrizos, PVLDB 2022). SMD cites the
  existing OmniAnomaly ref [30].
- §4.3(iii) residual-gate ratios updated to the clean pipeline: 4.95/1.23/0.85 -> 3.41/1.22/0.76 (WADI off,
  HAI/SWaT on).
- Figure 3: WADI bars redrawn to the new difficult AUROC; WADI whiskers removed (WADI Table 3 no longer
  reports per-seed SD); HAI/SWaT LatAD bars + whiskers nudged (0.849->0.845, 0.840->0.837). Caption numbers
  updated. GDN is not drawn in the figure (6-method bar chart kept; GDN is in the tables).
- §7 coverage-diagnosis paragraph: base anchors 0.949/0.849 -> 0.948/0.845 for parity with Table 3 (the
  improved-view absolutes 0.965/0.890 from that secondary run left as reported).

## Self-check
- SC-3 abstract<->body parity: HAI All 0.948, SWaT All 0.938, HAI diff 0.845, SWaT diff 0.837, WADI diff
  0.771, USAD/TranAD 0.44-0.48 all match Table 3. PASS.
- Every significance value matches clean_recompute (difficult and double-hard, all three). PASS.
- WADI internal consistency: 0.771 fused / 0.795 HC_coh / 0.634 global appear consistently in Table 3,
  Table A1, §6, §7, Appendix A. PASS.
- GDN in Table 3 (all three), Table 4 ("-"), and prose (§2.4, §5.5, §7). PASS.
- Table/span tags balanced (10/10 tables; spans matched).

## Unresolved / flagged for the human
1. **Table 2 (WADI single-model decomposition, 0.816 recon / 0.718 nearest / 0.803 joint) left UNCHANGED.**
   It is a single trained model on the pre-clean (dirty, FIX-2-bug) WADI difficult subset and was NOT
   re-run in clean_recompute (only the 5-seed Table 6 was). It now visibly disagrees with the clean Table 6
   WADI column (recon 0.740, nearest 0.603). Options: re-run the single-model Table 2 on the clean 30-window
   WADI subset (~5 min CPU, cannot run here - raw datasets absent from this environment), or drop/annotate
   the WADI column of Table 2 as pre-clean. Recommend a focused re-run before build.
2. **GDN double-hard cell (Table 4) is "-".** The single-seed GDN rescore was aggregated only for
   All/Easy/Difficult (gdn_rescore_clean.py); a double-hard cell needs the LinRes train-q99 mask on the same
   grid, which was not available in this environment (raw WADI/HAI/SWaT CSVs are not present here, so
   gdn_rescore_clean.py cannot be re-run). Per the task's stated fallback, GDN is in Table 3 only and "-" in
   Table 4 with a caption note.
3. **New references appended as [54]-[58], NOT first-appearance-renumbered.** MDPI first-appearance order
   would place BDG2 (first cited in the §1 scope note) and MetroPT (§7/Appendix C) ahead of the §2.5
   citations. Renumbering all 58 refs by hand risks errors; the paper-build/human should run the MDPI
   first-appearance renumber pass at build time.
4. **Cranfield** (used in the A8 observation-space screen and Appendix C) is cited by name only; it has no
   entry in the reference list. Add a Cranfield multiphase-flow-facility citation if the reviewer expects
   one, or keep as an internal-screen dataset name.
5. **§7 coverage paragraph improved-view numbers (0.965 / 0.890 / 0.916)** are from the pre-clean secondary
   run; only the base anchors were updated to 0.948/0.845. If that secondary analysis is re-run on the clean
   pipeline the deltas may shift by ~0.001-0.005.
6. GDN Table 3 numbers (WADI 0.801/0.963/0.660, HAI 0.845/0.970/0.481, SWaT 0.876/0.997/0.665) are taken
   from the task brief / gdn_rescore_clean.py output; they could not be independently re-run here (raw data
   absent). Verify by running `python _diagnostics/gdn_rescore_clean.py` on the data host before build.

DOCX intentionally left unbuilt (HTML edits only).
