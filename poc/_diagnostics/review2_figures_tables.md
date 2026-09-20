# Review 2: figures, tables, captions (IoT2.html)

Scope: `poc/paper/IoT2.html` (1395 lines). Source of truth: `poc/_diagnostics/clean_recompute.md` plus the GDN
and Table 2 numbers supplied by the coordinator. Every table cell and every SVG bar/whisker was read from the
markup; SVG text extents were measured in a browser (`getBBox`) against each `viewBox`. Report only; nothing edited.
Change-marking (`rev2-highlight` / `rev-del`) is not flagged.

Severity: **S1** = wrong or misleading number shown to the reader; **S2** = inconsistency a reviewer will catch;
**S3** = polish.

## Bottom line

- **No stale headline numbers anywhere.** Figure 3 encodes 0.771 / 0.845 / 0.837, not 0.824 / 0.849 / 0.840. All
  WADI baseline bars encode the clean 30-window values. Every cell of Tables 3, 4, 5, 6, A1 matches
  `clean_recompute.md`; GDN rows (Table 3 difficult 0.660 / 0.481 / 0.665, Table 4 0.550 / 0.469 / 0.615) are present
  and correct; WADI difficult = 30 and double-hard 19 / 84 / 59 everywhere; the only "43" in the file is the
  "43-to-30" explanation at line 861. All bold/`win` cells are the true column maxima (WADI-All LinRes 0.834,
  WADI-Difficult LatAD 0.771, SWaT-Easy tie 1.000/1.000).
- **One real figure defect (S1): Figure 3 was regenerated with the wrong y-axis scale.** The new bars (all six WADI
  bars, and the LatAD bar on HAI and SWaT) map 1.0 to y=64 (the tick *label* baseline) while the axis gridlines,
  the tick line at y=60, and the untouched HAI/SWaT baseline bars map 1.0 to y=60. Read against the axis the
  reader sees, LatAD shows as WADI 0.761, HAI 0.833, SWaT 0.826 (true 0.771 / 0.845 / 0.837), and the HAI margin
  over the AutoEncoder appears as +0.076 instead of +0.088. Details below.
- **One stale table column (S2): Table A2's WADI column** (0.673 / 0.675 / +0.002) was computed on the old
  43-window difficult mask; `clean_recompute.md` says explicitly "WADI A2 was not re-run in this pass" and
  recommends a ~5 min, 5-seed re-run.
- Several caption/consistency items (S2/S3), listed per element.

---

## FIGURES

### Figure 3 (lines 934-989): difficult-subset AUROC bar chart

| element | issue | stale value -> correct value / fix |
|---|---|---|
| **S1. Bar heights vs axis** | Two pixel scales in one chart. Axis: 0.3 at y=250, gridlines every 27.1 px, 1.0 at y=60 (271.4 px per unit). Old HAI/SWaT baseline bars use that scale. The regenerated bars use 1.0 at y=64 (265.7 px per unit), so they are drawn ~1 % short. | Read against the drawn axis: WADI `IF 0.627, AE 0.621, LinRes 0.741, USAD 0.573, TranAD 0.607, LatAD 0.761` (should be 0.634 / 0.628 / 0.750 / 0.579 / 0.613 / **0.771**); HAI `LatAD 0.833` (should be **0.845**; its baselines read correctly: 0.627 / 0.757 / 0.586 / 0.477 / 0.444); SWaT `LatAD 0.826` (should be **0.837**; baselines 0.625 / 0.727 / 0.779 / 0.656 / 0.653, drawn at a third scale, 270 px/unit, ~0.002 low). **Fix:** regenerate every bar and whisker with one mapping, `y = 250 - (v - 0.3) * (190 / 0.7)` (1.0 at y=60), and add the missing 1.0 gridline at y=60. Exact tops after the fix: WADI LatAD y=122.1, HAI LatAD y=102.1, SWaT LatAD y=104.3, WADI LinRes y=127.9. |
| S1 (consequence) | Visual margins are understated where the paper claims significance: HAI LatAD-AE gap drawn as 20.8 px = 0.077 (true 0.088 = 23.9 px); SWaT LatAD-LinRes gap 12.6 px = 0.047 (true 0.055 = 14.9 px); WADI LatAD-LinRes 5.5 px = 0.020 (correct by luck, both bars on the same wrong scale). | Fixed by the regeneration above. |
| S2. GDN absent | Table 3 now carries GDN on all three datasets (difficult 0.660 / 0.481 / 0.665) but the figure and its legend stop at six methods; caption says "by method". | Add a GDN bar (no whisker, single seed) per dataset, or state in the caption "GDN (single seed) is reported in Table 3 only". Caption claim "ahead of every learned detector on WADI" stays true with GDN 0.660. |
| S2. WADI whiskers | Caption: "whiskers show ±1 SD over seeds where it exceeds zero, for Isolation Forest, AutoEncoder, USAD, TranAD, and LatAD". No WADI bar has a whisker, yet WADI LatAD is a 5-seed learned model with SD 0.040 on the double-hard subset (Table 4), so its difficult-subset SD is not zero; the SDs are simply not in `clean_recompute.md` (WADI block has no ±). | Either draw WADI whiskers from the per-seed `clean_ensemble_WADI.json`, or say in the caption that WADI seed SDs are not shown. Same omission affects Table 3 (see below). |
| S3. Truncated axis | y-axis starts at 0.3, not 0; caption does not say so. Harmless but reviewers like it stated. | Add "(axis starts at 0.3)" or a break mark. |
| S3. Whisker check | HAI/SWaT whiskers decode correctly on the old scale: HAI IF ±0.005, AE ±0.003, USAD ±0.018, TranAD ±0.010, LatAD ±0.018; SWaT IF ±0.007, AE ±0.007, LatAD ±0.004. All match Table 3. | Regenerate with the bars so their centres stay on the bar tops. |
| Caption text (OK) | "HAI (0.845, +0.088 over the AutoEncoder, 95% CI [0.042, 0.157])", "SWaT (0.837 vs 0.782, P = 0.007)", "WADI (0.771, tying the linear baseline over 8 episodes)", "USAD and TranAD fall to 0.44-0.48" all match `clean_recompute.md`. "Tying" is a fair reading of +0.020, P = 0.46. | none |
| Reference (OK) | Cited at line 845 "(Figure 3)". No orphan. | none |
| Legibility (OK) | Largest text right edge 602 of 720; nothing overlaps the legend (y 24-51) or the bars (y >= 100). | none |

### Figure 2 (lines 357-442): LatAD workflow

| element | issue | fix |
|---|---|---|
| S3. Caption vs figure tags | Caption assigns the whitened-residual head to A7 and the figure footer also tags "[A7] auto-gated whitened residual"; Table 1 row A7 lists both "whitened residual (§4.3, iii)" and "community factorization (§4.4)", so this is consistent. The figure gives the correlation-community block the tag [A7] as well, so A7 appears twice in the legend line. | Optional: tag the community block "[A7, §4.4]" and the residual head "[A7, §4.3 iii]" so the two uses are distinguishable. |
| S3. Caption omission | Caption never says which boxes are the base score vs the fused score, nor that the dashed box means optional (the figure does, in-line). | Add "dashed box = optional head" to the caption. |
| Legibility (OK) | Longest string (footer legend line) ends at x=627 of 680; subtitle at 598. Nothing clipped; the stacked "communities G1 ... GK" cards and the residual-head dashed box render inside the frame. | none |
| Reference (OK) | Cited at lines 348, 458, 530, 601. | none |

### Figure 1 (lines 94-157): IIoT monitoring context

| element | issue | fix |
|---|---|---|
| S3. Caption/figure agreement (OK) | Caption describes exactly the five blocks, the arrow labels (measurements, multivariate telemetry, windowed stream, surprise score, alerts + diagnosis) and the grey/blue legend, all present in the SVG. | none |
| S3. Redundancy | The SVG carries its own title, subtitle, and a three-line footer that repeats the caption ("Model fitting and calibration use train-normal telemetry only..."). In print the same sentence appears twice within 2 cm. | Drop the SVG title/subtitle/footer (or the caption sentence). Same pattern in Figure 2. |
| Legibility (OK) | Subtitle right edge 689.7 of 720; "diagnosis" label 689.5. Inside the frame with 30 px margin. | none |
| Reference (OK) | Cited at line 92 "(Figure 1)". | none |

---

## TABLES

### Table 3 (lines 783-835): per-dataset, per-subset results

Numbers: **all 27 method rows x 6 cells verified against `clean_recompute.md`** (WADI rows to the 2-decimal F1 of the
source; HAI/SWaT F1 at 3 decimals are finer than the source's 2 and round to it). GDN difficult 0.660 / 0.481 / 0.665
correct. GDN All/Easy (0.801 / 0.963, 0.845 / 0.970, 0.876 / 0.997) are not in the source of truth and could not be
checked. Group headers: WADI "56 = 26 + 30", HAI "652 = 485 + 167", SWaT "233 = 148 + 85" all correct.
Bold: WADI All LinRes 0.834, Easy trivial 0.999, Difficult LatAD 0.771; HAI All 0.948, Easy 0.984, Difficult 0.845;
SWaT All 0.938, Easy trivial + LinRes 1.000 (joint), Difficult 0.837. All correct.

| element | issue | fix |
|---|---|---|
| S2. Caption vs WADI block | Caption: "Learned detectors report AUROC as mean±standard deviation over five seeds (omitted where it rounds to 0.000)". The entire WADI block (IF, AE, USAD, TranAD, LatAD global, LatAD regime-community) shows no ± at all, and these SDs are not 0.000 (Table 4 shows WADI LatAD ±0.040, global ±0.013). | Add the WADI seed SDs (from `clean_ensemble_WADI.json` / `scores_WADI_clean.npz`), or add "WADI seed SDs not shown" to the caption. |
| S2. "±0.000" contradicts caption | SWaT AutoEncoder Easy "0.996±0.000" and SWaT LatAD (regime-community) Easy "0.995±0.000" print the SD the caption says is omitted. | Drop the "±0.000" on both cells. |
| S3. F1 precision mixed within columns | HAI and SWaT baselines give F1 to 3 decimals (0.666, 0.803...) while the LatAD (regime-community) rows give 2 (0.76, 0.82, 0.36; 0.83, 0.92, 0.57) and the whole WADI block gives 2. | One precision per table (3 decimals, or 2 everywhere). |
| S3. "Group" column undefined | Caption does not define Baseline / SOTA / Ours. | One clause: "Group: classical baseline, deep SOTA, or ours." |
| S3. Header "trivial max\|u\|" | Caption never expands the trivial rule (it is defined in §5.5 / "Validity of the difficulty split"). | Optional pointer "(univariate max-mean rule, §5.5)". |
| Body prose (OK) | Lines 838-864: 0.948 / 0.938, 0.827 vs 0.834, 0.717 / 0.933 / 0.927, WADI baselines 0.750 / 0.634 / 0.628 / 0.613 / 0.579, +0.020 CI [-0.161, 0.205] P = 0.46, 8 episodes, HC_coh 0.795, global 0.634: all match. | none |

### Table 4 (lines 903-924): double-hard subset

Numbers: all 9 rows x 3 verified, including GDN 0.550 / 0.469 / 0.615 and sizes WADI 19 / 7, HAI 84 / 19,
SWaT 59 / 18 episodes. Bold 0.662 / 0.814 / 0.773 are the column maxima. Caption CIs and P-values (+0.084
[0.019, 0.189] P = 0.0005; +0.085 [0.030, 0.146] P = 0.0005; +0.056 [-0.244, 0.305] P = 0.43) match.

| element | issue | fix |
|---|---|---|
| S2. Caption vs WADI column | "five-seed mean±standard deviation for the learned detectors", but WADI IF 0.530, AE 0.535, USAD 0.466, TranAD 0.489 carry no SD while the HAI/SWaT counterparts do (and WADI LatAD rows do). | Add the WADI baseline SDs or state the omission. |
| S3. Bold undefined | Caption does not say what bold marks (Table 3's does). | "Bold marks the best AUROC per column." |
| S3. GDN unlabeled | GDN row has no "single seed" marker here, unlike Table 3's caption note. | "GDN single seed, as in Table 3." |

### Table 5 (lines 995-1002): source of gain

Numbers: 0.656±0.016 / 0.639±0.004 / +0.017; 0.789±0.018 / 0.589±0.004 / +0.200; 0.795±0.010 / 0.654±0.002 / +0.141.
All match. n = 30 / 85 / 167 correct.

| element | issue | fix |
|---|---|---|
| S3. Caption order vs column order | Caption lists "WADI, canonical SWaT, and HAI difficult subsets (30 / 85 / 167)" but the columns run WADI, HAI, SWaT. The counts are right for the caption's own order, but a reader mapping "85" to the HAI column will think it is wrong. | Reorder the caption to "WADI, HAI, and SWaT (30 / 167 / 85)". |
| S3. Bold semantics | `win` on +0.200 and +0.141 but not +0.017; bold here means "gain that matters", not "best in column", and the caption does not say so. | "Bold marks gains that survive the significance test" or drop the bolding. |
| Prose (OK) | Line 992 "+0.200 on HAI and +0.141 on canonical SWaT ... WADI +0.017" matches. | none |

### Table 6 (lines 1048-1061): score-head ablation

Numbers: all 5 rows x 3 verified (WADI 0.740 / 0.656 / 0.603 / 0.634 / 0.634; HAI 0.695 / 0.802 / 0.797 / 0.801 /
0.820; SWaT 0.835 / 0.794 / 0.723 / 0.775 / 0.807, with SDs). Caption claim "reproduces its Table 3 row within seed
variation" holds: WADI 0.634 = 0.634; HAI 0.820±0.010 vs 0.811±0.016; SWaT 0.807±0.008 vs 0.804±0.007.

| element | issue | fix |
|---|---|---|
| S3. Bold row is not the column best | Bold last row (0.634 / 0.820 / 0.807); the WADI and SWaT column maxima are the dropped reconstruction head (0.740, 0.835). Bold marks "the shipped configuration", but the caption does not say so and Table 3's convention is "best in column". | Say "bold = the configuration shipped as LatAD (global density)". |
| S3. Caption defines "recon" but the row label says "reconstruction residual (dropped term)" | Term used in caption (`recon`) never appears in the table. | Align: either label the row "recon" or change the caption. |
| Prose (OK) | Lines 1064-1072: 0.695 / 0.802, WADI 0.740 / 0.656 / 0.634, HAI 0.801->0.820, SWaT 0.775->0.807, 0.835 all match. Table 2 caption's cross-reference ("as Table 6 ... show[s] it does not generalize") consistent with base+resid = base on WADI. | none |

### Table 2 (lines 507-519): two natural score terms

Numbers: WADI 0.742 / 0.621 / 0.733, HAI 0.689 / 0.760 / 0.689: match the supplied values. Caption states
"single trained model per dataset" (n = 1 seed) and points to Table 6 for the multi-seed version.

| element | issue | fix |
|---|---|---|
| S3. No SWaT column | Table covers WADI and HAI only; the caption acknowledges this indirectly ("three-dataset head ablation is reported in Table 6"). | Fine as is; optionally say "SWaT omitted here, see Table 6". |
| S3. Caption is an argument, not a description | Two of its three sentences interpret the result (mechanism, gating). Acceptable in this venue's style but long for a 3-row table. | Optional trim. |
| S3. Row (b) vs Table 6 | "(b) latent NLL (nearest diagonal component)" 0.621 on WADI vs Table 6 "nearest-component NLL" 0.603±0.010; different run (single model vs 5-seed), within plausible spread. | none, or note "single model" in the row label. |

### Table 1 (lines 281-330): assumptions

| element | issue | fix |
|---|---|---|
| S3. Caption sentence 1 vs grouping | "A1-A8 shape the instantaneous window; A9-A10 shape the trajectory" and then "Rows are grouped into A1-A7 ... and A8-A10". Both are true (A8 is a window assumption that is not realized), but a reader sees A8 on two sides of two different splits within one caption. | "A1-A8 concern the instantaneous window and A9-A10 the trajectory; the table is grouped by realization status, A1-A7 realized, A8-A10 specified only." |
| Columns (OK) | Five headers (ID, Assumption, CPS rationale, Motivates, Realized by) match the five cells per row and the caption's "last column names the section". Numbers "dim 10-16" and "3.4-3.7 decades" are qualitative descriptors; no source to check. | none |
| Reference (OK) | Cited at line 271. | none |

### Table A1 (lines 1205-1225): factorization / aggregation ladder

Numbers: all 8 rows verified (0.634 / 0.811 / 0.804; 0.795 / 0.801 / 0.822; 0.771 / 0.845 / 0.837; 0.698 / 0.827 /
0.805; 0.712 / 0.809 / 0.804; 0.760 / 0.793 / 0.826; 0.771 / 0.845 / 0.837; 0.749 / 0.770 / 0.811). Caption
"first four ... last fuses by a maximum" matches the 4 + 1 rows of the aggregation block.

| element | issue | fix |
|---|---|---|
| S3. `win` marks split across duplicated row | The headline configuration appears twice (factorization block row 3 and aggregation block row 4, same numbers). HAI/SWaT `win` is on the aggregation copy only; WADI `win` correctly on density-only 0.795. Per-column "best" is therefore marked once for HAI/SWaT although two rows tie. | Mark both copies, or say "bold = best per block". |
| S3. Missing ladder rung | `clean_recompute.md` Table A1 has a "cohesion-max (+LatAD)" row (0.741 / 0.832 / 0.829) that the paper omits. Not an error; it is the rung between max and HC. | Optional add. |
| Prose (OK) | Line 1198 "0.634 to 0.795 ... fused headline 0.771" matches. | none |

### Table A2 (lines 1239-1250): feature-representation check

| element | issue | stale -> correct / fix |
|---|---|---|
| **S2. WADI column stale (old 43-window mask)** | Cells `0.673±0.006 / 0.675±0.015 / +0.002`. `clean_recompute.md` §Table A2: "WADI A2 was not re-run in this pass; a focused re-run (~5 min, 5 seeds CPU) would refresh its WADI column." Every other WADI difficult number in the paper is on the 30-window subset (and the LatAD-global six-statistics reference is now 0.634, not 0.67x). The caption's "fixed difficulty mask" is now a different mask from the rest of the paper. | Re-run the WADI A2 pair on the 30-window mask (~5 min agent time, CPU) and replace the three WADI cells; or add "WADI column computed on the pre-revision 43-window mask" to the caption. HAI 0.743 / 0.814 / +0.072 and SWaT 0.750 / 0.776 / +0.026 are confirmed unchanged. |
| S3. Row label "(headline)" | "six summary statistics (headline)" while the caption says these "are not the headline numbers of Tables 3-5". Two meanings of "headline" in one table. | Rename the row "six summary statistics (used by LatAD)". |

### Table B1 (lines 1264-1276): computational cost

No source-of-truth file; checked against the Appendix B prose (lines 1254-1261). Train 3.2 / 16 / 3.5 vs TranAD 231 /
898 / 163 gives 72x / 56x / 47x, matching "47 to 72 times". GPU 0.7-13.0 (global), 17.2-32.4 (community), CPU-edge
0.6-14.6 / 14.5-25.0, params 100k-209k / 0.78M-1.25M: all match the prose.

| element | issue | fix |
|---|---|---|
| S3. Caption states a quantity the table lacks | "run within about 3.5-3.9 GB of host memory" has no column. | Add a "Host mem GB" column or move the sentence to the prose only. |
| S3. "0.0 / 0.0" train seconds | AutoEncoder train "0.2 / 0.0 / 0.0 s" reads as zero. | "<0.1". |
| S3. Prose says "0.15 ms for a plain autoencoder", table says 0.16 / 0.15 / 0.15 | Rounding only. | none |

### Table C1 (lines 1300-1317): A8 overlap screen

No source-of-truth file; caption defines all four columns and the 0.61 positive-control anchor matches the row.

| element | issue | fix |
|---|---|---|
| S3. Prose promises two controls, table shows one | Line 1291: "Two synthetic controls calibrate them: a 30-percent-overlap regime pair, and a drifting-onto-a-neighbour excursion with a matched ... negative control"; line 1293 "and the controls". The table has a single row "positive control". | Add the drift-excursion (and its negative control) rows, or change the prose to "the overlap control". |
| S3. Prose range vs table | Line 1295 "close-distinct mass 0.94 to 1.00" but the table's minimum is 0.99 (SWaT); 0.94 must come from the broader screen not shown. | "0.99 to 1.00 in Table C1 (0.94 to 1.00 in the broader screen)". |
| S3. Mixed precision | Isolated-missed "0.06 / 0.005", "0.80 / 0.13", "0.997 / 0.006". | Three decimals throughout the column. |

---

## Cross-cutting checks

- **43 vs 30**: only occurrence of "43" as a count is line 861 ("43-to-30 change"), which is the intended explanation.
  Table 3 header, Table 5 caption, Figure 3 prose (line 853), Table A1 prose all say 30. Double-hard 19 / 84 / 59 in
  Table 4 caption and prose (lines 896, 909).
- **Old headline numbers** (0.824, 0.849, 0.840, 0.734, 0.787, 0.681, 0.739, 0.722, 0.685, 0.819, 0.775 as a
  double-hard value, 0.092, 0.058, 0.037, 11 episodes, P = 0.003): none present. The two hits for "0.775" are the
  correct Table 6 SWaT base head.
- **Orphans**: none. Figures 1, 2, 3 and Tables 1-6, A1, A2, B1, C1 are each cited at least once in the body.
- **Clipping**: none. Maximum text right-edge per SVG: Fig 1 689.7/720, Fig 2 626.8/680, Fig 3 602.1/720.

## Ranked to-do (shortest path)

1. Regenerate Figure 3 bars + whiskers on the single mapping `y = 250 - (v - 0.3) * 271.43`; add the 1.0 gridline. (S1)
2. Decide on GDN in Figure 3: add bars or say "Table 3 only" in the caption. (S2)
3. Re-run Table A2 WADI on the 30-window mask, or caption the old mask. (S2)
4. Supply WADI seed SDs in Tables 3 and 4 and WADI whiskers in Figure 3, or state the omission in each caption. (S2)
5. Delete the two "±0.000" cells in Table 3; unify F1 precision. (S2/S3)
6. Reorder the Table 5 caption dataset list to match the columns. (S3)
7. Define bold in Tables 4, 5, 6 (each uses a different meaning). (S3)
8. Table C1 second control row, Table B1 memory column, Table A2 "headline" label. (S3)
