# Review-3 line-by-line accuracy audit of `poc/paper/IoT2.html`

Scope: every line of the file (1396 lines) read sequentially, reconciled against
`_diagnostics/clean_recompute.md` and `_diagnostics/review2_recompute.md`. Change-marking was mentally
accepted (green kept, red dropped) and the resulting clean text checked for fragments.

## What checks out (no defect)

- Every headline number matches the source of truth: difficult 0.845 / 0.837 / 0.771, HC_coh 0.795,
  LatAD-global WADI 0.634, All 0.948 / 0.938 / 0.827 (LinRes 0.834 leads WADI-All), significance
  HAI +0.088 [0.042, 0.157] P~0, SWaT +0.055 [0.012, 0.104] P=0.007, WADI vs LinRes +0.020 [-0.161, 0.205]
  P=0.46, WADI vs learned detectors AE +0.142 P=0.001 / USAD +0.192 / TranAD +0.157 / IF +0.136 / GDN +0.111
  (Sec 6, lines 857-865), GDN difficult 0.660/0.481/0.665 and double-hard 0.550/0.469/0.615, Table A2 WADI
  0.634 / 0.626 / -0.009, Table 2 WADI 0.742 / 0.621 / 0.733, Tables 4, 5, 6, A1 all cell-for-cell.
- All WADI SD cells in Tables 3 and 4 match review2_recompute Section 2.
- Figure 3: all 21 bars decoded from the SVG with y = 250 - (v - 0.3) x 271.43 reproduce Table 3 to 3 dp;
  whiskers match the SDs; GDN bars present; legend has 7 entries.
- Table B1 arithmetic: 231/3.2, 898/16, 163/3.5 = 72x, 56x, 47x, matching "47 to 72 times".
- Every "Section / Table / Figure / Equation / Appendix" cross-reference resolves to an existing target
  that says what the sentence claims. No orphan float: Tables 1-6, A1, A2, B1, C1 and Figures 1-3 are
  each cited in prose at least once. Equations (4), (5), (7) are cited; (1)-(3), (6), (8), (9) are inline.
- In-text citation numbers [1]-[59] all match the position of the target `id` in the reference list.
- Accept-all: no unclosed span, no doubled word, no broken sentence or math after accepting every mark.
  The deleted paragraph at lines 228-232 leaves an empty `<p>`, harmless.

## Defects (ranked)

| # | Severity | Line / section | Issue (quoted) | Correct value or fix |
|---|---|---|---|---|
| 1 | HIGH | 82, Abstract | "ahead of every learned detector on WADI (0.771), where it ties the linear baseline **on the full set**" | Wrong subset. On the full set LinRes is ahead (0.834 vs 0.827; Sec 6 line 840-841 and 864-865 say so explicitly). The tie (P=0.46) is on the difficult subset. Change to "where it ties the linear baseline (P = 0.46)" or "...on the difficult subset". As written the abstract contradicts Sec 6. |
| 2 | MEDIUM | 1329 vs 1135/1155/1302; reference list | SKAB is reference [54] but is first cited at line 1329 (Appendix C), after [55] HI-VAE, [56] VAEM, [57] TABOR (line 1135), [58] CalexNet (line 1155) and [59] Cranfield (line 1302). | First-appearance order (MDPI) is broken for 54-59. Correct order: 54 hivae, 55 vaem, 56 tabor, 57 calexnet, 58 cranfield, 59 skab. Renumber the six list entries and the seven in-text citations (lines 1135 x3, 1155, 1302, 1329). |
| 3 | MEDIUM | 1216-1218, Table A1 caption vs rows 1227-1228 | Caption: "Top block: the factorization axis (single global density, then the regime-community ensemble with **density-only versus full per-community heads**)". Row labels: "cohesion-weighted HC, **no whole-plant-expert fusion**" (0.795/0.801/0.822) vs "cohesion-weighted HC **+ whole-plant expert** (headline)". | The row labels (and Appendix A prose, line 1208, and Sec 6 line 862) attribute the row-2/row-3 difference to the whole-plant-expert fusion, which matches the source of truth (HC_coh vs HCcoh+LatAD = same HC, plus the LatAD null term). The caption still describes the old "density-only vs full heads" contrast. Rewrite caption top-block sentence to "...then the regime-community ensemble without and with the whole-plant-expert fusion". |
| 4 | MEDIUM | 749-751, Sec 5.4 | Block length "ceil(W/stride) + 1 windows (3 on WADI, 4 on HAI and SWaT)" | With W = 60 and stride 30 "unified across all three datasets" (lines 698, 704, 714) the formula gives 3 on every dataset. Either the formula, the stated lengths, or the "unified" claim is wrong; check the bootstrap script and fix whichever is stale. |
| 5 | MEDIUM | 245, 736-739, 762 vs Table 3 rows 807-808, 829-830 and Fig 3 | Text: USAD and TranAD are run "over five seeds" and AUROC is "reported as a five-seed mean for the learned detectors". Table 3 WADI and SWaT USAD/TranAD rows carry no SD (only HAI does), Figure 3 draws no USAD/TranAD whiskers on WADI or SWaT, and review2_recompute.md Section 2 lists "Single-run methods (LinRes, trivial, USAD, TranAD, GDN) carry no SD". | Verify how many seeds USAD/TranAD actually have on WADI and SWaT. If single-run there, say so in Sec 5.5 and the Table 3 caption (the caption's "SD omitted where it rounds to 0.000" would otherwise be read as five identical seeds). |
| 6 | LOW | 1009, Table 5 caption | "Bold marks the cross-channel gains that survive the significance test." | No significance test of the Table 5 gains is reported anywhere; the only bootstraps are headline-vs-baseline (Sec 5.4/6). Either drop the sentence or replace with "Bold marks the two datasets on which LatAD's difficult-subset margin is significant (Sec 6)". |
| 7 | LOW | 1159, 1270-1271, 1276-1277 (Appendix B text, Table B1 caption, Sec 7) | "both configurations run within about 3.5 to 3.9 GB of host memory" | Table B1 has columns Params / Train s / GPU ms/win / Peak GPU MB / CPU-edge ms/win; there is no host-memory column, so the 3.5-3.9 GB figure is stated three times with no supporting cell. Add a "Host RAM GB" column or drop the claim from the caption. |
| 8 | LOW | 1041, 1078 | "the whitened reconstruction residual is the weakest **base head** on HAI (0.695)"; "informative but the weakest base head (0.695)" | Sec 4.3 and the auto-gating note (598-599) define the base heads as (i)-(ii) only; the residual is the optional head. Say "the weakest of the three heads" or "weaker than either base head". |
| 9 | LOW | 1250-1254, Table A2 caption | "the absolute values use a reduced global-density configuration and are not the headline numbers of Tables 3-5" | The WADI six-statistics cell (0.634 +/- 0.011) is identical to the Table 3 LatAD (global density) difficult cell and the Table 6 base row (the residual is gated off on WADI, so the reduced and headline global configurations coincide). The caveat is true only for HAI and SWaT; qualify it ("on HAI and SWaT") or the reader will see a contradiction. |
| 10 | LOW | 803-811 vs 814-833, Table 3 | WADI F1 cells are 2 dp ("0.62", "0.39") while HAI and SWaT F1 cells are 3 dp ("0.666", "0.418"). | Format the WADI F1 column to 3 dp (values available in clean_ensemble_WADI.json) or round HAI/SWaT to 2 dp. |
| 11 | LOW | 258, Sec 2.5 | After accepting the marks the whole paragraph reduces to "(Per-dataset details are given in Section 5.1.)" | A lone parenthetical paragraph. Fold it into the next paragraph or un-parenthesize ("Per-dataset details are given in Section 5.1."). |
| 12 | LOW | 191, contribution (3) | "comparing against classical baselines and the deep detectors USAD and TranAD re-computed with the same raw metrics" | GDN is also benchmarked (Sec 2.4, 5.5, Table 3, Fig 3). Add "and GDN" for consistency with the rest of the paper. |
| 13 | LOW | 1226-1234, Table A1 | Bold `win` marks: WADI bold on top-block row 2 (0.795), HAI/SWaT bold on the aggregation-block headline row (0.845/0.837) while the identical top-block headline row (1228) is unbolded; the caption states no rule for bold. | Either bold the column maximum in each block consistently or add "Bold marks the column maximum" to the caption and apply it to both occurrences. |

## Newly edited spots specifically checked

- Sec 6 WADI significance sentence (857-865): all six numbers and both P-values match review2_recompute
  Section 3; CI for LinRes uses the clean_recompute run ([-0.161, 0.205]); consistent with Sec 7 (1025-1029)
  and the Conclusion (1180). No defect.
- Table A1 row-2 label (1227): label and values (0.795 / 0.801 / 0.822) correct; only the caption is stale (item 3).
- Sec 7 A8 paragraph (1093): percentages (94-100, 30, 80, 13) agree with Appendix C and Table C1; citation
  targets resolve. No defect.
- Table B1 GPU-memory column (1278-1284): present, labeled "Peak GPU MB", consistent with the caption
  definition and with the Appendix B prose "tens of MB ... peak 20 to 29 MB". No defect (host-memory
  claim is item 7).
- Table A2 (1257-1259) and the WADI SD cells (Tables 3, 4): all match. No defect.
