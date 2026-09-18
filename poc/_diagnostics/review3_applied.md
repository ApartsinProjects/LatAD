# Review-3 fix pass: applied edits

Scope: HTML edits only. No DOCX rebuild. No reference renumbering (citation numbers left untouched;
user renumbers after). Change-marking in the manuscript uses `<span class="rev2-highlight">` (green) for
new/changed text and `<span class="rev-del">` (red strike) for deletions; prior yellow `rev-highlight` kept.
Letters have no change-mark CSS, so letter edits are applied as clean text (letters are response documents,
not the change-tracked manuscript).

Self-check (all pass): no em-dash / `&mdash;` / double-hyphen introduced in prose; no "honestly/frankly/candidly";
"Apartsin" spelled correctly (2x each file); `<span>` open/close balanced 293/293; `<math>`/`</math>` balanced
53/53 (one sanctioned MathML deletion, see edit 2); 3 `<svg>` present; Figure 3 SVG untouched.

## Manuscript `poc/paper/IoT2.html`

### 1. Abstract WADI subset (verified resolution 1; accuracy #1, H1, letters A#1)
- Line 82. `where it ties the linear baseline on the full set` -> struck; inserted (green) `statistically tied
  with the linear baseline on the difficult subset (P = 0.46)`. Rest of abstract left byte-identical
  (did NOT drop "canonical", per user override of H1's secondary suggestion).

### 2. Section 5.4 block length (verified resolution 2; accuracy #4, M13, letters A#2)
- Lines 749-751. Struck `with block length ⌈W/stride⌉+1 windows (3 on WADI, 4 on HAI and SWaT)` including the
  `<math>` formula element (sanctioned MathML deletion, -1 math element); inserted (green) `with a block length
  of 3 windows on WADI and 4 on HAI and canonical SWaT, exceeding the window-overlap span`. Formula dropped;
  values kept per user's verified resolution.

### 3. USAD/TranAD seed count (verified resolution 3; accuracy #5, finding #10)
- Sec 2.4 (line 245): struck `over five seeds` -> green `(five seeds on HAI, a single run on WADI and
  canonical SWaT; §5.5)`.
- Sec 5.5 (line 761): struck `over five seeds` -> green `(over five seeds on HAI, where per-seed standard
  deviations are reported, and as a single run on WADI and canonical SWaT, where no standard deviation is shown)`.
- Sec 5.4 AUROC bullet (line 737): appended green qualifier naming IF/AE/LatAD as five-seed on all three and
  USAD/TranAD five-seed on HAI only, single-run on WADI/SWaT (§5.5).
- Table 3 caption (line 786): struck the old `Learned detectors report AUROC ... omitted where it rounds to
  0.000.` -> green replacement stating IF/AE/LatAD five-seed on all three; USAD/TranAD five-seed on HAI (SD shown),
  single run on WADI and canonical SWaT (no SD).

### 4. Table B1 host-memory column (verified resolution 4; accuracy #7, finding #3, letters A#3)
- Added `Host RAM GB (W/H/S)` header column (green) and one cell per row (green):
  LatAD (global density) 3.5/3.7/3.9; LatAD (regime-community) 3.5/3.8/3.9; TranAD/USAD/AutoEncoder "–".
  Values are the CPU-edge-proxy peak host RAM from `_diagnostics/e3_rows_cpu.jsonl`
  (3529.6/3725.6/3865.7 MB global; 3529.6/3753.5/3865.7 MB community), which is what the "3.5 to 3.9 GB"
  prose refers to. Caption updated (green) to define the column and note deep-baseline host memory not measured.
  The "3.5-3.9 GB host memory" claim in prose (§7, App B) and all three letters is now backed by a real column.
- Also renamed the Table B1 row label `LatAD (global)` -> `LatAD (global density)` (L16 unification).

### 5. Table 5 caption significance clause (verified resolution 5; accuracy #6, M17, finding #4)
- Line 1009. Struck `Bold marks the cross-channel gains that survive the significance test.` -> green
  `Bold marks the two datasets on which LatAD's difficult-subset lead over the strongest baseline is
  significant (§6).`

### 6. Table A1 caption + prose (verified resolution 6; accuracy #3, H5, H6)
- Caption top block (line 1217): struck `with density-only versus full per-community heads` -> green
  `without and with the whole-plant-expert fusion`.
- Caption (line 1219, H6): struck doubled adjective `global null` -> green `whole-plant` (now reads "fused with
  the whole-plant expert").
- Prose (line 1211, H5): replaced the un-parsing closing parenthetical. `leads on all three <once the community
  experts and the whole-plant expert are fused (and, on WADI, the community aggregation without that fusion leads
  outright)>` -> `leads on all three datasets` + green `(all aggregation rules here fused with the whole-plant
  expert)`; the WADI "carries the whole effect" point is already stated in the preceding kept clauses, so no loss.

### 7. Section 3 "each assumption motivates" (verified resolution 7; H7)
- Line 272: struck `and each motivates` -> green `and A1–A7 each motivate` a specific design choice.
- Line 272 green sentence: struck `each property` -> green `each of A1–A7`; appended green
  `; A8–A10 are specified and measured but not realized (§7)`. (Table 1 caption already accurate; not re-done.)

### 8. Section 7 self-contradiction (verified resolution 8; H2, finding #5)
- Lines 1138-1140: struck the whole `Because the community construction ... developed on these three benchmarks
  ... fixed before the WADI and SWaT evaluations reported here ... out-of-sample confirmations of the design.`
  -> green replacement keeping ONE accurate statement: `The community construction and combiner settings were
  fixed before the WADI and canonical SWaT evaluations, so those two results are out-of-sample confirmations of
  the design. Validating the regime-community realization on additional CPS systems is the next test; because its
  build and calibration use train-normal data only, that test requires no labels.` Dropped the contradictory
  "developed on these three benchmarks" and the "reported here" timeline narration.

### 9. Section 7 A8 paragraph + Appendix C (verified resolution 9; H3, H4)
- Bold lead-in (line 1092): struck `: specified, over-coverage defended, masking coupled to A10` ->
  `Assumption A8 (between-regime overlap).`
- Inserted green gloss after "stays in range.": defines the two failure modes on first use, over-coverage
  (coarse density head spreading mass across the gap) and masking (anomaly absorbed by an adjacent regime).
- Screen-measure count: struck `(overlap coefficient, density-valley ratio, cross-regime nearest-neighbour share,
  and discriminant error)` -> green `(two of which are tabulated in Table C1: the overlap coefficient and the
  density-valley ratio)`, consistent with what Table C1 shows.
- `The one A8 variant the reported model already defends is over-coverage:` -> green `The over-coverage mode is
  already defended by the reported model:`; `Masking, the remaining variant,` -> green `The masking mode`.
- Line 1127 (H4d): struck A8 alias `between-regime proximity and masking` -> green `between-regime overlap`
  (one consistent A8 name throughout).
- Appendix C SKAB (line 1329, H3): struck the responsibility-entropy / variance-floor / basin-head debugging
  sentence entirely; kept the forward statements (precondition absent on SKAB; A8 specified but not demonstrated).
  Removed the dead reference to the withdrawn basin head.

### 10. Section 4.3(iv) forward statement (verified resolution 10; H8, M4)
- Line 596: struck `The stack has no head for A8 (between-regime overlap): that assumption is specified in Table 1
  (§3) but not realized by this model, so Equation (7) is the detector's complete window score.` -> green forward
  statement `Equation (7) is the detector's complete window score; A8 (between-regime overlap) is specified in
  Table 1 (§3) and addressed in §7 rather than by a dedicated head.` Struck the trailing colon clause
  `: the residual head fires on HAI and SWaT and stays off on WADI` (now ends "...no test-set tuning.").

### 11. Smaller bundle items
- CPS/IIoT double definition (M1, line 90): struck `networked` -> `such` sensors; struck `: the
  sensing-and-actuation fabric of the Industrial Internet of Things (IIoT)` (CPS/IIoT already defined at line 88).
- Residual-gate restatement reduced (M4/M5): removed the 596 restatement (edit 10) and struck the redundant tail
  `, with the auto-gated residual head adding to them on HAI and SWaT (Table 6)` at line 449. Measured statement
  (§4.3 iii ratios) and the auto-gating note box kept.
- "both deep detectors on both" (M14, line 839): -> green `all three deep detectors on each`.
- Duplicated WADI full-set sentence (M14, line 864): struck `On the full set the linear baseline is marginally
  ahead of LatAD (0.834 vs 0.827).` (already stated in the Overall-detection paragraph, line 840-841).
- "illusion of success" -> "illusion of progress" (M16, line 935), matching the Wu-and-Keogh phrase in §2.2.
- Table 6 gate causal slip (M20, line 1083): struck `, which is why the gate keeps it on there` -> green
  `; the gate, which sees only train-normal, keeps the head on there because the residual generalizes to held-out
  normal`.
- Contribution (3) (accuracy #12, line 191): struck `USAD and TranAD` -> green `USAD, TranAD, and GDN`.
- SWaT filename token (M12, line 685): struck `<code>SWaT_Dataset_Attack_v0</code>` -> green `release-v0`.
- LatAD naming unified to `LatAD (global density)` (L16): §7 line 1157, App B line 1265, Table B1 caption
  line 1276, and Table B1 row label (edit 4).
- UK->US spelling (L21), applied unmarked as orthographic-only changes (marking each would clutter; documented
  here as the sanctioned exception): mislabelling->mislabeling, manoeuvres->maneuvers (x2),
  initialisation->initialization (x3), neighbour->neighbor (x4), labelled->labeled (x4). Reference titles
  (e.g. "behavioural taxonomy") left as the authors published them.

## Letters (`review_round_1/`)

- `letter_editor.html`: (a) A8 control wording `synthetic 50 percent-overlap positive control` -> `synthetic
  overlap positive control (30 to 2 percent Bayes error)`, aligning to the paper's Appendix C sweep. (b) Req 4
  WADI under-sell fixed: added that on WADI LatAD is significantly ahead of every learned detector (AE, USAD,
  TranAD, IF, GDN, each P < 0.05) and ties only the linear baseline, matching the paper and the R1 letter.
  (Host-memory attribution is now real after the Table B1 column, so no phantom remains; left as is.)
- `letter_reviewer1.html`: A8 control wording `synthetic 50 percent-overlap positive control` -> `synthetic
  overlap positive control (30 to 2 percent Bayes error)`.
- `letter_reviewer2.html`: (a) C8 location fixed: `Section 4 opens by stating the relationship explicitly:` ->
  `Section 4 states the relationship explicitly, immediately after the workflow figure:` (the quoted sentence is
  the third paragraph of §4, not the opener). (b) C10 list now includes `peak GPU memory` (R2 asked for it and
  Table B1 holds it).
- Cross-reference re-verification: every Section / Table / Figure / Appendix / equation-count / number citation
  in all three letters was checked against the manuscript. All resolve correctly (GDN in Tables 3-4 and Fig 3 and
  §§2.4/5.5/7; Table 6 nearest-component 0.603/0.797/0.723; Table B1 train times 50/195/38 and 3.2/16/3.5;
  significance CIs on HAI/SWaT/WADI and double-hard; Table 5 +0.200/+0.141/+0.017; Table A2 −0.009/+0.072/+0.026;
  nine equations; §4 quote). No further corrections needed.

## Deliberately NOT done (with reason)

- Reference renumbering (accuracy #2, L18, L22, finding #21; SKAB [54] first-appearance order): user runs the
  renumber afterward; citation numbers left untouched by instruction.
- Items requiring new experiments / new data / rerun code, out of scope for an HTML text pass and (where relevant)
  already declined in the letters: finding #6 (headline model on six-statistic split for all three datasets),
  #11 (localization top-k hit-rate), #12 (hyperparameter table), #18 (GDN F1 / five seeds), #19 (percentile
  sensitivity), #20 (A5 imbalance sweep), #7/#8/#9/#13 argument-strengthening rewrites of §7 that hinge on new
  numbers or claims not verified by the user.
- Figure 2 SVG missing `[A7]` tag on the residual block (finding #22): SVG edit deferred to avoid touching the
  drawing; the caption/footnote already state the A7 tagging. Not a text defect.
- Optional prose-polish items in the language report not included in the user's specified fix set (M2, M3, M6-M11,
  M15, M18, M19, M21-M29; L1-L15, L17-L20): left for a later style pass. All substantive accuracy, internal-
  consistency, and the user-enumerated language fixes are applied above.
