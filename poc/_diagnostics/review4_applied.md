# Review 4 applied: change log

Round-4 fixes applied to `poc/paper/IoT2.html` (the active paper) and the three response letters
in `review_round_1/`. HTML only; no DOCX rebuild, no reference renumbering, no Figure A1 curve or
axis-label change (the author regenerates the axis label separately). Change-marking preserved
(green `rev2-highlight` = new, red `rev-del` = struck). Every string quoted below is the current
(accepted) reading; where the touched text was already inside a `rev2-highlight` span (new-in-rev2
content), it was edited in place and stays green, so no red strike is added for text that never
appeared in the previously submitted version.

## Paper edits (`poc/paper/IoT2.html`)

### Fix 1 + 2 - Section 7 density sentence garbled; "matches GDN"; "deployed"->"deep"
Location: Section 7, "Why reconstruction fails and density wins" paragraph (density head sentence),
and Table 6 caption.
- old: "Scoring by density rather than reconstruction is the right default **for the deployed reconstruction detectors**: the single latent-density head ... exceeds **USAD and TranAD** on the difficult subset of all three plants (...) **and matches GDN on WADI (0.660)**."
- new: "Scoring by density rather than reconstruction is the right default: the single latent-density head ... exceeds **the reconstruction-based deep detectors USAD and TranAD** on the difficult subset of all three plants (...) **and is within noise of GDN on WADI (0.656 versus 0.660)**."
- Table 6 caption: "the **deployed** reconstruction detectors of Table 3 score **0.44 to 0.66**" -> "the **deep** reconstruction detectors of Table 3 score **0.44 to 0.67**" (GDN SWaT 0.665 rounds to 0.67; also covers Fix 6 Table 6 rounding).

### Fix 3 - Sections 1 and 8 universal reconstruction-blindness claim nuanced
Location: Section 1 (two sentences) and Section 8 (Conclusion mechanism paragraph). Change-marked
(struck old, green new).
- Section 1, difficult-faults sentence: "...invisible to any per-channel range rule, and, as we show, `[del]`to a reconstruction residual as well`[/del]` `[new]`to the reconstruction residual the deep detectors score with, which on HAI places these faults no better than chance`[/new]`, because the question is not whether a flexible decoder can reproduce the state but whether normal operation is likely to occupy it."
- Section 1, obstacles paragraph: `[del]`"and a reconstruction residual is blind to exactly the joint-structure faults that matter, because a flexible decoder reconstructs them faithfully."`[/del]` `[new]`"and the reconstruction residual the deep detectors score with misses exactly the reconstructable-but-improbable joint-structure faults that matter, because a flexible decoder reconstructs them faithfully; where a fault does leave a reconstruction signal, a regime-conditioned residual stays competitive and is auto-gated back in."`[/new]`
- Section 8: "Treating multimodal normal as a single blob`[del]`, or as a reconstruction target, is a modeling failure.`[/del]``[new]`, or scoring it by reconstruction alone, misses the reconstructable-but-improbable faults, starkest on HAI, where the deep detectors fall to chance; where a fault leaves a reconstruction signal the residual is competitive and LatAD gates it back in.`[/new]`"

### Fix 4 - two over-dense sentences split
- H2 (Section 7 density paragraph, the ~95-word 7-number sentence): split into three sentences
  ("The margin is largest where the faults are reconstructable but improbable." / "On HAI ... in density." / "On WADI that share is zero, and 70 percent ... carries the WADI gain."). Same numbers, same content.
- H5 (Appendix A opener, the ~150-word semicolon-chained "Factorizing the global density over regime communities lifts all three datasets..." sentence): split into four sentences. Semicolons became sentence breaks ("...a global density dilutes." / "On WADI the community aggregation carries the whole effect: ... settles at 0.771." / "On HAI and SWaT the whole-plant-expert fusion instead lifts the score, ... correlation-conditional." / "Among the sparsity-adaptive combiners, ... adapting to the fault's extent."). Same content and numbers; green `rev2-highlight` span and the `rev-del` strike inside it preserved. "is mildly counterproductive there" softened to "costs a little" (inside the already-green text).

### Fix 5 - HAI localization percentage, duplicate top-3, dangling participle, coverage-aware gloss
Location: Section 7 "Subsystem localization" paragraph (entirely new-in-rev2, edited in place).
- "40 percent of HAI attacks" -> "39 percent of HAI attacks" (Table 7 shows 0.39 = 15/38).
- Removed the leftover opener "The per-community surprises also localize." and its dangling "also".
- Removed the duplicated top-3 statement ("...and a three-community shortlist contains it in 79 and 68 percent (random 49 and 56 percent).") - the top-3 headline is now stated once, in the opening sentence.
- Fixed the dangling participle ("Ranking ..., the named community contains"); rewritten as "The ranking is the one the Higher Criticism combiner already computes: communities are ordered by ... and the top community is named once per attack episode. That single named community contains ...".
- Glossed "coverage-aware" on first prose use: "against a coverage-aware random baseline (the chance that a randomly named community of the same size contains an attacked channel) of 22 and 25 percent".

### Fix 6 - Table 3 / Table 4 / Table 6 captions
- Table 3 caption (green text): re-inserted "(the standard deviation is omitted where it rounds to 0.000)" after "...over five seeds on all three datasets" (four five-seed Easy cells still show no SD).
- Table 4 caption: added green sentence "USAD and TranAD are single-run on WADI and canonical SWaT and GDN is single-seed throughout, as in Table 3; the standard deviation is omitted where it rounds to 0.000."
- Table 6 caption: "0.44 to 0.66" -> "0.44 to 0.67" (done together with Fix 1 above).

### Fix 7 - Table C1 sentence named a column that does not exist
Location: Section 7, A8 paragraph.
- old (green): "(two of which are tabulated in Table C1: the overlap coefficient and the density-valley ratio)"
- new (green): "(the overlap coefficient is tabulated in Table C1, alongside its close-distinct-mass, masking-rate, and isolated-anomaly over-coverage columns)"
- Verified against Table C1 columns: Close-distinct mass (min D), Overlap coeff. mean (max), Masking rate / base, Isolated-missed wide / 80-comp. There is no density-valley-ratio column.

### Fix 8 - Table 7 size-weighted baseline quantified
Location: Section 7 "Subsystem localization" limit (ii).
- old: "so the shortlist is reported at its size; against a size-weighted random baseline the SWaT margin holds while HAI's top-1 margin is small, which is why the top-3 shortlist is the comfortable headline."
- new: "so the named-community size is reported and the top-3 headline is stated against the coverage-aware baseline; against a size-weighted random baseline (a community drawn with probability proportional to its size) the SWaT top-1 margin holds (0.46 versus 0.35) while HAI's top-1 margin is small (0.42 versus 0.37, WADI 0.43 versus 0.19), which is why the three-community shortlist is the headline figure."
- Numbers from `framing_brainstorm.md` caveat (i): rank top-1 vs size-weighted 0.458/0.350 (SWaT), 0.417/0.374 (HAI), 0.43/0.19 (WADI). Printed per the user's resolution (0.46/0.35, 0.42/0.37, 0.43/0.19). The HAI size-weighted detector value 0.42 differs from Table 7's coverage-aware 0.39 because they are two different baselines/measurements; both are stated in their own contexts, as instructed.

### Fix 9 - Figure 1 SVG output label
Location: Figure 1 SVG, second-line text run.
- old: `<text ...>diagnosis</text>`  ->  new: `<text ...>triage shortlist</text>` (label reads "alerts + triage shortlist"). Only that text run changed; no other SVG element touched. Fits inside viewBox width 720 (label anchored at x=651).

### Fix 10 - Figure A1 caption closing softened
- old: "This is a synthetic mechanism illustration, not a claim about the benchmarks."
- new: "This is a synthetic mechanism illustration; the benchmark results are in Section 6."
- "synthetic" labeling kept (caption still opens "Synthetic illustration" and says "toy data with five Gaussian regimes"). Axis label and curves untouched.

### Fix 11 - six smaller items
- (a) Figure 3 SVG: removed the leftover HTML comment `<!-- rev2: bars regenerated ... WADI whiskers added -->` (source-only revision narration; drawing untouched).
- (b) Section 4.4: doubled "rank ... ranked" fixed - green "**ranked** by their train-normal upper-tail p-value" -> "**ordered** by their train-normal upper-tail p-value" (the surrounding "rank the plant's subsystems" stays).
- (c) Section 2.5: removed the orphan lone-parenthetical paragraph "(Per-dataset details are given in Section 5.1.)". Merged the struck old dataset paragraph and the green Benchmark-scale paragraph into one `<p>`; the pointer now reads, un-parenthesized, "Per-dataset details are given in §5.1. The water-treatment and ICS-security datasets ..." (no empty paragraph left after accept-all).
- (d) Section 2.1: "reinforcing our **regime-latent factorization** over a single plant-wide model" -> "reinforcing our **factorization of the density over correlation-community subsystems** rather than a single plant-wide model" (plain language, matches the paper's standard term).
- (e) Appendix A: added a one-sentence green lead-in after Table A2 introducing Figure A1: "Figure A1 closes the appendix with a synthetic illustration of the reachability-versus-probability mechanism discussed in §7: as a fault moves from off-manifold to between-regime, density overtakes reconstruction, and the crossover comes earlier for better-separated regimes."
- (f) Section 4.3 (iv): trimmed the bare A8 disclaimer - green "Equation (7) is the detector's complete window score; A8 (between-regime overlap) is specified in Table 1 (§3) and addressed in §7 rather than by a dedicated head." -> "Equation (7) is the detector's complete window score." (A8 is already deferred earlier at the residual-head paragraph.)

## Letter edits (`review_round_1/`)

### MUST-FIX - SKAB "variance-floor artifact" phantom (editor Req 1 and R1 C1)
The paper strikes the SKAB variance-floor explanation; both letters cited it. Reworded to the paper's
current basis (Appendix C: SKAB has no close-distinct regime pairs, 0 of 12, single-loop rig).
- `letter_editor.html`: "...that screen rated the regimes as maximally crisp, `[was]` and it also proved to be a variance-floor artifact of the estimator on SKAB's 400-window sample (Appendix C, Table C1). `[now]` and SKAB is the one screened dataset with no close-distinct regime pair at all (a single-loop 400-window rig; Appendix C, Table C1)."
- `letter_reviewer1.html`: "...it rated the regimes as maximally crisp, `[was]` and on SKAB the signal proved to be a variance-floor artifact of the estimator on a 400-window sample (Appendix C, Table C1). `[now]` and SKAB is the one screened dataset with no close-distinct regime pair at all (a single-loop 400-window rig; Appendix C, Table C1)."
- Appendix C wording verified: "SKAB [59], a 400-window rotor rig, has no close-distinct regime pairs (0 of 12; it is a single-loop rig)"; Table C1 SKAB row shows close-distinct mass 0.00.

### Editor letter - new-additions bullet in "Summary of the principal changes"
Added a "New validated additions" bullet announcing Table 7 (subsystem-localization triage: three-community
shortlist contains the attacked subsystem in 68 to 79 percent of HAI and SWaT attacks vs 49 to 56 percent
random), the sharpened Section 7 density-vs-deep-detector comparison (single latent-density head beats USAD
and TranAD on the difficult subset of all three plants), and the new synthetic Figure A1.

### R1 letter - one-line pointer in Comment 5 (representation)
Added a closing sentence pointing at the head-level 3/3 result: "The revised Section 7 makes this most
directly: the single latent-density head ... beats the deep reconstruction detectors USAD and TranAD on
the difficult subset of all three plants (Table 6), which isolates the density mechanism from both the
representation and the factorization."

Tone plain/thankful; no em-dash, double-hyphen, or "honest*"; Apartsin spelled correctly.

## Deferred (flagged for the authors, NOT fixed here, per instruction)
These are the deeper reviewer carry-overs the user explicitly deferred:
1. Headline (regime-community) model in the six-statistic robustness split - Section 6 reports the split
   for HAI and the global-density ablation only; the headline detector on all three datasets with the
   episode-block bootstrap is not run (review4 A.18 / review-3 item 6). Answer would also feed §5.3.
2. Hyperparameter table (encoder/decoder widths, optimizer, LR, epochs, variance floor, DAGMM weight,
   batch size, per-community K / latent-dim rule) in Appendix B (A.20 / review-3 item 12).
3. Gate threshold stated a priori - §4.3 (iii) cut-off 1.5 between observed 1.22 and 3.41 is not stated
   as fixed before evaluation (A.21 / review-3 item 13).
4. "Per-channel marginals already suffice" wording on WADI in §6 / Table 5 caption (A.19 / review-3 item 8).

Also left untouched by instruction: Abstract; Data Availability statement; Figure 3 and Figure A1 SVG curve
drawings; Figure A1 axis label (author regenerates); all reference/citation numbers.

## Self-check
- Span balance: `<span>` open = close = 307 in the paper (0 diff). Merged §2.5 paragraph verified
  well-formed (`<p>`[rev-del][rev2-highlight]`</p>`); no empty paragraph and no orphan parenthetical
  after accept-all.
- Style: 0 em-dash characters, 0 `&mdash;`, 0 prose "`--`" (the 64 `--` hits are CSS custom properties
  and SVG transform math, unchanged), 0 "honest*" - in paper and all three letters. "Apartsin" correct;
  no stray "Partsin".
- MathML preserved: 53 `<math>` tags intact; no equation text edited.
- Figure 3 / Figure A1 curve drawings untouched (only the Figure 3 source comment removed and the
  Figure A1 caption softened; no path, polyline, or coordinate changed). Figure 1 change limited to the
  single "diagnosis" text run.
- Citation numbers untouched (no reference renumbering; anchors in the localization paragraph preserved).
- "should be gone" strings absent: "matches GDN", "right default for the deployed", "40 percent of HAI",
  "deployed reconstruction detectors", "density-valley ratio)" in the Table C1 sentence,
  "regime-latent factorization", "per-community surprises also localize", "bars regenerated".
- "should be present" strings confirmed: "within noise of GDN", "39 percent of HAI",
  "deep reconstruction detectors of Table", "0.42 versus 0.37", Figure 1 "triage shortlist" text run,
  "the benchmark results are in Section".

