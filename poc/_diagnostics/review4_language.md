# Review 4: line-by-line language, clarity, tone, flow (IoT2.html, read after accept-all)

Scope: full sequential read of `poc/paper/IoT2.html` (3004 lines). Line numbers refer to the current file. Text is quoted as it reads once green insertions are accepted and strikes are removed. Green/strike marking itself is not flagged.

Mechanical style scan: PASS. No em-dash or double hyphen in prose (the only `--` hits are CSS variables and one SVG source comment); no "honestly/frankly/candidly"; no "seems to / appears to / arguably / admittedly / we acknowledge"; author name spelled "Apartsin" throughout.

Ranked by reader impact. Severity: HIGH = misreads or factual mismatch; MED = stumbles, dense, or leftover; LOW = polish.

---

## HIGH

### H1 | line 1041, Section 7 "Why reconstruction fails and density wins" | garbled clause: says density is the right default *for the reconstruction detectors*
**Text:** "Scoring by density rather than reconstruction is the right default for the deployed reconstruction detectors: the single latent-density head, with no residual term and no factorization, exceeds USAD and TranAD on the difficult subset of all three plants ..."
**Problem:** "the right default for the deployed reconstruction detectors" reads as advice to those detectors; the intended meaning is that density beats them. "Deployed" also suggests production use (same word in the Table 6 caption, line 1076).
**Rewrite:** "Scoring by density rather than reconstruction is the better default, judged against the published reconstruction detectors themselves: the single latent-density head, with no residual term and no factorization, exceeds USAD and TranAD on the difficult subset of all three plants ..." and in the Table 6 caption: "the deep reconstruction detectors of Table 3 score 0.44 to 0.66 on the same difficult subsets."

### H2 | lines 1044-1049, same paragraph | one 95-word sentence carrying seven numbers
**Text:** "The margin is largest where the faults are reconstructable but improbable: on HAI the reconstruction detectors place the difficult attacks at the median of normal operation while the density head places them at the 93rd percentile, and 27 percent of HAI's difficult attacks reconstruct at least as well as a typical normal window yet fall above the 90th normal percentile in density; on WADI that share is zero and 70 percent of the difficult attacks are weak under both scores, which is why the community factorization, not the head, carries the WADI gain."
**Rewrite (three sentences):** "The margin is largest where the faults are reconstructable but improbable. On HAI the reconstruction detectors place the difficult attacks at the median of normal operation, whereas the density head places them at the 93rd percentile; 27 percent of HAI's difficult attacks reconstruct at least as well as a typical normal window yet fall above the 90th normal percentile in density. On WADI that share is zero, and 70 percent of the difficult attacks are weak under both scores, which is why the community factorization rather than the head carries the WADI gain."

### H3 | lines 1162-1169, Section 7 "Subsystem localization" | leftover fragment + dangling participle + unglossed "coverage-aware" + top-3 numbers stated twice
**Text:** "... against 49 to 56 percent for a random shortlist of the same size. The per-community surprises also localize. Ranking communities by their train-normal upper-tail p-value (the quantity the Higher-Criticism combiner already uses; communities whose calibration surprise is degenerate carry no p-value and are not ranked) and naming the top community per attack episode, the named community contains an attacked channel in 46 percent of SWaT attacks (11 of 24, coverage-aware random 22 percent, P = 0.004) and 40 percent of HAI attacks (15 of 38, random 25 percent, P = 0.02), and a three-community shortlist contains it in 79 and 68 percent (random 49 and 56 percent)."
**Problems:** (a) "The per-community surprises also localize." was the old opener; after the new headline sentence it is a non-sequitur. (b) "Ranking ..., the named community contains" dangles. (c) "coverage-aware" is first used here and only glossed in the Table 7 caption. (d) The top-3 figures repeat the opening sentence. (e) HAI top-1 is "40 percent" in text but 0.39 in Table 7 (15/38 = 0.395).
**Rewrite:** "A three-community shortlist ranked by train-normal p-value contains the attacked subsystem in 68 to 79 percent of HAI and SWaT attacks, against 49 to 56 percent for a random shortlist of the same size (Table 7). The ranking is the one the Higher Criticism combiner already computes: communities are ordered by their train-normal upper-tail p-value (communities whose calibration surprise is degenerate carry no p-value and are not ranked), and the top community is named once per attack episode. That single named community contains an attacked channel in 46 percent of SWaT attacks (11 of 24) and 39 percent of HAI attacks (15 of 38), against a coverage-aware random baseline (the chance that a randomly named community of the same size contains an attacked channel) of 22 and 25 percent (P = 0.004 and P = 0.02)."

### H4 | line 1103, Section 7 A8 paragraph vs Table C1 | text claims a column that the table does not have
**Text:** "under four observation-space measures (two of which are tabulated in Table C1: the overlap coefficient and the density-valley ratio) no benchmark approaches ..."
**Problem:** Table C1 has columns close-distinct mass, overlap coefficient, masking rate / base, isolated-missed. There is no density-valley-ratio column. Appendix C prose (line 2905) also lists "the nearest-pair overlap coefficient and density-valley ratio" as measured.
**Rewrite (choose one):** "(the overlap coefficient is tabulated in Table C1 alongside the masking and over-coverage measures)"; or add the density-valley-ratio column to Table C1.

### H5 | line 1246-1250, Appendix A opening | one 150-word sentence chained by semicolons
**Text:** "Factorizing the global density over regime communities lifts all three datasets, most on WADI, whose difficult faults are confined to small correlated subsystems that a global density dilutes; on WADI the community aggregation carries the whole effect, lifting the single-latent global density from 0.634 to 0.795 through cohesion-weighted Higher Criticism over the per-community experts, and fusing in the whole-plant expert (the headline fusion) is mildly counterproductive there, so the fused headline settles at 0.771; on HAI and SWaT the whole-plant-expert fusion instead lifts the score, most on HAI, where the hard valve and flow deviations are correlation-conditional; and among the sparsity-adaptive combiners, cohesion-weighted Higher Criticism is the one that leads on all three datasets (all aggregation rules here fused with the whole-plant expert), because it weights a violation by the coupling strength of the subsystem it occurs in (A7) while adapting to the fault's extent."
**Rewrite (four sentences):** "Factorizing the global density over regime communities lifts all three datasets, most on WADI, whose difficult faults are confined to small correlated subsystems that a global density dilutes. On WADI the community aggregation carries the whole effect: cohesion-weighted Higher Criticism over the per-community experts lifts the single-latent global density from 0.634 to 0.795, and fusing in the whole-plant expert costs a little, so the fused headline settles at 0.771. On HAI and SWaT the whole-plant-expert fusion instead lifts the score, most on HAI, where the hard valve and flow deviations are correlation-conditional. Among the sparsity-adaptive combiners (all fused with the whole-plant expert in the table), cohesion-weighted Higher Criticism leads on all three datasets because it weights a violation by the coupling strength of the subsystem it occurs in (A7) while adapting to the fault's extent."

---

## MEDIUM

### M1 | line 657, Section 4.4 | doubled "rank ... ranked" from the insertion; "Section 7" vs house "§" style; "Higher-Criticism" hyphenation
**Text:** "the per-community surprises rank the plant's subsystems by how implicated they are, ranked by their train-normal upper-tail p-value (the quantity the Higher-Criticism combiner already uses), so a plant-wide alarm becomes a subsystem triage shortlist; Section 7 (Table 7) validates that ..."
**Rewrite:** "the per-community surprises, ordered by their train-normal upper-tail p-value (the quantity the Higher Criticism combiner already uses), rank the plant's subsystems by how implicated they are, so a plant-wide alarm becomes a subsystem triage shortlist; §7 (Table 7) shows that a three-community shortlist contains an attacked subsystem in 68 to 79 percent of HAI and SWaT attacks." Apply "Higher Criticism" (no hyphen) at line 1165 too.

### M2 | lines 1170-1177, "Subsystem localization" | WADI sentence and the two limits read awkwardly; "the pointer", "reported at its size", "comfortable headline"
**Text:** "WADI, with 9 attacks whose targets are published and 7 of them inside any community, is directionally the same (3 of 9 at top-1, random 13 percent) but too small to test. ... (i) A community whose surprise drifts above its calibration range is named disproportionately often even on normal windows, so the pointer is a triage shortlist rather than a diagnosis. (ii) Larger communities are easier to hit, so the shortlist is reported at its size; against a size-weighted random baseline the SWaT margin holds while HAI's top-1 margin is small, which is why the top-3 shortlist is the comfortable headline."
**Rewrite:** "WADI has 9 attacks with published targets, 7 of which hit a channel that belongs to some community; it points the same way (3 of 9 at top-1, random 13 percent) but is too small to test. The named community averages about six channels of 51 (SWaT) and 59 (HAI). Two limits apply. (i) A community whose surprise drifts above its calibration range is named disproportionately often even on normal windows, so the named community is a triage lead rather than a diagnosis. (ii) Larger communities are easier to hit, so Table 7 reports the named-community size; against a size-weighted random baseline the SWaT margin holds while HAI's top-1 margin is small, which is why the three-community shortlist is the headline figure."

### M3 | Table 7 caption, line 1181 | fragment "rank rule"; "coverable" unglossed
**Text:** "Subsystem localization at the attack-episode level, rank rule (communities ordered by their train-normal upper-tail p-value; degenerate-calibration communities are unranked)."
**Rewrite:** "Subsystem localization at the attack-episode level. Communities are ranked by their train-normal upper-tail p-value (communities with degenerate calibration are unranked). ... WADI's "7 coverable" counts the attacks whose target channel belongs to some community."

### M4 | flow: Table 7 splits the deployment discussion | lines 1152-1198
**Problem:** "Implications for IIoT deployment" (1152) is followed by the localization paragraph and Table 7, then the edge-inference paragraph (1189) resumes the deployment topic with no lead-in, so it appears to continue the localization result.
**Fix (minimal):** give line 1189 a bold lead-in: "**Edge inference cost.** Pushing inference to the network edge ..." Alternative: move "Subsystem localization" + Table 7 to precede "Implications for IIoT deployment" and change "(validated below, Table 7)" to "(Table 7)".

### M5 | line 1158, "Implications for IIoT deployment" | redundant "plant-wide ... plant-wide"
**Text:** "the per-community score turns a plant-wide alarm into a subsystem triage shortlist rather than only a plant-wide flag (validated below, Table 7)."
**Rewrite:** "the per-community score turns a plant-wide alarm into a subsystem triage shortlist (validated below, Table 7)."

### M6 | Figure A1 caption, line 2868, and axis label | "in-envelope" coinage; defensive closing sentence; figure never introduced inside Appendix A
**Text:** "Density overtakes reconstruction as faults become more in-envelope, and the crossover is earlier for better-separated regimes. This is a synthetic mechanism illustration, not a claim about the benchmarks." Axis label reads "fault in-envelope-ness (0 = off-manifold, 1 = between-regime)".
**Rewrite (caption):** "Synthetic illustration of the reachability-versus-probability mechanism. On toy data with five Gaussian regimes, the difficult-subset gap between a latent-density score and a reconstruction score (AUROC of density minus AUROC of reconstruction) is plotted as a fault moves from off-manifold (outside every channel's normal range, so reconstruction catches it) to between-regime (inside every channel's range but between two regimes, so it reconstructs yet is improbable), at three regime separations. Density overtakes reconstruction as the fault moves inside the per-channel envelope, and the crossover comes earlier for better-separated regimes. The panel illustrates the mechanism on synthetic data; the benchmark evidence is Tables 5 and 6."
**Axis label (regenerate figure):** "fault position (0 = off-manifold, 1 = between-regime)".
**Introduce it:** add one sentence at the end of the Table A2 paragraph (line 1284): "Figure A1 closes the appendix with a synthetic illustration of the reachability-versus-probability crossover discussed in §7." (The §7 forward reference at line 1049 is fine; the figure itself currently appears with no sentence pointing to it.)

### M7 | line 1052-1056, end of the density paragraph | trailing sentences repeat §4.2 and the paragraph's own opening; ambiguous "which"
**Text:** "Scoring by density in the clustered latent replaces reachability with probability, which is exactly what reconstruction cannot see. This is why the base score omits reconstruction and scores in the jointly learned latent, with a high-resolution density head for non-Gaussian fringes (A3) and a rare-regime-safe nearest-component likelihood (A5); reconstruction is added back only through the optional, auto-gated residual head where it demonstrably helps (§4.3 iii)."
**Rewrite:** "Scoring by density in the clustered latent therefore replaces reachability with probability, the quantity reconstruction cannot see; the base score works in the latent (§4.3 i-ii) and reconstruction re-enters only through the auto-gated residual head where it generalizes to held-out normal (§4.3 iii)."

### M8 | line 1116-1119, "Benchmark coverage as a diagnosable property" | "lifts a ... view" is garbled
**Text:** "Excluding that regime, a label-free operation that removes 18.4% of test-normal windows and no anomaly, lifts a secondary coverage-diagnostic view of the headline model (in that pass all-subset AUROC 0.948 to 0.965, difficult-subset 0.845 to 0.890, and the true-positive rate at a 5% false-alarm budget 0.789 to 0.916), with LatAD leading throughout."
**Rewrite:** "Excluding that regime, a label-free operation that removes 18.4% of test-normal windows and no anomaly, gives a secondary coverage-diagnostic view in which the headline model's all-subset AUROC rises from 0.948 to 0.965, its difficult-subset AUROC from 0.845 to 0.890, and its true-positive rate at a 5% false-alarm budget from 0.789 to 0.916, with LatAD leading throughout."

### M9 | line 1137, "Benchmark scale and the trajectory assumptions" | broken parallelism
**Text:** "with one nominal operating mode (though the mixture discovers many sub-regimes within it) and days of data there is little between-regime masking to observe and short trajectories with limited path history."
**Rewrite:** "with one nominal operating mode (though the mixture discovers many sub-regimes within it) and days of data, there is little between-regime masking to observe, and the trajectories are too short to carry much path history."

### M10 | line 596, Section 4.3 (iv) | A8 clause is revision-journey residue in the scoring section
**Text:** "Equation (7) is the detector's complete window score; A8 (between-regime overlap) is specified in Table 1 (§3) and addressed in §7 rather than by a dedicated head."
**Problem:** A reader of the scoring stack has no reason to expect A8 here; line 448 already defers it.
**Rewrite:** "Equation (7) is the detector's complete window score."

### M11 | line 258, Section 2.5 | orphan parenthetical paragraph after accept-all
**Text:** "(Per-dataset details are given in Section 5.1.)" stands alone as the whole first paragraph of §2.5.
**Rewrite:** delete the parentheses and fold into the next paragraph's opening: "Per-dataset details are given in §5.1. The water-treatment and ICS-security datasets used here and elsewhere ..."

### M12 | line 861 and 1067 | the 0.795 community figure exceeds the 0.771 headline without saying why until Appendix A
**Text (861):** "The WADI lead is carried by the community density aggregation (cohesion-weighted Higher Criticism over the community experts, 0.795)"
**Rewrite:** "(cohesion-weighted Higher Criticism over the community experts alone, 0.795 before fusion with the whole-plant expert; Table A1)". Same gloss at line 1067.

### M13 | line 622, Section 4.4 | "its" has no clear antecedent after a 60-word parenthetical
**Text:** "... plus one global expert). We read its calibrated surprise s_G (the train-normal upper-tail negative log-probability), and we include the global detector ..."
**Rewrite:** "We read each community's calibrated surprise s_G (the train-normal upper-tail negative log-probability, that is, how far into the normal tail its score falls), and we include ..."

### M14 | line 643, Section 4.4 | "train-normal upper-tail p-value" used before it is glossed
**Text:** "Writing the per-community upper-tail p-values for the S community experts in ascending order ..."
**Rewrite:** "Writing the per-community upper-tail p-values (for each expert, the fraction of train-normal windows whose surprise is at least as large) for the S community experts in ascending order ..."

### M15 | line 175, Introduction | inverted, and "the critique" has no referent yet
**Text:** "The evaluation obstacle it addresses by adopting the critique's protocol, raw metrics with difficulty stratification, which we use rather than claim as a contribution."
**Rewrite:** "The evaluation obstacle it addresses by adopting the protocol the evaluation critique of §2.2 prescribes, raw metrics with difficulty stratification; we use that protocol rather than claim it as a contribution."

### M16 | line 206, Section 2.1 | "regime-latent factorization" is a coinage used nowhere else
**Text:** "reinforcing our regime-latent factorization over a single plant-wide model."
**Rewrite:** "reinforcing our regime-community factorization over a single plant-wide model."

### M17 | line 341-344, Section 3 | "regime" used in two senses in one paragraph; "pays off" twice
**Text:** "This ordering tracks where the design pays off: the regime-community factorization gains most on WADI, and by a similar margin on HAI and SWaT, since a single global density blurs local structure (§6). This is the regime in which a jointly learned latent, scored by a mixture density and factorized over subsystems, pays off and a global detector or a raw reconstruction residual does not."
**Rewrite:** "This ordering tracks where the design pays off: the regime-community factorization gains most on WADI, and by a similar margin on HAI and SWaT, because a single global density blurs local structure (§6). In this setting a jointly learned latent, scored by a mixture density and factorized over subsystems, succeeds where a global detector or a raw reconstruction residual does not."

### M18 | line 1129-1133, Trajectory assumptions | 60-word A10 sentence
**Text:** "A10 (path dependence) is measured absent rather than merely unobserved: three conditional scorers (a linear autoregressor on the two preceding windows, a nearest-neighbor successor model, and a regime-transition matrix), validated on a synthetic positive control and guarded by an order-shuffle control, find no window that is marginally normal yet improbable given its history above the shuffle floor, on all three benchmarks, once windows whose history already contains an attack are excluded."
**Rewrite:** "A10 (path dependence) is measured absent rather than merely unobserved. Three conditional scorers (a linear autoregressor on the two preceding windows, a nearest-neighbor successor model, and a regime-transition matrix), each validated on a synthetic positive control and guarded by an order-shuffle control, find no window on any of the three benchmarks that is marginally normal yet improbable given its history above the shuffle floor, once windows whose history already contains an attack are excluded."

---

## LOW

### L1 | line 1193-1198, edge paragraph | CalexNet sentence and its restatement say the same thing twice
**Text:** "... for example CalexNet [57], a cascade-aligned early-exit method reported to reduce inference cost by roughly 30 to 70 percent, which applies directly to the per-community expert stack on constrained edge hardware. ... The CalexNet early-exit path above is a further-reduction route on top of these figures for the per-community stack on constrained edge hardware."
**Rewrite (last sentence):** "Early exit is a further reduction on top of these figures."

### L2 | line 838, "Overall detection" | "clearing every baseline ... including the strong classical baselines" is redundant
**Rewrite:** "ahead of every baseline and all three deep detectors on both, including the strong linear baseline on the more separable SWaT."

### L3 | line 272, Section 3 | "each ... and A1-A7 each" clunky
**Text:** "each is grounded in a physical property of CPS operation and A1-A7 each motivate a specific design choice in the detector of §4."
**Rewrite:** "each is grounded in a physical property of CPS operation, and each of A1-A7 motivates a specific design choice in the detector of §4."

### L4 | line 243, Section 2.4 | "reconstructable-but-improbable" first appears as a compound without a gloss
**Rewrite:** "the same blind spot on the reconstructable-but-improbable joint faults (faults a flexible decoder reproduces yet normal operation rarely occupies) that define our difficult subset."

### L5 | line 245, Section 2.4 | two back-to-back parentheticals
**Text:** "(five seeds on HAI, a single run on WADI and canonical SWaT; §5.5) (the re-scored comparison is in §6)."
**Rewrite:** "(five seeds on HAI, a single run on WADI and canonical SWaT, §5.5; the re-scored comparison is in §6)."

### L6 | line 177 | "Slowly-progressing" takes no hyphen after an -ly adverb
**Rewrite:** "Slowly progressing and between-regime faults ..."

### L7 | lines 526 and 1056 | "demonstrably" protests; the gate is the demonstration
**Rewrite:** drop "demonstrably" in both places ("which fires only where it generalizes to held-out normal").

### L8 | Table 2 caption (509-512) vs body (523-525) | the HAI 0.689 / 0.760 sentence appears nearly verbatim twice
**Fix:** keep the body sentence; shorten the caption to "On HAI the residual (0.689) drags the summed score down to the residual's own level; on WADI the residual is informative (0.742) but fails the held-out-normal gate (§4.3 iii, Table 6) and is not used."

### L9 | line 946, Figure 3 SVG source | scaffolding comment "rev2: bars regenerated ... GDN bar added ... WADI whiskers added"
**Fix:** delete the HTML comment before submission (invisible when rendered, but it is revision narration in the source that ships).

### L10 | line 258 and 657 | "Section 5.1", "Section 7" vs the paper's "§" convention
**Fix:** use "§5.1", "§7".

### L11 | Table A2 caption, line 1290 | "within-harness" is internal jargon
**Rewrite:** "This is a self-contained comparison inside one evaluation pipeline, isolating the representation axis; ..."

---

## What reads well (no action)
- The density reframe lands: §4.2 states the mechanism, §7 quantifies it (median vs 93rd percentile), Figure A1 is referenced before it appears. Only H1/H2/M7 stand between the paragraph and a clean read.
- The localization result leads its paragraph with the headline number and states both caveats afterward; the caveats do not bury it. H3/M2/M3 are about fluency, not structure.
- Abstract, contributions paragraph, Table 3 discussion, significance-testing paragraph, Limitations, Conclusion, Appendices B and C: clear, forward-only, no apologetic tone.
- Glosses now present on first use: whitened (Mahalanobis) residual (Table 1 A7), Higher Criticism (§1 and §4.4), Ledoit-Wolf, F1, AUROC, GMM, HIL, NLL.
