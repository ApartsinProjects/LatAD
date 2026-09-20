# Review 2: readability and tone of `poc/paper/IoT2.html`

Scope: the whole manuscript read as a careful reader. Report only; no edits made. Line numbers refer to `IoT2.html` as of this review. Green/strike change-marking is ignored.

Mechanical checks passed: no em-dashes or double hyphens in prose (the `--` hits are all CSS variables), no "honestly/frankly/candidly", no "acknowledge/admittedly/arguably", no "seems/appears" on a measured result. The problems are structural: repetition, revision-journey narration, one enormous paragraph, and a scope note that arrives before its setup.

Severity: **H** = hurts a first read or leaks project history; **M** = slows the reader; **L** = polish.

---

## Ranked findings

### 1. [H | §7 "Assumption A8", line 1080] One 450-word paragraph narrating how the A8 conclusion was reached

**Problem.** The paragraph is the longest in the paper, mixes four measurements, two synthetic controls, five datasets and a retracted earlier metric, and narrates the revision journey ("The evidence earlier read as A8 came from a cluster-responsibility screen that is blind by construction", "the SKAB signal that appeared to witness it"). A reader who wants the conclusion has to wade through the audit trail. "Shipped 80-component density head" is software-product jargon (also at lines 1296, 1308).

**Suggested rewrite.** Keep the conclusion in §7, move every measurement to Appendix C (which already contains them). Replacement for the whole paragraph:

> **Assumption A8 (between-regime overlap).** A8 posits regimes lying so close in observation space that a window can be improbable while every channel stays in range. Its precondition, adjacent-but-distinct regimes, holds on every multi-loop plant we screened (WADI, HAI, SWaT, Cranfield, MetroPT, SMD: 94 to 100 percent of train-normal mass has a valley-separated neighbour within twice its own envelope). The overlap itself is absent: under four observation-space measures (overlap coefficient, density-valley ratio, cross-regime nearest-neighbour share, discriminant error) no benchmark approaches the 30-percent-overlap positive control, and the labelled anomaly-masking rate never exceeds its regime-switch base rate (Appendix C, Table C1). The one A8 variant the reported model already defends is over-coverage: a wide two-component head misses about 80 percent of WADI's isolated anomalies, the 80-component density head of §4.3 (i) about 13 percent. Masking, the remaining variant, is identifiable only through the path a window arrived by, so it belongs with A10 and is left to operational-scale data (Appendix C).

Then in Appendix C, replace "An earlier version of this screen measured overlap by cluster responsibility ... that metric is invalid for A8" (line 1280) with a forward statement:

> Cluster responsibility (how confidently the fitted mixture assigns a window to one regime) is not a valid overlap measure: a softmax posterior is crisp both when a window sits at a component and when it is far from all of them, so it cannot see a between-mode pocket. On a synthetic regime pair whose Bayes error runs from 30 to 2 percent the responsibility entropy stays flat, and it assigns a genuine between-mode window (density percentile 0.000) responsibility 1.000. We therefore measure A8 in observation space with four label-free quantities ...

And replace the SKAB paragraph heading and its journey language (line 1320: "does not survive scrutiny", "no longer framed as a near-witness", "appeared to exhibit") with:

> **SKAB.** SKAB [54], a 400-window rotor rig, has no close-distinct regime pairs (0 of 12; it is a single-loop rig), so it does not exhibit the A8 precondition. Its responsibility entropy of 0.29 is a variance-floor artifact: the VaDE component log-variance floor of log(0.05) manufactures apparent overlap on so few windows, and refitting with empirically estimated component variances drops the entropy to 0.045 to 0.048, the same as WADI (0.053 to 0.057), with the basin-head lift on the difficult subset going to zero on every seed.

### 2. [H | whole paper] "The model has no A8 head" is stated eight times

**Locations.** Abstract (line 82), §1 scope note (177), Table 1 caption and A8 row (283, 318-321), §3 (272, 343), §4 opener (445-448), §4.3 (iv) (596), auto-gating note (599), Table 6 caption (1053), §7 (1080, 1100), Appendix C (1328).

**Problem.** Each restatement was added to answer a reviewer, and together they make A8, an assumption the detector does not use, the most-discussed item in the paper. A first-time reader concludes A8 is the paper's main subject.

**Suggested fix.** State it in exactly three places: Table 1 (the A8 row), one sentence in §4's opener, and the §7 paragraph above (with Appendix C). Delete from: the abstract clause "of which seven are realized by the detector and validated here and three are specified for future work"; the §1 scope-note parenthetical (see item 4); §3 line 272 "so a property that a corpus does not exhibit (here A8) simply leaves the corresponding component inactive rather than degrading the detector"; §3 line 343 parenthetical "(This low silhouette reflects a partition slicing a continuous regime ridge, not heavy mode overlap ...)"; §4.3 (iv) lines 596 second half ("The stack has no head for A8 ... Equation (7) is therefore the detector's complete window score."); the auto-gating note sentence "The reported model has no overlap head ... its precondition notwithstanding (§7)"; Table 6 caption "The model has no A8 overlap head, so none appears here."

§4 opener replacement (lines 445-452):

> The reported detector realizes A1 to A7. The mixture representation (A1, A6), the high-resolution density head (A2, A3), the low-dimensional latent (A4), the nearest-component score (A5) and per-feature standardization (A7) are always active; the whitened-residual head (A7) is auto-gated on train-normal signals and activates on HAI and SWaT (Table 6). A8 is specified in Table 1 but has no head: the overlap it posits is absent from the datasets studied (§7, Appendix C). A9 and A10 shape the trajectory rather than the window and are left to future work (§7).

### 3. [H | §6 line 861-862] Revision-journey sentence about the WADI count

**Quote.** "The 43-to-30 change in the WADI difficult count removes twelve constant-channel-flip windows that a difficulty-stratification artifact (an inconsistent clip across datasets, now fixed) had left in the difficult subset."

**Problem.** Refers to a number (43) that appears nowhere else in the paper and to a bug ("now fixed"). Pure project history.

**Fix.** Delete the sentence. The clipping rule is already stated in §5.1.

### 4. [H | §1 "A note on scope", line 177] A8/A9/A10 status parenthetical arrives before the assumptions exist

**Quote.** "... motivates the assumptions we examine in §7 (A8, between-regime overlap, specified but not realized by the reported model, its masking form coupled to A10; A9, multiscale structure, present in the data yet exploited here only as a complementary lever; and A10, path dependence, measured absent on these snapshot-detectable benchmarks)."

**Problem.** The reader has not yet met A1 to A10 (§3), "masking form", "snapshot-detectable" or "complementary lever". This is the paper's clearest claim-before-setup.

**Rewrite.** End the paragraph at: "... which shapes what any method can be exercised on and motivates the trajectory assumptions examined in §7." Also split the paragraph: the sentence "These public testbeds are also small: SWaT (51 channels, 11 days) ..." belongs in §2.5, which already makes the same point.

### 5. [H | Abstract, line 82] Final sentence packs six numbers and three qualifiers

**Quote.** "On three real IIoT testbeds LatAD attains the best overall AUROC on HAI (0.948) and SWaT (0.938) and leads the difficult subset of all three: significantly over the strongest baseline on HAI (0.845) and canonical SWaT (0.837), and ahead of every learned detector on WADI (0.771), where it ties the linear baseline on the full set; on HAI the deep detectors USAD and TranAD fall to 0.44–0.48. Code and checkpoints are released (Zenodo)."

**Rewrite.**

> On three real IIoT testbeds LatAD attains the best overall AUROC on HAI (0.948) and SWaT (0.938) and leads the difficult subset, the anomalies a per-channel rule cannot separate, on all three (HAI 0.845, SWaT 0.837, WADI 0.771), with statistically significant margins over the strongest baseline on HAI and SWaT. On HAI's joint-structure faults the deep detectors USAD and TranAD fall to 0.44 to 0.48, near chance. Code and checkpoints are released on Zenodo.

Also drop the clause "of which seven are realized by the detector and validated here and three are specified for future work" (item 2), and gloss "difficult subset" as above, since the abstract uses it before any definition.

### 6. [H | §1 lines 88-90] CPS and IIoT are defined twice in consecutive paragraphs

**Quote (para 1).** "A cyber-physical system (CPS) couples networked sensors, actuators, and controllers with a physical process, forming the sensing-and-actuation layer of the Industrial Internet of Things (IIoT) ..."
**Quote (para 2).** "A modern CPS (a water-treatment plant, an industrial control loop, a rotating machine) couples hundreds of networked sensors, controllers, and actuators into a unit that continuously senses its own state and acts on it: the sensing-and-actuation fabric of the Industrial Internet of Things (IIoT)."

**Fix.** Keep para 1's opening sentence as the definition. Open para 2 with "Such a system (a water-treatment plant, an industrial control loop, a rotating machine) couples hundreds of channels, and this complexity puts its faults beyond enumeration: ..." and drop the second IIoT expansion. Para 1 also lost its closing thesis sentence to a strike; it now ends on the problem without the paper's answer. Add one line: "This paper scores probability instead of reconstruction, and does so per subsystem."

### 7. [H | §6 lines 845-864] The "Difficult, joint-structure faults" paragraph runs 330 words over three datasets

**Problem.** HAI, SWaT and WADI results, three bootstrap results, the HC_coh ablation and the deleted 43-to-30 sentence are one paragraph. Readers lose which dataset a CI belongs to.

**Fix.** Split into three short paragraphs headed by dataset, and move the mechanism sentence ("The WADI lead is carried by the community density aggregation ... supplies the WADI signal") to §7 "Why local density beats global density", where it is already restated (line 1041). Suggested WADI paragraph:

> On WADI LatAD reaches 0.771 (30 difficult windows over 8 attack episodes), ahead of every learned detector (linear cross-channel 0.750, Isolation Forest 0.634, AutoEncoder 0.628, TranAD 0.613, USAD 0.579). The lead over the linear baseline is numerical (+0.020, 95% CI [−0.161, 0.205], P = 0.46); with 8 episodes the test has little power. On the full set the linear baseline is marginally ahead (0.834 vs 0.827).

Note a logic slip in the same passage (line 857): "so LatAD is significantly ahead of every learned detector on WADI and ties only the linear baseline." Only the linear-baseline comparison was bootstrapped; "significantly" is not supported for the learned detectors. Use "ahead of every learned detector".

### 8. [M | 5 places] `HC_coh` is an internal variable name

**Locations.** Lines 858, 1014, 1044, 1198, 1217.

**Fix.** Replace with the paper's own term: "cohesion-weighted Higher Criticism over density-only experts" on first use, then "the density-only community aggregation". In Table A1 rename the row "LatAD (regime-community), density-only experts, cohesion-weighted HC".

### 9. [M | §7 line 1128-1130] "a design frozen on their earlier versions"

**Quote.** "The community construction and combiner settings were fixed before the artifact-free WADI and canonical SWaT evaluations reported here, so those two datasets are effectively out-of-sample confirmations of a design frozen on their earlier versions."

**Problem.** "Earlier versions" tells the reader there were other versions of the datasets in play. Revision narration.

**Rewrite.** "The community construction and combiner settings were fixed before the WADI and SWaT evaluations reported here, so those two results are out-of-sample confirmations of the design."

Same paragraph, line 1125-1127 is apologetic: "should be validated on additional, untouched CPS systems before its per-dataset gains are treated as established". Rewrite: "Validation on additional CPS systems is the next test of the regime-community realization; its build and calibration use train-normal only, so that test requires no labels."

### 10. [M | whole paper] "artifact-free WADI" and "canonical SWaT" as standing qualifiers

**Problem.** "artifact-free" (8 uses) and "canonical" (13 uses) read as "as opposed to the version we used before". Once §5.1 defines the datasets, the qualifiers carry no information for the reader.

**Fix.** Define once in §5.1 ("WADI, 122 channels after dropping the rescaled analyzer channel"; "SWaT, the December 2015 iTrust release") and use plain "WADI" and "SWaT" everywhere else.

### 11. [M | whole paper] "headline" carries two meanings

**Problem.** "Headline" means the benchmark aggregate number (lines 172, 197, 224) and also the main model ("headline realization", "headline detector", "headline fusion": 15 uses). "Headline number" is pejorative in §1 and §2, then "headline detector" is the paper's own model.

**Fix.** Keep "headline number" for the metric sense. Rename the model sense "full detector" or "LatAD (regime-community)" throughout.

### 12. [M | §4.4 line 624, §4.4 line 659, Fig. 2] "null expert"

**Quote.** "we include the global detector as an unfactorized *null* expert so the detector retains sensitivity to dense whole-system faults".

**Problem.** "Null" suggests a null hypothesis or an empty model; the expert is the whole-plant density. A non-specialist reads it as "the do-nothing expert".

**Fix.** Call it the "whole-plant expert" (and relabel the Figure 2 box "Whole-plant expert").

### 13. [M | §4.2 Table 2, lines 508-528] Table 2 duplicates Table 6 and its caption argues

**Problem.** Table 2 is a single-model two-dataset preview of Table 6 (five seeds, three datasets). Its caption is a 90-word argument ("on WADI the residual is in fact informative (0.742), but (as Table 6 and §4.3 iii show) it does not generalize ..."), and the paragraph after it repeats the HAI numbers a second time ("This is what happens on HAI, where the residual term (0.689) trails the latent term (0.760) ...").

**Fix.** Drop Table 2 and cite Table 6 from §4.2, or keep it with a one-line caption: "Difficult-subset AUROC of the two natural score terms, single model per dataset; the five-seed decomposition is Table 6." Delete "in fact" (also at line 1066). Cut the paragraph at line 522 to: "A flexible decoder reconstructs a correlation-break window faithfully because its marginals are in range, so the residual carries little signal; on HAI the residual (0.689) trails the latent term (0.760) and drags the summed score down. The base score therefore works entirely in the latent; reconstruction re-enters only through the auto-gated residual head (§4.3 iii)."

### 14. [M | auto-gate story told six times]

**Locations.** §4.2 (line 525-528), §4.3 iii (584-588), §4.3 iv (596), the auto-gating note (598-599), §7 "Auto-gating adapts" (1074-1078), Table 6 paragraph (1064-1072).

**Fix.** Explain the gate once in §4.3 (iii) with the three ratios; keep the boxed note (it is a good summary); delete the "Auto-gating adapts one architecture" paragraph in §7 (its content is in the Table 6 paragraph immediately above it) and the trailing sentence of line 596 ("The reported model configures its heads purely from train-normal statistics ... stays off on WADI.").

### 15. [M | §1 line 175] Defensive framing of the evaluation protocol

**Quote.** "The evaluation obstacle it addresses by adopting the critique's protocol, raw metrics with difficulty stratification, which we use rather than claim as a contribution."

**Rewrite.** "For evaluation we adopt the protocol the critique implies: raw metrics with difficulty stratification."

Related pre-emptive phrases to cut: line 769 "This is deliberate:" (start the sentence at "Point adjustment inflates F1 ..."); line 897 "The subset removes easily separated signal rather than selecting a favorable slice:" (replace with "The subset is harder, not merely different:"); line 884 "so the HAI advantage is not an artifact of the particular statistic that defines difficulty" (replace with "so the HAI advantage holds under either definition of difficulty"); line 738 "(an oracle, field-standard threshold, disclosed plainly)" (replace with "(the best threshold chosen on the test set, the field-standard convention)").

### 16. [M | §2 line 203 and §2.5 line 258] Orphan one-sentence paragraphs after deletions

**Problem.** §2's opening paragraph is now a single sentence with no link to LatAD, and §2.5 begins with a stand-alone parenthetical "(Per-dataset details are given in Section 5.1.)".

**Fix.** Fold the §2 sentence into §2.1's first paragraph. Delete the §2.5 parenthetical and open §2.5 directly with "The water-treatment and ICS-security datasets used here ...".

### 17. [M | §3 line 336-344] Silhouette paragraph adds a defensive parenthetical and a "regime ridge"

**Quote.** "(This low silhouette reflects a partition slicing a continuous regime ridge, not heavy mode overlap between distinct regimes, which Appendix C finds absent.)"

**Problem.** "Regime ridge" is undefined; the parenthetical pre-empts an objection nobody has raised at this point.

**Fix.** Delete the parenthetical (Appendix C is cited from §7). The paragraph then reads cleanly: low silhouette on WADI/HAI, higher on SWaT, and that ordering tracks where factorization gains most.

### 18. [M | §4.4 line 622] 70-word parenthetical inside a sentence

**Quote.** "... on that community's channels alone (a *separate* small VaDE per community, each with its own encoder, Gaussian mixture, and scoring heads, its regime count and latent dimension scaled to the community size |G|; nothing is shared across communities except the input window features, so the model is a stack of independent per-subsystem experts, tens per plant, plus one global expert). We read its calibrated surprise ..."

**Rewrite.** Make it its own sentence: "For each community G we fit the full detector of §4.1 to 4.3 on that community's channels alone. Each community gets its own small VaDE (encoder, Gaussian mixture, scoring heads), with regime count and latent dimension scaled to |G|; nothing is shared across communities but the input features, so the model is a stack of independent subsystem experts, tens per plant, plus one whole-plant expert. We read each expert's calibrated surprise s_G ..."

### 19. [M | jargon without gloss on first use]

| Term | First use | Suggested gloss |
|---|---|---|
| whitened (Mahalanobis) residual | Table 1 A7 row (line 315), §4.3 iii | "whitened (scaled by the normal-data covariance, so correlated channels are not double-counted)" |
| Ledoit–Wolf-shrunk precision | line 235, 576 | "(a covariance estimate regularized toward the identity, stable when channels outnumber samples)" |
| nearest-component NLL | Table 1 A5 row (line 308), before NLL is expanded at line 501 | expand "negative log-likelihood (NLL)" in the Table 1 row |
| Higher Criticism | abstract, §1 line 188 | on first body use: "Higher Criticism (a statistic that asks whether more subsystems are mildly surprised than chance allows)"; the gloss now sits only at line 641 |
| hardware-in-the-loop | Table caption/§5.1 | "(a real controller driving a simulated plant)" |
| construct-matched | lines 246, 767, 787 | "scored on the same windows and difficulty subsets" |
| variance-floor artifact | line 1080, 1322 | explained only in Appendix C; if item 1 is adopted the §7 use disappears |
| overlap coefficient, density-valley ratio, Bayes error | §7 line 1080 | move to Appendix C (item 1); gloss "Bayes error (the misclassification rate of an ideal classifier between the two regimes)" there |
| one-hot encoded | line 760 | "(each actuator state as its own binary input)" |
| z-score units | line 659 | "standardized units" |

### 20. [L | §5.1 WADI bullet, line 673-676] Channel-drop sentence overloaded

**Quote.** "(its train-normal mean 9.09 and standard deviation 0.16 shift to a test-normal mean of 4503, a 27,359σ break, unique among the dataset's 123 raw channels)"

**Rewrite.** "One channel, 2B_AIT_002_PV, is rescaled between the normal and attack recordings (train-normal mean 9.09, test-normal mean 4503) and is dropped; all WADI results use the remaining 122 channels." The 27,359σ figure adds nothing a reader can use.

### 21. [L | §7 line 1093-1094] Misplaced adverb

**Quote.** "Excluding that single train-empty regime label-free (no anomaly is removed; 18.4% of test-normal windows) raises ..."

**Rewrite.** "Excluding that regime, a label-free operation that removes 18.4% of test-normal windows and no anomaly, raises ..."

### 22. [L | Appendix B, line 1254-1261] Twelve numbers per sentence

Acceptable for an appendix, but the paragraph repeats every cell of Table B1. Cut to the two conclusions (training 47 to 72 times faster than TranAD; per-window latency within gateway budgets on GPU and CPU edge) and let the table carry the rest.

### 23. [L | §4.3 ii, line 563-567] Role-of-the-head sentence is long and slightly defensive

**Quote.** "This term is a rare-regime safety head: its role is to keep a valid window in a sparsely populated regime from being flagged for rarity alone (A5), not to maximize difficult-subset AUROC. On a dataset without such rare-regime false alarms it can leave the base score marginally below the density head alone (Table 6, WADI), and it earns its place by protecting the false-alarm rate on rare regimes rather than by raising the aggregate score."

**Rewrite.** "This head protects rare regimes (A5): it keeps a valid window in a sparsely populated regime from being flagged for rarity alone. Where no such false alarms occur it can sit marginally below the density head alone (Table 6, WADI)."

---

## Logical arc and key explanations (items 5 and 6 of the brief)

**Arc.** Problem (§1) → assumptions (§3) → method (§4) → protocol (§5) → results (§6) → mechanism and limits (§7) is the right order and each section's job is clear. Two breaks: the §1 scope note discusses A8 to A10 status before §3 exists (item 4), and §7 spends more words on the assumption the detector does not use (A8) than on the two mechanisms that produce the result (items 1, 2). After items 1 to 4 the arc reads cleanly.

**Central thesis.** Stated clearly and carried through: line 162 ("Reconstruction measures reachability; detection needs probability."), §7 "Why reconstruction fails and density wins", and the Conclusion's "The mechanism is simple to state." This is the paper's strongest writing; nothing to change.

**Key explanations, each once and plainly?**

- *VaDE latent density* (§4.1): clear, single place, plain preamble before Eq. 2 and 3. Good.
- *Regime-community factorization + cohesion-weighted HC* (§4.4): the physical motivation (subsystems, locality, dilution) is excellent. The mechanics are dense: the ⊕ combiner, the cohesion weight, HC, the p-value exponent, and the whole-plant fusion arrive in one paragraph (lines 634-664). Split after "Higher Criticism ... interpolates between the two without assuming the extent in advance." and start a new paragraph for Eq. 9 and the weighting. Rename "null expert" (item 12).
- *Auto-gated residual head*: clear in §4.3 iii, then repeated five more times (item 14).
- *Difficulty stratification*: §5.3 is clear and short. The term is used from the abstract on; add the eight-word gloss in the abstract (item 5). The intro's gloss at line 196-197 is good.
- *MIIM A1 to A10*: Table 1 does the job. The A8 row is the longest row in the table and could be halved once §7 and Appendix C carry the detail: "Adjacent regimes could overlap so that a window is ambiguous while every channel is in range. Precondition present, overlap absent on the datasets studied (§7, Appendix C)." / "No head; specified for future work."

## Summary of the highest-value edits, in order

1. Replace the §7 A8 paragraph with the ~150-word version (item 1) and move the audit to Appendix C with forward-only wording.
2. Delete five of the eight "no A8 head" restatements (item 2).
3. Delete the "43-to-30 ... now fixed" sentence (item 3).
4. Cut the §1 scope-note parenthetical (item 4).
5. Rewrite the abstract's last sentence (item 5).
6. Merge the duplicate CPS/IIoT definitions (item 6).
7. Split the §6 difficult-subset paragraph by dataset and fix "significantly ahead of every learned detector" (item 7).
8. Replace `HC_coh`, "artifact-free", "canonical", "headline detector", "null expert", "shipped" (items 8, 10, 11, 12).
