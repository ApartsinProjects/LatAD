# Review 3: language, clarity, tone, flow (IoT2.html, read as accepted-changes text)

Scope: every sentence of `poc/paper/IoT2.html` (1,396 lines), read top to bottom as it will read once all
green insertions are accepted and all struck text is removed. Change-marking itself is not flagged.

Mechanical style check (whole file): no em-dashes or `&mdash;`; no double hyphens in prose (the `--` hits are
CSS custom properties and one SVG comment); no "honestly / frankly / candidly"; author spelled Apartsin;
no "seems / arguably / admittedly / we acknowledge / it should be noted" on a measured result. The one
"appears" (line 1104) is spatial ("appears as a contiguous block"), not a hedge.

Verdict on the four regions singled out: the split Section 6 difficult-subset block reads cleanly; the
"whole-plant expert" relabel is consistent everywhere except two spots (H6, M9); the "cohesion-weighted
Higher Criticism" usages are consistent; the Section 7 A8 paragraph parses but carries two unglossed terms
and one measure list that does not match Appendix C (H4); the Table A1 prose has one sentence that no
longer parses on first reading (H5). Two genuine self-contradictions remain (H1, H2).

Ranking: H = hurts the reader or contradicts the paper; M = noticeable friction; L = polish.

---

## HIGH

**H1 | Abstract, line 82 | Factual self-contradiction with Section 6.**
Text: "ahead of every learned detector on WADI (0.771), where it ties the linear baseline on the full set"
Section 6 (lines 840-841, 864-865) and Table 3 say the linear baseline is *ahead* on the full set (0.834 vs
0.827); the tie is on the *difficult* subset (0.771 vs 0.750, P = 0.46).
Rewrite: "and ahead of every learned detector on WADI (0.771), where it ties the linear baseline; on HAI
the deep detectors USAD and TranAD fall to 0.44-0.48."
Also drop "canonical" from the abstract ("canonical SWaT" is defined only in Section 5.1).

**H2 | Section 7, lines 1138-1140 | Two adjacent sentences contradict each other, and the second is revision-journey narration.**
Text: "Because the community construction and the cohesion-weighted Higher Criticism combiner were developed
on these three benchmarks, validation on additional CPS systems is the next test ... The community
construction and combiner settings were fixed before the WADI and SWaT evaluations reported here, so those
two results are out-of-sample confirmations of the design."
If the settings were fixed before WADI and SWaT were evaluated, they were not "developed on these three
benchmarks". "fixed before the ... evaluations reported here" narrates the project timeline.
Rewrite: "The community construction and combiner settings were fixed on HAI; the WADI and SWaT results are
therefore out-of-sample confirmations of the design. Validation on further CPS systems is the next test, and
because build and calibration use train-normal data only, that test requires no labels."

**H3 | Appendix C, line 1329 | Leftover reference to a removed head, plus debugging narration.**
Text: "Its responsibility entropy of 0.29 is a variance-floor artifact: the VaDE component log-variance floor
of log(0.05) manufactures apparent overlap on so few windows, and refitting with empirically estimated
component variances drops the entropy to 0.045 to 0.048, the same as WADI (0.053 to 0.057), with the
basin-head lift on the difficult subset going to zero on every seed."
The "basin head" was deleted from the model (struck at lines 320 and 596); no reader of the accepted text
knows what it is. The whole sentence diagnoses a measure (responsibility entropy) that the same appendix has
already declared invalid, so it reads as lab-notebook narration.
Rewrite the paragraph to: "**SKAB.** SKAB [54], a 400-window single-loop rotor rig, has no close-distinct
regime pairs (0 of 12), so it does not exhibit the A8 precondition. A8 is specified for completeness but not
demonstrated on any dataset in this study."

**H4 | Section 7 A8 paragraph, line 1093 | Reads smoothly but two terms are unglossed and one list does not match Appendix C.**
(a) Bold lead-in "Assumption A8 (between-regime overlap): specified, over-coverage defended, masking coupled to A10."
is a telegraphic run-on. Rewrite: "**Assumption A8 (between-regime overlap).**"
(b) "The one A8 variant the reported model already defends is over-coverage: a wide two-component head misses
about 80 percent of WADI's isolated anomalies ..." and "Masking, the remaining variant, is identifiable only
through the path ..." Neither "over-coverage" nor "masking" is defined before this point; Appendix C defines
them a page later. Insert glosses on first use: "over-coverage (a density head so coarse that its normal
mass spreads across the gap between regimes and covers an isolated anomaly)" and "masking (an anomaly of one
regime absorbed by an adjacent regime's density)".
(c) "under four observation-space measures (overlap coefficient, density-valley ratio, cross-regime
nearest-neighbour share, and discriminant error)" but Appendix C and Table C1 report only the overlap
coefficient and density-valley ratio. Either name the same measures in both places or write "under four
observation-space measures, two of which are tabulated in Table C1 (overlap coefficient and density-valley
ratio)".
(d) Section 7 line 1127 renames A8 as "between-regime proximity and masking"; Table 1 and every other mention
say "between-regime overlap". Use one name.

**H5 | Appendix A prose, lines 1206-1212 | One 130-word sentence; the closing parenthetical no longer parses on first read.**
Text: "... and among the sparsity-adaptive combiners, cohesion-weighted Higher Criticism is the one that
leads on all three once the community experts and the whole-plant expert are fused (and, on WADI, the
community aggregation without that fusion leads outright), because it weights a violation ..."
The parenthetical mixes the aggregation axis with the fusion axis mid-sentence and the reader cannot tell
which row "leads outright".
Rewrite as four sentences: "Factorizing the global density over regime communities lifts all three datasets,
most on WADI, whose difficult faults are confined to small correlated subsystems that a global density
dilutes. On WADI the community aggregation carries the whole effect: cohesion-weighted Higher Criticism over
the per-community experts lifts the single-latent global density from 0.634 to 0.795, and fusing in the
whole-plant expert (the headline fusion) costs 0.024, so the headline settles at 0.771. On HAI and SWaT the
whole-plant-expert fusion instead lifts the score, most on HAI, where the hard valve and flow deviations are
correlation-conditional. Among the aggregation rules, all fused with the whole-plant expert, cohesion-weighted
Higher Criticism leads on all three datasets because it weights a violation by the coupling strength of the
subsystem it occurs in (A7) while adapting to the fault's extent."

**H6 | Table A1 caption, line 1219 | Relabel left a doubled adjective.**
Text: "Every aggregation-block row is fused with the global whole-plant expert"
Rewrite: "Every aggregation-block row is fused with the whole-plant expert".

**H7 | Section 3, lines 271-272 and Table 1 caption line 282 | Claim contradicts the A8-A10 rows.**
Text: "each is grounded in a physical property of CPS operation and each motivates a specific design choice
in the detector of Section 4" and "LatAD is built to exploit each property when it is present and to reduce
to a standard latent-density model when it is absent."
A8-A10 motivate no design choice in the reported detector (their "Motivates" cells say so), and A8 has no
component to "exploit".
Rewrite: "each is grounded in a physical property of CPS operation, and A1-A7 each motivate a specific design
choice in the detector of Section 4. ... LatAD is built to exploit each of A1-A7 when it is present and to
reduce to a standard latent-density model when it is absent; A8-A10 are specified and measured but not
realized (Section 7)."

**H8 | Section 4.3 (iv), line 596 | Residue of the deleted basin head reads as an unexplained disclaimer.**
Text: "The stack has no head for A8 (between-regime overlap): that assumption is specified in Table 1
(Section 3) but not realized by this model, so Equation (7) is the detector's complete window score."
A reader who never saw the deleted head wonders why the paper stresses the absence of one. The A8 statement
already appears at line 448. Delete this sentence; keep "The reported model configures its heads purely from
train-normal statistics, with no test-set tuning." and drop its trailing colon clause (see M4).

---

## MEDIUM

**M1 | Introduction, lines 88-90 | CPS and IIoT are each defined twice in consecutive paragraphs.**
Line 88 (green): "A cyber-physical system (CPS) couples networked sensors, actuators, and controllers with a
physical process, forming the sensing-and-actuation layer of the Industrial Internet of Things (IIoT)".
Line 90: "A modern CPS (...) couples hundreds of networked sensors, controllers, and actuators into a unit
that continuously senses its own state and acts on it: the sensing-and-actuation fabric of the Industrial
Internet of Things (IIoT)."
Rewrite line 90's opening: "A modern CPS (a water-treatment plant, an industrial control loop, a rotating
machine) couples hundreds of such sensors, controllers, and actuators into a unit that continuously senses
its own state and acts on it."

**M2 | Introduction, line 175 | Inverted word order and a dangling "the critique".**
Text: "The evaluation obstacle it addresses by adopting the critique's protocol, raw metrics with difficulty
stratification, which we use rather than claim as a contribution."
Rewrite: "For the evaluation obstacle it adopts the protocol that critique implies, raw point-wise metrics
with difficulty stratification; we use that protocol rather than claim it as a contribution."

**M3 | Section 2 opener, line 203 | After deletions the section opens on one orphan sentence.**
Remaining: "On time series, anomaly detection must respect temporal and cross-channel dependence [12], and
on cyber-physical and industrial control systems it underpins physics-aware intrusion and fault detection [13]."
Rewrite: "Four lines of prior work frame this paper: anomaly detection in the Industrial IoT, the critique of
time-series anomaly-detection evaluation, latent and clustering detectors, and deep multivariate CPS
detectors. On time series, anomaly detection must respect temporal and cross-channel dependence [12], and on
cyber-physical and industrial control systems it underpins physics-aware intrusion and fault detection [13]."

**M4 | Sections 4, 4.2, 4.3, note box | "residual on for HAI and SWaT, off for WADI" is stated five times.**
Lines 448-449, 527-528, 587-588, 596, 599. Keep the measured statement at 587-588 (with the ratios) and the
one-line auto-gating note; delete the clause at 596 ("the residual head fires on HAI and SWaT and stays off
on WADI") and shorten 448-449 to: "the whitened-residual head (A7) is auto-gated on train-normal signals
(Section 4.3 iii)."

**M5 | Section 4 realization paragraph, lines 445-452 | Repetition and an interrupting A8 sentence.**
Text: "... activates on HAI and SWaT (Table 6). A8 (...) is deferred to future work (Section 7, Appendix C).
The gated residual head did not reduce performance on any of these datasets, and the reported gains come from
the always-active heads, with the auto-gated residual head adding to them on HAI and SWaT (Table 6)."
Rewrite: "... the whitened-residual head (A7) is auto-gated on train-normal signals; the reported gains come
from the always-active heads, and the gated residual adds to them where it fires (HAI and SWaT, Table 6).
A8 (between-regime overlap) is specified in Table 1, but the overlap it posits is absent from the datasets
studied, so it is deferred to future work (Section 7, Appendix C). Assumptions A9-A10 ..."

**M6 | Section 4.2, Table 2 caption (509-512) and following paragraph (523-525) | Verbatim duplication.**
Both say: HAI residual 0.689 trails latent 0.760, drags the joint score down, so demoting helps. Keep it in
the paragraph; trim the caption to: "Difficult-subset AUROC of the two natural score terms (single trained
model per dataset). A multi-seed, three-dataset head ablation is reported in Table 6."

**M7 | Section 4.3 (iii), lines 585-586 | Gate logic reads backwards.**
Text: "if held-out normal scores much higher (the residual overfits or drifts), the ratio q95(B)/q95(A)
exceeds 1.5 and the head is switched off."
Rewrite: "the head is switched off when the ratio q95(B)/q95(A) exceeds 1.5, that is, when held-out normal
scores much higher than the fitting split (the residual overfits or drifts)."

**M8 | Section 4.4, line 622 | Pronoun stranded after a 60-word parenthetical; three names for one object.**
Text: "(... plus one global expert). We read its calibrated surprise s_G ..."
"its" is far from "community". Also "global expert" (622), "global detector" (624) and "whole-plant expert"
(624) name the same model in one paragraph.
Rewrite: "(... plus one whole-plant expert). We read each community's calibrated surprise s_G (the
train-normal upper-tail negative log-probability), and we include the global detector of Sections 4.1-4.3, the
LatAD (global density) model, as an unfactorized whole-plant expert so the detector retains sensitivity to
dense whole-system faults."
Same fix at line 349: "plus a global expert" -> "plus a whole-plant expert".

**M9 | Section 4 preview, line 350-351, vs 4.4 heading, line 601 | "Two stages" but three named stages.**
Line 350: "Two stages implement it. A representation stage ... a scoring stage ..." Line 601 heading: "(the
subsystem-factorization stage ...)".
Rewrite line 350: "Three stages implement it: a representation stage (a VaDE that jointly learns a latent and
a Gaussian-mixture regime prior), a scoring stage (a stack of density heads in that latent, calibrated on
train-normal), and a subsystem-factorization stage (Section 4.4)."

**M10 | Section 3, lines 340-344 | "This ordering tracks" is not supported by the numbers that follow, and "regime" is used in a non-technical sense.**
Text: "This ordering tracks where the design pays off: the regime-community factorization gains most on
WADI, and by a similar margin on HAI and SWaT ... This is the regime in which a jointly learned latent ...
pays off"
HAI (0.08) and SWaT (0.29) differ in silhouette but gain the same, so the ordering does not track; "pays off"
appears twice; "regime" here collides with the paper's technical term.
Rewrite: "Where regimes are least separated the design gains most: the regime-community factorization lifts
the WADI difficult subset by +0.14 and HAI and SWaT by about +0.03 each (Section 6, Table A1). Overlapping,
imbalanced regimes are the setting in which a jointly learned latent, scored by a mixture density and
factorized over subsystems, wins and a global detector or a raw reconstruction residual does not."

**M11 | Section 5.1 WADI bullet, lines 676-677 | Clipping example does not explain itself.**
Text: "clipped to +-10 sigma to cap sensor glitches (for example a channel constant at zero in the 14-day
normal record)."
Rewrite: "clipped to +-10 sigma so that a channel with near-zero train-normal variance (one is constant at zero
throughout the 14-day normal record) cannot produce unbounded standardized values on a single glitch."

**M12 | Section 5.1 SWaT bullet, line 685 | Filename token in prose.**
Text: "the `SWaT_Dataset_Attack_v0` attack recording"
Rewrite: "the December-2015 attack recording (release v0)". The file name can stay in the Data Availability
statement if needed.

**M13 | Section 5.4 significance, lines 750-751 | Number does not follow from the stated formula.**
Text: "block length ceil(W/stride)+1 windows (3 on WADI, 4 on HAI and SWaT)". With W = 60 and stride = 30
unified across datasets (Section 5.2), ceil(60/30)+1 = 3 everywhere. Either the formula or "4 on HAI and
SWaT" is wrong; verify against the bootstrap script and make the text match.

**M14 | Section 6 overall, lines 838-843 and 864-865 | "both deep detectors" with three in the table; duplicated WADI full-set sentence; "on both ... on both".**
Text: "clearing every baseline and both deep detectors on both, including the strong classical baselines on
the more separable SWaT."
Rewrite: "clearing every baseline and all three deep detectors on each, including the strong classical
baselines on the more separable SWaT."
Delete the second occurrence at 864-865: "On the full set the linear baseline is marginally ahead of LatAD
(0.834 vs 0.827)."

**M15 | Section 6 WADI paragraph, line 862, and Section 7 lines 1027, 1057 | 0.795 appears without saying why it differs from the reported 0.771.**
Text: "(cohesion-weighted Higher Criticism over the community experts, 0.795)"
Rewrite at first use (line 862): "(cohesion-weighted Higher Criticism over the community experts alone,
0.795; 0.771 after fusion with the whole-plant expert, Table A1)". The two later mentions can then stay short.

**M16 | Section 6 deep SOTA, line 935 | Wrong quotation.**
Text: "the concrete face of the 'illusion of success'"
The cited phrase (Wu and Keogh, Section 2.2, line 219) is "illusion of progress". Rewrite accordingly.

**M17 | Table 5 caption, line 1009 | Claims a significance test that is not described.**
Text: "Bold marks the cross-channel gains that survive the significance test."
No test of the cross-channel gain itself is reported; the tests are LatAD vs baseline.
Rewrite: "Bold marks the two datasets on which LatAD's difficult-subset lead over the strongest baseline is
significant (Section 6)."

**M18 | Section 6 "Where the gain comes from", line 1005 | Channel id repeated in prose; sentence explains an exclusion already made.**
Text: "a rescaled analyzer channel (`2B_AIT_002_PV`) in the public WADI release inflates single-channel
separability and is excluded from the evaluation (Section 5.1)."
Rewrite: "(the rescaled analyzer channel excluded in Section 5.1 would otherwise inflate single-channel
separability further)."

**M19 | Section 7 reconstruction paragraph, lines 1038-1040 | "whose" attaches to the wrong noun.**
Text: "fall to near chance on HAI (USAD 0.477, TranAD 0.444, and the graph detector GDN 0.481, which learns
inter-channel relations yet collapses on the same faults), whose difficult faults are reconstructable but
improbable."
Rewrite: "fall to near chance on HAI, whose difficult faults are reconstructable but improbable (USAD 0.477,
TranAD 0.444; the graph detector GDN, which learns inter-channel relations, collapses on the same faults at
0.481)."

**M20 | Section 7 Table 6 discussion, lines 1082-1084 | Causal slip about the gate.**
Text: "consistent with SWaT's difficult faults being larger multivariate deviations that reconstruction
catches, which is why the gate keeps it on there."
The gate sees only train-normal; it keeps the head on because the residual generalizes to held-out normal,
not because the faults are large.
Rewrite: "consistent with SWaT's difficult faults being larger multivariate deviations that reconstruction
catches; the gate, which sees only train-normal, keeps the head on there because the residual generalizes to
held-out normal."

**M21 | Section 7 benchmark coverage, lines 1106-1110 | Awkward verb and hedged close.**
Text: "lifts a secondary coverage-diagnostic view of the headline model (in that pass all-subset AUROC 0.948
to 0.965, ...), with LatAD leading throughout. The main tables report the full test set; we note the effect
as a secondary view and as evidence that ..."
Rewrite: "raises the headline model's all-subset AUROC from 0.948 to 0.965, its difficult-subset AUROC from
0.845 to 0.890, and its true-positive rate at a 5% false-alarm budget from 0.789 to 0.916, with LatAD leading
throughout. The main tables report the full test set; this secondary view shows that the regime-community
model diagnoses its own coverage gaps."

**M22 | Section 7 A9-A10 paragraph, lines 1115-1122 | Two over-dense sentences.**
(a) "On HAI a longer 600-sample window adds about +0.06 difficult-subset AUROC on the part of the difficult
subset whose history is attack-free, a slow persistent fault that accumulates over the longer horizon,
corroborating the +0.072 that explicit temporal features add in Appendix A; on WADI and canonical SWaT longer
windows do not help."
Rewrite: "On HAI a longer 600-sample window adds about +0.06 difficult-subset AUROC on the attack-free-history
part of the difficult subset, where a slow persistent fault accumulates over the longer horizon; this
corroborates the +0.072 that explicit temporal features add in Appendix A. On WADI and SWaT longer windows do
not help."
(b) "three conditional scorers (a linear autoregressor on the two preceding windows, a nearest-neighbour
successor model, and a regime-transition matrix), validated on a synthetic positive control and guarded by an
order-shuffle control, find no window that is marginally normal yet improbable given its history above the
shuffle floor, on all three benchmarks, once windows whose history already contains an attack are excluded."
Rewrite: "Three conditional scorers (a linear autoregressor on the two preceding windows, a nearest-neighbour
successor model, and a regime-transition matrix) were validated on a synthetic positive control and guarded by
an order-shuffle control. On all three benchmarks, after excluding windows whose history already contains an
attack, none finds a window that is marginally normal yet improbable given its history above the shuffle
floor."

**M23 | Section 7, paragraphs at 1112 and 1127 | Same conclusion stated twice in consecutive paragraphs.**
Both end on "needs operational-scale / history-dependent data". Merge: delete the last sentence of the A9-A10
paragraph ("Extending the detector ... requires data that contains a history-dependent fault.") and let the
"Benchmark scale" paragraph carry it.

**M24 | Section 7 IIoT deployment, lines 1150 and 1157-1160 | Appendix B pointed to twice; closing sentence repeats the one before it.**
Delete line 1150's "The training and inference cost of this configuration is measured in Appendix B
(Table B1)." Delete the final sentence of the next paragraph ("The CalexNet early-exit path above is a
further-reduction route ..."), which restates "which applies directly to the per-community expert stack on
constrained edge hardware" from two sentences earlier.

**M25 | Section 2.5, line 259 | Repeats line 224 verbatim and cites A8-A10 before Section 3 defines them.**
Text: "We therefore read every method per dataset and per difficulty subset, and flag operational scale as
the setting in which the trajectory assumptions A9 and A10 and the masking form of A8 become testable (Section 7)."
Rewrite: "Operational scale is therefore the setting in which the trajectory assumptions of Section 3 (A8-A10)
become testable (Section 7)."

**M26 | Section 2.5 first paragraph, line 258 | A parenthetical stands alone as a paragraph.**
Remaining text: "(Per-dataset details are given in Section 5.1.)"
Fold it into the next paragraph's opening: "Per-dataset details are given in Section 5.1; here we place the
three testbeds on the scale axis. The water-treatment and ICS-security datasets ..."

**M27 | Conclusion, lines 1174 and 1180-1181 | "restated" and a misattached "whose".**
"We restated the structural assumptions" -> "We stated the structural assumptions".
"and a lead over every learned detector on WADI (tying the linear baseline), whose difficult faults are
reconstructable but improbable." -> "and a lead over every learned detector on WADI, where it ties the linear
baseline. In every case the difficult faults are reconstructable but improbable."

**M28 | Section 4.3 (ii), lines 563-567 | Defensive pre-emption; "marginally" understates 0.022.**
Text: "On a dataset without such rare-regime false alarms it can leave the base score marginally below the
density head alone (Table 6, WADI), and it earns its place by protecting the false-alarm rate on rare regimes
rather than by raising the aggregate score."
Rewrite: "Its role is protection against rare-regime false alarms (A5), not difficult-subset AUROC; on WADI,
which has no such false alarms, the base score sits 0.02 below the density head alone (Table 6)."

**M29 | Section 6 validity paragraph, line 879 | "ceilings" is unclear.**
Text: "so all three are discriminative tests of joint-structure detection rather than ceilings."
Rewrite: "so all three are discriminative tests of joint-structure detection rather than subsets on which
every method is already saturated."

---

## LOW

**L1 | Abstract, line 82 | Two 60+-word sentences.** Split the first at the semicolon: "... three are
specified for future work. A fault can be individually normal on every sensor ..." Split the method sentence:
"... scores by probability rather than reconstruction. It factorizes the density over correlation-community
subsystems, combines per-subsystem surprises by cohesion-weighted Higher Criticism, and re-admits
reconstruction only through a residual head gated on held-out normal data."

**L2 | Introduction, line 177 | Hyphenated -ly adverb.** "Slowly-progressing" -> "Slowly progressing".

**L3 | Introduction, line 192 | ROC unglossed.** "AUROC (area under the ROC curve)" -> "AUROC (area under the
receiver-operating-characteristic curve)".

**L4 | Section 2.1, line 206 | Abrupt fragment-like close.** "Process-level, subsystem-localized detection
from normal-only telemetry, under regime imbalance, remains open." -> "What remains open, and what LatAD
addresses, is process-level, subsystem-localized detection from normal-only telemetry under regime imbalance."

**L5 | Section 2.3, line 235 | "Standard tools used later" is a list, not a sentence.** -> "We also draw on
standard tools: Ledoit-Wolf covariance shrinkage [33] in the residual head (Section 4.3), and Isolation Forest
[34] and deep autoencoders [35] as baselines (Section 5.5)."

**L6 | Section 2.4, lines 245-248 | A result (GDN 0.481 on HAI) is reported inside Related Work, before the
protocol is set up.** Move the clause "on the difficult subset it collapses on HAI (0.481) ..." to Section 6
and end the 2.4 sentence at "(Section 5.5)".

**L7 | Section 3, line 338 | BIC optimum sits at the edge of the grid.** "(BIC-optimal K* of 22, 24, and 25
on WADI, HAI, and SWaT, on a component grid extended to 25)" -> "(BIC keeps improving up to the edge of a
25-component grid, with optima at 22, 24, and 25 on WADI, HAI, and SWaT)".

**L8 | Section 4.3 (i) heading, line 536 | "KDE" is defined and never used again.** Drop the acronym:
"(a parametric kernel-density estimate for non-Gaussian pockets)".

**L9 | Section 5.4, line 745 | "which we flag" is odd.** "only the F1 threshold uses the test-swept oracle,
as disclosed above."

**L10 | Section 6 stronger-split paragraph, line 887 | Reader will ask why the headline detector is absent.**
Add "(the headline regime-community model was not re-run on this split)" or report it.

**L11 | Table 4 caption vs body text, lines 895-903 and 912-916 | Same six numbers in both.** Keep them in
the text; cut the caption to the definition plus window/episode counts.

**L12 | Table 6 caption, line 1063 | Caption labels ("recon", "resid") do not match row labels
("reconstruction residual (dropped term)", "base + resid (auto)").** Use one label set.

**L13 | Limitations, line 1166 | "train-empty regime" is jargon.** -> "a regime absent from training".

**L14 | Section 7, line 1112 | "leverages"** -> "uses".

**L15 | Appendix B, lines 1265, 1268 | Vendor name in prose; per-window latency rising with batch is
confusing.** "one Modal A10G GPU" -> "one cloud A10G GPU (24 GB, Modal)". "17 to 32 ms per window (batch 1;
up to about 52 ms at batch 256)" -> state whether 52 ms is per window or per batch; if per batch, say
"(0.2 ms per window at batch 256)".

**L16 | Appendix B and Section 7, "LatAD-global" / "LatAD (global)" / "LatAD (global density)"** name the
same configuration three ways. Use "LatAD (global density)" throughout, or define the short form once.

**L17 | Appendix C, lines 1294-1300 | One sentence lists four measures with nested parentheticals.** Split
into a short lead sentence and a four-item list.

**L18 | Appendix C, line 1302 | SKAB and the Paderborn testbed are uncited on first mention.** Add "[54]"
after SKAB at line 1302 and a citation for Paderborn (or drop it from the list).

**L19 | Table C1 caption, line 1313 | "=" used as a word.** "higher mass, smaller D = more adjacency" ->
"higher mass and smaller D mean closer adjacency".

**L20 | Table A2 caption vs Table 3 | "not the headline numbers"** yet the six-statistics WADI value
(0.634 +- 0.011) is identical to Table 3's global-density value while HAI and SWaT differ. Say "a reduced
configuration on HAI and SWaT" or align the caption with what was run.

**L21 | Whole paper | Mixed UK/US spelling.** US dominates ("labeled", "standardization", "modeling");
UK forms survive at "labelled" (1093, 1298, 1302, 1310, 1315), "initialisation" (493, 496), "neighbour(ing)"
(319, 1120, 1296, 1312), "manoeuvres" (306, 1188), "mislabelling" (218). Pick US and normalize.

**L22 | Reference order | [54] SKAB first appears (line 1329) after [55]-[59].** Under the first-appearance
scheme, either cite SKAB at its first mention (line 1302, still after [59]) or renumber Cranfield/SKAB to
follow MetroPT [53].

---

## Confirmations (no change needed)

- Split Section 6 difficult-subset block (lines 845-868): HAI, SWaT, WADI paragraphs each carry one dataset
  and its test; the one-sentence closing paragraph reads as a clean summary. Only M15 applies.
- "whole-plant expert": consistent in Figure 2, Section 4.4, Table A1 rows and Appendix A prose; the only
  slips are the doubled adjective (H6) and the one remaining "global expert" (M8).
- "cohesion-weighted Higher Criticism": every usage is consistent; the "(HC)" gloss at first use (line 188)
  is the right place, and Figure 2 / Table A1 use "HC" only after it.
- No revision-journey narration survives in the main text apart from H2 and H8; the struck paragraphs at
  lines 87-91 (auto-gating) and 228-232 (protocol) disappear cleanly.
- The struck inline "(A8, between-regime overlap, specified but not realized ...)" at line 177 and the
  struck "and cyber-physical" / green ". Cyber-physical" at line 160 both read correctly once accepted.
