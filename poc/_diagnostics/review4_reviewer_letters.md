# Review 4: skeptical-reviewer critique of IoT2.html (new claims) and audit of the three response letters

Scope: `poc/paper/IoT2.html` (3004 lines; all prose, tables, captions, Figures 1 to 3 SVG text, Figure A1 caption and in-figure labels read; the Figure A1 glyph paths skipped) and `review_round_1/letter_editor.html`, `letter_reviewer1.html`, `letter_reviewer2.html` (read in full), checked against `Review1.txt`, `Review2.txt`, `Editor review.txt`.

Known-correct numbers confirmed in the paper: HAI 0.845 (sig, P about 0), SWaT 0.837 (sig, P = 0.007), WADI 0.771 (sig vs AE/USAD/TranAD/IF/GDN, ties LinRes P = 0.46); density head beats USAD and TranAD on the difficult subset of all three (0.802 / 0.794 / 0.656 vs 0.477, 0.444 / 0.658, 0.655 / 0.579, 0.613); Table 7 top-1 SWaT 0.46, HAI 0.39, WADI 0.33 and top-3 0.79 / 0.68 / 0.56 (11/24, 15/38, 3/9 and 19/24, 26/38, 5/9 reproduce them). Change-marking is intentional and not flagged.

Mechanical checks (all pass): no em-dash, double hyphen, "honest*", "we acknowledge", "admittedly", "seems to / appears to / arguably / to some extent" in the paper or the letters; Apartsin spelled correctly everywhere; nine equations (1) to (9); references now in first-appearance order ([53] MetroPT at §7, [54]-[56] at §7, [57] CalexNet, [58] Cranfield and [59] SKAB in Appendix C); `paper_mdpi.docx` (21:11) and the three letter DOCX files (21:03) are newer than their HTML sources.

Review-3 items now applied: abstract WADI subset wording; block-length sentence; Host RAM column in Table B1; Table 5 caption; "developed on these three benchmarks" clause struck; USAD/TranAD seed disclosure; localization evaluation (Table 7); reference order; editor letter now states WADI significance over every learned detector; letters use the "30 to 2 percent Bayes error" control wording; R2 C8 location corrected; R2 C10 now names peak GPU memory.

Review-3 items still open (carried below): six-statistic split shown for HAI and the global model only (now A.7); "per-channel marginals already suffice" (A.9); "tracks" in §3 (A.15); hyperparameter table (A.16); gate threshold fixed a priori (A.17); community-stack cost ratio beside the global ratio (A.14); Figure 2 residual block lacks its [A7] tag (A.19); HAI and KL abbreviations (A.20).

---

## PART A. Paper, ranked

Severity: **MUST** = factual or internal-consistency defect a reviewer or copy-editor will catch. **STRONG** = a gap a skeptical reviewer will attack; fixable with wording or persisted outputs. **OPT** = optional strengthening.

### A.1 Table 7 / subsystem localization (task question 2)

Verdict: the result is bounded in the right way. Both caveats (calibration-drift naming bias; community-size confound) are stated in the same paragraph as the numbers, the output is called a "triage shortlist rather than a diagnosis", and the coverage-aware random baseline is defined in the caption. A reviewer would accept top-3 on SWaT and HAI as a validated triage output. The attack surface is the top-1 row and the missing size-weighted numbers, plus three factual inconsistencies.

| # | Sev | Location | Objection | Fix |
|---|---|---|---|---|
| 1 | MUST | Figure 1 SVG, line 147: arrow label "alerts + diagnosis" | The paper now says three times that the pointer is "a triage shortlist rather than a diagnosis" (§4.4 line 657, §7 line 1158, §7 line 1174). The figure says "diagnosis". | Change the label to "alerts + triage shortlist" (or "+ subsystem shortlist"). |
| 2 | MUST | §7 localization text, line 1168 "40 percent of HAI attacks (15 of 38 ...)" vs Table 7 HAI top-1 **0.39** | 15/38 = 0.395; the text rounds up, the table rounds down. Same number, two printed values on one page. | Print 0.39 / 39 percent in both places (or 0.40 / 40 in both). |
| 3 | MUST | §7 localization paragraph, lines 1162 to 1169 | The paragraph states the top-3 result twice in full: opening sentence ("68 to 79 percent ... against 49 to 56 percent") and again at line 1169 ("79 and 68 percent (random 49 and 56 percent)"). The second sentence, "The per-community surprises also localize.", is a leftover topic sentence that no longer follows anything. | Delete the opening summary sentence (the §4.4 and deployment paragraphs already carry it) or delete the repeat at 1169; drop "also". |
| 4 | STRONG | §7 line 1175 to 1177: "against a size-weighted random baseline the SWaT margin holds while HAI's top-1 margin is small" | The size-weighted baseline is the one that answers the confound the paragraph itself raises, and its numbers are not printed anywhere. "Small" is unquantified; a reviewer will ask whether HAI top-1 clears it at all, and whether top-3 clears it on either dataset. Table 7 bolds HAI top-1 0.39, which reads as a supported margin. | Add two columns to Table 7 (size-weighted random top-1 and top-3) or one sentence with the four numbers and their P-values; if HAI top-1 does not clear the size-weighted baseline, unbold it and say so. |
| 5 | STRONG | §7 line 1167 to 1168: P = 0.004 (SWaT) and P = 0.02 (HAI), top-1 only | The test is not named (one-sided binomial against the episode-averaged coverage-aware chance? a permutation over community labels?). The "comfortable headline", top-3, carries no P-value at all; only the row the paper calls weaker is tested. | Name the test in the caption and report P for top-3 on SWaT and HAI (the same test on the same episodes; no retraining). |
| 6 | STRONG | Table 7 column "Attack episodes (targets known)": SWaT 24, HAI 38, WADI 9 | The selection rule is not stated. HAI 20.07 has 38 attacks, so HAI uses all of them; SWaT's published table names a target for all 41 attacks (36 effective), and WADI's for 15, yet the paper uses 24 and 9. A reviewer will ask what excluded the other 17 SWaT and 6 WADI attacks (no anomaly window after 10x downsampling? target not among the retained channels? target inside no community?). The WADI cell "9 (7 coverable)" shows the third criterion exists but does not define it. | One sentence in the caption or text: "An episode is scored if it yields at least one anomaly window on the window grid and its published target maps to a retained channel; 'coverable' means the target lies inside at least one discovered community." Give the counts excluded by each rule. |
| 7 | STRONG | Table 7 evaluated on all attack episodes; the paper's thesis is the difficult subset | Most scored episodes are easy (SWaT 148 of 233 anomaly windows are easy). Localization on the joint-structure faults, the ones the paper is about, is not reported. A reviewer may argue the shortlist is validated only where a per-channel rule already points at the channel. | Report top-1 / top-3 restricted to episodes with at least one difficult window (SWaT 23, HAI 26 episodes exist per §6), or state in the paragraph that the localization test is episode-level over all attacks and leave the difficult-subset cut to future work. |
| 8 | OPT | Placement: the localization paragraph and Table 7 sit between the two halves of the "Implications for IIoT deployment" discussion (line 1152 deployment, 1162 localization, 1180 Table 7, 1189 "Pushing inference to the network edge" which continues the deployment thread) | A new quantitative result with its own table is inside the Discussion, interrupting a paragraph pair. | Move localization + Table 7 to §6 as a short "Subsystem localization" results paragraph (Table 7 becomes a results table), or place it after the edge paragraph under its own bold lead. |
| 9 | OPT | R2 letter C7 and paper line 1162 say "of HAI and SWaT attacks" | The denominator is the scored episodes with a known, coverable target (24 and 38), not all attacks. | "of the HAI and SWaT attack episodes with a known target". |

### A.2 Density-vs-reconstruction reframe (task question 1)

Verdict: the 3/3 claim is clean at the head level and the paper does not claim a predictive law. Table 6's caption now says explicitly that its reconstruction row is the whitened residual and that the deployed detectors score 0.44 to 0.66 on the same subsets, and §7 says the whitened residual overtakes the density head on WADI and SWaT, so there is no self-contradiction. Remaining problems are wording and two untabulated numbers.

| # | Sev | Location | Objection | Fix |
|---|---|---|---|---|
| 10 | MUST | §7 line 1041 to 1042: "Scoring by density rather than reconstruction is the right default **for the deployed reconstruction detectors**" | As written this says density is the right default for USAD and TranAD. The intended meaning is "relative to". | "Density, not reconstruction, is the better default score: the single latent-density head ... exceeds the deployed reconstruction detectors USAD and TranAD on ..." |
| 11 | STRONG | §7 line 1044: "and matches GDN on WADI (0.660)" | The density head is 0.656; GDN is 0.660. "Matches" a number that is ahead is the kind of phrasing a reviewer quotes. The gap is inside the head's seed SD (0.016), which is the defensible statement. | "and is within seed variation of GDN on WADI (0.656 vs 0.660, SD 0.016)". |
| 12 | STRONG | §7 lines 1045 to 1048: "place the difficult attacks at the median of normal operation while the density head places them at the 93rd percentile, and 27 percent of HAI's difficult attacks reconstruct at least as well as a typical normal window yet fall above the 90th normal percentile in density; on WADI that share is zero and 70 percent ... are weak under both scores" | Five numbers (median, 93rd, 27 percent, zero, 70 percent) appear in no table or appendix, and "the reconstruction detectors" does not say which one (USAD, TranAD, both, which seed). These are the paper's most quotable mechanism numbers and a reviewer cannot locate them. | Either add a three-row table to Appendix A (per dataset: median normal-percentile of difficult attacks under USAD, TranAD, and the density head; share reconstructable-yet-improbable; share weak under both) or attribute inline ("under USAD and TranAD, seed-averaged"). |
| 13 | STRONG | §1 line 162 "invisible ... and, as we show, to a reconstruction residual as well"; line 166 "a reconstruction residual is blind to exactly the joint-structure faults that matter"; §8 line 1221 "Treating multimodal normal ... as a reconstruction target, is a modeling failure" | Universal statements about "a reconstruction residual", while Table 6 shows the paper's own whitened residual is the strongest single head on SWaT (0.835) and informative on WADI (0.740). What the paper shows is that the deployed detectors' unwhitened, regime-agnostic residual collapses on HAI and trails everywhere, and that a regime-conditioned whitened residual is a useful gated term. The Discussion says this correctly; the Introduction and Conclusion still say the universal version. | §1: "to the reconstruction residual the deployed deep detectors score with"; §8: "as a reconstruction target alone" or "an unreliable default". The abstract sentence ("reconstruction-based detectors measure reachability rather than probability") can stay as the thesis. |
| 14 | OPT | §7 line 1195 and Appendix B: "LatAD (global density) trains roughly 47 to 72 times faster than TranAD" | Now correctly labelled as the base ablation, but the headline regime-community ratio (about 4.3 to 4.6x faster to train; 9 to 16x slower per window on GPU than TranAD's 2 ms) is left for the reader to compute, and "gateway-class real-time budgets" is still undefined. | Add the community ratio in the same sentence and define the budget: one window per 30-sample stride (30 s on HAI, 300 s on WADI and SWaT), so 52 ms is under 0.2 percent duty. |
| 15 | OPT | §3 line 341: "This ordering tracks where the design pays off: the regime-community factorization gains most on WADI, and by a similar margin on HAI and SWaT" | Silhouette order WADI 0.06 < HAI 0.08 < SWaT 0.29; gains +0.137 / +0.034 / +0.033. HAI and SWaT gain identically across a 3.6x silhouette difference, so "tracks" rests on one point. Review-3 item 14, still present. | Drop "This ordering tracks where the design pays off:" and keep the factual clause. |

### A.3 Figure A1 (task question 3)

Verdict: clearly labelled synthetic. The caption opens "Synthetic illustration", says "toy data with five Gaussian regimes", and closes "This is a synthetic mechanism illustration, not a claim about the benchmarks"; §7 references it as "on synthetic data"; no benchmark number is attributed to it. Two small points.

| # | Sev | Location | Objection | Fix |
|---|---|---|---|---|
| 16 | OPT | Figure A1 SVG in-figure text (axis labels "fault in-envelope-ness (0 = off-manifold, 1 = between-regime)", "AUROC(density) - AUROC(reconstruction)", legend "regime separation 4 / 6 / 10 (silhouette 0.51 / 0.65 / 0.78)"); no in-figure title | The word "synthetic" exists only in the caption. Lifted without its caption (a slide, a DOCX float that separates), the figure has no marker. Also, the three synthetic silhouettes (0.51 to 0.78) are far above every benchmark's (0.06 to 0.29, §3); a reviewer could note the illustration lives in a separation regime none of the plants occupy. | Add an in-figure title "Synthetic five-regime toy (not benchmark data)"; add one caption clause: "the benchmarks' silhouettes (0.06 to 0.29) lie below this range, so the figure illustrates the mechanism only". |
| 17 | OPT | Figure A1 sits at the end of Appendix A ("Ablation over the design axes") and is not mentioned in the Appendix A text; Data Availability says the study "created no new data" | The figure is orphaned in an appendix about a different topic, and the toy generator is new (tiny) data. | Give it a one-line lead-in in Appendix A or its own "Appendix D. Synthetic mechanism illustration"; note in Data Availability that the Figure A1 generator script is in the Zenodo release. |

### A.4 Remaining overreach, unsupported claims, missing limitations (task question 4)

| # | Sev | Location | Objection | Fix |
|---|---|---|---|---|
| 18 | STRONG | §6 "Robustness to a stronger difficulty definition", line 886: six-statistic split reported for HAI only and for LatAD (global density) only (0.675 vs 0.622) | Unchanged since review 3 (item 6). This is the paper's defense against "you chose the statistic that defines difficulty"; it covers one dataset and the ablation model, not the headline detector, with no CI. §5.3 line 724 says "§6 confirms the HAI advantage persists under a stronger split", which is true only for the base model. | Report the headline regime-community model on the six-statistic split for all three datasets with the episode-block bootstrap (persisted scores; minutes of agent time). If only HAI/global exists, say so in §5.3 as well. |
| 19 | STRONG | §6 line 1004 and Table 5 caption: "per-channel marginals already suffice" on WADI | Still present (review-3 item 8). Marginals score 0.639 and the trivial rule 0.601 against the headline 0.771 and the community aggregation 0.795; they do not suffice, they merely equal the cross-channel latent (0.656). The same paragraph then attributes the WADI gain to community factorization, and §7 line 1064 to subsystem-local coordinated drift, which is a cross-channel mechanism. | "capture as much as the single cross-channel latent does (0.639 vs 0.656); the WADI gain comes from estimating the density within each community and combining by Higher Criticism (0.634 to 0.795), not from cross-channel modeling at plant scale." Apply in both places. |
| 20 | OPT | §4.1, §5.2 reproducibility | No encoder/decoder widths, optimizer, learning rates, epochs, warm-up length, variance floor, DAGMM weight, batch size, or the per-community K / latent-dim scaling rule. Code on Zenodo mitigates; MDPI reviewers expect a table. (Review-3 item 12.) | Compact hyperparameter table in Appendix B. |
| 21 | OPT | §4.3 iii line 586: gate cut-off 1.5 between observed 1.22 and 3.41 | Nothing says 1.5 was fixed before evaluation. (Review-3 item 13.) | "fixed a priori" or "any threshold in [1.3, 3.3] gives the same gating on all three datasets". |
| 22 | OPT | Figure 2 SVG lines 407 to 412 vs footnote line 440 and caption | Whitened-residual block carries no [A7] tag in the drawing; the footnote and caption say it does. | Add the tag to the block. |
| 23 | OPT | Abbreviations (R2 asked "etc.") | HAI (HIL-based Augmented ICS) at §5.1 and KL (Kullback-Leibler) at eq. (3) still unexpanded. | Expand at first use. |
| 24 | OPT | Limitations (§7 line 1200) | The two localization limits live in the localization paragraph, which is fine, but the Limitations paragraph does not mention that localization is episode-level over all attacks (not the difficult subset) and untested on WADI; nor that USAD/TranAD are single-run on WADI and SWaT (disclosed in captions only). | One sentence each in Limitations. |
| 25 | OPT | Table 3 F1 columns | Never discussed; on WADI LinRes beats LatAD on F1 in all three subsets; WADI F1 printed to 2 decimals, HAI/SWaT to 3. (Review-3 item 15.) | One sentence plus unified decimals. |

No unsupported "significant" or "best" was found: every "significant" is tied to a reported CI or P (HAI, SWaT difficult; HAI, SWaT double-hard; WADI per-detector; SWaT/HAI top-1 localization), and every "best" is scoped to HAI and SWaT All-subset AUROC. The only untested word doing significance work is "the SWaT margin holds" against the size-weighted baseline (item 4).

### A.5 Summary judgment

- Density-vs-reconstruction: convincing at the head level and consistent across abstract, §4.2, Table 2, Table 6, and §7 once items 10, 11, and 13 are applied. No predictive law is claimed; Figure A1 is a mechanism illustration and says so.
- Table 7: a reviewer accepts top-3 on SWaT and HAI as a validated triage shortlist. The exposed flank is HAI top-1 against the size-weighted baseline (numbers absent, item 4) and the untested top-3 (item 5). Items 1 to 3 are trivial but visible.
- The single biggest carry-over defect is item 18 (headline model missing from the six-statistic robustness check), because it is the paper's answer to the most obvious methodological attack.

---

## PART B. Letter audit

Legend: Loc = cited section/table/figure exists and says what the letter claims. Num = every number matches the paper. Phantom/Over = a described change not in the paper, or a result overstated. Must-fix items are marked **MUST**.

### B.1 Response to the Academic Editor (`letter_editor.html`)

Tone: thankful, plain, non-apologetic. Apartsin correct (2x). No forbidden punctuation or words. All five direct requests, the forwarded R2 items, and the point-by-point request are answered.

| Editor point | Cited location correct? | Number correct? | Phantom / overclaim? | Issue |
|---|---|---|---|---|
| Summary: GDN in Tables 3, 4, Figure 3, §2.4, 5.5, 7 | yes | 0.481 HAI (§2.4, §7) yes | none | |
| Summary: WADI difficult 43 to 30 windows; HAI 0.845 / SWaT 0.837 / WADI 0.771; rescaled analyzer channel; canonical SWaT (§5.1) | yes | yes (30 windows in Table 3 header) | none | "43" is history, not in the paper; acceptable. |
| Summary: A8/A9/A10 remeasured (Table 1, §7, Appendix C) | yes | yes | none | |
| Summary: benchmark-scale discussion (§1, §2.5, §7) | yes | n/a | none | |
| Summary: change-marked manuscript | yes | n/a | none | **Gap, not phantom**: the summary does not announce the three additions made after the letters were drafted: Table 7 (subsystem localization), the density-head vs deployed-detector comparison in §7, and Figure A1. The editor will see them highlighted in green with no letter sentence pointing at them. Add one bullet: "New validated result: a subsystem-localization evaluation (Table 7, §7); the head-level density-versus-reconstruction comparison in §7 is sharpened and illustrated on synthetic data (Figure A1)." |
| Req 1 (A3/A9/A10): A8 renumbering; responsibility screen blind on a 30-to-2-percent Bayes-error control; **"it also proved to be a variance-floor artifact of the estimator on SKAB's 400-window sample (Appendix C, Table C1)"**; A9 3.4 to 3.7 decades, +0.06, +0.072; A10 three scorers | Table 1, §7, Appendix C, Table C1, Table A2: yes, **except the SKAB variance-floor sentence** | yes | **MUST (phantom)**: the paper's SKAB variance-floor explanation is now struck (line 2937, `rev-del`: "Its responsibility entropy of 0.29 is a variance-floor artifact ... log(0.05) ... drops the entropy to 0.045 to 0.048"). Appendix C now says only that SKAB has no close-distinct regime pairs. The letter cites a finding the revised paper no longer contains. | Either restore one clause in Appendix C ("its earlier apparent responsibility signal was a variance-floor artifact of the log(0.05) component floor on 400 windows") or reword the letter to what the paper says: "and SKAB, a single-loop 400-window rig, has no close-distinct regime pairs at all (Appendix C, Table C1)". Same fix in the R1 letter (below). |
| Req 2 (nearest-component): Table 6 0.603 / 0.797 / 0.723; A5 in Table 1; §4.3 ii | yes | yes | none | Declines the imbalance sweep plainly; second-round risk, not a defect. |
| Req 3 (cost): Appendix B, Table B1 with host memory; 3.2 / 16 / 3.5 s; 47 to 72x; 0.7 to 13 ms; 17 to 32 ms; 0.6 to 14.6 ms CPU; 20 to 29 MB; 3.5 to 3.9 GB | yes (Host RAM GB column now exists) | yes | none | |
| Req 4 (time-aware test): §5.4, §6, Table 4, Figure 3; HAI +0.088 [0.042, 0.157]; SWaT +0.055 [0.012, 0.104] P = 0.007; WADI significant vs AE, USAD, TranAD, IF, GDN at P < 0.05; LinRes +0.020 [-0.161, 0.205] P = 0.46, 8 episodes | yes | yes | none | Under-sell from review 3 is fixed. |
| Req 5 (ablations): Table 5 +0.200 / +0.141 / +0.017; Table 6; Table A1; Table A2 | yes | yes | none | "essentially nothing on the artifact-free WADI difficult subset (+0.017)" matches. The letter does not say "marginals suffice", so paper item A.19 does not leak into the letter. |
| Forwarded R2 items: abstract, §1, §2.1 to 2.5, Figures 1 and 2, z / std / u, nine equations, DOCX indentation | yes | nine equations confirmed | none | Indentation is a DOCX-stage property; verify in `paper_mdpi.docx` before upload. |
| Point-by-point | yes | n/a | none | |

### B.2 Response to Reviewer 1 (`letter_reviewer1.html`)

Tone: thankful, plain, non-apologetic. Apartsin correct. No forbidden punctuation or words. All five comments plus the English/figures/conclusions line are answered, each quoting the comment.

| R1 point | Cited location correct? | Number correct? | Phantom / overclaim? | Issue |
|---|---|---|---|---|
| C1 (A3/A9/A10): A8 renumbering; Table 1 blocks; 30-to-2-percent control; **"on SKAB the signal proved to be a variance-floor artifact of the estimator on a 400-window sample (Appendix C, Table C1)"**; 80 percent vs 13 percent isolated-missed; A9 3.4 to 3.7 decades, 600-sample window +0.06, +0.072; A10 three named scorers, positive control, order-shuffle; §2.5 | yes except the SKAB sentence | yes (Table C1 WADI 0.80 / 0.13) | **MUST (phantom)**: same struck SKAB variance-floor finding as the editor letter. | Apply the same fix as B.1 Req 1. |
| C2 (nearest-component): Table 6 numbers; A5 quote | yes | yes | none | The Table 1 A5 quote is a paraphrase inside quotation marks ("never the weighted mixture" vs the paper's "never the pi-weighted mixture"); make it verbatim or drop the quotation marks. |
| C3 (cost): 50 / 195 / 38 s; 17 to 32 ms, up to 52 ms at batch 256; global 3.2 / 16 / 3.5 s, 0.7 to 13 ms; TranAD 231 / 898 / 163 s; 20 to 29 MB; 3.5 to 3.9 GB; §4.4 architecture; §7 pointer | yes | yes | none | R1 asked about the community stack's cost; the letter gives its absolute times but the "47 to 72 times faster" is the global model's. Add the community ratio (about 4.3 to 4.6x faster to train than TranAD) so the reviewer is not left to compute it (paper item A.14 covers the same in the text). |
| C4 (time-aware test): §5.4 definition; HAI 167 / 26; SWaT 85 / 23; WADI per-detector P (AE 0.001, USAD 0.000, TranAD 0.0005, IF 0.000, GDN 0.000); LinRes +0.020 [-0.161, 0.205] P = 0.46; double-hard HAI +0.084 [0.019, 0.189] P = 0.0005; SWaT +0.085 [0.030, 0.146] P = 0.0005; WADI 7 episodes | yes | yes | none | |
| C5 (representation): Table 5 +0.200 / +0.141 / +0.017; Table A2 +0.072 HAI, +0.026 SWaT, immaterial WADI; Table 6; Table A1; no raw per-timestep variant | yes | yes | none | Plain about the unrun sequential variant. Optional: the new §7 head-level evidence (the single density head beats USAD and TranAD on all three difficult subsets) is the most direct answer to "cross-channel density or representation?" and is not cited here; one sentence pointing at §7 and Table 6 would strengthen the response. Table A2 uses a reduced configuration (caption); optional to say so. |
| English / figures / conclusions: Figures 1 to 3; Tables 3 to 6 and A1; §8 statement | yes | n/a | none | "Tables 3 to 6 and A1 were regenerated" is accurate; the letter does not list Table 7 or Figure A1 as new figures/tables. Optional: add "and a new Table 7 and Figure A1". |

### B.3 Response to Reviewer 2 (`letter_reviewer2.html`)

Tone: thankful, plain, non-apologetic. Apartsin correct. No forbidden punctuation or words. All twelve comments plus the closing line are answered, each quoting the comment.

| R2 point | Cited location correct? | Number correct? | Phantom / overclaim? | Issue |
|---|---|---|---|---|
| C1 abbreviations: LatAD in abstract; CPS / IIoT / SCADA in §1; VaDE, GMM, NLL, HC, BIC, AUROC, F1, ICS, SOTA, KDE in main text | yes (lines 82, 88, 92, 183, 291, 308, 188, 336, 192, 169, 235, 239, 536) | n/a | none | HAI and KL remain unexpanded in the paper (A.23); the reviewer wrote "etc.". |
| C2 abstract rebalanced; quoted sentence; 0.948 / 0.938 | yes, quote verbatim (line 82) | yes | none | |
| C3 first-paragraph sentence struck; CPS opening; contributions later | yes (line 88 strike and opening; line 179) | n/a | none | |
| C4 "Unsupervised anomaly detection for CPS telemetry is often framed as ..." | yes, verbatim (line 160) | n/a | none | |
| C5 two-obstacles paragraph | yes, verbatim (line 175) | n/a | none | |
| C6 Related Work: definition moved to §1; three self-descriptions struck; §2.3 rewritten with Deep SVDD, OmniAnomaly, THOC; §2.5 replaced by "Benchmark scale" with pointer to §5.1 | yes (lines 90, 203, 206, 228, 235, 257 to 259) | n/a | none | |
| C7 Figure 1 system model; **"Section 7 (new Table 7) reports that a three-community shortlist ranked by train-normal p-value contains the attacked subsystem in 68 to 79 percent of HAI and SWaT attacks, against 49 to 56 percent for a random shortlist of the same size"** | yes (Table 7 top-3 0.79 / 0.68; random 0.49 / 0.56; §7 line 1162) | yes | none | Matches the paper's new result. Two optional precisions: "attack episodes with a known target" rather than "attacks" (A.9), and the Figure 1 arrow the letter describes still reads "alerts + diagnosis" in the SVG (A.1), which the paper now disavows; fix the figure, not the letter. |
| C8 MIIM vs LatAD: §3 checklist sentence; "Section 4 states the relationship explicitly, immediately after the workflow figure" | checklist yes (line 272); the quoted sentence is the second paragraph after Figure 2 (line 454), after the "reported detector realizes A1 to A7" paragraph | n/a | minor location imprecision | "in the paragraphs following the workflow figure". Trivial. |
| C9 §4 headings; Figure 2 | yes (lines 458, 530, 601; Figure 2 at 357 to 443, referenced at 348) | n/a | none | |
| C10 separate VaDE per community; Appendix B / Table B1 with parameter count, training time, latency, peak GPU memory, host memory; 3.2 / 16 / 3.5 s; 47 to 72x; 0.7 to 13 ms; 17 to 32 ms; 3.5 to 3.9 GB | yes (§4.4 line 622; Table B1 has all five columns) | yes | none | |
| C11 z / std / u renaming; Tables 3 and 4; §5.3, §5.5, §6; §4.4 and Table A1 "standardized units" | yes (line 721 parenthetical; "trivial max|u|" in Tables 3 and 4; line 659; Table A1 caption) | n/a | none | |
| C12 equations (1) to (9); indentation at DOCX build | nine equations confirmed | n/a | indentation not verifiable in HTML | Verify `paper_mdpi.docx`. |
| Results / English / figures: Table 5 new; GDN in Tables 3, 4, Figure 3; Figures 1 and 2 added; Figure 3 renumbered | yes | n/a | none | Optional: mention Table 7 and Figure A1 here as well. |

### B.4 Letter-level summary

- **One must-fix, in two letters**: the editor letter (Req 1) and the R1 letter (C1) both cite the SKAB "variance-floor artifact ... (Appendix C, Table C1)" finding, which the revised paper strikes at line 2937. Restore a one-clause version in Appendix C or reword both letters to the paper's current statement (no close-distinct regime pairs on SKAB).
- **No number in any letter disagrees with the paper**, including the new Table 7 numbers in the R2 letter (C7) and every cost, CI, P-value, and ablation figure.
- **No phantom edit other than the SKAB sentence.** The Host RAM column, the WADI significance statement, the control wording, and the R2 C8 location from review 3 are all fixed.
- **Coverage gap (optional but recommended)**: none of the three letters announces Table 7, the sharpened density-vs-reconstruction comparison in §7, or Figure A1 as additions, except the single Table 7 sentence in R2 C7. Add one bullet to the editor letter's "Summary of the principal changes" and one clause to R1 C5 (density head beats USAD and TranAD on all three difficult subsets, §7 and Table 6), so the reviewers are told what the green passages contain.
- **Two declined requests stated plainly** (A5 imbalance sweep; raw sequential-input variant). Second-round risk, not defects.
- Tone, spelling, and punctuation pass in all three letters.

---

## Priority order for the fix pass

Must-fix (paper): A.1 Figure 1 "diagnosis" label; A.2 HAI 39 vs 40 percent; A.3 duplicated top-3 sentence and leftover "also"; A.10 "right default for the deployed reconstruction detectors" wording.

Must-fix (letters): B.1/B.2 SKAB variance-floor phantom (restore in Appendix C or reword both letters).

Strong (paper): A.4 size-weighted baseline numbers; A.5 name the test and give top-3 P; A.6 episode selection rule for Table 7; A.7 localization on difficult-subset episodes or an explicit scope sentence; A.11 "matches GDN"; A.12 source the 93rd-percentile / 27 percent / 70 percent numbers; A.13 universal "reconstruction residual" in §1 and §8; A.18 headline model on the six-statistic split; A.19 "marginals already suffice".

Recommended (letters): editor summary bullet for Table 7, §7 reframe, Figure A1; R1 C5 pointer to the head-level 3/3 result; R1 C3 community-stack cost ratio.

Optional: A.8, A.9, A.14 to A.17, A.20 to A.25; B.3 C8 wording; R1 C2 verbatim quote.
