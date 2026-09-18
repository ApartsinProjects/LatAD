# Review 3: line-by-line reviewer critique of IoT2.html and audit of the three response letters

Scope: `poc/paper/IoT2.html` (1396 lines, read in full) and `review_round_1/letter_editor.html`, `letter_reviewer1.html`, `letter_reviewer2.html` (read in full), checked against `Review1.txt`, `Review2.txt`, `Editor review.txt`.
Reference numbers used for checking: HAI 0.845 (sig, P about 0), SWaT 0.837 (sig, P = 0.007), WADI 0.771 (sig vs every learned detector, ties LinRes P = 0.46), GDN weaker on all three, A8 precondition present / not exercised, A9 present, A10 measured absent, MIIM 7 realized + 3 specified. All of these are stated correctly in the paper body and tables. Change-marking is intentional and not flagged.

Mechanical checks run (all pass unless listed in Part A): no em-dash, double hyphen, "honest*", "we acknowledge", "admittedly", "seems to / appears to / arguably" in paper or letters; Apartsin spelled correctly in all three letters; Figure 3 bar heights reproduce every Table 3 difficult-subset value under the stated transform; nine display equations numbered (1) to (9); `paper_mdpi.docx` exists (2026-09-18).

---

## PART A. Reviewer findings, ranked

Severity: **MUST** = a reviewer or copy-editor will catch it and it is a factual or internal-consistency defect. **STRONG** = an argument gap a skeptical reviewer will attack; fixable with existing persisted scores or wording. **OPT** = optional strengthening.

### A.1 Must-fix defects

| # | Sev | Location | Objection | Fix |
|---|---|---|---|---|
| 1 | MUST | Abstract, line 82: "ahead of every learned detector on WADI (0.771), **where it ties the linear baseline on the full set**" | Wrong subset. On the full set the linear baseline is *ahead* (0.834 vs 0.827; Table 3 bolds LinRes in the WADI All column, and §6 says "marginally ahead"). The tie (P = 0.46) is on the *difficult* subset. The abstract contradicts Table 3 on the first page. | "ahead of every learned detector on WADI (0.771), where it ties the linear baseline (P = 0.46); on the full set the linear baseline is marginally ahead." |
| 2 | MUST | §5.4 Significance testing, lines 750-751: block length "⌈W/stride⌉ + 1 windows (3 on WADI, 4 on HAI and SWaT)" | With W = 60 and stride 30 "unified across all three datasets" (§5.2), the formula gives 3 everywhere. Either the formula, the numbers, or the "unified" claim is wrong. | State the true block lengths and why they differ (if HAI/SWaT used 4 because of the native-vs-downsampled horizon, say so), or correct to 3. |
| 3 | MUST | Appendix B text (line 1270), Table B1 caption (line 1276), §7 (line 1159), and all three letters | Host memory "about 3.5 to 3.9 GB" is attributed to Table B1, but Table B1 has columns Params / Train s / GPU ms/win / Peak GPU MB / CPU-edge ms/win only. No host-memory column exists. The editor, R1 and R2 letters each say Table B1 reports host memory. | Add a `Host GB (W/H/S)` column per method to Table B1 (the numbers exist since the text quotes them), or drop the "(Table B1)" attribution and correct the three letters. |
| 4 | MUST | Table 5 caption, line 1009: "Bold marks the cross-channel gains that **survive the significance test**" | No significance test of the Table 5 differences (+0.200, +0.141, +0.017) is reported anywhere. The body text (line 1005) means "the two datasets where LatAD's margin over the strongest baseline is significant", a different statement. As written the caption asserts an untested significance. | Either run the paired episode-block bootstrap on the Table 5 difference (persisted scores, no retraining) and report CI/P, or reword: "Bold marks the two datasets on which LatAD's difficult-subset margin over the strongest baseline is significant (§6)." |
| 5 | MUST | §7 "Alternative realizations", lines 1138-1140 | Two adjacent sentences contradict: "the community construction and the cohesion-weighted Higher Criticism combiner **were developed on these three benchmarks**" then "settings **were fixed before the WADI and SWaT evaluations** reported here, so those two results are out-of-sample confirmations." A reviewer reads the first as design-on-test-set. | State the actual history once: on which dataset(s) the construction and combiner were developed (HAI? synthetic?), that they were frozen, and that WADI and SWaT were scored afterwards. Delete the "developed on these three benchmarks" clause if it is not literally true. |
| 6 | MUST (borderline STRONG) | §6 "Robustness to a stronger difficulty definition", lines 881-889 | The six-statistic split is reported for **HAI only** and for **LatAD (global density) only** (0.675 vs 0.622), not for the headline regime-community model and not for WADI or SWaT, with no CI. This paragraph is the paper's main defense of the difficulty split against "the authors chose a favorable statistic"; as written it covers one dataset and the ablation model. A reviewer will ask why the headline model is missing exactly here. | Report the headline model on the six-statistic split for all three datasets, with the same bootstrap (scores are persisted). If the headline was not run, say so and run it; the cost is minutes. |

### A.2 Weakest links in the argument (a skeptical reviewer's attack list)

| # | Sev | Location | Objection | Fix |
|---|---|---|---|---|
| 7 | STRONG | Title thesis vs Table 6; §7 heading "Why reconstruction fails and density wins" (line 1033) | The head-level evidence supports "density beats reconstruction" on **HAI only**. Table 6: on SWaT the whitened reconstruction residual alone (0.835) equals the full headline detector (0.837); on WADI the residual alone (0.740) beats the entire single-latent model (0.634) and every base head. On two of three datasets the paper's own reconstruction head is the strongest single head. The paper discloses this in the Table 6 prose, but the §7 heading and the abstract's "scores by probability rather than reconstruction" read as a universal mechanism. Second, on WADI the gate switched **off** a head scoring 0.740 in favour of a base scoring 0.634; the train-split q95 ratio (3.41) did not predict test AUROC, so "the gate recovers this pattern from train-normal alone" (line 588) is true of the ratio but not of the outcome. | (a) Retitle: "Why the reconstruction-based deep detectors fail and latent density wins on joint-structure faults". (b) State the claim precisely: the deep detectors' *unwhitened, regime-agnostic* residual collapses on HAI; a *regime-conditioned, whitened* residual is informative, and on SWaT/WADI the design's gain comes from community factorization and calibration rather than from demoting reconstruction. (c) Add one sentence on the WADI gate trade-off: the gate is a false-alarm guard on held-out normal, it costs difficult-subset AUROC on WADI relative to residual-on (0.740 vs 0.634), and the community factorization recovers the lead (0.771). |
| 8 | STRONG | WADI mechanism, lines 1005 vs 1054-1058 | Two incompatible explanations for WADI. Line 1005: "WADI's difficult anomalies are single-channel or linear, so **per-channel marginals already suffice**". Line 1054: "a coordinated drift confined to a small correlated subsystem is improbable within that subsystem's own density yet unremarkable in the whole-plant density". And "suffice" is contradicted by the numbers: product of marginals 0.639 and trivial rule 0.601 vs 0.771 / 0.795. | Replace "suffice" with "capture as much as the cross-channel latent does (0.639 vs 0.656)". Reconcile: on WADI the cross-channel *latent* adds nothing over marginals, but estimating the marginals *within each community and combining by HC* is what lifts 0.634 to 0.795; the gain is in the factorization/aggregation, not in cross-channel modeling. Say that in one place. |
| 9 | STRONG | §7 deployment paragraph (lines 1157-1159), Appendix B (1265-1267) | The "47 to 72 times faster than TranAD" figure is for LatAD-**global**, the base ablation. The headline regime-community trains in 50 / 195 / 38 s, about 4.3 to 4.6 times faster than TranAD, and its per-window GPU latency (17 to 32 ms) is 9 to 16 times **slower** than TranAD's 2 ms. The paragraph foregrounds the ablation's speed under the headline's name. "Gateway-class real-time budgets" is never defined. | Quote the headline's ratios beside the global's. Define the budget: one window per 30-sample stride (30 s on HAI, 300 s on WADI/SWaT), so 52 ms is under 0.2 percent duty on any of them. |
| 10 | STRONG | Table 3, USAD and TranAD rows on WADI and SWaT | No ±SD in any of the six cells on WADI and SWaT, but ±0.018 / ±0.010 on HAI. The caption says SD is omitted where it rounds to 0.000; five-seed SD rounding to 0.000 in all six cells for two adversarially trained models is implausible and a reviewer will ask whether five seeds were run there. Figure 3 caption is consistent (whiskers only "where it exceeds zero") but the question stands. | State explicitly under Table 3: "USAD and TranAD on WADI and SWaT: five seeds, SD < 0.0005" if true; if a single seed was used, say so. |
| 11 | STRONG | Localization claim: Figure 1, Figure 2, §4.4 (line 656), §7 (line 1148), Conclusion | "the most-surprised subsystem" is sold as an operator output in two figures and three sections but is **never evaluated**. WADI and SWaT attack lists name the attacked sensor/actuator, so a hit-rate (attacked channel inside the top-1 / top-3 community) is cheap and would be the single most IIoT-relevant number in the paper. | Add a short localization evaluation (top-k community hit-rate on WADI and SWaT attacks with a named target), or state in Limitations that localization is a design property not evaluated here. |
| 12 | STRONG | Reproducibility (§4.1, §5.2) | Given: K, latent dim, M = 80 (capped at 1/10 of train windows), PCA 30, Ledoit-Wolf, W = 60, stride 30, gate 1.5, communities 3 to 25 by average linkage, cohesion weight, HC. Missing: encoder/decoder architecture (layers, widths, activations), optimizer, learning rates (main and the "slower" mixture rate), epochs for pretraining and joint training, warm-up schedule length, variance-floor value (log 0.05 appears only in Appendix C), DAGMM penalty weight, batch size, per-community K and latent-dim scaling rule. Code on Zenodo mitigates, but MDPI reviewers expect a hyperparameter table. | Add a compact hyperparameter table (Appendix A or B), including the per-community scaling rule for K and latent dim. |
| 13 | STRONG | Gate threshold, §4.3 iii, line 586 | The cut-off 1.5 sits between the observed ratios 1.22 (on) and 3.41 (off). Nothing says it was fixed before evaluation. | State that 1.5 was fixed a priori, or add "any threshold in [1.3, 3.3] gives the same gating on all three datasets." |
| 14 | OPT | §3, lines 339-343: "This ordering tracks where the design pays off" | Silhouette WADI 0.06 < HAI 0.08 < SWaT 0.29, but factorization gains are WADI +0.137, HAI +0.034, SWaT +0.033: HAI and SWaT gain identically despite a 3.6x silhouette difference. "Tracks" rests on one point. | "The factorization gains most on WADI, the least-separated plant." Drop "tracks". |
| 15 | OPT | Table 3 F1 columns | F1 is never discussed, and on WADI the linear baseline beats LatAD on F1 in all three subsets (0.68 / 0.73 / 0.53 vs 0.57 / 0.58 / 0.42); on SWaT Easy LinRes F1 0.976 vs 0.920. A reviewer will notice the secondary metric is unmentioned exactly where it is unfavourable. Also WADI F1 is printed to 2 decimals, HAI and SWaT to 3. | One sentence: "At the oracle threshold the linear baseline's F1 is higher on WADI, consistent with the AUROC tie there." Unify decimals. |
| 16 | OPT | §5.2 "The window size and stride are unified" | W = 60 samples is 60 s on HAI (native) and 600 s on WADI/SWaT (10x downsampled). The A9 result "a 600-sample window adds +0.06 on HAI" is exactly equalizing HAI's horizon to WADI/SWaT's. | State horizons in seconds and note the asymmetry in one clause. |
| 17 | OPT | Table 2 vs Table 6 | Table 2 (single model: WADI latent 0.621, HAI 0.760) and Table 6 (five seeds: 0.603, 0.797) are different runs of the same heads; Table 2 is now redundant. Its HAI joint NLL equals the residual exactly (0.689), which means the unstandardized residual term dominates the sum, a point the text does not make. | Drop Table 2 (Table 6 carries the argument), or keep it and add "the joint score is scale-dominated by the residual term, which is why standardized heads are used in §4.3." |
| 18 | OPT | §2.4 / §5.5 GDN | Single seed, AUROC only, F1 "not computed". Disclosed, but with scores persisted F1 is free and five seeds are cheap. | Add F1 or state why not. |
| 19 | OPT | Limitations, line 1167-1170 | "a different percentile would redraw the boundary" is admitted but not shown. | Add a two-line sensitivity (95th and 99.9th percentile) for the headline model on HAI; the ranking either holds or it does not. |
| 20 | OPT | Editor request 2 (A5 sweep) | The paper offers Table 6 head isolation only; the editor asked explicitly for an imbalance sweep and the letters decline it. Risk of a second-round request. | A cheap sweep exists: subsample one discovered regime in train-normal to 1 / 0.1 / 0.01 percent and report the false-alarm rate on that regime's held-out windows under mixture NLL vs nearest-component NLL. One afternoon of agent time; no new data. |

### A.3 Formatting and wording (low)

| # | Sev | Location | Objection | Fix |
|---|---|---|---|---|
| 21 | OPT | References [53]-[59] | First-appearance order is 53 (line 1093), 55, 56, 57 (1135-1136), 58 (1155), 59 (1302), **54 (1329)**. SKAB [54] is first cited after [59]. MDPI numbers by first appearance. Note that "SKAB" is named without a citation at line 1302 (after Cranfield [59] on the same line). | Renumber: SKAB becomes [59] and Cranfield [54] if the Appendix C sentence order stays; or cite SKAB at its first mention in Appendix C and renumber accordingly. |
| 22 | OPT | Figure 2 SVG vs caption | The whitened-residual block carries no `[A7]` tag in the drawing, but the caption and the footnote line say it is tagged A7. | Add the tag to the block. |
| 23 | OPT | §7 lines 1041 and 1078: "the weakest **base** head (0.695)" | Table 6 defines *base* = density + nearest and calls the residual "the dropped term". | "the weakest head". |
| 24 | OPT | Abbreviations (R2 asked for "etc.") | HAI (HIL-based Augmented ICS), KL (Kullback-Leibler) in eq. (3), SMD / MSL / SMAP / PSM in §2.5 are never expanded. | Expand HAI at §5.1, KL at eq. (3); the dataset names can stay as names. |
| 25 | OPT | Table 1 caption vs A8 row wording | Caption says "A8's precondition present but its heavy overlap not exercised"; A8 row says overlap "is absent under valid observation-space metrics". "Not exercised" and "absent" are different statuses. | Use "absent" in both places (that is the measured finding). |

### A.4 Contributions, honesty, and what is missing (summary judgment)

- **Contributions stated and delivered.** The three numbered contributions (MIIM list, LatAD, fair-protocol evaluation) are delivered. Contribution (1) is weakened by the paper's own reframing: three of ten assumptions are unrealized, and the text says so consistently; a reviewer may still ask why the list is ten rather than seven. The paper's answer (a checklist a dataset realizes to varying degrees) is adequate.
- **Difficulty split defended?** Partly. The split is a fixed, pre-specified univariate rule with a train-normal threshold, independent of the detector, plus two harder splits (six-statistic; double-hard) and the trivial rule's drop on Difficult. The defense is undercut by finding 6 (six-statistic split shown for HAI and the ablation only) and finding 19 (no percentile sensitivity).
- **Honesty.** WADI is owned correctly in the body: "significantly ahead of every learned detector ... ties only the linear baseline" (§6, §7, Conclusion). No apologetic tone anywhere. Overclaims: finding 1 (abstract subset error), finding 4 (Table 5 "survive the significance test"), finding 7 (§7 heading), finding 14 ("tracks"). Under-selling: none in the paper; the editor letter under-sells WADI (Part B).
- **Missing.** Localization evaluation (finding 11); headline model on the six-statistic split (finding 6); hyperparameter table (finding 12); a TPR at 5 percent false-alarm budget table for all methods (the number appears once, for LatAD only, in the coverage paragraph at line 1108, and is the operator-facing metric for the IIoT framing).

---

## PART B. Letter audit

Legend: Loc = cited section/table/figure exists and says what the letter claims. Num = every number matches the paper. Phantom/Over = a described change that is not in the paper, or a result overstated.

### B.1 Response to the Academic Editor (`letter_editor.html`)

Tone: thankful, plain, non-apologetic. Apartsin correct (2x). No em-dash, double hyphen, or "honestly". All five editor requests plus the forwarded R2 items and the point-by-point request are answered.

| Editor point | Loc | Num | Phantom / Over | Issue |
|---|---|---|---|---|
| Summary: GDN in Tables 3, 4, Figure 3, §2.4, 5.5, 7 | OK | OK (0.481 HAI in §2.4 and §7) | none | |
| Summary: WADI difficult 43 to 30 windows; HAI 0.845 / SWaT 0.837 / WADI 0.771 | OK | OK (30 windows, Table 3 header) | none | "43" is history, not in the paper; acceptable. |
| Summary: A8/A9/A10 remeasured (Table 1, §7, App C) | OK | OK | none | |
| Summary: benchmark-scale discussion (§1, §2.5, §7) | OK | n/a | none | |
| Req 1 (A3/A9/A10): renumbered A8; SKAB variance-floor artifact; 50 percent synthetic control; 80-component defense; A9 3.4 to 3.7 decades, +0.06, +0.072; A10 three scorers | OK (Table 1, §7, App C, Table C1, Table A2) | OK | none | Note: the letter's "50 percent-overlap positive control" (responsibility-screen control) and the paper's "30-percent-overlap positive control" (Table C1) are two different controls; the paper's Appendix C describes the responsibility test on "a synthetic pair whose Bayes error runs from 30 to 2 percent", not "50 percent". Minor mismatch; align the letter to "a synthetic overlap positive control". |
| Req 2 (nearest-component support): Table 6 0.603 / 0.797 / 0.723; A5 in Table 1; §4.3 ii | OK | OK | none | Declines the sweep plainly. Risk, not defect (see A.20). |
| Req 3 (cost): App B, Table B1; 3.2 / 16 / 3.5 s; 47 to 72x; 0.7 to 13 ms; 17 to 32 ms; 0.6 to 14.6 ms CPU; 20 to 29 MB; 3.5 to 3.9 GB host | OK except host memory | OK | **PHANTOM: Table B1 has no host-memory column** (Part A #3). Also "parameter count ... peak GPU memory, and host memory" lists five columns; only four exist. | Fix Table B1 or the sentence. |
| Req 4 (time-aware test): §5.4, §6, Table 4, Fig 3; HAI +0.088 [0.042, 0.157]; SWaT +0.055 [0.012, 0.104] P = 0.007; WADI +0.020 [-0.161, 0.205] P = 0.46, 8 episodes | OK | OK | none | **Under-sells WADI.** The editor letter says only "on WADI the lead over the linear baseline is numerical" and omits the paper's (and R1 letter's) statement that LatAD is significantly ahead of every learned detector on WADI (P < 0.05 each). Add that sentence so the editor sees the same result as R1. |
| Req 5 (ablations): Table 5 +0.200 / +0.141 / +0.017; Table 6; Table A1; Table A2 | OK | OK | none | |
| Forwarded R2 items: abstract, §1, §2.1-2.5, Figures 1-2, z / std / u, nine equations, DOCX indentation | OK | nine equations confirmed | Indentation is "applied at the DOCX build stage": the HTML cannot show it; verify the DOCX before submission. | |
| Point-by-point | OK | n/a | none | |
| Abbreviations | | | | GDN, AUROC, NLL, CI, GPU, SKAB, VaDE are not expanded in the letter. Optional: expand at first use per letter (the user's checklist asks for it). |

### B.2 Response to Reviewer 1 (`letter_reviewer1.html`)

Tone: thankful, plain, non-apologetic. Apartsin correct. No forbidden punctuation. All five comments plus the English/figures/conclusions line are answered, each quoting the comment.

| R1 point | Loc | Num | Phantom / Over | Issue |
|---|---|---|---|---|
| C1 (A3/A9/A10): A8 renumbering; Table 1 blocks; App C Table C1; 80 percent vs 13 percent isolated-missed; A9 3.4-3.7 decades, 600-sample window +0.06, +0.072; A10 three named scorers, positive control, order-shuffle; §2.5 scale | OK (Table C1 WADI 0.80 / 0.13; §7 A9-A10 paragraph) | OK | none | Same "50 percent-overlap" vs paper's "30 to 2 percent Bayes error" control wording as the editor letter; align. |
| C2 (nearest-component): Table 6 numbers; A5 quote from Table 1 | OK | OK; the Table 1 A5 quote is paraphrased ("score against the nearest component, never the weighted mixture" vs paper's "never the pi-weighted mixture") | none | Acceptable paraphrase inside quotation marks; consider making it verbatim. Declines the sweep plainly (risk A.20). |
| C3 (cost): 50 / 195 / 38 s; 17 to 32 ms, up to 52 ms at batch 256; global 3.2 / 16 / 3.5 s, 0.7 to 13 ms; TranAD 231 / 898 / 163 s; 20 to 29 MB; 3.5 to 3.9 GB host; §4.4 architecture; §7 pointer | OK except host memory | OK | **PHANTOM: host memory not in Table B1** (Part A #3). | Also the letter's "47 to 72 times faster" is the global model's ratio; R1 asked about the *regime-community* cost. The letter gives community training times, which is fair, but should also give its ratio (about 4.3 to 4.6x faster than TranAD) so the reviewer is not left to compute it. |
| C4 (time-aware test): §5.4 definition; HAI 167 windows / 26 episodes; SWaT 85 / 23; WADI per-detector P-values (AE 0.001, USAD 0.000, TranAD 0.0005, IF 0.000, GDN 0.000); LinRes +0.020 [-0.161, 0.205] P = 0.46; double-hard HAI +0.084 [0.019, 0.189] P = 0.0005; SWaT +0.085 [0.030, 0.146] P = 0.0005; WADI 7 episodes | OK | OK | none | Correctly owns WADI. Note Part A #2 (block length 3 vs 4) is a paper defect the reviewer may raise when reading §5.4 that this letter points to. |
| C5 (representation): Table 5 +0.200 / +0.141 / +0.017; Table A2 +0.072 HAI, +0.026 SWaT, immaterial WADI; Table 6; Table A1; no raw per-timestep variant | OK | OK | none | Plain about the unrun sequential variant. Table A2 uses a *reduced* global configuration (caption says so); the letter does not mention this. Optional: add "at a reduced configuration" so the reviewer is not surprised by the lower absolutes. |
| English / figures / conclusions | OK (Figures 1-3; Tables 3-6, A1; §8) | n/a | "a language pass over the whole manuscript is applied to the submitted version": unverifiable from the HTML | Ensure the DOCX reflects it. |

### B.3 Response to Reviewer 2 (`letter_reviewer2.html`)

Tone: thankful, plain, non-apologetic. Apartsin correct. No forbidden punctuation. All twelve comments plus the closing line are answered, each quoting the comment.

| R2 point | Loc | Num | Phantom / Over | Issue |
|---|---|---|---|---|
| C1 abbreviations: LatAD in abstract; CPS / IIoT / SCADA in §1; VaDE, GMM, NLL, HC, BIC, AUROC, F1, ICS, SOTA, KDE in main text | OK (lines 82, 88, 92, 183, 291, 308, 188, 336, 192, 169, 235, 239, 536) | n/a | none | HAI and KL remain unexpanded in the paper (Part A #24); the reviewer wrote "etc.". |
| C2 abstract rebalanced; quoted sentence | OK, quote verbatim (line 82) | 0.948 / 0.938 OK | none | |
| C3 first-paragraph sentence struck; CPS background; contributions later | OK (line 88 strike; line 88 opening; line 179) | n/a | none | |
| C4 "Unsupervised anomaly detection for CPS telemetry is often framed as ..." | OK verbatim (line 160) | n/a | none | |
| C5 two-obstacles paragraph | OK verbatim (line 175) | n/a | none | |
| C6 Related Work: definition moved; three self-descriptions struck; §2.3 rewritten with Deep SVDD, OmniAnomaly, THOC; §2.5 replaced by Benchmark scale with pointer to §5.1 | OK (lines 90, 203, 206, 228, 235, 257-259) | n/a | none | |
| C7 Figure 1 system model | OK; caption text matches | n/a | none | |
| C8 MIIM vs LatAD: §3 checklist sentence; "Section 4 opens by stating ..." | Checklist OK (line 272). The quoted sentence is the **third** paragraph of §4 (line 454, after the preamble, Figure 2, and the "reported detector realizes A1-A7" paragraph), not the opening. | n/a | Minor location imprecision | "Section 4 states explicitly, immediately after the workflow figure, ..." |
| C9 §4 headings; Figure 2 | OK (lines 458, 530, 601, 357-443, referenced at 348) | n/a | none | |
| C10 separate VaDE per community; cost in App B / Table B1 incl. host memory; 3.2 / 16 / 3.5 s; 47 to 72x; 0.7 to 13 ms; 17 to 32 ms; 3.5 to 3.9 GB | §4.4 OK (line 622) | OK | **PHANTOM: host memory not in Table B1** (Part A #3). Also "GPU memory" that R2 asked for is in the table (peak MB) but the letter's list omits it while listing host memory, the reverse of what the table holds. | Fix the table or the list; mention peak GPU memory explicitly since R2 asked for it. |
| C11 z / std / u renaming; Tables 3-4; §5.3, §5.5, §6; §4.4 and Table A1 "standardized units" | OK (line 721 parenthetical; "trivial max|u|" in Tables 3-4; line 659; line 1220 strike) | n/a | none | |
| C12 equations (1)-(9); indentation at DOCX build | Nine equations confirmed | n/a | Indentation not verifiable in HTML | Verify DOCX before submission. |
| Results / English / figures | OK | n/a | none | |
| Abbreviations in the letter itself | | | | GDN, AUROC, NLL, VaDE, GPU unexpanded; optional. |

### B.4 Letter-level summary

- **No promised change is absent from the paper** except the host-memory column of Table B1 (claimed in all three letters and in the paper's own Appendix B/§7 text). This is the one must-fix in Part B.
- **No number in any letter disagrees with the paper.**
- **One location imprecision** (R2 C8 "Section 4 opens").
- **One control-wording mismatch** (editor and R1 C1: "50 percent-overlap positive control" vs the paper's "30 to 2 percent Bayes error" sweep and "30-percent-overlap" Table C1 control).
- **One under-sell** (editor Req 4 omits WADI's significance over every learned detector).
- **Two declined requests stated plainly** (A5 imbalance sweep; raw sequential-input variant). These are risks for a second round, not defects; A.20 gives a cheap way to close the first.
- Tone, spelling, and punctuation pass in all three letters.

---

## Priority order for the fix pass

1. Abstract WADI subset wording (A #1).
2. Table B1 host-memory column, then re-check the three letters (A #3).
3. Block-length sentence in §5.4 (A #2).
4. Table 5 caption "survive the significance test" (A #4).
5. "developed on these three benchmarks" vs "fixed before" (A #5).
6. Six-statistic split for the headline model on all three datasets (A #6).
7. §7 heading and the WADI gate sentence (A #7), WADI "suffice" (A #8), headline cost ratio (A #9), USAD/TranAD SD note (A #10).
8. Editor letter: add WADI learned-detector significance; align the A8 control wording in editor and R1 letters; fix R2 C8 location.
9. Everything in A.2 marked OPT and A.3, as time allows; the localization hit-rate (A #11) is the highest-value optional item for an IoT journal.
