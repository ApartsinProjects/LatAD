# Round-2 triangulation audit: reviews vs response letters vs IoT2.html

Audited 2026-09-18 against `poc/paper/IoT2.html` (commit e7198da), the three letters in `review_round_1/`,
`letters_summary.md`, and the raw reviews. Report only; nothing was edited.

Legend: **Addressed** = the edit exists in the paper at the cited place. **Letter accurate** = every
section/table/figure/number the letter cites checks out. Issue codes (P1..P15) are detailed below the tables.

## Verdict in one paragraph

All numbers the letters quote (HAI 0.845, SWaT 0.837, WADI 0.771; CIs and P values; Table 6 heads; Table B1
timings; Table 5 gains; Table A2 deltas; timescale decades; the 80-percent / 13-percent over-coverage figures)
match the paper. GDN is in Tables 3 and 4 and in Sections 2.4, 5.5, 7. A8 is reframed (Table 1, Section 7,
Appendix C), A9 is present, A10 is measured absent. Tone is polite, no em-dashes or double hyphens, no
"honestly/frankly", authors spelled Apartsin. The problems are: one phantom edit (P1), one requested
measurement that exists in the raw data but is not in the paper (P2, GPU memory), one unnamed hardware
setting (P3), one rebuttal whose argument contradicts the paper's own Table 1 (P4), one "significantly"
that no reported test supports in either paper or letter (P5), one silently substituted experiment (P6),
Related Work self-descriptions that were added rather than removed (P7), abbreviations still unexpanded
including in the abstract (P8), and several small wrong cross-references (P9, P10, P11, P14).

## Editor

| Editor request | Addressed in paper? Where | Letter accurate? | Issue |
|---|---|---|---|
| Validate unused assumptions (A3, A9, A10) | Yes. Table 1 regouped A1-A7 / A8-A10 with "Realized by" column; §3 checklist sentence; §4 preamble; §7 "Assumption A8", "Trajectory assumptions A9-A10", "Benchmark scale" paragraphs; Appendix C + Table C1. Old A3 "Hard envelopes" (basin head) is indeed now A8 (verified against paper.html Table 1). | Yes, cites match; numbers (3.4-3.7 decades, +0.06 HAI, +0.072 App. A, three scorers) match §7. | P15b (50 vs 30 percent control mismatch inside paper, repeated by letter) |
| Direct support for nearest-component likelihood | Partly. Table 6 isolates the head (0.603/0.797/0.723); Table 1 A5; §4.3(ii) "safety head" sentence. No imbalance sweep, no rare-regime FPR measurement. | Numbers correct. Argument overreaches: see P4. | **P4** |
| Training and inference cost for edge | Yes. Appendix B, Table B1; §7 deployment paragraph sentence "We report training and inference cost in Appendix B". | Numbers correct. But (a) Table B1 has no memory column, memory is prose/caption only; (b) "The former 'no measurements' sentence ... now points to these numbers" is a phantom; (c) hardware not named. | **P1, P2, P3, P14** |
| Time-aware statistical test | Yes. §5.4 "Significance testing" paragraph (moving-block + episode bootstrap, CI and P defined); results in §6, Table 4, Figure 3. | Numbers correct. Cites "Sections 6, Table 4, Figure 3" for the definition, which lives in §5.4. | P11 |
| Ablations isolating source of gain | Yes. Table 5 (§6 "Where the gain comes from"), Table 6, Table A1, Table A2. | Yes. +0.200 / +0.141 / +0.017 and +0.072 match. | none |
| Forwarded R2 items | See R2 table. | Summary paragraph accurate except P9 (clause "removed"). | P8, P9 |
| Point-by-point response + revised manuscript | Yes; change-marking is real (`rev2-highlight` = green #86e29a, `rev-del` = strikethrough). | Yes. Note `rev-highlight` (yellow, first IIoT pass) is a second highlight color the letter does not mention. | minor |

## Reviewer 1

| R1 point | Addressed in paper? Where | Letter accurate? | Issue |
|---|---|---|---|
| 1. A9/A10 unused, basin head (A3) inactive; activate or scope | Yes, as for Editor 1. Additional datasets were screened for A8 (Cranfield, MetroPT, SMD, SKAB, Paderborn; Appendix C). | Yes. "wide two-component head misses about 80 percent ... 80-component head about 13 percent" matches §7 and Table C1 (0.80 / 0.13). | P15b |
| 2. Imbalance-ratio sweep: mixture NLL vs nearest NLL vs other imbalance-aware scores; rare-regime FPR | **Not run.** Only the pre-existing head isolation (Table 6) and A5 rationale. No imbalance ratios, no FPR of rare normal regimes, no third scoring method. | Rebuttal is labelled as such (good), but its premise is not in the paper and contradicts it. | **P4** |
| 3. Cost of many-model design under same hardware as baselines | Yes in substance (Appendix B, Table B1, §4.4 architecture sentence, §7 pointer). Hardware never named. | Numbers correct (50/195/38 s, 17-32 ms, 52 ms b256, TranAD 231/898/163). | **P3**, P2, P14 |
| 4. Time-aware test on all three; clarify WADI CI and P | Yes (§5.4 definitions, §6 numbers, Table 4, Figure 3). | Numbers correct. "LatAD is significantly ahead of every learned detector on WADI" has no test behind it. Definition cited to §6 instead of §5.4. | **P5**, P11 |
| 5. Raw sequential / temporal / spectral input to isolate source of gain | Partly. Table 5 (cross-channel vs channel-independent) and Table A2 (six stats vs ten temporal/spectral features). **No raw-sequential-input experiment.** | Table 5 / A2 numbers correct. Raw sequential input is never mentioned. | **P6** |
| English / figures / conclusions | Yes. Figures 1-3, Tables 3-6 and A1 rebuilt, §8 rewritten. | "Tables 3 to 6 and A1 were regenerated": Table 2 is also regenerated (0.742/0.621/0.733, marked new) though the letter does not claim it; fine. Opening line misattributes a compliment. | P13, P15a, P15c |

## Reviewer 2

| R2 point | Addressed in paper? Where | Letter accurate? | Issue |
|---|---|---|---|
| 1. Expand abbreviations at first use | Partly. LatAD (abstract), CPS/IIoT/SCADA (§1), VaDE/HC/AUROC/F1 (§1), GMM (Table 1), BIC (§3), NLL (§4.2) are expanded. Still unexpanded at first use: VaDE, AUROC, IIoT in the abstract; SOTA, ICS, NLL (Table 1 before §4.2), KDE in the body. | Overclaims "The remaining body abbreviations ... are now expanded at first textual use". | **P8** |
| 2. Abstract: highlight LatAD principle | Yes. Abstract mechanism sentence is verbatim as quoted; headline numbers and Zenodo closer present. | Yes. | none |
| 3. Intro para 1: delete method sentence, no contributions, add CPS background | Yes. Sentence is struck (`rev-del`); CPS opener verbatim; contributions in a later paragraph. | Yes. | none |
| 4. "the task is often framed as" | Yes. §1: "Unsupervised anomaly detection for CPS telemetry is often framed as ..." verbatim. | Yes. | none |
| 5. Two obstacles: which does LatAD address | Yes. §1 dedicated paragraph, verbatim as quoted. | Yes. | none |
| 6. Related Work restructure | Partly. Definition moved to §1 (old sentence struck); the three named sentences struck; §2.3 rewritten with Deep SVDD [29], THOC [30], OmniAnomaly [31]; old §2.5 struck and replaced by "Benchmark scale" (replacement, not deletion; letter says so). But Related Work still carries LatAD self-descriptions, several newly added. | Accurate on what was struck; silent on what was added. | **P7** |
| 7. System-model diagram | Yes. Figure 1 (inline SVG) with caption as quoted; referenced "(Figure 1)" in §1. | Yes. | none |
| 8. MIIM (§3) vs LatAD (§4) relationship | Yes. §3 new "checklist" sentence; Table 1 "Realized by" column + grouped rows; §4 "One realization of the assumptions" paragraph (pre-existing). | "Section 3 now states ... MIIM is the specification and LatAD is one realization": that sentence is in §4 and was already in the previous version. | P10 |
| 9. §4 readability + workflow figure | Yes. §4.1, §4.3, §4.4 headings tagged with stage + Figure 2 pointer; Figure 2 (inline SVG) referenced from §4 preamble. | Yes. | none |
| 10. Per-community vs shared encoder; training time, GPU memory, latency | Architecture: yes (§4.4 parenthetical, verbatim). Training time and latency: yes (Table B1). **GPU memory: no** (host memory only). | Letter says "host memory" (true) and never says GPU memory was not reported, although `poc/sota_bundle/results/e3_cost_cuda.json` records `peak_gpu_mb` (62.7 MB on the first entry). | **P2**, P14 |
| 11. z notation conflict | Yes. Latent z kept; `std(.)` operator in Eq. after (5); trivial rule is `max|u|` in §5.3, §5.5 prose, Tables 3-4. | "The former disambiguation clause ... is removed": a new clause was added instead, §5.3 "(u denotes the standardized per-channel window mean, distinct from the latent z of §4.3)". "z-score units" survives in §4.4 and the Table A1 caption. | P9 |
| 12. Typesetting: indentation, numbered equations | Equations (1)-(9) numbered, right-aligned `eqno`. Indentation: not in the HTML (no `text-indent` rule); claimed for the DOCX build. `paper_mdpi.docx` exists (2026-09-18 16:09) but was not opened here. | Claim is unverifiable from IoT2.html; needs a render check of the DOCX. | P12 |
| Results clarity / English / figures | Yes (§6 rewritten, Table 5, GDN in Table 3, Figures 1-3). | Yes. | P15 |

## Issues in detail (quotes are verbatim)

### P1. Phantom edit (Editor letter, request 3)
Letter: "The former 'no measurements' sentence in the deployment paragraph of Section 7 now points to these numbers."
No version of IoT2.html (236c78d, 5b454d6, e7198da) ever contained a "no measurements" sentence in §7; the only
"not measured" string is the Table B1 caption. The sentence that did exist, and still exists unchanged in the §7
deployment paragraph, is: "This is a deployment architecture rather than an experimentally evaluated configuration."
It now sits two sentences before "We report training and inference cost in Appendix B" and reads as a contradiction.
Fix: either strike that sentence in the paper or reword the letter to say the pointer sentence was added.

### P2. GPU memory requested, not reported, although measured (R2 comment 10; Editor 3)
R2: "The paper does not report training time, GPU memory, or inference latency". Paper reports "3.5 to 3.9 GB of host
memory" in Appendix B prose and the Table B1 caption only; Table B1 has columns Params / Train s / GPU ms/win / CPU-edge
ms/win and no memory column. The source `e3_cost_cuda.json` carries `peak_gpu_mb` and `peak_host_ram_mb` per entry, so
GPU memory is available. The R2 letter ("host memory") and Editor letter ("Table B1 report parameter count, training
wall-clock, per-window inference latency, and host memory") are accurate about what is there but never say the
requested GPU figure is absent. Fix: add a GPU-memory column (or sentence) and say so in the letters.

### P3. "Same hardware setting" never specified (R1 comment 3)
R1: "evaluate the practical cost of this design under the same hardware setting as the main baselines." Appendix B and
Table B1 say only "one shared GPU machine plus a CPU edge proxy"; no GPU model, CPU model, or cloud instance type appears
anywhere in IoT2.html (searched RTX, GeForce, Tesla, A100, T4, L4, Xeon, Core, Ryzen). The diagnostics reference Modal.
The R1 letter repeats "one shared GPU machine" without naming it. A reviewer who asked for "the same hardware setting"
will ask what it was.

### P4. Nearest-component rebuttal contradicts the paper (R1 comment 2; Editor 2)
Letter: "On these three benchmarks the nearest-component and mixture scores are close, because the benchmarks lack the
operating-regime imbalance that separates them: their normal windows sit almost entirely within one dominant regime, the
same reason the A8 overlap is absent."
(a) Nothing in the paper says the benchmarks lack regime imbalance or that normal sits in one dominant regime. The
paper says the opposite: §3 "cluster sizes are heavy-tailed", and Table 1 lists A5 "Heavy tail (Zipf imbalance)" under
"Realized and validated (A1-A7)". The letter therefore argues the requested experiment is moot because the data lacks
A5, while the paper claims A5 is validated on that data.
(b) "we tie the missing stress case to the new benchmark-scale discussion (Section 2.5, Section 7), which argues that
rare-regime behavior surfaces at operational scale": §2.5 and the §7 "Benchmark scale" paragraph name only A8, A9, A10;
neither mentions A5, rare regimes, or the nearest-component head.
(c) "close": Table 6 nearest-component vs latent density is 0.603 vs 0.656 (WADI) and 0.723 vs 0.794 (SWaT); the
base (density + nearest) is below density alone on WADI (0.634 vs 0.656) and SWaT (0.775 vs 0.794). The paper's §4.3(ii)
concedes this for "(Table 6, WADI)" only.
(d) The experiment R1 asked for (imbalance-ratio sweep, rare-regime FPR, other imbalance-aware scores) was not run.
The letter is candid that this is a rebuttal, which is right; the supporting argument needs to match the paper.

### P5. "Significantly ahead of every learned detector on WADI" has no test (R1 letter comment 4; paper §6)
R1 letter: "LatAD is significantly ahead of every learned detector on WADI and ties only the linear baseline."
Paper §6: "so LatAD is significantly ahead of every learned detector on WADI and ties only the linear baseline."
The only WADI bootstrap reported anywhere (paper, `clean_recompute.md`, `manuscript_reconcile.md`) is LatAD vs LinRes:
+0.020, 95% CI [-0.161, 0.205], P = 0.46, 8 episodes. No CI or P versus TranAD (0.613), USAD (0.579), AutoEncoder
(0.628) or Isolation Forest (0.634) exists. With a 0.37-wide interval on 8 episodes, "significantly" is an unbacked
claim in both letter and paper. The abstract, Figure 3 caption and §8 use "ahead of every learned detector" without
"significantly", which is the defensible wording.

### P6. Raw sequential input silently dropped (R1 comment 5)
R1: "Experiments with raw sequential input and temporal or spectral information are needed". The paper adds Table A2
(ten hand-built temporal/spectral features) and Table 5; no raw-sequence (per-timestep) input variant of LatAD was
run. The letter answers "Response: change made, the source of the gain is now isolated" and never mentions raw
sequential input. It should say plainly that the raw-sequence variant was not run and why the feature-level check
answers the question.

### P7. Related Work still describes LatAD, with new self-descriptions added (R2 comment 6)
R2's principle: "since LatAD is the authors' novel work, it should not be described in detail in Related Work." The
three named sentences are struck (verified). Remaining or newly added in §2:
- §2.1 (new): "a setting LatAD shares"; "reinforcing our regime-latent factorization over a single plant-wide model";
  "Process-level, subsystem-localized detection from normal-only telemetry, under regime imbalance, remains open."
- §2.3 (new): "none estimates density per physical subsystem under regime imbalance, the gap this paper addresses";
  "whereas we factorize a learned latent density over correlation communities and combine them by Higher Criticism."
- §2.4 (new): "whereas we route per-community density experts by cohesion-weighted Higher Criticism rather than a
  learned gate"; "the mechanism analysis of §7 suggests they inherit the same blind spot ... that define our difficult
  subset"; "We also benchmark the cross-channel GDN ... (0.481) ...".
- §2.4 (untouched): "LatAD generalizes this applied line to unsupervised, multimodal-latent anomaly detection."
The letter's "The three LatAD self-descriptions ... are struck" is true but incomplete; a reviewer re-reading §2 will
find more LatAD than before in §2.1 and §2.3.

### P8. Abbreviations still unexpanded at first use (R2 comment 1)
- Abstract: "latent Gaussian mixture (VaDE)" (VaDE not expanded); "best overall AUROC" (not expanded); "three real IIoT
  testbeds" (the first abstract sentence writes "The Industrial Internet of Things" without "(IIoT)").
- Body: "SOTA" first clean use is §2.4 "deep SOTA"; its only expansion "state-of-the-art (SOTA)" is inside the struck
  §2.2 paragraph and will vanish when deletions are accepted. "ICS" (§2.3 "flag ICS anomalies") never expanded. "NLL"
  appears in Table 1 A5 ("Nearest-component NLL head") before its §4.2 expansion. "KDE" (§4.3 heading) precedes
  "kernel-density estimate". IIoT is expanded twice in §1 (paragraphs 1 and 2).
Letter: "The remaining body abbreviations (VaDE, GMM, NLL, HC, BIC, AUROC, F1) are now expanded at first textual use."
True for the body order only if Table 1 is not counted; false for the abstract.

### P9. "Disambiguation clause removed" but a new one added (R2 comment 11)
Letter: "The former disambiguation clause is no longer needed and is removed." Paper §5.3 now reads "max|u| (u denotes
the standardized per-channel window mean, distinct from the latent z of §4.3)". The old clause is gone; a new one is
there. Also "train-normal z-score units" remains in §4.4 and the Table A1 caption. Harmless, but the letter should say
"replaced" not "removed".

### P10. Wrong section for the specification/realization statement (R2 comment 8)
Letter: "Section 3 now states the relationship explicitly: MIIM is the specification and LatAD is one realization of
it". That statement is §4 "One realization of the assumptions. A1-A10 specify what a model of CPS normal must capture,
not how to implement it; the detector below is one concrete realization of A1-A7", and only "A1-A7" is marked new.
What §3 newly adds is the "checklist" sentence and the Table 1 column. "They are closely related and evaluated
together" does not appear in the paper.

### P11. Bootstrap definition cited to §6; it is in §5.4 (Editor 4; R1 comment 4)
Both letters: "(Section 6, Table 4, Figure 3)" for the episode-block bootstrap and the CI / P definitions. The
definitions ("Significance testing." paragraph: block length, 2000/1000 resamples, percentile CI, one-sided P) are in
§5.4 Comparable metrics. §6 only reports the outcomes.

### P12. Indentation claim deferred to DOCX (R2 comment 12; Editor)
Letter: "Paragraph-indentation is unified ... applied at the DOCX build stage". IoT2.html has no `text-indent` rule, so
the HTML the reviewers may see is not unified. `poc/paper/paper_mdpi.docx` (2026-09-18 16:09) is the candidate build;
it was not rendered for this audit. Render and check before submission, or the claim is unverified.

### P13. Misattributed compliment (R1 letter opening)
R1 letter: "Thank you for a careful and constructive reading and for finding the research content interesting."
Reviewer 1 wrote no such thing; Reviewer 2 wrote "The research content of the paper is very interesting." Move the
phrase to the R2 letter or drop it.

### P14. "Table B1 reports ... host memory" (Editor 3; R2 comment 10)
Table B1 has no memory column; memory is stated in the Appendix B paragraph and caption. Say "Appendix B" for memory.

### P15. Paper-internal inconsistencies met while checking (affect "results clearly presented")
a. §6 "even while they remain strong on Easy (0.90-0.97)" for USAD/TranAD; Table 3 Easy values are 0.962-0.998.
b. §7: "on a synthetic 50-percent-overlap positive control"; Appendix C: "a 30-percent-overlap regime pair" and Table C1
   "30-percent-Bayes-error control (0.61)". Both letters repeat "50 percent". One of the two numbers is wrong.
c. Figure 3 omits GDN although Table 3 and Table 4 include it; the R1 letter says only that GDN was added to Table 3,
   so the letters are accurate, but the figure and table now disagree on the baseline set.
d. §7 deployment paragraph: "This is a deployment architecture rather than an experimentally evaluated configuration."
   contradicts the Appendix B measurements two sentences later (see P1).

### Stale entries in letters_summary.md (mapping file, not the letters)
- "Table 2 ... still on pre-clean numbers; letters do not cite it": Table 2 in IoT2.html is marked new with
  0.742 / 0.621 / 0.733 and its caption cites Table 6 consistently.
- "GDN double-hard cells are '-'": Table 4 now carries GDN 0.550 / 0.469 / 0.615.
Neither affects the letters, but the mapping file should be refreshed so it does not mislead a later pass.

## Tone and style check (all three letters)
- Em-dash, en-dash, double hyphen: none found (grep for "—", "–", "--").
- "honestly / frankly / candidly": none.
- Author spelling: "Alexander Apartsin" in all three signatures and headers.
- Register: polite and thankful throughout; plain English. One spaced hyphen used as a dash in the R2 letter heading
  for comment 4 ("... framed as ..." - state which task) and in the quoted reviewer text ("XXX"-what task), where the
  reviewer's em-dash was replaced by a bare hyphen; consider a colon or comma.

## Suggested minimal fixes, in priority order
1. P5: delete "significantly" from the WADI learned-detector sentence in the R1 letter and in §6, or run and report
   the bootstrap vs TranAD/USAD/AE on WADI.
2. P1 + P15d: strike "This is a deployment architecture rather than an experimentally evaluated configuration." in §7
   and reword the Editor letter sentence.
3. P2: add GPU memory (peak_gpu_mb is already in e3_cost_cuda.json) to Table B1 and to both letters.
4. P3: name the GPU and CPU (or cloud instance) in Appendix B.
5. P4: rewrite the nearest-component rebuttal so it does not deny A5 on the benchmarks; anchor it on §4.3(ii) and
   Table 6 as written, and drop the "Section 2.5, Section 7" pointer or add one A5 sentence there.
6. P6, P7, P8, P13: one-line letter edits plus the abstract abbreviations and the §2.1/§2.3 self-descriptions.
7. P9, P10, P11, P14: cross-reference corrections in the letters.
8. P12, P15a, P15b: render the DOCX; fix 0.90-0.97; reconcile 50 vs 30 percent.
