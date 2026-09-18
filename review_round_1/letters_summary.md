# Round-2 response letters: point-to-change map

Three HTML letters written in `review_round_1/`: `letter_editor.html`, `letter_reviewer1.html`,
`letter_reviewer2.html`. All reflect the FINAL verified numbers (`poc/_diagnostics/clean_recompute.md`,
`manuscript_reconcile.md`) and cite the revised paper `poc/paper/IoT2.html`. DOCX intentionally not built.

## Final story anchors used in all three letters
- Headline detector: fused HCcoh+LatAD (LatAD regime-community).
- Difficult-subset AUROC: HAI 0.845 (+0.088, CI [0.042,0.157], P about 0, significant); SWaT 0.837
  (+0.055, CI [0.012,0.104], P=0.007, significant); WADI 0.771 (significantly ahead of every learned detector
  by a per-detector episode bootstrap, all P<0.05: AE +0.142, USAD +0.192, TranAD +0.157, IF +0.136, GDN
  +0.111; ties only the linear baseline LinRes +0.020, CI [-0.161,0.205], P=0.46, 8 episodes).
- Best overall AUROC HAI 0.948, SWaT 0.938; WADI full set linear baseline marginally ahead (0.834 vs 0.827).
- WADI difficult subset re-derived 43 -> 30 (12 constant-channel-flip windows, difficulty-stratification
  artifact fixed). WADI difficult lead carried by community aggregation HC_coh 0.795; single-latent global
  density collapses to 0.634.
- GDN added as a real construct-matched baseline (Table 3); collapses to 0.481 on HAI-difficult like
  USAD/TranAD. Replaces the earlier inaccurate "GDN did not complete" statement.
- A8 reframed with valid observation-space metrics (Appendix C, Table C1): precondition present, heavy
  overlap absent, masking form coupled to A10, over-coverage form defended by the 80-component density head.
- A9 present and partly exploitable (HAI +0.06 longer window); A10 measured absent with passing positive
  control. "Snapshot detectable" now a measured finding.
- MIIM reframed: ten assumptions, seven realized+validated (A1-A7), three specified for future work
  (A8-A10). Former "A3" between-regime -> renumbered A8.
- Dataset-scale scholarship: Intro, new Section 2.5 "Benchmark scale", Section 7, new refs (MetroPT [53],
  BATADAL [46], N-CMAPSS [47], BDG2 [11], TSB-UAD [48]).
- Cost: Appendix B / Table B1. LatAD-global 3.2/16/3.5 s, ~47-72x faster than TranAD; latency within
  gateway budgets; ~3.5-3.9 GB host memory.

## Reviewer 1 (mostly experimental) -> changes
| R1 point | Final change | Location |
|---|---|---|
| 1. A9/A10 unused, basin head (A3) inactive; scope or activate | MIIM reframed 7 realized (A1-A7) + 3 future (A8-A10); "A3" -> A8; A8 remeasured (valid observation-space metrics, responsibility screen shown blind by positive control); A9 present/partly exploitable; A10 measured absent | Table 1, Section 7, Appendix C |
| 2. Nearest-component NLL lacks support; imbalance sweep + rare-regime FPR | Rebuttal: head isolated in Table 6 (WADI 0.603/HAI 0.797/SWaT 0.723); A5 design rationale; benchmarks lack regime imbalance (same reason A8 overlap absent); distinctive benefit tied to operational scale | Table 6, Table 1 (A5), Sections 2.5, 7 |
| 3. Many-model cost vs baselines | Cost benchmarked; per-community architecture stated; LatAD-global 47-72x faster than TranAD, ~3.5-3.9 GB | Appendix B, Table B1, Section 4.4, Section 7 |
| 4. Time-aware test all three + WADI CI/P | Episode-block bootstrap uniform on all three; CI/P defined in Section 5.4; HAI/SWaT significant; WADI significantly ahead of every learned detector (per-detector bootstrap P<0.05), ties LinRes +0.020 P=0.46 (8 ep) | Section 5.4, Section 6, Table 4, Figure 3 |
| 5. Six static stats; isolate source of gain | New Table 5 source-of-gain (cross-channel +0.200 HAI, +0.141 SWaT, +0.017 WADI); Table A2 representation check (+0.072 HAI); localized by Tables 6, A1 | Section 6, Tables 5/6/A1/A2 |
| English / figures / conclusion | Language pass; Figures 1-3; tables rebuilt with GDN; Conclusion rewritten to scoped claims | Sections 1, 8, Figures 1-3, Table 3 |

## Reviewer 2 (mostly writing/structure) -> changes
| R2 point | Final change | Location |
|---|---|---|
| 1. Expand abbreviations at first use | LatAD/CPS/IIoT/SCADA + body abbreviations expanded | Abstract, Section 1 |
| 2. Abstract: highlight LatAD principle | Abstract rebalanced to mechanism, one headline number per claim | Abstract |
| 3. Intro para 1: delete method sentence, no contributions, add CPS background | "This paper detects them by..." struck; CPS/IIoT opener; contributions moved later | Section 1 |
| 4. "the task is often framed as..." define task | Named: unsupervised anomaly detection for CPS telemetry | Section 1 |
| 5. Two obstacles: which does LatAD address | Dedicated paragraph: modeling obstacle = detector; evaluation = adopted protocol | Section 1 |
| 6. Related work restructure | AD definition -> Section 1; 3 self-descriptions struck; Section 2.3 rewritten as survey (Deep SVDD/OmniAnomaly/THOC); old 2.5 replaced by "Benchmark scale" | Sections 1, 2.1-2.5 |
| 7. System-model / scenario diagram | Figure 1 added and referenced | Section 1 |
| 8. MIIM(3) vs LatAD(4) relationship | Explicit "specification vs one realization"; Table 1 "Realized by" column + grouped rows | Section 3, Table 1 |
| 9. Section 4 readability + workflow figure | 4.1-4.4 headings name pipeline stages; Figure 2 added | Section 4, Figure 2 |
| 10. Per-community vs shared encoder; cost | Separate VaDE per community (encoder not shared); cost reported | Section 4.4, Appendix B, Table B1 |
| 11. z notation conflict | Latent z kept; standardizer -> std(.); trivial-rule value -> u; clause removed | Sections 4.3, 5.3, 5.5, 6, Tables 3-4 |
| 12. Typesetting + numbered equations | Nine equations numbered (1)-(9); indentation unified at DOCX build | Sections 3, 4.1, 4.3, 4.4 |
| Results clarity / English / figures | Results rewritten on corrected numbers; GDN added; Figures 1-3; language pass | Section 6, Figures 1-3, Table 3 |

## Editor -> changes
| Editor request | Final change | Location |
|---|---|---|
| Validate unused assumptions (A3, A9, A10) | 7 realized + 3 future framing; A8 (ex-A3) remeasured; A9 present; A10 measured absent | Table 1, Section 7, Appendix C |
| Direct support for nearest-component likelihood | Head isolated (Table 6) + A5 rationale + scale argument | Table 6, Sections 2.5, 7 |
| Training/inference cost for edge | Appendix B, Table B1 | Appendix B |
| Time-aware statistical test | Episode-block bootstrap uniform, CI/P defined | Section 6, Table 4 |
| Ablations isolating source of gain | Table 5 (+ Tables 6, A1, A2) | Section 6 |
| (Fwd R2) abstract/intro, related work, figures, z, equations | All done as in R2 map | Sections 1, 2, 3, 4, Figures 1-2 |
| Point-by-point response + revised manuscript | These three letters; change-marked IoT2.html | - |

## Notes / open items (updated round-2)
- Table 2 (WADI single-model decomposition) is now on the clean numbers (WADI 0.742 / 0.621 / 0.733, marked new);
  letters do not cite it.
- GDN double-hard cells are now populated in Table 4 (WADI 0.550 / HAI 0.469 / SWaT 0.615); Figure 3 also carries a
  GDN bar. Letters say GDN is in Table 3 and Table 4.
- New refs are integrated with first-appearance numbering already applied: MetroPT [53], BATADAL [46],
  N-CMAPSS [47], BDG2 [11], TSB-UAD [48].
- Cost table (Table B1) now names the hardware (Modal A10G GPU, 24 GB, plus a Modal 8-vCPU CPU edge proxy) and adds
  a peak-GPU-memory column; letters (Editor, R1) report the GPU memory and named hardware.
