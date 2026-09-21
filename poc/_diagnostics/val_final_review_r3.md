# Final acceptance check — IoT-4522779 (revision3)

Adversarial MDPI *IoT* associate-editor pass. Verified from scratch against
`poc/paper/IoT2.html`, the three response letters, and the three review files.
READ-ONLY; nothing in the submission was modified.

## 1. Coverage + well-addressed (per-comment)

Legend: **Answered** = letter response present; **Backed** = a real, matching paper change exists.

| Comment | Letter response | Paper change (verified) | Answered | Backed |
|---|---|---|---|---|
| R1-1 (A9/A10 unused, A3 basin head inactive) | Rev1 C1 | Table 1 regrouped A1–A7 / A8–A10 + "Realized by" col; A3→A8 renumber; §8, App. C | Y | Y |
| R1-2 (nearest-component support) | Rev1 C2 | App. E + Table E1 (3W); head isolated in Table A6 | Y | Y |
| R1-3 (regime-community cost) | Rev1 C3 | App. B + Table B1 | Y | Y |
| R1-4 (time-aware test, WADI CI/p) | Rev1 C4 | §5.4 episode-block bootstrap; §6, Fig 3 | Y | Y |
| R1-5 (source of gain, six stats) | Rev1 C5 | Table A5 (cross-channel), Table A2 (representation) | Y | Y |
| R1 English/figures/conclusions | Rev1 tail | Figs 1–2, Fig 3 renumber, §9 rewrite | Y | Y |
| R2-1 (expand abbreviations) | Rev2 C1 | Abstract LatAD; §1 CPS/IIoT/SCADA | Y | Y |
| R2-2 (abstract = mechanism) | Rev2 C2 | Abstract rebalanced (exact quote matches) | Y | Y |
| R2-3 (intro logic; delete method sentence; CPS bg) | Rev2 C3 | Line 89 "This paper detects them by…" struck; CPS opener | Y | Y |
| R2-4 (line 71 "the task") | Rev2 C4 | §1: "Unsupervised anomaly detection for CPS telemetry is often framed…" | Y | Y |
| R2-5 (two obstacles / what problem) | Rev2 C5 | §1 C1/C2/C3 (both axes) | Y | Y |
| R2-6 (restructure Related Work) | Rev2 C6 | Definition→§1; 3 self-descriptions struck; §2.3 rewritten (SVDD/OmniAnomaly/THOC); §2.5→"Benchmark scale" | Y | Y |
| R2-7 (system-model diagram) | Rev2 C7 | Figure 1 + Table 5 triage | Y | Y |
| R2-8 (MIIM↔LatAD relationship) | Rev2 C8 | §3 checklist sentence; §4 "A1–A10 specify what…"; Table 1 "Realized by" | Y | Y |
| R2-9 (§4 readability, workflow fig) | Rev2 C9 | Figure 2; §4.1/4.3/4.4 headings name pipeline stage | Y | Y |
| R2-10 (per-community fit? cost) | Rev2 C10 | §4.4 explicit ("separate small VaDE…encoder not shared"); App. B | Y | Y |
| R2-11 (z symbol conflict) | Rev2 C11 | latent z kept; operator→std(·); trivial value→u (Tables 2/A4, §5.3/5.5/6) | Y | Y |
| R2-12 (typesetting, number equations) | Rev2 C12 | Eqs (1)–(9) all present; indentation at DOCX build | Y | Y |
| R2 results/English/figures | Rev2 tail | Table A5, GDN in Table 2/Fig 3 | Y | Y |
| ED-1 (validate A3/A9/A10) | Editor req 1 | Table 1, §8, App. C | Y | Y |
| ED-2 (nearest-component support) | Editor req 2 | App. E, Table E1 | Y | Y |
| ED-3 (train/inference cost) | Editor req 3 | App. B, Table B1 | Y | Y |
| ED-4 (time-aware test) | Editor req 4 | §5.4, §6 | Y | Y |
| ED-5 (ablations isolating gain) | Editor req 5 | Table A5, A6, A1, A2 | Y | Y |
| ED reviewer-2 forwards + point-by-point | Editor §"Requests forwarded" | Covered via Rev2 letter | Y | Y |

All 22 numbered comments (R1-1..5, R2-1..12, ED-1..5) answered and backed. No hand-waves; each rebuttal (nearest-component defensive role; A8 scoping) states the limitation plainly and points to a table.

## 2. Tone

All three letters open with thanks, use "Response: change made" / plain rebuttal, close with thanks. No defensive or apologetic phrasing; no "honestly/frankly". Limitations (WADI ties LinRes; A8 not demonstrated; SWaT ranking uninformative under drift) are stated plainly, not protested. PASS.

## 3. Number / cross-reference audit

Specifically-requested spots — all verified correct:
- **Table A5 HAI cross-channel cell = 0.789** (line 1337); channel-independent 0.589; gain **+0.200 = 0.789−0.589** (line 1339). Correct. WADI 0.656−0.639=+0.017; SWaT 0.472−0.522=−0.050. All correct.
- **Appendix E / Table E1 seed count = three** (rows seed 0/1/2 + mean; line 3033–3036); prose "three seeds" (App. E line 3028; both letters). Consistent. FP counts 17,17,23→11,9,16 match text; TPR@5% mean +0.007±0.068 matches.
- **Reviewer 2 letter comment 5** points to live §1 C1/C2/C3, quotes no deleted text, and correctly frames C2 (difficulty stratification) as a claimed contribution while only the raw-metric protocol is "adopted from the critique." Matches live C2. PASS.
- **Appendix D.2** reads "the standardized sum of that cohesion-weighted Higher-Criticism score and the whole-plant … upper-tail score, each standardized against its train-normal reference" (line 3010). PASS.
- **Reviewer 1 letter TranAD P value** = "+0.157 (P = 0.0005)" (line 129) matches paper §6 line 837. PASS.

Table base/value/delta arithmetic:
- Table A2: −0.008 / +0.071 / +0.241 all equal value−base. Correct.
- Table A5: +0.017 / +0.200 / −0.050 all correct.
- Table E1: mean rows 19 / 12 consistent with per-seed.

Subset-count arithmetic (Table 2 headers): WADI 26+30=56 ✓; HAI 485+167=652 ✓; SWaT 142+91=233 ✓.

Seed counts vs table rows: five-seed claims (Tables 2, A2, A4, A5, A6) match; GDN single-seed everywhere; Table E1 three-seed. Consistent.

Cross-references: Tables 1–5, A1–A6, B1, C1, D1, D2, E1 each defined exactly once; Figures 1, 2, 3, A1 all present and referenced; equations (1)–(9) all numbered. Reference [n] citations checked against the 63-entry list (HC=49, VaDE=27, USAD=36, TranAD=37, GDN=38, WADI=50, HAI=51, SWaT=52, 3W=63, etc.) — all live citations resolve to the intended entry. Deleted §2.5 text still carries old [38]/[39]/[40] dataset numbers but is struck (rev-del), so it does not ship.

Key headline numbers reconciled across abstract / letters / body / tables: HAI difficult 0.845 (USAD 0.497, TranAD 0.445, AE 0.757, GDN 0.481); WADI difficult 0.771 (LinRes 0.750, GDN 0.660); SWaT difficult 0.524→0.723 (Table 3), double-hard 0.178→0.671, All 0.788→0.880; drift 1.5/39/69%; false alarm 79%→6.5%; residual gate ratios 3.41/1.22/0.76 vs 1.5; communities 44/28/24; cost 3.2/16/3.5 s (global) vs TranAD 231/898/163 s (47–72×). All internally consistent.

Minor (non-blocking) notes:
- WADI prose margins are stated to 3 dp from seed-mean AUROCs, so a few differ by 0.001 from displayed-value minus displayed-base (e.g. +0.157 vs 0.771−0.613=0.158; +0.136 vs 0.137; +0.020 vs 0.021). The letters and body agree with each other; the discrepancy is display rounding, not a stale number.
- Editor letter (lines 53–54): "re-derived from 43 to 30 windows … 12 trivially separable constant-channel-flip windows." 43−30=13. Reconciles only if the clip fix removed 12 and the dropped analyzer channel removed the 13th (the next sentence bundles the two earlier corrections). Wording is tight but admits a consistent reading; advisory clarity nit, not a wrong result.

## 4. Highlighting integrity

Stack-checked all `rev-del` / `rev-highlight` / `rev2-highlight` spans:
- **Green-addition nested inside a red rev-del deletion: 0 occurrences** (the DOCX-corrupting pattern). PASS.
- Spans are balanced (every open closed).
- Three instances of the reverse (a `rev-del` deletion nested inside a `rev2-highlight` insertion) at lines 167, 272, 842 — deletion-inside-insertion, which OOXML tracked changes tolerates. Not the flagged corrupting pattern; advisory clean-up only.
- No live (non-struck) stale terms: "basin-agreement", "null expert", "z-score units", "did not complete" appear only inside rev-del or not at all; the sole "z-score" is struck (line 1245).

---

FINAL VERDICT: ACCEPT-READY (zero blocking)
