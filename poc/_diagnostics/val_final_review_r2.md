# Independent final acceptance check — IoT-4522779 (revision3)

Reviewer: fresh MDPI *IoT* associate editor. Sources verified line-by-line:
`review_round_1/Review1.txt`, `Review2.txt`, `Editor review.txt`; paper
`poc/paper/IoT2.html`; letters `letter_editor.html`, `letter_reviewer1.html`,
`letter_reviewer2.html`. Nothing assumed from earlier rounds.

## 1–2. Per-comment coverage and adequacy

| # | Comment (short) | Letter answers | Paper contains change | Adequate? |
|---|---|---|---|---|
| R1-1 | A9/A10 unused, basin/A3 head inactive; validate or scope | Yes (R1 C1) | Table 1 regrouped A1–A7 realized / A8–A10 specified, "Realized by" col; old A3→A8; A8 screen App C+Table C1; A9 timescales 3.4–3.7 dec, +0.06 HAI; A10 measured absent (§8) | Yes — remeasured, not just scoped |
| R1-2 | Nearest-component lacks direct support; imbalance sweep, rare-normal FPR | Yes (R1 C2) | Table A6 standalone head (0.603/0.797/0.479); **Appendix E + Table E1** on 3W (rare-regime FP 17,17,23→11,9,16; TPR@5% +0.007±0.068) | Yes — direct experiment added |
| R1-3 | Regime-community cost vs baselines, same hardware | Yes (R1 C3) | Appendix B + Table B1 (params, train s, GPU/CPU ms, peak GPU MB, host RAM), one Modal A10G + 8-vCPU | Yes |
| R1-4 | Time-aware test all 3 datasets; clarify WADI CI/p | Yes (R1 C4) | §5.4 episode-block/moving-block bootstrap; §6 per-detector WADI p-values; defs of CI and P | Yes |
| R1-5 | 6 static stats; raw-seq/temporal/spectral to isolate gain | Yes (R1 C5) | Table A5 (cross-channel vs marginals) + Table A2 (temporal/spectral rep); raw per-timestep variant honestly declined | Yes — plainly stated limit |
| R1-eng | English/figures/conclusions | Yes | Fig 1/2/3, Conclusion rewritten | Yes |
| R2-1 | Expand abbreviations | Yes (R2 C1) | LatAD/CPS/IIoT/SCADA expanded (abstract, §1); rest at first use | Yes |
| R2-2 | Rebalance abstract to LatAD mechanism | Yes (R2 C2) | Abstract rewritten (VaDE, community factorization, HC), one headline result per claim | Yes |
| R2-3 | Intro logic; delete "This paper detects them by…"; CPS background | Yes (R2 C3) | §1 opens with CPS/IIoT; method sentence struck (rev-del, line 89); contributions moved later | Yes |
| R2-4 | p.2 l.71 "the task" — name it | Yes (R2 C4) | §1 "Unsupervised anomaly detection for CPS telemetry is often framed as…" | Yes |
| R2-5 | Two obstacles — LatAD contributes to both? | Yes (R2 C5) | §1 C1 (modeling) + C2 (evaluation) explicit | Yes |
| R2-6 | Restructure Related Work; move definition; delete 3 self-descriptions; rewrite 2.3; delete 2.5 | Yes (R2 C6) | Definition→§1; 3 self-descriptions struck (l.204, 207, 229); §2.3 neutral survey (Deep SVDD/THOC/OmniAnomaly); §2.5 → "Benchmark scale" | Yes |
| R2-7 | System-model diagram | Yes (R2 C7) | Figure 1 + validated triage (Table 5) | Yes |
| R2-8 | MIIM (§3) ↔ LatAD (§4) relationship | Yes (R2 C8) | §3 checklist sentence; §4 "A1–A10 specify what… realization of A1–A7"; Table 1 "Realized by" col | Yes |
| R2-9 | §4 readability; map 4.1–4.4 to stages; workflow figure | Yes (R2 C9) | Headings tagged with pipeline stage; Figure 2 | Yes |
| R2-10 | Per-community fit / encoder shared? cost | Yes (R2 C10) | §4.4 separate VaDE per community, own encoder; Appendix B | Yes |
| R2-11 | z symbol conflict | Yes (R2 C11) | latent z kept; std(·) operator; trivial-rule value u (Table 2/A4, §5.3/5.5/6); "z-score"→"standardized" | Yes |
| R2-12 | Typeset indentation; number equations | Yes (R2 C12) | 9 equations numbered (1)–(9); indentation unified at DOCX build (not HTML-verifiable) | Yes* |
| R2-res | Results/English/figures | Yes | GDN added Table 2/Fig 3; Fig 1/2/3 | Yes |
| ED-1 | Validate unused A3/A9/A10 | Yes (ED §1) | = R1-1 evidence | Yes |
| ED-2 | Direct support nearest-component | Yes (ED §2) | Appendix E + Table E1 | Yes |
| ED-3 | Training/inference cost for edge | Yes (ED §3) | Appendix B + Table B1 | Yes |
| ED-4 | Time-aware significance test | Yes (ED §4) | §5.4 + §6 | Yes |
| ED-5 | Gain ablation (Table A5) | Yes (ED §5) | Table A5 | Yes |
| ED-R2 | Forward R2 writing/structure items | Yes (ED forwarded section) | Covered in R2-1..12 | Yes |

*R2-12 indentation is claimed to be applied at the DOCX build stage; it cannot be
confirmed from the HTML tracked-changes source but is a reasonable place to assert it.

All 22 numbered comments (R1-1..5, R2-1..12, ED-1..5) are answered in the letters and
each claimed change is present in the paper. No comment is uncovered.

Special attention to the Editor's five:
- **A3/A9/A10 validation:** old between-regime A3 renumbered A8 and *remeasured*
  (observation-space screen, App C/Table C1, synthetic 30→2% control); A9 present
  (timescales 3.4–3.7 decades, +0.06 HAI); A10 measured absent with passing positive
  control + order-shuffle guard. Substantive, not a hand-wave.
- **Nearest-component rare-normal-FPR (Appendix E):** present as Table E1 on 3W;
  invariant (can only remove, never add, a rare-regime FP) stated and holds every seed.
- **Edge cost (Table B1):** complete (params, train s, GPU+CPU latency, peak GPU MB,
  host RAM), one shared hardware set.
- **Time-aware significance:** episode-block + moving-block bootstrap on all three
  datasets; CI and P defined; WADI per-detector p-values given.
- **Gain ablation (Table A5):** cross-channel vs marginal product; +0.200 HAI /
  +0.017 WADI / −0.050 SWaT.

## 3. Letter tone

All three letters open with thanks, credit the reviewers/editor, and close politely
("Thank you again", "we have stated it plainly"). Where the authors rebut (R1-2
nearest-component is defensive-by-design; R1-5 raw-per-timestep variant not run), they
state it factually and back it with an added experiment or an explicit limitation, never
dismissively. No defensive or combative language. PASS.

## 4. Stale content / number audit

Base/value/delta tables (exact check):
- **Table A5:** WADI 0.656−0.639=+0.017 ✓; HAI 0.789−0.589=+0.200 ✓; SWaT 0.472−0.522=−0.050 ✓.
- **Table A2:** WADI 0.626−0.634=−0.008 ✓; HAI 0.814−0.743=+0.071 ✓; SWaT 0.488−0.247=+0.241 ✓.
- **Table E1:** mixture mean (17+17+23)/3=19 ✓; nearest (11+9+16)/3=12 ✓; TPR (0.064+0.046−0.089)/3=+0.007 ✓.
- **Table 3 drift transitions:** SWaT diff 0.524→0.723=+0.199≈+0.200 ✓; double-hard 0.178→0.671=+0.493 ✓; All 0.788→0.880=+0.092 ✓.

Letter↔paper cross-checks (all match the current paper):
- HAI difficult 0.845 vs AutoEncoder 0.757 = +0.088, CI [0.042,0.157], 167 win/26 ep ✓.
- SWaT difficult 91 win/23 ep; raw ties linear, typed 0.723 (+0.200, CI [0.018,0.355], P=0.014) ✓.
- SWaT double-hard 0.178→0.671 (+0.493, CI [0.349,0.619], P<0.001) ✓.
- Cost: 3.2/16/3.5 s; 47–72× vs TranAD (231/898/163); GPU 0.7–13 (global), 17–32 (community);
  CPU 0.6–14.6; GPU mem ~20–29 MB; host 3.5–3.9 GB — all match Table B1/App B ✓.
- Nearest-component Table A6 (0.603/0.797/0.479) ✓; Appendix E figures ✓.
- Table 5 localization: HAI top-3 0.68 vs 0.56 random; WADI 0.56 vs 0.31 — matches editor
  and R2-7 letters ("68% HAI, 56% WADI, against 56 and 31%") ✓.
- Abstract: HAI 0.845, USAD/TranAD 0.44–0.50, WADI 0.771, SWaT 0.723, FA 79%→6.5%,
  drift 1.5/39/69% — all match §6/§7 ✓.
- Community counts 44/28/24 (§5.2 = App D.2) ✓; K* 22/24/25 ✓; K 20/40/40, latent 10/16/16 ✓;
  feature dims 732/354/306 ✓; WADI 122 channels (123−1 dropped) ✓.

SWaT is presented as a drift-aware win with correct numbers (raw difficult 0.524 tie →
typed 0.723; double-hard 0.178→0.671; All 0.788→0.880); FA 79%→6.5% consistent across
abstract, §7, Table 4, and all three letters.

Deprecated framing: none survives in live text — grep confirms 0 live occurrences of
"did not complete", "z-score", "null expert", "basin-agreement", "43 windows",
"K*≥22". The old SWaT-localization claim ("68 to 79% of HAI and SWaT") survives only as
struck rev-del text in §4.4; live text and Table 5 correctly use HAI+WADI and omit SWaT
(drift). Table/figure pointers in the letters (Tables 1,2,5,A1,A2,A5,A6,B1; Figs 1,2,3,A1;
Apps B,C,E; §§2.4,2.5,5.4,5.5,7,8) all resolve to the correct objects.

NON-BLOCKING note (rounding, not stale): three WADI difficult in-prose margins differ by
0.001 from the rounded cell subtraction — TranAD stated +0.157 (0.771−0.613=0.158),
Isolation Forest +0.136 (0.771−0.634=0.137), LinRes +0.020 (0.771−0.750=0.021). These are
paired differences of full-precision five-seed-mean AUROCs, so a 1-ULP gap from the
rounded display is expected and correct; the letter and paper agree with each other
exactly, and the dedicated base/value/delta tables (A5, A2) are exact. Not a defect.

## 5. Highlighted edits

- Substantive additions are marked green (`rev-highlight`/`rev2-highlight`); deletions are
  red strikethrough (`rev-del`). New figures/tables/appendices (Fig 1/2/3/A1, Tables 3/4/5,
  B1/C1/D1/D2/E1, App B/C/E), the restructured intro/related-work, and every reworked
  results sentence carry the highlight classes.
- **No green addition span is nested inside a red `rev-del` deletion.** Programmatic scan
  (span-level and block-level p/td/th/li/caption/h*/figcaption/text/div): 0 dangerous
  green-in-del nestings; all `<span>` tags balanced (0 leftover). Three benign red-in-green
  cases exist (rev-del inside a green span/para at lines 167, 272, 842) — the safe
  direction, which does not corrupt the DOCX tracked-changes render.

## FINAL VERDICT

**ACCEPT-READY** — zero blocking items. All 22 comments covered and substantively
addressed; letters polite and non-defensive; numbers, table/section pointers and quoted
results consistent between the three letters and the current paper; SWaT correctly framed
as a drift-aware recovery; highlighting complete with no green-inside-red nesting. The only
observations (three prose margins off by 0.001 from rounded-cell subtraction; the DOCX-only
indentation claim) are non-blocking.
