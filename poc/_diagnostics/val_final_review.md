# Final acceptance review — IoT-4522779 (revision3)

Read line-by-line against `poc/paper/IoT2.html`, the three response letters, and the
three review files. Role: MDPI *IoT* associate editor, final pre-acceptance check.

## 1-2. Per-comment coverage and adequacy

| ID | Covered (letter + paper)? | Well-addressed? | Letter tone OK? | Stale / issues |
|----|---------------------------|-----------------|-----------------|----------------|
| R1-1 A3/A9/A10 unused | Yes — R1 c1; Table 1 regrouped A1–A7 / A8–A10, "Realized by" col, §8, Appendix C (Table C1) | Yes — A8 renumbered, remeasured in obs-space; A9 measured present; A10 measured absent w/ positive control | Yes | none |
| R1-2 nearest-component support | Yes — R1 c2; Table A6 standalone head, Appendix E (Table E1, 3W) | Adequate but see defect | Yes | **Appendix E prose+caption+both letters say "five seeds" but Table E1 lists only seeds 0/1/2 + mean (3 seeds; "17,17,23"→"11,9,16")** |
| R1-3 cost of community design | Yes — R1 c3; Appendix B, Table B1 | Yes — params/train/latency/GPU-MB/host-RAM on shared A10G + CPU proxy | Yes | none |
| R1-4 time-aware test + WADI CI/p | Yes — R1 c4; §5.4 episode-block bootstrap, §6, Fig 3 | Yes — HAI +0.088 CI[0.042,0.157]; SWaT typed +0.200 CI[0.018,0.355] P=0.014; WADI per-detector + tie w/ LinRes +0.020 P=0.46 | Yes | minor: R1 letter writes "TranAD +0.157 (P < 0.00055)"; paper §6 says "P = 0.0005" |
| R1-5 source of gain | Yes — R1 c5; Table A5, A2, A6, A1 | Yes in structure | Yes | **Table A5 HAI gain: 0.802−0.589 = 0.213, but stated (and cited in letter/main text) as +0.200** |
| R2-1 expand abbreviations | Yes — R2 c1; abstract "LatAD (Latent-density Anomaly Detector)", §1 CPS/IIoT/SCADA | Yes | Yes | none |
| R2-2 abstract highlights LatAD | Yes — R2 c2; abstract rebalanced to mechanism + one result/claim | Yes | Yes | none |
| R2-3 intro logic | Yes — R2 c3; "This paper detects them by…" struck, CPS background opens §1, contributions later | Yes | Yes | none |
| R2-4 "the task" (line 71) | Yes — R2 c4; §1 "Unsupervised anomaly detection for CPS telemetry is often framed as…" | Yes | Yes | none |
| R2-5 two obstacles / does LatAD address both | Covered by live §1 (two obstacles) + C1/C2 | Weak in letter | Yes | **Letter c5 quotes the "LatAD addresses the modeling obstacle…" paragraph as a live addition, but that paragraph (line 175) is `rev-del` = deleted; letter also says evaluation is "adopted, not claimed as a contribution", which conflicts with the paper's C2 that DOES claim the difficulty stratification as a contribution** |
| R2-6 restructure Related Work | Yes — R2 c6; definition moved to §1, three self-descriptions struck, §2.3 rewritten (Deep SVDD/OmniAnomaly/THOC), §2.5 → "Benchmark scale" | Yes | Yes | none |
| R2-7 system/scenario diagram | Yes — R2 c7; Figure 1 + validated triage (Table 5) | Yes | Yes | none |
| R2-8 MIIM↔LatAD relationship | Yes — R2 c8; §3 checklist sentence, §4 "A1–A10 specify what… A1–A7 realized", Table 1 "Realized by" col | Yes | Yes | none |
| R2-9 §4 readability + workflow figure | Yes — R2 c9; Figure 2 (assumption-tagged), §4.1/4.3/4.4 headings name pipeline stage | Yes | Yes | none |
| R2-10 per-community arch + cost | Yes — R2 c10; §4.4 (separate VaDE/community, encoder not shared), Appendix B | Yes | Yes | none |
| R2-11 z notation conflict | Yes — R2 c11; latent keeps z, std(·) operator, trivial value renamed u (Table 2, §5.3) | Yes | Yes | minor: Appendix D.2 (new text) still uses "z-sum"/"z-standardized" |
| R2-12 typesetting + numbered eqns | Yes — R2 c12; equations (1)–(9) numbered; indentation unified at DOCX build | Yes | Yes | none |
| ED-1 validate A3/A9/A10 | = R1-1 | Yes | Yes | none |
| ED-2 nearest-component support | = R1-2 | see R1-2 | Yes | Appendix E seed-count defect |
| ED-3 edge cost / Table B1 | = R1-3 | Yes | Yes | none |
| ED-4 time-aware significance | = R1-4 | Yes | Yes | none |
| ED-5 gain ablation Table A5 | = R1-5 | Yes | Yes | Table A5 HAI +0.200 vs 0.213 |

All 22 comments are covered by both a letter response and a corresponding paper change. No comment is unaddressed.

## 3. Letter tone
All three letters pass. Each opens with genuine thanks (editor: "Thank you for handling
our manuscript and for a clear, actionable set of requests"; R1: "Thank you for a careful
and constructive reading"; R2: "Thank you for the thorough reading…"), stays courteous
throughout, and closes with thanks. The one rebuttal (R1 c2, nearest-component) is a plain
confident rebuttal, not defensive or dismissive. No curt or grovelling passage found.

## 4. Stale content / number audit
Cross-checked every number, table pointer, and section reference in the three letters
against the current paper. Matches confirmed: HAI difficult 0.845, WADI difficult 0.771,
SWaT difficult 0.524→0.723, double-hard 0.671 (+0.493, CI[0.349,0.619], P<0.001), drift
1.5/39/69%, false-alarm 79%→6.5%, cost 3.2/16/3.5 s global vs 231/898/163 s TranAD,
Table 5 (HAI 68 vs 56, WADI 56 vs 31), Table A6 head numbers (0.603/0.797/0.479),
WADI re-derived 43→30 windows. SWaT is consistently presented as a drift-aware win (raw
ties → typed 0.723 / double-hard 0.671 / +0.493). No deprecated "best overall AUROC
HAI/SWaT" framing and no old table number survives.

Two genuine number defects (both echoed in the letters, so they fail a strict audit):
- **Table A5, HAI cross-channel gain:** operands 0.802 and 0.589 subtract to **0.213**,
  but the gain cell, the main-text sentence (§6), and the editor + R1 letters all say
  **+0.200**. Reconcile (either fix the gain to +0.213 or an operand).
- **Appendix E "five seeds":** the prose, the Table E1 caption, and both the editor and R1
  letters say "five seeds", but Table E1 lists only seeds 0/1/2 + mean, and the prose lists
  three values (17,17,23 → 11,9,16, mean 19→12). Add the two missing seeds or relabel as
  three seeds throughout (paper + both letters).

Minor (non-blocking): R1 letter "P < 0.00055" (paper: P = 0.0005); Appendix D.2 retains
"z-sum"/"z-standardized"; R1 letter describes the Conclusion as "best overall AUROC on HAI"
(true of Table 2's HAI All 0.948, but not literally stated in §9).

## 5. Highlighted edits
Additions are marked green (`rev-highlight`/`rev2-highlight`) and deletions red strikethrough
(`rev-del`) throughout — coverage is extensive and the two-colour convention is applied.
**Coherence fails, though:** a nesting-aware scan finds 7 places where a green addition span
is stranded *inside* a `rev-del` deletion (lines 231, 247, 448 ×3, 450, 1188 — e.g. the green
"(SOTA)" inside the struck "LatAD adopts this critique…" paragraph, and the green
"LatAD (global density)" inside the struck CalexNet passage). Green-inside-deletion is exactly
the pattern that produces a contradictory insertion-within-deletion in a DOCX tracked-changes
render. These islands should be folded into their surrounding deletion (struck, not green)
before building `paper_mdpi.docx`.

## FINAL VERDICT
**NOT ACCEPT-READY — 4 blocking items (all quick fixes):**
1. Table A5 HAI cross-channel gain reads +0.200 but 0.802−0.589 = 0.213; reconcile the number in Table A5, §6 text, and the editor + Reviewer 1 letters.
2. Appendix E claims "five seeds" but Table E1 / prose give only three (seeds 0–2); fix the seed count in the paper and in the editor + Reviewer 1 letters.
3. Highlighting coherence: 7 green additions nested inside red deletions (lines 231, 247, 448, 450, 1188) will corrupt the DOCX tracked-changes render; make those fragments part of the deletion.
4. Reviewer 2 letter comment 5 quotes a deleted (`rev-del`) paragraph as the live change and states evaluation is "not claimed as a contribution", which contradicts the paper's C2; re-point the response to the live §1 two-obstacles text and C1/C2.
