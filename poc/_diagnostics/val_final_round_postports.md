# Final consistency validation — post-ports round (revision3)

Scope: strict, read-only final gate. Files audited: `poc/paper/IoT2.html`,
`review_round_1/letter_editor.html`, `letter_reviewer1.html`, `letter_reviewer2.html`,
against `review_round_1/Review1.txt`, `Review2.txt`, `Editor review.txt`.

---

## Check 1 — Number consistency (paper <-> letters) and delta arithmetic — PASS (one rounding note)

Every headline number quoted in a letter matches the current manuscript:

- HAI difficult 0.845; +0.088, 95% CI [0.042, 0.157], P < 0.0005; 167 windows / 26 episodes
  — paper S6/S8/abstract, editor req.1/4, reviewer1 C4, reviewer2 C2. MATCH.
- SWaT difficult raw 0.524 -> typed 0.723; +0.200, 95% CI [0.018, 0.355], P = 0.014; 91 windows / 23 episodes
  — paper S6/S7/Table 3, editor bullet+req.4, reviewer1 C4. MATCH.
- WADI difficult 0.771; vs AE +0.143 (P=0.001), USAD +0.192 (P<0.0005), TranAD +0.157 (P=0.0005),
  IF +0.136 (P<0.0005), GDN +0.111 (P<0.0005); ties LinRes 0.750 +0.020 [-0.161,0.205] P=0.46 / 8 episodes
  — paper S6, editor req.4, reviewer1 C4. MATCH (paper and letters identical).
- Double-hard: WADI 0.763 (+0.021, 8 ep), HAI 0.648 (+0.101, 13 ep), SWaT 0.178 -> 0.671
  (+0.493, [0.349,0.619], P<0.001); windows/ep 29/8, 55/13, 28/9 — Table A4, S6, S7, reviewer1 C4. MATCH.
- Cost: LatAD-global 3.2/16/3.5 s; community 50/195/38 s; TranAD 231/898/163; 47-72x; GPU 20-29 MB;
  host 3.5-3.9 GB; CPU-edge 0.6-14.6 ms global — Table B1, editor req.3, reviewer1 C3, reviewer2 C10. MATCH.
- Nearest-component standalone 0.603/0.797/0.479 (W/H/S) — Table A6, editor req.2, reviewer1 C2. MATCH.
- Localization Table 5: HAI top-1 0.39 vs 0.25 (P=0.024, 15/38), top-3 0.68 vs 0.56; WADI top-1 0.33 vs 0.13,
  top-3 0.56 vs 0.31 (P=0.086, 9 attacks) — S8, editor bullet, reviewer2 C7. MATCH (HAI/WADI mapping correct;
  no live SWaT-localization claim; the old "68 to 79% HAI and SWaT" survives only inside a rev-del).

Base/value/delta tables re-derived:
- Table A5 (source of gain): HAI 0.789-0.589 = +0.200; WADI 0.656-0.639 = +0.017; SWaT 0.472-0.522 = -0.050.
  All three exact. MATCH.
- Table E1 counts 17/17/23 -> 11/9/16, mean 19->12, TPR +0.007±0.068. Exact.
- Table E2 5%: nearest 0.148 < uniform 0.204 < tempered 0.205 < mixture 0.210 < kNN 0.216. MATCH letters.
- Table 3: All +0.092 (0.880-0.788), Difficult +0.199~0.200 (0.723-0.524), Double-hard +0.493 (0.671-0.178). OK.
- Anomaly counts add up: WADI 26+30=56, HAI 485+167=652, SWaT 142+91=233. OK.

Rounding note (NOT a paper/letter conflict, not blocking): the WADI per-detector paired deltas
(+0.157, +0.136, +0.020) differ by 0.001 from a naive subtraction of the 3-dp displayed Table 2 values
(0.158, 0.137, 0.021). Both paper and letters carry the identical figures; the deltas are computed from
full-precision seed-mean AUROCs, so this is expected and internally consistent.

## Check 2 — Newly-added evidence internally consistent — PASS

- Time-aware block length: moving-block 3 (WADI) / 4 (HAI,SWaT); ACF-selected 264 (HAI) / 3 (WADI) / 4 (SWaT);
  "seven headline comparisons keep the same significant-or-tie verdict" (paper S5.4). The 264/3/4 numbers
  appear identically in editor and reviewer1 letters. MATCH. ("seven headline comparisons" phrasing lives in
  the paper only; letters summarise as "verdicts unchanged" — acceptable.)
- Rare-regime sweep (Appendix E / Table E2): nearest 0.148 vs mixture 0.210 at 5%, alternatives in between;
  invariant 0/30 at <=5%, 1/30 at 10%; Table E1 three seeds (labelled three seeds), Table E2 five-seed
  (caption confirms). Editor + reviewer1 letters match. MATCH.
- Raw-sequential: hurts by 0.068/0.230/0.060 (W/H/S) — Appendix A + reviewer1 C5. MATCH. Reviewer1 letter now
  affirmatively describes running it (no "did not run" anywhere; grep clean).
- Drift operating point: recall 0.687 overall / 0.330 difficult at 6.5% FPR (vs 0.918/0.791 untyped);
  24 h sensitivity 0.757/0.723/0.706 at 12/24/48 h — S7 + abstract. MATCH.
- Drift spectrum 1.5/39/69% and typing validation (74.8/88.5/61.9% changepoint; 97.4/92.5% drift; FA 79->6.5%)
  consistent across S7.1, Table 4, abstract. MATCH.

## Check 3 — Coverage of all 22 original comments — PASS (22/22, no point dropped)

Per-comment coverage:
R1-1 A9/A10/A3(->A8) validate/scope = letter_reviewer1 C1 (answered). R1-2 nearest-component sweep = C2.
R1-3 regime-community cost = C3. R1-4 time-aware test + WADI CI/p clarify = C4. R1-5 raw-sequential/temporal
source-of-gain = C5. R1-English/figures/conclusions = final section.
R2-1 abbreviations = C1. R2-2 abstract = C2. R2-3 intro logic = C3. R2-4 "the task" (p2 l71) = C4.
R2-5 two obstacles = C5. R2-6 related-work restructure/delete 2.5 = C6. R2-7 system-model figure = C7.
R2-8 MIIM<->LatAD relation = C8. R2-9 S4 readability + workflow figure = C9. R2-10 per-community fit + cost = C10.
R2-11 z symbol conflict = C11. R2-12 typesetting/numbered eqs = C12. R2-results/English/figures = final.
ED-1 validate A3/A9/A10 = editor req.1. ED-2 nearest-component support = req.2. ED-3 edge cost = req.3.
ED-4 time-aware test = req.4. ED-5 source-of-gain ablation = req.5. (Forwarded R2 items + point-by-point also
addressed.) All resolutions carry a concrete change + location.

## Check 4 — Highlight nesting scan — PASS on the flagged criterion; 3 mirror-direction nestings raised

Full span parse: all rev spans balanced (open-stack leftover = 0).
Green-addition-inside-red-deletion (the corrupting case explicitly named): 0 occurrences. PASS.

Raised for verification (mirror direction — a red rev-del nested inside a green rev2-highlight,
i.e. a tracked deletion inside a tracked insertion): 3 occurrences:
- IoT2.html:170  `<span class="rev2-highlight">... faults that matter<span class="rev-del">, because a flexible decoder reconstructs them faithfully</span>; where a fault ...</span>`
- IoT2.html:275  `<span class="rev2-highlight">... LatAD is built to exploit <span class="rev-del">each property</span><span class="rev2-highlight">each of A1-A7</span> when it is present ...</span>`
- IoT2.html:838-845  `<span class="rev2-highlight">On WADI LatAD reaches 0.771 ... supplies the WADI signal.<span class="rev-del"> On the full set the linear baseline is marginally ahead of LatAD (0.834 vs 0.827).</span></span>`
If the DOCX build maps rev2-highlight -> w:ins and rev-del -> w:del, a w:del nested inside a w:ins is invalid
OOXML and can corrupt tracked changes. Fix: close the outer rev2-highlight before each rev-del and reopen it
after, so the insertion and deletion are siblings, not nested. Verify against the tracked-changes builder.

## Check 5 — Tone — PASS

All three letters open and close with thanks (editor: "grateful for the care ... Thank you again for your
time and guidance"; reviewer1: "careful and constructive reading ... detailed and helpful review";
reviewer2: "thorough reading ... improved the clarity and structure considerably"). Added warmth
("Good point", "a fair thing to check", "the numbers are reassuring") is measured, not effusive. No defensive
or apologetic language; limitations stated plainly. No "honest/frankly/candidly" anywhere (grep clean).

## Check 6 — Stale / style / cross-references — PASS with 2 deprecated-phrasing findings

- em-dash / double-hyphen: none in prose of paper or letters (grep clean; en-dash &ndash; / &minus; used).
- "honestly/frankly/candidly/in truth/to be honest": none.
- "P about 0" / "P approx 0": none (all p-values numeric: P<0.0005, P=0.014, etc.).
- Cross-references all resolve: Tables 1-5, A1-A6, B1, C1, D1, D2, E1, E2; Figures 1,2,3,A1; Eqs (1)-(9);
  Sections up to 9; Appendices A-E. No ref to a non-existent element (no Table 6 / Figure 4 / Section 10 /
  Appendix F). z-score purged from live text (only inside a rev-del at line 1248; live text uses
  "standardized units"). Community counts 44/28/24 and feature dims 732/354/306 consistent across S5.2 and
  Appendix D. SWaT Dec-2015 / WADI April-2017, 122-channel WADI, all consistent paper<->letters.

Two deprecated-phrasing findings (factually correct and consistent paper<->letters, so not number errors,
but the wording the task flagged as deprecated):
- (a) "best overall AUROC" survives in letter_reviewer1.html:186 ("best overall AUROC on HAI"). The paper's
  live text uses the careful form "best All-subset AUROC of any learned detector on HAI (0.948)". Recommend
  aligning the letter to that wording. (On HAI-All, LatAD 0.948 is in fact the max, so the claim is true.)
- (b) "Measured absent" for A10 remains in live Table 1 (caption IoT2.html:286; A10 row IoT2.html:332) and in
  the letters (editor 58/94; reviewer1 65), whereas S8 (IoT2.html:1123) uses the softened, more defensible
  "A10 (path dependence) shows no evidence under these tests rather than merely unobserved." Table 1 and S8
  differ in claim strength; recommend harmonising Table 1 to the S8 phrasing.

---

## FINAL VERDICT

Numbers, arithmetic, evidence blocks, comment coverage, tone, and cross-references are clean. The explicitly
flagged green-inside-red nesting is absent and all spans balance. Remaining items:

1. BLOCKING (verify before DOCX build) — 3 rev-del deletions nested inside rev2-highlight insertions at
   IoT2.html lines 170, 275, and 838-845. Un-nest them (close the green span before each rev-del, reopen
   after) so tracked insert/delete are siblings; a w:del inside a w:ins corrupts DOCX tracked changes.
2. ADVISORY (non-blocking, deprecated wording) — letter_reviewer1.html:186 "best overall AUROC on HAI" ->
   "best All-subset AUROC of any learned detector on HAI (0.948)".
3. ADVISORY (non-blocking, deprecated wording) — "Measured absent" for A10 in Table 1 (IoT2.html:286, 332)
   and letters (editor 58/94; reviewer1 65) -> harmonise with S8's "shows no evidence under these tests
   rather than merely unobserved."

Status: NOT ACCEPT-READY pending item 1 (build-corruption risk); items 2-3 are wording clean-ups.
If the tracked-changes builder flattens rev-del to strikethrough run-formatting rather than true w:del,
item 1 is harmless and the package is ACCEPT-READY with only advisories 2-3.
