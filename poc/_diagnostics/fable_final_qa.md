# Final adversarial QA: IoT2.html + response_letter.md (read-only sweep)

Scope: `poc/paper/IoT2.html` (1266 lines) and `review_round_1/response_letter.md` (repo root, not under `poc/`).
Line numbers refer to IoT2.html unless marked "letter". Every finding carries a verbatim quote, a location, and a concrete fix.
Provenance checks used `_diagnostics/headline_full_HCcoh_{WADI_clean,SWaT_canon}.json`, `_diagnostics/a3_rho_screen.json`,
`_diagnostics/a3_screen_smd_pu.json`, `_diagnostics/rev4_sota_ms.json`, `sota_bundle/sota_scores/`, `build_scores_table.py`.

Summary: 4 BLOCKING groups (scaffolding filenames in §7; revision-journey narration in live prose; letter scaffolding/"pending"/"coordinator note"; five-seed claim for USAD/TranAD not backed on WADI/SWaT), 16 substantive, ~15 minor. Categories 1 (reference order/anchors), 3 (float contiguity/equations) and most of 6/7 are clean.

---

## BLOCKING

### B1. Internal diagnostic filenames and the "fable" codename rendered in live prose (cat 5)
- Line 1047 (§7, live green paragraph): `... every latent or PCA dimension we tested (<code>_diagnostics/fable_a3_spaces.md</code>)` and `... (<code>_diagnostics/a3_rho_screen.json</code>, <code>a3_screen_smd_pu.json</code>) finds the same absence`.
- Line 1049 (§7): `... on every seed (<code>_diagnostics/fable_a3_spaces.md</code> &sect;7)`.
- Fix: delete all four `<code>` path references (and the trailing "§7" that points into an internal document). If a pointer is wanted, say "(supplementary analysis)" or nothing.

### B2. Revision-journey narration in LIVE (non-rev-del) prose (cat 8) and dead-end reporting (cat 9)
Each item: quote, location, forward rewrite.

1. Line 590, §4.3(iv), live green: `An earlier version of the stack carried an optional basin-agreement head meant to flag between-regime overlap (A8), gated by ... That overlap was not observed in any dataset studied ... The head is therefore withdrawn from the shipped model, so Equation (7) is the detector's complete window score.`
   Rewrite: "Equation (7) is the complete window score. No head addresses between-regime overlap (A8): the train-normal fraction of ambiguously assigned windows (maximum responsibility below 0.5) is ρ ≤ 0.02 on WADI, HAI and SWaT (0.01 / 0.02 / 0.00), and A8 is not observed in any dataset studied (§7)."
   Also drop the rev-del bracket sentence `[An optional basin-agreement rescue head ... was described here.]` from the marked copy if a clean build is produced; it is an editorial placeholder, not struck manuscript text.

2. Line 705, §5.2 Model configuration: `The overlap head of &sect;4.3 is withdrawn from the shipped model and carries no configuration here; the train-normal ambiguous-window fraction that would have gated it is &le; 0.02 ...`
   Rewrite: delete (the fraction is already in §4.3), or "No overlap head is configured; the train-normal ambiguous-window fraction is ≤ 0.02 (0.01 / 0.02 / 0.00 on WADI / HAI / SWaT)."

3. Line 963, §6 "Where the gain comes from": `the earlier +0.343 WADI figure was an artifact of the rescaled <code>2B_AIT_002_PV</code> channel, now removed (&sect;5.1)`.
   Rewrite: delete the clause. A first-time reader has never seen +0.343; it exists only in the response letter. Keep: "WADI's difficult anomalies are single-channel or linear, so per-channel marginals already suffice."
   Same paragraph: `on the artifact-free WADI difficult subset` → "on the WADI difficult subset".

4. Line 767, Table 3 caption: `GDN is omitted here: no run on the canonical SWaT stream is available within our compute budget, and its earlier mirror-SWaT run is not comparable to the canonical figures reported here (&sect;5.5).`
   "mirror-SWaT" is never defined anywhere in the paper and leaks the pre-correction data setup. Rewrite: "GDN is not benchmarked (§5.5)."

5. Lines 245-247 (§2.4) and 746-748 (§5.5), both live green: `We also attempted the cross-channel GDN under the identical raw protocol, but its per-window graph training did not complete on the canonical SWaT stream within our compute budget, so it is not reported (&sect;5.5).` and `We also attempted the graph-based GDN [37] under the identical protocol; its per-window graph training did not complete ...`
   Cat 9 (attempt/dead-end reporting, stated twice). Rewrite once, in §5.5 only: "GDN [37] is outside the comparison: its per-window graph training exceeds our compute budget on the SWaT stream." Delete the §2.4 sentence.

6. Line 1020, Table 6 caption: `The overlap head (A8, &sect;4.3) is withdrawn from the shipped model and is not shown.` → delete.

7. Line 446, §4 preamble: `A8 (between-regime overlap) is specified for completeness but was not observed in the datasets studied: the shipped model carries no dedicated overlap head, and &sect;7 shows that the earlier apparent SKAB signal attributed to A8 is a variance-floor artifact of the estimator. It is left as future work with A9&ndash;A10.`
   Rewrite: "A8 (between-regime overlap) is specified for completeness; it is not observed in the datasets studied (§7), so the model carries no overlap head, and it is left with A9–A10 to future work."

8. Line 317, Table 1 A8 row: `Across eight real CPS datasets this overlap was not observed (responsibility entropy &le; 0.06); the earlier SKAB signal is a variance-floor artifact of the estimator on n = 400 (&sect;7).`
   Rewrite: "Not observed across the real CPS datasets screened (responsibility entropy ≤ 0.06; §7)." (see S3 for the count and the 0.06 bound). Line 318 `not realized by the shipped model` is acceptable.

9. Line 1047, §7 bold lead: `Assumption A8 (between-regime overlap) was not observed, and the earlier SKAB witness is an estimator artifact.`
   Rewrite: "Assumption A8 (between-regime overlap) is not observed."

10. Line 1049, §7, whole paragraph: `The one dataset that appeared to exhibit A8, SKAB [49] (a 400-window rotor rig), does not survive scrutiny. Its high responsibility entropy (0.29) and the apparent basin-head lift on the difficult subset are a variance-floor artifact ... manufactures the overlap ... both the &rho; gate and the basin-head lift collapse to about zero ... The overlap head is therefore withdrawn: A8 is specified for completeness but not demonstrated on any dataset in this study ... Because the head was auto-gated off (&lambda; = 0) on WADI, HAI, and SWaT, its removal changes no benchmark number in Tables 3&ndash;5.`
   Rewrite (forward statement of fact): "The responsibility-based overlap statistic is confounded by the estimator's variance floor on small samples. On SKAB [49] (400 training windows) a VaDE fitted with the default component log-variance floor of log(0.05) reports responsibility entropy 0.29; refitting the same latent with empirically estimated component variances, or a floor of 0.01, gives 0.045–0.048 on every seed, indistinguishable from WADI (0.053–0.057). Small-sample entropy is therefore not evidence of A8, which remains unobserved on every dataset studied and is left to future work with A9–A10."
   This also removes the undefined "basin-head lift", "ρ gate" and "λ = 0", and the "changes no benchmark number in Tables 3–5" changelog sentence.

11. Line 1105, §7: `We now report these measurements (Table&nbsp;7), benchmarked on one shared machine plus a CPU edge proxy.`
    Rewrite: "Table 7 reports training and inference cost, benchmarked on one shared machine plus a CPU edge proxy."

12. Lines 1086-1088, §7: `The community construction and combiner settings were fixed before the artifact-free WADI and canonical SWaT evaluations reported here, so those two datasets are effectively out-of-sample confirmations of a design frozen on their earlier versions.`
    "their earlier versions" narrates the correction. Rewrite: "The community construction and combiner settings were fixed before the WADI and SWaT evaluations reported here." (Drop the "out-of-sample" claim: the design was developed on the same two plants, so a reviewer will not accept "out-of-sample".)

13. Line 593, auto-gating note: `The shipped model has no overlap head: A8 (between-regime overlap) was not observed in any dataset studied (&sect;7).` → "No overlap head is used: A8 is not observed (§7)." (minor wording, listed for completeness).

14. "artifact-free WADI" / "cleaned WADI" presuppose a prior dirty version: lines 963 (×1), 967 Table 5 caption (`on the artifact-free WADI, canonical SWaT, and HAI difficult subsets` and `on cleaned WADI it is essentially zero`), 1087, 1133 (`WADI results are on the artifact-free 122-channel set`).
    Rewrite: "WADI (122 channels, §5.1)" / "on WADI". §5.1 line 669 `it is dropped` → "it is excluded" (already a factual scope statement; fine otherwise).

15. Line 446-448, §4: `The gated residual head did not reduce performance on any of these datasets, and the reported gains come from the always-active heads, with the auto-gated residual head adding to them on HAI and SWaT (Table 6).`
    Defensive/non-result framing (cat 7/9). Rewrite: "The reported gains come from the always-active heads; the auto-gated residual head adds to them on HAI and SWaT (Table 6)."

### B3. Response letter: scaffolding, "pending" items, and stale cross-references (cat 5, letter)
- Letter line 82 (R1-Eng): `A dedicated language pass ... is applied to the submitted version. (See coordinator note.)` and `Change made in part; full pass pending.`
- Letter line 142 (R2-12): `indentation pending in the DOCX build ... (See coordinator note.)`
  Fix: remove "(See coordinator note.)" and "pending"; state what was done ("A full language pass was applied"; "Paragraph indentation is unified in the submitted DOCX").
- Letter line 9: `our internal action tags (A-##, B-##) are cited so cross-references between comments are traceable` plus every heading tag: `(items A-12, A-13, B-01)` (25), `(item B-05 / Ed-near)` (39), `(items B-03, A-30 / Ed-cost)` (47), `(items A-22, A-23, A-24 / Ed-stat)` (59), `(items B-04, A-25, A-29 / Ed-abl)` (72), `(item A-01)` (96), `(item A-02 / Ed-abs)` (100), `(item A-03)` (104), `(item A-04)` (108), `(item A-05)` (112), `(items A-06, A-08, A-09, A-10 / Ed-RW)` (116), `(item A-16 / Ed-fig)` (120), `(item A-14)` (124), `(items A-15, A-17 / Ed-fig)` (128), `(items A-21, A-30, B-03)` (132), `(item A-18 / Ed-z)` (136), `(items A-19, A-20 / Ed-eq)` (140), `Scope stated (A-12)` (31), `(B-01)` (33), `(A-23, A-24, §6)` (67), and the `(Ed-A)` ... `(Ed-resp)` suffixes on E1-E11.
  Fix: strip every internal tag; keep the R1-n / R2-n / E-n numbering only.
- Letter lines 33, 154: `` `_diagnostics/fable_a3_spaces.md` §7 ``; line 35: `` `_diagnostics/a3_rho_screen.json`, `a3_screen_smd_pu.json` ``; line 76: `` `_diagnostics/e5_gain_clean.json` ``. Fix: delete; a reviewer cannot open them and "fable" is a codename.
- Letter lines 43, 45, 158: `Table 5 reports nearest-component NLL as a standalone head (WADI 0.719, HAI 0.797, SWaT 0.723)`, `the Table 5 parity result`, `isolated in Table 5` → these numbers are in **Table 6** (line 1025). Line 78 already says "head-level ablation (Table 6)", so the letter contradicts itself.
- Letter line 118: `now cites Deep SVDD, OmniAnomaly, and THOC (refs 43-45)` → they are [28], [29], [30].
- Letter line 78: `the benchmark literature [ref 9] shows SWaT/WADI anomalies are univariate` → that is [20] (Pinet et al.); [9] is Wu and Keogh.
- Letter line 54 `17-52 ms for the full regime-community stack` (batch 1) vs line 162 `17-32 ms at batch 1, up to ~52 ms at batch 256` and paper line 1108. Fix line 54 to "17-32 ms (batch 1)".
- Letter lines 45, 158 (R1-2 / E2): `We therefore support it with a controlled imbalance sweep (mixture-NLL vs nearest-component NLL vs imbalance-aware variants, measuring the false-positive rate on rare normal regimes at π < 2%), with the invariant that ...` and `it is supported by a controlled imbalance sweep`. No number from this sweep is given in the letter or the paper. Either report the sweep's result (numbers, seeds) or reword as a design argument plus the Table 6 parity: "supported by the Table 6 parity and the A5 design argument".
- Letter line 78: `A prior temporal-feature run gave no improvement (WADI 0.726 to 0.672)`: 0.726 matches no current table (global density is 0.734); it is a pre-correction number. Drop it or label it as pre-correction.
- Letter line 9: `Green highlighting in the revised manuscript marks the new text` but the manuscript also carries yellow (`rev-highlight`) and red strikethrough (`rev-del`). Add one legend sentence: green = this revision, yellow = IIoT-scope text added in the previous submission, red strikethrough = deleted.

### B4. "Five seeds" claimed for USAD/TranAD, but the WADI and SWaT rows are single-seed (cat 1/2)
- Line 244-245 (§2.4): `We benchmark USAD and TranAD ... under the same raw, difficulty-stratified metrics over five seeds`.
- Line 742-743 (§5.5): `we re-run USAD [35] and TranAD [36] through the TranAD evaluation harness over five seeds`.
- Line 765-767 (Table 3 caption): `Learned detectors report AUROC as mean&plusmn;standard deviation over five seeds`.
- Evidence: Table 3 WADI/SWaT USAD/TranAD cells carry no ± (lines 787-788, 807-808) while the HAI cells do (797-798); Figure 3 draws USAD/TranAD whiskers only on HAI; `headline_full_HCcoh_{WADI_clean,SWaT_canon}.json` store USAD/TranAD with sd 0.0; `sota_bundle/sota_scores/` holds one vector per dataset (`score_USAD_WADI_clean.npy`, `score_USAD_SWaT.npy`); `build_scores_table.py` line 7: "SOTA (USAD/TranAD/GDN) are single per-window vectors"; only `rev4_sota_ms.json` (HAI, plus pre-correction WADI/SWaT) has n_seed 5.
- Fix: either re-score USAD/TranAD over five seeds on the 122-channel WADI and canonical SWaT streams (then print ±), or state the truth in §5.5 and the Table 3 caption: "USAD and TranAD are five-seed on HAI and single-seed on WADI and SWaT". The abstract/§1 "five-seed mean" statements are about LatAD and stay true.

---

## SUBSTANTIVE

### S1. SWaT confidence-interval bounds are not those of the persisted artifact (cat 2, verify-before-report)
- Paper: line 829 `(+0.058, 95% CI [0.015, 0.107], P&nbsp;=&nbsp;0.003)`; lines 864-865 and 880-881 `+0.087 (95% CI [0.029, 0.151], P&nbsp;=&nbsp;0.0005)`; Figure 3 caption line 960 and §7 line 984, §8 line 1149 repeat P = 0.003. Letter lines 66, 68 repeat both CIs.
- Artifact that backs every other WADI/SWaT number in Tables 3-4 (`headline_full_HCcoh_SWaT_canon.json`, `significance`): Difficult diff 0.058, CI **[0.017, 0.107]**, P 0.0025; DoubleHard diff 0.087, CI **[0.030, 0.153]**, P 0.0005. No saved artifact contains [0.015, 0.107] or [0.029, 0.151] (`headline_HCcoh_SWaT_canon.json` gives 0.045 / [0.000, 0.097]; `headline_full_SWaT_canon.json` gives 0.040 / [−0.001, 0.092]). WADI's [−0.088, 0.181] / P 0.32 and double-hard [−0.134, 0.267] / P 0.22 match the artifact exactly.
- Fix: print the artifact values ([0.017, 0.107] and [0.030, 0.153]) in paper and letter, or persist the bootstrap run that produced the printed bounds. The conclusion (significant) is unaffected.

### S2. A8 statistics disagree with each other inside the paper (cat 1)
- (a) Ambiguous fraction: line 590 and 705 say `&rho; &le; 0.02 ... (0.01 / 0.02 / 0.00)`; line 1047 says `ambiguous fraction &rho; &le; 0.05` with `mean maximum responsibility 0.94 / 0.85 / 0.99`. `a3_rho_screen.json` gives rho 0.01 / **0.049** / 0.0 and mean_maxresp 0.944 / 0.846 / 0.989, so §7 is from that screen and §4.3/§5.2 (HAI 0.02) are from a different run. Print one HAI value.
- (b) Responsibility entropy for WADI: line 1047 `stays near 0.03 / 0.08 / 0.01` vs line 1049 `indistinguishable from WADI&rsquo;s 0.053&ndash;0.057`.
- (c) Line 1047 `responsibility entropy &le; 0.06 throughout` (and Table 1 A8 row `&le; 0.06`) while the same paragraph gives HAI 0.08. `a3_screen_smd_pu.json` only covers SKAB, SMD×4, Paderborn (max 0.056); `a3_rho_screen.json` has no entropy field for the benchmarks. Either raise the bound to ≤ 0.08 or restrict "throughout" to the screened non-benchmark datasets.
- (d) Line 1047 `A broader eight-dataset scarcity screen (MetroPT, four SMD server-machine entities, and the Paderborn bearing testbed alongside the three benchmarks ...)` lists 1 + 4 + 1 + 3 = **nine** datasets (letter line 35 repeats "eight"). `a3_rho_screen.json` shows TEP, BATADAL and PSM errored (file not found), so they must not be counted. Fix the count to nine, or state "six additional datasets".

### S3. "Overlapping regimes" in §3 vs "overlap not observed" in §7; factorization-gain claim overstated for HAI (cat 1/4)
- Line 337-341: `The regimes overlap on WADI and HAI (silhouette 0.06 and 0.08 ...) ... the two overlapping datasets are where the regime-community factorization gains most ... and the more separable SWaT gains less but still improves (&sect;6).` Line 963: `most on the overlapping-regime datasets (&sect;3)`.
- Table A1 factorization lift (global → full per-community): WADI 0.734→0.824 (+0.090), HAI 0.811→0.849 (+0.038), SWaT 0.804→0.840 (+0.036). HAI and SWaT gain the same; only WADI is "most".
- The word "overlap" here collides with §7's "We do not observe [overlap] on any dataset studied".
- Fix: "Regimes are less separated on WADI and HAI (silhouette 0.06 and 0.08) than on SWaT (0.29); the factorization gains most on WADI (+0.09 difficult AUROC) and about +0.04 on HAI and SWaT (Table A1)." Apply the same to line 963.

### S4. Wrong assumption range in §7 (cat 1, renumbering)
- Line 1069: `The reported model is window-only and leverages the instantaneous-structure assumptions A1&ndash;A8.` Line 443 and Table 1 say the model realizes A1–A7 and does not realize A8. Fix: "A1–A7".
- All other A-tags were checked line by line against the new numbering (A1 mixture, A2 explosion, A3 thin fringes, A4 few levers, A5 heavy tail, A6 hidden regimes, A7 mixed signals, A8 overlap, A9 clocks, A10 path): lines 274, 443-445, 459, 535, 550, 560, 603-604, 632, 691, 700, 1000-1001, 1079, 1159, 1177, 1180, Figure 2 tags and caption, Table 1 "Realized by" column are all consistent. No other misattribution found.

### S5. Stale Easy range for the deep detectors (cat 2)
- Line 898: `even while they remain strong on Easy (0.90&ndash;0.97)`. HAI Easy is USAD 0.969 / TranAD 0.968; nothing is 0.90. Fix: "(0.97)".

### S6. SOTA expansion lives only inside struck text (cat 1)
- Line 230, inside `<span class="rev-del">`: `state-of-the-art (SOTA)`. First live use is line 239 `These define a widely used deep SOTA` and the §5.5 heading. Fix: expand at line 239.

### S7. §2.5 becomes an orphan sentence once deletions are applied (cat 3/1)
- Line 256-257: the `2.5 Benchmark datasets` heading and its paragraph are rev-del, leaving only `(Dataset details are given in Section 5.1.)` live under §2.4. Fix: delete that sentence too (Table 3/§5.1 already cover it) or append "Dataset details are in §5.1." to the end of §2.4. Also "Section 5.1" vs the "§" style used everywhere else.

### S8. Table 7 AutoEncoder training time "0.0 / 0.0" and "—" placeholders (cat 2/7)
- Line 1127: `<td>0.2 / 0.0 / 0.0</td>` reads as unmeasured. Report with resolution ("<0.1") or as measured seconds.
- Lines 1120, 1125-1126: `Deep-baseline CPU-edge latency was not measured (&mdash;).` and `&mdash;` cells: em-dashes (house rule) used as a symbol. Fix: "n/m" (not measured) and define it in the caption.

### S9. "Between-regime pocket" survives as an observed fault type while A8 says between-regime overlap is unobserved (cat 1)
- Line 88 `a state that falls between two legitimate regimes`; line 162 `a between-regime pocket`; line 500 `(correlation breaks and between-regime pockets)`; line 993 `an improbable between-regime or in-regime-tail combination`. Table 1 no longer carries the old "hard envelopes / thin pockets" rationale, so this fault type is now unanchored, and §7 says "we do not observe this on any dataset studied".
- Fix: in §4.2 (line 500) and §7 (line 993) say "correlation breaks and low-density pockets"; keep "between two legitimate regimes" in §1 only as the general motivation, or qualify it as hypothetical.

### S10. Table 3 ± formatting contradicts its caption (cat 1)
- Caption line 766: `the standard deviation is omitted where it rounds to 0.000`, yet `0.999&plusmn;0.000` (line 785), `0.996&plusmn;0.000` (805), `0.999&plusmn;0.000` (810) and Table 6 `0.835&plusmn;0.000` (1023) are printed, while HAI AE Easy `0.980` (795) and TranAD Easy `0.968` (798) omit it. Harmonize (print ±0.000 everywhere for five-seed rows, or omit everywhere).

### S11. Table 5's "cross-channel latent density" arm is not identifiable (cat 1/4)
- Line 970: `cross-channel latent density 0.743&plusmn;0.013 / 0.789&plusmn;0.018 / 0.795&plusmn;0.010` matches no Table 6 row (base 0.743 / 0.801 / 0.775; density 0.758 / 0.802 / 0.794). Fix: name the configuration in the caption (which heads, global single-VaDE, seeds) so the reader can reconcile it with Table 6.

### S12. Table 2 covers WADI and HAI only (cat 4)
- Line 512: `<th>WADI</th><th>HAI</th>`; caption says "single trained model per dataset" and the §4.2 prose then argues about SWaT gating. Add a SWaT column or state that SWaT is covered by Table 6.

### S13. Robustness paragraph reports the base ablation, not the headline detector (cat 4)
- Line 853: `the single-latent LatAD (global density) still leads every baseline (0.675 ...)` on the 110-window HAI subset. The headline detector's number is missing. Add it or say why only the global model is reported.

### S14. Apologetic / pre-emptive sentences (cat 7)
- Line 1083-1085: `the regime-community realization should be validated on additional, untouched CPS systems before its per-dataset gains are treated as established` → "Validation on additional CPS systems is the next step; build and calibration use train-normal data only, so the extension is directly testable."
- Line 1097-1098: `This is a deployment architecture rather than an experimentally evaluated configuration.` Contradicts the following paragraph and Table 7, which do evaluate cost. Delete.
- Line 175: `which we use rather than claim as a contribution` → "which we adopt".
- Line 1051-1056 "The residual frontier on WADI": `No evaluated snapshot detector, ours or the deep baselines, separates these at a low false-alarm budget` is a negative result (cat 9). If kept as a limitation, shorten to one forward sentence: "A minority of WADI difficult windows carry near-absent instantaneous fault content; reaching them requires information beyond the window (process residuals or temporal features), left to future work."

### S15. "Tables 3–5" and undefined λ (cat 1)
- Line 1049: `changes no benchmark number in Tables 3&ndash;5` (results tables are 3–6 and A1) and `auto-gated off (&lambda; = 0)` (λ never defined). Both disappear with the B2.10 rewrite.

### S16. Response-letter internal inconsistencies not already in B3
- Letter line 86: `Tables 3-7 and A1 were regenerated on the corrected, construct-matched data` is fine, but line 118's "refs 43-45" and lines 43/45/158's "Table 5" show the letter was not re-synced after the final renumber; re-read the whole letter against the final table/ref numbers after the paper fixes land.

---

## MINOR

- M1. Em-dash in live prose, line 1047: `0.03 / 0.08 / 0.01 &mdash; at or below the no-overlap floor` → comma or colon. (Table 7 dashes: see S8.)
- M2. Line 177 nested parentheses: `&sect;7 (<span ...>A8 (between-regime overlap), specified but not observed ...; the trajectory assumptions A9&ndash;A10 left to future work</span>)` → "§7: A8 (between-regime overlap) is specified but not observed; A9–A10 are left to future work."
- M3. "canonical SWaT" in the abstract (line 82), §1 and §8 (lines 1149, 960, 984): undefined for a first reader until §5.1. Use "SWaT" outside §5.1/§6.
- M4. Abbreviations: NLL appears in Table 1 (line 306 `Nearest-component NLL head`) before its expansion at line 499; KDE (534), KL (490), ICS (235), GNN (238), SMD (1047) never expanded.
- M5. Keywords line 84: "IoT anomaly detection" and "anomaly detection" both listed.
- M6. "left to future work" appears six times (247, 322, 450, 700, 1049, 1056); vary or consolidate.
- M7. Equation (1) MathML is `display="inline"` (line 262) while (2)–(9) are `display="block"`; the eqno span precedes the math in DOM order for all nine (CSS grid places it right in HTML; check the DOCX render puts numbers on the right).
- M8. Figure 2 SVG legend (line 438) tags A7 twice (`[A7] correlated channels / subsystems` and `[A7] auto-gated whitened residual`); consistent with Table 1 but reads as a typo. Consider "[A4/A7]" on the community block.
- M9. Struck (rev-del) text carries old reference numbers with no anchors: line 203 `[1, 9, 5]`, line 257 `[38]`, `[39]`, `[40]`, `[6, 8]`. In the marked copy [38]–[40] now visually point at graphmoe/SensitiveHUE/CATCH. `edit_pass_report.md` records this as deliberate; acceptable only if the reviewer copy is understood as a diff. Renumber inside the struck text if cheap.
- M10. Line 1097 vs 1105: "deployment architecture rather than an experimentally evaluated configuration" immediately followed by measured costs (see S14).
- M11. §7 coverage paragraph (lines 1058-1066) reports 0.949→0.965, 0.849→0.890, TPR 0.789→0.916 and "every detector improves" with no table; acceptable as a secondary view, but "every detector improves" is unbacked. Either add a one-line table or drop that clause.
- M12. Table 2 caption (line 507-510) is long and argumentative for a caption; the same text is repeated in the prose at lines 521-526. Trim the caption to the table's content.
- M13. Line 336 `on a component grid extended to 25` with K* = 25 on SWaT means the optimum is at the grid edge; say so or extend the grid.
- M14. Letter line 1 header "Response to Reviewers — Manuscript IoT-4522779" uses an em-dash.
- M15. Doclinks div (line 68-72) references `IoT2_cover_letter.html`; it is `no-docx no-print`, so it does not reach the DOCX. Fine; noted so nobody renames it into visible text.

---

## CLEAN (checked, nothing to fix)

- References: 53 entries, every `<a href="#r-...">N</a>` matches its list position, first-appearance order is strictly increasing, no uncited entries, no missing anchors (script check).
- Floats: Tables 1, 2, 3, 4, 5, 6, 7, A1 each captioned once and referenced in prose (Table 1 l.269; 2 l.501; 3 l.757; 4 l.861; 5 l.963; 6 l.448/511/997/1031; 7 l.1105; A1 l.963/1010/1174). Note: the coordinator's expected list "1,3,4,5,6,7,A1" omits Table 2, which exists and is correctly used; numbering is contiguous. Figures 1, 2, 3 each referenced (l.92, 346, 821); no orphans, no duplicates.
- Equations: nine display equations numbered (1)–(9), all present.
- Figure 3 bar heights decode to Table 3 difficult values within ±0.003 for all 18 bars; whiskers only where Table 3 prints ±.
- Table A1 aggregation rows match `headline_full_HCcoh_*` (sum 0.790/0.812, max 0.811/0.803, HC 0.812/0.829, HCcoh 0.824/0.840, null+HC 0.806/0.823). Headline rows in Tables 3 and 4 match `HCcoh+LatAD` exactly (WADI 0.864/0.996/0.824/0.743; SWaT 0.941/0.999/0.840/0.775) and WADI CIs match the artifact.
- Abstract numbers (0.864, 0.949, 0.941, 0.849, 0.44–0.48) are all backed by Table 3; subset sizes (WADI 56 = 13 + 43; HAI 652 = 485 + 167; SWaT 233 = 148 + 85; Table 5 "43 / 85 / 167"; Table 4 27/10, 84/19, 59/18; episodes 11 / 26 / 23) agree across prose, tables and captions.
- Stale pre-correction values: none of 0.796, 0.690, 0.862, +0.12, 0.30–0.48, 0.500, 0.605, ρ = 0.58, "single episode", "ceiling" (only the corrected "rather than ceilings"), "validated on SKAB", "near-significant" survive. 0.993 / 0.969 / 0.605 hits are legitimate different cells (LinRes WADI Easy AUROC; F1 cells). The only +0.343 is the changelog clause in B2.3.
- No `<!-- -->` comments, no TODO/FIXME/placeholder, no A-##/B-##/R1-/E# tags, no "registry"/"agent", no `.npz`/`e5_gain`/`a3_witness` in the manuscript.
- AI-tone fillers: none of "plays a crucial role", "it is important to note", "in today's", "In conclusion", "Note that", "we believe" found. No "honestly/frankly/candidly", "we acknowledge", "admittedly", "it should be noted", "seems to/appears to/arguably". No literal "—" or "--" outside the &mdash; entities listed (M1, S8, M14).
- Cost arithmetic: 231/3.2 = 72, 163/3.5 = 47 → "47–72×" correct; latency, parameter and memory ranges in the prose match Table 7.
- Data Availability, author contributions, and MDPI back-matter present; Zenodo DOI stated (not verified online).
