# Round-2 review fixes applied (2026-09-18)

Scope: HTML-only edits to `poc/paper/IoT2.html` and the three letters in `review_round_1/`
(`letter_editor.html`, `letter_reviewer1.html`, `letter_reviewer2.html`, `letters_summary.md`).
No DOCX rebuild, no pandoc. Change-marking: new/changed text in `rev2-highlight` (green), deletions in
`rev-del` (red strike), prior yellow `rev-highlight` left intact.

Self-check (all pass): 0 em-dashes / prose double-hyphens introduced; 0 "honestly/frankly/candidly";
"Apartsin" spelled correctly (2x per file); span tags balanced (237/237 paper); `<p>` 97/97; `<caption>`
10/10. SVG: Figure 1 identical, Figure 2 changed only by the one sanctioned label rename, Figure 3 untouched
by me (its changes are the other agent's regeneration). MathML: one inline `<math>ρ≤0.02</math>` pair removed
as a byproduct of a sanctioned sentence deletion (balanced open+close, delta -1 pair, `<math` 54/54); all other
MathML byte-for-byte preserved.

## Resolutions (overrides) applied
1. **WADI "significantly ahead of every learned detector" KEPT + per-detector bootstrap added** (§6 WADI
   paragraph). Added: vs AutoEncoder +0.142 (P=0.001), USAD +0.192 (P=0.000), TranAD +0.157 (P=0.0005),
   Isolation Forest +0.136 (P=0.000), GDN +0.111 (P=0.000); ties only LinRes 0.750 (+0.020, CI [-0.161,0.205],
   P=0.46, 8 episodes). Same support propagated to R1 letter comment 4 and letters_summary.
2. **Table A2 WADI column** -> stats 0.634±0.011, temporal 0.626±0.017, temporal-minus-stats -0.009 (was
   0.673/0.675/+0.002). §5.2 prose WADI "(+0.002)" -> "(-0.009)". HAI/SWaT unchanged.
3. **WADI seed SDs added** to Table 3 (IF 0.725±0.005 / 0.829±0.007 / 0.634±0.006; AE 0.792±0.002 /
   0.981±0.001 / 0.628±0.004; LatAD-global 0.717±0.006 / 0.813±0.016 / 0.634±0.011; LatAD reg-comm 0.827±0.009
   / 0.892±0.009 / 0.771±0.023) and Table 4 DoubleHard (IF 0.530±0.009, AE 0.535±0.004; LatAD rows already had
   SDs). Single-run methods (trivial, LinRes, USAD, TranAD, GDN) keep no SD.
4. **Data Availability / Zenodo statement: NOT touched** (verified byte-identical).
5. **Abstract: NOT touched** (verified byte-identical).
6. **Figure 3: NOT touched** (verified: my only SVG edit is the Figure 2 label).

## Paper accuracy fixes
- **§1 contribution 3**: "best overall AUROC ... on `every dataset`" -> struck, "on HAI and SWaT" (green).
- **Table A1 mislabel (HIGH #3)**: row-2 label -> "cohesion-weighted HC, no whole-plant-expert fusion";
  row-3 label -> "cohesion-weighted HC + whole-plant expert (headline)". Appendix A prose rewritten: the
  WADI 0.795->0.771 drop is the whole-plant-expert fusion (not a per-community residual head / not
  "whitened-residual fusion counterproductive"), which no longer contradicts §4.3. §6 and §7 "HC_coh 0.795"
  references reworded to "cohesion-weighted Higher Criticism ... over the community experts".
- **Table 1 caption (#5)**: "specified for completeness but not observed" -> "specified but not realized by
  the reported window-only model (A8 precondition present but heavy overlap not exercised, A9 present in the
  data, A10 measured absent)".
- **§5.4 (#7)**: "From 2000 resamples (1000 on HAI)" -> "From 2000 resamples" (HAI double-hard P=0.0005=1/2000).
- **§6 coverage paragraph (#9)**: cross-pipeline delta reframed as a single secondary-pass view; "every
  detector improves" deleted; misplaced adverb fixed ("Excluding that regime, a label-free operation that
  removes 18.4% ...").
- **§7 "Auto-gating adapts one architecture" paragraph (#10 + readability 14)**: struck whole paragraph
  (`<p class="rev-del">`), removing the "WADI faults are pure low-density pockets" contradiction; its content
  already lives in the Table 6 paragraph above.
- **§7 Benchmark-scale (#13)**: "single nominal regime ... few independent clocks" -> "one nominal operating
  mode (though the mixture discovers many sub-regimes) ... little between-regime masking and short
  trajectories"; "Verifying A8 through A10 ... become testable" -> "Exercising A8 through A10 on
  history-dependent faults".
- **§7 deployment (P1/P15d)**: "This is a deployment architecture rather than an experimentally evaluated
  configuration." struck; replaced with "The training and inference cost of this configuration is measured in
  Appendix B (Table B1)."
- **§6 "Easy 0.90-0.97" (P15a)** -> "Easy 0.96-1.00" (matches Table 3).
- **§6/§3 "SWaT gains less" / "overlapping-regime" (#11)** -> "gains most on WADI, and by a similar margin on
  HAI and SWaT"; §3 "regime ridge" parenthetical deleted.
- **50-percent vs 30-percent control (P15b)**: §7 A8 paragraph now uses the 30-percent control (consistent
  with Appendix C / Table C1).
- **Conclusion (#21)**: future-work list "(A9, A10)" struck -> "(A9, A10, and the masking form of A8 coupled
  to A10)".
- **§4 preamble A9/A10 (#14)**: "are not exercised" -> "only weakly exercised (A9) or measured absent (A10)".
- **Appendix A / §5 WADI mechanism (#19, partial)**: Appendix A "sparse coordinated-drift faults" ->
  "difficult faults confined to small correlated subsystems". (§7 line-1050 mechanism sentence left as-is: it
  describes locality/dilution, which is consistent with Table 5's single-channel/linear reading; see NOT
  changed.)

## Paper readability fixes
- **§7 A8 paragraph replaced** (~450 -> ~150 words), forward-only: no "evidence earlier read as A8", no
  "SKAB signal that appeared to witness it", no "shipped/earlier version"; 30-percent control; ends "The
  reported model carries no dedicated overlap head."
- **Appendix C** rewritten forward: "An earlier version of this screen ..." -> "Cluster responsibility ... is
  not a valid overlap measure ..."; SKAB paragraph de-narrated (header "SKAB.", no "does not survive
  scrutiny", "no longer framed as a near-witness").
- **"no A8 head" restatements reduced** to ~3 (Table 1 A8 row, §4.3 iv, §7): deleted from §1 scope note
  (parenthetical struck), §3 line-272 clause (struck), §3 silhouette parenthetical, §4 opener (rewritten to a
  single "specified in Table 1, absent, deferred" sentence), §4.3 iv (trimmed to one sentence, ρ≤0.02 math
  removed), auto-gating note (A8 sentence deleted), Table 6 caption (A8 sentence deleted). Abstract left
  untouched per resolution 5.
- **"43-to-30 ... now fixed" clause**: deleted (was §6).
- **§1 scope-note A8/A9/A10 parenthetical**: struck (arrives before §3).
- **§6 difficult-subset paragraph split by dataset** (HAI / SWaT / WADI / closing mechanism).
- **HC_coh renamed** in prose to "cohesion-weighted Higher Criticism" (2 §7 spots; 0 "HC_coh" remain).
- **"null expert" -> "whole-plant expert"** throughout: Figure 2 SVG label, Figure 2 caption, §4.4 prose (2),
  Table A1 caption (3), Table A1 row label. (The one surviving "null" is inside a `rev-del` strike.)
- **"z-score units" -> "standardized units"** (§4.4 prose, Table A1 caption; struck+green marked).
- **"shipped"** removed (§7 A8 rewrite; Table C1 prose; Table C1 caption).
- **Abbreviation first-use expansions**: ICS (§2.3), SOTA (§2.4, first surviving use), NLL (Table 1 A5 row),
  KDE (§4.3 i heading).
- **First-use glosses added**: Higher Criticism (§1), construct-matched (§5.5), Ledoit-Wolf (§4.3 iii),
  hardware-in-the-loop (§5.1). (whitened/Mahalanobis already glossed in Table 1 A7 row.)
- **"artifact-free" removed** (4 -> 0). "canonical" reduced at edited spots (Table 5 caption, §6/§7 WADI
  prose); remaining "canonical SWaT" uses left (defined once in §5.1, not incorrect) - see NOT changed.

## Paper figures/tables fixes
- **Table 3**: two "±0.000" cells removed (SWaT AE Easy, SWaT LatAD Easy); F1 precision made column-uniform
  (HAI & SWaT LatAD rows padded to 3 dp to match their columns; WADI column already uniform 2 dp).
- **Table B1**: added a "Peak GPU MB (W/H/S)" column (LatAD-global 19.9/19.4/19.7; LatAD-comm 28.7/25.1/24.8;
  TranAD 62.7/42.3/38.9; USAD 12.2/19.4/19.2; AE 28.9/45.6/24.2, from `e3_cost_cuda.json` peak_gpu_mb).
  Hardware named: "one Modal A10G GPU (24 GB) plus a Modal 8-vCPU CPU edge proxy" (caption + Appendix B
  prose). AE train "0.0 / 0.0" -> "<0.1 / <0.1".
- **Table 5 caption**: dataset list reordered to column order "WADI, HAI, and SWaT (30 / 167 / 85)"; bold
  defined ("gains that survive the significance test"); "artifact-free/canonical" dropped.
- **Table 6 caption**: bold clarified ("the LatAD (global density) configuration, not a column maximum").
- **Table A2 row label**: "six summary statistics (headline)" -> "(used by LatAD)".
- **Table C1 prose**: "Two synthetic controls ... drift-into-empty-space negative control" -> "A synthetic
  control ... (the positive control in Table C1)"; close-distinct mass range "0.94 to 1.00" -> "0.99 to 1.00
  in Table C1, 0.94 to 1.00 across the broader screen".

## Letter fixes
**Editor** (`letter_editor.html`):
- Phantom "no measurements sentence in §7 now points to these numbers" removed; replaced with plain statement
  that cost is added in Appendix B / Table B1 and the deployment paragraph is reworded not to deny it.
- Request 3: named hardware (Modal A10G GPU 24 GB + Modal 8-vCPU CPU proxy), added peak GPU memory
  (tens of MB, ~20-29 MB for LatAD).
- Request 2 (nearest-component, P4): removed the phantom "benchmarks lack operating-regime imbalance / one
  dominant regime" claim (contradicted A5-validated); reframed as a rare-regime safety term whose benefit
  needs the imbalance sweep, which was not run.
- Request 4: bootstrap definition cited to §5.4 (was implicitly §6).
- GDN bullet: "reported in Tables 3 and 4 and in Figure 3".

**Reviewer 1** (`letter_reviewer1.html`):
- Opening: removed "for finding the research content interesting" (that was Reviewer 2).
- Comment 2 (P4): same honest nearest-component reword; states imbalance sweep not run.
- Comment 3: named hardware + peak GPU memory.
- Comment 4: bootstrap def cited to §5.4; WADI bullet rewritten to the per-detector significance result
  (Resolution 1) - significantly ahead of every learned detector, ties LinRes.
- Comment 5 (P6): states plainly that no raw per-timestep sequential-input variant was run and that the
  temporal/spectral part is answered at the feature level by Table A2; "canonical/artifact-free" dropped.
- Figures line: GDN "in Tables 3 and 4 and in Figure 3".

**Reviewer 2** (`letter_reviewer2.html`):
- Comment 1 (P8): abbreviation claim de-overclaimed; lists the now-expanded set (adds ICS, SOTA, KDE), notes
  the abstract keeps compact forms.
- Comment 8 (P10): specification/realization statement correctly attributed to §4 (§3 adds the checklist
  sentence + Table 1 column), not §3; dropped the unsupported "closely related and evaluated together".
- Comment 11 (P9): "clause ... removed" -> "replaced"; notes "z-score units" -> "standardized units".
- Results line: GDN "in Tables 3 and 4 and to Figure 3".

**`letters_summary.md`**: refreshed the stale open-items (Table 2 now clean 0.742/0.621/0.733; GDN double-hard
cells populated in Table 4 and a GDN bar in Figure 3); ref numbers corrected (MetroPT [53], BATADAL [46],
N-CMAPSS [47], BDG2 [11], TSB-UAD [48]); WADI story anchor + R1 map updated to the per-detector significance;
added a Table B1 hardware/GPU-memory note.

## Deliberately NOT changed (with reason)
- **Abstract, Data Availability/Zenodo, Figure 3**: per resolutions 5, 4, 6 (verified untouched).
- **"canonical SWaT" remaining ~14 uses**: defined once in §5.1 and not incorrect; a full marked sweep of
  every occurrence is high-churn polish with low reader value. Reduced only at spots already being edited.
- **CPS/IIoT double definition in §1 (report readability item 6)** and **Table 2 trim / "headline" dual
  meaning (items 11, 13)**: front-matter restructure that overlaps the deliberate R2 responses; left to avoid
  re-opening settled reviewer answers and the off-limits abstract.
- **§7 line-1050 WADI mechanism sentence ("coordinated drift confined to a small correlated subsystem")**:
  left as-is; it describes locality/dilution (why community density beats global density), which is
  consistent with Table 5's "single-channel or linear" reading (a single-channel drift in one small community
  is diluted globally). Aligning it to "single-channel or linear" would misdescribe the factorization
  mechanism.
- **§4.2 second "(NLL)" expansion**: harmless duplicate now that Table 1 expands NLL first; removing it needs
  marking for no reader gain.
- **F1 3-dp on WADI LatAD rows**: WADI F1 column is uniformly 2 dp already; padding it would imply precision
  not present in the source.
- **Table 3 F1 "twelve/thirteen windows" (#8)**: moot - the "43-to-30" sentence was deleted per readability.
- **DOCX rebuild / pandoc**: excluded by instruction (human rebuilds).
