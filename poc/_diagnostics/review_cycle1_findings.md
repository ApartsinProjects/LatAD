# Cycle-1 consolidated fixes for IoT2.html (verified)

Apply all. Keep the tracked-changes markup (rev2-highlight green = new, rev-del red strike = deleted)
as the revision format, BUT remove rev-del spans whose content is editorial/lab-notebook (see B5).
No CONTENT (live or struck) may narrate a leak, fix, recompute, or pipeline version.

## A. BLOCKING correctness
A1. HAI USAD number is a copy from the SWaT row. Verified clean: HAI USAD All=0.849, Difficult=0.497
    (paper has 0.843 / 0.477). Fix Table 3 HAI USAD cells and every prose quote of "USAD (0.477)".
    Cascade: abstract "USAD and TranAD fall to 0.44-0.48" -> "0.44-0.50"; §8 line ~1113 same.
    HAI TranAD Difficult 0.444 -> 0.445 (verified). HAI AE 0.757 is fine (keep).
    (NOTE: HAI double-hard USAD 0.351 / TranAD 0.296 are CORRECT - do NOT change.)
A2. Table 8 drift-rate definition: use ONE definition and state it precisely:
    "fraction of test-normal windows whose per-window mean community surprise exceeds the
    train-normal 99th percentile" -> WADI 1.5%, HAI 39%, SWaT 69% (verified). Fix the WADI
    "n/a (no drifted normals)" cell to 1.5%. In §7.1 REMOVE the clause claiming this is "the same
    calibration the headline detector uses" (the fused-score flag rate differs). Keep 1.5/39/69
    everywhere (abstract, §7.1, Table 8).
A3. Five stale "§8" refs that mean the drift section §7: lines ~712, 859, 862, 885, 1357 -> "§7".

## B. BLOCKING project-history / vocabulary (live text)
B1. "canonical SWaT" / "canonical-SWaT" / "the canonical iTrust release" (~30x) -> "SWaT". State the
    release ONCE in §5.1: "the December 2015 iTrust SWaT release". Line ~887 "The canonical split"
    (meaning the §5.3 split) -> "the §5.3 split".
B2. Line ~691 "USAD and TranAD are re-scored on this identical canonical stream" -> "USAD and TranAD
    are scored on the same stream."
B3. Line ~688 "the SWaT_Dataset_Attack_v0 release-v0 attack recording" -> "the SWaT attack recording".
B4. Line ~1062 "the fresh GDN" -> "GDN". Line ~1064 "(inside the pre-set +-0.02 tolerance)" ->
    "(within seed variation; Table 6)". Lines ~1225-1226 "settings were fixed before the WADI and
    SWaT evaluations ... out-of-sample confirmations" -> "All community-construction and combiner
    settings are shared across datasets and not tuned per dataset (§5.2)."
B5. Remove rev-del spans that are editorial/lab-notebook (delete entirely, not as struck text):
    line ~595 "[An optional basin-agreement rescue head ... was described here.]"; line ~2997 SKAB
    "responsibility entropy 0.29 is a variance-floor artifact ... log(0.05) ... drops to 0.045-0.048";
    line ~1234 garbled duplicate "rather than issuing only a plant-wide flag turns a plant-wide alarm
    into a subsystem triage shortlist rather than only a plant-wide flag".

## C. Localization Table 9 (clean, from localization_clean.md)
C1. HAI row unchanged (top-1 0.395 / top-3 0.684, p=0.024). WADI row unchanged (0.333 / 0.556,
    p=0.086, directional/underpowered). DROP SWaT as a localization win: SWaT clean is 0.333 / 0.458
    (top-3 below random 0.542). Replace the SWaT Table 9 row accordingly and mark it as drift-defeated.
C2. Prose: "68 to 79 percent of HAI and SWaT attacks" -> HAI-only (~68% top-3). Remove SWaT-specific
    localization claims ("11 of 24", "0.46 versus 0.35", "P=0.004"); present SWaT localization as the
    drift-defeated case that §7 typing addresses. Scope the localization contribution to HAI
    (significant) with WADI as directional support.

## D. Substantive tone/style/content
D1. "hardened ... difficulty stratification" -> "two-stage ... difficulty stratification" (abstract ~82, ~186).
D2. "HAI ... little drift" / Table 7 caption "low-drift records (HAI, WADI)" / §8 "where the record is
    stationary" -> "WADI (stationary) and HAI (moderate, front-loaded drift)" (lines ~1063-65, 1071, 1115).
D3. "learned detector" where LinRes is excluded -> "nonlinear detector" (abstract ~82; ~842, 864-868).
    Keep the explicit LinRes tie statement.
D4. Table A2 arithmetic: row = displayed(value) - displayed(base): HAI +0.071, WADI -0.008; co-round
    prose (~710-711, 1205, 1356) to match. SWaT A2 row 0.247/0.488/+0.241 stays.
D5. Table A1 line ~1321 "0.811 to 0.845" -> "0.801 to 0.845" (no-fusion row is 0.801).
D6. Coverage-stable clip prose (~683-686): one plain sentence, no post-hoc-justification tone, and note
    WADI channel was DROPPED not clipped: "Standardized features are clipped to +-10 sd on SWaT and WADI
    so recalibrated analyser channels [DAICS] do not dominate per-channel range statistics; HAI is not
    clipped." Ref to this axis should say §5.1 not §5.3 (line ~1044).
D7. Seed asymmetry: add one sentence (compute budget of the TranAD/USAD harness) explaining USAD/TranAD
    single-run on WADI/SWaT vs five-seed HAI; add a clause to Limitations.
D8. Trim over-repetition of "the drift-aware typing of §7 recovers ...": keep one forward pointer in §6,
    the Table 4 caption, and the full statement in §7; remove the rest (~10 -> 3).
D9. "headline detector/score/number" internal term -> define once in §4 as "the reported detector,
    LatAD (regime-community)" then use "reported"/model name.
D10. Drop self-certifying / defensive phrases: "deliberately fair"/"fair protocol" -> "raw-metric,
    difficulty-stratified protocol" (~1113, 1292); "earns its keep" -> "is the subset the design targets"
    (~850); "disclosed plainly"/"stated plainly" -> remove the adverb (~743, 1101); "answers the
    simple-baseline critiques ... directly" -> cut clause (~189); "which makes our numbers lower ... by
    design" -> "no point adjustment is applied to any method" (~1285); "we note the effect as a secondary
    view" -> "The main tables report the full test set; the excluded-regime view is secondary" (~1197).

## E. Minor (batch)
E1. Define at first use: "era / era-local" (~1102-05; define or rename to "recording segment"), CI (spell
    out at first use ~860), SD (Fig 3 caption), LOO (Table 4 row), PLCs (Fig 1 caption), the AIT/FIT/PIT
    tag families (~1040 add "(analyser, flow, pressure-transmitter tags)").
E2. Spelling/casing: pick "analyser" (not analyzer) throughout; "AutoEncoder" in prose to match tables;
    Table 3 WADI F1 to 3 decimals like HAI/SWaT.
E3. "two-day recording" for WADI test is fine, but line ~676 "14-day normal record" is correct - write
    "WADI's two-day TEST record" where the test is meant (~1035, 1108).
E4. Duplicates: seed policy stated 4x (keep §5.5 + captions, drop §2.4 ~246 and §5.4 ~741); "reconstruction
    measures reachability; detection needs probability" 3x -> keep Intro + Conclusion (drop ~1118).
E5. Line ~207 "The field therefore leans on ..." lost its antecedent after a deletion -> drop "therefore".
E6. Fig 3 SWaT bar group drawn at a slightly different pixel scale -> regenerate the SWaT bars from Table 3
    clean values (community 0.524, etc.).
E7. Reference order (SC-5): after accepting the redline, renumber [N] in first-appearance order; [48]
    TSB-UAD is orphaned (only cite inside rev-del) - drop it or add a live cite. New refs DAICS/AnoShift/D3R
    already validated.
E8. "difficult attacks" in the mechanism paragraph (~1128-31) -> "difficult anomalies (labeled attacks)".

## REJECTED reviewer items (verified NOT issues)
- HAI double-hard USAD 0.351 / TranAD 0.296: correct (match doublehard_pca_all3.json rows_pca). No change.
- §7 opening "Everything that departs from training-normal is an anomaly, but ... two kinds": CORRECT and
  matches the paper's intended framing (drift IS an anomaly, of a distinct type). Keep as is.

## Verify after edit
bibtest passes; balanced spans/tables; grep confirms no "canonical SWaT", "re-scored", "fresh GDN",
"pre-set", "were fixed before", em-dash, honest/frankly/admittedly; all §refs resolve; abstract <=200 words.
Do NOT rebuild DOCX.
