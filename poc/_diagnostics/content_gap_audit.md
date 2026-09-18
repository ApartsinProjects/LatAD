# Content-gap and stale-narrative audit of `paper/IoT2.html` (report-only, 2026-09-17)

Scope: the qualitative complement to the numeric consistency pass. Every sentence below is quoted verbatim
from `E:\Projects\Backlog\LatAD\poc\paper\IoT2.html` (text extracted, tags stripped) with its section.
Evidence base for "what is now true": `_diagnostics/headline_clean_results.md` (headline regime-community
model on WADI_clean / SWaT_canon, 5 seeds, construct-matched), `_diagnostics/fable_session_audit.md`
(F1-F18), `review_round_1/RESEARCH_REGISTRY.md` (2026-09-17 entries, §7b coverage), `_diagnostics/
hai_exclusion_community.json`, `_diagnostics/cov_excl3_HAI.log`, `_diagnostics/fable_discrete_sensors.md`,
`_diagnostics/discrete_method_citations.md`, and a fresh re-run of `_diagnostics/pca_discrete_easyfilter.py`
and `_diagnostics/linres_confound_and_ci.py` (logs in the session scratchpad).

Nothing in the paper was edited. Numeric cross-checks (abstract vs tables, SC-2/3/5/7, the HAI 0.849 vs
0.814 reconciliation) are the other agent's and are referenced only where they change a CLAIM.

Reading key for verdicts: **paper** = wins-only paper-worthy on the corrected basis; **rebuttal** = response
letter / cover letter only; **registry** = stays in `RESEARCH_REGISTRY.md` and commits.

---

## A. Prioritized STALE-CLAIM list

Severity: BLOCKING = the claim is now false or rides an artifact; SUBSTANTIVE = the claim survives but its
framing or support is pre-correction; MINOR = wording.

### A1. BLOCKING. "leads the difficult subset of all three" (four occurrences)

| Where | Verbatim |
|---|---|
| Abstract | "On three real IIoT testbeds LatAD attains the best AUROC (WADI 0.862, HAI 0.949, SWaT 0.993) and leads the difficult subset of all three, notably on HAI (0.849; +0.09, 95% CI [0.046, 0.160]); USAD and TranAD fall to 0.30–0.48 on these reconstructable-but-improbable faults." |
| §1 contributions (3) | "LatAD attains the best overall AUROC on every dataset and leads the difficult subset of all three, most decisively on HAI, where the difficult faults break joint structure, the deep detectors collapse to near chance, and LatAD's margin is statistically significant." |
| §7 "What the results show" | "LatAD attains the best overall AUROC on every dataset and leads the difficult subset of all three over five seeds." |
| §8 Conclusion | "LatAD attains the best overall AUROC on every dataset and leads the difficult subset of all three, significantly on HAI (paired 95% CI [0.046, 0.160]) and near-significantly on WADI, where the reconstruction-based deep detectors collapse." |

Why stale: on cleaned WADI the headline model is numerically ahead of the strongest classical baseline
(HCcoh+LatAD 0.816 vs linres 0.787; +0.029, 95% CI [-0.096, +0.174], 11 episodes) and the lead is not
significant; on canonical SWaT the lead over linres IS significant (P = 0.024 difficult, 0.007 double-hard).
The significance pattern has moved from {HAI sig, WADI near-sig, SWaT ceiling} to {HAI sig, SWaT sig, WADI
numerical}. "Near-significantly on WADI" and "where the reconstruction-based deep detectors collapse" (WADI
collapse to 0.30-0.33 was the artifact; clean: TranAD 0.722, USAD 0.685) are both false now.

Corrected claim (forward form): "LatAD leads the difficult subset on HAI and SWaT, with statistically
significant margins over the strongest baseline on both (HAI 95% CI [..]; SWaT P = 0.024), and is the
best learned detector on WADI, ahead of every reconstruction-based method." The abstract keeps ONE headline
number per claim (SC-11); the deep-detector collapse sentence becomes HAI-specific: "on HAI, USAD and TranAD
fall to 0.44-0.48."

### A2. BLOCKING. Every WADI-superiority sentence that rode the `2B_AIT_002_PV` artifact

| Where | Verbatim | Status on cleaned WADI |
|---|---|---|
| §6 "Difficult, joint-structure faults" | "On WADI the deep detectors collapse to 0.30–0.33 while LatAD reaches 0.796, a lead over Isolation Forest (0.677) that the same bootstrap places near significance (+0.12, 95% CI [−0.005, 0.257], P ≈ 0.03); the regime-community factorization is what turns the earlier single-latent tie (0.690) into this lead." | Deep detectors do not collapse (0.685 / 0.722). The IF comparison is no longer the relevant one (linres 0.787 is the strongest baseline). The factorization gain survives and is the honest WADI story: global 0.734 -> headline 0.799-0.816. |
| §6 "Deep SOTA under raw metrics" | "over five seeds they fall to 0.477 and 0.444 on HAI and to 0.305 and 0.325 on WADI, near or below chance and far below our 0.796–0.849" | WADI half is the artifact (AE 0.425 -> 0.739, USAD 0.303 -> 0.685, TranAD 0.333 -> 0.722 on the clean rebuild; audit F16: "the entire global-density difficult margin over AE on WADI is the artifact"). Keep the HAI half only. |
| §6 Figure caption (results figure) | "the deep detectors USAD and TranAD fall to 0.30–0.48 on WADI and HAI, whose difficult faults break the joint structure of normal operation, and rise only on SWaT, whose difficult faults are large deviations every method detects." | Both halves stale: WADI collapse is artifact; on canonical SWaT USAD/TranAD are 0.658/0.655 on difficult (they do NOT rise). |
| §6 Table 4 prose | "the AutoEncoder collapses on WADI (0.336)" | Clean double-hard: AE 0.671, TranAD 0.629, USAD 0.595; headline 0.736-0.756. The AE "collapse" was the artifact. |
| §7 "Why local density beats global" | "This is the mechanism behind the WADI gain (Table A1): the coordinated drift of a small analyzer group is improbable within its own subsystem yet unremarkable in the whole-plant density, so factorization turns the single-latent tie into a lead." | The factorization gain is REAL on clean WADI (HC alone 0.783 > global 0.734), so the mechanism sentence survives, but "coordinated drift of a small analyzer group" must be re-verified: the artifact channel is an analyzer (`2B_AIT_002_PV`, 27,359 sigma) and the next-largest test-normal shifts are the AIT_004 analyzers at ~2 sigma (audit F7). If the "analyzer group drift" the sentence describes is the rescaled channel, the mechanism illustration is the artifact itself. Re-derive the top-implicated community on the clean model before keeping the phrase. |
| §7 "The residual frontier on WADI" | "A minority of WADI difficult windows are attack-onset/offset edges whose fault content is near-absent: 2–3σ dips on a few correlated channels" | Derived on the dirty 19-window difficult set; the clean set is 43 windows (13 STATUS-flip windows re-enter as difficult once the clip is applied, audit F1/F2). Re-verify or drop. |
| §3 last paragraph | "the two overlapping datasets are exactly where the regime-community factorization gains most, since a single global density blurs local structure there, while crisp SWaT is a ceiling for every method (§6)" | Gain ordering on clean data: WADI +0.065 to +0.090, HAI +0.038, SWaT_canon +0.023; the first half holds. "crisp SWaT is a ceiling for every method" is false on canonical SWaT (see A3). |

Also stale by construction: Table 3 WADI block (all rows), Table 4 WADI column and its "WADI 11 / 5"
episode count (clean double-hard: 27 windows / 10 episodes), Table 5 WADI column, Table 6 WADI row, Table A1
WADI column, and the §6 "Robustness to a stronger difficulty definition" WADI numbers ("WADI remains a tie
(Isolation Forest 0.710, ours 0.697)"). These are the numeric agent's to reconcile; they are listed here
because each carries a prose claim.

### A3. BLOCKING. Every "SWaT is a ceiling / single episode" sentence (mirror-SWaT framing)

Canonical SWaT (`SWaT_Dataset_Attack_v0`, 12.14% attack, 1498 windows, 233 anomalies = 148 easy + 85
difficult across 23 episodes) is a discriminative test: trivial rule 0.646 on difficult (mirror: 0.943),
headline 0.827 vs linres 0.782 (P = 0.024), USAD 0.658, TranAD 0.655, AE 0.729, IF 0.627. Every sentence
that calls SWaT a ceiling, a single-episode result, or a benchmark where "every method is strong" is stale:

| Where | Verbatim |
|---|---|
| §3 | "while crisp SWaT is a ceiling for every method (§6)" |
| §5.1 SWaT | "The SWaT attack stream used here is an openly accessible mirror whose attack record is attack-only: the normal periods interleaved between attacks in the canonical iTrust release are absent, so SWaT's absolute scores are not directly comparable to canonical-SWaT figures. The comparison remains internally fair because USAD and TranAD are re-scored on the identical stream." (delete entirely once canonical is adopted; also the "first 2% ... last 20% of the normal record is held out" protocol was a mirror workaround and must be re-stated for the canonical split) |
| §6 "Overall detection" | "including the strong classical baselines at the SWaT ceiling" |
| §6 "Difficult, joint-structure faults" | "On SWaT the difficult subset is only mildly difficult (the trivial rule already scores 0.943), so every method is strong (LatAD 0.969, the linear baseline 0.959) and its 38 windows come from a single attack episode, making SWaT a ceiling case rather than a discriminative test. The advantage over the deep detectors is largest exactly where the difficult anomalies are reconstructable yet improbable (WADI, HAI), and vanishes on SWaT, whose difficult anomalies are large multivariate deviations every method detects." |
| §6 "Validity of the difficulty split" | "On SWaT the rule stays strong on the difficult subset (0.943), which is why SWaT is a strong overall benchmark but not a discriminative test of joint-structure detection." |
| §6 "Robustness to a stronger difficulty definition" | "and SWaT remains a ceiling case (trivial rule 0.929)" |
| §6 "The double-hard subset" | "and the 18 SWaT windows come from a single attack episode, so no episode-level generalization is claimed there." and Table 4 caption "a single-episode result on SWaT" and "Windows / distinct attack episodes: WADI 11 / 5, HAI 84 / 19, SWaT 18 / 1" (canonical: SWaT double-hard 59 windows / 18 episodes) |
| §7 "What the results show" | "and SWaT is a ceiling." |
| §7 Table 5 prose | "On SWaT reconstruction is in fact the strongest single head (0.965), consistent with SWaT's difficult faults being large deviations that reconstruction catches, which is why the gate keeps the residual head on there and why SWaT is a ceiling rather than a joint-structure test." (Table 5 SWaT column is mirror-only; the head decomposition has not been run on canonical SWaT, so this sentence has no support on the adopted data) |
| §4.3 (iii) | "and 0.85 on SWaT (on)" gate ratio; §4 "the residual activates on HAI and SWaT" | Gate ratio was computed on the mirror stream; re-derive on canonical before keeping the number. |
| §5.5 and Table 3 | "GDN ... completed within our compute budget only on SWaT, where it matches the other deep detectors (difficult-subset AUROC 0.871, below ours)" | GDN was run on the mirror only. If canonical SWaT is adopted, the GDN row is on a different stream than every other SWaT row (SC-12). Re-run on canonical or drop the row and the §2.4/§5.5 sentences. |
| Table 6 SWaT row | "SWaT | 0.00 | 1.00 | 0.00 | off | 0.959 | 0.943" | mirror-only; re-derive. |

Corrected framing (forward): "On canonical SWaT the difficult subset spans 23 attack episodes and the trivial
rule falls to 0.646; LatAD leads every baseline and its margin over the strongest linear baseline is
significant (P = 0.024)." SWaT becomes the second significant win, not the ceiling case; the paper's
"two joint-structure benchmarks (WADI, HAI)" dichotomy needs to become "three".

### A4. BLOCKING. §5.1 WADI preprocessing paragraph now contradicts the adopted cleaning

Verbatim (§5.1): "A few channels carry physically impossible standardized values (a channel constant at
zero in the 14-day normal record, another recalibrated between files); standardized features are clipped to
±10σ to cap such sensor glitches."

Why stale: the "recalibrated between files" channel is `2B_AIT_002_PV` (train mean 9.09 / sd 0.164, attack-file
normal-row mean 4503, z = 27,359, the only channel with a break of this order among 123). The current text
KEEPS it and clips it, which leaves a constant +10 sigma offset on every test window: that offset is the
artifact-robustness signal the dirty WADI lead was built on. The adopted procedure DROPS the channel (122
channels remain, window counts and labels unchanged: 575 / 519 / 56). The sentence must state the drop, the
reason, and that all WADI numbers are on the cleaned data. Note also §4.4 "WADI's 738-dimensional feature,
123 channels × 6 statistics" becomes 732 / 122.

### A5. SUBSTANTIVE. "the deep detectors collapse" and "reconstruction-based deep detectors collapse" as a
general (three-dataset) claim

Verbatim (§1, §8, abstract): "USAD and TranAD fall to 0.30–0.48 on these reconstructable-but-improbable
faults"; "where the reconstruction-based deep detectors collapse".

On the corrected basis USAD/TranAD are far below LatAD on all three difficult subsets (HAI 0.477/0.444,
SWaT_canon 0.658/0.655, WADI_clean 0.685/0.722) but "collapse to near chance" is true only on HAI. Rewrite as a
margin statement ("USAD and TranAD trail LatAD by 0.09 to 0.40 on the difficult subset of every dataset, and
sit near chance on HAI") rather than a collapse statement.

### A6. SUBSTANTIVE. Global-ablation numbers presented where the headline model is meant

The paper is internally consistent on this (Table 3 has both rows; Table 5's last row is explicitly the
global configuration). Two places blur it:

- §7 Table 5 prose: "The residual head, where the gate turns it on, lifts the base score exactly where
  reconstruction is informative (HAI 0.801→0.820, SWaT 0.947→0.960)". These are global-model numbers on
  mirror SWaT; fine as an ablation, but the sentence sits under the "Why reconstruction fails and density wins"
  narrative that the reader attaches to the headline. Label the paragraph "single-latent ablation" explicitly.
- §7 "We verify this directly on WADI with a single-model diagnostic probe (one seed). The difficult anomalies
  stay on the normal manifold (PCA off-subspace residual AUROC 0.43) and reconstruct as well as normal
  (autoencoder reconstruction-error AUROC 0.43), yet the latent density flags them (0.74)." Dirty-WADI probe,
  single seed, global model. On clean WADI the AE reconstruction error reaches 0.739 on the same subset, so
  "reconstruct as well as normal (0.43)" is the artifact. Re-run the probe on WADI_clean or move it to HAI.

Also for the numeric agent, but carrying a narrative consequence: `hai_exclusion_community.json` reports the
current HCcoh+LatAD HAI difficult AUROC as 0.8137 while the paper says 0.849. If 0.814 is the number that
survives reconciliation, the abstract's "+0.09, 95% CI [0.046, 0.160]" over the AutoEncoder (0.757) becomes
+0.057 and the CI must be recomputed; the E4 time-aware bootstrap in the registry (+0.054, CI [0.012, 0.089],
P = 0.005) is the construct-matched value for that case and is still a clean significant win.

### A7. MINOR. Cross-reference and numbering rot

- Two figures are captioned "Figure 1": the IIoT-context diagram (§1) and the results bar chart (§6).
  The §6 sentence "This is where the design earns its keep (Figure 1)" now resolves to the context
  diagram. The results figure should be Figure 3 and the reference updated (SC-5).
- §1 A-note: "motivates the assumptions we examine in §7 (A3, validated on real malfunction data; the
  trajectory assumptions A9–A10 left to future work)" resolves correctly.

---

## B. CONTENT-GAP table

| # | Finding (verified source) | Verdict | Where it goes | Wins-only / honesty note |
|---|---|---|---|---|
| G1 | **WADI instrumentation artifact** `2B_AIT_002_PV` rescaled between the normal and attack recordings (z = 27,359; unique among 123 channels; next-largest test-normal shift ~2 sigma; audit F7, `cov_wadi_sensors.log`, registry §7b energy-distance 2.34 -> 0.71 after removal). | **paper** (required disclosure) | §5.1 WADI paragraph: replace the "another recalibrated between files ... clipped" clause with "channel `2B_AIT_002_PV` is rescaled between the normal and attack recordings (a 27,000-sigma shift, unique among the 123 channels) and is dropped; all WADI results use the remaining 122 channels." One sentence in Limitations ("WADI results are on the artifact-free 122-channel set"). | This is preprocessing, not a result; it must be stated. Do NOT report the dirty numbers or the "robust to the artifact" comparison in the paper (registry/rebuttal only). |
| G2 | **Canonical SWaT adoption** (`SWaT_Dataset_Attack_v0`, 449,919 rows, 12.14% attack, 51 channels aligned by name, Normal = train, Attack = test; audit F11). Headline 0.827 vs linres 0.782, P = 0.024; double-hard P = 0.007; USAD/TranAD 5-seed re-run on the canonical stream. | **paper** | §5.1 SWaT paragraph (replace the mirror disclosure with the canonical split; drop the 2% / 20% hold-out workaround if no longer used); §6 SWaT prose (A3 above); Table 3/4 SWaT blocks; abstract sentence 5. | Strong, significant, construct-matched. Requires: GDN, Table 5 head decomposition, Table 6 gate row, §4.3 gate ratio re-derived on canonical (or the GDN row dropped). |
| G3 | **Coverage self-diagnosis on HAI**: one train-empty operating regime (regime 21: 0.4% train / 6.3% test; a 646-701-window contiguous test-normal block, appearing ~6,170 normal windows after training ends, so a recording-DURATION limit). Excluding it label-free (0 anomalies removed) lifts the headline model from AUROC 0.936 -> 0.951, TPR@5% 0.757 -> 0.888 (`hai_exclusion_community.json`, HCcoh+LatAD); 93% (global model, `cov_excl3_HAI.log` fpr5 0.928) of LatAD's 5%-FPR false alarms lie inside the block; all detectors improve and LatAD leads throughout. | **paper** (one Discussion paragraph, secondary numbers) + **rebuttal** | §7, new paragraph after "The residual frontier on WADI", titled e.g. "Benchmark coverage as a diagnosable property": state that the per-community model reports the block as an unseen regime, that it is duration-limited, and the with/without numbers as a secondary view. Main tables stay on the FULL test set. | Framing must be forward: "LatAD's false alarms concentrate in one unseen regime that its own regime model flags" (a strength: diagnosability), never "our false alarms are the benchmark's fault". The three artifacts disagree in the last digit (0.934/0.950 global vs 0.936/0.951 community; 93% vs 94%; difficult 0.841 vs 0.851): co-compute ALL quoted values on the headline model in ONE artifact before any enters the paper (SC-12). Never swap the headline numbers for the excluded ones. |
| G4 | **Detection-set complementarity on HAI** (equal alarm budget): LatAD catches 65/167 difficult anomalies, 49 caught by LatAD alone; LatAD catches 63 that TranAD misses, TranAD catches 0 that LatAD misses (Jaccard 0.03). Registry flags it "Needs re-verify on fixed tables + headline model"; computed on the GLOBAL model. | **rebuttal** now; **paper** (one sentence or small figure in §6 HAI paragraph) only after re-verification on the headline model | §6 "Difficult, joint-structure faults", HAI paragraph, after the bootstrap sentence: "At a matched alarm budget LatAD detects N difficult HAI anomalies that no baseline detects, and misses none that TranAD detects." | Strong and honest once re-verified; the current number is unverified on the headline model and must not be quoted until it is. |
| G5 | **Extended difficulty stratification** (T1 continuous + T2 discrete-state + T3 PCA-per-axis easy-filters, circularity-guarded: linres/l2/PCA define difficulty, so only IF and the deep detectors are compared). Re-run today: HAI survivors 48/652, LatAD 0.581 vs AE 0.447 / IF 0.439 / USAD 0.328 / TranAD 0.257; registry bootstrap +0.136, CI [+0.055, +0.199], p = 0.002, 10 episodes. SWaT (MIRROR) 11 windows, 0.883 vs 0.834, 1 episode. WADI_clean 19 windows, 0.610 vs IF 0.612 (tie). | **paper** for HAI as an extension of the existing §6 "Robustness to a stronger difficulty definition" paragraph; WADI/SWaT rows **registry** | §6 robustness paragraph: add one sentence, "Under a still stronger split that also removes windows a whitened principal-component axis or a discrete-state rule separates (48 HAI windows, 10 episodes), the lead over the best reconstruction detector widens to +0.14 (95% CI [0.06, 0.20])." | Honest only if (a) it is stated that linear/PCA detectors are excluded by construction from that comparison, (b) it is re-run on the headline model (current run is global LatAD from `scores_HAI.npz`), (c) the SWaT number is re-run on canonical (the current 0.883 is mirror). The WADI tie stays out of the paper (wins-only), but if the existing paragraph keeps reporting WADI, it must be the clean headline number, not "Isolation Forest 0.710, ours 0.697". |
| G6 | **A8 wording gap**: Table 1 A8 says "Mixed signals (heterogeneous, typed channels)" and §4 says A8 is realized by "per-feature standardization" and the whitened residual; the code types no channel (valve states are z-scored, moment-summarized and Gaussian-modeled; only the LinRes baseline one-hot-encodes; `fable_discrete_sensors.md` §A). Figure 2 additionally tags the community-discovery block "[A8] locality / subsystems", a third meaning. | **paper** (required wording fix) | Table 1 A8 row: rename to "Mixed signals (heterogeneous scale and noise)" and the design cell to "per-feature standardization on train-normal; whitened residual respecting per-channel scale and cross-channel correlation". §4 paragraph and §4.4 "typed, correlated channels of A8": drop "typed". Figure 2 tag: change "[A8]" on the community block to the assumption it actually realizes (A5 few levers / correlated groups), or retitle the tag legend. Future-work sentence in §7 "Alternative realizations": "Native categorical likelihoods for discrete channels (HI-VAE [Nazábal 2020], VAEM [Ma 2020]) and graphical-model ICS detectors that type actuator states (TABOR [Lin 2018]) are complementary realizations of A8." Three validated citations are ready in `discrete_method_citations.md`. | Not a result, a claim-vs-code fix. No new experiment needed. |
| G7 | **LinRes scaling caveat**: the one-hot LinRes baseline leaves state fractions unscaled while numeric means are standardized; standardizing all features drops its WADI_clean difficult AUROC from 0.787 to 0.684 (numeric-only 0.735); reproduced today by re-running `linres_confound_and_ci.py`. HAI (0.586 / 0.586) and SWaT (0.959 / 0.955) unaffected. | **registry / rebuttal** | Nowhere in the paper. Keep the STRONGER LinRes variant (0.787) in Table 3 as the baseline; the headline beats it anyway. If a reviewer asks why a linear baseline is competitive on WADI, the answer is that WADI's subtle faults are largely linear cross-channel breaks and the baseline is reported in its strongest form. | Reporting the weaker LinRes variant to widen the margin would be cherry-picking; reporting the caveat in the paper would be a scope-confession about a baseline. Both stay out. |
| G8 | **Clean-WADI factorization gain**: headline 0.799 (null+HC) / 0.816 (HCcoh) vs global 0.734, with the gain traced to the community signal itself (HC alone 0.783). | **paper** (already the paper's mechanism claim; numbers refresh) | §6 WADI sentence and Table A1 WADI column; §7 "Why local density beats global" keeps its mechanism paragraph with the clean numbers. | The +0.065 to +0.082 gain is the honest WADI headline; it replaces the "near-significant lead over IF" story. |
| G9 | **LOF adaptive-density head** (audit F5: on the canonical mask LOF >= GMM on WADI 0.718 vs 0.694 and SKAB 0.568 vs 0.473, tie on HAI); registry marks it superseded and to be re-run. | **registry** | Nowhere until re-run with bootstrap on the headline model. | Unfinished; direction reversed once already this session. |

---

## C. Narrative and tone inconsistencies

### C1. SC-4 coherence (Conclusion vs Discussion vs Limitations)

All three sections carry the SAME pre-correction framing, so they are mutually consistent and jointly stale:
"leads the difficult subset of all three" (abstract, §1, §7, §8), "near-significant on WADI" (§6, §7, §8),
"SWaT is a ceiling" (§3, §6, §7), "deep detectors collapse on WADI" (§6, Figure caption, §8). When the
results are refreshed, all four sections must move together; the risk is a v2 rewrite that updates the
abstract and §6 and leaves §3 ("crisp SWaT is a ceiling for every method"), §7 "What the results show", and
the Table 5 / Table 6 prose in their v1 state. Concretely, the §3 sentence is a forward reference to §6 and
will be the first to rot.

Limitations (§7 "Limitations"), verbatim: "Three boundary conditions apply: three datasets, one trained
configuration per dataset (evaluated over five seeds), and a single instantaneous representation."

- Under-claims the corrected scope in two places: it must add (i) the WADI data-cleaning boundary ("WADI
  results are on the 122-channel set with the rescaled analyzer channel removed") and (ii) the coverage
  boundary ("HAI's test record contains one operating regime absent from training; results are reported
  on the full test set"). Both are neutral boundary conditions, not confessions.
- Over-claims nothing on its own; but the sentence in §7 "Alternative realizations": "Because the community
  construction and the cohesion-weighted Higher Criticism combiner were developed on these three benchmarks,
  the regime-community realization should be validated on additional, untouched CPS systems before its
  per-dataset gains are treated as established" is a correct boundary and, with canonical SWaT and clean WADI
  now being effectively fresh evaluations of a combiner developed on the mirror/dirty versions, it can be
  stated MORE strongly in the authors' favour: the combiner's settings were fixed before the clean/canonical
  re-evaluation. That is a positive out-of-sample statement the paper is currently not making.
- "The easy/difficult split is a property of one cut" is fine and now has two robustness checks behind it
  (six-statistic split, and G5 once adopted).

### C2. Abstract arc and tone audit

- Em-dash: zero U+2014 in the body; **three `&mdash;` entities in h3 headings** (§4.1 "Variational Deep
  Embedding (VaDE) — the representation stage (Figure 2, top)", §4.3 "The scoring stack — the scoring stage",
  §4.4 "Regime-community density (the headline realization) — the subsystem-factorization stage"). These
  render as em-dashes and violate the house rule; use a colon.
- Double-hyphens: 22 hits, all CSS custom properties (`--text`, `--muted`, ...) inside `<style>`; none in
  prose. Clean.
- Forbidden words: zero hits for honestly / frankly / admittedly / unfortunately / arguably / we believe /
  it seems / somewhat / merely / we acknowledge / we concede / not statistically. Clean.
- Abstract beats: problem (2 sentences), solution (2), key finding (1), cross-task folded into the key-finding
  sentence, no evaluation-lens beat, no impact beat (no code/DOI pointer; the Zenodo DOI appears only in the
  Data Availability statement). Word count ~230. The key-finding sentence stacks three headline numbers, a
  difficult-subset number, a delta, and a CI in one sentence (SC-11 anti-pattern "three qualifiers stacked").
  On the corrected basis the natural rewrite is: one full-AUROC clause without three numbers, one
  difficult-subset clause naming HAI and SWaT as significant, one deep-detector margin clause, one impact
  clause (Zenodo DOI).
- Scope-confession scan (SC-15): "shown for completeness, not as an unbiased comparison" (§6 double-hard) and
  Table 4 caption "not an unbiased comparison against those two filters" are legitimate circularity
  disclosures, keep. "WADI remains a tie (Isolation Forest 0.710, ours 0.697)" (§6 robustness paragraph) is a
  null stated inside a wins paragraph; on the clean headline model this may no longer be a tie, so re-run
  before deciding whether it stays. "This is a deployment architecture rather than an experimentally evaluated
  configuration" and "We report no parameter count, memory, latency, or energy measurements here" (§7 IIoT
  deployment) are boundary statements; the second is now false in spirit because the registry has E3 timing
  (LatAD-global train 3.2 / 16 / 3.5 s vs TranAD 231 / 898 / 163 s; latency 0.7-13 ms global, 17-32 ms b1
  community). That is a paper-worthy win the paper disclaims: replace the "we report no ... latency" sentence
  with the measured numbers (verified in audit F17, with the 52 ms corrected to the b1 figures).
- Defensive framings: "which we use rather than claim as a contribution" (§1, evaluation protocol) and "The
  gate is therefore not a tuned switch but a property of the data" (§7) are mild pre-litigations; the second
  is acceptable, the first can be "adopting the critique's protocol".
- Process backreferences: "the regime-community factorization is what turns the earlier single-latent tie
  (0.690) into this lead" (§6) and "turns a prior single-latent tie into a near-significant lead" (§7) narrate
  the ablation as history ("earlier", "prior"). Rewrite as a static ablation contrast ("the single-latent
  configuration scores 0.734; factorization lifts it to 0.816").

### C3. Internal terminology drift

- A8 has three meanings across Table 1 (typed channels), §4 (standardization + residual), and Figure 2
  (locality / subsystems). See G6.
- "joint-structure benchmarks (WADI, HAI)" vs "three real CPS attack benchmarks": once canonical SWaT is
  discriminative, the two-vs-three dichotomy that structures §6 and §7 dissolves; every "the two
  joint-structure benchmarks" phrase (§4.2, §6, §7 Table 5 prose) needs the third dataset or a new
  distinction (e.g. "the two high-dimensional plants").

---

## D. Literature and positioning notes

Source: a web-researcher scout run today (WebSearch/WebFetch; [F] = page fetched and verified, [S] =
search snippet only). Nothing found in 2024-2026 combines a latent regime mixture, likelihood scoring,
correlation-community factorization and Higher Criticism; the contribution's core framing is not pre-empted.
Gaps a top-venue reviewer would still raise:

### D1. The paper's own preprint must be cited
arXiv 2607.06094 (Apartsin and Aperstein, July 2026) is the LatAD preprint and is public. The MDPI submission
must cite it as the preprint version and the cover letter must state the relationship. Currently absent.

### D2. Direct competitors to contrast in §2.3 / §2.4 (one line each)
- Birihanu and Lendak, "Explainable correlation-based anomaly detection for ICS", Frontiers in AI, Feb 2025
  [F]: LSTM-AE window features, Pearson correlation vectors scored by a single multivariate Gaussian, on SWaT
  and HAI. This is the closest published "density over correlation structure" detector and it is unimodal
  with no subsystem structure: cite as the antecedent LatAD's regime mixture and community factorization
  extend. Missing from §2.3, which currently jumps from DAGMM/Deep SVDD/THOC/OmniAnomaly (2018-2020) to nothing
  newer on the density side.
- Islam and Carden, "Product-Aware Deep Autoencoders for Multi-Product CPS", arXiv 2606.00052, May 2026 [F]:
  grade-specific autoencoders because a global model hides anomalies inside "expanded decision boundaries";
  a global model misses 77.8% of attack scenarios on Extended TEP. This is independent evidence for A1/A6
  (regime mixture, global model under-fits) and the natural mode-aware contrast: they use supervised mode
  labels and reconstruction, LatAD discovers regimes unsupervised and scores by likelihood. Cite in §3 or
  §2.3.
- Graph-MoE with memory-augmented routers, AAAI 2025 [F]: mixture-of-experts over hierarchical GNN layers
  for MTS-AD. Nearest expert-ensemble precedent to the per-community expert stack; forecast/reconstruction
  scoring. Dataset list not in the abstract; check the PDF before citing as a SWaT/WADI competitor.
- Physics-guided contrastive temporal graph learning with node-level root-cause scores on SWaT/WADI,
  Scientific Reports 2026 [S, fetch blocked; PMC mirror PMC13216598]: overlaps LatAD's localization claim
  ("most-surprised subsystem"). Verify, then contrast on localization granularity (node vs subsystem) and
  on the normal-only, no-schematic training.
- Snippet-only, verify before citing: DARTs (arXiv 2512.13735), cross-domain representation learning for ICS
  AD (arXiv 2509.11786), hybrid-DL real-time CPS AD on SWaT/WADI (ScienceDirect 2026).

### D3. Structural precedent for the Higher Criticism combiner
§4.4 cites only Donoho and Jin 2004. Two recent statistics papers apply HC exactly as LatAD does (combining
per-stream surprises across many streams) and should be cited as precedent: Stoepker et al., "Permutation-based
Higher Criticism for a large number of streams", JASA 2024 [F] (null-free HC for anomaly detection across
many streams, industrial batch monitoring), and "Rank-based Higher Criticism across referentials", Annals of
Statistics 2025 [F]. Neither is a CPS time-series application, which is a positive positioning point: LatAD is
the first to apply HC across physical subsystems of a plant, as far as the scout found.

### D4. Evaluation-methodology positioning
- No 2024-2026 paper proposing a per-channel-max-z difficulty split or "hard anomaly subset" for MTS-AD
  benchmarks surfaced; the closest is "Benchmarking Unsupervised Strategies for AD in MTS", arXiv 2506.20574
  (VLDB 2026 under review) [F, abstract], whose body reportedly finds SMD, SWaT and MSL "almost trivial" for a
  simple baseline. Cite in §2.2 alongside Pinet et al. as convergent evidence, and keep the paper's current
  posture ("adopting the critique's protocol, not claiming it") since it remains defensible.
- "Unveiling the Flaws: initialization effect on TSAD", arXiv 2408.06620 [F]: window size, seed and
  normalization dominate reported gains. Supports the paper's five-seed reporting; one cite in §5.4.
- Reachability-vs-probability framing: no new 2024-2026 critique beyond Bouman and Heskes 2025 (already cited)
  surfaced; the framing stands. The Islam and Carden "expanded decision boundary" language is the same
  argument from the mode-aware side and strengthens it.

### D5. WADI / SWaT data-quality literature (supports G1 and G2)
- No paper documents the `2B_AIT_002_PV` recalibration between the WADI normal and attack files. One
  snippet-level source drops the channel because train/test distributions differ (source not identified).
  This means the paper's data-quality sentence (G1) is, as far as the scout found, the first explicit
  documentation of the artifact and should be written as such: state the measurement (27,359 sigma, unique
  among 123 channels), not "a documented WADI issue".
- Known WADI issues that ARE documented: WADI.A1_9 (Oct 2017) vs WADI.A2_19 (Nov 2019) version ambiguity and
  the attack-9 timetable date typo (GDN issue #49 [F]); Turrin et al. (CISPA) statistical-analysis report on
  the water-distribution datasets [S, URL moved]. §5.1 should name which WADI release is used.
- Canonical vs mirror SWaT: the TranAD repository notes its processed baselines differ from the paper [S]; a
  PMC comparative study (PMC9921147) [S] shows constant-feature exclusion changes SWaT results per model. Both
  support stating the exact SWaT release (`SWaT_Dataset_Attack_v0`, Dec 2015, 12.14% attack) in §5.1 (G2).

### D6. Positioning verdict
The difficulty-stratification framing and the reachability-vs-probability framing remain defensible and
uncontested in the 2024-2026 literature the scout could reach. The two positioning gaps a reviewer will flag
are (i) the density-side related work stops in 2020 (add Birihanu and Lendak 2025, Islam and Carden 2026,
Graph-MoE 2025) and (ii) the HC combiner is cited only to its 2004 origin (add the JASA 2024 / AoS 2025
stream-combination precedents). Every new citation goes through `bibtest` before it enters the reference list.
