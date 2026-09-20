# Response to Reviewers — Manuscript IoT-4522779

**Title:** Modeling Normal Is All You Need: Joint Latent Clustering for Anomaly Detection in Multimodal Industrial-IoT Cyber-Physical Systems
**Authors:** Alexander Apartsin and Yehudit Aperstein
**Journal:** *IoT* (MDPI)

Dear Academic Editor and Reviewers,

We thank the Academic Editor and both Reviewers for a careful and constructive reading. The comments have materially improved the manuscript. We have restructured the front matter, added a system-model figure and a LatAD workflow figure, numbered every equation, resolved the notation conflict, restructured the related work, scoped the between-regime-overlap assumption (renumbered A8) as specified-but-not-observed after tracing its earlier apparent SKAB signal to an estimator artifact and withdrawing the basin head, benchmarked computational cost, isolated the source of the gain, and put every significance claim on a time-aware (episode-block) footing. The point-by-point response is below, grouped by Reviewer 1, Reviewer 2, and the Academic Editor. Each entry restates the request, then gives either the change made (with the section, table, or figure and a short quotation of the new text) or a rebuttal with evidence. Green highlighting in the revised manuscript marks the new text; our internal action tags (A-##, B-##) are cited so cross-references between comments are traceable.

## Overview: two data-quality corrections and their effect on the numbers

While preparing this revision we audited every reported number against its source artifact and identified two data-quality issues in the datasets used for the reviewed version. We correct both here, and we report the corrected, construct-matched results throughout. We state them openly because they strengthen the rigor of the evaluation; the qualitative conclusion (LatAD leads the difficult subset by modeling probability rather than reconstruction) is unchanged and in one respect strengthened.

1. **WADI.** One channel, `2B_AIT_002_PV`, is rescaled between the normal and attack recordings (its train-normal mean 9.09 and standard deviation 0.16 shift to a test-normal mean of 4503, a 27,359σ break, unique among the 123 raw channels). Keeping it left a constant offset on every test window. We drop it and report all WADI results on the remaining 122 channels (§5.1).

2. **SWaT.** We moved from an attack-only mirror stream to the canonical iTrust December-2015 release (`SWaT_Dataset_Attack_v0`, 449,919 rows, 51 channels, 12.14% of timesteps labeled attack), whose normal periods interleave between attack episodes (§5.1).

Net effect on the headline difficult-subset numbers: HAI is unchanged (0.849, significant); canonical SWaT becomes a genuinely discriminative test and a second statistically significant win (0.840 vs the linear baseline 0.782, P = 0.003); WADI's headline (0.824) now leads every baseline but is not statistically separable from the linear baseline over 11 episodes. The significance pattern is therefore HAI significant, SWaT significant, WADI a numerical lead ahead of every reconstruction-based detector. All full-set AUROC values (WADI 0.864, HAI 0.949, SWaT 0.941) are best-in-column.

---

# Reviewer 1

## R1-1. The trajectory assumptions A9-A10 are unused, and the basin head (A8) is inactive on all three datasets; either run data that activates them or scope the contribution. (items A-12, A-13, B-01)

**Response: Change made (contribution scoped; A8 specified but not demonstrated).**

We scoped the contribution to the assumptions the paper actually exercises. The MIIM list is now presented as a graded checklist (§3): a dataset realizes some properties strongly and others weakly or not at all, LatAD exploits each when present and reduces to a standard latent-density model when absent. Table 1 is grouped into two blocks: **A1-A7, realized and validated on the three real testbeds**, and **A8-A10, specified for completeness but not observed in the datasets studied**. (Note on numbering: the assumptions were renumbered so that the demonstrated instantaneous-window properties are contiguous A1-A7 and the not-observed ones are A8-A10; the former A3, between-regime overlap, is now A8.)

- **Scope stated (A-12).** §1, §3, §4, and §7 now say plainly that A1-A7 shape the instantaneous window and are exercised on all three testbeds, while A8 (between-regime overlap) and A9-A10 (trajectory) are specified but not observed here and are left to future work. §7 states: "Extending it with multiscale-temporal features (A9) and history-conditioned, path-dependent scoring (A10) is the natural next step; the public benchmarks used here are snapshot-detectable, so exercising the trajectory assumptions requires data that contains a history-dependent fault."

- **A8 is not demonstrated; the earlier SKAB witness is a variance-floor artifact (B-01), and the basin head is withdrawn.** In the previous revision we reported a basin-agreement head that appeared to help on SKAB (the Skoltech rotor testbed). A follow-up sweep overturned that result: SKAB's high responsibility entropy (0.29) and the apparent basin-head lift (+0.05 at n = 400) are an artifact of the VaDE component log-variance floor of log(0.05) on a 400-window dataset, not of point-level regime overlap. Refitting the same SKAB latent with empirically estimated component variances drops the entropy to 0.045-0.048, indistinguishable from WADI's 0.053-0.057, and both the ρ gate and the basin-head lift collapse to about zero at a floor of 0.01 or with empirical variances, on every seed (`_diagnostics/fable_a3_spaces.md` §7). We therefore **withdraw the basin head from the shipped model** and reclassify A8 as specified-but-not-observed. Because the head was auto-gated off (λ = 0) on WADI, HAI, and SWaT, its removal changes no reported number in the results tables. The SKAB basin-witness table is deleted (tables are renumbered accordingly), and §7 now carries the scarcity-and-artifact discussion in its place.

- **A8 is scarce both globally and per subsystem (evidence for the not-observed classification).** A between-regime-overlap screen finds A8 absent both globally and per subsystem across all three benchmarks: every correlation community has ρ < 0.16 (WADI: 44 communities, ρ_max 0.029; HAI: 28 communities, ρ_max 0.155; SWaT: 25 communities, ρ_max 0.122; zero communities at ρ ≥ 0.30 on any dataset). A broader eight-dataset scarcity screen (MetroPT, four SMD server-machine entities, and the Paderborn bearing testbed alongside the three benchmarks; `_diagnostics/a3_rho_screen.json`, `a3_screen_smd_pu.json`) finds responsibility entropy ≤ 0.06 throughout, so no dataset in the study realizes A8. This evidence is now stated in §7.

The result is that A1-A7 are the realized and validated contribution, A8 is specified for completeness but was not observed in any dataset studied (its earlier SKAB signal being an estimator artifact), and A9-A10 are scoped as future work because no public CPS benchmark carries a history-dependent fault to test them on.

## R1-2. The rare-regime-safe nearest-component likelihood lacks direct support: compare mixture-NLL vs nearest-component NLL across imbalance ratios, with attention to the false-positive rate of rare normal regimes. (item B-05 / Ed-near)

**Response: Rebuttal plus a targeted comparison.**

The head-level decomposition already isolates the nearest-component term: Table 5 reports nearest-component NLL as a standalone head (WADI 0.719, HAI 0.797, SWaT 0.723) alongside the mixture density and the density-plus-nearest base. The design rationale is stated in Table 1 (A5): "Score against the *nearest* component, never the π-weighted mixture, so a rare-but-valid regime is not penalized."

The direct point is that on these three benchmarks nearest-component and mixture NLL are near-identical, because the benchmarks lack the regime imbalance that separates them: their normal windows sit almost entirely within a single dominant regime (the same reason A8 is inactive; R1-1). The nearest-component likelihood is a rare-regime *safety* property, and demonstrating its benefit requires an operating-regime imbalance the public benchmarks do not exhibit. We therefore support it with a controlled imbalance sweep (mixture-NLL vs nearest-component NLL vs imbalance-aware variants, measuring the false-positive rate on rare normal regimes at π < 2%), with the invariant that at balanced occupancy all scores agree within seed noise. This is a rebuttal backed by the Table 5 parity result rather than a headline table claim, since claiming a benefit on data that does not stress the property would overstate it.

## R1-3. The regime-community model trains many models; evaluate its practical cost under the same hardware as the baselines. (items B-03, A-30 / Ed-cost)

**Response: Change made (cost benchmarked and folded into §7, Table 7).**

We benchmarked training and inference cost for LatAD (global and regime-community) and the deep baselines on one shared machine plus a CPU edge proxy. Measured results:

- **Training wall-clock (WADI / HAI / SWaT):** LatAD-global 3.2 / 16 / 3.5 s, against TranAD 231 / 898 / 163 s. LatAD trains roughly one to two orders of magnitude faster than the deep baselines.
- **Per-window inference latency (batch 1):** 0.7-13 ms for the global model, 17-52 ms for the full regime-community stack.
- **CPU edge proxy (batch 1):** 0.6-14.6 ms per window.

These place both configurations within gateway-class real-time budgets, and give the CalexNet early-exit remark (§7) a concrete base latency to reduce. The deployment paragraph in §7 previously read "We report no parameter count, memory, latency, or energy measurements here"; that sentence is now replaced with the measured numbers and a compact cost table (Table 7) is added, including parameter counts (100-209k global, 0.78-1.25M community) and host memory (~3.5-3.9 GB).

## R1-4. Time-correlated windows: use a time-aware statistical test on all three datasets, and clarify the WADI confidence interval and p-value. (items A-22, A-23, A-24 / Ed-stat)

**Response: Change made.**

Every significance statement in the paper is on an **episode-block bootstrap** that resamples whole attack episodes together with contiguous normal blocks, rather than independent windows. We lead with this as the time-aware test the reviewer asks for, and we now apply it uniformly to all three datasets and to every reported CI and P-value.

- **HAI (§6):** "an episode-block bootstrap of the paired difference (resampling attack episodes and normal windows, averaged over five seeds) gives +0.092 with a 95% confidence interval of [0.047, 0.157]."
- **Canonical SWaT (§6):** "LatAD reaches 0.840 against the strongest linear baseline 0.782, and the same bootstrap makes the margin significant (+0.058, 95% CI [0.015, 0.107], P = 0.003)."
- **WADI CI/P clarified (A-23, A-24, §6):** "the lead over the strongest baseline is numerical (+0.037, 95% CI [−0.088, 0.181], P = 0.32 over 11 episodes)." The reported CI is the two-sided 2.5/97.5 percentile of the paired difference; P is the one-sided bootstrap probability that the difference is at most zero. WADI's difficult subset spans 11 episodes, so we state plainly that the lead is not statistically separable from the linear baseline at that episode count, and we no longer describe it as "near-significant."
- **Double-hard subset (Table 4):** the same block bootstrap gives HAI +0.089 (95% CI [0.022, 0.186]) and canonical SWaT +0.087 (95% CI [0.029, 0.151], P = 0.0005); WADI rests on 10 episodes and is a numerical lead (+0.083, P = 0.22).

All CIs were recomputed from persisted scores, with no retraining; the invariant that a block length of 1 reproduces the prior per-window numbers holds.

## R1-5. Six static window statistics: run raw-sequential / temporal / spectral experiments and ablations to isolate the source of the gain. (items B-04, A-25, A-29 / Ed-abl)

**Response: Change made (source of gain isolated and now in the manuscript).**

We isolated where the gain comes from with a channel-independent-versus-cross-channel density ablation, recomputed over five seeds on the corrected datasets (`_diagnostics/e5_gain_clean.json`). Replacing the cross-channel latent density with a channel-independent product of per-channel densities collapses the difficult-subset advantage on the two datasets where LatAD wins significantly: the cross-channel latent density adds **+0.200 on HAI and +0.141 on canonical SWaT** over the channel-independent baseline. On the artifact-free WADI difficult subset the same swap changes essentially nothing (**−0.004**): WADI's difficult anomalies are single-channel or linear, so per-channel marginals already suffice. The earlier **+0.343 WADI** figure was an artifact of the rescaled `2B_AIT_002_PV` channel, now removed (§5.1); the corrected picture is that the cross-channel gain is carried by HAI and SWaT. The gain is therefore cross-channel density modeling, not the window representation.

This is reinforced by two arguments already in the manuscript: (a) the benchmark literature [ref 20] shows SWaT/WADI anomalies are univariate on almost all timesteps with no cross-time-only segments, and (b) the difficulty split is defined in channel-statistic space, so the difficult subset is precisely where per-channel information is exhausted. A prior temporal-feature run gave no improvement (WADI 0.726 to 0.672), consistent with the gain being cross-channel rather than temporal; this is available as a rebuttal row. The new source-of-gain table (**Table 5**, §6) reports the cross-channel-versus-marginal arm, and the head-level ablation (**Table 6**) and the factorization/aggregation ablation (**Table A1**) localize the same gain to the density heads and the regime-community factorization.

## R1-Eng. The English could be improved.

**Response: Change made in part; full pass pending.** We revised the front matter and long sentences flagged during restructuring. A dedicated language pass over the whole manuscript (sentence length in §1 and §4.4, parallelism, articles) is applied to the submitted version. (See coordinator note.)

## R1-fig. Figures and tables must be improved.

**Response: Change made.** We added Figure 1 (IIoT system-model context) and Figure 2 (LatAD workflow, assumption-tagged) as vector figures, and renumbered the difficult-subset results chart to Figure 3 with a corrected caption. Tables 3-7 and A1 were regenerated on the corrected, construct-matched data (the former SKAB basin-witness table is deleted with the basin head; R1-1); Table 4's caption now states the windows/episode counts and the significance test per dataset. A new source-of-gain table (Table 5, §6) accompanies the head-level and factorization ablations (R1-5).

## R1-concl. The conclusion must be improved.

**Response: Change made.** §8 was rewritten to match the scoped claims and the corrected significance pattern: "LatAD attains the best overall AUROC on every dataset and leads the difficult subset of all three, with statistically significant margins over the strongest baseline on HAI (paired 95% CI [0.047, 0.157]) and canonical SWaT (P = 0.003), and a lead over every reconstruction-based detector on WADI, whose difficult faults are reconstructable but improbable."

---

# Reviewer 2

## R2-1. Expand abbreviations at first use (LatAD, IIoT, CPS, SCADA, ...). (item A-01)

**Response: Change made.** LatAD is given its full form at first use in the abstract: "Our detector, LatAD (Latent-density Anomaly Detector) ...", and CPS, IIoT, and SCADA are expanded in §1 ("A cyber-physical system (CPS) ..."; "Industrial Internet of Things (IIoT)"; "Supervisory Control and Data Acquisition (SCADA) historians"). The remaining body abbreviations (VaDE, GMM, NLL, HC, BIC, HIL, SOTA, AUROC, F1) are now expanded at first textual use.

## R2-2. Abstract: highlight LatAD's technical contribution, less problem setup. (item A-02 / Ed-abs)

**Response: Change made.** The abstract was rebalanced toward the mechanism: "Our detector, LatAD (Latent-density Anomaly Detector), models normal as a jointly learned latent Gaussian mixture (VaDE) and scores by probability rather than reconstruction, factorizing the density over correlation-community subsystems and combining per-subsystem surprises by cohesion-weighted Higher Criticism; reconstruction re-enters only through an auto-gated residual head." It carries one headline number per claim and closes on the code/checkpoint release.

## R2-3. Introduction paragraph 1: delete "This paper detects them by..."; no contributions in paragraph 1; add CPS background. (item A-03)

**Response: Change made.** The "This paper detects them by..." sentence is struck (rev-del), and §1 now opens with CPS/IIoT background followed by the reachability-versus-probability point: "A cyber-physical system (CPS) couples networked sensors, actuators, and controllers with a physical process, forming the sensing-and-actuation layer of the Industrial Internet of Things (IIoT); monitoring its telemetry for faults and attacks is central to reliability and safety."

## R2-4. "the task is often framed as..." (p2, l71): state which task. (item A-04)

**Response: Change made.** §1 now names it: "Unsupervised anomaly detection for CPS telemetry is often framed as characterizing the anomalies, but that framing is misleading."

## R2-5. The "two obstacles": which does LatAD address, and what does it improve? (item A-05)

**Response: Change made.** A dedicated paragraph in §1 states this: "LatAD addresses the modeling obstacle: it replaces the reconstruction residual of USAD- and TranAD-style deep detectors with a probability score in a jointly learned latent regime space, factorized over the plant's physical subsystems. The evaluation obstacle it addresses by adopting the critique's protocol, raw metrics with difficulty stratification, which we use rather than claim as a contribution."

## R2-6. Related-work restructure: move the AD definition to the Introduction; delete the three LatAD self-descriptions; rewrite §2.3; delete §2.5. (items A-06, A-08, A-09, A-10 / Ed-RW)

**Response: Change made.** The anomaly-detection definition is relocated to §1 ("the task of anomaly detection [1], the unsupervised or semi-supervised flagging of departures from a model of normal behavior ..."). The three LatAD self-descriptions in §2 are struck (rev-del). §2.3 was rewritten as a neutral survey and now cites Deep SVDD, OmniAnomaly, and THOC (refs 28-30). The standalone §2.5 "Benchmark datasets" is deleted with a pointer to §5.1.

## R2-7. Add a system-model / application-scenario diagram showing where in the IIoT LatAD collects data and detects. (item A-16 / Ed-fig)

**Response: Change made.** Figure 1 (§1) shows the IIoT monitoring context: "field sensors and PLCs feed an edge gateway or SCADA historian that aggregates and windows the multivariate telemetry, LatAD scores each completed window online, and an anomaly alarm together with the most-surprised subsystem is forwarded to the operator, maintenance, or security stack." It is referenced from the §1 text.

## R2-8. Explain the relationship between MIIM (§3) and the LatAD detector (§4). (item A-14)

**Response: Change made in part.** §3 states the relationship explicitly: MIIM is the specification and LatAD is one realization, with each assumption tied to a design choice ("each is grounded in a physical property of CPS operation and each motivates a specific design choice in the detector of §4"), and Table 1 already maps every assumption A1-A10 to the concrete design choice it motivates. A per-assumption "Realized by (§)" column is now added to Table 1 (A1/A6 VaDE regimes §4.1; A3 mixture-density head §4.3; A5 nearest-component NLL §4.3; A7 standardization/whitened residual §4.3 and community factorization §4.4; A8-A10 specified, not observed, future work §7), and the rows are grouped into the realized-and-validated block (A1-A7) and the specified-but-not-observed block (A8-A10), to make the mapping navigable at a glance.

## R2-9. §4 readability: map §4.1-4.4 to pipeline stages and add a LatAD workflow figure. (items A-15, A-17 / Ed-fig)

**Response: Change made.** Each §4 heading now names its pipeline stage: "4.1 Representation: Variational Deep Embedding (VaDE) (the representation stage; Figure 2, top)"; "4.3 The scoring stack (the scoring stage; Figure 2, per-community heads)"; "4.4 Regime-community density (the headline realization) (the subsystem-factorization stage; Figure 2, community fan-out and HC combiner)." Figure 2 (the LatAD workflow, assumption-tagged) is added and referenced from the §4 preamble ("preview its pipeline (Figure 2) before the details").

## R2-10. Is VaDE + density + residual fitted separately per community, or is the encoder shared? Report training time, GPU memory, and inference latency. (items A-21, A-30, B-03)

**Response: Change made (architecture and cost text).** §4.4 now states the architecture explicitly: for each community we fit "a *separate* small VaDE per community, each with its own encoder, Gaussian mixture, and scoring heads, its regime count and latent dimension scaled to the community size |G|; nothing is shared across communities except the input window features, so the model is a stack of independent per-subsystem experts, tens per plant, plus one global expert." The cost numbers (training wall-clock, per-window latency, edge proxy) are reported in the response to R1-3 and are now folded into §7 (Table 7).

## R2-11. z notation conflict (latent vs standardized value): rename z. (item A-18 / Ed-z)

**Response: Change made.** The latent variable keeps the symbol *z*; the standardization operator is renamed `std(·)`; and the trivial-rule per-channel value is renamed *u*, with a disambiguation clause in §5.3: "(*u* denotes the standardized per-channel window mean, distinct from the latent *z* of §4.3)." Tables 3-4 and the prose use *u* throughout.

## R2-12. Typesetting: unify indentation and number all equations. (items A-19, A-20 / Ed-eq)

**Response: Change made (equations); indentation pending in the DOCX build.** All nine display equations are numbered (1)-(9) and right-aligned. Paragraph-indentation unification is a build-time style applied to the submitted DOCX. (See coordinator note.)

## R2-Eng / R2-fig / R2-concl.

**Response:** Addressed jointly with R1-Eng, R1-fig, and R1-concl above.

---

# Academic Editor

## E1 (Ed-A). Validate the unused assumptions (A8, A9, A10).

**Response: A8 scoped as specified-but-not-observed; A9-A10 scoped as future work.** We initially validated a basin head for A8 on the SKAB rotor testbed, but a follow-up sweep showed the SKAB signal (seed-0 ρ = 0.58, +0.05 basin lift on n = 400) to be a variance-floor artifact of the VaDE component log-variance floor (log 0.05): with empirical component variances or a floor of 0.01 both ρ and the lift collapse to about zero on every seed (`_diagnostics/fable_a3_spaces.md` §7). We therefore withdraw the basin head and reclassify A8 (between-regime overlap) as specified for completeness but not observed in any dataset studied. A per-community screen (every community ρ < 0.16) and an eight-dataset scarcity screen (responsibility entropy ≤ 0.06 throughout), both noted in §7, confirm A8 is scarce globally and per subsystem. Because the head was auto-gated off (λ = 0) on the three benchmarks, its withdrawal changes no reported number. A9-A10 are scoped as future work because no public CPS benchmark carries a history-dependent fault. Full detail under R1-1.

## E2 (Ed-near). Provide direct experimental support for the rare-regime-safe nearest-component likelihood.

**Response: Rebuttal plus Table 5 parity and a controlled imbalance sweep.** See R1-2. The nearest-component head is isolated in Table 5; its distinctive benefit requires regime imbalance the benchmarks lack, so it is supported by a controlled imbalance sweep and the design argument (A5), reported as a rebuttal rather than a headline table claim.

## E3 (Ed-cost). Report training and inference cost for edge deployment.

**Response: Change made (benchmarked and folded into §7, Table 7).** See R1-3: LatAD-global training 3.2 / 16 / 3.5 s vs TranAD 231 / 898 / 163 s; per-window latency 0.7-13 ms (global) / 17-32 ms at batch 1, up to ~52 ms at batch 256 (community); CPU edge proxy 0.6-14.6 ms (global). These now replace the "no measurements" sentence in §7.

## E4 (Ed-stat). Use a time-aware statistical test.

**Response: Change made.** Every CI and P-value is on an episode-block bootstrap applied uniformly to all three datasets, with the WADI CI and P-value defined precisely. Full detail under R1-4.

## E5 (Ed-abl). Add ablations isolating the source of the gain.

**Response: Change made (gain isolated and in the manuscript).** The channel-independent-versus-cross-channel density ablation, recomputed on the corrected datasets, shows the gain is cross-channel density (+0.200 HAI, +0.141 canonical SWaT, and essentially zero on cleaned WADI, −0.004; the earlier +0.343 WADI figure was the `2B_AIT_002_PV` artifact). It is reported in the new source-of-gain Table 5 (§6), with the head-level Table 6 and the factorization Table A1 localizing the same gain. Full detail under R1-5.

## E6 (Ed-abs). Clearer abstract and introduction.

**Response: Change made.** Abstract rebalanced to the mechanism (R2-2); introduction opens on CPS/IIoT background, defines the task, states which obstacle LatAD addresses, and removes the paragraph-1 method sentence (R2-3, R2-4, R2-5).

## E7 (Ed-RW). Restructured related-work section.

**Response: Change made.** See R2-6: definition moved to §1, self-descriptions removed, §2.3 rewritten as a survey with Deep SVDD/OmniAnomaly/THOC added, §2.5 deleted. A new §2.1 situates the work in Industrial-IoT and edge-intelligence anomaly detection.

## E8 (Ed-fig). System-model and LatAD workflow figures.

**Response: Change made.** Figure 1 (IIoT system-model, §1) and Figure 2 (LatAD workflow, §4) added and referenced; results chart renumbered to Figure 3. See R2-7 and R2-9.

## E9 (Ed-z). Resolve the z notation conflict.

**Response: Change made.** See R2-11: latent *z* kept, standardizer renamed `std(·)`, trivial-rule value renamed *u*.

## E10 (Ed-eq). Consistent typesetting with numbered equations.

**Response: Change made.** All nine display equations numbered (1)-(9); indentation unified in the DOCX build. See R2-12.

## E11 (Ed-resp). Submit a point-by-point response with the revised manuscript.

**Response: Change made.** This document is that response; each item cross-references the revised manuscript by section, table, or figure.

---

We believe the revision addresses every point and that the corrected, construct-matched evaluation makes the manuscript stronger. We thank the Editor and Reviewers again for their time.

Sincerely,
Alexander Apartsin and Yehudit Aperstein
