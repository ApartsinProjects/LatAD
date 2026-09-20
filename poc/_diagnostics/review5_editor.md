# Review 5: Handling-Editor Assessment (MDPI IoT)

Manuscript: "Modeling Normal Is All You Need: Joint Latent Clustering for Anomaly Detection in Multimodal Industrial-IoT Cyber-Physical Systems" (Apartsin, Aperstein). Source read: `poc/paper/IoT2.html`, all revision marks accepted. Line numbers below are HTML source lines of that file.

Role: handling editor deciding what to do with the manuscript before/after peer review. Report only.

---

## 1. Scope fit

**Verdict: in scope. The IIoT framing is genuine at the application level and retrofitted at the method level; that is acceptable for this journal, but the seams show.**

- The problem (unsupervised anomaly detection from multivariate sensor/actuator telemetry of industrial control testbeds, WADI/HAI/SWaT) is squarely the journal's Industrial IoT / CPS security-and-reliability remit. §5.1 (L667) makes the case explicitly: "All three are instrumented Industrial-IoT / industrial-control-system testbeds."
- Where the IIoT framing is real: Figure 1 (L108-157) places the detector in the edge/historian tier; §7 "Implications for IIoT deployment" (L1147-1154) and Appendix B (L2853-2865) measure training cost and per-window latency on a CPU edge proxy and state host-RAM footprint. Table 7 subsystem localization (L1177-1182) is an operator-facing output. These are not decorative.
- Where it is bolted on: nothing in the method (§4) is shaped by an IoT constraint. The design is driven by a statistical argument (reconstruction measures reachability, density measures probability) that would read identically in a generic time-series venue. §2.1 (L205) cites five papers from this journal ([4], [5], [6], [16], [17]), several of which concern network-packet intrusion detection, not process telemetry; a reviewer will read that paragraph as venue-pleasing. The edge-inference paragraph (L1185-1194) invokes the authors' own CalexNet [57] as a "further-reduction route" that is never applied.
- Net: the editor would not desk-reject on scope. The paper belongs in the journal's CPS/IIoT anomaly-detection stream and the deployment material was clearly written for this readership.

## 2. Contribution

**What the editor understands the contribution to be after one reading:** a normal-only detector for CPS telemetry that scores each window by its probability under a jointly learned latent Gaussian-mixture model (VaDE) instead of by reconstruction error, factorizes that density over unsupervised correlation-community subsystems and combines the per-subsystem surprises by cohesion-weighted Higher Criticism; under raw (non-point-adjusted) metrics on the anomalies a per-channel range rule cannot separate, it beats USAD, TranAD and GDN on HAI and SWaT with bootstrap significance, and the deep detectors collapse to near chance on HAI.

**Clarity:** clear. The three-contribution list (L178-193) and the one-line mechanism in the Conclusion (L1217-1219: "Reconstruction scores reachability; a density in a clustered latent scores probability") are quotable and consistent with the tables.

**Novelty:** moderate, and of the recombinant kind. VaDE [27], DAGMM-style GMM energy [28], Ledoit-Wolf whitening, agglomerative community detection, and Higher Criticism [49] are all off-the-shelf. The new thing is (a) the diagnostic argument that reconstructable-but-improbable faults are the discriminative minority on CPS benchmarks, backed by the reachability/probability decomposition (Table 2, Table 6, the 93rd-percentile vs median statistic at L1041-1043), and (b) the subsystem factorization with an HC combiner as a localizing, sparsity-adaptive fusion. Contribution (1), the MIIM assumption list A1-A10, will be read by reviewers as a taxonomy, not a contribution; three of the ten are explicitly "not realized" (Table 1, L315-327) and one of the seven realized ones (A2, "regime explosion") is realized only by "use a high K".

**Oversold or undersold?** Both, in different places.
- Oversold: the MIIM apparatus. The abstract spends a full clause on "ten ... assumptions, of which seven are realized ... and three are specified for future work" (L82), Table 1 carries three rows of things the model does not do, §7 has three long paragraphs on A8/A9/A10 (L1098, L1117-1130, L1132), and Appendix C is a whole appendix demonstrating that A8 is absent from every dataset ("A8 is specified for completeness but not demonstrated on any dataset in this study", L2935). That is a null result presented at the scale of a finding.
- Oversold: "leads the difficult subset of all three" (L192, L1017, L1212). On WADI the lead over the linear leave-one-channel-out baseline is +0.020 with 95% CI [-0.161, 0.205], P = 0.46 (L857-858), and the same linear baseline beats LatAD on WADI All (0.834 vs 0.827, L837-838). The abstract says "statistically tied"; the Discussion opener (L1017) drops the qualifier.
- Undersold: the two things an IIoT practitioner would actually cite. (i) Training cost: LatAD (global) trains in 3 to 16 s vs 163 to 898 s for TranAD (L2853-2855), i.e. 47 to 72 times faster, on a model with 100 to 209 k parameters; this is in Appendix B. (ii) Subsystem localization: a three-community shortlist contains the attacked subsystem in 68 to 79% of HAI/SWaT attacks vs 49 to 56% random (Table 7); this is buried two-thirds of the way through the Discussion.

## 3. Significance and interest to the readership

**Would readers cite it?** Yes, for two reasons: it is one of the few CPS papers that re-scores USAD/TranAD/GDN under raw metrics with difficulty stratification on all three standard testbeds, and it ships a cheap, label-free detector with a subsystem pointer. The raw-metric re-computation alone (Table 3, L781-830; Table 4) is a service to the field.

**Strongest selling point:** the HAI difficult-subset result. LatAD 0.845 ± 0.018 vs AutoEncoder 0.757, USAD 0.477, TranAD 0.444, GDN 0.481 (L812-819), with a paired episode-block bootstrap giving +0.088, 95% CI [0.042, 0.157] (L845-847), robust to a stronger six-statistic difficulty definition (L878-886) and to the double-hard intersection subset (Table 4). The mechanism evidence (Table 5: cross-channel gain +0.200 on HAI, +0.141 on SWaT, +0.017 on WADI) matches the story exactly where the wins are significant. Editors like a result whose ablation and mechanism agree.

**Weakest link a reviewer will attack:** the linear baseline. LinRes (a leave-one-channel-out linear regression, L756-758) is within noise of the headline on WADI Difficult, ahead on WADI All, second on SWaT All (0.920 vs 0.938), and beats every deep detector on every dataset's difficult column. A reviewer versed in Sarfraz et al. [25] and Garg et al. [8] will say: the paper re-discovers that reconstruction-based deep detectors underperform simple baselines under raw metrics (already known), and the new detector is the best of the simple models on 2 of 3 datasets. The paper's own Table 5 (L1007-1009) concedes that on WADI "per-channel marginals already suffice." The authors have pre-empted this fairly (the abstract states the tie; §6 L857 states the CI) but the framing "the deep detectors collapse" carries the paper further than the WADI numbers allow.

Secondary attack surfaces:
- Sample sizes on WADI: 56 anomaly windows, 30 difficult, 8 attack episodes (L799, L854) after 10x downsampling. Table 4's WADI double-hard subset rests on 19 windows / 7 episodes (L910).
- Single-seed SOTA on two of three datasets: "USAD and TranAD five-seed on HAI only and single-run on WADI and canonical SWaT" (L735, L759); GDN single-seed everywhere (L764). The headline claim "ahead of every learned detector on WADI" is then a comparison of a five-seed mean against single runs.
- The AutoEncoder baseline is close on HAI All (0.923 vs 0.948, L813/L819); the All-subset headline in the abstract is a 0.025 margin over a plain autoencoder.

## 4. Readability for the journal's audience

**Overall:** well written at the sentence level, self-contained, and consistent in its numbers across abstract, tables and text (I cross-checked every headline number in the abstract against Tables 3, 4 and 6; all match). But it is long, dense, and front-loaded with framing; a busy editor loses the thread in three places.

Where the thread is lost:
1. **The Introduction before the contributions** (L88-176) runs roughly 1,400 words and restates the reachability/probability point four times (L88, L162, L166, L174) before the contribution list. The "note on scope" paragraph (L176) about benchmark scale belongs in §2.5, where the same material appears again (L257).
2. **§3 and Table 1** (L259-342). The reader must hold ten assumptions, a "Motivates" column and a "Realized by" column, then learn that three assumptions are not realized, then absorb a paragraph of BIC and silhouette diagnostics (L333-342) whose relevance to the method is asserted ("This ordering tracks where the design pays off") rather than shown. The BIC-optimal K* "of 22, 24, and 25 ... on a component grid extended to 25" (L335) means the SWaT optimum sits at the grid ceiling, which a careful reader will notice.
3. **The proliferation of near-identical model names and numbers.** The paper reports, for what is nominally the same single-latent detector on HAI Difficult: 0.760 (Table 2, single model), 0.789 (Table 5), 0.802 (Table 6 "latent density"), 0.811 (Table 3 / Table A1), 0.820 (Table 6 "base + resid = LatAD (global density)"), and 0.743 (Table A2, "reduced global-density configuration"). Each is explained locally ("an independent five-seed run", "reproduces its Table 3 row within seed variation", "not the headline numbers"), but the cumulative effect is that the reader cannot say which number is *the* number. Likewise on WADI the community result is quoted as 0.795 (HC only) and 0.771 (fused headline) in the same paragraph (L859-862), and the Discussion repeats the pair (L1022-1024, L1062-1063).

Other readability notes:
- §7 Discussion (L1013-1205) is roughly 2,600 words and contains a results table (Table 7) and a new quantitative finding (localization); it is longer than §6 Results. The A8/A9/A10 paragraphs (L1098, L1117-1132) and the "residual frontier" paragraph (L1100-1105) are future-work material.
- Table 2 covers only WADI and HAI (L512-516); SWaT is absent without explanation, and the caption (L506-511) is a paragraph of argument that repeats the body text at L520-526.
- Acronym load is high (MIIM, VaDE, GMM, KDE, NLL, HC, HIL, PLC, SCADA, ICS, SOTA, AUROC, LinRes) and there is no Abbreviations list.
- Figure 3 (L937-997) is a bar chart of one column of Table 3; it adds no information beyond the table.

## 5. Presentation and compliance

- **Title** (L6): "Modeling Normal Is All You Need" is a meme title; the subtitle carries the content. MDPI will accept it; some reviewers will not. Not a blocker.
- **Abstract** (L82): 229 words against the journal's 200-word limit. The second sentence is about 90 words long and contains the "ten assumptions, seven realized, three future work" bookkeeping, which is not abstract material. Otherwise the abstract has the right arc (problem, mechanism, method, three-dataset result, code release) and its numbers are correct.
- **Keywords** (L84): eight, within the 3-10 range; "anomaly detection" and "IoT anomaly detection" overlap.
- **Figures:** four (Figs 1, 2, 3, A1). Figure 1 and Figure 2 are inline SVG block diagrams; readable, captions complete. Figure A1 is a matplotlib SVG; its caption (L2851 region) states it is synthetic. The embedded SVG metadata carries a generation timestamp (L2305) and Matplotlib version string; harmless but should be stripped in the production file. Figure 3 is redundant with Table 3 (see above).
- **Tables:** eleven (Tables 1-7, A1, A2, B1, C1). Two compliance issues:
  - *Bold means three different things.* Table 3: best AUROC per column (L785). Table 5: "the two datasets on which LatAD's lead ... is significant" (L1004). Table 6: "the LatAD (global density) configuration, not a column maximum" (L1071). Table 7: "the two datasets with enough episodes to test the margin" (L1177). A journal copy-editor will flag this; a reader will misread Table 6.
  - *F1 precision is inconsistent within Table 3:* two decimals on WADI (0.62, 0.39, 0.55 ...) and three on HAI and SWaT (0.666, 0.418 ...).
  - Table 3 is dense (9 methods x 3 datasets x 6 numbers) but is the right table; keep it.
- **Equations:** (1)-(9), numbered and referenced; Eq. (8) is stated as an approximation ("over-complete pooling rather than an exact factorization", L623-624), which is the right level of care. Eq. (7) is called "the detector's complete window score" (L594) but the headline detector's score is the HC fusion of §4.4 (L655-658); the sentence at L594 is true only of the global-density ablation.
- **References:** 59 entries in MDPI style with DOIs. Six author self-citations ([42]-[45], [57], and the arXiv-free ones are fine). Three 2026 arXiv preprints ([18], [21], [23]) carry a load-bearing claim each (multi-product blind spots; "anomalies are mostly univariate"; "autoencoders are unreliable"); reviewers may ask for published versions. Five citations to this journal in one paragraph (§2.1) look like venue-pleasing.
- **Section balance:** Discussion (~2,600 words) longer than Results (~1,500); Introduction (~1,400) longer than Method sections need it to be. Appendices A-C total ~1,200 words plus four tables; Appendix C documents a null result.
- **Back matter (L1228-1233):** Author Contributions, Funding, IRB, Informed Consent, Data Availability, Conflicts of Interest are present in MDPI form. Missing: an **Abbreviations** list (recommended for a paper with this acronym density) and a **generative-AI use** statement, which MDPI now requires when AI tools were used in drafting; the editor would ask the authors to confirm one way or the other.
- **Accessibility of the difficulty split:** §5.3 (L718-727) defines it cleanly and the paper is candid that F1 uses a test-swept oracle threshold (L736-743). Good.

## 6. Red flags

Ranked by how the editor would weigh them.

1. **Data Availability Statement does not match the deposit (hard pre-review blocker).** L1232 states: "The trained model checkpoints (one per dataset) and the source code supporting the reported results are openly archived on Zenodo at https://doi.org/10.5281/zenodo.21821524 ... each is trained with a fixed seed and reproduces that seed's scores, and the released scripts reproduce the five-seed aggregates reported in the tables." The project's own diagnostic (`poc/_diagnostics/review2_stale_gaps.md`, item 12, fetched 2026-09-18) records that the live record contains WADI, HAI and SKAB checkpoints of a single-latent configuration, no SWaT checkpoint, no regime-community (headline) model, and a WADI model predating the 122-channel clipped retrain. As written, the statement promises artifacts the record does not contain. This is the one item an editor treats as an integrity matter rather than a revision request: the statement must describe what the record holds, or the record must be updated (v2 deposit) before the manuscript goes to reviewers. The same statement also says "created no new data" while the paper generates synthetic controls (Figure A1; the 30%-overlap positive control of Table C1) and uses Cranfield, SKAB, MetroPT, SMD and Paderborn in Appendix C (L2890-2892) without saying where those are obtained.

2. **Model-selection protocol is not stated.** §5.2 (L709-715) gives per-dataset VaDE component counts and latent dimensions (WADI 20/10, HAI 40/16, SWaT 40/16) and says settings are "shared across datasets except the per-dataset mixture size and latent dimension," but never says how those two were chosen. §7 (L1143) says "The community construction and combiner settings were fixed before the WADI and canonical SWaT evaluations, so those two results are out-of-sample confirmations," which implies HAI served as the development set. If so, the HAI margin (the paper's strongest result) is in-sample with respect to design decisions, and the "no test-set tuning" sentence at L594 needs a precise scope. This is a standard major-revision request, not a rejection ground, but it must be answered explicitly.

3. **Headline wording exceeds the WADI evidence.** "Leads the difficult subset of all three" appears without qualifier at L1017 and L1212 while the WADI comparison against LinRes is P = 0.46 with a CI spanning zero (L857-858). The abstract's "statistically tied with the linear baseline" is correct; the body should be brought to the same standard everywhere.

4. **Single-seed deep baselines on two of three datasets** (L735, L759, L764). Not a red flag on its own, but combined with item 3 it means the WADI "ahead of every learned detector" claim rests on five-seed LatAD vs one-run USAD/TranAD/GDN. Reviewers will ask for five seeds or a stated reason.

5. **Missing method detail for reproduction from the paper alone.** No encoder/decoder architecture (layer widths, depth), no optimizer, learning rate, epoch counts or pretraining schedule for the VaDE (§4.1 says only "plain-VAE pretraining, GMM initialization, then joint optimization", L494-495); no statement of how the dendrogram is cut to obtain "subtrees of size 3 to 25" (L609-611, L713-714) or how nested/overlapping communities are selected; no statement of how per-community p-values are calibrated (empirical train-normal tail? fitted distribution?). The Zenodo code would normally cover this, which makes item 1 worse.

6. **Results-versus-claims consistency:** checked and clean. Every number in the abstract, §6 and §7 appears in a table with the same value; the significance statements (HAI CI [0.042, 0.157]; SWaT P = 0.007; WADI P = 0.46; Table 4 HAI P = 0.0005, SWaT P = 0.0005, WADI P = 0.43) are internally consistent. No sign of an unsupported headline.

7. **Ethics:** public testbeds, no human data; IRB/consent "Not applicable" is correct.

## 7. Verdict

**Editorial lean: send to reviewers after a compliance-and-integrity fix; expected reviewer outcome is Major Revision, with a credible path to acceptance.** Not a desk-reject: scope fits, the central result is real, significant and mechanistically supported, and the evaluation protocol is more rigorous than the journal's norm. Not a minor revision either: the deposit statement, the model-selection disclosure, and the length/framing problems are substantive.

**The three things the editor would require before sending it back to reviewers:**

1. **Make the Data Availability Statement true.** Either publish a v2 Zenodo deposit containing the SWaT checkpoint, the regime-community (headline) models for all three datasets, the post-fix 122-channel WADI model, and the scripts that reproduce Tables 3-7, or reword the statement to describe exactly what the current record holds. State where the synthetic controls and the Appendix C datasets (Cranfield, SKAB, MetroPT, SMD, Paderborn) come from, and delete "created no new data" or qualify it.

2. **Add a model-selection and development-set paragraph to §5.** State which dataset(s) were used to choose K, latent dimension, M = 80, the 1.5 gate ratio, window/stride and the community size bounds; state which results are in-sample with respect to those choices and which are out-of-sample; and either run USAD/TranAD/GDN over five seeds on WADI and SWaT or say why not. Add the missing training hyperparameters (architecture, optimizer, learning rates, epochs, pretraining schedule) and the p-value calibration rule.

3. **Cut to journal form.** Abstract to 200 words or fewer (drop the "ten/seven/three assumptions" bookkeeping). Move Table 7 and the localization paragraph into Results. Compress the A8/A9/A10 material (three §7 paragraphs plus Appendix C) to one paragraph of stated boundaries; a null screen does not need its own appendix in the main submission. Unify the bold convention across Tables 3, 5, 6 and 7 and the F1 decimal precision in Table 3. Add an Abbreviations list and the generative-AI statement. Bring every "leads the difficult subset of all three" to the abstract's "tied with the linear baseline on WADI" standard.

**The one sentence the editor would write to the authors:**

"The manuscript presents a well-supported and useful result for IIoT anomaly detection, but before it can be sent for review the Data Availability Statement must accurately describe the Zenodo deposit, the model-selection protocol must be stated explicitly, and the abstract and framing must be trimmed to journal form with the WADI claim stated at the same strength throughout."
