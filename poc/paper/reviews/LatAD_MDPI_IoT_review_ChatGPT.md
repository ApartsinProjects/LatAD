# Referee Report for MDPI *IoT*

**Manuscript:** *Modeling Normal Is All You Need: Joint Latent Clustering for Anomaly Detection in Multimodal Cyber-Physical Systems* (LatAD)

**Recommendation: Major Revision**

## Overall assessment

The manuscript has a potentially publishable core idea: evaluate multivariate CPS anomaly detectors under raw metrics, separate trivially univariate anomalies from harder cases, and ask whether a clustered latent-density score detects low-probability joint states that reconstruction-based scores miss. The paper is also unusually explicit about weaknesses of common time-series anomaly-detection evaluation, and Section 6 does not hide the fact that SWaT is largely non-discriminative under the proposed split. The HAI result is the strongest empirical evidence: on the difficult subset the proposed detector reaches AUROC 0.811±0.016 versus 0.757±0.003 for the plain AutoEncoder, 0.627±0.005 for Isolation Forest, 0.497 for USAD, and 0.445 for TranAD (Table 3).

However, the current manuscript turns this useful empirical observation into a mechanism claim that is stronger than the evidence supports. The central “reconstruction = reachability, latent density = probability” narrative is plausible as an intuition, but it is not established as a general mechanism by the experiments presented. More importantly, several results in the manuscript directly complicate that story: (i) the plain AutoEncoder is strong on HAI-difficult (0.757), so reconstruction does not generally “collapse” there; (ii) the headline model itself activates a reconstruction-residual head on HAI and SWaT (Section 4.3(iv)), despite repeated claims that reconstruction is dropped and the score is “latent-only”; and (iii) Table 2 labels reconstruction as “at or below chance” although its HAI difficult-subset AUROC is 0.689. These are not cosmetic issues because they affect the causal interpretation of the contribution.

The proposed difficulty split is leak-free with respect to model fitting, but it is not yet sufficiently validated as a method-independent definition of “joint-structure difficulty.” It conditions the evaluation on failure of one specific univariate statistic (maximum absolute standardized channel mean), and failure of that statistic does not imply that an anomaly is cross-channel, between-mode, or dynamic. The difficult subsets are also small (19 anomalous windows for WADI and 38 for SWaT), while overlapping windows make these counts even less informative as independent sample sizes. Five random seeds quantify some optimization variability but do not establish inferential significance or dataset/episode uncertainty.

I therefore recommend **Major Revision**, not Reject. The HAI result and the raw-metric evaluation are worth preserving, but the paper needs a more falsifiable mechanism analysis, a more robust difficulty protocol, modernized baselines, and statistical treatment that matches the strength of the claims.

---

## 1. Central mechanism: “reachability versus probability”

### What is sound

The conceptual distinction in Section 7 is useful. A sufficiently expressive autoencoder can reconstruct inputs that were not probable under the training distribution, so low reconstruction error is not, in general, equivalent to high probability under normal data. The paper correctly connects this to earlier concerns about autoencoder anomaly detection in Section 2.1 and then provides a model-specific decomposition in Table 2. On WADI-difficult, the contrast is meaningful: the VaDE reconstruction residual is essentially chance (0.490), whereas the nearest-component latent NLL reaches 0.663. The WADI diagnostics in Section 7 further report reconstruction-error AUROC 0.43 versus latent-density AUROC 0.74, which is directionally consistent with the hypothesis.

The between-mode probe is also the right *kind* of experiment. Constructing low-density combinations from two normal modes and asking whether reconstruction and density rank them differently directly targets the proposed mechanism. The reported result—reconstruction AUROC 0.08 and latent-density AUROC 0.72 for averaged cross-mode normal windows—shows that the specific trained model can assign low reconstruction error to deliberately interpolated inputs that its latent density regards as atypical.

### Why the evidence is not yet sufficient

The manuscript currently presents the mechanism as if it were almost definitional: Section 7 states that reconstruction “measures reachability” while latent density “scores probability,” and the Conclusion says this is “the mechanism” and that latent scoring is “the fix.” That is too strong. Reconstruction error is a property of an encoder-decoder, its bottleneck, regularization, objective, architecture, and training trajectory; it is not a formal reachability functional. Likewise, a density in an encoder latent is not automatically the data probability \(p(x)\). The encoder can distort volumes, and the manuscript’s final score is a combination of a high-K latent GMM NLL and an unweighted nearest-component NLL, not a calibrated likelihood of the observed CPS state. The paper should call this a **latent typicality/probability surrogate**, unless it provides a stronger probabilistic derivation.

There are also internal empirical counterexamples to the broad claim. In Table 2, HAI reconstruction residual AUROC is 0.689—clearly above chance—while the latent NLL is 0.760. The Table 2 caption nevertheless says reconstruction is “at or below chance on the correlation-break faults.” That statement is false for HAI as reported. More importantly, Table 3 shows a plain AutoEncoder at 0.757±0.003 on HAI-difficult, only 0.054 below LatAD. Thus the evidence supports a narrower claim: **some reconstruction objectives/models fail badly on some hard anomalies, and latent scoring helps in the tested VaDE model**, especially on WADI. It does not support the categorical claim that reconstruction cannot separate HAI difficult faults from normal.

The strongest inconsistency is architectural. Section 4.2 says “we drop reconstruction”; Section 7 says “we drop reconstruction entirely”; the Abstract calls the contribution a “latent-only anomaly score.” Yet Section 4.3(iv) defines an explicit responsibility-weighted whitened reconstruction residual, and the auto-gating summary states that this head is **on for HAI and SWaT**. Therefore the headline `VaDE-hard+resid(auto)` result on HAI—the paper’s lead result—is not latent-only. The residual may be transformed and mode-conditioned, but it is still derived from \(x-\hat{x}\). The manuscript must either (a) change the claim to “latent-primary scoring with an optional reconstruction-residual head,” or (b) make the strictly latent-only model the headline and show that the conclusions survive.

The WADI between-mode probe is promising but under-described. Section 7 introduces new AUROCs (PCA off-subspace 0.43, reconstruction 0.43, latent density 0.74, synthetic cross-mode reconstruction 0.08, latent density 0.72) without a corresponding protocol in Section 5. It is unclear how many probe samples were created, how “different modes” were defined, whether mode assignment comes from the same VaDE whose density is being tested, how pairs were sampled, whether averaging occurred in raw time-series space or the six-statistic window-feature space, how physical feasibility was checked, and whether the numbers are single-seed or multi-seed. If “different modes” are defined by the same model used to demonstrate low latent density, the probe risks partial circularity.

### Required mechanism experiments

The authors should convert the reachability/probability story into a falsifiable hypothesis and add at least the following:

1. **Full score ablation over five seeds on all three datasets.** Report, separately, VaDE reconstruction residual; VaDE prior/nearest NLL; high-K latent density; nearest-component NLL; density+nearest; density+nearest+residual; and the final auto-gated configuration. This is necessary to establish which head causes each gain. Table 2 is currently single-model and only covers WADI/HAI.
2. **A controlled interpolation test with negative controls.** Sample pairs within the same mode and across different modes; interpolate with \(x_\alpha=(1-\alpha)x_i+\alpha x_j\) for several \(\alpha\) values; plot reconstruction error and latent score against \(\alpha\). A genuine between-mode mechanism predicts a systematic density valley for cross-mode interpolation but not within-mode interpolation.
3. **Mode definitions independent of LatAD.** Where possible, use actuator/setpoint regimes or an independently fitted clustering to define the probe pairs. If VaDE modes are used, explicitly acknowledge the dependence and repeat with an external partition.
4. **Marginal-preserving joint-structure corruption.** Construct anomalies by channel permutation, cross-channel pairing from mismatched normal regimes, or other perturbations that preserve per-channel marginals while breaking dependence. This would test the precise claim better than averaging alone.
5. **Architecture/capacity check.** Repeat the reconstruction-vs-density comparison for at least a plain AE and the VaDE decoder (ideally at two bottleneck capacities). If the effect vanishes with decoder capacity or model family, the paper should present it as model-dependent rather than universal.
6. **Explicit falsification criterion.** State what result would disconfirm the mechanism—for example, if a reconstruction model consistently separates independently defined marginal-preserving joint faults as well as the latent score, or if the latent score’s advantage disappears under a mode-independent probe.

Until such experiments are added, I would treat the mechanism as an informed explanation, not an established causal finding.

---

## 2. Difficulty stratification: principled, leak-free, but not yet construct-valid

Section 5.3 defines the trivial score as the maximum absolute standardized per-channel **window mean** and labels an anomalous window easy if that score exceeds the 99th percentile of train-normal. This is training-leakage-free in the conventional sense: the threshold is calibrated only on train-normal, and the LatAD score is not used to define its own test subset. That is a strength.

However, the stronger interpretation given in Sections 5.3 and 6 is not justified. The paper states that difficult windows are “therefore the correlation/dynamics faults that no univariate range rule separates.” The word “therefore” is doing too much work. Failure of a max-absolute-*mean* rule can occur for many reasons that are still entirely univariate: a variance change with stable mean, an altered range, a transient spike diluted by the window mean, slope/trend changes, frequency changes, or modest single-channel deviations below the chosen threshold. Indeed, the model input itself uses six per-channel statistics (mean, standard deviation, minimum, maximum, first-to-last difference, and range; Section 5.2), while the difficulty rule uses only one of them. A window can thus be “difficult” by definition while being trivially detected by another univariate statistic already available to the model.

This matters because the paper interprets success on the difficult subset as evidence of detecting **joint structure**. The current split establishes only “not detected by this particular max-mean rule.” It does not establish cross-channel dependence violation. The 2026 work already cited by the manuscript (Pinet et al.) explicitly distinguishes univariate deviations from cross-channel correlation changes on a per-segment basis. The present stratification should either adopt a comparable diagnostic taxonomy or use more cautious language.

There is also a selection-bias issue. Because the split is defined by one baseline’s failure, that baseline must look poor on the difficult subset by construction, and methods whose scores are strongly correlated with the same marginal statistic are disadvantaged. This is not leakage, but it means the difficult column is a **conditional benchmark chosen relative to one detector**. That can be scientifically useful if framed as such, but it cannot by itself establish a universal notion of anomaly difficulty.

The one-hot linear residual (LinRes) is a useful attempt to challenge the joint-structure interpretation, but it is under-specified in Section 5.5. The manuscript should define exactly what is predicted (each of six summaries separately or a vector), how discrete actuator channels are identified and one-hot encoded after window aggregation, what regularization is used, whether fitting is per target/channel, how residuals are standardized and aggregated, and whether all fitting is strictly on train-normal. On SWaT-difficult LinRes reaches 0.959, essentially equal to LatAD’s 0.960, which is important evidence that the subset is not uniquely suited to nonlinear latent modeling.

### Required stratification revisions

1. **Sensitivity analysis over the split threshold.** Repeat the difficult-subset AUROC at, for example, the 95th, 97.5th, 99th, and 99.5th train-normal percentiles. The paper’s own limitation paragraph notes that the boundary changes with the percentile; this should be demonstrated rather than asserted to be non-knife-edge.
2. **A stronger univariate difficulty detector.** At minimum define difficulty using the maximum anomaly score across all six standardized per-channel window statistics, not only the mean. Better still, use a simple per-channel one-dimensional detector and aggregate its score. If LatAD retains its advantage after excluding anomalies detectable by *any* simple univariate statistic, the “joint-structure” claim becomes much stronger.
3. **Independent anomaly taxonomy.** For each attack/event, report whether it contains (a) univariate excursion, (b) cross-channel relation change, (c) temporal/dynamic change, or combinations thereof. This can be based on a published diagnostic method or a transparent analysis independent of LatAD.
4. **Report event counts, not only anomalous-window counts.** WADI has only 19 difficult anomalous windows and SWaT only 38 (Table 3). Because windows overlap (W=60, stride=30), these are not independent observations. The manuscript should state how many distinct attack episodes contribute to each subset.
5. **Do not call the split “fair” without qualification.** “Train-normal calibrated and method-independent with respect to LatAD” is defensible; “fair” is too broad until alternative split definitions give similar conclusions.

---

## 3. WADI / HAI / SWaT evidence and the cross-dataset story

The cross-dataset results are more coherent than a superficial reading suggests, but the paper overstates how uniformly they support the proposed model.

### HAI

HAI is the strongest result and should remain the lead example. Table 3 gives LatAD 0.933±0.007 overall AUROC and 0.811±0.016 on the difficult subset. The difficult-subset margin over the plain AutoEncoder is 0.054 and the margin over USAD/TranAD is very large. If the difficult subset is validated as genuinely cross-channel and the statistical uncertainty is handled correctly, this is a meaningful contribution.

At the same time, the HAI result does **not** support the strongest reconstruction-failure language. The AutoEncoder reaches 0.757±0.003, and the VaDE reconstruction term in Table 2 reaches 0.689. The story is therefore “LatAD improves on reconstruction,” not “reconstruction cannot separate these faults.”

### WADI

WADI supports the mechanism analysis better than it supports a performance-superiority claim. The proposed detector obtains 0.690±0.027 on the difficult subset versus Isolation Forest 0.677±0.011. A 0.013 mean difference is smaller than one standard deviation of the LatAD runs and is not evidence of a statistically established win. The manuscript does say that Isolation Forest is “the closest competitor,” which is fair, but Figure 1 and the Conclusion still describe the detector as leading every difficult column in a way that visually/rhetorically suggests a meaningful superiority. On WADI the honest conclusion is **statistical tie or near-tie unless a paired uncertainty analysis shows otherwise**.

This is also scientifically important: Isolation Forest does not implement the proposed joint latent clustering mechanism. Its near-tie implies that WADI-difficult performance may reflect generic low-density/outlier structure rather than something uniquely enabled by VaDE. That does not invalidate LatAD, but it weakens any claim that the WADI result identifies the proposed mechanism.

### SWaT

The manuscript is commendably candid that SWaT is not a strong discriminative test: the trivial rule already scores 0.943 on the difficult subset, LinRes scores 0.959, and LatAD scores 0.960±0.006. This should be emphasized even more strongly. On SWaT the proposed model is effectively tied with a simple linear cross-channel baseline, and the “difficult” subset is not difficult by the paper’s intended construct. SWaT therefore provides evidence that LatAD does not catastrophically degrade on an easy benchmark, not evidence for the latent-density mechanism.

### Cross-dataset conclusion

A defensible synthesis is:

- **HAI:** clear performance evidence for the full method; mechanism still needs ablation.
- **WADI:** useful qualitative/mechanistic case study; performance advantage over Isolation Forest is unestablished.
- **SWaT:** ceiling/triviality case; little evidence for method-specific benefit.

That is a coherent three-dataset story and is actually more credible than the current universal wording. The Abstract and Conclusion should not imply three independent strong wins on difficult joint-structure faults.

---

## 4. Baselines: reasonably honest reporting, but not adequate for a 2026 submission

The paper deserves credit for including simple baselines, especially Isolation Forest, a plain AutoEncoder, the trivial max|z| rule, and LinRes. This is exactly the right response to the evaluation critique summarized in Section 2.1. The authors also re-score USAD and TranAD under the same raw protocol rather than comparing against incompatible point-adjusted values, which is methodologically sound.

The reporting of the Isolation Forest near-tie on WADI is reasonably honest in Section 6: the text explicitly calls it the closest competitor. The problem is not concealment; it is interpretation. A 0.690 versus 0.677 result does not support a strong “best method” claim without an inferential test, and it weakens the claim that explicit latent clustering is necessary on WADI.

The larger baseline problem is age and relevance. By 2026, USAD (2020) and TranAD (2022) cannot by themselves define “modern SOTA,” particularly when the manuscript’s key claim concerns cross-channel structure and reconstruction. Section 2.3 even discusses GDN, a graph-based cross-channel model, but does not evaluate it. This omission is conspicuous because GDN targets exactly the dependency structure that the difficult-subset argument claims to isolate.

The 2024–2026 literature also makes the positioning incomplete. At minimum, the manuscript should discuss:

- **SensitiveHUE (KDD 2024)**, which explicitly modifies reconstruction-based multivariate anomaly detection to increase sensitivity to normal patterns and models heteroscedastic uncertainty. Its existence directly argues against treating “reconstruction-based” as one homogeneous class represented by USAD/TranAD.
- **PATE (KDD 2024)** and related evaluation work, which extends the critique of point-adjusted F1 by providing event/proximity-aware evaluation rather than simply reverting to point-wise metrics.
- **CATCH (ICLR 2025)**, a recent channel-aware MTS anomaly detector that models channel correlations in the frequency domain and remains reconstruction-based. This is particularly relevant to the manuscript’s claim that reconstruction is fundamentally the wrong score for joint anomalies.
- **mTSBench (2025)** and **TAB (2025)** benchmark studies, which emphasize that detector rankings are dataset-dependent and that unified evaluation/model-selection protocols matter. This context is highly relevant to a paper making general conclusions from three classic CPS benchmarks.
- **Pinet et al. (2026)**, already cited, should be integrated more deeply into the difficulty-construct argument rather than used mainly to justify benchmark triviality.

I am not asking the authors to reproduce every 2025–2026 model. A reasonable revision would add **at least one modern channel/dependency method and one recent strong reconstruction model** under the same raw protocol, plus GDN if implementation is feasible. CATCH and SensitiveHUE are particularly informative because they stress-test the paper’s mechanism claim. If reproduction is impossible, the paper should stop calling USAD/TranAD “the modern SOTA” and instead call them “representative historical deep baselines.”

The paper would also benefit from one strong non-deep density/distance baseline on the same window features (e.g., robust Mahalanobis, kNN density/distance, or one-class SVM), because Isolation Forest’s WADI result suggests that a generic density/outlier principle may explain part of the gain.

---

## 5. Statistical rigor and metric choices

### Multi-seed reporting

Five-seed mean±standard deviation for LatAD, Isolation Forest, and the AutoEncoder is better than single-run reporting. However, optimization-seed variability is only one uncertainty source. The key difficult subsets contain 19 WADI and 38 SWaT anomalous windows (Table 3), and those windows overlap by 50%. The dominant uncertainty may therefore come from which attack episodes are present, not from random initialization.

USAD and TranAD are reported without seed variation, despite being stochastic learned models. This prevents a statistically comparable statement such as “LatAD exceeds TranAD by X.” All stochastic learned baselines should be run over the same seed budget, or the paper should explicitly justify why deterministic checkpoints/runs are used.

### No significance is established

The manuscript repeatedly uses language such as “leads,” “best,” and “clears every baseline,” but gives no confidence intervals or significance tests. This is especially problematic on WADI-difficult (0.690±0.027 vs 0.677±0.011 for Isolation Forest) and SWaT-difficult (0.960±0.006 vs deterministic LinRes 0.959). These are not established wins. Mean±SD across seeds cannot answer whether the finite test-set AUROC difference is significant.

The appropriate analysis should account for dependence in the time series. A naive bootstrap over overlapping windows would be invalid or overly optimistic. I recommend an **attack-episode/block bootstrap**, ideally nested with model seeds, or an episode-level paired bootstrap of AUROC/AUPRC differences. DeLong’s test is less attractive here because windows are serially dependent; a block/episode bootstrap is more defensible. At minimum provide 95% confidence intervals for each primary AUROC and for pairwise differences against the strongest baseline.

### F1

The paper is transparent that F1 is the best raw point-wise F1 over test thresholds, i.e., an oracle. That transparency is good, but the metric should not be used as evidence of deployable threshold performance. Table 3 also provides only one F1 number per method with no seed dispersion, and it is unclear whether that F1 comes from one representative seed, the best seed, an averaged score, or some other aggregation. This must be specified.

A deployable evaluation should add a threshold selected without test labels, for example a train-normal or held-out-normal quantile, and report the resulting F1/precision/recall or TPR at a fixed FPR. The oracle F1 can remain as a secondary “score separability” statistic if clearly labelled.

### AUROC alone is insufficient for rare anomalies

Raw AUROC is preferable to point-adjusted F1 for avoiding the known inflation problem, but it is not sufficient by itself. CPS anomalies are rare, and AUROC can look favorable even when precision is poor at operational thresholds. The paper should add **AUPRC** and at least one **low-FPR operating metric** (for example TPR at 1% FPR or FPR at a chosen recall). Because the paper makes a deployment-oriented IoT/CPS argument, false alarms are operationally central.

The recent evaluation literature also supports an event/proximity-aware metric such as PATE or another carefully chosen range/event metric. I would not require the authors to abandon raw point-wise metrics; rather, the paper should show that its ranking is not an artifact of *any one* metric family.

### Sample-size reporting

For every dataset and subset, report:

- total normal windows used for AUROC;
- anomalous windows;
- number of independent attack episodes contributing those windows;
- episode-duration distribution or at least median/range;
- seed count per stochastic method.

Without episode counts, “n=19” and “n=38” are difficult to interpret and may represent only a handful of attacks.

---

## 6. Presentation, 2024–2026 positioning, and reproducibility

### Presentation and claim discipline

The manuscript is generally readable, but several statements should be corrected before publication:

1. **Abstract:** “Anomaly scores are computed directly in latent space rather than from a density estimate or reconstruction error” conflicts with Section 4.3(i), which is explicitly a latent mixture-density NLL, and with Section 4.3(iv), which uses reconstruction residuals. Rewrite this sentence.
2. **Table 2 caption:** “reconstruction residual is at or below chance” is inconsistent with HAI=0.689. Correct the caption and the surrounding text.
3. **Sections 4.2, 7, and 8:** “drop reconstruction entirely” and “latent-only” are inaccurate for the headline model because the residual head is active on HAI and SWaT. Use precise terminology.
4. **Section 6:** calling difficult WADI/HAI “precisely” joint-structure anomalies is too strong until the split is independently validated.
5. **Conclusion:** “jointly learning the latent and its modes, then scoring in that latent, is the fix” overgeneralizes from one clear dataset win plus two near-ties/ceilings.

A related technical issue is the claim in Section 4.3(iii) that the nearest-component score is “rare-mode-safe.” Dropping mixture weights \(\pi_c\) does remove a direct penalty for low-prior components. However, the base score simultaneously includes the high-K density head in Section 4.3(i), whose mixture weights \(w_m\) can again penalize low-mass regions. The paper should empirically verify the A6 claim by stratifying **normal false-positive rate versus mode frequency**. Otherwise “rare-mode-safe” is an intention, not demonstrated behavior.

The term “parametric KDE” for the high-K Gaussian mixture in Section 4.3(i) is also potentially confusing. A fitted finite GMM is a parametric/semi-parametric density estimator, but it is not conventional kernel density estimation in which kernels are typically centered on samples with bandwidth control. A clearer term would be “high-component Gaussian-mixture density estimator.”

### Related-work positioning

Section 2 is strong on evaluation critique but comparatively weak on current methods. It moves from GDN/USAD/TranAD to the proposed method without engaging the major 2024–2025 evolution in multivariate TSAD. In particular, recent channel-aware, frequency-domain, uncertainty-aware, and modern benchmark work should be discussed. The literature does not invalidate LatAD, but it makes the novelty claim narrower: **the novelty is not simply “use latent density instead of reconstruction,” but the specific combination of joint VaDE representation, multimodal scoring, train-normal auto-gating, and difficulty-conditioned evaluation on CPS data.** That is a more defensible position.

The authors should also be aware of a naming collision: a 2024 time-series anomaly-detection work already uses the acronym/name **LATAD** (“Self-Supervised Time-Series Anomaly Detection Using Learnable Data Augmentation”). If the present manuscript intends to brand the method as LatAD, a distinct name would reduce confusion in search and citation databases.

### Reproducibility

The Data Availability Statement is positive: it states that code, trained checkpoints, and supporting results are archived on Zenodo. However, the paper itself omits many implementation details needed to understand or independently reimplement the method. Across Sections 4–5 I could not find the encoder/decoder layer widths, activation functions, batch size, optimizer, epoch counts, learning rates, exact latent dimension by dataset, VaDE prior component count, high-K selection rule beyond “K=80 by default,” warm-up schedule, variance-floor value, collapse-penalty coefficient, residual dimension-reduction method, gate threshold for the q95 ratio, basin parameters \(R\), \(\lambda_0\), and \(\delta\), or complete seed specification.

For MDPI *IoT*, relying on code alone is not sufficient. Add a **reproducibility table** with all dataset-specific and global hyperparameters, preprocessing details, architecture details, and calibration rules. The exact WADI/SWaT downsampling procedure and HAI score-grid correction should be specified algorithmically, because score alignment can materially affect point-wise labels and AUROC.

The authors should also release or archive **seed-level raw anomaly scores and labels** used for Table 3. This would allow reviewers/readers to reproduce AUROC, confidence intervals, threshold curves, and future metrics without retraining every model.

Finally, the basin-agreement head is inactive on all three benchmarks (Section 4.3(v)). It therefore has no empirical support in the present paper. Either remove it from the claimed contribution, move it to an appendix/future-work note, or add a benchmark/synthetic experiment where it activates and demonstrate that it improves performance without harming normal rare modes.

---

## Ranked blocking concerns

1. **Mechanism claim is overgeneralized and internally inconsistent.** The paper claims reconstruction is dropped/latent-only, yet the headline model uses a reconstruction-residual head on HAI and SWaT; HAI reconstruction is also substantially above chance. The reachability-vs-probability narrative needs explicit ablations and falsifiable controls.
2. **Difficulty split lacks construct validity for “joint-structure faults.”** It is leak-free but defined by failure of one max-channel-mean statistic. It may select low-amplitude univariate or other simple anomalies and condition the benchmark around one baseline. Robustness to split definition and an independent anomaly taxonomy are required.
3. **Statistical evidence is insufficient for the strength of the “best/lead” claims.** WADI-difficult is effectively a near-tie with Isolation Forest and SWaT-difficult is effectively tied with LinRes; difficult-set positive counts are small and overlapping. No confidence intervals, paired tests, episode bootstrap, or seed variation for USAD/TranAD are provided.
4. **Baseline set is dated for a 2026 paper and does not adequately stress-test the mechanism.** GDN is cited but omitted; recent methods such as SensitiveHUE and CATCH are directly relevant, and recent benchmarking work should temper universal claims.
5. **Reproducibility and ablation are incomplete.** Core architectural/training/gating hyperparameters are absent from the prose, Table 2 is single-model rather than multi-seed, the full scoring stack is not ablated, and the basin head is claimed without ever activating on the evaluated datasets.

---

## Concrete required revisions

I would require the following before reconsideration:

1. Rewrite the Abstract, Sections 4.2/7/8, and Table 2 caption so that “latent-only,” “drop reconstruction entirely,” and “at or below chance” match the actual method and numbers.
2. Add a five-seed ablation table across WADI, HAI, and SWaT isolating every score head and the auto-gating contribution.
3. Fully specify and expand the between-mode probe: sample counts, mode definition, space of interpolation, seeds, confidence intervals, within-mode controls, interpolation sweep, and at least one marginal-preserving joint-structure corruption.
4. Validate the difficulty construct with alternative percentiles and a stronger univariate detector using all six per-channel statistics; report an independent per-attack anomaly taxonomy if feasible.
5. Report the number of independent attack episodes in Easy/Difficult/All and use episode/block bootstrap confidence intervals for primary metrics and pairwise AUROC/AUPRC differences.
6. Run every stochastic learned baseline with the same multi-seed protocol, including USAD/TranAD, or provide a compelling reason why not.
7. Add AUPRC and at least one fixed low-FPR operational metric; add a non-oracle thresholded result. Keep best-F1 only as explicitly oracle/secondary.
8. Add at least one modern cross-channel/dependency baseline and one recent strong reconstruction-based method; at minimum reconsider GDN, SensitiveHUE, and CATCH in the positioning and, where feasible, experiments.
9. Reframe the cross-dataset claim: HAI is the clear win; WADI is a near-tie with Isolation Forest unless significance is demonstrated; SWaT is a ceiling/triviality case.
10. Add a complete hyperparameter/reproducibility table and archive seed-level anomaly scores/labels. Clarify preprocessing/downsampling/alignment in enough detail to reproduce Table 3 independently.
11. Either empirically validate the basin-agreement head on a dataset where it activates or remove it from the main claimed contributions.
12. Consider renaming “LatAD” to avoid collision with the existing 2024 LATAD time-series anomaly-detection method.

---

## Minor and editorial comments

- Section 3’s MIIM assumptions are a useful organizing device, but A2 (“mode explosion”) and several other assumptions are stated more strongly than directly evidenced by three benchmarks. Distinguish engineering hypotheses from empirically verified properties.
- The claim that BIC “keeps improving past several dozen components” and silhouettes are 0.08–0.19 is potentially informative, but the corresponding analysis should be shown in a figure/table, with the feature space and clustering method specified.
- A9–A10 are not exercised by the reported window-only model. They should be presented as scope/future design principles, not validated assumptions of the current detector.
- WADI clipping at ±10σ and the treatment of near-constant channels may materially affect both the trivial split and learned scores. A sensitivity check without clipping, or with an alternative robust scaler, would strengthen the result.
- The 5% anomalous-timestep rule for labeling a window should be justified and sensitivity-tested, especially because W=60 and stride=30 can change the number of difficult windows substantially.
- Section 5.5 says USAD/TranAD per-timestep scores are aggregated onto the proposed window grid, including an “integer-ratio correction” for HAI. This needs an explicit formula/algorithm; otherwise score alignment is difficult to audit.
- The paper mentions FPR in Section 2.1 as part of its adopted evaluation philosophy but does not report FPR in Table 3. Either report it at a defined threshold or remove that implication.
- Figure 1 should include uncertainty for learned methods. A bar plot without confidence intervals visually exaggerates differences such as WADI 0.690 versus 0.677.
- The use of one model/checkpoint per dataset is appropriate for dataset-specific unsupervised learning, but the phrasing “single trained model per dataset” is potentially confusing alongside five-seed results. Clarify whether five independent models are trained for evaluation while one fixed-seed checkpoint is archived as the reproducibility artifact.

---

## Final recommendation

**Major Revision.** The manuscript contains a worthwhile HAI result and a useful evaluation perspective, but the current version does not yet establish its headline mechanism or the generality implied by the Abstract and Conclusion. The strongest path to publication is not to add more rhetoric around the current numbers; it is to narrow the claim, validate the difficulty construct independently, perform full score ablations and falsification-oriented mechanism probes, quantify episode-level uncertainty, and update the baseline/evaluation context to the 2024–2026 literature. If those revisions preserve the HAI advantage and show that the WADI mechanism survives independent controls, the paper would become substantially more convincing for *IoT*.
