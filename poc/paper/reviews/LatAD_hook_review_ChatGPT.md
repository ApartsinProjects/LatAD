# Focused hook/readability review for MDPI *IoT*

## Bottom line

**The hook now works.** A busy IoT/CPS engineer who reads the title, first two abstract sentences, and first Introduction paragraph will understand that the paper targets a specific failure mode of reconstruction-based anomaly detection: a CPS state can be easy for a deep model to reconstruct yet still be improbable under normal joint operation. The revised first Introduction paragraph also now connects that principle to subsystem factorization, so the paper no longer feels like “latent density first, community method later.”

The remaining risk is **not conceptual confusion; it is cognitive load**. Immediately after an unusually clear reachability-versus-probability hook, the prose accelerates into MIIM, VaDE, Gaussian mixtures, correlation communities, tail probabilities, cohesion weighting, Higher Criticism, auto-gated residuals, and several evaluation strata. A non-ML IoT reviewer can follow the paper, but must work harder than necessary. The best final revision is therefore not another conceptual rewrite. It is a **front-loaded simplification and skimmability pass**: make the title mechanism-forward, simplify the abstract’s method sentence, gloss each statistical term at first use, convert Results/Discussion topic sentences into real subheadings, and shorten the Table 3 caption.

---

# 1. THE HOOK

## What a reader thinks the paper is about after 30 seconds

**After 30 seconds, the reader thinks: this paper argues that hard CPS anomalies can remain sensor-wise normal and reconstruct well, so anomaly detection should score how probable a state is under normal operating regimes, preferably at subsystem scale, rather than rely on reconstruction error.**

That is the right takeaway.

## Title audit

**Current title:**

> “Modeling Normal Is All You Need: Joint Latent Clustering for Anomaly Detection in Multimodal Cyber-Physical Systems”

### What works
- “Modeling Normal” points in the correct direction.
- “Cyber-Physical Systems” makes the application domain visible.

### What weakens the hook
The title does **not expose the paper’s most memorable mechanism**. “Joint Latent Clustering” is an ML implementation phrase; it does not tell an IoT engineer why the method exists. “All You Need” is catchy but also slightly at odds with the paper’s optional reconstruction residual, community factorization, calibration, and future temporal extensions.

For a 60-second editorial skim, the title is currently the weakest element of the front door.

### Stronger title

> **Reconstructable Yet Improbable: Regime-Community Latent Density for Anomaly Detection in Industrial IoT Cyber-Physical Systems**

This immediately gives:
1. the paradox;
2. the method family;
3. the IoT/CPS venue fit.

A slightly shorter alternative is:

> **Reconstructable Yet Improbable: Subsystem-Factorized Latent Density for CPS Anomaly Detection**

---

## Abstract first two sentences

**Current, Abstract sentences 1–2:**

> “A cyber-physical system (CPS) can enter a faulty state that looks individually normal on every sensor, and that a deep model reconstructs accurately, yet is highly improbable under normal joint operation.”

> “Such faults expose the central weakness of reconstruction-based anomaly detection: reconstruction measures whether a state can be reproduced (reachability), not whether normal operation is likely to occupy it (probability).”

### Verdict

**Excellent hook. The reachability-versus-probability idea is unmistakable.**

The first sentence is slightly long, but the logical contradiction is immediately graspable. The second sentence is the paper’s strongest explanatory sentence and should remain essentially intact.

The only improvement would be to shorten sentence 1 so the paradox lands faster.

### Sharper first 2–3 sentences

> **A cyber-physical system can be faulty even when every sensor stays in its normal range and a deep model reconstructs the state accurately. Reconstruction therefore answers the wrong question: can the state be reproduced? Anomaly detection needs a different one: would normal operation actually occupy it?**

If you want the method in the third sentence:

> **LatAD answers that question by scoring probability over normal operating regimes and preserving subsystem-local evidence that a whole-plant score can dilute.**

This is probably the strongest possible front-door formulation for an IoT reader.

---

## First Introduction paragraph

**Location: §1, paragraph 1, beginning “A cyber-physical system can be faulty…”**

The revision is significantly stronger because it now closes with:

> “This paper detects them by modelling that joint structure, scoring probability rather than reconstruction, and, because faults are as local as the physics that produces them, factorizing that probability over the plant’s own subsystems.”

That sentence now integrates the community/subsystem story with the reachability/probability story.

### What still blunts it

Two phrases are heavier than needed:

> “a correlation break, or a state that falls between two legitimate regimes”

and

> “because faults are as local as the physics that produces them”

The first is fine for a technical reader but slows the sentence. The second is memorable but sounds stronger than the evidence warrants, because not every fault is subsystem-local.

### Recommended first three Introduction sentences

> **A cyber-physical system can be faulty even when every sensor stays within its familiar range and a deep model reconstructs the full state accurately. Reconstruction answers whether a state can be reproduced; anomaly detection needs to know whether normal operation is likely to occupy it. LatAD therefore models the probability of normal operating regimes and evaluates that probability within tightly coupled channel communities, so a local joint violation is not diluted by unaffected parts of the plant.**

Then continue with one concrete example of a correlation break or between-regime state.

### Hook verdict

**Yes: the hook now lands.** The remaining issue is that the paper becomes much more jargon-dense immediately after the hook than the hook itself promises.

---

# 2. READABILITY FOR A NON-ML IoT READER: 10 EXACT STALL POINTS

## 1. Abstract, sentence 3: MIIM arrives too early

**Quoted text:**

> “We model CPS normal behaviour instead as a union of many imbalanced, sparsely populated operating regimes, a structure we formalize as ten assumptions summarized as Massive, Implicit, Imbalanced Multimodality (MIIM).”

**Why a non-ML reader stalls:** The sentence introduces a conceptual model, a ten-assumption framework, and an acronym before the reader has seen the detector.

**One-line fix:**

> “We model normal CPS telemetry as a collection of common and rare operating regimes; §3 states the corresponding structural assumptions explicitly as MIIM.”

Keep the acronym, but do not make the abstract carry all four expanded words.

---

## 2. Abstract, sentence 4: VaDE/Gaussian-mixture jargon stack

**Quoted text:**

> “Our detector, LatAD, jointly learns a latent representation and a Gaussian-mixture clustering of these regimes (VaDE) and scores anomalies by density in the clustered latent rather than by reconstruction.”

**Why the reader stalls:** “latent representation,” “Gaussian-mixture clustering,” “VaDE,” and “density in the clustered latent” all arrive in one sentence.

**One-line fix:**

> “LatAD learns a compact representation in which normal operating regimes form explicit clusters, then scores how probable each new state is within those clusters rather than how well it reconstructs.”

Put “implemented with VaDE” at the end or in Methods.

---

## 3. Abstract, sentence 5: the densest sentence in the abstract

**Quoted text:**

> “Because a CPS is an assembly of physically coupled subsystems, we factorize that density over correlation-community subsystems recovered from the train-normal correlation graph and combine the per-community surprises by a cohesion-weighted, sparsity-adaptive statistic, concentrating a local fault that a single global density dilutes.”

**Why the reader stalls:** This asks the reader to parse factorization, correlation communities, train-normal graph construction, “surprises,” cohesion weighting, and sparse adaptation simultaneously.

**One-line fix:**

> “Because many CPS faults are local, LatAD also learns data-driven communities of strongly correlated channels and scores them separately, preserving a local anomaly that a whole-plant density can dilute.”

Name Higher Criticism only later.

---

## 4. §1, paragraph 2, first sentence

**Quoted text:**

> “A modern CPS, a water-treatment plant, an industrial control loop, or a rotating machine, couples hundreds of networked sensors, embedded controllers, and physical actuators into a unit that continuously senses its own state and acts on it; it is the sensing-and-actuation fabric of the industrial Internet of Things (IoT).”

**Why the reader stalls:** The appositive list makes the grammar awkward, and the sentence is doing both CPS definition and IoT positioning.

**One-line fix:**

> “Industrial IoT systems couple sensors, controllers, and actuators into continuously monitored CPSs such as treatment plants, industrial control loops, and rotating machinery.”

---

## 5. §3 opening paragraph, immediately before/after the mixture equation

**Quoted text:**

> “where each component \(p_k\) is a bounded, oriented, curved patch of observation space.”

**Why the reader stalls:** The equation appears before a plain-English picture, and “bounded, oriented, curved patch of observation space” is mathematically evocative but not operationally concrete.

**One-line fix:**

> “In engineering terms, each component represents one normal operating regime, such as a load, setpoint, or control configuration, with its own allowable multivariate range.”

Then show the equation.

---

## 6. §4 opening pipeline

**Quoted text:**

> “windowed telemetry → correlation-community subsystems → one regime-density model per community plus a global expert → calibrated per-community tail probabilities → cohesion-weighted Higher Criticism → anomaly score.”

**Why the reader stalls:** The pipeline is useful, but the last three nodes are statistical terms that are not yet explained.

**One-line fix:**

> “windowed telemetry → groups of strongly coupled channels → one regime-probability model per group plus a whole-plant model → normalized local anomaly scores → sparsity-adaptive aggregation → final anomaly score.”

Then give the formal terms in parentheses.

---

## 7. §4.1, VaDE objective and the paragraph beginning “where γc … are the responsibilities”

**Quoted text:**

> “Three standard measures prevent the well-known cluster collapse… a KL/cluster warm-up… a DAGMM-style penalty on tiny component variances… and a slower learning rate on the mixture parameters…”

**Why the reader stalls:** The paper moves from a large ELBO equation straight into optimizer-level anti-collapse details before restating the engineering meaning.

**One-line fix:** Add immediately before the training details:

> “For the IoT reader, the key point is simply that training organizes normal telemetry into explicit operating-regime clusters while learning the compact representation.”

The equations can remain for reproducibility.

---

## 8. §4.3 heading “Latent mixture-density head (parametric KDE for non-Gaussian pockets)”

**Quoted text:**

> “Latent mixture-density head (parametric KDE for non-Gaussian pockets)”

**Why the reader stalls:** “mixture-density,” “parametric KDE,” and “non-Gaussian pockets” are three levels of statistical terminology in one heading.

**One-line fix:** Rename:

> **Fine-grained latent density: modeling non-uniform operating regimes**

Put “parametric KDE” in the body.

---

## 9. §4.3(iv), first paragraph

**Quoted text:**

> “The optional residual head reduces the per-channel residual r to 30 dimensions, fits a Ledoit–Wolf-shrunk precision per regime, and combines the per-regime whitened energies by responsibility…”

**Why the reader stalls:** This is implementation-heavy and assumes familiarity with precision matrices, shrinkage covariance, whitening, and responsibilities.

**One-line fix:** Lead with purpose:

> “When reconstruction error is stable on held-out normal data, LatAD retains it as a secondary, covariance-normalized signal; the formula below gives the implementation.”

---

## 10. §4.4, final paragraph: Higher Criticism arrives after several other operations

**Quoted text:**

> “Higher Criticism, a sparse-signal statistic that asks whether more communities are mildly surprised than chance allows, interpolates between the two without assuming the extent in advance.”

**Why the reader stalls:** The new gloss is good, but it comes only after “cohesion,” “tail probabilities,” summation, maximum, and the \(\oplus\) combiner.

**One-line fix:** Put the intuition first:

> “Because the number of affected subsystems is unknown, the combiner must detect either one strong local alarm or several weaker ones. We use Higher Criticism for that sparse-to-distributed aggregation.”

This is a good example of where **intuition-before-formula** would materially improve readability.

---

# 3. IoT-JOURNAL FIT

## Verdict

**The manuscript is substantively well aligned with an IoT/CPS readership, but the explicit IoT framing is thinner than the CPS framing.**

The paper already has:
- “Internet of Things” and “industrial IoT” in the keywords;
- an explicit sentence in §1 paragraph 2 calling CPS the “sensing-and-actuation fabric of the industrial Internet of Things”;
- repeated emphasis on networked sensor/actuator telemetry;
- WADI, HAI, and SWaT as real CPS/industrial-control testbeds;
- operational language around condition monitoring and subsystem localization.

So the venue fit is present. The issue is **visibility during a skim**. “IoT” does not appear in the title or abstract body, and the strongest method/result framing is written primarily as a CPS anomaly-detection paper.

## Best places to sharpen IoT relevance without overclaiming

### A. Title

Use “Industrial IoT Cyber-Physical Systems” or “Industrial IoT Monitoring” once.

Recommended:

> **Reconstructable Yet Improbable: Regime-Community Latent Density for Anomaly Detection in Industrial IoT Cyber-Physical Systems**

This is the highest-value IoT-facing edit because editors see the title before anything else.

---

### B. Abstract, first or third sentence

Do not insert a generic “IoT is important” sentence. Instead connect the actual data modality:

> “Industrial IoT monitoring produces multivariate CPS telemetry in which a faulty joint state can remain individually plausible on every sensor.”

or, less intrusive:

> “We model normal multivariate IoT telemetry as a collection of operating regimes…”

That is venue-facing and scientifically relevant.

---

### C. §1 paragraph 2

The current IoT sentence is conceptually right but syntactically heavy.

Replace the first sentence with:

> “Industrial IoT deployments expose CPS operation as continuous multivariate telemetry from sensors, controllers, and actuators.”

Then explain why normal-only modeling is needed.

This makes IoT telemetry the data object, not merely a label.

---

### D. §2 opening or §2.3

Add one sentence connecting the methodological problem to IoT monitoring:

> “For industrial IoT monitoring, the practical object is therefore not an isolated sensor trace but the joint telemetry state created by coupled sensing, actuation, and control.”

This directly motivates cross-channel density and avoids generic venue padding.

---

### E. Conclusion / impact paragraph

Current:

> “Two properties make the detector deployable for condition monitoring and predictive maintenance…”

This overstates deployment readiness.

Replace with:

> “Two properties are attractive for industrial IoT condition monitoring: rare valid operating modes are protected from unnecessary alarms, and community-level scores preserve subsystem-local violations that remain inside individual sensor limits.”

Then, if desired:

> “This positions LatAD as a data-driven anomaly layer that could complement fixed alarms or physics-based monitoring in an IoT/CPS stack.”

That is strong without claiming predictive-maintenance validation.

---

# 4. SKIMMABILITY

## Can the contribution and result be reconstructed from abstract + headings + Table 3 + Figure 1 + conclusion?

**Mostly, but not cleanly enough yet.**

### What the skimmer gets

From the abstract:
- reconstructable-but-improbable fault;
- density instead of reconstruction;
- community factorization;
- difficult-subset evaluation;
- best headline results.

From Table 3:
- LatAD regime-community is the main method;
- global density is the base ablation;
- the method leads overall and on difficult subsets.

From Figure 1:
- difficult-subset advantage is the main empirical story;
- HAI is strongest;
- WADI is also favorable;
- SWaT is a ceiling case;
- reconstruction methods are weak on the hard WADI/HAI cases.

From the conclusion:
- reachability vs probability;
- regime/community factorization;
- operational motivation.

That is enough to reconstruct the paper.

### What is missing in the headings

Current visible headings include:

> “3 The MIIM assumptions about CPS normal data”

> “4 Method”

> “4.1 Representation: Variational Deep Embedding (VaDE)”

> “4.2 Scoring: why we demote reconstruction”

> “4.3 The scoring stack”

> “4.4 Regime-community density (the headline realization)”

> “6 Results”

> “7 Discussion”

Only §4.2 and §4.4 reveal the story. “Method,” “Results,” and “Discussion” are generic, and the Results/Discussion use bold topic sentences rather than actual subheadings.

## Recommended heading changes

### Section 3
Current:

> “The MIIM assumptions about CPS normal data”

Better:

> **CPS normality as imbalanced operating regimes**

Introduce “MIIM” in the first paragraph rather than making the acronym the heading.

### Section 4
Current:

> “Method”

Better:

> **LatAD: regime probability with subsystem-local aggregation**

### §4.1
Current:

> “Representation: Variational Deep Embedding (VaDE)”

Better:

> **Learning normal operating regimes in a compact latent space**

VaDE can remain in parentheses.

### §4.2
Current:

> “Scoring: why we demote reconstruction”

Better:

> **Why reconstruction fails: reachability is not probability**

This should be one of the most visible headings in the paper.

### §4.3
Current:

> “The scoring stack”

Better:

> **Probability scores and train-normal-gated auxiliary evidence**

### §4.4
Current:

> “Regime-community density (the headline realization)”

Better:

> **Preserving local faults with subsystem-factorized density**

“Headline realization” sounds like manuscript-development language, not a final-paper heading.

### Section 6
Convert the current bold topic sentences into true subsections:

- **6.1 Overall detection performance**
- **6.2 Difficult in-envelope faults: the discriminative test**
- **6.3 Robustness to stronger difficulty definitions**
- **6.4 Double-hard stress test**
- **6.5 Reconstruction-based deep detectors under raw metrics**

### Section 7
Likewise:

- **7.1 Why reconstruction fails: reachability versus probability**
- **7.2 Why community factorization helps: local faults versus global dilution**
- **7.3 When reconstruction still helps: train-normal gating**
- **7.4 Limits and next validation**

This alone would substantially improve the editor’s 60-second skim.

---

## Table 3 caption

**Current location: Table 3 caption in §6.**

The caption is self-contained but too long. It explains:
- AUROC/F1;
- leak-free versus oracle;
- five seeds;
- GDN compute limitation;
- bolding;
- “Ours”;
- global-density ablation;
- trivial baseline;
- difficulty split;
- LinRes.

A skimmer has to read a paragraph before reaching the table.

### Better caption

> **Table 3. Raw anomaly-detection performance on WADI, HAI, and SWaT. Values are [AUROC, best raw F1] for All, Easy, and Difficult subsets; learned detectors report five-seed AUROC mean±SD. LatAD (regime-community) is the proposed model; LatAD (global density) is its one-community ablation. The Difficult subset contains anomalies not separated by the train-normal univariate range rule. Bold marks the best AUROC in each column.**

Move GDN compute detail and LinRes definition to a note below the table or Methods.

---

## Figure 1 caption

**Current Figure 1 caption is strong and nearly self-contained.**

Its one weakness is that it mixes result and causal interpretation:

> “whose difficult faults are reconstructable-but-improbable combinations”

For a skimmer, that is useful, but it reads as if the figure itself establishes the mechanism.

A slightly cleaner caption is:

> **Figure 1. Difficult-subset AUROC under raw metrics. LatAD leads all three datasets, significantly on HAI (0.849; +0.09 over AutoEncoder, 95% CI [0.046, 0.160]) and numerically on WADI (0.796 vs 0.677 for Isolation Forest). USAD and TranAD fall to 0.30–0.50 on WADI and HAI; SWaT is a ceiling case on which most methods remain strong. Whiskers show ±1 SD for multi-seed detectors.**

Put the reconstructable-but-improbable interpretation in the Results paragraph immediately after.

---

# 5. TEN HIGHEST-IMPACT EDITS, RANKED

## 1. Replace the title with the mechanism, not the implementation

**Location:** Title.

**Problem:**

> “Modeling Normal Is All You Need: Joint Latent Clustering…”

**Why high impact:** The current title hides the most memorable idea from the editor’s first glance.

**Concrete rewrite:**

> **Reconstructable Yet Improbable: Regime-Community Latent Density for Anomaly Detection in Industrial IoT Cyber-Physical Systems**

---

## 2. Simplify the abstract method sentence before naming VaDE

**Location:** Abstract, sentence 4.

**Problem:**

> “jointly learns a latent representation and a Gaussian-mixture clustering of these regimes (VaDE)…”

**Concrete rewrite:**

> “LatAD learns a compact representation in which normal operating regimes form explicit clusters, then scores each new state by its probability within those clusters rather than by reconstruction error; the implementation uses VaDE.”

---

## 3. Rewrite the abstract community sentence in engineering language

**Location:** Abstract, sentence 5.

**Problem:**

> “factorize that density over correlation-community subsystems recovered from the train-normal correlation graph and combine the per-community surprises by a cohesion-weighted, sparsity-adaptive statistic…”

**Concrete rewrite:**

> “Because many CPS faults are local, LatAD learns groups of strongly correlated channels as subsystem proxies, scores each group separately, and combines those local scores so unaffected parts of the plant do not dilute a fault.”

If desired, add “using cohesion-weighted Higher Criticism” in the next sentence or Methods.

---

## 4. Make the Introduction’s first three sentences shorter and irreversible

**Location:** §1 paragraph 1.

**Problem:** The current paragraph is strong but still contains long clauses before the method appears.

**Concrete rewrite:**

> “A cyber-physical system can be faulty even when every sensor stays within its familiar range and a deep model reconstructs the state accurately. Reconstruction answers whether a state can be reproduced; anomaly detection needs to know whether normal operation is likely to occupy it. LatAD therefore scores probability over normal operating regimes and preserves that evidence at subsystem scale, where local faults are not diluted by unaffected channels.”

---

## 5. Rename §4 and §4.2 so the mechanism survives a heading-only skim

**Location:** §4 and §4.2 headings.

**Problem:**

> “4 Method”

> “4.2 Scoring: why we demote reconstruction”

**Concrete rewrite:**

> **4 LatAD: regime probability with subsystem-local aggregation**

> **4.2 Why reconstruction fails: reachability is not probability**

This is a major skimmability gain for almost zero cost.

---

## 6. Turn Results and Discussion bold lead-ins into actual subheadings

**Location:** §§6–7.

**Problem:** The manuscript contains excellent topic phrases such as:

> “Difficult, joint-structure faults.”

> “Why reconstruction fails and density wins.”

but they are embedded inside paragraphs rather than exposed in the document hierarchy.

**Concrete fix:** Promote them to numbered h3 headings, especially:
- difficult in-envelope faults;
- stronger difficulty split;
- double-hard stress test;
- reachability vs probability;
- community factorization;
- auto-gating/limits.

---

## 7. Remove statistical jargon from the §4 pipeline preview

**Location:** §4 opening paragraph.

**Problem:**

> “calibrated per-community tail probabilities → cohesion-weighted Higher Criticism”

**Concrete rewrite:**

> “normalized local anomaly scores → sparsity-adaptive aggregation”

Then add:

> “The reported aggregator is cohesion-weighted Higher Criticism (§4.4).”

The pipeline should orient before it formalizes.

---

## 8. Add one IoT-facing sentence to the abstract and one to the impact paragraph

**Location:** Abstract and §8 final paragraph.

**Problem:** Explicit IoT language is largely confined to keywords and §1 paragraph 2.

**Concrete abstract insertion:**

> “We model normal multivariate industrial-IoT telemetry as a collection of operating regimes rather than a single global distribution.”

**Concrete conclusion rewrite:**

> “For industrial IoT condition monitoring, the resulting community scores preserve subsystem-local violations while nearest-regime scoring reduces alarms on rare but valid operating modes.”

---

## 9. Shorten Table 3 caption by about half

**Location:** Table 3 caption.

**Problem:** The caption is technically self-contained but too operationally detailed for a skim.

**Concrete rewrite:** Use the short caption proposed in §4 above and move GDN/compute and LinRes details into a footnote or Methods.

---

## 10. Soften three overclaims that can distract a skeptical reviewer

### A. §4.4

Current:

> “Agglomerative clustering … recovers these subsystems as nested communities…”

Rewrite:

> “Agglomerative clustering … produces nested channel communities that serve as data-derived proxies for tightly coupled subsystems…”

### B. §8

Current:

> “Two properties make the detector deployable for condition monitoring and predictive maintenance…”

Rewrite:

> “Two properties are operationally attractive for industrial IoT condition monitoring…”

### C. §8

Current:

> “the CPS faults that matter are reconstructable but improbable.”

Rewrite:

> “the difficult in-envelope faults targeted here can be reconstructable yet improbable.”

These changes increase credibility without weakening the story.

---

# 6. REWRITTEN TITLE + ABSTRACT FOR A FAST IoT READER

## Recommended title

**Reconstructable Yet Improbable: Regime-Community Latent Density for Anomaly Detection in Industrial IoT Cyber-Physical Systems**

## Rewritten abstract

A cyber-physical system can be faulty even when every sensor stays within its familiar range and a deep model reconstructs the state accurately. Reconstruction therefore answers the wrong question: can the state be reproduced? Anomaly detection needs to know whether normal operation is likely to occupy it. We model normal multivariate industrial-IoT telemetry as a collection of common and rare operating regimes and build LatAD to score probability within that regime structure rather than reconstruction error. LatAD learns a compact regime-structured representation with VaDE and, because many faults are local, applies the density model to data-derived communities of strongly correlated channels that serve as subsystem proxies. Their calibrated anomaly scores are combined with a sparsity-adaptive, cohesion-weighted statistic so that a local violation is not diluted by unaffected parts of the plant; a whole-plant expert retains sensitivity to distributed faults. We evaluate LatAD on WADI, HAI, and SWaT using raw point-wise metrics and a difficulty split that isolates anomalies not separated by a simple per-channel rule. LatAD achieves the highest overall AUROC on all three benchmarks and leads their difficult subsets, with a statistically significant margin on HAI and a numerical lead on WADI; SWaT is a ceiling case. Reconstruction-based USAD and TranAD fall near chance on the difficult WADI and HAI faults. The results support a simple monitoring principle: **for in-envelope CPS anomalies, score probability rather than reconstructability, and preserve that evidence at subsystem scale.**

---

# Final focused verdict

**Will a busy IoT/CPS reader get the hook? Yes.** The revised manuscript now makes the central idea visible within the first two abstract sentences and the first Introduction paragraph: **a state can be reconstructable yet improbable, so anomaly detection should score probability rather than reconstruction.** The subsystem-factorization sentence at the end of the first Introduction paragraph now connects the headline regime-community method to that same logic.

**Will a non-ML reviewer find the paper readable enough to accept comfortably? Almost, but one final simplification pass would materially improve the odds.** The manuscript’s conceptual spine is clearer than its terminology. The front door is simple; the method section is not. The highest-return changes are therefore title/abstract simplification, intuition-before-formula wording in §4, true Results/Discussion subheadings, a shorter Table 3 caption, and slightly stronger explicit industrial-IoT framing.

With those edits, the paper can be skimmed as one coherent story:

**industrial IoT telemetry contains hard in-envelope faults → reconstruction tests reachability, not probability → LatAD models probability over normal regimes → community factorization preserves local fault evidence → raw difficult-subset evaluation shows the gain where reconstruction-based methods lose signal.**
