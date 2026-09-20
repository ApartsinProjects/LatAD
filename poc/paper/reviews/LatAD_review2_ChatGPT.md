# LatAD manuscript narrative and scientific-writing review

**Scope of this review.** This review is restricted to narrative arc, storytelling, impact framing, structure, style, tone, captions/headings, and readability for a technically strong CPS/controls reader who is not an ML specialist. I do not re-check numerical correctness or re-adjudicate the science. The manuscript was read top to bottom, including the abstract, every main-text paragraph, section and subsection headings, table and figure captions, the limitations, conclusion, and Appendix A.

## 1. Overall verdict and the single biggest narrative weakness

### Verdict

The paper has a strong core idea and, at sentence level, several unusually memorable formulations. The best of them is already present: **“Reconstruction measures reachability; detection needs probability.”** That is the paper’s intellectual hook, the easiest part for a CPS engineer to remember, and the right organizing principle for the manuscript.

The current manuscript is nevertheless **one structural revision short of telling the strongest possible story**. It often reads as if the paper first presents one method, LatAD as a global regime-density detector, and then in §4.4 introduces a second, stronger method, LatAD as a regime-community detector with correlation communities and Higher Criticism. Because §4.4 is explicitly called “the headline realization,” this creates a retrospective redefinition of what the reader thought LatAD was.

### Single biggest narrative weakness

**The headline method arrives too late and therefore feels bolted onto the reachability-versus-probability paper rather than inevitable from it.**

The manuscript spends the Introduction, §3, and §§4.1–4.3 teaching a clean story:

> “Reconstruction measures reachability; detection needs probability.”

and

> “The base scoring stack works entirely in the jointly learned latent.”

Then §4.4 changes the center of gravity:

> “Its headline realization estimates density *per subsystem* instead…”

followed by correlation communities, cohesion weights, a global null expert, and Higher Criticism. These are not minor implementation details. They are the distinguishing architecture of the headline detector.

The result is a two-stage reader experience:

1. **Paper A:** CPS normality is multimodal; therefore learn regime-structured latent density and demote reconstruction.
2. **Paper B:** CPS faults are subsystem-local; therefore factorize density over communities and combine sparse local surprises.

These two ideas are compatible and can become one excellent paper, but the manuscript currently connects them too late.

### Highest-leverage structural correction

Make the method’s logic explicitly **two-level from the first page**:

1. **Probability principle:** reconstructability does not imply normality, so model the probability of normal operating regimes.
2. **Locality principle:** CPS faults often disturb only one tightly coupled group, so estimate that probability locally and aggregate sparse subsystem evidence without dilution.

Then make **LatAD (regime-community)** the method from the first Method paragraph onward. Rename **LatAD (global density)** consistently as the **global ablation/base configuration**, not as an equally primary version of the detector.

The architecture should be previewed before VaDE. A CPS engineer should know the whole pipeline before encountering the first ELBO equation:

**windowed telemetry → correlation communities → one regime-density model per community + global expert → calibrated community tail probabilities → cohesion-weighted Higher Criticism → anomaly score**

That one move would resolve most of the current narrative fragmentation.

---

## 2. One-sentence STORY SPINE, where it is lost, and whether the paper reads as one or two papers

### Recommended one-sentence story spine

**Hard CPS faults are often locally improbable joint states that remain individually in-range and reconstructable, so LatAD learns the probability of normal operating regimes inside correlation-defined subsystem communities and aggregates sparse community-level surprises instead of relying on global reconstruction error.**

This sentence contains all four ideas that currently compete for prominence:

- hard faults can be in-range;
- reconstruction is the wrong question;
- normality is regime-structured probability;
- fault evidence is often subsystem-local and must not be globally diluted.

### The three places where the spine is most lost

#### 1. §3, especially the transition from MIIM to the method

The MIIM section gives a detailed ten-assumption ontology, but the headline subsystem factorization is not made inevitable by the ontology. A5 and A8 are eventually invoked in §4.4:

> “the few independent levers of A5 and the typed, correlated channels of A8 surface as blocks of the channel correlation graph.”

That bridge is plausible, but it appears after the reader has already absorbed MIIM as a story primarily about **regimes**, not about **modular/local structure**.

**Fix:** At the end of the opening paragraph of §3, add one short bridge:

> “Two consequences matter for detection: normality is multimodal in regime space, and deviations are often local to tightly coupled channel groups. LatAD therefore models regime probability at community scale rather than only at whole-plant scale.”

Do not add an eleventh assumption. Instead, explicitly state that **subsystem locality is an operational consequence of A5 and A8**.

#### 2. Opening of §4 and §§4.1–4.3

The first Method paragraph says:

> “it estimates density either over the whole channel set … or per subsystem …”

and then:

> “We describe the representation and heads first, then the regime-community realization.”

This is structurally backwards for the headline model. It asks the reader to learn the global base first and only later discover the architecture that wins.

**Fix:** Insert an **Architecture overview** before VaDE. Present the regime-community detector first; explain that every community expert uses the common VaDE/scoring machinery detailed next. The global model is then simply the one-community ablation.

#### 3. §7 Discussion

The Discussion’s strongest mechanistic paragraph is:

> “Why reconstruction fails and density wins.”

This is excellent, but it again centers only the first half of the final method. The equally important second mechanism, **why local density beats global density**, is scattered across “What the results show,” Appendix A, and some WADI discussion.

**Fix:** Make the Discussion explicitly parallel:

- **7.1 Why reconstruction fails: reachability is not probability**
- **7.2 Why global density misses local faults: subsystem evidence is diluted**
- **7.3 When reconstruction still helps: train-normal-gated residual evidence**
- then limitations/operational implications.

That structure makes the two mechanisms feel deliberately designed rather than accumulated.

### Does the regime-community story integrate cleanly with reachability-versus-probability?

**Conceptually yes; narratively not yet.**

They are not rival stories. The clean hierarchy is:

- **Reachability versus probability** answers **what anomaly evidence should mean**.
- **Regime-community factorization** answers **where that probability should be evaluated in a modular CPS**.
- **Higher Criticism** answers **how sparse local evidence should be aggregated when fault extent is unknown**.

That is a coherent progression: **what → where → how**.

The manuscript presently presents them more like **method → extension → combiner**, which makes the community contribution feel like a second paper. Recasting the architecture as “probability first, locality second, sparse aggregation third” would integrate it.

---

## 3. Abstract arc audit and full rewritten abstract

### Current abstract arc

The abstract has the right ingredients and a strong first two sentences. Its arc is:

1. paradoxical fault;
2. reachability versus probability;
3. MIIM;
4. VaDE latent density;
5. subsystem factorization;
6. evaluation critique;
7. benchmark results;
8. contribution list.

The main problem is **compression density**. The reader must absorb MIIM, VaDE, latent density, correlation-community factorization, cohesion weighting, Higher Criticism, raw metrics, difficulty stratification, three benchmarks, a confidence interval, and deep-detector collapse in one paragraph. The abstract is not unclear sentence by sentence, but the number of conceptual nouns makes it feel like a results-heavy mini-paper rather than a single causal story.

Three changes would improve it:

- Present LatAD as a **two-part answer** to the opening paradox: regime probability + subsystem locality.
- Mention MIIM after the method logic, not before it, or demote the acronym entirely from the abstract if space is tight.
- Replace the final contribution inventory with one mechanistic interpretation. The current last sentence repeats the method instead of landing the impact.

### Recommended rewritten abstract, 230 words

Cyber-physical systems can enter faulty states in which every channel remains within its familiar range and a high-capacity decoder reconstructs the observation accurately. Such states expose a mismatch between reconstruction and detection: reconstruction tests reachability, whereas anomaly detection requires probability under normal joint operation. We model normal CPS behavior as many imbalanced operating regimes distributed across locally coupled sensor-actuator communities. LatAD jointly learns a regime-structured latent representation with VaDE, scores windows by latent density, and factorizes that score over correlation-defined communities that proxy plant subsystems. Per-community tail probabilities are combined with cohesion-weighted Higher Criticism so that sparse local violations are not diluted by unaffected channels, while a global expert retains sensitivity to plant-wide deviations. Optional reconstruction evidence is used only when train-normal validation indicates that it generalizes. We evaluate on WADI, HAI, and SWaT using raw point-wise metrics and difficulty stratification that separates anomalies detectable by simple univariate rules from those requiring joint-structure modeling. LatAD achieves the highest overall AUROC on all three benchmarks and is strongest on the difficult subset, with the clearest margin on HAI; reconstruction-based USAD and TranAD deteriorate sharply on WADI and HAI difficult faults. Mechanism analyses show that these faults can reconstruct as well as normal observations while remaining low-density in the learned latent. The results support a two-part CPS monitoring principle: model probability over normal regimes, then preserve subsystem locality when aggregating anomaly evidence.

### Why this abstract is stronger

It gives the reader a single chain:

**fault paradox → wrong question → probability model → subsystem-local probability → sparse aggregation → evaluation designed for the hard case → result → mechanism.**

It also avoids calling the correlation communities literal subsystems as an established fact; “proxy plant subsystems” is more defensible and easier to accept.

---

## 4. Opening/hook audit and rewritten opening paragraph

### What works now

The first paragraph of §1 is one of the strongest parts of the manuscript. The opening sentence is concrete:

> “A cyber-physical system can be faulty even when every sensor reads within its normal range and a deep model reconstructs the full sensor vector accurately.”

The next sentence immediately defines the conceptual distinction:

> “reconstruction asks whether a state can be reproduced, whereas detection needs to know whether normal operation is ever likely to occupy it.”

The closing sentence also gives a clear promise:

> “This paper detects them by modelling that joint structure and scoring probability rather than reconstruction.”

This is a real hook. Do not replace it with a generic “CPS are increasingly important” opening.

### What is missing

The opening explains **why density matters** but not **why the headline detector is community-factorized**. The community/locality idea appears only much later. Because §4.4 is the headline method, the first paragraph should plant that seed.

The phrase:

> “These are the faults that matter most for condition monitoring and predictive maintenance”

is also stronger than the manuscript’s evidence. The datasets are attack benchmarks, not longitudinal degradation or remaining-useful-life datasets. The motivation is relevant to predictive maintenance, but the paper demonstrates anomaly detection, not prognostics.

### Recommended opening paragraph

A cyber-physical system can be faulty without any individual sensor leaving its familiar range, and a high-capacity decoder can still reconstruct the full observation accurately. The reason is simple: reconstruction asks whether a state can be reproduced, whereas anomaly detection asks whether normal operation is likely to occupy that state. In a controlled plant, vehicle, or machine, normal behavior occupies many operating regimes constrained by physics and control logic, and component-level faults often disturb only one tightly coupled group of channels. A correlation break or between-regime state can therefore be globally reconstructable yet locally improbable. Such in-envelope violations are precisely the cases that fixed alarm limits and reconstruction residuals are least equipped to expose. We address them by modeling probability over normal regimes at subsystem scale, then combining the resulting local anomaly evidence without allowing unaffected channels to dilute it.

### Follow-on opening structure

After this paragraph, the Introduction should move in this order:

1. **Why normal-only learning is operationally necessary.**
2. **What CPS normality looks like: regimes + local coupling.**
3. **Two failure modes of existing detectors: reconstruction and global dilution.**
4. **One evaluation failure mode: easy anomalies dominate headline metrics.**
5. **Contributions.**

The current generic CPS-definition paragraph should be shortened or moved after the conceptual hook.

---

## 5. Design-space framing, §4.4, and the community=subsystem motivation

### Is the four/five-axis framing clear?

**No. It is currently over-engineered and internally hard to count.**

The sentence at the end of §4.4 says:

> “The detector varies one point on each of five design axes, representation (VaDE), density estimation (single global vs a regime-community ensemble: construction plus aggregation), scoring heads, and train-normal calibration; the Appendix ablates each axis in turn.”

A reader counting the nouns sees four top-level items:

1. representation;
2. density estimation;
3. scoring heads;
4. calibration.

The text reaches five only by treating “construction plus aggregation” as two hidden subaxes inside density estimation. That is not obvious. It also says the Appendix “ablates each axis in turn,” while Appendix A, as narrated, focuses on factorization and aggregation, and Table 5 handles score heads in the Discussion. Even without checking the numbers, the **story of the ablation architecture does not match the claimed organization**.

### Recommended five-decision framing

If five is important, name all five explicitly and use the same names everywhere:

1. **Representation:** how normal windows are embedded, here VaDE.
2. **Density factorization:** one global model or a set of community models.
3. **Community aggregation:** how community tail probabilities are combined, here cohesion-weighted Higher Criticism plus a global expert.
4. **Score composition:** density, nearest-regime, optional residual, and optional basin evidence.
5. **Calibration/adaptation:** train-normal normalization and train-normal gating.

Then write:

> “LatAD makes five separable design decisions: representation, density factorization, cross-community aggregation, score composition, and train-normal calibration. Table 5 isolates score composition; Appendix A isolates factorization and aggregation; representation and calibration are fixed in the reported implementation.”

That is much clearer than claiming that one appendix ablates “each axis.”

### Is §4.4 itself clear?

The physical intuition is good, but it does too much in three paragraphs. It introduces:

- subsystem locality;
- correlation graph;
- agglomerative clustering;
- nested communities;
- community sizes;
- per-community copies of the detector;
- calibrated tail surprise;
- a global null expert;
- an additive decomposition;
- cohesion weights;
- sum versus max logic;
- Higher Criticism;
- five design axes.

For a non-ML CPS reader, that is the densest conceptual section of the paper and should be slower, not faster.

### Recommended §4.4 structure

Split §4.4 into four short subsections or bolded steps:

**4.4.1 Why localize the density**  
Explain dilution with one physical example.

**4.4.2 Building correlation communities**  
Define them as data-derived proxies for tightly coupled channel groups.

**4.4.3 Scoring each community**  
State that each community reuses the same regime-density machinery.

**4.4.4 Combining sparse community evidence**  
Explain cohesion weighting, Higher Criticism, and the global expert.

The reader should understand each step before seeing the next.

### Is community = subsystem persuasive?

**The motivation is persuasive; the equality is too strong.**

The current wording says:

> “Agglomerative clustering on the train-normal correlation graph recovers these subsystems as nested communities…”

That sentence moves from physical argument to empirical identity without qualification. Unless the manuscript directly validates the learned communities against plant schematics or known process units, the safer and more credible wording is:

> “Agglomerative clustering on the train-normal correlation graph produces nested communities that serve as data-derived proxies for tightly coupled plant subsystems.”

Likewise, instead of:

> “Faults inherit the same locality…”

write:

> “Many component-level faults initially produce localized violations, so a community-scale score can preserve evidence that a whole-plant density dilutes.”

This is not weaker storytelling. It is stronger because the reader does not have to resist an overclaim.

### The most useful conceptual sentence to add

At the top of §4.4:

> **“The latent-density principle solves the wrong-score problem; community factorization solves the dilution problem.”**

That one sentence joins the two halves of the paper.

---

## 6. Impact framing: predictive maintenance, digital twins, and CPS operations

### Current state

The paper is **simultaneously under-sold and over-sold** in different places.

It is under-sold because the operational bridge is mostly compressed into isolated phrases:

> “condition monitoring and predictive maintenance”

> “from paging an operator”

> “deployable”

The paper never explains how the detector fits into an actual CPS monitoring stack.

It is over-sold because phrases such as:

> “These are the faults that matter most for condition monitoring and predictive maintenance”

and

> “Two properties make the detector deployable for condition monitoring and predictive maintenance”

imply evidence beyond attack-benchmark anomaly detection. Predictive maintenance normally entails degradation, temporal progression, maintenance decisions, lead time, or prognostic value. The manuscript does not demonstrate those outcomes.

### Stronger but restrained operational framing

The right claim is:

**LatAD is an anomaly/condition-monitoring layer whose modeling choices are relevant to predictive-maintenance and digital-twin stacks; the present experiments establish detection behavior, not prognostic value or full digital-twin capability.**

A clean Discussion paragraph could say:

> “Operationally, LatAD can be viewed as a data-driven normal-behavior model: it estimates which multivariate operating states are probable within learned channel communities and raises evidence when the plant enters an unlikely local regime. Such a model can serve as the anomaly layer of a condition-monitoring or digital-twin stack, complementing physical residuals and fixed alarms. The present benchmarks test detection, not degradation forecasting, remaining useful life, or maintenance decision quality; those predictive-maintenance claims require longitudinal fault-progression data.”

This gives the paper a real digital-twin connection without calling LatAD itself a digital twin.

### Three concrete operational implications worth stating

1. **Alarm burden:** nearest-regime scoring is motivated by avoiding false alarms on rare-but-valid modes such as startup or unusual maneuvers.
2. **Localization:** community scores can identify which tightly coupled channel group produced the anomaly evidence, even if the current paper evaluates detection rather than diagnosis.
3. **Integration:** a data-driven regime model can complement model-based residuals or physics-based digital twins where exact plant models are incomplete.

### Phrases to avoid

Avoid:

- “make the detector deployable”
- “faults that matter most”
- “directly deployable”
- “predictive maintenance” as if the benchmark demonstrates prognosis.

Prefer:

- “operationally relevant”
- “supports condition-monitoring use”
- “can serve as an anomaly layer”
- “motivates evaluation on longitudinal degradation data.”

---

## 7. Prioritized top-10 highest-leverage prose/structure fixes

### 1. Move the regime-community architecture to the front of §4

**Location:** §4 opening, before §4.1.

**Current problem quote:**

> “We describe the representation and heads first, then the regime-community realization.”

**Why it hurts:** The manuscript postpones the actual headline detector and makes §4.4 feel like an extension.

**Concrete fix:**

Replace the opening with:

> “LatAD is a regime-community anomaly detector. It first partitions channels into correlation-defined communities that proxy tightly coupled subsystems, fits a regime-structured latent-density model to each community and to the whole plant, and then combines calibrated community surprises with cohesion-weighted Higher Criticism. The global-density model is the one-community ablation. Sections 4.1–4.3 define the common representation and score used by each expert; §4.4 defines community construction and aggregation.”

### 2. Unify the paper around two failure modes, not one

**Location:** end of §1 opening and contribution paragraph.

**Current problem:** The Introduction foregrounds the reconstruction problem but barely seeds the global-dilution problem.

**Concrete rewrite to add after the reachability/probability distinction:**

> “A second difficulty follows from CPS modularity: a local fault can be improbable within one tightly coupled sensor group yet weak when averaged over the full plant. LatAD therefore addresses both failures, replacing reconstruction with regime probability and replacing a single global score with sparse aggregation of community-level probability.”

### 3. Demote MIIM from co-headline contribution to explanatory framework

**Location:** Abstract, §1 contribution paragraph, §3.

**Current problem quote:**

> “The contributions are the MIIM assumptions, the difficulty-stratified evaluation protocol, and LatAD…”

**Why it hurts:** The reader is asked to treat the acronymized assumption list as a contribution equal to the detector, although the manuscript’s memorable contribution is the detection principle and regime-community realization.

**Concrete rewrite:**

> “We make three contributions: a regime-community detector that scores probability rather than reconstruction; a CPS-normality framework that makes its design assumptions explicit; and a raw, difficulty-stratified evaluation that tests the in-envelope faults those assumptions target.”

This keeps MIIM but makes it serve the method.

### 4. Fix the four/five-axis inconsistency

**Location:** final sentence of §4.4 and Appendix A opening.

**Current quote:**

> “five design axes, representation (VaDE), density estimation (single global vs a regime-community ensemble: construction plus aggregation), scoring heads, and train-normal calibration…”

**Concrete rewrite:**

> “The implementation separates five design decisions: representation, density factorization, cross-community aggregation, score composition, and train-normal calibration. Table 5 isolates score composition; Appendix A isolates factorization and aggregation.”

Do not say the Appendix ablates every axis unless it actually does.

### 5. Qualify correlation community as a subsystem proxy

**Location:** §4.4 first paragraph.

**Current quote:**

> “Agglomerative clustering on the train-normal correlation graph recovers these subsystems as nested communities…”

**Concrete rewrite:**

> “Agglomerative clustering on the train-normal correlation graph produces nested communities that serve as data-derived proxies for tightly coupled plant subsystems, without requiring a plant schematic.”

This preserves the physical intuition and avoids making a stronger identity claim than the narrative establishes.

### 6. Remove the inactive basin head from the main story or compress it heavily

**Location:** §4.3(v).

**Current quote:**

> “it stays inactive on all three benchmarks here and is therefore not part of the empirical contribution…”

**Why it hurts:** A full mathematical subsection is spent on a component that contributes no empirical signal in the reported benchmarks. For a non-ML CPS reader this increases perceived complexity exactly before the headline §4.4.

**Concrete fix:** Reduce §4.3(v) to a short paragraph or move its details to an appendix:

> “An optional basin-agreement head is available for heavily overlapping regimes. It is gated entirely from train-normal ambiguity and is inactive on all three reported benchmarks; details are provided in Appendix X.”

### 7. Reorganize the Discussion around the two mechanisms

**Location:** §7.

**Current problem:** “Why reconstruction fails and density wins” is excellent, but the factorization mechanism is not given equal conceptual treatment.

**Concrete structure:**

- 7.1 Reachability is not probability.
- 7.2 Local probability is lost in a global density.
- 7.3 Train-normal gating admits reconstruction only when useful.
- 7.4 Operational implications for condition monitoring.
- 7.5 Limits and next validation.

Move the strongest factorization evidence from Appendix A into the second paragraph of the Discussion narrative, while retaining the table in the Appendix if desired.

### 8. Reduce adversarial evaluation rhetoric

**Locations:** §§2.1, 5.5, 6, 7, 8.

**Current phrases:**

> “The illusion of progress…”

> “fair protocol”

> “collapse”

> “the concrete face of the ‘illusion of success’”

> “deliberately fair”

**Concrete rewrite style:**

Use neutral language:

- “evaluation inflation under point adjustment”
- “common raw-metric protocol”
- “performance falls to near chance”
- “the raw-metric comparison exposes limited difficult-subset signal.”

This sounds more authoritative and less prosecutorial.

### 9. Tighten predictive-maintenance and deployability claims

**Locations:** §1 first paragraph; §7 alternative validation; §8 final paragraph.

**Current quotes:**

> “These are the faults that matter most for condition monitoring and predictive maintenance”

> “the extension is directly deployable for that test”

> “Two properties make the detector deployable…”

**Concrete rewrite:**

> “These in-envelope deviations are operationally important for condition monitoring because they can precede threshold violations.”

and

> “These properties are attractive for condition-monitoring deployment: rare valid regimes are less likely to trigger alarms, and local joint violations remain visible.”

Then explicitly state that predictive-maintenance/prognostic value requires longitudinal degradation data.

### 10. Shorten the Results narrative and make each paragraph answer one question

**Location:** §6, especially “Difficult, joint-structure faults,” “Robustness…,” and “double-hard subset.”

**Current problem:** Several paragraphs combine interpretation, dataset-by-dataset numbers, statistical qualification, mechanism claims, and limitations.

**Concrete fix:** Use a consistent results grammar:

1. one-sentence result;
2. one supporting number per dataset;
3. one interpretation;
4. one qualification.

For example, begin the difficult-subset paragraph:

> “The regime-community model separates the difficult faults most clearly on HAI and WADI, the two datasets in which the simple range rule loses signal. On HAI, LatAD reaches 0.849 versus 0.757 for the AutoEncoder and 0.497/0.445 for USAD/TranAD. On WADI it reaches 0.796 versus 0.677 for Isolation Forest. SWaT remains a ceiling case. This pattern is consistent with the proposed mechanism: local density helps most where anomalies remain marginally ordinary.”

Then put bootstrap details in the next sentence or caption.

---

## 8. Twenty-four of the weakest sentences, with tighter rewrites

These are selected for narrative load, overclaim, density, tone, or readability rather than scientific correctness.

### 1. §1, paragraph 2

**Original:**

> “A modern CPS, a water-treatment plant, an industrial control loop, or a rotating machine, couples hundreds of networked sensors, embedded controllers, and physical actuators into a unit that continuously senses its own state and acts on it; it is the sensing-and-actuation fabric of the industrial Internet of Things (IoT).”

**Rewrite:**

> “A modern CPS couples sensors, controllers, and actuators into a closed sensing-and-control loop, generating continuous multivariate telemetry.”

### 2. §1, paragraph 2

**Original:**

> “That behaviour is richly observable: vibration, current, temperature, pressure, and position, together with internal control signals, stream continuously as IoT telemetry and carry an early signature of almost any developing fault.”

**Rewrite:**

> “Vibration, current, temperature, pressure, position, and control signals provide continuous telemetry in which developing faults may appear before a fixed alarm threshold is crossed.”

### 3. §1, paragraph 3

**Original:**

> “The task is often framed as characterising the anomalies, but that framing is misleading.”

**Rewrite:**

> “In unsupervised CPS monitoring, the practical modeling problem is not to enumerate faults but to characterize normal operation.”

### 4. §1, paragraph 3

**Original:**

> “The usable information lies on the other side of the boundary, in the normal data, which is abundant and rarely exploited in full.”

**Rewrite:**

> “The strongest training signal is therefore the normal record, which is abundant relative to labeled faults.”

### 5. §1, paragraph 3

**Original:**

> “A detector that mis-models this structure fails however it thresholds.”

**Rewrite:**

> “If the model collapses this structure, threshold tuning cannot recover the missing notion of normality.”

### 6. §1, paragraph 4

**Original:**

> “A fault is anything that leaves this union of normal regimes, and the operationally difficult ones stay inside the marginal envelope of every channel while breaking the joint structure: a correlation break, a between-regime pocket, a history-inconsistent state.”

**Rewrite:**

> “The difficult anomalies are those that leave the joint support of normal regimes while remaining inside familiar per-channel ranges, for example a correlation break or a between-regime state.”

### 7. §1, paragraph 5

**Original:**

> “If normal is a union of many imbalanced regimes, a global anomaly model under-fits the regime structure, a single deep autoencoder alarms on rare-but-valid regimes (the trust-eroding false positive), and a reconstruction residual is blind to exactly the joint-structure faults that matter, because a flexible decoder reconstructs them faithfully.”

**Rewrite:**

> “Three errors follow from this structure: a global model can blur distinct regimes, a detector can mistake rare valid regimes for faults, and a flexible decoder can reconstruct joint-structure anomalies with little residual.”

### 8. §1, contribution paragraph

**Original:**

> “We build LatAD, a detector whose whole design is modelling that normal law: a jointly learned latent with explicit Gaussian-mixture regime clustering (VaDE), scored in the latent by a mixture-density head and a rare-regime-safe nearest-component likelihood (§4).”

**Rewrite:**

> “We build LatAD to model that normal law directly: VaDE learns a regime-structured latent, and latent-density scores measure how probable each window is relative to normal regimes.”

### 9. §2.3

**Original:**

> “Both remain reconstruction-based, so the mechanism analysis of §7 suggests they inherit the same blind spot on the reconstructable-but-improbable joint faults that define our difficult subset.”

**Rewrite:**

> “Because both retain reconstruction-based scoring, the mechanism examined in §7 suggests that they may face the same failure mode on reconstructable but low-density joint states.”

### 10. §3, final paragraph

**Original:**

> “The regimes overlap on WADI and HAI (mean silhouette 0.06 and 0.08) but separate more crisply on SWaT (0.29), and this ordering tracks where the design pays off: the two overlapping datasets are exactly where the regime-community factorization gains most, since a single global density blurs local structure there, while crisp SWaT is a ceiling for every method (§6).”

**Rewrite:**

> “WADI and HAI show more overlapping regime structure than SWaT. The largest gains from community factorization occur on those two datasets, consistent with the idea that a global density can blur local structure.”

### 11. §4 opening

**Original:**

> “We call the detector LatAD. It has a representation stage … and a scoring stage … and it estimates density either over the whole channel set … or per subsystem…”

**Rewrite:**

> “LatAD is a regime-community detector: it estimates normal-regime probability within correlation-defined channel communities and combines their calibrated anomaly evidence, with a whole-plant model retained as a global expert.”

### 12. §4, “One realization of the assumptions”

**Original:**

> “A1–A10 specify what a model of CPS normal must capture, not how to implement it; the detector below is one concrete realization of A1–A8, reported in full so that every choice is auditable, and §7 surveys the design space the same assumptions admit.”

**Rewrite:**

> “A1–A10 describe the structure to model; LatAD is one implementation of A1–A8, while §7 discusses alternative estimators compatible with the same assumptions.”

### 13. §4.2

**Original:**

> “On the difficult faults (correlation breaks and between-regime pockets) this is exactly wrong.”

**Rewrite:**

> “On the difficult faults examined here, that joint score can be counterproductive because the reconstruction term contributes little or misleading signal.”

### 14. §4.2

**Original:**

> “Demoting reconstruction from the default score is a universal difficult-subset improvement and also lifts the combined and easy columns on HAI.”

**Rewrite:**

> “Across the reported difficult-subset ablations, removing reconstruction from the default score improves the base detector; reconstruction is added back only where train-normal gating supports it.”

### 15. §4.2

**Original:**

> “reconstruction re-enters only through the optional, auto-gated residual head (§4.3 iv), which fires only where it provably generalises to held-out normal…”

**Rewrite:**

> “reconstruction re-enters only through an optional residual head, activated when its score remains stable on held-out normal data.”

### 16. §4.3(iv)

**Original:**

> “the residual carries the fault on HAI but is dead on WADI, and the gate recovers exactly this pattern from train-normal alone.”

**Rewrite:**

> “The held-out-normal gate activates the residual on HAI and suppresses it on WADI, matching the dataset-specific usefulness observed in the evaluation.”

### 17. §4.3(v)

**Original:**

> “Between-regime pockets (A3) are the dangerous false negative on datasets whose regimes overlap.”

**Rewrite:**

> “Between-regime pockets are a potential false-negative mode when regime boundaries overlap.”

### 18. §4.4

**Original:**

> “Agglomerative clustering on the train-normal correlation graph recovers these subsystems as nested communities…”

**Rewrite:**

> “Agglomerative clustering on the train-normal correlation graph produces nested channel communities that serve as data-derived proxies for tightly coupled subsystems.”

### 19. §4.4

**Original:**

> “Faults inherit the same locality: a stuck valve, a drifting analyzer, or a fouled sensor perturbs its own subsystem while the rest of the plant runs normally…”

**Rewrite:**

> “Many component-level faults initially perturb a limited set of coupled channels while much of the plant remains near normal.”

### 20. §4.4

**Original:**

> “The detector varies one point on each of five design axes, representation (VaDE), density estimation (single global vs a regime-community ensemble: construction plus aggregation), scoring heads, and train-normal calibration; the Appendix ablates each axis in turn.”

**Rewrite:**

> “The implementation separates five design decisions: representation, density factorization, cross-community aggregation, score composition, and train-normal calibration. Table 5 and Appendix A isolate the score-composition, factorization, and aggregation choices.”

### 21. §5.1

**Original:**

> “We use three real CPS datasets and no synthetic data in the results tables.”

**Rewrite:**

> “We evaluate on three real CPS attack testbeds: WADI, HAI, and SWaT.”

### 22. §5.4

**Original:**

> “F1 is the best raw point-wise F1 over score thresholds (an oracle, field-standard threshold, disclosed plainly).”

**Rewrite:**

> “F1 is reported as the best raw point-wise F1 over test-swept thresholds and is therefore treated as a secondary oracle metric.”

### 23. §5.5

**Original:**

> “This is deliberate: point adjustment inflates F1 so severely that a random score outscores every deep model (§2), so a fair comparison must use raw metrics for all methods…”

**Rewrite:**

> “We therefore use raw metrics for every method, avoiding the point-adjustment inflation discussed in §2.”

### 24. §6

**Original:**

> “This is the concrete face of the ‘illusion of success’: models that look strong under point-adjusted F1 have little difficult-fault signal once the metric is raw.”

**Rewrite:**

> “The raw difficult-subset results show that strong point-adjusted scores do not necessarily imply useful signal on the in-envelope anomalies targeted here.”

### Additional sentences that should also be softened

**§6:**

> “This is where the design earns its keep.”

Use:

> “This subset is the most discriminative test of the proposed design.”

**§7:**

> “No snapshot detector, ours or the deep baselines, separates these at a low false-alarm budget; they are an intrinsic property of the data rather than of any one detector.”

Use:

> “These onset/offset windows remain difficult for all evaluated snapshot detectors, suggesting that additional temporal or physics-based context may be required.”

**§8:**

> “Treating multimodal normal as a single blob, or as a reconstruction target, is a modelling failure.”

Use:

> “For these benchmarks, collapsing multimodal normality into a single global representation or relying only on reconstruction misses important joint-structure anomalies.”

**§8:**

> “Two properties make the detector deployable for condition monitoring and predictive maintenance…”

Use:

> “Two properties are operationally attractive for condition monitoring…”

---

## 9. Tone audit, process language, hedging, and punctuation checks

### Punctuation/word checks requested

In the manuscript’s visible prose:

- **Em dash (—): none found.**
- **Double hyphen (--): none found in visible prose.**
- **“honestly”: none found.**
- **“frankly”: none found.**
- **“candidly”: none found.**

The manuscript uses en dashes appropriately in ranges and compounds such as A1–A10 and 0.30–0.50.

### Tone that is too adversarial or self-certifying

The bigger tone issue is not hedging. It is **argumentative self-certification**.

Repeated examples:

> “fair protocol”

> “a fair comparison must…”

> “deliberately fair”

> “illusion of progress”

> “illusion of success”

> “collapse”

> “earns its keep”

> “provably generalises”

> “recovers exactly this pattern”

These phrases ask the reader to accept the authors’ evaluation of their own rigor or competitors rather than simply observe the evidence.

#### Replace self-certification with procedure

Instead of:

> “Under a fair protocol…”

write:

> “Under raw point-wise metrics, difficulty stratification, and train-normal-only calibration…”

Instead of:

> “deliberately fair…”

write:

> “common raw-metric, difficulty-stratified…”

The procedure is stronger than the adjective.

### Process/apologetic phrasing

The paper contains several places where the prose sounds like a response to reviewers rather than a finished article:

> “reported in full so that every choice is auditable”

> “left to future work”

> “no synthetic data in the results tables”

> “disclosed plainly”

> “within our compute budget”

> “shown for completeness, not as an unbiased comparison”

> “should be validated on additional, untouched CPS systems before its per-dataset gains are treated as established”

Some qualification is necessary, but it should be consolidated.

#### Recommended policy

- Keep the important methodological limitations.
- Remove language explaining the authors’ intentions.
- State facts once, in the natural location.
- Move repeated future-work statements into one compact limitations paragraph.

For example, “basin head inactive here,” “A9–A10 future,” “newer baselines future,” “additional CPS systems future,” and “physics-informed residual future” currently produce a cumulative sense of unfinished architecture. The method will feel more mature if these are consolidated.

### Hedging

The paper is not excessively hedged. In fact, several claims are stronger than necessary. The most useful tone shift is **from absolute to bounded**:

- “universal improvement” → “improves the reported difficult-subset ablations”
- “recovers subsystems” → “produces subsystem proxies”
- “faults inherit the same locality” → “many component-level faults are local”
- “deployable” → “operationally attractive”
- “intrinsic property of the data” → “remains difficult for the evaluated snapshot detectors.”

### Colloquial phrases to remove

- “single statistical blob”
- “earns its keep”
- “dead on WADI”
- “paging an operator”
- “whole design”
- “one-line univariate rule” can stay once, but repeated use becomes rhetorical.

“Blob” is memorable, but for the final journal version “single global mode” or “single undifferentiated density” is more professional.

---

## 10. Readability for a non-ML CPS/controls engineer, plus heading and caption punch-ups

### Where the CPS reader is most likely to get lost

#### A. §3: MIIM as ten assumptions before the architecture is visible

A controls engineer can understand regimes, constraints, setpoints, and local coupling. The problem is not the concepts; it is the **mapping burden** of A1–A10, each tied to later head numbers and future mechanisms.

**Fix:** Put a one-paragraph method preview before or after Table 1:

> “LatAD uses these assumptions in three layers: VaDE represents operating regimes; density heads score how probable a window is within those regimes; community factorization preserves local subsystem violations. A1–A8 support the reported detector, while A9–A10 define temporal extensions.”

That gives the assumption table a destination.

#### B. §4.1: VaDE equations arrive before the reader knows the full system

The ELBO is standard for ML readers but interrupts the story for CPS readers.

**Fix:** Add an architecture overview figure or boxed five-step pipeline before the equations. Then say what the reader needs to retain:

> “The important point for the detector is that the latent is explicitly organized into normal operating regimes during training.”

The equation can follow as implementation detail.

#### C. §4.3: scoring-head enumeration

The numbering “(i)–(ii), (iii), (iv), (v)” is hard to remember, particularly because (i)–(ii) are combined in one heading and the basin head is inactive.

**Fix:** Name the heads by function and put active status in a compact table:

| Head | Purpose | Always active? | What problem it addresses |
|---|---|---:|---|
| Fine latent density | local probability | yes | non-Gaussian regime interiors |
| Nearest regime | rare-mode protection | yes | rare valid operation |
| Whitened residual | reconstruction evidence | gated | datasets where residual generalizes |
| Basin agreement | boundary ambiguity | gated | overlapping regimes |

The main text can then explain only the logic; formulas remain for reproducibility.

#### D. §4.4: Higher Criticism

For a controls engineer, Higher Criticism is likely the least familiar component. The current sentence is mathematically concise but conceptually abrupt.

**Fix:** Add a plain-language sentence before naming it:

> “The number of affected subsystems is unknown: some faults are local, others plant-wide. We therefore need an aggregator that is sensitive to a few extreme community scores without ignoring broader distributed evidence.”

Then introduce Higher Criticism as the chosen statistic.

#### E. §5.3: difficulty split

The split is central to the paper but its operational meaning could be clearer.

**Fix:** Rename:

- **Easy** → **marginally separable**
- **Difficult** → **in-envelope / joint-structure**

If changing labels would disrupt tables, at least define once:

> “We use ‘difficult’ as shorthand for anomalies not separated by the train-normal univariate range rule; it is an evaluation stratum, not a claim of universal difficulty.”

#### F. §6: too many robustness strata in sequence

Easy/difficult, stronger difficulty, and double-hard are all legitimate, but the reader can lose track of which one carries the main claim.

**Fix:** State the hierarchy:

1. primary: difficult subset;
2. robustness: stronger univariate exclusion;
3. stress test: double-hard subset.

Use those exact labels in headings.

### Recommended title revision

Current title:

> “Modeling Normal Is All You Need: Joint Latent Clustering for Anomaly Detection in Multimodal Cyber-Physical Systems”

The first phrase is catchy but has two drawbacks:

- “all you need” is inconsistent with the optional residual head and future trajectory extensions;
- “Joint Latent Clustering” does not name the distinctive regime-community factorization.

Best title for the actual paper:

> **Reconstructable but Improbable: Regime-Community Latent Density for Anomaly Detection in Cyber-Physical Systems**

More conventional alternative:

> **LatAD: Regime-Community Latent Density for Multimodal Cyber-Physical Anomaly Detection**

Mechanism-forward alternative:

> **From Reconstruction to Probability: Subsystem-Factorized Latent Density for CPS Anomaly Detection**

### Heading punch-ups

**Current:** “2.1 The illusion of progress in time-series anomaly-detection evaluation”  
**Better:** “2.1 Evaluation inflation and benchmark triviality in time-series anomaly detection”

**Current:** “3 The MIIM assumptions about CPS normal data”  
**Better:** “3 CPS normality as imbalanced operating regimes”  
Then introduce MIIM in the first paragraph.

**Current:** “4 Method”  
**Better:** “4 LatAD: regime probability with subsystem-local aggregation”

Add before current §4.1:

> **4.1 Architecture overview: probability first, locality second**

Then renumber:

- 4.2 Regime-structured latent representation
- 4.3 Latent-density scoring
- 4.4 Train-normal-gated auxiliary evidence
- 4.5 Community factorization and sparse aggregation

**Current:** “4.2 Scoring: why we demote reconstruction”  
**Better:** “Latent-density scoring: reachability is not probability”

**Current:** “4.3 The scoring stack”  
**Better:** “Score components and train-normal gating”

**Current:** “4.4 Regime-community density (the headline realization)”  
**Better:** “Subsystem-factorized regime density and sparse aggregation”

Avoid “headline realization”; it sounds like manuscript process language.

### Caption punch-ups

#### Table 1

Current caption is already useful but can be more compact.

**Suggested:**

> **Table 1. CPS normal-data assumptions and their role in LatAD.** A1–A8 describe instantaneous regime structure used by the reported detector; A9–A10 describe temporal structure reserved for trajectory-aware extensions.

#### Table 2

Current caption interprets too much and points forward to Table 5.

**Suggested:**

> **Table 2. Reconstruction and latent-density signal on the difficult subset.** AUROC for a single trained model shows that latent likelihood carries more difficult-subset signal than the reconstruction residual on WADI and HAI; Table 5 reports the multi-seed head ablation.

#### Table 3

The current caption is overloaded with nearly a paragraph of protocol details.

**Suggested:**

> **Table 3. Raw anomaly-detection performance on marginally separable, difficult, and full test subsets.** Values are [AUROC, best raw F1]. Learned detectors report five-seed AUROC mean±SD; GDN is single-run on SWaT. LatAD denotes the regime-community model; the global-density row is its one-community ablation. Bold marks the best AUROC in each dataset/subset column.

Move LinRes and split definitions to table notes.

#### Table 4

**Suggested:**

> **Table 4. AUROC on the double-hard stress-test subset.** The subset retains anomalies missed by both the univariate range rule and LinRes at train-normal 99th-percentile thresholds. Learned detectors report five-seed mean±SD. Episode counts limit inferential strength on WADI and especially SWaT.

#### Figure 1

Current caption is informative but too interpretive.

**Suggested:**

> **Figure 1. Difficult-subset AUROC under raw metrics.** Bars show five-seed means; whiskers show ±1 SD for multi-seed detectors. The largest separation occurs on HAI and WADI, where the difficult subset is weakly separable by the univariate rule; SWaT remains a ceiling case.

#### Table 5

**Suggested:**

> **Table 5. Difficult-subset ablation of LatAD score components.** Values are five-seed AUROC mean±SD. The comparison isolates the contribution of latent density, nearest-regime scoring, and the train-normal-gated residual before community factorization.

#### Table A1

**Suggested:**

> **Table A1. Ablation of density factorization and community aggregation.** The top block compares global and community-factorized density; the bottom block compares aggregation rules for full community experts. The reported LatAD uses full community heads with cohesion-weighted Higher Criticism.

---

# Paragraph-by-paragraph narrative audit

The following audit records the narrative job, the main issue, and the recommended action for each substantive paragraph/caption in manuscript order.

## Abstract

### Abstract paragraph

**Role:** States paradox, principle, method, evaluation, result, and contributions.

**Diagnosis:** Strong opening and complete coverage, but concept density is too high. MIIM, VaDE, community factorization, cohesion weighting, Higher Criticism, difficulty stratification, and detailed results arrive with almost no breathing room.

**Action:** Use the rewritten abstract above. Preserve the opening paradox and reachability/probability distinction. Present the regime-community method as one two-level solution rather than as an added clause after VaDE.

---

## §1 Introduction

### Paragraph 1, “A cyber-physical system can be faulty…”

**Role:** Hook and central conceptual distinction.

**Diagnosis:** Excellent. This is the manuscript’s best paragraph. The only missing element is subsystem locality, which is essential to the headline model. The predictive-maintenance wording is slightly too strong for attack-benchmark evidence.

**Action:** Keep the paradox, add locality, soften “faults that matter most,” and end on “probability at subsystem scale.”

### Paragraph 2, “A modern CPS…”

**Role:** Broad CPS motivation and telemetry context.

**Diagnosis:** Too generic after a strong hook, and the first sentence is syntactically overloaded. “carry an early signature of almost any developing fault” overstates observability.

**Action:** Cut by about 40%. Define CPS only as much as needed for the anomaly problem. Preserve continuous telemetry and unlabeled fault diversity.

### Paragraph 3, “The task is often framed…”

**Role:** Justifies normal-only modeling and introduces multimodal normality.

**Diagnosis:** Important but over-argued. “framing is misleading,” “usable information lies on the other side,” and “fails however it thresholds” add rhetorical force without adding structure.

**Action:** Convert to a clean normal-only argument: faults are sparse and incomplete; normal data are abundant; normal operation consists of multiple constrained regimes.

### Paragraph 4, “A fault is anything…”

**Role:** Defines the hard anomaly class and restates the central mechanism.

**Diagnosis:** Strong conceptual bridge, but “a fault is anything that leaves this union” is too categorical, especially because A10 later allows history-conditioned abnormality at the same instantaneous state.

**Action:** Define the target class rather than all faults: “The difficult anomalies studied here leave the joint support of normal regimes while staying within marginal ranges.”

### Paragraph 5, “Two obstacles… The first is modelling.”

**Role:** Summarizes modeling failures.

**Diagnosis:** The sentence contains three different failure modes: global regime underfit, rare-valid false positives, and reconstruction blindness. This is exactly where the subsystem dilution story should appear, but “global anomaly model” is too vague.

**Action:** Split into three short sentences and explicitly name global dilution.

### Paragraph 6, “The second is evaluation.”

**Role:** Motivates raw metrics and difficulty stratification.

**Diagnosis:** Strong, but “high headline number certifies little” and “easy majority/discriminative minority” are slightly prosecutorial.

**Action:** Keep the logic, neutralize tone, and define the difficult subset as the evaluation stratum that tests joint-structure detection.

### Paragraph 7, contribution paragraph

**Role:** States contributions and previews results.

**Diagnosis:** Too long, and the headline regime-community architecture is missing from contribution (2). The paragraph describes the global density machinery more fully than the actual headline factorization.

**Action:** Make contribution 1 the regime-community detector, contribution 2 the explicit CPS-normality assumptions, contribution 3 the difficulty-stratified raw evaluation. Mention the global density model only as ablation.

### Paragraph 8, “Because aggregate benchmark scores…”

**Role:** Sets evaluation priority and discloses scope.

**Diagnosis:** Useful but reads like reviewer-response housekeeping: “single trained configuration,” “oracle disclosed.”

**Action:** Move implementation/scope disclosures to Methods. End the Introduction with the scientific test, not caveats.

---

## §2 Related work

### Opening paragraph

**Role:** Places LatAD in unsupervised multivariate CPS anomaly detection.

**Diagnosis:** Clear but generic.

**Action:** Shorten to two sentences and end with the two gaps the paper addresses: reconstruction-based scoring and whole-system aggregation.

### §2.1 paragraph 1, evaluation critique

**Role:** Establishes problems with point adjustment and benchmark triviality.

**Diagnosis:** Evidence-rich but very dense. It stacks many papers and conclusions in one long paragraph.

**Action:** Split into “metric inflation” and “benchmark triviality.” Rename the subsection neutrally.

### §2.1 paragraph 2, reconstruction critique

**Role:** Connects prior reconstruction failures to latent scoring.

**Diagnosis:** Good conceptual bridge. The Sarfraz/Wagner sentence is overloaded.

**Action:** Keep the first half almost unchanged; split the final sentence and avoid “plagued.”

### §2.1 paragraph 3, “LatAD adopts this critique…”

**Role:** Converts literature critique into protocol.

**Diagnosis:** Useful but repeats §5.

**Action:** One sentence is enough: “Accordingly, we use raw metrics and a difficulty split, and re-score all compared deep models under the same protocol.”

### §2.2 paragraph

**Role:** Locates VaDE/DAGMM and classical components.

**Diagnosis:** Reads as a component inventory rather than related-work argument.

**Action:** Explain what is inherited versus novel: VaDE supplies regime-structured representation; novelty lies in scoring and community factorization.

### §2.3 main paragraph

**Role:** Positions against deep multivariate baselines.

**Diagnosis:** Strong coverage, but the claim that newer reconstruction methods “inherit the same blind spot” is too definitive without direct evaluation. Compute-budget wording also belongs in limitations/method notes.

**Action:** Use “may face the same failure mode.” Keep USAD/TranAD/GDN comparison concise.

### §2.3 applied-condition-monitoring paragraph

**Role:** Connects benchmark anomaly detection to operational monitoring.

**Diagnosis:** Too thin for the impact burden placed on predictive maintenance elsewhere.

**Action:** Expand by one or two sentences explaining the operational role of a normal-behavior model, but explicitly distinguish detection from prognosis.

### §2.4 benchmark paragraph

**Role:** Introduces datasets and links them to multimodality.

**Diagnosis:** Efficient. “strongly multimodal” followed by BIC counts is clear, but the final clause asks the reader to connect MIIM before §3.

**Action:** Keep; simplify the closing bridge.

---

## §3 MIIM assumptions

### Opening paragraph, “We model the normal law…”

**Role:** Formalizes multimodal normality and introduces MIIM.

**Diagnosis:** Clear enough for an ML reader, slightly abstract for a controls reader. It frames everything around regimes but not yet around subsystem-local structure.

**Action:** Add the two-consequence bridge: multimodal probability + local coupling.

### Table 1 caption and table

**Role:** Maps assumptions to physical causes and design choices.

**Diagnosis:** This is a useful organizing device. The weakness is that some rows point to optional heads or future trajectory mechanisms, making the table look like a full theory while the reported detector uses only part of it.

**Action:** Make the caption explicitly distinguish “used here” from “future trajectory extension.” Consider visually grouping A1–A8 and A9–A10.

### Final paragraph, “Two standing qualifiers…”

**Role:** Gives empirical evidence for MIIM and anticipates where factorization helps.

**Diagnosis:** It jumps from descriptive regime statistics to a strong interpretation of factorization before the factorization has been explained. “This is the regime in which…” is also confusing because “regime” now means both operating mode and favorable dataset condition.

**Action:** Keep the empirical observations, but defer the “where the design pays off” argument to Results/Discussion. End §3 with a transition to the architecture.

---

## §4 Method

### Opening paragraph

**Role:** Defines LatAD and the global/community variants.

**Diagnosis:** This is the main structural failure. It gives equal conceptual status to global and community versions, then postpones the headline version.

**Action:** Rewrite around the regime-community architecture first.

### Paragraph 2, assumption-to-component mapping

**Role:** Maps A1–A10 to active and optional detector pieces.

**Diagnosis:** Too many cross-references and parenthetical IDs before the reader has seen the architecture. It also foregrounds future A9/A10.

**Action:** Move the A9/A10 sentence to limitations. Keep only a compact “A1–A8 are used in the reported detector.”

### Paragraph 3, “One realization of the assumptions”

**Role:** Limits the framework-to-implementation claim.

**Diagnosis:** Sounds defensive and process-oriented.

**Action:** Reduce to one direct sentence or merge into §3.

### §4.1 VaDE paragraph 1

**Role:** Defines the regime-structured latent.

**Diagnosis:** Technically clear, but equations arrive before the reader knows how many community models exist and what their outputs feed.

**Action:** Precede with the architecture overview. Add one sentence after the equation stating what matters conceptually.

### §4.1 VaDE paragraph 2

**Role:** Explains anti-collapse training.

**Diagnosis:** Dense implementation detail, but legitimate for reproducibility.

**Action:** Keep, but consider moving learning-rate/variance-floor detail to a hyperparameter or implementation subsection if page pressure exists.

### §4.2 opening paragraph

**Role:** Shows why naive VaDE reconstruction+latent scoring is wrong for the target anomalies.

**Diagnosis:** Strong turning point. “exactly wrong” is too absolute.

**Action:** Keep the section early and rename it around reachability versus probability.

### Table 2 caption

**Role:** Interprets the score-term decomposition.

**Diagnosis:** Too much argumentative text in the caption, especially “drags the summed joint score down.”

**Action:** Use the concise caption proposed above.

### §4.2 mechanism paragraph

**Role:** Explains reconstruction failure and gating.

**Diagnosis:** One of the core paragraphs. “universal difficult-subset improvement” and “provably generalises” overstate.

**Action:** Tighten and bound to reported data.

### §4.3 opening paragraph

**Role:** Defines score normalization.

**Diagnosis:** Clear.

**Action:** Keep.

### Density-head paragraph

**Role:** Motivates fine latent density beyond one Gaussian per regime.

**Diagnosis:** “parametric KDE” may distract a non-ML reader, because the conceptual role is simply to model non-Gaussian regime interiors more flexibly.

**Action:** Explain purpose first, label second.

### Nearest-component paragraph 1

**Role:** Protects rare valid regimes.

**Diagnosis:** Excellent operational intuition.

**Action:** Keep and connect explicitly to startup/shutdown rare modes.

### Nearest-component paragraph 2

**Role:** Explains why nearest regime complements mixture density.

**Diagnosis:** Slightly repetitive.

**Action:** Compress to one sentence and move the base-score formula immediately after.

### Residual-head paragraph 1

**Role:** Defines optional reconstruction evidence.

**Diagnosis:** Technically dense but coherent.

**Action:** Keep after stating the plain-language purpose: reconstruction is retained only when it adds stable normal-generalizing signal.

### Residual-head paragraph 2

**Role:** Explains train-normal gating.

**Diagnosis:** Good methodological story; “dead on WADI” and “exactly this pattern” are colloquial/strong.

**Action:** Neutral rewrite.

### Basin-head paragraph 1

**Role:** Defines a boundary-ambiguity rescue.

**Diagnosis:** Conceptually interesting but not part of the empirical story.

**Action:** Move formula/detail to appendix.

### Basin-head paragraph 2

**Role:** Discloses that the head is inactive.

**Diagnosis:** This paragraph effectively says the previous subsection is not an empirical contribution.

**Action:** Compress in main text to one sentence.

### Auto-gating note

**Role:** Summarizes which heads are active.

**Diagnosis:** Helpful after a complicated scoring section.

**Action:** Keep, but convert to a tiny table if possible.

### §4.4 paragraph 1

**Role:** Introduces physical modularity and correlation communities.

**Diagnosis:** Essential headline material placed too late. “recovers these subsystems” is too strong.

**Action:** Move the conceptual version to the architecture overview and use “proxy.”

### §4.4 paragraph 2

**Role:** Motivates local fault concentration and defines community experts/global expert.

**Diagnosis:** Strong physical intuition, but “Faults inherit the same locality” is absolute. The global “null expert” term may confuse non-statistical readers.

**Action:** Use “global expert” and explain it as protection against plant-wide anomalies.

### §4.4 paragraph 3

**Role:** Defines cohesion weights, Higher Criticism, and design axes.

**Diagnosis:** This is the most overloaded paragraph in the manuscript. The final design-axis sentence is structurally inconsistent.

**Action:** Split aggregation from design-space framing. Explain unknown fault extent before naming Higher Criticism. Fix the five-axis list.

---

## §5 Experimental methodology

### §5.1 opening sentence

**Role:** Introduces data.

**Diagnosis:** “and no synthetic data in the results tables” sounds like a rebuttal to a concern not present in the narrative.

**Action:** Delete unless synthetic data are otherwise central.

### WADI bullet

**Role:** Dataset handling.

**Diagnosis:** Necessary but “attack carriers, not noise” sounds argumentative.

**Action:** Say “near-constant channels are retained because attacks can act through them.”

### HAI bullet

**Role:** Dataset handling.

**Diagnosis:** Clear.

**Action:** Keep.

### SWaT bullet

**Role:** Dataset handling.

**Diagnosis:** Clear.

**Action:** Keep.

### §5.2 windows/features

**Role:** Defines model input.

**Diagnosis:** Clear but long. The final A9 future-work sentence repeats §4.

**Action:** Remove future-work material from Methods and keep preprocessing only.

### §5.3 difficulty split

**Role:** Defines the central evaluation stratum.

**Diagnosis:** Very important and mostly clear. “The difficult column is the discriminative one” sounds editorial.

**Action:** Say “We treat difficult-subset AUROC as the primary test of joint-structure detection.”

### §5.4 opening

**Role:** Introduces metric comparability.

**Diagnosis:** Fine.

**Action:** Keep.

### AUROC bullet

**Role:** Defines primary metric.

**Diagnosis:** Clear.

**Action:** Keep.

### F1 bullet

**Role:** Defines secondary oracle metric.

**Diagnosis:** “field-standard threshold, disclosed plainly” is defensive.

**Action:** Use the tighter rewrite above.

### Calibration paragraph

**Role:** Establishes no test leakage except oracle F1 threshold.

**Diagnosis:** Important but repeats “train-normal” many times.

**Action:** Compress to two sentences.

### §5.5 baselines paragraph

**Role:** Defines baselines and re-scoring.

**Diagnosis:** Overloaded. It combines baseline taxonomy, score-grid alignment, compute limitation, raw-metric justification, and pedagogical trivial baseline.

**Action:** Split into “Baselines” and “Common evaluation protocol.” Move compute limitation to a parenthetical note or limitations.

---

## §6 Results

### Opening paragraph

**Role:** Tells reader how to read Table 3.

**Diagnosis:** Useful.

**Action:** Keep, but eliminate details already in caption.

### Table 3 caption

**Role:** Encodes protocol and labels.

**Diagnosis:** Far too long.

**Action:** Use the shorter caption proposed above plus table notes.

### “Overall detection” paragraph

**Role:** Gives headline result and factorization gain.

**Diagnosis:** Strong but promotional wording: “clearing every baseline.”

**Action:** State rankings and the global-to-community delta neutrally.

### “Difficult, joint-structure faults” paragraph

**Role:** Main results paragraph.

**Diagnosis:** Important but too dense. It contains five distinct arguments: HAI result, HAI inference, WADI result, WADI inference, SWaT ceiling, and mechanism interpretation.

**Action:** Split into two paragraphs: HAI/WADI as discriminative cases; SWaT as ceiling/contrast case.

### “Trivially separable anomalies”

**Role:** Shows easy subset is non-discriminative.

**Diagnosis:** Clear and short.

**Action:** Keep.

### “Validity of the difficulty split”

**Role:** Shows the split behaves as intended.

**Diagnosis:** Useful but repeats the definition.

**Action:** Reduce to one sentence per dataset class.

### “Robustness to a stronger difficulty definition”

**Role:** Tests dependence on max-mean split.

**Diagnosis:** Necessary robustness, but the sentence defining the stronger split is long.

**Action:** State the purpose first, then the construction.

### “double-hard subset”

**Role:** Secondary stress test.

**Diagnosis:** This is the most complicated Results paragraph. It carries method definition, results, bootstrap interpretation, small-episode caveats, and bias disclaimer.

**Action:** Put the construction in Methods or table caption; Results should state only what changed and what remained.

### Table 4 caption

**Role:** Defines double-hard subset and inferential limits.

**Diagnosis:** Better than Table 3 but still long.

**Action:** Use shorter caption; retain episode counts as table note.

### “Deep SOTA under raw metrics”

**Role:** Summarizes behavior of USAD/TranAD.

**Diagnosis:** Numerically clear, rhetorically combative.

**Action:** Replace “collapse” and “illusion of success” with neutral description.

### Figure 1 caption

**Role:** Visual takeaway.

**Diagnosis:** Too much causal interpretation in caption.

**Action:** Let the figure show ranking; put mechanism interpretation in text.

---

## §7 Discussion

### “What the results show”

**Role:** High-level synthesis.

**Diagnosis:** Strong summary, but “fair protocol” self-certifies and the last sentence compresses all mechanisms.

**Action:** Remove “fair,” then split probability and factorization mechanisms into separate paragraphs.

### “Why reconstruction fails and density wins”

**Role:** Core mechanism analysis.

**Diagnosis:** This is the strongest Discussion paragraph and should be elevated. It directly supports the story spine with real and controlled examples.

**Action:** Keep almost all substance. Shorten the final sentence listing heads. Consider ending on: “The anomaly is reachable by the decoder but improbable under normal regime density.”

### Table 5 caption

**Role:** Defines head ablation.

**Diagnosis:** Too much cross-reference and configuration explanation.

**Action:** Use shorter caption.

### Table 5 interpretation paragraph

**Role:** Shows when density versus reconstruction contributes.

**Diagnosis:** Strong and nuanced. This paragraph prevents the paper from becoming dogmatic about reconstruction.

**Action:** Keep; trim repeated “which is why” clauses.

### “Auto-gating adapts…”

**Role:** Explains adaptive architecture.

**Diagnosis:** Useful but slightly redundant with the preceding head-ablation paragraph.

**Action:** Merge with it or shorten by half.

### “The residual frontier on WADI”

**Role:** Explains remaining hard cases.

**Diagnosis:** Valuable limitation, but “intrinsic property of the data” goes beyond what the narrative establishes.

**Action:** Say these windows remain unresolved by the evaluated snapshot detectors and likely require additional context.

### “Trajectory assumptions A9–A10”

**Role:** Connects unused assumptions to future work.

**Diagnosis:** Repeats earlier future-work disclosures.

**Action:** Merge into Limitations.

### “Alternative realizations and validation”

**Role:** Opens broader design space and qualifies external validation.

**Diagnosis:** The first half risks making the paper sound like a framework paper in addition to a detector paper. The second half is appropriately cautious but “directly deployable for that test” is unnecessary.

**Action:** Keep only the most relevant alternatives and external-validation need.

### “Limitations”

**Role:** Consolidates scope constraints.

**Diagnosis:** Good content, but several limitations have already been stated multiple times.

**Action:** Make this the single authoritative home for them; remove duplicates upstream.

---

## §8 Conclusion

### Paragraph 1

**Role:** Restates contributions/results.

**Diagnosis:** Accurate summary but “deliberately fair” is self-congratulatory, and the sentence is long.

**Action:** Split method and evidence into two sentences. Use procedure rather than “fair.”

### Paragraph 2

**Role:** Restates mechanism.

**Diagnosis:** Excellent first two clauses. Final sentence, “Treating… is a modelling failure,” is too sweeping.

**Action:** Keep the reachability/probability sentence almost verbatim; bound the final claim to the target anomaly class.

### Paragraph 3

**Role:** Operational implications and next steps.

**Diagnosis:** The nearest-regime and local-fault implications are exactly the right impact points. “make the detector deployable” overstates evidence, and predictive maintenance is stronger than the benchmark demonstrates.

**Action:** Say “operationally attractive for condition monitoring” and add the digital-twin/PHM integration sentence proposed in §6 of this review.

---

## Appendix A

### Opening paragraph

**Role:** Explains factorization/aggregation ablation and mechanism.

**Diagnosis:** This contains key narrative evidence for why the headline regime-community method exists. It is too important to leave entirely in an appendix.

**Action:** Keep the table in Appendix A if space requires, but move a two-sentence synthesis into the main Discussion under “Why global density misses local faults.”

### Table A1 caption

**Role:** Defines factorization and aggregation comparisons.

**Diagnosis:** Clearer than several main-text captions, but still refers to “headline detector.”

**Action:** Replace “headline detector” with “reported regime-community configuration.”

---

# Final editorial prescription

The paper should not be rewritten around a new idea. It should be rewritten around the idea it already has.

The most compelling version of the manuscript is not:

**MIIM → VaDE → several score heads → optional gates → regime-community realization → evaluation protocol.**

It is:

**A hard CPS fault can be reconstructable yet improbable. Probability therefore replaces reconstruction as the primary anomaly evidence. Because that improbability is often local to a tightly coupled subsystem, probability is estimated per correlation community rather than only globally. Because fault extent is unknown, sparse community surprises are aggregated adaptively. Raw, difficulty-stratified evaluation then tests exactly the in-envelope cases for which those two design choices were made.**

That arc makes §4.4 feel inevitable, turns the global model into a clean ablation instead of a competing identity for LatAD, gives the controls reader a physical reason for every statistical component, and strengthens the condition-monitoring/digital-twin relevance without claiming predictive-maintenance evidence the benchmarks do not provide.
