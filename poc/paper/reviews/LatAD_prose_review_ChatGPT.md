# LatAD prose, narrative, and storytelling review

## Scope and editorial verdict

This review evaluates narrative arc, story throughline, impact framing, structure, style, tone, and readability for a CPS or controls engineer who is technically strong but not necessarily an ML specialist. It does not re-check numerical results, statistical validity, or scientific correctness.

The manuscript contains a genuinely strong organizing insight, but it currently presents that insight too late and surrounds it with too much architecture detail, evaluation process narration, and future-method speculation. The most memorable idea is not "VaDE with several heads." It is this: **a CPS fault can be reconstructable yet improbable; reconstruction measures whether a decoder can reproduce a state, while density measures whether normal operation is likely to occupy that state.** Everything else should be made to serve that sentence.

A second editorial issue is impact framing. The manuscript is convincingly about anomaly detection in CPS and has a natural bridge to condition monitoring and predictive maintenance. It does not currently demonstrate a digital twin, and it should not imply that it does. If digital-twin relevance is important, frame the learned normal-regime distribution as a probabilistic normal-behavior layer that could support a data-driven digital twin, rather than claiming a digital-twin contribution.

### Front-matter note: title

The current title, **"Modeling Normal Is All You Need: Joint Latent Clustering for Anomaly Detection in Multimodal Cyber-Physical Systems,"** is memorable but rhetorically risky. "All You Need" sounds slogan-like and is not fully aligned with a model that can activate a reconstruction-residual head and explicitly leaves trajectory modeling to future work. A title that carries the actual conceptual breakthrough would be stronger.

Preferred direction: **"Reconstructable but Improbable: Latent Regime Density for Anomaly Detection in Cyber-Physical Systems."**

More conservative direction: **"Modeling Multimodal Normal Operation for Anomaly Detection in Cyber-Physical Systems."**

Mechanism-led direction: **"From Reachability to Probability: Joint Latent Regime Modeling for CPS Anomaly Detection."**

## Paragraph-by-paragraph pass

### 1. Abstract, paragraph

**LOCATION:** Abstract, "Fault detection in networked CPS"

**ROLE:** Establishes the problem, introduces MIIM and the detector, summarizes the evaluation protocol and results, and states the contributions.

**DIAGNOSIS:** The abstract begins with a familiar rare-fault argument and delays the paper's most distinctive idea until near the end. It also carries too many result clauses, subset definitions, confidence intervals, and caveats for an abstract. The narrative therefore reads as "problem, assumptions, architecture, protocol, scoreboard" rather than "surprising failure mode, explanation, remedy, evidence." The phrase "existing AI methods often fail because they ignore the unique structure of CPS" is broad and combative. The term MIIM arrives before the reader has been shown why the regime structure matters. For a controls engineer, "latent-primary anomaly score with an auto-gated reconstruction-residual head" is too implementation-heavy for the closing sentence.

**FIX:** Lead with the reconstructable-but-improbable failure mode. Then explain that CPS normal operation is a union of regimes, making probability in a regime-aware representation more informative than reconstruction error for certain faults. Introduce MIIM only after that intuition. Report one overall result pattern and one decisive difficult-subset result, not every secondary robustness slice. Close on the conceptual contribution, not the head stack.

Original: "Fault detection in networked cyber-physical systems (CPS), which form the sensing and actuation layer of industrial Internet-of-Things deployments, is difficult because faults are rare and poorly represented in available data."

Tighter: "A CPS fault can remain reconstructable by a model of normal data while still being an improbable combination of otherwise normal sensor values."

### 2. Abstract, keywords

**LOCATION:** Abstract metadata, "Keywords: anomaly detection"

**ROLE:** Improves indexing and tells readers what conceptual territory the paper occupies.

**DIAGNOSIS:** The list is technically accurate but weighted toward model machinery. "Variational deep embedding" and "Gaussian mixture model" are implementation terms, while the broader application story is underrepresented. The paper repeatedly invokes operating regimes, condition monitoring, and fault detection, yet "operating regimes" and "predictive maintenance" are absent.

**FIX:** Favor searchable problem and mechanism terms over low-level architecture terms. If the journal permits nine keywords, keep one model term but add the CPS concept that drives the paper.

Original: "anomaly detection; cyber-physical systems; Internet of Things; industrial IoT; unsupervised learning; variational deep embedding; Gaussian mixture model; sensor time series; fault detection"

Tighter: "anomaly detection; cyber-physical systems; operating regimes; unsupervised learning; latent density; multivariate time series; condition monitoring; predictive maintenance; fault detection"

### 3. Introduction, paragraph 1

**LOCATION:** Section 1, "A modern cyber-physical system"

**ROLE:** Establishes the industrial setting, explains why faults cannot be enumerated, and motivates normal-only anomaly detection.

**DIAGNOSIS:** This is a long, conventional opening with multiple nested clauses before the paper's distinctive question appears. The first sentence tries to define CPS, IoT, sensing, actuation, and system coupling at once. The second sentence combines combinatorial failure space, gradual degradation, downtime, and safety. The fourth sentence adds a telemetry inventory. The result is technically respectable but generic. A reviewer can read this paragraph and still not know what this paper sees that others miss.

**FIX:** Open with the surprising detection failure, then zoom out to the CPS setting. Move the broad "hundreds of sensors" description after the conceptual hook. Keep one consequence sentence linking the problem to predictive maintenance. The opening should create a contradiction the paper resolves: "the model can reproduce the fault, yet the state is not normal."

Original: "A modern cyber-physical system (CPS), whether a water-treatment plant, an industrial control loop, or a rotating machine, integrates hundreds of networked sensors, embedded controllers, and physical actuators, the sensing-and-actuation fabric of the industrial Internet of Things (IoT), into a single tightly coupled unit that continuously senses its own state and acts upon it."

Tighter: "In a modern CPS, a faulty state can look individually normal on every sensor and can even be reconstructed accurately by a deep model, yet still be an implausible state of the coupled physical process."

### 4. Introduction, paragraph 2

**LOCATION:** Section 1, "The task is commonly framed"

**ROLE:** Reframes anomaly detection as learning normal behavior rather than cataloging faults, then introduces multimodal operating regimes.

**DIAGNOSIS:** This paragraph contains the paper's real problem formulation, but it is buried behind the phrase "The task is commonly framed as characterising the anomalies," which sounds like a literature argument rather than an engineering insight. The strongest sentence is "The information lies on the other side of the boundary, in the normal data," but it is surrounded by a long counterfactual about supervised classification and model selection. "A detector that mis-models this structure fails however it thresholds" is punchy but absolute.

**FIX:** Promote this paragraph to the front of the introduction, immediately after the reconstructable-but-improbable hook. Reduce the supervised-learning detour to one sentence. Make the regime structure concrete with operating examples such as startup, steady-state load, transition, and shutdown. End by stating the design consequence: model the union of normal regimes explicitly.

Original: "The information lies on the other side of the boundary, in the normal data, which is abundant and rarely exploited in full."

Tighter: "Because representative faults are scarce, the usable information is concentrated in normal operation, which must be modeled as a set of distinct operating regimes rather than one global distribution."

### 5. Introduction, paragraph 3

**LOCATION:** Section 1, "A fault is anything"

**ROLE:** Identifies the difficult anomaly class: joint-structure violations that remain inside per-channel normal ranges.

**DIAGNOSIS:** This is one of the best conceptual paragraphs in the paper, but it is too short and arrives after two much longer setup paragraphs. "A fault is anything that leaves this union" is broader than needed for the story and sounds definitional. "The interesting ones" is informal and undersells the operational significance. The paragraph also stops one step before the key distinction between reachability and probability.

**FIX:** Make this the opening conceptual paragraph or the second paragraph. Replace "interesting" with "operationally difficult" or "stealthy." Add the reachability-versus-probability sentence here. This paragraph should become the thesis the reader remembers.

Original: "These faults are invisible to any per-channel range rule, and, as we show, to a reconstruction residual as well."

Tighter: "These faults can evade both per-channel range checks and reconstruction error because the issue is not whether the state can be reproduced, but whether normal operation is likely to occupy it."

### 6. Introduction, paragraph 4

**LOCATION:** Section 1, "Two problems make progress"

**ROLE:** Introduces two barriers: misleading evaluation and models that do not match multimodal normal operation.

**DIAGNOSIS:** Two different arguments are compressed into one paragraph. The evaluation critique is about whether progress can be measured; the modeling critique is about why methods fail. Each deserves its own paragraph because each drives a separate contribution. Parenthetical phrases such as "(a single blob under-fits the regime structure)" and "(the trust-eroding false positive)" add rhetorical color but make the paragraph feel argumentative. The sentence becomes a list of failure modes rather than a causal story.

**FIX:** Split this into two paragraphs. First: why normal CPS structure breaks global and reconstruction-based scoring. Second: why standard evaluation hides that failure because easy anomalies dominate and point adjustment inflates scores. In that order, the evaluation protocol becomes a necessary test of the mechanism rather than a separate crusade.

Original: "Two problems make progress on this task hard to measure."

Tighter: "Two obstacles obscure this failure mode: many detectors model normal operation too coarsely, and common benchmarks reward them mostly on anomalies that simple marginal rules already detect."

### 7. Introduction, paragraph 5

**LOCATION:** Section 1, "This paper makes three contributions"

**ROLE:** States the three contributions and previews the main result.

**DIAGNOSIS:** The contribution paragraph is overloaded with architecture terms, parenthetical section references, scoring-head names, calibration details, baseline names, and result qualifications. A non-ML engineer must decode VaDE, latent mixture density, nearest-component likelihood, auto-gating, point-wise metrics, and train-normal-only calibration before reaching the claim. The three contributions are not parallel in form: one is a conceptual assumption set, one is a model implementation, and one is an evaluation protocol plus result.

**FIX:** Make each contribution answer one question. First: What structure characterizes normal CPS operation? Second: How should anomalies be scored when reconstruction can succeed on faults? Third: How should the claim be tested without easy anomalies hiding the signal? Put architecture names after the conceptual verbs, not before them.

Original: "We build a detector, one concrete realization of these assumptions, whose whole design is faithfully modeling that normal law: a jointly learned latent plus explicit Gaussian-mixture regime clustering (VaDE), scored in the latent by a mixture density head and a rare-regime-safe nearest-component likelihood."

Tighter: "We realize these assumptions with a jointly learned regime-aware latent model and score anomalies by latent typicality, using reconstruction only when a normal-only calibration test shows that it adds stable information."

### 8. Introduction, paragraph 6

**LOCATION:** Section 1, "We are explicit about scope"

**ROLE:** States study boundaries and metric choices.

**DIAGNOSIS:** This is process narration rather than story. "We are explicit about scope" and "disclosed in §5" sound defensive. The paragraph interrupts the introduction immediately after the contribution climax. The content is important, but most of it belongs in Experimental Methodology or Limitations. The sentence "we treat the difficult subset, not the headline, as the discriminative test" is conceptually important and should be framed positively, not as a disclaimer.

**FIX:** Keep one sentence in the introduction about the difficult subset, because it defines what the paper considers a meaningful test. Move seed, threshold, and scope details to Sections 5 and 7.

Original: "We are explicit about scope."

Tighter: "Because aggregate benchmark scores are dominated by easy marginal excursions, the central test is performance on anomalies that a simple per-channel detector cannot separate."

### 9. Related work, opening paragraph

**LOCATION:** Section 2, "Anomaly detection is the"

**ROLE:** Places the work in anomaly detection, time-series modeling, and CPS fault detection.

**DIAGNOSIS:** This is a textbook definition paragraph. It repeats motivation already established in the introduction and delays the literature gap. The final sentence, "LatAD sits squarely in this line," is generic positioning and also introduces a name that is not consistently used as the headline model elsewhere.

**FIX:** Replace the generic definition with a two-axis map of the literature: evaluation reliability and scoring mechanism. State immediately where the paper differs. Use one model name consistently throughout the manuscript.

Original: "Anomaly detection is the unsupervised or semi-supervised task of flagging departures from a model of normal behaviour."

Tighter: "Prior work relevant to this study falls into two connected lines: how multivariate time-series anomalies are scored, and whether benchmark protocols reveal the faults those scores are meant to detect."

### 10. Related work 2.1, paragraph 1

**LOCATION:** Section 2.1, "A line of critical work"

**ROLE:** Establishes that point adjustment and benchmark triviality can create misleading impressions of progress.

**DIAGNOSIS:** At roughly a quarter-page, this is a citation cascade. It gives each paper its own result, then adds benchmark critique, then dataset dependence. The reader has to remember many names while the manuscript's own story temporarily disappears. The phrase "illusion of progress" is inherited from cited work, but repeated emphasis on field failure can make the paper sound prosecutorial.

**FIX:** Split or compress. One paragraph should make a single point: raw evaluation matters because point adjustment can reward weak scores. A second, shorter paragraph should make the second point: many benchmark anomalies are marginally easy, so aggregate scores may not test cross-channel structure. End with one sentence explaining why this paper stratifies difficulty.

Original: "The critique now spans metrics and benchmarks alike."

Tighter: "Together, these studies show two distinct evaluation problems: score inflation from point adjustment and weak discrimination when benchmark anomalies are already separable by simple marginal rules."

### 11. Related work 2.1, paragraph 2

**LOCATION:** Section 2.1, "A parallel critique targets"

**ROLE:** Establishes the reconstruction failure mode and connects it to broader criticism of deep anomaly detection.

**DIAGNOSIS:** This paragraph contains the literature foundation for the paper's central mechanism, yet it is presented as "a parallel critique" after the evaluation survey. The first two sentences are far more important than the later position-paper catalog. The final sentence packs two papers, several claims, and a field-level judgment into one long structure.

**FIX:** Move the reconstruction argument before the metric critique, or at least make it the first subsection of Related Work. Use the prior studies only to establish that reconstruction error is not guaranteed to increase for anomalies. Then state the unresolved CPS-specific question: what if abnormal states are reconstructable but low probability under the normal regime structure?

Original: "This motivates scoring in a clustered latent rather than through a reconstruction residual."

Tighter: "These results motivate the question tested here: can a regime-aware latent density identify abnormal CPS states that a decoder reconstructs accurately?"

### 12. Related work 2.1, paragraph 3

**LOCATION:** Section 2.1, "LatAD adopts this critique"

**ROLE:** Bridges the literature critique to the paper's evaluation protocol.

**DIAGNOSIS:** The bridge is useful, but it repeats material already stated in the introduction and later in Section 5. Phrases such as "adopts this critique" and "same raw metrics" emphasize procedural fairness more than scientific logic.

**FIX:** Compress to one forward-looking sentence that links the literature gap to the experimental design.

Original: "LatAD adopts this critique as its evaluation protocol."

Tighter: "Accordingly, the experiments use raw metrics and separate marginally easy anomalies from the subset that actually tests multivariate structure."

### 13. Related work 2.2, paragraph

**LOCATION:** Section 2.2, "Our representation is Variational"

**ROLE:** Positions VaDE relative to DAGMM and names supporting components and baselines.

**DIAGNOSIS:** The paragraph reads like a parts inventory. VaDE, DAGMM, Ledoit-Wolf, Isolation Forest, an autoencoder, and the difficulty detector all appear in one block. The reader is told what is borrowed but not why VaDE is conceptually suited to the CPS regime story. The baseline list belongs in Methods, not Related Work.

**FIX:** Lead with the conceptual fit: VaDE jointly discovers a low-dimensional representation and mixture components, which maps naturally to hidden operating regimes. Then distinguish DAGMM in one sentence. Move covariance-estimator and baseline details to Section 4 or 5.

Original: "Our representation is Variational Deep Embedding (VaDE), a VAE with a Gaussian-mixture latent prior that learns representation and clusters jointly."

Tighter: "VaDE is useful here because it learns the representation and its mixture components jointly, allowing hidden CPS operating regimes to emerge in the same latent space used for anomaly scoring."

### 14. Related work 2.3, paragraph 1

**LOCATION:** Section 2.3, "USAD couples two"

**ROLE:** Positions the work against modern deep multivariate anomaly detectors.

**DIAGNOSIS:** The paragraph becomes a mini-survey of USAD, TranAD, GDN, SensitiveHUE, and CATCH, then predicts that newer reconstruction-based methods will miss the paper's difficult faults. The prediction is rhetorically stronger than necessary and invites a reviewer to demand experiments the paper does not provide. The compute-budget explanation for GDN is a Methods detail. "Deep SOTA" is also specialist shorthand.

**FIX:** Organize by scoring principle, not model chronology. Say that several influential methods still rely on reconstruction or prediction error, while GDN explicitly models channel relationships. Explain that the present study tests representative methods under one raw protocol. Mention unevaluated newer methods as scope, not as predicted failures.

Original: "Both remain reconstruction-based, so the mechanism analysis of §7 predicts they too will miss the reconstructable-but-improbable joint faults that define our difficult subset."

Tighter: "Because these methods still rely substantially on reconstruction, the failure mode studied here remains relevant to them; a direct raw-metric comparison is left for future work."

### 15. Related work 2.3, paragraph 2

**LOCATION:** Section 2.3, "Beyond public benchmarks"

**ROLE:** Connects benchmark anomaly detection to real condition monitoring and fault prediction.

**DIAGNOSIS:** This is the manuscript's main impact bridge to predictive maintenance, but it arrives late in Related Work and is only one sentence plus citations. "LatAD generalises this applied line" is vague and stronger than the paragraph demonstrates. The operational implication is not developed: learning normal regimes can support condition monitoring when labeled faults are scarce.

**FIX:** Move a version of this paragraph to the Introduction, after the problem statement. In Discussion, return to the same bridge with restrained language about deployment. If digital-twin relevance is desired, this is where to explain that the learned normal-regime distribution can be interpreted as a data-driven model of admissible operating behavior, without claiming a full digital twin.

Original: "LatAD generalises this applied line to unsupervised, multimodal-latent anomaly detection."

Tighter: "The same normal-only regime modeling is directly relevant to condition monitoring, where labeled failure examples are scarce but continuous multivariate telemetry is abundant."

### 16. Related work 2.4, paragraph

**LOCATION:** Section 2.4, "We evaluate on three"

**ROLE:** Introduces WADI, SWaT, and HAI and explains their relevance.

**DIAGNOSIS:** Dataset descriptions are duplicated with Section 5.1. Related Work should explain why these benchmarks matter historically or methodologically, not repeat channel counts and testbed details. The BIC statement points forward to Section 3 and mixes empirical characterization into literature positioning.

**FIX:** Merge most of this into Section 5.1. If Section 2.4 remains, keep two sentences: these are standard multivariate CPS benchmarks and they are central to the evaluation debate. Save channel counts, attack details, and multimodality evidence for Methods or Results.

Original: "We evaluate on three real CPS attack testbeds."

Tighter: "WADI, SWaT, and HAI are widely used multivariate CPS testbeds and therefore provide a useful setting for testing whether raw, difficulty-stratified evaluation changes the apparent ranking of detectors."

### 17. Section 3, opening paragraph

**LOCATION:** Section 3, "We model the normal law"

**ROLE:** Formalizes normal CPS operation as a mixture over reachable operating regimes and defines MIIM.

**DIAGNOSIS:** The section opens with an equation before the engineering intuition has been restated. "Normal law" and "bounded, oriented, curved patch of observation space" are compact ML language but not immediately transparent to a controls engineer. MIIM is then unpacked letter by letter after the equation, making the acronym feel like a label imposed on the math rather than a summary of a physical picture.

**FIX:** Start with one plain-language sentence: a plant spends time in multiple operating regimes, each occupying a constrained region of sensor-actuator space, with very unequal dwell times. Then present the mixture equation as a compact mathematical representation of that picture. Define MIIM after the intuition.

Original: "We model the normal law of a CPS as a mixture over reachable operating regimes."

Tighter: "Normal CPS operation is better pictured as a collection of reachable operating regimes than as one cloud of sensor values; each regime occupies a constrained region, and the system visits some regimes far more often than others."

### 18. Table 1 caption

**LOCATION:** Section 3, "Table 1. The ten CPS"

**ROLE:** Explains what the assumptions table contains and distinguishes instantaneous from trajectory assumptions.

**DIAGNOSIS:** The caption is clear, but it implies all ten assumptions play an equivalent role in the reported detector even though the manuscript later emphasizes that A9 and A10 are not exercised in the current experiments. That becomes a narrative expectation problem even if it is scientifically qualified later.

**FIX:** Make the distinction visible in the caption so readers understand the paper's empirical center of gravity before reading ten rows.

Original: "A1–A8 shape the instantaneous window; A9–A10 shape the trajectory."

Tighter: "A1–A8 motivate components used in the reported detector; A9–A10 describe temporal structure reserved for future trajectory-aware extensions."

### 19. Section 3, paragraph 2

**LOCATION:** Section 3, "Two standing qualifiers apply"

**ROLE:** Adds stationarity and sample-size qualifiers, then claims empirical support for MIIM structure.

**DIAGNOSIS:** This paragraph mixes assumptions, exploratory empirical results, and method justification. The reader has not yet reached the dataset methodology, yet is given BIC and silhouette values. "This is precisely the regime in which..." jumps from descriptive statistics to a strong method claim, making the paragraph sound like results inside the conceptual section.

**FIX:** Keep only the standing assumptions here. Move BIC, cluster-size, and silhouette evidence to a short dataset-characterization subsection in Section 5 or to Results. Then refer back: "These measurements support the regime assumptions used in Section 3." This will give the paper a cleaner progression from premise to method to evidence.

Original: "This is precisely the regime in which a jointly learned latent, scored in the latent by a mixture density, pays off."

Tighter: "The empirical extent to which these assumptions hold in WADI, HAI, and SWaT is characterized in the experimental section before the detector results are interpreted."

### 20. Section 4, opening paragraph

**LOCATION:** Section 4, "The main model is"

**ROLE:** Names the detector and divides it into representation and scoring stages.

**DIAGNOSIS:** The model name, "VaDE-hard+resid(auto)," is an internal configuration label, not a reader-friendly scientific object. It makes the method feel like an ablation configuration rather than a coherent detector. "A stack of heads" is ML jargon and "We describe each in turn" is process narration. The conceptual reason for the pipeline is absent.

**FIX:** Give the model one stable human-readable name, then summarize it in three verbs: learn regimes, score latent typicality, optionally add residual information when normal-only validation supports it. The reader should understand the causal design before seeing component labels.

Original: "It has a representation stage ... and a scoring stage ... We describe each in turn."

Tighter: "The detector first learns a regime-structured latent representation of normal windows, then scores how typical each new window is within that latent structure; reconstruction contributes only when a held-out-normal check indicates that its residual is stable."

### 21. Section 4, paragraph 2

**LOCATION:** Section 4, "The reported detector operationalizes"

**ROLE:** Maps assumptions A1–A8 to active model components and separates them from A9–A10.

**DIAGNOSIS:** The paragraph asks the reader to hold eight assumption labels, five scoring components, and three activation rules in working memory. It is correct as a crosswalk but ineffective as prose. The future-work discussion of A9–A10 arrives before the core method has been explained. "So an inapplicable head cannot backfire" is informal and competitive.

**FIX:** Replace the assumption-by-assumption prose list with one short roadmap plus a compact mapping table or the existing Table 1 references. Move A9–A10 entirely to Discussion. Keep the method story centered on two ideas: represent regimes and score probability.

Original: "The reported detector operationalizes assumptions A1–A8."

Tighter: "The implementation uses the instantaneous MIIM assumptions in two stages: regime-aware representation learning and normal-only anomaly scoring."

### 22. Section 4, paragraph 3

**LOCATION:** Section 4, "One realization of the assumptions"

**ROLE:** Argues that MIIM is a portable conceptual contribution and VaDE is only one implementation.

**DIAGNOSIS:** This paragraph is defensive and arrives at the wrong time. Before the reader has seen the method, the manuscript explains all the other methods it could have used. That weakens commitment to the presented design and creates a second design-space story. The alternative estimators are revisited again in Discussion, producing substantial repetition.

**FIX:** Move this argument to Discussion after the evidence. In Method, keep one sentence: "The implementation below is one concrete realization of the MIIM assumptions." After the results, discuss which elements are portable and which are implementation-specific.

Original: "The detector described below is one concrete realization, reported in full so that each of its choices is auditable; but each choice is a point in a larger design space that the same assumptions admit."

Tighter: "The detector below is one concrete realization of the MIIM assumptions; Section 7 discusses which design choices can be replaced without changing the underlying normal-regime view."

### 23. Section 4.1, paragraph 1

**LOCATION:** Section 4.1, "VaDE is a variational"

**ROLE:** Explains VaDE and defines the latent mixture model and training objective.

**DIAGNOSIS:** The first sentence is accessible, but the paragraph immediately becomes equation-dense. For a controls engineer, the important intuition is that each latent Gaussian is intended to represent a region of normal operating behavior, while the encoder maps high-dimensional windows into a lower-dimensional regime space. That interpretation is not stated plainly before the notation.

**FIX:** Add two intuition sentences before the equation. Separate architecture definition from objective definition. Use the equation after the reader knows what each variable means physically.

Original: "VaDE is a variational autoencoder whose latent prior is a Gaussian mixture, so the representation and the operating-regime clusters are learned together rather than one after the other."

Tighter: "VaDE maps each sensor window into a low-dimensional latent space while simultaneously learning a mixture of normal regions in that space. In this paper, those mixture regions serve as data-driven approximations to operating regimes."

### 24. Section 4.1, paragraph 2

**LOCATION:** Section 4.1, "where γc are the responsibilities"

**ROLE:** Explains the posterior responsibilities, anti-collapse measures, and training sequence.

**DIAGNOSIS:** The paragraph contains three stabilization techniques and the full training schedule in one long sentence. "Before the prior is pulled around" is colloquial. "Refined, not destroyed" is vivid but not journal tone. A reader unfamiliar with VaDE may also not know why component collapse matters operationally.

**FIX:** Use a short introductory sentence explaining the failure mode, then present the three measures as a numbered sentence or compact list. End with the training sequence. This is one place where controlled enumeration improves readability.

Original: "Three standard measures prevent the well-known cluster collapse in which many components are abandoned during joint training."

Tighter: "Joint mixture training can abandon components, which is undesirable when rare operating regimes must remain represented. We therefore use three stabilization measures: KL warm-up, a variance floor with collapse regularization, and a lower learning rate for mixture parameters."

### 25. Section 4.2, paragraph 1

**LOCATION:** Section 4.2, "The natural VaDE anomaly"

**ROLE:** Introduces the central scoring decision to demote reconstruction and points to Table 2.

**DIAGNOSIS:** This is a high-value paragraph and should feel like the conceptual hinge of the Method. "This is exactly wrong" is rhetorically sharp but slightly adversarial. More importantly, the paragraph assumes the reader already understands why a correlation break can reconstruct well. The reachability-versus-probability language still has not appeared.

**FIX:** Make this the explicit design hinge: reconstruction answers a different question from the one the detector needs. Then use Table 2 as evidence for the design choice.

Original: "On the difficult faults ... this is exactly wrong."

Tighter: "For the difficult faults of interest, reconstruction answers the wrong question: a decoder may reproduce the window accurately even when that window is improbable under normal regime occupancy."

### 26. Table 2 caption

**LOCATION:** Section 4.2, "Table 2. Difficult-subset AUROC"

**ROLE:** Interprets the reconstruction and latent score terms.

**DIAGNOSIS:** The caption does too much argumentative work. "Drags the summed joint score down" is conversational, and the final sentence points to a later ablation. A table caption should let the reader decode the table and state the central pattern, not narrate the paper's whole scoring argument.

**FIX:** Shorten to the comparison and the interpretation. Put the multi-seed caveat in the surrounding paragraph.

Original: "The reconstruction residual is near chance on WADI ... and drags the summed joint score down."

Tighter: "On the difficult subset, the latent term is more informative than the reconstruction residual on both reported datasets; combining the two without gating can reduce discrimination."

### 27. Section 4.2, paragraph 2

**LOCATION:** Section 4.2, "The mechanism is that"

**ROLE:** Explains why reconstruction can fail and how the score is redesigned.

**DIAGNOSIS:** This is the method's conceptual center, but the best language is still missing. "Demoting reconstruction from the default score is a universal difficult-subset improvement" is broader than needed and sounds like a general theorem. "Provably generalises" is also too strong rhetorically for a held-out-normal gate description. The sentence about combined and easy columns interrupts the mechanism.

**FIX:** State the mechanism in reachability-versus-probability terms, then state the design consequence. Move dataset-specific score effects to Results. Replace "provably" with a direct description of the validation rule.

Original: "The mechanism is that a flexible decoder reconstructs the fault faithfully."

Tighter: "A flexible decoder can reproduce a fault because reconstruction tests representational reachability, not normal-state probability. The base detector therefore scores latent typicality first and adds residual information only when it remains stable on held-out normal data."

### 28. Section 4.3, opening paragraph

**LOCATION:** Section 4.3, "Let z = μ(x)"

**ROLE:** Defines the base score as two standardized latent components.

**DIAGNOSIS:** This paragraph is too terse to orient a non-ML reader. The phrase "two latent heads" appears before the heads have been given conceptual roles. "Operating scale is set by normal" is useful but abstract.

**FIX:** Name the two questions before the formulas: "How dense is this point under normal latent occupancy?" and "How far is it from the nearest learned regime component?" Then explain that both are normalized using training-normal data.

Original: "The base score sums two latent heads."

Tighter: "The base score combines two normality tests in latent space: global latent density and distance to the nearest learned regime component."

### 29. Section 4.3, density-head paragraph

**LOCATION:** Section 4.3, "Because each regime has"

**ROLE:** Introduces the fine-grained latent-density score.

**DIAGNOSIS:** "Parametric KDE for non-Gaussian pockets" and "high-K diagonal Gaussian mixture" are ML-centric labels. The controls reader needs the physical reason: a regime is not uniformly occupied, so even inside a broad operating region some sensor combinations are much more typical than others. The paragraph begins with that idea but moves to implementation too quickly.

**FIX:** Keep the physical intuition in the first two sentences and then present the estimator. Consider renaming the subheading to "Fine-grained latent density" and moving "parametric KDE" into a parenthetical implementation note.

Original: "Because each regime has a non-uniform interior with thin fringes, one Gaussian per regime is too coarse."

Tighter: "Even within one operating regime, the plant spends much more time in some states than others, so a single Gaussian can blur meaningful low-density pockets."

### 30. Section 4.3, nearest-component paragraph 1

**LOCATION:** Section 4.3, "Under heavy-tailed imbalance"

**ROLE:** Motivates a score that does not penalize valid rare regimes simply because they are rarely occupied.

**DIAGNOSIS:** The engineering intuition is good, but "low-π regime" requires the reader to remember mixture weights. The important distinction is between rarity of a regime and abnormality within that regime.

**FIX:** State that distinction before the notation. This is one of the strongest practical ideas in the score and deserves plain language.

Original: "Under heavy-tailed imbalance a valid point in a rare, low-π regime must not be flagged solely for being rare."

Tighter: "A rare operating regime is not a fault simply because the plant visits it infrequently, so the score must separate low occupancy from poor fit to the regime itself."

### 31. Section 4.3, nearest-component paragraph 2

**LOCATION:** Section 4.3, "Using the maximum over"

**ROLE:** Explains why nearest-component likelihood complements the mixture-density score and defines the base score.

**DIAGNOSIS:** This repeats the rationale from the preceding paragraph and uses informal phrasing such as "washes out." The score equation is the new information; the conceptual explanation can be merged into paragraph 30.

**FIX:** Merge paragraphs 30 and 31. Use one sentence to contrast mixture-weighted rarity with within-regime fit, then give the base-score equation.

Original: "Using the maximum over components rather than the Bayesian mixture sum keeps a rare-regime signal that the mixture sum washes out."

Tighter: "Taking the best-fitting component preserves evidence that a point is normal within a rarely occupied regime instead of penalizing that regime through its small mixture weight."

### 32. Section 4.3, residual-head paragraph 1

**LOCATION:** Section 4.3, "On some datasets the"

**ROLE:** Introduces the optional reconstruction-residual score and its regime-conditioned whitening.

**DIAGNOSIS:** This is technically dense and arrives after the paper has emphasized demoting reconstruction, so the reader may perceive a contradiction. The paragraph needs to state clearly that reconstruction is not rejected; it is treated as complementary evidence when stable. "Responsibility-weighted whitened residual" is an implementation phrase, not an intuitive description.

**FIX:** Start with the reconciliation: some faults do produce structured reconstruction residuals, so residual information can be useful, but it should not dominate by default. Then explain whitening as scale and correlation correction.

Original: "On some datasets the fault does surface in reconstruction."

Tighter: "Reconstruction is not useless: some faults leave a stable residual pattern, so we retain residual evidence as an optional secondary signal rather than the primary anomaly score."

### 33. Section 4.3, residual-head paragraph 2

**LOCATION:** Section 4.3, "The head is auto-gated"

**ROLE:** Defines the held-out-normal gate and reports which datasets activate it.

**DIAGNOSIS:** The method rule and the observed dataset outcomes are mixed together. The parenthetical "(the residual overfits / drifts)" is informal and the slash construction is visually awkward. "Recovers exactly this pattern" sounds retrospective and result-driven even though the point is normal-only calibration.

**FIX:** Separate rule from outcome. In Method, describe the gate using held-out normal only. In Results, report which datasets activated it and what happened. This improves trust and reduces the appearance of tuning around test behavior.

Original: "Empirically the ratio is 5.24 on WADI (off) and 1.17 on HAI (on): the residual carries the fault on HAI but is dead on WADI, and the gate recovers exactly this pattern from train-normal alone."

Tighter: "The gate is determined from held-out normal data only; the dataset-specific activation pattern is reported with the experimental results."

### 34. Section 4.3, basin-head paragraph 1

**LOCATION:** Section 4.3, "Between-regime pockets are"

**ROLE:** Introduces a perturbation-based agreement score intended to protect against between-regime false negatives.

**DIAGNOSIS:** The intuition is interesting, but this component is not active on the evaluated datasets. Giving it a full subheading, detailed procedure, and equation makes the main detector look more complicated than the evidence requires. It also competes with the much stronger density story.

**FIX:** Move the basin mechanism to an optional-extension subsection or supplement unless there is a benchmark where it materially contributes. In the main paper, one sentence is enough to state that the architecture includes a normal-only gate for overlapping-regime cases.

Original: "Between-regime pockets (A3) are the dangerous false negative on datasets whose regimes overlap."

Tighter: "For datasets with strongly overlapping learned regimes, an optional perturbation-agreement term can identify points that sit unstably between regime basins."

### 35. Section 4.3, basin-head paragraph 2

**LOCATION:** Section 4.3, "where ρ is the"

**ROLE:** Defines the gate and states that the basin head is inactive on all three benchmarks.

**DIAGNOSIS:** Most of this paragraph explains why a just-described component does not contribute to the reported results. That is a major narrative cost. Phrases such as "exact no-op," "not part of the empirical contribution," and "left to future work" turn the reader's attention from the tested method to an untested extension.

**FIX:** Remove the inactive mechanism from the headline model description. If it must remain for completeness, put the equation and gate in an appendix and state in the main text that the gate remained inactive on all three benchmarks. The core paper should not spend more prose on an inactive head than on the central density mechanism.

Original: "The basin head is a design provision for datasets with heavily overlapping regimes; it stays inactive on all three benchmarks here and is therefore not part of the empirical contribution."

Tighter: "The overlap gate remained inactive on all three evaluated benchmarks, so the reported empirical gains come from the latent score and, where activated, the residual term."

### 36. Section 4.3, auto-gating note

**LOCATION:** Section 4.3 note, "Auto-gating summary"

**ROLE:** Summarizes which score components are always active and which are gated.

**DIAGNOSIS:** The summary is useful because the preceding text is complex, but needing this note is itself evidence that the method presentation has too many moving parts. Number references "(i–iii), (iv), (v)" force backtracking.

**FIX:** Keep the note, but recast it around conceptual roles rather than component numbers. If the basin head is moved out of the main method, the note can become one sentence.

Original: "Base heads (i–iii) run on every dataset."

Tighter: "Every dataset uses the two latent normality scores; reconstruction residuals are added only when held-out normal data show that the residual model remains stable."

### 37. Section 5.1, opening paragraph

**LOCATION:** Section 5.1, "We use three real"

**ROLE:** Introduces the datasets.

**DIAGNOSIS:** "And no synthetic data in the results tables" sounds defensive because the manuscript has not raised synthetic data as a concern. It spends the only prose sentence in the subsection on what is not used rather than why these datasets are appropriate.

**FIX:** State dataset selection in terms of CPS diversity and benchmark relevance.

Original: "We use three real CPS datasets and no synthetic data in the results tables."

Tighter: "We evaluate on three widely used CPS attack testbeds spanning water distribution, water treatment, and a multi-process industrial control system."

### 38. Section 5.1, WADI bullet

**LOCATION:** Section 5.1, "WADI, a water-distribution testbed"

**ROLE:** Describes WADI and the preprocessing choices needed for it.

**DIAGNOSIS:** The bullet mixes dataset identity, interpretation of near-constant channels, anomalous raw values, clipping, and downsampling. "Attack carriers, not noise" sounds argumentative. For a narrative reader, this level of preprocessing detail interrupts the dataset overview.

**FIX:** Keep identity, channel count, and one sentence on preprocessing in the main text. Move the rationale for clipping specific anomalies and file recalibration details to a reproducibility appendix or preprocessing table.

Original: "Roughly a quarter of channels are near-constant in normal operation; these are attack carriers, not noise, and are kept."

Tighter: "Near-constant channels are retained because they can change during attacks and therefore remain diagnostically relevant."

### 39. Section 5.1, HAI bullet

**LOCATION:** Section 5.1, "HAI, a hardware-in-the-loop"

**ROLE:** Identifies the HAI benchmark and notes that clipping is not applied.

**DIAGNOSIS:** This bullet is much shorter and structurally different from WADI and SWaT, which makes the dataset presentation feel uneven. "No clipping" is preprocessing detail without the parallel context used elsewhere.

**FIX:** Use a consistent template for all datasets: system, channels, evaluation stream, and material preprocessing difference.

Original: "No clipping (its large excursions are real attack signal)."

Tighter: "No clipping is applied because the large excursions in HAI are retained as part of the measured attack behavior."

### 40. Section 5.1, SWaT bullet

**LOCATION:** Section 5.1, "SWaT, the Secure Water"

**ROLE:** Identifies SWaT and notes downsampling and preprocessing.

**DIAGNOSIS:** The bullet is clear but continues the uneven pattern. It names the institution, number of stages, attack status, clipping, and downsampling without explaining which facts matter for the paper's regime story.

**FIX:** Make the three bullets parallel and reserve low-level transformations for a common preprocessing paragraph.

Original: "SWaT is downsampled 10× on both streams, with the train and test horizons matched."

Tighter: "SWaT is downsampled using the same train/test treatment applied to WADI so that the windowing protocol is consistent across datasets."

### 41. Section 5.2, paragraph

**LOCATION:** Section 5.2, "Each raw channel is"

**ROLE:** Defines standardization, windowing, feature extraction, labeling, and the omission of explicit temporal features.

**DIAGNOSIS:** Five methodological decisions are packed into one paragraph. The six statistics are listed, but their engineering meaning is only partly glossed. Standardization appears twice. The final future-work sentence about A9 repeats earlier caveats and dilutes the methods description.

**FIX:** Split into two paragraphs: one for window construction and features, one for labels and normalization. Explain the six statistics in terms of level, variability, extrema, trend, and spread. Move the A9 future-work sentence to Discussion.

Original: "Each window is summarised by six per-channel statistics ... concatenated into one flat feature vector."

Tighter: "Each window is reduced to six per-channel summaries capturing level, variability, extrema, net change, and spread, then concatenated into a single multichannel feature vector."

### 42. Section 5.3, paragraph

**LOCATION:** Section 5.3, "A trivial detector scores"

**ROLE:** Defines easy and difficult anomaly subsets and explains why the difficult subset is the discriminative test.

**DIAGNOSIS:** This is central to the paper's contribution, but the paragraph starts with formula mechanics rather than the purpose of the split. "Trivial" is repeated and can sound dismissive of a useful baseline. "Hand-picked statistic" is defensive. The key sentence is the last one.

**FIX:** Start with the evaluative question: does the model detect anomalies that cannot be separated by any single-channel excursion? Then define the simple marginal detector and threshold. Use "simple marginal baseline" instead of "trivial detector" in prose.

Original: "A trivial detector scores each window by the maximum absolute standardised per-channel window mean."

Tighter: "To separate obvious marginal excursions from anomalies that require multivariate structure, we first apply a simple per-channel range baseline to every window."

### 43. Section 5.4, opening paragraph

**LOCATION:** Section 5.4, "We report two raw"

**ROLE:** Introduces AUROC and F1.

**DIAGNOSIS:** "And are explicit about their comparability" is process language. The paragraph can simply state the metric hierarchy.

**FIX:** Use a direct priority statement.

Original: "We report two raw, point-wise metrics and are explicit about their comparability across subsets."

Tighter: "AUROC is the primary metric because it is threshold-free and comparable across the three difficulty subsets; raw F1 is reported as a secondary thresholded measure."

### 44. Section 5.4, AUROC bullet

**LOCATION:** Section 5.4, "AUROC is prevalence-independent"

**ROLE:** Explains why AUROC is primary.

**DIAGNOSIS:** The bullet is clear and appropriately short. The only readability issue is "prevalence-independent," which is familiar to ML readers but can be stated more plainly for engineers.

**FIX:** Keep the content, simplify the first clause.

Original: "AUROC is prevalence-independent, hence directly comparable across Easy/Difficult/All."

Tighter: "AUROC does not depend on the anomaly fraction in the same way as F1, so it is the primary metric for comparing Easy, Difficult, and All subsets."

### 45. Section 5.4, F1 bullet

**LOCATION:** Section 5.4, "F1 is the best"

**ROLE:** Defines the reported F1 and limits how it should be compared.

**DIAGNOSIS:** "An oracle, field-standard threshold, disclosed plainly" reads like self-defense. The key fact is simple: the threshold is selected on the test scores and therefore F1 is descriptive rather than deployment-ready.

**FIX:** State the limitation once, neutrally.

Original: "F1 is the best raw point-wise F1 over score thresholds (an oracle, field-standard threshold, disclosed plainly)."

Tighter: "F1 is reported at the threshold that maximizes raw point-wise F1 on the test set, so it is a descriptive secondary metric rather than a deployable threshold estimate."

### 46. Section 5.4, paragraph 2

**LOCATION:** Section 5.4, "All model fitting and"

**ROLE:** States that training and score calibration use only normal training data, while F1 alone uses a test-swept threshold.

**DIAGNOSIS:** The content is important, but the list of every calibrated component is too detailed and repeats earlier normal-only claims. "AUROC is leak-free" and "which we flag" are defensive phrases.

**FIX:** State the rule at the level that matters to trust, then point to component definitions if needed.

Original: "AUROC is leak-free; only the F1 threshold uses the test-swept oracle, which we flag."

Tighter: "Training and score calibration use training-normal data only; the sole test-dependent operation is selection of the reported best-F1 threshold."

### 47. Section 5.5, paragraph

**LOCATION:** Section 5.5, "Classical baselines are Isolation"

**ROLE:** Defines baseline families, explains raw-metric recomputation, mentions GDN scope, and repeats the point-adjustment rationale.

**DIAGNOSIS:** This paragraph does too many jobs. It mixes baseline definitions, implementation alignment, score-grid conversion, compute limitations, metric philosophy, and the difficulty baseline. It also repeats the point-adjustment critique for a third time. The main experimental fairness principle gets lost inside operational details.

**FIX:** Split into two paragraphs. First: baseline families and why each is informative. Second: common evaluation treatment and score alignment. Put the GDN compute limitation in a footnote or compact sentence. Do not re-argue point adjustment here.

Original: "For modern SOTA we re-run USAD and TranAD through the TranAD evaluation harness over five seeds and score them with the same raw point-wise metrics."

Tighter: "All learned baselines are re-evaluated on the same window grid with the same raw metrics, so differences reflect the scoring methods rather than point-adjustment conventions."

### 48. Section 6, opening paragraph

**LOCATION:** Section 6, "Table 3 reports AUROC"

**ROLE:** Orients the reader to the main results table.

**DIAGNOSIS:** This is mechanically correct but does not tell the reader what question to ask when reading the table. Because the paper argues that overall scores can be misleading, the result section should explicitly direct attention first to the difficult subset.

**FIX:** Lead with the interpretive order: difficult subset first, overall second, easy subset as context.

Original: "Table 3 reports AUROC and best raw point-wise F1 for every method on the three datasets and the three subsets."

Tighter: "Table 3 reports all three subsets, but the primary comparison is the Difficult column because it removes anomalies already separated by the simple marginal baseline."

### 49. Table 3 caption

**LOCATION:** Section 6, "Table 3. Per-dataset, per-subset"

**ROLE:** Defines the table's metrics, averaging, tie convention, model labels, and baseline meanings.

**DIAGNOSIS:** The caption is overloaded. It includes metric caveats, five-seed policy, a GDN compute note, tie formatting, the meaning of "Ours," the role of the max|z| baseline, and the definition of LinRes. The reader must process a paragraph before reaching the table.

**FIX:** Keep only what is necessary to decode the table. Put methodological caveats in Section 5 and method definitions in a footnote or legend.

Original: "The trivial max|z| row is the univariate range rule that defines the difficulty split: strong on Easy and near chance on Difficult, which is what makes the split meaningful."

Tighter: "The max|z| row is the simple marginal baseline used to define Easy and Difficult subsets; LinRes is the linear cross-channel baseline."

### 50. Section 6, "Overall detection" paragraph

**LOCATION:** Section 6, "Overall detection. The headline"

**ROLE:** States overall AUROC performance across datasets.

**DIAGNOSIS:** The paragraph is concise, but it uses scoreboard language such as "clears every baseline," "strongest," and "leads." More importantly, the paper has already argued that the All column is not the discriminative test, so opening the narrative with the overall scoreboard creates tension with the paper's own evaluation philosophy.

**FIX:** Put the difficult-subset paragraph first. Then summarize overall performance as a consistency check, not the central evidence.

Original: "The detector thus leads the combined column on all three datasets."

Tighter: "Overall AUROC remains competitive or best across all three datasets, showing that the gain on difficult faults does not require sacrificing aggregate detection performance."

### 51. Section 6, "Difficult, joint-structure faults" paragraph

**LOCATION:** Section 6, "Difficult, joint-structure faults"

**ROLE:** Presents the main discriminative evidence, significance result, WADI tie, SWaT ceiling case, and mechanism interpretation.

**DIAGNOSIS:** This is the most important results paragraph, but it is too long and contains four separate stories. "This is where the design earns its keep" is colloquial. "Collapse" is repeated as competitive rhetoric. Statistical detail, episode counts, WADI interpretation, SWaT ceiling behavior, and mechanism are all compressed into one block. The reader can miss the clean result pattern: HAI demonstrates the advantage; WADI shows parity with a strong classical baseline while deep reconstruction methods fail; SWaT is too easy to discriminate methods.

**FIX:** Split into three paragraphs by dataset role, not just dataset name. First: HAI as the decisive discriminative case. Second: WADI as a tie at the top but strong rejection of reconstruction scoring. Third: SWaT as a ceiling case that validates the paper's warning about benchmark difficulty. End with one synthesis sentence connecting performance to reconstructable-but-improbable faults.

Original: "This is where the design earns its keep."

Tighter: "The difficult subset provides the clearest test of the proposed mechanism because it removes anomalies that a simple marginal detector already separates."

### 52. Section 6, "Trivially separable anomalies" paragraph

**LOCATION:** Section 6, "Trivially separable anomalies"

**ROLE:** Explains why easy-subset rankings are not scientifically informative.

**DIAGNOSIS:** The paragraph is clear but largely repeats the definition of Easy from Section 5.3. As a standalone paragraph, it interrupts the more important result flow.

**FIX:** Merge this point into the difficulty-split validation paragraph or use it as a one-sentence transition after the main difficult-subset results.

Original: "On the easy subset every method is strong and the ranking is uninformative, by construction."

Tighter: "Easy-subset rankings add little discrimination because these anomalies are defined by their separability under a simple marginal rule."

### 53. Section 6, "Validity of the difficulty split" paragraph

**LOCATION:** Section 6, "Validity of the difficulty split"

**ROLE:** Shows that the simple marginal baseline behaves differently on Easy and Difficult subsets and identifies SWaT as a ceiling case.

**DIAGNOSIS:** This is useful interpretation, but it repeats both Section 5.3 and paragraph 52. The core insight is not the exact baseline scores; it is that WADI and HAI actually contain a difficult slice where marginal detection fails, while SWaT mostly does not.

**FIX:** Combine paragraphs 52 and 53 into one compact result: the split succeeds as a discriminative filter on WADI and HAI, but not on SWaT. That is a more memorable benchmark story.

Original: "The difficult subset is precisely the anomalies this univariate range rule cannot see, so a method that scores well there is detecting joint structure the rule misses."

Tighter: "On WADI and HAI, the split isolates anomalies that the marginal baseline cannot separate, making those subsets the meaningful tests of multivariate structure; SWaT remains close to a ceiling case."

### 54. Section 6, "Robustness to a stronger difficulty definition" paragraph

**LOCATION:** Section 6, "Robustness to a stronger"

**ROLE:** Tests whether the HAI result depends on using window mean as the difficulty filter.

**DIAGNOSIS:** The logic is important, but the paragraph carries a full definition of the stronger split plus multiple dataset results. "Hand-picked" and "artefact" are defensive. The result should be framed as a robustness question, not a response to an accusation.

**FIX:** State the concern in one sentence, define the stronger filter compactly, then give the main conclusion. Detailed values can stay in a table or parenthetical.

Original: "A window could in principle be 'difficult' yet be separable by a different univariate statistic."

Tighter: "To test whether the split depends on window mean, we repeat it using the maximum over all six per-channel summary statistics."

### 55. Section 6, "The double-hard subset" paragraph

**LOCATION:** Section 6, "The double-hard subset"

**ROLE:** Introduces a stricter stress test that removes anomalies separable by either a marginal or simple linear cross-channel baseline.

**DIAGNOSIS:** This paragraph is too long for a secondary robustness analysis and spends substantial space explaining what the subset is not. The phrase "intersect the failures" is compact but not immediately intuitive. The paragraph also combines subset construction, validity caveat, all three results, significance, sample-size caveats, and interpretation. That makes the main paper feel defensive and benchmark-centric.

**FIX:** Put this in a clearly labeled robustness subsection after the main results. Use two paragraphs: definition and purpose, then results and limits. State the caveat once. Most importantly, distinguish "defined independently of our model" from "unbiased for comparing the baselines used to define it."

Original: "This subset is not an unbiased benchmark against the two detectors that define it, which fail on it by construction; it isolates the anomalies that remain once both marginal and simple linear cross-channel structure is removed."

Tighter: "This stress test is defined without reference to our detector, but it is not suitable for ranking the two baselines used to construct it; its purpose is to isolate anomalies that evade both marginal and simple linear cross-channel rules."

### 56. Table 4 caption

**LOCATION:** Section 6, "Table 4. Double-hard subset"

**ROLE:** Defines the double-hard subset and summarizes the reliability of the three dataset results.

**DIAGNOSIS:** The caption contains a wording tension: it calls the subset a "baseline-independent stress test" and immediately says it is not an unbiased comparison against the two filters that define it. A careful reviewer may stop on this phrasing even if the intended logic is sound. The caption is also long.

**FIX:** Use "independent of the proposed detector" rather than "baseline-independent." Then keep the sample-size caveat short.

Original: "It is a baseline-independent stress test, not an unbiased comparison against those two filters."

Tighter: "The subset is constructed independently of the proposed detector; because two baselines define the filter, their scores on this subset are descriptive rather than an unbiased comparison."

### 57. Section 6, "Deep SOTA under raw metrics" paragraph

**LOCATION:** Section 6, "Deep SOTA under raw"

**ROLE:** Synthesizes the failure of reconstruction-oriented deep baselines on the difficult WADI and HAI subsets.

**DIAGNOSIS:** This repeats the strongest numbers from paragraph 51 and uses charged language: "collapse" and "illusion of success." The paper's point is stronger when it sounds analytical rather than victorious. The real message is not that deep models are bad; it is that easy anomalies and point adjustment can mask weak signal on reconstructable-but-improbable faults.

**FIX:** Make this the synthesis sentence at the end of the main results subsection and use neutral mechanism language.

Original: "This is the concrete face of the 'illusion of success': models that look strong under point-adjusted F1 have little difficult-fault signal once the metric is raw."

Tighter: "The ranking changes sharply on raw difficult-subset AUROC, indicating that strong aggregate or point-adjusted results can coexist with weak discrimination on the faults that motivate this study."

### 58. Figure 1 caption

**LOCATION:** Section 6, "Figure 1. Difficult-subset AUROC"

**ROLE:** Explains the plotted difficult-subset comparison and interprets the dataset-specific margins.

**DIAGNOSIS:** The caption is effectively a mini Discussion paragraph. It reports significance, ties, deep-model margins, fault morphology, and a causal explanation. Captions should be self-contained, but this much interpretation duplicates surrounding text.

**FIX:** Keep plot definition, five-seed whiskers, and one sentence naming the main pattern. Put the reconstructable-versus-large-deviation explanation in the body.

Original: "The margin over the deep detectors ... is largest on HAI and WADI, the two benchmarks whose difficult faults are low-magnitude improbable combinations that reconstruction error cannot separate from normal."

Tighter: "The largest separation from reconstruction-based deep baselines occurs on WADI and HAI; SWaT is near the ceiling for all methods."

### 59. Discussion, "Executive summary" paragraph

**LOCATION:** Section 7, "Executive summary. Under a"

**ROLE:** Restates the main benchmark findings and attributes the difficult-subset advantage to latent regime scoring.

**DIAGNOSIS:** Discussion begins by repeating the abstract and results scoreboard almost verbatim. "Under a fair protocol" is self-evaluative. This delays the most insightful paragraph in the manuscript, which follows immediately. A strong Discussion should begin by explaining what the results mean, not by restating every winning number.

**FIX:** Delete or reduce this paragraph to two sentences. Open Discussion with the reachability-versus-probability mechanism from the next paragraph. Then use HAI and WADI as evidence for that interpretation.

Original: "Under a fair protocol ... the VaDE-based latent-plus-clustering detector attains the best overall AUROC on every dataset and is strongest on the difficult subset over five seeds."

Tighter: "The main result is not the aggregate ranking but the change in behavior on difficult faults: regime-aware latent density retains signal where reconstruction-based scores often do not."

### 60. Discussion, "Why reconstruction fails and density wins" paragraph

**LOCATION:** Section 7, "Why reconstruction fails and"

**ROLE:** States the reachability-versus-probability mechanism and supports it with WADI observations and a controlled between-regime probe.

**DIAGNOSIS:** This is the manuscript's strongest paragraph and the one that should define the paper. It arrives too late. The distinction "reconstruction measures reachability, not probability" is memorable, intuitive, and accessible to a controls engineer. The controlled probe is also a strong storytelling device because it makes the mechanism visible. The only issue is density: the paragraph includes manifold residual, reconstruction AUROC, synthetic averaging, latent-density AUROC, design implications, A4, A6, and gating in one block.

**FIX:** Move the conceptual first half into the Introduction and a shortened version into the Abstract. Consider moving the controlled probe into Results as an early mechanism experiment, potentially before the benchmark scoreboard. In Discussion, retain a concise interpretation. If a conceptual figure can be added, show two normal regimes with a low-density gap point that the decoder can reproduce.

Original: "Reconstruction measures reachability (can the decoder reproduce this vector?), not probability (is this vector likely under normal?)."

Tighter: "Reconstruction asks whether the model can reproduce a state; anomaly detection asks whether normal operation is likely to occupy that state."

### 61. Table 5 caption

**LOCATION:** Section 7, "Table 5. Difficult-subset AUROC"

**ROLE:** Defines the score-head ablation across datasets.

**DIAGNOSIS:** The caption is mostly clear, but it reintroduces the name "LatAD" while the headline model elsewhere is "VaDE-hard+resid(auto)." That naming inconsistency makes the paper feel like it is switching between a method name and an experiment label. "Independent five-seed run" and "within seed variation" add process detail.

**FIX:** Choose one model name and use it everywhere. Define each row in plain language. Keep the caption focused on the ablation.

Original: "LatAD is the reported model (base+resid+basin, auto-gated)."

Tighter: "The final row is the reported detector: latent density plus nearest-component score, with optional residual and overlap terms controlled by normal-only gates."

### 62. Discussion, Table 5 interpretation paragraph

**LOCATION:** Section 7, "Table 5 decomposes the"

**ROLE:** Shows which score component carries performance on each dataset and supports the mechanism.

**DIAGNOSIS:** The paragraph is informative but number-heavy. The central pattern is simple: latent density dominates on WADI and HAI; residual is useful on SWaT and adds value on HAI; the gate keeps it off on WADI. "Provably generalises" again sounds stronger than necessary. The narrative would be clearer if the first sentence stated the pattern before the exact values.

**FIX:** Lead with the mechanism conclusion, then cite representative numbers. Avoid repeating every table entry.

Original: "The decomposition supports the mechanism at the head level: the latent density is what carries the joint-structure benchmarks, and reconstruction contributes only where it provably generalises."

Tighter: "The ablation supports the mechanism at the component level: latent density carries most of the difficult-fault signal on WADI and HAI, while the residual term helps only on datasets where its normal-only stability check keeps it active."

### 63. Discussion, "Auto-gating adapts one architecture" paragraph

**LOCATION:** Section 7, "Auto-gating adapts one architecture"

**ROLE:** Interprets the gate as a way to adapt the same architecture across datasets.

**DIAGNOSIS:** Much of this repeats Section 4.3 and the previous paragraph. "The winning signal is dataset-dependent" sounds competitive. The real operational point is valuable: the system can decide from normal data whether reconstruction residuals are stable enough to use.

**FIX:** Recast this as an implication for deployment rather than a second method description. One sentence on normal-only adaptation is enough.

Original: "The winning signal is dataset-dependent, so rather than hand-select we gate two optional heads by purely train-normal signals."

Tighter: "Because the useful anomaly evidence differs across systems, the detector uses normal-only checks to decide whether residual information should supplement latent typicality."

### 64. Discussion, "The residual frontier on WADI" paragraph

**LOCATION:** Section 7, "The residual frontier on"

**ROLE:** Identifies a class of WADI windows that remain difficult and motivates richer temporal or physics-informed information.

**DIAGNOSIS:** This is a good limitation-to-future-work bridge, but it overstates what can be concluded from the evaluated methods. "No snapshot detector ... separates these" and "they are an intrinsic property of the data" sound universal. The reader only needs the narrower observation that all evaluated snapshot methods struggle on these windows.

**FIX:** Keep the failure description, narrow the claim, and tie it directly to the next temporal-assumptions paragraph.

Original: "No snapshot detector, ours or the deep baselines, separates these at a low false-alarm budget; they are an intrinsic property of the data rather than of any one detector."

Tighter: "These onset and offset windows remain difficult for all evaluated snapshot-based detectors, suggesting that additional temporal or process-model information may be needed."

### 65. Discussion, "Trajectory assumptions A9–A10" paragraph

**LOCATION:** Section 7, "Trajectory assumptions A9–A10"

**ROLE:** Explains that A9 and A10 are not exercised and motivates future temporal extensions.

**DIAGNOSIS:** The paragraph is clear but generic. After paragraph 64, there is an opportunity to make the future direction concrete: temporal context is not just an item on the assumption list; it addresses the observed residual frontier where instantaneous state is insufficient.

**FIX:** Connect the future work explicitly to the limitation just described and to predictive maintenance, where degradation trajectories often matter more than isolated windows.

Original: "Extending it with multiscale-temporal features (A9) and history-conditioned, path-dependent scoring (A10) is the natural next step."

Tighter: "The unresolved onset and offset cases motivate the next extension directly: incorporate multiscale dynamics and path history so that a state can be judged by how the system arrived there, not only by its instantaneous window."

### 66. Discussion, "Alternative realizations of the assumptions" paragraph

**LOCATION:** Section 7, "Alternative realizations of the"

**ROLE:** Proposes other clustering and grouped-density implementations of the MIIM assumptions.

**DIAGNOSIS:** This begins a second paper. It introduces correlation-structure clustering, hierarchies, grouped density models, sparse versus dense fault aggregation, and a new design space after the manuscript has already established its main contribution. The material is intellectually interesting but weakens closure. It also shifts the contribution from "we tested this mechanism" toward "here is a broader research program," which can make reviewers question where the validated paper ends.

**FIX:** Reduce this to one short future-work paragraph or move it to a separate perspective section or supplement. Use the reclaimed space for operational implications: how a regime-aware normality model could fit into condition monitoring, predictive maintenance, or a data-driven digital-twin pipeline.

Original: "Because the MIIM assumptions fix the structure to model rather than the estimator, they open a family of detectors beyond the one reported here, and two routes are especially promising."

Tighter: "The MIIM view is not tied to VaDE; future work can test alternative regime partitions and lower-dimensional density models without changing the central principle of scoring probability within normal operating structure."

### 67. Discussion, "A multiscale regime-density decomposition" paragraph 1

**LOCATION:** Section 7, "A multiscale regime-density decomposition"

**ROLE:** Formalizes a new grouped-density decomposition as future work.

**DIAGNOSIS:** A new mathematical framework appears very late in the Discussion, after the main method and results are complete. This is structurally disruptive. Readers may wonder whether this is part of the contribution, an untested extension, or the intended next paper. The equation increases cognitive load without supporting the current results.

**FIX:** Remove this equation from the main manuscript. If the authors want to preserve the idea, place it in a short "Future directions" appendix or develop it in a separate paper. In the main Discussion, one sentence from paragraph 66 is enough.

Original: "The route above can be stated precisely."

Tighter: "A promising extension is to estimate normality within correlated channel groups rather than only in one global latent space."

### 68. Discussion, "A detector scores a window" paragraph

**LOCATION:** Section 7, "A detector scores a"

**ROLE:** Extends the speculative grouped-density method with weighting and sparse-versus-dense aggregation choices.

**DIAGNOSIS:** This paragraph compounds the structural problem in paragraph 67. It introduces a new operator, group weights, factorized likelihood, maximum aggregation, a union-bound interpretation, Higher Criticism, sparsity adaptation, and a relationship to the current model. For the target engineer reader, this is an abrupt move into a new statistical design space. It also leaves the paper ending on speculation rather than the demonstrated insight.

**FIX:** Cut from the main paper. The Discussion should move from mechanism to practical meaning to limitations to focused future work. If retained elsewhere, give it its own formal section with motivation and evaluation in a future manuscript.

Original: "The combination ⊕ encodes the fault's expected extent."

Tighter: "Future grouped-density models could adapt how evidence is combined according to whether faults are expected to affect one subsystem or many."

### 69. Discussion, "Limitations" paragraph

**LOCATION:** Section 7, "Limitations. Three boundary conditions"

**ROLE:** Summarizes dataset, model, representation, threshold, and difficulty-split limitations.

**DIAGNOSIS:** The paragraph is compressed and partially repeats caveats already stated in the Introduction and Methods. It mixes generalization limits with metric disclosure. "The split is not knife-edge" is argumentative. The final sentence again contrasts raw and point-adjusted leaderboard values, repeating a point already established several times.

**FIX:** Organize limitations by what they constrain: external validity, representation scope, and evaluation interpretation. State each once and explain what it means for the claim. Remove repeated defense of raw metrics.

Original: "Three boundary conditions apply: three datasets, one trained model per dataset, and a single instantaneous representation."

Tighter: "The conclusions are bounded in three ways: they come from three public CPS testbeds, they evaluate one regime-modeling implementation, and they use window-level rather than trajectory-level representations."

### 70. Conclusion, paragraph

**LOCATION:** Section 8, "We restated the structural"

**ROLE:** Recaps the assumptions, model, evaluation, results, mechanism, and future work.

**DIAGNOSIS:** At more than 250 words, the conclusion is a second abstract. It repeats multiple numerical result qualifications, the double-hard subset, the mechanism, and future-work items. The strongest conceptual sentence, about reachability versus probability, is buried in the middle. The ending is about extensions rather than the contribution the reader should remember.

**FIX:** Cut to roughly 120 to 150 words. Use three moves: problem insight, demonstrated solution, implication. Mention HAI as the strongest discriminative case without reproducing all confidence-interval detail. End with the conceptual takeaway for CPS monitoring.

Original: "The mechanism is that reconstruction scores reachability while a density in a clustered latent scores probability, and the CPS faults that matter are reconstructable but improbable."

Tighter: "The central lesson is that reconstruction and normality are different: a CPS state may be easy to reproduce yet unlikely under normal operating regimes, so anomaly scoring should estimate typicality rather than treat reconstruction error as the default proxy for normality."

### 71. Back matter, Author Contributions

**LOCATION:** Back matter, "Author Contributions: Conceptualization, methodology"

**ROLE:** Provides the journal-required contribution statement.

**DIAGNOSIS:** This is standard administrative prose and does not affect the scientific narrative. "The authors contributed equally to all aspects of the work" is clear.

**FIX:** No narrative rewrite is needed unless the journal requires CRediT formatting or role-specific distinctions.

Original: "The authors contributed equally to all aspects of the work."

Tighter: "Both authors contributed equally to all reported aspects of the work."

### 72. Back matter, Funding

**LOCATION:** Back matter, "Funding: This research received"

**ROLE:** States funding status.

**DIAGNOSIS:** Standard and clear.

**FIX:** Leave unchanged unless journal style requires a different formula.

Original: "This research received no external funding."

Tighter: "This research received no external funding."

### 73. Back matter, IRB statement

**LOCATION:** Back matter, "Institutional Review Board Statement"

**ROLE:** States ethics-review applicability.

**DIAGNOSIS:** Standard and clear.

**FIX:** Leave unchanged.

Original: "Not applicable."

Tighter: "Not applicable."

### 74. Back matter, Informed Consent

**LOCATION:** Back matter, "Informed Consent Statement"

**ROLE:** States consent applicability.

**DIAGNOSIS:** Standard and clear.

**FIX:** Leave unchanged.

Original: "Not applicable."

Tighter: "Not applicable."

### 75. Back matter, Data Availability

**LOCATION:** Back matter, "Data Availability Statement: This"

**ROLE:** Explains dataset access, code/checkpoint availability, and what the checkpoints contain.

**DIAGNOSIS:** The statement is complete but somewhat long for back matter. The sentence about each checkpoint containing learned parameters and a fixed seed is useful reproducibility information, but it could be separated for scanability. This is not a narrative problem.

**FIX:** Use three short sentences: public datasets, code/checkpoint archive, checkpoint contents and reproducibility.

Original: "The trained model checkpoints (one per dataset) and the source code supporting the reported results are openly archived on Zenodo at https://doi.org/10.5281/zenodo.21821524."

Tighter: "Source code and trained checkpoints for the reported experiments are archived on Zenodo at the stated DOI."

### 76. Back matter, Conflicts of Interest

**LOCATION:** Back matter, "Conflicts of Interest: The"

**ROLE:** States conflict status.

**DIAGNOSIS:** Standard and clear.

**FIX:** Leave unchanged.

Original: "The authors declare no conflicts of interest."

Tighter: "The authors declare no conflicts of interest."

## Story spine

**One-sentence STORY SPINE:** A CPS fault can be reconstructable yet improbable, so anomaly detection should model the probability structure of normal operating regimes rather than treat reconstruction error as a proxy for normality, and it should be evaluated on faults that simple marginal rules cannot already detect.

### The three places where the spine is most lost

1. **Related Work, especially Sections 2.2 to 2.4.** The paper shifts from the reachability-versus-probability problem into a catalog of models, estimators, datasets, and citations. The reader stops following a scientific question and starts following literature categories.

2. **Method opening and the scoring-stack presentation.** The central idea becomes obscured by assumption labels, configuration names, five heads, gating details, and an inactive basin mechanism. The method should read as a consequence of the insight: learn regimes, score typicality, add residual evidence only when normal-only validation supports it.

3. **Discussion paragraphs on alternative realizations and multiscale grouped density.** These paragraphs begin a new theoretical research program after the main story should be closing. They make the paper feel less finished and displace space that should be used for operational implications, limitations, and the predictive-maintenance bridge.

## Rewritten abstract

A cyber-physical system can enter a faulty state that is individually plausible on every sensor and still be unlikely under normal joint operation. Such faults expose a weakness of reconstruction-based anomaly detection: a flexible decoder may reproduce an abnormal sensor combination accurately, because reconstruction measures whether a state can be represented, not whether normal operation is likely to occupy it. We model normal CPS behavior instead as a collection of hidden, unequally occupied operating regimes, summarized by ten structural assumptions termed Massive, Implicit, Imbalanced Multimodality (MIIM). A jointly learned latent representation and Gaussian-mixture regime model are used to score latent typicality, with reconstruction residuals added only when a held-out-normal stability check supports them. To test the mechanism rather than benchmark trivialities, we use raw point-wise metrics and separate marginally easy anomalies from difficult anomalies that a simple per-channel detector cannot identify. Across WADI, HAI, and SWaT, the detector achieves the strongest overall AUROC in the reported comparison. The clearest discriminative case is HAI, where it reaches 0.811 AUROC on difficult faults and significantly exceeds the next-best baseline, while reconstruction-based deep detectors fall near chance. WADI shows the same reconstruction failure mode but a tie with the strongest classical baseline, while SWaT is largely a ceiling case. The results support a simple principle for CPS anomaly detection: model the probability structure of normal operating regimes, not reconstruction error alone.

## Rewritten opening paragraph

A cyber-physical system can be faulty even when every sensor value looks individually normal and a deep autoencoder reconstructs the full sensor vector accurately. The reason is that reconstruction and normality answer different questions. Reconstruction asks whether a model can reproduce a state; anomaly detection asks whether normal operation is likely to occupy that state. In a tightly coupled plant, vehicle, or machine, normal behavior is distributed across many operating regimes created by physical constraints, controller logic, setpoints, and operating conditions. A correlation break or a state that falls between two legitimate regimes can therefore be reconstructable but improbable. This distinction matters directly for condition monitoring and predictive maintenance: the most valuable early warnings may be states that remain reachable within familiar sensor ranges but violate the joint probability structure of normal operation.

## Single highest-impact structural change

**Make the reachability-versus-probability distinction the organizing axis of the entire paper, and move its controlled mechanism evidence forward.** Specifically, open the Introduction with the rewritten mechanism paragraph, place the normal-regime argument immediately after it, and present the WADI between-regime probe early in Results as the mechanism test before the large benchmark table. Then let MIIM, VaDE, the latent-density score, difficulty stratification, and the benchmark results appear as consequences of that single question. This change would make the manuscript feel like one paper with one discovery rather than a framework paper, a detector paper, an evaluation-critique paper, and a future-method paper stacked together.

## Tone and style checklist

### Apologetic or process-centered phrasing to remove or reduce

The manuscript repeatedly narrates how carefully the authors are behaving rather than simply stating the scientific rule. Examples include "We are explicit about scope," "disclosed plainly," "This is deliberate," "which we flag," "reported in full so that each of its choices is auditable," and repeated statements that a component is or is not part of the empirical contribution. These phrases can sound defensive even when the underlying methodological choice is good.

Preferred pattern: state the rule directly. For example, replace "We are explicit about scope" with the scope statement itself; replace "which we flag" with the exact limitation; replace "This is deliberate" with the scientific reason for the choice.

### Competitive or overheated phrasing to soften

Several phrases make the manuscript sound like it is arguing with the field rather than explaining a mechanism: "exactly wrong," "earns its keep," "collapse," "illusion of success," "winning signal," "clears every baseline," and repeated uses of "fair protocol." The strongest version of this paper is calm and mechanistic. Use measured descriptions such as "falls near chance on the difficult subset," "the ranking changes under raw metrics," and "the difficult subset is the primary discriminative comparison."

The title phrase "All You Need" belongs in the same category. It is catchy, but it raises the burden of proof and conflicts rhetorically with the optional residual head and unmodeled temporal assumptions.

### Hedging and qualification

The manuscript is not excessively hedged overall. The larger issue is uneven certainty. Some statements are very strong, such as "universal difficult-subset improvement," "provably generalises," "No snapshot detector ... separates these," and predictions that unevaluated reconstruction models will also miss the same faults. Elsewhere the paper uses softer phrases such as "in principle," "could in principle," "approximately," and "we regard." Standardize the confidence level.

Preferred pattern: make claims exactly as broad as the evaluated evidence. Use "across the evaluated datasets," "passes the held-out-normal stability check," "all evaluated snapshot detectors," and "the same failure mode may remain relevant."

### Jargon accessibility for CPS and controls readers

The manuscript should gloss the following terms at first use in plain engineering language: latent space, mixture responsibility, negative log-likelihood, high-K mixture, parametric KDE, whitening, BIC, silhouette, oracle threshold, episode-block bootstrap, and z-normalization. The prose is strongest whenever it translates these terms into operating behavior, such as rare regimes, between-regime pockets, and correlated sensor combinations.

A useful editorial rule is: **every ML term should be followed once by the physical or operational question it answers.** For example, "nearest-component likelihood" answers "does this window fit at least one legitimate operating regime?" "Latent density" answers "how typical is this combination under normal operation?" "Whitened residual" answers "is there structured reconstruction error after accounting for channel scale and correlation?"

### Digital-twin and predictive-maintenance impact framing

The manuscript already has a legitimate predictive-maintenance bridge because it learns normal operating structure from telemetry without requiring representative fault labels. That should be stated earlier and returned to in Discussion.

Digital-twin language requires more restraint. The current manuscript does not build or validate a digital twin. A safe and useful framing is: "The learned regime distribution can serve as a probabilistic normal-behavior layer within a data-driven digital twin or condition-monitoring system." Avoid stronger formulations such as claiming a new digital-twin architecture unless the paper adds explicit twin state, synchronization, prediction, or decision-support content.

### Hard-rule check

Visible manuscript prose contains **no em dash** and **no double hyphen**. Preserve that.

The five prohibited candor markers specified for this review are **absent** from the visible manuscript prose. Preserve that.

The manuscript does use en dashes in ranges and compound contrasts, which is consistent with the stated rule.

## Recommended narrative order after revision

A strong final ordering would be:

1. **Introduction:** reconstructable-but-improbable fault, reachability versus probability, normal as operating regimes, predictive-maintenance relevance, then evaluation problem and contributions.
2. **Related Work:** reconstruction-based scoring failure, regime and latent models, then evaluation critique. Keep datasets out of Related Work.
3. **MIIM model of normal operation:** plain-language regime picture first, assumptions second, with A1–A8 visually separated from A9–A10.
4. **Method:** one conceptual diagram or short roadmap, then representation, latent scoring, optional residual gate. Move the inactive basin mechanism to supplementary material unless it contributes empirically.
5. **Experimental Methodology:** datasets, windows, difficulty split, metrics, baselines.
6. **Results:** mechanism probe first, difficult-subset comparison second, overall results third, robustness analyses last.
7. **Discussion:** mechanism interpretation, operational implication for condition monitoring and predictive maintenance, limits, then concise future work.
8. **Conclusion:** one conceptual takeaway, one evidence sentence, one implication sentence.

That ordering would make the paper read as a causal argument: **a known scoring proxy fails for a specific class of CPS faults; the structure of normal operation explains why; a regime-aware probability score addresses the failure; a difficulty-aware protocol exposes the difference.**
