Inconsistency — the advertised “two-stage” difficulty protocol is not the protocol behind the main Difficult results. The Abstract says, “We evaluate under a two-stage, detector-agnostic difficulty stratification that strips trivially and linearly separable anomalies,” and C2 says the channel rule plus PCA separate easy/linear cases. But §5.3 defines the primary Difficult subset only by the max-absolute standardized channel mean; the PCA filter creates a separate later “double-hard” subset.

IoT2 +1

 This is important because the paper then interprets the main Difficult results as evidence about joint/nonlinear structure. Fix: describe the primary analysis as a one-stage max-mean split, and the PCA-filtered double-hard analysis as a secondary robustness test. Do not say the headline Difficult subset has had linear anomalies removed.

Inconsistency — “factorized density” is mathematically stronger than what the method actually implements. §4.4 explicitly states that communities “are nested subtrees and can overlap” and that their sum is “over-complete pooling rather than an exact factorization,” yet Equation (8) still writes −logp(x)≈∑
G
	​

s
G
	​

(x), and the paper repeatedly calls LatAD a “community-factorized latent-density detector.”

IoT2

 Since s
G
	​

 is a calibrated tail-surprise score and communities overlap, this is not a valid density factorization as written. Fix: call it “community-local density modeling with surprise aggregation” or a “community ensemble,” and rewrite Eq. (8) as a score definition rather than an approximation to −logp(x).

Inconsistency — Equation (7) omits the auto-gate that the text says is essential. §4.3 says the residual is added “where the residual head is gated on,” but Eq. (7), immediately called “the detector's complete window score,” always gives s=s
0
	​

+s
resid
	​

. WADI is explicitly a case where the residual head is off.

IoT2

 Fix: put the gate into the equation, e.g. s=s
0
	​

+gstd(s
resid
	​

), g∈{0,1}, and define the standardization consistently.

Project leftover / inconsistency — the live Appendix still contains a removed “basin head.” Table 1 says A8 is “specified, not realized,” and the revised architecture removes the dedicated overlap/basin mechanism; nevertheless Appendix D says: “The basin head is auto-scaled by a train-normal ambiguity ratio and is a no-op on all three crisp-mode datasets.”

IoT2 +1

 This is a clear residue from an earlier model version. Fix: delete this sentence and any remaining basin-head parameters/references. If the basin head truly exists, then the method, equations and ablations must all be restored consistently.

Inconsistency / project-process residue — “AUROC is leak-free” is too categorical given dataset-specific preprocessing informed by the attack recordings. §5.1 says WADI channel 2B_AIT_002_PV is dropped because it “is rescaled between the normal and attack recordings,” while HAI clipping is avoided because its large excursions “are real attack signal.” Later §5.4 says simply “AUROC is leak-free.”

IoT2 +1

 Even if these choices are legitimate, the wording reads like an internal leakage audit and invites a reviewer to ask whether test/attack information influenced preprocessing. Fix: replace “leak-free” with a precise data-access statement, and state whether each preprocessing rule was fixed from dataset documentation before examining labels/results. Ideally add a sensitivity result retaining the WADI channel.

Inconsistency — Appendix D incorrectly says all detectors use the same features. It states: “All detectors share the same windowing and features.” But §5.5 says USAD, TranAD and GDN produce native per-timestep scores that are subsequently mapped or aggregated to the LatAD evaluation grid; Table D2 even gives GDN its own window length 5.

IoT2 +1

 Fix: say all methods are compared on the same labels/evaluation grid, while LatAD and local feature baselines use the six-statistic windows and the deep sequence baselines retain their native inputs.

Inconsistency — the density-head component count is presented as globally fixed at 80 although community heads use another rule. §5.2 says “the density head is a diagonal mixture of 80 components,” while Appendix D.2 says each community uses min(50, max(12, floor(n_fit/12))).

IoT2 +1

 Fix: state explicitly in §5.2 that the global density head uses M=80, while community experts use the size-dependent rule.

Inconsistency — “ties the linear baseline” is not what the statistics show. The WADI text says “LatAD ties only the linear baseline LinRes (0.750),” but LatAD is 0.771 versus 0.750, with a nonsignificant +0.020 difference and CI crossing zero.

IoT2

 Fix: say “LatAD is not significantly different from LinRes” rather than “ties.” Also replace P=0.000 from 2,000 bootstrap replicates by the appropriate inequality, e.g. p<0.0005.

Inconsistency — Table A2 points to the wrong main-results table. Its caption says the reduced global-density values “are not the headline numbers of the main results table (Table A3).” Table A3 is an appendix decomposition of score terms, not the main results table.

IoT2

 Fix: change “Table A3” to “Table 2.”

Inconsistency — Table A3 says “per dataset” but omits SWaT. The caption reads “single trained model per dataset,” yet the columns contain only WADI and HAI.

IoT2

 Fix: either add SWaT or explicitly say “for WADI and HAI.”

Inconsistency — Appendix B’s caption contradicts its own table. It says “Deep-baseline CPU-edge latency and host memory were not measured,” but the AutoEncoder row reports CPU-edge latency of 0.08/0.07/0.07 ms/window.

IoT2

 Fix: say CPU measurements were unavailable for USAD/TranAD/GDN specifically, not for all deep baselines.

Inconsistency — Appendix C says a failure mode is “absent” although Table C1 reports a nonzero rate. The text says “the … over-coverage miss of the 80-component head are all absent,” but WADI has an isolated-missed rate of 0.13 for the 80-component head.

IoT2

 Fix: use “substantially reduced” or report the actual rates; “absent” is factually too strong.

Language/tone — several claims are written as promotional conclusions rather than neutral scientific statements. Examples include “those detectors collapse to near chance,” “where it matters most,” “headline number,” and later “The mechanism is simple to state.”

IoT2 +1

 Fix: replace the rhetoric with the actual values and uncertainty: e.g. “USAD and TranAD achieve AUROC X and Y on the HAI Difficult subset.” The manuscript will sound substantially more credible.

Language — several technical terms are imprecise. The Abstract says “a fault can read within range on every sensor and reconstruct accurately”; a fault neither “reads” nor reconstructs. Use “a faulty state can remain within every sensor’s normal range and still be reconstructed accurately.”

IoT2

 Also, §4.3 calls the 80-component Gaussian mixture a “parametric kernel-density estimate, KDE.” KDE normally denotes a nonparametric estimator. Fix: call it a “high-resolution Gaussian-mixture density estimator.” Likewise, “nearest-component NLL” is not the mixture NLL once the mixture weights are deliberately omitted; rename it more precisely.

Duplication — the reachability-versus-probability argument is repeated too many times. Variants of “reconstruction measures reachability; detection needs probability” appear in the Abstract, Introduction, §4.2, Discussion and Conclusion; the Conclusion again says “Reconstruction scores reachability; a density in a clustered latent scores probability.”

IoT2 +1

 Fix: retain the conceptual statement once in the Introduction, provide the evidence in §4.2/Results, and use only one short interpretive sentence in the Discussion or Conclusion.

AI-tell — the manuscript is excessively scaffolded by formulaic mini-narratives. Repeated constructions such as “This paper makes three contributions,” “Two standing qualifiers apply throughout,” “The mechanism is simple to state,” “Alternative realizations and validation,” and many bold paragraph-openers give the text a checklist/LLM cadence. C1 itself is an unusually dense inventory of method, motivation, mechanism and outcome in one paragraph.

IoT2

 Fix: reduce the number of bold mini-headings and numbered rhetorical setups. Let Methods state the method and Results state the result without repeatedly telling the reader how to interpret each block.

Project leftovers — the live manuscript explicitly records recomputation/rerun history. §5.5 is titled “Baselines and SOTA re-computation” and says “we re-run USAD and TranAD through the TranAD evaluation harness”; Table A6 calls itself “an independent five-seed run” that “reproduces its Table 2 row within seed variation.”

IoT2 +1

 This reads like a development notebook describing successive reruns rather than a frozen manuscript protocol. Fix: describe one final evaluation protocol in the present tense, give the exact implementation/version/seed policy, and remove “re-computation,” “re-run,” “independent run,” and “reproduces.”

Project leftovers — internal implementation shorthand and tracked/development residue remain in both live and struck text. Table D2 contains “imperial-qore/TranAD harness,” “same harness,” and “harness-default top-k and embedding dimension,” which is not publication-grade reproducibility language.

IoT2

 The struck text also still contains obsolete architecture names such as “basin-agreement head,” “LatAD-global,” “null expert,” and “canonical” split terminology, while the HTML itself contains author-workspace links to “DOCX,” “Cover letter to editor,” and “Journal info (ranking, APC).”

IoT2 +1

 Fix: generate a clean submission version with all struck material physically removed, delete the workspace/navigation elements, replace repository/harness shorthand by a versioned public implementation citation or commit plus explicit parameter values, and run a final search for leak, re-comput, re-run, canonical, basin, null expert, and obsolete model names before submission.