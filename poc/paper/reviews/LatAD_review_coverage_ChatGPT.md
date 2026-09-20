(A) Reviewer/editor coverage
Editor

1. Validate the currently unused assumptions (A3/A9/A10): PARTIALLY ADDRESSED.
The revision substantially restructures this issue. Table 1 now explicitly separates A1–A7 as “realized and validated” from A8–A10 as “specified, not realized by the reported window-only model.” A9 is examined through measured multiscale time constants and an additional longer-window experiment; A10 is tested with history-conditioned scorers and a synthetic positive control, with the conclusion that path dependence is absent in these benchmarks. However, the originally inactive basin-agreement mechanism is not actually activated on real data; it has effectively been moved/reframed under the revised A8 discussion. So the scope is now much more honest, but the missing assumptions are not fully experimentally validated.

2. Direct experimental support for the rare-regime-safe nearest-component likelihood: NOT ADDRESSED.
Section 4.3 explains the rationale, and the scoring-head ablation reports nearest-component NLL as one component, but the specific experiment requested is absent. There is no controlled comparison across regime-imbalance ratios, no direct mixture NLL vs nearest-component NLL vs another imbalance-aware method, and no false-positive analysis specifically on rare normal regimes. This remains the clearest unresolved request.

3. Training and inference cost for edge deployment: ADDRESSED.
Appendix B / Table B1 reports parameter count, training wall-clock time, GPU latency per window, peak GPU memory, CPU latency, and RAM for LatAD-global and LatAD-community, together with baseline GPU measurements. Section 4.4 also clarifies that communities use separate models.

4. Use a time-aware statistical test: ADDRESSED.
Section 5.4 now uses a moving-block bootstrap for normal windows and episode-level bootstrap for attacks, with 2000 resamples, percentile 95% confidence intervals, and an explicit one-sided p-value definition. Sections 6–7 use this procedure for the reported significance tests.

5. Add ablations isolating the source of the gain: ADDRESSED.
Appendix A now contains multiple targeted ablations: Table A1 factorization/aggregation, A2 representation, A3 reconstruction versus latent scoring, A5 cross-channel versus channel-independent density, and A6 individual score heads. This is a substantial response.

6. Clearer abstract and introduction: ADDRESSED.
The abstract now explains LatAD’s actual mechanism: VaDE/GMM latent modeling, probability rather than reconstruction scoring, community factorization, Higher Criticism, difficulty stratification, and drift handling. The Introduction now begins with CPS/IIoT background and moves the explicit contribution statement later.

7. Restructure Related Work: PARTIALLY ADDRESSED.
Most of Reviewer 2’s requested restructuring is done: anomaly-detection definition is moved to the Introduction; several passages describing LatAD itself are deleted; Section 2.3 is rewritten as a literature comparison. However, the reviewer explicitly suggested deleting Section 2.5; it remains as “Benchmark scale.”

8. Add system/application diagram: ADDRESSED.
Figure 1 shows the IIoT pipeline from physical process and PLCs to edge gateway/SCADA historian, LatAD scoring, and operator/security output.

9. Add LatAD workflow figure: ADDRESSED.
The revised paper includes a workflow figure mapping the representation, density/scoring heads, community models, and aggregation.

10. Resolve the z notation conflict: ADDRESSED.
The standardized per-channel value in the evaluation section is renamed, so it no longer conflicts with latent z.

11. Number equations and make typesetting consistent: ADDRESSED.
The revised mathematical expressions are numbered and formatting is much more consistent.

12. Submit a point-by-point response: NOT PRESENT IN THE FOUR FILES.
The Editor explicitly requested one, but among the four supplied files there is no response letter.

Reviewer 1

1. A9/A10 unused; inactive basin-agreement assumption; either test them or limit claims: ADDRESSED, mainly by limiting the claims.
The revised Introduction and Table 1 now say explicitly that the reported model realizes A1–A7, while A8–A10 are not realized by the reported window-only model. Section 8 adds diagnostics for A8–A10. This satisfies the reviewer’s stated fallback option of restricting the contribution to assumptions actually validated. The caveat is that the former inactive basin-agreement issue is reclassified rather than demonstrated experimentally.

2. Rare-regime-safe nearest-component likelihood needs direct support: NOT ADDRESSED.
The requested imbalance-ratio study and rare-normal false-positive analysis are still missing. The individual-head ablation does not establish the specific claimed advantage for rare regimes.

3. Community design may be too expensive; report practical cost: ADDRESSED.
Section 4.4 clarifies that each community has its own VaDE/encoder/mixture/scoring heads. Appendix B, Table B1 quantifies training time, memory and latency.

4. Sliding windows are temporally correlated; use a time-aware test and clarify WADI CI/p-value: ADDRESSED.
Section 5.4 provides a block/episode bootstrap and explicit definitions of CI and p-value.

5. Determine whether improvement comes from cross-channel density modeling or the six-statistic window representation: PARTIALLY ADDRESSED.
Table A2 compares the basic six-statistic window representation with a richer temporal/spectral feature representation. Table A5 compares cross-channel density against an independent per-channel density product. These are useful and directly relevant. However, Reviewer 1 specifically requested an experiment using raw sequential input, and that experiment is still absent.

Reviewer 2

1. Expand abbreviations at first appearance: PARTIALLY ADDRESSED.
CPS, IIoT and SCADA are now expanded appropriately. LatAD itself is still not explicitly expanded or identified as simply a model name, even though the reviewer specifically mentioned it.

2. Abstract focuses too much on deficiencies of prior methods and not enough on LatAD: ADDRESSED.
The revised abstract now spends substantial space on the proposed architecture and its main results.

3. Introduction logic unclear; add CPS background and remove “This paper detects them…” from the first paragraph: ADDRESSED.
That sentence is deleted, CPS/IIoT background is added up front, and the contributions appear later.

4. Clarify what “the task” refers to: ADDRESSED.
The text now explicitly says “Unsupervised anomaly detection for CPS telemetry…”

5. Clarify whether LatAD contributes to modeling, evaluation, or both, and what problem it solves: ADDRESSED.
The revised contribution paragraph separates C1 LatAD detector, C2 detector-agnostic difficulty stratification, and C3 drift-aware anomaly typing.

6. Move anomaly-detection definition to the Introduction: ADDRESSED.

7. Remove detailed descriptions of the authors’ own method from Related Work: ADDRESSED.
The requested self-descriptive passages are removed or substantially rewritten.

8. Rewrite Section 2.3: ADDRESSED.
It is now a real related-work section covering VaDE, DAGMM, Deep SVDD, THOC, OmniAnomaly and related methods, followed by a concise positioning of LatAD.

9. Delete Section 2.5: NOT ADDRESSED LITERALLY.
Section 2.5 remains, although substantially shortened and renamed “Benchmark scale.”

10. Add an IIoT system/application diagram showing where LatAD operates: ADDRESSED.
Figure 1 directly answers this.

11. Explain the relationship between MIIM and LatAD: ADDRESSED.
Table 1 now contains a “Realized by (§)” column linking each assumption to a detector mechanism, and the text explicitly states which assumptions LatAD implements.

12. Improve readability of Section 4 and relate subsections to the LatAD pipeline; add workflow figure: ADDRESSED.
The representation, scoring, and community-factorization stages are now explicitly connected to the LatAD workflow and illustrated.

13. Clarify whether community models share the encoder, and report time/memory/latency: ADDRESSED.
Section 4.4 states that each community has its own encoder, mixture, and scoring heads; Appendix B gives the computational measurements.

14. Resolve the two uses of z: ADDRESSED.

15. Fix inconsistent formatting and add equation numbers: ADDRESSED.

(B) Most important remaining or weak points

1. The nearest-component likelihood claim is still the biggest unresolved issue.
Both the Editor and Reviewer 1 explicitly asked for direct evidence that nearest-component NLL protects rare-but-normal regimes. The revision still does not test the claim that matters: whether rare normal regimes obtain fewer false positives than under mixture NLL. A second-round reviewer can repeat the same objection almost unchanged.

2. The raw-sequence comparison requested by Reviewer 1 is still missing.
The richer temporal/spectral feature ablation is useful, but it is still a handcrafted window representation. Therefore the paper has not fully demonstrated that the advantage comes from cross-channel density modeling rather than from the chosen representation. Any strong statement to that effect should be softened.

3. The assumption renumbering/reframing needs very careful explanation.
The round-1 reviewer referred to an inactive basin-agreement head associated with A3. In the revision, A3 is now “thin fringes,” while the overlap/basin problem is placed under A8 and declared not realized. Scientifically that restructuring may be defensible, but without an explicit old-to-new mapping it can look as though the objection was removed by changing the taxonomy.

4. A9/A10 are discussed more than they are experimentally demonstrated.
Section 8 reports timescale ranges, a longer-window result, A10 conditional scorers, and a synthetic positive control, but these do not have the same transparent table-level reporting as the main ablations. In particular, the claim that A10 is “measured absent” would be stronger with explicit dataset-by-dataset numbers and control results.

5. The time-aware bootstrap may still use blocks that are too short.
The new test is clearly better than ordinary iid resampling, but blocks of roughly 3–4 windows mainly address mechanical sliding-window overlap. The paper itself argues that CPS channels have much longer physical time constants. A reviewer may therefore ask whether the bootstrap really captures the relevant serial dependence.

6. WADI significance claims may be stronger than the effective sample size supports.
The difficult WADI subset contains only a small number of independent attack episodes. Very small p-values against several methods, especially with multiple pairwise comparisons and a single-seed GDN result, may attract scrutiny.

7. Edge-deployment claims should remain modest.
Appendix B now addresses the requested cost question well, but the CPU experiment is a proxy machine rather than a specified industrial gateway, and CPU latency for all competing methods is not given. The paper supports “computationally practical” better than a strong claim of demonstrated deployment on constrained edge hardware.

8. Development-versus-test provenance needs to be completely explicit.
The manuscript says the community construction/aggregation design was developed using these benchmarks while also describing WADI/SWaT as out-of-sample confirmations and stressing no test-set tuning. A reviewer may ask exactly which datasets/results influenced architectural choices and which were genuinely untouched evaluation sets.

9. The headline aggregation choice is not uniformly best in the ablations.
In Table A1, alternative aggregation variants outperform the final configuration on some individual datasets. This is not necessarily a problem if the final architecture was fixed prospectively, but it makes the development-provenance issue above more important.

10. Two easy Reviewer-2 issues remain: explicitly define what “LatAD” stands for, or state that it is the model name, and consider whether Section 2.5 is important enough to retain after the reviewer specifically suggested deleting it.