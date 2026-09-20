# Under-claim audit: LatAD manuscript (IoT2.html)

Read-only audit. Scope: does the prose/structure sell genuinely strong, already-verified results as
hard as the evidence supports? No manuscript edits made. Ranked most-under-sold first.

---

## 1. UNDER-CLAIMED / BURIED — SWaT false-alarm collapse under drift typing (79% -> 6.5%)

**Evidence.** Table 4 and the §7.2 text: at the train-calibrated threshold, the normal-window false-alarm
rate on the severely drifting SWaT record falls from 79% (reported detector) to 6.5% (changepoint-typed
detector) once drift is separated from changepoint. This is independently reproduced in
`_diagnostics/drift_changepoint_typing.md` ("The normal-window false-alarm rate at the train-calibrated
threshold falls from 79% to 6.5%") with the same number, confirming it is not a stray draft figure.

**Why it is strong.** A 79% false-alarm rate is operationally useless (an operator would ignore the
alarm entirely); 6.5% is close to a deployable budget. This is the single most dramatic *practical*
number in the whole drift-typing result — more visceral than the AUROC deltas (0.524->0.723,
0.178->0.671) that currently carry the headline, because AUROC is abstract to a non-ML reader while
"79% of alarms were noise, now 6.5% are" is immediately legible to the IIoT-operator audience the paper
explicitly targets (Figure 1, "Implications for IIoT deployment").

**Current placement/framing.** Only in running text of §7.2 (line ~998) and Table 4. Absent from the
abstract, the C3 contribution description, the §6/§8/§9 "what the results show" summaries, and the
conclusion. The abstract states the AUROC recovery (0.723) but not the false-alarm number.

**Verdict: UNDER-CLAIMED.**

**Elevation suggestion.** Add the false-alarm figure next to the AUROC figure wherever the SWaT recovery
is stated: e.g. abstract sentence becomes "...a causal drift-aware anomaly typing brings the difficult
subset to 0.723 and cuts the normal-window false-alarm rate from 79% to 6.5%." Repeat in the Conclusion's
SWaT sentence and in the §8 "What the results show" paragraph. This one number does more to justify
"drift-aware typing" as a first-class contribution (not a rescue) than the AUROC gain alone.

---

## 2. UNDER-CLAIMED / BURIED — Subsystem localization (HAI top-3 68% vs 56% random, p=0.024; WADI directional 2.5x/1.8x)

**Evidence.** Table 5 and the single "Subsystem localization" paragraph in §8 (Discussion): HAI top-1
0.39 vs 0.25 random (P=0.024), top-3 0.68 vs 0.56 random; WADI top-1 0.33 vs 0.13 random (2.5x), top-3
0.56 vs 0.31 (1.8x), directionally consistent but underpowered (9 episodes, P=0.086). Confirmed current
and consistent with `_diagnostics/localization_clean.md`, which independently re-derives the same numbers
from the clean pipeline and explicitly flags WADI's lift as "the largest lifts" among the three datasets
even though it doesn't reach significance at n=9.

**Why it is strong.** This is a real, statistically significant (on HAI) operator-facing capability — not
just a detection score but a *triage* answer ("which of the plant's ~10 subsystems is implicated"),
exactly the kind of output the paper's own Figure 1 deployment diagram promises ("Anomaly alarm + most-
surprised subsystem"). Few anomaly-detection papers demonstrate localization with a random-baseline
comparison and a p-value; this one does, on a real testbed, and it works.

**Current placement/framing.** It is not one of the three stated contributions (C1-C3 in §1 describe
detection, difficulty stratification, and drift typing; localization is not named). It appears once, as
a single paragraph deep in §8 Discussion, after ten other subsections (A8 assumption, residual frontier,
benchmark coverage, trajectory assumptions, alternative realizations, deployment implications). The
abstract does not mention it at all. The conclusion does not mention it at all, despite the conclusion's
closing paragraph explicitly invoking "anomalous behavior is local to a coupled physical subsystem" as a
motivation — the capability that delivers on that motivation (localization) is never named as a
delivered result.

**Verdict: BURIED** (real, significant, on-theme result with zero contribution-list or abstract/conclusion
presence).

**Elevation suggestion.** (a) Add one clause to the abstract: "...and localizes 68% of HAI attacks to a
three-subsystem shortlist against 56% by chance (P=0.024)." (b) Either fold localization into C1 ("...so a
local fault... is concentrated and localized to its subsystem, delivering a three-community operator
shortlist that contains the attacked subsystem in 68% of HAI attacks against 56% random") or add it as an
explicit fourth capability sentence in the contributions paragraph. (c) Give it one sentence in the
Conclusion's closing paragraph, next to the existing "local to a coupled physical subsystem" motivation,
so the claim and its evidence sit together.

---

## 3. UNDER-CLAIMED / BURIED — Training/inference efficiency (47-72x faster than TranAD, sub-30ms latency, <4GB RAM)

**Evidence.** Appendix B / Table B1: LatAD (global density) trains in 3.2-16s vs TranAD's 163-898s (47-72x
faster) and USAD's 95-592s; per-window latency is 0.7-13ms (global) / 17-32ms (full regime-community
stack) on GPU and 0.6-25ms on an 8-vCPU CPU edge proxy; peak host memory 3.5-3.9GB; peak GPU memory 20-29MB
(vs TranAD's 39-63MB). Params are 0.78-1.25M for the full model, smaller than TranAD's 1.27M on two of
three datasets.

**Why it is strong.** The entire manuscript is framed around IIoT edge deployment: the title says
"Industrial-IoT," Figure 1 is an edge/SCADA deployment diagram, and §8 has a dedicated "Implications for
IIoT deployment" paragraph. A 47-72x training-time advantage and single-digit-millisecond, sub-4GB-RAM
inference is a concrete, differentiating deployability claim against the deep SOTA baselines this paper
already beats on accuracy — a genuine "wins on both accuracy and cost" story, which is rare and
compelling.

**Current placement/framing.** One sentence in §8's edge-deployment paragraph pointing to Appendix B;
otherwise confined entirely to Appendix B, after Appendix A's long ablation tables. Not in the abstract,
not in the introduction's motivation (§1 mentions edge/SCADA infrastructure at length but never promises
a cost result), not in the conclusion.

**Verdict: BURIED.**

**Elevation suggestion.** Add one clause to the abstract or the "Implications for IIoT deployment"
paragraph's opening, stated as a headline number rather than an appendix pointer: "LatAD also trains
47-72x faster than TranAD and scores a window in under 30ms on an 8-vCPU CPU edge proxy within 4GB of
host memory (Appendix B), making the accuracy gain compatible with gateway-class hardware." Consider
promoting this to a fourth, brief sentence near C1 ("...a design that is also cheap to train and run at
the edge, Appendix B") so a reader skimming the contributions list sees the deployability claim, not just
the accuracy claim.

---

## 4. UNDER-CLAIMED — WADI community-factorization effect size (0.634 -> 0.795) stated but not framed as the largest single ablation effect

**Evidence.** §8 "Why local density beats global density" states the number once in prose: "factorization
lifts the WADI difficult-subset score (Table A1: 0.634 -> 0.795)." Table A1 in Appendix A carries the full
ablation. This +0.161 AUROC jump from a single design choice (community factorization vs. single global
latent) is larger than any other single-axis ablation delta reported in Table A1/A2/A5/A6.

**Why it is strong.** It is the cleanest mechanistic demonstration in the paper that subsystem
factorization — not the VaDE representation, not the residual head, not the combiner choice — is what
carries the WADI result. The paper already argues this qualitatively ("the community factorization rather
than the single latent supplies the WADI signal," §6) but never states the ablation magnitude as a
headline number outside one appendix-referencing clause.

**Current placement/framing.** Correctly cited once in Discussion, but as a parenthetical clause rather
than a called-out effect size; the full context (that the pure-community score, 0.795, actually exceeds
the fused headline, 0.771, because whole-plant fusion costs a little on WADI) is explained only in
Appendix A prose (line ~1233), never in the main text.

**Verdict: UNDER-CLAIMED** (correct but under-sold as a magnitude).

**Elevation suggestion.** In the §8 mechanism paragraph, state it as an effect size rather than a bare
before/after: "...factorization alone lifts the WADI difficult-subset score by +0.161 AUROC (0.634 to
0.795, Table A1), the largest single design-axis effect in the ablation study." Optionally add one
clause noting that the community score without whole-plant fusion (0.795) exceeds the fused headline
(0.771) on WADI specifically — a useful nuance for readers choosing a configuration on a stationary,
subsystem-driven plant.

---

## Checked and found already PROMINENT (no action needed)

- **HAI difficult-subset win (0.845 vs 0.44-0.50 deep detectors, significant).** Stated in the abstract,
  in contribution C1/C3 framing, as the headline of Table 2, illustrated in Figure 3, restated in §8
  "What the results show," and in the Conclusion. This is already sold as hard as the evidence supports.
- **"Illusion of progress" / deep SOTA collapse to near chance under raw metrics.** Explicitly named as
  "the concrete face of the illusion of progress" in §6, tied to related work (§2.2), and repeated in
  Discussion and Conclusion. Already a headline framing device, not a footnote.
- **Reachability-vs-probability mechanism.** Stated in the abstract, introduction (twice), §4.2, §8 (two
  full paragraphs plus Table A6 decomposition), Conclusion, and illustrated synthetically in Figure A1.
  If anything this is repeated more than necessary (consistent with the project's own prior
  de-duplication pass); it is not under-sold.
- **Drift-spectrum + causal drift/changepoint typing as a method contribution.** It has its own numbered
  top-level section (§7, not a subsection), its own contribution letter (C3), two dedicated tables (3, 4),
  and appears in the abstract and conclusion. It is not "buried as a rescue" structurally — the AUROC
  recovery (0.524->0.723, 0.178->0.671, P=0.014) is well presented. (Its most under-sold *sub-result*,
  the false-alarm collapse, is Item 1 above.)
- **A8 (between-regime overlap) treatment.** Correctly kept as a deferred/negative result in Appendix C
  and §8 ("specified but not realized"). The diagnostics (`a8_drift_lens_verdict.md`) confirm this is the
  right resting state — the hypothesized masking failure mode is absent on all three real datasets and
  present only on synthetic positive controls. No elevation is warranted here; promoting it would violate
  the wins-only rule the paper correctly follows.

---

## Summary ranking (most under-sold first)

1. SWaT false-alarm collapse, 79% -> 6.5% (drift-typed detector) — BURIED, add to abstract/conclusion.
2. Subsystem localization, HAI top-3 68% vs 56% random (P=0.024) — BURIED, no contribution-list presence.
3. Training/inference efficiency, 47-72x faster than TranAD, <30ms/<4GB edge footprint — BURIED in Appendix B.
4. WADI community-factorization effect size, 0.634->0.795 (+0.161, largest single ablation effect) — UNDER-CLAIMED, state as a magnitude in main text.
