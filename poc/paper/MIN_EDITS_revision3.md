# Minimum edit list — LatAD (MDPI *IoT*) round-1 revision

## TL;DR — minimal SET to get accepted
Every reviewer- and editor-required change is **already implemented and audited on revision2** (IoT2.html +
`letter_{editor,reviewer1,reviewer2}.html`, verified across review cycles 1-5). The minimal path is therefore:
1. **Port revision2's revised `IoT2.html` and the 3 response letters onto revision3** (one copy, not a rewrite).
2. **One QA pass**: numbers match across abstract/tables/letters; equations numbered; z-notation split; two figures present; abbreviations expanded.
3. **Reframe the cross-dataset claim honestly** (the single most acceptance-critical point): HAI = clear significant win; WADI = tie with Isolation Forest / LinRes (state the p); SWaT = significant but near a ceiling. Reviewers will reject "best/leads everywhere"; they accept an honest HAI-led claim with the ties stated.

The **blocking five** (reviewers re-check these; all done on rev2, just port): ED4 time-aware significance,
ED3 edge-cost table, ED5 per-head ablation, ED2 nearest-component FPR-by-mode, ED1 scope the unused
assumptions. Presentation items (R2.*) are cheap and expected. Do NOT add new experiments beyond porting.

Full mapping below.

---


Target: `poc/paper/IoT2.html` on **revision3** (the submitted version). Reference implementation for every
item already exists on the **revision2** branch (`git show revision2:poc/paper/IoT2.html`, plus the three
response letters `review_round_1/letter_{editor,reviewer1,reviewer2}.html`). This list is the *minimum* set
of edits to satisfy the actual Reviewer 1, Reviewer 2, and Editor letters (`review_round_1/Review1.txt`,
`Review2.txt`, `Editor review.txt`). Each row: comment → minimal edit → revision2 reference → type.

Legend — **T** = text/presentation only; **E** = needs an experiment/number/table/figure (all already
computed on revision2, so this is a *port*, not new work).

---

## Editor letter

| # | Requirement | Minimal edit | revision2 reference | Type |
|---|---|---|---|---|
| ED1 | Validate unused assumptions A3, A9, A10 | Add one appendix subsection: activate A3 (basin/residual), A9 (multiscale), A10 (history) on a dataset that exercises them, OR narrow the contribution text to the validated assumptions. | rev2 Appendix C (SKAB/Cranfield adds) + revision2 `_diagnostics/e51_ablation` (A9/A10 novel-vs-base) | E |
| ED2 | Direct support for rare-regime-safe nearest-component | Add the mixture-NLL vs nearest-component false-positive-rate-by-mode-frequency experiment. | rev2 nearest-component stratification; revision2 `inv_a5_rarity` (FP-by-occupancy, invariant) | E |
| ED3 | Report training + inference cost (edge) | Add a cost table (params, train time, peak GPU mem, inference latency, host RAM) per dataset. | rev2 Table B1 (Host RAM column already added) | E |
| ED4 | Time-aware statistical test | Replace naive CI with an **attack-episode/block bootstrap**; report 95% CI + pairwise p vs strongest baseline. | rev2 significance: HAI P~0, SWaT P=0.007, WADI ties LinRes P=0.46 | E |
| ED5 | Ablations isolating the source of the gain | Add the per-head five-seed ablation (residual / density / nearest / density+nearest / +residual / auto-gate) on all 3 datasets. | rev2 ablation; density head beats USAD/TranAD on difficult (0.802/0.794/0.656 vs 0.477/0.444…) | E |

## Reviewer 1 (technical)

| # | Comment | Minimal edit | rev2 reference | Type |
|---|---|---|---|---|
| R1.1 | A9/A10 unused, A3 basin inactive → limit to validated, or add experiments | Same as ED1: scope the claim to validated assumptions + appendix activation. | rev2 Appendix C; e51 | E |
| R1.2 | Nearest-component lacks direct support; test imbalance ratios, rare-normal FPR | Same as ED2. | `inv_a5_rarity` | E |
| R1.3 | Regime-community cost vs single global detector | Add per-community cost rows to the cost table (ED3). | rev2 Table B1 | E |
| R1.4 | Time-correlated windows → time-aware test; clarify WADI CI/p | Same as ED4; add one sentence defining the episode-bootstrap CI and the WADI p=0.46 tie. | rev2 WADI significance wording | E/T |
| R1.5 | Six static stats → is the gain from cross-channel density or the representation? | Add the raw-sequence / temporal-feature comparison showing the gain is cross-channel density, not the window rep. | revision2 `e42` (temporal rep) + feat_temporal ablation | E |

## Reviewer 2 (presentation / structure — mostly text)

| # | Comment | Minimal edit | rev2 reference | Type |
|---|---|---|---|---|
| R2.1 | Expand abbreviations on first use (LatAD, IIoT, CPS, SCADA) | Spell out each acronym at first occurrence in abstract + body. | rev2 IoT2.html | T |
| R2.2 | Abstract over-weights prior-method problems, under-explains LatAD | Rewrite abstract: 1 problem sentence, then LatAD's principle + technical contributions + headline HAI number. | rev2 abstract | T |
| R2.3 | Intro para 1: delete "This paper detects them by…"; no contributions in para 1; add CPS background | Move contributions out of para 1; open with CPS background. | rev2 intro | T |
| R2.4 | p2 l71 "the task is often framed as…" — what task? | Name the task explicitly (unsupervised multivariate CPS anomaly detection). | rev2 | T |
| R2.5 | "two obstacles" — does LatAD address both? what does it improve? | Add one sentence stating LatAD contributes to both modeling (latent-mode density) and evaluation (difficulty split). | rev2 | T |
| R2.6 | Related Work: move AD definition to Intro; delete LatAD self-refs; rewrite §2.3; delete §2.5 | Strike the 3 named self-description sentences; rewrite §2.3 as related work (not LatAD); remove §2.5. | rev2 §2 | T |
| R2.7 | Add system-model / application-scenario diagram (where in IIoT LatAD runs) | Add **Figure 1: system model** (IIoT stack → where LatAD ingests telemetry + flags). | rev2 Figure (system model SVG) | E(fig) |
| R2.8 | MIIM (§3) vs LatAD (§4) relationship | Add a bridging paragraph: each MIIM assumption → the LatAD component that exploits it. | rev2 §3→§4 bridge | T |
| R2.9 | §4 readability; map representation/scoring to pipeline; add workflow figure | Add **Figure 2: LatAD workflow** (representation stage → scoring stage) + label each §4.x with its pipeline stage. | rev2 workflow figure | E(fig) |
| R2.10 | Shared encoder or per-community? report train time / GPU mem / latency | State encoder sharing explicitly; add the cost table (= ED3). | rev2 Table B1 | E/T |
| R2.11 | z-notation conflict (§4.3 latent vs §5.3 standardized) | Rename the standardized value in §5.3 (e.g. to `s` or `ζ`); keep z for the latent only. | rev2 §5.3 | T |
| R2.12 | Typesetting: unify indentation; number all equations | Unify paragraph indentation; number every display equation (1)…(9). | rev2 (9 numbered eqs) | T |

## Response letters (required)
Point-by-point letters to Editor, R1, R2 — port verbatim from revision2 `review_round_1/letter_editor.html`,
`letter_reviewer1.html`, `letter_reviewer2.html` (already audited in `review4_reviewer_letters.md`).

---

## Suggested order (minimize effort)
1. **All T edits** (R2.1–2.6, 2.8, 2.11, 2.12, R2.4/2.5) — pure prose/typesetting, ~1 pass, port from rev2.
2. **Two figures** (R2.7 system model, R2.9 workflow) — port the SVGs from rev2.
3. **Five evidence blocks** (ED1–ED5 / R1.1–1.5) — port the tables/numbers already computed on rev2 (cost table, episode-bootstrap CIs, per-head ablation, nearest-component FPR-by-mode, raw-vs-window rep).
4. **Response letters** — port from rev2.

Everything here already exists on revision2; revision3 keeps only the minimum, auditable change set on top of
the submitted paper. Optional (not required by reviewers): the LatAD naming-collision note (a 2024 "LATAD" exists).
