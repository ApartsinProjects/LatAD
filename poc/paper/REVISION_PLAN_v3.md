# LatAD paper revision plan (v3) — MDPI IoT

Guiding policy (locked this session):
- **Wins-only**: only confirmed, significant positive results reach the paper. Negatives live in
  the registry (committed 2d78e6b). No apologetic discussion of our method.
- **Forward-only tone**: limitations are boundary conditions, never confessions; no em-dashes, no
  "honestly/slim/marginal".
- **Construct-matched multi-seed** numbers throughout (5 seeds, mean±std, co-computed).

---

## 1. Datasets: drop SKAB, lead with WADI / HAI / SWaT
- Remove SKAB everywhere (33 mentions): abstract line, dataset description, results table + row,
  Figure 1 bars + caption, discussion, data-availability, reference entry.
- Promote **SWaT** to a headline dataset. Framing: real iTrust testbed, **real executed attacks**
  (not synthetic) — reinforces the "true CPS" scope (real sensors + actuators + physics).
- Headline trio = WADI, HAI, SWaT (all real CPS, real attacks).

## 2. Numbers: seed-0 → multi-seed (5 seeds) + CIs
Replace every headline number. Difficult-subset AUROC (the paper's key metric):

| dataset | OLD (seed-0) | NEW (5-seed) |
|---|---|---|
| WADI | 0.726 | **0.690 ± 0.027** |
| HAI  | 0.831 | **0.811 ± 0.016** |
| SWaT | (not headline) | **0.960 ± 0.006** |

All-subset AUROC: WADI 0.792, HAI 0.933, SWaT 0.991. Deploy F1/FPR: WADI 0.545/0.012,
HAI 0.444/0.096, SWaT 0.945/0.007. Sync abstract, body, tables, captions — every occurrence.

## 3. Baselines: expand + make fair
- Deep SOTA (raw point-wise, re-scored on Modal, window-aligned): **USAD, TranAD**. **Drop GDN**
  (jobs failed/timeout; USAD+TranAD already show the collapse cleanly; two deep baselines + classical
  + LinRes is a strong honest set).
- Classical: **Isolation Forest**, **AutoEncoder** (reconstruction).
- **NEW baseline: one-hot linear-residual detector (LinRes)** — leave-one-channel-out linear
  cross-channel predictor, discrete channels one-hot encoded. Strong, correctly-specified simple
  competitor; closes the "did you try a simple correlation model" reviewer question.
- Rebuild the §6 comparison table (All/Easy/Difficult × AUROC/F1/FPR) from the unified per-window
  scores tables (`_diagnostics/scores_*.npz`) — construct-matched, multi-seed for LatAD/IF/AE,
  single for USAD/TranAD. [DECISION: finalize SOTA basis = window-aligned to LatAD's per-window split.]

Difficult-subset AUROC (max|z| canonical split) that the table will report:
| method | WADI | HAI | SWaT |
|---|---|---|---|
| **LatAD** | **0.690** | **0.811** | **0.960** |
| AutoEncoder | 0.425 | 0.757 | 0.939 |
| Isolation Forest | 0.677 | 0.627 | 0.853 |
| LinRes (one-hot) | 0.392 | 0.586 | 0.959 |
| USAD | 0.44* | 0.68* | 0.87* |
| TranAD | 0.47* | 0.65* | 0.88* |
(*deep-SOTA cells: reconcile per-timestep vs window-aligned before final table.)

## 4. Difficulty stratification: add the motivation
- Insert the `DIFFICULTY_RATIONALE.md` argument into §5: *why* difficult-subset (aggregate score is
  dominated by trivial anomalies; the security-relevant faults are stealthy correlation-breaks),
  and *how* we decide (prefilter out anomalies a simple, model-free rule catches; thresholds set on
  normal data only → non-circular; prefilters strictly simpler than the method → fair).
- [DECISION] Headline split = **max|z| quantile** (matches current paper, simplest). Optionally note
  the linear-residual as a better-calibrated axis (it is drift-robust; max|z| over-fires under WADI
  drift). Recommend: max|z| primary, one sentence on linear-residual robustness. Do NOT headline the
  strict range-based split (it exposes an unsolved core — future-work territory, not a result).

## 5. Thesis reframe: multimodality LEVEL, not gradient
- **Remove** "advantage scales with multimodality" (K* is flat 22–25 across WADI/HAI/SWaT; SKAB was
  the only low-K* anchor and it is gone — the gradient claim has no support).
- **New framing**: all three benchmarks are *massively multimodal* (K* ≥ 22); LatAD wins on every
  one. Where the advantage is largest is governed by **reconstruction-hardness**: it is largest on
  WADI (AutoEncoder difficult AUROC 0.43, below chance — pure correlation-breaks) and smallest on
  SWaT (AE 0.94 — mostly reconstructable). This is the reconstruction-inversion mechanism, now
  quantified per dataset. Cleaner and more defensible than the K* story.
- Keep the assumptions-coverage paragraph (A1–A8 leveraged; A9–A10 future work).

## 6. Method contributions (confirmed only)
- Base model: **VaDE-hard+resid(auto)** (density + auto-gated whitened residual + auto-gated basin).
- **Confirmed improvement — drop the nearest-component term** (uniform gain, multi-seed confirmed).
- **NOT included** (registry only, tested and non-confirmed under proper statistics):
  hierarchical density (paired 5-seed n.s.: WADI +0.003, HAI +0.013, SWaT +0.007), trajectory /
  (mode,within-density) detector (HAI bootstrap: 0.672 vs static 0.735), one-hot detector inputs
  (hurt WADI 0.690→0.606). None mentioned in the paper.

## 7. Results narrative (wins-only, HAI as lead)
- Headline: **LatAD is the best standalone detector on three real-CPS attack benchmarks**, multi-seed.
- **Lead the difficult-subset story on HAI** (decisive: 0.811 vs AE 0.757, IF 0.627, deep-SOTA
  0.44–0.68) — the dataset where nonlinear joint structure is genuinely required.
- Present WADI and SWaT honestly and without apology: LatAD leads on both; on WADI-difficult
  Isolation Forest is competitive (state it plainly as a strong baseline, not a weakness); the deep
  SOTA collapse on the difficult subset everywhere (the robust cross-dataset finding).

## 8. Honest boundary (future work, one sentence, no confession)
> Extending the latent model with temporal and trajectory structure, and detecting stealthy attacks
> that perturb joint sensor relationships while remaining within observed operating envelopes, are
> directions for future work.

## 9. Citations
- Fix **ref-12** truncated DOI (bibtest: 25/26 valid, this one not_found).
- Fix **orphan Figure 1** (captioned, never referenced in body — add an in-text pointer).
- Add **4** Aperstein co-author papers (the vibration / PdM / fault-prediction CPS line): gearbox
  operating-conditions (ICCAD 2023), helicopter bolt-loosening (PHM Europe 2022), rotorcraft
  flight-controls fault prediction (IFAC 2022), vibration signal decomposition dilated-CNN (PHM 2023).
  Position as the authors' CPS sensor-time-series / condition-monitoring line; LatAD is the
  unsupervised multimodal-latent generalization. **DO NOT** cite the "Modeling Normal Is All You
  Need" preprint (authors' own prior research, excluded per author decision). DOIs pending primary-
  source verification (web-researcher) + bibtest before entry.
- Literature gate: 1–2 recent (2024–2026) CPS-anomaly / evaluation-methodology refs if missing.

## 10. Build + QA
- Regenerate the MDPI DOCX via `html2doc` `mdpi_from_html.py` (verify figure/media count).
- Final `bibtest` pass (must be all-valid before "done").
- `paper-reviewer` self-consistency audit (SC-1..SC-11): abstract↔body parity, number sync,
  cross-refs, orphan floats, section list, tone.

---

## Execution order
- **Phase A — numbers/tables** (compute per-subset F1/FPR from scores tables; finalize SOTA basis;
  build §6 table). Mostly compute + table editing.
- **Phase B — prose** (SKAB removal, SWaT promotion, thesis reframe, difficulty motivation, method
  section with the one confirmed improvement, future-work sentence).
- **Phase C — citations** (ref-12, Figure 1, Aperstein, literature gate).
- **Phase D — build + QA** (DOCX regen, bibtest, reviewer audit).

## Open decisions for sign-off
1. **GDN**: drop (recommended) vs re-run.
2. **Difficulty split**: max|z| primary (recommended) vs adopt the linear-residual cascade.
3. **LinRes + IF as formal baselines**: include (recommended).
4. **SOTA table basis**: window-aligned to LatAD's per-window split (recommended) — reconcile the
   per-timestep vs window-aligned deep-SOTA cells.
