# LatAD — Experiment Log

Running record of what we tried, what it showed, and what we shelved. Newest sections at
the bottom of each list. "Ours" = the A→B→C trajectory-aware detector; each dataset gets its
**own** A/B/C weights (shared architecture, not shared weights).

---

## 0. Thesis & headline

Normal CPS data is **MIIM** (Massive, Implicit, Imbalanced, Multimodal). Three models:
**A** mode encoder (VaDE) → γ, d, π; **B** trajectory encoder → context c_t; **C** conditional
detector running the C1–C4 scoring stack. Goal: catch snapshot *and* history-dependent faults.

**Headline result.**
- **Synthetic (labelled):** ours beats LOF / IsolationForest / AutoEncoder on snapshot faults
  **and uniquely** catches the history-dependent `bad_transition` (0.08 → ~0.32 TPR@5%FPR;
  baselines ≈ chance).
- **Real (SKAB/HAI/WADI):** competitive on snapshot; the **trajectory branch adds ~nothing**,
  because these public benchmarks' attacks are snapshot-detectable, not history-dependent.
  This is the empirical justification for the synthetic generator.

---

## 1. Datasets

| dataset | kind | status | notes |
|---|---|---|---|
| `temporal_data.py` (make_temporal_miim) | synthetic | ✓ | clean `bad_transition` floor 0.057 |
| `miim_gen.py` (unified A1–A10) | synthetic | ✓ | labelled: fringe (normal), pocket, near_boundary, drift, ood, wrong_for_regime, bad_transition |
| SKAB | real CPS | ✓ loader | small, contaminated, 35% anom, hard for all |
| HAI | real CPS | ✓ loader | 4% anom; snapshot-detectable |
| WADI (A2) | real CPS | ✓ loader (`wadi.py`) | 6.2% anom; snapshot-detectable; we're behind IF |
| SWaT (A4–A12 tabular) | real CPS | downloaded, **not wired** | no clean single normal+attack CSV; A9 has attack-patterns to parse |

Labelled synthetic categories (miim_gen): **(a) fringe** = hard NORMAL near-boundary (|u|∈[1.7,2)σ);
**(b) pocket** = between two modes; **(c) near_boundary** = just OUTSIDE one mode (|u|∈(2,3]σ);
**(d) bad_transition** = forbidden-path trajectory fault. (a)+(c) are a matched inside/outside-rim pair.

---

## 2. Chronological experiments

### Generator
- Implemented `miim_gen.py` from the spec (Phase A instantiate system, Phase B semi-Markov walk,
  anomaly injection). Adaptive compatibility relation (p_forbid too aggressive → 0 modes; relaxed).
  Result: K≈63 modes, coverage 61/63, 5–7 anomaly families. `premature_depletion` still unwired.

### The decisive experiment (does B help catch `bad_transition`?)
1. **Baseline agent build** — full A/B/C ran end to end. bad_transition 0.267 = 0.267 (C+B == C-alone).
   Root cause: **B undertrained** (memory-horizon probe ~0.35 at k=1).
2. **B → fidelity** — *focused near-offset training* (offsets ≤64) + *clipped rarity weights* +
   more epochs lifted the probe to **0.88@k=1 / 0.77@k=4**. But C+B **still** == C-alone (lift 0).
   → **FiLM falsified:** conditioning the window decoder scores p(window|c_t); a `bad_transition`
   window is genuinely normal, so no window-decoder can move it.
3. **Concatenation falsified** — VaDE over [window(144)⊕c_t(96)]: window dominates the clustering,
   latent-NLL for bad_transition stays at the floor.
4. **Per-mode context density works** — score p(c_t|mode): distance of c_t from the mode's normal
   arrival-context. **Per-mode (not pooled)** covariance is essential (pooling washes out the tight
   arrival structure). bad_transition 0.16 → ~0.32.
5. **Fusion matters** — z-sum lets the heavy-tailed context drown snapshot faults; an **asymmetric
   p-value OR** (window branch keeps most FPR budget, context a thin tail) preserves both.
   **PCA-reduce c_t** (24 dims) cut the χ² tail → cleaner tradeoff.
6. **Soft (argmax-free) context** — γ-weighted per-mode density (no hard assignment) ≈ hard-argmax,
   marginally better, removes the poisoning path → **adopted**.
7. **Rarity / coverage weighting — inert so far.** On temporal_data (weights std 0.016) and miim_gen
   (std 0.010) it had ~no effect: streams are within-dwell-dominated and my weighting targets
   *transition* windows, not settled windows that *arrived via* a rare edge. Needs re-implementation
   (arrival-edge weighting) + data with a real rare-valid-edge tail.
8. **Ported to miim_gen** — B even more faithful (probe 0.91@k=1, 0.70@k=8); context lifts
   bad_transition **0.079 → 0.32**; mechanism generalises to the full A1–A10 data.
9. **Semi-Markov EVENT detector (compression idea)** — run-length (mode,duration) events (debounced)
   + transition surprise −log P(current|predecessor). **Matches the neural context density on
   bad_transition (0.325 vs 0.322) with NO B training** — cheap, interpretable, pure-transition
   branch (≈0 on snapshot). Also emits a **dwell-surprise** signal (unvalidated).

### Baseline comparisons
- **Temporal (per-type TPR@5%FPR, AUROC):** ours window-VaDE beats LOF/IF/AE on snapshot + AUROC
  (0.876 vs .80/.71/.70); ours+context uniquely catches bad_transition (0.244) with best AUROC (0.912).
- **miim_gen (labelled categories):** ours ties LOF on AUROC (.854 vs .853), beats IF/AE; only
  ours+context catches bad_transition (.322 vs ~.10). `near_boundary` **hard for ALL (~.06)**.
- **Real (SKAB/HAI/WADI):** ours best TPR@5%/F1 on SKAB & HAI; behind IF on WADI (AUROC .629 vs .748).
  Trajectory branches add ~nothing everywhere → real benchmarks are snapshot-only.

---

## 3. Validated techniques (keep)
- B fidelity: **focused near-offset training + clipped rarity weights** (probe gate ~0.7@k≤8).
- **Per-mode** (not pooled) Ledoit-Wolf context density `p(c_t|mode)`; **soft γ-weighted** variant.
- **PCA-reduce c_t** before the density (cut χ² tail).
- **Asymmetric p-value OR** fusion (window budget-heavy, context thin tail).
- **Semi-Markov event model** as a cheap, sharp transition branch.
- Three complementary branches: **window VaDE (snapshot) + event (discrete transition) + c_t context (continuous)**.

---

## 4. Backlog — ideas shelved / queued

**Queued now (priority):**
1. **`near_boundary` hard-band tooling** — the just-outside-rim case is the weakest spot across ALL
   methods (~0.06). Needs the C1/`ModeDistance` σ/E∈[1,2] machinery applied at detection.
2. **Continuous trajectory anomaly** — build a *smooth* trajectory drift (not a discrete forbidden
   edge) to decide **event-model vs neural-B** (the event model can't express continuous trajectory
   faults; this is the test that justifies B over the free event model).
3. **kNN-retrieval context density** (needle-in-haystack) — replace the parametric per-mode Gaussian
   with kNN-to-normal-arrival (the `TrueNormalOracle` pattern in context space). Cheap; may sharpen
   bad_transition/fringe by handling rare-valid arrivals via retrieval instead of smearing.
4. **3-branch fusion tuning** — best single operating point across window+event+context.
5. **Dwell-anomaly labels** — add a "stuck-too-long / left-too-early" family to validate the event
   model's dwell-surprise signal.

**Other shelved ideas:**
- **Multi-head B** — add a **dwell head** (time-in-state) and **recency head** (time-since-last-visit)
  + presence features as auxiliary inputs (the "how long / when" the trajectory must encode).
- **Attention / KV-memory B** + **surprise-gated / landmark writes** — retain rare needles vs GRU
  compression forgetting (needle-in-haystack transfer).
- **Learned-transition-graph balanced sampling** — soft transition matrix (γ outer-products, lag≥2)
  + arrival-edge rarity weighting for coverage augmentation (needs clean graph from settled windows
  to avoid poisoning; can't cover never-observed edges).
- **Rarity/coverage weighting** — re-implement as *arrival-edge* weighting; validate on rare-edge data.
- **`premature_depletion`** anomaly family — wire the accumulator injection in miim_gen (6th family).
- **WADI tuning** — we're behind IF (.629 vs .748); less downsample / better K / feature choice.
- **SWaT wiring** — parse A9 (2022) attack CSVs / Attack Patterns.txt to complete the real set.
- **Full B/C on SKAB & HAI** — currently only A + event ran there (`--noB`); run the full A+B+C for
  completeness (expected: trajectory adds ~nothing, since snapshot-only).

---

## 5. Open questions / honest limits
- **Is B worth its cost?** The free **event model** matches neural-B on `bad_transition`. B's only
  justification is **continuous/contextual** trajectory faults an event model can't express — untested
  (see backlog #2).
- **Graph-free ceiling:** false alarms come from **rare-valid transitions looking forbidden**
  (trajectory-domain E1). Fundamental to graph-free detection; fixed only by coverage of rare-valid edges.
- **Real benchmarks lack history-dependent anomalies** — so the trajectory contribution can only be
  demonstrated on the synthetic generator (which is why it exists).

---

## 6. Key files
- Generators: `data.py` (static), `temporal_data.py`, `miim_gen.py` (unified A1–A10).
- Architecture: `ldt_a.py` / `ldt_b.py` / `ldt_c.py`, `run_ldt.py` (decisive experiment).
- Tuning/exploration: `tune_b.py`, `explore_c.py` (soft context), `explore_miim.py`, `event_detector.py`.
- Baselines/real: `compare_baselines.py`, `compare_miim.py`, `run_real_abc.py`, `wadi.py`, `skab.py`, `hai.py`.
- Doc: `synthetic_data.html` (A1–A10 spec), `architecture.html` (A/B/C spec).
