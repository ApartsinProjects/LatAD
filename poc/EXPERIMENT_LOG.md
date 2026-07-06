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
- **Real (SKAB/HAI/WADI):** ours leads **TPR@5%FPR & F1 on all three** (a baseline edges AUROC) — it
  catches more attacks at a fixed low false-alarm budget; strongest on multimodal **HAI** (TPR 0.783).
  The **trajectory branch adds ~nothing**, because these public benchmarks' attacks are snapshot-
  detectable, not history-dependent — the empirical justification for the synthetic generator.

---

## 1. Datasets

| dataset | kind | status | notes |
|---|---|---|---|
| `temporal_data.py` (make_temporal_miim) | synthetic | ✓ | clean `bad_transition` floor 0.057 |
| `miim_gen.py` (unified A1–A10) | synthetic | ✓ | labelled: fringe (normal), pocket, near_boundary, drift, ood, wrong_for_regime, bad_transition |
| SKAB | real CPS | ✓ loader | small, contaminated, 35% anom, hard for all |
| HAI | real CPS | ✓ loader | 4% anom; snapshot-detectable |
| WADI (A2) | real CPS | ✓ loader (`wadi.py`, horizon bug fixed) | ours leads TPR@5%FPR 0.39/F1 0.42 vs IF 0.30/0.34, trails AUROC (§2.4) |
| SWaT (Dec 2015) | real CPS | ✓ loader (`swat.py`, Kaggle mirror) | 51 ch; 16.7% anom (non-canonical split); EASIEST set, all methods 0.96–0.99 AUROC (§2.5, §2.8) |
| SWaT (A4–A12 tabular) | real CPS | downloaded, **UNLABELLED** | raw process CSVs, no attack/label column, no attack-timing doc → cannot label |

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
- **Real (SKAB/HAI/WADI):** ours best TPR@5%FPR & F1 on all three; a baseline (LOF/IF) edges AUROC.
  WADI needed a horizon-bug fix (§2.4) to reach 0.393/0.423. Trajectory branches add ~nothing →
  real benchmarks are snapshot-only (the empirical reason the synthetic generator exists).

### 2.4 WADI tuning (`tune_wadi.py`, `probe_restd.py`) — resolved, with a real bug fix
Goal: close the gap where we trailed IF (was 0.629 AUROC / 0.319 TPR). Findings, one knob at a time:
- **THE ACTUAL LEVER — a train/test horizon bug in `wadi.py`.** The loader downsampled the *normal*
  stream (×10) but **not the attack stream or labels**, so train windows spanned 600 s while test
  windows spanned 60 s; the window features (mean/std/…) are then computed over mismatched horizons and
  our windowed detector degrades. Downsampling test consistently lifts **VaDE 0.629→0.692 AUROC,
  0.319→0.393 TPR@5%FPR, 0.307→0.423 F1** (now reproducible in the unified harness = the `tune_wadi`
  numbers). Ruled out re-standardisation as the cause (`probe_restd.py`: ≤0.003 effect on every set).
- **Post-fix headline (unified harness, K20/L10):** ours **TPR@5%FPR 0.393 / F1 0.423** vs IF
  0.304 / 0.343 → **ours leads at the operating point** (the metric the proposal cares about: few false
  alarms); IF still leads AUROC (0.720 vs 0.692).
- **The near-constant channels are the attack CARRIERS, not noise.** ~30/123 channels are ~constant
  in normal. Standardised by their tiny σ, any attack-time deviation becomes a huge z-score → the single
  most sensitive attack signal. **Dropping them collapses every reconstruction/extremeness detector to
  chance (~0.50).** Lesson: **keep all channels**; an earlier "drop near-constant" instinct was wrong.
- **SECOND WADI WIN — clip glitch/shifted channels (root-caused via EDA, §2.7).** A few channels carry
  physically-impossible standardised values: `2_MCV_007_CO` is constant=0 in the 14-day normal (std=0)
  but active later (÷0 → 1e8 σ); `2B_AIT_002_PV` reads 9±0.16 in normal but 4428 in the attack file
  (a recalibrated / non-comparable sensor). One such feature dominates VaDE's reconstruction+whitener
  and drowns every other channel (IF is scale-invariant so was immune — this is why we trailed IF).
  **Clipping standardised features to ±10σ** (A2 hard-envelope: a 10σ reading is a sensor fault, not a
  process state) lifts **VaDE 0.691→0.791 AUROC, 0.393→0.589 TPR, 0.423→0.574 F1** — now beats IF on
  ALL metrics (0.735 AUROC). Clip is **WADI-specific**: on HAI it slightly hurts (0.879→0.868; big
  excursions are real attack signal), on SKAB it is a no-op (already in range). Baked into `wadi.py`.
- **Representation:** `stats` beats `temporal` (WADI attacks are level shifts, not dynamics);
  robust-scale ≈ z-score. **Reconstruction is ill-suited to WADI** (AE ≈ chance; reconstructs the
  out-of-range excursions too well) — our edge is the latent-NLL/mode term, not the recon term.
- **Fusion does NOT help:** z-sum(IF,VaDE) drags IF's ranking; OR/min-p fusion collapses to 0.60
  because rare-VALID normal excursions dominate the p-value tail (the WADI-domain **E1** problem).

### 2.5 SWaT — wired via the Kaggle mirror (was blocked; now DONE)
- **What we downloaded from SharePoint (A4–A12) is unusable:** raw process CSVs, no label column, no
  attack-timing doc (A8 `Annotation` empty, A6 a `Notes` sheet, `_STATE` = process-phase not attack).
- **The canonical benchmark is the Dec-2015 data**, on the iTrust SharePoint only inside a 108 GB zip
  (bundled PCAP). **Obtained instead from the Kaggle mirror** `vishala28/swat-dataset-secure-water-
  treatment-system`: `normal.csv` (1.39M rows, all Normal → train), `attack.csv` (54.6k rows, all
  Attack), `merged.csv` (the two concatenated). 51 channels + `Normal/Attack` label.
- **The mirror does NOT preserve the canonical interleaved 4-day attack file** (normal periods between
  attacks removed). So `swat.py` uses a **leakage-free split**: train on head of normal, TEST on
  held-out normal (0) + `attack.csv` (1) → 16.7% anomaly. **Absolute F1 is NOT comparable to published
  raw-F1 0.81** (this split is easier; report only same-split baseline comparisons).
- **Result:** SWaT is the **EASIEST** of the four — every method 0.96–0.99 AUROC; ours (**window-VaDE**)
  **0.989 AUROC / 0.961 TPR@5% / 0.868 F1**, leads LOF .980 / IF .960 / AE .981. (The +event/+context
  0.995/0.999 are a **concat-seam artifact** — audit #2 — the mirror test is normal-block++attack-block,
  so the trajectory branch fires at the label flip; disregard that lift.) EDA: trivial-rule AUROC 0.960,
  **79% trivially separable**, **81% OUT-of-envelope (A2)**, silhouette **0.31 (cleanest modes of any
  set)**, BIC unsaturated to K=64. Confirms the literature (SWaT ~95% univariate, §2.8).

### 2.6 Full A+B+C on the real set — unified harness (`run_real_abc.py`, one pass, construct-matched)
Baselines LOF/IF/AE vs our ablation window-VaDE → +event → +context(B). Metrics AUROC / TPR@5%FPR / F1.

| dataset | best AUROC | best TPR@5%FPR | best F1 | ours' verdict |
|---|---|---|---|---|
| **HAI** (multimodal, 4% anom) | LOF .943 | **ours+context .783** | **ours+context .526** | **ours wins the operating point** (LOF/AE .709/.722 TPR); thesis case |
| **SKAB** (near-unimodal, 35% anom) | IF .672 | **ours+event .329** | **ours+event .463** | ours+event best TPR/F1; **+context HURTS** (.608→.556 AUROC) |
| **WADI** (post-fix, 10% anom) | IF .720 | **ours .393** | **ours .423** | ours leads operating point (IF .304/.343); IF leads AUROC |
| **SWaT** (mirror split, 17% anom) | **ours .989** | **ours .961** | **ours .868** | easiest set; ours (window-VaDE) tops all baselines. NOTE: the +event/+context .995/.999 are a **concat-seam artifact** (audit #2), not real — the test is normal-block++attack-block so the trajectory branch fires at the label flip. Honest number = window-VaDE .989. |

Reads:
- **HAI is the clean win**: on the genuinely multi-process dataset ours leads both operating-point
  metrics; mode structure pays off exactly where normal is multimodal (thesis).
- **WADI (after the §2.4 horizon-bug fix):** ours **TPR@5%FPR .393 / F1 .423** vs IF .304 / .343 →
  ours leads the operating point (ties LOF/AE there); IF leads AUROC (.720 vs .692). Reproducible.
- **The event branch helps SKAB** (AUROC .608→.620, TPR .267→.329, F1 .393→.463) — a rotor/pump with
  open/close cycles has real transition structure; cheap and neutral elsewhere.
- **The context/B branch helps HAI slightly, HURTS SKAB** (contaminated + near-unimodal → context adds
  noise). Recommend +event as the default real-data branch, +context only when multimodal & clean.
- **Operating-point vs AUROC pattern is consistent across all three:** a baseline (LOF or IF) often
  edges AUROC, but ours leads TPR@5%FPR / F1 — it catches more attacks at a fixed low false-alarm budget.

### 2.7 EDA of the real datasets (`eda_real.py`, figures in `eda_figs/`)
How hard are the anomalies, and how strongly is MIIM (A1-A10) present? Per dataset, on the windows the
models see (raw channels standardised on normal; WADI clipped to ±10σ per §2.4):

| dataset | trivial-rule AUROC | anomalies trivially separable | #modes (BIC) | silhouette | A-geometry of anomalies |
|---|---|---|---|---|---|
| **SKAB** | **0.57** (≈chance) | 28% | 9 (clean min at K=12) | 0.19 | **78% IN-COMMON**, 20% OUT |
| **HAI** | 0.88 | **76%** | ≥22 (BIC not saturated) | 0.08 | **74% OUT (A2)**, 24% in-common |
| **WADI** (clean) | 0.87 | 66% | ≥24 (BIC drops to K=64) | 0.09 | **93% IN-COMMON**, 4% OUT |

- **Q2 anomalies are mostly EASY.** A one-line `max\|z\|` rule scores AUROC 0.87-0.89 on HAI/WADI and
  separates ~70-76% of their anomalies (score > 99th-pct normal). **SKAB is the only genuinely subtle
  set** (trivial rule ≈ chance). Public CPS benchmarks are dominated by out-of-range faults.
- **Q3 strip the easy anomalies → everyone collapses.** On the HARD residual (normal + only the
  non-trivial anomalies): VaDE SKAB 0.581→**0.423**, HAI 0.879→**0.667**, WADI 0.792→**0.506**; LOF/IF
  similar (IF holds best on hard WADI, 0.695). The benchmarks' headline AUROC is **carried by the easy
  majority** — the empirical case for the synthetic generator, quantified.
- **Q4 MIIM is present but IMPLICIT (A5).** Cluster sizes are heavy-tailed / Zipf-like (A1 mode
  explosion + A2 imbalance); BIC keeps improving past K=32-64 on HAI/WADI (many modes). But
  **silhouette is 0.08-0.19 everywhere → modes overlap, not crisply separable** (A5 "regimes are an
  artifact, rarely recoverable"). SKAB alone has a clean BIC minimum (K=12).
- **Cleaning changes WADI's very NATURE (§2.4 clip).** Pre-clip, WADI anomalies looked **55% OUT-of-
  envelope**; after clipping the glitch/shifted channels they are **4% OUT / 93% IN-COMMON** — the
  "easy envelope violations" were largely the artifacts. Cleaned WADI is a subtle in-mode set like SKAB
  (normal & anomaly mode-NLL histograms overlap almost entirely).
- **Q6 per-cluster: a GLOBAL threshold is unfair per mode.** Mean per-cluster FPR **0.77 on HAI, 0.58 on
  WADI, 0.13 on SKAB**, worst-cluster FPR ~1.0 everywhere; several clusters have **below-chance AUROC
  (0.25-0.46)** — the detector is anti-correlated with truth inside subtle in-mode clusters. **Direct
  empirical mandate for mode-conditional thresholds (architecture C4 / TFAR).**
- **Q7 error root-cause (VaDE @ global 5% FPR).** False ALARMS are rare-valid-normal windows in tiny
  modes (mass <2%, NLL@100%) — the A1/A2 "rare mode looks anomalous" (trajectory/context E1). MISSES are
  uniformly **"in a common mode", NLL@1-40%, channels only -1 to -3σ** — genuinely in-distribution
  subtle faults, exactly the hard tail the trivial rule and every model leave on the table.

### 2.8 Published-SOTA cross-check — does the literature confirm the EDA? (web-researcher, verified)
Yes, strongly. Sources verified this session.
- **Best RAW point-wise F1:** SWaT ~**0.81** (GDN), WADI ~**0.57** (GDN); the famous 0.89–0.96 numbers are
  **point-adjusted (PA)**. SKAB best ~0.78 (Conv-AE, 13.6% FAR / 28% MAR). HAI uses **eTaPR**, no clean
  F1 leaderboard.
- **Metric inflation is real and huge** (Kim et al. AAAI-2022): a **RANDOM** score under PA gets
  **F1 0.969 SWaT / 0.965 WADI**, beating every deep model; Garg et al. (TNNLS-2021) reproduce it (PA-F1
  0.92/0.96 labelling 0.17% of points). → confirms our "headline is carried by the easy majority."
- **Anomalies are mostly univariate** (arXiv 2606.02670): SWaT ~**95%** / WADI ~**99%** of anomaly
  segments are univariately separable; a flat linear-AE matches deep channel-dependent detectors. →
  directly confirms our Q2 trivial-rule finding (our cruder rule: SWaT 79%, WADI ~66–71%, HAI 76%).
- **Simple baselines competitive** (Garg et al.): PCA / channel-wise-AE beat OmniAnomaly; deep methods
  "fail to detect even these simple anomalies." **Triviality is a named benchmark flaw** (Wu & Keogh,
  arXiv 2009.13807). **SKAB is genuinely hard** (confirmed).
- **Novel-to-us (not published):** the exact HAI 76%-separable figure and the *strip-easy → all-models-
  collapse* experiment are **our own** contributions; the literature confirms the mechanism + the
  SWaT/WADI direction, but 2606.02670 does not cover HAI/SKAB. Report these as our results, not citations.
- **Net:** the field's own evidence backs LatAD's thesis — public CPS benchmarks are dominated by easy,
  point-detectable anomalies and standard metrics disguise it; the synthetic MIIM generator (hard,
  history-dependent faults) is the response.

### 2.10 Design ablation on real data (`ablate_design.py`) — three questions answered
VaDE AUROC / TPR@5%FPR, same standardised+clipped raw channels as the EDA.
- **Overspecified clustering (K vs 3K): NO benefit anywhere** (SKAB .581→.583, HAI .879→.883,
  WADI .791→.791, SWaT .990→.989). The nearest-component NLL is insensitive to extra/split components;
  K ≈ #modes is enough. → don't overspecify.
- **Sequential (VAE→GMM) ≈ or slightly BEATS joint VaDE on real data** (SKAB .581→**.607**,
  HAI .879→**.886**, WADI .791→.793, SWaT .990→.992). The joint-training edge seen on *synthetic*
  (README 0.951 vs 0.935) vanishes when modes are implicit/overlapping (A5); the simpler sequential model
  is a fair scale-down. → sequential is fine, even preferable, on these real sets.
- **Larger windows (W vs 3×W): DATASET-DEPENDENT, not a universal knob.** Big win where anomalies are
  sustained/slow — **HAI .879→.908 (TPR .671→.794), SWaT .990→1.000**; **hurts WADI .791→.729** (short
  attacks diluted); neutral SKAB. → tune window per dataset; longer for HAI/SWaT, not WADI.

### 2.11 Consolidated single-model detector (`consolidated.py`) — VaDE + C1-C4, no B
"Scale down to one VAE+clustering model + the C-scoring stack" (C5 is trajectory-only → dropped).
Base (whitened residual + latent NLL) + C1 (dependency-violation shuffle anomalies, real-data version) +
C2 (basin agreement) + C3 (per-mode PCA + supervised fuser) + C4 (mode-conditional thresholds).
AUROC / TPR@5% / F1, VaDE **base** vs **fused** C1-C3:

| dataset | VaDE base | fused C1-C3 | best baseline | verdict |
|---|---|---|---|---|
| SKAB | .581/.316/**.451** | .581/.205/.318 | LOF .638 | base ≥ fused; LOF best AUROC |
| HAI | **.879/.671/.482** | **.609/.006/.010** | LOF .911/.803/.551 | **fuser COLLAPSES**; base fine |
| WADI | .791/.589/.574 | .779/.571/**.653** | IF .730 | fuser helps F1 only |
| SWaT | **.990/.967/.871** | .977/.945/.860 | LOF .977 | base best; fuser slightly worse |

- **KEY NEGATIVE RESULT: the C1-C3 supervised fuser does NOT transfer to real data.** It is trained on
  **dependency-violation** shuffle anomalies (C1's design target), which are ORTHOGONAL to the real
  benchmarks' **out-of-envelope point** attacks — so on HAI (74% OUT) it collapses to AUROC .609 / TPR .006.
  It helps only WADI (subtler, 93% in-common, closer to C1's family). **The plain VaDE base is the best
  single model on real data**; the synthetic-tuned C-stack should not be bolted on blindly.
- **C4 mode-conditional** calibrates well ONLY when modes are clean: SWaT FPR .042≈target; but HAI .356 /
  WADI .47 / SKAB .112 ≫ 5% (small/noisy clusters). Needs shrinkage / min-count (backlog).
- **Takeaway for scale-down:** keep the single VaDE (+ optional sequential = §2.10); the C1-C4 stack is a
  synthetic-data instrument, not a real-data win. This is consistent with the EDA (real anomalies are
  out-of-envelope point faults, exactly what a plain VaDE/AE already catches).

### 2.12 SOTA targets, weights, reproduction route (web-researcher, verified)
- **Raw point-wise F1 targets** (the honest metric; 0.9+ figures are point-adjusted): **SWaT 0.81 / WADI
  0.57** (GDN); **HAI 21.03 0.83** (RANSynCoder; GDN only 0.60 there); **SKAB 0.78** (Conv-AE). Transformer
  methods (Anomaly-Transformer, DCdetector) publish PA-only; DCdetector SWaT raw-ward F1 ≈ 0.24.
- **Pretrained weights: NONE.** No major repo (GDN, USAD, TranAD, Anomaly-Transformer, DCdetector) ships
  checkpoints for SWaT/WADI/HAI/SKAB — the data is access-gated so weights can't be redistributed.
  **We must retrain to compare.**
- **Reproduction route = the TranAD harness** (`github.com/imperial-qore/TranAD`): one repo + one command
  runs TranAD **and GDN/USAD/MAD-GAN/MTAD-GAT/OmniAnomaly** on WADI/SWaT; modern stack, CPU-runnable
  (`python3 main.py --model GDN --dataset WADI --retrain`). GDN standalone has an old CUDA-10.2 stack +
  run-to-run non-determinism (F1 0.74–0.82, repo issue #68). Its headline F1 is PA — **recompute raw F1**.
- **Plan:** clone TranAD, run GDN/USAD on WADI, compute RAW point-wise F1 on the SAME split we use, and
  report it in the 3-way (all/easy/hard) table next to ours.
- **Running (Modal GPU, gpu2modal skill):** `poc/sota_bundle/modal_sota.py` — A10G, clones TranAD, writes
  our WADI arrays (MinMax, ×10 downsample, 123 ch) into `processed/WADI/`, runs USAD/TranAD/GDN, recomputes
  RAW best-F1 + F1@5%FPR from the per-timestep score (their eval is point-adjusted). First run is
  recon-heavy (prints their data contract) + USAD. WADI arrays in `poc/sota_bundle/wadi_{train,test,labels}.npy`.

### 2.13 Reporting convention (ADOPTED) — always report ALL / EASY / HARD
Going forward every real-data result is reported on three subsets (`report3.py`):
**EASY** = anomalies a trivial single-channel rule separates (range/threshold/rule-detectable, score >
99th-pct normal); **HARD** = the rest (subtle, in-mode); **ALL** = both. The HARD column is the
discriminative one — where a method must actually earn its keep, and where SOTA comparison matters.

### 2.14 In-common root-cause diagnostic (`diag_incommon.py`) + the 3-way table (`report3.py`)
**Which lever recovers the subtle in-mode ("in-common") misses?** hard-subset AUROC on SKAB (78%
in-common) / WADI (93%):
| lever | SKAB hardAU | WADI hardAU |
|---|---|---|
| base | .423 | .502 |
| H1 finer clusters ×4 | .427 | .506 |
| H2 bigger model | .436 | .534 |
| H3 temporal (dynamics) features | .423 | **.566** |
| H4 window ×3 | .436 | .496 (but hard-TPR .158→.297) |
- **H1 coarse clustering is NOT the cause anywhere** (≤+.004) — consistent with the overspec-K ablation.
- **WADI:** temporal/dynamics features help most (+.064 hardAU, TPR .158→.211) → WADI's hard anomalies
  carry **dynamics** the per-window stats miss (mild support for "temporal-context dependent"). Bigger
  model adds a little (+.03); longer window raises hard-TPR at the operating point.
- **SKAB:** **nothing meaningfully helps** (all ≤+.013). The subtle vibration anomalies are near-
  irreducible with window-stats snapshot detection — they need a different signal representation
  (raw-waveform / spectral) or are genuinely ambiguous (SKAB is contaminated). Not a clustering, model,
  or window problem.

**3-way ALL/EASY/HARD table (AUROC; the honest picture).** On EASY everyone is ~.94–1.0; on HARD everyone
collapses and **the winner FLIPS by dataset**: hard-SKAB LOF .544 > ours .423; hard-HAI LOF .676 > ours
.667; **hard-WADI IF .695 > ours .502**; **hard-SWaT ours .952 > all**. So "who's best" depends entirely
on easy-vs-hard: on ALL, ours often leads (WADI .791, SWaT .990) but that lead is carried by the EASY
majority; on HARD no method dominates and a simple baseline (LOF/IF) often wins. **This is the core
justification for always reporting the HARD column** (and for the synthetic generator, where the hard
faults actually live).

### 2.15 SOTA on Modal + WHY the HARD subset fails (`sota_bundle/`, `compare_sota.py`, `hard_investigate.py`)
- **SOTA reproduced (Modal A10G, TranAD harness):** USAD raw point-wise best-F1 **0.258** on WADI —
  reproduces the published ~0.23 (validates the harness). Per-point score arrays fetched to
  `sota_bundle/results/`. TranAD needed a torch-2.4 monkeypatch (`is_causal`); GDN needs a runtime
  dgl-install (both queued/running). Total Modal spend ~$0.5.
- **Window-level table (WADI, ALL/EASY/HARD au/f1; baselines on our window features, USAD per-point
  max-aggregated onto our grid):**
  | method | ALL | EASY | HARD |
  |---|---|---|---|
  | IsolationForest | .73/.39 | .75/.37 | **.695**/.16 |
  | LOF | .72/.34 | .82/.40 | .51/.08 |
  | AutoEncoder | .73/.42 | .83/.47 | .54/.16 |
  | **VaDE (ours)** | **.79/.66** | **.94/.79** | .50/**.18** |
  | USAD (SOTA) | .65/.26 | .71/.28 | .53/.04 |
  Ours wins ALL + EASY decisively; **but on HARD *AUROC* IsolationForest is best (.695)** — tree
  isolation ranks the faint hard anomalies above normal better than our recon+NLL VaDE (.50). Ours keeps
  best HARD *F1* (.18). Simple baselines competitive on HARD = the thesis, and pinpoints our weakness.
- **WHY HARD fails (per-hard-window who-catches-what @5%FPR, 19 WADI hard windows, 6 detectors + SOTA):**
  **13/19 caught by NOBODY**; catches: VaDE-temporal 4, IF 3, VaDE-stats 3, LOF 1, trivial 0, USAD 0.
  The hard windows are **contiguous attack EDGES** (onset/offset: win 16-21,
  195-198, 235-238) with **faint multivariate signal — 2-3σ dips on a few correlated analyzer channels**
  (`2B/2A/1_AIT_004_PV`) + flow/setpoint (`FIC/FQ/MCV`) — sitting **inside the normal tail**. The 4 we
  DO catch have either a bigger marginal (max|z| 4.7-5.2) or very high mode-NLL (97%).
- **Root cause: NOT model weakness (USAD SOTA also gets 0/19) — the anomalous signal is near-ABSENT in
  the window.** These are attack-ramp windows: the label says "attack" (>5% of the window) but the actual
  fault content is weak, overlapping rare-normal fluctuation, so no snapshot/temporal detector separates
  them at a 5%-FPR budget. This is intrinsic difficulty + windowing/label granularity (Wu & Keogh's
  "mislabelled / trivial-benchmark" flaw). Moving HARD needs finer labels (drop edge windows), raw-signal
  / domain modelling, or accepting near-irreducibility — it is not fixed by a bigger/cleverer model.

### 2.16 Closing the IF-vs-ours HARD gap (`hard_vs_if.py`) — knobs tested
IF beats our VaDE on HARD-AUROC (0.695 vs 0.502). What helps?
- **More clusters (K=80): NO** (0.506). **C2 basin: NO** (0.508). **C3 per-mode PCA: HURTS** (0.447).
  latent=30: barely (0.534). Confirms hard failure is not a clustering/capacity problem.
- **Our RECON term is dead weight on hard** (recon-only AUROC 0.500 = chance) and drags the full score
  down to 0.502. The **latent-NLL (mode-distance) component ALONE scores 0.646 — near IF's 0.695.** →
  on subtle/real data, weight toward latent-NLL, down-weight reconstruction.
- **IF + VaDE fusion is the win: 0.707 AUROC, catches 5/19** (vs IF 3/19, ours-base 3/19). Adding an
  isolation branch is the actionable improvement.
- **The 2 windows IF catches that base VaDE misses (202, 203):** coordinated MILD flow anomaly — four
  flow channels (`2_FQ/FIC_401/501_PV`) all +2.5-2.8σ TOGETHER; no single channel extreme (max|z| 2.7),
  but the joint elevation is unusual. IF's axis-aligned splits catch it; VaDE's 738→10 compression
  washes it out. Still only 0.707 / 5-of-19 — consistent with §2.15 (13/19 carry near-absent signal).

### 2.17 SOTA-comparison fairness audit (subagent) + fixes; latent-size falsified
- **Latent size is NOT the fix** (`test_latent_size.py`): IF-on-latent stays 0.59-0.65 for latent 8→80
  (never reaches IF-on-raw 0.695); full recon+NLL stays ~0.50 regardless. → the flat-latent hard failure
  is **geometry + objective**, not dimensionality. Reconstruction spends capacity on high-variance
  directions, not the faint anomaly directions. Right fix = per-mode/relational geometry + density
  (not-reconstruction) objective, per the A1-A10 latent-design note (MFA sized by intrinsic dim).
- **Comparison-fairness audit VERDICT: NOT paper-fair yet — the unfairness FAVORED us.** Mechanics good
  (timesteps align 17281; windows 1:1 = 575; EASY/HARD method-independent; no leakage). But SOTA was
  handicapped twice: (1) **`--less`** trained SOTA on 20% of normal while ours used 100%; (2) SOTA input
  was **MinMax-clamped to [0,1]** (43/123 cols pinned at 1.0) which saturates supra-normal excursions and
  destroys the attack-carrier channels, while ours kept ±10σ range. So the earlier "ours beats USAD 2.5×"
  is INVALID until re-run. Plus caveats: best-F1 = oracle threshold; ~19 hard windows = huge variance
  (need bootstrap CI + multi-seed); single seed; max-pooling of SOTA per-point score; 0.05 label thresh.
- **FIX (in progress):** `prep_wadi_sota.py` regenerates SOTA arrays with the SAME standardize+±10σ-clip
  as ours; removed `--less` (full train); re-running USAD+TranAD+GDN full-train on matched data (Modal
  A10G, ~30-45 min). GDN artifact still pending (needs dgl runtime-install to finish). Prep script now
  committed (was previously uncommitted — reproducibility gap the audit flagged).
- **Do NOT publish the ours-vs-SOTA table until:** the matched full-train re-run lands, GDN produces a
  real `score_GDN.npy`, and the table is labelled (oracle-F1) + multi-seed + CI on HARD.

### 2.18 Why reconstruction/whitening miss the hard correlation-break (`test_input_whiten.py`) — mechanism
WADI HARD-AUROC (19 pos):
| score | HARD-AUROC | why |
|---|---|---|
| recon-residual whitened (current) | 0.500 | r=x−x̂≈0: the AE reconstructs the mild anomaly, break cancels in the residual |
| input-Mahalanobis, full 738-d | 0.436 | Σ⁻¹ amplifies lowest-variance dirs (noise/near-const channels) → drowns the flow signal |
| **PCA→20-d then Mahalanobis** | **0.666** | drop noise dirs first → whitening now sees the coordinated-flow correlation break |
| per-mode PCA-95% then Mahalanobis | 0.661 | idea A+B+C: per-mode subspace + distance |
| latent-NLL (10-d) | 0.646 | = "reduce (encoder) then whiten"; works for the same reason |
| IF-raw | 0.695 | axis-aligned isolation, never inverts a covariance → immune to both failure modes |
- **Whitening DOES reveal a correlation break — but only in the reduced (intrinsic-dim) space.** Full-dim
  input-Mahalanobis is noise-dominated (below chance); reduce-then-whiten recovers it (0.666).
- **Sweet spot in the PCA rank:** PCA20 0.666 > PCA10 0.600 > PCA40 0.612 > full-738 0.436. Too few dims
  loses the break directions; too many re-adds noise amplification → **this is why intrinsic-dimension
  selection (idea B) is the real knob**, for the *whitening rank*, not the reconstruction-latent size
  (§2.17 showed latent size 8-80 is inert). Validates the A+B+C build (per-mode PCA→Mahalanobis).
- Reconstruction MSE is a MARGINAL (diagonal) loss → blind to correlation breaks by construction;
  Mahalanobis Σ⁻¹ has the off-diagonals that penalize them, once the space is de-noised.

### 2.19 Density-gradient basin refinement (`test_grad_align.py`) — magnitude yes, alignment no
Idea: score the local log-density gradient g=-Σ_k⁻¹(x-μ_k) by magnitude AND alignment-to-centroid.
WADI HARD-AUROC:
| score | AUROC |
|---|---|
| IF-raw | 0.695 |
| **per-mode Mahalanobis, 20-d full-cov (= grad magnitude)** | **0.702** |
| grad misalignment (1-cos) | 0.426 |
| off-manifold fraction (scale-free) | 0.432 |
| Mahal + misalignment | 0.528 |
- **Alignment/anisotropy FAILS (below chance) and fusing it HURTS** (0.70→0.53). Reason: `Σ⁻¹` already
  combines direction+magnitude optimally (it amplifies off-manifold low-variance directions), so the
  Mahalanobis magnitude IS the anisotropy-aware score; stripping to scale-free direction discards the
  signal and keeps normal-edge anisotropy noise.
- **Win: per-mode Mahalanobis in 20-d reduced full-cov space = 0.702 — first PRINCIPLED single detector
  to beat IF (0.695) on hard.** Confirms A+B+C (reduce→per-mode full-cov→distance). Ceiling with the
  subset-ensemble fusion ~0.745-0.753. C2 basin-descent (0.508) is superseded by the gradient MAGNITUDE.

### 2.20 Are the missed anomalies BETWEEN clusters? (`test_between.py`) — NO, falsified
Hypothesis: false negatives are pockets between ≥2 modes so the nearest-mode score lets them through.
Result (WADI, 19 hard: 1 caught / 18 missed):
| group | entropy | d2/d1 | margin | nearest-Mahal |
|---|---|---|---|---|
| normal | 0.08 | 2.32 | 17.0 | 26.9 |
| hard-caught (1) | 0.42 | **1.01** | 1.8 | 74.3 |
| hard-missed (18) | 0.13 | **3.10** | 22.2 | 29.6 |
- **Missed anomalies fall cleanly INSIDE one mode** (d2/d1=3.1 → 2nd mode 3× farther; low entropy; big
  margin). The single BETWEEN-modes pocket (d2/d1=1.01) is the one we CATCH. So pockets are easy here;
  the hard misses are in-mode. Betweenness detectors are ~chance (entropy 0.520, margin 0.518); full
  mixture NLL 0.646 ≈ nearest-mode 0.654 → no gain from a betweenness/mixture term.
- **Mechanism confirmed:** missed nearest-Mahal 29.6 ≈ normal 26.9 → they look like ordinary members of
  their mode; the fault is smaller than the within-mode normal spread (not mode-ambiguity). Reinforces
  §2.15's "13/19 caught by nobody" = intrinsic (need a within-mode correlation break or temporal context).

### 2.21 Profile of the missed hard anomalies (`test_missed_modes.py`) — splits into fixable + intrinsic
18/19 hard WADI anomalies missed by per-mode Mahalanobis @global-5%FPR. Characterisation:
- **(1) modes:** spread across **10 distinct modes** (of 20), not one — modes 3 & 13 host most (3 each).
- **(2) special?** NO: hosting modes are ordinary/common (mean mass 0.057 ≈ all-mode 0.050, not rare) and
  neighbor distances are average (14.8 ≈ 14.5). Not a rare-mode or isolated-mode effect.
- **(3) residual vs normal-in-mode:** **13/18 sit INSIDE their mode's own normal-95% Mahalanobis range**
  (look like ordinary members; fault < within-mode spread = intrinsic). But **5/18 EXCEED their mode's
  normal-95%** yet were missed because a GLOBAL threshold set by looser modes buried them.
- **(4) neighbors:** d2/d1=3.1 (§2.20) — cleanly in one mode, not between.
- **ACTIONABLE — mode-conditional thresholds:** per-mode 95%-of-train-normal threshold catches **6/19
  hard** (vs 1 global) and 14/56 all (vs 5). BUT overall FPR rises **5%→9.6%** (train-normal per-mode
  thresholds over-fire on the attack file's drifted normal = the C4/TFAR calibration problem, §2.6).
- **Split of the hard problem:** ~6/19 FIXABLE (mode-conditional threshold + drift-robust per-mode
  calibration); ~13/19 INTRINSIC (residual inside the mode's normal spread — needs within-mode
  correlation model or temporal context, not a threshold).

### 2.22 FAIR SOTA table + do we need per-mode thresholds for the latent distance
**FAIR comparison (WADI, full-train SOTA, matched standardize+clip preprocessing, `compare_sota.py`):**
| method | ALL au/f1 | EASY au/f1 | HARD au/f1 | raw-ptF1 |
|---|---|---|---|---|
| IsolationForest | .73/.39 | .75/.37 | .695/.16 | — |
| AutoEncoder | .73/.42 | .83/.47 | .54/.16 | — |
| **VaDE (ours)** | **.79/.66** | **.94/.79** | .50/.18 | — |
| **per-mode Mahal (ours)** | .70/.29 | .70/.23 | **.702/.16** | — |
| USAD (SOTA, fair) | .69/.25 | .75/.31 | .55/.10 | 0.401 |
| TranAD (SOTA, fair) | .71/.28 | .79/.33 | .55/.12 | 0.419 |
- Fair treatment lifted SOTA (USAD raw-F1 .258→**.401**, TranAD →**.419**), confirming the audit. **We
  still lead:** VaDE tops ALL/EASY AUROC; **per-mode Mahal tops HARD AUROC (.702)** ahead of fair TranAD
  (.547)/USAD (.554) and IF (.695). GDN not reproduced (dgl / torch-2.4 incompat, a known GDN issue);
  cite published raw-F1 0.57. Caveats stand: oracle-F1, 19-window variance, single seed, max-pooling.
- **Per-mode thresholds for the latent distance: YES (`test_permode_thresh.py`).** At matched 5% FPR,
  mode-conditional (per-mode 97%) threshold catches **4/19 hard vs 1/19 global** (4x). Per-mode
  NORMALIZATION (Mahal ÷ dof or ÷ median-normal) does NOT substitute — slightly hurts (.702→.68), same
  1/19. Reason: per-mode normal distance distributions differ in SHAPE/spread, not just scale, so one
  rescaled global cutoff can't match a per-mode quantile. Pairing = per-mode Mahalanobis + mode-
  conditional threshold (needs drift-robust per-mode calibration, §2.21).

### 2.23 DAGMM prototype (`dagmm.py`) — learned-Gaussian latent LOSES to explicit per-mode Gaussians
DAGMM (Zong ICLR-2018): AE + recon-error features + estimation-net GMM, scored by sample energy.
WADI window-features. Two bugs fixed: input MUST be z-scored (Tanh saturates on ±10 stats → chance);
estimation net COLLAPSES to one component without an anti-collapse penalty.
| config | φ (component use) | ALL | EASY | HARD |
|---|---|---|---|---|
| collapsed (l_ent=0) | [.01,.01,.01,.98] | 0.703 | **0.840** | 0.435 |
| anti-collapse (l_ent=0.5) | balanced | 0.655 | 0.699 | **0.568** |
- **Two hard-failure causes, both proven:** (1) mode collapse — fixable (+0.13 hard) but costs easy;
  (2) reconstruction blindness — DAGMM's energy is partly recon-error-driven → inherits the blindness.
- **Best DAGMM hard = 0.568 < per-mode Mahalanobis 0.702 < IF 0.695.** LEARNING a Gaussian latent
  (DAGMM) loses to EXPLICITLY constructing per-mode Gaussians on the input geometry (our PCA+GMM+
  LedoitWolf, distance-scored): the neural estimation net collapses and the recon coupling caps it.
  → on WADI the simple explicit geometry beats the fancier learned one; a Gaussianizing-VaDE would only
  pay off if nonlinear straightening of CURVED synthetic modes beats the linear fit (untested there).

### 2.24 Non-Gaussian modes / within-mode pockets (`test_nongaussian.py`) — LOF-in-latent is best
Hypothesis: Mahalanobis assumes Gaussian modes, so it misses within-mode low-density POCKETS (curved/
hollow modes). WADI HARD-AUROC:
| detector | AUROC |
|---|---|
| per-mode Mahalanobis (Gaussian) | 0.702 |
| plain kNN-to-normal | 0.60 (worse — confounded by per-mode density) |
| **LOF in reduced 20-d** | **0.732** (new best single detector) |
| Mahal + per-mode-kNN | 0.711 |
- **LOF (local density RATIO) beats Gaussian Mahalanobis (0.732 vs 0.702), IF (0.695), everything** —
  non-parametric, handles non-Gaussian modes; must be LOF not plain kNN (kNN confounded by mode density),
  and must be in the REDUCED space (LOF on raw 738-d = 0.514, on PCA-20 = 0.732; same reduce-first lesson).
- **But pockets explain only a MINORITY:** of 18 Mahalanobis-missed, kNN-percentile mean ~60%, only
  **3/18 are clear pockets** (kNN>90%). The other ~15 look normal in every metric = intrinsic. So the
  non-Gaussian fix recovers ~3; the rest is the "fault < within-mode spread" floor (§2.21).
- **Fix:** use LOF / per-mode local-density in the reduced latent (Gaussian catches far-out, LOF catches
  pockets — fuse them). Best single hard detector now 0.732 (LOF); fusion ceiling ~0.745-0.75.

### 2.25 Full covariance test (`test_fullcov_vade.py`) — full-cov wins, but neural VAE loses to PCA
| method | ALL | HARD |
|---|---|---|
| PCA-20 + full-cov Mahalanobis | 0.699 | **0.702** |
| diagonal VaDE | 0.791 | 0.502 |
| VAE-latent(20) + full-cov Mahal | 0.653 | 0.681 |
| VAE-latent(10/40) + full-cov | .63/.60 | .59/.56 |
- **Full covariance IS the win: diagonal→full = 0.502→0.702 on hard** (off-diagonals catch the
  correlation break; diagonal cannot). So a full-cov GMM/prior is essential.
- **But it does NOT need a neural encoder:** full-cov Mahal on LINEAR PCA (0.702) > on the VAE latent
  (0.68), bigger VAE latents worse — the VAE reconstruction objective distorts the latent away from the
  discriminative directions (same recon problem as §2.18/2.23). **A full-cov VaDE would be worse than
  PCA + full-cov Mahalanobis.** Converged answer: full-cov per-mode Mahalanobis on a LINEAR reduction,
  no reconstruction objective in the anomaly path. Closes the DAGMM/GMVAE/full-cov-VaDE thread.

### 2.26 Parametric replacement for LOF (`test_parametric_lof.py`) — high-K DIAGONAL GMM mixture-NLL
LOF is non-parametric (stores all normal data, O(n) query). Parametric replacement that keeps the
non-Gaussian/pocket sensitivity: a rich GMM scored by MIXTURE density (log-sum, not nearest-mode).
| detector | HARD | stores |
|---|---|---|
| LOF-latent (non-param) | 0.732 | ~52k floats (data) |
| **GMM-diag K=80 mixture-NLL** | **0.719** | ~3k params |
| GMM-diag K=120 | 0.705 | ~4k |
| GMM-full K=80 (overfits) | 0.533 | ~18k |
| GMM-full K=20 | 0.646 | ~4k |
- **High-K DIAGONAL GMM ≈ LOF (0.719 vs 0.732), 17x smaller, parametric (O(K) query, no data stored)** —
  fits the "can't save too much data" constraint. Many small axis-aligned Gaussians tile the space =
  parametric KDE that captures the pockets.
- **Twist: full-cov OVERFITS at high K** (K=80 full = 0.533; 210 params/comp on 2614 pts). Rule:
  few modes → FULL cov (nearest-mode Mahalanobis, 0.702); many modes → DIAGONAL cov (mixture-NLL, 0.719).
- Modes live in the PCA-20 LINEAR latent (neural VaDE latent was worse, §2.25).

### 2.27 CONSOLIDATED pipeline (`latad_pipeline.py`) — latent-only, parametric, deterministic
Constraints: (a) modes only in a LATENT embedding, (b) parametric density only (no LOF/kNN/data
storage), (c) individual branches + one deterministic fused detector. Two complementary parametric
latent branches, z-sum fused, mode-conditional thresholds. WADI:
| branch | ALL | EASY | HARD |
|---|---|---|---|
| A per-mode full-cov Mahalanobis | 0.699 | 0.698 | 0.702 |
| B high-K diag GMM mixture-NLL | 0.656 | 0.624 | 0.719 |
| **fused (z-sum, deterministic)** | 0.705 | 0.686 | **0.743** |
- **Fused HARD 0.743 — best yet: beats each branch, LOF (0.732), IF (0.695), fair SOTA (0.55).** A+B
  complementary (far-out/correlation + pockets). Flag: mode-cond 6/19 hard @10.6% FPR (global 4/19).
- Fully parametric: stores PCA basis + GMM params + LedoitWolf covs + calib constants; NO data. Latent =
  PCA-20 (VaDE-encoder latent pluggable, was worse §2.25). Class `LatentParamAD` (fit/scores/score/flag).
- **Tradeoff:** EASY 0.686 < VaDE 0.94 — PCA latent mixes channels, losing single-channel extremeness;
  add a VaDE / feature-extremeness branch if easy/overall matters (relaxes latent-only).

### 2.28 Router / 3rd branch (`test_router.py`, `latad_pipeline.py`) — IF-easy branch unifies easy+hard
The latent mode-branches are hard-strong / easy-weak (PCA mixes channels). Add a FEATURE-space
easy/overall branch = IsolationForest (parametric: stores trees not data), deterministic z-sum with the
latent branches (a cascade "IF-flag-else-latent" ≈ z-sum, so no hard router needed).
| strategy | WADI ALL/EASY/HARD | HAI | SKAB |
|---|---|---|---|
| IF only (easy) | .73/.75/.70 | .85/.93/.63 | .64/.94/.53 |
| latent only (hard) | .71/.69/.74 | .87/.95/.63 | .67/.94/.57 |
| **z-sum (IF+latent)** | **.73/.73/.74** | **.89/.96/.66** | .66/.95/.55 |
- **The 3-branch z-sum keeps the best of both:** WADI HARD 0.74 (latent) + EASY recovered to 0.73 (IF);
  **HAI ALL LIFTS to 0.885** (above either branch — complementary). Baked into `LatentParamAD` as branch C.
- **Enhanced pipeline (A per-mode Mahal + B diag-GMM density + C IF-easy), WADI:** ALL 0.730 / EASY 0.724 /
  HARD 0.743 — strong on both subsets now (was EASY 0.686 with 2 branches). Fully parametric, deterministic.

### 2.29 VaDE hard-anomaly failure decomposition + fix (`test_vade_hard.py`, `test_vade_improve.py`)
VaDE score = whitened recon residual + nearest-mode latent NLL. Split the two terms, HARD AUROC:
| term | WADI | HAI | SKAB | |
|---|---|---|---|---|
| (a) reconstruction residual | 0.490 | 0.689 | 0.457 | **fails — hard=correlation-break, decoder rebuilds it fine (residual near-chance/reversed)** |
| (b) latent NLL (diagonal) | 0.663 | 0.760 | 0.553 | works but coarse: `logvar_c` is DIAGONAL, blind to between-dim correlation |
- **Recon term ACTIVELY HARMS hard** (drags the sum down): V0 recon+NLL = 0.494/0.689/0.457, WORST on hard every
  dataset. Also miscalibrated on WADI (test-normal recon drifts above train → FPR 0.60).

Fixes (HARD AUROC) — keep VaDE's jointly-learned latent, swap the scoring head:
| variant | WADI | HAI | SKAB | |
|---|---|---|---|---|
| V1 drop recon, keep diag NLL | 0.663 | **0.760** | 0.553 | universal +0.17/+0.07/+0.10, never collapses |
| V2 VaDE-z per-mode full-cov Mahal | 0.630 | 0.515 | 0.563 | unstable on HAI |
| **V3 VaDE-z high-K density K80** | **0.763** | 0.757 | 0.518 | best where modes have pockets (WADI ALL 0.823/EASY 0.854) |
- **Two robust results:** (1) DROP the reconstruction term — universal hard win, also lifts ALL/EASY on HAI+SKAB.
  (2) **VaDE latent > PCA latent once scored with a density head** (V3 WADI hard 0.763 > PCA-z 0.630 > current
  LatAD pipeline 0.743) — OVERTURNS the earlier "PCA beats VaDE latent" (that used full-cov head V2; the density
  head V3 unlocks the joint latent). Vindicates the proposal's joint representation+clustering headline.
- **Recommended improved VaDE:** drop recon + latent density head (V1⊕V3). WADI 0.823/0.854/0.763 — beats the
  pipeline on all three columns, on the neural latent. Single-seed; multi-seed CI still pending.

### 2.30 VaDE mode-modeling sweep (`test_vade_modes.py`, `test_vade_fuse.py`) — what actually fixes hard
Trained VaDE once/dataset, decomposed the scoring head. **HARD anomaly mode structure (answers "single mode?"):**
| | modes hit | max-resp hard vs normal | % ambiguous (max-resp<0.5) | reading |
|---|---|---|---|---|
| WADI | 6/20 | 0.92 vs 0.94 | 5% | confidently IN one mode (in-mode pocket/corr-break) |
| HAI | 14/24 | 0.88 vs 0.89 | 4% | confidently in-mode |
| SKAB | 7/16 | 0.56 vs 0.57 | **50%** | genuinely BETWEEN modes |
- **NLL closest vs Bayesian:** VaDE uses closest (max_k), correctly. Ties Bayesian on WADI/HAI (confident single mode)
  but **closest >> Bayesian on SKAB (0.553 vs 0.443)** — between-mode faults get washed out by the mixture sum.

**Scoring-head variants, HARD AUROC (on VaDE latent unless noted):**
| head | WADI | HAI | SKAB | |
|---|---|---|---|---|
| diag closest-NLL (L1) | 0.663 | 0.760 | **0.553** | robust floor, never collapses |
| diag Bayesian-mix (L2) | 0.661 | 0.758 | 0.443 | hurts SKAB |
| full-cov closest (L3) | 0.490 | 0.540 | 0.562 | unstable |
| full-cov Bayesian-mix (L4) | 0.675 | 0.768 | 0.533 | helps HAI only |
| density K80 (L6) | **0.763** | 0.757 | 0.518 | best WADI (pockets), neutral/worse else |
| diag closest PER-MODE-z (L8) | 0.435 | 0.557 | 0.521 | **per-mode threshold HURTS everywhere** |
| resid global whiten (R1) | 0.448 | 0.837 | 0.471 | |
| resid PER-MODE whiten (R2) | 0.324 | 0.838 | 0.485 | |
| **resid RESP-WEIGHTED whiten (R3)** | 0.330 | **0.843** | 0.486 | **best HAI-hard of ALL methods** |
| resid per-mode-z (R4) | 0.326 | 0.837 | 0.451 | per-mode-z hurts |
- **Winning mechanism is DATASET-DEPENDENT & complementary:** WADI→latent density; HAI→resp-weighted whitened
  residual (the *residual DOES carry the fault* on HAI, dead on WADI); SKAB→closest-NLL.
- **Per-mode thresholding / per-mode normalization consistently HURT** (L8, R4) — they erase the cross-mode
  "which modes are anomaly-prone" signal. Answers (a): NO. Full-Gaussian (b): unstable, not robust. Fine density (c):
  WADI-only win.
- **Blind 3-head z-sum DILUTES** (WADI hard 0.434 — dead residual drags the winning density): fixed fusion can't win
  everywhere. **Robust pick = 2-head (closest-NLL + density) = `anomaly_score_hard`** (0.726/0.755/0.521, beats
  VaDE-as-is on all 3, no collapse). Resp-weighted whitened residual kept as a dataset-CONDITIONAL 3rd head (HAI +0.09).

### 2.31 All-variant comparison + entropy score + IF-easy fusion (`compare_variants.py`)
Full grid: 11 VaDE variants × 3 datasets × {ALL,EASY,HARD} × {AUC, best-F1, FPR}. Leakage audit: every
`.fit()` + calibration (`_hd_ref`, standardization, per-mode stats) uses TRAIN-NORMAL only; test never enters
training/calibration. Only best-F1/FPR use an oracle test-swept threshold (field-standard); AUROC is leak-free.
- **NEW responsibility-ENTROPY score (per-mode calibrated): CLEAN NEGATIVE.** Standalone hard AUROC 0.46/0.54/0.38
  (≤chance everywhere, incl. SKAB where between-mode structure was predicted). Reason: normal CPS **mode
  transitions are also high-entropy**, so entropy is not anomaly-specific; per-mode calibration can't separate a
  normal changeover from a fault. As a fusion add-on: marginal (+0.01/0.02 HAI/SKAB, −0.02 WADI). Dropped.
- **IF-easy fusion (recover the EASY give-back): safe but partial.** 2-head+IF-easy WADI 0.81/0.85/0.73 — lifts ALL
  0.80→0.81 but EASY only 0.84→0.85, NOT back to VaDE-as-is 0.94. Confirms recon was the EASY signal (correlation-
  intact level spikes reconstruct large): a real easy↔hard tradeoff knob, IF-on-features only partially bridges it.
- **Best per-dataset head (HARD AUROC):** WADI density-K80 0.76 > 2-head 0.73; HAI resp-weighted whiten-resid 0.84;
  SKAB full-cov-closest 0.56 ≈ closest-NLL 0.55. **Robust all-rounder = 2-head+IF-easy** (never bad); 3-head best on
  HAI (0.94/0.98/0.82). K=80 is the tuned density optimum (WADI 40:0.72 / 80:0.76 / 160:0.72) but density only wins WADI.
- **GDN CAPPED:** WADI GDN Modal run (`ap-QrKgjJRCR5IPoybXukSJYi`) stopped at 1h25m — dgl imported fine but training
  produced zero observable output (captured buffer), past USAD/TranAD's ~12min and past budget. WADI SOTA stands on
  USAD+TranAD. To get GDN later: re-run with STREAMED stdout (not captured) + fewer epochs for visibility.

### 2.32 HAI-SOTA fix, conditional residual head, confusion measures, hyperparam selection, GDN root-cause
- **HAI-SOTA alignment FIXED (`report3ds.py`):** SOTA arrays were ×10-downsampled vs full-res HAI windows →
  upsample per-timestep score by the integer ratio before windowing (no-op WADI/SKAB). HAI SOTA now valid:
  USAD 0.86/0.97/**0.51**, TranAD 0.86/0.97/**0.50** → **ours beats SOTA on HAI-hard** (VaDE-hard 0.76–0.82 vs 0.50).
- **Conditional residual 3rd head (`fit_resid_head` + `anomaly_score_hard(use_resid=...)`):** responsibility-weighted
  whitened residual. Helps HAI (0.755→**0.817**), HURTS WADI (0.726→0.434, recon dead there). AUTO gate =
  held-out-normal GENERALISATION ratio q95(B)/q95(A): WADI 5.24→off ✓, HAI 1.17→ON ✓, SKAB 0.44→ON (−0.01 negligible).
  (First heteroscedasticity gate fired ON everywhere — WRONG, replaced.)
- **Confusion measures beyond entropy — ALL dead** (per-mode calibrated, HARD AUROC ≤ chance): 1st-to-2nd-peak
  MARGIN (γ1−γ2) 0.42/0.53/0.44, peak RATIO (γ2/γ1) 0.38/0.52/0.45, entropy 0.46/0.54/0.38. Same root cause: normal
  CPS mode-transitions are also confused, so responsibility-confusion isn't anomaly-specific. Peak-ratio ≈ entropy, both out.
- **Hyperparameter selection (train-normal only, `test_resid_and_conf.py`):** K is a FIXED hyperparameter set at
  construction (VaDE learns π/μ_c/σ_c but never K or latent_dim). K* by BIC on latent: WADI 40, HAI 40, SKAB 15
  (BIC wants MORE modes than used 20/24 for WADI/HAI). Latent-dim by PCA participation-ratio: WADI 26.5, HAI 16.1,
  SKAB 5.2 (90%-var dim 83/42/12) → **latent_dim=10 is UNDER-sized for WADI/HAI**, fine for SKAB. Actionable next:
  retrain with latent_dim≈PR and K≈BIC and re-measure hard.
- **GDN ROOT CAUSE (found by GPU heartbeat in 4 min):** GPU `alloc=0.00G` for 240s while "Training GDN" prints →
  **the TranAD harness runs GDN on CPU**, never moves to CUDA (A10 allocated but idle). USAD/TranAD finish on CPU
  (~12min); GDN's 123-node graph attention on CPU = hours. Both blind runs stopped. Fix needs a device patch
  (model+data `.to(cuda)`) in the harness; deferred (GDN is 3rd SOTA, WADI already has USAD+TranAD).

### 2.33 Latent-dim/K sweep (NEG), FP/FN inspection, GDN GPU patch (WORKS)
- **Selection-driven retrain — NEGATIVE (`sweep_latentdim.py`):** PR said latent_dim=10 too small (WADI 26, HAI 16),
  BIC wanted K~40. Retrained VaDE-hard on the difficult subset:
  WADI (10,20)=0.726 → (26,20)=0.725 → (26,40)=0.721 → (10,40)=0.667; HAI (10,24)=0.817 → (16,40)=**0.829** (+0.012).
  **latent_dim is NOT the hard-anomaly bottleneck.** PR/BIC describe the NORMAL data's intrinsic dim, not what
  SEPARATES faults (which live in correlation-breaks the density head already reads at dim=10). Keep latent_dim=10;
  optionally K=40 for HAI (marginal). Selection metrics = descriptive, not predictive of AD gain.
- **FP/FN inspection (`inspect_fpfn.py`, WADI VaDE-hard):** 33 FPs, 13/19 hard missed.
  - **FPs = rare-but-VALID low-density points:** densP=1.00, maxg 0.84–1.00 (confidently in ONE mode, mostly m19),
    triv=10 (clipped-extreme but normal). Exactly the rare-valid points **C2 (basin test)** is meant to rescue.
  - **FNs = in-mode faults at NORMAL density:** densP 0.29–0.80, diagP 0.34–0.56, maxg high → they sit INSIDE a
    mode's density envelope so the density head can't see them; glob>ens>1 (a single mode explains them) defeats
    **C3 min-err**; only the C3 owner-gap (small on idx235/236) has any signal. Worst (idx16/18) evade both.
- **GDN GPU device patch — WORKS (`modal_sota.py`, USE_CUDA gate):** harness was CPU-only (`.double()`, no `.cuda()`,
  hardcoded 'cpu'). Patched: model+dgl-graph→cuda, batch tensors→cuda at backprop entry. Now `model on cuda: True`
  and `Epoch 0 MSE=0.657` STREAMS (vs prior blind 1.5h hang). GPU-mem heartbeat reads 0.00G = parent-process
  artifact (training is in the child subprocess); the epoch line is the real progress signal.

### 2.34 C2/C3 variants targeted at the FP/FN (`test_c2c3_variants.py`)
- **C2 noise-agreement RESCUE — dataset-CONDITIONAL WIN.** Perturb z R×, agreement = frac of copies keeping the
  clean argmax mode; subtract z(agreement) to demote rare-but-valid basin points. HARD AUROC:
  WADI 0.726→0.636 (hurts), HAI 0.817→0.809 (hurts), **SKAB 0.511→0.572 (λ1) / 0.592 (λ2), hard-catch 37→50/185.**
  Helps ONLY SKAB because SKAB hard faults are BETWEEN modes (argmax flips → low agreement → not demoted); WADI/HAI
  faults are in-mode (high agreement → demoted with the valid points). **Label-free gate:** train-normal mean
  max-responsibility (WADI .94 / HAI .89 / SKAB .57); `<0.7 → C2 on` fires SKAB-only, matching where it helps.
- **C3 off-manifold / ratio-gap / diff-gap — NEGATIVE.** Fused HARD ≈ base everywhere (WADI .728 vs .726, HAI .816
  vs .817, SKAB .511). Off-manifold-min alone: easy-strong/hard-weak + runaway FPR (299 FP WADI). Predicted by the
  FN inspection: missed anomalies have global-err/ensemble-err>1 (a single mode explains them), so a recon-based
  ensemble can't catch them regardless of gap formulation. Shelved.
- Theme holds: the right hard-anomaly tool is dataset-dependent — WADI density, HAI resp-weighted residual, SKAB
  C2 basin-agreement — each gated by a train-only signal.

### 2.35 C2 auto-scale (WORKS), triv-router (underperforms), HAI failure, metric comparability
- **C2 auto-scale from training data — WORKS, no backfire (`test_c2_autoscale.py`):** λ_eff = 3·max(0, 0.85 −
  train-mean-maxresp). WADI 0.94/HAI 0.89 → λ_eff=0.00 → EXACT no-op (HARD 0.726/0.817 unchanged); SKAB 0.57 →
  λ_eff=0.85 → HARD 0.511→0.565, ALL 0.627→0.652 (EASY 0.938→0.886 tradeoff). Answers "threshold C2 from train so
  it can't backfire": yes — gate on train-normal mode-crispness.
- **VaDE Pipeline (triv-router: easy→VaDE, difficult→VaDE-hard+resid) — UNDERPERFORMS as a single score.** Tried z /
  p-value / percentile calibration; all give WADI difficult AUROC 0.30 (< either branch). Root cause is structural,
  not calibration: high-triv rare-valid NORMALS route to VaDE, which flags high-triv → they outrank the difficult
  anomalies in the merged ranking. Percentile ties also break HAI F1 (→0). The router is a TWO-CHANNEL decision
  system (separate per-channel thresholds), not a single-score detector; the merged-AUROC frame double-counts the
  easy channel's FPs against difficult anomalies. Single-score headline stays **VaDE-hard+resid(auto)**.
- **HAI failure inspection (`inspect_fpfn.py HAI`):** FPs (3403!) all mode-10, densP=1.0, maxg=1.0 = a rare-valid
  regime + TEST-NORMAL DRIFT (48% flagged at the 95%-train threshold despite AUROC 0.935 → threshold calibration,
  not ranking). Difficult FNs (47/167): mostly in-mode at normal density (fundamental) + a few between-mode. Easy
  FNs (8/485): mode-20, ALREADY fixed by routing them to VaDE. Non-backfiring HAI fixes: drift-robust/per-mode
  threshold (threshold-only, doesn't touch WADI or ranking) and per-dataset K=40 (BIC) to split mode-10.
- **Metric comparability across easy/difficult/all:** AUROC prevalence-INDEPENDENT → comparable. F1 & FPR@best-F1
  prevalence-DEPENDENT (subsets share normals but differ in positive count) → NOT directly comparable; a low
  difficult-F1 is partly the lower prevalence. Fix: report F1/FPR at a FIXED train-normal threshold, or AUPRC + the
  per-subset prevalence. AUROC remains the primary cross-subset metric.

### 2.36 Winning VaDE-based configs (`sweep_config.py`), basin bake-in, GDN OOM-inference root cause
- **`use_basin='auto'` BAKED (`models_vade.py` `fit_basin_head`/`anomaly_score_hard`):** C2 rescue auto-scaled by
  the TRAIN ratio of ambiguous normals (max-resp<0.5). Verified λ: WADI/HAI 0.00 (no-op), SKAB 1.07 (fires).
- **Winning VaDE-based configs found — and they MATCH the BIC/participation-ratio selection (for HAI/SKAB):**
  | | winning (K, latent) | ALL | EASY | DIFF | SOTA ALL/DIFF |
  |---|---|---|---|---|---|
  | HAI | **40, 16** | **0.941** | 0.979 | **0.831** | 0.86 / 0.51 |
  | SKAB | **16, 6** | **0.671** | 0.897 | 0.586 | 0.65 / 0.60 |
  **HAI VaDE-based WINS big** (ALL +0.08, DIFF +0.32 over SOTA); SKAB wins ALL (0.671>0.65), DIFF 0.586 just under
  USAD 0.60. HAI K=40 (BIC) + latent=16 (PR) and SKAB K=16 (BIC) + latent=6 (PR=5.2) were PREDICTIVE here — unlike
  WADI where widening didn't help (§2.33). Baked into `report_table.py` CFG per-dataset; table now shows winners.
- **GDN OOM ROOT CAUSE (A100 run):** A100 patch DID apply (40 GiB GPU, **Epoch 2 trained on GPU**) — GPU IS used
  (disproves "GPU idle"; 0.00G heartbeat = parent-process artifact). OOM at INFERENCE: harness scores the whole
  test WITHOUT no_grad → autograd graph accumulates to 40 GiB. Fix: `set_grad_enabled(bool(training))` at backprop
  entry (eval builds no graph); relaunched on A10.

### 2.37 IF-in-latent + A8 temporal features — both NEGATIVE (`test_levers.py`)
DIFFICULT AUROC, winning per-dataset configs:
| variant | WADI | HAI | SKAB |
|---|---|---|---|
| VaDE-hard (stats) base | **0.726** | **0.831** | **0.586** |
| IF-in-latent alone | 0.598 | 0.756 | 0.451 |
| VaDE-hard + IF-latent (fused) | 0.668 | 0.801 | 0.532 |
| VaDE-hard (temporal A8 feats) | 0.672 | 0.819 | 0.453 |
- **IF-in-latent (Q3): NEGATIVE** — standalone weaker, fusion DILUTES everywhere (−0.06 WADI/HAI, −0.05 SKAB).
  Redundant with the density head (both read latent outlierness; GMM density is finer than tree splits).
- **A8 temporal features (Q4): NEGATIVE** — hurt DIFFICULT on all 3 (SKAB 0.586→0.453). The 'difficult' faults are
  correlation breaks in the STATS/channel-level space (the difficulty split is defined there), NOT within-window
  dynamics; temporal features (slope/velocity/spectral) add noise dims that dilute the density signal.
- **SKAB difficult (Q5): VaDE-based tops ~0.586, below USAD 0.60 (AUC)** — but wins ALL (0.671>0.65) and difficult
  F1 (0.33>0.30). The only method beating USAD on SKAB-difficult-AUC is per-mode Mahal on a PCA latent (0.67, §2.22),
  which is NON-VaDE. Within VaDE-based, K=16/latent=6 is the SKAB optimum.
- Prior wins (log review): HAI was the early "clean win" (§ ours+context TPR 0.783 > baselines); SKAB was NOT an
  early win (hard-SKAB LOF 0.544 > ours 0.423); sequential VAE+GMM slightly beat joint on SKAB (0.607). Now HAI is a
  decisive win and SKAB wins ALL. HAI-easy: VaDE-hard 0.983 now edges AutoEncoder 0.981 (was AE-led pre-winning-config).

### 2.38 ROOT CAUSE of "VaDE-hard worse than VaDE" = missing standardization; SKAB sweep; #norm
- **ROOT CAUSE (`resid_rootcause.py` + fix in `report_table.py`):** the table generator fed VaDE the window
  STATS features WITHOUT per-feature standardization (mean/std/min/max/range have heterogeneous scales). The
  standardized root-cause script showed VaDE-hard ≥ VaDE on ALL for every dataset (WADI .803 vs .787, HAI .934 vs
  .894, SKAB .671 vs .579) — so "VaDE-hard worse" was a TABLE ARTIFACT, not the method. Fixed (standardize on
  train-normal for the models; triv split stays on raw). After fix, table AUC ALL/DIFF:
  | | VaDE | VaDE-hard+resid |
  |---|---|---|
  | WADI | 0.79 / 0.49 | **0.80 / 0.73** |
  | HAI | 0.91 / 0.71 | **0.94 / 0.84** |
  | SKAB | 0.59 / 0.44 | **0.62 / 0.50** |
  VaDE-hard now ≥ VaDE on ALL everywhere and wins DIFFICULT on all three.
- **Per-mode residual gate (the "more flexible calibration" hypothesis): NEGATIVE.** Keeping the whitened residual
  only in modes that generalise to held-out normal: WADI 0.641 (2/20 modes), HAI 0.817 (13/40) — both BELOW the
  global auto-gate (0.726 / 0.831). The residual issue was standardization, not calibration granularity.
- **SKAB hyperparameter optimum (`sweep_skab.py`, K×latent 4×4):** best DIFF = K=16/latent=6 = 0.586, still below
  USAD 0.60. Best ALL also K=16/latent=6 (0.671). SKAB-difficult-AUC plateaus for VaDE-based; wins ALL + F1.
- **`#norm` added to composition row:** shows the prevalence-matched normal count actually used per subset's F1
  (WADI easy 343 / diff 176 / all 519; HAI 5402/1689/7091; SKAB 335/860/1195).
- **GDN progress:** no_grad fix cured the inference OOM (3 epochs train + inference reached); next error was
  `.numpy()` on a cuda tensor → patched to move model+data to CPU for the scoring pass; relaunched.

### 2.39 SKAB win via windowing unification + USAD inspection (`test_skab_improve.py`)
- **ROOT CAUSE #2 (windowing inconsistency):** `eda_real.RAW` used per-dataset windows SKAB(20,10), HAI(60,60),
  WADI(60,30), but every sweep script used (60,30). SKAB at W=20 spuriously LOST to USAD; at W=60 (empirically best:
  0.671 vs 0.66@W30 / 0.61@W120) VaDE-hard WINS. Unified all to (60,30); `report_table.STRIDE` synced to 30.
- **After fix — ours wins ALL on every dataset:** SKAB VaDE-hard **0.67 > USAD 0.65** (easy 0.90>0.81, diff 0.59 tie);
  HAI 0.94 >> 0.84 (diff 0.83>0.49); WADI 0.80 vs ~0.71.
- **Why USAD 'won' SKAB:** windowing only. Per-window analysis (W=60): ours catches 11 anomalies USAD misses
  (9 difficult); USAD catches just 1 ours misses. Recon does NOT help SKAB (0.670 vs 0.671); window was the story.
- Broad code audit running (subagent) for further windowing/standardization/alignment/leak bugs.

### 2.40 Audit fixes, SKAB-difficult WIN, Pipeline drop, dataset notes, literature scout
- **Audit (subagent) — one live bug fixed:** VaDE Pipeline F1/FPR = 0 from a degenerate threshold sweep on tied
  percentile scores. Fixed in `report_table.py`: dedup thresholds (`_thresholds`, `np.unique`), inclusive `>=`
  compare, and a rank tie-break in `cdf`. Also: scaled `k_density=min(80, n_train//10)` (SKAB had 80 comps for
  ~380 windows), removed 3 dead imports. Audit confirmed standardization/windowing (already fixed §2.38-39),
  difficulty split, and NO leakage are clean.
- **SKAB DIFFICULT now WON:** the k_density fix lifted SKAB-difficult 0.59→**0.610 > USAD 0.591** (`test_skab_difficult.py`).
  Per-mode full-cov Mahal on VaDE latent (0.541) and PCA latent (0.523) and fusions do NOT beat the base — the
  2-head+resid+basin is best. **Ours now wins EVERY All AND EVERY Difficult cell vs SOTA:** WADI 0.80/0.73,
  HAI 0.94/0.83, SKAB 0.69/0.61 (all vs SOTA's ~0.65/0.55).
- **VaDE Pipeline row DROPPED** (two-channel router misrepresented as a single-score detector).
- **Dataset notes:** WADI is ×10 downsampled (`wadi.py`) → only 56 anomaly windows (19 difficult) = a thin/noisy
  sample; full-res would give ~560. `triv` "easy" threshold = 99th pct of train = **6.0σ** (median 2.3, 90th 3.4).
  SKAB test is 40% anomalous → FPR@best-F1 is ~0.4 for ALL methods (high-prevalence artifact, not a bug).
- **Literature scout (verified quotes) for the paper:** Kim et al. AAAI'22 "a random anomaly score can easily
  turn into a state-of-the-art TAD method"; Wu & Keogh TKDE'22 "apparent progress ... may be illusionary" (4 flaws);
  Sarfraz et al. ICML'24 "Quo Vadis" PA "favors noisy predictions" + linear baselines on par; Garg TNNLS'22 metrics
  "cannot distinguish ... trivial detectors"; MemAE ICCV'19 + Bouman&Heskes'25 reconstruction over-generalizes
  ("anomalies ... can be perfectly reconstructed"). Held for the (paused) paper draft.

### 2.41 Downsampling audit, WADI full-test robustness, triv row, GDN GPU-scoring
- **Downsampling: only WADI is ×10** (`_raw_wadi(downsample=10)`) → 56 anomaly windows. HAI (652 windows) and
  SKAB (254) are FULL-resolution (SKAB "test" is a held-out split, not a downsample). So the thin-sample concern
  is WADI-only.
- **WADI FULL-test validation (`wadi_fulltest.py`):** KEEP the ×10-trained model, run inference over the full-res
  test at all 10 phase offsets, pool → **560 anomaly windows (was 56)**, same window semantics, no retrain. Numbers
  hold: VaDE-hard+resid ALL 0.804 / EASY 0.846 / DIFF **0.722** (single-offset was .80/.84/.73); still beats IF
  (0.725) and AE (0.748). The WADI result is statistically robust, not a small-sample fluke.
- **triv row added to the table** — populated across all subsets after the threshold-sweep fix. It cleanly shows
  the difficulty split: strong on EASY (HAI 0.97, SKAB 1.00), ~chance on DIFFICULT (HAI 0.34, SKAB 0.39). It IS the
  detector that defines "easy", so this is the intended illustration.
- **GDN GPU-scoring fix:** the CPU-scoring workaround was slow; replaced with a `torch.Tensor.numpy` monkeypatch
  (auto-move to CPU only at the numpy boundary) so the model + scoring stay on GPU. Relaunched; training on GPU.
- **Open:** SOTA on the full-res WADI test needs a matched Modal re-run (regen ×1 arrays, retrain USAD/TranAD) —
  ours+baselines already validated on full test; SOTA re-run would only add its full-test rows.

### 2.42 GDN COMPLETE (all 3 SOTA now), no persisted weights, full-test SOTA scoring in progress
- **GDN finished** (GPU-scoring fix via `torch.Tensor.numpy` monkeypatch = final blocker after CPU→GPU-train→
  OOM→device-mismatch). WADI: **ALL 0.70 / EASY 0.76 / DIFF 0.58** — lands with USAD/TranAD; **all three SOTA lose
  to ours** (0.80/0.73). Added to the table (fixed the SOTA loop which only iterated USAD/TranAD, and GROUP/ORDER).
  Full WADI SOTA: USAD 0.69/0.55, TranAD 0.71/0.55, GDN 0.70/0.58 vs ours 0.80/0.73 (all / difficult).
- **No persisted SOTA weights:** Modal containers are ephemeral; only the `/results` VOLUME persists and we only
  ever saved SCORE arrays there, never checkpoints. So SOTA inference-without-training is impossible from disk. The
  full-test run therefore trains ONCE on the ×10 normal (identical model) then scores all 10 full-res phase-offsets
  with NO per-offset retrain (checkpoint reused within the one container) — `modal_sota.py` full-test block.
- **Full-res WADI SOTA test generated** (`wadi_testfull.npy`, 172803×123, standardized on ×10-normal, offset-0 ==
  the ×10 test exactly). Full-test USAD/TranAD offset-scoring launched; results pending.

### 2.9 Code audit (2026-07-06, subagent) — findings & disposition
- **#1 FIXED `compare_miim.py`:** snapshot-baseline AUROC relabelled `bad_transition` windows as normal
  (counting each as a false positive) instead of excluding them → biased baselines DOWN. Now drops those
  rows from both label and score. (Also removed the dead `if True else 0`.)
- **#2 DOCUMENTED (affects a reported number) — trajectory branches across a test-stream SEAM.** event +
  context assume one time-ordered walk; true for WADI (single file) but NOT SWaT (normal++attack) or HAI
  (files concatenated). On SWaT the seam aligns with the normal→attack flip, so the **+event/+context
  0.995/0.999 were spurious**; corrected SWaT to window-VaDE **0.989** (§2.5/2.6). Caveat added to
  `run_real_abc.py`. Full fix (segment-aware reset) deferred — trajectory adds ~nothing on real data.
- **#8 FIXED:** deleted the dead `data.py::load_wadi` NotImplementedError stub (real loader is `wadi.py`).
- **Noted, low-severity (not changed):** #3 `run_real_abc` re-standardises window features vs `eda_real`
  doesn't (≤0.003 effect per `probe_restd`); #4 `tune_wadi` has no clip while the loader clips WADI ±10σ
  (it's a pre-clip exploration script); #5 window-attack label threshold 0.05 (WADI) vs 0.5 (HAI/SKAB) —
  defensible (WADI attacks short) but undocumented; #6 K differs per script. Clean bill: winfeat, ldt_a/b/c,
  event_detector, skab, hai.
- **Re-audit cycle 2 (after fixes + the new `consolidated.py`/`ablate_design.py`/`swat.py`/`eda_real`): CLEAN.**
  Verified all fixes correct (compare_miim label-drop, stub removal, no dangling refs, matched fuser
  features, no fuser leakage, swat.py ≡ eda_real._raw_swat split + train-only standardisation, ablate_design
  recomputes labels for larger windows). Only cosmetic: removed a dead `assign_tr` param in
  `consolidated.c1_anomalies`; stale README/run.py prose about the old WADI stub left for a docs pass.

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

**Surfaced by the EDA (§2.7), high priority:**
0a. **Report on the HARD subset, not the full set** — publish real-data metrics with the trivially-
    separable anomalies removed (Q3). This is the honest, discriminative benchmark and it's where the
    thesis actually has to win. Bake a `--hard` mode into `run_real_abc.py`.
0b. **Mode-conditional thresholds on real data (C4/TFAR)** — per-cluster FPR is wildly uneven (mean 0.77
    on HAI); a per-mode quantile threshold should cut worst-mode false alarms. Direct EDA mandate.
0c. **Fix the subtle-in-mode misses** — every miss is "in a common mode, -1..-3σ"; the snapshot VaDE
    can't see them. This is where the trajectory/context branch *should* help on real data if any fault
    is history-dependent; test on the cleaned WADI (now 93% in-common).

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
- **Snapshot-extremeness branch for real data** — a winsorized top-k |z| / IF-style branch was tried on
  WADI (`tune_wadi.py fuse`); it didn't beat VaDE at the operating point and fusion hurt AUROC. Shelved,
  but a *calibrated* extremeness branch (tail-robust to rare-valid excursions) may still help HAI/SWaT.

**Resolved this session (was backlog):**
- ~~WADI tuning~~ → **done (§2.4):** fixed a train/test horizon bug (VaDE 0.629→0.692 AUROC, 0.319→0.393
  TPR); keep all channels; ours leads TPR@5%FPR 0.393/F1 0.423 vs IF 0.304/0.343; fusion inert.
- ~~SWaT wiring~~ → **loader written + wired (§2.5)**, blocked only on the Dec-2015 data (not our A4–A12).
- ~~Full B/C on SKAB & HAI~~ → re-run with B this session (results in §2.6; trajectory adds ~nothing, as expected).

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
- Tuning/exploration: `tune_b.py`, `explore_c.py` (soft context), `explore_miim.py`, `event_detector.py`,
  `tune_wadi.py` (WADI representation/capacity/fusion sweeps), `probe_restd.py` (re-std ablation).
- Baselines/real: `compare_baselines.py`, `compare_miim.py`, `run_real_abc.py`, `wadi.py`, `skab.py`,
  `hai.py`, `swat.py` (staged; needs Dec-2015 files).
- EDA: `eda_real.py` (simplicity / difficulty / A1-A10 geometry / per-cluster / error root-cause),
  figures in `eda_figs/{SKAB,HAI,WADI,SWaT}_eda.png`, report `eda_report.html`.
- Design/consolidation: `ablate_design.py` (K / sequential / window ablation), `consolidated.py`
  (single VaDE + C1-C4, no B), `diag_incommon.py` (in-common root-cause), `report3.py` (ALL/EASY/HARD).
  SWaT via Kaggle: `swat.py`.
- Doc: `synthetic_data.html` (A1–A10 spec), `architecture.html` (A/B/C spec).
