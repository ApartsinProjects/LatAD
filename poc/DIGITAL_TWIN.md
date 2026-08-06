# Latent-space digital twin — design & training protocol

Branch `latent-digital-twin`. Goal: move from *window/snapshot* anomaly detection
to *dynamics/trajectory* anomaly detection by learning a **predictive state
encoder** — a latent "digital twin" whose state absorbs history and predicts the
future. This is the unification of three threads the synthetic/real work surfaced:
assumption **A8** (multiscale temporal dynamics), the **slow state-machine /
invalid-transition** problem, and the finding that snapshot encoders (MLP-stats,
Path-B conv) cannot see history.

## Why (motivation)
A snapshot (even a long window or a conv over it) has **no memory of the trajectory
that led here**. Two anomaly classes are invisible to any snapshot:
- a value normal *now* but reached via an **invalid path** (state transition), and
- a **slow drift** whose evidence spans far more than one window.
The fix is a learned **sufficient statistic of unbounded history** at bounded
dimension — a recurrent/state-space state `s_t`, made meaningful by a **prediction**
objective.

## Architecture (causal)
- per-step encoder `e(x_t)`;
- state model `s_t = f(s_{t-1}, e(x_t))` (GRU / temporal-conv / SSM / causal Transformer);
- multi-horizon prediction heads `g_h(s_t)` for `h ∈ {1,4,16,64,...}` (and/or
  downsampled/averaged futures) — multiscale = A8.

## Objective (self-supervised, NORMAL data only)
`L = Σ_h λ_h · ℓ(target_{t+h}, g_h(s_t))`, with one of:
- **A regression** (obs space, whitened residual) — simple, interpretable surprise;
- **B contrastive (CPC/InfoNCE)** in latent space — avoids blurry-mean & collapse;
- **C probabilistic (RSSM)** — predict `p(z_{t+h}|s_t)`, NLL/ELBO, connects to VaDE.

## The three make-or-break pitfalls
1. **Persistence shortcut** (copy `x_t`): use large horizons + predict changes.
2. **Blurry-mean L2**: prefer contrastive (B) or probabilistic (C) for branching dynamics.
3. **Latent collapse** (constant state): stop-gradient/EMA target (BYOL), CPC negatives, or VICReg.

## Data protocol
Normal only; reset state at run/file boundaries; burn-in context before scoring;
per-channel standardization on train; teacher-force the state, open-loop the heads.

## Optimization & (label-free) model selection
Adam ~1e-3, grad clip, seq len ~100–500. Select state dim / context / horizons by
**held-out normal prediction loss** + detection on **C1-generated anomalies** — no
real labels needed.

## Inference — anomaly scores (freeze encoder, run causally)
- **prediction surprise** — whitened `||x_{t+1} - g_1(s_t)||²` (multi-horizon agg);
  spikes at onsets → suits SKAB's NAB metric.
- **off-manifold state** — nearest-mode density of `s_t` (VaDE on normal states).
- **invalid transition** — low `p(s_t | s_{t-1})` under the dynamics head.
Combine by standardizing each on normal and summing.

## Hook into the framework
Freeze twin → extract states `{s_t}` → fit **VaDE mode model on states** (modes =
operating regimes/attractors) → layer components: **C1** generate anomalous rollouts
(valid trajectory + spliced invalid transition), **C2** basins in the dynamics,
**C3** ensemble of per-regime predictors, **C4** per-regime surprise control.

## Grounding (compose, don't reinvent)
Contrastive Predictive Coding (van den Oord 2018); recurrent state-space / world
models (RSSM, PlaNet/Dreamer); forecasting-based CPS AD (OmniAnomaly, GDN, TranAD);
classical state-space/Kalman; digital twins. Novel angle: MIIM mode/regime structure
+ the four components on top of the latent dynamics.

## Plan (this branch)
1. `twin.py` — GRU state + 2-horizon prediction head; train on HAI normal.
2. Score = whitened prediction surprise; evaluate on HAI (vs static-stats baseline).
3. Add nearest-regime density on `s_t` (VaDE on states) + invalid-transition prob.
4. If it beats static-stats on HAI, add C1 rollouts / C3 per-regime predictors.
5. SKAB under its own F1/NAB protocol (onset detection is what NAB rewards).
