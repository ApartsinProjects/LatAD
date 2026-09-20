# LatAD revision2 reframe plan (option B: typing as a dedicated section)

Verified state after the SWaT leak fix + stale-expert catch + drift/changepoint typing.
Spine unchanged (community-factorized latent density + HAI clean win); drift-aware typing
is its own section, not the headline.

## Contributions
- C1. Community-factorized latent-density detector: VaDE latent Gaussian mixture (normal-only)
  + correlation-based regime communities + cohesion-weighted Higher Criticism fusion; localizes
  a violation to its subsystem.
- C2. Hardened, detector-agnostic difficulty stratification: channel-based max|z| (difficult) and
  PCA T2+SPE component-axis filters (double-hard) strip trivially- and linearly-detectable
  anomalies. Answers the simple-baseline critiques (Sarfraz Quo Vadis; Garg TNNLS) up front.
- C3. Drift-aware anomaly typing (new section): everything off training-normal is an anomaly; a
  CAUSAL per-community slow/fast decomposition (slow = rolling median prev 24h; fast = residual /
  max(rolling MAD, calib MAD)) types each window gradual-drift vs abrupt-changepoint. Scoring the
  changepoint component recovers attack detection under drift.

## Narrative arc
Problem (non-stationary, easy-dominated benchmarks) -> C1 -> C2 hardened eval -> HAI clean
significant win -> measured drift spectrum (WADI 1.5% / HAI 30% / SWaT 67% test-normal shift),
physically grounded (RO membrane fouling + analyser electrode drift), literature-backed (DAICS,
D3R, AnoShift) -> C3 typing recovers SWaT + unifies the spectrum -> impact.

## Per-dataset landing
- HAI: flagship. community 0.845 difficult, significant vs all learned baselines; moderate drift
  where fixed channel-wise/deep predictors collapse and density holds.
- WADI: comparable, no drift (1.5%). community 0.771 numerically leads (8 episodes, NS); linear
  competes (WADI faults low-rank/linear).
- SWaT: severe-drift case motivating C3. raw headline collapses (0.524 difficult under 67% drift);
  drift-aware typing recovers attack detection (All 0.880 / Diff 0.723 / DH 0.671).

## Key clean numbers (difficult subset unless noted)
- HAI: community 0.845; AE 0.758; USAD 0.497; TranAD 0.445; GDN 0.481; LinRes 0.586; boosted 0.321.
- WADI: community 0.771; LinRes 0.750; GDN 0.660; boosted 0.660; TranAD 0.613; USAD 0.579; AE 0.629; IF 0.634.
- SWaT (clean official, raw headline): community 0.524; IF 0.550; AE 0.517; USAD/TranAD 0.477; GDN 0.473(clean); LinRes 0.464; boosted 0.412. trivial floor 0.627.
- SWaT (changepoint typing): All 0.880 / Diff 0.723 / DH 0.671 (vs headline +0.200 diff P=0.014).
- Drift-spectrum FA rates: WADI 1.5% / HAI 39% / SWaT 69% (community-surprise, test-normal > train-q99).

## Manuscript changes (edit pass)
1. Replace all SWaT numbers with clean official; remove leak-driven claims; correct GDN double-hard (leaked dump -> clean 0.115).
2. Add drift-spectrum subsection (measured FA rates + physical grounding + DAICS/D3R/AnoShift citations; SWaT = single continuous 11-day run, non-stationary, NOT two sessions).
3. Add C3 drift/changepoint typing section (recovery table + typing validation + caveats).
4. Move A8-A10 to future work; keep A8 valley as characterization (rare, no detection edge).
5. Terminology: anomaly/malfunction; "attack" only for dataset description.
6. Double-hard = PCA T2+SPE (LinRes now a detector). Coverage-stable difficulty axis = measurement-artifact removal (WADI 2B_AIT_002_PV precedent + DAICS AIT201 recalibration).
7. Add boosted LOO baseline (collapses clean).
8. Rewrite abstract to the new arc (<=200 words).

## Caveats to state plainly (no apologetic tone)
- SWaT typing masks 2/25 episodes (long-attack 24h shadow); deployment needs a rolling threshold.
- WADI synthetic-ramp invariance -0.055 on the 2-day record (no-op in practice; WADI has no real drift).
- SWaT official normal is small (~1616 train windows), data-starved fit.
- WADI lead underpowered (8 episodes).

## Data-integrity guards now in place
- eda_real._raw_swat_canonical: official normal + _assert_swat_no_leak (0% overlap fail-fast) + CLIP=10.
- Use sota_bundle/experts_full (S=24) for SWaT, NEVER experts_variants/cur_avg25 (stale S=25).
- rev4_doublehard_pca_all3.py GDN_FILE[SWaT_canon] points at a stale leaked dump - fix before reuse.
