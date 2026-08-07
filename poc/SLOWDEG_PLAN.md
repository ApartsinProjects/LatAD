# Plan: slow-degradation detection via TTE-labeled trend alerting

Status: PLAN ONLY (not implemented). Separate detection module from the paper core
(WADI/HAI/SWaT event task). Enters the write-up only if it yields a positive
lead-time result; otherwise stays a registry/diagnostics entry (wins-only rule).

## Contract
Slow-degradation datasets carry **time-to-event** labels, not per-window
correlation-break events. Task: does a monotone drift in the health signal precede
failure, and can we alert early at a controlled false-alarm rate?

Datasets in scope (once downloaded): Scania Component X (native `train_tte.csv`),
Wind Turbine, ALFA (UAV), MetroPT (already prepped, but likely too abrupt).

## Stage 1 - TTE labeling
- Per unit/run: `tte(t) = t_failure - t` from the failure event (Scania ships it
  directly; MetroPT/Wind/ALFA derive from published failure windows).
- Units with no failure = right-censored ("survived", no positive event).
- Define run-in-to-failure horizon H (last 20-30% of each unit's life) as the
  "degrading" region; earlier = "healthy". Ground truth for scoring alerts
  (early = good, late/miss = bad).

## Stage 2 - per-regime/mode anomaly score
- Reuse VaDE latent + per-mode density head.
- Assign each window to its operating regime: VaDE hard-assignment component, OR an
  explicit regime variable when shipped (Scania `*_specifications.csv`, Wind
  operating state, load/RPM bins).
- Score **within-regime**: normalize density-NLL against the train-normal
  distribution of that same regime (per-mode z-score), so a benign regime switch
  (e.g. high load) does not masquerade as degradation. Key methodological point:
  drift measured against like-for-like operating conditions.

## Stage 3 - trend fit per unit
- On each unit's regime-normalized score series, fit a monotone trend. Two
  estimators, co-computed in one pass:
  - **Mann-Kendall** trend test (non-parametric, robust to noise/seasonality) -> S
    statistic + p-value.
  - **Theil-Sen slope** -> trend magnitude (score units/hour), a degradation rate.
- Smooth first (rolling median over regime-normalized score) so single-window
  spikes don't drive the slope.

## Stage 4 - alert rule + statistical gate
- Alert fires when Mann-Kendall p < alpha (one-sided, positive trend) AND Theil-Sen
  slope exceeds a minimum effect size tau, sustained over persistence window of k
  consecutive windows (suppress transient blips).
- Calibrate (alpha, tau, k) on healthy (censored) units only: hold per-unit
  false-alarm rate at a target (e.g. <= 1 false alert / unit-year). Deploy-threshold
  analogue of `improve_multiseed.py`.

## Stage 5 - evaluation (early-warning metrics, not point-wise AUROC)
- Detection: fraction of failing units alerted before failure, at calibrated FA rate.
- Lead time: distribution of (t_failure - t_alert) for true alerts (headline PdM number).
- False-alarm rate on censored units (the held constraint).
- Multi-seed over VaDE seeds; per-dataset.

## Landing
- New script `poc/slowdeg_trend.py` (Stages 1-5), consuming `eda_real`-style loaders.
- Smoke-test Stage 3 on Scania first (cleanest native TTE) before building the full harness.

## Risk
MetroPT showed 0 discriminative windows at fine resolution -> air-leak degradation may
be too abrupt for a trend test. Trend approach targets Scania/Wind (genuine gradual wear).
