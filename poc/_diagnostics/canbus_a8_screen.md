# A8 (between-regime overlap) screen — real CAN-bus / OBD-II driving telemetry

## (a) Dataset

- **Source**: Kaggle `cephasax/obdii-ds3` ("OBD-II datasets"), CC0-1.0, https://www.kaggle.com/datasets/cephasax/obdii-ds3
- **File used**: `exp2_19drivers_1car_1route.csv` — 19 drivers, 1 car, 1 shared route (Buenos Aires), logged with an OBD-II Bluetooth dongle during real on-road driving. No `TROUBLE_CODES` rows anywhere in the file (pure normal driving, no fault/intrusion data).
- **Local path**: `poc/datasets/_new/CANdrive/exp2_19drivers_1car_1route.csv`
- **Rows**: 8261 raw, 7822 after dropping malformed OBD glitch rows (e.g. a raw hex frame `1:6032000007E803` in place of a barometric reading, and a few truncated `ENGINE_RUNTIME` stamps).
- **Trips**: 19 continuous per-driver segments (`VEHICLE_ID` = s1..s19), 125–640 rows each. Windows are built per-trip and never cross a trip boundary, then pooled (2581 windows total).
- **Sample rate**: `ENGINE_RUNTIME` advances by 4s per row (0.25 Hz).
- **Channels (8, all continuous real vehicle-dynamic signals)**: SPEED, ENGINE_RPM, THROTTLE_POS, ENGINE_LOAD, ENGINE_COOLANT_TEMP, INTAKE_MANIFOLD_PRESSURE, MAF, TIMING_ADVANCE. Raw values carried unit suffixes and comma-decimal formatting (e.g. `"34,90%"`, `"2124RPM"`); parsed to floats before use.
- **Windowing**: W=6 samples (24s) / stride=3 (12s), `winfeat.window_features(...,"stats")` (6 stats/channel × 8 channels = 48 features/window), standardized on the pooled window set.
- **Method**: identical to the established WADI/HAI/SWaT/Cranfield/SKAB A8 screens — `models_vade.train_vade` (VaDE, latent_dim=8), K-sweep {8,16,32,64,128}, mean max-responsibility / normalized responsibility entropy, then the mandatory variance-floor sanity check on the highest-entropy K across 3 seeds.

## (b) K-sweep

| K | n_windows | mean_maxresp | H_norm | rho(maxresp<0.5) | rho(maxresp<0.6) |
|---|---|---|---|---|---|
| 8   | 2581 | 0.899 | 0.132 | 0.023 | 0.073 |
| 16  | 2581 | 0.841 | 0.158 | 0.072 | 0.152 |
| 32  | 2581 | 0.821 | 0.142 | 0.069 | 0.170 |
| 64  | 2581 | 0.806 | 0.132 | 0.098 | 0.188 |
| 128 | 2581 | 0.870 | 0.077 | 0.040 | 0.105 |

Entropy stays in a narrow **0.077–0.158** band across the whole K-sweep, with mean max-responsibility consistently high (0.81–0.90). This is squarely inside the **WADI/HAI/SWaT/Cranfield "no overlap" reference band (0.01–0.15)**, not the SKAB pre-refit band (0.06 rising to 0.53). There is no K at which entropy climbs the way SKAB's apparent witness did.

## (c) Variance-floor sanity check (best K=16, 3 seeds)

| seed | H_floor | H_emp (empirical-variance refit) | rho_floor | rho_emp | frac components at floor | collapse (H_emp < 0.5·H_floor)? |
|---|---|---|---|---|---|---|
| 0 | 0.158 | 0.107 | 0.072 | 0.029 | 0.06 | False |
| 1 | 0.175 | 0.112 | 0.071 | 0.035 | 0.06 | False |
| 2 | 0.169 | 0.112 | 0.080 | 0.039 | 0.02 | False |

Only 2–6% of the K=16 components sit at the logvar floor, and empirical component SD (~0.52) is close to the floored SD (~0.60–0.61) — i.e. the floor is not doing the work here. Entropy drops moderately (~0.16 → ~0.11) under the empirical refit but does **not collapse** by the project's own >2x threshold. This is the opposite failure mode from SKAB: there, a *high* entropy collapsed under the refit, exposing a floor artifact. Here entropy was never high to begin with, and the modest additional drop under the empirical refit only tightens an already-low number.

## (d) Verdict

**A8 (between-regime overlap) is NOT observed on this CAN-bus driving dataset.** Responsibility entropy across the full K=8..128 sweep (0.077–0.158) sits in the same low band established for WADI/HAI/SWaT/Cranfield, and the one seemingly elevated point (K=16, H_norm≈0.16–0.18) is not a floor artifact — it survives the variance-floor check largely intact — but its absolute magnitude is simply too low to count as genuine overlap by the project's own SKAB-calibrated positive threshold (SKAB pre-refit reached 0.53; nothing here approaches that). Mean max-responsibility stays above 0.80 at every K, meaning the VaDE consistently assigns the large majority of each window's probability mass to a single component: driving windows on this dataset cluster into well-separated discovered regimes (idle/low-load vs. accelerating vs. cruising vs. high-RPM segments show up as crisp components), not a continuously blended manifold.

This does not resolve the scout's underlying hypothesis about driving dynamics in general — a single-route, single-vehicle, 19-trip, 0.25 Hz OBD-II log is a narrow slice (short trips, uniform road/vehicle, coarse 4s sampling that could be smoothing over the sub-4s transitions where blending would actually appear) — but on the data obtained, the between-regime overlap signature the project is looking for is absent, joining WADI/HAI/SWaT/Cranfield rather than providing the first genuine A8 witness.
