# A3 screen: SMD + Paderborn PU bearing (extending the "which real CPS datasets exhibit A3" evidence)

Report-only. Method is identical to the existing A3 screen so numbers are comparable:
load train-NORMAL, build the standard 6-stat window features (W=60, stride=30), standardize the
window features on train-normal, train VaDE (n_clusters=16, latent_dim=8, epochs=30, warmup=6,
seed=0, cpu), then compute

- **H_norm** = mean over windows of `-sum(g*log g)/log K` from `v._responsibilities(Z)` (K=16),
- **rho** = fraction of train-normal windows with max-responsibility < 0.5.

**A3 verdict rule:** A3 if `H_norm >= ~0.15`. Reference scale (existing screen, same method):
SKAB `H_norm ~0.27` (A3 holds); WADI / HAI / SWaT / MetroPT / WindSCADA `H_norm <= 0.09` (no A3).

Reproduced this run as an anchor: **SKAB H_norm = 0.268, rho = 0.102** (matches the ~0.27 reference).

Script: `_diagnostics/a3_screen_smd_pu.py`; raw numbers: `_diagnostics/a3_screen_smd_pu.json`.

## Results

| dataset | n_train | feat_dim | H_norm | rho(<0.5) | mean_maxresp | verdict |
|---|---:|---:|---:|---:|---:|---|
| **SKAB** (positive reference) | 400 | 48 | **0.268** | 0.102 | 0.658 | **A3-WITNESS** |
| SMD machine-1-1 | 948 | 228 | 0.049 | 0.006 | 0.946 | no-A3 |
| SMD machine-1-4 | 789 | 228 | 0.039 | 0.001 | 0.957 | no-A3 |
| SMD machine-2-1 | 788 | 228 | 0.048 | 0.008 | 0.949 | no-A3 |
| SMD machine-3-7 | 955 | 228 | 0.056 | 0.008 | 0.940 | no-A3 |
| **Paderborn PU** (K001-K005 healthy) | 1200 | 78 | 0.047 | 0.000 | 0.948 | no-A3 |

## Dataset 1 - SMD (Server Machine Dataset)

Downloaded `ServerMachineDataset` (train / test / test_label) from the OmniAnomaly repo
(github.com/NetManAIOps/OmniAnomaly) into `datasets/_new/OmniAnomaly/ServerMachineDataset/`.
Screened machine-1-1 plus 1-4, 2-1, 3-7 (38 channels each, per-timestep labels).

**Verdict: no A3 on all four machines** (H_norm 0.039-0.056, all inside the <=0.09 benchmark
no-A3 band; rho ~0, mean max-responsibility ~0.95 = crisp single-mode assignment). This confirms
the LOW A3 prior for SMD: server-telemetry normal operation is effectively one tight regime, not
a set of overlapping basin-heads. Consistent with the other high-dimensional SCADA/telemetry
benchmarks. Loader `_raw_smd(machine)` already existed in `eda_real.py` and was used unchanged.

## Dataset 2 - Paderborn PU bearing (KAt DataCenter, raw ~64 kHz vibration)

Downloaded (public, no auth) the 5 healthy/undamaged bearings K001-K005 from
`https://groups.uni-paderborn.de/kat/BearingDataCenter/<CODE>.rar` (~160 MB each) into
`datasets/_new/Paderborn/`, extracted with 7-Zip. Each bearing = 4 operating conditions x 20
recordings of ~4 s (`vibration_1` at 64 kHz):

| condition | speed | load torque | radial force |
|---|---|---|---|
| N09_M07_F10 | 900 rpm | 0.7 Nm | 1000 N |
| N15_M01_F10 | 1500 rpm | 0.1 Nm | 1000 N |
| N15_M07_F04 | 1500 rpm | 0.7 Nm | 400 N |
| N15_M07_F10 | 1500 rpm | 0.7 Nm | 1000 N |

**Feature mapping (raw vibration -> standard window features).** This is a different modality
(univariate high-rate vibration, not a SCADA multivariate stream), so the raw signal is mapped to a
per-frame multichannel "regime stream" that then flows through the identical W=60 / stride=30 /
6-stat pipeline: each recording is framed (non-overlapping 2048-sample frames), and each frame ->
log band-power in 12 log-spaced bands (500 Hz..32 kHz) + log broadband RMS = 13 channels. Channels
are standardized on the pooled healthy set; windows are built **within each recording** (never
crossing recording boundaries); the A3 screen then standardizes the window features as usual. The
"regimes" are thus the operating conditions (band-power level/shape shift with speed/load/force),
pooled across the 5 nominally-identical healthy bearings.

**Verdict: no A3. Paderborn is NOT a second real A3 witness.** H_norm = 0.047, rho = 0.000, mean
max-responsibility 0.948 - inside the benchmark no-A3 band, nowhere near SKAB's 0.27. The operating
conditions form crisply separable clusters rather than overlapping basin-heads.

**Sanity checks (guarding against a feature-map artifact, per verify-before-report).** The reference
SKAB anchor reproduced at 0.268 in the same run, so the low bearing value is not a broken pipeline.
A nearest-centroid condition-separability check gives purity 0.679 (5 bearings pooled) and 1.000
within a single condition - the conditions ARE distinct and the healthy bearings within one condition
collapse to a single tight mode, so the low entropy is genuine crisp separation, not a degenerate map
that made windows indistinguishable. Robustness across feature-map choices (all far below the 0.15
A3 threshold):

| variant | n_win | H_norm | rho | verdict |
|---|---:|---:|---:|---|
| base (L=2048, 12 bands, no overlap) | 1200 | 0.047 | 0.000 | no-A3 |
| coarse (L=4096, 8 bands) | 400 | 0.022 | 0.000 | no-A3 |
| overlap (L=2048, hop=L/2) | 2802 | 0.071 | 0.004 | no-A3 |
| fine (L=1024, 20 bands, hop=L/2) | 6010 | 0.104 | 0.046 | weak |
| single-condition N15_M07_F10 x5 bearings | 1502 | 0.037 | 0.000 | no-A3 (purity 1.0) |

The only variant that rises toward the threshold is the finest resolution (0.104, still < 0.15 and
< half of SKAB), where shorter frames inject within-recording nonstationarity that mildly spreads
responsibilities; rho stays ~0.05 vs SKAB's 0.10 and max-resp stays 0.89 vs SKAB's 0.66, so it is
not the SKAB-style basin-head ambiguity. The high-A3 prior (rotating machinery under varying
conditions would blend into overlapping modes) is not borne out at the band-power window-feature
level: distinct operating conditions separate crisply, and within a condition healthy vibration is
a single mode.

## Loaders added to eda_real.py

- `_raw_paderborn(bearings=(...), signal="vibration_1")` - **new, additive.** Reads the extracted
  Paderborn `.mat` recordings and returns per-recording raw signals tagged with bearing / operating
  condition / run. Deliberately **not** registered in `RAW`/`CLIP` (raw vibration is not a SCADA
  stream and does not go through `load()`'s continuous windowing); the band-power framing lives in
  `_diagnostics/a3_screen_smd_pu.py`. No existing dataset or behavior was changed.
- `_raw_smd(machine)` - pre-existing, used unchanged; SMD data added under
  `datasets/_new/OmniAnomaly/ServerMachineDataset/`.

## Bottom line

Two more real CPS datasets screened. Both are **no-A3**: SMD (4 machines, expected) and the
Paderborn PU healthy bearing (against its high prior). **SKAB remains the sole real A3 witness**;
the extended evidence now spans WADI, HAI, SWaT, MetroPT, WindSCADA, SMD (x4) and Paderborn on the
no-A3 side. The bearing result is a clean negative: real rotating-machinery vibration under varying
speed/load/force does not exhibit the overlapping-basin-head geometry, so A3 in this corpus stays
specific to SKAB rather than generic to CPS.
