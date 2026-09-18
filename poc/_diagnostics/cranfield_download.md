# Cranfield Three-Phase (Multiphase) Flow Facility — download & verification

**Status: SUCCESS.** A verified, no-auth copy of the canonical Ruiz-Cárcel benchmark is on disk.

## Source that worked (no auth, source #1)

MATLAB Central File Exchange submission **50938** ("Cranfield Multiphase Flow Facility",
packaged by Yi Cao as the CVA benchmark tutorial for Ruiz-Cárcel et al.'s
*Statistical Process Monitoring of a Multiphase Flow Facility*, Control Engineering Practice 2015).

Direct download endpoint (curl -L, HTTP 200, no login):
```
https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/50938/versions/1/download/zip/CUcasestudy.zip
```
ZIP: 16,773,163 bytes. License: BSD-style, Copyright (c) 2015 Yi Cao (see `license.txt`).

Sources #2/#3 (GitHub/Zenodo/figshare, IEEE DataPort) not needed. Kaggle
`afrniomelo/cranfield` remains the auth-gated fallback (needs `~/.kaggle/kaggle.json`);
not attempted, per constraint.

## Location on disk

```
E:\Projects\Backlog\LatAD\poc\datasets\_new\Cranfield\
├── CUcasestudy.zip                      (16.0 MB, kept)
└── CUcasestudy\CUcasestudy\
    ├── Training.mat        4,422,337 B   (normal-operation data)
    ├── FaultyCase1.mat     1,913,575 B
    ├── FaultyCase2.mat     2,130,815 B
    ├── FaultyCase3.mat     3,453,687 B
    ├── FaultyCase4.mat     2,027,255 B
    ├── FaultyCase5.mat     1,806,268 B
    ├── FaultyCase6.mat     1,021,031 B
    ├── CUBenchmark.m / .asv   (CVA benchmark driver)
    ├── cvatutor.m             (CVA detector tutorial)
    └── html\CUBenchmark.html + PNGs   (rendered tutorial)
```

## Verification (scipy.io.loadmat)

**Process-variable count: 24** — confirmed. Every data array has exactly 24 columns
(float64), matching the 24 process variables of the facility (air/water/oil flow rates,
pressures, temperatures, densities, level indicators, valve positions per the 2015 paper).
Sampled at 1 Hz. Values are physical/engineering units (e.g. Training T1 first columns
range ~0.95–1.68), not normalized.

**Normal / training data — `Training.mat`** (3 normal-operation regimes):
| array | shape (rows × vars) |
|---|---|
| T1 | 10372 × 24 |
| T2 |  9825 × 24 |
| T3 | 13200 × 24 |

**Fault data — `FaultyCase{1..6}.mat`** (6 fault types, the canonical Ruiz-Cárcel set):
each file holds 2–3 experiment "Set" runs (24-var data) plus a matching `EvoFault*` label
vector giving the fault-evolution / severity stage per timestep.

| file | data arrays (rows × 24) | label vectors |
|---|---|---|
| FaultyCase1 | Set1_1 5811, Set1_2 4467, Set1_3 4321 | EvoFault1_1/_2/_3 |
| FaultyCase2 | Set2_1 9192, Set2_2 3496, Set2_3 3421 | EvoFault2_1/_2/_3 |
| FaultyCase3 | Set3_1 9090, Set3_2 6272, Set3_3 10764 | EvoFault3_1/_2/_3 (float64) |
| FaultyCase4 | Set4_1 7208, Set4_2 4451, Set4_3 3661 | EvoFault4_1/_2/_3 |
| FaultyCase5 | Set5_1 2541, Set5_2 10608 | EvoFault5_1/_2 |
| FaultyCase6 | Set6_1 2800, Set6_2 4830 | EvoFault6_1/_2 |

Label check (FaultyCase1): `EvoFault1_1` unique values `{10,15,20,25,30,40,90}` (all-fault run,
stage-coded), `EvoFault1_2` unique `{0,10,15,...,90}` with 480/4467 timesteps at 0 (normal
preamble then fault ramp). So labels are staged fault-severity codes, not plain 0/1 — 0 = normal,
>0 = fault stage. Total ~17 fault-run "Set" segments across the 6 cases.

## Operating conditions / regime structure

This CVA packaging exposes **3 distinct normal-operation regimes (T1, T2, T3)** as the training
data, plus per-case operating-condition/severity codes inside the fault runs. It does NOT split
out the full ~20-point (4×5 air/water flow-rate grid) normal matrix as separate arrays; the three
Training sets are the regime-labelled normal data here. If the finer 4×5 = 20 operating-point
breakdown is later needed, the Kaggle `afrniomelo/cranfield` version organizes normal data by those
individual flow-rate set-points — enable it by placing a token at `~/.kaggle/kaggle.json`.

## Ready for A8 (between-regime overlap)?

**Yes, usable.** The three normal regimes T1/T2/T3 (24-var, ~33k rows total) give distinct
operating regimes to measure between-regime overlap on, and the fault sets add further
operating-condition-coded segments. This is enough to run the A8 overlap screen. Caveat: regimes
here are coarse (3 normal training blocks) rather than the full 20 flow-rate set-points; if A8
wants the finer grid, the Kaggle version is the fallback. No loader built and no screen run yet,
per instructions — data is downloaded and verified only.
