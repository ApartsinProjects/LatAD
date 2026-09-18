# Scout: many-regime CPS datasets (for A2 regime-explosion + A8 between-regime overlap)

Cited counts. Excludes already-screened (WADI/SWaT/HAI/MetroPT/Wind-SCADA/SMD/Paderborn, all ≤K≈25, crisp).

## Ranked candidates
| Rank | Dataset | Modality / #ch | Regimes (evidence) | Real? | Download | Why A2 / A8 |
|---|---|---|---|---|---|---|
| 1 | **Cranfield Three-Phase (Multiphase) Flow Facility** | SCADA 1 Hz, **24 vars** | **20 normal operating conditions** (4 air × 5 water set-points) + 6 faults; multiphase flow-regime map (bubbly→slug→churn→annular) with transition zones | **Real** rig | no-auth **Kaggle** `afrniomelo/cranfield`; IEEE DataPort; MATLAB Central | 20-point continuous flow grid + gas-liquid regime transitions = clearest place for genuine BLENDED/overlapping modes (A8) |
| 2 | **Building Data Genome Project 2 (BDG2)** | hourly meters, **3053 meters / 1636 buildings** | season × occupancy × weekday modes across 1636 buildings; very high multiplicity (no crisp K) | Real | open: GitHub, Zenodo 3887306, Kaggle | soft seasonally-drifting overlapping modes; pushes A2 far past K≈25 (needs mode-discovery pass) |
| 3 | **Automotive CAN-bus driving** (KCID 2025; "This Car is Mine!", 51 feat; Kreutz Tesla-M3) | raw high-freq multivariate | continuous driving regimes (idle/accel/cruise/brake/gear), emergent count | Real | registration (IEEE DataPort) / GitHub | driving states blend continuously → strong A8 overlap candidate |
| 4 | **N-CMAPSS (NASA turbofan)** | sim, 14 sensors + 4 op-cond | continuous operating conditions; C-MAPSS FD004 = 6 discrete; N-CMAPSS clusterable into many flight regimes | **hi-fidelity sim** | open (NASA PCoE/DASHlink) | continuous envelope → regime count is a modeling choice; flight-phase transitions give overlap |

## Top picks to pull + screen
1. **Cranfield Multiphase Flow Facility** — BEST (real × overlap × trivial download, small). Kaggle no-auth. Screen: fit K* on normal, check whether the 20 set-points collapse into OVERLAPPING clusters at slug/churn transitions (responsibility entropy vs SKAB-ref). **Single most likely place to find genuine A8 between-regime overlap.**
2. **BDG2** — best for raw regime multiplicity / A2 scale; needs mode-discovery; pick a few buildings.
3. **N-CMAPSS** (secondary) — tests A2 regime-count directly; flag as simulation.

## Caveats
- Cranfield "20 conditions (4×5) / 24 vars / 6 faults" from benchmark descriptions (Ruiz-Cárcel); primary PDF 403'd — verify flow-regime labels on pull.
- CAN regime count emergent (not published); BDG2 per-building mode counts unpublished; C-MAPSS FD004 only 6 conditions (N-CMAPSS supersedes).

## Sources
Cranfield: sciencedirect S0967066115000866; ieee-dataport three-phase-flow-facility; kaggle afrniomelo/cranfield. BDG2: nature s41597-020-00712-x; github buds-lab; zenodo 3887306. CAN: arXiv 2301.04988, 2510.25856; IEEE DataPort car-mine. N-CMAPSS: arXiv 2302.01704; PHM 2021 Data Challenge.
