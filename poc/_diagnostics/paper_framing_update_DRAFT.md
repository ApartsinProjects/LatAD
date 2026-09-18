# Paper-framing update — DRAFT for sign-off (nothing written to IoT2.html yet)

Basis: corrected, clipped, canonical (subtle-anomaly) difficult subset, construct-matched, 5-seed,
**headline `LatAD (regime-community)` model** (not the ablated global). Cleaned WADI (drop the rescaled
artifact channel `2B_AIT_002_PV`) + canonical SWaT (`SWaT_Dataset_Attack_v0`, 12.14% attack).

## 1. Corrected difficult-subset table (what the paper should report)

| difficult-subset AUROC | LatAD (regime-community) | linres | AE | TranAD | USAD | IF |
|---|---|---|---|---|---|---|
| **WADI (cleaned)** | **0.816** (HCcoh) / 0.824 (cohmax) | 0.787 | 0.739 | 0.722 | 0.685 | 0.681 |
| **SWaT (canonical)** | **0.827** | 0.782 | 0.729 | 0.655 | 0.658 | 0.627 |
| **HAI** | 0.814 → 0.851 excl. uncovered regime | 0.586 | 0.447 | 0.257 | 0.328 | 0.439 |

Significance (episode-block bootstrap): **SWaT LatAD > linres P=0.024 (difficult), P=0.007 (double-hard) — significant.**
WADI LatAD > linres +0.036, CI [−0.097,+0.188], p=0.31 — **numerically ahead, not significant** (only 11 anomaly episodes).

## 2. Abstract — proposed change

CURRENT: "...LatAD attains the best AUROC (WADI 0.862, HAI 0.949, SWaT 0.993) and leads the difficult
subset of all three, notably on HAI (0.849; +0.09, 95% CI [0.046, 0.160])..."

PROBLEMS: (a) WADI 0.862 rode the instrumentation artifact (channel rescaled between recordings);
(b) SWaT 0.993 used the non-canonical mirror; (c) "leads all three difficult subsets" is only
significant on HAI and SWaT, numerical on WADI.

PROPOSED (numbers to be finalized after the reconciliations in §5):
> "On three real IIoT testbeds — with WADI's known instrumentation artifact removed and SWaT's canonical
> attack set — LatAD leads the difficult subset on HAI and SWaT and is competitive on cleaned WADI. On the
> hardest, jointly-improbable faults (which USAD and TranAD fall to 0.26-0.66 on) LatAD's advantage is
> largest: on HAI it detects anomalies no reconstruction or linear detector catches, and on SWaT its lead
> over the strongest linear baseline is significant (P=0.024)."

## 3. WADI results paragraph — proposed framing
State plainly and without apology:
- WADI's difficult anomalies are largely **linear cross-channel breaks**; a discrete-aware linear residual
  (linres) is a strong baseline there. LatAD's regime-community model is **numerically ahead** of it
  (0.816 vs 0.787) and **beats every learned baseline** (AE/TranAD/USAD/IF), but with only 11 anomaly
  episodes the margin over linres is not statistically distinguishable.
- Disclose the **data-quality cleaning**: channel `2B_AIT_002_PV` is rescaled between the normal and attack
  recordings (27,359σ shift, unique among 123 channels; a documented WADI issue). We drop it; results are
  reported on the cleaned data. (The uncleaned "0.862 / leads" number is robustness to that artifact, which
  we do not claim as detection performance.)
- Note the linres caveat honestly: linres's apparent WADI edge is **scaling-sensitive** (0.787 with unscaled
  one-hot features → 0.684 under standard all-feature standardization).

## 4. A8 (Mixed signals / typed channels) — REQUIRED wording fix
The design does NOT type channels: discrete/actuator states are z-scored and Gaussian-modeled like continuous
channels. Narrow the A8 sentence to what runs — e.g.:
> "LatAD handles the *scale* heterogeneity of typed channels via per-channel standardization and the whitened
> residual; the per-window statistics implicitly encode discrete-state fractions for low-cardinality channels.
> Native categorical likelihoods for discrete channels (HI-VAE [Nazábal 2020], VAEM [Ma 2020]) are left to future work."
New validated citations ready: HI-VAE, VAEM, TABOR (see `discrete_method_citations.md`).

## 5. Consistency reconciliations REQUIRED before wiring in
- **HAI difficult:** abstract says 0.849; current pipeline gives 0.814. Reconcile (config/subset/run difference)
  and use one number everywhere (SC-2/SC-7).
- **Full-AUROC row:** update WADI 0.862 → clean (~0.85, All-subset HCcoh 0.857), SWaT 0.993 → canonical, HAI
  0.949 → current (~0.94). Recompute the full-set AUROC row on the corrected data.
- Regenerate all derived tables/figures from the corrected tables; re-run bibtest.

## 6. Optional supporting additions (strong, honest, on the corrected basis)
- **Detection-set complementarity (HAI):** LatAD catches 49 difficult anomalies that no classical/ML/deep
  method catches; TranAD catches 0 that LatAD misses. A Venn/complementarity figure.
- **Coverage self-diagnosis (HAI):** excluding the uncovered operating regime (0 anomalies removed) lifts
  LatAD to 0.951 AUROC / 0.888 TPR@5%; 94% of its false alarms are in that block. (All detectors improve;
  LatAD leads throughout.)
- **Extended difficulty stratification (PCA-per-axis + discrete easy-filter):** on the joint-nonlinear
  survivors LatAD's HAI lead is significant (+0.136, CI [+0.055,+0.199]).
- Report **AUPRC + TPR@low-FPR** alongside AUROC everywhere (more honest for rare anomalies).
