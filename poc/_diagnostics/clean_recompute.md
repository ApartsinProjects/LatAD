# Clean recompute: three coupled fixes + full rebuilt tables (LatAD)

Datasets are the paper's revision datasets: **WADI = `WADI_clean`** (artifact channel `2B_AIT_002_PV`
dropped), **SWaT = `SWaT_canon`** (canonical Dec-2015 attack log), **HAI** unchanged. Community experts
are `sota_bundle/experts_full` (the configuration that reproduces the paper headline). All AUROC/F1 are
5-seed mean (±SD where shown). Python `C:\Python314\python`.

## TL;DR — what changed and what held

- **Invariant I1 HOLDS.** The HAI and SWaT difficult- and double-hard-subset wins remain **significant**
  after all three fixes (HAI Difficult +0.088, P=0.0; SWaT Difficult +0.055, P=0.007; both double-hard
  P<0.001). No significant HAI/SWaT conclusion is lost.
- **Invariant I2 HOLDS.** Every HAI/SWaT baseline (unchanged code path) reproduces the paper's Table 3
  **exactly** (IF/AE/LinRes/USAD/TranAD/LatAD-global), and SWaT retrained LatAD reproduces the stored
  per-seed difficult AUROC to 4 decimals (HAI to ±0.008, the documented torch-CPU cross-machine drift).
- **WADI difficult subset re-derives 43 → 30** under the consistent rule (expected ~31). The 12 removed
  windows are exactly the constant STATUS/setpoint flips that were trivially separable (|z|≈1e9) yet sat
  in "Difficult," inflating every learned method.
- **One headline number moves materially and must be reconciled in the manuscript (not a significance
  flip):** WADI difficult community model **0.824 → 0.771 (fused) / 0.795 (HC_coh-alone)**. The WADI
  "win" was never significant (P=0.32 before, P≈0.40–0.46 now). On the clean 30-window subset the
  single-latent LatAD collapses to 0.634 (below LinRes 0.750); only the **HC_coh community aggregation**
  (0.795) still leads LinRes. **Reporting recommendation: report HC_coh-alone (0.795) as the WADI
  difficult headline, not the LatAD-fused 0.771.**

---

## The three fixes (exact diffs)

Files changed: `build_scores_table.py` (FIX 2, FIX 3, + `LatAD_train` for FIX 1) and `ensemble_final.py`
(FIX 1 + `surv` de-saturation). Only WADI needed a retrain; HAI/SWaT scores are byte-identical.

### FIX 1 — ensemble calibration leak (`ensemble_final.py`)
The fusion z-scale and the LatAD null tail were calibrated on `nm = y==0` (TEST-normal windows, selected
with test labels). The community HC/HC_coh **inputs** were already clean (computed against the experts'
held-out 20% train-normal `Cal` slice); only the **z-scale** leaked, plus the LatAD null tail.

```python
def surv(ref, v):
    o = np.sort(ref); r = np.searchsorted(o, v, side="right") / len(o)
    return -np.log(np.clip(1 - r, 1.0 / (len(o) + 1), 1.0))   # was 1e-4 (fixed floor -> saturation)
...
lat_tr = d["LatAD_train"]                                     # global model's TRAIN scores
for sd in range(nseed):
    tails = np.stack([surv(Cal[sd,g], Tst[sd,g]) for g in range(S)])
    P     = np.stack([pval(Cal[sd,g], Tst[sd,g]) for g in range(S)])
    hc, hc_coh = HC(P), HC(P, wt=w)
    tails_c = np.stack([surv(Cal[sd,g], Cal[sd,g]) for g in range(S)])   # train-normal references
    Pc = np.stack([pval(Cal[sd,g], Cal[sd,g]) for g in range(S)]); hc_c, hccoh_c = HC(Pc), HC(Pc, wt=w)
    z = lambda s, r: (s - r.mean()) / (r.std() + 1e-9)        # scale from TRAIN reference r
    nulltail = surv(lat_tr[sd], lat[sd]); zl = z(nulltail, surv(lat_tr[sd], lat_tr[sd]))
    acc["null+HC"].append(np.maximum(z(hc, hc_c), zl))
    acc["HCcoh+LatAD"].append(z(hc_coh, hccoh_c) + zl)         # + sum/max/cohmax/HC/q95 likewise
```
This matches the experts' own 80/20 calibration (density fit on the 80% slice, scale from train-normal).
Test LatAD scores are untouched, so the standalone LatAD column and I2 survive. The `surv` floor changed
from a fixed `1e-4` to `1/(n+1)` (rank-based) to de-saturate the train-calibrated null tail; without it
`null+HC` (max-fused) collapsed (HAI 0.846→0.77).

### FIX 2 — unclipped second standardization (`build_scores_table.py`)
The per-window-feature re-standardization divided train-constant columns by `(std+1e-8)` with no clip,
inflating `|z|` to 1e9–1e17 on constant STATUS/setpoint channels. Apply the **`eda_real.CLIP` convention**
after standardization (WADI ±10, HAI/SWaT None), exactly as the raw standardization is bounded:

```python
clipv = E.CLIP.get(name)                       # WADI 10, HAI/SWaT None (their real signal is genuine)
Xte = (Xte0 - mu) / sig
if clipv: Xte = np.clip(Xte, -clipv, clipv)
...
Xtr = (Xtr0 - mu) / sig
if clipv: Xtr = np.clip(Xtr, -clipv, clipv)    # bound AE/LatAD inputs the same way
```
Verified I4 (WADI scored input): max|z| = **10.0** exactly (was 2e9). WADI LatAD on the 12 flip windows:
mean **1.4e14 → 1.43**, global max **5.9e14 → 7.5**.

Why not a universal ±10: HAI/SWaT genuine attack signal reaches **thousands of sigma on REAL channels**
(HAI real-channel max |z| = 3860; 1428 test windows > 10) — the documented reason `CLIP=None` there. A
universal ±10 would destroy real signal and break I1/I2. The blow-up FIX 2 kills is specifically the
÷1e-8 constant-column artifact. On HAI/SWaT the residual constant-column blow-ups (35 / 7 windows) live
exclusively in **Easy** windows and enter no difficult/double-hard scored input.

### FIX 3 — consistent difficulty stratification (`build_scores_table.py`)
Compute the difficulty axis `maxz` on the **UNCLIPPED** train-standardized windowed features for **every**
dataset (byte-identical rule, no per-dataset branch), so the SAME physical constant-channel flip
stratifies the same way everywhere:

```python
Du = E.load(name, clip=None)                    # unclipped for the difficulty axis, ALL datasets
Xtr0u, Xte0u = Du["Xn_w"], Du["Xa_w"]; C6 = Xte0u.shape[1] // 6
maxz     = np.abs(Xte0u[:, :C6]).max(1)
maxz_thr = np.quantile(np.abs(Xtr0u[:, :C6]).max(1), 0.99)
```
`eda_real.load` clips WADI raw ±10 before windowing but leaves HAI/SWaT unclipped; feeding that into
`maxz` capped WADI's flips at ~10 (→ Difficult) while HAI/SWaT saw them at 1e7–1e9 (→ Easy). With
`clip=None` the rule is identical (I3) and re-derives the subset with **no hand-removal**:

| dataset | difficult (clipped, old) | difficult (unclipped, FIX 3) | moved to Easy |
|---|---|---|---|
| **WADI_clean** | 43 | **30** | 13 (incl. the 12 constant-flip windows) |
| HAI | 167 | **167** | 0 (identical, thr 3.84) |
| SWaT_canon | 85 | **85** | 0 (identical, thr 5.03) |

The rule is confirmed byte-identical and HAI/SWaT are exactly unchanged (thresholds reproduce to full
precision), so I2/I3 hold.

**The 12 removed WADI windows** (constant STATUS/setpoint flips, |z|≈1e9 under the old FIX-2 bug):
windows 204,248,283,285,490,491,498,501,502,503,504,506 on channels `1_MV_002_STATUS`, `2_MCV_007_CO`,
`1_P_006_STATUS`, `2_PIC_003_SP`. On these 12 alone every learned method scored ≈0.99 AUROC (trivially
separable), vs 0.86 for LinRes — including them in "Difficult" inflated the community model from a clean
0.758 to 0.824.

---

## Table 3 — per-dataset, per-subset [AUROC / best-F1], 5 seeds (CLEAN)

`LatAD` = single-latent global density; `HC_coh` = cohesion-weighted HC community aggregation (density
only); `HCcoh+LatAD` = the fused community+LatAD headline.

### WADI_clean  (All 575 / Easy 26 / Difficult **30** / DoubleHard 19)
| method | All | Easy | Difficult | DoubleHard |
|---|---|---|---|---|
| trivial max\|z\| | 0.786/0.62 | 0.999/0.98 | 0.601/0.13 | 0.493/0.03 |
| IsolationForest | 0.725/0.39 | 0.829/0.42 | 0.634/0.17 | 0.530/0.11 |
| AutoEncoder | 0.792/0.55 | 0.981/0.82 | 0.628/0.18 | 0.535/0.12 |
| LinRes (one-hot) | 0.834/0.68 | 0.932/0.73 | 0.750/0.53 | 0.606/0.21 |
| USAD | 0.757/0.49 | 0.962/0.73 | 0.579/0.16 | 0.466/0.08 |
| TranAD | 0.786/0.54 | 0.986/0.74 | 0.613/0.20 | 0.489/0.08 |
| LatAD (global density) | 0.717/0.34 | 0.813/0.32 | 0.634/0.18 | 0.571/0.12 |
| **HC_coh (community)** | 0.843/0.61 | 0.898/0.60 | **0.795/0.47** | 0.696/0.29 |
| **HCcoh+LatAD (headline)** | 0.827/0.57 | 0.892/0.58 | 0.771/0.42 | 0.662/0.23 |

### HAI  (All 14819 / Easy 485 / Difficult 167 / DoubleHard 84)
| method | All | Easy | Difficult | DoubleHard |
|---|---|---|---|---|
| trivial max\|z\| | 0.806/0.67 | 0.966/0.80 | 0.340/0.00 | 0.349/0.00 |
| IsolationForest | 0.844/0.42 | 0.919/0.47 | 0.627/0.06 | 0.635/0.03 |
| AutoEncoder | 0.923/0.73 | 0.980/0.80 | 0.757/0.34 | 0.730/0.25 |
| LinRes (one-hot) | 0.779/0.50 | 0.846/0.60 | 0.586/0.12 | 0.465/0.00 |
| USAD | 0.843/0.69 | 0.969/0.82 | 0.477/0.02 | 0.442/0.01 |
| TranAD | 0.834/0.69 | 0.968/0.82 | 0.444/0.01 | 0.418/0.01 |
| LatAD (global density) | 0.933/0.71 | 0.975/0.78 | 0.811/0.36 | 0.806/0.25 |
| HC_coh (community) | 0.932/0.75 | 0.977/0.81 | 0.801/0.39 | 0.737/0.24 |
| **HCcoh+LatAD (headline)** | 0.948/0.76 | 0.984/0.82 | **0.845/0.36** | 0.814/0.25 |

### SWaT_canon  (All 1498 / Easy 148 / Difficult 85 / DoubleHard 59)
| method | All | Easy | Difficult | DoubleHard |
|---|---|---|---|---|
| trivial max\|z\| | 0.871/0.77 | 1.000/0.97 | 0.646/0.21 | 0.581/0.12 |
| IsolationForest | 0.821/0.68 | 0.933/0.87 | 0.627/0.26 | 0.578/0.15 |
| AutoEncoder | 0.899/0.80 | 0.996/0.98 | 0.729/0.35 | 0.666/0.19 |
| LinRes (one-hot) | 0.920/0.82 | 1.000/0.98 | 0.782/0.39 | 0.688/0.17 |
| USAD | 0.873/0.80 | 0.996/0.98 | 0.658/0.27 | 0.609/0.14 |
| TranAD | 0.873/0.80 | 0.998/0.98 | 0.655/0.27 | 0.604/0.14 |
| LatAD (global density) | 0.927/0.80 | 0.998/0.95 | 0.804/0.44 | 0.728/0.27 |
| HC_coh (community) | 0.919/0.80 | 0.974/0.91 | 0.822/0.55 | 0.757/0.42 |
| **HCcoh+LatAD (headline)** | 0.938/0.83 | 0.995/0.92 | **0.837/0.57** | 0.773/0.43 |

## Table 4 — double-hard subset AUROC (in the DoubleHard column above)
DoubleHard sizes: WADI **19** (was 27), HAI 84, SWaT 59. Headline HCcoh+LatAD: WADI 0.662, HAI 0.814,
SWaT 0.773; HC_coh-alone WADI 0.696.

## Table 5 — source of gain (cross-channel latent vs channel-independent marginal product), difficult subset
| dataset (n_diff) | cross-channel | marginal product | gain |
|---|---|---|---|
| WADI_clean (30) | 0.656 ± 0.016 | 0.639 ± 0.004 | **+0.017** (≈0) |
| SWaT_canon (85) | 0.795 ± 0.010 | 0.654 ± 0.002 | **+0.141** |
| HAI (167) | 0.789 ± 0.018 | 0.589 ± 0.004 | **+0.200** |

Cross-channel density carries the gain on HAI and SWaT; on cleaned WADI it is essentially zero (WADI's
difficult anomalies are single-channel/linear). Conclusion unchanged from the paper (paper WADI gain was
−0.004 at 43 windows; +0.017 at the clean 30).

## Table 6 — difficult-subset AUROC by score head, 5 seeds
| head | WADI_clean (30) | HAI (167) | SWaT_canon (85) |
|---|---|---|---|
| reconstruction residual (dropped) | 0.740 ± 0.002 | 0.695 ± 0.004 | 0.835 ± 0.000 |
| latent density | 0.656 ± 0.016 | 0.802 ± 0.011 | 0.794 ± 0.006 |
| nearest-component NLL | 0.603 ± 0.010 | 0.797 ± 0.008 | 0.723 ± 0.015 |
| base (density + nearest) | 0.634 ± 0.011 | 0.801 ± 0.010 | 0.775 ± 0.009 |
| base + resid (auto) = LatAD-global | 0.634 ± 0.011 | 0.820 ± 0.010 | 0.807 ± 0.008 |

WADI recomputed on the clean 30-window subset (FIX 2 clip); HAI/SWaT unchanged (I2). Head ordering
preserved on WADI (recon > density > base; resid/basin gate off, consistent with the paper).

## Table A1 — factorization / aggregation ladder (difficult-subset AUROC, headline experts_full)
| variant | WADI_clean | HAI | SWaT_canon |
|---|---|---|---|
| LatAD-global, no factorization | 0.634 | 0.811 | 0.804 |
| sum of community surprises (+LatAD) | 0.698 | 0.827 | 0.805 |
| max community (+LatAD) | 0.712 | 0.809 | 0.804 |
| cohesion-max (+LatAD) | 0.741 | 0.832 | 0.829 |
| HC unweighted (+LatAD) | 0.760 | 0.793 | 0.826 |
| **cohesion-weighted HC (+LatAD), headline** | **0.771** | **0.845** | **0.837** |
| HC + null expert, max-fused (null+HC) | 0.749 | 0.770 | 0.811 |
| HC_coh density-only (no LatAD) | 0.795 | 0.801 | 0.822 |

Factorizing the density over regime communities still lifts the difficult subset most on the
overlapping-regime datasets (HAI, SWaT). On WADI the aggregation ladder rises from LatAD-global 0.634 to
HC_coh 0.795, i.e. the WADI signal is the community density aggregation, not the LatAD post-processing.

## Table A2 — feature-representation check (single-latent global density, difficult subset)
HAI/SWaT unchanged (I2): six-statistics 0.743 / 0.750, ten-feature temporal/spectral 0.814 / 0.776
(temporal − statistics +0.072 / +0.026). WADI's single-latent global density shifts with the 30-window
subset (LatAD-global 0.634); the qualitative conclusion (representation is a secondary, smaller lever;
the latent-density mechanism carries the result) is unchanged. WADI A2 was not re-run in this pass; a
focused re-run (~5 min, 5 seeds CPU) would refresh its WADI column.

---

## Significance (episode + moving-block bootstrap, 2000–4000 reps, 95% CI, one-sided P)

**Difficult subset — HCcoh+LatAD vs strongest baseline:**
| dataset | headline | baseline | diff | 95% CI | P(diff≤0) | episodes |
|---|---|---|---|---|---|---|
| HAI | 0.845 | AE 0.757 | **+0.088** | [0.042, 0.157] | **0.000** | 26 |
| SWaT_canon | 0.837 | LinRes 0.782 | **+0.055** | [0.012, 0.104] | **0.007** | 23 |
| WADI_clean | 0.771 | LinRes 0.750 | +0.020 | [−0.161, 0.205] | 0.460 | 8 |

**Double-hard subset — HCcoh+LatAD vs strongest baseline:**
| dataset | diff | 95% CI | P(diff≤0) | episodes |
|---|---|---|---|---|
| HAI | **+0.084** | [0.019, 0.189] | **0.0005** | 19 |
| SWaT_canon | **+0.085** | [0.030, 0.146] | **0.0005** | 18 |
| WADI_clean | +0.056 | [−0.244, 0.305] | 0.426 | 7 |

**WADI difficult — both reporting choices vs LinRes (clean 30-window / 19-window subsets):**
| method | Difficult diff (CI, P) | DoubleHard diff (CI, P) |
|---|---|---|
| **HC_coh-alone** (0.795) | **+0.044** [−0.150, 0.234] P=0.40 | +0.090 [−0.230, 0.344] P=0.33 |
| HCcoh+LatAD fused (0.771) | +0.020 [−0.161, 0.205] P=0.46 | +0.056 [−0.244, 0.305] P=0.43 |
| LatAD-global (0.634) | −0.116 [−0.315, 0.090] P=0.86 | −0.035 [−0.308, 0.199] P=0.64 |

WADI is not significant either way (8 episodes, wide CI) — as in the paper (P=0.32). HC_coh-alone is the
better clean point estimate and does not depend on the LatAD fusion, which is actively harmful on WADI's
single-channel/linear difficult anomalies.

---

## Per-table reconciliation vs current paper (old → new)

**Difficult-subset headline (Table 3 / Fig 3), community model:**
| dataset | old | new | note |
|---|---|---|---|
| HAI | 0.849 | **0.845** | FIX 1 only; FIX 2/3 no-op. Still significant vs AE (P=0.0). |
| SWaT_canon | 0.840 | **0.837** | FIX 1 only. Still significant vs LinRes (P=0.007). |
| WADI_clean | 0.824 | **0.771 fused / 0.795 HC_coh** | FIX 1+2+3; 43→30 subset. Never significant. |

**Difficult-subset baselines (Table 3):** HAI/SWaT **identical** to the paper (I2). WADI every method drops
because the 12 trivially-separable flip windows left "Difficult": IF 0.681→0.634, AE 0.739→0.628,
LinRes 0.787→0.750, USAD 0.685→0.579, TranAD 0.722→0.613, LatAD-global 0.734→0.634.

**Significance (Fig 3):** HAI +0.092 [0.047,0.157] → **+0.088 [0.042,0.157] P=0.0**; SWaT +0.058
[0.015,0.107] P=0.003 → **+0.055 [0.012,0.104] P=0.007**; WADI +0.037 (11 ep) → +0.020 (8 ep), still n.s.

**Table 5:** HAI +0.200, SWaT +0.141 unchanged; WADI −0.004 (43) → +0.017 (30), still ≈0.
**Table 6:** HAI/SWaT unchanged; WADI heads recomputed on 30 clean windows (recon 0.813→0.740, density
0.758→0.656, base/LatAD 0.743→0.634); ordering preserved.
**Table 4 double-hard:** HAI 0.814, SWaT 0.773 (FIX 1, from 0.819/0.775); WADI 27→19 windows, LatAD/HC_coh
0.696, fused 0.662.
**Table A1:** headline row 0.824/0.849/0.840 → **0.771/0.845/0.837**; WADI HC_coh density-only 0.795.

---

## Invariants

- **I1 (HAI/SWaT difficult wins stay significant): PASS.** HAI Difficult P=0.0, DoubleHard P=0.0005; SWaT
  Difficult P=0.007, DoubleHard P=0.0005.
- **I2 (unchanged paths reproduce stored per-seed scores): PASS.** HAI/SWaT baselines reproduce Table 3
  exactly; SWaT retrained LatAD reproduces stored difficult AUROC to 4 dp (per seed 0.8039/0.8111/0.8077/
  0.7917/0.8050), HAI to ±0.008 (rank-corr ≥0.988; torch-CPU cross-machine drift, documented).
- **I3 (difficulty rule byte-identical across datasets): PASS.** Single expression
  `maxz=|Xu[:, :C6]|.max(1)`, `thr=quantile(|Xu_tr[:, :C6]|.max, .99)` on `E.load(name, clip=None)`; no
  per-dataset branch. HAI/SWaT thresholds reproduce to full precision.
- **I4 (no |z| above the clip in any scored input): PASS for WADI** (max|z| = 10.0 exactly; the flip
  blow-up 5.9e14 → 7.5). For HAI/SWaT the clip is `None` by the documented `eda_real.CLIP` convention
  (their large |z| are genuine real-channel attack signal, not the ÷1e-8 artifact); the residual
  constant-column blow-ups there sit only in Easy windows and enter no difficult/double-hard scored
  input. This is the scoped reading of I4 that is consistent with the empirically-necessary `CLIP=None`.
- **I5 (removing the leak cannot spuriously RAISE a number): PASS, explained.** FIX 1 *lowers* the fused
  HAI/SWaT headline (0.849→0.845, 0.840→0.837), as expected from removing a small favorable leak. The one
  increase in the intermediate analysis (WADI dirty-43 fused 0.824→0.833 under train-normal calibration)
  is a legitimate fusion-scale correction, not label leakage: the test-normal scale was slightly
  mis-calibrated for WADI. In the FINAL clean numbers the 43→30 subset change dominates and WADI drops to
  0.771/0.795. No final number rises for a spurious reason.

## Headline-conclusion flags

- **No HAI/SWaT win is lost.** Both remain significant on difficult and double-hard.
- **WADI difficult community headline moves 0.824 → 0.771 (fused) / 0.795 (HC_coh-alone).** The paper text
  "LatAD reaches 0.824 on WADI, ahead of every baseline; the regime-community factorization lifts the
  single-latent global density (0.734) to this WADI lead" needs rewording: on the clean subset the
  single-latent LatAD is 0.634 (below LinRes 0.750) and the lead is carried by the **HC_coh community
  density aggregation (0.795)**, not by LatAD. The WADI lead was and remains **not statistically
  significant**. Recommend reporting **HC_coh-alone 0.795** for WADI difficult.

## Artifacts written (all under `poc/_diagnostics/`)
- `scores_WADI_clean.npz` rebuilt (FIX 2 + FIX 3 + `LatAD_train`); `scores_HAI.npz`/`scores_SWaT_canon.npz`
  gained `LatAD_train` (test scores untouched). Originals in `clean_recompute_backup/`.
- `clean_ensemble_WADI.json`, `clean_ensemble_HAI_SWaT_sig.json` (Tables 3/4 + significance).
- `clean_wadi_tables56.json` + `clean_wadi_tables56.py` (WADI Tables 5/6).
- `fable_leak_calib_*.json` (prior FIX-1 validation, reused).
- Reproduce: `EXPERTS_DIR=sota_bundle/experts_full HEAD=HCcoh+LatAD BOOT_REPS=2000 python ensemble_final.py <DS>`.

---

# Auto-gated fusion of the LatAD-null term — VERDICT: NOT PRINCIPLED (fall back to fused-everywhere)

**Goal.** Auto-gate the LatAD-null fusion term `zl` in `HCcoh+LatAD = z(hc_coh) + zl`: ON where it
generalizes (HAI/SWaT, where it makes the win significant), OFF on WADI (revert to HC_coh, since on clean
WADI the term hurts: fused 0.827 All / 0.771 Diff vs HC_coh 0.843 / 0.795). Hard constraint: the gate
must decide from **train-normal data only**, no peeking at test scores/labels or the WADI difficult
result.

**Gate rule (mirrors `models_vade.fit_resid_head`'s existing auto-gate).** Fit the global VaDE on the
80% train-normal slice A (with the FIX-2 clip), score the fit slice A (in-sample) and the held-out 20%
slice B, and test whether the global LatAD score generalizes to held-out normal:
`ratio = q95(sB) / q95(sA)`, gate ON iff `ratio < 1.5` (the residual head's own threshold). Uses the
FIX-1 train-normal split; no test data touched. Variants R2 (`frac(sB>q99(sA))`), R3
(`(mean sB-mean sA)/std sA`), R4 (`q99(sB)/q99(sA)`) reported for robustness.

**Decision variable per dataset (train-normal only, 5-seed mean; `clean_gate_probe.json`):**
| dataset | R1 q95(B)/q95(A) | R2 tail-exceed q99(A) | R3 z-shift | R4 q99(B)/q99(A) | gate (ON if R1<1.5) | NEEDED |
|---|---|---|---|---|---|---|
| **WADI_clean** | **1.10** ± 0.02 | 0.018 (≈ideal 0.01) | +0.20 | 1.16 | **ON** | **OFF** |
| HAI | **2.66** ± 0.31 | 0.170 | +0.88 | 1.89 | **OFF** | ON |
| SWaT_canon | 0.59 ± 0.05 | 0.000 | −0.51 | 0.52 | ON | ON |

**This is the exact OPPOSITE of the required pattern, on every variant (R1–R4).** WADI's global LatAD
score generalizes to held-out normal **better** than HAI's (WADI ratio 1.10, near-ideal 1.8% tail
exceedance; HAI ratio 2.66, 17% tail exceedance). The residual-head gate would turn the term **ON for
WADI and OFF for HAI** — precisely wrong, and it would kill the biggest fused win (HAI). No monotone
threshold on any of R1–R4 puts WADI on one side and both HAI and SWaT on the other.

**Why (root cause, from the recompute).** WADI's failure of the LatAD-null term is an **anomaly-side**
property: its difficult attacks are single-channel / linear, so the cross-channel global density adds no
discrimination (Table 5 WADI gain +0.017; LatAD-global difficult 0.634, below LinRes 0.750). That is a
property of the *attacks*, which do not exist in train-normal, so it is unobservable without test data.
Conversely HAI's train-normal is genuinely non-stationary (regime shifts), so held-out normal looks
"anomalous" (poor generalization), yet the global LatAD still discriminates real attacks strongly
(0.811). Held-out-normal generalization therefore measures false-alarm calibration, not the
attack-geometry mismatch that actually hurts WADI — and it is anti-correlated with the target here.

**Would any other train-normal criterion work?** No principled one was found. Deciding "will the anomalies
be single-channel" requires anomalies; train-normal has none. There is no degeneracy signal either (WADI's
held-out spread is mild, R3 +0.20). Choosing the *reverse* direction ("gate OFF when it generalizes
well") would be reverse-engineering the sign from the test outcome (it would also gate off SWaT), which
the honesty constraint forbids.

**VERDICT: NOT PRINCIPLED.** A train-normal-only gate cannot independently produce off-WADI / on-HAI-SWaT.
Do not ship an auto-gate reverse-engineered from the WADI test result.

**Recommended fallback (per the coordinator's stated contingency): report the fused `HCcoh+LatAD`
everywhere, state WADI honestly.**
| dataset | fused HCcoh+LatAD, difficult | significance vs strongest baseline |
|---|---|---|
| HAI | 0.845 | +0.088, 95% CI [0.042, 0.157], **P=0.000** (significant) |
| SWaT_canon | 0.837 | +0.055, 95% CI [0.012, 0.104], **P=0.007** (significant) |
| WADI_clean | 0.771 (All 0.827) | +0.020, 95% CI [−0.161, 0.205], P=0.46 (numerical, n.s.) |

The fused detector is the single reported architecture (it is what makes HAI/SWaT significant; HC_coh-alone
is n.s.). On WADI the LatAD-null term is mildly counterproductive (the community HC_coh 0.795 would lead
0.771), but WADI is not significant either way, so state it plainly: on WADI's single-channel/linear
difficult anomalies the cross-channel/global term adds nothing and the fused detector's numerical lead
over the linear baseline is not significant.

**Artifacts:** `clean_gate_probe.py`, `clean_gate_probe.json`, `clean_gate_probe.jsonl` (per-seed,
incremental). Reproduce: `python _diagnostics/clean_gate_probe.py`.
