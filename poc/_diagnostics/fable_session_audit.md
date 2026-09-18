# Adversarial audit of the 2026-09-17 revision-2 research session (Fable, report-only)

Scope: every claim in the audit brief plus the coordinator's mid-audit correction. Nothing in the paper,
models, checkpoints or registry was modified. All verification ran in-process on CPU from the stored
artifacts; the only retraining was a light LOF re-check (WADI and SKAB 5 seeds, HAI 2 seeds, ~3 min).
Scratch output: `E:/tmp/claude/E--Projects-Backlog-LatAD/6ae0c3f3-.../scratchpad/lof_recheck.json`.

Canonical difficult-subset definition used throughout (the paper's, `improve_multiseed.py:43`,
`build_scores_table.py`): `triv = max|z|` over the first sixth of the window features (per-channel means),
`thr = 99th pct of TRAIN triv`, `easy = (y==1) & (triv > thr)`, `hard = (y==1) & ~easy`,
`difficult-AUROC = AUROC over (y==0) | hard` (all test normals vs the subtle anomalies). For WADI the paper's
loader clips standardized inputs at +-10 sigma (`eda_real.CLIP["WADI"] = 10.0`) BEFORE windowing, so `triv` is
computed on clipped features.

---

## 0. Prioritized findings

| # | Severity | Finding | Files |
|---|---|---|---|
| F1 | **BLOCKING (definition bug, three occurrences)** | The "difficult subset" was inverted in three places this session: (a) the coordinator's cleaned-WADI re-score, (b) the coordinator's canonical-SWaT numbers (0.895 / 0.895 / 0.790), (c) `adaptive_density.py` (the LOF study). All three used `mask = triv > thr` over ALL windows, i.e. the EASY anomalies (plus the few normals above threshold), not the subtle ones. | see F1 detail |
| F2 | **BLOCKING (loader bug, unfixed)** | `eda_real.CLIP` has no `"WADI_clean"` key, so `E.load("WADI_clean")` runs unclipped. `scores_WADI_clean.npz` (rebuilt 12:48, now with USAD/TranAD) is that unclipped build: LatAD/AE scores reach 1e32 / 1e36, 13 subtle anomalies are mis-filed as "easy" by STATUS channels with train std 1e-8 flipping to |z| = 7e6..7e9, and the file mixes unclipped LatAD/AE/IF/linres/l2 with USAD/TranAD that Modal scored on the CLIPPED `wadi_clean_*.npy`. The coordinator's "corrected" 30-window numbers (LatAD 0.620 = AE 0.620, linres 0.723, TranAD 0.613, USAD 0.579) come from this broken table. | `poc/eda_real.py:201`, `poc/_diagnostics/scores_WADI_clean.npz` |
| F3 | **construct-match (Modal HARD mask)** | `sota_bundle/wadi_clean_triv_test.npy` and `wadi_clean_triv_thr.npy` are byte-identical to the DIRTY `wadi_triv_test.npy` / `wadi_triv_thr.npy` (thr 18.90, values up to 1e10, median 49 252, driven by the dropped `2B_AIT_002_PV`). Every `raw_HARD` / `raw_EASY` cell for `WADI_clean` in `sota_matrix.json` (TranAD ~0.747, USAD ~0.702, n_easy_hard 561/436) is computed on a mask defined by a channel that is not in the data, at timestep resolution. Those cells are void. `raw_ALL` (per-timestep) is valid but not the paper's window-level construct. | `poc/sota_bundle/wadi_clean_triv_*.npy`, `poc/sota_bundle/results/sota_matrix.json`, `poc/sota_bundle/prep_sota_general.py` (`pfx` / `CLIP.get` logic never produced a clean triv file) |
| F4 | **construct-match (which LatAD)** | Every comparison this session (clean WADI, canonical SWaT, LOF, E2, E5 reference arm) scores the `LatAD (global density)` configuration (`train_vade` + `anomaly_score_hard`). The paper's headline row is `LatAD (regime-community)` (WADI difficult 0.796 / all 0.862; global density is 0.690 / 0.792). No clean-WADI or canonical-SWaT number exists for the headline model. "LatAD ties AE on clean WADI" is a statement about the ablated global-density model. | `poc/paper/IoT2.html` Table 3 rows 802-803, `poc/gen_tables.py` (`HCcoh+LatAD`) |
| F5 | **BLOCKING for the registry entry (conclusion reversed)** | Re-run on the canonical mask, LOF is NOT worse than the fixed GMM head on the subtle anomalies: WADI 0.718 vs 0.694 (5 seeds), SKAB 0.568 vs 0.473 (5 seeds), HAI 0.789 vs 0.790 (2 seeds). The registry sentence "difficult-AUROC lof < gmm everywhere" describes the EASY subset (my inverted-mask recompute reproduces the JSON: WADI 0.812 vs 0.842, SKAB 0.914 vs 0.947, HAI 0.961 vs 0.973). The stored `boot_adap_minus_*` CIs are on the wrong subset too. | `poc/_diagnostics/adaptive_density.py:150-152`, `adaptive_density.json`, `adaptive_density_SKAB.json`, `RESEARCH_REGISTRY.md` §5 |
| F6 | **overclaim (metric at its ceiling)** | `sparse_fpr` in the LOF study is capped at 0.10 by construction (1 % of all normals flagged / 10 % sparsest normals). HAI 0.1002 and SKAB 0.1053 for all three scorers are the ceiling, not a measurement; only WADI (0.115 -> 0.019) and SWaT (0.108 -> 0.088) moved. "LOF reduces sparse-normal FPR where designed (HAI/SKAB unchanged)" should read "metric saturated on HAI/SKAB". | `adaptive_density.py` (`sparse_fpr`) |
| F7 | **SOLID (with one wording fix)** | WADI artifact: `2B_AIT_002_PV` train mean 9.09 / std 0.164, attack-file normal-row mean 4503, z = 27 359. It is the only channel with a break of this order; the next largest test-normal shifts are the three AIT_004 analyzers at about 2 sigma (dirty-bundle rank: ch 102 clipped 5.36, ch 104 -2.03, ch 3 -2.02, ch 100 -1.83; only 3 of 123 channels exceed 2 sigma). Dropping exactly one channel is not cherry-picking; it is the single instrumentation break. Window counts unchanged (575 test / 519 normal / 56 anomalies, labels identical). The difficulty mask changes correctly ONLY on the clipped build (19 dirty-difficult -> 43 clean-difficult, dirty set is a strict subset). | `poc/eda_real.py:_raw_wadi_clean`, `cov_wadi_sensors.log`, `cov_coverage.json` |
| F8 | **SOLID** | Clean-WADI tie (global-density LatAD vs AE) holds on the clipped, construct-matched table and under an episode-block bootstrap: LatAD 0.734 +- 0.008 vs AE 0.739 +- 0.003; iid-window CI [-0.034, +0.026] (reproduced), episode-block CI [-0.022, +0.011] (14 episodes resampled, normals fixed) and [-0.027, +0.015] (episodes + normal blocks of 10), P(diff <= 0) 0.65-0.71; per-seed paired differences [-0.001, -0.004, +0.009, -0.017, -0.012]. LatAD vs IF +0.05 to +0.06, block-bootstrap p 0.07-0.08 (not significant). Seeds are paired correctly (same resample, per-seed AUROC then mean, both methods). | `fable_cleanwadi_scores_clip10_default.npz`, `fable_cleanwadi_analysis.json` |
| F9 | **corrected numbers (new, construct-matched)** | Clean WADI, clipped, canonical 43 subtle vs 519 normals, all seven methods on identical windows (USAD/TranAD = Modal per-timestep dumps on the clipped clean arrays, window-averaged exactly as `build_scores_table.py` does): **linres 0.787, l2 0.751, AE 0.739, LatAD-global 0.734, TranAD 0.722, USAD 0.685, IF 0.681.** LatAD-global minus TranAD +0.012 CI [-0.050, +0.074]; minus USAD +0.048 CI [-0.018, +0.115]. The whole field is a tie inside 0.68-0.79; the linear residual leads. | this audit, section 2 |
| F10 | **corrected numbers (canonical SWaT)** | `scores_SWaT_canon.npz` on the canonical mask (85 subtle of 233 anomalies, 1265 normals): **LatAD-global 0.804 +- 0.007, linres 0.782, AE 0.729 +- 0.007, l2 0.690, IF 0.627 +- 0.007; all-AUROC 0.927 / 0.920 / 0.899 / 0.883 / 0.821.** The brief's 0.895 / 0.895 / 0.790 is exactly `auc(y[triv>thr], s[triv>thr])` = 148 easy anomalies vs 9 normals (reproduced to three decimals). On the correct subset LatAD-global leads AE by +0.075 and IF by +0.18, not a tie. No USAD/TranAD/regime-community numbers exist on canonical SWaT yet. | `scores_SWaT_canon.npz` |
| F11 | **SOLID (loader)** | Canonical SWaT alignment is by channel NAME (`sens` from the npz, 51 channels, `df[sens]` on our normal.csv), split is the mandated one (Normal recording = train, `SWaT_Dataset_Attack_v0` = test). npz: 449 919 rows, 54 621 attack rows = 12.14 % (the published figure; the `"A ttack"` label variant is therefore included), 35 contiguous attack segments after adjacent merges (36 documented attacks). Config (40, 16) matches `build_scores_table.CFG["SWaT"]` and the exported `vade_SWaT.pt`. The "median 0.2 % per-channel diff" scale check is not in any stored log (only the conversion log survives); it is a docstring assertion, unverified here. | `poc/eda_real.py:_raw_swat_canonical`, `swat_canon_cache.log` |
| F12 | **SOLID (genuine null)** | E2 nearest vs mixture: from `e2_fable_HAI.npz`, log pi spans [-10.07, -2.78] (pi 4e-5 to 0.062, so pi is far from uniform), the mixture-minus-nearest penalty on test is median 3.1 nats, p95 5.5, max 8.0, Spearman(mixture, nearest) = 0.993, and the train-99 % threshold flags differ on 27 of 14 167 test windows. The null is real: pi changes scores by a few nats but not their ranking. Registry wording "4-8 nats" should be "median 3, up to 8". | `e2_fable_HAI.npz`, `e2_nearest.json` |
| F13 | **minor overclaim** | "The detection head never reads VaDE pi." True for the base score (`anomaly_score_hard` = sklearn `latent_gmm` density, K=80, plus `max_c log N`, no pi) and for the basin head (argmax). False when the auto-gated residual head fires: `_resid_score` weights per-mode Mahalanobis by `G = softmax(log pi + log N)`; the gate is ON for HAI and for SWaT (`export_swat.log`: `resid_auto=True`). So on HAI/SWaT pi enters the shipped score through the residual weights. E2 remains a null for the reason in F12, not because pi is unused. | `poc/models_vade.py:197-212, 295-299` |
| F14 | **SOLID** | "Four anti-collapse measures already ship": KL warm-up (`beta = min(1, (ep+1)/warmup)`), DAGMM-style `cov_reg` on 1/var, hard variance floor `logvar_floor = log 0.05` via `_lvc()` clamp, mixture params at `0.1 x lr` from a `n_init=3` GMM initialisation. All four verified in `train_vade` / `VaDE`. | `poc/models_vade.py:81, 94-96, 106-133, 380-440` |
| F15 | **SOLID** | HAI starvation sweep: all 12 rows (2 seeds x 6 ratios, 9.7 % -> 0.014 % of train) have `fpr_val_target = 0.0` at the train-99 % threshold and at matched 1 % other-regime FPR, for both heads. The target regimes score BELOW the rest at ratio 1 (median nearest-NLL 9.7 vs 14.3), i.e. they are core, in-support regimes; the memo's reading (no dynamic range, uninformative null) is correct. | `e2_fable_sweep.jsonl` |
| F16 | **registry: WADI line conflates two constructs** | Registry §2/§3: "Excluding it moves LatAD AUROC by 0.001 but lifts AE/USAD/TranAD by +0.1 to +0.24" is the test-normal EXCLUSION on the dirty model. The actual clean RE-RUN moves LatAD-global difficult 0.690 -> 0.734 and AE 0.425 -> 0.739: the entire global-density difficult margin over AE on WADI is the artifact. The registry has no clean-WADI table yet; when added it must be F9, and the line "part of LatAD's WADI margin is artifact-robustness" must become "the global-density WADI difficult margin over AE/USAD/TranAD is artifact-robustness; the regime-community margin is untested on clean WADI". | `RESEARCH_REGISTRY.md` §2 row 5, §3 loader audit |
| F17 | **registry: minor mismatches** | (a) E5 "5-seed": WADI has 4 seeds (1-4; seed 0 absent), HAI/SWaT 5. Gains reproduce: +0.343 / +0.206 / +0.078. (b) E3 "community 17-52 ms": cuda rows give b1 latency 32.4 / 18.5 / 17.2 ms; 52 is the WADI b256 batch latency (51.6 ms), not b1. (c) A3 SKAB 0.500 -> 0.605 is the FORCED lambda = 1.0 arm; the auto gate gives lambda = 1.075 and no auto-arm row is stored. (d) E2 "4-8 nats" (see F12). (e) LOF paragraph: see F5/F6. Everything else checked (E4 HAI +0.054 CI [0.012, 0.089] P 0.005; WADI 5 / SWaT 1 episodes; E3 train_only_s TranAD 231 / 898 / 163, USAD 96 / 592 / 95; LatAD 3.2 / 16 / 3.5; CPU community 25 / 16 / 14.5; E2 HAI 0.3552 vs 0.3516) matches its JSON. | `rev4_timeaware.json`, `e5_gain_cuda.json`, `e3_rows_*.jsonl`, `a3_skab_witness.json` |
| F18 | **minor (bookkeeping)** | `fable_cleanwadi_report.md` §6 still contains `SWEEP_PLACEHOLDER`. `anomaly_score_hard` docstring says the reported model is density-only while `build_scores_table.py` calls it with `use_near=True` (density + nearest); the paper text also says "mixture density plus nearest-component likelihood", so the code path is right and the docstring is stale. | |

---

## 1. F1 in detail: the inverted difficulty subset

Three implementations this session computed `mask = triv > thr` over all windows and then either
`roc_auc_score(y[mask], s[mask])` (coordinator) or `AUROC over (y==0) | mask` (`adaptive_density.py`).
Both are the EASY anomalies:

| Artifact | Construction | What it measured | Reported | Canonical (correct) |
|---|---|---|---|---|
| coordinator clean-WADI re-score (unclipped table) | `auc(y[triv>thr], s[triv>thr])` | 26 easy anomalies vs 4 normals | LatAD 0.731, AE 0.779 | 43 subtle vs 519 normals: LatAD 0.734, AE 0.739 (clipped) |
| coordinator canonical-SWaT | same | 148 easy anomalies vs 9 normals | LatAD 0.895, AE 0.895, IF 0.790 | 85 subtle vs 1265 normals: LatAD 0.804, AE 0.729, IF 0.627 |
| `adaptive_density.py` (`hard = maxz > thr`, then `_au(y, s, hard)`) | all normals + windows with triv > thr | easy anomalies vs normals; `n_diff` counts normals too (WADI 325 "difficult" of 575 windows, but WADI has 56 anomalies) | gmm 0.859 / lof 0.822 (WADI) etc. | see F5 |

Invariant that exposes each of them: the paper's WADI trivial rule scores 0.687 on the easy subset and 0.307 on
the difficult one; any "difficult" AUROC that comes out ABOVE the same method's all-anomaly AUROC (0.859 > 0.806
for the LOF-study GMM head; 0.895 > 0.927 fails but 0.998 easy does) is on the wrong subset. Fable's
`fable_cleanwadi_analysis.json` "canonical_difficult" (43 windows) IS the `improve_multiseed` definition; it
differs from the coordinator's 30-window count only because Fable rebuilt with `clip = 10` (paper convention) and
the coordinator's file is unclipped (F2). The 13 windows that move are exactly the STATUS/CO flip windows
(`1_MV_002/003_STATUS`, `2_MCV_007_CO`, `1_P_006_STATUS`, `2_PIC_003_SP`): unclipped `triv` 7.5e6 to 7.5e9,
clipped 3.0 to 6.0.

Consumers of `build_scores_table` outputs must use: `hard = (label==1) & (maxz <= maxz_thr)`, score over
`(label==0) | hard`, per-seed AUROC then mean over the seed axis. Other datasets checked: `scores_SWaT.npz`
(mirror) reproduces the paper's global-density 0.960 under this rule; `e2_nearest.py` uses the canonical
`hard = (yw==1) & ~easy` (correct); `rev4_stats_timeaware.py` numbers reproduce (correct); `adaptive_density.py`
is the only script with the inversion.

---

## 2. Corrected numbers

### 2a. Clean WADI (clipped +-10 sigma, W=60 / stride 30, 575 windows, 56 anomalies = 13 easy + 43 difficult)

| method | difficult (43 vs 519) | easy (13) | all (56) |
|---|---|---|---|
| linres (one-hot LOCO) | 0.787 | 0.993 | 0.834 |
| l2 | 0.751 | 0.999 | 0.809 |
| AE (5 seeds) | 0.739 +- 0.003 | 0.999 | 0.799 |
| LatAD global density (5 seeds) | 0.734 +- 0.008 | 0.991 | 0.793 |
| TranAD (Modal, clipped clean arrays, window-avg) | 0.722 | 0.998 | 0.786 |
| USAD (same) | 0.685 | 0.994 | 0.757 |
| IF (5 seeds) | 0.681 +- 0.006 | 0.871 | 0.725 |
| LatAD regime-community (paper headline) | **not run** | | |

Paired bootstraps (seed-mean, same resample for both methods): LatAD-AE -0.005 [-0.034, +0.026] iid,
[-0.022, +0.011] episode-block; LatAD-TranAD +0.012 [-0.050, +0.074]; LatAD-USAD +0.048 [-0.018, +0.115];
LatAD-IF +0.052 [-0.011, +0.125]; LatAD-linres -0.053 [-0.142, +0.038]. Effective N = 14 attack episodes.

Dirty WADI, same construction, for reference (paper): LatAD-global 0.690, IF 0.677, AE 0.425, l2 0.445,
linres 0.392, TranAD 0.333, USAD 0.303.

### 2b. Canonical SWaT (1498 windows, 233 anomalies = 148 easy + 85 difficult, 1265 normals)

| method | difficult (85 vs 1265) | easy (148) | all (233) |
|---|---|---|---|
| LatAD global density (5 seeds) | 0.804 +- 0.007 | 0.998 | 0.927 |
| linres | 0.782 | | 0.920 |
| AE (5 seeds) | 0.729 +- 0.007 | 0.996 | 0.899 |
| l2 | 0.690 | | 0.883 |
| IF (5 seeds) | 0.627 +- 0.007 | 0.933 | 0.821 |
| USAD / TranAD / regime-community | **not run** | | |

Mirror SWaT (paper) for reference: LatAD-global 0.960, linres 0.959, AE 0.939, USAD 0.871, TranAD 0.867,
IF 0.853 (38 difficult from one episode). Canonical SWaT is a materially harder and more discriminative test
(85 subtle windows across many episodes) and, unlike the mirror, separates the methods.

### 2c. LOF vs fixed GMM density head, canonical mask (density-only head, `k_density` as shipped)

| dataset | seeds | GMM difficult | LOF difficult | GMM easy | LOF easy | stored JSON "diff" (= easy) |
|---|---|---|---|---|---|---|
| WADI (19 subtle) | 5 | 0.694 | **0.718** | 0.842 | 0.812 | 0.859 / 0.822 |
| SKAB (185 subtle) | 5 | 0.473 | **0.568** | 0.947 | 0.914 | 0.947 / 0.915 |
| HAI (167 subtle) | 2 | 0.790 | 0.789 | 0.973 | 0.961 | 0.972 / 0.960 |

Per-seed WADI LOF minus GMM: +0.036, +0.011, +0.011, +0.035, +0.025 (5 of 5 positive). SKAB: +0.097, +0.104,
+0.088, +0.120, +0.070. This is a rerun with the scripts' own recipe, not a bootstrap; it establishes only that
the registry's direction is wrong, not that LOF is a win (SKAB is near chance for both, WADI has 19 windows,
HAI is a tie, and the shipped score is density + nearest, not density alone). The mechanism paragraph in the
registry ("hardest anomalies are globally far, LOF removes that signal") was derived from the EASY subset and
must be withdrawn.

---

## 3. Bootstrap validity on clean WADI (brief item 2)

The iid window bootstrap treats 43 overlapping windows (stride 30 on W=60, 14 episodes) as independent. I re-ran
the same paired, seed-mean statistic resampling (i) episodes only and (ii) episodes plus blocks of 10 normal
windows. The CIs did not widen (they narrowed slightly, because resampling 14 unequal episodes concentrates mass
on the long ones) and the conclusion (tie with AE, no significant lead over IF or over the deep baselines) is
unchanged. The `rev4_stats_timeaware.py` episode-block machinery already used for E4 should be the one quoted
in any rebuttal; the numbers above are consistent with it. Five-seed pairing is correct in both Fable's script
and my re-run (per-seed AUROC on the same resample, then averaged, difference taken per resample).

---

## 4. Construct-match sweep (brief item 6): comparisons NOT co-computed on the same windows / model / split

| Comparison as stated this session | Mismatch | Fix |
|---|---|---|
| Clean-WADI LatAD 0.734 / AE 0.739 vs TranAD 0.747 / USAD 0.702 | LatAD/AE: 43 windows, clipped, window-level. TranAD/USAD: 436 timesteps under a mask built from the dirty `2B_AIT_002_PV` triv array, on the CLIPPED arrays, timestep-level. | Use section 2a (window-averaged Modal dumps on the same 43-window mask): TranAD 0.722, USAD 0.685. |
| `scores_WADI_clean.npz` internal (LatAD/AE/IF/linres/l2 vs USAD/TranAD) | first group unclipped (scores 1e32-1e36), second group clipped | add `"WADI_clean": 10.0` to `eda_real.CLIP`, rebuild with `build_scores_table.py WADI_clean`; result will equal `fable_cleanwadi_scores_clip10_default.npz` plus the two SOTA columns. |
| "LatAD ties AE on clean WADI" vs the paper's WADI row | session = global density; paper headline = regime-community (0.796 difficult, 0.862 all) | run the `HCcoh+LatAD` pipeline (`gen_tables.py` path) on `WADI_clean` and `SWaT_canon` before any rebuttal sentence about the headline model. |
| Canonical SWaT 0.895 = AE 0.895 > IF 0.790 | inverted subset | section 2b. |
| Canonical SWaT vs USAD/TranAD | SOTA not run on `SWaT_canon` | Modal run on a `SWaT_canon` export (needs `prep_sota_general.py` fixed to emit a per-dataset triv file). |
| Registry "LatAD delta 0.001, baselines +0.1..0.24" vs clean re-run | exclusion-on-dirty-model vs retrain-on-clean-data | quote both with their constructs, or only 2a. |
| LOF "difficult" vs GMM "difficult" | both on the easy subset | section 2c; the two heads ARE co-computed on the same latent, so once the mask is fixed the comparison is valid. |
| E2 `diff_auroc` (HAI 0.807 / 0.808) vs paper 0.811 | E2 scores VaDE's own mixture / nearest components, not the shipped `latent_gmm` head | fine as a design-note; do not present E2 numbers as LatAD numbers. |
| E5 reference arm WADI 0.683 (4 seeds) vs paper 0.690 (5 seeds) | seed 0 missing on WADI | note "4 seeds" or re-run seed 0. |
| dq_audit "effective N = episodes 5/26/1" vs clean-WADI "14 episodes" | dirty 19-window subset vs clean 43-window subset | both correct for their subset; the rebuttal must name which. |

Co-computed and valid: `fable_cleanwadi_scores_clip10_default.npz` (all local methods, clipped, one pass);
`scores_SWaT_canon.npz` (all local methods, one pass); section 2a's USAD/TranAD columns (same windows, same
clip, same labels, verified identical to the window-averaging in `build_scores_table.py`); `rev4_timeaware.json`;
`e5_gain_cuda.json` within dataset; `a3_skab_witness.json` within dataset.

---

## 5. Invariants checked (stated before computing)

1. A method's difficult-subset AUROC must not exceed its all-anomaly AUROC by more than noise when the easy
   subset is near 1.0. Violated by every inverted-mask number (F1); satisfied by every canonical number in §2.
2. Clipped and unclipped WADI_clean tables must share labels and window count (575 / 56): true. The unclipped
   difficult set must be a subset of the clipped one: true (30 of 43), the 13 extra are STATUS flips.
3. Window-averaging the Modal per-timestep dumps must reproduce the `USAD` / `TranAD` columns that
   `build_scores_table.py` wrote into `scores_WADI_clean.npz`: `allclose` true for both.
4. The coordinator's `auc(y[triv>thr], s[triv>thr])` must reproduce 0.731 / 0.779 (clean WADI, unclipped) and
   0.895 / 0.895 / 0.790 (SWaT canon) if the inversion hypothesis is right: reproduced to 3 decimals.
5. The inverted mask in `adaptive_density.py` must reproduce its JSON: WADI 0.842 vs 0.859 (density-only head,
   different `k_density` draw, within seed noise), SKAB 0.947 exact, HAI 0.973 vs 0.972.
6. The dropped channel must be the unique large break: only 3 of 123 channels exceed 2 sigma in test-normal
   mean; the artifact channel is at 27 359 sigma (clipped 10), the next two at 2.0.
7. Canonical SWaT attack fraction must equal the published 12.1 %: 54 621 / 449 919 = 12.14 %.
8. Episode-block CI must contain the iid point estimate and not flip sign: true.

---

## 6. Publication-ready vs needs-work

| Session conclusion | Status | What must happen first |
|---|---|---|
| `2B_AIT_002_PV` is an instrumentation artifact (27 359 sigma rescale); dropping it is justified and unique | **ready** (rebuttal + data-quality section) | none; cite `cov_wadi_sensors.log` numbers |
| Dirty-WADI global-density difficult lead over AE/USAD/TranAD is artifact-robustness | **ready** as a scoped statement | say "global-density configuration"; do not extend to the regime-community row |
| Clean WADI: global-density LatAD ties AE (0.734 vs 0.739), field bunched 0.68-0.79, linres leads | **ready** with the §2a table (7 methods, one construct) | regenerate `scores_WADI_clean.npz` with the CLIP fix so the archived artifact matches; retire the unclipped file |
| "Corrected" 30-window clean-WADI numbers (0.620 / 0.620 / 0.723 ...) | **withdraw** | unclipped table (F2); superseded by §2a |
| Clean-WADI headline (regime-community) row | **needs work** | run `HCcoh+LatAD` on `WADI_clean`; unknown direction |
| Canonical SWaT: LatAD-global 0.804 vs AE 0.729 / IF 0.627 / linres 0.782 | **ready for the global-density row only** | USAD/TranAD/regime-community on `SWaT_canon` before adopting canonical SWaT as the paper's SWaT; "0.895 ties AE" must be withdrawn |
| E2 nearest-vs-mixture null | **ready** (rebuttal design note) | wording: "median 3, up to 8 nats"; "pi enters the shipped score only through the auto-gated residual weights (HAI, SWaT)" |
| LOF adaptive density "no-win, reinforces design" | **withdraw from registry, rerun** | fix `hard = (y==1) & (maxz <= thr)` in `adaptive_density.py`, rerun 5 seeds x 4 datasets with the bootstrap; current evidence says LOF >= GMM on subtle anomalies for WADI/SKAB, tie on HAI |
| Sparse-normal FPR "LOF helps where designed" | **withdraw** | metric is at its 0.10 ceiling on HAI/SKAB; redefine (e.g. fraction of flagged normals that are sparse) |
| Imbalance-design memo (`fable_imbalance_design.md`) | **ready as a design memo** | its three factual anchors (HAI sweep flat at 0, anti-collapse measures, pi-only counterfactual 0) verified |
| Backbone memo (`fable_backbone_alternatives.md`) | **ready as a design memo, two edits** | "code fact 1" needs the residual-head caveat (F13); "lesson 3" (LOF loses on hard anomalies) is void (F5) and the memo's Fact A rests partly on it |
| Registry E4 / E5 / E3 / A3 headline numbers | **ready** | E5 WADI is 4-seed; E3 "52 ms" is b256; A3 auto-lambda arm not stored |
| Modal `sota_matrix.json` WADI_clean HARD/EASY cells | **void** | regenerate `wadi_clean_triv_*.npy` from the clean arrays (or drop the timestep HARD split entirely and rely on window-level §2a) |
