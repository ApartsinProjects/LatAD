# Baseline fairness / test-leak audit (symmetric check to the LatAD ensemble leak)

Scope: every scored method in the §6 comparison table (`_diagnostics/comparison_table.json`, built by
`build_comparison_table.py` from `_diagnostics/scores_<DS>.npz`, plus the 5-seed USAD/TranAD arrays
in `_diagnostics/scores_sota_ms_<DS>.npz`). The LatAD ENSEMBLE leak in `ensemble_final.py:119-139`
(`z()` uses `nm = y==0`, i.e. test-normal) is out of scope here; it is being quantified and fixed
separately. This audit asks: after that fix, is every OTHER number in the comparison calibrated on
train-normal only and AUROC-fair?

Method: code inspection of the exact scoring paths, plus two cheap numeric checks (no retraining).
Every verdict cites the line it rests on.

## Shared preprocessing (all methods)

| Step | Code | Statistic source | Verdict |
|---|---|---|---|
| Raw per-timestep standardization | `eda_real.py:248-249` `mu, sd = Xn.mean(0), Xn.std(0)+1e-8; Xn, Xa = (Xn-mu)/sd, (Xa-mu)/sd` | TRAIN-normal only | clean |
| Clip (WADI/WADI_clean/WindSCADA only, +-10) | `eda_real.py:240, 250-251` | fixed constant | clean |
| Windowing into 6 stats/channel | `eda_real.py:252-261`, `winfeat.py:15-18` | none | clean |
| Window label | `eda_real.py:258` `y[i:i+W].mean() > 0.05` | labels used only to define the target | clean |
| SOTA export (USAD/TranAD inputs) | `sota_bundle/prep_sota_general.py:20-30` same `mu, sd` from `Xn`, same clip table | TRAIN-normal only | clean |
| Second re-standardization of window features | `build_scores_table.py:51-52, 61` `mu, sig = Xtr0.mean(0), Xtr0.std(0)+1e-8; Xte = (Xte0-mu)/sig; Xtr = (Xtr0-mu)/sig` | TRAIN-normal window-feature stats | clean as a leak question; see "Artifact" below for the no-clip side effect |

No code path standardizes test windows with test statistics. `Xte0`/`Xa_w` are never re-centred on
themselves anywhere in `eda_real.load`, `build_scores_table.build`, `onehot_filter.build_feats`, or
`prep_sota_general.prep`.

## Difficulty mask and LinRes threshold

| Quantity | Code | Source |
|---|---|---|
| `maxz` (difficulty axis) | `build_scores_table.py:46-47` `np.abs(Xte0[:, :C6]).max(1)` on the train-standardized window MEANS | test features, train scale (no test stats) |
| `maxz_thr` | `build_scores_table.py:48` `np.quantile(np.abs(Xtr0[:, :C6]).max(1), 0.99)` | TRAIN-normal 99th pct |
| `lin_thr` (double-hard) | `ensemble_final.py:149` `lin_thr = float(np.quantile(r_tr, 0.99))` with `r_tr` = LOCO residual on `Fn` (train) | TRAIN-normal 99th pct |
| SOTA-side trivial threshold | `prep_sota_general.py:24` `thr = np.quantile(np.abs(zn).max(1), 0.99)` | TRAIN-normal 99th pct |

The Easy/Difficult/Double-hard subsets use test LABELS only to select which anomalies are in the
subset (`y==1 & maxz<=thr`), which is the definition of the subset, not a calibration. Shipped values
(`scores_<DS>.npz`): WADI `maxz_thr`=6.000, HAI 3.842, SWaT 5.187; n_easy/n_anom = 37/56, 485/652,
144/182.

## Per-method table

Key discrimination: AUROC is invariant to any strictly monotone (in particular affine) transform of a
SINGLE score array. So a method whose final score is one array, however it is scaled, cannot have its
AUROC changed by the choice of standardization statistics. The only way scaling statistics can move
AUROC is when two or more differently-scaled components are SUMMED or MAXED (the ensemble-leak
mechanism). The columns below make that explicit.

| Method | Test-window standardization source | Multi-component fusion? | Any test-label / test-anomaly use | AUROC affected by stat source? | Verdict |
|---|---|---|---|---|---|
| trivial max\|z\| | train-normal raw `mu, sd` (`eda_real.py:248-249`); score `= d["maxz"]` (`build_scores_table.py:47`, `build_comparison_table.py:13`) | No: max over channels of ONE train-scaled array | none (best-F1 oracle only) | No | clean |
| Isolation Forest | `Xtr, Xte` from train window stats (`build_scores_table.py:51-52, 61`); `-IsolationForest(...).fit(Xtr).score_samples(Xte)` (`:69`) | No: single score | none | No | clean |
| AutoEncoder | same `Xtr, Xte`; `ae_scores(Xtr, Xte)` trains on `Xtr` only, returns `((ae(zt)-zt)**2).sum(1)` (`compare_baselines.py:23-33`) | No: unweighted sum of squared residuals is one array (no per-term z-scoring) | none | No | clean |
| LinRes (one-hot) | `onehot_filter.build_feats`: continuous cols scaled by TRAIN window-mean `mu, sd` (`:41-43`); discrete states from `np.unique(Xn[:,c])` (`:33`); regressors `.fit(Fn, ...)` train only (`loco_residual :51-54`) | Sum of squared residuals over columns, all on the same train-fit scale; no test-scaled z-terms | none | No | clean |
| USAD | inputs `wadi/HAI/SWaT_test.npy` standardized on train (`prep_sota_general.py:20-21`); harness `load_dataset` loads the arrays as-is (`TranAD/main.py:26-43`, no normalize call in `main.py`); test score `loss = 0.1*l(ae1s,data) + 0.9*l(ae2ae1s,data)` per timestep (`main.py:179-180`), dumped BEFORE `pot_eval` (`modal_sota.py:113-118`); window map `ts[i:i+W].mean()` (`rev4_sota_aggregate.py:24-25, 50`) | Fixed-weight sum of two MSEs on the same scale, no z-scoring | `pot_eval` uses labels but only for the harness's own POT metrics, after the dump; `metrics()` in `modal_sota.py:158-169` uses `score[y==0]` for a 5%-FPR F1/TPR that the paper does not report (grep of IoT2.html: no "5% FPR"/"TPR@"/"F1@") | No | clean |
| TranAD | same inputs; test score `loss = l(z, elem)[0]` per timestep (`main.py:274-281`); same dump and window map | No: single MSE array | same as USAD | No | clean |
| base LatAD s0 (`anomaly_score_hard`) | encoder sees `Xte` (train-scaled, above) | YES: `score = (dens-dm)/ds + (diag_nll-nm)/ns` (`models_vade.py:286-290`) | none | Would be, IF the reference stats were test-derived | clean: `dm, ds, nm, ns = self._hd_ref` are set in `fit_latent_density(x)` from `self._hard_components(x)` on the TRAIN argument `Xtr` (`models_vade.py:187-188`; called with `Xtr` at `build_scores_table.py:66`) |
| LatAD s = s0 + s_resid (auto) | as above | YES, third z-term `(rs-rm)/rsd` (`models_vade.py:296-299`) | none | same class of risk | clean: `rm, rsd = self._rd_ref` set in `fit_resid_head(x)` on train residuals (`:228`), called with `Xtr` (`build_scores_table.py:67`); the auto gate `_resid_auto` is a train 80/20 held-out ratio (`:233-242`) |
| LatAD basin term (auto) | as above | YES, subtractive z-term (`models_vade.py:300-302`) | none | same class of risk | clean: `am, asd = self._basin_ref` from `fit_basin_head(Xtr)` (`:271-272`, `build_scores_table.py:67`); `_basin_lam` from train ambiguity ratio (`:268-270`) |
| LatAD `use_recon` term | n/a | would z-score against the TEST batch: `(r - r.mean())/(r.std()+1e-9)` (`models_vade.py:291-294`) | n/a | YES if enabled | NOT USED: default `use_recon=False`, and `build_scores_table.py:68` does not pass it. Flagged so nobody turns it on without moving the reference to train. |
| Community experts (inputs to the ensemble) | per-expert `m2, s2 = Ff.mean(0), Ff.std(0)` from the first 80% of train (`modal_experts.py:66-67`); expert z-score `cm, cs` from the held-out train slice `Xn[nfit:]` (`:78-80`) | per-expert z, train-calibrated | none | No | clean (the leak is downstream, in `ensemble_final.py:129`, not here) |

Reference for what the shipped LatAD row is: `build_scores_table.py:68` calls
`anomaly_score_hard(Xte, use_resid="auto", use_basin="auto")` with the default `use_near=True`, so the
shipped s0 is z_train(density) + z_train(nearest-mode NLL). The docstring at `models_vade.py:279-281`
says the reported model is "density-only base"; the code default and the call site are density+near.
Both terms are train-calibrated, so this is a documentation mismatch, not a leak, but the paper's
description of s0 should be checked against `use_near=True`.

Cross-check on the trivial row: since `trivial = maxz` is also the difficulty axis, every anomaly in
the Difficult subset has `maxz <= maxz_thr` by construction, so trivial's Difficult AUROC is depressed
by design. That is the intended control and the paper discloses it; not a fairness issue.

## Artifact (not a leak): the un-clipped second re-standardization

`build_scores_table.py:51-52, 61` re-standardizes the WINDOW features with train window-feature
stats and `+1e-8`, with no clip. Window features of channels that are constant in train (std, range,
trend of a constant channel are exactly 0) get divided by 1e-8, so any test movement in such a
channel becomes a value of 1e9-1e17. This is computed from TRAIN statistics, so it is not a test
leak, but it is a hidden "train-constant channel moved" detector that reaches exactly the three
methods that consume `Xte`: IF, AE and LatAD (`:68-70`). LinRes drops constant channels
(`onehot_filter.py:31-32`), USAD/TranAD see the raw-timestep arrays, trivial uses `Xte0`.

Measured (`scratchpad/restd_check.jsonl`, `restd_effect.jsonl`; "blown" = any train-std<1e-6
column with |Xte|>10 in the window):

| Dataset | train-std<1e-6 cols / total | max abs Xte | normal windows blown | anomaly windows blown | Difficult anomalies blown | blown-indicator AUROC (All / Difficult) |
|---|---|---|---|---|---|---|
| SWaT | 50 / 306 | 1.0e16 | 0.0% | 3.8% (all Easy) | 0 / 38 | 0.519 / 0.500 |
| HAI | 48 / 354 | 1.3e17 | 0.0% | 5.4% (all Easy) | 0 / 167 | 0.527 / 0.500 |
| WADI | 188 / 738 | 2.0e9 | 0.19% | 39.3% (91% of them Easy) | 2 / 19 | 0.695 / 0.552 |

Reading: on HAI and SWaT the artifact touches only anomalies that `maxz` already classes as Easy and
no normal window, so it cannot change any Difficult-subset AUROC and cannot inflate any method
relative to another there. On WADI it carries a small train-derived signal into the Difficult subset
(2 of 19 anomalies, 1 of ~519 normals), available equally to IF (the WADI competitor in
`ensemble_final.COMPET`), AE and LatAD. How much each of the three actually exploits a 1e9 feature is
model-specific (IF is roughly scale-free per split; an MSE autoencoder and the VaDE encoder are not),
so this is a residual "same input, different sensitivity" caveat on WADI-Difficult only, not a leak.
Recommended (out of scope for this audit, no change made): clip `Xte`/`Xtr` after line 61, or use a
std floor instead of `+1e-8`, then re-run `build_scores_table.py` for the three learned methods.

## Best-F1 oracle

`build_comparison_table.py:14-16` `bestf1`: thresholds are quantiles of the (subset) test score,
labels select the best one. `ensemble_final.py:68-72` and `modal_sota.py:164-165` do the same on a
0.80-0.999 grid. Labels enter only through the max over thresholds, which affects F1 and never AUROC,
and the paper discloses the oracle. Applied identically to every method.

## Bottom line

1. All seven baselines (trivial, IF, AE, LinRes, USAD, TranAD; GDN where present) are calibrated on
   train-normal statistics only and each produces a SINGLE score array (or a fixed-weight sum of
   same-scale residuals). Their AUROC cannot be moved by any standardization choice, and no path uses
   test-normal statistics or test labels beyond the disclosed best-F1 oracle. They are AUROC-fair.
2. The base/global LatAD score (s0 and the auto-gated s0 + s_resid - basin) is a genuine multi-term
   z-sum, so it is the same CLASS of object as the leaked ensemble, but every one of its reference
   pairs (`_hd_ref`, `_rd_ref`, `_basin_ref`) is computed on the training argument
   (`models_vade.py:188, 228, 272`) and the call sites pass `Xtr` (`build_scores_table.py:66-67`).
   It does NOT share the ensemble leak. The shipped `scores_<DS>.npz["LatAD"]` arrays are clean.
3. The only test-normal z-scoring in the pipeline is `ensemble_final.py:129` (out of scope, being
   fixed). The only latent hazard in LatAD's own code is the disabled `use_recon` branch
   (`models_vade.py:291-294`), which z-scores against the test batch and must not be enabled as-is.
4. After the ensemble fix, the comparison is apples-to-apples: same `Xte`, same window grid, same
   train-derived `maxz_thr`/`lin_thr`, same oracle-F1 protocol, every reference statistic from train.
