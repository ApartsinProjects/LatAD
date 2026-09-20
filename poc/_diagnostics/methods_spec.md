# LatAD — reproducibility spec (insert-ready, extracted from code)

All values below are read directly from the source; each carries its `file:line`.
Active datasets in revision2 are **WADI_clean, HAI, SWaT_canon**; the older
WADI/SWaT variants are noted only where the numbers differ. Anything genuinely
absent from code is tagged **[NOT IN CODE]**.

Global constants that recur:
- Window/stride: **W = 60, stride = 30**, identical for every dataset (`eda_real.py:273-276`, `RAW`).
- Per-channel window features: **6 stats** = [mean, std, min, max, last−first, max−min], concatenated channel-blocked (block b = stat b over all channels): `winfeat.py:15-18` (`feat_stats`); windowing in `eda_real.py:298-307`.
- Feature dimension = 6·C: **WADI_clean 122 ch → 732**, **HAI 59 → 354**, **SWaT_canon 51 → 306** (verified from `ens_bundle/bundle_*.npz`).
- Window label rule: window is anomalous iff >5% of its raw timesteps are labelled attack (`eda_real.py:304`, `int(y[i:i+W].mean() > 0.05)`).
- Seeds: **0,1,2,3,4** for all multi-seed detectors (`build_scores_table.py:20`, `SEEDS=[0,1,2,3,4]`).

---

## 1. VaDE hyperparameter table

Architecture (`models_vade.py:56-74`, `_mlp` + `VaDE.__init__`):
- Encoder MLP: `[C6, 128, 64]` with **ReLU between hidden layers only** (no activation on the final linear), then two heads `fc_mu`, `fc_logvar`: `Linear(64 → latent_dim)`. `C6` = 6·C = input feature dim.
- Decoder MLP: `[latent_dim, 64, 128, C6]`, ReLU between hidden layers, linear output.
- `hidden=(128,64)` default (`models_vade.py:66`); never overridden by the callers, so 128/64 for all datasets.
- Mixture prior parameters learned jointly: `pi_logit(K)`, `mu_c(K,latent)` init `randn*0.5`, `logvar_c(K,latent)` init 0 (`models_vade.py:76-78`).

| Item | WADI_clean | HAI | SWaT_canon | Source |
|---|---|---|---|---|
| K (mixture components) | 20 | 40 | 40 | `build_scores_table.py:19` `CFG` |
| latent dim | 10 | 16 | 16 | `build_scores_table.py:19` `CFG` |
| pretrain epochs (plain-VAE) | 30 | 30 | 30 | default `pretrain_epochs=30`, `models_vade.py:381`; callers do not override |
| joint epochs | 40 | 40 | 40 | `build_scores_table.py:88` `epochs=40` |
| beta (KL/cluster) warmup length | 8 | 8 | 8 | `build_scores_table.py:88` `warmup=8`; beta = min(1,(ep+1)/warmup) `models_vade.py:434` |
| batch size | 256 | 256 | 256 | `_loader(..., batch=256)` `models_vade.py:373` |
| optimiser | Adam | Adam | Adam | `models_vade.py:398,428` |
| learning rate | 1e-3 | 1e-3 | 1e-3 | default `lr=1e-3`, `models_vade.py:381` |
| mixture-param lr ratio | 0.1 | 0.1 | 0.1 | `mix_lr_scale=0.1` `models_vade.py:382,430` (mixture params get lr·0.1) |
| LR schedule | StepLR step 20, γ 0.5 | same | same | `models_vade.py:432` |
| variance floor (logvar_c) | log(0.05) | same | same | `models_vade.py:81` |
| cov-reg penalty weight | 1e-3 | 1e-3 | 1e-3 | `build_scores_table` uses default `cov_reg=1e-3` `models_vade.py:382`; penalty = cov_reg·Σexp(−logvar_c) `models_vade.py:132` |
| GMM init (mixture prior) | diag, reg_covar 1e-4, n_init 3 | same | same | `models_vade.py:416-417` |
| seeds | 0-4 | 0-4 | 0-4 | `build_scores_table.py:20` |

**K selection rule (state exactly this in the paper).** K is a **fixed per-dataset
constant** hard-coded in `CFG` (`build_scores_table.py:19`); it is **not** selected on any
split and never touched by test data. The BIC-optimal K\* (WADI 22 / HAI 24 / SWaT 25,
`eda_real.py:346-352` `main`, saved to `multimodality.json`) is computed **only for dataset
characterisation** (a diagnostic of regime multiplicity on train-normal windows) and is
**not** the K fed to the detector. So "no test tuning" is accurate: K=20/40/40 are author-set
constants, K\*=22/24/25 are a separate BIC description of the data. **Recommended wording:**
"K was fixed a priori per dataset (20/40/40); we separately report the BIC-optimal component
count on train-normal windows as a characterisation of regime multiplicity, and do not tune K."
**[Flag]** the code does not contain a script that *derives* 20/40/40 from anything; they are
manual constants. If the paper implies K was chosen by a rule, that rule is not in code —
state it as a fixed hyperparameter.

Per-community experts train a *different, smaller* VaDE — see §3.

---

## 2. Scoring-head specifications

The reported per-community/global detector uses `anomaly_score_hard` (`models_vade.py:276-303`):
`score = z(density) [+ z(nearest)] [+ z(residual, auto)] [− λ·z(basin, auto)]`, each term
z-normalised against a stored **train-normal** reference (`_hd_ref`, `_rd_ref`, `_basin_ref`).
The headline base is **density-only + nearest** with residual/basin **auto-gated**
(`build_scores_table.py:91`, `use_resid="auto", use_basin="auto"`; `use_near=True` default).

**(i) Density head** (`fit_latent_density`, `models_vade.py:177-189`):
- High-K **diagonal** GMM on the train-normal latent means, `covariance_type="diag"`, `reg_covar=1e-3`, `random_state=seed` (seed = model seed, not fixed 0).
- M (=`k_density`): default 80 (`models_vade.py:177`), but the global caller caps it:
  `kd = min(80, max(20, n_train//10))` (`build_scores_table.py:85`). Resulting M:
  **WADI_clean** n_train 2614 → min(80,261)=**80**; **HAI** 18359 → **80**; **SWaT_canon** 1616 → min(80,161)=**80**. (All three land at 80; the cap only bites for tiny sets.)
- Score = `−GMM.score_samples(z)` (NLL), z-standardised by stored train mean/std (`_hd_ref`, `models_vade.py:188`).

**(ii) Nearest-component head** (`models_vade.py:99-104,174`):
- `diag_nll = −max_k log N(z | mu_c_k, diag var_c_k)` over the K jointly-learned mixture
  components (the **closest** component, not the π-weighted mixture — a rare-but-valid mode is
  not penalised for being rare). Diagonal covariance, log-variance floored at log(0.05).

**(iii) Residual head** (`fit_resid_head`, `models_vade.py:214-243`; scoring `_resid_score` `:210-212`):
- Residual definition: **r = x − x̂**, x̂ = `decode(encode_mean(x))` (decoder of the posterior mean, `models_vade.py:206-208`).
- PCA reduce residual to **red_dim = 30** dims (`min(30, R.shape[1])`, `models_vade.py:224`).
- Per-regime **Ledoit-Wolf** precision: assign each window to its argmax responsibility mode; fit one LW covariance per mode with **≥ min_mode = 30** samples, else fall back to the **global** LW (`models_vade.py:227,237`).
- Score = responsibility-weighted whitened residual: `Σ_k γ_k · mahalanobis_k(Q)` (`_resid_score`, `:210-212`), z-standardised (`_rd_ref`).
- **Auto-gate** (`models_vade.py:233-242`): fit per-mode precision on the **first 80%** of train-normal windows (chronological, `nA = int(0.8*len(Q))`), score the **last 20%**; ratio = q95(last-20%)/q95(first-80%). Gate ON iff **ratio < 1.5** (`_resid_auto`). Measured gate ratios (from BUILD_REFERENCE, older sets): WADI 5.24 → off; HAI 1.17 → on; SWaT ~0.85 → on. **[Recompute for the clean/canon sets before quoting these exact ratios.]**

**(iv) Basin head** (`fit_basin_head`, `models_vade.py:262-273`; used only when `use_basin` and λ>0):
- Perturb z with Gaussian noise **R = 16** copies, σ = `frac·latent_std` with **frac = 0.5**; agreement = fraction of copies keeping the clean argmax mode (`_noise_agreement`, `:250-260`).
- λ = `base_lam · max(0, frac_ambiguous − deadzone)` with **base_lam = 2.5**, **deadzone = 0.15**, ambiguity level **amb_level = 0.5** (window "between modes" if max responsibility < 0.5). Crisp-mode data → frac_amb ≲ deadzone → **λ = 0** (no-op on all three datasets).

---

## 3. Per-community sizing rules and community construction

Both live in `sota_bundle/modal_experts.py` (Modal runner that trains the expert library).

**Community construction** (`modal_experts.py:44-64`):
- Data: train-normal **window means only** — the level stat block `Xn[:, :nch]` (first 6th of the feature vector), standardised per channel.
- **Near-constant channels dropped**: active = channels with train std > 1e-6 (`:47`, `active = where(sdc > 1e-6)`); communities are indexed back to original channel ids via `active`.
- Correlation type: **Pearson** — `np.corrcoef` on standardised window means (`:49`). **NaN handling**: `np.nan_to_num` on the correlation matrix (`:49`); diagonal zeroed.
- Distance: **1 − |ρ|**, symmetrised `(dist+dist.T)/2` (`:50`).
- Linkage: **HAC average linkage** on the condensed distance (`:51`, `linkage(squareform(dist), method="average")`).
- Communities = **every internal dendrogram subtree whose leaf count is in [3, 25]** (`MAXSZ=25`, `:33,58`), de-duplicated (`:56-63`). These are **nested** (a subtree and its parent can both qualify) — this is intentional, so S > C.
- Resulting community count S (verified from `experts_full/expert_*.npz`):
  **WADI_clean S = 44**, **HAI S = 28**, **SWaT_canon S = 24** (this is the 44/28/24 the brief asks about). Community sizes range 3-25 (medians ≈5-6). (Older sets: WADI 45, SWaT 26, SWaT_canon-experts(non-full) 25.)
- **Cohesion** per community = **mean pairwise |ρ|** within G = `sub.sum() / (|G|·(|G|−1))` (`:64`).

**Per-community VaDE sizing** (functions of |G|, `modal_experts.py:78-82`):
- K(|G|) = `min(20, max(6, |G|))`
- latent(|G|) = `min(8, max(3, |G|//2))`
- M(|G|) density-head k_density = `min(50, max(12, nfit//12))` (nfit = 80% of train-normal window count; depends on dataset, not |G|)
- epochs = 15, warmup = 4 (`:79`)
- Residual-PCA dim: default **30** (`fit_resid_head` default; `use_full=1` path only, `:82`). Per-community residual/basin heads are added **only in the `experts_full` build** (`use_full`, `:81-83`); the plain `experts` build is **density-only** per community. The paper's headline uses `experts_full` (`BUILD_REFERENCE.md:16`, `EXPERTS_DIR=sota_bundle/experts_full`).
- Each community also gets the auto-gate for residual/basin exactly as in §2 (auto). A **global VaDE** (full §1 config) is the unfactorized "null expert".

**Per-community calibration split**: `nfit = len(Xn)*4//5` (`:46`). First 80% train-normal = fit slice; **last contiguous 20% = calibration (`Cal`)** used as the p-value/survival reference (`:66-68`). Per-expert scores are z-standardised by the expert's own fit-slice mean/std then re-referenced to Cal (`:86-89`).
- Failed experts (non-finite scores) are zero-filled, not excluded (`:91-93`).

---

## 4. Aggregation operationalisation

All in `ensemble_final.py`.

**Per-community upper-tail p-value / survival** (`ensemble_final.py:35-47`):
- Reference set = the expert's **held-out 20% train-normal calibration slice** `Cal[sd,g]` (NOT test-normal; FIX 1, `:125-132`).
- `surv(ref,v)` = `−log(clip(1 − rank, floor, 1))`, rank = `searchsorted(sort(ref), v, 'right')/n`, **floor = 1/(n+1)** (the finest resolution the size-n reference supports; replaced a fixed 1e-4 that saturated the tail) (`:35-42`).
- `pval(ref,v)` = `clip(1 − searchsorted(...,'left')/n, 1e-4, 1)` (`:45-47`). Tie rule: `surv` uses side='right', `pval` uses side='left'.

**Higher Criticism** (`HC`, `ensemble_final.py:50-57`):
- Input P is (S communities, n windows). Optional cohesion weighting raises each community's p to the power `w_G/mean(w)` (`:52-53`).
- Sort p ascending → `Ps`; `i = (1..S)/S`; `HC_i = √S·(i − Ps)/√(Ps(1−Ps))` (`:54-55`).
- **Edge case p_(i) ≥ 1/2**: those entries set to −∞ (`hc[Ps>=0.5] = -inf`, `:56`) so only left-tail (significant) order statistics contribute; take **max over communities** (`:57`).
- **When no community has p < 1/2** (all −∞): `nan_to_num(neginf=0.0)` → HC score = **0** for that window (`:57`).
- Degenerate/failed communities are **not excluded** — their zero-filled surprise gives a mid p-value that simply does not enter the left tail.

**Cohesion weight**: `w_G = cohesion_G · √|G|` (`ensemble_final.py:123`, `w = coh*np.sqrt(size)`); the **√size exponent is 0.5**. `HC_coh` applies it via the `P**(w/mean(w))` reweighting (`:52-53`).

**Fusion terms**:
- `null+HC` (ensemble_final default headline, `HEAD` env `:30`): `max( z(HC), z(LatAD-null-tail) )` — a **max** of two z-standardised scores, **unit weights** (`:152`). z is against the calib-vs-calib reference for HC (`hc_c`, `:147`) and against the train-normal null tail for LatAD (`zl`, `:149-150`).
- `HCcoh+LatAD` (BUILD_REFERENCE / drift / PCA headline, `HEADKEY`): **z-sum** `z(HC_coh) + z(LatAD-null-tail)` (`:158`, and `rev4_doublehard_pca.py:34`, `drift_changepoint_typing.py:46`).
- **[Flag a naming inconsistency to resolve in the paper]**: `ensemble_final.py` defaults `HEAD="null+HC"` (a max-fusion), but `BUILD_REFERENCE.md:3` and the double-hard / drift scripts treat **`HCcoh+LatAD`** (a z-sum with cohesion weighting) as "LatAD (regime-community)". State which one is the reported headline; they are different combiners.
- **Null / whole-plant expert**: the global VaDE's test score `d["LatAD"][sd]`; its tail = `surv(LatAD_train[sd], LatAD[sd])`, z-standardised against `surv(LatAD_train, LatAD_train)` (train-normal, no test labels) (`:149-150`). This is the **z-standardization + unit-weight** fusion the brief refers to.

---

## 5. PCA double-hard second filter

`rev4_doublehard_pca.py` (+ `_all3.py`). Filter B flags a difficult window if T2 or SPE exceeds its 99th train-normal percentile.
- **Variance retained: 95%** (`VAR_KEEP = 0.95`, `:36`); sensitivity reported at **90/95/99%** (`VAR_SENS`, `:36`).
- Input space = the **standardised windowed feature matrix the detectors consume** (build_scores_table FIX-2 standardisation + eda_real ±10 clip on WADI/SWaT_canon; **no one-hot indicators**), re-centred on the train mean (`detector_inputs` `:93-102`, `pca_stats` `:105-118`).
- k retained at 95% (verified from `pca_filter_doublehard.json`): **WADI_clean k = 113 / 732**, **HAI k = 60 / 354**, **SWaT_canon k = 43 / 306**.
- Statistics (`:113-117`): **T2** (Hotelling) = `Σ_{i≤k} score_i² / λ_i`; **SPE / Q** = `‖Z_centered‖² − Σ_{i≤k} score_i²` (energy in dropped components), clamped ≥0.
- Thresholds = **99th percentile of the train-normal** T2 and SPE (`thr_t2`, `thr_spe`, `:152`). Verified values: WADI_clean T2=406.81 / SPE=125.27; HAI 173.02 / 45.64; SWaT_canon 535.03 / 85.33. Train flag rate ≈ 1-2% by construction.
- Double-hard = anomaly & maxz≤maxz_thr & T2≤thr & SPE≤thr (`:154`). Resulting sizes: WADI_clean 29 win/8 ep, HAI 55/13, SWaT_canon 58/18.

---

## 6. Drift / changepoint typing

`_diagnostics/drift_changepoint_typing.py`.
- **Rolling-median window: 24 h a priori** (`APRIORI = 24`, `:43`), converted to windows: `K24 = int(24·3600/step)` (`:152`). Step per window = stride·row period: **HAI 30 s** (1 Hz native, stride 30), **SWaT_canon / WADI_clean 300 s** (10× downsample → 10 s rows, stride 30) (`STEP_S`, `:42`). Sensitivity **6/12/24/48 h** reported, not selected (`HOURS`, `:43`); plus a 6 h-lagged variant (`GAP_H=6`).
- **slow_g(t)** = causal median of the community surprise over the previous K windows: `rolling(K, min_periods=max(3,K//4)).median().shift(1+gap)` (`causal_median`, `:53-56`); before enough history it uses `init` = the community's **train (calib) median** (`:56,102`).
- **fast_g(t)** = `s_g(t) − slow_g(t)` (`decompose`, `:59-72`). **Primary v2 (level+scale)** additionally divides by a slow scale: `fast = (x − slow) / max(rolling MAD, calib MAD)`, where rolling MAD = 1.4826·causal-median|x−slow| and **calib MAD** = 1.4826·median|C − median(C)| + 1e-6 (`:70-71,75-76`). The scale is **floored at the calibration MAD** so a quiet community is never made more sensitive than at train level.
- **Record start (no 24 h history)**: `causal_median` returns NaN until `min_periods = max(3, K//4)` past windows exist; those NaNs are replaced by `init` (train level), so `fast = x − train-level` there and the scale floor = calib MAD (`:56,70`). No look-ahead (`shift(1+gap)` guarantees strictly-past).
- **Changepoint score CP** = the headline `HCcoh+LatAD` fusion applied to the **fast** residuals: `z(HC_coh over fast p-values) + z(fast-detrended LatAD global tail)` (`scores`, `:109-117`). **Drift score DR** = the same HC_coh fusion on the **slow** baselines vs the raw calib reference (`:118-123`).
- **Thresholds** = 99th percentile of the **calib-vs-calib** score (no test labels): `thr_head`, `thr_cp`, `thr_drift` (`:255`).
- **Per-window type collapse (S communities → one label)**: the HC_coh fusion already reduces the S community p-values to **one scalar CP score per window**; a window flagged by the headline is typed **changepoint** if `CP > thr_cp`, else **drift** (`:17-18,258-261`). A window is additionally "drifted" if its slow baseline exceeds the per-community calib 99th percentile in **≥1 community** (`fd`, `:259`).
- **Gate**: identity check I1 — `K=None` (no decomposition) reproduces the headline exactly (`:157-161`); this is the "gate" that guarantees the typing layer never changes the headline detection.

---

## 7. Baseline settings table

| Baseline | Settings | Source |
|---|---|---|
| Isolation Forest | n_estimators **200**, `random_state=seed`; **default** max_samples ('auto' = min(256,n)), contamination 'auto'; score = `−score_samples` | `build_scores_table.py:95` |
| AutoEncoder | arch **C6-64-16-64-C6**, ReLU; Adam **lr 1e-3**; **40 epochs**; train loss = **mean** MSE; anomaly score = **sum** of squared residuals per window; per-seed | `compare_baselines.py:23-33` |
| LinRes (linear LOO) | **OLS** `LinearRegression` (no ridge), **leave-one-CHANNEL-out** (drops all features of the held-out channel); features: continuous → standardised window mean, discrete (≤ **6** distinct normal values) → per-state window fraction (soft one-hot), constant channel dropped; score = mean over channels of squared residual | `onehot_filter.py:17,29-57`; used at `build_scores_table.py:74-77` |
| Boosted LinRes | `HistGradientBoostingRegressor(max_iter=**150**, learning_rate=**0.1**, random_state=0)`, default depth (max_leaf_nodes 31), same leave-one-channel-out structure; single seed | `_diagnostics/clean_swat_boosted_loo.py:36-46` |
| USAD | imperial-qore/TranAD harness; **30 epochs** (minibatched, `SOTA_BS=128`), fp32, n_window per harness; 5 seeds | `modal_sota.py:271,275,94-96` |
| TranAD | same harness; **5 epochs** (harness default, already minibatched); 5 seeds | `modal_sota.py:271,275` |
| GDN | same harness; **5 epochs**, **n_window = 5**, train windows optionally strided (test dense, stride 1); single seed; **top-k / embedding dim = harness defaults** | `modal_sota.py:271,291`, `modal_gdn_fast.py`; **[top-k, embed-dim NOT IN OUR CODE — they are imperial-qore/TranAD `src/` constants for GDN; cite the harness repo or state "harness defaults"]** |
| trivial max\|z\| | `max` of \|standardised window mean\| over the first-6th (level) block | `eda_real.py:318-323`, `build_scores_table.py:54-56` |

- **Per-timestep → per-window aggregation** for SOTA: per-timestep score dumped by the harness, averaged over the window's timesteps: `mean` over each window (`build_scores_table.py:108-110`, `[ts[i:i+W].mean() for i in starts]`); if a per-timestep score is 2-D it is first averaged over the feature axis (`ts.mean(1)`).
- **HAI integer-ratio correction**: HAI is used at **native rate (no downsample)**; SOTA scores computed on a coarser grid are **integer-ratio upsampled to the LatAD window grid** before the per-window average (`BUILD_REFERENCE.md:82`). **[The upsampling code path is in the SOTA-aggregation script, not build_scores_table; verify the exact ratio logic in `rev4_sota_aggregate.py` before quoting.]**

---

## 8. Data spec

Preprocessing order (`eda_real.py:289-309`, `load`): raw channels → **standardise per channel on train-normal** (μ,σ from Xn, σ+1e-8) → **clip to ±10** (WADI/WADI_clean/SWaT_canon only; HAI/SWaT None) → **window (W=60, stride=30)** → **6 stats per channel** (`window_features`). A **second** per-window-feature standardisation + the same clip is applied where the detectors consume the matrix (`build_scores_table.py:66-84`, FIX-2).

| Dataset | Files / split | Downsample | Clip | Channels | Train-normal windows | Test windows | Test anomaly windows | Source |
|---|---|---|---|---|---|---|---|---|
| WADI_clean | itrust WADI.A2 `WADI_14days_new.csv` (train), `WADI_attackdataLABLE.csv` (test); drops the rescaled artifact channel `2B_AIT_002_PV` | 10× | ±10 | 122 | 2614 | 575 | 56 | `eda_real.py:37-63` |
| HAI (20.07) | `hai-20.07/train*.csv.gz` (all), `test*.csv.gz` (all); label = `attack`; sensor cols = all minus {time, attack, attack_P1/P2/P3} | native (none) | None | 59 | 18359 | 14819 | 652 | `eda_real.py:86-97` |
| SWaT_canon | official iTrust Dec-2015 `swat_normal_canonical.npz` (495k rows) as train, `swat_attack_canonical.npz` as test; 2% warmup drop; leak guard removes 1 coincidental row-collision, fail-fast `_assert_swat_no_leak` | 10× | ±10 | 51 | 1616 | 1498 | 233 | `eda_real.py:100-169` |

- WADI train = full 14-day normal recording; test = the attack recording (interleaved normals + attacks, label = −1 → anomaly) (`eda_real.py:40-47`).
- SWaT_canon **train/test-normal separation** is enforced by exact rounded row-hash set-difference; the earlier Kaggle-mirror leak (100% test-normal in train) is fixed (`eda_real.py:125-169`; commit `1224d49`).
- maxz_thr (99th pct train-normal max\|z\| over level block) verified: WADI_clean **6.008**, HAI **3.842**, SWaT_canon **3.163**.
- **[Older-set caveat]**: BUILD_REFERENCE Table 3/4 anomaly counts (WADI 56, HAI 652, SWaT 182) mix WADI_clean/HAI with the *old* SWaT; SWaT_canon has **233** anomaly windows. Use the canonical numbers.

---

## 9. Symbol-definition fixes (correct definitions to use)

- **Eq 6 Σ_k (residual whitening).** Correct definition: **Σ_k = Ledoit-Wolf-shrunk COVARIANCE of the PCA-reduced (30-dim) residual over the windows assigned to regime k** (argmax responsibility), with a global-covariance fallback for regimes with <30 windows. The whitened residual energy is the **Mahalanobis form r_kᵀ Σ_k⁻¹ r_k** (a precision, i.e. inverse-covariance, quadratic). Code: `sklearn.covariance.LedoitWolf().fit(...)` returns a **covariance** estimator and `.mahalanobis(r)` internally uses its **precision** (`models_vade.py:37-44, 227, 210-212`). **Resolve the precision-vs-covariance inconsistency by writing Σ_k as the covariance and the score as rᵀΣ_k⁻¹r** (do not call Σ_k a precision matrix). Residual is r = x − x̂, x̂ = decode(posterior mean).
- **Eq 3 index j.** j ranges over the **6·C features** (6 stats × C channels), not over channels: the feature vector is the concatenation of 6 per-channel stat blocks (`winfeat.py:15-18`). State "j = 1..6C".
- **Eq 4 mixture terms.** Name them explicitly: **w_m = π_m** (softmax of `pi_logit`, mixture weight), **a_m = μ_c,m** (component mean, `mu_c`), **b_m = diag Σ_c,m = exp(logvar_c,m)** floored at 0.05 (component diagonal variance) (`models_vade.py:76-81, 99-104, 121`).
- **s(x) reuse.** The symbol s(x) is used for both a per-expert score and the fused score. **Rename**: per-community/per-expert score `s_G(x)` (community G's z-normalised NLL) vs the fused detector score `S(x)` = HCcoh+LatAD (or null+HC). Do not reuse one glyph.
- **Fig 2 K → S.** The number of **communities is S** (44/28/24), distinct from the VaDE **mixture-component count K** (20/40/40). Relabel the community count in Fig 2 as **S** to avoid collision with the GMM K.

---

### Values flagged as NOT in code (infer, cite external, or ask)
1. K=20/40/40 and latent=10/16/16 are **manual constants** (`CFG`); no code derives them. If the paper claims a selection rule, it must be stated as "fixed a priori" — no rule exists in the repo.
2. GDN **top-k** and **embedding dim**: live in the imperial-qore/TranAD harness (`src/` constants), not in this repo. Cite the harness or say "GDN harness defaults, n_window=5, 5 epochs".
3. HAI SOTA integer-ratio **upsampling ratio**: referenced in BUILD_REFERENCE but the exact code is in `rev4_sota_aggregate.py` (not read here) — verify before quoting the ratio.
4. Residual auto-gate ratios (WADI 5.24 / HAI 1.17 / SWaT 0.85) are from BUILD_REFERENCE against the **older** datasets; recompute on WADI_clean/HAI/SWaT_canon before quoting.
