# LatAD Round-2 Revision — Research Registry

Branch `revision2`. Started 2026-09-17. This is the durable record of every research
direction, decision, and result from the round-2 revision effort (responding to the
MDPI *IoT* round-1 reviews). It preserves negatives/exploratory paths too (wins-only
keeps those out of the paper but they stay here). Artifact `.json/.npz` are gitignored;
paths below let you re-open them. **Maintained live — update as jobs land.**

Active manuscript: `poc/paper/IoT2.html` (edits green `.rev2-highlight`). Plans:
`review_round_1/IoT2_review_response_plan.html`, `IoT2_editor_plan.html`.

---

## 1. Required experiments (editor E-items)

| ID | What | Verdict | Key result | In paper? | Artifacts / scripts |
|----|------|---------|------------|-----------|---------------------|
| **E4** | Time-aware significance test (A-22/23/24) | **WIN** | HAI difficult-subset lead survives autocorrelation-block bootstrap: diff **+0.054, 95% CI [0.012, 0.089], P(diff≤0)=0.005**. WADI (5 episodes) / SWaT (1 episode) non-significant, as already framed. Invariant L=1 reproduces iid. | not yet | `poc/rev4_stats_timeaware.py`, `poc/_diagnostics/rev4_timeaware.json` |
| **E5** | Gain ablation: source of the gain (A-25/B-04) | **WIN** | Cross-channel latent density vs channel-independent product-of-marginals, difficult-subset AUROC gain **WADI +0.343, HAI +0.206, SWaT +0.078** (5-seed). Isolates that the gain is cross-channel density, not the window representation. | not yet | Modal `poc/sota_bundle/modal_e5_gain.py`, `poc/sota_bundle/results/e5_gain_cuda.json` |
| **E2** | Nearest-component vs π-weighted mixture, rare-regime safety (B-05) | **NO WIN** | Parity on all 3 benchmarks (rare-regime FPR HAI 0.3552 vs 0.3516; WADI/SWaT 0). Fable adversarial audit: no bug; penalty real (4-8 nats) but never flips a flag at the operating threshold; forced imbalance to 0.01% still no win (VaDE won't allocate a low-π component to a very rare regime); synthetic miim_gen shows a small 1-2 pp effect only. Concrete: ~10 HAI valid windows saved (regimes 8/22), 0 the other way. → **design-note rebuttal only, no table.** | no (rebuttal) | `poc/e2_nearest.py`, `poc/_diagnostics/e2_nearest.json`, Fable scratch `poc/_diagnostics/e2_fable_*` |
| **E3** | Compute/cost table for edge deployment (B-03/A-30) | **DONE** | `train_only_s`: LatAD-global **3.2/16/3.5 s** (WADI/HAI/SWaT) vs TranAD 231/898/163 s (~47-72× cheaper), USAD 96/592/95 s, AE ~0. GPU latency b1: LatAD-global 0.7-13 ms, community 17-52 ms (CalexNet motivation), AE 0.15 ms. **CPU edge:** LatAD-global b1 0.6/14.6/14.5 ms, community 25/16/14.5 ms, AE 0.07 ms; ~3.5-3.9 GB RAM. Params: global 100-209k, community 0.78-1.25M. All cells clean (15 cuda + 9 cpu). | not yet | Modal `poc/sota_bundle/modal_e3_cost.py`, `poc/_diagnostics/e3_rows_{cuda,cpu}.jsonl` |
| **E1/A3** | Validate A3 basin head | **WIN — IN PAPER** | SKAB (real rotor testbed): gate fires (ρ=0.58, λ=1.07), difficult AUROC 0.500→0.605; provable exact no-op on WADI/HAI/SWaT. §7 witness + **Table 6** + ref [47]. | **yes** | `poc/a3_witness.py`, `poc/_diagnostics/a3_skab_witness.json` |

Other E-items already applied in IoT2.html (green): **E6** abstract/intro, **E7** related-work, **E8** Figures 1&2 (inline SVG), **E9** z-notation (`std`/`u`), **E10** equations numbered (1)-(9). Editor E11 (point-by-point response letter): **not started.**

---

## 2. Exploratory investigations (registry-only unless promoted)

| Direction | Verdict | Finding | Artifacts |
|-----------|---------|---------|-----------|
| **Rare-regime modeling fixes** (shrinkage / variance floor / per-regime threshold / kNN / merge) | No fix | Tiny regimes (6-14 samples) are **out of support** (99.8th pct kNN distance), a **data-coverage limit not a modeling one**. Every lever that cuts rare-regime FP also hides anomalies. HAI's big "rare FPR" is one **drifted regime (21)** + anomaly sink, not small-sample failure. | `poc/_diagnostics/e2_rare_*` |
| **kNN-in-latent scoring** (lead: SWaT 0.918→0.969) | No general win | Construct-matched 5-seed: helps **SWaT only** (+0.016, the easiest/least-discriminative set), **hurts WADI** (−0.025), neutral HAI. Full head best overall. Lead did not hold. | `poc/knn_latent_verify.py`, `poc/_diagnostics/e2_knn_latent.json` |
| **Data-quality audit** (mislabels, duplicate anomalies) | Reassuring | Mislabel/transition effect small (~0.1-0.4 FPR pts). Difficult subset is 89% overlap-adjacent → **effective N = attack episodes (5/26/1)**, exactly what the episode-block bootstrap uses → **validates E4**. No feature-space duplicate inflation. | `poc/_diagnostics/dq_audit.json` |
| **New-normal-regime discovery** (partition flags → unseen normal vs fault) | **Promising extension** | Works on HAI: the drift block recovered as one cluster (647-run, cohesive, far-from-train, correlation-consistent, **99% y=0**); folding it back cuts FPR **8.2%→3.2%** without masking attacks. **Key discriminator = physical correlation structure (PCA SPE)** (ties to A8). WADI/SWaT: none (correctly). One instance → future-work "regime-or-fault triage" stage. | `poc/_diagnostics/nr_discovery.json`, `nr_members.npz` |
| **Benchmark coverage & rarefaction** | PARTIAL (WADI done, HAI pending) | Reframes drift as measurable **training-set coverage** via non-circular measures (energy distance + perm p; occupancy KL/TV; C2ST-with-null; distance-OOS demoted to relative footnote), a **rarefaction curve** (#regimes vs data size), and a **with/without-exclusion decomposition**. **WADI finding (important):** the C2ST "test-only" mass is **62% of WADI test-normal**, an **instrumentation artifact** (sensor `2B_AIT_002_PV` train mean 9.1 → 4503 in the attack file, a rescale/break), NOT a regime. Excluding it moves **LatAD AUROC by 0.001 but lifts max\|z\|/AE/USAD/TranAD by +0.1 to +0.24** → LatAD is robust to the artifact, baselines are not; part of LatAD's WADI margin is artifact-robustness. **HAI** = genuine unseen operating regime (regime 21, 0.4% train / 6.3% test). SWaT covered. Also affected sensors `1_AIT_004_PV`, `2B_AIT_004_PV` (~2 SD). | `poc/_diagnostics/cov_coverage.json`, `cov_rarefaction.json` |

---

## 3. Decisions

- **Wins-only gate:** experiment results NOT wired into the paper until confirmed wins; E4/E5 confirmed, awaiting go-ahead to wire. E2 = rebuttal note only.
- **E3 decisions 1-3:** (1) run **both** A10G GPU + CPU; (2) AE = the paper's own `compare_baselines.ae_scores` (64-16-64, full-batch), aligned for construct-match; (3) SWaT `(K, latent)=(40,16)` verified authoritative from `build_scores_table.py`, `vade_SWaT.pt` checkpoint exported (`export_checkpoints.py`).
- **USAD:** first dropped (train=0.0s, latency shape error), then **restored** after Fable found the root cause: `_patch_harness` is **not idempotent** and the reused clone got double-patched into a `SyntaxError`, silently zeroing **every second deep cell**. Fix = **fresh git clone per deep cell** + module purge + Transformer forward shim. **Use `train_only_s` not `train_s`** (`train_s` inflates with import + POT scoring; TranAD "275s" real training is ~231s).
- **Early-exit / CalexNet [42]:** future work, **no experiment**; E3 supplies the per-community cost that motivates it.
- **Coverage OOS distance** flagged as **circular** (threshold-dependent) → switched to threshold-free energy distance + occupancy KL (headline); C2ST kept only with a permutation null.
- **Test-set exclusion analysis** (measure performance excluding the uncovered regime): allowed only with guardrails — label-free pre-registered exclusion, exclude only normals (0 anomalies removed, verified), report both with/without.
- **Train/test split** is **dataset-mandated** (WADI/HAI fully; SWaT anomaly side mandated, normal-test tail is a chosen 20% slice) → the HAI drift is inherent to the benchmark, not fixable by re-splitting.
- **Modal auth** from gitignored `modal_keys.txt` (env-extracted at run time, never echoed; the raw `modal token set` was declined per credential policy).

### Loader audit (eda_real.py) — triggered by the WADI artifact
- Alignment is **by name** on WADI/HAI/SWaT (`sens` = named columns present in both files), so no misalignment bug; the WADI 9→4503 and the HAI regime-21 drift are REAL, not loader errors.
- **WADI:** real documented artifact. Data-driven diagnostic (train vs test-normal per-channel shift): the ONLY genuinely broken channel is **`2B_AIT_002_PV`** (train mean 9.09 / std 0.16 → test-normal mean 4503; a real rescale). The "~30 zero-variance" channels (STATUS/actuator flags) are legitimately constant in normal and their flips are REAL attack signal → **kept** (dropping them, GDN-style, would lose signal for our pipeline). The two `AIT_004` analyzers are only ~2 SD drift → kept. **DONE:** added `_raw_wadi_clean` (drops `2B_AIT_002_PV` only) + `RAW["WADI_clean"]` + `build_scores_table` CFG; **re-running** WADI LatAD + IF/AE/linres (`scores_WADI_clean.npz`). Cleaning also corrects the **difficulty split** (`max|u|` was saturated by the artifact). USAD/TranAD-clean = a Modal follow-up if the margin materially changes. LatAD AUROC delta ~0.001; baselines +0.1..0.24 when the artifact is removed.
- **SWaT:** `ya=ones(len(Xa))` is **correct** — the accessible mirror's `attack.csv` is pre-filtered to attack-only. §5.1 mirror disclosure added (green, interim). **CANONICAL OBTAINED:** user had iTrust access; downloaded `SWaT_Dataset_Attack_v0.xlsx` (116 MB) from the iTrust SharePoint via the browser → `poc/datasets/_new/SWaT_canonical/`, cached to `swat_attack_canonical.npz` (**449,919 rows, 51 ch, 12.14% attack** = canonical). Scale-check: our existing SWaT normal is the SAME recording (channel names identical, median per-channel mean diff 0.2%), so **no Normal_v1 download needed**. Wired `_raw_swat_canonical` + `RAW["SWaT_canon"]` + CFG; **re-running** `scores_SWaT_canon.npz` (LatAD + baselines). On completion: compare canonical vs mirror, adopt canonical as primary, drop the §5.1 disclosure. USAD/TranAD-canonical = Modal follow-up.
- **HAI** correct (mandated split, consistent columns). **SKAB / PSM / MetroPT / TEP / BATADAL / SMD** auxiliary (not in paper), loaders standard.

---

## 4. Datasets

- **SKAB** — real rotor testbed, in registry; used as the A3 witness (in paper, Table 6).
- **Wind-SCADA (EDP Open Data)** — downloaded to `poc/datasets/_new/wind_scada/` (gitignored). 5 turbines, 2016-2017, ~521,800 10-min records, 83 channels + a 23-event failure log (generator/gearbox/transformer/hydraulic/bearing). For **A3/A9/A10 + incipient** future work.
- **MetroPT** — already in repo (`poc/datasets/_new/MetroPT3/metropt_data.npz`); real metro-compressor, slow air-leak degradation + operating modes → best real-data **A9/A10** candidate.
- **Paderborn PU bearing** — how-to documented (KAt datacenter, per-bearing `.rar`, real-damage codes KA04/KI04/…); **not downloaded**.
- Scout report: MetroPT / Paderborn / wind-SCADA are the top real, open picks for the assumptions WADI/HAI/SWaT cannot exercise; N-CMAPSS/TEP are simulated (complementary only).

---

## 5. Open threads / pending

- **Running:** coverage + rarefaction + exclusion study. (E3 GPU+CPU both complete.)
- **Paper:** wire E4/E5/E3 wins into IoT2.html (green) once approved; draft **E11 response letter**; remaining Group-A edits (A-23 WADI CI/P definition [E4 backs it], A-24, A-26, A-27 conclusion, A-29 raw-sequential rebuttal, A-07 English pass, A-20 indentation). Done: A-01/A-14/A-15/A-21 (+ E6-E10, A3).
- **Rebuild DOCX + render-QA** after edits land.
- **Possible new contribution:** a **"benchmark data-quality & coverage"** discussion section (energy-distance coverage + rarefaction + regime-or-fault triage + drift decomposition), honest and measured.
- **Optional bigger:** validate **A9/A10 on MetroPT / wind-SCADA** (real drift/trajectory data) to move them from future-work to demonstrated. Datasets on disk: MetroPT3 (best A9/A10 candidate), wind_scada (EDP). Paderborn not downloaded.
- **QUEUED (offline, report-only, HOLD until Modal `bnff19a89` + Fable clean-WADI both report):** clean-WADI adversarial re-analysis (Fable `a77b4ef09a7c89195`) — is LatAD's shortfall vs AE a coverage artifact (test-normals out-of-support: dirty-WADI audit had 19% OOS normals, 83% of LatAD false alarms on them) rather than real misses; HAI-style exclusion legitimacy check (must remove ~0 real anomalies).
- **QUEUED (offline, report-only, same hold):** **coverage-vs-anomaly triage** — second label from the latent structure splitting alarms into (a) suspected under-represented normal regime [novel latent region BUT low whitened-residual (A8) + basin-agreement holds (A3) + coherent/persistent cluster] vs (b) suspected fault/attack [high residual / basin-disagree / isolated]. Reuses existing heads; validate on HAI (regime-21 block = known new-normal, 0 anomalies, vs real attacks) → confusion matrix + "FPR-after-triage" as a method OUTPUT (automates the manual 0.934→0.950 exclusion). Risks: incipient-fault boundary case (also coherent+persistent; residual is the discriminator), circularity (must not relabel real anomalies as new-regime), batch/transductive not streaming. Do NOT wire into paper until confirmed win.

### Regime-imbalance single-knob experiment — NULL (flat-score trap), registry-only (`fable_sampler_audit.md`)
Single knob β: downsample each regime to (n_i/n_max)^β; retrain; measure rare-regime FPR vs β, imbalanced vs uniform-N control. Sampler audited + fixed (β=0 bitwise baseline, non-circular matched threshold on non-rare normals, frozen k_density/mask/std, AUDC + contrast + guards). **Key finding:** a naive reading shows a LatAD "win" (rare-FPR degrades slower than AE, contrast −0.32 on SKAB) but the guard exposes the GRACEFUL-FPR TRAP — LatAD's rare-vs-hard AUROC drops 0.62→0.45 and hard-TPR falls to 7-9%: the score goes FLAT (everything looks normal), not stable. No positive claim under wins-only. Also KREG=16 global tilt makes 15/16 regimes "rare" → measures dominant-regime collapse not targeted starvation. Conclusion: NULL; do not run full grid; do not wire to paper. [[e2-nearest]] theme: rare-regime robustness doesn't convert to a real win on these data.

### Per-community A3 ρ (subsystem-level A3 screen) — NEGATIVE, strengthens scope (`per_community_rho.json`)
Tested whether the correlation-community factorization (A8) hides A3 (thin between-regime pockets) at the subsystem level even though global ρ is low. Per-community VaDE (exact stored communities + expert-builder K/latent), ρ_G = frac(train max-resp<0.5): WADI(clean) 44 comm ρ_max 0.029; HAI 28 comm ρ_max 0.155; SWaT(canon) 25 comm ρ_max 0.122; **0 communities ≥0.30 on any benchmark** (vs SKAB global ρ≈0.58). Conclusion: A3 absent BOTH globally AND per-subsystem on all 3 benchmarks; A8 and A3 are independent; SKAB remains the sole A3 witness. Also confirms per-community complementarity: full-head HAI LatAD catches 67/167, strictly dominates deep (TranAD 0 unique) but only 8 LatAD-exclusive once AE included (49→8 correction). Rebuttal/scope evidence.

### 2026-09-17 FINAL construct-matched FULL-HEAD headline (supersedes the density-only numbers below)
All three headline numbers from ONE full-head config (experts_full/, whitened-residual+basin inside each community), matching HAI 0.849. HCcoh+LatAD difficult: **WADI_clean 0.824, SWaT_canon 0.840, HAI 0.849** (cohmax 0.828/0.837; null+HC 0.806/0.823). Real CIs (BOOT_REPS=2000). vs linres: SWaT SIGNIFICANT (+0.058 P=0.0025; DoubleHard +0.087 P=0.0005), WADI ahead +0.037 not sig (11 episodes), HAI leads. This is the FINAL Table-3 headline row for the corrected paper.

### 2026-09-17 HEADLINE regime-community model on clean data (density-only — SUPERSEDED by full-head above)
The paper's headline `LatAD (regime-community)` (= null+HC / HCcoh+LatAD / cohmax+LatAD), 5-seed, canonical difficult subset, construct-matched (bundles from clipped/canonical loader, verified label-for-label vs scores_*.npz):
- **WADI_clean: LEADS.** HCcoh+LatAD 0.816 / cohmax+LatAD 0.824 / null+HC 0.799 vs global-LatAD 0.734 (+0.065-0.090), linres 0.787, AE 0.739, TranAD 0.722, USAD 0.685, IF 0.681. Gain is real community signal (HC alone 0.783 > global). Ahead of linres but NOT significant (11 episodes, underpowered). DoubleHard: cohmax+LatAD 0.756 leads clearly (linres 0.660, AE 0.671).
- **SWaT_canon: SIGNIFICANT WIN.** HCcoh+LatAD 0.827 / cohmax+LatAD 0.828 vs global 0.804, linres 0.782, USAD 0.658, TranAD 0.655. Margin over linres significant: Difficult P=0.024, DoubleHard P=0.007.
- **Key lesson:** every OTHER session number used the ablated GLOBAL model, understating LatAD ~0.07-0.09. On the actual headline model WADI is a LEAD (not tie/loss) and SWaT a significant win. Global-LatAD 0.734 reproduced exactly → pipeline cross-validated.
- Artifacts: `sota_bundle/ens_bundle/bundle_{WADI_clean,SWaT_canon}.npz`, `experts/expert_{WADI_clean,SWaT_canon}.npz` (S=44/25, 5 seeds), USAD/TranAD added to `scores_SWaT_canon.npz`. Config-key edits only to ensemble_final/modal_experts/modal_sota (COMPET=linres verified strongest classical).

### 2026-09-17 corrections & findings (MAJOR — several supersede earlier entries)
Full-session Fable audit (`fable_session_audit.md`) + two clean-WADI Fable analyses + easy-filter prototype. Key:
- **BUG (fixed): `CLIP` missing `WADI_clean`** → `scores_WADI_clean.npz` was built UNCLIPPED; STATUS channels (train std ~1e-8) reached |z|=1e9, mis-filing 13 subtle anomalies as easy, blowing scores to 1e30+. Fixed: added `"WADI_clean":10.0` (+`"SWaT_canon":None`) to `eda_real.CLIP`; regenerated. Correct stratum now 43 subtle / 13 easy.
- **BUG (unfixed, rank-safe): second re-standardization in `build_scores_table.py` has no clip** → constant channels reinflate to feature-z ~1e9 at model input (global/null expert; community experts DO clip after their 2nd standardization). Cleanup TODO, does not change AUROC.
- **BUG (mine): inverted difficult subset.** My re-scores + `adaptive_density.py` used `maxz>thr` (the EASY subset). Canonical = `(y==1)&(maxz<=thr)` (subtle), scored vs all normals. All affected numbers recomputed.
- **CORRECTED cleaned-WADI difficult (clipped, canonical, 43 subtle vs 519 normals, construct-matched):** linres 0.787, l2 0.751, AE 0.739, **LatAD 0.734**, TranAD 0.722, USAD 0.685, IF 0.681. LatAD **ties AE** (block-bootstrap over 14 episodes), **beats deep baselines**, trails linres/l2.
- **linres WADI lead is a SCALING ARTIFACT:** `build_feats` leaves one-hot fractions unscaled while numeric is standardized. Standardize all features → linres 0.787→**0.684 (below LatAD 0.734)**; numeric-only 0.735. HAI/SWaT unaffected. So "linres beats LatAD on WADI" is not robust; under principled scaling LatAD ties/leads. (`linres_confound_and_ci.py`)
- **Extended difficulty stratification (PCA-per-axis T3 + discrete-state T2 easy-filters, circularity-guarded: compare vs IF/deep only):** on the joint-nonlinear survivors, **HAI = SIGNIFICANT LatAD win** — LatAD 0.585 vs best-deep AE 0.445, Δ+0.136, CI[+0.055,+0.199], p=0.002, 10 episodes. SWaT directional (0.888 vs 0.828) but only 1 episode (weak). WADI tie (0.609 vs 0.610). Strongest pro-LatAD result of the session. (`pca_discrete_easyfilter.py`, `linres_confound_and_ci.py`)
- **HAI detection-set complementarity** (equal budget): LatAD catches 65/167 difficult anomalies, **49 caught by LatAD ALONE**; LatAD catches 63 that TranAD misses, TranAD catches 0 LatAD misses (near-disjoint, Jaccard 0.03). Needs re-verify on fixed tables + headline model.
- **A8 gap:** paper claims "typed/discrete channels" but NO channel typing exists in LatAD (valve states Gaussian-modeled); only linres types channels. Narrow A8 wording. New validated cites ready: HI-VAE (Nazábal 2020), VAEM (Ma 2020), TABOR (Lin 2018) — `discrete_method_citations.md`.
- **Triage pilot FAILED** (registry): can't separate rare-valid-normal from incipient-attack (hides 199 anomaly windows). `fable_triage_pilot.md`.
- **Backbone verdict:** no swap / no in-place VaDE fix / no head change (`fable_backbone_alternatives.md`). Caveat: Fable clean-WADI found VaDE DOES allocate low-π components (π to 0.005), so "VaDE collapses rare regimes" is too strong; E2 still null (both scores 0 FPR there).
- **⚠ ALL session numbers use the ABLATED `LatAD (global density)` model.** Headline `LatAD (regime-community)` (WADI difficult 0.796 vs global 0.690) NOT yet run on clean data — Modal agent building experts now. This is the decisive open number.
- Canonical SWaT (audit, correct subset): **LatAD LEADS** 0.804 vs linres 0.782, AE 0.729, IF 0.627. USAD/TranAD on SWaT_canon pending (agent).

### Locally-adaptive density head (LOF) — CORRECTED (canonical rerun `adaptive_density_canonical.log`)
**The original "LOF hurts everywhere" verdict was a SUBSET BUG (inverted easy mask). Corrected canonical rerun (5-seed, difficult subset): LOF vs gmm(current head) — SKAB 0.568 vs 0.475 (+0.093, helps, sig vs fixed p=0); WADI_clean 0.753 vs 0.743 (+0.010 tie); HAI 0.797 vs 0.789 (+0.008 tie); SWaT_canon 0.740 vs 0.795 (−0.055 HURTS). Coherent story: LOF helps where rare regimes exist (SKAB), neutral on WADI/HAI, hurts on well-covered SWaT. Not a universal win → registry-only, but NOT "hurts everywhere." Design invariant still passes (LOF treats diffuse-valid regimes fairly).**
--- (original superseded numbers below, inverted-subset, DO NOT USE) ---
Tested swapping the fixed high-K GMM density head for a locally-adaptive local-density-ratio (LOF) in latent space, construct-matched, 5 seeds. Synthetic control PASSED the invariant (LOF penalty ratio on a diffuse-but-valid mode = 1.0 vs fixed-KDE 1.33 → LOF treats rare-valid regimes fairly). But on real data it does NOT convert to better hard-anomaly detection. **all 4 datasets** (5 seeds): difficult-AUROC lof < gmm everywhere — WADI 0.822<0.859 (ns), HAI 0.960<0.972 (CI [−0.016,−0.009] sig), SWaT 0.989<1.000 (CI [−0.016,−0.007] sig), SKAB 0.915<0.947 (CI [−0.053,−0.014] sig). LOF DOES reduce sparse-normal FPR where designed (WADI 0.11→0.019, SWaT 0.108→0.088; HAI/SKAB unchanged) but pays with hard-anomaly AUROC. **Mechanism:** hardest anomalies are globally-far points; LOF's local normalization removes exactly the "far from everything" signal → dilutes extreme anomalies. Conclusion: the fixed high-K GMM density head is the correct choice; adaptive-density is a no-win and REINFORCES the current design. (`poc/_diagnostics/adaptive_density.py`, `adaptive_density_SKAB.json`, `adaptive_density.json`.) [[e2-nearest]] shares the theme: rare-regime-safety mechanisms don't help on these benchmarks because the hard anomalies aren't rare-regime normals.

### E2 on SKAB (negative, registry-only)
E2 nearest-component run on SKAB (`e2_nearest.py SKAB`, K=16 LD=6): **NOT a win.** nearest vs mixture rare-regime FPR identical (0.417 both; only 12 rare-normal test windows, unreliable); difficult AUROC near-chance (nearest 0.495 / mixture 0.449). Reason: SKAB witnesses **A3** (thin envelopes), not **A6** (rare low-π regimes) which E2 targets — different assumptions, so SKAB adds no second empirical leg to E2. E2 stays design-note/rebuttal only; SKAB stays the A3 witness (Table 6), NOT a 4th benchmark (it is a deliberate negative control for the core detector: 8-dim, factorization has nothing to exploit → promoting it would violate wins-only).

---

## 6. Headline numbers (for quick reference)

- E4: HAI difficult diff +0.054, CI [0.012, 0.089], P=0.005 (time-aware).
- E5: cross- vs channel-independent density, difficult AUROC +0.343 / +0.206 / +0.078 (WADI/HAI/SWaT).
- E3: LatAD-global train 3.2/16/3.5 s vs TranAD 231/898/163 s; latency 0.7-13 ms (global), 17-52 ms (community, 45/28/26 communities).
- A3/SKAB: difficult AUROC 0.500→0.605, gate ρ=0.58 λ=1.07 (SKAB) vs ρ≈0.05 λ=0 (WADI/HAI/SWaT).
- HAI drift block: test-normal 6436..7081 (646 windows), regime 21 = 0.4% train / 6.3% test; excluding it FPR 8.2%→3.2%.
- Data-quality: effective N = episodes 5/26/1 (validates E4).

---

## 7. Capstone: MIIM validation matrix + benchmark coverage (both DONE)

### 7a. MIIM assumption × dataset matrix (`poc/_diagnostics/miim_matrix.json`)
Verdicts (HOLDS / PARTIAL / ABSENT / NOT-EXERCISED) with the measurement per cell:

| A# | measure | WADI | HAI | SWaT | SKAB |
|----|---------|------|-----|------|------|
| A1 Regime mixture | BIC K*, participation ratio | HOLDS (K*64, PR26) | HOLDS (K*128, PR15) | HOLDS (K*64, PR14) | HOLDS (K*8, PR5.2) |
| A2 Regime explosion | BIC saturation vs K | PARTIAL | HOLDS (unsat@128) | PARTIAL | PARTIAL (n-limited) |
| A3 Hard envelopes | ambiguous frac ρ | ABSENT (.013) | ABSENT (.023) | ABSENT (.000) | **HOLDS (.58)** |
| A4 Thin fringes | high-K density gain | HOLDS (+78 nats) | HOLDS (+24) | HOLDS (+36) | HOLDS (+5.3) |
| A5 Few levers | PCA PR / raw dim | HOLDS (4.8%) | HOLDS (4.8%) | HOLDS (5.5%) | PARTIAL (11%) |
| A6 Heavy tail | Zipf slope, Gini | PARTIAL (−.74) | PARTIAL (−.99) | HOLDS (−1.36) | HOLDS (−2.40) |
| A7 Hidden regimes | silhouette | HOLDS (.06/.09) | HOLDS (.08/.07) | PARTIAL (.27/.53) | PARTIAL (.27; 53% between-mode) |
| A8 Mixed signals | #communities, Q | HOLDS (45, Q.42) | HOLDS (28, Q.44) | HOLDS (26, Q.59) | PARTIAL (1, Q.06) |
| A9 Many clocks | snapshot-separable % | NOT-EXERCISED (66%) | NOT-EXERCISED (74%) | NOT-EXERCISED (79%) | **HOLDS (27%)** |
| A10 Path dependence | history-dep faults | NOT-EXERCISED | NOT-EXERCISED | NOT-EXERCISED | PARTIAL (50% between-mode) |

Reading: WADI/HAI/SWaT satisfy the instantaneous-structure assumptions (A1,A4,A5,A7,A8), lack between-regime (A3) and trajectory (A9/A10) structure; **SKAB is the one set exercising A3 + partially A9/A10**. Confirms the paper's scope (A1-A8 realized/validated; A3 on SKAB; A9/A10 future). A8 community counts reproduce E3 exactly (45/28/26). Caveat: A10 has no direct history-fault labels (inferred from A9 + SKAB between-mode).

### 7b. Benchmark coverage & rarefaction (`cov_coverage.json`, `cov_rarefaction.json`)
Model-free, block-permutation nulls calibrated against each train set's own temporal variability (so a p-value alone ≠ coverage gap; the **ratio to within-train variability** is the honest statement):
- **Energy distance test-vs-train ÷ max within-train**: SWaT 0.35 (covered), HAI 0.54 (covered as a whole distribution), **WADI 2.34 (outside) → 0.71 once `2B_AIT_002_PV` removed** (artifact, not operation).
- **Occupancy**: WADI 55% of test-normal in a train-empty mode → 0.6% after artifact removal (instrumentation); **HAI one train-empty mode at every K (regime 21: 0.4% train / 6.3% test), the 646-window block [6436..7081]**; SWaT none.
- **Rarefaction**: SWaT plateaus (covered); HAI coarse near-plateau but the **time-prefix** curve adds the block regime only after the test period begins — it first appears ~6170 normal windows past the training end (would need ~34% more training *duration* to see once) → HAI's limit is recording **duration**, not sample count.
- **With/without-exclusion (HAI, 701-window block, 0 anomalies removed)**: LatAD AUROC 0.934→0.950, difficult 0.814→0.841, FPR@1% 0.0100→0.0033, TPR@5% 0.65→0.89; **93% of LatAD's 5%-FPR false alarms are inside the block** (AE 91%, IF 59%, USAD/TranAD 0% — coverage-attributable share is detector-specific). SWaT exclusion empty (unchanged). WADI "block" = the artifact; LatAD robust to it, raw-reconstruction detectors not.
- **Verdict: HAI = genuine unseen operating regime (duration-limited); WADI = instrumentation artifact; SWaT = covered.** C2ST demoted (it certifies non-stationarity, separating any two time periods, so not a clean coverage measure).
</content>

### 2026-09-17 A3 WITHDRAWN — SKAB witness is a variance-floor artifact (DECISIVE, `fable_a3_spaces.md`)
Full investigation (H1 merging + H2 latent-space + floor sensitivity, synthetic-controlled). Verdict: **A3 (between-regime overlap) is NOT demonstrated on any dataset.**
- H1 (cluster merging) REJECTED: benchmark entropy flat/falling across K=8-64 (WADI 0.067→0.031); HAI regime-21 not merged. H2 (coarse latent) REJECTED: A3 absent from raw-738d → PCA-{5,20,50} → VaDE-LD{6,16,32}; synthetic control detects real 2.5σ overlap in those same spaces.
- **SKAB witness = artifact:** ρ=0.58 AND the basin lift +0.05 are the `logvar_floor=log(0.05)` guard on n=400. Floor→0.01: ρ 0.58→0.00, lift +0.072→+0.007 (one seed neg); empirical variances: ρ≤0.005, lift mean +0.010. OFF base at chance every floor. Invariant met (default floor reproduces ρ=0.58/+0.105). SKAB kNN-ambiguity 0.07 = as crisp as benchmarks.
- **8 real datasets screened, A3 absent in all:** WADI/HAI/SWaT/MetroPT/WindSCADA/SMD(×4)/Paderborn (entropy ≤0.06) vs SKAB's artifactual 0.27. New loaders added: WindSCADA (`_raw_wind_scada`, T06), Paderborn (`_raw_paderborn`, not in RAW). MetroPT/SMD data fetched.
- **DECISION:** basin head is auto-off (λ=0) on all benchmarks → removing it changes NO headline. **Renumber MIIM** (old A4-A8→A3-A7, old A3→A8; A9/A10 unchanged) so A1-A7=demonstrated/validated, A8-A10=specified-not-observed. Remove basin head from paper (code untouched), drop Table 6, graded A1-A10 framing, rewrite §4.3(iv)/§7/Fig2/Table1, fix letter R1-1. Executing (agent). Core method (regime-community factorization, latent density, HC) + all wins untouched.
- E5 source-of-gain also corrected on clean data: WADI +0.343→−0.004 (artifact), SWaT +0.141, HAI +0.200 → cross-channel-density gain is HAI/SWaT only.
