# Non-win hunt: bugs, measurement, and convert-to-win options for the LatAD paper

Scope: diagnosis and proposals only. Nothing in `paper/`, `models_vade.py`, the exported checkpoints or
the shipped score tables was modified. Every number below was recomputed in this session from the
stored artifacts (`_diagnostics/scores_<DS>.npz`, `sota_bundle/experts_full/expert_<DS>.npz`,
`_diagnostics/heads_<DS>.npz`) or from cheap local retrains, and every script persists its output
incrementally and is resumable. Paper numbers quoted are from `paper/IoT2.html` as extracted here
(Tables 2-6, A2, C1) and were reproduced to three decimals before anything else was run.

New artifacts (all under `poc/_diagnostics/`):

| script | output | what it does |
|---|---|---|
| `fable_leak_calib.py` | `fable_leak_calib_{WADI_clean,SWaT_canon,HAI}.json`, `fable_leak_trainscores_*.npz` | fusion-calibration leak: recomputes the headline under train-only calibration |
| `fable_wadi_nonwin_audit.py` (run with `EXPERTS_DIR=sota_bundle/experts_full`) | `fable_wadi_nonwin_audit.json` | WADI leak-window audit, per-episode decomposition, achievable-significance sweep, F1 audit |
| `fable_a8_positive_control.py` | `fable_a8_positive_control.jsonl`, `_ramp.jsonl` | positive control for the A8 responsibility-entropy screen |
| `fable_wadi_commsize_sweep.py` | `fable_wadi_commsize_sweep.{jsonl,json}` | community-size sweep with a label-free selection criterion |
| `fable_recon_gate.py` | `fable_recon_gate.jsonl` | would a train-only gate admit the plain whitened residual |

---

## 0. TOP PRIORITY: the fusion-calibration leak in `ensemble_final.py` (coordinator item)

### 0.1 What the code does

`ensemble_final.ensemble_scores()` lines 119-139:

```python
lat = d["LatAD"]; ...; nm = y == 0                      # y is the TEST label vector
...
nulltail = surv(lat[sd][nm], lat[sd])                   # LatAD null tail referenced on TEST normals
z = lambda s: (s - s[nm].mean()) / (s[nm].std() + 1e-9) # z-scale from TEST-normal statistics
acc["null+HC"].append(np.maximum(z(hc), zl))
acc["HCcoh+LatAD"].append(z(hc_coh) + zl)               # headline
```

Confirmed: `nm` is selected with the test labels, so the fusion scale of every fused aggregator
(headline `HCcoh+LatAD`, `null+HC`, `cohmax+LatAD`, ...) comes from test-normal statistics. AUROC is
rank-invariant to a monotone transform of ONE score, so `HC_coh` alone is clean (its difficult AUROC
is identical under every calibration below: 0.805 / 0.822 / 0.801). The leak enters only through the
RELATIVE scale of the two fused terms. It contradicts the paper's sentence "All model fitting and all
calibration ... use train-normal data only."

### 0.2 Quantification (`fable_leak_calib.py`)

Calibration variants, difficult-subset AUROC of the headline `HCcoh+LatAD` (5-seed mean +- sd):

| variant | reference for z(HC_coh) | reference for LatAD null tail | WADI | SWaT | HAI |
|---|---|---|---|---|---|
| `test_normal` (paper, current code) | test normals (label-selected) | test normals | **0.824** +-0.012 | **0.840** +-0.004 | **0.849** +-0.018 |
| `test_all` (label-free, transductive) | all test windows | all test windows | 0.819 | 0.838 | 0.852 |
| `train_cal` (fix, full model) | experts' held-out train slice (`calib_surprise`, last 20% of train) | global model's own train windows (in-sample; retrained, identical config) | **0.833** +-0.015 | **0.837** +-0.004 | **0.845** +-0.019 |
| `train_heldout80` (fix, construct-matched with the experts) | same | global model fit on first 80% of train, tail on last 20% | **0.846** +-0.012 | **0.841** +-0.007 | **0.830** +-0.010 |

Paired episode-block bootstrap vs the strongest baseline (2000 reps), Difficult subset:

| variant | WADI vs LinRes | SWaT vs LinRes | HAI vs AE |
|---|---|---|---|
| `test_normal` (paper) | +0.037 [-0.088, 0.181] P=0.32 | +0.058 [0.016, 0.105] P=0.0035 | +0.263 [0.152, 0.366] P=0 |
| `train_cal` | +0.046 [-0.085, 0.203] P=0.26 | +0.055 [0.012, 0.101] P=0.0035 | +0.260 [0.149, 0.364] P=0 |
| `train_heldout80` | +0.060 [-0.080, 0.224] P=0.22 | +0.058 [0.015, 0.107] P=0.0065 | +0.244 [0.131, 0.359] P=0 |

Double-hard: WADI 0.743 -> 0.754 (train_cal) / 0.772 (heldout80); SWaT 0.775 -> 0.773 / 0.776;
HAI 0.819 -> 0.814 / 0.788. All-subset: WADI 0.864 -> 0.871 / 0.881; SWaT 0.941 -> 0.938 / 0.938;
HAI 0.949 -> 0.948 / 0.944.

Invariants tested: I1 the `test_normal` variant reproduces 0.824 / 0.840 / 0.849 and the paper's
CIs/P-values exactly (PASS). I2 the retrained global model reproduces the stored per-seed LatAD
difficult AUROC (WADI and SWaT: identical to 4 decimals, rank correlation 1.000; HAI: within 0.008,
rank corr 0.988-0.994) (PASS). I3 `HC_coh` alone is invariant across variants (PASS).

### 0.3 Verdict

1. **The leak is real but it did not buy the headline.** Removing test-normal statistics from the
   fusion moves WADI UP (+0.009 to +0.022), SWaT by -0.003/+0.001, HAI by -0.004 (in-sample train
   reference) or -0.019 (fully held-out 80/20 model, which also weakens the global expert by training
   it on less data: LatAD global 0.811 -> 0.800). All three headline leads survive; the two significant
   results stay significant (SWaT P <= 0.0065, HAI P = 0); WADI stays non-significant.
2. **How much the scale matters at all**: under train calibration, sweeping the relative weight
   a in `a*z(HC_coh) + zl` over 0.25-4 spans 0.799-0.840 (WADI), 0.836-0.840 (SWaT), 0.825-0.851
   (HAI). The scale is a real degree of freedom, so it must be set label-free; the paper's current
   setting is inside that band and not at its top on any dataset.
3. **The max-fused `null+HC` variant is fragile under the fix** (HAI 0.846 -> 0.770 in-sample /
   0.751 held-out; WADI 0.806 -> 0.821 / 0.832; SWaT 0.823 -> 0.811 / 0.811). Cause: `surv()` clips
   the tail at 1e-4, so every test window above the reference maximum ties at -log(1e-4) = 9.21; with
   a train reference many HAI test normals (the train-empty regime) saturate. The sum-fused headline
   is robust to this; `null+HC` is not. Recommend dropping `null+HC` from Table 6's aggregation block
   or replacing the clipped survival by a smooth tail (see fix).

### 0.4 Minimal correct code change

Two files.

(a) `build_scores_table.py`: give the global null expert a held-out train-normal calibration slice,
construct-matched with the experts (which fit on the first 80% of train windows and calibrate on the
last 20%; `modal_experts.py` `nfit = len(Xn) * 4 // 5`). Fit the global VaDE on `Xtr[:nfit]`, store
`LatAD_cal = score(Xtr[nfit:])` next to `LatAD` (shape (n_seed, n_cal)). This is the
`train_heldout80` variant above. (The `train_cal` variant instead keeps the full-train model and stores
its in-sample train scores; it is cheaper and keeps HAI at 0.845, but an in-sample reference is a
weaker protocol and should be stated as such if used.)

(b) `ensemble_final.ensemble_scores()`: replace the four `nm`-based lines by train-only references:

```python
lat_cal = d["LatAD_cal"]                                   # (nseed, ncal) held-out train-normal scores
for sd in range(nseed):
    P     = np.stack([pval(Cal[sd, g], Tst[sd, g]) for g in range(S)])
    P_cal = np.stack([pval(Cal[sd, g], Cal[sd, g]) for g in range(S)])   # calib slice vs itself
    hc, hc_coh         = HC(P), HC(P, wt=w)
    hc_cal, hc_coh_cal = HC(P_cal), HC(P_cal, wt=w)
    tail, tail_ref = surv(lat_cal[sd], lat[sd]), surv(lat_cal[sd], lat_cal[sd])
    z = lambda s, ref: (s - ref.mean()) / (ref.std() + 1e-9)
    zl = z(tail, tail_ref)
    acc["HCcoh+LatAD"].append(z(hc_coh, hc_coh_cal) + zl)   # headline
    acc["null+HC"].append(np.maximum(z(hc, hc_cal), zl))
    ...
```

and in `surv()` replace the fixed `1e-4` clip by `1.0 / (len(ref) + 1)` (or a log-linear
extrapolation) so windows beyond the reference maximum keep their order. Then re-run
`ensemble_final.py` (EXPERTS_DIR=sota_bundle/experts_full) and re-derive Table 3/4/6 and the bootstrap
sentences. Agent execution: ~15 min (WADI/SWaT retrain 30 s/seed, HAI 90 s/seed, bootstraps ~5 min).

Paper consequence: with (a)+(b) the headline row becomes WADI 0.846 / SWaT 0.841 / HAI 0.830
(held-out) or 0.833 / 0.837 / 0.845 (in-sample reference). Either way the "train-normal only"
sentence becomes true and the conclusions do not change.

---

## A. WADI: the only dataset without a significant win

### A.1 Bug / measurement hunt

**Finding A-1 (measurement, real): 12 of the 43 "difficult" WADI windows are trivially detectable
constant-channel flips that the difficulty mask misfiles, and the second standardization inflates
them to |z| = 1e9.**

Code path. `eda_real.load("WADI_clean")` standardizes the raw stream on train, clips to +-10, then
windows into six blocks (mean, std, min, max, trend, range). The difficulty statistic
(`build_scores_table.py` line 47) is `maxz = |Xte0[:, :C6]|.max(1)`, i.e. the **window MEAN block
only**, on the **clipped** stream, thresholded at the train 99th (6.0). A channel that is constant in
train (30 such STATUS/CO/SP channels on WADI_clean: `1_MV_002_STATUS`, `1_MV_003_STATUS`,
`1_P_006_STATUS`, `2_MCV_007_CO`, `2_PIC_003_SP`, ...) has train std 1e-8, so any flip is +-1e8 sigma,
clipped to +-10; if the flip occupies k of the 60 rows, the window mean is 10k/60 and stays below 6.0
for k < 36 rows, so the window is filed as "difficult". The same flip on HAI or SWaT (CLIP = None)
gives maxz = 1e7-1e9 and is filed as "easy" (that is why `maxz` maxes at 1e7 on SWaT_canon and 1.3e9
on HAI, but at exactly 10.0 on WADI). The difficulty construction is therefore not the same rule
across datasets.

Then `build_scores_table.py` lines 51-52 re-standardize the window features with `Xtr0.std(0) + 1e-8`
and **no clip**: 191 train-constant features (30 channels x 6 blocks + 11) get |z| up to 2e9 in test
(27 features from the 5 channels above). AE, IF, the global LatAD and the recon head all see these; the
experts do not (`modal_experts.py` clips at +-10 after its own standardization); LinRes drops
train-constant channels entirely (`onehot_filter.build_feats`: `if k <= 1: continue`).

Invariants tested (all PASS, `fable_wadi_nonwin_audit.json["leak"]`):
- exactly 23 test windows carry |z| > 100 on a train-constant feature: 12 difficult, 10 easy, 1 normal
  (window 505, a `2_PIC_003_SP` setpoint change inside a normal stretch);
- a one-line trivial rule "max |z| over ALL six blocks > train-99th (17.9)" catches exactly those 12
  difficult windows and 1 normal window (the difficult set it leaves is identical to "difficult minus
  leak": every method's AUROC agrees to three decimals on both);
- on the leak-only subset (12 vs 519 normals) AE, global LatAD and `null+HC` score 0.994-0.999
  (numerically forced), the headline 0.992, LinRes 0.859, HC_coh 0.816; the 12 windows sit at
  normal-percentile 0.998-1.000 under global LatAD/AE for every seed (zero seed variance = the
  overflow symptom already noted in `fable_cleanwadi_report.md`).

What it does to the comparison (Difficult subset, 5-seed mean, full-head experts):

| subset | n | LinRes | AE | LatAD global | HC_coh alone | null+HC | **HCcoh+LatAD** |
|---|---|---|---|---|---|---|---|
| Difficult, paper (43) | 43 | 0.787 | 0.739 | 0.734 | 0.805 | 0.806 | **0.824** |
| Difficult minus the 12 leak windows | 31 | 0.758 | 0.638 | 0.631 | **0.801** | 0.733 | 0.758 |
| leak windows only | 12 | 0.859 | 0.999 | 0.999 | 0.816 | 0.994 | 0.992 |

Episode bootstrap on the 31 (9 episodes): headline - LinRes = 0.000 [-0.170, 0.170] P = 0.53;
HC_coh alone - LinRes = +0.043 (not bootstrapped separately; same 9 episodes); LatAD global - LinRes
= -0.127, AE - LinRes = -0.120.

Reading: the paper's +0.037 WADI margin over LinRes is carried by windows that a per-channel rule on
the std/min/max blocks catches and that LinRes cannot see because it drops constant channels. On
the 31 windows that are difficult under a consistent rule, the community factorization alone
(HC_coh 0.801) still leads LinRes by +0.043, but fusing it with the global null expert (0.631 there)
pulls the headline down to a tie. The per-episode table confirms it: of the three episodes where the
headline beats LinRes (ep 0: +0.40, ep 5: +0.19, ep 7: +0.58), two (5 and 7) consist entirely of leak
windows; leave-one-episode-out shows the sign of the overall margin depends on episode 0 alone
(drop it: -0.022).

Is this a "bug"? The flips are genuine attack signal and the paper is explicit that constant STATUS
channels are kept. The problems are (i) the difficulty rule is applied to a clipped stream on WADI
and an unclipped one elsewhere, so "difficult" does not mean the same thing across the three datasets,
and (ii) a 1e-8 floor without a clip turns a model comparison into an overflow comparison on those
windows. Both are one-line fixes: define `maxz` over all six blocks on the re-standardized features
(or, equivalently, use the pre-clip z as `prep_sota_general.py` already does for the SOTA bundle),
and clip after the second standardization as the experts do.

**Finding A-2 (statistics, not a bug): the CI is a sample-size limit, and significance is not
achievable by any plausible method change.** `fable_wadi_nonwin_audit.json["achievable"]`:

| shift of headline's difficult-window scores (in normal SDs) | headline Difficult AUROC | margin vs LinRes | P(diff <= 0) |
|---|---|---|---|
| 0 (as reported) | 0.824 | +0.037 | 0.32 |
| +0.25 | 0.854 | +0.068 | 0.21 |
| +0.50 | 0.884 | +0.097 | 0.098 |
| **+0.75** | **0.913** | **+0.126** | **0.018** |
| +1.00 | 0.935 | +0.148 | 0.0005 |

At 11 episodes with the observed between-episode heterogeneity (per-episode headline AUROC ranges
0.01 to 0.99), P < 0.05 needs a difficult AUROC of about **0.90-0.91**, i.e. a margin of roughly
+0.12 over LinRes, three times the current one. Conversely, holding the current +0.037 margin fixed
and resampling more episodes: 22 -> P 0.24, 44 -> 0.16, 88 -> 0.08, **176 -> 0.035**. WADI would need
about sixteen times its attack-episode count. No configuration knob on this dataset moves +0.08
(see D); the answer to the KEY QUESTION is: significance vs LinRes on WADI is not achievable at n = 11.

The bootstrap code itself (`rev4_stats.py` / `ensemble_final.boot`) was re-read: episode resampling
with replacement restricted to difficult windows, moving-block resampling of normals with block
L = ceil(W/stride)+1 = 3, seed-mean AUROC inside each replicate, paired difference. No error found; it
reproduces the paper's CIs exactly.

### A.2 Convert-to-win options (ranked by payoff x cheapness)

1. **Re-stratify consistently and report the 31-window difficult set** (cost: 10 min agent time,
   offline). Hypothesis: with the 12 flip windows moved to "easy" (where HAI/SWaT already put them),
   the WADI story becomes "factorization alone leads LinRes (+0.043), the global null expert is dead
   weight on WADI". Sanity check: the 12 must all land in Easy and every raw-feature method must
   score ~1.0 on them. Dead-end risk: the fused headline then ties LinRes (0.758 = 0.758), so the win
   would have to be claimed for `HC_coh`, which is not the headline aggregator on HAI/SWaT; and n
   drops to 31 / 9 episodes.
2. **Gate the global null expert out label-free on WADI.** Hypothesis: the null expert only helps where
   the global latent generalizes; use the same held-out-normal criterion as the residual gate
   (q95 held-out / q95 fit of the LatAD score). Observed ratios in this session for the global model:
   WADI 3.1-5.4 (does not generalize), SWaT 0.78-0.84, HAI 1.19-1.36. A gate at 1.5 would drop the
   null expert on WADI only, giving HC_coh 0.805 (43) / 0.801 (31) vs the fused 0.824 / 0.758.
   Sanity: the gate must stay ON for SWaT/HAI, where fusion adds +0.018 / +0.048. Dead-end risk: on
   the paper's 43-window set this LOWERS WADI (0.824 -> 0.805), because the null expert is what ranks
   the leak windows; so option 2 only pays after option 1.
3. **Community-size cap** (`fable_wadi_commsize_sweep.json`, 3 seeds, train-only criterion = excess
   false-alarm rate of HC_coh on the second half of the calibration slice at the first half's 99th):

   | MAXSZ | #communities | criterion (lower is better) | HC_coh 43 | HC_coh 31 | fused 43 | fused 31 |
   |---|---|---|---|---|---|---|
   | 12 | 39 | 0.251 | 0.786 | 0.778 | 0.831 | 0.769 |
   | 25 (paper) | 44 | 0.045 | 0.798 | 0.795 | 0.831 | 0.768 |
   | 40 | 45 | **0.039** | 0.803 | 0.803 | 0.836 | 0.775 |

   The label-free criterion picks 40, worth +0.005 (fused, train-calibrated). Not a lever.
4. Correct-and-unfixable framing (recommended regardless): "On WADI the regime-community model is
   numerically ahead of every baseline on the difficult subset; with 11 attack episodes the
   episode-block bootstrap has power to detect only a margin of about 0.12, so the WADI lead is
   reported as a point estimate with its interval." Do not describe the CI as "underpowered" in a way
   that implies more data would fix it unless the +0.037 is expected to hold; state the 0.91 number.

---

## B. WADI F1 (All 0.679 vs LinRes 0.680; Difficult 0.579 vs 0.595)

### B.1 Bug / measurement hunt (`fable_wadi_nonwin_audit.json["f1"]`)

- Threshold grid: the paper's 60-point 0.80-0.999 quantile grid vs an exhaustive sweep over every
  distinct score: LinRes 0.680 -> 0.687 (All), 0.595 -> 0.595 (Difficult); headline 0.679 -> 0.683,
  0.579 -> 0.582. Not a grid artifact (invariant: exhaustive >= grid, PASS everywhere).
- Construct mismatch: the learned detectors take the oracle per seed then average; LinRes is a single
  vector. Oracle on the seed-mean rank score instead: headline 0.686 (All) / 0.583 (Difficult). Same
  picture.
- Noise: episode-bootstrap of best-F1(headline) - best-F1(LinRes) on Difficult: -0.001 [-0.155, 0.175],
  P(<= 0) = 0.52. The F1 "loss" is a coin flip on 43 positives.
- Operating point (the informative part): F1 at a fixed 1% test-normal false-alarm threshold
  (no oracle): Difficult LinRes **0.515**, `null+HC` 0.542, headline **0.393**, HC_coh 0.413; All LinRes
  0.637, `null+HC` 0.667, headline 0.568. On the 31 non-leak windows: LinRes 0.520, headline 0.131,
  HC_coh 0.403. The headline's AUROC lead over LinRes on WADI lives in the mid-FPR range; at a 1%
  false-alarm budget LinRes separates more difficult windows. `rev4_stats.deployable` computes exactly
  this TPR@1%FPR and it is not in the paper.

Verdict: clean, and real. The F1 tie is not an oracle artifact; the fixed-budget numbers say the WADI
lead is not at the operating point a plant would use.

### B.2 Options

1. Frame F1 as the secondary within-subset metric it already is, and add one sentence that the
   difference is inside the bootstrap interval (-0.001 [-0.155, 0.175]).
2. If a deployable metric is wanted, report TPR@1%FPR / F1@normal-q99 from `rev4_stats.py` for all
   methods and accept that LinRes wins it on WADI (and check HAI/SWaT before adding the column: on HAI
   the exclusion analysis already gives headline TPR@1%FPR 0.712 vs global 0.625).
3. No method-side fix is credible for F1 on 43 positives; do not torture it.

---

## C. Head-level results vs the "demote reconstruction" thesis

### C.1 Bug / measurement hunt

Table 2 (WADI recon 0.816 vs latent NLL 0.718) and Table 6 (WADI recon 0.813, SWaT recon 0.835) are
produced by `table2_clean.py` / `rev4_ablation.py`: the plain global whitened residual
`0.5 r^T Sigma^-1 r` on the re-standardized 732/306-dim features, z-scored on train. On WADI that head
sees the same 1e9 blow-ups as AE/LatAD (leak windows score 0.999 under every head). Recomputed on the
31 non-leak windows (`heads_WADI_clean.npz`):

| head, WADI Difficult | 43 (paper) | 31 (no leak) |
|---|---|---|
| reconstruction residual (dropped) | 0.813 | **0.741** |
| latent density | 0.758 | 0.665 |
| nearest-component NLL | 0.719 | 0.610 |
| base (density + nearest) | 0.743 | 0.644 |
| base + resid(auto) = LatAD global | 0.743 | 0.644 |
| forced base + plain recon (z-sum) | 0.802 | 0.726 |

SWaT_canon has no leak windows (its constant-channel flips are already "easy"), so Table 6's SWaT
column stands: recon 0.835 > base+resid 0.807 > density 0.794 > base 0.775 > nearest 0.723; forced
base + plain recon 0.821.

So the head-level facts are: on WADI and SWaT the plain reconstruction residual alone beats the
single-latent LatAD score on the difficult subset by +0.10 (31-window WADI) and +0.03 (SWaT); on HAI it
is the weakest head (0.695 vs 0.802). The auto-gate is NOT the resolution on SWaT: the gated head is a
different, weaker residual (PCA-30, responsibility-weighted per-mode Ledoit-Wolf, `fit_resid_head`)
and the gate admits it (ratio 0.78-0.84) to reach 0.807, still 0.028 below the plain residual it
replaced. On WADI the gate is off (ratio 3.1-5.4 for the per-mode head), consistent with Table 6.

Gate experiment (`fable_recon_gate.jsonl`, model fit on the first 80% of train windows, held-out
20%, 2 seeds; the same q95(held-out)/q95(fit) ratio the existing gate uses, computed for the PLAIN
residual; test AUROCs are on the same 80% model and are for reporting only):

| dataset | ratio plain | ratio per-mode (existing gate) | plain admitted at 1.5? | base | plain alone | base + plain | base + per-mode (auto) |
|---|---|---|---|---|---|---|---|
| WADI_clean (43) | 6.0e11 / 6.1e11 | 2.31 / 2.17 (off) | **no** | 0.736 / 0.774 | 0.794 / 0.793 | 0.792 / 0.804 | 0.736 / 0.774 |
| WADI_clean (31 no-leak) | same | same | no | 0.635 / 0.687 | 0.714 / 0.713 | 0.712 / 0.729 | same as base |
| SWaT_canon (85) | 0.66 / 0.60 | 1.01 / 0.80 (on) | **yes** | 0.782 / 0.785 | **0.835 / 0.834** | 0.825 / 0.820 | 0.813 / 0.815 |
| HAI (167) | 4.61 / 4.41 | 1.50 / 1.26 (off / on) | **no** | 0.771 / 0.801 | 0.687 / 0.679 | 0.659 / 0.666 | 0.771 / 0.819 |

Invariant (stated in advance): the gate must reject the plain residual on HAI, where it is the
weakest head and fusing it hurts (base + plain 0.66 vs base 0.77-0.80). PASS. It also rejects it on
WADI, where the held-out train slice itself contains train-constant features that blow up under the
first-80% standardization (ratio 6e11: the residual does not generalize, which is the same overflow as
A-1 seen from the train side). It admits it on SWaT with margin (0.6 vs 1.5). So the existing
generalization gate IS a safe label-free selector for the plain residual; the paper's model simply
never offered it that head. Payoff on SWaT: base + plain 0.820-0.825 vs base + per-mode 0.813-0.815,
about +0.01 for the global model, and still 0.01 below the plain residual alone. Note the per-mode
gate's seed instability on HAI at 80% training data (1.50 vs 1.26 straddles the threshold).

The thesis as stated in the abstract ("demoting reconstruction from the default score") is supported
only by HAI at the single-latent level. What the data support everywhere is narrower: the WIN comes from
factorizing the latent density over communities (WADI 31-window: HC_coh 0.801 vs recon 0.741 vs base
0.644; SWaT: 0.840 vs 0.835 vs 0.775; HAI: 0.849 vs 0.695 vs 0.801). Reconstruction is demoted at the
level of the base score; it is not dominated as a head.

### C.2 Options

1. **Let the gate choose among {off, per-mode residual, plain residual} by the same held-out
   generalization ratio** (cost: the code exists; one re-run of `build_scores_table.py`, ~15 min).
   Measured (table above): the gate admits the plain residual on SWaT only (ratio 0.60-0.66) and
   rejects it on WADI (6e11) and HAI (4.4-4.6), seed-stable on all three. Payoff: SWaT global
   0.807 -> ~0.82; the community model's SWaT number (0.840) would move by at most a similar amount
   (not run; the experts would need the same three-way gate, ~20 min on Modal or ~1 h local).
   Dead-end risk: the fused score stays below the plain residual alone on SWaT (0.835), so this does
   not rescue the single-latent thesis on SWaT; it only tightens the gap.
2. Rewrite the mechanism paragraph: "the latent density is the base; reconstruction re-enters as an
   auto-gated head; on SWaT the plain residual is itself a strong head (0.835) and the community
   model's lead over it is +0.005". State Table 2's WADI 0.816 with the caveat that 0.741 is the value
   on the consistently-difficult windows, or drop Table 2's WADI column in favour of Table 6.
3. The docstring of `anomaly_score_hard` says "the reported model uses density-only base"; the code
   (`use_near=True` default) and the paper (base = density + nearest) both include the nearest head.
   Fix the docstring; no science changes.

---

## D. Components adding nothing on WADI (nearest head, cross-channel gain, representation swap)

### D.1 Bug / measurement hunt

- Nearest-component head lowering the base (WADI 0.758 -> 0.743; SWaT 0.794 -> 0.775; HAI 0.802 ->
  0.801) is real and consistent across seeds and also on the 31-window WADI set (0.665 -> 0.644).
  Nothing wrong with the measurement; the head costs 0.015-0.02 on two of three datasets and is kept
  for the A5 (rare-regime) rationale, not for AUROC.
- Cross-channel gain -0.004 (Table 5) and representation swap +0.002 (Table A2) are computed on the
  43-window set, where the 12 leak windows are at 0.999 under BOTH arms of each comparison (the
  marginal-product density and the temporal-feature model see the same 1e9 features). They therefore
  compress every WADI difference toward zero by 12/43 of the range. Not a bug in the arms, but the
  WADI cells of Tables 5 and A2 are measured on a subset that dilutes them. Recomputing both on the
  31-window set is a 20-minute agent job (`e5_gain_clean.py`, `repr_ablation.py` with a mask
  argument) and is the honest first step before claiming "WADI's difficult anomalies are
  single-channel or linear".
- The factorization gain itself is large once the leak is removed: HC_coh 0.801 vs global LatAD 0.631
  on the 31 (+0.17), vs +0.07 on the 43.

### D.2 Is there a WADI configuration that makes the gain significant, without test tuning?

No. Three train-only sweeps exist now: (i) global K x latent (`fable_cleanwadi_sweep_summary.json`:
9 configs, held-out NLL selects K=30, test range 0.730-0.751); (ii) community-size cap (C.1 table
above: label-free criterion selects MAXSZ=40, +0.005); (iii) density resolution is fixed by
`k_density = min(80, n/10)` = 80 and the experts' `min(50, nfit/12)` = 50, both at their caps. The
achievable-significance sweep (A-2) says +0.12 is needed; the combined reach of every knob is about
+0.02. Correct-and-unfixable at this n.

---

## E. A8 (between-regime overlap) "not observed" on 5 + 8 datasets

### E.1 Bug hunt: is the screen blind by construction? **YES (positive control fails).**

The A8 screen (Appendix C, Table C1; `per_community_rho.py`, `canbus_a8_screen.py`,
`cranfield_a8_screen.py`, `a3_screen_smd_pu.py`) measures mean max-responsibility, normalized
responsibility entropy H_norm and rho = frac(max resp < 0.5) from a trained VaDE, with a NO-overlap
synthetic floor and an empirical-variance refit as controls. There was no control in which overlap is
present by construction. `fable_a8_positive_control.py` builds one: 6 regimes in 24 channels, dwell
switching, and a fraction b of dwell segments placed at a random convex combination of two regime
centres (between-regime pockets), then runs the identical pipeline (`window_features("stats")`,
`train_vade`, responsibilities on train, refit).

| bridge mass b | sep | K | H_norm (VaDE) | rho | refit H_norm | refit rho |
|---|---|---|---|---|---|---|
| 0 | 4.0 | 8 / 20 | 0.004-0.017 | 0.000 | 0.002-0.009 | 0.000 |
| 0.2 | 4.0 | 8 / 20 | 0.007-0.023 | 0.001-0.002 | 0.005-0.015 | 0.000-0.002 |
| 0.3 | 2.5 | 8 / 20 | 0.038-0.057 | 0.002-0.007 | 0.024-0.038 | 0.001-0.005 |
| **0.5** | 2.5 | 8 / 20 | **0.040-0.086** | **0.005-0.012** | 0.025-0.063 | 0.002-0.012 |

Model-free check that the overlap really is in the window-feature space the screen sees (KMeans K=6 on
PCA-20 of the same standardized window features; between = d1/d2 > 0.67): b=0 -> 0.12 of windows, b=0.5
-> **0.52**. So half the mass sits between regimes, and the VaDE screen reports H_norm <= 0.09 and rho <=
0.012, against the paper's "rho >= 0.30 would indicate material overlap" threshold and its benchmark
band (H_norm 0.01-0.08). P0 (negative control reproduced) PASS; P1 (screen detects constructed overlap)
FAIL. A transit-only variant (linear ramps between centres, 9-50% of windows in transit) also gives
H_norm <= 0.06, but the model-free measure does not flag ramps either, so that variant is inconclusive
and is not relied on.

Mechanism: VaDE's objective (the KL of gamma to pi plus the clustering term) and the encoder's freedom
to warp between-regime windows onto a regime's latent cluster make crisp responsibilities a property
of the trained model, not of the data; a dwelt bridge simply becomes its own component (K >= 8 always
has spare components), and the empirical-variance refit inherits the same assignment. The SKAB
reversal in Section 7 ("variance-floor artifact") uses this same estimator and should be treated with
the same caution.

### E.2 Options

1. **Reframe now** (no experiment needed): "A8 is not exercised by the reported model; the
   responsibility-entropy screen does not detect overlap constructed at 50% mass in a positive control,
   so its negative result on the eight datasets is not evidence of absence." Drop "not observed" from
   Table 1 and the abstract-level claim; keep A8 as specified-and-untested. This is the honest call.
2. If an A8 statement is wanted, replace the screen by a model-free one that passes the positive
   control: KMeans/GMM at BIC-selected K in PCA space, between-fraction d1/d2 with the
   over-segmentation control (I0c in `fable_a3_measures.md` already shows chord measures inflate under
   over-clustering, so K must be selected label-free, not swept). Cost: ~1 h agent time across the
   eight datasets. Dead-end risk: the earlier pilot (`fable_a3_measures.md` 3a) already found between
   0.15-0.34 on WADI/HAI/SWaT under KMeans, which would then contradict "not observed"; the paper would
   have to say overlap exists at the 15-35% level and the basin head does nothing with it.

---

## F. GDN missing on canonical SWaT

### F.1 Fact check

- No canonical-SWaT GDN run exists anywhere: `grep` over every `_diagnostics/*.log`, `sota_bundle/results`,
  `sota_bundle/sota_scores`, `EXPERIMENT_LOG.md` finds GDN artifacts only for dirty WADI
  (`score_GDN_WADI.npy`, rc=124 after 9778 s, watchdog) and mirror SWaT (`rev4_sota_matrix.log` line
  6360: `DONE GDN_SWaT_s0 rc=0 2125.7s`, raw ALL 0.961, HARD 0.879; `rev4_sota_ms.json`: difficult
  0.871 vs USAD/TranAD 0.867 and LatAD global 0.96 on that split). `sota_swat_canon.log` contains only
  USAD and TranAD.
- The paper says (twice) that GDN's "per-window graph training did not complete on the canonical SWaT
  stream within our compute budget". I find no attempt to complete or fail. The mirror run finished in
  35 min; the canonical test stream (44,992 rows after downsampling) is smaller than the mirror test
  (~55k rows) with the same train, so a canonical run is a ~40-minute, ~$2 Modal job
  (`modal run sota_bundle/modal_sota.py --smoke "SWaT_canon:GDN"`, PFX already includes SWaT_canon).

### F.2 Options

1. **Run it** (cheapest honest fix; agent time ~10 min to launch and re-window, ~40 min external
   wall-clock). Invariant: the mirror-SWaT GDN difficult 0.871 must reproduce within seed noise if the
   same code is pointed at the mirror bundle first (5-minute smoke). Risk: on the mirror GDN trailed
   LatAD global by 0.09 while USAD/TranAD trailed by 0.09 too; USAD/TranAD then fell 0.21 on canonical
   and LatAD global fell 0.16. GDN's cross-channel graph could fall less and land near the community
   model's 0.840. That is a risk to the SWaT win, and the reason it must be run rather than described.
2. If not run, the sentence must change to "GDN is not benchmarked" (already flagged in
   `fable_final_qa.md`), because "did not complete" describes an attempt that the record does not show.

---

## G. Limits: WADI residual frontier and the HAI train-empty regime

- Residual frontier: it is episode 6 of the per-episode table (windows 360-362, three difficult windows;
  headline AUROC 0.01, LinRes 0.445; every method at normal-percentile 0.02-0.06 per
  `fable_cleanwadi_report.md` section 4, no channel shifted by more than 0.37 sigma) plus parts of
  episodes 0, 3 and 10 (headline 0.66-0.77). Correct and unfixable with snapshot features; the paper's
  framing is right. One addition: with the 12 flip windows re-stratified, the frontier is a larger
  share of a smaller difficult set (roughly 6-9 of 31), and the sentence "a minority of WADI difficult
  windows" should be quantified.
- HAI train-empty regime: the label-free exclusion (18.4% of test normals; headline 0.849 -> 0.890,
  `hai_exclusion_community.json`) is a secondary view and is stated as such. Two things to state
  plainly: (i) the effect is on every detector; (ii) under the construct-matched held-out
  calibration (Section 0, `train_heldout80`) HAI's headline is 0.830, i.e. the HAI number is sensitive
  to how much of the train stream the global expert sees, which is consistent with the coverage story
  and should be disclosed alongside it.

---

## Prioritized action list

1. **Fix the fusion calibration leak** (Section 0): store a held-out train-normal LatAD calibration
   slice in `build_scores_table.py`, reference all fusion z-scores and the null tail on train-only
   statistics in `ensemble_final.py`, smooth the `surv` clip, re-run, and re-derive Tables 3/4/6.
   Headline becomes 0.846 / 0.841 / 0.830 (held-out protocol); all conclusions survive; the
   "train-normal only" sentence becomes true. ~15 min agent time.
2. **Make the difficulty rule consistent across datasets and remove the 1e-8 overflow** (A-1): define
   maxz on all six blocks (or on pre-clip z), clip after the second standardization, re-stratify WADI
   (43 -> 31 difficult, 11 -> 9 episodes) and recompute Tables 3-6, 5 and A2 on it. Expect: WADI
   headline ties LinRes, HC_coh alone leads it by +0.04, global LatAD/AE lose to it; report WADI as
   "numerically ahead / tie, underpowered by construction (P < 0.05 needs AUROC ~0.91 at 11
   episodes)". ~30 min agent time. This is the single largest honesty item after item 1.
3. **Run GDN on canonical SWaT** (F): ~40 min external Modal wall-clock, ~$2; replace "did not
   complete" with a number either way.
4. **Reframe A8** (E): the screen fails its positive control; state A8 as specified-and-untested and
   drop "not observed" and the SKAB variance-floor reversal as evidence. No compute needed.
5. **Rewrite the reconstruction paragraph and, optionally, gate the plain residual** (C): the
   measured gate is seed-stable and rejects the plain residual on WADI/HAI while admitting it on SWaT
   (+0.01 for the global model). Adopting it is a small, safe gain; the larger item is the text: the
   single-latent "demote reconstruction" claim holds on HAI only, and the paper's actual win is the
   community factorization (which beats the plain residual on all three datasets). ~15 min agent time
   for the code path; also correct the `anomaly_score_hard` docstring.

Speculative (flagged as such): the claim that GDN "could land near 0.840" on canonical SWaT is an
extrapolation from the mirror split, not a measurement; the model-free between-fraction on real
WADI/HAI/SWaT quoted in E.2 comes from the earlier pilot, not from this session.
