# All-wins brainstorm: routes to a significant WADI win, and what the model can shed

Read-only adversarial pass, 2026-09-18. Nothing in the paper, the model, or the expert library was
touched. Every number below is either read from an existing artifact (named) or produced by a scratch
check on cached scores/experts (scripts in the session scratchpad; the relevant outputs are quoted
inline). Scratch checks are labelled PILOT and are test-peeking by construction; none of them is a
result, they bound what a real experiment could return.

Standing (full-head regime-community model, difficult subset, 5 seeds, `headline_full_*.json`,
`ensemble_full.json`): WADI_clean 0.824 vs linres 0.787 (P=0.32, 11 episodes); SWaT_canon 0.840 vs
linres 0.782 (P=0.0025); HAI 0.849 vs AE 0.757 (CI [0.047, 0.157]).

---

## Part 1. Routes to a clean significant WADI win

### 1.0 Verdict first: the WADI tie is correct, and no detector change can turn it into a significant
### win under the paper's own test. Do not torture it.

Three independent pieces of evidence, all from cached scores (`chk_wadi_episodes.py`):

**(a) Per-episode picture.** Seed-mean percentile against the 519 test normals on the 11 difficult
episodes (headline HCcoh+LatAD / linres):

| ep | n | headline | linres | who wins |
|---|---|---|---|---|
| 0 | 6 | 0.68 | 0.26 | headline |
| 1 | 4 | 0.95 | 1.00 | tie |
| 2 | 3 | 0.98 | 0.98 | tie |
| 3 | 4 | 0.79 | 0.90 | linres |
| 4 | 2 | 0.99 | 0.99 | tie |
| 5 | 2 | 0.99 | 0.80 | headline |
| 6 | 3 | 0.01 | 0.45 | linres (both fail) |
| 7 | 2 | 0.99 | 0.41 | headline |
| 8 | 9 | 0.99 | 0.99 | tie |
| 9 | 4 | 0.99 | 1.00 | tie |
| 10 | 4 | 0.73 | 0.77 | tie/linres |

3 wins, 3 losses, 5 ties. The methods alternate episode by episode (Spearman between them on difficult
windows 0.60, on normals 0.25). Episode 6 (segment 7 of `fable_cleanwadi_report.md`, no channel moves
more than 0.37 sigma against context) is undetectable by anything; episode 0 tops out at 0.68 for every
method. That alternation IS the bootstrap variance.

**(b) The oracle bound.** Fusing the headline with linres itself in test-normal z units (the most
favourable thing any "add a linear residual" route could achieve) gives difficult AUROC 0.840 (max) /
0.832 (sum), and the episode-block bootstrap still returns diff +0.053, CI [-0.019, 0.155], P=0.09.
If even the baseline's own scores added to ours do not clear the CI, no in-model reproduction of that
signal will.

**(c) The arithmetic.** The CI half-width is about 0.13 on this subset. For the lower bound to clear
zero the headline would need roughly 0.92 difficult AUROC. 14 of the 43 difficult windows are OOS
STATUS/CO flips that every method already ranks at 0.998-1.000; the remaining 29 windows include
episode 6 (3 windows, ceiling 0.45) and episode 0 (6 windows, ceiling 0.68). The reachable ceiling is
about 0.86-0.87. The gap to 0.92 is not closable with this test on this data.

**Recommended paper move:** keep WADI as "leads every baseline; not statistically separable from the
linear residual over 11 episodes", and add the sentence the paper is currently missing (see 1.2):
on WADI the headline is significant against every learned baseline. That is a true, already-computed,
construct-matched claim and it is stronger than what the paper says now.

### 1.1 Rank 1 (bug class; protects all three wins; cheap). Fusion z-scores and the null expert's
### survival function are computed on TEST normals, while the paper says train-normal units.

`ensemble_final.py`, `ensemble_scores()` lines 119-132: `nm = y == 0` is the test label vector;
`nulltail = surv(lat[sd][nm], lat[sd])` ranks the global LatAD score against test normals, and
`z = lambda s: (s - s[nm].mean()) / (s[nm].std() + 1e-9)` standardises every fused row (`null+HC`,
`HCcoh+LatAD`, `cohmax+LatAD`, ...) on test normals. The paper (IoT2.html line 593 "All calibration
uses train-normal only"; Table A1 caption "in train-normal z-score units") says otherwise. The
per-community p-values and tails are clean (they use the 20% calibration slice, `Cal`); only the
fusion scale and the null expert's tail are affected. `HC_coh` alone is unaffected.

- Hypothesis: the fused numbers move by less than one seed-sd when the scale comes from the
  calibration slice; the claim in the paper becomes true.
- Experiment: train the global LatAD expert on `Xn[:nfit]` (the same 80% the community experts use;
  `train_vade` costs 3-16 s per fit), score the last 20% as its calibration slice, and derive both the
  survival reference and the z mean/std from that slice. `scores_*.npz` and `heads_*.npz` store test
  scores only, so this needs the retrain; do not score the existing 100%-train checkpoints in-sample
  (their in-sample std is optimistically small and would over-weight the null expert).
- Invariants: (i) the `HC_coh` row is bit-identical before and after; (ii) every single-expert AUROC
  is unchanged; (iii) `use_near=False`/no-residual expert score equals z(density) exactly (already
  verified once in the cleanwadi report, re-assert).
- Cost: 15-25 min of agent time, a few CPU-minutes (5 seeds x 3 datasets x one small VaDE fit),
  then `ensemble_final.py` at BOOT_REPS=2000 (about 1 min per dataset).
- Risk: the WADI/SWaT fused numbers shift by about 0.01 either way. That is the price of a correct
  claim; a reviewer who reads `ensemble_final.py` finds this in five minutes and it taints all three
  wins, not only WADI.
- Payoff for the WADI significance: none. Payoff for the paper: closes the one objection that can
  sink every significant row at once.

### 1.2 Rank 2 (metric/reporting; zero compute; already computed here). Report WADI significance
### against the learned baselines.

Same test, same episodes, same subset, comparator swapped (`boot()` with `d[ck]`):

| comparator | diff | 95% CI | P(diff <= 0) |
|---|---|---|---|
| linres | +0.037 | [-0.088, 0.181] | 0.32 |
| AE | +0.085 | [0.029, 0.150] | 0.0015 |
| TranAD | +0.101 | [0.026, 0.205] | 0.0005 |
| USAD | +0.138 | [0.045, 0.260] | 0.0000 |
| IF | +0.143 | [0.060, 0.263] | 0.0000 |

- Experiment: rerun `ensemble_final.py` with `COMPET["WADI_clean"]` set to each learned baseline (or a
  loop) so the JSON artifact exists; report the row in Table 3's significance column as "vs linres
  P=0.32; vs AE P=0.002; vs TranAD/USAD/IF P<0.001".
- Invariant: the linres row must reproduce +0.037 / P=0.32 from `headline_full_WADI_clean.json`.
- Risk: none. This is not a new claim about linres; it is the missing half of the WADI row. Check
  whether Table 3 or section 5 already states it; if not, add it.

### 1.3 Rank 3 (test choice; zero payoff; listed to close it). Do not swap the significance test.

A paired sign-flip permutation over per-episode AUROC contributions, a cluster-robust DeLong, or a
per-episode Wilcoxon all face the same 11-episode power, and the 3-3-5 per-episode table above shows
what they would return. Switching tests after seeing the result would be the definition of torture.
The block length L=3 and the episode resampling in `boot()` are fine as they are.

### 1.4 Rank 4 (data; near-zero payoff). More WADI episodes do not exist in the recording.

`wadi_labelsfull.npy` (full 1 Hz) holds 14 attack runs; the W=60/stride=30 windowing at 10x
downsampling yields 11 window episodes (three runs merge through window overlap). Reducing the
downsample or the stride recovers at most 14 episodes, which shrinks the CI half-width by
sqrt(11/14)=0.89, to about 0.115; still spanning zero at +0.04. A second WADI recording (the 2017
A1 release) is a different dataset, not more of this one, and would have to be added as a fourth
benchmark under the paper's inclusion rule; expensive, and it does not change the WADI_clean row.

### 1.5 Rank 5 (model; cheap pilot; low ceiling). Per-community linear residual as an extra A7 expert.

Mechanism: the WADI difficult windows are single-channel setpoint/flow steps of 1.5-3 sigma with the
plant otherwise intact (`fable_cleanwadi_report.md` section 5), exactly what a leave-one-channel-out
linear residual isolates; the VaDE whitened residual is gated off in the large communities because the
decoder residual drifts (ratios 1.6-13 measured in `chk_gates.py`), while it stays ON in the tight
3-4 channel ones (ratios 1.06, 1.39). A linear residual per community generalises better and would
pass the same held-out gate in more communities.

- Experiment: for each HAC community fit `loco_residual` on the 80% fit slice, calibrate on the 20%
  slice, add the S linear-residual experts to the HC alongside the S VaDE experts with the same
  cohesion weights; same rule applied to all three datasets; gate each linear expert by the same
  1.5 held-out ratio. No neural training; minutes on CPU.
- Invariant: on the Easy subset every added expert must score ~1.0; with the linear experts removed
  the HC must reproduce `HC_coh` 0.805 exactly.
- Ceiling: the oracle in 1.0(b) says at most ~0.84 and P~0.09. It will not deliver significance vs
  linres and it invites the "you fused the baseline in" objection. Mostly noise-chasing for the
  significance question; mechanism-grounded for +0.01 on the mean. Only worth running if the paper
  wants a "linear instantiation of A7" story, and only if HAI does not lose (linres is 0.586 on HAI
  and a bad expert adds noise to the HC maximum).

### 1.6 Closed routes (tested here or in the registry; do not re-run)

- **Score persistence (causal EMA over consecutive windows, all methods equally)**, `chk_ema.py` on
  cached scores: WADI headline 0.829 / linres 0.775 at alpha=0.5, bootstrap P=0.24 (0.27 at
  alpha=0.33); SWaT stays significant (P=0.009); HAI weakens (P=0.025 at 0.5, P=0.086 at 0.33,
  because AE gains more from smoothing than we do). Dead end; also changes the evaluation protocol.
- **Richer temporal/spectral representation**: +0.002 on WADI (`repr_ablation.json`). Dead.
- **K x latent sweep with train-only selection**: whole range 0.730-0.751 for the global model
  (`fable_cleanwadi_sweep_summary.json`). Dead.
- **Forcing the residual head on**: rejected by the train-only gate; adopting it is a test-label choice.
- **Coverage / OOS exclusion**: 5 of 519 test normals OOS; excluding them changes nothing.
- **Dropping AIT_004**: identical verdict. Not justified by the data.

### 1.7 One model change that is worth a Modal run, but as streamlining, not as a WADI fix

Dropping the nearest-component head inside the experts (see 2.1) may lift WADI a little because the
global-model head ablation shows the nearest head subtracts value on every dataset. Run it for the
simpler model; if WADI moves, that is a bonus to be reported, not a route to chase.

---

## Part 2. Model streamlining: what earns its place

Evidence per component, then the minimal model.

### 2.1 Nearest-component NLL head: drop (evidence says it costs, and the code contradicts its own docstring)

Table 6 (`rev4_ablation_clean.json`, `rev4_ablation.json`), global model, difficult subset:

| head | WADI | HAI | SWaT |
|---|---|---|---|
| latent density alone | 0.758 | 0.802 | 0.794 |
| base = density + nearest | 0.743 | 0.801 | 0.775 |

Adding the nearest head is -0.015 / -0.001 / -0.019. `models_vade.anomaly_score_hard` says in its
docstring "the reported model uses density-only base", but its default is `use_near=True` and
`modal_experts.py` calls it with the default, so every shipped expert and the global null expert
include the head. Either the docstring is stale or the shipped model carries a head the ablation says
to remove; both should not stand. The paper's A5 narrative (rare-regime-safe nearest scoring) is
also carried by the high-K density head, which the code comment already states.

- Experiment: `modal_experts.py --full 1` with `use_near=False` threaded through (one keyword), all
  three datasets, 5 seeds; rerun the global expert the same way (combine with 1.1 so there is one
  retrain). About one Modal run (seeds in parallel, 10-20 min wall-clock, a few dollars) plus
  `ensemble_final.py`.
- Invariant: expert score with `use_near=False`, residual gate off, basin off must equal z(density)
  exactly. HAI must stay >= 0.849 - 1 sd (0.831) with the CI vs AE still excluding zero; SWaT must
  stay significant vs linres.
- Risk: the residual head is z-added on top of the base, so removing the nearest term changes the
  base's scale; on HAI the base+resid combination could shift. Low risk given the table.

### 2.2 Basin-agreement head: remove from the code path (paper already says it is not in the model)

IoT2.html line 318 and 590: "not realized by the reported model", "the reported model carries no
dedicated overlap head". `modal_experts.py --full 1` still calls `fit_basin_head` and scores with
`use_basin="auto"`. Measured on 8 WADI_clean communities spanning sizes 3-21 and cohesion 0.45-1.00
(`chk_gates.py`): `basin_frac_amb` 0.000-0.004 against a 0.15 deadzone, `basin_lam` = 0.000 on every
one. It is inert, so removing the call changes no number, but a reviewer diffing code against text
sees a head the paper says does not exist.

- Experiment: log `_basin_lam` for every community of every dataset in the 2.1 rerun; if any is
  non-zero, the paper text is wrong, not the code, and that has to be resolved first. Then delete the
  basin methods from the shipped path (keep them in git history; SKAB, the only dataset that ever
  triggered them, is out of the paper).

### 2.3 Whitened-residual head (auto-gated): keep; it is the HAI win

Table A1 factorization block: density-only experts 0.816 / 0.814 / 0.827 vs full heads
0.824 / 0.849 / 0.840. HAI gains +0.035 from the residual head; without it HAI's margin over AE
(0.757) drops to 0.057 and the CI likely touches zero. Two wording corrections for the paper: the gate
is evaluated per community, and on WADI it is ON in the tight small communities (ratios 1.06 and 1.39
for the two 3-4 channel communities sampled) and OFF in the large ones (1.6-13). "Off on WADI" is
true of the global model only.

### 2.4 Cohesion weighting vs plain HC: keep (zero parameters, consistent small gain)

Table A1 aggregation block: HC 0.812 / 0.828 / 0.829 vs HC_coh 0.824 / 0.849 / 0.840, i.e.
+0.012 / +0.021 / +0.011, roughly one seed-sd on WADI and HAI and 2.5 sd on SWaT. It is the only row
that leads on all three. It adds no hyperparameter (weight = cohesion x sqrt(size), both measured on
train-normal). Keep, but the paper should present plain HC as the minimal variant and cohesion as the
refinement, which Table A1 already does.

### 2.5 Null-expert fusion: keep (it is what carries HAI), after fixing 1.1

HC_coh alone 0.805 / 0.801 / 0.822 vs HCcoh+LatAD 0.824 / 0.849 / 0.840. HAI depends on the global
expert (+0.048). The max-fused `null+HC` variant (0.806 / 0.846 / 0.823) is strictly dominated; keep it
in Table A1 only.

### 2.6 VaDE inside the communities vs a linear PCA + high-K GMM density: keep VaDE (HAI needs it)

PILOT (`pilot_gmm_experts.py`): same HAC communities, same 80/20 fit/calibration split, same
latent size and K as the VaDE experts, sklearn PCA + diagonal GMM(K=50) per community, same HC_coh
aggregator; 5 seeds. Invariant passed: the VaDE-full fused row reproduces the headline exactly
(0.824+-0.012, 0.840+-0.004, 0.849+-0.018).

| expert type, difficult AUROC | WADI_clean | SWaT_canon | HAI |
|---|---|---|---|
| VaDE density-only, HC_coh | 0.800+-0.009 | 0.781+-0.009 | 0.777+-0.005 |
| VaDE full heads, HC_coh | 0.805+-0.012 | 0.822+-0.014 | 0.801+-0.020 |
| PCA+GMM, HC_coh | 0.786+-0.008 | 0.801+-0.007 | 0.720+-0.008 |
| PCA+GMM, HCcoh+LatAD | 0.815+-0.007 | 0.847+-0.002 | 0.786+-0.014 |
| VaDE full, HCcoh+LatAD (headline) | 0.824+-0.012 | 0.840+-0.004 | 0.849+-0.018 |

On WADI and SWaT a linear community expert is within noise of the neural one (SWaT even edges it
once fused). On HAI it loses 0.063, because the HAI signal lives in the per-community whitened
residual, which needs a decoder; a GMM has none. So the neural expert earns its place through HAI,
not through WADI. This is the correct answer to "is the VaDE baroque": it is needed on exactly the
dataset where the residual head is needed, which is a clean story for the paper (and matches the
"linear on WADI" reading of `fable_cleanwadi_report.md`). Do not ship the GMM variant; do consider
citing this as the reason the community expert is not linear.

### 2.7 Untested simplification worth one run: flat partition instead of 44 nested subtrees

The expert library uses every HAC subtree of size 3-25 (WADI 44, SWaT 25, HAI 28 communities,
overlapping by nesting). A flat cut (one partition at, say, 8-12 communities) would make the model
3-5x cheaper and easier to describe. There is no ablation of this axis in the registry.

- Experiment: same `modal_experts.py` with `comms` replaced by the leaves of a single `fcluster` cut
  chosen on train-normal (e.g. the cut maximising mean cohesion subject to size >= 3); rerun HC_coh
  and the fusion.
- Invariant: nested and flat must agree on the Easy subset (~1.0); the flat HC has fewer p-values so
  HC's sqrt(S) scale changes, which the z-standardisation absorbs.
- Risk: the WADI gain is attributed to multiscale coverage (small tight communities catch the sparse
  faults, large ones the coordinated drifts); the flat cut may lose WADI's +0.07 over the global model.
  Unknown payoff; the only streamlining axis whose answer is not already in the registry.

### 2.8 Code-only dead weight (no numbers change)

`VaDE.anomaly_score` (recon + nearest legacy path), `use_recon`, `fit_residual_whitener` when the
resid head is not used, `PlainVAE` (sequential ablation only). Move to an `ablations/` module or
drop from the shipped checkpoint export.

### 2.9 The minimal model that keeps both significant wins and the WADI lead

HAC nested correlation communities (train-normal) -> per-community VaDE with two heads only:
high-K latent density + auto-gated per-community whitened residual (no nearest, no basin) ->
cohesion-weighted Higher Criticism over calibration-slice p-values -> z-sum with the global expert
(same two heads), every z and survival reference taken from the calibration slice (1.1).
Heads go from four to two, gates from two to one, and every remaining piece has a measured
contribution on at least one dataset: density (all three), residual (HAI, SWaT), factorization
(WADI +0.07 over global), cohesion (+0.01 to +0.02 everywhere), null fusion (HAI +0.05).

Order of operations, so that each step has one retrain and a stated invariant: 1.1 and 2.1 together
in one Modal/CPU run (global expert on the 80% slice, `use_near=False`, basin removed, log
`_basin_lam` and `_resid_gen_ratio` per community); then `ensemble_final.py` at 2000 reps with the
learned-baseline comparators added (1.2); then, optionally, 2.7. Expected agent time for the first
two steps: 30-45 minutes plus one Modal run (10-20 min wall-clock, a few dollars); no paper edits
until the invariants in 1.1 and 2.1 pass.
