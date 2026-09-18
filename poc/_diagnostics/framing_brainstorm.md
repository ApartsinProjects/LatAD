# Framing brainstorm: strongest defensible claims for two weak spots (report only, paper untouched)

Scripts and artifacts (all in `_diagnostics/`, all re-runnable, nothing retrained on GPU):

| item | file |
|---|---|
| Issue 1 index test | `framing_issue1_index.py` -> `framing_issue1_index.json`, `framing_issue1_index_sensitivity.log` |
| Issue 1 synthetic crossover | `framing_issue1_synth.py` -> `framing_issue1_synth.json`, `.svg`, `.png`, `.log` |
| Issue 2 ground-truth + calibrated localization | `framing_issue2_localization.py` -> `framing_loc/loc2_summary.json`, `loc2_<ds>.jsonl` (per window), `loc2_episodes_<ds>.jsonl` (per attack), `size_check.json/.log`, `perseed_rank.log` |
| SWaT attack table (text copy of iTrust List_of_attacks_Final, 36 labelled attacks) | `swat_attack_list_deepsentinel.csv` |

Ranked by payoff:

1. **Issue 1, reframing (no new experiment, numbers already in Tables 3 and 6): "density beats reconstruction" is 3-for-3 against the deployed reconstruction detectors; it is only 1-for-3 against the paper's own whitened residual head.** Highest payoff, zero risk.
2. **Issue 2, positive localization result with ground-truth targets and a train-normal-only calibration rule**: episode-level top-1 significantly above the coverage-aware random baseline on SWaT and HAI, positive but underpowered on WADI. Real result, moderate size, with two caveats that must be printed next to it.
3. Issue 1, synthetic mechanism figure: clean, supports the mechanism, does not by itself license a law.
4. Issue 1, a per-dataset "reconstructable-but-improbable" index as a predictor: not defensible as a law at n=3 (details below); usable as a diagnostic decomposition only.

---

## Issue 1. "Density beats reconstruction" holds cleanly only on HAI (Table 6)

### Diagnosis

Table 6 head gaps (density minus whitened reconstruction residual, difficult subset): HAI +0.107, SWaT -0.041, WADI -0.084. The recomputation from `heads_<ds>.npz` under the canonical 30-window WADI mask gives WADI 0.655 vs 0.732 (-0.077), HAI 0.802 vs 0.695 (+0.106), SWaT 0.794 vs 0.835 (-0.040); same conclusion, the paper's WADI pair (0.656/0.740) is a separate 5-seed run.

The key observation is **which reconstruction score the density head is being compared with**. Table 6 compares it with the model's own *whitened* residual (residual whitened on train-normal covariance, the A8 head), which is a far stronger reconstruction score than the reconstruction the deployed detectors actually use. Against the deployed reconstruction detectors on the same difficult subsets (Table 3 values, canonical masks):

| difficult subset | density head (Table 6) | USAD | TranAD | GDN | density minus best deep recon |
|---|---|---|---|---|---|
| HAI | 0.802 | 0.477 | 0.444 | 0.481 | **+0.321** |
| SWaT | 0.794 | 0.658 | 0.655 | 0.665 | **+0.129** |
| WADI | 0.656 | 0.579 | 0.613 | 0.660 | +0.043 vs TranAD, tie vs GDN (-0.004) |

So the single density head, with no residual, no nearest-component term and no community factorization, beats USAD and TranAD on all three difficult subsets and matches GDN on WADI. The sign flips in Table 6 come from the whitened residual, i.e. from a reconstruction term the paper itself strengthened and then gated in. That dissolves the apparent contradiction rather than softening it.

Per-window decomposition (`framing_issue1_index.json`, external reconstruction = mean USAD/TranAD percentile against normal windows, density = LatAD density-head percentile), difficult anomalies only:

| dataset | n | median percentile under external recon | median percentile under density | reconstructable-but-improbable share (recon pct <= 0.5, density pct >= 0.9) | seen by density (>= 0.9) | seen by external recon (>= 0.9) | seen by neither |
|---|---|---|---|---|---|---|---|
| HAI | 167 | **0.46** (chance) | 0.93 | **0.27** | 0.62 | 0.02 | 0.38 |
| SWaT | 85 | 0.72 | 0.94 | 0.16 | 0.60 | 0.32 | 0.35 |
| WADI | 30 | 0.65 | 0.71 | **0.00** | 0.20 | 0.23 | 0.70 |

HAI is the textbook reconstructable-but-improbable case: the deployed reconstruction detectors place its difficult attacks at the median of normal (0.46), the density head places them at the 93rd percentile, and 27 percent of them sit in the "reconstructs like a normal window, improbable under density" quadrant (the converse quadrant is empty on every dataset). WADI's difficult anomalies are different in kind: 70 percent are seen by neither score at the 90th percentile; they are not reconstructable-but-improbable, they are weak under everything, which is exactly why the community factorization (not the head choice) is what lifts WADI (Table A1: 0.634 -> 0.795).

### Cheap tests run

**(a) Does a per-dataset index predict the Table 6 gap?** Three candidates, n = 3:

| index | HAI | SWaT | WADI | Spearman with Table 6 gap (+0.107, -0.041, -0.084) |
|---|---|---|---|---|
| reconstruction-detector collapse (mean USAD/TranAD difficult AUROC) | 0.471 | 0.656 | 0.596 | **-0.50** (mis-orders WADI vs SWaT) |
| reconstructable-but-improbable share (0.5 / 0.9 thresholds) | 0.27 | 0.16 | 0.00 | +1.00 |
| threshold-free: density-head AUROC minus external-recon AUROC | +0.33 | +0.14 | +0.06 | +1.00 |

The natural independent index (how far USAD/TranAD collapse) separates HAI from the other two but does not order WADI against SWaT, so it cannot predict the sign of the two negative gaps. The two indices that do order all three share the density head with the quantity they predict, and the share index is threshold-sensitive (`framing_issue1_index_sensitivity.log`: HAI's share runs from 0.02 at a 0.95 density threshold to 0.59 at 0.9/0.9; the HAI > SWaT > WADI ordering holds in 4 of 5 threshold settings). SKAB and Cranfield add nothing here: every head on SKAB is at chance (density 0.46, density+recon 0.456, `fix_recon_SKAB.json`), and Cranfield has no labelled faults in the A8 screens.

**(b) Synthetic crossover** (`framing_issue1_synth.py`, CPU, ~2 min). K = 5 regimes on a 6-dim manifold inside R^20, separation delta in {4, 6, 10} (silhouette 0.51 / 0.65 / 0.78, the multimodality knob), MLP autoencoder + GMM-in-latent trained on normal only. Faults displace a normal point by (1-p) x off-manifold (magnitude fixed at +4 sd of normal reconstruction error) + p x in-manifold toward the midpoint of the gap to the nearest regime (reconstructable but improbable). Seed-mean over 3 seeds:

| p (in-envelope-ness) | delta = 4: recon / density / gap | delta = 6 | delta = 10 |
|---|---|---|---|
| 0.0 | 0.93 / 0.51 / -0.42 | 0.92 / 0.50 / -0.42 | 0.91 / 0.50 / -0.41 |
| 0.4 | 0.75 / 0.62 / -0.12 | 0.75 / 0.74 / -0.02 | 0.78 / 0.91 / +0.12 |
| 0.5 | 0.68 / 0.65 / -0.03 | 0.70 / 0.81 / +0.11 | 0.76 / 0.96 / +0.20 |
| 0.6 | 0.62 / 0.72 / +0.10 | 0.68 / 0.87 / +0.19 | 0.78 / 0.98 / +0.19 |
| 1.0 | 0.51 / 0.81 / +0.29 | 0.71 / 0.96 / +0.25 | 0.90 / 1.00 / +0.10 |

Invariants stated in advance: (i) p = 0 -> reconstruction wins: holds (0.91-0.93 vs 0.50). (ii) p = 1 -> density wins and reconstruction falls toward chance: holds for the winner at every separation; reconstruction falls to chance only at delta = 4 (0.51) and stays at 0.90 at delta = 10, because the nonlinear decoder does not fully interpolate across a wide gap. (iii) the crossover moves little with separation: only partly; it shifts from p = 0.5 (delta 4) to 0.3 (delta 10), so better-separated regimes let density win earlier, but the sign is set by fault geometry at every separation. Left panel of `framing_issue1_synth.png` is the mechanism figure; the right panel (gap vs reconstruction-detector AUROC, with the three real datasets overlaid) shows the real points do **not** lie on the synthetic curves, which is the visual form of the n = 3 caveat in (a).

### Strongest defensible framing and proposed wording

Section 4.3 / 6 ("Why reconstruction fails and density wins"), replacing the Table 6 sentence that currently concedes the point on two datasets:

> Scoring by density rather than reconstruction is the right default for the deployed reconstruction detectors: the single latent-density head, with no residual term and no factorization, exceeds USAD and TranAD on the difficult subset of all three plants (HAI 0.802 against 0.477 and 0.444; SWaT 0.794 against 0.658 and 0.655; WADI 0.656 against 0.579 and 0.613) and matches GDN on WADI (0.660). The margin is largest where the faults are reconstructable but improbable: on HAI the reconstruction detectors place the difficult attacks at the median of normal operation while the density head places them at the 93rd percentile, and 27 percent of HAI's difficult attacks reconstruct at least as well as a typical normal window yet fall above the 90th normal percentile in density; on WADI that share is zero and 70 percent of the difficult attacks are weak under both scores, which is why the community factorization, not the head, carries the WADI gain. A reconstruction residual *whitened on train-normal* (Table 6) is a stronger term than the reconstruction the deep detectors score with; it overtakes the density head on WADI and SWaT, and this is precisely why LatAD keeps it as a gated head where it generalizes to held-out normal instead of discarding reconstruction.

Table 6 caption: add "the reconstruction row is the *whitened* residual; the reconstruction detectors of Table 3 score 0.44 to 0.66 on the same subsets". A small figure from the synthetic left panel (gap vs in-envelope-ness at three separations) can go in the appendix as the mechanism illustration, captioned as synthetic.

### Verdict

A quantified conditional law ("density wins when index X exceeds Y") is **not defensible** at n = 3: the only index that is independent of the density head mis-orders WADI against SWaT, and the indices that order all three either share the density head or are threshold-sensitive. What is defensible, and stronger than the current text, is the three-dataset statement against deployed reconstruction detectors plus the per-window quadrant numbers that make the mechanism concrete, with the synthetic crossover as an illustration. Do not add SKAB or Cranfield.

---

## Issue 2. Subsystem localization failed at top-1 on WADI (chance) and SWaT (below chance)

### Diagnosis (what was wrong with the earlier evaluation)

Three separate things, all now measured.

1. **SWaT's two "loud" communities are a numerical degeneracy, not a signal.** Communities 1 {P402, UV401, P501} and 17 {MV101, AIT201, MV201, P201} (plus 16) have calibration surprise with standard deviation 0 to 1e-14 in most seeds (near-constant discrete channels in calibration), so the per-community z-scoring inside `modal_experts.py` divides by ~0 and their test surprise is 1e9 to 1e21 on **every** window, normal or attacked (community 1: test-normal median = p90 = max = 1.6e17 in seed 0). That is why they were argmax in 100 percent of attacked windows. Their train-normal p-value is undefined, so a train-normal-only rule ("a community with degenerate calibration cannot be ranked") removes them without touching labels.
2. **The earlier HAI "3.2x chance" was a proxy artefact.** With ground-truth targets (HAI 20.07 timetable, 38 attacks, one-to-one with the 38 labelled runs) the raw argmax scores 0.238 at window level against a coverage-aware random baseline of 0.248, i.e. chance; the earlier 0.44 came from the proxy target (max-|z| channel), which shares its deviation driver with the surprise.
3. **WADI coverage**: 2 of the 11 attack episodes target channels in no community (2_MCV_007, 1_P_006), two more (11 Oct 00:07 and 12:16 runs, attacks 13/14) are absent from every text copy of the attack table found, so WADI has 9 target-known episodes, 7 coverable.

Ground truth used: SWaT iTrust attack list (36 physically labelled attacks; timestamps align with the canonical raw label runs to the second, e.g. attack 1 at 10:29:14 = raw second 1754); HAI 20.07 timetable pp. 39-41 of the technical-details PDF, cross-checked against the `attack_P1/P2/P3` columns; WADI 13-row table shipped with TranAD, re-dated by time of day against the raw label runs (its dates are mangled).

### Fix tested: calibrated per-community surprise

Scores compared (all seed-averaged, all from `expert_<ds>.npz`): **raw** (the original); **excl** (raw with degenerate communities removed); **rank** (per seed, each community's test surprise converted to its upper-tail empirical p-value within its own calibration surprise, then seed-averaged; degenerate communities excluded; this is the same quantity the paper's Higher-Criticism aggregator consumes); **onset** (rank minus its median over the previous 10 windows: which community *rose*); **oracle** (z against test-normal, not deployable, diagnostic only).

**Episode level** (contiguous attacked windows, score averaged over the episode, target = any community containing any attacked channel; random = coverage-aware `1 - C(S-kc,k)/C(S,k)` averaged over episodes; p = Monte-Carlo probability of at least the observed hits under that baseline, 20,000 draws):

| dataset (episodes with known target, coverage) | random top-1 / top-3 | raw top-1 | **rank top-1** (p) | rank top-3 | excl top-1 (p) | onset top-1 (p) |
|---|---|---|---|---|---|---|
| SWaT (24, 1.00) | 0.215 / 0.490 | 0.083 | **0.458** (0.004) | 0.792 | 0.667 (< 5e-5) | 0.250 (0.42) |
| HAI (38, 0.95) | 0.246 / 0.557 | 0.263 | **0.395** (0.024) | 0.684 | 0.263 (0.47) | 0.447 (0.005) |
| WADI (9, 0.78) | 0.131 / 0.308 | 0.333 | 0.333 (0.086) | 0.556 | 0.333 (0.085) | 0.222 (0.33) |

Per-seed stability of the rank rule (episode top-1 over coverable episodes, `perseed_rank.log`): SWaT 0.475 +/- 0.020, HAI 0.433 +/- 0.028, WADI 0.457 +/- 0.057 (3 of 7).

**Window level, difficult subset** (rank rule): SWaT 0.575 vs random 0.163 (n = 80), HAI 0.539 vs 0.277 (n = 167), WADI 0.222 vs 0.085 (n = 36). Window level, all attacks: SWaT 0.303 vs 0.178, HAI 0.500 vs 0.248, WADI 0.245 vs 0.108.

Controls run on this code path: synthetic positive/negative pointer control passes on all three datasets (`size_check.log`); hits are spread across attacks, not carried by one community (SWaT excl: 12 distinct top communities across 24 episodes, 12 of 14 hits occur when the pointer is *not* the modal community; HAI rank: 15 distinct top communities, 11 of 26 hits off the modal community; `loc2_episodes_<ds>.jsonl` lists every attack with the named community's channels).

### Two caveats that must be printed next to any positive claim

**(i) Community size.** Communities are nested HAC subtrees, so the largest one is easy to hit: a *constant* pointer that always names the largest community would score 0.83 on SWaT (24 of 51 channels) and 0.74 on HAI (12 of 59), above the detector. The detector's named community is small (mean 6.5 channels on SWaT, 5.6 on HAI, 5.4 on WADI, out of 51 / 59 / 122), so the right comparison is at matched resolution: against a random community drawn with probability proportional to its size (`size_check.json`), rank top-1 is 0.458 vs 0.350 on SWaT (excl 0.667 vs 0.350), 0.417 vs 0.374 on HAI (onset 0.472), and 0.43 vs 0.19 on WADI (7 episodes). The SWaT margin survives; the HAI margin against the size-weighted baseline is small. Channel-level precision of a top-1 hit is 0.2 to 0.36 (the named 5-to-8-channel community contains one or two attacked channels).

**(ii) Calibration drift.** Even after exclusion, one community is the top-1 on 78 percent of SWaT *normal* windows (community 18, stage 1/2 dosing) and 39 percent of HAI normal windows (community 1, {P1_PCV02Z, P2_SD01, P2_SIT01}), because their test-normal surprise sits far above their calibration range. Localization is only invoked when the plant-wide alarm fires, so this does not create false alarms, but it means the top-1 pointer has a drift-driven prior; the off-modal hit counts above show the result does not rest on it. The `onset` (change-based) rule removes the prior entirely (flat normal profile, max 10-17 percent) and is the best rule on HAI, but it is at chance on SWaT, so it is not the single rule to report.

### Strongest defensible framing and proposed wording

Report **one rule** (rank calibration with degenerate communities excluded), episode level, with the coverage-aware baseline, shortlist size, and the two caveats. Section 7 (deployment) and the sentence in 4.4 that currently asserts localization:

> The per-community surprises also localize. Ranking communities by their train-normal upper-tail p-value (the quantity the Higher-Criticism combiner already uses; communities whose calibration surprise is degenerate carry no p-value and are not ranked) and naming the top community per attack episode, the named community contains an attacked channel in 46 percent of SWaT attacks (11 of 24, coverage-aware random 22 percent, p = 0.004) and 40 percent of HAI attacks (15 of 38, random 25 percent, p = 0.02), and a three-community shortlist contains it in 79 and 68 percent (random 49 and 56 percent). The named community averages six channels of 51 (SWaT) and 59 (HAI). WADI, with 9 attacks whose targets are published and 7 of them inside any community, is directionally the same (3 of 9 at top-1, random 13 percent) but too small to test. Two limits apply: a community whose surprise drifts above its calibration range is named disproportionately often even on normal windows, so the pointer is a shortlist for triage rather than a diagnosis; and larger communities are easier to hit, so the shortlist is reported at its size.

Small table for Section 7 (episode level, rank rule):

| dataset | attack episodes (targets known) | named-community size (channels / plant) | top-1 | random top-1 | top-3 | random top-3 |
|---|---|---|---|---|---|---|
| SWaT | 24 | 6.5 / 51 | **0.46** | 0.22 | **0.79** | 0.49 |
| HAI | 38 | 5.6 / 59 | **0.39** | 0.25 | **0.68** | 0.56 |
| WADI | 9 (7 coverable) | 5.4 / 122 | 0.33 | 0.13 | 0.56 | 0.31 |

Also correct the earlier internal number: the HAI "0.44 vs 0.14 (3.2x)" from `localization_eval.md` was proxy-based and should not be reused; the ground-truth raw-argmax figure on HAI is 0.24 (chance).

### Verdict

**Stronger claim achievable, but modest, and only in the calibrated, episode-level, size-reported form above.** It is a positive localization result on two datasets (p < 0.05 against the coverage-aware baseline on both, stable across seeds), directionally positive on the third, with the top-3 shortlist the more comfortable headline (0.79 / 0.68 / 0.56 vs 0.49 / 0.56 / 0.31). It should not be stated as "the most-surprised community identifies the attacked subsystem": at top-1 the margin over a size-matched baseline on HAI is small, and the pointer carries a drift prior. If the authors prefer a single sentence, the defensible one is: "a three-community shortlist ranked by train-normal p-value contains the attacked subsystem in 68 to 79 percent of HAI and SWaT attacks, against 49 to 56 percent for a random shortlist of the same size."

### What would raise the localization claim further (not run; each is cheap)

- Recalibrate the per-community z inside `modal_experts.py` with a floor on the calibration standard deviation (or rank-calibrate there), which fixes the degeneracy at the source and would also let the existing `max (most-surprised community)` aggregator row in the Table 7 ablation be re-measured on SWaT (currently 0.804).
- Re-fit calibration on a later normal slice (or a rolling normal baseline) to remove the drift prior, then re-run `framing_issue2_localization.py`; the `oracle` column (test-normal z) bounds what this can give: episode top-1 0.708 SWaT, 0.342 HAI.
- Non-nested communities (a partition instead of HAC subtrees) would make hit@k directly interpretable and remove the size confound.
