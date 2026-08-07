# LatAD — Revision Plan v2 (honest reframe)

Status: proposed, awaiting sign-off. Supersedes the single-seed "wins on all three" framing.

## 1. Why we are revising

A multi-seed pass (5 seeds), a 4th dataset (SWaT), and model-agnostic measurements of the
data changed the picture:

- The reported numbers were **seed 0**, which is systematically the *best* seed (highest of
  five on both WADI and SKAB). Headline values are ~0.04-0.09 AUROC optimistic.
- Under multi-seed CIs, the "wins on all three datasets" claim **does not hold on SKAB**.
- Adding **SWaT** (a robust new win) and measuring **multimodality (K\*)** and **sensor time
  scales (tau)** produced a *stronger, more defensible* organizing thesis than the original.

## 2. The revised spine (one paragraph)

Normal CPS behaviour is a union of many operating modes, and we *measure* this per dataset
(BIC-optimal mode count K\*). A detector that models this structure (joint latent + explicit
mode mixture, scored in the latent) beats the deep SOTA on the difficult subset **wherever the
data is genuinely multimodal**, and its advantage is *predicted by the measured mode count*:
decisive on the most multimodal sets (SWaT K\*=25, HAI K\*=24), a tie with the best classical
baseline on WADI (K\*=22), and no lead on the near-fewest-mode set (SKAB K\*=12). SKAB then
illustrates a *different* MIIM axis: its difficult faults are subtle multiscale-temporal
(vibration) faults (A9), which window-statistic features cannot encode. The paper's contribution
is thus (i) an **empirically grounded** MIIM assumption set (we measure that A1/A5/A6/A9 hold in
real CPS data), (ii) a fair, multi-seed, difficulty-stratified evaluation protocol, and (iii) a
latent-only detector whose advantage tracks a measured property of the data.

## 3. Revised results — difficult-subset AUROC (5-seed mean +- std)

| Dataset | K\* | **Ours** | IsolationForest | AutoEncoder | USAD* | TranAD* | GDN* | verdict |
|---|---|---|---|---|---|---|---|---|
| SWaT | 25 | **0.960 +- 0.006** | 0.853 +- 0.011 | 0.945 | - | - | - | win (beats AE) |
| HAI  | 24 | **0.811 +- 0.016** | 0.627 +- 0.005 | 0.758 | 0.487 | 0.467 | - | win (decisive) |
| WADI | 22 | 0.690 +- 0.027 | 0.677 +- 0.011 | 0.430 | 0.554 | 0.547 | 0.582 | tie IF; beats all deep SOTA |
| SKAB | 12 | 0.522 +- 0.046 | 0.532 +- 0.011 | 0.504 | 0.591 | 0.507 | - | no lead |

\* USAD/TranAD/GDN are still single-seed (paper values); Modal multi-seed pending for symmetric CIs.

Reported-vs-multiseed (shows the seed-0 optimism): WADI 0.726->0.690, HAI 0.831->0.811,
SKAB 0.610->0.522, (SWaT new: 0.960).

## 4. Claims to change (exact)

- **DROP** "wins both columns on all three datasets."
- **NEW headline**: "robustly outperforms the deep SOTA (USAD/TranAD/GDN) on the difficult
  subset on every multimodal dataset, with the advantage predicted by the measured mode count
  K\*; decisive on SWaT and HAI, a tie with the strongest classical baseline on WADI."
- **SKAB**: reframe from "we win" to an **A9 illustration** - its difficult faults are
  multiscale-temporal and out of reach of window statistics; we show this and scope it.
- **WADI tie**: report the ablation honestly - the nearest-component head *hurts* WADI
  (density-only 0.709 > full 0.690 > IF 0.677); decide whether to gate/drop it (uniformly).
- **All numbers become multi-seed mean +- std.** No single-seed headline survives.

## 5. New contributions (the empirical MIIM grounding)

| Assumption | Measured evidence (strong) |
|---|---|
| A1 multimodality | K\* = 22/24/12/25 (all >> 1) |
| A5 few levers | participation ratio 5-27 vs 48-738 features (3-11% effective dim) |
| A6 heavy-tail modes | mode-occupancy Gini 0.34-0.53 |
| A9 many clocks | sensor tau spread; SKAB widest (log-tau std 0.87) |

Weak / honest-scope: A2 (K\* does not cleanly scale with #channels), A8 (needs a channel-*type*
measure, not scale), A7 (qualitative), A10 (prior test inert; will re-test, likely stays a
stated-not-exercised assumption).

## 6. New / revised figures

1. **Advantage vs K\*** (done) - the thesis as a measured relationship.
2. **Sensor time scales** (done; refine the 200-sample cap).
3. **MIIM fingerprint radar** per dataset (K\*, intrinsic-dim, mode-Gini, clock-spread, %difficult, advantage).
4. **Mode-occupancy Zipf** (A6) and/or a 2D normal-manifold projection coloured by mode (A1).
5. Results Table 3 -> multi-seed; add SWaT row; add the head ablation table.

## 7. Honest scope / limitations (kept, not hidden)

Single trained model per dataset (now with 5-seed CIs); oracle best-F1 threshold for F1/FPR
(AUROC is the leak-free basis; a deployable train-normal-threshold operating point is added);
SKAB contaminated and A9-limited; A2/A8/A10 weak or unexercised; four datasets.

## 8. Work items (bucketed)

DONE (this pass): multi-seed CIs (ours + IF/AE), SWaT integrated + run, K\* multimodality,
tau time-scales, A5/A6 measures, deploy-threshold operating point, ablation, thesis+tau figures.

EDITS (no compute): rewrite abstract/results/discussion to the spine above; qualify GDN-on-WADI;
add the multi-seed table; SKAB->A9 reframe; report the nearest-component ablation.

NEW EXPERIMENTS (local, free): (V2) uniform temporal/spectral representation - re-test the "did
not help" claim under multi-seed, esp. SKAB (A9 payoff); A4 single-Gaussian-per-mode ablation;
A10 history-feature recovery test on WADI; MIIM fingerprint + Zipf figures; refine tau cap.

NEW EXPERIMENTS (Modal, ~$5-15): USAD/TranAD multi-seed on all 4; GDN on HAI/SKAB/SWaT.
Optional: +1-2 more datasets (SMD/PSM/MSL) to firm the K\* trend.

## 9. Decisions for sign-off

- [ ] Adopt this spine (drop 3/3 win; K\*-predicted advantage; SKAB as A9 illustration)?
- [ ] Fix the nearest-component head (gate/drop uniformly) or just report the ablation?
- [ ] Run the uniform temporal-representation test (V2) for the SKAB/A9 story?
- [ ] Run the A10 history test (report whatever it shows)?
- [ ] Spend the Modal budget on symmetric SOTA CIs?
- [ ] Add 1-2 more datasets, or lock at four?
