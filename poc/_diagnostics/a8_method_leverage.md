# A8 method leverage: kNN (local-density) head and expected-regime (context) head on the clean pipeline

Scripts: `a8_method_leverage.py` (`fit`: refit of the shipped recipe, heads, gates, per-seed npz; `eval`: fusion into
the clean headline, 5 seeds, subsets, paired episode bootstrap), `a8_method_leverage_ctl.py` (persistence controls,
per-episode breakdown), `a8_method_leverage_summary.py` (tables). Artifacts: `a8_leverage_<ds>_seed<s>.npz`,
`a8_leverage_<ds>_feat.npz`, `a8_leverage_subsets_<ds>.npz`, `a8_method_leverage.jsonl`,
`a8_method_leverage_eval.json` (WADI, SWaT), `a8_method_leverage_eval_HAI.json`, `a8_method_leverage_ctl.json`,
`a8_method_leverage_tables.md` (every table in full, including the per-window WADI dump). Paper and shipped model
files untouched.

Everything is on the CLEAN pipeline of `clean_recompute.md`: FIX 2 clip (WADI ±10, HAI/SWaT none), FIX 3 subsets
(WADI_clean difficult **30** / double-hard **19**; HAI 167 / 84; SWaT_canon 85 / 59), community experts
`sota_bundle/experts_full`, FIX 1 train-normal calibration of the fusion. Bases are the stored `scores_<ds>.npz`
(5 seeds) and the headline `HCcoh+LatAD` recomputed with `ensemble_final.py`'s construction; they reproduce the
clean tables (WADI 0.771 / 0.662, HAI 0.846 / 0.814, SWaT 0.837 / 0.773; HC_coh-alone WADI 0.795 / 0.696).
The heads come from a refit of the shipped recipe (`build_scores_table.py`: K/latent per CFG, 40 epochs, warmup 8,
k_density 80, resid/basin auto), one refit per stored seed. Refit check against the stored LatAD per seed: SWaT
corr 1.000 (bit-identical CPU path), HAI 0.982 to 0.993, WADI 0.977 to 0.984; refit difficult AUROC 0.627 / 0.812 /
0.803 vs stored 0.634 / 0.811 / 0.804. Fusion rule fixed a priori: robust z (median / IQR of the TRAIN-normal
reference, clipped at ±20) added with unit weight to the base, the same convention as the headline's z-sum; the
A8 study's nonlinear max-fusions (max of robust z, max of train percentiles) are also reported.

## Bottom line

**Neither lever is a paper-ready win. Both are properly controlled nulls against the clean headline.**

- **Lever B (kNN / local density): dead end.** Added to the headline it changes the difficult AUROC by −0.023 to
  −0.015 on WADI (P(diff ≤ 0) 0.98 to 1.00), −0.004 to +0.001 on HAI (P 0.48 to 0.85), −0.007 to −0.004 on SWaT
  (P 0.85 to 0.94); double-hard likewise. The head passes its own invariant (it catches isolated anomalies: Easy
  AUROC 0.95 to 1.00 in feature space) and passes the pre-declared train-normal gate on every dataset, and still
  adds nothing, because the shipped 80-component density head already IS a local-density estimate: Spearman
  correlation between the density head and the k=10 kNN distance on train-normal is 0.90 to 0.93 and the share of
  train windows "covered by the density head yet kNN-isolated" is 0.0 to 0.5%. The over-coverage that A8 found in a
  wide K=2 head is resolved by resolution, and there is nothing left for kNN to add. On WADI the kNN head is worse
  than the density head on the difficult windows (0.589 vs 0.656) and pulls the community score down.
- **Lever A (expected-regime context): future work at best, and the earlier WADI lift does not survive the clean
  pipeline.** Every label-free rule lowers WADI (prev1 −0.043 to −0.059, transition-weighted −0.008 to −0.011) and is
  within noise on SWaT (−0.015 to +0.002). The one positive number, the self-gated rule on HAI (headline +0.012 on
  difficult, P=0.08; +0.024 on double-hard, P=0.004; neutral on SWaT; −0.047 on WADI), is reproduced by a
  regime-free persistence control: a running max of the LatAD score during its own alarm (`hold`) gives
  +0.008 / +0.022, a 3-window rolling max gives +0.020 / +0.027, a 5-window one +0.038 / +0.042, and a rolling max of
  the headline itself +0.029 / +0.026 (P=0.004). Head-to-head, selfgate vs hold is +0.003 (P=0.20) and selfgate vs
  rmax3 is −0.009 (P=0.92). The lift is alarm persistence (temporal smoothing) on HAI's long episodes, not regime
  geometry. The A8 study's WADI difficult 0.73 → 0.80 came from three things that are gone here: a label-using
  expectation rule (last window with y=0), the dirty 43-window subset, and LatAD (0.634) rather than the community
  aggregation (0.795) as the base. On the clean subset even the label-using oracle rule does not lift HC_coh on WADI
  (−0.013 to −0.020, P 0.64 to 0.67), and the transition head that beats LatAD alone (0.668 vs 0.634) still loses
  to HC_coh when fused (0.779 vs 0.795).
- **WADI stays where clean_recompute.md left it: HC_coh-alone 0.795 / 0.696 is the best point estimate**, and no
  head raises it. The per-window dump explains why: the residual WADI frontier (windows 360 to 362) sits below the
  normal median on every head including kNN and context (test-normal percentile 0.00 to 0.21), windows 16 to 21 and
  544 to 547 sit inside their own regime on every axis, and the windows the heads do rank high (17, 202, 203, 494,
  495) are the ones the headline already ranks at the 0.96 to 1.00 percentile.
- **Invariant: no HAI/SWaT margin is lowered by anything recommended**, because nothing is recommended for the
  method. The significant margins (HAI +0.088 vs AE, SWaT +0.055 vs LinRes) are untouched.

## Lever B: kNN / local-density head

Prototype. `d_k(x)` = Euclidean distance from the window to its k-th nearest TRAIN-normal window, in three spaces:
the VaDE latent of the refit seed (10-D WADI, 16-D HAI/SWaT), the standardised (FIX-2-clipped) window-feature space
(732 / 354 / 306 dims), and PCA-20 of the features. Train reference is a temporal leave-out (neighbours within ±2
windows excluded, because 50%-overlapping windows are otherwise their own nearest neighbours and the normal scale is
understated). k = 10 pre-declared as primary; k in {1, 5, 20, 50} reported as sensitivity. Score standardised on
train-normal (median / IQR), fused with unit weight.

Difficult / double-hard AUROC, 5 seeds (mean ± sd):

| score | WADI Diff | WADI DH | HAI Diff | HAI DH | SWaT Diff | SWaT DH |
|---|---|---|---|---|---|---|
| LatAD (global) | 0.634±0.011 | 0.571±0.013 | 0.811±0.016 | 0.806±0.019 | 0.804±0.007 | 0.728±0.010 |
| HC_coh (community) | 0.795±0.022 | 0.696±0.041 | 0.801±0.020 | 0.738±0.022 | 0.822±0.014 | 0.757±0.015 |
| **HCcoh+LatAD (headline)** | 0.771±0.023 | 0.662±0.040 | 0.846±0.019 | 0.814±0.024 | 0.837±0.004 | 0.773±0.006 |
| head: kNN latent k10 | 0.589±0.007 | 0.518±0.011 | 0.784±0.008 | 0.768±0.004 | 0.729±0.002 | 0.645±0.004 |
| head: kNN feature k10 | 0.623 | 0.524 | 0.784 | 0.761 | 0.770 | 0.688 |
| head: kNN PCA-20 k10 | 0.619 | 0.549 | 0.788 | 0.772 | 0.740 | 0.660 |
| HCcoh + kNN latent | 0.762±0.021 | 0.652±0.040 | 0.832±0.012 | 0.797±0.016 | 0.823±0.007 | 0.754±0.008 |
| HCcoh + kNN feature | 0.768±0.022 | 0.654±0.040 | 0.834±0.011 | 0.791±0.015 | 0.826±0.006 | 0.757±0.007 |
| headline + kNN latent | 0.748±0.018 | 0.636±0.031 | 0.844±0.013 | 0.822±0.014 | 0.830±0.006 | 0.762±0.007 |
| headline + kNN feature | 0.752±0.019 | 0.636±0.033 | 0.842±0.011 | 0.814±0.013 | 0.834±0.005 | 0.766±0.007 |
| headline + kNN PCA-20 | 0.756±0.017 | 0.645±0.030 | 0.846±0.013 | 0.824±0.015 | 0.833±0.006 | 0.767±0.007 |
| zmax(headline, kNN latent) | 0.757±0.025 | 0.643±0.043 | 0.833±0.015 | 0.807±0.019 | 0.837±0.005 | 0.773±0.006 |
| pmax(headline, kNN latent) | 0.756±0.024 | 0.642±0.042 | 0.845±0.015 | 0.815±0.019 | 0.795±0.021 | 0.743±0.018 |

Paired episode bootstrap of headline + kNN vs headline (diff, 95% CI, P(diff ≤ 0)): WADI latent −0.023
[−0.04, −0.01] P=1.00 / DH −0.026 P=1.00; feature −0.018 P=0.98 / −0.026 P=0.96; PCA-20 −0.015 P=0.99 / −0.018
P=0.97. HAI latent −0.002 [−0.01, +0.01] P=0.66 / DH +0.007 P=0.15; feature −0.004 P=0.85 / −0.000 P=0.50; PCA-20
+0.001 P=0.48 / +0.010 P=0.11. SWaT latent −0.007 P=0.94 / −0.011 P=0.96; feature −0.004 P=0.85 / −0.007 P=0.90.

k sensitivity (latent, head alone): monotone in k on every dataset, k=1 best (WADI 0.653, HAI 0.780, SWaT 0.822
difficult), k=50 worst (0.544 / 0.777 / 0.670); HCcoh + kNN k=1 reaches 0.774 / 0.838 / 0.849, i.e. still below
the headline on WADI and equal to it on HAI/SWaT. A smaller k moves the head toward the density head; it does not
add information.

Invariant and gate. Head-alone Easy AUROC: feature-space kNN 0.982 (WADI) / 0.962 (HAI) / 0.998 (SWaT), latent kNN
0.769 / 0.952 / 0.996: isolated anomalies are caught, the instrument works. Pre-declared train-normal gate (the
residual head's rule transplanted: q95 of held-out last-20%-of-train / q95 of in-sample temporal-leave-out, ON if
< 1.5): latent 1.05 / 1.50 / 0.38, feature 1.04 / 1.41 / 0.63, PCA-20 1.02 / 1.72 / 0.45. The gate passes on every
dataset except HAI PCA-20 and does not predict where the head helps, because it never helps.

Why it cannot help (train-normal diagnostics, 5 seeds): Spearman(density-head NLL, kNN k10) on train-normal 0.93
(WADI) / 0.91 (HAI) / 0.90 (SWaT); share of train windows below the median density NLL yet above the kNN q90 is
0.001 / 0.000 / 0.005. The 80-component diagonal GMM on the latent is already a local-density estimator; the A8
over-coverage failure (wide K=2 head missing 80% of isolated WADI anomalies, `a8_masking_allsets.md`) is a
resolution effect that the shipped head has already removed (13% miss), and a kNN head is redundant with it. On
HAI, kNN added to HC_coh-alone does give +0.030 to +0.037 (P 0.02 to 0.04) and +0.054 to +0.068 on double-hard
(P < 0.01), which is exactly what the LatAD null tail already contributes to the headline (HC_coh 0.801 → headline
0.846); kNN is a substitute for that term, not an addition to it.

Verdict: **dead end**, with the mechanism explained. Not worth a paper sentence beyond "the density head's resolution
already provides the local-density behaviour; an explicit kNN head is redundant (Spearman 0.90 to 0.93 on
train-normal)".

## Lever A: expected-regime (context) head

Prototype. Regimes = the VaDE's own mixture components (responsibility argmax; diagonal Gaussians; per-component
q99 of its own train members gives a scale-free distance), and as a model-free replicate a PCA-10 / full-covariance
GMM-16 partition (Ledoit-Wolf per-regime Gaussians, as in `a8_masking.py`), one partition per seed. Scores, all
higher = more anomalous, all calibrated on the contiguous train stream with the prev1 rule:
`exp` = distance to the EXPECTED regime / that regime's q99; `trans` = NLL under the mixture re-weighted by the
train transition row of the expected regime; `switch` = 1 if the nearest regime differs from the expected one.
Expectation rules (label-free, deployable): **prev1** = regime of the previous window; **selfgate** = regime of the
last window the detector itself scored normal (LatAD ≤ train q95). Reference only, NOT deployable: **oracle** = regime
of the last window with y = 0 (this is the rule the A8 study used; it reads test labels). Controls: **ctl_rand** =
expected regime drawn from the train regime marginal; **ctl_shuf** = expectation taken from a time-shuffled copy of
the test regime sequence; **switch** alone (the regime-switch base rate as a score).

Difficult / double-hard AUROC of the headline + head (5 seeds), with the head-alone value in parentheses:

| head (rule) | WADI Diff | WADI DH | HAI Diff | HAI DH | SWaT Diff | SWaT DH |
|---|---|---|---|---|---|---|
| headline (no head) | 0.771 | 0.662 | 0.846 | 0.814 | 0.837 | 0.773 |
| VaDE exp prev1 (0.483 / 0.515 / 0.575) | 0.711 | 0.589 | 0.838 | 0.817 | 0.834 | 0.768 |
| VaDE exp selfgate (0.500 / 0.788 / 0.701) | 0.724 | 0.609 | **0.857** | **0.838** | 0.838 | 0.774 |
| VaDE trans prev1 (0.619 / 0.794 / 0.715) | 0.760 | 0.650 | 0.847 | 0.825 | 0.834 | 0.768 |
| obs exp prev1 (0.622 / 0.615 / 0.547) | 0.728 | 0.621 | 0.826 | 0.799 | 0.822 | 0.755 |
| obs exp selfgate (0.619 / 0.660 / 0.626) | 0.738 | 0.641 | 0.831 | 0.810 | 0.828 | 0.763 |
| obs trans prev1 (0.668 / 0.725 / 0.792) | 0.762 | 0.649 | 0.839 | 0.803 | 0.840 | 0.776 |
| control: VaDE exp ctl_rand | 0.660 | 0.553 | 0.818 | 0.788 | 0.832 | 0.767 |
| control: VaDE exp ctl_shuf | 0.657 | 0.559 | 0.808 | 0.776 | 0.833 | 0.767 |
| control: obs exp ctl_shuf | 0.614 | 0.527 | 0.744 | 0.710 | 0.822 | 0.757 |
| control: obs switch prev1 | 0.753 | 0.638 | 0.788 | 0.767 | 0.833 | 0.767 |
| oracle (labels): VaDE exp | 0.777 | 0.675 | 0.892 | 0.872 | 0.853 | 0.795 |
| oracle (labels): obs exp | 0.772 | 0.691 | 0.846 | 0.831 | 0.833 | 0.767 |
| HC_coh-alone (WADI reference) | 0.795 | 0.696 | 0.801 | 0.738 | 0.822 | 0.757 |
| HC_coh + best label-free head (VaDE/obs trans) | 0.779 | 0.675 | 0.836 | 0.803 | 0.836 | 0.773 |

Paired bootstrap vs the headline (diff, 95% CI, P(diff ≤ 0)), label-free rules:

| head | WADI Diff | WADI DH | HAI Diff | HAI DH | SWaT Diff | SWaT DH |
|---|---|---|---|---|---|---|
| VaDE exp prev1 | −0.059 [−0.11,−0.02] 1.00 | −0.073 1.00 | −0.007 [−0.02,+0.01] 0.80 | +0.002 0.39 | −0.003 0.76 | −0.005 0.76 |
| VaDE exp selfgate | −0.047 [−0.11,+0.01] 0.95 | −0.053 0.92 | **+0.012 [−0.00,+0.03] 0.08** | **+0.024 [+0.01,+0.04] 0.00** | +0.001 0.43 | +0.001 0.43 |
| VaDE trans prev1 | −0.011 [−0.02,−0.01] 1.00 | −0.012 1.00 | +0.002 0.38 | +0.011 [+0.00,+0.02] 0.02 | −0.003 0.76 | −0.005 0.79 |
| obs exp prev1 | −0.043 0.99 | −0.041 0.94 | −0.020 1.00 | −0.015 0.97 | −0.015 1.00 | −0.018 1.00 |
| obs exp selfgate | −0.033 0.84 | −0.021 0.69 | −0.014 0.98 | −0.004 0.69 | −0.009 0.96 | −0.010 0.94 |
| obs trans prev1 | −0.008 0.83 | −0.013 0.88 | −0.006 0.86 | −0.011 0.93 | +0.002 [−0.01,+0.01] 0.29 | +0.003 0.33 |
| kNN + VaDE exp prev1 (combined) | −0.065 1.00 | −0.076 1.00 | −0.005 0.77 | +0.010 0.09 | −0.009 0.96 | −0.013 0.97 |

Controls read correctly. The shuffled and random-regime expectations are strictly worse than the real ones on every
dataset (WADI headline + obs ctl_shuf 0.614 vs + obs exp prev1 0.728; HAI 0.744 vs 0.826), and the switch
indicator alone is at chance (0.49 to 0.54), so the prev1 context scores do carry temporal information; they are
simply worse than what the community score already has, and fusing them costs AUROC. The A8 study's "expected-regime
score beats the nearest-regime score on the windows LatAD misses" is consistent with this: head-alone, the
observation-space transition score (0.668) and the oracle context (0.693) do beat the single-latent LatAD (0.634)
on WADI's clean difficult windows, but neither beats HC_coh (0.795), and unit-weight fusion with HC_coh lands at
0.779 to 0.785 (all P ≥ 0.85 for a lift). The A8 finding was real relative to LatAD and is moot relative to the
community aggregation that now carries WADI.

### The one positive: self-gated context on HAI is alarm persistence

The selfgate rule freezes the expectation while the detector's own alarm is on. On HAI the expectation differs from
prev1 on 54% of difficult windows (11% of normals); on SWaT 47% / 12%; on WADI 9% / 4%. Persistence controls
(`a8_method_leverage_ctl.py`), same fusion, no regime geometry, vs the headline:

| control (headline + …) | HAI Diff | HAI DH | SWaT Diff | SWaT DH | WADI Diff | WADI DH |
|---|---|---|---|---|---|---|
| selfgate context (the head) | +0.012 P=0.09 | +0.024 P=0.00 | +0.001 P=0.43 | +0.001 P=0.45 | −0.047 P=0.98 | −0.053 P=0.91 |
| hold (running max of LatAD z during its own alarm) | +0.008 P=0.25 | +0.022 P=0.06 | −0.000 P=0.54 | +0.000 P=0.46 | −0.006 P=0.94 | −0.005 P=0.81 |
| rmax2 (rolling max of LatAD z, 2 windows) | +0.013 P=0.08 | +0.021 P=0.01 | +0.003 P=0.23 | +0.005 P=0.24 | −0.005 P=0.84 | −0.003 P=0.69 |
| rmax3 | +0.020 P=0.07 | +0.027 P=0.01 | +0.002 P=0.39 | +0.004 P=0.34 | −0.006 P=0.87 | −0.004 P=0.72 |
| rmax5 | +0.038 P=0.05 | +0.042 P=0.03 | +0.001 P=0.53 | +0.003 P=0.52 | −0.009 P=0.93 | −0.008 P=0.90 |
| ewma (alpha 0.5) | +0.014 P=0.11 | +0.022 P=0.02 | +0.005 P=0.14 | +0.006 P=0.14 | −0.008 P=0.98 | −0.009 P=0.94 |
| rolling max of the headline itself (3 windows) | +0.029 P=0.00 | +0.026 P=0.01 | −0.003 P=0.61 | +0.005 P=0.45 | +0.022 P=0.07 | +0.028 P=0.04 |

Head-to-head: selfgate vs hold +0.003 [−0.004, +0.013] P=0.20 (difficult), +0.002 P=0.40 (double-hard); selfgate vs
rmax3 −0.009 [−0.02, +0.004] P=0.92, −0.003 P=0.66. Per episode (26 HAI difficult episodes), selfgate lifts 11 and
lowers 9; the two largest lifts (episodes at windows 3059 and 8457, +0.067 and +0.021 in test-normal percentile) are
matched or exceeded by the hold control (+0.076, +0.065). The context head carries no information beyond "an alarm
that was on a window ago is probably still on". That is temporal smoothing, a generic post-processing step; it
helps on HAI, is neutral on SWaT and costs 0.005 to 0.009 on WADI (difficult-window counts per episode have a median
of 4 on both HAI and WADI, so episode length does not explain the difference; HAI's drifting test normals, on which
LatAD fires 19 to 20% of the time at its train q95, are the likelier reason a held alarm is rewarded there). It does
not belong in the method as a regime-context contribution; if temporal
smoothing were ever added it should be added to every method in the comparison, and it is not part of this study's
recommendation.

### Gate

The pre-declared gate (residual rule, ratio < 1.5) for the context heads: VaDE exp 1.00 / 1.00 / 1.07, VaDE trans
1.05 / 1.05 / 0.48, obs exp 0.72 / 1.22 / 1.56, obs trans 1.06 / 1.43 / 0.83 (WADI / HAI / SWaT). It is ON almost
everywhere and OFF only for obs exp on SWaT; it does not separate WADI (hurt) from HAI (persistence lift) from SWaT
(neutral). No other train-normal statistic measured here does either: normal switch rate 0.42 / 0.28 / 0.55 (VaDE
regimes), share of train windows beyond their own regime's q99 comparable across datasets. The only statistic that
lines up with "does not hurt" is the shipped residual gate itself (WADI 3.41 OFF, HAI 1.22 ON, SWaT 0.76 ON); with
three datasets that is a coincidence to note, not a gate to adopt, and since the head is not recommended it is moot.

Verdict: **future work at most.** A context head is only worth revisiting if (i) the base is a single-latent
detector rather than the community aggregation, (ii) the expectation rule is label-free and its lift is measured
against a persistence control of equal memory, and (iii) the dataset has long episodes and rare legal switches.
On WADI, the dataset this lever was meant for, it is a controlled negative (−0.008 to −0.059, all label-free rules,
all partitions, including the oracle).

## Invariants checked

- I-A (no test data in the method): every head is calibrated on train-normal only (temporal leave-out kNN, prev1
  context on the contiguous train stream, robust z from train); the only label-using score is the oracle, which is
  labelled as a reference and recommended for nothing. Subsets and bases are the clean_recompute definitions.
- I-B (instrument works): kNN head Easy AUROC 0.96 to 1.00 in feature space, 0.77 (WADI) to 1.00 in the latent;
  context heads fused into the headline sit 0.03 to 0.11 AUROC above their shuffle / random controls on WADI/HAI.
- I-C (shuffle / base-rate control): every context lift over its shuffled control is real, but no context head lifts
  the headline except selfgate on HAI, and that one is reproduced by regime-free persistence.
- I-D (HAI/SWaT margins): unchanged, because no head is added. The largest label-free effect on either is +0.012
  (HAI, selfgate), which is persistence, and the largest loss is −0.020 (HAI, obs exp prev1), which is not adopted.
- I-E (refit fidelity): SWaT refit is identical to the stored scores (corr 1.000), HAI 0.98 to 0.99, WADI 0.98,
  difficult AUROC of the refit within 0.007 of the stored value on every dataset.

## Ranked recommendation

1. **Do not add either head to the method.** Report, if anything, one sentence in the A8 discussion: the density
   head's 80-component resolution already provides local-density behaviour (Spearman 0.90 to 0.93 with a kNN
   distance on train-normal; an explicit kNN head changes the difficult AUROC by −0.023 to +0.001), and a
   train-normal-calibrated, label-free expected-regime head does not lift the community score on any testbed
   (−0.059 to +0.002), the one apparent HAI lift being temporal persistence.
2. **WADI headline stays HC_coh-alone 0.795 / 0.696** (clean_recompute.md recommendation); nothing here makes WADI a
   stronger lead. The residual frontier (360 to 362, 16 to 21, 544 to 547) is inside its regime and below the normal
   median on every axis tried, including local density and regime context.
3. **Future-work sentence for A10 (optional):** "a context-conditioned regime score beats the single-latent
   detector on WADI's difficult windows (transition-weighted 0.668 vs 0.634, label-free) but not the community
   aggregation; separating regime context from alarm persistence requires a persistence-matched control".
4. **Correction to the A8 deep study's headline number:** the WADI difficult 0.73 → 0.80 fusion used the
   last-normal-window rule (reads test labels), the 43-window subset, and LatAD as base. Under label-free rules on
   the clean 30-window subset the same fusion is 0.626 to 0.660 with LatAD (vs 0.634) and 0.728 to 0.762 with the
   headline (vs 0.771). That number should not be quoted as a method lead.

## Caveats

- n: WADI 30 difficult windows in 8 episodes (double-hard 19 in 7); WADI CIs are ±0.05 to ±0.1 wide and every WADI
  statement is a point estimate. HAI (167 / 26 episodes) and SWaT (85 / 23) are the datasets where the null is
  tight.
- Partition dependence: one PCA-10 / GMM-16 partition per seed (5 partitions), not the 18-partition sweep of the A8
  study; the transition head's sd across partitions is 0.01 to 0.03. K=24 was not run; the A8 study's best partition
  (K=24, "previous window" rule, 0.87 fused) was against LatAD on the dirty subset and is superseded by the clean
  comparison above.
- Selfgate threshold fixed at the train q95 of LatAD; other thresholds were not run (the per-seed npz stores scores,
  not the regime distances, so a threshold sweep needs a refit).
- Unit-weight fusion, fixed a priori. A tuned weight is test-set tuning and was not done; the max-fusions cover
  the nonlinear alternative and give the same picture (WADI 0.755 to 0.787 vs HC_coh 0.795; HAI/SWaT at or below
  the headline).
- The heads pair the refit's latent with the STORED LatAD / community scores of the same seed index; the refit's own
  LatAD is within 0.007 difficult AUROC and 0.98 to 1.00 correlation of the stored one, so the pairing is sound.
- Agent execution: about 2.5 hours of tool calls; local CPU about 30 minutes of fits (three datasets in parallel,
  HAI 2 min per seed) plus about 90 minutes of bootstrap evaluation (WADI/SWaT and HAI in parallel); no cloud cost.
