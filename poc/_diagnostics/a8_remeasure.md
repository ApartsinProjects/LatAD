# A8 re-measurement: is between-regime overlap absent, or only invisible to the VaDE-responsibility metric?

Scripts: `a8_remeasure.py` (synthetic invariant test, observation-space measures, transition inspection,
sub-regime probe; results in `a8_remeasure.jsonl`, one line per item) and `a8_remeasure_transit.py`
(Cranfield "improbable under both neighbouring setpoints" test; appended to the same jsonl). Per-dataset
logs `a8_remeasure.<name>.log`. Paper and shipped model untouched. Single VaDE seed (0) throughout, same
recipe as the A8 screens (`train_vade`, K=16, latent 8, 40 epochs, warm-up 8, "stats" window features).

## Verdict in three lines

1. **The metric is invalid, in both directions.** VaDE responsibility entropy does not track observation-space overlap: on a synthetic pair of regimes whose Bayes-posterior entropy runs from 0.84 (30% Bayes error) to 0.09 (2% error), VaDE K=16 entropy stays flat at 0.35, 0.31, 0.32, 0.28. Max-responsibility is also blind to between-mode pockets by construction (softmax normalisation): Cranfield window 2977 sits at density percentile 0.000 of steady operation and gets responsibility 1.000.
2. **"Heavy mode overlap" (A8 as the paper defines it) is nevertheless genuinely absent**, now under label-free observation-space measures on all five datasets and under physical labels on Cranfield: the 53 setpoint segments are 41 Mahalanobis units apart at window level (106 at 1 Hz sample level), pairwise LDA error at most 3.3%, overlap coefficient at most 0.016, deep density valleys between every pair.
3. **What IS present, and what the responsibility screens could not see, is sparse between-regime transit.** 18% of all Cranfield train-normal windows (600 of 728 windows inside setpoint steps, 82%) lie beyond the 99% Mahalanobis contour of BOTH the setpoint before and the setpoint after the step, while VaDE assigns them with mean max-responsibility 0.92. On WADI/HAI/SWaT, 6 to 10% of the strongest within-window-trend windows fall below the 1st percentile of steady-state density. These are normal windows improbable under every mode; they are the false-positive side of the original A8 row ("a between-regime point... every channel in range"), and the paper's own nearest-mode NLL already scores them in the top 20% (Cranfield: percentile 0.79 on average, 0.99 or above on the examples below).

So: A8-as-overlap is absent and the paper's conclusion survives, but the evidence in Appendix C does not establish it and must be replaced; A8-as-transit is present at the 5 to 18% level and must be stated. Details follow.

## 1. Metric-bias diagnosis (code and numbers)

### 1a. What `_responsibilities` computes

`models_vade.py:197-203`: `gamma = softmax_k( log pi_k + log N(z | mu_k, var_k) )` at the encoder mean `z`. The softmax divides out the absolute density: a window at Mahalanobis distance 25 from its nearest component and 30 from every other gets `gamma_max ~ 1`, exactly like a window at distance 0.5. Mean max-responsibility, normalised entropy and rho (`max < 0.5`) therefore measure only whether a window is equidistant from two components, never whether it is far from all of them. That is the pocket geometry A8 was meant to capture. Concrete: Cranfield window 2977 (T3, t0 = 9600 s, ch12 jumping 26.1 -> 153.4 inside the window) has VaDE top-2 responsibilities (1.000, 0.000), observation-space GMM top-2 (1.000, 0.000), observation log-density percentile 0.000 among steady windows, nearest-neighbour distance percentile 1.000, VaDE nearest-mode NLL percentile 1.000. The responsibility metric reports it as the crispest possible assignment.

### 1b. Does the ELBO push responsibilities to be crisp?

`loss()` (`models_vade.py:106-133`): `term_a` is the gamma-weighted expected NLL of z under each component, `term_b = sum gamma (log gamma - log pi)`, `term_c` the encoder-entropy term. With gamma set to the posterior, `term_a + term_b` equals minus the mixture log-density of z (Jensen with equality), so the objective rewards the encoder for placing z where the mixture is dense, which is at component centres; nothing rewards placing a window between two components. Over training the encoder is free (it is a nonlinear map) to move a window that is ambiguous in x to the centre of whichever component claims it. This is the qualitative mechanism; the quantitative test is the synthetic experiment in 1d.

### 1c. The logvar floor does NOT cause crispness

`logvar_floor = log(0.05)` (`models_vade.py:81`) is a lower bound on component variance, so floored components are wider, not narrower, than the fit wants; it inflates entropy (the SKAB artefact, `fable_a3_floor_resolution.py`) and cannot produce false crispness. 5.5% (WADI), 6.2% (CAN-bus), 7.0% (HAI), 8.6% (Cranfield), 16.4% (SWaT) of components sit at the floor in these fits; component sd (0.60 to 0.97) is smaller than latent sd (0.84 to 1.81), so the fitted components are tight relative to the latent spread. The user's suspicion that the floor biases the entropy downward is not supported; the bias is in the softmax normalisation and in the encoder's freedom to separate.

### 1d. Synthetic invariant test (`synthetic_two_regime`, `synthetic_continuum` rows)

Invariant stated in advance: a valid overlap metric applied to two Gaussian regimes must reproduce the Bayes posterior entropy, which is known exactly, and must decrease monotonically as the separation Delta grows.

Case B, realistic geometry: 3 latent factors -> 48 features (rank-3 plus 0.1 noise, like window statistics), 2500 windows per regime, regimes shifted by Delta in factor 1 (unit within-regime sd).

| Delta | Bayes: maxresp / H / error | obs-GMM K=2 (PCA-free, full cov): H / ARI | VaDE K=2: H / ARI | VaDE K=8: H | VaDE K=16: H | VaDE K=16 merged by true regime: H | label-free obs measures on true labels: OVL / valley / LDA-CV error |
|---|---|---|---|---|---|---|---|
| 1 | 0.69 / 0.84 / 0.30 | 0.50 / 0.07 | 0.73 / 0.00 | 0.43 | 0.35 | 0.39 | 0.61 / 1.00 / 0.302 |
| 2 | 0.84 / 0.51 / 0.16 | 0.40 / 0.45 | 0.78 / 0.10 | 0.34 | 0.31 | 0.33 | 0.32 / 1.00 / 0.158 |
| 3 | 0.93 / 0.24 / 0.06 | 0.21 / 0.75 | 0.42 / 0.70 | 0.40 | 0.32 | 0.24 | 0.13 / 0.70 / 0.061 |
| 4 | 0.98 / 0.09 / 0.02 | 0.07 / 0.91 | 0.28 / 0.84 | 0.36 | 0.28 | 0.11 | 0.05 / 0.29 / 0.024 |

- VaDE K=8/16 entropy is flat (0.28 to 0.43) while the true posterior entropy changes 10-fold. At Delta = 1 the truth is 0.84 and VaDE reports 0.35 (or 0.39 after merging its components by true regime); at Delta = 4 the truth is 0.09 and VaDE reports 0.28. The metric neither ranks nor scales with overlap. 62.5% of components sat at the floor in every Case-B fit, so the floor was active and still did not make the heavy-overlap cases look overlapping.
- The observation-space GMM (any fitted mixture) also understates entropy when overlap is heavy (0.50 vs 0.84 at Delta = 1, ARI 0.07: EM finds a sharper partition than the generating one). Fitted-model responsibilities are therefore not the fix; label-free geometry is.
- The label-free measures track the truth: LDA cross-validated error equals the Bayes error to three decimals at every Delta (0.302/0.301, 0.158/0.157, 0.061/0.062, 0.024/0.024); OVL is 2x the Bayes error as theory says; the valley ratio (minimum mixture density between the two projected means divided by the lower peak) is 1.0 (unimodal) at Delta = 1, 2 and drops to 0.29 at Delta = 4.

Case A (isotropic 10-D noise, shift in one dimension) is a control: the recipe does not find the structure at all (VaDE ARI 0.02 to 0.18 even at Delta = 4 where a GMM reaches 0.92; latent sd collapses to 0.5 with components wider than that) and reports entropy 0.63 to 0.97 regardless of Delta. Same conclusion from the other side: the entropy reports the fit state of the VaDE, not the data.

Case C (a continuum: one uniform factor plus two nuisance factors through the same 48-feature map, no regimes at all): VaDE K=8/16 max-resp 0.72/0.68, entropy 0.33/0.31, rho 0.11/0.19. A continuum with zero modes gets entropy 0.31, which is 2 to 40 times higher than any of the five real datasets (0.007 to 0.157). Observation-space: nearest-pair valley 0.95 to 0.97 (no dips, as expected), nearest-pair OVL 0.09, LDA error 0.04 to 0.05, 7 to 10% of points have their nearest neighbour in another cluster. These continuum values are the reference "slices of one ridge" signature used below.

## 2. Observation-space overlap measures on the real windows

All in the PCA-10 projection of the standardised "stats" window features (same windows as the A8 screens; PCA-10 keeps the geometry the mixture heads see). Pairwise measures are between each cluster and its nearest partner (highest Bhattacharyya coefficient); clusters come from a full-covariance K=16 GMM in observation space (the `pairwise_obsgmm` rows; the VaDE-label and k-means-8 variants are in the jsonl and tell the same story). Cross-NN excludes temporally adjacent windows (|Delta index| <= 2 in the same stream).

| Dataset (windows) | VaDE K16 maxresp / H (old metric) | obs-GMM K16 maxresp / H | silhouette: k-means8 / GMM16 / VaDE16 labels | nearest-pair OVL mean (max) | valley ratio mean / share >= 0.5 | LDA-CV error mean (max) | NN1 in another cluster | closer to another cluster than own |
|---|---|---|---|---|---|---|---|---|
| Cranfield (3336) | 0.953 / 0.047 | 0.988 / 0.012 | 0.26 / 0.21 / 0.16 | 0.09 (0.27) | 0.49 / 55% | 0.11 (0.21) | 6.9% | 12.7% |
| CAN-bus (2581) | 0.840 / 0.157 | 0.887 / 0.112 | 0.20 / 0.04 / 0.10 | 0.13 (0.20) | 0.88 / 92% | 0.08 (0.14) | 26.0% | 26.1% |
| HAI (18359) | 0.842 / 0.155 | 0.969 / 0.031 | 0.18 / 0.12 / 0.11 | 0.09 (0.26) | 0.57 / 64% | 0.08 (0.25) | 9.0% | 10.9% |
| SWaT (3623) | 0.993 / 0.007 | 0.999 / 0.001 | 0.46 / 0.45 / 0.33 | 0.02 (0.05) | 0.13 / 0% | 0.04 (0.10) | 1.1% | 3.8% |
| WADI (2614) | 0.938 / 0.059 | 0.979 / 0.022 | 0.27 / 0.14 / 0.06 | 0.09 (0.27) | 0.66 / 69% | 0.08 (0.17) | 13.7% | 14.8% |
| reference: synthetic continuum (VaDE16 labels) | 0.683 / 0.305 | 0.676 / 0.318 | 0.13 | 0.09 (0.19) | 0.95 / 100% | 0.05 (0.13) | 10.3% | 12.0% |
| reference: two regimes, Delta = 1 (true labels) | (see 1d) | | 0.07 | 0.61 | 1.00 | 0.30 | | |

Reading:
- Observation-space GMM responsibilities are even crisper than VaDE's on every dataset, so the "VaDE latent hides overlap that observation space would show via responsibilities" version of the hypothesis is false. Responsibilities are crisp in any space; that is a property of mixture posteriors in 10 dimensions and of EM's preference for sharp partitions, not evidence about the data.
- The label-free geometry says: nearest discovered clusters on CAN-bus, WADI and HAI have no density valley between them (valley ratio ~1 for 64 to 92% of nearest pairs, same as the continuum), but their overlap coefficient is small (mean 0.09 to 0.13, max 0.27) and a linear classifier separates each nearest pair with 8% mean error. That is the signature of adjacent slices of one elongated mode, not of two overlapping modes. A genuinely overlapping pair (Delta = 1 reference) would show OVL 0.6 and LDA error 0.3; nothing in any dataset comes within a factor of two of that (worst pair anywhere: HAI GMM clusters 5/12, OVL 0.26, LDA error 0.25; WADI VaDE clusters 2/9, LDA error 0.38 with OVL 0.20).
- SWaT is crisp by every measure (valleys everywhere, 1% cross-cluster neighbours). CAN-bus is the least separated (26% of windows have their nearest non-adjacent neighbour in another cluster), consistent with driving being a blend; but even there the nearest pairs are linearly separable at 8% error.
- The silhouettes reproduce the paper's Section 3 numbers (VaDE-label silhouette 0.06 on WADI, 0.11 on HAI, 0.33 on SWaT vs the paper's 0.06 / 0.08 / 0.29). Silhouette 0.06 means the K-partition slices a continuous ridge, not that two physical regimes coincide; see Section 6 on the resulting wording conflict inside the paper.

### 2a. Cranfield with physical labels (the decisive test; `cranfield_true_labels`, `cranfield_sample_level` rows)

| Labels | level | n classes | silhouette | nearest-pair BC max | OVL max | valley max | LDA-CV error mean (max) | Mahalanobis separation of nearest pairs (mean) | NN1 in another class |
|---|---|---|---|---|---|---|---|---|---|
| file T1/T2/T3 | windows | 3 | 0.13 | 0.029 | 0.010 | 0.04 | 0.043 (0.086) | 7.3 | 2.6% |
| setpoint segment | windows | 47 (>= 15 windows) | 0.24 | 0.005 | 0.016 | 0.09 | 0.003 (0.033) | 41.5 | 5.4% |
| file | 1 Hz samples (8000) | 3 | 0.16 | 0.000 | 0.000 | 0.00 | 0.000 (0.000) | 19.6 | |
| setpoint segment | 1 Hz samples | 50 | 0.19 | 0.008 | 0.030 | 0.08 | 0.001 (0.016) | 105.7 | |

Setpoint segments were detected from the step channels (7, 12, 13: the flow-rate setpoints; a 60 s backward-vs-forward mean shift > 0.5 sd, transition zone dilated +-30 s), 53 segments over the three files. Two flags: channel 23 is constant within each file at a different value per file (0.15 / 0.08 / 0.13), so the file-level separation is partly a labelling artefact; the setpoint-level numbers are within-file and unaffected. Sample-level separation (106) is larger than window-level (41), so window averaging is not hiding overlap; the raw process points are further apart than the window statistics.

Conclusion for A8-as-overlap: the physical operating points of a real multiphase rig are separated by tens of within-regime standard deviations. Nothing overlaps. "No A8" is true on Cranfield by direct measurement, with labels, at both time scales.

## 3. Transition windows: they exist, they are improbable under every mode, and responsibility cannot see them

`transition_summary` rows. "steady" = all other windows; density is the full-covariance K=16 GMM on PCA-10 (fit on all windows; for the between-setpoint test, on steady windows only).

| Dataset, transition definition | share of windows | VaDE maxresp: trans vs steady | VaDE rho (trans) | obs density percentile of trans within steady | share of trans below steady p05 / p01 | VaDE nearest-mode NLL percentile of trans |
|---|---|---|---|---|---|---|
| Cranfield, window overlaps a setpoint step | 25.6% | 0.914 vs 0.966 | 1.1% | 0.20 | 47% / 19% | 0.79 |
| CAN-bus, speed swing > 20 km/h in 24 s | 50.2% | 0.827 vs 0.852 | 9.0% | 0.38 | 8% / 2% | 0.70 |
| CAN-bus, net ramp > 15 km/h | 41.9% | 0.842 vs 0.838 | 7.6% | 0.41 | 9% / 3% | 0.66 |
| HAI, obs-GMM assignment boundary | 23.4% | 0.816 vs 0.850 | 5.8% | 0.38 | 11% / 3% | 0.51 |
| HAI, top-5% within-window trend | 5.0% | 0.828 vs 0.843 | 3.7% | 0.46 | 13% / 7% | 0.54 |
| SWaT, assignment boundary | 81.2% (cyclic switching, not transitions) | 0.994 vs 0.992 | 0.0% | 0.31 | 8% / 2% | 0.52 |
| SWaT, top-5% trend | 4.8% | 0.985 vs 0.994 | 0.0% | 0.24 | 27% / 10% | 0.91 |
| WADI, assignment boundary | 49.2% | 0.934 vs 0.942 | 2.0% | 0.31 | 22% / 6% | 0.64 |
| WADI, top-5% trend | 4.8% | 0.918 vs 0.939 | 1.6% | 0.28 | 21% / 6% | 0.75 |

The pattern is the same everywhere: transition windows barely move the responsibility metric (max-resp drops by 0 to 0.05, rho stays in single digits) while their observation-space density and the VaDE's own nearest-mode NLL flag them as tail events. The responsibility screens were reading the wrong quantity.

### 3a. Cranfield: improbable under BOTH neighbouring setpoints (`cranfield_between_setpoint_test`, `cranfield_step`)

For each of the 40 steps with a steady segment on both sides (>= 12 windows each), a Gaussian was fitted (PCA-10) to the preceding and following steady segments, and each transition window's Mahalanobis distance to both was compared with that segment's own 99th percentile.

- 600 of 728 transition windows (82%) are beyond the 99% contour of both neighbouring setpoints; every one of the 40 steps contributes at least one. That is 18.0% of ALL 3336 Cranfield train-normal windows.
- Control: 3.0% of steady windows are beyond their own segment's q99 (above the nominal 1% because q99 is in-sample on 12 to 100 windows in 10-D; the transition rate is 27x the control).
- VaDE mean max-responsibility on those 600 windows: 0.920 (0.93 over all transition windows). Median separation between the two setpoints of a step: 28.7 Mahalanobis units (range 12 to 126).

Window-by-window course of the step at T1 t0 = 410 to 520 s (`cranfield_step_course`, first entry; ch12 / ch13 are the (first, last) raw values in the window; d_prev, d_next are Mahalanobis distances divided by that setpoint's q99, so > 1 means outside the setpoint's 99% envelope):

| window idx | t0 (s) | VaDE component / max-resp | obs density percentile | VaDE NLL percentile | d_prev / q99 | d_next / q99 |
|---|---|---|---|---|---|---|
| 40 | 400 | 7 / 1.000 | 0.40 | 0.54 | 1.13 | 2.47 |
| 41 | 410 | 13 / 0.999 | 0.000 | 0.996 | 13.5 | 2.9 |
| 42 | 420 | 13 / 0.983 | 0.000 | 0.998 | 31.8 | 4.4 |
| 44 | 440 | 13 / 1.000 | 0.000 | 0.994 | 13.4 | 2.6 |
| 47 | 470 | 13 / 0.982 | 0.000 | 0.997 | 11.4 | 1.9 |
| 48 | 480 | 15 / 0.745 | 0.000 | 0.988 | 13.2 | 1.4 |
| 50 | 500 | 15 / 0.521 | 0.000 | 0.994 | 16.2 | 1.2 |
| 52 | 520 | 15 / 0.867 | 0.017 | 0.961 | 21.2 | 0.8 |
| 54 | 540 | 15 / 0.639 | 0.28 | 0.92 | 25.4 | 0.4 |

For 70 s the rig is 11 to 32 q99-units from the setpoint it left and 1.2 to 4.4 from the one it is approaching, i.e. outside both envelopes; VaDE responsibility is 0.98 to 1.00 on a component (13) that recurs as the "in transit" component at other steps too (windows 179 to 186 and 230 to 231 in the other courses), and only becomes ambiguous (0.52 to 0.75) once the window is already inside the destination envelope. Responsibility ambiguity marks the arrival, not the transit. The density percentile marks the transit exactly.

Other concrete examples (`transition_example` rows): Cranfield 3174 (T3, t0 = 11570, ch12 26.8 -> 110.1, ch13 992.5 -> 952.7): VaDE 1.000, density percentile 0.001. Cranfield 3213 (T3, t0 = 11960, ch12 110.9 -> 39.3, ch13 299.7 -> 369.4): VaDE 0.998, density percentile 0.001. Counter-examples chosen at random inside the dilated transition zone but before the step actually moves (1298, 1529: ch12/ch13 flat) sit at density percentiles 0.51 and 0.66 with VaDE 0.995 to 1.000, so the +-30 s dilation over-counts transitions; the 82% figure above is therefore a lower bound on the share of true in-transit windows that are improbable under both.

### 3b. CAN-bus: ramps are the regimes, not the gaps between them

Half of all windows contain a > 20 km/h speed swing and 42% a net ramp > 15 km/h, so a K=16 mixture allocates components to accelerating/decelerating windows (the window features include trend and range). Transition windows are consequently not tail events (density percentile 0.38 to 0.41; only 2 to 3% below p01). The ones that are tail events are rare manoeuvres: idx 1279 (trip s17, launch from standstill, speed 0,0,0,32,35,49, rpm 854 -> 2020): VaDE 1.000, density percentile 0.000; idx 902 (s14, hard stop 50,63,44,17,3,0): VaDE 0.655/0.301 (this one IS ambiguous), density 0.000; idx 605 (s12, 44,37,31,22,25,40 with rpm 1911 -> 790 -> 2277, a stop-and-go): VaDE 0.381/0.342, density percentile 0.15. CAN-bus is the one dataset where VaDE ambiguity (rho 7.8%) roughly co-locates with observation-space ambiguity (26% cross-cluster neighbours), and it is also the dataset with the shallowest valleys (92% of nearest pairs have none). It is a blended system, but blended along a continuum, not by two coincident modes.

### 3c. HAI / SWaT / WADI

No physical labels, so transitions were defined as assignment boundaries of the observation-space GMM and as the top-5% within-window trend. On HAI the boundary examples 10319 / 18239 / 7319 (density percentiles 0.000, VaDE 1.000 on component 9) are of the same kind as Cranfield's; the WADI top-trend windows 691 / 1820 / 466 (density percentile <= 0.002, VaDE 0.84 to 1.00) likewise. SWaT's 81% boundary share is window-to-window switching between components (tank fill/drain cycling at the 60-step window), not transit; its true transit windows are the top-5% trend set, 27% of which sit below the steady p05 and which VaDE's own NLL puts at percentile 0.91.

## 4. Sub-regime (nested) structure (`subregime` rows; Ward linkage on PCA-10, 4000-window subsample)

| Dataset | Ward silhouette at K = 2 / 3 / 4 / 8 / 12 / 16 / 24 | child-16 pairs inside the same Ward-4 parent: BC mean (max) | across parents: BC mean (max) | within-parent nearest-pair valley mean / share >= 0.5 | within-parent LDA error | labels |
|---|---|---|---|---|---|---|
| Cranfield | 0.25 / 0.17 / 0.19 / 0.22 / 0.26 / 0.25 / 0.26 | 0.003 (0.026) | 0.003 (0.046) | 0.19 / 0% | 0.024 | ARI(Ward-16, setpoints) 0.21; ARI(VaDE-16, setpoints) 0.25; VaDE-16 purity w.r.t. 53 setpoints 0.43 |
| CAN-bus | 0.29 / 0.16 / 0.16 / 0.15 / 0.13 / 0.12 / 0.10 | 0.092 (0.281) | 0.030 (0.319) | 0.77 / 88% | 0.064 | |
| HAI | 0.29 / 0.29 / 0.20 / 0.17 / 0.17 / 0.16 / 0.16 | 0.027 (0.125) | 0.003 (0.107) | 0.33 / 0% | 0.039 | |
| SWaT | 0.88 / 0.32 / 0.35 / 0.45 / 0.45 / 0.50 / 0.52 | 0.000 (0.001) | 0.000 (0.000) | 0.02 / 0% | 0.003 | |
| WADI | 0.17 / 0.21 / 0.24 / 0.22 / 0.22 / 0.22 / 0.22 | 0.013 (0.096) | 0.005 (0.091) | 0.43 / 47% | 0.039 | |

- Nesting is real: on Cranfield the silhouette rises again at K = 12 after the K = 3 minimum, VaDE-16 clusters are unions of 2 to 4 setpoints each (purity 0.43 against 53 setpoints), and the setpoint substructure inside a VaDE regime is invisible to responsibility entropy at K = 16 because the union is treated as one component. On HAI, WADI and CAN-bus, children inside a parent overlap 3 to 10 times more (Bhattacharyya) than children across parents, which is the hierarchical signature.
- But the sub-modes are themselves separated: within-parent LDA error is 2 to 6%, within-parent OVL 0.02 to 0.11, and only on CAN-bus (and half of WADI) are the within-parent valleys absent. So "regimes are unions of sub-regimes" holds; "sub-regimes overlap heavily" does not, except in the continuum sense on CAN-bus.

## 5. Reconciling with physical intuition

Why controlled CPS telemetry gives crisp regimes at the window level, stated so the paper can say it plainly:

1. **Setpoint control makes operating points tight relative to their spacing.** Cranfield's flow-rate setpoints differ by 12 to 126 Mahalanobis units at window level (28.7 median between consecutive setpoints), because within-setpoint variation is instrument noise around a regulated value while the grid spacing is a design choice. Overlap would require setpoints closer than a few noise sd; the rig's grid is 10 to 100 times coarser.
2. **Transitions are brief and traverse empty space.** A detected step run spans 11 to 41 windows (about 2 to 7 minutes at 10 s stride, including the +-30 s dilation) between segments lasting many minutes; 25% of Cranfield windows touch a step, but they pass through a corridor of near-zero density rather than through a region where two setpoint densities overlap. Hence "between-regime windows" exist (18% of all windows beyond both envelopes) without any "mode overlap".
3. **Window averaging does not delete the transit; it dilutes it into a distinct window.** Sample-level separation (106) exceeds window-level (41), and a window straddling a step becomes a point with a large trend/range feature that lies off both setpoint manifolds. The averaging hypothesis is not supported.
4. **Where blending is continuous (driving), the mixture absorbs it as extra components** (accelerating, decelerating, stop-and-go), giving the many-shallow-valley, low-silhouette geometry of a sliced continuum. This is the paper's "implicit MIIM" (silhouette 0.06 to 0.15, BIC still improving at K = 25), and it is exactly what the continuum reference reproduces (silhouette 0.13, valley 0.95).

Speculation, flagged as such: the residual difference between CAN-bus (26% cross-cluster neighbours) and the industrial testbeds (1 to 15%) is probably the absence of a controller holding a setpoint in driving; a longer, multi-route driving corpus at higher sample rate could show genuine between-mode overlap (e.g. idle vs creeping-in-traffic). Not tested here.

## 6. What this means for the paper (no edits made)

Internal tension to resolve: Section 3 states that "the regimes overlap on WADI and HAI (silhouette 0.06 and 0.08...)" while Appendix C states A8 (between-regime overlap) "is not demonstrated on any dataset". Both sentences are numerically right and semantically at odds because "overlap" is undefined. Under the measurements above: silhouette 0.06 means the K-partition slices a continuous ridge (adjacent clusters touch; no valleys); A8-as-heavy-mode-overlap means two physical regimes coincide, which does not happen (OVL <= 0.27 on discovered clusters, <= 0.016 on physical setpoints).

Recommended measurement of A8, replacing the responsibility-based Table C1:
1. **Nearest-pair overlap in observation space** (label-free): Fisher-projection overlapping coefficient and valley ratio between each discovered cluster and its nearest partner, with the continuum and the Delta = 1 synthetic as the two reference rows. Report the mean and the max; all five datasets sit at OVL 0.02 to 0.13 mean, max 0.27, against 0.61 for a 30%-Bayes-error overlap.
2. **Cross-regime nearest-neighbour share** excluding temporal neighbours: 1% (SWaT) to 14% (WADI), 26% on driving data.
3. **Physically labelled control** (Cranfield): LDA cross-validated error between operating points 0.3% mean, 3.3% max; Mahalanobis separation 41.
4. **Between-mode pocket share** (this is the quantity A8 was written for): fraction of train-normal windows beyond the 99% Mahalanobis envelope of every regime (or below the 1st percentile of steady-state density). Cranfield 18% of all windows (82% of in-step windows); WADI / HAI / SWaT 6 to 10% of the top-trend windows, 2 to 6% of assignment-boundary windows.
5. Drop mean max-responsibility, responsibility entropy and rho as evidence for or against A8; keep them, if at all, only as a description of the fitted mixture. The SKAB paragraph stays correct (it is a floor artefact) but should no longer be framed as the one near-witness, since the metric it uses cannot witness A8 either way.

Recommended statement (substance, not wording): "Operating regimes in the three benchmarks and in the Cranfield rig are separated modes: no pair of discovered clusters, and no pair of physically labelled operating points, overlaps materially in observation space (overlapping coefficient at most 0.27 for discovered clusters, 0.02 for labelled setpoints; a linear classifier separates every nearest pair with at most 25% error, and labelled setpoints with at most 3%). Between-regime windows nevertheless exist in normal operation as transit through low-density corridors: on Cranfield 18% of normal windows lie outside the 99% envelope of both adjacent setpoints, and the density head scores them in the top quintile. Assumption A8 (heavy mode overlap) is therefore not observed; the transit-window phenomenon belongs with the trajectory assumptions A9 and A10, and it bounds the false-positive rate of any density head on regime-switching plants." The last clause is the loud flag: the transit windows are normal data that the reported detector's latent term scores as anomalous (Cranfield NLL percentile 0.79 mean, 0.99 on the step itself), which is a real, measurable cost the paper currently does not state.

## 7. Caveats and things not done

- One VaDE seed per dataset (the A8 screens showed seed spread of ~0.02 in entropy; the qualitative gaps here are 5 to 30 times larger).
- Cranfield channel identities are by index; channels 7, 12, 13 were taken as the setpoint channels from their step structure, not from documentation. Channel 23 is a per-file constant and inflates file-level (not setpoint-level) separation.
- Step detection uses a 60 s two-sided mean shift with +-30 s dilation; it over-counts transition windows at the edges (Section 3a), which makes the 82% "improbable under both" a lower bound and the 25.6% "share of windows in transition" an upper bound. The 18% of all windows beyond both envelopes is a direct count and stands.
- The Mahalanobis q99 envelopes are in-sample 10-D Gaussians on 12 to 100 windows (control exceedance 3%, not 1%); the 27x enrichment is robust to this.
- No test-set or fault windows were examined; this audit is about normal-data geometry only.
- Agent execution: about 40 minutes of tool calls; local CPU compute about 5 minutes total across the seven runs (six datasets/synthetic in parallel at ~15 to 85 s each, plus the transit test); no cloud cost.
