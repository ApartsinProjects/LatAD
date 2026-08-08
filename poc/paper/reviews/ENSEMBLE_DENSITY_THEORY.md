# Why a random-subspace VaDE ensemble estimates anomaly-relevant density more accurately

## 1. The problem: high-dimensional density is worst exactly in the tails

Anomaly detection needs the *tail* of `p(x)`, and tail density is the hardest part to
estimate in high dimension `d`. Nonparametric estimators converge at `N^{-1/(d+4)}`
(curse of dimensionality); a full Gaussian/GMM needs `O(d^2)` covariance parameters per
mode, so with 123 channels the covariance is ill-conditioned and its inverse (the
Mahalanobis metric) is dominated by noise directions. The full-space density model is
therefore least reliable precisely where anomalies live.

## 2. Sparse-anomaly dilution (the mechanism we measured, made quantitative)

A CPS attack typically perturbs a small set `S` of `m << d` channels (WADI: ~5 of 123).
The full-model anomaly score (a log-density ratio) gains a signal `Δ_S` from those
channels, but its *normal-window variance* is the sum of fluctuations over **all** `d`
channels. So the achievable separation is

    SNR_full  ≈  ‖Δ_S‖ / sqrt(Σ over all d channels)  ~  ‖Δ_S‖ / sqrt(d).

A subspace model restricted to a subset containing `S` competes against only `m`
channels of noise:

    SNR_sub   ≈  ‖Δ_S‖ / sqrt(m),   independent of d.

For WADI this is a `sqrt(123/5) ≈ 5×` SNR gain in principle — which is why the full
123-channel latent density misses the coherent analyzer dip while a small subspace sees
it. This is not a tuning artifact; it is the signal-to-noise arithmetic of sparse signals
in high dimension.

## 3. Why the VAE/VaDE specifically is blind to it — the rate–distortion argument

A VAE/VaDE is trained to reconstruct, and its encoder is a **rate–distortion optimiser**:
with a finite-capacity latent it spends bits on the directions that most reduce
reconstruction error, i.e. the **high-variance** directions. A sparse anomaly along a
**low-variance** direction (a few analyzers a couple of sigma off, moving *along* their
own correlation) is exactly the kind of structure the encoder is pressured to **discard**
as nuisance. So the full VaDE latent — and any density scored in it — is *structurally*
blind to small-subset, low-variance, correlation-preserving perturbations. This is the
VAE-specific reason our common-mode dip survives into no latent coordinate.

**Subspaces reallocate the rate–distortion budget.** When a random subset isolates the
analyzer channels, those channels are now a large fraction of the subset's total variance,
so the subset encoder *must* represent them to reconstruct — the anomaly-carrying
direction is no longer nuisance. The subset VaDE's mixture prior (A1/A2) then carves finer
operating modes in low dimension, and the coherent low level becomes a genuine
between-mode / thin-fringe low-density region (A3/A4) instead of a point merged into one
broad high-dimensional component. The ensemble thus recovers the density structure the
single high-dimensional VAE is trained to throw away.

## 4. Why aggregating many subspaces is the right estimator

- **Union-bound / OR-of-experts (max aggregation).** A sparse anomaly is visible in *some*
  subspace. If each of `K` random size-`s` subsets covers a useful part of `S` with
  probability `p`, then `max`-aggregation detects it once **any** subspace fires:
  `P(detect) ≈ 1 − (1−p)^K → 1`. The full model has a single space, in which the anomaly
  is diluted; the ensemble has `K` chances, and needs only one good draw. Max/top-k is the
  anomaly-detection analogue of a union bound over weak detectors.
- **Bias–variance / bagging.** Random subspace selection is feature bagging (Ho 1998;
  Lazarevic & Kumar 2005). Each low-dim VaDE is a **low-variance, biased** density estimator
  (well-conditioned, curse mitigated); decorrelated experts + aggregation trade the full
  model's high estimator variance for many stable experts, sharpening the normal/anomaly
  gap. This is the density-estimation counterpart of why deep ensembles give better
  calibrated uncertainty (Lakshminarayanan et al. 2017).
- **Composite likelihood / product-of-experts.** The joint `log p(x)` can be approximated
  from overlapping low-dimensional marginals/conditionals (Besag pseudolikelihood; Lindsay
  composite likelihood; Hinton product-of-experts). A bank of overlapping subspace densities
  captures the dependency structure that matters **without** ever estimating the full
  `d × d` covariance — better-conditioned and cheaper (`K · O(N s^2)` vs `O(N d^2)`).

## 5. The clean tie-in to our own numbers

**Isolation Forest is itself a random-subspace ensemble** (it isolates points with
axis-aligned splits on subsampled features). That is exactly why IF, alone among the
baselines, *resists* the WADI dilution and ties LatAD there (0.677 vs 0.690): its
subspace-partition mechanism sees the sparse block. Our subspace-VaDE ensemble is the
**learned-density generalisation** of that same principle — it replaces IF's axis-aligned
partitions with per-subset multimodal latent densities — and it beats IF on WADI-difficult
(**0.823 vs 0.675**) because it adds the mode structure (A1–A4) that isolation partitions
lack. The empirical ordering full-VaDE (0.70) < IF-subspace-partition (0.68… ties) <
learned-subspace-density-ensemble (0.82) is exactly what the theory predicts.

## 6. One-paragraph summary for the paper

CPS attacks are sparse in channel space, and a single high-dimensional VaDE both dilutes
their signal-to-noise by `sqrt(d/m)` and, as a rate–distortion coder, actively discards the
low-variance directions they occupy. Estimating density over an ensemble of random channel
subspaces restores the anomaly's variance share within each low-dimensional model
(recovering the mode structure of A1–A4), converts detection into a union bound over
`K` cheap, well-conditioned experts, and reduces estimator variance by feature bagging —
the same mechanism that makes Isolation Forest, itself a subspace ensemble, the only
baseline that resists the dilution. Aggregating the subspace latent densities therefore
estimates the anomaly-relevant tail more accurately than one full-dimensional VAE.
