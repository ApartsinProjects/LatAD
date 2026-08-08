# Difficulty stratification: motivation and procedure

## Why we do not report on all anomalies

An aggregate detection score over every labelled anomaly is dominated by the anomalies that are
trivial to catch. In a cyber-physical process most labelled attack timestamps drive at least one
sensor far outside its normal operating range (a valve forced open, a pump switched on, a level
sensor railing). Any per-channel alarm flags these, so a detector can reach a high all-subset AUROC
while contributing nothing on the anomalies that actually require a model of normal behaviour. The
headline number then measures the dataset's easy majority, not the method's capability. This is the
same failure mode that point-adjustment inflation produces: an impressive score that does not
reflect performance on the operationally important cases.

The operationally important cases in CPS security are the stealthy ones: manipulations that keep
every individual sensor inside its normal range while violating the joint relationships between
sensors (a correlation break). A per-channel rule cannot see them by construction. Whether a method
detects these is exactly what an aggregate score hides, so we stratify by difficulty and report the
difficult subset as the headline for joint-structure capability, alongside the full set.

## How we decide what is difficult: a cascade of simple, model-free prefilters

An anomaly is "easy" if a simple detector that uses no learned model of the data already catches it.
We remove the easy anomalies with a cascade of prefilters, each targeting one class of trivially
detectable deviation. What survives every prefilter is the difficult subset.

1. **Per-channel magnitude.** `maxz` = the largest standardized deviation over channels within the
   window. Flags any single sensor grossly outside its normal range (railing, spike, out-of-range).
2. **Linear cross-channel structure.** For each channel we fit, on normal data only, a linear
   regression predicting that channel's window-average from the other N-1 channel averages. The
   per-window residual MSE flags windows where the normal linear correlations between sensors break,
   and it also catches large coherent multi-channel shifts (many sensors moving together). This is
   the prefilter that removes the large distributed deviations a per-channel rule misses.

An anomaly is **difficult** only if it passes both prefilters: no single sensor is grossly off, and
the linear cross-channel structure is (near) intact. These are the anomalies that are invisible to
marginal and linear-multivariate rules and can only be flagged by modelling the nonlinear, multimodal
joint distribution of normal behaviour.

## How the thresholds are set

Each prefilter's threshold is the **99th percentile of that statistic on train-normal data**. This
fixes a per-filter false-positive rate of about 1% on normal operation, is a principled operating
point, and, critically, is computed **without ever looking at the anomaly labels**. The split is
therefore not circular: difficulty is defined entirely by normal-data statistics, independent of
which anomalies exist and independent of the detector under test.

## Why the prefilters must be simple (and why this is fair)

The prefilters are deliberately restricted to the simplest model classes: per-channel magnitude
(marginal) and linear cross-channel regression. They are never the method under test, and never a
learned nonlinear detector. The logic is incremental value over simple baselines: we discard the
anomalies a simple detector already handles and ask whether the nonlinear, multimodal method adds
anything on the remainder. If the prefilter were itself a strong nonlinear model (a deep
autoencoder, a full-covariance Mahalanobis rule), it could remove the very anomalies the method is
meant to catch, and the comparison would be meaningless. Keeping the prefilters strictly simpler than
the method is what makes the difficult-subset result a fair test of the method's added capability.

## What the difficult subset means, per dataset

Because difficulty is defined by simple detectors, the size of the difficult subset is itself a
property of the dataset, and it varies:

- **WADI**: almost all difficult anomalies survive both prefilters. Its stealthy attacks are true
  correlation breaks whose aggregate magnitude is at or below normal, so no simple rule sees them.
  This is the cleanest joint-structure benchmark.
- **HAI**: about half survive; the rest are linearly detectable multi-channel shifts.
- **SWaT**: only a small fraction survive. Most SWaT attacks are large distributed deviations (for
  example a pump switched on with valves opened), which the linear prefilter catches even though no
  single channel exceeds the per-channel threshold. SWaT contains few genuine correlation-break
  anomalies.

## Reporting

We report the difficult-subset metric with its subset size n and multi-seed confidence intervals,
and we state n explicitly wherever it is small (a small difficult subset gives a wide interval and
must not be read as a precise point estimate). We report the full-set metric alongside for
completeness. The difficult subset is the headline because it isolates the capability the method
claims: detecting anomalies that break joint structure while remaining invisible to every simple rule.
