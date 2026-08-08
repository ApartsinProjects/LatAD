The WADI failure is more specific than “correlation anomaly.” It is a coherent block translation: a small correlated subspace moves along its own normal common-mode direction, so internal residuals stay small. The useful statistic is therefore the level of that block relative to external plant context, preferably with a sparse/group scan so five channels are not diluted by the other ~118.

1. Leave-group-out common-factor residual — highest priority

(a) Mechanism. Learn correlated channel groups G from train-normal only using a correlation graph/community clustering on window-level means or the relevant six statistics. Do not hard-code the mined WADI channels.

For each group, extract its normal common factor, e.g. PC1:

g
G
	​

=u
G
⊤
	​

x
G
	​

.

Then predict this whole-group factor from channels outside the group:

g
^
	​

G
	​

=f
G
	​

(x
¬G
	​

),

starting with ridge/PLS rather than another deep model. Score

s
G
	​

=
σ
^
G
	​

∣g
G
	​

−
g
^
	​

G
	​

∣
	​

,S
block
	​

=
G
max
	​

s
G
	​

,

with 
σ
^
G
	​

 and the maximum-over-groups distribution calibrated on held-out train-normal.

(b) Why this targets WADI. Leave-one-channel-out LinRes fails for exactly the expected reason: when four analyzers move down together, each anomalous analyzer is predicted from the other anomalous analyzers, so its residual stays small.

Leaving the entire correlated block out removes that masking mechanism. The external process state may imply “these analyzers should currently be near level L,” while the whole block sits at L−2σ.

There is also useful amplification. If m similarly standardized channels each shift by −δ along an approximately equal-weight common factor,

u≈(1,…,1)/
m
	​

,

then the factor moves by approximately

−δ
m
	​

.

Four analyzers each shifted only 2σ give a common-factor displacement around 4σ; five channels give 4.5σ. The individual marginals can all remain sub-threshold while the coherent block displacement becomes extreme.

That is the statistic currently missing.

(c) HAI/LinRes integration. Generalize this head to include both singleton groups and learned blocks. Singleton G={j} reduces essentially to your existing LinRes signal; multi-channel G catches WADI-style common-mode masking.

Use it as a rescue-only head:

S=S
LatAD
	​

+λ[z
block
	​

−τ]
+
	​

,

where τ, e.g. the 97.5th or 99th train-normal percentile, and λ are fixed from normal calibration. Thus ordinary block residuals cannot reshuffle LatAD rankings; only genuinely extreme conditional inconsistencies contribute.

(d) Failure risk. If the WADI analyzer block can legitimately take those low levels under external plant states that are poorly measured, the group-out predictor will also regard them as normal. And discovering only the mined five-channel group from test failures would be test-driven method design; the grouping procedure must operate uniformly on train-normal data.

2. Mode-conditioned coherent-shift scan

This is related but asks a different question and does not require predicting the block from hundreds of outside features.

(a) Mechanism. Within every train-normal VaDE operating mode k, learn correlated groups and their first one or two PCA directions. For group G,

a
Gk
	​

=u
Gk
⊤
	​

(x
G
	​

−μ
Gk
	​

).

Estimate the train-normal distribution of a
Gk
	​

, preferably robustly, and score its conditional tail:

s
Gk
	​

=
σ
Gk
	​

∣a
Gk
	​

∣
	​


or an empirical mode-conditioned quantile. Soft-average across VaDE responsibilities, then take a calibrated maximum across groups.

Crucially, score the common-mode component, not the within-group contrast residual.

(b) Why WADI. Your global latent density answers, roughly, “is this whole 123-channel state populated somewhere?” That can miss a five-channel coherent displacement because most coordinates remain completely ordinary.

This statistic instead asks:

Given this operating regime, how extreme is the shared level of this particular correlated subsystem?

Again, four channels at roughly −2σ can create a −4σ excursion along their shared direction even while each marginal remains legal and the contrasts between them remain normal.

It is effectively a sparse projection scan, preventing a small affected block from being averaged away by the full-system representation.

(c) HAI. Include singleton/very-small groups among the candidate blocks, or separately retain the thresholded LinRes rescue described above. HAI's valve/flow inconsistency should then generate either a singleton conditional residual or a small process-group residual.

(d) Failure risk. If the low analyzer common factor is itself frequent within the same VaDE mode, this will still miss it. That would indicate that the VaDE mode is too coarse with respect to the operational context relevant to these analyzers, making Direction 1 preferable.

3. Conditional block Mahalanobis in observation space, not latent space

Your negative full-covariance experiment was on the VaDE latent. I would not repeat that. The potentially useful covariance is local to the affected channel block.

(a) Mechanism. For each train-normal group G, predict its vector jointly from outside channels:

x
^
G
	​

=B
G
	​

x
¬G
	​

+b
G
	​

,r
G
	​

=x
G
	​

−
x
^
G
	​

.

Estimate a shrinkage covariance Σ
r,G
	​

 of these block residuals and score

s
G
	​

=r
G
⊤
	​

Σ
r,G
−1
	​

r
G
	​

.

Scan/calibrate max
G
	​

s
G
	​

 against held-out normal.

(b) Why WADI. Standard LinRes conditions each analyzer on the others in the drifting group. This conditional block model forbids that leakage by predicting all members jointly from the rest of the process.

If all four analyzers are −2σ lower than external process conditions imply, the residual vector is approximately

r
G
	​

=(−2,−2,−2,−2),

and its energy along the low-variance common residual direction can be very large despite every individual residual being modest.

This is fundamentally different from your failed full-covariance latent Mahalanobis experiment.

(c) HAI. This naturally subsumes LinRes. HAI groups could be valve/flow process blocks; a valve state inconsistent with surrounding process measurements produces a large conditional residual. Again use a normal-thresholded rescue contribution so it cannot degrade already well-ranked examples.

(d) Failure risk. High-dimensional external prediction can overfit and plant drift can inflate residuals. Ridge/PLS, small predefined block sizes, cross-fitted train-normal residuals, and shrinkage covariance are necessary. If external channels simply do not encode expected analyzer level, there is nothing to condition on.

4. Hierarchical/local LatAD heads to prevent sparse-block dilution

(a) Mechanism. Retain your existing global VaDE unchanged. Add small group-specific normal models for correlation communities learned from train-normal: either tiny VaDE/GMM heads or, more simply, 1–3 dimensional PCA-factor density models.

Produce one calibrated anomaly score per subsystem and aggregate with

S
local
	​

=
G
max
	​

F
G
	​

(s
G
	​

),

where calibration of the maximum itself is performed on held-out normal to account for multiple testing.

(b) Why WADI. Five abnormal channels constitute only a few percent of WADI's dimensions. A global representation is explicitly encouraged to retain dominant structure, so a locally unusual subsystem state can live in a well-populated global latent region.

A local analyzer-group model removes that dilution. The observed vector

(AIT
1
	​

,AIT
2A
	​

,AIT
2B
	​

,…)

may have ordinary shape but unusually low common level. In a local two-dimensional representation—common factor plus contrast factor—it would appear as:

contrast: normal;

common factor: unusually negative.

That decomposition is nearly tailored to the mined signature while remaining train-normal-defined and generic across subsystems.

(c) HAI. Include the thresholded singleton LinRes rescue separately; I would not force every HAI complementary signal through the local density models.

(d) Failure risk. Multiple local heads dramatically increase opportunities for false alarms. The maximum must therefore be calibrated after scanning all groups. Also, if you construct groups specifically around the discovered WADI analyzers rather than using a generic normal-data grouping algorithm, reviewers can reasonably call it test-set engineering.

What I would actually implement

I would start with Direction 1, using perhaps 5–20 train-normal-derived correlation communities and a simple ridge predictor of each community's PC1 from its complement. Before training anything complex, test the mined WADI windows diagnostically.

For the analyzer group, compute these three numbers:

z
common
	​

=
σ(u
G
⊤
	​

x
G
	​

)
u
G
⊤
	​

(x
G
	​

−μ
G
	​

)
	​

,
z
external
	​

=
σ
normal residual
	​

u
G
⊤
	​

x
G
	​

−
u
G
⊤
	​

x
G
	​

	​

(x
¬G
	​

)
	​

,

and

z
contrast
	​

=∥(I−u
G
	​

u
G
⊤
	​

)(x
G
	​

−μ
G
	​

)∥
Σ
−1
	​

.

Your hypothesis predicts a very distinctive pattern:

large negative z
common
	​

, large ∣z
external
	​

∣, but ordinary z
contrast
	​

.

If that pattern is not present in the mined WADI failures, I would stop: the proposed mechanism is wrong. If it is present repeatedly across the five attack episodes, you have both an implementable extension and a much stronger mechanistic result.

For HAI, I would add the existing one-hot LinRes as a thresholded rescue head rather than ordinary score fusion:

S
final
	​

=S
LatAD
	​

+λmax(0,z
LinRes
	​

−τ),

with τ entirely train-normal-calibrated and preferably high. That matches your observation: LinRes should intervene only on exactly those rare HAI cases where LatAD is unexpectedly permissive, rather than perturbing the ranking everywhere.

The most promising new idea, therefore, is not “better density.” It is conditional common-mode block inconsistency: the affected channels agree with one another, but as a group they disagree with what the rest of the system says their level should be.