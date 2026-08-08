One methodological caveat first: the fitting/calibration below is train-normal-only and therefore leakage-free in the narrow sense, but because the WADI failure signatures were mined from labeled test anomalies, these cycles are test-informed method development. Any final “new method” result should therefore be described as exploratory unless the frozen extension is subsequently evaluated on untouched data.

I would use one common safety wrapper for every auxiliary head. Convert baseline and auxiliary scores to cross-fitted train-normal percentiles u
0
	​

,u
h
	​

. Let an auxiliary head act only as a rare miss rescue:

u
∗
	​

={
max(u
0
	​

,u
h
	​

),
u
0
	​

,
	​

u
0
	​

<0.99 and u
h
	​

>0.995
otherwise.
	​


Thus the new head never lowers LatAD's score, does not touch points LatAD already regards as extreme, and can affect at most roughly 0.5% of calibration-normal windows. If multiple heads survive, calibrate the maximum auxiliary score jointly on normal so the family-wise rescue rate remains 0.5%. This is the most defensible way I see to retain the WADI common-mode gain without letting an auxiliary head globally reorder SWaT.

Cycle 1 — A5: observation-space common-lever rescue

1. Assumption. A5 “few levers”: many correlated sensors are manifestations of a small number of common physical factors. Your WADI failure is almost a textbook realization: several analyzers translate together along one common factor.

2. Statistic. Start from the successful common-mode block head, but formalize it as a train-normal factor model. For each train-normal-derived correlated block G,

x
G
	​

=μ
G
	​

+U
G
	​

f
G
	​

+ϵ
G
	​

,

with q=1−3 factors chosen using train-normal explained variance, capped at 3. Score the factor coordinates, not the orthogonal residual:

H
A5
	​

(x)=
G
max
	​

k∈K(x)
min
	​

[f
Gk
T
	​

Var(f
Gk
	​

)
−1
f
Gk
	​

],

where the factor distribution is conditioned on VaDE mode k. Use the conservative minimum across plausible modes so a point is not anomalous merely because it belongs to a rare mode.

Feed this through the miss-rescue wrapper above rather than ordinary additive fusion.

3. Target. WADI's analyzer block should have a large common-factor displacement while its orthogonal factor residual remains ordinary. This explicitly encodes:

large common-factor excursion+small within-block contrast.

It is therefore tailored to the coherent-translation signature. The ambiguity-only rescue is the important SWaT protection: SWaT anomalies that LatAD already ranks highly retain their original score even if this factor head behaves badly.

4. Kill criterion. Since the existing block head reaches 0.77, this formalized/safe version should retain most of that gain. I would kill it if either:

WADI difficult AUROC falls below about 0.74, or loses the advantage over IF; or

SWaT falls below about 0.95 or HAI below about 0.80.

If the safe wrapper destroys most of the WADI benefit, you have evidence that the original 0.77 depended on broad score reordering rather than genuine rescue of the missed cases.

Cycle 2 — A3: mode-conditioned hard envelope of the common factor

1. Assumption. A3 says each operating mode occupies a bounded region and anomalous pockets can exist inside global marginal ranges. Your current basin head checks mode ambiguity in latent space; this instead tests the actual observation-space boundary of the troublesome block coordinate.

2. Statistic. For each block G, use its leading common-mode coordinate

c
G
	​

=u
G
T
	​

x
G
	​

.

For each VaDE mode k, estimate weighted train-normal bounds, e.g.

L
Gk
	​

=Q
0.005
	​

(c
G
	​

∣k),U
Gk
	​

=Q
0.995
	​

(c
G
	​

∣k).

Define normalized envelope exceedance

e
Gk
	​

(x)=max(0,
s
Gk
−
	​

L
Gk
	​

−c
G
	​

	​

,
s
Gk
+
	​

c
G
	​

−U
Gk
	​

	​

),

where s
±
 are train-normal robust tail scales. Then

H
A3
	​

(x)=
G
max
	​

k:γ
k
	​

>η
min
	​

e
Gk
	​

(x).

Use η≈0.1 and a minimum effective mode sample size; otherwise back off to a pooled block envelope.

3. Target. This asks the precise question that global max∣z∣ does not:

Is the analyzer group's shared level outside the legal envelope for the current operating mode, even though every individual analyzer remains inside its global range?

If the low WADI state occurs normally only in other operating regimes, this should be substantially stronger than the current block head. SWaT is protected by the same rescue-only integration.

4. Kill criterion. Before looking at aggregate AUROC, check the mined WADI episodes. Kill the A3 hypothesis if the coherent dips do not exceed the 99–99.5% mode-conditioned block envelope in at least roughly 4 of the 5 episodes. At the aggregate level, abandon if it adds <0.02 WADI AUROC over baseline LatAD.

This cycle is especially valuable because failure is informative: it would show that the attacked low level is genuinely inside the normal mode envelope, not merely hidden by global aggregation.

Cycle 3 — A8: per-mode whitened residual in observation space

1. Assumption. A8 says typed channels have correlated noise and therefore deviations should be judged with covariance-aware residuals. Your present whitened residual is derived from reconstruction. This experiment removes the encoder/decoder from the residual definition entirely.

2. Statistic. Work in small train-normal-derived process blocks, not the full 6C-dimensional vector. For block G and mode k, let y
G
	​

 contain its six per-channel summary statistics. Estimate

μ
Gk
	​

,Σ
Gk
	​


with Ledoit-Wolf shrinkage from normal windows assigned softly to mode k. Score

D
Gk
2
	​

=(y
G
	​

−μ
Gk
	​

)
T
Σ
Gk
−1
	​

(y
G
	​

−μ
Gk
	​

),

and use

H
A8
	​

=
G
max
	​

k:γ
k
	​

>η
min
	​

D
Gk
2
	​

.

For discrete actuator blocks, keep one-hot representation rather than treating states numerically.

3. Target. This is an explicit test of whether the VaDE encoder has discarded a small but informative observation-space direction. It may catch WADI if the common analyzer displacement is unusual within its local operating mode even though the global latent representation regards the total state as populated.

It may be even more useful on HAI: the hard valve/flow cases are caught by linear observation-space relationships but missed by both latent density and reconstructed residual. A per-mode observation-space residual is exactly the missing representation there.

I would not expect it necessarily to beat Cycle 2 on WADI because high positive correlation can make a common-direction excursion relatively cheap under Mahalanobis geometry. That makes this a useful falsification experiment rather than a guaranteed gain.

4. Kill criterion. Kill if both are true:

mined WADI misses remain below roughly the 95th train-normal percentile of this head; and

the mined HAI valve/flow misses are not strongly elevated.

At the dataset level, if neither WADI nor HAI improves by at least ~0.01 AUROC under the rescue wrapper, stop.

Cycle 4 — A9: sustained-tail dwell score, not generic trajectory scoring

1. Assumption. A9 concerns characteristic time scales and dwell. The distinctive feature of your WADI misses is not a transition: they are flat, sustained plateaus. Generic trajectory/change-point models are therefore solving the wrong temporal problem.

2. Statistic. Apply temporal scoring only to the block common-mode coordinate from Cycles 1–2.

For every block/mode, transform the common factor to its train-normal conditional percentile

q
t
	​

=F
Gk
	​

(c
t
	​

).

Define lower-tail states, for example q
t
	​

<0.10,0.05,0.01, and learn from train-normal the empirical survival distribution of consecutive run lengths:

P
normal
	​

(L≥ℓ∣q≤q
0
	​

,k,G).

For a test window with current lower-tail depth q
t
	​

 and run length L
t
	​

,

H
A9
	​

(t)=−logP
normal
	​

(L≥L
t
	​

,Q≤q
t
	​

∣k,G).

This is preferable to another LSTM/transition model: it asks specifically whether a moderately low block level has persisted abnormally long.

3. Target. A single 2σ analyzer dip can be normal. A series of overlapping windows all sitting at approximately the same 2σ negative common-mode displacement may be extremely rare. This directly accumulates the piece of evidence your snapshot score discards.

Unlike onset detection, it remains elevated throughout the flat attack because the statistic depends on dwell length.

SWaT again receives only rare-tail rescue; its already high LatAD anomalies should largely remain unchanged.

4. Kill criterion. Inspect the five WADI episodes first. If their plateau run probabilities are not unusually small—for example, most remain p>0.01 under the normal dwell distribution—then A9 does not explain these misses and I would abandon it immediately.

For the full result, I would require WADI difficult AUROC around ≥0.73 and preferably a gain over the static common-mode score itself. If it merely reproduces the same ranking, it adds complexity without evidence.

Cycle 5 — A8: typed, mode-conditioned LinRes rescue for the HAI residual misses

1. Assumption. Again A8, but now operationalized through channel type and cross-channel conditional structure rather than covariance around a centroid.

2. Statistic. Upgrade the current LinRes baseline into a train-normal auxiliary head:

fit leave-one-channel-out ridge/elastic-net models within VaDE mode;

one-hot discrete actuators;

standardize continuous sensor/flow targets;

obtain cross-fitted normal residuals;

whiten related residual vectors within process blocks.

For block G,

r
G
	​

=y
G
	​

−
y
^
	​

G
	​

,H
Lin
	​

(x)=
G
max
	​

r
G
T
	​

Ω
G
−1
	​

r
G
	​

.

Do not add this globally to LatAD. Give it exactly the same rare miss-rescue privilege as the WADI common-mode head.

3. Target. This is specifically for the HAI failures you already diagnosed: P1 valve positions and flows for which ordinary LinRes reaches 0.61–0.99 percentile while LatAD remains permissive.

It should contribute almost nothing to the WADI coherent-block translations because those anomalies are internally self-consistent and LinRes already scores them low. That is desirable: the WADI and HAI rescue mechanisms become complementary rather than one head being forced to solve both.

For SWaT, the high normal-tail threshold plus base-ambiguity gate should sharply limit opportunities to disturb the ceiling result.

4. Kill criterion. Require a visible improvement on the actual HAI residual-miss set and at least a small difficult-AUROC improvement without damaging the existing significant HAI lead. If HAI stays essentially 0.811 or declines, the extra head is unnecessary. Any >0.01 loss on WADI or SWaT is also grounds to remove it.

Cycle 6 — A1/A7 fallback: hierarchical micro-regimes in the problematic block

I would run this only if Cycles 2–4 fail.

1. Assumption. A1/A7: regimes are multimodal and hidden. The hypothesis would be that the global VaDE representation has merged operational states that are distinct with respect to analyzer level.

2. Statistic. Within each global VaDE mode k, fit a tiny train-normal GMM to the low-dimensional block-factor coordinates from Cycle 1:

p(f
G
	​

∣k)=
m=1
∑
M
Gk
	​

	​

w
m
	​

N(f
G
	​

;μ
m
	​

,Σ
m
	​

),

with M
Gk
	​

∈{1,…,6} chosen by train-normal BIC and requiring bootstrap-stable components.

Use nearest-microcomponent NLL rather than mixture-weighted NLL:

H
A1/A7
	​

=
m
min
	​

−logN(f
G
	​

;μ
m
	​

,Σ
m
	​

),

preserving the A6 rare-mode principle.

3. Target. If the low WADI analyzer level is globally populated only because different operating states were collapsed together, hierarchical micro-regimes should separate the “legal low” states from the attack's inappropriate low state.

The structural gate—BIC improvement plus stable train-normal microclusters—also gives a more assumption-specific reason to leave the head inactive on datasets where no meaningful microstructure exists.

4. Kill criterion. If the problematic WADI windows remain close to a normal microcomponent, the “mode merging” explanation is false. Abandon if WADI gains <0.02 or if the microcomponent structure is unstable across train-normal resamples.

Recommended order

Run 1 → 2 → 3 → 4 → 5, with 6 only as a fallback. The scientific logic is useful even if some cycles fail:

A5: can the already-successful common-mode signal be integrated safely?

A3: is the attack outside a mode-specific block envelope?

A8: did the latent representation discard an observation-space covariance violation?

A9: is abnormal dwell the missing evidence?

A8: add the clearly complementary HAI linear-conditional signal without forcing it onto WADI.

A1/A7: only then test whether hidden micro-regimes explain the remaining WADI ambiguity.

The strongest possible result is not merely another WADI AUROC increase. It would be a sequence where each failed or successful cycle tells you which MIIM assumption actually explains the remaining errors, while the rescue-only wrapper keeps those targeted additions from undoing the already-solved SWaT case.