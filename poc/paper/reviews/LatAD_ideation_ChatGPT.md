The most promising path is not to add more snapshot-density machinery indiscriminately. Your WADI diagnostics say the missing information is primarily trajectory information: the attack points are individually plausible but occur as low-magnitude correlated excursions at abnormal onset/offset locations in the path. That is exactly what A9/A10 currently claim but the model does not exploit.

I would rank the directions as follows.

1. Add a latent trajectory / transition-surprise branch exploiting A9–A10

(a) Modification. Keep the existing VaDE representation, but treat the latent means z
t
	​

 or responsibilities γ
t
	​

 as a time series. Add two train-normal-only scores:

mode-transition surprise

s
trans
	​

(t)=−log
ij
∑
	​

γ
t−1,i
	​

P
ij
	​

γ
t,j
	​

,

where P
ij
	​

 is a smoothed transition matrix estimated from normal training data;

latent innovation/change score, e.g.

z
t
	​

=A
k
	​

[z
t−1
	​

,…,z
t−p
	​

]+b
k
	​

+ϵ
t
	​


within mode k, with shrinkage covariance for ϵ
t
	​

, and score the Mahalanobis innovation.

I would also test a very simple branch before anything sophisticated: per-mode standardized Δz
t
	​

=z
t
	​

−z
t−1
	​

, optionally Δ
2
z
t
	​

, followed by CUSUM/EWMA or Mahalanobis scoring. A simple temporal innovation score is easier to defend than an elaborate sequence model.

(b) Why it should help. This directly targets WADI's failure mode. A 2–3σ dip can be perfectly plausible as a state yet highly implausible as the next state given the previous trajectory. Attack onset and recovery are precisely where Δz
t
	​

, transition probabilities, or conditional innovations should fire. It is also much more consistent with your mechanistic narrative than trying to make the static density arbitrarily complicated.

SWaT may benefit less because its difficult subset is already almost trivial, but trajectory scoring could establish that LatAD captures a qualitatively different signal from max∣z∣ and LinRes.

(c) Fair evaluation. Predefine the temporal branch and its hyperparameters using train-normal only. Estimate transition/dynamics parameters on one portion of normal training data and calibrate scores on held-out normal. Report:

difficult/all raw AUROC;

partial AUROC at a prespecified low-FPR range, e.g. FPR ≤1%;

TPR at train-normal-calibrated FPR targets;

paired episode bootstrap wherever multiple attack episodes exist.

Keep the present snapshot difficulty definition unchanged for the primary comparison so the temporal detector does not redefine the problem it solves.

(d) Risk. Normal plant transitions can also be abrupt. With overlapping windows, a change-point score can simply become an onset detector rather than an anomaly detector. And if WADI's attacks follow trajectories also seen during legitimate control transitions, it may not improve much.

Priority: highest. This is the cleanest scientific extension because it activates assumptions already present in the paper rather than adding an unrelated trick.

2. Use causal multiscale windows rather than one fixed W=60

(a) Modification. Encode several nested windows ending at the same timestamp, for example W={30,60,120,240}, using the same six channel statistics initially. Either:

run the same encoder/scorer independently at each scale and combine calibrated tail probabilities; or

concatenate a small set of multiscale statistics before the VaDE encoder.

I prefer parallel scale-specific scores because they are easier to ablate and less likely to produce an enormous input vector.

A particularly useful extension is to include statistics of differences, not just levels: mean Δx, range of Δx, or early-vs-late window mean difference.

(b) Why it should help. WADI's low-amplitude dip may be weak over 60 samples but conspicuous relative to the previous 120–240 samples. Short windows preserve attack boundaries; longer windows establish the local operating context.

This also complements the temporal branch: multiscale features measure what changed over different horizons, while transition scoring measures whether the trajectory was likely.

(c) Fair evaluation. Freeze the scale set in advance and use causal windows only if the intended detector is online. All baselines that consume window summaries should receive equivalent multiscale information in a supplementary fairness experiment. Continue using identical raw labels and evaluation timestamps.

(d) Risk. Longer windows can artificially increase separability by including more attack samples or blur boundaries. They also create more correlated test observations. A large sweep over window sizes would look like test-set optimization, so use a small physically motivated scale set rather than searching dozens of values.

Priority: very high.

3. Replace diagonal latent distances with shrinkage full-covariance, mode-conditioned geometry

(a) Modification. Preserve the VaDE mixture, but estimate a full covariance Σ
k
	​

 for each sufficiently populated normal latent mode and use

d
k
2
	​

(z)=(z−μ
k
	​

)
⊤
Σ
k
−1
	​

(z−μ
k
	​

).

Use Ledoit–Wolf or another fixed shrinkage estimator. For rare modes with insufficient effective sample size, fall back automatically to diagonal or tied covariance. I would apply this first to the nearest-mode head, not convert the entire high-K GMM to unrestricted full covariance.

A useful compromise is low-rank-plus-diagonal covariance.

(b) Why it should help. A correlated WADI dip may lie inside every marginal latent range while moving in a direction that normal data have extremely low variance along. A diagonal density largely misses exactly that geometry. Full Mahalanobis distance can turn a modest marginal excursion into a strong anomaly if it violates a learned correlation direction.

This is also conceptually much closer to your CPS argument than simply increasing mixture component count.

(c) Fair evaluation. Fit covariance entirely on train-normal. Predefine minimum effective membership required for full covariance and the fallback rule. Report diagonal versus shrinkage-full covariance as a direct ablation across all five seeds.

(d) Risk. Latent dimensions are only 6–16, which helps, but high-K components can still have very few effective samples. Full covariance may improve WADI by overfitting normal local geometry and then produce unstable tails. Shrinkage and cross-normal validation are essential.

Priority: high.

4. Add a local-support latent score: kNN first, LOF second; flow only as a secondary experiment

(a) Modification. On the train-normal latent representation, add a local-support head such as:

s
kNN
	​

(z)=d(z,kth nearest normal neighbour)

preferably within or conditional on the nearest VaDE mode. Alternatively use LOF in novelty-detection mode.

Use empirical train-normal survival probabilities rather than raw distances so that scores from modes of different density become comparable.

I would test a normalizing flow only after kNN/LOF. If you use a flow, use it inside the learned latent rather than on the original 738-dimensional WADI summary vector.

(b) Why it should help. Your current mixture density imposes smooth Gaussian structure. WADI anomalies may occupy a locally unsupported pocket while still receiving tolerable probability from a broad Gaussian component. kNN/LOF asks a different question: is there actually normal support nearby?

That is plausibly part of what Isolation Forest is exploiting.

(c) Fair evaluation. Choose k from a very small prespecified set or using only held-out-normal stability—not anomaly AUROC. Compare:

VaDE density;

kNN/LOF alone on the same latent;

their fixed calibrated combination.

Report score correlations and which WADI attack windows each detects.

(d) Risk. kNN can simply identify rare-but-valid modes, violating A6, unless it is mode-conditioned or locally normalized. LOF can be noisy. Flows have an additional conceptual problem: high likelihood does not necessarily correspond to semantic normality, and sufficiently expressive flows can reproduce the same overgeneralization problem you criticize in reconstruction.

Priority: high for kNN; moderate/low for flows.

5. Explicitly capture what Isolation Forest is finding, then decide whether to incorporate it

(a) Modification. Before adding IF to LatAD, perform a disagreement analysis between WADI difficult scores:

LatAD high / IF low;

IF high / LatAD low;

both high;

both low.

For IF-high/LatAD-low attack windows, examine which of the 6×channel summary coordinates are responsible, their nearest-normal distances, local latent density, mode membership, and temporal derivatives.

Then test two principled variants:

Latent-IF: Isolation Forest on z, making it another latent normal-support head;

cross-space partition head: IF on the original standardized summary vector, combined with LatAD through fixed normal-calibrated score fusion.

(b) Why it should help. Isolation Forest does not estimate a smooth probability law. Recursive axis-aligned partitioning gives short paths to observations in sparsely occupied combinations of features. A low-amplitude 2–3σ shift in two or three coordinated variables may therefore look unremarkable to a Gaussian density but unusually isolated in the actual normal support.

If Latent-IF closes the WADI gap, the problem is the density model. If only raw-feature IF works, the encoder may be discarding precisely the subtle tail structure needed for WADI.

That diagnostic distinction is scientifically valuable.

(c) Fair evaluation. Do not select the combination weight by WADI test AUROC. Convert each score to an empirical normal-tail probability using held-out training normal and use a predetermined combination such as equal-weight rank sum, Fisher combination, or maximum tail significance.

Critically, compare:

IF;

LatAD;

LatAD + IF;

latent-IF;

density + latent-IF.

If LatAD+IF wins only because IF dominates, state that.

(d) Risk. Adding the strongest competing baseline as an auxiliary head can look like benchmark engineering. If the final score is essentially “LatAD plus Isolation Forest,” reviewers can reasonably ask what LatAD contributes. This direction should be used only if the disagreement analysis reveals a coherent complementary signal.

Priority: medium-high diagnostically; medium as final architecture.

6. Replace raw z-score summation of heads with normal-tail probability fusion

(a) Modification. For every head s
h
	​

, estimate its empirical survival function on held-out train-normal data,

p
h
	​

(x)=P
normal
	​

(S
h
	​

≥s
h
	​

(x)).

Then combine heads using a fixed rule such as:

minimum calibrated p;

Fisher combination;

equal-weight sum of −logp
h
	​

.

Finally calibrate the combined statistic again on independent normal data or by cross-fitting.

(b) Why it should help. Your heads are heterogeneous. Density NLL, Mahalanobis residual, trajectory surprise, and kNN distance can have completely different tail shapes. Mean/SD z-normalization is weak precisely in the extreme tail where WADI's low-FPR problem matters.

Tail calibration lets a modest absolute score become important when it is genuinely unusual under normal operation.

It could also stop SWaT's strong reconstruction head from being diluted by weaker latent heads.

(c) Fair evaluation. Use cross-fitting within train-normal so that empirical tails are not evaluated on the same observations used to fit the scorer. Fix the fusion rule across datasets. Evaluate both full AUROC and prespecified partial AUROC.

(d) Risk. Fusion cannot manufacture information that none of the heads contain. Empirical extreme-tail estimation can also be unstable, especially when targeting FPR 10
−3
 or lower. It should be a calibration improvement, not the main claimed innovation.

Priority: medium.

7. Make SWaT scientifically discriminative through the evaluation question, not by squeezing another 0.01 AUROC

This one is mostly an evaluation redesign, but I think it is essential.

(a) Modification. Keep your present difficult split as the primary preregistered result, but add one or both of the following uniformly across all datasets:

a double-hard subset: anomaly windows below train-normal-derived thresholds for both the six-stat univariate detector and LinRes;

raw partial AUROC / TPR at very low FPR, e.g. FPR ≤1% and possibly ≤0.1%, on the complete test set.

The double-hard definition should be fixed identically for WADI, HAI, and SWaT, regardless of how many examples survive.

(b) Why it helps. SWaT difficult AUROC of 0.960 versus LinRes 0.959 is not a meaningful contest, and max∣z∣=0.943 confirms that there is little remaining discrimination to measure. Worse, all 38 windows belong to one attack episode.

A stricter benchmark can answer the real question: are there SWaT anomalies that neither simple marginal nor linear cross-channel structure explains?

If the answer is “almost none,” that is an important finding rather than a failure.

(c) Fair evaluation. Define the rule before looking at method scores under that subset. Report the number of anomaly windows and number of distinct episodes surviving. If only one episode remains, explicitly state that no episode-level significance claim is possible.

Do not describe a 0.98 versus 0.96 numerical difference as a robust win when n
episodes
	​

=1.

(d) Risk. The stricter SWaT subset may contain almost nothing. But that result would substantiate your argument that SWaT is intrinsically close to a ceiling benchmark for this particular question.

Priority: essential for claims, even though it may not improve the headline number.

What I would implement first

The highest-value combined architecture would be:

VaDE representation → static latent-density heads + shrinkage full-covariance nearest-mode head + latent temporal innovation/transition head, with a small causal multiscale feature bank.

I would not initially add flows or raw-feature Isolation Forest. First determine whether the missing WADI signal can be recovered naturally from A9/A10 and covariance geometry. That produces a much cleaner paper.

A plausible score stack would be:

S=F(s
dens
	​

,s
near−fullcov
	​

,s
transition
	​

,s
innovation
	​

,s
resid
	​

),

where F is a fixed train-normal tail-probability fusion and the residual remains normal-gated.

That would also resolve a conceptual weakness in the current paper: A9/A10 would become actual model assumptions rather than acknowledged but unused motivation.

Candid verdict on the three datasets

HAI: already a defensible win. Do not destabilize this result chasing tiny additional gains. Any new component should preserve the significant advantage over AE and the large advantage over USAD/TranAD.

WADI: I think a genuine win is plausible. The current evidence strongly suggests that the remaining failure is not “density versus reconstruction” but snapshot versus trajectory/local geometry. Low-magnitude correlated onset/offset dips are exactly where latent innovation, transition surprise, multiscale context, and full-covariance geometry should have an advantage. Moving from 0.69 to a reproducibly higher result than IF is scientifically plausible, but significance will depend on the limited number of difficult attacks rather than just the numerical AUROC.

SWaT: a strong, defensible statistical win on the existing difficult subset is probably not achievable in a meaningful sense, regardless of model quality. With trivial max∣z∣=0.943, LinRes=0.959, LatAD=0.960, and all 38 difficult windows from one attack episode, the benchmark has essentially exhausted its discriminative information. You might raise LatAD to 0.98–0.99, but that would still not establish generalizable superiority because the effective episode sample size remains one.

So I would not set the scientific target as “statistically significant win on all three.” The stronger target is:

significant HAI win + independently significant or clearly larger WADI improvement attributable to trajectory modeling + explicit demonstration that SWaT is a ceiling case where sophisticated and trivial/simple methods are statistically indistinguishable.

That is more defensible than engineering a nominal 0.001–0.01 SWaT advantage and calling it a three-dataset victory.