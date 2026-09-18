# Discrete sensors in LatAD: audit, options, verdict

Report-only analysis (no model, paper or checkpoint touched). Evidence: code read of `winfeat.py`,
`models_vade.py`, `onehot_filter.py`, `eda_real.py`, `build_scores_table.py`, `sota_bundle/modal_experts.py`,
paper `IoT2.html` (A8 rows and section 5.2), plus one read-only numpy smoke on the cached standardized arrays
(`sota_bundle/wadi_clean_*.npy`, `SWaT_*.npy`, `ds/HAI_*.npy`; script in the session scratchpad,
`discrete_smoke.py`, W=60 / stride 30, DMAX=6, same window-label rule as `eda_real.load`).

## A. Current state: how a valve state flows through the pipeline today

**There is no channel typing anywhere in the LatAD path.** The only discreteness tests in the repo are in
`onehot_filter.py` (the LinRes baseline/filter, `DMAX = 6`) and in the diagnostic
`_diagnostics/miim_matrix_measure.py` (`n_discrete_le10vals`, a measurement, not a model input). Every LatAD
stage treats every channel as a continuous real-valued signal:

1. `eda_real.load`: per-channel z-score on train-normal with `sd + 1e-8`; WADI/WADI_clean clipped at +-10, HAI and
   SWaT unclipped. A channel that is constant in train has `sd = 0`, so any test flip becomes |z| = 1e8+ (clipped
   to 10 on WADI, left at 1e8 to 1e10 on HAI/SWaT).
2. `winfeat.feat_stats`: six per-channel window statistics (mean, std, min, max, last-first, range) with no
   awareness of type. For a two-state channel the window mean IS the state fraction (up to an affine map), std
   is `sqrt(p(1-p))` times a constant, and min/max say which states were visited; for three states the (mean, std)
   pair still pins down the two free fractions; for 4 to 6 states the block is a lossy summary of the fractions.
3. `build_scores_table.build` (the reported null expert): the window features are standardized AGAIN
   (`Xtr0.std(0) + 1e-8`) with NO clip. For the 30 WADI channels that are constant in train, all six window
   statistics are constant in train, so the second `sig` is 1e-8 and a flip in test lands at the model input
   as a feature-z of ~1e9 even though the raw stream was clipped at 10. The community experts
   (`modal_experts.py` line 76) do clip at +-10 after their own second standardization, so the two LatAD paths
   see different inputs for exactly these channels.
4. `models_vade.VaDE`: MLP encoder, Gaussian-mixture latent prior, decoder trained with MSE (a Gaussian
   likelihood on every feature, including state fractions). The reported score (`anomaly_score_hard`) is the
   high-K diagonal GMM density on the latent plus the nearest-component NLL; reconstruction is dropped; the
   whitened-residual head (the paper's A8 item) is auto-gated and fires on HAI and SWaT, not on WADI.

So, bluntly: a valve state is z-scored as a continuous variable, summarized by moments, re-standardized, encoded
by an MLP and scored by a latent Gaussian mixture. Nothing in the code knows it is a state. The MSE decoder is a
mis-specified likelihood on a fraction, but since the reported base score drops reconstruction, the discrete
channels reach the score only through the encoder's latent, where the K=80 density head is flexible enough that
"Gaussian on a fraction" is not, by itself, the mechanism that loses detection.

**The A8 claim versus the code.** The paper's A8 row (IoT2.html line 309-311) says "Mixed signals
(heterogeneous, typed channels): the bus carries sensors, actuators, discrete states and setpoints, each with its
own range, resolution and correlated noise" and gives as the design response "per-feature standardization on
train-normal; a whitened (Mahalanobis) residual". Section 4 then states A8 is "always active" through
standardization and "auto-gated" through the residual head. What is implemented is the SCALE half of A8
(range, noise covariance). The TYPE half (discrete states, resolution) is named in the assumption text but has
no realization: no typing, no categorical likelihood, no state features. A reviewer who reads the A8 row and then
section 5.2 ("six per-channel statistics ... concatenated") can catch this in one sentence. This is a gap between
claim wording and code, not a bug in the numbers; the honest fix is either to narrow the A8 label to what is
realized ("heterogeneous scale and noise") or to implement typing (options below).

**Smoke numbers (per-channel unique train values, DMAX=6).**

| dataset | constant in train | discrete 2..6 states (3+ states) | continuous | constants that flip in test | discrete channels with train-unseen states |
|---|---|---|---|---|---|
| WADI_clean (122 ch) | 30 | 26 (10) | 66 | 5 (`2_MCV_007_CO`, `1_P_006_STATUS`, `1_MV_002/003_STATUS`, `2_PIC_003_SP`) | 0 |
| SWaT (51 ch) | 8 | 18 (6) | 25 | 2 | 0 |
| HAI (59 ch) | 8 | 1 (0) | 50 | 1 | 0 |

Feature-z actually received by the null expert (after the second standardization), max over test: WADI constant
block 1e9, SWaT constant block 1e16 and discrete block 1e10, HAI constant block 1e17. On WADI, 12 of the 43
"difficult" anomalies (maxz <= 6 on the clipped raw mean, i.e. a partial-window STATUS flip) carry a |feature-z|
> 50 at the model input; 1 of 519 test normals does. On SWaT and HAI the constant-channel flips are all in the
EASY bucket already (unclipped raw z is 1e8, so maxz screens them), so the re-inflation there changes no stratum.
HAI has a separate, continuous-channel version of the same effect (40/167 difficult windows with |feature-z| > 50
from std/min/max features of continuous channels, 18/14167 normals); that is a scale issue, not a typing one, and
is out of scope here.

## B. Options, three-axis assessment

Axes: (i) plausible gain on HARD-anomaly detection; (ii) strengthens A8 / difficulty stratification /
correctness; (iii) cost in a round-2 revision (agent execution time, not human time).

**(1) One-hot / state-fraction features for discrete channels, fed into VaDE.**
(i) Null expected. For 2-state channels the mean block already IS the state fraction; for 3-state channels
(mean, std) already determine it. WADI has only 10 channels with 3+ states, SWaT 6, HAI 0. Detection on the
difficult bucket does not move for HAI (0 such channels) and SWaT (LatAD 0.95 already ties LinRes-onehot 0.949).
On WADI the LinRes numeric-to-onehot lift (0.614 -> 0.723, quoted this session; I could not find the run log)
is confounded: `onehot_filter.build_feats` standardizes the numeric mean by its train std but leaves one-hot
fractions unscaled in [0, 1], so the two representations weight residuals differently (a rarely flipping
channel gets a huge numeric residual and a bounded one-hot residual). Part or all of the lift can be that
re-weighting, not "discrete-state signal". Not established, so do not carry it into the paper as evidence.
(ii) Modest: it lets the A8 row say "state fractions for discrete channels" truthfully. It does not fix the
1e9 re-inflation (constant channels still have zero train variance in any representation).
(iii) Small: one branch in `winfeat`, feature count changes, all tables regenerate (5 seeds x 3 datasets of
VaDE on CPU, roughly 1 to 2 hours wall-clock), plus the SOTA bundle is unaffected (it consumes raw arrays).

**(2) Separate categorical / multinomial likelihood head for discrete channels, combined with the continuous density.**
This is the principled mixed-type VAE (HI-VAE, Nazabal et al. 2020; VAEM, Ma et al. 2020): per-type decoder
likelihoods, categorical for states. (i) Null on the reported score: the reported base score does not use the
decoder likelihood at all (reconstruction demoted), so a categorical decoder changes only the training signal
for the encoder. On WADI the residual head is gated off; on HAI there are no discrete channels to speak of. The
one place it could matter is SWaT's residual head, which is already at 0.95 on difficult. (ii) Strongest A8
story ("typed likelihoods"), and it is what a reviewer means by "typed channels". (iii) Largest cost: new decoder
heads, loss bookkeeping, per-type whitening in `fit_resid_head`, community experts on Modal need the same
change, every artifact regenerates. Several agent-hours plus a Modal re-run; not a round-2 item unless (i) is
shown non-null first.

**(3) Route discrete channels only through the residual / community structure, exclude from the Gaussian latent.**
(i) Likely negative on WADI: the near-constant STATUS channels are the attack carriers (EXPERIMENT_LOG line
99-102: dropping them collapses every detector to chance), and the residual head is gated OFF on WADI, so
excluding them from the latent removes them from the score entirely. On SWaT/HAI the community experts already
drop channels with window-mean std <= 1e-6 from the correlation graph (`modal_experts.py` line 47), so constant
channels are already outside the community latents. (ii) Weak; it makes A8 read as "we ignore discrete channels
in the mixture", which is not a strength. (iii) Cheap to prototype, expensive to justify. Not recommended.

**(4) STATUS / near-constant channel policy: drop vs clip vs treat-as-categorical (the CLIP finding).**
The mechanism is two-stage: raw clip at 10 (WADI only) followed by an unclipped second standardization with
`+1e-8` in `build_scores_table.py`. Drop is ruled out by the carrier finding. Clip-after-second-standardization
(what the community experts already do at +-10) is the consistent fix: it bounds the null expert's input to the
same range the community path sees. Treat-as-categorical is option (1)/(2) and does not remove the zero-variance
problem on its own. (i) Null on ranking: the affected windows are STATUS flips that every method already scores
at the top (analyze log: LatAD 1.000, AE 1.000 percentile on w204/w248/w490/w491); bounding them to 10 keeps
them at the top. (ii) Correctness: yes, it removes a null-vs-community input inconsistency and the 1e9 inputs
that make scores non-portable across seeds/hardware. (iii) Trivial: one `np.clip` line plus a table regeneration.
It is cleanup, and it changes no headline number by more than seed noise (to be confirmed by the regeneration,
not assumed).

**(5) Leave modeling alone; add a discrete-state TRIVIAL detector (T2) to the difficulty stratification.**
T2 as prototyped in the smoke: per window, max over typed (constant or <= 6-state) channels of the fraction of
rows in a train-unseen or train-rare (< 0.1 % of train rows) state; threshold at the train p99.

| dataset | T2 train-p99 thr | anomalies flagged | of which currently DIFFICULT | test normals flagged |
|---|---|---|---|---|
| WADI_clean | 0.0167 | 24 / 56 | 14 / 43 | 5 / 519 |
| SWaT | 0.0 | 20 / 182 | 0 / 38 | 0 / 905 |
| HAI | 0.0 | 35 / 652 | 0 / 167 | 0 / 14167 |

On WADI the 14 T2-flagged difficult windows include all 12 windows whose model input carries the 1e9 feature-z,
i.e. T2 identifies exactly the partial-flip STATUS windows that the raw-mean maxz axis lets through as
"difficult" while the model sees them as trivially separable. On SWaT and HAI T2 moves nothing (the flips are
already easy), so the stratification becomes uniform across datasets by construction. (i) No detection gain and,
on WADI, a probable LOSS in the reported difficult-AUROC for LatAD and AE: the 12 to 14 windows leave the
difficult set, and they were scored at percentile ~1.0 by LatAD/AE but at 0.39 to 0.43 by LinRes on two of them
(analyze log w490/w491), so the 29-window difficult set will most likely widen LinRes's lead. State this before
running it. (ii) This is the correctness win: the paper's "difficult = not caught by a trivial univariate rule"
definition is currently violated by 12 windows on WADI that a one-line rule catches. It also gives the A8 row a
truthful, implemented sentence: "discrete channels are typed for the difficulty stratification (T2)". (iii)
Small: T2 is ~20 lines in `build_scores_table.py` (store `t2`, `t2_thr` alongside `maxz`), `slice_scores.hard_mask`
gains one filter, difficult-bucket tables regenerate from the existing `scores_*.npz` without retraining anything
(minutes of agent time, no Modal). Also note `easy` should then be defined as `maxz > thr OR t2 > t2_thr`.

## C. Verdict

What actually helps: nothing here is a detection gain on hard anomalies. The pattern of this session holds: the
discrete-channel evidence (LinRes one-hot lift) is confounded by residual re-weighting, the multi-state channels
are few, the 2- and 3-state channels are already represented by the moment block, and the datasets where discrete
channels are numerous (SWaT) show LatAD tied with the discrete-aware baseline on the difficult bucket.

What is cleanup: option (4), clip after the second standardization in the null-expert path so it matches the
community experts. Do it, expect no headline change, verify by regeneration.

The honest win is the difficulty-stratification correctness fix, option (5), and it comes with a cost on the
WADI-difficult table. Recommended single next step, sized as a small offline pilot (no retraining):

1. Add T2 to `build_scores_table.py` as a stored column (`t2`, `t2_thr`) and `t2` to `slice_scores.hard_mask`
   (`easy = maxz > thr | t2 > t2_thr`). Regenerate the difficult-bucket tables for all methods from the existing
   `scores_WADI_clean.npz` / `scores_SWaT.npz` / `scores_HAI.npz` (after the CLIP fix already identified, so the
   WADI table is clip=10 end to end).
2. Invariants stated in advance: SWaT and HAI difficult counts unchanged (38, 167); WADI difficult drops from 43
   to 29 (+-1 depending on the T2 rarity cut); the removed windows are a subset of the 14 T2-flagged ones and every
   method's difficult-AUROC on WADI moves DOWN or stays (they were top-scored windows). Any method whose
   difficult-AUROC goes UP on WADI is a bug.
3. In the same pilot, kill or confirm the LinRes confound in one pass: run `onehot_filter.build_feats` numeric
   with the mean feature left unscaled in [0,1] (bounded like the one-hot columns). If that recovers most of
   0.614 -> 0.723, the lift is re-weighting and the "discrete-state signal is real" sentence must not be written.
4. Paper wording: narrow the A8 row's response to what runs ("per-feature standardization and a whitened
   residual handle heterogeneous scale and noise; discrete channels are typed in the difficulty stratification
   (T2)"), and drop "typed channels" from the label unless option (2) is implemented.

Estimated agent execution: 30 to 60 minutes of tool time, CPU only, no Modal.

## D. The sharpest counter-argument

"The continuous Gaussian on discrete channels already works; leave it." The case: (a) information-wise, the
six-statistic block already carries the state fraction exactly for two-state channels and, through (mean, std),
for three-state ones; only 10 WADI and 6 SWaT channels have more states, HAI none. (b) The reported score is a
non-parametric-grade latent density (K=80 diagonal GMM) on top of a jointly trained encoder; the decoder's
Gaussian likelihood on fractions is not in the reported score, so the mis-specification is confined to a training
signal. (c) Empirically, the dataset with the most discrete channels (SWaT, 18 of 51) is where LatAD ties the
discrete-aware LinRes on difficult anomalies (0.95 vs 0.949); the dataset with the largest LatAD-vs-LinRes gap
(WADI, 0.734 vs 0.787, paired bootstrap CI [-0.142, +0.038]) has its LinRes advantage confounded by residual
scaling and by 12 STATUS-flip windows sitting in the difficult set, and once those windows are re-bucketed the
comparison changes for every method, not for LatAD alone. (d) The one real defect the discrete lens found is a
scale defect (zero-variance re-inflation), fixed by a clip, not by typing. Under this argument, options (1) to
(3) are effort with no expected hard-anomaly return, and the only actions with a return are the clip (correctness)
and T2 (stratification honesty). I agree with the counter-argument on modeling and disagree with it on wording:
the A8 row still promises typed handling the code does not do, and that sentence must change either way.

## References to run through bibtest later

- VaDE: Jiang, Zheng, Tan, Tang, Zhou. "Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering", IJCAI 2017.
- DAGMM: Zong et al. "Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection", ICLR 2018.
- HI-VAE: Nazabal, Olmos, Ghahramani, Valera. "Handling incomplete heterogeneous data using VAEs", Pattern Recognition 2020.
- VAEM: Ma, Tschiatschek, Turner, Hernandez-Lobato, Zhang. "VAEM: a Deep Generative Model for Heterogeneous Mixed Type Data", NeurIPS 2020.
- TABOR: Lin, Adepu, Verwer, Mathur. "TABOR: A Graphical Model-based Approach for Anomaly Detection in Industrial Control Systems", AsiaCCS 2018 (actuator states modeled as timed automata; the closest prior for typed actuator handling in ICS).
- GDN: Deng, Hooi. "Graph Neural Network-Based Anomaly Detection in Multivariate Time Series", AAAI 2021.
- Higher Criticism: Donoho, Jin. "Higher criticism for detecting sparse heterogeneous mixtures", Annals of Statistics 2004.
- Ledoit, Wolf. "A well-conditioned estimator for large-dimensional covariance matrices", J. Multivariate Analysis 2004.
