# Backbone alternatives to VaDE for fine latent structure: adversarial analysis

Report-only. No model, checkpoint, or paper file touched. Written 2026-09-17 against the state of the
`_diagnostics` registry (E2, adaptive-density, coverage, cleanwadi runs).

## 0. What the question actually is

The question is NOT "which generative model has the finest latent structure". It is: given that

- the fixed high-K GMM density head already wins on HARD anomalies and every local-density variant loses
  (adaptive_density.log: WADI difficult-subset AUROC fixed 0.857 vs LOF 0.822; HAI 0.969 vs 0.960;
  SKAB control shows LOF removes the rare-normal penalty 1.33 -> 1.00 while costing overall AUROC),
- VaDE merges rare regimes into a dominant neighbour (E2), and the only measurable symptom on a real
  benchmark is a rare-normal false-positive tax (e2_fable_analyze_HAI: at matched 1% overall test-normal
  FPR, rare-regime normals fire at 5.1% vs common-regime normals at 0.0%; 100% of the top-1% scoring
  normals sit in rare regimes against a 19.6% base rate), while hard-anomaly AUROC is untouched
  (0.807 vs 0.808),

which backbone, if any, changes either the detection number or the triage capability enough to justify a
round-2 edit.

Two facts frame every answer below.

**Fact A (from lesson 3): the hard anomalies are globally far.** Any backbone whose contribution is
"score relative to the local neighbourhood" is a null or a loss for detection. Only changes that preserve
an absolute density scale can even be candidates for a detection gain.

**Fact B (from E2 census on HAI): the rare regimes are rare in an absolute sense.** Occupancy is Zipf
(slope -1.10); 13 of 40 regimes hold under 2% of the training mass, 6 hold under 0.1%, and the three
regimes that matter at test time (8, 19, 21) are 3x to 15x over-represented in test-normal relative to
train. A regime with 0.04% of 300k training windows is roughly 120 windows. No density estimator, of any
family, gives a tight density estimate from 120 points in a 16-d to 64-d latent; the question is only
whether the estimator gives that region a *code* (a name) rather than a *good density*.

Fact B is why the question splits cleanly: backbones can help with **naming** (triage), and essentially
cannot help with **scoring** (detection), because the scoring problem on rare regimes is a sample-size
problem, not a model-capacity problem.

## A. Ranked shortlist

Ranking criterion: (expected triage gain) x (probability it survives Fact A) / (round-2 cost).

| Rank | Option | What it fixes about VaDE collapse | Main risk |
|---|---|---|---|
| 1 | **Discrete regime codebook layered on top of the frozen VaDE latent** (k-means / VQ over z with a large code count, then a coverage test per code). Not a backbone swap. | Gives every occupied latent region a code regardless of pi; VaDE's merging of pi is irrelevant because the codebook is fitted on the embedding geometry, not on VaDE's responsibilities. | Code count and dead-code handling are new hyperparameters; a code is a partition, not a density, so it must stay off the detection path. |
| 2 | **Dirichlet-process / stick-breaking mixture on top of the frozen latent** (truncated DP-GMM, Blei and Jordan 2006 variational DP; Nalisnick and Smyth 2017 "Stick-Breaking Variational Autoencoders" if end-to-end). | Grows the number of components from data; a well-separated 120-point cluster gets its own component with an honest small weight instead of being absorbed. | On Zipf data the DP concentration alpha becomes the knob that decides whether the tail is split or merged; alpha is fitted on the same data that under-represents the tail, so it under-splits by construction unless prior-tuned. |
| 3 | **VQ-VAE backbone** (van den Oord, Vinyals, Kavukcuoglu 2017; VQ-VAE-2 Razavi et al. 2019; codebook-collapse fixes: EMA updates, dead-code restarts, or FSQ, Mentzer et al. 2023). | A rare regime gets a nearest code; no mixture weight to collapse. | (a) VQ-VAE has no density; scoring anomalies needs a separate prior (PixelCNN/transformer over codes) or a distance-to-code score, both of which are local measures and fall under Fact A. (b) VQ codebook collapse is the well-known dual of VaDE's collapse: unused codes die unless restarted, so the "no collapse" claim is not free. |
| 4 | **VampPrior VAE** (Tomczak and Welling 2018). | Prior = mixture of posteriors at learned pseudo-inputs; with K pseudo-inputs larger than the regime count, a rare regime can be pinned by a pseudo-input. | Pseudo-inputs are gradient-fitted to maximise ELBO, i.e. dominated by mass exactly like VaDE's pi. Same failure, different parameterisation. |
| 5 | **GMVAE** (Dilokthanakul et al. 2016; Jiang et al. 2017 VaDE is itself the ELBO-tightened cousin). | Nothing new relative to VaDE; same fixed-K Gaussian mixture prior with a categorical latent, and Dilokthanakul et al. report the same over-regularisation/cluster-merging. | Lateral move. Listed only to close the question. |
| 6 | **Normalizing flow latent density** (MAF, Papamakarios et al. 2017; IAF, Kingma et al. 2016; neural spline flows, Durkan et al. 2019), replacing the high-K GMM head or the VaDE prior. | Component-budget-free density, can in principle put mass on a thin tail region. | Flows are known to assign HIGH likelihood to out-of-distribution inputs (Nalisnick et al. 2019 "Do deep generative models know what they don't know?"; Kirichenko, Izmailov, Wilson 2020). This is the single worst property for a detector whose wins come from globally-far points. Also: a flow with 120 tail points learns the tail no better than a GMM. |
| 7 | **Hierarchical / ladder VAE** (Sonderby et al. 2016 LVAE; NVAE, Vahdat and Kautz 2020; Havtorn et al. 2021 use the hierarchy for OOD). | Multi-scale latents; a top-level coarse regime code plus fine local latents. | Two to three orders of magnitude more training cost and hyperparameters than the current VaDE; the OOD literature on LVAEs (Havtorn et al. 2021, "Hierarchical VAEs Know What They Don't Know") shows the gain comes from likelihood-ratio tests between levels, which is again a local/relative score (Fact A). |

### A2. In-place VaDE fixes (keep the backbone, stop the merging)

Two code facts from `models_vade.py` govern this whole subsection.

**Code fact 1: the detection score never reads VaDE's pi.** `fit_latent_density` fits a separate
sklearn diagonal GMM (K=80, reg_covar=1e-3) on the encoded means; `anomaly_score_hard` scores with
that GMM's `score_samples`. VaDE's mixture (pi_logit, mu_c, logvar_c) enters the score only through
(a) the encoder geometry it shaped during joint training, (b) the A8 responsibilities used to weight
per-mode residual precisions, and (c) the A3 basin argmax. So any fix to pi changes the HARD number
only through second-order effects on where the encoder places points. That is why E2, which changed
the pi treatment at score time, was a null: it operated on a quantity the head does not use.

**Code fact 2: four anti-collapse measures already ship.** KL warm-up (`beta` annealed over `warmup`
epochs), a DAGMM-style `cov_reg` penalty on 1/var, a hard variance floor (`logvar_floor = log 0.05`),
and mixture parameters trained at 0.1x the network learning rate from a 3-restart GMM initialisation.
The merging observed in E2 happens *despite* these, which means it is not the classic
"40 modes collapse to 5" optimisation failure those measures target; it is the ELBO doing what the ELBO
does on Zipf data, assigning a 0.04% regime to the nearest large component because the KL cost of a
thin component exceeds its likelihood gain at that sample count.

With those two facts, each proposed fix:

| Fix | What it does to the merging | Already present? |
|---|---|---|
| F1 Entropy floor / minimum-pi / component-preservation penalty on pi | Forces every pi_k >= pi_min. Keeps the component *alive* but does not move its mean to the rare regime; a live component with no data drifts to wherever the mixture-level gradient pushes it, typically the fringe of a dense component. Keeping a weight nonzero is not the same as allocating it. | No (only the variance floor, which is the covariance analogue). |
| F2 Load-balancing loss (MoE-style, Shazeer et al. 2017 / Fedus et al. 2022 Switch) on batch-mean gamma | Pushes batch-average responsibilities toward uniform. On Zipf data this is actively wrong: it forces 40 components to take 2.5% each, splitting the dense regimes into arbitrary halves and still giving no guarantee that a 0.04% regime gets a component (the balancing pressure is global, the rare regime is 1/60 of a balanced share). The `dagmm.py` entropy term (`l_ent`) in this repo is a softer version of the same thing. | Partially (DAGMM variant exists in the repo, not used in VaDE). |
| F3 Explicit component birth-death (split high-variance / low-likelihood components, kill empty ones, during training; Ueda et al. 2000 SMEM style) | Actually targets the mechanism: a merged rare regime shows up as excess variance or a bimodal residual inside one component, and a split step names it. This is the only in-place fix whose mechanism matches the E2 finding. | No. |
| F4 Dirichlet prior on pi with alpha < 1 | Sparsity prior: pushes pi toward *fewer* nonzero components. This is the opposite of what is wanted; alpha < 1 accelerates merging. With alpha > 1 it becomes F1. | No. |
| F5 Free-bits / per-component KL annealing (Kingma et al. 2016 free bits; Razavi et al. 2019 delta-VAE) | Free bits protect the *encoder posterior* from collapsing to the prior, i.e. posterior collapse of z. The VaDE merging is a *prior-side* mixture-weight collapse; free bits on the categorical KL (term_b in `loss`) would cap the cost of keeping gamma non-degenerate per point, which weakens the ELBO's pull toward assigning rare points to big components. Mechanism is plausible but indirect; the warm-up already does the annealing half. | Half (warm-up yes, free bits no). |
| F6 Higher K paired with F1/F3 | Raw higher K without allocation control gives more empty components (the user's own observation). With F3 it is the right dose: the split step decides where the extra components go. With F1 alone, more components pinned alive at pi_min all drift to dense fringes. | K is already a per-dataset setting. |
| F7 Student-t components (Peel and McLachlan 2000 t-mixtures; Abiri and Ohlsson 2020 for VAE priors) | Heavy tails make a large component *more* willing to absorb a distant thin cluster (a t with nu=3 assigns non-negligible likelihood 5 sigma out), so a rare TIGHT regime is swallowed *more* easily, not less. The intent (robustness to the broad neighbour) is inverted: the broad neighbour is the one that gains reach. Reject on mechanism. | No. |
| F8 Two-stage: coarse VaDE fit, then split high-variance components post hoc (offline F3) | Same mechanism as F3 without touching training. A split component gives a name and a local Gaussian; the name is what triage needs. | No, but trivially addable offline. |
| F9 Keep VaDE coarse; move all fine handling into the density head (tune k_density, reg_covar, covariance type) | This is what the method already does. The head is a near-KDE (K=80 diag on a 10-d latent); k_density is the only knob with any scoring leverage, and the ablation in the paper already selected it. | Yes, this is the shipped design. |

Three-axis assessment of the in-place fixes:

- (i) HARD detection. **All null, by code fact 1.** None of F1 to F8 changes `latent_gmm`. The only
  path to a HARD change is via encoder geometry, and the direction of that change is unsigned: a fix
  that forces the encoder to keep a rare regime separated can equally pull a hard anomaly closer to
  a rare-regime component. The one measured instance of a training-side change (the K and warm-up
  sweep now running) is the right way to find out, and its result should be read before any of
  F1 to F8 is tried; if K/warm-up move HARD by less than the seed spread, the in-place fixes will
  too. F9 (head tuning) is the only lever that touches the score directly, and it is already tuned.
- (ii) TRIAGE / rare-regime robustness. F3 and F8 (split) are the only fixes whose mechanism matches
  what E2 found; F5 is a plausible indirect help; F1 and F6 keep weights alive but do not place them;
  F2, F4, F7 are wrong-signed on Zipf data. F8 is a post-hoc partition of the latent, which makes it
  the same object as option B1 (a codebook) with Gaussian rather than Voronoi cells.
- (iii) Cost. F1, F2, F4, F5, F7: small code edits but each is a *training change*, so every stored
  checkpoint, every five-seed baseline row and the A3/A8 gates must be re-run and re-verified; that
  is medium cost in a round-2 revision even when the diff is ten lines. F3 and F6-with-F3: medium to
  large (a split/merge schedule inside the training loop is a new algorithm with its own ablation).
  F8 and F9: small (offline, no retraining).

The decisive comparison against the backbone alternatives: an in-place fix is cheaper than a swap, but
it is not cheaper than a post-hoc layer, and on axis (i) it has the same expected value (null) while on
axis (ii) the post-hoc split/codebook (F8 = B1) delivers the same partition without retraining anything.
The in-place fixes therefore sit strictly between the swap and the post-hoc layer on cost, and tie the
post-hoc layer on benefit, which dominates them out.

Added options not on the requested list, evaluated and rejected:

- **Bayesian GMM with sparsity-promoting Dirichlet prior on pi (sklearn BayesianGaussianMixture,
  weight_concentration_prior small)**: this is the finite-K approximation of option 2 and is what one
  would actually run in a pilot; it is not a separate idea.
- **Prototype / memory networks (MemAE, Gong et al. 2019)**: a memory of prototypes over the latent.
  Same behaviour as option 1 (a codebook) but entangled with reconstruction, which LatAD deliberately
  does not score on. Dominated by option 1.
- **Density-ratio / "known unknowns" split via an auxiliary open-set classifier**: needs labelled
  regimes; the paper's setting is unsupervised. Out of scope.

## B. The three blunt questions, per option

Notation: HARD = difficult-subset AUROC on WADI/HAI/SWaT, the number the fixed GMM head currently owns.
TRIAGE = the proposed second label (under-represented valid regime vs fault), measured by
rare-normal-vs-anomaly separation and rare-normal FPR at matched overall FPR (the E2 metrics).

### B1. Discrete codebook on top of frozen VaDE latent
- (i) HARD: **null by design.** It never touches the score. This is the point.
- (ii) TRIAGE: **plausibly yes, and it is the only option whose success can be checked without
  retraining anything.** A high-scoring window whose latent lands in a code that (a) has nonzero
  training occupancy, (b) has low whitened residual in A8, and (c) is entered by a temporally coherent
  block of neighbours, is a rare-valid-regime candidate. The HAI regime-21 block is the ready-made
  positive witness (train share 0.4%, test-normal share 6.5%). Codebook resolution is decoupled from
  VaDE's pi, so merging does not propagate.
- (iii) Cost: **small.** Post-hoc, offline, CPU, reuses cached latents.

### B2. DP / stick-breaking mixture on top of frozen latent
- (i) HARD: null if kept off the score path. If it REPLACES the high-K GMM as the density head, the
  expected outcome is a null-to-slight-loss: a DP posterior with a small alpha yields fewer effective
  components than K=high, so density in dense regions becomes coarser, and the hard-anomaly margin is
  a global-distance effect that is insensitive to the component count in the first place (the fixed
  high-K head is already a near-KDE).
- (ii) TRIAGE: yes in principle, and it produces a per-code *weight* which the plain codebook does not;
  but on Zipf occupancy, variational DP fits are known to prune components with tiny responsibility
  mass (the "rich get richer" behaviour is the stick-breaking prior itself). The 0.04%-mass regimes are
  the ones most likely to be pruned, i.e. the same failure as VaDE, softened. Only a fixed
  concentration prior tuned to expected regime count (not fitted) avoids it, at which point it is a
  codebook with soft assignments.
- (iii) Cost: **small** as a post-hoc layer (sklearn BayesianGaussianMixture with dirichlet_process
  prior); **large** as an end-to-end SB-VAE swap (Nalisnick and Smyth 2017 uses Kumaraswamy
  reparameterisation, needs its own training recipe and a new ablation table).

### B3. VQ-VAE backbone
- (i) HARD: **expected loss.** VQ-VAE gives no likelihood. The two available scores are distance to
  nearest code (a local measure, Fact A applies: a globally-far point and a locally-off point both get
  "large distance to nearest code" but with no calibrated global scale, and the codebook's Voronoi
  cells are unbounded so a point 10 sigma out and 3 sigma out land in the same cell) or a learned
  autoregressive prior over code sequences (rewrites the detector as a sequence model; not the method).
  Everything that makes the high-K GMM head win would have to be re-added on top.
- (ii) TRIAGE: yes, but no better than B1, because B1 IS a VQ layer, fitted after the fact, on a latent
  that already has the regime geometry. End-to-end VQ only adds the possibility that the encoder
  reorganises the latent around codes, and adds codebook collapse as a failure mode to manage.
- (iii) Cost: **large.** New backbone, new training recipe, all ablations (A3, A8, community
  factorisation) re-run, all five-seed baselines re-compared, paper method section rewritten.

### B4. VampPrior
- (i) HARD: null. The head still needs a density; the VampPrior changes the prior used for the ELBO, not
  the post-hoc high-K GMM on the aggregate posterior. If the VampPrior itself were the score, it is a
  K-component mixture at learned pseudo-inputs, i.e. a GMM with tied covariances and worse
  optimisation; no mechanism for beating the current head.
- (ii) TRIAGE: no. Pseudo-inputs are chosen by ELBO gradient, which is mass-weighted. There is no
  pressure to park a pseudo-input on a 0.04% regime; Tomczak and Welling report pseudo-inputs
  converging to prototypical (dense) inputs.
- (iii) Cost: medium (backbone retrain, prior swap only) for zero expected return.

### B5. GMVAE
- (i) HARD: null (same object as VaDE up to ELBO term arrangement).
- (ii) TRIAGE: no; documented cluster merging in the original paper.
- (iii) Cost: medium. Reject.

### B6. Normalizing flow as density head or as latent prior
- (i) HARD: **expected loss, and a specific one.** The flow OOD failure (Nalisnick et al. 2019;
  Kirichenko et al. 2020: flows model local pixel/feature correlations rather than semantic content,
  and assign high likelihood to simpler inputs) is the exact inverse of what LatAD needs. A sensor
  window during a stuck-at attack is *smoother* than normal operation; a flow density will tend to rate
  it as more likely. The GMM head cannot make this mistake because its density is a function of
  distance to training mass, not of input complexity.
- (ii) TRIAGE: no. A flow gives a scalar density, not a partition; there is no code to name a regime
  with. A rare regime under a flow is just a region of moderate density, indistinguishable from the
  fringe of a common regime.
- (iii) Cost: medium for the head swap, large if the prior is swapped. Reject on (i) alone.

### B7. Hierarchical / ladder VAE
- (i) HARD: null-to-loss. The published OOD gains for hierarchical VAEs (Havtorn et al. 2021) come from
  a likelihood-ratio between hierarchy levels, which normalises away the global-distance signal in the
  same way LOF did (lesson 3). Using the top-level latent alone with a GMM head reduces to the current
  method with a worse-trained encoder.
- (ii) TRIAGE: in principle the top level is a regime code; in practice it is a continuous latent and
  needs a codebook on top, i.e. B1 again after a far more expensive training run.
- (iii) Cost: large. Reject.

Summary grid:

| Option | HARD detection | Triage | Round-2 cost |
|---|---|---|---|
| B1 codebook on frozen latent | null (untouched) | plausible, testable now | small |
| B2 DP mixture on frozen latent | null | plausible, worse tail behaviour than B1 | small (post-hoc) / large (SB-VAE) |
| B3 VQ-VAE backbone | loss | no better than B1 | large |
| B4 VampPrior | null | no | medium |
| B5 GMVAE | null | no | medium |
| B6 flows | loss (OOD likelihood failure) | no | medium/large |
| B7 ladder VAE | null/loss | B1 after an expensive detour | large |
| F1 min-pi / entropy floor (in-place) | null (head never reads pi) | keeps weight alive, does not place it | medium (retrain + re-verify all rows) |
| F2 load-balancing loss (in-place) | null | wrong-signed on Zipf occupancy | medium |
| F3 birth-death split in training (in-place) | null | yes, mechanism matches E2 | medium/large |
| F4 Dirichlet alpha<1 (in-place) | null | wrong-signed (sparsity accelerates merging) | medium |
| F5 free bits on categorical KL (in-place) | null | indirect, plausible | medium |
| F6 higher K + F1/F3 (in-place) | null | only with F3 | medium/large |
| F7 Student-t components (in-place) | null | wrong-signed (heavy tails absorb more) | medium |
| F8 post-hoc split of high-variance components | null (untouched) | yes, same object as B1 | small |
| F9 tune the density head only | already tuned; the only lever on the score | no partition, no triage | small |

Reading the grid by the four routes the coordinator asked to choose among:

- **Backbone swap** (B3 to B7): axis (i) null-or-loss, axis (ii) no better than a post-hoc partition,
  cost large. Dominated.
- **In-place VaDE fix** (F1 to F7): axis (i) null by code fact 1, axis (ii) only F3 has the right
  mechanism, cost medium because every training change re-opens every reported number. Dominated by
  the post-hoc layer on cost at equal benefit.
- **Head-only change** (F9): the only route that touches the score, already exhausted by the paper's
  own k_density ablation and by lesson 3 (every local-density variant lost). Nothing left to tune that
  has a stated mechanism for beating the current head on globally-far anomalies.
- **Post-hoc partition for triage only** (B1 = F8, with B2 as its soft variant): axis (i) untouched by
  construction, axis (ii) the only route with a testable positive witness (HAI regime 21), cost small.
- **Do nothing**: see section D.

## C. Verdict and one concrete next step

**Chosen route: post-hoc partition for triage only (B1 / F8). Not a backbone swap, not an in-place
VaDE fix, not further head tuning.** The reasoning, route by route:

- Backbone swap: every end-to-end alternative either (a) cannot beat the fixed GMM on hard anomalies
  because the hard anomalies are globally far and the alternative scores locally (VQ, ladder-VAE ratio
  tests), (b) has a documented failure mode that is the exact inverse of the detection need (flows), or
  (c) reproduces VaDE's mass-driven merging under another name (VampPrior, GMVAE, DP with fitted alpha).
- In-place VaDE fix: the detection head never reads VaDE's pi (code fact 1), so no pi-side fix has a
  first-order path to the HARD number; four anti-collapse measures already ship (code fact 2) and the
  merging survives them because it is the ELBO's correct answer on Zipf occupancy, not an optimisation
  failure; and every training-side change re-opens every stored checkpoint, five-seed baseline row and
  A3/A8 gate for re-verification, which makes a ten-line diff a medium-cost revision item. The one fix
  with the right mechanism (F3, split/merge) has a post-hoc twin (F8) that delivers the same partition
  for free. If the running K/warm-up sweep shows HARD moving by less than the seed spread, that is the
  empirical confirmation that training-side changes are null on detection, and F1 to F7 are closed.
- Head-only: the shipped high-K GMM is already the tuned head; lesson 3 closed the local-density
  direction, and no remaining knob (k_density, reg_covar, covariance type) has a stated mechanism for
  moving a globally-far anomaly's score.

The finer-structure need is a naming need, not a density need, and naming is a partition of the latent,
which can be done post hoc: a **regime codebook (or a post-hoc split of high-variance VaDE components)
over the frozen VaDE latent, used only to produce the triage label, with the detection score left
byte-identical.** This keeps every reported number in the submitted paper unchanged (nothing in the score
path moves), adds one figure/table for the triage capability, and is defensible as a round-2 addition
rather than a method change.

### Recommended pilot (offline, CPU, no retraining, one script)

Inputs: cached HAI latents for train and test (the same tensors E2 consumed, `e2_fable_HAI.pt`), the
existing per-window score, the A8 whitened residual, the test labels, and the E2 regime map (the 40
regime ids with their train and test-normal occupancies).

1. Fit a codebook of M codes on the training latent (k-means with M in {64, 128, 256}; report all
   three; no tuning on test). Record per-code training occupancy n_m.
2. For each test window compute: code id, code occupancy n_m, A8 residual, and a run-length of
   consecutive windows sharing the code (temporal coherence).
3. Define the triage label on windows above the detection threshold at the paper's operating point:
   RARE-VALID if n_m >= n_min (say 20 training windows), residual below the A8 gate, and run-length
   >= L (say 10); FAULT otherwise. Both constants are stated before looking at results.
4. Invariants stated in advance:
   - The detection score path is untouched: overall and difficult-subset AUROC must reproduce the
     stored numbers exactly (to the last digit). If they do not, the pilot has a bug.
   - The regime-21 test-normal block (train share 0.4%, test share 6.5%) must be labelled RARE-VALID
     at a rate well above the labelled-anomaly windows; the E2 census identifies it in advance.
   - A degenerate codebook (M = 1) must label nothing RARE-VALID, and the label must not depend on the
     seed of k-means beyond a stated tolerance across 5 seeds.
5. Report: (a) among alarms at the operating point, the fraction of rare-regime false positives that
   receive RARE-VALID; (b) the fraction of true anomalies that receive RARE-VALID (the triage's own
   false-negative rate, which must stay small: the win condition is roughly "reclassify >= half of
   the rare-normal FPs while mislabelling <= 5% of labelled anomalies"); (c) the same on SKAB and
   miim_gen as witnesses where rare regimes are synthetic and known.
6. Compare three partitions under the same label rule and the same pre-stated constants:
   (P0) VaDE's own responsibility argmax as the code (the free baseline; K components);
   (P1) F8: VaDE components with any component whose within-component latent variance exceeds
   2x the median split by a 2-component GMM, applied recursively at most twice (so at most 4K codes);
   (P2) the k-means codebook of step 1.
   If P0 already clears the bar on (a) at equal (b), no new object is needed and the triage label is
   built from VaDE's own components; if P1 clears it and P2 adds nothing, ship the split (it keeps the
   paper's own mixture as the vocabulary); only if P2 is needed does a separate codebook enter the
   paper. This ordering keeps the added machinery at the minimum the witness requires.

Agent execution estimate: about 20 to 40 minutes of tool calls, CPU only, one script, no cloud cost. It
does not compete with the running VaDE sweep for the GPU and can be launched after the sweep finishes if
CPU contention matters.

Decision rule after the pilot: adopt the triage label into the paper only if invariants hold and (a)/(b)
clear the pre-stated bar on HAI and on at least one witness; otherwise the whole line stays in the
registry as a design note, alongside E2.

**Pilot outcome (2026-09-17, see `fable_triage_pilot.md`): FAILS the pre-stated bar.** Detection path
reproduced to the digit on all 5 seeds; the degenerate control labelled nothing. Best partition (VaDE
argmax) reclassified >= 50% of rare-normal FPs on 3 of 5 seeds only, and labelled 6.1% of anomalies
(8-10% on two seeds) RARE-VALID; those are HARD attack windows that land in the rare regime's component
with low residual, indistinguishable from rare-valid normals on every guard statistic. Registry only.

## D. The sharpest case for doing none of this

The three real benchmarks do not measure this. On WADI and SWaT the rare-regime structure is absent or
not exercised (lesson 4), and on HAI the entire measurable effect is a rare-normal false-positive
surplus at low FPR that does not move hard-anomaly AUROC at all (0.807 vs 0.808). A triage label whose
only positive witness is one HAI regime block plus two synthetic-or-toy datasets (SKAB, miim_gen) is a
capability the paper cannot evaluate at the standard its main results are held to. A round-2 reviewer
who asked for the paper's claims to be tightened will read a new capability with a one-dataset witness
as scope creep, and the wins-only rule says a result that is positive on one real block and null
elsewhere does not reach the paper.

The stronger version of the counter-argument: the submitted method already handles the rare-regime
cost in the right place. The fixed high-K GMM head pays a rare-normal FP tax at the 1% operating point on
HAI, and that tax is what buys the hard-anomaly margin on all three benchmarks (lesson 3 shows that
removing it removes the margin). Reducing the tax by naming rare regimes is a second-order operational
convenience for an operator, not a detection result, and it is a convenience that only exists on plants
whose normal operation has a heavy-tailed regime distribution, which the paper cannot show is the
general case with the data it has. The paper's round-2 effort is better spent on what the reviewers
actually asked for than on a triage feature that no reviewer requested and no benchmark can score.

If the user wants the triage line pursued anyway, the pilot in section C is the minimum honest test and
its result decides the question in under an hour of agent time.

## References to be bibtest-checked before any of the above reaches the paper

- Jiang, Z., Zheng, Y., Tan, H., Tang, B., Zhou, H. (2017). Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering. IJCAI.
- Dilokthanakul, N. et al. (2016). Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders. arXiv:1611.02648.
- van den Oord, A., Vinyals, O., Kavukcuoglu, K. (2017). Neural Discrete Representation Learning. NeurIPS.
- Razavi, A., van den Oord, A., Vinyals, O. (2019). Generating Diverse High-Fidelity Images with VQ-VAE-2. NeurIPS.
- Mentzer, F., Minnen, D., Agustsson, E., Tschannen, M. (2023). Finite Scalar Quantization: VQ-VAE Made Simple. arXiv:2309.15505.
- Nalisnick, E., Smyth, P. (2017). Stick-Breaking Variational Autoencoders. ICLR.
- Blei, D. M., Jordan, M. I. (2006). Variational Inference for Dirichlet Process Mixtures. Bayesian Analysis 1(1).
- Tomczak, J. M., Welling, M. (2018). VAE with a VampPrior. AISTATS.
- Papamakarios, G., Pavlakou, T., Murray, I. (2017). Masked Autoregressive Flow for Density Estimation. NeurIPS.
- Kingma, D. P. et al. (2016). Improved Variational Inference with Inverse Autoregressive Flow. NeurIPS.
- Durkan, C., Bekasov, A., Murray, I., Papamakarios, G. (2019). Neural Spline Flows. NeurIPS.
- Nalisnick, E., Matsukawa, A., Teh, Y. W., Gorur, D., Lakshminarayanan, B. (2019). Do Deep Generative Models Know What They Don't Know? ICLR.
- Kirichenko, P., Izmailov, P., Wilson, A. G. (2020). Why Normalizing Flows Fail to Detect Out-of-Distribution Data. NeurIPS.
- Sonderby, C. K., Raiko, T., Maaloe, L., Sonderby, S. K., Winther, O. (2016). Ladder Variational Autoencoders. NeurIPS.
- Vahdat, A., Kautz, J. (2020). NVAE: A Deep Hierarchical Variational Autoencoder. NeurIPS.
- Havtorn, J. D., Frellsen, J., Hauberg, S., Maaloe, L. (2021). Hierarchical VAEs Know What They Don't Know. ICML.
- Gong, D. et al. (2019). Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection. ICCV.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR. (load-balancing loss, F2)
- Fedus, W., Zoph, B., Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. JMLR 23. (load-balancing loss, F2)
- Ueda, N., Nakano, R., Ghahramani, Z., Hinton, G. E. (2000). SMEM Algorithm for Mixture Models. Neural Computation 12(9). (split-merge EM, F3/F8)
- Peel, D., McLachlan, G. J. (2000). Robust Mixture Modelling Using the t Distribution. Statistics and Computing 10. (t-mixtures, F7)
- Abiri, N., Ohlsson, M. (2020). Variational Auto-Encoders with Student's t-Prior. ESANN 2019 / arXiv:2004.02581. (Student-t VAE prior, F7; verify venue)
- Zong, B. et al. (2018). Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection. ICLR. (DAGMM, source of the cov_reg penalty already in train_vade)
