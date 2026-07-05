# LatAD — Physically-Grounded Anomaly Detection for Massively-Multimodal CPS Normal Data

**Read the paper:** <https://apartsinprojects.github.io/LatAD/> (rendered from [`index.html`](index.html))

A proof-of-concept study on anomaly detection for predictive maintenance in transportation
cyber-physical systems (CPS). The premise: the hard problem is not characterising *anomalies* (which
are rare and unrepresentative) but modelling *normal* behaviour, which in CPS is **Massive, Implicit,
Imbalanced, Multimodal (MIIM)** — a very large number of unlabeled operating modes whose frequencies
span orders of magnitude. That structure induces three detector errors, and the framework here targets
each one.

## The framework

A joint variational latent model with a Gaussian-mixture prior (**VaDE**) discovers the operating
modes; on top of it three components each attack one MIIM error:

| | Component | Targets | Mechanism |
|---|---|---|---|
| **C1** | Physically-guided anomaly generation | calibration / ranking without labels | shuffle channels across modes → oracle-filter → tune difficulty ([`poc/generate.py`](poc/generate.py)) |
| **C2** | Basin-of-attraction test | false **positive** on rare-but-valid modes | latent gradient-descent; rare-valid = one stable basin, pocket = split ([`poc/component2.py`](poc/component2.py)) |
| **C3** | Cluster-based ensemble | false **negative** on over-interpolated pockets | per-mode reconstruction experts, min-over-experts ([`poc/component3.py`](poc/component3.py)) |

Plus a whitened (Ledoit-Wolf Mahalanobis) reconstruction score in [`poc/models_vade.py`](poc/models_vade.py).

## Headline results (synthetic MIIM benchmark)

- Joint latent+clustering **foundation AUROC 0.94**, full framework **0.997**, vs 0.74–0.87 for
  Isolation Forest / LOF / AutoEncoder.
- On physically-valid **dependency-violation faults**, Isolation Forest and a single AutoEncoder are
  **blind** (TPR 0.00 @ 5% FPR — they score these faults as *more normal than normal*); the framework
  detects **0.92**.
- **C2** lowers rare-mode false-positive rate 0.93 → 0.69 (in its dedicated configuration).
- **C3**: a single global reconstruction model scores AUROC 0.451 on pockets (over-interpolates);
  the per-mode ensemble scores 0.996.
- **Whitening is neutral** on this isotropic-noise synthetic (expected to help on real
  correlated-channel data — not yet demonstrated).

Full ablation, tables, and honest limitations are in the [paper](index.html).

## Run

```bash
python poc/run_seeds.py --seeds 5     # main benchmark, baselines vs foundation + C2
python poc/eval_generated.py          # C1 generated faults: standard detectors are blind
python poc/component3.py              # C3: single-global vs cluster ensemble on pockets
python poc/ablation.py --seeds 3      # each component, separately and combined
python poc/make_figures.py            # regenerate results/figure.png
```

Requires: `torch`, `scikit-learn`, `pyod`, `numpy`, `scipy`, `matplotlib` (CUDA optional).

## Data

- **Synthetic MIIM** generator ([`poc/data.py`](poc/data.py)) — controlled, ground-truth modes.
- **Real CPS datasets** are *not committed* (size / redistribution). SKAB and HAI 20.07/21.03 are
  fetched locally under `poc/datasets/` (git-ignored); **WADI/SWaT are access-gated** via the
  [iTrust request form](https://www.sutd.edu.sg/itrust/request-for-datasets/). A dataset-agnostic
  loader stub (`load_wadi`) is the single switch point.

## Status

Preliminary proof of concept on **synthetic data**. Real-testbed validation (WADI/SWaT/HAI) is the next
step. See the paper's *Limitations* section for the honest scope. Derived from the project
[executive summary](executive-summary.html).
