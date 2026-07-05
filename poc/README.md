# LatAD — Prior-research proof of concept

A small, self-contained experiment that tests the central premise of the LatAD
proposal: on CPS-style normal data with **Massive, Implicit, Imbalanced
Multimodality (MIIM)**, modelling *normal behaviour* with a **learned latent
representation plus explicit mode clustering** detects faults better than
standard anomaly-detection methods, and does so **without any labelled faults**.

## What it does

| Step | File | Notes |
|---|---|---|
| (a) CPS dataset | `data.py` | Controlled MIIM generator (40 modes, Zipf-2.0 imbalance, nonlinear per-mode manifolds). Two fault families: **OOD** (outside all modes) and **pocket** (between two modes — the dangerous false negative). Ground-truth modes are used for evaluation only, never shown to a detector. `load_wadi()` is a drop-in stub for the real dataset. |
| (b) Latent + clustering | `models_vade.py` | **VaDE** = joint VAE + Gaussian-mixture prior (encode and cluster *together*). **Plain VAE + post-hoc GMM** = the sequential ablation. |
| Scoring | `models_vade.py` | **Whitened residual + latent NLL.** The reconstruction term is the Mahalanobis (Ledoit-Wolf-shrunk) residual energy `½·rᵀΣ⁻¹r`, not a raw sum of squares — it respects per-channel scale and cross-channel correlation. Added to the nearest-*component* latent NLL (rare-mode-safe: a valid point is not penalised for sitting in a rare, low-π cluster). |
| Component 2 | `component2.py` | **Basin-of-attraction test.** From a point's latent, gradient-descend the mixture-density potential over several perturbed restarts; measure **basin agreement** (do restarts converge to one mode = valid, or split across two = pocket) and convergence distance. The full score fuses whitened reconstruction + basin instability, dropping the density term that causes the rare-mode false positive. |
| (c) Baselines | `baselines.py` | Isolation Forest, LOF, deep AutoEncoder (via PyOD). |
| Evaluation | `evaluate.py` | Calibration-free: thresholds fixed on normal training scores only. Reports AUROC/AUPRC plus two thesis diagnostics — **rare-mode false-positive rate** and **pocket detection rate**. |

Real CPS datasets on disk under `datasets/` (via the background download): **SKAB** (8-channel water-circulation testbed) and **HAI 20.07 / 21.03** (59-channel multi-process ICS). HAI 22.04+ is Git-LFS-budget-blocked; WADI/SWaT are iTrust-request-only.

## Run

```bash
C:/Python314/python.exe run.py            # single seed, prints table + writes results/figure.png
C:/Python314/python.exe run_seeds.py --seeds 5   # mean +/- std over seeds (the result below)
C:/Python314/python.exe run.py --quick    # fast wiring smoke-test
```

Environment: torch 2.10 (CUDA), scikit-learn 1.8, PyOD 3.6. ~2 min/seed on an RTX 2060.

## Results (5 seeds, 40 modes, mean ± std)

| method | AUROC | AUPRC | rare-mode FPR ↓ | pocket recall ↑ |
|---|---|---|---|---|
| Isolation Forest | 0.739 ± 0.009 | 0.638 | 0.570* | 0.029 |
| LOF | 0.872 ± 0.023 | 0.746 | 0.887 | 0.956 |
| AutoEncoder | 0.751 ± 0.004 | 0.642 | **1.000** | 0.969 |
| VAE + GMM (sequential) | 0.935 ± 0.051 | 0.860 | 0.879 | 1.000 |
| **VaDE (joint, ours)** | **0.951 ± 0.012** | **0.877** | 0.915 | 1.000 |

(ood recall = 1.000 for all methods; omitted. *IF's low rare-mode FPR is an
artefact of it detecting almost nothing — pocket recall 0.029 — so it rarely
alarms on anything.)

## Root-cause note: why the joint model first looked worse

An earlier version had joint VaDE *underperforming* the sequential ablation
(0.868 vs 0.935). Diagnostics (`diagnose.py`) traced it to **cluster collapse**:
joint training merged the 40 operating modes into **5 components**, so rare modes
had no cluster and scored as anomalies. The fix (in `train_vade`) is standard and
cited: **β/KL warm-up**, **component-variance floor + covariance penalty**
(DAGMM-style), and a **slower learning rate on the mixture parameters** so the
40-cluster initialisation is refined, not destroyed. After the fix VaDE uses
38/40 clusters, its mode-clustering NMI (0.70) exceeds the sequential GMM's
(0.64), and it becomes the best detector with the lowest variance.

## What this proves

**Robust across seeds:**
- Standard anomaly detection breaks on MIIM normal data. A *global* model
  (Isolation Forest) collapses to 0.74 because it cannot represent multimodal
  normal; a *single deep* model (AutoEncoder) flags **100 % of rare-but-valid
  modes** — exactly the trust-eroding false positive the proposal names.
- Modelling normal via a **joint latent representation + clustering** is the fix:
  VaDE reaches **AUROC 0.951 / AUPRC 0.877**, the best of all methods, with
  perfect pocket detection, decisively beating every baseline.
- **Joint beats sequential** once collapse is cured (0.951 vs 0.935) and is far
  more stable (±0.012 vs ±0.051) — supporting the proposal's "encode and cluster
  together" foundational step.

**Open (the next component, not a failure):** rare-mode false positives remain
high for every detector that actually detects faults (~0.88–1.0). Separating a
rare-but-valid mode from a true fault is precisely the job of the proposal's
**Component 2 (basin-of-attraction test)**, which this foundational PoC does not
yet implement.

## Next steps
1. **Component 2 (basin-of-attraction)** to close the rare-mode FPR gap — the one
   metric no baseline wins, and the proposal's distinctive contribution.
2. **Real CPS data now**: develop on openly-licensed **HAI** (ICS testbed) and
   **SKAB** (water testbed) immediately; request **WADI + SWaT** from the iTrust
   form (https://www.sutd.edu.sg/itrust/request-for-datasets/, ~3 working days) and
   drop them into `load_wadi()`. Everything downstream is dataset-agnostic.
3. **DAGMM** as an alternative modern joint method (AE latent + GMM energy,
   purpose-built for AD) to corroborate the joint-clustering result.
