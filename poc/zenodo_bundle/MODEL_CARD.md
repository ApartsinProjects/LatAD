# Model Card: LatAD VaDE-hard+resid checkpoints (WADI, HAI, SKAB)

Trained anomaly-detection models accompanying the paper *"Modeling Normal Is All You
Need: Joint Latent Clustering for Anomaly Detection in Multimodal Cyber-Physical Systems
and the Industrial IoT"* (Apartsin & Aperstein).

## What this is
One checkpoint per cyber-physical-system benchmark. Each is the reported main model,
**VaDE-hard+resid(auto)**: a Variational Deep Embedding (VAE with a Gaussian-mixture
latent prior) whose anomaly score is computed entirely in the latent (mixture density +
nearest-component negative log-likelihood), with two optional heads (a responsibility-
weighted whitened residual and a basin-agreement head) auto-gated on train-normal signals.
Reconstruction is dropped from scoring. All fits use train-normal windows only.

## Files
| File | Dataset | K | latent dim | k_density | Auto-gated heads active |
|------|---------|---|-----------|-----------|--------------------------|
| `checkpoints/vade_WADI.pt` | WADI (123-ch water distribution) | 20 | 10 | 80 | none (density only) |
| `checkpoints/vade_HAI.pt`  | HAI (59-ch HIL industrial control) | 40 | 16 | 80 | whitened residual |
| `checkpoints/vade_SKAB.pt` | SKAB (8-ch rotor/water circulation) | 16 | 6  | 40 | basin agreement (lambda 1.08) |

Common: window length 60, stride 30, per-channel window-statistic features, per-feature
standardisation on train-normal, 40 epochs, 8 warm-up epochs, **seed 0** (so the
checkpoints reproduce the paper's numbers exactly).

## Checkpoint contents (`torch.load`)
A dict with keys: `model` (the full fitted object; needs `code/models_vade.py` on the
import path to unpickle), `state_dict` (portable weights), `standardization` (`mu`,
`sig`), `config` (all hyperparameters), `resid_auto_on`, `basin_lambda`.

## Intended use
Reproducing and extending the paper's results; a starting point for CPS/industrial-IoT
anomaly-detection research. The score is raw and point-wise (no point adjustment).

## Inputs
Window-statistic feature vectors of shape `(n_windows, n_channels * 6)`, standardised with
the checkpoint's `mu`/`sig`. See `code/load_and_score.py` for the exact pipeline.

## Out-of-scope / limitations
Window-only: the trajectory assumptions A9-A10 (multiscale-temporal, path-dependence) are
not modelled here (future work). Trained on three specific testbeds; transfer to other
plants is untested. The F1/FPR operating point in the paper is a disclosed oracle;
only AUROC is threshold-free.

## Data
These checkpoints contain **learned parameters only, never raw data**. The training data
(WADI, HAI, SKAB) are third-party benchmarks obtained from their original providers
(WADI: iTrust, SUTD, on request; HAI: ETRI public release; SKAB: public repository) and
are **not** redistributed here.

## License
MIT (see `LICENSE`).

## Citation
Cite the accompanying paper and this Zenodo record (DOI on the record landing page).
