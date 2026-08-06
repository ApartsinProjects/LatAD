# LatAD trained checkpoints (WADI, HAI, SKAB)

Trained anomaly-detection models for the paper *"Modeling Normal Is All You Need:
Joint Latent Clustering for Anomaly Detection in Multimodal Cyber-Physical Systems
and the Industrial IoT"* (Apartsin & Aperstein).

## Contents
- `checkpoints/vade_{WADI,HAI,SKAB}.pt` - one fitted VaDE-hard+resid(auto) model per dataset.
- `code/models_vade.py` - model + scoring definitions (needed to unpickle the checkpoints).
- `code/winfeat.py` - window-feature extraction used at train/inference time.
- `code/load_and_score.py` - minimal load-and-score example.
- `MODEL_CARD.md` - architecture, per-dataset config, intended use, limitations.
- `zenodo.json` - deposit metadata. `LICENSE` - MIT.

## Quick start
```bash
pip install torch numpy scikit-learn scipy
cd code
python load_and_score.py ../checkpoints/vade_SKAB.pt
```

## Reproducibility
Every checkpoint is trained with seed 0 and the exact configuration in `MODEL_CARD.md`,
so it reproduces the paper's reported numbers. Checkpoints hold learned parameters only;
the WADI/HAI/SKAB datasets are third-party benchmarks obtained from their original
providers and are not redistributed here.
