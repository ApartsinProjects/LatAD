"""Minimal example: load a LatAD checkpoint and score window features.

    python load_and_score.py ../checkpoints/vade_SKAB.pt

Requires `models_vade.py` (shipped in this folder) on the import path, plus torch and
numpy. Feature vectors must be window statistics of shape (n_windows, n_channels*6),
built the same way as training (see winfeat.window_features(..., "stats")).
"""
import sys, numpy as np, torch


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    v = ckpt["model"]                      # fully fitted VaDE-hard+resid object
    mu = ckpt["standardization"]["mu"]; sig = ckpt["standardization"]["sig"]
    return v, mu, sig, ckpt["config"]


def score(v, mu, sig, X):
    """X: raw window-statistic features (n, n_channels*6). Returns anomaly scores (n,)."""
    Xs = ((X - mu) / sig).astype(np.float32)
    return v.anomaly_score_hard(Xs, use_resid="auto", use_basin="auto")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../checkpoints/vade_SKAB.pt"
    v, mu, sig, cfg = load(path)
    print("loaded", cfg["dataset"], "| K", cfg["K"], "latent", cfg["latent_dim"],
          "| expects feature dim", mu.shape[0], "(=", mu.shape[0] // 6, "channels x 6)")
    # demo on random standard-normal features of the right width
    rng = np.random.default_rng(0)
    demo = rng.standard_normal((5, mu.shape[0])).astype(np.float32) * sig + mu
    print("demo scores:", np.round(score(v, mu, sig, demo), 3))
