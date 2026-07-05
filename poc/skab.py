"""
SKAB loader (real CPS: Skoltech rotor/pump testbed, 8 sensors at ~1 Hz).

This is the first real-data adapter. SKAB is a time series, so we WINDOW it: each
example is a length-W window of the 8 channels, flattened to W*C features. The
flattened window exposes per-channel temporal shape (fast vs slow channels), the
multiscale property of assumption A8. Training uses the pure anomaly-free run;
test uses the labelled runs (a window is anomalous if the majority of its rows are
labelled anomalous). Standardisation is per-channel on the training rows only.

Produces the same CPSDataset the synthetic pipeline uses, so every detector and
component runs unchanged. SKAB has no ground-truth operating modes, so mode labels
are left empty (mode-dependent metrics that need true modes do not apply; C4 uses
the DISCOVERED clusters and still works).
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from data import CPSDataset

CHANNELS = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
            "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"]


def _feat(win):
    """Per-channel window features: mean, std, min, max, trend, range (6xC)."""
    return np.concatenate([win.mean(0), win.std(0), win.min(0), win.max(0),
                           win[-1] - win[0], win.max(0) - win.min(0)])


def _windows(df, W, stride, rep="flatten"):
    X = df[CHANNELS].to_numpy(dtype=np.float64)
    y = (df["anomaly"].to_numpy(dtype=np.float64) if "anomaly" in df.columns
         else np.zeros(len(X)))                        # anomaly-free file has no label
    wins, labs = [], []
    for i in range(0, len(X) - W + 1, stride):
        w = X[i:i + W]
        wins.append(_feat(w) if rep == "stats" else w)
        labs.append(1 if y[i:i + W].mean() > 0.5 else 0)
    if not wins:
        dim = 6 * len(CHANNELS) if rep == "stats" else (W, len(CHANNELS))
        return np.empty((0,) + (dim if isinstance(dim, tuple) else (dim,))), np.empty((0,), int)
    return np.stack(wins), np.array(labs, int)


def load_skab(root="datasets/SKAB", W=20, stride=10, rep="flatten") -> CPSDataset:
    """File-level split of the labelled runs. The separate anomaly-free run is a
    DIFFERENT operating regime (channel means differ up to 2x), so training on it
    and testing on the labelled runs is an invalid distribution shift; instead we
    alternate the labelled files into train/test so both splits see every regime
    and there is no window-overlap leakage across the split. Train uses only the
    NORMAL windows of the train files; test uses all windows of the test files."""
    root = Path(root)
    groups = [sorted(glob(str(root / f"data/{g}/*.csv"))) for g in ("valve1", "valve2", "other")]
    train_files, test_files = [], []
    for g in groups:                                       # alternate within each regime group
        train_files += g[0::2]; test_files += g[1::2]

    Xtr = []
    for f in train_files:
        w, l = _windows(pd.read_csv(f, sep=";"), W, stride, rep)
        if len(w):
            Xtr.append(w[l == 0])                          # normal windows only
    Xn = np.concatenate(Xtr)

    Xt, yt = [], []
    for f in test_files:
        w, l = _windows(pd.read_csv(f, sep=";"), W, stride, rep)
        if len(w):
            Xt.append(w); yt.append(l)
    Xt = np.concatenate(Xt); yt = np.concatenate(yt)

    if rep == "stats":                                     # (N, 6C) feature vectors
        mu, sd = Xn.mean(0), Xn.std(0) + 1e-8
        x_train = ((Xn - mu) / sd).astype(np.float32)
        x_test = ((Xt - mu) / sd).astype(np.float32)
    else:                                                  # (N, W, C) -> per-channel
        C = len(CHANNELS)
        mu = Xn.reshape(-1, C).mean(0); sd = Xn.reshape(-1, C).std(0) + 1e-8
        x_train = (((Xn - mu) / sd).reshape(len(Xn), W * C)).astype(np.float32)
        x_test = (((Xt - mu) / sd).reshape(len(Xt), W * C)).astype(np.float32)
    atype = np.where(yt == 1, "anomaly", "none")
    return CPSDataset(
        x_train=x_train, x_test=x_test, y_test=yt,
        mode_train=np.zeros(len(x_train), int), mode_test=np.where(yt == 1, -1, 0),
        atype_test=atype, name=f"SKAB(W={W})")


if __name__ == "__main__":
    ds = load_skab()
    print(f"{ds.name}: train={len(ds.x_train)} test={len(ds.x_test)} "
          f"feats={ds.n_features} anomaly-frac={ds.y_test.mean():.2f}")
