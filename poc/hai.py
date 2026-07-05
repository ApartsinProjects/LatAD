"""
HAI loader (real CPS: HIL-based Augmented ICS testbed, 59 sensors/actuators over
four coupled processes - boiler, turbine, water treatment - at 1 Hz). Unlike SKAB
this is genuinely multi-process, so it is the better test of the MIIM thesis.

Train files are attack-free; test files carry an 'attack' label. Same testbed, so
train and test share the operating regimes (no distribution shift). Windowed like
SKAB (per-channel statistical features by default; 59x6 = 354 dims). Standardised
on train. Returns the standard CPSDataset so the framework runs unchanged.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from data import CPSDataset

LABELS = {"time", "attack", "attack_P1", "attack_P2", "attack_P3"}


def _feat(win):
    return np.concatenate([win.mean(0), win.std(0), win.min(0), win.max(0),
                           win[-1] - win[0], win.max(0) - win.min(0)])


def _read(path):
    df = pd.read_csv(path, sep=";", compression="gzip")
    df.columns = [c.strip() for c in df.columns]
    return df


def _windows(df, sensors, W, stride, rep):
    X = df[sensors].to_numpy(dtype=np.float64)
    y = df["attack"].to_numpy(dtype=np.float64) if "attack" in df.columns else np.zeros(len(X))
    wins, labs = [], []
    for i in range(0, len(X) - W + 1, stride):
        w = X[i:i + W]
        wins.append(_feat(w) if rep == "stats" else w.ravel())
        labs.append(1 if y[i:i + W].mean() > 0.5 else 0)
    if not wins:
        return np.empty((0, 0)), np.empty((0,), int)
    return np.stack(wins), np.array(labs, int)


def load_hai(root="datasets/hai/hai-20.07", W=60, stride=60, rep="stats") -> CPSDataset:
    root = Path(root)
    train_files = sorted(glob(str(root / "train*.csv.gz")))
    test_files = sorted(glob(str(root / "test*.csv.gz")))
    sensors = [c for c in _read(train_files[0]).columns if c not in LABELS]

    Xtr = []
    for f in train_files:
        w, l = _windows(_read(f), sensors, W, stride, rep)
        if len(w):
            Xtr.append(w[l == 0])
    Xn = np.concatenate(Xtr)

    Xt, yt = [], []
    for f in test_files:
        w, l = _windows(_read(f), sensors, W, stride, rep)
        if len(w):
            Xt.append(w); yt.append(l)
    Xt = np.concatenate(Xt); yt = np.concatenate(yt)

    mu, sd = Xn.mean(0), Xn.std(0) + 1e-8
    x_train = ((Xn - mu) / sd).astype(np.float32)
    x_test = ((Xt - mu) / sd).astype(np.float32)
    return CPSDataset(
        x_train=x_train, x_test=x_test, y_test=yt,
        mode_train=np.zeros(len(x_train), int), mode_test=np.where(yt == 1, -1, 0),
        atype_test=np.where(yt == 1, "attack", "none"),
        name=f"HAI(W={W},{rep})")


if __name__ == "__main__":
    ds = load_hai()
    print(f"{ds.name}: train={len(ds.x_train)} test={len(ds.x_test)} "
          f"feats={ds.n_features} attack-frac={ds.y_test.mean():.3f}")
