"""WADI A2 loader (iTrust). Normal = WADI_14days_new.csv; test = WADI_attackdataLABLE.csv
with a per-row attack label (1 no-attack, -1 attack). Windowed, time-ordered, standardised
on the normal channels. Returns a CPSDataset (mode/atype are dummy for real data)."""
from __future__ import annotations
import os, numpy as np, pandas as pd
from data import CPSDataset
from winfeat import window_features

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
DIR = os.path.join(_ROOT, "datasets", "itrust", "WADI", "WADI.A2_19 Nov 2019")


def _prep(df, cols):
    X = df[cols].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    return X.values.astype(np.float32)


def load_wadi(W=60, stride=30, rep="stats", downsample=10, clip=10.0):
    nrm = pd.read_csv(f"{DIR}/WADI_14days_new.csv")        # header on row 0
    atk = pd.read_csv(f"{DIR}/WADI_attackdataLABLE.csv", skiprows=1)  # extra index row first
    nrm.columns = [c.strip() for c in nrm.columns]
    atk.columns = [c.strip() for c in atk.columns]
    lc = atk.columns[-1]                                   # "Attack LABLE ..."
    meta = set(list(atk.columns[:3]) + list(nrm.columns[:3]) + [lc])
    sensors = [c for c in atk.columns if c not in meta and c in nrm.columns]
    sensors = [c for c in sensors if nrm[c].isna().mean() < 0.5]   # drop mostly-NaN cols

    Xn = _prep(nrm, sensors)[::downsample]
    Xa = _prep(atk, sensors)[::downsample]                 # downsample TEST too (was a bug:
    ya = (pd.to_numeric(atk[lc], errors="coerce").values == -1).astype(int)[::downsample]
    # train/test windows must span the SAME physical horizon, else features mismatch)
    # KEEP the ~30 near-constant-in-normal channels: standardised by their tiny sigma
    # they are the most sensitive attack carriers (an actuator forced during an attack).
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-8                  # standardise on normal channels
    Xn, Xa = (Xn - mu) / sd, (Xa - mu) / sd
    if clip:                                               # A2 hard envelope: a >10sigma reading is a
        Xn, Xa = np.clip(Xn, -clip, clip), np.clip(Xa, -clip, clip)  # sensor fault / glitch / unit
    # shift (2_MCV_007_CO const-in-normal->active; 2B_AIT_002_PV 9->4428), not a process state.

    def win(X, y=None):
        Xw, yl = [], []
        for i in range(0, len(X) - W + 1, stride):
            Xw.append(window_features(X[i:i + W], rep))
            if y is not None:
                yl.append(int(y[i:i + W].mean() > 0.05))
        return np.asarray(Xw, np.float32), (np.asarray(yl, int) if y is not None else None)

    Xtr, _ = win(Xn)
    Xte, yte = win(Xa, ya)
    n, m = len(Xtr), len(Xte)
    return CPSDataset(Xtr, Xte, yte, np.full(n, -1), np.full(m, -1),
                      np.array(["none"] * m, object), "WADI")


if __name__ == "__main__":
    d = load_wadi()
    print(f"WADI: train={d.x_train.shape} test={d.x_test.shape} feat={d.n_features} "
          f"anom-frac={d.y_test.mean():.3f}")
