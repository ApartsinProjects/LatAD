"""SWaT loader (iTrust Secure Water Treatment, Dec 2015 canonical benchmark).

Data source. The A4-A12 collections on the iTrust SharePoint are raw process CSVs with
no labels; the labelled Dec-2015 benchmark lives only inside a 108 GB zip. We instead
use the Kaggle CSV mirror `vishala28/swat-dataset-secure-water-treatment-system`:
    normal.csv  (~1.39M rows, all 'Normal')          -> training data
    attack.csv  (~54.6k rows, all 'Attack')          -> the extracted attack seconds
    merged.csv  ( = normal ++ attack concatenated )
51 sensor/actuator channels + a `Normal/Attack` label; column names have leading spaces.

NOTE the mirror does NOT preserve the canonical *interleaved* 4-day attack file (normal
periods between attacks are removed), so absolute F1 is not directly comparable to papers
that test on the interleaved file. We use a clean, leakage-free split: train on the first
part of normal, TEST on held-out normal (label 0) + attack.csv (label 1). Standardised on
train-normal; windowed like WADI/HAI.
"""
from __future__ import annotations
import os, numpy as np, pandas as pd
from data import CPSDataset
from winfeat import window_features

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(_ROOT, "datasets", "itrust", "SWaT", "Dec2015")
LABEL = "Normal/Attack"


def _read(name, usecols=None):
    df = pd.read_csv(os.path.join(DIR, name), usecols=usecols)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_swat(W=60, stride=30, rep="stats", downsample=10, clip=None,
              test_normal_frac=0.2, warmup_drop=0.02):
    npath, apath = os.path.join(DIR, "normal.csv"), os.path.join(DIR, "attack.csv")
    if not (os.path.exists(npath) and os.path.exists(apath)):
        raise FileNotFoundError(
            f"SWaT CSVs not found in {DIR}. Need normal.csv + attack.csv from the Kaggle mirror "
            "vishala28/swat-dataset-secure-water-treatment-system.")
    nrm, atk = _read("normal.csv"), _read("attack.csv")
    drop = {"Timestamp", LABEL}
    sensors = [c for c in nrm.columns if c not in drop and c in atk.columns
               and pd.api.types.is_numeric_dtype(nrm[c])]
    prep = lambda df: df[sensors].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0).values.astype(np.float32)
    Xn_all = prep(nrm); Xa = prep(atk)[::downsample]
    Xn_all = Xn_all[int(len(Xn_all) * warmup_drop):]        # drop plant-stabilisation head
    Xn_all = Xn_all[::downsample]
    cut = int(len(Xn_all) * (1 - test_normal_frac))
    Xn_tr, Xn_te = Xn_all[:cut], Xn_all[cut:]               # leakage-free normal train/test split

    mu, sd = Xn_tr.mean(0), Xn_tr.std(0) + 1e-8             # standardise on TRAIN normal
    norm = lambda X: np.clip((X - mu) / sd, -clip, clip) if clip else (X - mu) / sd
    Xn_tr, Xn_te, Xa = norm(Xn_tr), norm(Xn_te), norm(Xa)

    def win(X, lab):
        Xw, yl = [], []
        for i in range(0, len(X) - W + 1, stride):
            Xw.append(window_features(X[i:i + W], rep)); yl.append(lab)
        return (np.asarray(Xw, np.float32) if Xw else np.empty((0, 0), np.float32),
                np.asarray(yl, int))
    Xtr, _ = win(Xn_tr, 0)
    Xte_n, yn = win(Xn_te, 0); Xte_a, ya = win(Xa, 1)       # window each segment separately
    Xte = np.concatenate([Xte_n, Xte_a]); yte = np.concatenate([yn, ya])
    n, m = len(Xtr), len(Xte)
    return CPSDataset(Xtr, Xte, yte, np.full(n, -1), np.full(m, -1),
                      np.where(yte == 1, "attack", "none").astype(object), "SWaT")


if __name__ == "__main__":
    d = load_swat()
    print(f"SWaT: train={d.x_train.shape} test={d.x_test.shape} feat={d.n_features} "
          f"anom-frac={d.y_test.mean():.3f}")
