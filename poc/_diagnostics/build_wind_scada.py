"""Build a wind-SCADA (EDP Open Data) anomaly-detection tensor for one turbine, cached to npz,
mirroring the MetroPT loader contract (Xn normal / Xa test / ya per-timestep / ch names).
Turbine T06 (6 failures). Normal = rows with NO failure within +-14 days; anomaly = the 72h window
BEFORE each failure timestamp (incipient degradation). 10-min SCADA, 81 numeric channels.
Report-only exploration (for the A3 screen); does not touch paper/model.
"""
from __future__ import annotations
import os, glob, numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "_new", "wind_scada")
TURB = "T06"; PRE_H = 72; HEALTHY_DAYS = 14
STEP_MIN = 10
PRE = PRE_H * 60 // STEP_MIN            # 432 steps
HEALTHY = HEALTHY_DAYS * 24 * 60 // STEP_MIN

sig = pd.concat([pd.read_csv(f, sep=";") for f in sorted(glob.glob(os.path.join(ROOT, "wind-farm-1signals-training-*.csv")))],
                ignore_index=True)
sig = sig[sig["Turbine_ID"] == TURB].copy()
sig["Timestamp"] = pd.to_datetime(sig["Timestamp"], utc=True)
sig = sig.sort_values("Timestamp").reset_index(drop=True)

meta = ["Turbine_ID", "Timestamp"]
chan = [c for c in sig.columns if c not in meta]
X = sig[chan].apply(pd.to_numeric, errors="coerce").ffill().bfill().to_numpy(float)
keep = X.std(0) > 1e-6                      # drop constant/dead channels
X = X[:, keep]; chan = [c for i, c in enumerate(chan) if keep[i]]
ts = sig["Timestamp"].to_numpy()

fl = pd.read_csv(os.path.join(ROOT, "wind-farm-1-failures-training.csv"), sep=";")
fl["Timestamp"] = pd.to_datetime(fl["Timestamp"], utc=True)
fails = fl[fl["Turbine_ID"] == TURB]["Timestamp"].to_numpy()
print(f"{TURB}: {len(X)} rows, {X.shape[1]} channels, {len(fails)} failures")

# per-row labels: anomaly = within [fail-PRE, fail]; near = within +-HEALTHY of any fail
n = len(X); ya = np.zeros(n, int); near = np.zeros(n, bool)
tsi = pd.Series(range(n), index=pd.to_datetime(ts))
for f in fails:
    lo = np.searchsorted(ts, f - np.timedelta64(PRE_H, "h")); hi = np.searchsorted(ts, f, side="right")
    ya[lo:hi] = 1
    a = np.searchsorted(ts, f - np.timedelta64(HEALTHY_DAYS, "D")); b = np.searchsorted(ts, f + np.timedelta64(HEALTHY_DAYS, "D"), side="right")
    near[a:b] = True

Xn = X[~near]                                # healthy: far from any failure
Xa = X.copy()                                # full stream with pre-failure windows labeled
print(f"  normal(healthy) rows {len(Xn)}, test rows {len(Xa)}, anomaly rows {int(ya.sum())} ({ya.mean():.3%})")
out = os.path.join(ROOT, f"wind_{TURB.lower()}.npz")
np.savez(out, Xn=Xn.astype(np.float32), Xa=Xa.astype(np.float32), ya=ya, sens=np.array(chan))
print("saved", out)
