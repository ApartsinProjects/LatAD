"""Export per-timestep SOTA-harness arrays for WADI/HAI/SWaT with the SAME preprocessing
our detector uses (standardize on normal; clip +-10 sigma only where the detector clips,
i.e. WADI). Emits {pfx}_train/test/labels.npy plus {pfx}_triv_test.npy + {pfx}_triv_thr.npy
so the easy/difficult raw split can be applied per dataset inside Modal. Preprocessing-fair
by construction (matches eda_real.CLIP)."""
from __future__ import annotations
import os, sys, numpy as np
import eda_real as E

OUT = os.path.dirname(os.path.abspath(__file__))


def prep(name):
    rawfn = E.RAW[name][0]
    Xn, Xa, ya, sens = rawfn()                      # timestep resolution (Xn train-normal, Xa test)
    Xn = np.asarray(Xn, np.float32); Xa = np.asarray(Xa, np.float32)
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-8
    zn = (Xn - mu) / sd; za = (Xa - mu) / sd
    # trivial max|z| split uses PRE-clip z so it does not saturate (WADI clips at 10 -> the
    # split would otherwise collapse: every point maxes at 10 and nothing exceeds the threshold).
    triv = np.abs(za).max(1).astype(np.float32)
    thr = float(np.quantile(np.abs(zn).max(1), 0.99))
    clip = E.CLIP.get(name)                          # WADI -> 10.0 ; HAI/SWaT -> None (match detector)
    if clip:
        zn = np.clip(zn, -clip, clip); za = np.clip(za, -clip, clip)
    zn = zn.astype(np.float32); za = za.astype(np.float32); ya = np.asarray(ya, np.int64)
    pfx = "wadi" if name == "WADI" else name
    np.save(f"{OUT}/{pfx}_train.npy", zn)
    np.save(f"{OUT}/{pfx}_test.npy", za)
    np.save(f"{OUT}/{pfx}_labels.npy", ya)
    np.save(f"{OUT}/{pfx}_triv_test.npy", triv)
    np.save(f"{OUT}/{pfx}_triv_thr.npy", np.float32(thr))
    print(f"{name:5} train{zn.shape} test{za.shape} anom={ya.mean():.3f} chan={zn.shape[1]} "
          f"clip={clip} triv_thr={thr:.2f}", flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI", "HAI", "SWaT"]):
        prep(nm)
