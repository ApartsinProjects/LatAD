"""Verify the nearest-component fix UNIFORMLY: per dataset x seed, one trained model,
compare FULL (use_near=True, current) vs FIXED (use_near=False, density-only base), both
with auto resid+basin. Adopt the fix only if it is >= current on every dataset (not a
WADI-specific tweak). Also records the A4 comparison (high-K density vs nearest-mode).
"""
from __future__ import annotations
import os, sys, json, numpy as np
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
import eda_real as E

CFG = {"WADI": (20, 10), "HAI": (40, 16), "SKAB": (16, 6), "SWaT": (40, 16)}
SEEDS = [0, 1, 2, 3, 4]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics"); os.makedirs(OUT, exist_ok=True)


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k])) if m.sum() >= 3 else None


def run(name):
    D = E.load(name); Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"]
    C6 = Xte0.shape[1] // 6; triv = np.abs(Xte0[:, :C6]).max(1); trn = np.abs(Xtr0[:, :C6]).max(1)
    hard = (y == 1) & ~(triv > np.quantile(trn, 0.99))
    K, LD = CFG[name]; mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
    full, fixed = [], []
    for seed in SEEDS:
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
        kd = min(80, max(20, len(Xtr) // 10))
        v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd)
        v.fit_resid_head(Xtr); v.fit_basin_head(Xtr)
        sF = v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto", use_near=True)
        sX = v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto", use_near=False)
        full.append(au(y, sF, hard)); fixed.append(au(y, sX, hard))
        print(f"[{name} seed{seed}] full={full[-1]:.3f} fixed(density-base)={fixed[-1]:.3f}", flush=True)
    r = {"dataset": name,
         "full_near_on":  {"mean": float(np.mean(full)),  "std": float(np.std(full))},
         "fixed_near_off": {"mean": float(np.mean(fixed)), "std": float(np.std(fixed))},
         "delta": round(float(np.mean(fixed) - np.mean(full)), 4),
         "per_seed_full": [round(x, 3) for x in full], "per_seed_fixed": [round(x, 3) for x in fixed]}
    json.dump(r, open(f"{OUT}/fix_verify_{name}.json", "w"), indent=1)
    print(f"=== {name}: full {np.mean(full):.3f} -> fixed {np.mean(fixed):.3f} (delta {r['delta']:+.3f})", flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI", "HAI", "SKAB", "SWaT"]):
        run(nm)
