"""V2 / A9 demonstration: re-test (multi-seed) whether a multiscale TEMPORAL/spectral window
representation lifts the DIFFICULT subset, especially on SKAB (vibration faults), without
hurting the others. Difficulty split is fixed (from the stats trivial detector) so both
representations are scored on the SAME hard windows. Uses the fixed density-base model.

feat_temporal adds within-window slope, velocity mean/std/spike, and low/high spectral band
power (A9) on top of level/variability/trend/range.
"""
from __future__ import annotations
import os, sys, json, numpy as np
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
from winfeat import window_features
import eda_real as E

W, ST = 60, 30
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SKAB": (16, 6), "SWaT": (40, 16)}
SEEDS = [0, 1, 2, 3, 4]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics"); os.makedirs(OUT, exist_ok=True)


def window(X, rep, y=None):
    A, B = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], rep))
        if y is not None:
            B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k])) if m.sum() >= 3 else None


def score_rep(Xn, Xa, y, hard, K, LD, seed):
    mu, sig = Xn.mean(0), Xn.std(0) + 1e-8
    Xtr = ((Xn - mu) / sig).astype(np.float32); Xte = ((Xa - mu) / sig).astype(np.float32)
    v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
    kd = min(80, max(20, len(Xtr) // 10))
    v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd)
    v.fit_resid_head(Xtr); v.fit_basin_head(Xtr)
    s = v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto", use_near=False)
    return au(y, s, hard)


def run(name):
    D = E.load(name)
    Xn = np.asarray(D["Xn_raw"], np.float32); Xa = np.asarray(D["Xa_raw"], np.float32); ya = D["ya_raw"]
    Xn_s, _ = window(Xn, "stats"); Xa_s, y = window(Xa, "stats", ya)
    Xn_t, _ = window(Xn, "temporal"); Xa_t, _ = window(Xa, "temporal", ya)
    y = np.asarray(y)
    C6 = Xa_s.shape[1] // 6; triv = np.abs(Xa_s[:, :C6]).max(1); trn = np.abs(Xn_s[:, :C6]).max(1)
    hard = (y == 1) & ~(triv > np.quantile(trn, 0.99))
    K, LD = CFG[name]
    stats, temp = [], []
    for seed in SEEDS:
        stats.append(score_rep(Xn_s, Xa_s, y, hard, K, LD, seed))
        temp.append(score_rep(Xn_t, Xa_t, y, hard, K, LD, seed))
        print(f"[{name} seed{seed}] stats={stats[-1]:.3f} temporal={temp[-1]:.3f}", flush=True)
    r = {"dataset": name, "n_difficult": int(hard.sum()),
         "stats":    {"mean": float(np.mean(stats)), "std": float(np.std(stats))},
         "temporal": {"mean": float(np.mean(temp)),  "std": float(np.std(temp))},
         "delta_temporal_minus_stats": round(float(np.mean(temp) - np.mean(stats)), 4),
         "per_seed_stats": [round(x, 3) for x in stats], "per_seed_temporal": [round(x, 3) for x in temp]}
    json.dump(r, open(f"{OUT}/v2_temporal_{name}.json", "w"), indent=1)
    print(f"=== {name}: stats {np.mean(stats):.3f} -> temporal {np.mean(temp):.3f} "
          f"(delta {r['delta_temporal_minus_stats']:+.3f})", flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["SKAB", "WADI", "HAI", "SWaT"]):
        run(nm)
