"""R1-5(b) representation ablation: does LatAD's gain depend on the FEATURE REPRESENTATION?
Compare the current 6-stat window features vs the 10-feature TEMPORAL/SPECTRAL set (winfeat
feat_temporal: level/variability/trend/range + within-window slope, velocity mean/std, spike,
low/high spectral band-power). Same VaDE config, same CANONICAL difficulty mask (from stats maxz,
fixed across reps for a fair comparison), 5 seeds. Global LatAD (density head) isolates the
representation effect. If stats >= temporal, the gain is NOT from a richer representation
(supports §6); if temporal helps, the representation matters (report honestly). Report-only.
"""
from __future__ import annotations
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eda_real as E
from winfeat import window_features
from models_vade import train_vade
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {"WADI_clean": (20, 10), "HAI": (40, 16), "SWaT_canon": (40, 16)}
SEEDS = [0, 1, 2, 3, 4]
REPS = ["stats", "temporal"]


def winfeats(X, W, stride, rep):
    F = np.stack([window_features(X[i:i + W], rep) for i in range(0, len(X) - W + 1, stride)]).astype(np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def au(y, s, hard):
    k = (y == 0) | hard
    return float(roc_auc_score(y[k], s[k])) if 0 < y[k].sum() < k.sum() else float("nan")


def run(name):
    D = E.load(name); fn, W, stride = E.RAW[name]; K, LD = CFG[name]
    Xn = np.asarray(D["Xn_raw"], np.float32); Xa = np.asarray(D["Xa_raw"], np.float32)
    y = np.asarray(D["ya_w"], int)
    # FIXED canonical difficulty mask from the stats representation (the paper's axis)
    Xte_stats = winfeats(Xa, W, stride, "stats"); Xtr_stats = winfeats(Xn, W, stride, "stats")
    C6 = Xte_stats.shape[1] // 6
    triv = np.abs(Xte_stats[:, :C6]).max(1); thr = float(np.quantile(np.abs(Xtr_stats[:, :C6]).max(1), 0.99))
    hard = (y == 1) & ~((y == 1) & (triv > thr))
    out = {"dataset": name, "n_diff": int(hard.sum())}
    for rep in REPS:
        Xtr0 = Xtr_stats if rep == "stats" else winfeats(Xn, W, stride, rep)
        Xte0 = Xte_stats if rep == "stats" else winfeats(Xa, W, stride, rep)
        mu, sg = Xtr0.mean(0), Xtr0.std(0) + 1e-8
        clp = lambda A: np.clip(np.nan_to_num((A - mu) / sg, nan=0.0, posinf=0.0, neginf=0.0), -10, 10).astype(np.float32)
        Xtr = clp(Xtr0); Xte = clp(Xte0)   # +-10 feature clip (matches the pipeline; stops temporal-feature blowup)
        kd = min(80, max(20, len(Xtr0) // 10))
        aus = []
        for sd in SEEDS:
            v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
            v.fit_latent_density(Xtr, k_density=kd)
            s = np.asarray(v.anomaly_score_hard(Xte, use_resid=False, use_basin=False))
            aus.append(au(y, s, hard))
        out[rep] = dict(dim=int(Xtr0.shape[1]), diff=round(float(np.mean(aus)), 4), sd=round(float(np.std(aus)), 4))
        print(f"  [{name}] {rep:9s} dim={Xtr0.shape[1]:4d} difficult={out[rep]['diff']:.4f}±{out[rep]['sd']:.4f}", flush=True)
    out["temporal_minus_stats"] = round(out["temporal"]["diff"] - out["stats"]["diff"], 4)
    print(f"  == {name}: temporal − stats = {out['temporal_minus_stats']:+.4f} "
          f"({'temporal helps' if out['temporal_minus_stats']>0.01 else 'stats sufficient' if out['temporal_minus_stats']>-0.01 else 'temporal hurts'})", flush=True)
    return out


if __name__ == "__main__":
    res = []
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI_clean", "HAI", "SWaT_canon"]):
        res.append(run(nm)); json.dump(res, open(os.path.join(HERE, "repr_ablation.json"), "w"), indent=1)
    print("\nInterpretation: temporal−stats ≤ +0.01 across datasets ⇒ the gain is NOT from the representation (supports §6).")
