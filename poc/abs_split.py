"""Recompute the difficult-subset AUROC under an ABSOLUTE-sigma easy/hard threshold instead of
the data-driven 99th-percentile of train max|z|. 'easy' = some raw channel is grossly off
(max|z| >= K sigma); 'hard' = no channel grossly off (candidate correlation-break). Compares
subset sizes + LatAD / IsolationForest / AutoEncoder difficult-AUROC across split definitions.
Seed 0 (direction check). Reuses the exact detector + baselines.
"""
from __future__ import annotations
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
from compare_baselines import ae_scores
import eda_real as E

CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}
SEED = 0


def au(y, s, mask):
    k = (y == 0) | mask
    return float(roc_auc_score(y[k], s[k])) if mask.sum() >= 3 else float("nan")


def run(name):
    D = E.load(name)
    Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"].astype(int)
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
    C6 = Xte0.shape[1] // 6
    zmax = np.abs(Xte[:, :C6]).max(1); zmax_tr = np.abs(Xtr[:, :C6]).max(1)   # standardized raw-channel max|z|
    q99 = float(np.quantile(zmax_tr, 0.99))

    # scores (seed 0): LatAD (reported head), IsolationForest, AutoEncoder (reconstruction)
    K, LD = CFG[name]
    v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=SEED, device="cpu")
    kd = min(80, max(20, len(Xtr) // 10))
    v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd)
    v.fit_resid_head(Xtr); v.fit_basin_head(Xtr)
    s_latad = v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto")
    iforest = IsolationForest(n_estimators=200, random_state=SEED).fit(Xtr)
    s_if = -iforest.score_samples(Xte)
    s_ae = ae_scores(Xtr, Xte, seed=SEED)
    scores = {"LatAD": s_latad, "IF": s_if, "AE": s_ae}

    splits = [("99pct(cur)", q99)] + [(f"abs{K}sig", float(K)) for K in (4, 5, 6)]
    print(f"\n=== {name}  (anom={int((y==1).sum())}, 99pct thr={q99:.1f} sigma) ===")
    hdr = f"  {'split':12} {'thr':>5} {'n_easy':>7} {'n_hard':>7} " + " ".join(f"{m+'_hard':>11}" for m in scores)
    print(hdr)
    for tag, thr in splits:
        easy = (y == 1) & (zmax > thr); hard = (y == 1) & ~easy
        row = f"  {tag:12} {thr:>5.1f} {int(easy.sum()):>7} {int(hard.sum()):>7} "
        row += " ".join(f"{au(y, s, hard):>11.3f}" for s in scores.values())
        print(row)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI", "HAI", "SWaT"]):
        run(nm)
