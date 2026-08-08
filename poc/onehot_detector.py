"""Does retraining OUR detector (VaDE/LatAD) on ONE-HOT-encoded discrete channels help?
Window features: continuous channel -> the usual 6 stats (mean/std/min/max/trend/range);
discrete channel (<=6 normal states) -> per-state window fraction (soft one-hot). Retrain VaDE on
this representation and compare difficult-subset AUROC to the continuous baseline, on the SAME fixed
max|z| difficulty split (from the scores tables). Seeds 0-2.
"""
from __future__ import annotations
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
import eda_real as E

CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}
SEEDS = [0, 1, 2]
DMAX = 6


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k])) if m.sum() >= 3 else float("nan")


def mixed_feats(X, W, stride, disc, states):
    """continuous -> 6 stats; discrete -> per-state fractions. One flat vector per window."""
    out = []
    for i in range(0, len(X) - W + 1, stride):
        w = X[i:i + W]; parts = []
        for c in range(X.shape[1]):
            if disc[c]:
                parts.append([np.mean(w[:, c] == s) for s in states[c]])
            else:
                col = w[:, c]
                parts.append([col.mean(), col.std(), col.min(), col.max(),
                              col[-1] - col[0], col.max() - col.min()])
        out.append(np.concatenate(parts))
    return np.array(out, np.float32)


def run(name):
    D = E.load(name); fn, W, stride = E.RAW[name]
    Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
    y = D["ya_w"].astype(int)
    nun = [np.unique(Xn[:, c]) for c in range(Xn.shape[1])]
    disc = np.array([1 < len(u) <= DMAX for u in nun])
    states = [nun[c] for c in range(len(nun))]
    Xtr0 = mixed_feats(Xn, W, stride, disc, states)
    Xte0 = mixed_feats(Xa, W, stride, disc, states)[:len(y)]

    # fixed canonical difficulty split from the scores table
    d = np.load(f"_diagnostics/scores_{name}.npz")
    hard = (d["label"] == 1) & ~(d["maxz"] > float(d["maxz_thr"]))
    base = np.nanmean([au(d["label"], d["LatAD"][i], hard) for i in range(d["LatAD"].shape[0])])

    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
    K, LD = CFG[name]; kd = min(80, max(20, len(Xtr) // 10))
    aucs = []
    for sd in SEEDS:
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
        v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd)
        v.fit_resid_head(Xtr); v.fit_basin_head(Xtr)
        s = v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto")
        aucs.append(au(d["label"], np.asarray(s), hard))
    print(f"  {name}: dim {Xtr0.shape[1]} (discrete {int(disc.sum())}) | "
          f"LatAD one-hot difficult={np.nanmean(aucs):.3f}+/-{np.nanstd(aucs):.3f}  "
          f"vs continuous baseline={base:.3f}", flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI", "HAI", "SWaT"]):
        run(nm)
