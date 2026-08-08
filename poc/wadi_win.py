"""Why does LatAD lose on the exact-range WADI subset (every sensor within normal envelope)?
(1) Component decomposition: which head (density / nearest / residual / basin) has any signal there.
(2) Representation test: static 'stats' vs 'temporal' (rates/slope/spectral band power, A8).
(3) Inspect a few raw examples: which channels deviate (within range) and their dynamics.
WADI only (small, fast). Seeds 0-2.
"""
from __future__ import annotations
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
from winfeat import window_features
import eda_real as E

K, LD, SEEDS = 20, 10, [0, 1, 2]


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k])) if m.sum() >= 3 else float("nan")


def winmat(X, W, stride, rep):
    return np.stack([window_features(X[i:i + W], rep) for i in range(0, len(X) - W + 1, stride)]).astype(np.float32)


D = E.load("WADI"); fn, W, stride = E.RAW["WADI"]
Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
y = D["ya_w"].astype(int)
lo, hi = Xn.min(0), Xn.max(0)
oor = np.array([bool(((Xa[i:i + W] < lo) | (Xa[i:i + W] > hi)).any())
                for i in range(0, len(Xa) - W + 1, stride)])[:len(y)]
hard = (y == 1) & ~oor
print(f"range-hard WADI anomalies: {int(hard.sum())} (of {int((y==1).sum())})")

for rep in ("stats", "temporal"):
    Xtr0, Xte0 = winmat(Xn, W, stride, rep), winmat(Xa, W, stride, rep)[:len(y)]
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr, Xte = ((Xtr0 - mu) / sig).astype(np.float32), ((Xte0 - mu) / sig).astype(np.float32)
    comp = {k: [] for k in ["density", "base", "full"]}
    for sd in SEEDS:
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
        kd = min(80, max(20, len(Xtr) // 10))
        v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd)
        v.fit_resid_head(Xtr); v.fit_basin_head(Xtr)
        dens_te, near_te = v._hard_components(Xte); dm, ds, nm, ns = v._hd_ref
        comp["density"].append(au(y, (dens_te - dm) / ds, hard))
        comp["base"].append(au(y, np.asarray(v.anomaly_score_hard(Xte, use_resid=False, use_basin=False)), hard))
        comp["full"].append(au(y, np.asarray(v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto")), hard))
    print(f"  rep={rep:8} dim={Xtr0.shape[1]:4} | " +
          "  ".join(f"{k}={np.nanmean(vv):.3f}" for k, vv in comp.items()))

# inspect 3 range-hard examples (which channels deviate within range + dynamics)
sens = list(D.get("ch", [f"c{i}" for i in range(Xn.shape[1])]))
mu_r, sd_r = Xn.mean(0), Xn.std(0) + 1e-8
idx = np.where(hard)[0][:3]
print("\n=== 3 range-hard examples (raw, within envelope) ===")
for wi in idx:
    st = wi * stride; w = Xa[st:st + W]
    z = (w.mean(0) - mu_r) / sd_r
    vel = np.abs(np.diff(w, axis=0)).mean(0)
    veln = vel / (np.abs(np.diff(Xn, axis=0)).mean(0) + 1e-9)     # velocity vs normal
    top = np.argsort(-np.abs(z))[:5]
    print(f"\n win {wi}: top |mean-shift| channels")
    for c in top:
        print(f"   {sens[c][:16]:16} z_mean={z[c]:+5.2f}  vel/normal={veln[c]:5.1f}x  "
              f"val[{w[:,c].min():.2f},{w[:,c].max():.2f}] normal[{lo[c]:.2f},{hi[c]:.2f}]")
