"""H1 loophole probe: a rare regime that OVERLAPS a big one and is rarer than 1/K would be merged at every
K in the sweep. Fit fine diag GMMs (K = paper, 64, 128) on the cached paper-config latents (and PCA-20 of
the features) and read (a) how many components carry < 1% train mass, (b) the responsibility entropy of
the train points sitting in those rare components vs the rest. Invariant (before running): if rare regimes
overlap big ones, rare-component points read HIGH entropy (>= SKAB's 0.27) once they get their own
component; if rare regimes are separated clumps, they read as low as the bulk (< 0.1).
SKAB reference latent: one VaDE fit at the paper config (K16/LD6, seed 0).
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fable_a3_spaces_lib import H_norm, HERE
sys.path.insert(0, os.path.dirname(HERE))

R = {}


def probe(Z, K, tag):
    g = GaussianMixture(K, covariance_type="diag", reg_covar=1e-4, random_state=0, n_init=1).fit(Z)
    G = g.predict_proba(Z); lab = G.argmax(1); H = H_norm(G)
    mass = np.bincount(lab, minlength=K) / len(lab)
    rare = mass < 0.01; in_rare = rare[lab]
    r = dict(K=K, n_rare_components_lt1pct=int(rare.sum()), mass_in_rare=float(in_rare.mean()),
             H_all=float(H.mean()), H_rare_points=float(H[in_rare].mean()) if in_rare.any() else None,
             H_bulk_points=float(H[~in_rare].mean()), frac_rare_points_H_gt_0p3=float((H[in_rare] > 0.3).mean()) if in_rare.any() else None,
             smallest5_mass=[float(m) for m in np.sort(mass)[:5]])
    print(f"  {tag} K={K}: {r}", flush=True)
    return r


def feats_pca20(X):
    return PCA(20, random_state=0).fit_transform(np.asarray(X, np.float64))


for name in ("WADI", "HAI", "SWaT"):
    e = np.load(os.path.join(HERE, f"e2_fable_{name}.npz")); ztr = e["ztr"].astype(np.float64); Kp = len(e["logpi"])
    R[name] = {"latent": {}, "pca20": {}}
    print(f"== {name} n={len(ztr)} LD={ztr.shape[1]}", flush=True)
    P = feats_pca20(e["Xtr_s"])
    for K in sorted({Kp, 64, 128}):
        R[name]["latent"][f"K{K}"] = probe(ztr, K, f"{name} latent")
        R[name]["pca20"][f"K{K}"] = probe(P, K, f"{name} pca20")
# SKAB reference
import eda_real as E, torch
from models_vade import train_vade
X = np.asarray(E.load("SKAB")["Xn_w"], np.float32); X = (X - X.mean(0)) / (X.std(0) + 1e-8)
v = train_vade(X, n_clusters=16, latent_dim=6, epochs=40, warmup=8, seed=0, device="cpu")
with torch.no_grad():
    z = v.encode(torch.as_tensor(X))[0].numpy().astype(np.float64)
G = v._responsibilities(X); R["SKAB"] = {"vade_own": dict(H_all=float(H_norm(G).mean())), "latent": {}, "pca20": {}}
print(f"== SKAB n={len(z)} vade H={R['SKAB']['vade_own']['H_all']:.3f}", flush=True)
P = feats_pca20(X)
for K in (16, 32, 64):
    R["SKAB"]["latent"][f"K{K}"] = probe(z, K, "SKAB latent"); R["SKAB"]["pca20"][f"K{K}"] = probe(P, K, "SKAB pca20")
json.dump(R, open(os.path.join(HERE, "fable_a3_spaces_rare.json"), "w"), indent=1)
print("done", flush=True)
