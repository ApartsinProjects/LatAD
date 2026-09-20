"""Per-community A3 rho: for each correlation-community expert, does that SUBSYSTEM have thin
between-regime pockets (A3) even when the GLOBAL rho is low? rho_G = frac(train-normal max VaDE
responsibility < 0.5) within community G, using the exact stored communities + the expert builder's
per-community VaDE config. Flags any subsystem with A3 structure hidden at the whole-plant level.
"""
from __future__ import annotations
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eda_real as E
from models_vade import train_vade
HERE = os.path.dirname(os.path.abspath(__file__))

DS = {"WADI_clean": "sota_bundle/experts_full/expert_WADI_clean.npz",
      "HAI": "sota_bundle/experts_full/expert_HAI.npz",
      "SWaT_canon": "sota_bundle/experts_full/expert_SWaT_canon.npz",
      "SKAB": "sota_bundle/experts/expert_SKAB.npz"}


def run(name, path, seed=0):
    if not os.path.exists(path):
        print(f"[{name}] expert missing ({path}) - skip"); return None
    D = E.load(name)
    Xw = np.asarray(D["Xn_w"], np.float32); nch = Xw.shape[1] // 6
    ex = np.load(path, allow_pickle=True)
    csize = np.asarray(ex["comm_size"]).astype(int); cch = np.asarray(ex["comm_channels"]).astype(int)
    rhos = []
    for gi, sz in enumerate(csize):
        G = cch[gi][:sz]
        cols = [b * nch + c for b in range(6) for c in G]
        Z = np.nan_to_num(Xw[:, cols])
        mu, sd = Z.mean(0), Z.std(0) + 1e-8; Z = ((Z - mu) / sd).astype(np.float32)
        K = min(20, max(6, sz)); LD = min(8, max(3, sz // 2))
        v = train_vade(Z, n_clusters=K, latent_dim=LD, epochs=15, warmup=4, seed=seed, device="cpu")
        mr = v._responsibilities(Z).max(1)
        rhos.append(float((mr < 0.5).mean()))
    rhos = np.array(rhos)
    hi = int((rhos >= 0.30).sum())
    res = dict(name=name, n_comm=len(rhos), rho_mean=round(float(rhos.mean()), 3),
               rho_median=round(float(np.median(rhos)), 3), rho_max=round(float(rhos.max()), 3),
               n_comm_A3_ge0p30=hi, n_comm_ge0p20=int((rhos >= 0.20).sum()),
               top5=[(int(csize[i]), round(float(rhos[i]), 3)) for i in np.argsort(-rhos)[:5]],
               per_comm_rho=[round(float(r), 3) for r in rhos])
    print(f"[{name}] {len(rhos)} communities: rho mean={res['rho_mean']} median={res['rho_median']} "
          f"max={res['rho_max']} | communities with A3 (rho>=0.30): {hi} | top5 (size,rho)={res['top5']}", flush=True)
    return res


if __name__ == "__main__":
    out = {}
    for nm, p in DS.items():
        r = run(nm, p)
        if r: out[nm] = r
        json.dump(out, open(os.path.join(HERE, "per_community_rho.json"), "w"), indent=1)
    print("\nGlobal-rho reference: SKAB ~0.53-0.58 (A3), WADI/HAI/SWaT ~0.01-0.05 (no global A3).")
