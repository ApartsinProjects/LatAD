"""Export the reported model (VaDE-hard+resid(auto)) as shareable checkpoints, one per
dataset, for a Zenodo deposit. Reconstructs the EXACT model report_table.py evaluates
(same CFG, seed 0, epochs, k_density gating), then serializes weights + fitted heads +
standardization + config. Checkpoints contain learned parameters only, never raw data,
so they are shareable even though WADI/HAI/SKAB themselves are gated third-party sets.

Usage:  python export_checkpoints.py            # all three
        python export_checkpoints.py WADI       # one
Outputs zenodo_bundle/checkpoints/vade_<DS>.pt  (torch.save dict).
"""
from __future__ import annotations
import os, sys, numpy as np, torch
from models_vade import train_vade
import eda_real as E

# mirror report_table.py exactly
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SKAB": (16, 6)}   # (K, latent_dim)
STRIDE = {"WADI": 30, "HAI": 30, "SKAB": 30}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zenodo_bundle", "checkpoints")


def export(name: str):
    D = E.load(name)
    Xtr = D["Xn_w"].astype(np.float32)
    K, LD = CFG[name]
    mu, sig = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s = ((Xtr - mu) / sig).astype(np.float32)
    v = train_vade(Xtr_s, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=0, device="cpu")
    kd = min(80, max(20, len(Xtr_s) // 10))
    v.fit_residual_whitener(Xtr_s); v.fit_latent_density(Xtr_s, k_density=kd)
    v.fit_resid_head(Xtr_s); v.fit_basin_head(Xtr_s)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"vade_{name}.pt")
    torch.save({
        "dataset": name,
        "model": v,                       # full fitted object (needs models_vade.py to unpickle)
        "state_dict": v.state_dict(),     # portable weights
        "standardization": {"mu": mu.astype(np.float32), "sig": sig.astype(np.float32)},
        "config": {"dataset": name, "K": K, "latent_dim": LD, "window": int(D["W"]),
                   "stride": STRIDE[name], "epochs": 40, "warmup": 8, "seed": 0,
                   "k_density": int(kd), "score": "anomaly_score_hard(use_resid='auto', use_basin='auto')"},
        "resid_auto_on": bool(getattr(v, "_resid_auto", False)),
        "basin_lambda": float(getattr(v, "_basin_lam", 0.0)),
    }, path)
    print(f"saved {path}  ({os.path.getsize(path)//1024} KB)  "
          f"K={K} latent={LD} kd={kd} resid_auto={getattr(v,'_resid_auto',None)} "
          f"basin_lam={getattr(v,'_basin_lam',None):.3f}")


if __name__ == "__main__":
    names = sys.argv[1:] or ["WADI", "HAI", "SKAB"]
    for nm in names:
        export(nm)
