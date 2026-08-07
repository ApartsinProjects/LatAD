"""Gate-free fix: combine standardized views by MAX (an OR ensemble) instead of a sum or a
learned gate. A window is anomalous if ANY view (latent density, reconstruction) flags it.
Tests whether max(z density, z recon) recovers PSM/SWaT (recon-signal) WITHOUT hurting
WADI/HAI (density-signal, recon uninformative). Difficult-subset AUROC, seeds 0-2.
"""
from __future__ import annotations
import os, sys, json, numpy as np
from sklearn.metrics import roc_auc_score
from models_vade import train_vade, _as_tensor, _recon_energy
import eda_real as E

CFG = {"WADI": (20, 10), "HAI": (40, 16), "SKAB": (16, 6), "SWaT": (40, 16), "PSM": (40, 16), "SMD": (40, 16)}
SEEDS = [0, 1, 2]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics"); os.makedirs(OUT, exist_ok=True)


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k])) if m.sum() >= 3 else float("nan")


def zt(s, ref): return (s - ref.mean()) / (ref.std() + 1e-9)
def tonp(t):
    try: return t.detach().cpu().numpy()
    except Exception: return np.asarray(t)


def run(name):
    D = E.load(name); Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"]
    C6 = Xte0.shape[1] // 6; triv = np.abs(Xte0[:, :C6]).max(1); trn = np.abs(Xtr0[:, :C6]).max(1)
    hard = (y == 1) & ~(triv > np.quantile(trn, 0.99))
    K, LD = CFG[name]; mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
    R = {k: [] for k in ["density", "sum_d_r", "max_d_r", "max_d_r_n"]}
    for seed in SEEDS:
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
        kd = min(80, max(20, len(Xtr) // 10)); v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd)
        dte, nte = v._hard_components(Xte); dtr, ntr = v._hard_components(Xtr)
        rte = tonp(_recon_energy(_as_tensor(Xte, v), v.decode(v.encode(_as_tensor(Xte, v))[0]), v.res_whitener))
        rtr = tonp(_recon_energy(_as_tensor(Xtr, v), v.decode(v.encode(_as_tensor(Xtr, v))[0]), v.res_whitener))
        sd, sr, sn = zt(dte, dtr), zt(rte, rtr), zt(nte, ntr)
        R["density"].append(au(y, sd, hard))
        R["sum_d_r"].append(au(y, sd + sr, hard))
        R["max_d_r"].append(au(y, np.maximum(sd, sr), hard))
        R["max_d_r_n"].append(au(y, np.maximum.reduce([sd, sr, sn]), hard))
    r = {"dataset": name, **{k: round(float(np.nanmean(v)), 3) for k, v in R.items()}}
    json.dump(r, open(f"{OUT}/maxcombo_{name}.json", "w"), indent=1)
    print(f"{name:5} density={r['density']:.3f} sum={r['sum_d_r']:.3f} "
          f"MAX(d,r)={r['max_d_r']:.3f} MAX(d,r,n)={r['max_d_r_n']:.3f}", flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["PSM", "SWaT", "SMD", "WADI", "HAI", "SKAB"]):
        run(nm)
