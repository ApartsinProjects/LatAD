"""Root-cause the non-wins (PSM/SMD/SKAB): decompose each candidate signal's difficult-subset
AUROC. Hypothesis: on server-IT data the VaDE encoder maps anomalies INTO the normal latent
(latent density blind), while a feature-space / reconstruction signal separates them. If so,
the 'drop reconstruction' rule is CPS-specific and an auto-detector could recover PSM/SMD.
"""
from __future__ import annotations
import os, sys, json, numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from models_vade import train_vade, _as_tensor, _recon_energy
import eda_real as E

CFG = {"WADI": (20, 10), "HAI": (40, 16), "SKAB": (16, 6), "SWaT": (40, 16), "PSM": (40, 16), "SMD": (40, 16)}
SEEDS = [0, 1, 2]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics"); os.makedirs(OUT, exist_ok=True)


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k])) if m.sum() >= 3 else float("nan")


def z(s, ref):
    return (s - ref.mean()) / (ref.std() + 1e-9)


def tonp(t):
    try: return t.detach().cpu().numpy()
    except Exception: return np.asarray(t)


def run(name):
    D = E.load(name); Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"]
    C6 = Xte0.shape[1] // 6; triv = np.abs(Xte0[:, :C6]).max(1); trn = np.abs(Xtr0[:, :C6]).max(1)
    hard = (y == 1) & ~(triv > np.quantile(trn, 0.99))
    K, LD = CFG[name]; mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
    acc = {k: [] for k in ["density", "nearest", "recon", "IF_feat", "anom_in_normal_latent"]}
    for seed in SEEDS:
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
        kd = min(80, max(20, len(Xtr) // 10))
        v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd)
        # component signals
        dens_te, near_te = v._hard_components(Xte); dens_tr, near_tr = v._hard_components(Xtr)
        acc["density"].append(au(y, z(dens_te, dens_tr), hard))
        acc["nearest"].append(au(y, z(near_te, near_tr), hard))
        rec_te = tonp(_recon_energy(_as_tensor(Xte, v), v.decode(v.encode(_as_tensor(Xte, v))[0]), v.res_whitener))
        rec_tr = tonp(_recon_energy(_as_tensor(Xtr, v), v.decode(v.encode(_as_tensor(Xtr, v))[0]), v.res_whitener))
        acc["recon"].append(au(y, z(rec_te, rec_tr), hard))
        sIF = -IsolationForest(n_estimators=200, random_state=seed).fit(Xtr).decision_function(Xte)
        acc["IF_feat"].append(au(y, sIF, hard))
        # diagnostic: are difficult anomalies in HIGH normal-latent-density? (blind spot)
        # density signal = NLL; low NLL (high density) for anomalies => encoder maps them to normal
        nll_anom = dens_te[hard].mean(); nll_norm = dens_te[y == 0].mean()
        acc["anom_in_normal_latent"].append(float(nll_anom < nll_norm))  # True = anomalies look MORE normal
    r = {k: {"mean": round(float(np.nanmean(vv)), 3), "std": round(float(np.nanstd(vv)), 3)} for k, vv in acc.items()}
    json.dump({"dataset": name, **r}, open(f"{OUT}/diag_fails_{name}.json", "w"), indent=1)
    print(f"{name:5} density={r['density']['mean']:.3f} nearest={r['nearest']['mean']:.3f} "
          f"recon={r['recon']['mean']:.3f} IF_feat={r['IF_feat']['mean']:.3f} "
          f"| anomInNormalLatent={r['anom_in_normal_latent']['mean']:.0%}", flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["PSM", "SMD", "SKAB", "WADI", "HAI", "SWaT"]):
        run(nm)
