"""Test a RECONSTRUCTION AUTO-GATE. The diagnostic shows recon is the best signal on
PSM/SWaT/SMD but harmful on WADI/HAI. Can a TRAIN-NORMAL-ONLY signal decide when to use it?

Candidate gate: does the reconstruction residual carry structure the latent density MISSES on
held-out normal? We split train-normal 80/20, and compare how well recon vs density RANK the
held-out-normal tail. If recon flags DIFFERENT held-out-normal points than density (low rank
correlation) AND generalises (held-out recon not inflated), recon is complementary -> gate on.
We report, per dataset: difficult AUROC of density-only, density+recon, and the gate's decision,
to see if a clean unsupervised rule separates recon-helps from recon-harmful.
"""
from __future__ import annotations
import os, sys, json, numpy as np
from scipy.stats import spearmanr
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


def recon_energy(v, X):
    xt = _as_tensor(X, v)
    return tonp(_recon_energy(xt, v.decode(v.encode(xt)[0]), v.res_whitener))


def run(name):
    D = E.load(name); Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"]
    C6 = Xte0.shape[1] // 6; triv = np.abs(Xte0[:, :C6]).max(1); trn = np.abs(Xtr0[:, :C6]).max(1)
    hard = (y == 1) & ~(triv > np.quantile(trn, 0.99))
    K, LD = CFG[name]; mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
    a_dens, a_drec, gate_on, gate_val = [], [], [], []
    for seed in SEEDS:
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
        kd = min(80, max(20, len(Xtr) // 10)); v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd)
        dens_te, _ = v._hard_components(Xte); dens_tr, _ = v._hard_components(Xtr)
        rec_te, rec_tr = recon_energy(v, Xte), recon_energy(v, Xtr)
        sd = zt(dens_te, dens_tr); sr = zt(rec_te, rec_tr)
        a_dens.append(au(y, sd, hard)); a_drec.append(au(y, sd + sr, hard))
        # ---- unsupervised gate on train-normal only ----
        n = len(Xtr); cut = int(n * 0.8); A, B = Xtr[:cut], Xtr[cut:]
        vd_A, _ = v._hard_components(A); vr_A = recon_energy(v, A)
        vd_B, _ = v._hard_components(B); vr_B = recon_energy(v, B)
        # (1) recon generalises? held-out recon tail not inflated vs train
        gen = np.quantile(vr_B, 0.95) / (np.quantile(vr_A, 0.95) + 1e-9)
        # (2) recon complementary to density? low rank-corr on held-out normal
        rho = abs(spearmanr(vd_B, vr_B).correlation)
        # gate on when recon generalises (gen<1.5) AND is complementary (rho<0.5)
        on = (gen < 1.5) and (rho < 0.5)
        gate_on.append(bool(on)); gate_val.append((round(float(gen), 2), round(float(rho), 2)))
    r = {"dataset": name,
         "density_only": round(float(np.mean(a_dens)), 3),
         "density+recon": round(float(np.mean(a_drec)), 3),
         "recon_helps": bool(np.mean(a_drec) > np.mean(a_dens) + 0.01),
         "gate_fires": round(float(np.mean(gate_on)), 2), "gate_vals": gate_val}
    json.dump(r, open(f"{OUT}/fix_recon_{name}.json", "w"), indent=1)
    print(f"{name:5} dens={r['density_only']:.3f} dens+recon={r['density+recon']:.3f} "
          f"recon_helps={r['recon_helps']!s:5} GATE_fires={r['gate_fires']:.0%}  gen/rho={gate_val[0]}", flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["PSM", "SWaT", "SMD", "WADI", "HAI", "SKAB"]):
        run(nm)
