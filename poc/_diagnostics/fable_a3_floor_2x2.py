"""Isolate WHERE the floor acts on the SKAB basin lift: train floor x score floor 2x2, seeds 0-2.
Train at floor 0.05 (default) or 0.01; then score the SAME trained model with the basin head refit under
(a) its own floored variances, (b) the other floor applied to logvar_c (re-clamped), (c) empirical
within-component variances (no floor). Only the basin head (_noise_agreement -> argmax of
_log_pz_given_c, which reads _lvc()) sees the change; the density base is fit once per model at its own
floor and kept fixed, so the OFF score is identical within a row. Local monkeypatch, no file edits.
Invariant: cell (train 0.05, score 0.05) must reproduce lift1 +0.105/+0.044/+0.068 for seeds 0/1/2.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, math
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import eda_real as E
import models_vade as MV
from a3_witness import win, au, CFG

FLOOR = {"floor": 0.05}
_orig_init = MV.VaDE.__init__
def _patched_init(self, *a, **k):
    _orig_init(self, *a, **k); self.logvar_floor = math.log(FLOOR["floor"])
MV.VaDE.__init__ = _patched_init

K, LD = CFG["SKAB"]; D = E.load("SKAB")
Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
Xtr_s = ((Xtr - m) / sd).astype(np.float32); Xte_s = ((Xte - m) / sd).astype(np.float32)
C6 = Xte.shape[1] // 6
triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy
kd = min(80, max(20, len(Xtr_s) // 10))
R = {}


def basin_lift(v, off_te):
    v.fit_basin_head(Xtr_s)
    ag = v._noise_agreement(Xte_s); am, asd = v._basin_ref
    on = lambda lam: off_te - lam * (ag - am) / asd
    a_off = au(yw, off_te, hard)
    G = v._responsibilities(Xtr_s)
    return dict(rho=round(float((G.max(1) < 0.5).mean()), 3), lift1=round(au(yw, on(1.0), hard) - a_off, 3),
                lift2=round(au(yw, on(2.0), hard) - a_off, 3), OFF=round(a_off, 3))


for tf in (0.05, 0.01):
    for seed in range(3):
        FLOOR["floor"] = tf
        v = MV.train_vade(Xtr_s, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
        v.fit_residual_whitener(Xtr_s); v.fit_latent_density(Xtr_s, k_density=kd); v.fit_resid_head(Xtr_s)
        off_te = np.asarray(v.anomaly_score_hard(Xte_s, use_resid="auto", use_basin=False))
        lv_trained = v.logvar_c.detach().clone()
        row = {}
        for sf in (0.05, 0.02, 0.01, "empirical"):
            with torch.no_grad():
                if sf == "empirical":
                    z = v.encode(torch.as_tensor(Xtr_s))[0].numpy().astype(np.float64)
                    v.logvar_floor = math.log(1e-4); v.logvar_c.copy_(lv_trained)
                    lab = v._responsibilities(Xtr_s).argmax(1)
                    lvc0 = v._lvc().detach().numpy()
                    var_hat = np.stack([z[lab == k].var(0) + 1e-4 if (lab == k).sum() > 1 else np.exp(lvc0[k]) for k in range(K)])
                    v.logvar_c.copy_(torch.as_tensor(np.log(var_hat), dtype=torch.float32))
                else:
                    v.logvar_floor = math.log(sf); v.logvar_c.copy_(lv_trained)
            row[f"score_{sf}"] = basin_lift(v, off_te)
        R[f"train{tf}_s{seed}"] = row
        print(f"train floor {tf} seed {seed}: " + " | ".join(f"score {k[6:]}: rho={r['rho']} lift1={r['lift1']:+.3f} lift2={r['lift2']:+.3f}" for k, r in row.items()) + f" (OFF {row['score_0.05']['OFF']})", flush=True)
json.dump(R, open(os.path.join(HERE, "fable_a3_floor_2x2.json"), "w"), indent=1)
print("done", flush=True)
