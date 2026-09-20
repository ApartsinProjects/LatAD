"""Is the SKAB A3 basin witness seed-robust, or a lucky-seed artifact? Replicates a3_witness.run
for SKAB seeds 0-4: OFF vs FORCED-ON(lam=1,2) difficult AUROC, the rho<0.5 gate, and the proposed
stable gate statistic (mean normalized responsibility entropy). Report-only.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eda_real as E
from models_vade import train_vade
from a3_witness import win, au, CFG
HERE = os.path.dirname(os.path.abspath(__file__))

name = "SKAB"; K, LD = CFG[name]
D = E.load(name)
Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
Xtr_s = ((Xtr - m) / sd).astype(np.float32); Xte_s = ((Xte - m) / sd).astype(np.float32)
C6 = Xte.shape[1] // 6
triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy
kd = min(80, max(20, len(Xtr_s) // 10))

rows = []
for seed in range(5):
    v = train_vade(Xtr_s, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
    v.fit_residual_whitener(Xtr_s); v.fit_latent_density(Xtr_s, k_density=kd)
    v.fit_resid_head(Xtr_s); v.fit_basin_head(Xtr_s)
    G = v._responsibilities(Xtr_s); maxr = G.max(1)
    rho = float((maxr < 0.5).mean())
    ent = float(np.mean(-(G * np.log(G + 1e-12)).sum(1) / np.log(G.shape[1])))  # mean normalized entropy
    basin_lam = float(getattr(v, "_basin_lam", 0.0))
    off_te = np.asarray(v.anomaly_score_hard(Xte_s, use_resid="auto", use_basin=False))
    off_tr = np.asarray(v.anomaly_score_hard(Xtr_s, use_resid="auto", use_basin=False))
    ag_te = v._noise_agreement(Xte_s); am, asd = v._basin_ref
    def on(lam): return off_te - lam * (ag_te - am) / asd
    au_off = au(yw, off_te, hard); au_on1 = au(yw, on(1.0), hard); au_on2 = au(yw, on(2.0), hard)
    r = dict(seed=seed, rho=round(rho, 3), entropy=round(ent, 3), basin_lam=round(basin_lam, 3),
             gate_fires=basin_lam > 0, diff_OFF=round(au_off, 3), diff_ON1=round(au_on1, 3),
             diff_ON2=round(au_on2, 3), lift1=round(au_on1 - au_off, 3), lift2=round(au_on2 - au_off, 3))
    rows.append(r)
    print(f"seed {seed}: rho={r['rho']} entropy={r['entropy']} lam={r['basin_lam']} gate={r['gate_fires']} "
          f"| OFF={r['diff_OFF']} ON(1)={r['diff_ON1']} lift={r['lift1']:+.3f} ON(2)={r['diff_ON2']} lift2={r['lift2']:+.3f}", flush=True)

arr = lambda k: np.array([r[k] for r in rows])
print(f"\nSUMMARY over 5 seeds:")
print(f"  rho: {arr('rho').mean():.3f} +- {arr('rho').std():.3f}  (gate rho>=... fires {int(arr('gate_fires').sum())}/5)")
print(f"  entropy: {arr('entropy').mean():.3f} +- {arr('entropy').std():.3f}  (stable? benchmark ref ~0.05-0.07)")
print(f"  FORCED-ON(1) lift: {arr('lift1').mean():+.3f} +- {arr('lift1').std():.3f}  (positive all seeds? {bool((arr('lift1')>0).all())})")
print(f"  diff OFF {arr('diff_OFF').mean():.3f} -> ON(1) {arr('diff_ON1').mean():.3f}")
json.dump(rows, open(os.path.join(HERE, "a3_seed_robustness.json"), "w"), indent=1)
