"""Recompute Table 2 (single trained model) WADI column on the CLEAN 30-window subset:
(a) reconstruction residual, (b) latent NLL nearest-component, (a)+(b) joint. HAI unchanged.
Verbatim head defs from clean_wadi_tables56.py; single model = seed 0."""
from __future__ import annotations
import os, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
POC = r"E:\Projects\Backlog\LatAD\poc"; os.chdir(POC); sys.path.insert(0, POC)
from sklearn.metrics import roc_auc_score
from models_vade import train_vade, _as_tensor, _recon_energy
import eda_real as E
NAME = "WADI_clean"; K, LD = 20, 10; OUT = os.path.join(POC, "_diagnostics")
D = E.load(NAME); Xn = D["Xn_w"].astype(np.float32); Xa = D["Xa_w"].astype(np.float32); y = D["ya_w"].astype(int)
clipv = E.CLIP.get(NAME); mu, sig = Xn.mean(0), Xn.std(0) + 1e-8
Zn = np.clip((Xn-mu)/sig, -clipv, clipv).astype(np.float32) if clipv else ((Xn-mu)/sig).astype(np.float32)
Za = np.clip((Xa-mu)/sig, -clipv, clipv).astype(np.float32) if clipv else ((Xa-mu)/sig).astype(np.float32)
kd = min(80, max(20, len(Xn)//10))
d = np.load(os.path.join(OUT, f"scores_{NAME}.npz")); thr = float(d["maxz_thr"]); maxz = d["maxz"]
keep = np.where((y == 0) | ((y == 1) & (maxz <= thr)))[0]
def z(v, r): return (v - r.mean())/(r.std()+1e-9)
au = lambda s: round(float(roc_auc_score(y[keep], np.nan_to_num(s)[keep])), 3)
v = train_vade(Zn, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=0, device="cpu")
v.fit_latent_density(Zn, k_density=kd, seed=0); v.fit_residual_whitener(Zn); v.fit_resid_head(Zn)
dens, diagnll = v._hard_components(Za); dens_tr, diagnll_tr = v._hard_components(Zn)
s_near = z(np.asarray(diagnll), np.asarray(diagnll_tr))
xt, xtr = _as_tensor(Za, v), _as_tensor(Zn, v)
r_te = np.asarray(_recon_energy(xt, v.decode(v.encode(xt)[0]), v.res_whitener))
r_tr = np.asarray(_recon_energy(xtr, v.decode(v.encode(xtr)[0]), v.res_whitener))
s_recon = (r_te - r_tr.mean())/(r_tr.std()+1e-9)
print(f"Table 2 WADI_clean (single model seed0, n_diff={int(((y==1)&(maxz<=thr)).sum())}):")
print(f"  (a) reconstruction residual : {au(s_recon)}")
print(f"  (b) latent NLL (nearest)    : {au(s_near)}")
print(f"  (a)+(b) joint NLL           : {au(s_recon + s_near)}")
