"""Recompute Table 5 (source-of-gain: cross-channel vs marginal) and Table 6 (head decomposition)
for WADI_clean on the FIXED difficult subset (FIX 3 -> 30 windows, read from scores_WADI_clean.npz)
with the FIX 2 clip applied to the learned inputs. HAI/SWaT are unchanged (their scores/difficulty
are byte-identical) so we only recompute WADI. Arms/heads verbatim from e5_gain_clean.py /
rev4_ablation.py; difficulty from the rebuilt npz maxz/maxz_thr.
"""
from __future__ import annotations
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
POC = r"E:\Projects\Backlog\LatAD\poc"
os.chdir(POC); sys.path.insert(0, POC)
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from models_vade import train_vade, _as_tensor, _recon_energy
import eda_real as E

NAME = "WADI_clean"; K, LD = 20, 10; NSEEDS = 5
OUT = os.path.dirname(os.path.abspath(__file__))

D = E.load(NAME)
Xn = D["Xn_w"].astype(np.float32); Xa = D["Xa_w"].astype(np.float32); y = D["ya_w"].astype(int)
clipv = E.CLIP.get(NAME)
mu, sig = Xn.mean(0), Xn.std(0) + 1e-8
Zn = np.clip((Xn - mu) / sig, -clipv, clipv).astype(np.float32) if clipv else ((Xn - mu) / sig).astype(np.float32)
Za = np.clip((Xa - mu) / sig, -clipv, clipv).astype(np.float32) if clipv else ((Xa - mu) / sig).astype(np.float32)
kd = min(80, max(20, len(Xn) // 10))

d = np.load(os.path.join(OUT, f"scores_{NAME}.npz"))
thr = float(d["maxz_thr"]); maxz = d["maxz"]
hard = (y == 1) & (maxz <= thr)
keep = np.where((y == 0) | hard)[0]
print(f"{NAME}: difficult (FIX3) = {int(hard.sum())} windows, normals = {int((y==0).sum())}", flush=True)


def z(v, ref):
    return (v - ref.mean()) / (ref.std() + 1e-9)


def marginal_product_score(Ztr, Zte, seed, k_marg=5, max_fit=20000):
    rng = np.random.default_rng(seed); n, dd = Ztr.shape
    fit = Ztr if n <= max_fit else Ztr[rng.choice(n, max_fit, replace=False)]
    nll = np.zeros(len(Zte))
    for j in range(dd):
        col = fit[:, j:j + 1]; kj = min(k_marg, max(1, len(np.unique(col))))
        try:
            g = GaussianMixture(n_components=kj, covariance_type="full", reg_covar=1e-3, random_state=seed).fit(col)
            nll += -g.score_samples(Zte[:, j:j + 1])
        except Exception:
            m, s = col.mean(), col.std() + 1e-9; nll += 0.5 * ((Zte[:, j] - m) / s) ** 2
    return nll


cc, mp = [], []          # Table 5
heads = {h: [] for h in ["recon", "density", "nearest", "base", "base+resid", "LatAD"]}  # Table 6
for seed in range(NSEEDS):
    # Table 5 cross-channel latent (density only)
    v = train_vade(Zn, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
    v.fit_latent_density(Zn, k_density=kd, seed=seed)
    s_cc = np.asarray(v.anomaly_score_hard(Za, use_near=False, use_recon=False, use_resid=False, use_basin=False), float)
    cc.append(roc_auc_score(y[keep], np.nan_to_num(s_cc)[keep]))
    mp.append(roc_auc_score(y[keep], np.nan_to_num(marginal_product_score(Zn, Za, seed))[keep]))
    # Table 6 heads (full model)
    v.fit_residual_whitener(Zn); v.fit_resid_head(Zn); v.fit_basin_head(Zn)
    dens, diagnll = v._hard_components(Za); dens_tr, diagnll_tr = v._hard_components(Zn)
    s_density = z(np.asarray(dens), np.asarray(dens_tr)); s_near = z(np.asarray(diagnll), np.asarray(diagnll_tr))
    xt, xtr = _as_tensor(Za, v), _as_tensor(Zn, v)
    r_te = np.asarray(_recon_energy(xt, v.decode(v.encode(xt)[0]), v.res_whitener))
    r_tr = np.asarray(_recon_energy(xtr, v.decode(v.encode(xtr)[0]), v.res_whitener))
    s_recon = (r_te - r_tr.mean()) / (r_tr.std() + 1e-9)
    s_base = np.asarray(v.anomaly_score_hard(Za, use_near=True))
    s_br = np.asarray(v.anomaly_score_hard(Za, use_near=True, use_resid="auto"))
    s_lat = np.asarray(v.anomaly_score_hard(Za, use_resid="auto", use_basin="auto"))
    for h, s in dict(recon=s_recon, density=s_density, nearest=s_near, base=s_base,
                     **{"base+resid": s_br}, LatAD=s_lat).items():
        heads[h].append(roc_auc_score(y[keep], np.nan_to_num(s)[keep]))
    print(f"  seed {seed} done", flush=True)

cc, mp = np.array(cc), np.array(mp)
res = {"name": NAME, "n_difficult": int(hard.sum()), "n_normal": int((y == 0).sum()),
       "table5_source_of_gain": {
           "crosschannel_latent_mean": round(float(cc.mean()), 4), "crosschannel_latent_sd": round(float(cc.std()), 4),
           "marginal_product_mean": round(float(mp.mean()), 4), "marginal_product_sd": round(float(mp.std()), 4),
           "gain": round(float(cc.mean() - mp.mean()), 4)},
       "table6_heads": {h: {"auroc": round(float(np.mean(v)), 3), "auroc_sd": round(float(np.std(v)), 3)}
                        for h, v in heads.items()}}
json.dump(res, open(os.path.join(OUT, "clean_wadi_tables56.json"), "w"), indent=1)
print("\nTable 5 (WADI clean, n_diff=%d):" % int(hard.sum()))
print(f"  cross-channel {cc.mean():.4f}+/-{cc.std():.4f}  marginal {mp.mean():.4f}+/-{mp.std():.4f}  gain {cc.mean()-mp.mean():+.4f}")
print("Table 6 heads (WADI clean):")
for h in ["recon", "density", "nearest", "base", "base+resid", "LatAD"]:
    print(f"  {h:12} {np.mean(heads[h]):.3f}+/-{np.std(heads[h]):.3f}")
print("saved -> clean_wadi_tables56.json", flush=True)
