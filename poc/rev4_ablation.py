"""Rev4 Tier-2 item 8a: full score-head ablation, MULTI-SEED, ALL THREE datasets.

Extends Table 2 (currently single-model, WADI+HAI only) to the reviewer's requested
decomposition: for each seed and dataset, difficult-subset AUROC of every score head and
the cumulative ladder, so the paper can show WHICH head causes each gain.

Heads:
  recon      = whitened reconstruction residual energy (the term LatAD drops)
  density    = high-K latent GMM density NLL
  nearest    = nearest-diagonal-component NLL
  base       = density + nearest (z-normalised sum)   [reported base]
  base+resid = base + auto-gated responsibility-weighted residual
  LatAD      = base + resid(auto) + basin(auto)        [reported model; matches npz]

Retrains VaDE 5 seeds x 3 datasets on CPU (matches build_scores_table config). Also dumps
per-head per-window scores to _diagnostics/heads_<DS>.npz for reuse. Difficult split =
canonical max|z| (from scores_<DS>.npz). Output -> _diagnostics/rev4_ablation.json.
"""
from __future__ import annotations
import json, os, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from models_vade import train_vade, _as_tensor, _recon_energy
import eda_real as E

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}
SEEDS = [0, 1, 2, 3, 4]


def diff_auroc(y, s, hard):
    keep = (y == 0) | hard
    return float(roc_auc_score(y[keep], s[keep]))


def z(v, ref):
    m, sd = ref
    return (v - m) / sd


ALL = {}
for name in ["WADI", "HAI", "SWaT"]:
    K, LD = CFG[name]
    D = E.load(name)
    Xtr0 = np.asarray(D["Xn_w"], float); Xte0 = np.asarray(D["Xa_w"], float)
    y = np.asarray(D["ya_w"], int)
    d = np.load(f"{OUT}/scores_{name}.npz")
    thr = float(d["maxz_thr"]); hard = (y == 1) & ~((y == 1) & (d["maxz"] > thr))
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
    kd = min(80, max(20, len(Xtr0) // 10))

    heads = {h: [] for h in ["recon", "density", "nearest", "base", "base+resid", "LatAD"]}
    dump = {h: [] for h in heads}
    for sd in SEEDS:
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
        v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd)
        v.fit_resid_head(Xtr); v.fit_basin_head(Xtr)
        # per-head raw components
        dens, diagnll = v._hard_components(Xte)
        dens_tr, diagnll_tr = v._hard_components(Xtr)
        dref = (float(dens_tr.mean()), float(dens_tr.std() + 1e-9))
        nref = (float(diagnll_tr.mean()), float(diagnll_tr.std() + 1e-9))
        s_density = np.asarray(z(dens, dref))
        s_near = np.asarray(z(diagnll, nref))
        # recon residual energy (the dropped term), z-normed on train
        xt, xtr_t = _as_tensor(Xte, v), _as_tensor(Xtr, v)
        r_te = np.asarray(_recon_energy(xt, v.decode(v.encode(xt)[0]), v.res_whitener))
        r_tr = np.asarray(_recon_energy(xtr_t, v.decode(v.encode(xtr_t)[0]), v.res_whitener))
        s_recon = (r_te - r_tr.mean()) / (r_tr.std() + 1e-9)
        s_base = np.asarray(v.anomaly_score_hard(Xte, use_near=True))
        s_base_resid = np.asarray(v.anomaly_score_hard(Xte, use_near=True, use_resid="auto"))
        s_latad = np.asarray(v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto"))
        cur = dict(recon=s_recon, density=s_density, nearest=s_near,
                   base=s_base, **{"base+resid": s_base_resid}, LatAD=s_latad)
        for h, s in cur.items():
            heads[h].append(diff_auroc(y, s, hard)); dump[h].append(s.astype(np.float32))
        print(f"  {name} seed {sd} done", flush=True)

    res = {}
    for h in heads:
        a = np.array(heads[h]); res[h] = dict(auroc=round(float(a.mean()), 3), auroc_sd=round(float(a.std()), 3))
    ALL[name] = {"n_difficult": int(hard.sum()), "heads": res}
    np.savez(f"{OUT}/heads_{name}.npz", **{h: np.stack(dump[h]) for h in dump}, label=y, hard=hard)
    print(f"\n=== {name} (difficult n={int(hard.sum())}) ===")
    for h in ["recon", "density", "nearest", "base", "base+resid", "LatAD"]:
        print(f"    {h:12} difficult-AUROC {res[h]['auroc']}±{res[h]['auroc_sd']}")

json.dump(ALL, open(f"{OUT}/rev4_ablation.json", "w"), indent=1)
print(f"\nsaved -> {OUT}/rev4_ablation.json")
