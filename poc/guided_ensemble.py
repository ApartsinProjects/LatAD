"""Guided HAC-community VaDE ensemble (method A). Members are the nested correlation
communities extracted by hierarchical agglomerative clustering on train-normal |rho| (every
dendrogram subtree of size [3, MAXSZ], overlapping by nesting). One VaDE latent-density model
per community; aggregate the community NLLs. Because coverage is DETERMINISTIC (the analyzer
block is always a member), robustness is over VaDE inits only (SEEDS), not over a random draw.

Theory: the aggregate estimates an anomaly score under a hierarchical FACTORIZATION of the
joint density p(x) ~ prod_G p_G(x_G); an anomaly in one factor is one violated term. We test
q90 (robust min-probability), max (strict OR), sum (full factorized NLL), and top-k. All member
scores are standardized on held-out train-normal; the aggregate is re-calibrated on held-out
train-normal (removes multiple-testing inflation).

Reports difficult-subset AUROC per seed (robustness) + tail-max fusion with LatAD (all three).
Output -> _diagnostics/guided_ensemble.json.
"""
from __future__ import annotations
import json, os, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
import eda_real as E

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
SEEDS = [0, 1, 2]
MAXSZ = 25
COMPET = {"WADI": "IF", "HAI": "AE", "SWaT": "linres"}


def hac_communities(Zactive, active, nch):
    C = np.nan_to_num(np.corrcoef(Zactive.T)); dist = 1 - np.abs(C)
    np.fill_diagonal(dist, 0.0); dist = (dist + dist.T) / 2
    L = linkage(squareform(dist, checks=False), method="average")
    _, nodes = to_tree(L, rd=True)
    def leaves(n): return [n.id] if n.is_leaf() else leaves(n.left) + leaves(n.right)
    seen, comms = set(), []
    for n in nodes:
        if not n.is_leaf():
            lv = sorted(leaves(n))
            if 3 <= len(lv) <= MAXSZ and tuple(lv) not in seen:
                seen.add(tuple(lv)); comms.append([int(active[i]) for i in lv])  # -> original channel idx
    return comms


def subset_feats(Xw, nch, S):
    return np.ascontiguousarray(Xw[:, [b * nch + c for b in range(6) for c in S]])


def agg_variants(Zc, Zt, y, keep):
    out = {}
    for nm, fn in [("max", lambda A: A.max(0)), ("q90", lambda A: np.percentile(A, 90, 0)),
                   ("q95", lambda A: np.percentile(A, 95, 0)), ("sum", lambda A: A.sum(0)),
                   ("top5", lambda A: np.sort(A, 0)[-5:].mean(0))]:
        ac, at = fn(Zc), fn(Zt); s = (at - ac.mean()) / (ac.std() + 1e-9)
        out[nm] = s
    return out


import gc
ALL = json.load(open(f"{OUT}/guided_ensemble.json")) if os.path.exists(f"{OUT}/guided_ensemble.json") else {}
for name in ["SWaT"]:
    D = E.load(name); nch = len(D["ch"])
    Xn = np.asarray(D["Xn_w"], float); Xa = np.asarray(D["Xa_w"], float)
    y = np.asarray(D["ya_w"], int)
    d = np.load(f"{OUT}/scores_{name}.npz"); thr = float(d["maxz_thr"])
    hard = (y == 1) & ~((y == 1) & (d["maxz"] > thr)); keep = np.where((y == 0) | hard)[0]
    lat = d["LatAD"]
    means = Xn[:, :nch]; sdc = means.std(0)
    active = np.where(sdc > 1e-6)[0]
    Zact = (means[:, active] - means[:, active].mean(0)) / sdc[active]
    comms = hac_communities(Zact, active, nch)
    nfit = len(Xn) * 4 // 5
    per_seed = {k: [] for k in ["max", "q90", "q95", "sum", "top5"]}
    fused_all = []
    for sd in SEEDS:
        Zc, Zt = [], []
        for G in comms:
            Ff, Fc, Ft = subset_feats(Xn[:nfit], nch, G), subset_feats(Xn[nfit:], nch, G), subset_feats(Xa, nch, G)
            m2, s2 = Ff.mean(0), Ff.std(0) + 1e-9
            v = train_vade(((Ff - m2) / s2).astype(np.float32), n_clusters=min(20, max(6, len(G))),
                           latent_dim=min(8, max(3, len(G) // 2)), epochs=15, warmup=4, seed=sd, device="cpu")
            v.fit_latent_density(((Ff - m2) / s2).astype(np.float32), k_density=min(50, max(12, nfit // 12)))
            cal = np.asarray(v.anomaly_score_hard(((Fc - m2) / s2).astype(np.float32), use_resid=False, use_basin=False))
            sc = np.asarray(v.anomaly_score_hard(((Ft - m2) / s2).astype(np.float32), use_resid=False, use_basin=False))
            cm, cs = cal.mean(), cal.std() + 1e-9
            Zc.append(np.nan_to_num((cal - cm) / cs)); Zt.append(np.nan_to_num((sc - cm) / cs)); del v; gc.collect()
        Zc, Zt = np.stack(Zc), np.stack(Zt)
        variants = agg_variants(Zc, Zt, y, keep)
        for k, s in variants.items():
            per_seed[k].append(round(float(roc_auc_score(y[keep], s[keep])), 3))
        # tail-max fusion of q90-ensemble with LatAD (test-normal calib proxy for LatAD)
        nm = y == 0
        def surv(ref, v):
            o = np.sort(ref); r = np.searchsorted(o, v, side="right") / len(o); return -np.log(np.clip(1 - r, 1e-4, 1))
        eh = variants["q90"]
        for li in range(lat.shape[0]):
            f = np.maximum(surv(lat[li][nm], lat[li]), surv(eh[nm], eh))
            fused_all.append(round(float(roc_auc_score(y[keep], f[keep])), 3))
        print(f"  {name} seed {sd}: " + "  ".join(f"{k} {per_seed[k][-1]}" for k in per_seed), flush=True)
    latau = round(float(np.mean([roc_auc_score(y[keep], lat[i][keep]) for i in range(lat.shape[0])])), 3)
    compet = d[COMPET[name]]; cau = round(float(np.mean([roc_auc_score(y[keep], compet[i][keep]) for i in range(compet.shape[0])]) if compet.ndim == 2 else roc_auc_score(y[keep], compet[keep])), 3)
    ALL[name] = dict(n_communities=len(comms), LatAD=latau, compet=COMPET[name], compet_auroc=cau,
                     ensemble={k: [round(np.mean(v), 3), round(np.std(v), 3)] for k, v in per_seed.items()},
                     fused_q90_tailmax=[round(np.mean(fused_all), 3), round(np.std(fused_all), 3)])
    print(f"=== {name}: {len(comms)} communities | LatAD {latau} (compet {COMPET[name]} {cau}) | "
          f"ens {ALL[name]['ensemble']} | FUSED {ALL[name]['fused_q90_tailmax']}", flush=True)
    json.dump(ALL, open(f"{OUT}/guided_ensemble.json", "w"), indent=1)
print(f"\nsaved -> {OUT}/guided_ensemble.json")
