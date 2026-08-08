"""Fuse LatAD with the subspace-bagged VaDE ensemble (both are full detectors, complementary:
LatAD best on HAI, ensemble best on WADI/SWaT). Symmetric fusions (not the too-strict rescue):
max(z), mean(z), and a soft-max of train-normal tail probabilities. Per-seed difficult AUROC +
episode-bootstrap significance vs each dataset's strongest baseline. Exploratory calibration.
"""
from __future__ import annotations
import json, os, numpy as np
from sklearn.metrics import roc_auc_score
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
RNG = np.random.default_rng(0)
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}
COMPET = {"WADI": "IF", "HAI": "AE", "SWaT": "linres"}


def episodes(y):
    eps, i, n = [], 0, len(y)
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1:
                j += 1
            eps.append(np.arange(i, j)); i = j
        else:
            i += 1
    return eps


ALL = {}
for name in ["WADI", "HAI", "SWaT"]:
    d = np.load(f"{OUT}/scores_{name}.npz"); y = d["label"].astype(int); thr = float(d["maxz_thr"])
    hard = (y == 1) & ~((y == 1) & (d["maxz"] > thr)); keep = np.where((y == 0) | hard)[0]
    es = np.load(f"{OUT}/ens_scores_{name}.npz"); ens = es["ens_top3"]
    ze = (ens - ens.mean()) / ens.std()
    lat = d["LatAD"]; compet = d[COMPET[name]]
    def au(s, idx=keep): return float(roc_auc_score(y[idx], s[idx]))

    nm = y == 0
    def surv(ref, v):
        order = np.sort(ref); r = np.searchsorted(order, v, side="right") / len(order)
        return -np.log(np.clip(1.0 - r, 1e-4, 1.0))          # -log tail prob on test-normal
    te = surv(ens[nm], ens)
    def fused_seed(kind):
        out = []
        for i in range(lat.shape[0]):
            zl = (lat[i] - lat[i].mean()) / lat[i].std()
            if kind == "max":
                out.append(np.maximum(zl, ze))
            elif kind == "mean":
                out.append(0.5 * (zl + ze))
            else:  # tailmax: max of calibrated -log tail probabilities (evidence)
                tl = surv(lat[i][nm], lat[i]); out.append(np.maximum(tl, te))
        return np.stack(out)

    res = {"LatAD": round(float(np.mean([au(lat[i]) for i in range(lat.shape[0])])), 3),
           "ensemble": round(au(ze), 3),
           "compet": COMPET[name],
           "compet_auroc": round(float(np.mean([au(compet[i]) for i in range(compet.shape[0])]) if compet.ndim == 2 else au(compet)), 3)}
    for kind in ["max", "mean", "tailmax"]:
        fs = fused_seed(kind)
        res[f"fused_{kind}"] = (round(float(np.mean([au(fs[i]) for i in range(fs.shape[0])])), 3),
                                round(float(np.std([au(fs[i]) for i in range(fs.shape[0])])), 3))
    # significance of fused_max vs competitor
    W, st = CFG[name]; L = int(np.ceil(W / st)) + 1
    norm = np.where(y == 0)[0]; heps = [e[hard[e]] for e in episodes(y) if hard[e].any()]
    fs = fused_seed("max"); cm = compet if compet.ndim == 2 else compet[None, :]
    def mAU(mat, idx): return float(np.mean([roc_auc_score(y[idx], mat[i][idx]) for i in range(mat.shape[0])]))
    dpt = mAU(fs, keep) - mAU(cm, keep); diffs = []
    for _ in range(2000):
        nb = int(np.ceil(len(norm) / L)); ss = RNG.integers(0, max(1, len(norm) - L + 1), size=nb)
        sn = np.concatenate([norm[s:s + L] for s in ss])[:len(norm)]
        pk = RNG.integers(0, len(heps), size=len(heps)); sh = np.concatenate([heps[k] for k in pk])
        idx = np.concatenate([sn, sh])
        if y[idx].sum() < 2 or (y[idx] == 0).sum() < 2:
            continue
        diffs.append(mAU(fs, idx) - mAU(cm, idx))
    diffs = np.array(diffs)
    res["fused_max_vs_compet"] = dict(diff=round(float(dpt), 3),
        ci=[round(float(np.quantile(diffs, 0.025)), 3), round(float(np.quantile(diffs, 0.975)), 3)],
        p_le_0=round(float((diffs <= 0).mean()), 4))
    ALL[name] = res
    print(f"{name:5}: LatAD {res['LatAD']}  ens {res['ensemble']}  fused_max {res['fused_max']}  "
          f"fused_mean {res['fused_mean']}  | vs {res['compet']} {res['compet_auroc']}: "
          f"diff {res['fused_max_vs_compet']['diff']} CI {res['fused_max_vs_compet']['ci']} "
          f"P(<=0)={res['fused_max_vs_compet']['p_le_0']}")
json.dump(ALL, open(f"{OUT}/ens_fuse.json", "w"), indent=1)
print(f"\nsaved -> {OUT}/ens_fuse.json")
