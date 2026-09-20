"""Partition variants for the community experts, TRAIN-NORMAL properties only (no training, no test).
Reproduces modal_experts' HAC construction from the local bundle and enumerates variants:
  nested cover: linkage in {average, complete, single, weighted}, MAXSZ in {15, 25, 40}, MINSZ in {3, 4}
  flat partition: fcluster on the same linkage at distance thresholds (1-|rho|) in {0.3, 0.5, 0.7}, size>=3
Reports per variant: S, size-weighted mean cohesion, plain mean cohesion, channel coverage, and (flat only)
Newman modularity on the |rho| graph. Also lists, for the CURRENT SWaT partition, which communities hold the
attacked channels of each difficult episode (train-normal structure question).
Usage: python partition_variants.py SWaT_canon [WADI_clean HAI]   ->  _diagnostics/partition_variants.json"""
from __future__ import annotations
import os, sys, json, numpy as np
from scipy.cluster.hierarchy import linkage, to_tree, fcluster
from scipy.spatial.distance import squareform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE = os.path.dirname(os.path.abspath(__file__))


def corr_graph(name):
    B = np.load(os.path.join(ROOT, "sota_bundle", "ens_bundle", f"bundle_{name}.npz"))
    Xn = B["Xn_w"].astype(float); nch = int(B["nch"])
    means = Xn[:, :nch]; sdc = means.std(0); active = np.where(sdc > 1e-6)[0]
    Z = (means[:, active] - means[:, active].mean(0)) / sdc[active]
    C = np.abs(np.nan_to_num(np.corrcoef(Z.T))); np.fill_diagonal(C, 0.0)
    return C, active, nch


def nested(C, active, method, maxsz, minsz):
    dist = 1 - C; dist = (dist + dist.T) / 2
    L = linkage(squareform(dist, checks=False), method=method); _, nodes = to_tree(L, rd=True)
    leaves = lambda n: [n.id] if n.is_leaf() else leaves(n.left) + leaves(n.right)
    seen, comms = set(), []
    for n in nodes:
        if not n.is_leaf():
            lv = tuple(sorted(leaves(n)))
            if minsz <= len(lv) <= maxsz and lv not in seen:
                seen.add(lv); comms.append([int(active[i]) for i in lv])
    return comms, L


def flat(C, active, L, thr, minsz=3):
    lab = fcluster(L, t=thr, criterion="distance")
    comms = [[int(active[i]) for i in np.where(lab == k)[0]] for k in np.unique(lab)]
    return [g for g in comms if len(g) >= minsz], lab


def stats(C, active, comms):
    a2p = {int(a): i for i, a in enumerate(active)}
    coh, sz = [], []
    for G in comms:
        pos = [a2p[c] for c in G]; sub = C[np.ix_(pos, pos)]
        coh.append(sub.sum() / (len(pos) * (len(pos) - 1))); sz.append(len(pos))
    coh, sz = np.array(coh), np.array(sz)
    covered = len(set(c for G in comms for c in G)) / len(active)
    return dict(S=len(comms), mean_coh=round(float(coh.mean()), 3), wmean_coh=round(float((coh * sz).sum() / sz.sum()), 3),
                median_size=float(np.median(sz)), coverage=round(float(covered), 3))


def modularity(C, lab):
    W = C; k = W.sum(1); m2 = W.sum()
    same = lab[:, None] == lab[None, :]
    return float(((W - np.outer(k, k) / m2) * same).sum() / m2)


OUT = {}
for name in (sys.argv[1:] or ["SWaT_canon"]):
    C, active, nch = corr_graph(name)
    rows = []
    for method in ("average", "complete", "single", "weighted"):
        for maxsz in (15, 25, 40):
            for minsz in (3, 4):
                comms, L = nested(C, active, method, maxsz, minsz)
                r = dict(kind="nested", method=method, maxsz=maxsz, minsz=minsz, **stats(C, active, comms))
                r["current"] = (method == "average" and maxsz == 25 and minsz == 3)
                rows.append(r)
        for thr in (0.3, 0.5, 0.7):
            comms, lab = flat(C, active, L if method else None, thr)
            r = dict(kind="flat", method=method, thr=thr, **stats(C, active, comms), modularity=round(modularity(C, lab), 3))
            rows.append(r)
    OUT[name] = rows
    print(f"\n=== {name}: {len(active)} active channels of {nch} ===")
    for r in rows:
        print("  ", r)
json.dump(OUT, open(os.path.join(HERE, "partition_variants.json"), "w"), indent=1)
