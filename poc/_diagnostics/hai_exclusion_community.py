"""HAI coverage-block exclusion on the HEADLINE regime-community model (re-run of cov_excl2 but for
the community ensemble, not just the global LatAD). Same K32 train-empty block set (0 anomalies removed).
Reports AUROC / difficult-AUROC / TPR@1%/5%FPR with -> without the uncovered-regime block, 5-seed mean,
for the community heads (null+HC, HCcoh+LatAD, cohmax+LatAD) vs global LatAD.
"""
from __future__ import annotations
import os, sys, json, numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ensemble_final import ensemble_scores

# exact HAI K32 exclusion blocks (from cov_coverage.json, criterion: train share <0.1%, runs>=30, y==0)
BLOCKS = [[0,177],[282,318],[599,658],[669,856],[878,1114],[1118,1184],[1191,1315],[1319,1378],
          [1490,1539],[1575,1698],[1765,1820],[1827,1965],[2041,2228],[2316,2347],[2352,2500],
          [2586,2618],[2668,2718],[6359,6416],[6436,7080],[13210,13242],[13399,13441],[14264,14314]]


def metrics(scr, y, hard, keep):
    """AUROC(all/difficult) and TPR at fixed 1%/5% FPR, on the kept windows."""
    yk = y[keep]; sk = scr[keep]
    au_all = roc_auc_score(yk, sk)
    hk = hard[keep]; sel = (yk == 0) | hk
    au_diff = roc_auc_score(yk[sel], sk[sel]) if 0 < yk[sel].sum() < sel.sum() else float("nan")
    neg = np.sort(sk[yk == 0])
    def tpr(f):
        t = neg[max(0, int((1 - f) * len(neg)) - 1)]
        return float((sk[yk == 1] > t).mean())
    return dict(auroc=au_all, auroc_difficult=au_diff, tpr_at_fpr1=tpr(0.01), tpr_at_fpr5=tpr(0.05))


def main():
    ens, y, d, nseed = ensemble_scores("HAI")
    triv = d["maxz"]; thr = float(d["maxz_thr"]); hard = (y == 1) & ~((y == 1) & (triv > thr))
    excl = np.zeros(len(y), bool)
    for a, b in BLOCKS: excl[a:b + 1] = True
    excl &= (y == 0)                                  # only normals (by construction all these are y==0)
    keep = ~excl
    assert (y[excl] == 0).all(), "exclusion touched an anomaly!"
    print(f"HAI: n={len(y)} anom={int((y==1).sum())} difficult={int(hard.sum())} "
          f"excluded_normals={int(excl.sum())} ({excl.sum()/(y==0).sum():.1%}) anoms_excluded=0")

    methods = {"LatAD (global)": d["LatAD"], "null+HC": ens["null+HC"],
               "HCcoh+LatAD": ens["HCcoh+LatAD"], "cohmax+LatAD": ens["cohmax+LatAD"]}
    out = {}
    for name, arr in methods.items():
        A = arr if arr.ndim > 1 else arr[None]
        ns = min(nseed, len(A))
        full = [metrics(A[s], y, hard, np.ones(len(y), bool)) for s in range(ns)]
        wo = [metrics(A[s], y, hard, keep) for s in range(ns)]
        agg = lambda L, k: float(np.mean([r[k] for r in L]))
        row = {k: (round(agg(full, k), 4), round(agg(wo, k), 4)) for k in full[0]}
        out[name] = row
        print(f"\n  {name}")
        for k, (f, w) in row.items():
            print(f"     {k:18s} {f:.4f} -> {w:.4f}   (delta {w-f:+.4f})")
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hai_exclusion_community.json"), "w"), indent=1)
    print("\nsaved hai_exclusion_community.json")


if __name__ == "__main__":
    main()
