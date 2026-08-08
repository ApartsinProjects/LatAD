"""Random-subspace density ensemble (user idea): estimate density over many random channel
subsets so a sparse coherent anomaly (a few channels) is not diluted by the other ~118.

For K random subsets S of size m: fit a shrinkage-covariance Gaussian on train-normal (half A),
score each test window's per-subset Mahalanobis D^2, standardize each subset's D^2 on held-out
train-normal (half B), aggregate across subsets (max = most-anomalous subset; also top-5 mean).
A window whose anomaly sits in a few channels lights up the subsets that contain them.

Reports difficult-subset AUROC vs LatAD / IF / the correlation-community common-mode head, and
the miss-rescue-wrapped fusion with LatAD (so it cannot harm the SWaT ceiling). Grounded in
A5 (few levers -> low-dim subsets capture them) and A2 (many modes -> many subsets).
Exploratory calibration (test-normal reference); see cycle_lib note.
"""
from __future__ import annotations
import json, os, numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
import cycle_lib as CL

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
K, MS = 400, [4, 8, 16]
RNG = np.random.default_rng(0)


def subspace_scores(A, B, Za, m, K):
    """max and top5-mean of standardized per-subset Mahalanobis over K random size-m subsets."""
    nch = A.shape[1]
    zt = np.zeros((K, len(Za))); ok = 0
    for _ in range(K):
        S = RNG.choice(nch, size=m, replace=False)
        lw = LedoitWolf().fit(A[:, S]); mu = A[:, S].mean(0)
        P = lw.precision_
        def maha(X):
            d = X[:, S] - mu; return np.einsum('ij,jk,ik->i', d, P, d)
        dB = maha(B); da = maha(Za)
        zt[ok] = (da - dB.mean()) / (dB.std() + 1e-9); ok += 1
    zt = zt[:ok]
    top5 = np.sort(zt, axis=0)[-5:].mean(0)
    return zt.max(0), top5


ALL = {}
for name in ["WADI", "HAI", "SWaT"]:
    bl = CL.load_blocks(name); y = bl["y"]
    Zn, Za = bl["Zn"], bl["Za"]; nA = len(Zn) // 2; A, B = Zn[:nA], Zn[nA:]
    d = np.load(f"{OUT}/scores_{name}.npz"); thr = float(d["maxz_thr"])
    hard = (y == 1) & ~((y == 1) & (d["maxz"] > thr)); keep = np.where((y == 0) | hard)[0]
    latad = d["LatAD"].mean(0); IFm = d["IF"].mean(0)
    cmh = CL.common_mode_head(bl)
    def A_(s): return round(float(roc_auc_score(y[keep], s[keep])), 3)
    row = dict(LatAD=A_(latad), IF=A_(IFm), common_mode_head=A_(cmh), by_m={})
    for m in MS:
        smax, stop5 = subspace_scores(A, B, Za, m, K)
        # rescue-wrap the max-subspace score with LatAD (per-seed mean latad here)
        resc, nf, nfn = CL.rescue_wrap(latad, smax, y)
        row["by_m"][f"m{m}"] = dict(sub_max=A_(smax), sub_top5=A_(stop5),
                                    rescued_LatAD=A_(resc), n_fire=nf, n_fire_normal=nfn)
    ALL[name] = row
    print(f"\n=== {name} (difficult n={int(hard.sum())}) ===  LatAD {row['LatAD']}  IF {row['IF']}  cmHead {row['common_mode_head']}")
    for m, r in row["by_m"].items():
        print(f"   {m}: sub_max {r['sub_max']}  sub_top5 {r['sub_top5']}  rescued_LatAD {r['rescued_LatAD']}  (fires {r['n_fire']}, {r['n_fire_normal']} on normal)")
json.dump(ALL, open(f"{OUT}/random_subspace.json", "w"), indent=1)
print(f"\nsaved -> {OUT}/random_subspace.json")
