"""Robustness of the train-normal vs test-normal mismatch to feature-space choices.
Variants: baseline (z-score, floored, clip 50) | drop floored channels | rank (train-CDF) transform | drop top-2 / top-10 shifted
channels | drop all channels with test-normal mean|z|>2 (diagnostic only; uses test-normal). Metrics: energy distance (even subsample,
block-perm p), within-train fifth-vs-rest max energy (calibration), C2ST blocked-LR AUC, pooled-KMeans32 mass in <1%-train modes,
relative OOS ratio at q99 (random 10% held-out). Merges into cov_coverage.json under 'robustness'.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cov_c2st import two_sample, c2st, occupancy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COV = os.path.join(ROOT, "_diagnostics", "cov_coverage.json")
DSS = sys.argv[1:] or ["WADI", "HAI", "SWaT"]


def measures(Zn, Zt, n_perm=300):
    n = len(Zn); out = {}
    ts = two_sample(Zn, Zt, seed=0, sub="even", n_perm=n_perm)
    out.update(energy=ts["energy"], H=ts["energy_coef_H"], p_block=ts["p_energy_blockperm"], null_block_max=ts["energy_null_block"]["max"])
    fe = []
    for j in range(5):
        lo, hi = (j * n) // 5, ((j + 1) * n) // 5; m = np.zeros(n, bool); m[lo:hi] = True
        fe.append(two_sample(Zn[m], Zn[~m], seed=0, sub="even", n_perm=30)["energy"])
    out["within_train_fifth_energy_max"] = float(max(fe)); out["within_train_fifth_energy_mean"] = float(np.mean(fe))
    out["energy_ratio_to_max_fifth"] = float(ts["energy"] / max(fe))
    out["c2st_auc"] = c2st(Zn, Zt)["auc"]
    pooled = np.vstack([Zn, Zt]); P = PCA(min(20, Zn.shape[1]), random_state=0).fit_transform(pooled)
    km = KMeans(32, n_init=4, random_state=0).fit(P); oc, _, _ = occupancy(km.labels_[:n], km.labels_[n:], 32)
    out["km32_tv"] = oc["tv"]; out["km32_mass_test_in_train_lt1pct"] = oc["mass_test_in_modes_with_train_lt_1pct"]
    hidx = np.random.default_rng(0).choice(n, int(0.1 * n), replace=False); msk = np.ones(n, bool); msk[hidx] = False
    nn = NearestNeighbors(n_neighbors=2, algorithm="brute", n_jobs=-1).fit(Zn[msk])
    dl = nn.kneighbors(Zn[msk])[0][:, 1]; t = np.quantile(dl, 0.99)
    a = (nn.kneighbors(Zn[hidx])[0][:, 0] > t).mean(); bb = (nn.kneighbors(Zt)[0][:, 0] > t).mean()
    out["oos_q99_heldout"] = float(a); out["oos_q99_test_normal"] = float(bb); out["oos_ratio_q99"] = float(bb / max(a, 1e-9))
    return out


cov = json.load(open(COV))
for ds in DSS:
    t0 = time.time()
    b = np.load(os.path.join(ROOT, "sota_bundle", "ens_bundle", f"bundle_{ds}.npz"))
    Xn, Xa, y = b["Xn_w"].astype(np.float64), b["Xa_w"].astype(np.float64), b["y"]; nrm = y == 0
    mu, sd = Xn.mean(0), Xn.std(0); fl = 0.01 * np.median(sd); floored = sd < fl; sdf = np.maximum(sd, fl)
    Zn, Zt = (Xn - mu) / sdf, np.clip((Xa[nrm] - mu) / sdf, -50, 50)
    shift = np.abs(Zt).mean(0); shift[floored] = 0; order = np.argsort(shift)[::-1]
    # rank transform: each channel mapped through the TRAIN empirical CDF (scale-free, outlier-proof)
    def rank_tf(X):
        R = np.empty_like(X)
        for j in range(X.shape[1]):
            R[:, j] = np.searchsorted(np.sort(Xn[:, j]), X[:, j], side="right") / len(Xn)
        return R
    Rn, Rt = rank_tf(Xn), rank_tf(Xa[nrm])
    keep_nf = ~floored
    if ds == "HAI":   # foreground budget: even-in-time 4000/side
        from cov_c2st import even
        Zn, Zt, Rn, Rt = even(Zn, 4000)[0], even(Zt, 4000)[0], even(Rn, 4000)[0], even(Rt, 4000)[0]
    variants = {
        "baseline_zscore_floor_clip50": (Zn, Zt),
        "drop_floored_channels": (Zn[:, keep_nf], Zt[:, keep_nf]),
        "rank_train_cdf_all_channels": (Rn[:, keep_nf], Rt[:, keep_nf]),
        "drop_top2_shifted": (np.delete(Zn, order[:2], 1), np.delete(Zt, order[:2], 1)),
        "drop_top10_shifted": (np.delete(Zn, order[:10], 1), np.delete(Zt, order[:10], 1)),
        "drop_shift_gt2sd": (Zn[:, shift <= 2], Zt[:, shift <= 2]),
    }
    R = dict(n_floored=int(floored.sum()), top10_shifted_feats=[(int(i), float(shift[i])) for i in order[:10]],
             n_chan_shift_gt2=int((shift > 2).sum()), n_chan_shift_gt1=int((shift > 1).sum()), variants={})
    print(f"== {ds}: floored {floored.sum()}, shift>2SD {R['n_chan_shift_gt2']}, >1SD {R['n_chan_shift_gt1']}, top {R['top10_shifted_feats'][:4]}")
    for name, (A, B) in variants.items():
        m = measures(A, B); m["dim"] = int(A.shape[1]); R["variants"][name] = m
        print(f"  {name:30s} d={m['dim']:4d} E={m['energy']:.3f} H={m['H']:.4f} p={m['p_block']:.3f} withinmax={m['within_train_fifth_energy_max']:.3f} ratio={m['energy_ratio_to_max_fifth']:.2f} AUC={m['c2st_auc']:.3f} km32 mass<1%={m['km32_mass_test_in_train_lt1pct']:.3f} TV={m['km32_tv']:.3f} OOSratio={m['oos_ratio_q99']:.1f} ({m['oos_q99_test_normal']:.3f}/{m['oos_q99_heldout']:.3f})")
    RB = os.path.join(ROOT, "_diagnostics", "cov_robust.json"); rb = json.load(open(RB)) if os.path.exists(RB) else {}; rb[ds] = R; json.dump(rb, open(RB, "w"), indent=1)
    cov = json.load(open(COV)); cov.setdefault(ds, {})["robustness"] = R; json.dump(cov, open(COV, "w"), indent=1)
    print(f"  [{time.time()-t0:.0f}s]")
