"""Follow-ups: strict OOS, enrichment ratios, HAI drift blocks / regime-21, SWaT duplicates, reverse-noise index lists."""
import json, numpy as np
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors
ROOT = "E:/Projects/Backlog/LatAD/poc"
J = json.load(open(f"{ROOT}/_diagnostics/dq_audit.json"))

def runs(mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0: return []
    br = np.flatnonzero(np.diff(idx) > 1)
    return [(int(s), int(e - s + 1)) for s, e in zip(np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]])]

def near_attack(y, k):
    a = y == 1; m = np.zeros_like(a)
    for d in range(1, k + 1):
        m[d:] |= a[:-d]; m[:-d] |= a[d:]
    return m

for ds in ["WADI", "HAI", "SWaT"]:
    print(f"\n=== {ds} ===")
    S = np.load(f"{ROOT}/_diagnostics/scores_{ds}.npz"); B = np.load(f"{ROOT}/sota_bundle/ens_bundle/bundle_{ds}.npz")
    F = np.load(f"{ROOT}/_diagnostics/e2_fable_{ds}.npz")
    y = S["label"].astype(int); n = len(y); lat = S["LatAD"].mean(0)
    Xn, Xa = B["Xn_w"].astype(np.float64), B["Xa_w"].astype(np.float64)
    mu, sd = Xn.mean(0), Xn.std(0); sd = np.maximum(sd, 0.01 * np.median(sd))  # floor near-constant train channels
    print(f'  channels at sd floor: {(Xn.std(0) < 0.01*np.median(Xn.std(0))).sum()} / {Xn.shape[1]}'); Zn, Za = (Xn - mu) / sd, (Xa - mu) / sd
    maxz, maxz_thr = B["maxz"], float(B["maxz_thr"]); diff = (y == 1) & (maxz <= maxz_thr)
    R = J[ds]
    # attack structure
    ar = runs(y == 1)
    print(f"  attack runs: {len(ar)}  lengths={[l for _,l in ar][:40]}  starts={[s for s,_ in ar][:40]}")
    R["attack_runs"] = dict(n=len(ar), starts=[s for s, _ in ar], lengths=[l for _, l in ar])
    # exact duplicates
    un_tr = np.unique(Xn.round(6), axis=0).shape[0]; un_te = np.unique(Xa.round(6), axis=0).shape[0]
    print(f"  exact-duplicate rows: train {len(Xn)-un_tr}/{len(Xn)}  test {len(Xa)-un_te}/{len(Xa)}")
    R["exact_duplicates"] = dict(train=int(len(Xn) - un_tr), test=int(len(Xa) - un_te))
    # distances
    nn = NearestNeighbors(n_neighbors=2).fit(Zn)
    d_loo = nn.kneighbors(Zn)[0][:, 1]; d_te = nn.kneighbors(Za, n_neighbors=1)[0][:, 0]
    oos99 = d_te > np.quantile(d_loo, 0.99); oos_max = d_te > d_loo.max()
    print(f"  OOS-strict (> max train LOO={d_loo.max():.2f}): test normals {oos_max[y==0].mean():.3f}, anomalies {oos_max[y==1].mean():.3f}")
    R["oos_strict_thr_train_loo_max"] = float(d_loo.max())
    R["oos_strict_rate_all_test_normals"] = float(oos_max[y == 0].mean())
    thr1, thr5 = np.quantile(lat[y == 0], 0.99), np.quantile(lat[y == 0], 0.95)
    mix = lambda L: -logsumexp(F["logpi"][None] + L, axis=1); s_tr, s_te = mix(F["logN_tr"]), mix(F["logN_te"]); thr99 = np.quantile(s_tr, 0.99)
    nn0 = int((y == 0).sum())
    for name, flag in [("latad_fpr1", (y == 0) & (lat > thr1)), ("latad_fpr5", (y == 0) & (lat > thr5)), ("train99_vade_mixture", (y == 0) & (s_te > thr99))]:
        nf = int(flag.sum()); row = R["audit1"][name]
        if nf == 0: continue
        a1, a3 = near_attack(y, 1), near_attack(y, 3)
        row["enrichment_adjacent_k1"] = float(a1[flag].mean() / max(a1[y == 0].mean(), 1e-9))
        row["frac_oos_strict"] = float(oos_max[flag].mean())
        row["base_oos_p99_all_normals"] = float(oos99[y == 0].mean()); row["base_oos_strict_all_normals"] = float(oos_max[y == 0].mean())
        row["enrichment_oos_p99"] = float(oos99[flag].mean() / max(oos99[y == 0].mean(), 1e-9))
        for lab, crit in [("k1", a1), ("k3", a3), ("oos_strict", oos_max), ("k1_or_oos_strict", a1 | oos_max), ("k3_or_oos_strict", a3 | oos_max), ("k3_or_oos_p99", a3 | oos99)]:
            rm = int((flag & crit).sum())
            row[f"n_{lab}"] = rm; row[f"fpr_corrected_{lab}"] = float((nf - rm) / (nn0 - rm))
        print(f"  [{name}] n={nf} adj-k1 enrich={row['enrichment_adjacent_k1']:.1f}x  OOS-strict={row['frac_oos_strict']:.2f} (base {row['base_oos_strict_all_normals']:.3f}) "
              f"OOS-p99 enrich={row['enrichment_oos_p99']:.1f}x | corrFPR: k1={row['fpr_corrected_k1']:.4f} k3={row['fpr_corrected_k3']:.4f} strict={row['fpr_corrected_oos_strict']:.4f} k3|strict={row['fpr_corrected_k3_or_oos_strict']:.4f} k3|p99={row['fpr_corrected_k3_or_oos_p99']:.4f}")
    # OOS-normal blocks (p99, gaps<=5 bridged) of length >= 30
    oosn = oos99 & (y == 0); filled = oosn.copy()
    for s, L in runs(~oosn):
        if L <= 5 and s > 0 and s + L < n: filled[s:s + L] = True
    big = [(s, L, int(oosn[s:s+L].sum()), float((lat[s:s+L] > thr1).mean()), float((s_te[s:s+L] > thr99).mean())) for s, L in runs(filled & (y == 0)) if L >= 30]
    R["oos_normal_blocks_ge30"] = [dict(start=s, length=L, n_oos=o, frac_flag_fpr1=f1, frac_flag_train99=f9) for s, L, o, f1, f9 in big]
    print(f"  OOS-normal blocks >=30 (gap<=5): {[(s,L,o,round(f1,2),round(f9,2)) for s,L,o,f1,f9 in big]}")
    print(f"  total test normals in such blocks: {sum(L for _,L,_,_,_ in big)} / {nn0}")
    R["n_test_normals_in_oos_blocks_ge30"] = int(sum(L for _, L, _, _, _ in big))
    # regime occupancy (e2_fable VaDE): test-normal regimes with ~zero train share
    r_tr = (F["logpi"][None] + F["logN_tr"]).argmax(1); r_te = (F["logpi"][None] + F["logN_te"]).argmax(1)
    K = F["logpi"].shape[0]; ctr = np.bincount(r_tr, minlength=K) / len(r_tr); cte = np.bincount(r_te[y == 0], minlength=K)
    rare = [(int(k), int(cte[k]), float(ctr[k])) for k in range(K) if cte[k] >= 50 and ctr[k] < 0.002]
    print(f"  test-normal regimes with train share <0.2%: {rare}  (k, n_test_normal, train_share)")
    R["drift_regimes_test_normal"] = [dict(k=k, n_test_normal=c, train_share=t) for k, c, t in rare]
    for k, c, t in rare:
        m = (r_te == k) & (y == 0); rr = runs(m)
        print(f"    regime {k}: {c} test normals, {len(rr)} runs, largest {max(l for _,l in rr)} @ {max(rr,key=lambda t:t[1])[0]}; flagged@fpr1 {(lat[m]>thr1).mean():.2f} @train99 {(s_te[m]>thr99).mean():.2f}; OOS-p99 {oos99[m].mean():.2f}")
        R["drift_regimes_test_normal"][[d["k"] for d in R["drift_regimes_test_normal"]].index(k)].update(
            n_runs=len(rr), largest_run=max(l for _, l in rr), frac_flag_fpr1=float((lat[m] > thr1).mean()), frac_flag_train99=float((s_te[m] > thr99).mean()), frac_oos_p99=float(oos99[m].mean()))
    # reverse noise: list anomaly windows closest to a normal
    nn0m = NearestNeighbors(n_neighbors=1).fit(Za[y == 0]); a_idx = np.flatnonzero(y == 1)
    d_a2n = nn0m.kneighbors(Za[a_idx])[0][:, 0]; dmin = np.minimum(d_a2n, d_te[a_idx]); eps50 = np.quantile(d_loo, 0.5)
    inl = a_idx[dmin < eps50]
    bnd = near_attack(1 - y, 1)
    print(f"  reverse inliers idx={inl.tolist()[:40]}  at-boundary={[bool(bnd[i]) for i in inl][:40]}  difficult={[bool(diff[i]) for i in inl][:40]}")
    R["audit2"]["reverse_label_noise"]["inlier_indices"] = inl.tolist()
    R["audit2"]["reverse_label_noise"]["inlier_runs"] = runs(np.isin(np.arange(n), inl))
    # difficult: which attack runs, how many difficult per run
    per = [(s, L, int(diff[s:s+L].sum())) for s, L in ar if diff[s:s+L].any()]
    R["audit2"]["difficult_per_attack_run"] = [dict(start=s, attack_len=L, n_difficult=d) for s, L, d in per]
    print(f"  difficult per attack run (start, attack_len, n_diff): {per}")
json.dump(J, open(f"{ROOT}/_diagnostics/dq_audit.json", "w"), indent=1, default=float)
print("saved")
