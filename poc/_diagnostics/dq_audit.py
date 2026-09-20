"""Offline data-quality audit for LatAD benchmarks (WADI, HAI, SWaT).
Audit 1: flagged normals (label-FP) -> transition adjacency, out-of-support, coherence.
Audit 2: difficult subset -> overlap duplicates, near-duplicate clusters, episodes, effective N.
No retraining; reuses scores_*.npz, bundle_*.npz, e2_fable_*.npz.
"""
import json, sys, numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

ROOT = "E:/Projects/Backlog/LatAD/poc"
OUT = {}


def runs(mask):
    """contiguous runs of True -> list of (start, length)"""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    br = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br], idx[-1]]
    return [(int(s), int(e - s + 1)) for s, e in zip(starts, ends)]


def near_attack(y, k):
    """mask: within +/-k windows of a y==1 window"""
    a = y == 1
    m = np.zeros_like(a)
    for d in range(1, k + 1):
        m[d:] |= a[:-d]
        m[:-d] |= a[d:]
    return m


def sanity(cond, msg):
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)


for ds in ["WADI", "HAI", "SWaT"]:
    print(f"\n{'='*90}\n{ds}\n{'='*90}")
    S = np.load(f"{ROOT}/_diagnostics/scores_{ds}.npz")
    B = np.load(f"{ROOT}/sota_bundle/ens_bundle/bundle_{ds}.npz")
    F = np.load(f"{ROOT}/_diagnostics/e2_fable_{ds}.npz")
    y = S["label"].astype(int)
    assert np.array_equal(y, B["y"]) and np.array_equal(y, F["yw"]), "row alignment"
    n = len(y)
    Xn, Xa = B["Xn_w"].astype(np.float64), B["Xa_w"].astype(np.float64)
    maxz, maxz_thr = B["maxz"], float(B["maxz_thr"])
    lat = S["LatAD"].mean(0)  # seed-mean LatAD score (higher = more anomalous)
    # standardize features by train-normal stats for distance work
    mu, sd = Xn.mean(0), Xn.std(0); sd = np.maximum(sd, 0.01 * np.median(sd))  # floor near-constant train channels
    print(f'  channels at sd floor: {(Xn.std(0) < 0.01*np.median(Xn.std(0))).sum()} / {Xn.shape[1]}')
    Zn, Za = (Xn - mu) / sd, (Xa - mu) / sd
    R = dict(n_test=int(n), n_normal=int((y == 0).sum()), n_anom=int((y == 1).sum()), n_train=int(len(Xn)))

    # ---------- threshold definitions ----------
    # (A) train-99th-pct threshold on a VaDE mixture score computed on TRAIN windows (e2_fable run)
    mix = lambda L: -logsumexp(F["logpi"][None] + L, axis=1)
    s_tr, s_te = mix(F["logN_tr"]), mix(F["logN_te"])
    thr_tr99 = float(np.quantile(s_tr, 0.99))
    rho = spearmanr(s_te, lat).correlation
    print(f"  e2_fable mixture score vs shipped LatAD seed-mean: Spearman rho={rho:.3f}  (test-normal rho={spearmanr(s_te[y==0], lat[y==0]).correlation:.3f})")
    # (B) paper operating points: 1% and 5% test-normal FPR on shipped LatAD seed-mean
    thr_fpr1 = np.quantile(lat[y == 0], 0.99)
    thr_fpr5 = np.quantile(lat[y == 0], 0.95)
    R["score_agreement_spearman"] = float(rho)

    # ---------- Audit 1 ----------
    # out-of-support: kNN distance (k=1) to train-normal, vs train leave-one-out 99th pct
    nn = NearestNeighbors(n_neighbors=2).fit(Zn)
    d_loo = nn.kneighbors(Zn)[0][:, 1]           # leave-one-out (skip self)
    d_te = nn.kneighbors(Za, n_neighbors=1)[0][:, 0]
    oos_thr = float(np.quantile(d_loo, 0.99))
    oos = d_te > oos_thr
    sanity((d_te[y == 0] > oos_thr).mean() < 0.9, "OOS rate of test normals is not ~100% (feature scaling sane)")
    R["oos_thr_train_loo_p99"] = oos_thr
    R["oos_rate_all_test_normals"] = float(oos[y == 0].mean())
    R["oos_rate_all_test_anoms"] = float(oos[y == 1].mean())
    print(f"  OOS (1-NN dist > train-LOO p99={oos_thr:.2f}): all test normals {oos[y==0].mean():.3f}, all anomalies {oos[y==1].mean():.3f}")

    A1 = {}
    for name, score, thr in [("train99_vade_mixture", s_te, thr_tr99),
                             ("latad_fpr1", lat, thr_fpr1), ("latad_fpr5", lat, thr_fpr5)]:
        flag = (y == 0) & (score > thr)
        nf = int(flag.sum()); nn0 = int((y == 0).sum())
        row = dict(threshold=float(thr), n_flagged=nf, fpr_label=nf / nn0)
        if nf == 0:
            A1[name] = row; print(f"  [{name}] no flagged normals"); continue
        adj = {}
        for k in (1, 2, 3):
            adj[k] = float(near_attack(y, k)[flag].mean())
        row["frac_adjacent_k"] = {str(k): v for k, v in adj.items()}
        # baseline: fraction of ALL normals adjacent (to see enrichment)
        row["base_adjacent_k1_all_normals"] = float(near_attack(y, 1)[y == 0].mean())
        row["frac_oos"] = float(oos[flag].mean())
        rr = runs(flag)
        row["n_runs"] = len(rr); row["largest_run"] = max(l for _, l in rr)
        row["largest_run_start"] = max(rr, key=lambda t: t[1])[0]
        row["runs_len_ge5"] = int(sum(l >= 5 for _, l in rr))
        row["frac_in_runs_ge5"] = float(sum(l for _, l in rr if l >= 5) / nf)
        plaus = (near_attack(y, 1) | oos) & flag
        plaus3 = (near_attack(y, 3) | oos) & flag
        row["n_plausible_mislabel_k1_or_oos"] = int(plaus.sum())
        row["n_plausible_mislabel_k3_or_oos"] = int(plaus3.sum())
        row["n_genuine_normal_flagged_k1"] = int(nf - plaus.sum())
        row["fpr_corrected_k1_or_oos"] = float((nf - plaus.sum()) / (nn0 - plaus.sum()))
        row["fpr_corrected_k3_or_oos"] = float((nf - plaus3.sum()) / (nn0 - plaus3.sum()))
        # if the flagged windows were ONLY adjacency-driven (no OOS), corrected FPR:
        row["fpr_corrected_k1_only"] = float((nf - (near_attack(y, 1) & flag).sum()) / (nn0 - (near_attack(y, 1) & flag).sum()))
        A1[name] = row
        print(f"  [{name}] thr={thr:.3f} flagged={nf}/{nn0} (FPR {nf/nn0:.4f}) adj k1/2/3={adj[1]:.2f}/{adj[2]:.2f}/{adj[3]:.2f} "
              f"(base k1 all-normals {row['base_adjacent_k1_all_normals']:.3f}) OOS={row['frac_oos']:.2f} runs={len(rr)} largest={row['largest_run']}@{row['largest_run_start']} "
              f"plausible(k1|oos)={plaus.sum()} corrFPR={row['fpr_corrected_k1_or_oos']:.4f}")
    R["audit1"] = A1

    # HAI drifted-block check: largest contiguous OOS block among test normals and its LatAD flag rate
    oos_norm_runs = runs(oos & (y == 0))
    if oos_norm_runs:
        s0, L = max(oos_norm_runs, key=lambda t: t[1])
        blk = np.zeros(n, bool); blk[s0:s0 + L] = True
        R["largest_oos_normal_block"] = dict(start=s0, length=L,
                                             frac_flagged_fpr1=float((lat[blk] > thr_fpr1).mean()),
                                             frac_flagged_train99=float((s_te[blk] > thr_tr99).mean()))
        print(f"  largest contiguous OOS-normal block: start={s0} len={L}; flagged@fpr1={R['largest_oos_normal_block']['frac_flagged_fpr1']:.2f} flagged@train99={R['largest_oos_normal_block']['frac_flagged_train99']:.2f}")
    # gap-tolerant version (allow gaps <=5 windows) of OOS-normal blocks
    oosn = oos & (y == 0)
    filled = oosn.copy()
    for s, L in runs(~oosn):
        if L <= 5 and s > 0 and s + L < n: filled[s:s + L] = True
    gr = runs(filled & (y == 0))
    if gr:
        s0, L = max(gr, key=lambda t: t[1])
        R["largest_oos_normal_block_gap5"] = dict(start=s0, length=L, n_oos_inside=int(oosn[s0:s0 + L].sum()))
        print(f"  largest OOS-normal block (gaps<=5 bridged): start={s0} len={L} ({oosn[s0:s0+L].sum()} OOS inside)")

    # ---------- Audit 2 ----------
    diff = (y == 1) & (maxz <= maxz_thr)
    di = np.flatnonzero(diff); nd = len(di)
    A2 = dict(n_difficult=nd)
    print(f"  difficult = {nd} / {int((y==1).sum())} anomalies")
    if nd > 0:
        consec = np.isin(di + 1, di) | np.isin(di - 1, di)
        A2["n_with_consecutive_neighbour"] = int(consec.sum()); A2["frac_consecutive"] = float(consec.mean())
        ep = runs(diff); A2["n_episodes"] = len(ep); A2["episode_lengths"] = [l for _, l in ep]
        # episodes with gap tolerance (attack windows between difficult ones): runs of y==1 that contain a difficult window
        att_runs = runs(y == 1)
        A2["n_attack_runs_containing_difficult"] = int(sum(diff[s:s + L].any() for s, L in att_runs))
        A2["n_attack_runs_total"] = len(att_runs)
        # near-duplicate clusters in feature space (standardized): eps = 5th pct of pairwise dist among train normals? use
        # two definitions: cosine dist < 0.01 and euclidean < 10th pct of train 1-NN LOO distance
        Zd = Za[di]
        if nd > 1:
            D = pdist(Zd, "euclidean"); C = pdist(Zd, "cosine")
            eps_e = float(np.quantile(d_loo, 0.10))  # a distance typical of *closest* train neighbours
            eps_e50 = float(np.quantile(d_loo, 0.50))
            lk = linkage(D, "single")
            cl_e10 = fcluster(lk, eps_e, "distance"); cl_e50 = fcluster(lk, eps_e50, "distance")
            cl_c = fcluster(linkage(C, "single"), 0.01, "distance")
            A2["near_dup"] = dict(eps_euclid_p10_trainLOO=eps_e, n_clusters_euclid_p10=int(cl_e10.max()),
                                  eps_euclid_p50_trainLOO=eps_e50, n_clusters_euclid_p50=int(cl_e50.max()),
                                  n_clusters_cosine_0p01=int(cl_c.max()),
                                  n_windows_with_dup_partner_euclid_p10=int(sum(np.bincount(cl_e10)[cl_e10] > 1)),
                                  median_pairwise_euclid=float(np.median(D)), min_pairwise_euclid=float(D.min()),
                                  median_pairwise_cosine=float(np.median(C)), min_pairwise_cosine=float(C.min()))
            # what fraction of near-dup pairs are consecutive windows (overlap-driven)?
            from scipy.spatial.distance import squareform
            Dm = squareform(D); iu = np.triu_indices(nd, 1)
            close = Dm[iu] < eps_e
            cons_pair = np.abs(di[iu[0]] - di[iu[1]])[close] == 1
            A2["near_dup"]["n_close_pairs"] = int(close.sum()); A2["near_dup"]["frac_close_pairs_consecutive"] = float(cons_pair.mean()) if close.sum() else None
            # 1-NN distance of difficult windows to other difficult vs to train normal
            np.fill_diagonal(Dm, np.inf)
            A2["near_dup"]["median_1nn_within_difficult"] = float(np.median(Dm.min(1)))
            A2["near_dup"]["median_1nn_to_train"] = float(np.median(d_te[di]))
            eff = dict(raw=nd, episodes=len(ep), clusters_e10=int(cl_e10.max()), clusters_e50=int(cl_e50.max()), clusters_cos=int(cl_c.max()))
            A2["effective_N"] = eff
            print(f"  consecutive-overlap: {consec.sum()}/{nd} ({consec.mean():.2f}); episodes={len(ep)} lens={[l for _,l in ep]}; attack-runs containing difficult={A2['n_attack_runs_containing_difficult']}/{len(att_runs)}")
            print(f"  near-dup: euclid eps p10={eps_e:.2f} -> {cl_e10.max()} clusters; p50={eps_e50:.2f} -> {cl_e50.max()}; cosine<0.01 -> {cl_c.max()}; close pairs={close.sum()} of which consecutive {A2['near_dup']['frac_close_pairs_consecutive']}")
            print(f"  median 1NN within-difficult={A2['near_dup']['median_1nn_within_difficult']:.2f} vs to-train={A2['near_dup']['median_1nn_to_train']:.2f}; min pairwise euclid={D.min():.3f} cos={C.min():.4f}")
    # (d) reverse: anomalies near-identical to normals (test normals + train normals)
    nn_te0 = NearestNeighbors(n_neighbors=1).fit(Za[y == 0])
    d_a2n = nn_te0.kneighbors(Za[y == 1])[0][:, 0]
    d_a2tr = d_te[y == 1]
    eps_e = float(np.quantile(d_loo, 0.10)); eps50 = float(np.quantile(d_loo, 0.50))
    inlier = np.minimum(d_a2n, d_a2tr) < eps50
    rev = dict(eps_p50_trainLOO=eps50, n_anom_within_p50_of_a_normal=int(inlier.sum()), frac=float(inlier.mean()),
               n_anom_within_p10=int((np.minimum(d_a2n, d_a2tr) < eps_e).sum()),
               n_anom_closer_to_normal_than_median_train_LOO=int((d_a2tr < np.median(d_loo)).sum()),
               of_which_difficult=int((inlier & diff[y == 1]).sum()))
    # how many anomalies are within +/-1 of a normal (edge windows): share of anomalies at attack boundaries
    a_idx = np.flatnonzero(y == 1); edge = near_attack(1 - y, 1)[a_idx]
    rev["n_anom_at_attack_boundary"] = int(edge.sum()); rev["n_inlier_at_boundary"] = int((inlier & edge).sum())
    A2["reverse_label_noise"] = rev
    print(f"  reverse: anomalies within train-LOO-median dist ({eps50:.2f}) of some normal: {inlier.sum()}/{len(a_idx)} (difficult among them {rev['of_which_difficult']}); at attack boundary {edge.sum()} (inlier&boundary {rev['n_inlier_at_boundary']})")
    R["audit2"] = A2
    OUT[ds] = R

json.dump(OUT, open(f"{ROOT}/_diagnostics/dq_audit.json", "w"), indent=1, default=float)
print("\nsaved dq_audit.json")
