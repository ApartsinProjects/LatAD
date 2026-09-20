"""nr_discovery: partition LatAD flags into latent 'regimes' and separate genuine anomalies from
NEW (unseen-in-training) NORMAL operating regimes. Offline; reuses e2_fable_*.npz, scores_*.npz,
bundle_*.npz. No retraining. Output: nr_discovery.json + stdout log.

Classification rule (label-free, fixed BEFORE looking at labels):
  NEW-NORMAL-REGIME  iff  longest contiguous run >= RUN_MIN
                     and  cohesion ratio (cluster RMS latent spread / median train-regime spread) <= COH_MAX
                     and  PCA-SPE ratio (median cluster SPE / train-normal p99 SPE) <= SPE_MAX
                          (correlation structure of train-normal is kept: off-subspace residual small)
  else GENUINE-ANOMALY.
Labels are used ONLY to validate the verdicts afterwards.
"""
import json, sys, warnings, numpy as np
warnings.filterwarnings('ignore')
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

ROOT = "E:/Projects/Backlog/LatAD/poc"
RUN_MIN, COH_MAX, SPE_MAX = 30, 2.0, 3.0
GAP = 2  # windows: gaps <= GAP are bridged when measuring contiguity
MEMB = {}
OUT = {"rule": dict(RUN_MIN=RUN_MIN, COH_MAX=COH_MAX, SPE_MAX=SPE_MAX, GAP=GAP)}


def runs(idx, gap=GAP):
    """sorted index array -> list of (start, end_inclusive, n_members)"""
    idx = np.sort(np.asarray(idx))
    if len(idx) == 0:
        return []
    br = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.r_[idx[0], idx[br + 1]]; ends = np.r_[idx[br], idx[-1]]
    cnt = np.r_[br + 1, len(idx)] - np.r_[0, br + 1]
    return [(int(s), int(e), int(c)) for s, e, c in zip(starts, ends, cnt)]


def rspread(Z):
    """robust spread: median euclidean distance to the coordinate-wise median"""
    return float(np.median(np.sqrt(((Z - np.median(Z, 0)) ** 2).sum(1)))) if len(Z) > 1 else 0.0


def analyse(ds, verbose=True):
    P = print if verbose else (lambda *a, **k: None)
    S = np.load(f"{ROOT}/_diagnostics/scores_{ds}.npz")
    B = np.load(f"{ROOT}/sota_bundle/ens_bundle/bundle_{ds}.npz")
    F = np.load(f"{ROOT}/_diagnostics/e2_fable_{ds}.npz")
    y = S["label"].astype(int); n = len(y)
    assert np.array_equal(y, B["y"]) and np.array_equal(y, F["yw"]), "row alignment"
    ztr, zte = F["ztr"].astype(np.float64), F["zte"].astype(np.float64)
    logpi, logN_tr, logN_te, mu_c = F["logpi"], F["logN_tr"], F["logN_te"], F["mu_c"].astype(np.float64)
    Xn, Xa = B["Xn_w"].astype(np.float64), B["Xa_w"].astype(np.float64)
    nch = int(B["nch"]) if "nch" in B.files else Xn.shape[1] // 6
    lat = S["LatAD"].mean(0)

    # ---------- scores / thresholds ----------
    mix = lambda L: -logsumexp(logpi[None] + L, axis=1)
    s_tr, s_te = mix(logN_tr), mix(logN_te)
    thr99 = float(np.quantile(s_tr, 0.99))
    thr_f1, thr_f5 = float(np.quantile(lat[y == 0], 0.99)), float(np.quantile(lat[y == 0], 0.95))
    R = dict(n_test=n, n_normal=int((y == 0).sum()), n_anom=int((y == 1).sum()), n_train=len(ztr), nch=nch,
             spearman_mix_vs_latad=float(spearmanr(s_te, lat).correlation))

    # ---------- reference quantities from TRAIN normal ----------
    # (a) per-regime latent spread: RMS distance of train points (argmax regime) from regime centroid
    reg_tr = np.argmax(logpi[None] + logN_tr, 1)
    spreads = []
    for c in range(len(logpi)):
        m = reg_tr == c
        if m.sum() >= 20:
            spreads.append(rspread(ztr[m]))
    spread_ref = float(np.median(spreads))
    # (b) latent support: 1-NN distance to train latents; train leave-one-out reference
    # clip blown-up latents (encoder extrapolation on OOD inputs) to 5x train p99 norm for clustering only
    Rcap = 5 * np.quantile(np.linalg.norm(ztr, axis=1), 0.99)
    zclip = zte * np.minimum(1.0, Rcap / np.maximum(np.linalg.norm(zte, axis=1), 1e-9))[:, None]
    R["n_latent_clipped"] = int((np.linalg.norm(zte, axis=1) > Rcap).sum())
    nn = NearestNeighbors(n_neighbors=2).fit(ztr)
    d_loo = nn.kneighbors(ztr)[0][:, 1]
    d1_te = nn.kneighbors(zte, n_neighbors=1)[0][:, 0]
    d1_ref = float(np.median(d_loo)); d1_p99 = float(np.quantile(d_loo, 0.99))
    dmu_te = np.sqrt(((zte[:, None] - mu_c[None]) ** 2).sum(2)).min(1)
    # (c) physical range on channel MEAN stat (first nch columns), tol 1% of train range
    Xn_m, Xa_m = Xn[:, :nch], Xa[:, :nch]
    lo, hi = Xn_m.min(0), Xn_m.max(0); rg = np.maximum(hi - lo, 1e-9)
    inrange_w = ((Xa_m >= lo - 0.01 * rg) & (Xa_m <= hi + 0.01 * rg)).mean(1)   # per window frac channels in range
    # (d) correlation structure: PCA on standardised train channel-means; SPE (off-subspace residual) and T2
    mu, sd = Xn_m.mean(0), Xn_m.std(0); keep = sd > 1e-6 * max(sd.max(), 1e-12)
    Zn, Za = (Xn_m[:, keep] - mu[keep]) / sd[keep], (Xa_m[:, keep] - mu[keep]) / sd[keep]
    pca = PCA(n_components=0.95, svd_solver="full").fit(Zn)
    k = int(pca.n_components_)
    spe = lambda Z: ((Z - pca.inverse_transform(pca.transform(Z))) ** 2).sum(1)
    t2 = lambda Z: ((pca.transform(Z) ** 2) / pca.explained_variance_[None]).sum(1)
    spe_tr, spe_te, t2_tr, t2_te = spe(Zn), spe(Za), t2(Zn), t2(Za)
    spe_ref, t2_ref = float(np.quantile(spe_tr, 0.99)), float(np.quantile(t2_tr, 0.99))
    Ctr = np.corrcoef(Zn, rowvar=False); iu = np.triu_indices(Zn.shape[1], 1)
    strong = np.abs(Ctr[iu]) > 0.5
    R["refs"] = dict(spread_ref=spread_ref, d1nn_train_loo_median=d1_ref, d1nn_train_loo_p99=d1_p99,
                     pca_k=k, spe_train_p99=spe_ref, t2_train_p99=t2_ref, n_strong_corr_pairs=int(strong.sum()))
    # sanity invariants on the references
    P(f"  refs: regime spread={spread_ref:.3f}  train 1NN LOO median={d1_ref:.3f} p99={d1_p99:.3f}  PCA k={k}/{Zn.shape[1]}  SPE p99={spe_ref:.2f}")
    P(f"  [{'OK' if np.median(spe_te[y==0])/spe_ref < 3 else 'WARN'}] median SPE of test normals / train p99 = {np.median(spe_te[y==0])/spe_ref:.2f}")
    P(f"  [{'OK' if np.median(spe_te[y==1]) > np.median(spe_te[y==0]) else 'FAIL'}] anomalies have larger median SPE than normals ({np.median(spe_te[y==1]):.1f} vs {np.median(spe_te[y==0]):.1f})")

    def cluster_features(idx, cid):
        idx = np.sort(idx); m = len(idx)
        rr = runs(idx)
        longest = max(r[2] for r in rr); span = max(r[1] - r[0] + 1 for r in rr)
        consec = float(np.isin(idx - 1, idx).mean()) if m > 1 else 0.0
        spread = rspread(zte[idx])
        f = dict(cluster=int(cid), size=m, n_y1=int(y[idx].sum()), frac_y1=float(y[idx].mean()),
                 time_start=int(idx[0]), time_end=int(idx[-1]),
                 n_blocks=len(rr), longest_run=int(longest), longest_span=int(span), frac_consecutive=consec,
                 spread=spread, cohesion_ratio=spread / spread_ref,
                 d1nn_train_median=float(np.median(d1_te[idx])), d1nn_ratio=float(np.median(d1_te[idx]) / d1_ref),
                 dmu_median=float(np.median(dmu_te[idx])),
                 inrange_frac=float(inrange_w[idx].mean()), inrange_full=float((inrange_w[idx] == 1).mean()),
                 spe_ratio=float(np.median(spe_te[idx]) / spe_ref), t2_ratio=float(np.median(t2_te[idx]) / t2_ref),
                 mix_score_median=float(np.median(s_te[idx])), latad_median=float(np.median(lat[idx])),
                 regime_argmax_top=int(np.bincount(np.argmax(logpi[None] + logN_te[idx], 1), minlength=len(logpi)).argmax()))
        if m >= 10:
            Zc_s = Za[idx]; ok = Zc_s.std(0) > 1e-6
            if ok.sum() >= 3:
                Cc = np.corrcoef(Zc_s, rowvar=False)
                sub = strong & ok[iu[0]] & ok[iu[1]]
                if sub.sum() >= 5:
                    f["corr_consistency"] = float(np.corrcoef(Ctr[iu][sub], np.nan_to_num(Cc[iu][sub]))[0, 1])
        f["corr_consistency"] = f.get("corr_consistency", float("nan"))
        # verdict (label-free)
        nr = (f["longest_run"] >= RUN_MIN) and (f["cohesion_ratio"] <= COH_MAX) and (f["spe_ratio"] <= SPE_MAX)
        f["verdict"] = "NEW-NORMAL" if nr else "ANOMALY"
        return f

    R["operating_points"] = {}
    for opname, score, thr in [("train99_mix", s_te, thr99), ("latad_fpr1", lat, thr_f1), ("latad_fpr5", lat, thr_f5)]:
        flag = score > thr; fi = np.flatnonzero(flag); nf = len(fi)
        P(f"\n  === {ds} @ {opname}: thr={thr:.3f} flagged={nf} (y1={int(y[fi].sum())}, y0={int((y[fi]==0).sum())}; FPR={((y==0)&flag).sum()/(y==0).sum():.4f}, TPR={((y==1)&flag).sum()/(y==1).sum():.3f})")
        op = dict(threshold=float(thr), n_flagged=nf, n_flag_y1=int(y[fi].sum()), n_flag_y0=int((y[fi] == 0).sum()),
                  fpr=float(((y == 0) & flag).sum() / (y == 0).sum()), tpr=float(((y == 1) & flag).sum() / (y == 1).sum()))
        if nf < 10:
            op["note"] = "too few flags to cluster"; R["operating_points"][opname] = op; continue
        mcs = int(max(5, min(15, nf // 20)))
        lab = HDBSCAN(min_cluster_size=mcs, min_samples=max(3, mcs // 3)).fit_predict(zclip[fi])
        op["hdbscan_min_cluster_size"] = mcs
        cl = []
        for cid in sorted(set(lab.tolist())):
            idx = fi[lab == cid]
            f = cluster_features(idx, cid)
            if cid == -1:
                f["verdict"] = "ANOMALY(noise)"
            cl.append(f)
        cl.sort(key=lambda f: -f["size"])
        hdr = f"  {'cid':>4} {'size':>5} {'y1%':>5} {'t0..t1':>13} {'blk':>4} {'run':>5} {'cons':>5} {'coh':>5} {'d1nn':>5} {'inrg':>5} {'SPE':>6} {'T2':>6} {'corr':>5} {'reg':>4}  verdict"
        P(hdr)
        for f in cl:
            P(f"  {f['cluster']:>4} {f['size']:>5} {100*f['frac_y1']:>5.0f} {f['time_start']:>6}..{f['time_end']:<6} {f['n_blocks']:>4} {f['longest_run']:>5} {f['frac_consecutive']:>5.2f} "
              f"{f['cohesion_ratio']:>5.2f} {f['d1nn_ratio']:>5.1f} {100*f['inrange_frac']:>5.0f} {f['spe_ratio']:>6.2f} {f['t2_ratio']:>6.1f} {f['corr_consistency']:>5.2f} {f['regime_argmax_top']:>4}  {f['verdict']}")
        # ---- confusion summary + FPR after folding NEW-NORMAL clusters back ----
        nn_idx = np.concatenate([fi[lab == f["cluster"]] for f in cl if f["verdict"] == "NEW-NORMAL"] or [np.array([], int)])
        an_idx = np.setdiff1d(fi, nn_idx)
        flag2 = flag.copy(); flag2[nn_idx] = False
        summ = dict(n_newnormal_clusters=int(sum(f["verdict"] == "NEW-NORMAL" for f in cl)),
                    n_in_newnormal=int(len(nn_idx)), frac_y0_in_newnormal=float((y[nn_idx] == 0).mean()) if len(nn_idx) else None,
                    n_in_anomaly=int(len(an_idx)), frac_y1_in_anomaly=float((y[an_idx] == 1).mean()) if len(an_idx) else None,
                    fpr_before=op["fpr"], fpr_after_fold=float(((y == 0) & flag2).sum() / (y == 0).sum()),
                    tpr_before=op["tpr"], tpr_after_fold=float(((y == 1) & flag2).sum() / (y == 1).sum()),
                    anomalies_lost_by_fold=int(y[nn_idx].sum()))
        P(f"  summary: NEW-NORMAL clusters={summ['n_newnormal_clusters']} windows={summ['n_in_newnormal']} (y0 share={summ['frac_y0_in_newnormal']}); "
          f"ANOMALY windows={summ['n_in_anomaly']} (y1 share={summ['frac_y1_in_anomaly']}); FPR {summ['fpr_before']:.4f} -> {summ['fpr_after_fold']:.4f}; TPR {summ['tpr_before']:.3f} -> {summ['tpr_after_fold']:.3f}")
        # ---- HAI reference block recovery ----
        if ds == "HAI":
            blk = np.arange(6436, 7082); inblk = np.isin(fi, blk)
            cov = {}
            for cid in sorted(set(lab.tolist())):
                m = (lab == cid)
                if (m & inblk).sum():
                    cov[int(cid)] = dict(block_members=int((m & inblk).sum()), cluster_size=int(m.sum()),
                                         purity=float((m & inblk).sum() / m.sum()))
            op["ref_block"] = dict(n_flagged_in_block=int(inblk.sum()), block_len=len(blk), clusters=cov)
            P(f"  ref block 6436..7081: flagged {inblk.sum()}/{len(blk)}; cluster coverage {cov}")
        op["clusters"] = cl; op["summary"] = summ
        MEMB[f"{ds}_{opname}_fi"] = fi; MEMB[f"{ds}_{opname}_lab"] = lab
        R["operating_points"][opname] = op
    return R


if __name__ == "__main__":
    dss = sys.argv[1].split(",") if len(sys.argv) > 1 else ["HAI", "WADI", "SWaT"]
    for ds in dss:
        print(f"\n{'='*100}\n{ds}\n{'='*100}")
        OUT[ds] = analyse(ds)
    np.savez(f"{ROOT}/_diagnostics/nr_members.npz", **MEMB)
    with open(f"{ROOT}/_diagnostics/nr_discovery.json", "w") as fh:
        json.dump(OUT, fh, indent=1, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    print("\nsaved nr_discovery.json")
