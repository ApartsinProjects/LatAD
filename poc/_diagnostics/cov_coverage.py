"""Model-free training-set coverage audit (k-NN support) for WADI / HAI / SWaT.
Reuses bundle_{DS}.npz feature arrays; no model, no retraining. Writes cov_coverage.json incrementally.
"""
import json, os, sys, time
import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "_diagnostics", "cov_coverage.json")
DSS = sys.argv[1:] or ["WADI", "HAI", "SWaT"]
FRACS = [0.1, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2]
K = 5


def knn_to(ref, q, k=K):
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", n_jobs=-1).fit(ref)
    d, _ = nn.kneighbors(q)
    return d  # (n, k)


def loo(ref, k=K):
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=-1).fit(ref)
    d, _ = nn.kneighbors(ref)
    return d[:, 1:]  # drop self


def runs(idx, bridge=2):
    """contiguous runs over sorted integer positions, merging gaps <= bridge."""
    idx = np.sort(np.asarray(idx))
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > bridge + 1:
            out.append((int(s), int(p)))
            s = i
        p = i
    out.append((int(s), int(p)))
    return out


def block_stats(oos_pos, n_oos):
    rs = runs(oos_pos)
    lens = np.array([b - a + 1 for a, b in rs]) if rs else np.array([0])
    big = [(a, b) for a, b in rs if b - a + 1 >= 30]
    n_in_big = sum(int(np.sum((oos_pos >= a) & (oos_pos <= b))) for a, b in big)
    big10 = [(a, b) for a, b in rs if b - a + 1 >= 10]
    n_in_big10 = sum(int(np.sum((oos_pos >= a) & (oos_pos <= b))) for a, b in big10)
    n_single = int(np.sum(lens == 1))
    return dict(n_runs=len(rs), longest_run=int(lens.max()), n_runs_ge30=len(big), n_runs_ge10=len(big10),
                frac_oos_in_runs_ge30=float(n_in_big / max(n_oos, 1)),
                frac_oos_in_runs_ge10=float(n_in_big10 / max(n_oos, 1)),
                frac_oos_singletons=float(n_single / max(n_oos, 1)),
                runs_ge30=[(a, b, b - a + 1) for a, b in big][:20])


res = json.load(open(OUT)) if os.path.exists(OUT) else {}

for ds in DSS:
    t0 = time.time()
    b = np.load(os.path.join(ROOT, "sota_bundle", "ens_bundle", f"bundle_{ds}.npz"))
    s = np.load(os.path.join(ROOT, "_diagnostics", f"scores_{ds}.npz"))
    Xn, Xa, y = b["Xn_w"].astype(np.float64), b["Xa_w"].astype(np.float64), b["y"]
    assert (s["label"] == y).all()
    mu, sd = Xn.mean(0), Xn.std(0)
    sd_floor = 0.01 * np.median(sd)
    n_floored = int((sd < sd_floor).sum())
    sd = np.maximum(sd, sd_floor)
    Zn, Za = (Xn - mu) / sd, (Xa - mu) / sd
    nrm, anm = np.where(y == 0)[0], np.where(y == 1)[0]
    R = dict(n_train=int(len(Zn)), n_test=int(len(Za)), n_test_normal=int(len(nrm)), n_test_anom=int(len(anm)),
             dim=int(Zn.shape[1]), n_channels_sd_floored=n_floored,
             za_absmax=float(np.abs(Za).max()))
    print(f"== {ds}: train {len(Zn)}, test {len(Za)} (normal {len(nrm)}), dim {Zn.shape[1]}, floored {n_floored}, |Za|max {R['za_absmax']:.1f}")

    # ---------- 1. OOS rate ----------
    d_loo = loo(Zn)                       # (n_train, K)
    d_te = knn_to(Zn, Za)                 # (n_test, K)
    thr = {}
    for kk, col in [("k1", 0), ("k5", K - 1)]:
        tl = d_loo[:, col]
        thr[kk] = dict(p99=float(np.quantile(tl, 0.99)), max=float(tl.max()),
                       mean=float(tl.mean()), median=float(np.median(tl)))
    oos = {}
    for kk, col in [("k1", 0), ("k5", K - 1)]:
        dt = d_te[:, col]
        for crit, t in [("p99", thr[kk]["p99"]), ("strict", thr[kk]["max"])]:
            oos[f"{kk}_{crit}"] = dict(test_normal=float((dt[nrm] > t).mean()),
                                       test_anom=float((dt[anm] > t).mean()),
                                       n_test_normal_oos=int((dt[nrm] > t).sum()))
    # held-out train slices (random 10 %, and temporal last 10 %) vs remaining 90 % with its own LOO threshold
    ho = {}
    n = len(Zn); m = int(0.1 * n)
    for name, hidx in [("random10", np.random.default_rng(0).choice(n, m, replace=False)),
                       ("last10_temporal", np.arange(n - m, n)),
                       ("first10_temporal", np.arange(0, m))]:
        mask = np.ones(n, bool); mask[hidx] = False
        ref = Zn[mask]; dl = loo(ref)[:, 0]; t99, tmax = np.quantile(dl, 0.99), dl.max()
        dq = knn_to(ref, Zn[hidx])[:, 0]
        ho[name] = dict(oos_p99=float((dq > t99).mean()), oos_strict=float((dq > tmax).mean()))
    R["thr"] = thr; R["oos"] = oos; R["train_heldout"] = ho
    print(f"  OOS test-normal: k1_p99 {oos['k1_p99']['test_normal']:.3%} strict {oos['k1_strict']['test_normal']:.3%} | k5_p99 {oos['k5_p99']['test_normal']:.3%} | anom k1_p99 {oos['k1_p99']['test_anom']:.3%} | heldout {ho}")

    # ---------- 3. block structure (k1 p99 and strict) ----------
    dt1 = d_te[:, 0]
    blk = {}
    for crit, t in [("p99", thr["k1"]["p99"]), ("strict", thr["k1"]["max"])]:
        oos_pos = nrm[dt1[nrm] > t]        # positions in test-window time order
        blk[crit] = block_stats(oos_pos, len(oos_pos))
        blk[crit]["n_oos"] = int(len(oos_pos))
        if ds == "HAI":
            inb = (oos_pos >= 6436) & (oos_pos <= 7081)
            nb_norm = int(((nrm >= 6436) & (nrm <= 7081)).sum())
            blk[crit]["hai_ref_block_6436_7081"] = dict(n_normal_windows=nb_norm, n_oos=int(inb.sum()),
                                                        frac_block_oos=float(inb.sum() / max(nb_norm, 1)),
                                                        share_of_all_oos=float(inb.sum() / max(len(oos_pos), 1)),
                                                        median_d1_in_block=float(np.median(dt1[6436:7082])),
                                                        median_d1_normal_outside=float(np.median(dt1[nrm[(nrm < 6436) | (nrm > 7081)]])))
    R["blocks"] = blk
    print(f"  blocks p99: {blk['p99']}")

    # ---------- 2. coverage vs training size ----------
    curve = []
    full_oos_set = set((nrm[dt1[nrm] > thr["k1"]["p99"]]).tolist())
    for f in FRACS:
        nf = max(int(round(f * n)), 50)
        subs = [("prefix", np.arange(nf))] + ([("random", np.random.default_rng(sd_).choice(n, nf, replace=False)) for sd_ in SEEDS] if f < 1.0 else [("random", np.arange(n))])
        for kind, idx in subs:
            ref = Zn[idx]
            dl = loo(ref)[:, 0]; t99 = float(np.quantile(dl, 0.99)); tmax = float(dl.max())
            dq = knn_to(ref, Za[nrm])[:, 0]
            o99 = dq > t99
            pos = nrm[o99]
            bs = block_stats(pos, int(o99.sum()))
            jacc = len(full_oos_set & set(pos.tolist())) / max(len(full_oos_set | set(pos.tolist())), 1)
            curve.append(dict(frac=f, n_sub=int(nf), kind=kind, thr_p99=t99, thr_max=tmax,
                              oos_p99=float(o99.mean()), oos_strict=float((dq > tmax).mean()),
                              oos_fixed_full_p99=float((dq > thr["k1"]["p99"]).mean()),   # absolute criterion: full-train threshold
                              median_test_normal_d1=float(np.median(dq)), p90_test_normal_d1=float(np.quantile(dq, 0.9)),
                              longest_run=bs["longest_run"], frac_in_runs_ge30=bs["frac_oos_in_runs_ge30"],
                              jaccard_vs_full=float(jacc), mean_loo_d1=float(dl.mean())))
    R["curve"] = curve
    # aggregate: random mean/sd per frac, prefix per frac
    agg = {}
    for f in FRACS:
        r = [c["oos_p99"] for c in curve if c["frac"] == f and c["kind"] == "random"]
        p = [c["oos_p99"] for c in curve if c["frac"] == f and c["kind"] == "prefix"][0]
        rf = [c["oos_fixed_full_p99"] for c in curve if c["frac"] == f and c["kind"] == "random"]
        pf = [c["oos_fixed_full_p99"] for c in curve if c["frac"] == f and c["kind"] == "prefix"][0]
        rm = [c["median_test_normal_d1"] for c in curve if c["frac"] == f and c["kind"] == "random"]
        rt = [c["thr_p99"] for c in curve if c["frac"] == f and c["kind"] == "random"]
        agg[str(f)] = dict(random_mean=float(np.mean(r)), random_sd=float(np.std(r)), prefix=float(p),
                           fixed_thr_random_mean=float(np.mean(rf)), fixed_thr_prefix=float(pf),
                           median_d1_random=float(np.mean(rm)), thr_p99_random=float(np.mean(rt)))
    R["curve_agg"] = agg
    # extrapolation: fit oos vs log2(n) over random means for f>=0.25
    xs = np.log2([int(round(f * n)) for f in FRACS if f >= 0.25]); ys = [agg[str(f)]["random_mean"] for f in FRACS if f >= 0.25]
    slope, icpt = np.polyfit(xs, ys, 1)
    cur = agg["1.0"]["random_mean"]
    doublings_to_1pct = (cur - 0.01) / (-slope) if slope < 0 else float("inf")
    R["extrap"] = dict(slope_per_doubling=float(slope), oos_at_full=float(cur),
                       doublings_to_reach_1pct=float(doublings_to_1pct),
                       drop_0p1_to_1=float(agg["0.1"]["random_mean"] - cur),
                       drop_0p5_to_1=float(agg["0.5"]["random_mean"] - cur),
                       jaccard_0p25_vs_full=float(np.mean([c["jaccard_vs_full"] for c in curve if c["frac"] == 0.25 and c["kind"] == "random"])))
    print("  curve:", {k: (round(v['random_mean'], 4), round(v['prefix'], 4)) for k, v in agg.items()}, "extrap", R["extrap"])

    # ---------- 4. false-alarm attribution ----------
    is_oos = dt1[nrm] > thr["k1"]["p99"]
    is_oos_strict = dt1[nrm] > thr["k1"]["max"]
    base = float(is_oos.mean())
    att = {}
    def attrib(flag_nrm, name):
        nfl = int(flag_nrm.sum())
        sh = float(is_oos[flag_nrm].mean()) if nfl else float("nan")
        shs = float(is_oos_strict[flag_nrm].mean()) if nfl else float("nan")
        att[name] = dict(fpr=float(flag_nrm.mean()), n_flagged=nfl, share_oos_p99=sh, share_oos_strict=shs,
                         enrichment_vs_base=float(sh / base) if base > 0 else float("nan"),
                         oos_rate_among_unflagged=float(is_oos[~flag_nrm].mean()),
                         recall_of_oos=float(flag_nrm[is_oos].mean()) if is_oos.any() else float("nan"))
    maxz = b["maxz"]; attrib(maxz[nrm] > float(b["maxz_thr"]), "maxz_train99")
    lat = s["LatAD"].mean(0)
    for q, name in [(0.99, "latad_fpr1"), (0.95, "latad_fpr5")]:
        t = np.quantile(lat[nrm], q); attrib(lat[nrm] > t, name)
    # also: test-normal window score vs support distance correlation
    from scipy.stats import spearmanr
    att["spearman_latad_vs_d1_testnormal"] = float(spearmanr(lat[nrm], dt1[nrm]).correlation)
    att["spearman_maxz_vs_d1_testnormal"] = float(spearmanr(maxz[nrm], dt1[nrm]).correlation)
    att["base_oos_rate"] = base
    R["fpr_attrib"] = att
    print("  attrib:", {k: v for k, v in att.items()})

    # ---------- 5. intrinsic dim / density ----------
    r1, r2 = d_loo[:, 0], d_loo[:, 1]
    ok = r1 > 0
    mu_ratio = r2[ok] / r1[ok]
    # TwoNN (Facco et al. 2017): fit -log(1-F(mu)) = d*log(mu), discard top 10 %
    mus = np.sort(mu_ratio); F = np.arange(1, len(mus) + 1) / len(mus); keep = F < 0.9
    twonn = float(np.polyfit(np.log(mus[keep]), -np.log(1 - F[keep]), 1)[0])
    R["density"] = dict(twonn_id_train=twonn, mean_loo_d1=float(r1.mean()), median_loo_d1=float(np.median(r1)),
                        p99_loo_d1=thr["k1"]["p99"], mean_loo_d1_per_sqrt_dim=float(r1.mean() / np.sqrt(Zn.shape[1])),
                        frac_exact_dup_train=float((r1 == 0).mean()),
                        median_test_normal_d1=float(np.median(dt1[nrm])), median_test_anom_d1=float(np.median(dt1[anm])))
    print("  density:", R["density"], f"  [{time.time()-t0:.0f}s]")

    res[ds] = R
    json.dump(res, open(OUT, "w"), indent=1)
print("saved", OUT)
