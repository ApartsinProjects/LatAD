"""Regime-discovery (rarefaction / species-accumulation) curves + with/without-exclusion performance decomposition.
Model-free; reuses bundle arrays, scores_{DS}.npz and the C2ST OOF block identification (cov_c2st_oof_{DS}.npz).
Writes cov_rarefaction.json and merges summary into cov_coverage.json.
"""
import json, os, sys, time, warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "_diagnostics", "cov_rarefaction.json")
COV = os.path.join(ROOT, "_diagnostics", "cov_coverage.json")
DSS = sys.argv[1:] or ["WADI", "HAI", "SWaT"]
FRACS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
SEEDS = [0, 1, 2]
MULTS = (2, 4, 8)
MIN_MEMBERS = 5


def leader_cluster(Z, eps, order):
    """Greedy leader clustering at fixed radius eps, visiting points in `order`. Returns leader id per point (in Z order)."""
    lab = np.full(len(Z), -1); leaders = []; L = np.zeros((0, Z.shape[1]))
    for i in order:
        if len(leaders):
            d = np.sqrt(((L - Z[i]) ** 2).sum(1)); j = int(np.argmin(d))
            if d[j] <= eps: lab[i] = j; continue
        leaders.append(i); L = np.vstack([L, Z[i]]); lab[i] = len(leaders) - 1
    return lab, np.array(leaders)


def count_leaders(Z, eps, order, min_members=MIN_MEMBERS):
    lab, _ = leader_cluster(Z, eps, order); cnt = np.bincount(lab)
    return int(len(cnt)), int((cnt >= min_members).sum())


def gmm_counts(P, seed=0):
    """BIC/AIC-selected diagonal GMM component count on the fixed PCA projection P; plus VB-GMM effective components."""
    n = len(P); grid = [k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256) if k <= max(2, n // 15)]
    bic, aic = {}, {}
    for k in grid:
        g = GaussianMixture(k, covariance_type="diag", n_init=1, random_state=seed, reg_covar=1e-3).fit(P)
        bic[k] = g.bic(P); aic[k] = g.aic(P)
    kb = min(bic, key=bic.get); ka = min(aic, key=aic.get)
    # stricter penalties: BIC penalty scaled 2x and 4x  (bic - aic = (log n - 2) * n_params, so penalty term is recoverable)
    plogn = {k: (bic[k] - aic[k]) * np.log(n) / (np.log(n) - 2) for k in grid}      # = n_params * log(n), the BIC penalty term
    bic2 = {k: bic[k] + 1 * plogn[k] for k in grid}; kb2 = min(bic2, key=bic2.get)   # penalty x2
    bic4 = {k: bic[k] + 3 * plogn[k] for k in grid}; kb4 = min(bic4, key=bic4.get)   # penalty x4
    kv = min(64, max(2, n // 25))
    vb = BayesianGaussianMixture(n_components=kv, covariance_type="diag", weight_concentration_prior_type="dirichlet_process",
                                 weight_concentration_prior=0.01, max_iter=300, random_state=seed, reg_covar=1e-3).fit(P)
    return dict(bic=int(kb), aic=int(ka), bic_2x=int(kb2), bic_4x=int(kb4), vb_dp=int((vb.weights_ > 0.01).sum()), grid_max=int(max(grid)),
                bic_capped=int(kb == max(grid)), bic_4x_capped=int(kb4 == max(grid)))


def fit_saturation(ns, S):
    """Michaelis-Menten S(n) = Smax * n / (k + n). Returns Smax, k, S(n_max)/Smax."""
    ns, S = np.asarray(ns, float), np.asarray(S, float)
    try:
        (Smax, k), _ = curve_fit(lambda n, a, b: a * n / (b + n), ns, S, p0=[S.max() * 1.5, ns.max() / 2], bounds=([0, 1e-6], [np.inf, np.inf]), maxfev=10000)
        return dict(Smax=float(Smax), half_n=float(k), frac_of_asymptote_at_full=float(S[-1] / Smax), fit_ok=True)
    except Exception as ex:
        return dict(Smax=float("nan"), half_n=float("nan"), frac_of_asymptote_at_full=float("nan"), fit_ok=False, err=str(ex))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cov = json.load(open(COV)) if os.path.exists(COV) else {}
    for ds in DSS:
        t0 = time.time()
        b = np.load(os.path.join(ROOT, "sota_bundle", "ens_bundle", f"bundle_{ds}.npz"))
        Xn, Xa, y = b["Xn_w"].astype(np.float64), b["Xa_w"].astype(np.float64), b["y"]
        mu, sd = Xn.mean(0), Xn.std(0); sd = np.maximum(sd, 0.01 * np.median(sd))
        Zn, Za = (Xn - mu) / sd, np.clip((Xa - mu) / sd, -50, 50)
        nrm = np.where(y == 0)[0]; Zt = Za[nrm]; ntr = len(Zn)
        stream = np.vstack([Zn, Zt])                       # full normal stream in time order (train period precedes test period)
        d_loo = NearestNeighbors(n_neighbors=2, algorithm="brute", n_jobs=-1).fit(Zn).kneighbors(Zn)[0][:, 1]
        eps0 = float(np.median(d_loo))
        pca = PCA(20, random_state=0).fit(stream)
        R = dict(n_train=int(ntr), n_test_normal=int(len(Zt)), eps_median_train_loo=eps0)
        print(f"== {ds}: train {ntr}, stream {len(stream)}, eps0 {eps0:.3f}")

        # ---------------- rarefaction curves ----------------
        curves = {}
        for src_name, Z in (("train_only", Zn), ("full_normal_stream", stream)):
            N = len(Z); P = pca.transform(Z); rows = []
            for f in FRACS:
                nf = max(int(round(f * N)), 40)
                subs = [("prefix", np.arange(nf))] + [("random", np.sort(np.random.default_rng(s_).choice(N, nf, replace=False))) for s_ in SEEDS]
                for kind, idx in subs:
                    row = dict(frac=f, n=int(nf), kind=kind)
                    order = np.arange(nf) if kind == "prefix" else np.random.default_rng(int(idx[0]) + nf).permutation(nf)
                    for mlt in MULTS:
                        tot, big = count_leaders(Z[idx], mlt * eps0, order)
                        row[f"leaders_x{mlt}"] = tot; row[f"leaders_x{mlt}_ge{MIN_MEMBERS}"] = big
                    row.update({f"gmm_{k}": v for k, v in gmm_counts(P[idx]).items()})
                    rows.append(row)
            keys = [k for k in rows[0] if k not in ("frac", "n", "kind")]
            agg = {}
            for k in keys:
                agg[k] = {str(f): dict(random_mean=float(np.mean([r[k] for r in rows if r["frac"] == f and r["kind"] == "random"])),
                                       random_sd=float(np.std([r[k] for r in rows if r["frac"] == f and r["kind"] == "random"])),
                                       prefix=int([r[k] for r in rows if r["frac"] == f and r["kind"] == "prefix"][0])) for f in FRACS}
            # saturation fits on the random-mean curve; late-slope = relative rise from f=0.5 to 1.0
            sat = {}
            for k in keys:
                ns = [int(round(f * N)) for f in FRACS]; S = [agg[k][str(f)]["random_mean"] for f in FRACS]
                s = fit_saturation(ns, S); s["rel_rise_0p5_to_1"] = float((S[-1] - S[4]) / max(S[4], 1e-9)); s["rel_rise_0p7_to_1"] = float((S[-1] - S[5]) / max(S[5], 1e-9))
                Sp = [agg[k][str(f)]["prefix"] for f in FRACS]; s["prefix_rel_rise_0p5_to_1"] = float((Sp[-1] - Sp[4]) / max(Sp[4], 1e-9)); s["prefix_rel_rise_0p7_to_1"] = float((Sp[-1] - Sp[5]) / max(Sp[5], 1e-9))
                sat[k] = s
            curves[src_name] = dict(rows=rows, agg=agg, saturation=sat)
            print(f"  [{src_name}]")
            for k in keys:
                print(f"    {k:22s} random:", [round(agg[k][str(f)]['random_mean'], 1) for f in FRACS], " prefix:", [agg[k][str(f)]['prefix'] for f in FRACS],
                      f" Smax={sat[k]['Smax']:.1f} at_full={sat[k]['frac_of_asymptote_at_full']:.2f} rise.5-1={sat[k]['rel_rise_0p5_to_1']:.2f} prefix rise.5-1={sat[k]['prefix_rel_rise_0p5_to_1']:.2f}")
        R["rarefaction"] = curves

        # ---------------- first-appearance of regimes along the full stream (prefix, time order) ----------------
        fa = {}
        for mlt in MULTS:
            lab, leaders = leader_cluster(stream, mlt * eps0, np.arange(len(stream)))
            cnt = np.bincount(lab); big = np.where(cnt >= MIN_MEMBERS)[0]
            born_in_test = leaders[big] >= ntr
            info = dict(n_regimes_ge5=int(len(big)), n_regimes_ge5_born_in_train=int((~born_in_test).sum()), n_regimes_ge5_born_in_test=int(born_in_test.sum()),
                        test_normal_mass_in_test_born_regimes=float(np.isin(lab[ntr:], big[born_in_test]).mean()),
                        first_test_born_regime_stream_index=int(leaders[big][born_in_test].min()) if born_in_test.any() else None)
            # regimes born in test: their birth position in test order and size
            births = sorted([(int(nrm[leaders[j] - ntr]), int(cnt[j])) for j in big[born_in_test]], key=lambda t: -t[1])
            info["test_born_regimes_pos_size_top10"] = births[:10]
            if ds == "HAI":
                blk_mask = (nrm >= 6436) & (nrm <= 7081)
                lab_blk = lab[ntr:][blk_mask]; regs_blk = np.unique(lab_blk)
                born_blk = [j for j in regs_blk if ntr <= leaders[j] and 6436 <= nrm[leaders[j] - ntr] <= 7081]
                info["hai_block"] = dict(n_regimes_in_block=int(len(regs_blk)), n_regimes_born_in_block=int(len(born_blk)),
                                         frac_block_windows_in_block_born_regimes=float(np.isin(lab_blk, born_blk).mean()),
                                         frac_block_windows_in_train_born_regimes=float(np.isin(lab_blk, [j for j in regs_blk if leaders[j] < ntr]).mean()),
                                         block_born_regime_members_outside_block=int(np.isin(lab, born_blk).sum() - np.isin(lab_blk, born_blk).sum()),
                                         first_birth_test_pos=int(min(nrm[leaders[j] - ntr] for j in born_blk)) if born_blk else None,
                                         extra_windows_needed_beyond_train=int(min(np.where(nrm == nrm[leaders[j] - ntr])[0][0] for j in born_blk) + 1) if born_blk else None,
                                         extra_as_frac_of_train=float((min(np.where(nrm == nrm[leaders[j] - ntr])[0][0] for j in born_blk) + 1) / ntr) if born_blk else None)
            fa[f"x{mlt}"] = info
            print(f"  first-appearance x{mlt}:", info)
        R["first_appearance"] = fa

        # ---------------- with/without-exclusion performance decomposition ----------------
        s = np.load(os.path.join(ROOT, "_diagnostics", f"scores_{ds}.npz")); lbl = s["label"]; assert (lbl == y).all()
        e = np.load(os.path.join(ROOT, "_diagnostics", f"e2_fable_{ds}.npz")); hard = e["hard"]
        oofz = np.load(os.path.join(ROOT, "_diagnostics", f"cov_c2st_oof_{ds}.npz")); oof, pos = oofz["oof"], oofz["pos"]
        # label-free block: contiguous runs (bridge 2) of C2ST test-only windows (p>0.9), length >= 30; exclude only y==0 windows inside them
        def runs(idx, bridge=2):
            idx = np.sort(idx); out = []
            if len(idx) == 0: return out
            st, pv = idx[0], idx[0]
            for i in idx[1:]:
                if i - pv > bridge + 1: out.append((int(st), int(pv))); st = i
                pv = i
            out.append((int(st), int(pv))); return out
        sel = pos[oof > 0.9]; blocks = [(a, bb) for a, bb in runs(sel) if bb - a + 1 >= 30]
        excl = np.zeros(len(y), bool)
        for a, bb in blocks: excl[a:bb + 1] = True
        excl &= (y == 0)
        n_anom_removed = int(((y == 1) & np.zeros(len(y), bool)).sum())  # by construction zero; verified below
        assert (y[excl] == 0).all()
        keep = ~excl
        def metrics(scr, mask, thr_fixed=None):
            yy, ss = y[mask], scr[mask]; nn_ = ss[yy == 0]; aa = ss[yy == 1]
            out = dict(n_normal=int((yy == 0).sum()), n_anom=int((yy == 1).sum()), auroc=float(roc_auc_score(yy, ss)))
            hm = hard[mask]; sub = (yy == 0) | ((yy == 1) & hm)
            out["auroc_difficult"] = float(roc_auc_score(yy[sub], ss[sub])) if (yy[sub] == 1).sum() > 0 else float("nan")
            for q, name in ((0.99, "fpr1"), (0.95, "fpr5")):
                t = np.quantile(nn_, q); out[f"tpr_at_{name}_own"] = float((aa > t).mean())
            if thr_fixed is not None:
                for name, t in thr_fixed.items():
                    out[f"fpr_at_{name}_fixedthr"] = float((nn_ > t).mean()); out[f"tpr_at_{name}_fixedthr"] = float((aa > t).mean())
            return out
        dec = dict(exclusion_blocks=[(a, bb, bb - a + 1) for a, bb in blocks], n_normal_excluded=int(excl.sum()), n_anom_excluded=n_anom_removed,
                   frac_test_normal_excluded=float(excl.sum() / (y == 0).sum()))
        det = {"LatAD": s["LatAD"].mean(0), "maxz": s["maxz"], "IF": s["IF"].mean(0), "AE": s["AE"].mean(0), "USAD": s["USAD"], "TranAD": s["TranAD"]}
        for name, scr in det.items():
            nn_full = scr[y == 0]
            thr = dict(fpr1=float(np.quantile(nn_full, 0.99)), fpr5=float(np.quantile(nn_full, 0.95)))
            if name == "maxz": thr["train99"] = float(s["maxz_thr"])
            full = metrics(scr, np.ones(len(y), bool), thr); wo = metrics(scr, keep, thr)
            dec[name] = dict(full=full, without_block=wo,
                             delta=dict(auroc=wo["auroc"] - full["auroc"], auroc_difficult=wo["auroc_difficult"] - full["auroc_difficult"],
                                        fpr_at_fpr1_fixedthr=wo["fpr_at_fpr1_fixedthr"] - full["fpr_at_fpr1_fixedthr"],
                                        fpr_at_fpr5_fixedthr=wo["fpr_at_fpr5_fixedthr"] - full["fpr_at_fpr5_fixedthr"],
                                        tpr_at_fpr1_own=wo["tpr_at_fpr1_own"] - full["tpr_at_fpr1_own"], tpr_at_fpr5_own=wo["tpr_at_fpr5_own"] - full["tpr_at_fpr5_own"]),
                             share_of_flagged_normals_in_block=dict(fpr1=float(excl[(y == 0) & (scr > thr["fpr1"])].mean()) if ((y == 0) & (scr > thr["fpr1"])).any() else float("nan"),
                                                                    fpr5=float(excl[(y == 0) & (scr > thr["fpr5"])].mean()) if ((y == 0) & (scr > thr["fpr5"])).any() else float("nan")))
            if name == "maxz":
                fl = (y == 0) & (scr > thr["train99"]); dec[name]["share_of_flagged_normals_in_block"]["train99"] = float(excl[fl].mean()) if fl.any() else float("nan")
        R["exclusion"] = dec
        print(f"  exclusion: blocks {dec['exclusion_blocks']} n_normal_excl {dec['n_normal_excluded']} ({dec['frac_test_normal_excluded']:.3%}) anom_excl {dec['n_anom_excluded']}")
        for name in det:
            d = dec[name]; print(f"    {name:7s} AUROC {d['full']['auroc']:.4f} -> {d['without_block']['auroc']:.4f} | diff {d['full']['auroc_difficult']:.4f} -> {d['without_block']['auroc_difficult']:.4f} | FPR@fixed1% {d['full']['fpr_at_fpr1_fixedthr']:.4f} -> {d['without_block']['fpr_at_fpr1_fixedthr']:.4f} | TPR@1%own {d['full']['tpr_at_fpr1_own']:.3f} -> {d['without_block']['tpr_at_fpr1_own']:.3f} | TPR@5%own {d['full']['tpr_at_fpr5_own']:.3f} -> {d['without_block']['tpr_at_fpr5_own']:.3f} | flagged-in-block share 1%/5%: {d['share_of_flagged_normals_in_block']}")

        res[ds] = R; json.dump(res, open(OUT, "w"), indent=1)
        cov = json.load(open(COV)) if os.path.exists(COV) else {}
        cov.setdefault(ds, {})["rarefaction_summary"] = {src: {k: dict(agg=curves[src]["agg"][k], sat=curves[src]["saturation"][k]) for k in curves[src]["agg"]} for src in curves}
        cov[ds]["first_appearance"] = fa; cov[ds]["exclusion"] = dec
        json.dump(cov, open(COV, "w"), indent=1)
        print(f"  [{time.time()-t0:.0f}s]")
    print("saved", OUT)


if __name__ == "__main__":
    main()
