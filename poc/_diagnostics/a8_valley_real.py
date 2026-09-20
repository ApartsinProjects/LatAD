"""A8 valley-smoothing test on real data (SWaT_canon, WADI_clean, HAI).

Hypothesis. Deep reconstructors (USAD, TranAD) smooth over the density valley between two close normal
regimes; an anomaly that is an extreme value of regime A landing in that valley reconstructs like normal
and is missed, while a latent-density head scores the valley as improbable and catches it.

Operational partition (label-free geometry, independent of every detector). winfeat 'stats' window
features on the train-standardised streams (W=60, ST=30, the detectors' window grid), feature-standardised
on train, PCA-10 fit on train, full-cov GMM on train with K by BIC (same recipe as a8_masking_allsets.py).
For a test window x: d_k = Mahalanobis distance to component k, h = nearest, j = second nearest,
R = q99.5 of the train nearest-component Mahalanobis distance (the normal envelope).
  in_cluster : d_h <= R                                   (inside the envelope of some regime)
  otherwise, in the (S_h+S_j)/2-whitened frame with axis v = mu_j - mu_h, u = x - mu_h,
  t = <u,v>/|v|^2 (position along the axis, 0 at h, 1 at j), perp = |u - t v|:
  valley     : 0 < t < 1 and perp <= R                    (between the two nearest centres, inside the tube)
  beyond     : (t <= 0 or t >= 1) and perp <= R           (on the axis but past a centre: an extreme away from j)
  off        : perp > R                                   (leaves the inter-regime tube: off-manifold)
  valley_distinct additionally requires the (h, j) pair to be density-valley-separated (mixture density
  along mu_h -> mu_j dips below 0.5 of the lower endpoint), so that h and j are two regimes, not one ridge.

Per partition and detector (LatAD seed-mean, USAD, TranAD, AE seed-mean, IF, linres, maxz), all from
scores_<ds>.npz: n, AUROC of the partition's anomalies against ALL test-normal windows, recall at the
q99 / q95 of test-normal scores (matched 1% / 5% FPR), median test-normal percentile. Interaction
statistic = (AUROC_LatAD - AUROC_deep)[valley] - (AUROC_LatAD - AUROC_deep)[off], bootstrap CI.
Miss enrichment: among anomalies, G = {deep <= q99_norm and LatAD > q99_norm}; valley share in G vs
outside G (Fisher exact). Magnitude-matched view: valley vs off within terciles of d_h.
Normals in the valley: FPR of each detector at its global q99 threshold restricted to valley normals.

Invariants stated in advance: (I1) in-cluster anomalies are near chance for every detector;
(I2) off-manifold anomalies with large d_h are caught by every detector (recall@1% high);
(I3) most test-normal windows are in_cluster; (I4) window count equals the scores npz length and the
window labels agree with the npz label vector; (I5) shuffling partition labels among anomalies kills
the interaction (null distribution centred at 0).

Rows appended to a8_valley_real.jsonl (resumable per (dataset, seed)); per-window table to
a8_valley_windows_<ds>_seed<s>.csv. Usage (from poc/): python _diagnostics/a8_valley_real.py [ds ...]
"""
from __future__ import annotations
import os, sys, json, time, warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.dirname(HERE)
sys.path.insert(0, POC); sys.path.insert(0, HERE); os.chdir(POC)
from winfeat import window_features
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact, rankdata

OUT = os.path.join(HERE, "a8_valley_real.jsonl")
CACHE = r"E:\tmp\claude\E--Projects-Backlog-LatAD\6ae0c3f3-0eaf-4beb-8a09-88ce1e618d2a\scratchpad"
KS, NPC, SEEDS, W, ST = (2, 3, 4, 6, 8, 12, 16), 10, (0, 1, 2), 60, 30
DETS = ["LatAD", "USAD", "TranAD", "AE", "IF", "linres", "maxz"]
PARTS = ["in_cluster", "valley", "beyond", "off"]
NBOOT = 1000


def _js(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def emit(row):
    row["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=_js) + "\n"); f.flush()
    print("EMIT", json.dumps(row, default=_js)[:400], flush=True)


def done():
    s = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try: r = json.loads(line); s.add((r["kind"], r["dataset"], r["seed"]))
            except Exception: pass
    return s


def windows(X, y=None):
    F, Y = [], []
    for i in range(0, len(X) - W + 1, ST):
        F.append(window_features(X[i:i + W], "stats"))
        if y is not None: Y.append(float(y[i:i + W].mean()))
    return np.nan_to_num(np.asarray(F, np.float32)), (np.asarray(Y) if y is not None else None)


def feats(ds):
    cf = os.path.join(CACHE, f"a8_valley_feat_{ds}.npz")
    if os.path.exists(cf):
        d = np.load(cf); return d["Ftr"], d["Fte"], d["yfrac"]
    d = np.load(os.path.join(CACHE, f"raw_{ds}.npz"), allow_pickle=True)
    Ftr, _ = windows(d["Xn"]); Fte, yfrac = windows(d["Xa"], d["ya"].astype(int))
    np.savez(cf, Ftr=Ftr, Fte=Fte, yfrac=yfrac); return Ftr, Fte, yfrac


def seedmean(v):
    return v.mean(0) if v.ndim == 2 else v


def load_scores(ds):
    z = np.load(os.path.join(HERE, f"scores_{ds}.npz"), allow_pickle=True)
    S = {k: np.asarray(seedmean(z[k]), np.float64) for k in DETS}
    return z["label"].astype(int), S


def auroc(y, s):
    if y.sum() == 0 or y.sum() == len(y): return float("nan")
    return float(roc_auc_score(y, s))


def pct_of_normal(s, s_norm):
    """fraction of test-normal scores strictly below s (rank-based, robust to 1e29 values)."""
    srt = np.sort(s_norm); return np.searchsorted(srt, s, side="left") / len(srt)


class Geo:
    def __init__(self, Ztr, seed):
        fits = [GaussianMixture(k, covariance_type="full", reg_covar=1e-3, random_state=seed, max_iter=300).fit(Ztr) for k in KS]
        bic = [g.bic(Ztr) for g in fits]; self.g = fits[int(np.argmin(bic))]; self.K = self.g.n_components
        self.mu, self.S = self.g.means_, self.g.covariances_
        self.Si = np.stack([np.linalg.inv(S) for S in self.S])
        # pooled whiteners per pair
        K = self.K; self.Wp = {}
        for i, j in itertools.combinations(range(K), 2):
            Sp = 0.5 * (self.S[i] + self.S[j]); w, V = np.linalg.eigh(Sp)
            self.Wp[(i, j)] = self.Wp[(j, i)] = V @ np.diag(1 / np.sqrt(w)) @ V.T
        # pairwise distance and valley ratio along the segment
        self.D = np.zeros((K, K)); self.valley = np.ones((K, K))
        ts = np.linspace(0, 1, 41)
        for i, j in itertools.combinations(range(K), 2):
            Wm = self.Wp[(i, j)]; self.D[i, j] = self.D[j, i] = np.linalg.norm(Wm @ (self.mu[j] - self.mu[i]))
            seg = self.mu[i][None] + ts[:, None] * (self.mu[j] - self.mu[i])[None]
            lp = self.g.score_samples(seg); v = float(np.exp(lp.min() - min(lp[0], lp[-1])))
            self.valley[i, j] = self.valley[j, i] = v

    def dist(self, Z):
        Dm = Z[:, None, :] - self.mu[None]
        return np.sqrt(np.einsum("nkd,kde,nke->nk", Dm, self.Si, Dm))

    def partition(self, Z, R):
        d = self.dist(Z); order = np.argsort(d, 1); h, j = order[:, 0], order[:, 1]
        dh, dj = d[np.arange(len(Z)), h], d[np.arange(len(Z)), j]
        t = np.zeros(len(Z)); perp = np.zeros(len(Z)); Dhj = np.zeros(len(Z)); vr = np.zeros(len(Z))
        for n in range(len(Z)):
            Wm = self.Wp[(h[n], j[n])]; u = Wm @ (Z[n] - self.mu[h[n]]); v = Wm @ (self.mu[j[n]] - self.mu[h[n]])
            vv = v @ v; t[n] = (u @ v) / vv; perp[n] = np.linalg.norm(u - t[n] * v); Dhj[n] = np.sqrt(vv); vr[n] = self.valley[h[n], j[n]]
        part = np.full(len(Z), "off", object)
        part[(perp <= R) & (t > 0) & (t < 1)] = "valley"
        part[(perp <= R) & ((t <= 0) | (t >= 1))] = "beyond"
        part[dh <= R] = "in_cluster"
        return dict(part=part, h=h, j=j, dh=dh, dj=dj, t=t, perp=perp, Dhj=Dhj, vr=vr, mixlogp=self.g.score_samples(Z))


def run(ds):
    Ftr, Fte, yfrac = feats(ds)
    lab, S = load_scores(ds)
    assert len(Fte) == len(lab), (len(Fte), len(lab))
    ywin = (yfrac > 0).astype(int)
    agree = float((ywin == lab).mean())
    print(f"[{ds}] windows train {len(Ftr)} test {len(Fte)}; label agreement (any-row rule vs npz) {agree:.4f}; anomalies npz {lab.sum()} mine {ywin.sum()}", flush=True)
    y = lab  # the npz label is the authoritative alignment with the detector scores
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    Ztr_full, Zte_full = (Ftr - mu) / sd, (Fte - mu) / sd
    pca = PCA(NPC, random_state=0).fit(Ztr_full); Ztr, Zte = pca.transform(Ztr_full), pca.transform(Zte_full)
    norm, anom = y == 0, y == 1
    for seed in SEEDS:
        if ("main", ds, seed) in done(): print(f"skip {ds} seed {seed}"); continue
        t0 = time.time(); G = Geo(Ztr, seed)
        dtr = G.dist(Ztr).min(1); R = float(np.quantile(dtr, 0.995))
        P = G.partition(Zte, R)
        part = P["part"]
        # invariants I3
        share_norm = {p: float((part[norm] == p).mean()) for p in PARTS}
        share_anom = {p: float((part[anom] == p).mean()) for p in PARTS}
        vdist = (part == "valley") & (P["vr"] < 0.5)
        share_anom["valley_distinct"] = float(vdist[anom].mean()); share_norm["valley_distinct"] = float(vdist[norm].mean())
        print(f"[{ds} s{seed}] K={G.K} R={R:.2f} normals {share_norm} anomalies {share_anom} ({time.time()-t0:.0f}s)", flush=True)
        # per-window table
        with open(os.path.join(HERE, f"a8_valley_windows_{ds}_seed{seed}.csv"), "w") as f:
            f.write("idx,label,part,valley_distinct,h,j,dh,dj,t,perp,Dhj,vr,mixlogp," + ",".join(f"{d}_pct" for d in DETS) + "\n")
            pcts = {d: pct_of_normal(S[d], S[d][norm]) for d in DETS}
            for n in range(len(Zte)):
                f.write(f"{n},{y[n]},{part[n]},{int(vdist[n])},{P['h'][n]},{P['j'][n]},{P['dh'][n]:.3f},{P['dj'][n]:.3f},{P['t'][n]:.3f},{P['perp'][n]:.3f},{P['Dhj'][n]:.3f},{P['vr'][n]:.3f},{P['mixlogp'][n]:.2f}," + ",".join(f"{pcts[d][n]:.4f}" for d in DETS) + "\n")
        # per-partition detection
        thr99 = {d: np.quantile(S[d][norm], 0.99) for d in DETS}; thr95 = {d: np.quantile(S[d][norm], 0.95) for d in DETS}
        det = {}
        parts_ext = PARTS + ["valley_distinct", "all"]
        for p in parts_ext:
            m = anom & ((vdist) if p == "valley_distinct" else (np.ones_like(anom) if p == "all" else (part == p)))
            det[p] = dict(n=int(m.sum()), dh_median=float(np.median(P["dh"][m])) if m.sum() else float("nan"))
            for d in DETS:
                if m.sum() == 0: det[p][d] = dict(auroc=float("nan"), rec99=float("nan"), rec95=float("nan"), pct_med=float("nan")); continue
                yy = np.r_[np.zeros(norm.sum()), np.ones(m.sum())]; ss = np.r_[S[d][norm], S[d][m]]
                det[p][d] = dict(auroc=auroc(yy, ss), rec99=float((S[d][m] > thr99[d]).mean()), rec95=float((S[d][m] > thr95[d]).mean()),
                                 pct_med=float(np.median(pcts[d][m])))
        # interaction with bootstrap and shuffle null
        inter = {}
        rng = np.random.default_rng(seed)
        idx_a = np.where(anom)[0]; s_norm = {d: S[d][norm] for d in DETS}
        def part_auroc(d, idx):
            if len(idx) == 0: return float("nan")
            yy = np.r_[np.zeros(len(s_norm[d])), np.ones(len(idx))]; return auroc(yy, np.r_[s_norm[d], S[d][idx]])
        for deep in ["USAD", "TranAD", "AE"]:
            iv = np.where(anom & (part == "valley"))[0]; io = np.where(anom & (part == "off"))[0]
            if len(iv) < 3 or len(io) < 3: inter[deep] = dict(n_valley=len(iv), n_off=len(io)); continue
            obs = (part_auroc("LatAD", iv) - part_auroc(deep, iv)) - (part_auroc("LatAD", io) - part_auroc(deep, io))
            boots = []
            for _ in range(NBOOT):
                bv = rng.choice(iv, len(iv)); bo = rng.choice(io, len(io))
                boots.append((part_auroc("LatAD", bv) - part_auroc(deep, bv)) - (part_auroc("LatAD", bo) - part_auroc(deep, bo)))
            nulls = []
            pool = np.r_[iv, io]
            for _ in range(NBOOT // 2):
                pp = rng.permutation(pool); nv, no = pp[:len(iv)], pp[len(iv):]
                nulls.append((part_auroc("LatAD", nv) - part_auroc(deep, nv)) - (part_auroc("LatAD", no) - part_auroc(deep, no)))
            nulls = np.array(nulls)
            inter[deep] = dict(n_valley=len(iv), n_off=len(io), obs=float(obs), ci=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                               null_mean=float(nulls.mean()), null_sd=float(nulls.std()), p_perm=float((np.abs(nulls) >= abs(obs)).mean()))
        # miss enrichment
        enrich = {}
        for deep in ["USAD", "TranAD", "AE"]:
            miss_deep = S[deep] <= thr99[deep]; catch_lat = S["LatAD"] > thr99["LatAD"]
            Gm = anom & miss_deep & catch_lat; Gc = anom & ~(miss_deep & catch_lat)
            Rm = anom & ~miss_deep & ~catch_lat  # reverse: deep catches, LatAD misses
            res = dict(n_G=int(Gm.sum()), n_notG=int(Gc.sum()), n_reverse=int(Rm.sum()))
            for p in ["valley", "valley_distinct", "off", "beyond", "in_cluster"]:
                pm = vdist if p == "valley_distinct" else (part == p)
                a, b = int((Gm & pm).sum()), int((Gm & ~pm).sum()); c, dd = int((Gc & pm).sum()), int((Gc & ~pm).sum())
                sh_G = a / max(1, a + b); sh_c = c / max(1, c + dd); sh_all = float(pm[anom].mean())
                res[p] = dict(share_in_G=sh_G, share_outside_G=sh_c, share_all_anom=sh_all, enrichment=(sh_G / sh_all if sh_all > 0 else float("nan")),
                              fisher_p=float(fisher_exact([[a, b], [c, dd]])[1]), share_in_reverse=float((Rm & pm).sum() / max(1, Rm.sum())))
            enrich[deep] = res
        # magnitude-matched: terciles of dh among anomalies outside the envelope
        mm = {}
        outside = anom & (part != "in_cluster")
        if outside.sum() >= 9:
            q = np.quantile(P["dh"][outside], [1 / 3, 2 / 3])
            for bi, (lo, hi) in enumerate([(-np.inf, q[0]), (q[0], q[1]), (q[1], np.inf)]):
                b = outside & (P["dh"] > lo) & (P["dh"] <= hi); row = dict(dh_range=[float(lo), float(hi)])
                for p in ["valley", "off", "beyond"]:
                    idx = np.where(b & (part == p))[0]; row[p] = dict(n=len(idx), **{d: part_auroc(d, idx) for d in ["LatAD", "USAD", "TranAD"]})
                mm[f"tercile{bi}"] = row
        # normals in the valley: FPR at global q99 thresholds
        vn = norm & (part == "valley"); on = norm & (part == "off"); bn = norm & (part == "beyond")
        fpr = {d: dict(valley=float((S[d][vn] > thr99[d]).mean()) if vn.sum() else float("nan"),
                       off=float((S[d][on] > thr99[d]).mean()) if on.sum() else float("nan"),
                       beyond=float((S[d][bn] > thr99[d]).mean()) if bn.sum() else float("nan"),
                       in_cluster=float((S[d][norm & (part == "in_cluster")] > thr99[d]).mean())) for d in DETS}
        emit(dict(kind="main", dataset=ds, seed=seed, K=G.K, R=R, label_agreement=agree, n_norm=int(norm.sum()), n_anom=int(anom.sum()),
                  share_norm=share_norm, share_anom=share_anom, n_norm_valley=int(vn.sum()), detection=det, interaction=inter,
                  enrichment=enrich, magnitude_matched=mm, fpr_normals_by_part=fpr,
                  pair_geometry=dict(minD=float(G.D[np.triu_indices(G.K, 1)].min()), n_valley_pairs=int((G.valley[np.triu_indices(G.K, 1)] < 0.5).sum()),
                                     n_pairs=int(G.K * (G.K - 1) / 2)), secs=time.time() - t0))


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["SWaT_canon", "WADI_clean", "HAI"]):
        run(ds)
