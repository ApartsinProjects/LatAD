"""A8 re-measurement: is between-regime overlap absent, or only invisible to the
VaDE-responsibility metric?  Observation/feature-space overlap measures computed
on the SAME windows the A8 screens used, plus a synthetic invariant test of the
metric itself, transition-window inspection, and a sub-regime (nested) probe.

Outputs (incremental, one JSON line per item, flushed):  a8_remeasure.jsonl
Log:                                                       a8_remeasure.log
Nothing in the paper or the shipped model is touched.
"""
from __future__ import annotations
import os, sys, json, time, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from scipy.io import loadmat
from scipy.stats import gaussian_kde
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
POC = os.path.dirname(HERE)
sys.path.insert(0, POC); sys.path.insert(0, HERE)
from models_vade import train_vade
from winfeat import window_features
import eda_real as E
from canbus_a8_screen import load_canbus_normal, W as CW, ST as CST

OUT = os.path.join(HERE, "a8_remeasure.jsonl")
LOG2PI = np.log(2 * np.pi)
RNG = np.random.default_rng(0)


def emit(kind, **row):
    row = dict(kind=kind, **row)
    with open(OUT, "a") as f:
        f.write(json.dumps(row, default=float) + "\n"); f.flush()
    print(f"[{kind}] " + json.dumps(row, default=float)[:600], flush=True)


def entropy_norm(G):
    K = G.shape[1]
    return float(np.mean(-(G * np.log(G + 1e-12)).sum(1) / np.log(K)))


def resp_stats(G):
    return dict(mean_maxresp=round(float(G.max(1).mean()), 4), H_norm=round(entropy_norm(G), 4),
                rho_lt0p5=round(float((G.max(1) < 0.5).mean()), 4))


def vade_resp(X, K=16, LD=8, epochs=40, warmup=8, seed=0):
    v = train_vade(X.astype(np.float32), n_clusters=K, latent_dim=LD, epochs=epochs, warmup=warmup, seed=seed, device="cpu")
    G = v._responsibilities(X.astype(np.float32))
    with torch.no_grad():
        z = v.encode(torch.as_tensor(X.astype(np.float32)))[0].numpy().astype(np.float64)
    return v, G, z


# --------------------------------------------------------------------------- pairwise overlap
def bhattacharyya(m1, S1, m2, S2):
    S = 0.5 * (S1 + S2)
    d = m1 - m2
    q = d @ np.linalg.solve(S, d)
    ld = np.linalg.slogdet(S)[1] - 0.5 * (np.linalg.slogdet(S1)[1] + np.linalg.slogdet(S2)[1])
    DB = 0.125 * q + 0.5 * ld
    return float(np.exp(-DB)), float(np.sqrt(q))          # BC in [0,1], Mahalanobis sep (pooled)


def fisher_1d(Za, Zb):
    """Project both clusters on the Fisher direction; return OVL, valley ratio, LDA CV error."""
    m1, m2 = Za.mean(0), Zb.mean(0)
    S = np.cov(Za.T) + np.cov(Zb.T) + 1e-3 * np.eye(Za.shape[1])
    w = np.linalg.solve(S, m1 - m2); w /= np.linalg.norm(w) + 1e-12
    pa, pb = Za @ w, Zb @ w
    lo, hi = min(pa.min(), pb.min()), max(pa.max(), pb.max())
    g = np.linspace(lo, hi, 400)
    try:
        fa, fb = gaussian_kde(pa)(g), gaussian_kde(pb)(g)
    except Exception:
        return dict(ovl=float("nan"), valley=float("nan"), lda_err=float("nan"))
    ovl = float(np.sum(np.minimum(fa, fb)) * (g[1] - g[0]))
    wa = len(pa) / (len(pa) + len(pb))
    fm = wa * fa + (1 - wa) * fb
    ia, ib = int(np.argmin(np.abs(g - pa.mean()))), int(np.argmin(np.abs(g - pb.mean())))
    lo_i, hi_i = min(ia, ib), max(ia, ib)
    if hi_i - lo_i < 2:
        valley = 1.0
    else:
        seg = fm[lo_i:hi_i + 1]
        valley = float(seg.min() / max(1e-12, min(fm[ia], fm[ib])))   # 1 = no dip (unimodal), ->0 deep valley
    Xp = np.r_[Za, Zb]; y = np.r_[np.zeros(len(Za)), np.ones(len(Zb))]
    n_min = min(len(Za), len(Zb))
    if n_min >= 10:
        bacc = cross_val_score(LinearDiscriminantAnalysis(), Xp, y, cv=min(5, n_min), scoring="balanced_accuracy").mean()
        lda_err = float(1 - bacc)
    else:
        lda_err = float("nan")
    return dict(ovl=round(ovl, 4), valley=round(min(valley, 1.0), 4), lda_err=round(lda_err, 4))


def pairwise_overlap(Z, lab, min_n=15, top=None):
    """For each cluster, overlap with its NEAREST cluster (by Bhattacharyya) and summary over all pairs."""
    ks = [k for k in np.unique(lab) if (lab == k).sum() >= min_n]
    stats = {k: (Z[lab == k].mean(0), np.cov(Z[lab == k].T) + 1e-4 * np.eye(Z.shape[1])) for k in ks}
    pairs = []
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            bc, msep = bhattacharyya(stats[a][0], stats[a][1], stats[b][0], stats[b][1])
            pairs.append(dict(a=int(a), b=int(b), bc=round(bc, 4), maha=round(msep, 3)))
    if not pairs:
        return dict(n_clusters=len(ks)), []
    # nearest partner per cluster + 1-D Fisher measures on those pairs
    near = {}
    for p in pairs:
        for x, y in ((p["a"], p["b"]), (p["b"], p["a"])):
            if x not in near or p["bc"] > near[x]["bc"]:
                near[x] = dict(partner=y, bc=p["bc"], maha=p["maha"])
    near_pairs = sorted({tuple(sorted((k, v["partner"]))) for k, v in near.items()})
    f1 = []
    for a, b in near_pairs:
        r = fisher_1d(Z[lab == a], Z[lab == b]); r.update(a=int(a), b=int(b), bc=near[a]["bc"] if near[a]["partner"] == b else near[b]["bc"])
        f1.append(r)
    bcs = np.array([p["bc"] for p in pairs])
    summ = dict(n_clusters=len(ks), n_pairs=len(pairs),
                bc_max=round(float(bcs.max()), 4), bc_mean=round(float(bcs.mean()), 4),
                frac_pairs_bc_gt0p2=round(float((bcs > 0.2).mean()), 4),
                frac_pairs_bc_gt0p5=round(float((bcs > 0.5).mean()), 4),
                nearest_bc_mean=round(float(np.mean([v["bc"] for v in near.values()])), 4),
                nearest_maha_mean=round(float(np.mean([v["maha"] for v in near.values()])), 3),
                nearest_ovl_mean=round(float(np.nanmean([r["ovl"] for r in f1])), 4),
                nearest_ovl_max=round(float(np.nanmax([r["ovl"] for r in f1])), 4),
                nearest_valley_mean=round(float(np.nanmean([r["valley"] for r in f1])), 4),
                frac_nearest_valley_gt0p5=round(float(np.nanmean([r["valley"] > 0.5 for r in f1])), 4),
                nearest_lda_err_mean=round(float(np.nanmean([r["lda_err"] for r in f1])), 4),
                nearest_lda_err_max=round(float(np.nanmax([r["lda_err"] for r in f1])), 4))
    return summ, sorted(f1, key=lambda r: -r["ovl"])[:8]


def cross_nn(Z, lab, stream, pos, k=5, excl=2):
    """Fraction of points whose nearest non-temporal-neighbour lies in another cluster, and the
    ratio d_cross/d_own (cross = nearest point of any other cluster, own = nearest of same cluster)."""
    nn = NearestNeighbors(n_neighbors=min(len(Z), 60)).fit(Z)
    d, idx = nn.kneighbors(Z)
    frac_other, ratio, nn1_other = [], [], []
    for i in range(len(Z)):
        m = ~((stream[idx[i]] == stream[i]) & (np.abs(pos[idx[i]] - pos[i]) <= excl))
        di, ii = d[i][m], idx[i][m]
        if len(ii) < k + 1:
            continue
        nn1_other.append(bool(lab[ii[0]] != lab[i]))
        frac_other.append(float((lab[ii[:k]] != lab[i]).mean()))
        own = di[lab[ii] == lab[i]]; oth = di[lab[ii] != lab[i]]
        if len(own) and len(oth):
            ratio.append(oth[0] / (own[0] + 1e-12))
        elif len(oth):
            ratio.append(0.0)
    fo = np.array(frac_other); r = np.array(ratio) if ratio else np.array([np.inf])
    return dict(frac_nn1_other=round(float(np.mean(nn1_other)), 4),
                mean_frac_k5_other=round(float(fo.mean()), 4),
                frac_ratio_lt1p2=round(float(np.mean(r < 1.2)), 4),
                frac_ratio_lt1=round(float(np.mean(r < 1.0)), 4),
                median_ratio=round(float(np.median(r[np.isfinite(r)])) if np.isfinite(r).any() else float("nan"), 3))


def validity(Z, lab):
    if len(np.unique(lab)) < 2:
        return dict(silhouette=float("nan"), davies_bouldin=float("nan"))
    sub = RNG.choice(len(Z), min(len(Z), 5000), replace=False)
    return dict(silhouette=round(float(silhouette_score(Z[sub], lab[sub])), 4) if len(np.unique(lab[sub])) > 1 else float("nan"),
                davies_bouldin=round(float(davies_bouldin_score(Z, lab)), 3))


# --------------------------------------------------------------------------- data builders
def build_cranfield():
    """Windows W=20/ST=10 as in cranfield_a8_screen, plus: file label, setpoint-segment label,
    transition flag (window overlaps a detected setpoint step), and per-sample arrays."""
    M = loadmat(os.path.join(POC, "datasets", "_new", "Cranfield", "CUcasestudy", "CUcasestudy", "Training.mat"))
    W, ST = 20, 10
    feats, file_lab, seg_lab, trans, stream, pos, tinfo, raw_rows = [], [], [], [], [], [], [], []
    samp_X, samp_file, samp_seg, samp_trans = [], [], [], []
    seg_id = 0
    for fi, key in enumerate(("T1", "T2", "T3")):
        X = np.asarray(M[key], np.float64)
        # setpoint steps from the two flow-rate channels (12, 13) + air-side ch7: 60 s backward vs forward mean
        step = np.zeros(len(X), bool)
        for c in (7, 12, 13):
            x = X[:, c]; sd = x.std() + 1e-9
            cs = np.cumsum(np.r_[0, x])
            for t in range(60, len(X) - 60):
                b = (cs[t] - cs[t - 60]) / 60; f = (cs[t + 60] - cs[t]) / 60
                if abs(f - b) > 0.5 * sd:
                    step[t] = True
        # dilate: transition zone = +-30 s around a step center; segments = runs between steps
        trans_t = np.convolve(step.astype(int), np.ones(61), "same") > 0
        seg = np.zeros(len(X), int); cur = seg_id; inside = False
        for t in range(len(X)):
            if trans_t[t]:
                inside = True; seg[t] = -1
            else:
                if inside:
                    cur += 1; inside = False
                seg[t] = cur
        seg_id = cur + 1
        samp_X.append(X); samp_file.append(np.full(len(X), fi)); samp_seg.append(seg); samp_trans.append(trans_t)
        for j, i in enumerate(range(0, len(X) - W + 1, ST)):
            win = X[i:i + W]
            feats.append(window_features(win, "stats")); file_lab.append(fi)
            s = seg[i:i + W]; ok = s[s >= 0]
            seg_lab.append(int(np.bincount(ok).argmax()) if len(ok) else -1)
            trans.append(bool(trans_t[i:i + W].any())); stream.append(fi); pos.append(j)
            tinfo.append(dict(file=key, t0=int(i), ch7=(round(win[0, 7], 1), round(win[-1, 7], 1)),
                              ch12=(round(win[0, 12], 1), round(win[-1, 12], 1)), ch13=(round(win[0, 13], 1), round(win[-1, 13], 1))))
    return dict(name="Cranfield", X=np.asarray(feats, np.float32), file=np.array(file_lab), seg=np.array(seg_lab),
                trans=np.array(trans), stream=np.array(stream), pos=np.array(pos), info=tinfo,
                samp=dict(X=np.concatenate(samp_X), file=np.concatenate(samp_file), seg=np.concatenate(samp_seg), trans=np.concatenate(samp_trans)))


def build_canbus():
    trips = load_canbus_normal()
    feats, trans, mid, stream, pos, info = [], [], [], [], [], []
    for si, (vid, X) in enumerate(trips.items()):
        for j, i in enumerate(range(0, len(X) - CW + 1, CST)):
            win = X[i:i + CW]
            feats.append(window_features(win, "stats"))
            sp = win[:, 0]
            trans.append(bool(sp.max() - sp.min() > 20))          # >20 km/h swing inside 24 s
            mid.append(bool(abs(sp[-1] - sp[0]) > 15))            # net ramp >15 km/h
            stream.append(si); pos.append(j)
            info.append(dict(trip=vid, t0=int(i), speed=sp.round(0).tolist(), rpm=win[:, 1].round(0).tolist(), throttle=win[:, 2].round(0).tolist()))
    return dict(name="CANbus", X=np.asarray(feats, np.float32), trans=np.array(trans), mid=np.array(mid),
                stream=np.array(stream), pos=np.array(pos), info=info)


def build_eda(name):
    D = E.load(name)
    X = np.asarray(D["Xn_w"], np.float32)
    n = len(X)
    return dict(name=name, X=X, stream=np.zeros(n, int), pos=np.arange(n), W=D["W"], raw=D["Xn_raw"], ch=D["ch"])


# --------------------------------------------------------------------------- synthetic invariant
def _two_regime_case(tag, delta, Xa, Xb, G_true, n_init=5):
    """Common evaluation of a 2-regime synthetic: Bayes (true) posterior vs obs-GMM vs VaDE, with ARI."""
    n = len(Xa)
    X = np.r_[Xa, Xb]; lab = np.r_[np.zeros(n), np.ones(n)].astype(int)
    Xs = ((X - X.mean(0)) / (X.std(0) + 1e-8)).astype(np.float32)
    gm = GaussianMixture(2, covariance_type="full", random_state=0, n_init=n_init, reg_covar=1e-3).fit(Xs); G_obs = gm.predict_proba(Xs)
    row = dict(case=tag, delta=delta, d=int(X.shape[1]), bayes=resp_stats(G_true),
               bayes_error=round(float((G_true.argmax(1) != lab).mean()), 4),
               obs_gmm_K2=resp_stats(G_obs), obs_gmm_K2_ari=round(float(adjusted_rand_score(lab, G_obs.argmax(1))), 3))
    sub = RNG.choice(len(Xs), 3000, replace=False)
    row["sil_true_obs"] = round(float(silhouette_score(Xs[sub], lab[sub])), 4)
    Z10 = PCA(min(10, X.shape[1]), random_state=0).fit_transform(Xs)
    summ, _ = pairwise_overlap(Z10, lab)
    row["obs_pairwise_true_labels"] = summ
    for K in (2, 8, 16):
        v, G, z = vade_resp(Xs, K=K, LD=8, epochs=40, warmup=8, seed=0)
        vl = G.argmax(1)
        r = resp_stats(G)
        r.update(ari=round(float(adjusted_rand_score(lab, vl)), 3),
                 frac_comp_at_floor=round(float((np.abs(v._lvc().detach().numpy() - v.logvar_floor) < 1e-6).mean()), 3),
                 latent_sd=round(float(z.std(0).mean()), 3), comp_sd=round(float(np.exp(0.5 * v._lvc().detach().numpy()).mean()), 3),
                 sil_true_latent=round(float(silhouette_score(z[sub], lab[sub])), 4),
                 # how much of the responsibility entropy survives if we MERGE components by true regime:
                 # G2 = responsibility mass on the components whose majority is regime 0 vs regime 1
                 purity=round(float(np.mean([np.bincount(lab[vl == k]).max() / max(1, (vl == k).sum()) for k in np.unique(vl)])), 3))
        maj = np.array([np.bincount(lab[vl == k], minlength=2).argmax() if (vl == k).sum() else 0 for k in range(K)])
        G2 = np.c_[G[:, maj == 0].sum(1), G[:, maj == 1].sum(1)] if (maj == 0).any() and (maj == 1).any() else None
        r["merged_by_true_regime"] = resp_stats(G2) if G2 is not None else None
        row[f"vade_K{K}"] = r
    emit("synthetic_two_regime", **row)


def synthetic_tests():
    n = 2500
    # (A) isotropic noise, mean shift in ONE of 10 dims (signal is 10% of the variance): control for
    #     whether the recipe even finds the structure.
    d = 10
    for delta in (2.0, 4.0):
        Xa = RNG.normal(size=(n, d)); Xb = RNG.normal(size=(n, d)); Xb[:, 0] += delta
        X = np.r_[Xa, Xb]
        la = -0.5 * ((X[:, 0]) ** 2); lb = -0.5 * ((X[:, 0] - delta) ** 2)
        G_true = np.exp(np.c_[la, lb] - logsumexp(np.c_[la, lb], 1, keepdims=True))
        _two_regime_case("A_isotropic_1of10dims", delta, Xa, Xb, G_true)
    # (B) low-rank, low-noise, like real window features: 3 latent factors -> 48 features, regimes
    #     shifted by delta (in factor-1 units, unit within-regime variance), noise sd 0.1.
    d, r = 48, 3
    A = RNG.normal(size=(r, d))
    for delta in (1.0, 2.0, 3.0, 4.0):
        Sa = RNG.normal(size=(n, r)); Sb = RNG.normal(size=(n, r)); Sb[:, 0] += delta
        Xa = Sa @ A + 0.1 * RNG.normal(size=(n, d)); Xb = Sb @ A + 0.1 * RNG.normal(size=(n, d))
        S = np.r_[Sa, Sb]
        la = -0.5 * (S[:, 0] ** 2); lb = -0.5 * ((S[:, 0] - delta) ** 2)
        G_true = np.exp(np.c_[la, lb] - logsumexp(np.c_[la, lb], 1, keepdims=True))
        _two_regime_case("B_lowrank_48feat", delta, Xa, Xb, G_true)
    # (C) continuum: NO regimes at all (1-D uniform latent mapped through the same low-rank map + 2 nuisance factors)
    n2 = 5000; t = RNG.uniform(0, 1, n2)
    S = np.c_[(t - 0.5) * 8.0, RNG.normal(size=(n2, 2))]
    X = S @ A + 0.1 * RNG.normal(size=(n2, d))
    Xs = ((X - X.mean(0)) / (X.std(0) + 1e-8)).astype(np.float32)
    row = dict(case="C_continuum_lowrank_48feat")
    Z10 = PCA(10, random_state=0).fit_transform(Xs)
    for K in (8, 16):
        v, G, z = vade_resp(Xs, K=K, LD=8, epochs=40, warmup=8, seed=0)
        r = resp_stats(G)
        gm = GaussianMixture(K, covariance_type="full", random_state=0, reg_covar=1e-3, n_init=2).fit(Z10)
        r["obs_gmm_full_pca10"] = resp_stats(gm.predict_proba(Z10))
        lab = G.argmax(1); sub = RNG.choice(n2, 3000, replace=False)
        r["sil_vadelabels_obs"] = round(float(silhouette_score(Z10[sub], lab[sub])), 4) if len(np.unique(lab[sub])) > 1 else None
        r["sil_vadelabels_latent"] = round(float(silhouette_score(z[sub], lab[sub])), 4) if len(np.unique(lab[sub])) > 1 else None
        summ, _ = pairwise_overlap(Z10, lab)
        r["obs_pairwise_vadelabels"] = summ
        r["crossnn_vadelabels"] = cross_nn(Z10, lab, np.zeros(n2, int), np.arange(n2), excl=0)
        r["corr_component_order_vs_t"] = round(float(abs(np.corrcoef(v.mu_c.detach().numpy()[lab, 0], t)[0, 1])), 3)
        row[f"vade_K{K}"] = r
    emit("synthetic_continuum", **row)


# --------------------------------------------------------------------------- per-dataset measure
def measure(D, K=16):
    name = D["name"]; X = D["X"]; n = len(X)
    Xs = ((X - X.mean(0)) / (X.std(0) + 1e-8)).astype(np.float32)
    t0 = time.time()
    v, G, z = vade_resp(Xs, K=K, LD=8, epochs=40, warmup=8, seed=0)
    vlab = G.argmax(1)
    Z10 = PCA(10, random_state=0).fit_transform(Xs)
    gm = GaussianMixture(K, covariance_type="full", random_state=0, reg_covar=1e-3, n_init=2).fit(Z10)
    Gobs = gm.predict_proba(Z10); olab = Gobs.argmax(1)
    gmd = GaussianMixture(K, covariance_type="diag", random_state=0, reg_covar=1e-3).fit(Xs)
    Gd = gmd.predict_proba(Xs)
    km = KMeans(8, n_init=5, random_state=0).fit(Z10); klab = km.labels_
    row = dict(dataset=name, n_windows=n, n_features=int(X.shape[1]),
               vade_K16=resp_stats(G), obs_gmm_full_pca10_K16=resp_stats(Gobs), obs_gmm_diag_fullfeat_K16=resp_stats(Gd),
               vade_frac_comp_at_floor=round(float((np.abs(v._lvc().detach().numpy() - v.logvar_floor) < 1e-6).mean()), 3),
               vade_latent_sd=round(float(z.std(0).mean()), 3), vade_comp_sd=round(float(np.exp(0.5 * v._lvc().detach().numpy()).mean()), 3),
               ari_vade_vs_obsgmm=round(float(adjusted_rand_score(vlab, olab)), 3))
    row["validity_obs_pca10"] = dict(vade_labels=validity(Z10, vlab), obsgmm_labels=validity(Z10, olab), kmeans8=validity(Z10, klab))
    row["validity_vade_latent"] = dict(vade_labels=validity(z, vlab))
    for tag, lab in (("vade", vlab), ("obsgmm", olab), ("kmeans8", klab)):
        summ, worst = pairwise_overlap(Z10, lab)
        row[f"pairwise_{tag}"] = summ
        row[f"pairwise_{tag}_top_overlap_pairs"] = worst[:4]
        row[f"crossnn_{tag}"] = cross_nn(Z10, lab, D["stream"], D["pos"])
    # same pairwise analysis in the VaDE LATENT (to show the latent pulls clusters apart)
    summ_lat, _ = pairwise_overlap(z, vlab)
    row["pairwise_vade_in_latent"] = summ_lat
    row["crossnn_vade_in_latent"] = cross_nn(z, vlab, D["stream"], D["pos"])
    row["secs"] = round(time.time() - t0, 1)
    emit("dataset_measures", **row)
    return dict(v=v, G=G, z=z, vlab=vlab, Z10=Z10, gm=gm, Gobs=Gobs, olab=olab, Xs=Xs)


# --------------------------------------------------------------------------- transitions
def inspect_transitions(D, R, trans, label, n_show=6, extra=None):
    name = D["name"]; G, Gobs, gm, Z10 = R["G"], R["Gobs"], R["gm"], R["Z10"]
    ld = gm.score_samples(Z10)
    nn = NearestNeighbors(n_neighbors=6).fit(Z10); dnn = nn.kneighbors(Z10)[0][:, 1:].mean(1)
    steady = ~trans
    pct = lambda a, ref: float((ref[None, :] <= a[:, None]).mean(1).mean())  # mean percentile of a within ref
    row = dict(dataset=name, label=label, n_trans=int(trans.sum()), frac_trans=round(float(trans.mean()), 4),
               vade_maxresp_trans=round(float(G.max(1)[trans].mean()), 4), vade_maxresp_steady=round(float(G.max(1)[steady].mean()), 4),
               vade_H_trans=round(entropy_norm(G[trans]), 4), vade_H_steady=round(entropy_norm(G[steady]), 4),
               vade_rho_trans=round(float((G.max(1)[trans] < 0.5).mean()), 4), vade_rho_steady=round(float((G.max(1)[steady] < 0.5).mean()), 4),
               obs_maxresp_trans=round(float(Gobs.max(1)[trans].mean()), 4), obs_maxresp_steady=round(float(Gobs.max(1)[steady].mean()), 4),
               obs_logdens_trans_mean=round(float(ld[trans].mean()), 2), obs_logdens_steady_mean=round(float(ld[steady].mean()), 2),
               obs_logdens_trans_pct_in_steady=round(pct(ld[trans], ld[steady]), 3),
               frac_trans_below_steady_p05=round(float((ld[trans] < np.quantile(ld[steady], 0.05)).mean()), 4),
               frac_trans_below_steady_p01=round(float((ld[trans] < np.quantile(ld[steady], 0.01)).mean()), 4),
               nn_dist_trans_pct_in_steady=round(pct(dnn[trans], dnn[steady]), 3),
               vade_latent_nll_nearest_trans_minus_steady=None)
    # VaDE's own nearest-mode NLL (its anomaly-score latent term) on transition vs steady windows
    v = R["v"]
    with torch.no_grad():
        nll = -v._log_pz_given_c(torch.as_tensor(R["z"], dtype=torch.float32)).max(1).values.numpy()
    row["vade_latent_nll_nearest_trans_minus_steady"] = round(float(nll[trans].mean() - nll[steady].mean()), 3)
    row["vade_latent_nll_trans_pct_in_steady"] = round(pct(nll[trans], nll[steady]), 3)
    if extra: row.update(extra)
    emit("transition_summary", **row)
    # concrete examples: the transition windows with the LOWEST obs-density percentile, and some random ones
    ti = np.where(trans)[0]
    if len(ti):
        order = ti[np.argsort(ld[ti])]
        picks = list(order[:n_show // 2]) + list(RNG.choice(ti, min(n_show - n_show // 2, len(ti)), replace=False))
        for i in picks:
            g2 = np.argsort(G[i])[::-1][:2]; o2 = np.argsort(Gobs[i])[::-1][:2]
            ex = dict(dataset=name, label=label, idx=int(i), info=D["info"][i] if "info" in D else None,
                      vade_top2=[(int(g2[0]), round(float(G[i, g2[0]]), 3)), (int(g2[1]), round(float(G[i, g2[1]]), 3))],
                      obs_top2=[(int(o2[0]), round(float(Gobs[i, o2[0]]), 3)), (int(o2[1]), round(float(Gobs[i, o2[1]]), 3))],
                      obs_logdens=round(float(ld[i]), 2), obs_logdens_pct_in_steady=round(float((ld[steady] <= ld[i]).mean()), 3),
                      nn_dist_pct_in_steady=round(float((dnn[steady] <= dnn[i]).mean()), 3),
                      vade_nll_pct_in_steady=round(float((nll[steady] <= nll[i]).mean()), 3))
            emit("transition_example", **ex)


# --------------------------------------------------------------------------- sub-regime probe
def subregime(D, R):
    name = D["name"]; Z10 = R["Z10"]
    sub = RNG.choice(len(Z10), min(len(Z10), 4000), replace=False); Zs = Z10[sub]
    sil = {}
    for k in (2, 3, 4, 6, 8, 12, 16, 24):
        lab = AgglomerativeClustering(k, linkage="ward").fit_predict(Zs)
        sil[k] = round(float(silhouette_score(Zs, lab)), 4)
    parent = AgglomerativeClustering(4, linkage="ward").fit_predict(Zs)
    child = AgglomerativeClustering(16, linkage="ward").fit_predict(Zs)
    # map child -> parent (Ward is nested, so each child sits in exactly one parent)
    cp = {c: int(np.bincount(parent[child == c]).argmax()) for c in np.unique(child)}
    stats = {c: (Zs[child == c].mean(0), np.cov(Zs[child == c].T) + 1e-4 * np.eye(10)) for c in np.unique(child) if (child == c).sum() >= 15}
    within, across = [], []
    ks = list(stats)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            bc, _ = bhattacharyya(stats[a][0], stats[a][1], stats[b][0], stats[b][1])
            (within if cp[a] == cp[b] else across).append(bc)
    row = dict(dataset=name, ward_silhouette_by_k=sil,
               child16_within_parent_bc_mean=round(float(np.mean(within)), 4) if within else None,
               child16_within_parent_bc_max=round(float(np.max(within)), 4) if within else None,
               child16_across_parent_bc_mean=round(float(np.mean(across)), 4) if across else None,
               child16_across_parent_bc_max=round(float(np.max(across)), 4) if across else None,
               n_within_pairs=len(within), n_across_pairs=len(across))
    # nearest-partner Fisher measures for within-parent child pairs
    f1 = []
    for a in ks:
        cands = [b for b in ks if b != a and cp[b] == cp[a]]
        if not cands: continue
        b = max(cands, key=lambda b: bhattacharyya(stats[a][0], stats[a][1], stats[b][0], stats[b][1])[0])
        r = fisher_1d(Zs[child == a], Zs[child == b]); r.update(a=int(a), b=int(b), parent=cp[a]); f1.append(r)
    if f1:
        row["within_parent_nearest_ovl_mean"] = round(float(np.nanmean([r["ovl"] for r in f1])), 4)
        row["within_parent_nearest_valley_mean"] = round(float(np.nanmean([r["valley"] for r in f1])), 4)
        row["within_parent_frac_valley_gt0p5"] = round(float(np.nanmean([r["valley"] > 0.5 for r in f1])), 4)
        row["within_parent_lda_err_mean"] = round(float(np.nanmean([r["lda_err"] for r in f1])), 4)
    if "file" in D:
        row["ari_ward4_vs_file"] = round(float(adjusted_rand_score(D["file"][sub], parent)), 3)
        row["ari_ward16_vs_setpoint_seg"] = round(float(adjusted_rand_score(D["seg"][sub], child)), 3)
        row["ari_vade16_vs_setpoint_seg"] = round(float(adjusted_rand_score(D["seg"], R["vlab"])), 3)
        row["ari_vade16_vs_file"] = round(float(adjusted_rand_score(D["file"], R["vlab"])), 3)
        row["n_setpoint_segments"] = int(len(np.unique(D["seg"][D["seg"] >= 0])))
    emit("subregime", **row)


# --------------------------------------------------------------------------- main
def main():
    global OUT
    which = sys.argv[1].split(",") if len(sys.argv) > 1 else ["synthetic", "Cranfield", "CANbus", "HAI", "SWaT", "WADI"]
    if len(which) == 1:
        OUT = os.path.join(HERE, f"a8_remeasure.{which[0]}.jsonl")
    torch.set_num_threads(2)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line); done.add((r["kind"], r.get("dataset", r.get("delta", ""))))
            except Exception:
                pass
    if "synthetic" in which and ("synthetic_continuum", "") not in done:
        synthetic_tests()
    for name in which:
        if name == "synthetic": continue
        if ("subregime", name) in done:
            print(f"cached {name}", flush=True); continue
        t0 = time.time()
        if name == "Cranfield": D = build_cranfield()
        elif name == "CANbus": D = build_canbus()
        else: D = build_eda(name)
        print(f"== {name}: {D['X'].shape} built in {time.time()-t0:.0f}s", flush=True)
        R = measure(D)
        Z10 = R["Z10"]
        if name == "Cranfield":
            # observation-space overlap using the TRUE labels (file, setpoint segment)
            for tag, lab in (("file", D["file"]), ("setpoint_seg", D["seg"])):
                m = lab >= 0
                summ, worst = pairwise_overlap(Z10[m], lab[m])
                emit("cranfield_true_labels", label=tag, validity=validity(Z10[m], lab[m]),
                     crossnn=cross_nn(Z10[m], lab[m], D["stream"][m], D["pos"][m]), pairwise=summ, top_overlap_pairs=worst[:5],
                     vade_purity=round(float(np.mean([np.bincount(lab[m][R["vlab"][m] == k]).max() / max(1, (R["vlab"][m] == k).sum()) for k in np.unique(R["vlab"][m])])), 3))
            inspect_transitions(D, R, D["trans"], "setpoint_step_windows")
            # sample level (1 Hz timesteps, no windowing): does averaging hide the transitions?
            S = D["samp"]; Xs = (S["X"] - S["X"].mean(0)) / (S["X"].std(0) + 1e-8)
            sub = RNG.choice(len(Xs), 8000, replace=False)
            P = PCA(10, random_state=0).fit(Xs[sub]); Zp = P.transform(Xs[sub])
            for tag, lab in (("file", S["file"][sub]), ("setpoint_seg", S["seg"][sub])):
                m = lab >= 0
                summ, worst = pairwise_overlap(Zp[m], lab[m])
                emit("cranfield_sample_level", label=tag, n=int(m.sum()), validity=validity(Zp[m], lab[m]), pairwise=summ,
                     top_overlap_pairs=worst[:5], frac_samples_in_transition=round(float(S["trans"].mean()), 4))
            # density of transition SAMPLES under a GMM on steady samples
            gmS = GaussianMixture(16, covariance_type="full", random_state=0, reg_covar=1e-3).fit(Zp[S["trans"][sub] == 0])
            ldS = gmS.score_samples(Zp); tr = S["trans"][sub]
            emit("cranfield_sample_level_transition_density", n_trans=int(tr.sum()),
                 logdens_trans_pct_in_steady=round(float((ldS[~tr][None, :] <= ldS[tr][:, None]).mean()), 3),
                 frac_trans_below_steady_p05=round(float((ldS[tr] < np.quantile(ldS[~tr], 0.05)).mean()), 4),
                 gmm_maxresp_trans=round(float(gmS.predict_proba(Zp[tr]).max(1).mean()), 4),
                 gmm_maxresp_steady=round(float(gmS.predict_proba(Zp[~tr]).max(1).mean()), 4))
        elif name == "CANbus":
            inspect_transitions(D, R, D["trans"], "speed_swing_gt20kmh", extra=dict(frac_mid_ramp_gt15=round(float(D["mid"].mean()), 4)))
            inspect_transitions(D, R, D["mid"], "net_ramp_gt15kmh")
        else:
            # transitions = windows where the obs-GMM assignment differs from BOTH temporal neighbours' shared label,
            # i.e. the window sits at a boundary between two runs of different regimes
            ol = R["olab"]; b = np.zeros(len(ol), bool)
            b[1:-1] = (ol[1:-1] != ol[:-2]) | (ol[1:-1] != ol[2:])
            D["info"] = [dict(idx=int(i), obs_lab_prev=int(ol[max(0, i - 1)]), obs_lab=int(ol[i]), obs_lab_next=int(ol[min(len(ol) - 1, i + 1)])) for i in range(len(ol))]
            inspect_transitions(D, R, b, "obsgmm_assignment_boundary")
            # also: windows straddling a change point of the raw signal (largest first-half vs second-half shift)
            X = D["X"]; C = X.shape[1] // 6
            trend = np.abs(X[:, 4 * C:5 * C]).max(1)   # |last-first| per channel, max over channels (standardised raw)
            hi = trend > np.quantile(trend, 0.95)
            inspect_transitions(D, R, hi, "top5pct_within_window_trend")
        subregime(D, R)
        print(f"== {name} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
