"""A9 (many clocks / multiscale) and A10 (path dependence) MEASUREMENT on the real testbeds.

The paper asserts A9/A10 are "specified but not realized; the public benchmarks are snapshot-
detectable". This script turns the assertion into numbers. Nothing in the paper or the shipped
model is touched. Results append to a9a10_measure.jsonl (one row per item, flushed); the write-up
is a9a10_measure.md.

Usage (from poc/):
  python _diagnostics/a9a10_measure.py timescales WADI_clean
  python _diagnostics/a9a10_measure.py wsweep     WADI_clean
  python _diagnostics/a9a10_measure.py a10        WADI_clean [--vade]
  python _diagnostics/a9a10_measure.py synth

Datasets: WADI_clean (paper config, 575 test windows), HAI, SWaT_canon.
Window grid: the paper's W=60 / stride=30 grid; labels = attack fraction > 5% in the W=60 slot.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from winfeat import window_features

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSONL = os.path.join(HERE, "a9a10_measure.jsonl")
CACHE = os.environ.get("A9A10_CACHE", r"E:\tmp\claude\E--Projects-Backlog-LatAD\6ae0c3f3-0eaf-4beb-8a09-88ce1e618d2a\scratchpad")
os.makedirs(CACHE, exist_ok=True)
W0, ST = 60, 30
RNG = np.random.default_rng(0)


def emit(row):
    row["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=_js) + "\n"); f.flush()
    print("EMIT", json.dumps(row, default=_js)[:400], flush=True)


def _js(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)


# ----------------------------------------------------------------------------- data
def load_raw(name):
    """Standardized (on train-normal), clipped raw streams; cached as npz."""
    p = os.path.join(CACHE, f"raw_{name}.npz")
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        return d["Xn"], d["Xa"], d["ya"], list(d["ch"])
    import eda_real as E
    fn, W, stride = E.RAW[name]
    clip = E.CLIP.get(name)
    Xn, Xa, ya, ch = fn()
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-8
    Xn, Xa = (Xn - mu) / sd, (Xa - mu) / sd
    if clip:
        Xn, Xa = np.clip(Xn, -clip, clip), np.clip(Xa, -clip, clip)
    np.savez(p, Xn=Xn.astype(np.float32), Xa=Xa.astype(np.float32), ya=ya.astype(int), ch=np.array(ch))
    return Xn.astype(np.float32), Xa.astype(np.float32), ya.astype(int), ch


def grid_ends(T):
    """Window END indices on the paper's grid (start i in range(0, T-W0+1, ST))."""
    return np.arange(0, T - W0 + 1, ST) + W0


def feats_at(X, ends, W):
    """Stats features for windows X[e-W:e]; rows with e-W<0 are NaN (kept for alignment)."""
    F = np.full((len(ends), 6 * X.shape[1]), np.nan, np.float32)
    for k, e in enumerate(ends):
        if e - W >= 0:
            F[k] = window_features(X[e - W:e], "stats")
    return F


def labels_at(y, ends):
    return np.array([int(y[e - W0:e].mean() > 0.05) for e in ends], int)


def pct_of(ref, x):
    """Percentile of x within the reference (train-normal) score distribution."""
    r = np.sort(ref)
    return np.searchsorted(r, x, side="right") / len(r)


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(s)
    if (y[m] == 1).sum() == 0 or (y[m] == 0).sum() == 0:
        return float("nan")
    return float(roc_auc_score(y[m], s[m]))


def episodes(yw):
    """Contiguous runs of label==1 on the window grid -> list of (start_idx, end_idx_exclusive)."""
    ep, i = [], 0
    while i < len(yw):
        if yw[i] == 1:
            j = i
            while j < len(yw) and yw[j] == 1:
                j += 1
            ep.append((i, j)); i = j
        else:
            i += 1
    return ep


# ============================================================================= A9.1 timescales
def acf_tau(x, lmax):
    x = x - x.mean(); v = float(np.dot(x, x))
    if v < 1e-9:
        return np.nan
    lmax = min(lmax, len(x) - 1)
    # vectorised via FFT
    n = len(x); f = np.fft.rfft(x, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:lmax + 1] / v
    below = np.where(ac[1:] < np.exp(-1))[0]
    return float(below[0] + 1) if len(below) else float(lmax)


def dominant_period(x, fs=1.0):
    from scipy.signal import welch
    x = x - x.mean()
    if x.std() < 1e-6:
        return np.nan
    nper = min(len(x), 8192)
    f, P = welch(x, fs=fs, nperseg=nper)
    P[0] = 0
    k = int(np.argmax(P))
    return float(1.0 / f[k]) if f[k] > 0 else np.nan


def run_timescales(name):
    Xn, Xa, ya, ch = load_raw(name)
    T = min(len(Xn), 200_000)
    Xn = Xn[:T].astype(np.float64)
    taus, pers = [], []
    for c in range(Xn.shape[1]):
        taus.append(acf_tau(Xn[:, c], lmax=5000)); pers.append(dominant_period(Xn[:, c]))
    taus = np.array(taus); pers = np.array(pers)
    act = np.isfinite(taus)
    t = taus[act]; p = pers[np.isfinite(pers)]
    lt = np.log10(np.maximum(t, 1))
    bins = {"tau<W/3(20)": float((t < 20).mean()), "W/3..W(20-60)": float(((t >= 20) & (t < 60)).mean()),
            "W..3W(60-180)": float(((t >= 60) & (t < 180)).mean()), "3W..10W(180-600)": float(((t >= 180) & (t < 600)).mean()),
            ">10W(>=600)": float((t >= 600).mean())}
    row = dict(item="A9_timescales", dataset=name, n_channels=len(ch), n_active=int(act.sum()),
               samples_used=T, acf_lmax=5000,
               tau_p05=float(np.percentile(t, 5)), tau_p25=float(np.percentile(t, 25)), tau_p50=float(np.median(t)),
               tau_p75=float(np.percentile(t, 75)), tau_p95=float(np.percentile(t, 95)),
               tau_decades_p95_p05=float(np.log10(np.percentile(t, 95) / max(np.percentile(t, 5), 1))),
               log10_tau_std=float(lt.std()), tau_bins_vs_W60=bins,
               period_p05=float(np.percentile(p, 5)), period_p50=float(np.median(p)), period_p95=float(np.percentile(p, 95)),
               period_decades=float(np.log10(np.percentile(p, 95) / max(np.percentile(p, 5), 1e-3))),
               fastest=[(ch[i], float(taus[i])) for i in np.argsort(np.where(act, taus, np.inf))[:5]],
               slowest=[(ch[i], float(taus[i])) for i in np.argsort(np.where(act, -taus, np.inf))[:5]])
    emit(row)
    return row


# ============================================================================= A9.2 window sweep
def fit_detectors(Ftr, seed=0, K=24, d=30):
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.ensemble import IsolationForest
    m = np.isfinite(Ftr).all(1); Ftr = Ftr[m]
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Z = (Ftr - mu) / sd
    pca = PCA(n_components=min(d, Z.shape[1]), random_state=seed).fit(Z)
    Ztr = pca.transform(Z)
    gmm = GaussianMixture(K, covariance_type="diag", random_state=seed, reg_covar=1e-2, max_iter=200).fit(Ztr)
    IF = IsolationForest(n_estimators=200, random_state=seed).fit(Z)
    def score(F):
        ok = np.isfinite(F).all(1); out = {k: np.full(len(F), np.nan) for k in ("gmm", "IF")}
        Zt = (F[ok] - mu) / sd
        out["gmm"][ok] = -gmm.score_samples(pca.transform(Zt)); out["IF"][ok] = -IF.decision_function(Zt)
        return out
    return score, dict(gmm=-gmm.score_samples(Ztr), IF=-IF.decision_function(Z)), m


def run_wsweep(name, Ws=(20, 60, 180, 600), seed=0):
    Xn, Xa, ya, ch = load_raw(name); C = len(ch)
    en, ea = grid_ends(len(Xn)), grid_ends(len(Xa))
    yw = labels_at(ya, ea)
    # fixed difficult set: the paper's W=60 max|u| rule
    F60n, F60a = feats_at(Xn, en, W0), feats_at(Xa, ea, W0)
    triv_n, triv_a = np.abs(F60n[:, :C]).max(1), np.abs(F60a[:, :C]).max(1)
    thr99 = np.quantile(triv_n[np.isfinite(triv_n)], 0.99)
    hard = (yw == 1) & (triv_a <= thr99)
    eps = episodes(yw)
    emit(dict(item="A9_wsweep_setup", dataset=name, n_train=len(en), n_test=len(ea), n_anom=int(yw.sum()),
              n_hard=int(hard.sum()), n_episodes=len(eps), thr99=float(thr99)))
    pct_scores = {}
    ep_rows = {}
    for W in Ws:
        t0 = time.time()
        Fn = F60n if W == W0 else feats_at(Xn, en, W)
        Fa = F60a if W == W0 else feats_at(Xa, ea, W)
        score, tr_scores, mtr = fit_detectors(Fn, seed=seed)
        te = score(Fa)
        te["trivial"] = np.abs(Fa[:, :C]).max(1); tr_scores["trivial"] = np.abs(Fn[mtr, :C]).max(1)
        row = dict(item="A9_wsweep", dataset=name, W=W, stride=ST, n_test_valid=int(np.isfinite(te["gmm"]).sum()))
        for k in ("trivial", "gmm", "IF"):
            s = te[k]; ok = np.isfinite(s)
            row[f"{k}_auroc_all"] = auroc(yw[ok], s[ok])
            msk = ok & ((yw == 0) | hard)
            row[f"{k}_auroc_hard"] = auroc(yw[msk], s[msk])
            thr = np.quantile(tr_scores[k], 0.95)
            row[f"{k}_tpr5_hard"] = float((s[ok & hard] > thr).mean()) if (ok & hard).any() else float("nan")
            row[f"{k}_fpr5_testnormal"] = float((s[ok & (yw == 0)] > thr).mean())
            pct_scores[(k, W)] = pct_of(tr_scores[k], s)
        # per-episode separability (gmm)
        pc = pct_scores[("gmm", W)]
        ep_rows[W] = []
        for (a, b) in eps:
            seg = pc[a:b]; seg = seg[np.isfinite(seg)]
            ep_rows[W].append(dict(ep=(a, b), n=b - a, n_hard=int(hard[a:b].sum()),
                                   med_pct=float(np.median(seg)) if len(seg) else float("nan"),
                                   frac_above95=float((seg > 0.95).mean()) if len(seg) else float("nan")))
        # contamination control: hard windows whose EXTRA history (the W-60 samples before the anchor
        # slot) is attack-free (strict) or contains no easy-anomaly slot (loose). Longer windows can
        # otherwise flag a window through an earlier easy anomaly sitting in its history.
        easy_slot = (yw == 1) & (triv_a > thr99)
        strict = np.array([ya[max(0, e - W):e - W0].sum() == 0 if e - W >= 0 else False for e in ea])
        nprev = max(0, (W - W0) // ST)
        loose = np.array([not easy_slot[max(0, i - nprev):i].any() if ea[i] - W >= 0 else False for i in range(len(ea))])
        for tag, hm in (("strict", strict), ("loose", loose)):
            for k in ("gmm", "IF"):
                s = te[k]; ok = np.isfinite(s)
                m = ok & ((yw == 0) | (hard & hm))
                row[f"{k}_auroc_hard_{tag}hist"] = auroc(yw[m], s[m])
                row[f"n_hard_{tag}hist"] = int((hard & hm & ok).sum())
                thr = np.quantile(tr_scores[k], 0.95)
                row[f"{k}_tpr5_hard_{tag}hist"] = float((s[ok & hard & hm] > thr).mean()) if (ok & hard & hm).any() else float("nan")
        row["secs"] = round(time.time() - t0, 1)
        emit(row)
    np.savez(os.path.join(HERE, f"a9a10_wsweep_{name}.npz"), y=yw, hard=hard, ends=ea,
             **{f"pct_{k}_W{W}": pct_scores[(k, W)] for (k, W) in pct_scores})
    # per-episode shift table
    shifts = []
    for i, (a, b) in enumerate(eps):
        d = {W: ep_rows[W][i]["frac_above95"] for W in Ws}
        shifts.append(dict(ep=(a, b), n=b - a, n_hard=int(hard[a:b].sum()), frac_above95_by_W={str(W): d[W] for W in Ws},
                           gain_vs_W60=float(np.nanmax([d[W] for W in Ws if W != W0]) - d[W0])))
    emit(dict(item="A9_wsweep_episodes", dataset=name, episodes=shifts))
    # multiscale fusion: max percentile over W (gmm), and over W for IF
    for k in ("gmm", "IF"):
        P = np.stack([pct_scores[(k, W)] for W in Ws]); ok = np.isfinite(P).all(0)
        fused = np.nanmax(P, 0)
        msk = ok & ((yw == 0) | hard)
        emit(dict(item="A9_wsweep_fusion", dataset=name, detector=k, Ws=list(Ws),
                  auroc_hard_fused_maxpct=auroc(yw[msk], fused[msk]),
                  auroc_hard_W60_pct=auroc(yw[msk], pct_scores[(k, W0)][msk]),
                  auroc_hard_by_W={str(W): auroc(yw[msk], pct_scores[(k, W)][msk]) for W in Ws},
                  n_hard_valid=int((msk & (yw == 1)).sum())))


# ============================================================================= A10 conditional
class CondModels:
    """History = the two preceding NON-overlapping windows on the grid (lags 2 and 4 = 60 and 120
    samples back). Three conditional scorers + Gaussian and GMM marginals, all on the same latent Z."""

    def __init__(self, K=24, knn=20, seed=0):
        self.K, self.knn, self.seed = K, knn, seed

    @staticmethod
    def hist(Z, lags=(2, 4)):
        n = len(Z); L = max(lags)
        H = np.full((n, Z.shape[1] * len(lags)), np.nan)
        for j, l in enumerate(lags):
            H[l:, j * Z.shape[1]:(j + 1) * Z.shape[1]] = Z[:n - l]
        return H

    def fit(self, Z, order=None):
        from sklearn.mixture import GaussianMixture
        from sklearn.cluster import KMeans
        from sklearn.neighbors import NearestNeighbors
        Zo = Z if order is None else Z[order]        # shuffle control = permute the ORDER before building history
        H = self.hist(Zo); ok = np.isfinite(H).all(1)
        Hs, Zs = H[ok], Zo[ok]
        d = Z.shape[1]
        # marginals
        self.gmm = GaussianMixture(self.K, covariance_type="diag", random_state=self.seed, reg_covar=1e-2, max_iter=200).fit(Z)
        self.mu0, self.C0 = Z.mean(0), np.cov(Z.T) + 1e-3 * np.eye(d)
        self.iC0 = np.linalg.inv(self.C0); self.ld0 = np.linalg.slogdet(self.C0)[1]
        # (a) ridge AR
        lam = 1e-2 * len(Hs)
        A = np.c_[Hs, np.ones(len(Hs))]
        self.B = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Zs)
        R = Zs - A @ self.B
        self.Cr = np.cov(R.T) + 1e-3 * np.eye(d); self.iCr = np.linalg.inv(self.Cr); self.ldr = np.linalg.slogdet(self.Cr)[1]
        self.ar_r2 = float(1 - R.var(0).sum() / Zs.var(0).sum())
        # (b) kNN successor model: neighbours in history space -> successor cloud
        self.nn = NearestNeighbors(n_neighbors=self.knn + 1).fit(Hs); self.Hs, self.Zs = Hs, Zs
        # (c) regime transitions
        self.km = KMeans(self.K, random_state=self.seed, n_init=4).fit(Z)
        c = self.km.predict(Zo); self.T = np.full((self.K, self.K), 0.5)   # additive smoothing 0.5
        for a, b in zip(c[:-2], c[2:]):
            self.T[a, b] += 1
        self.Tcount = self.T - 0.5
        self.P = self.T / self.T.sum(1, keepdims=True)
        return self

    def score(self, Z, exclude_self=False):
        H = self.hist(Z); ok = np.isfinite(H).all(1)
        d = Z.shape[1]; n = len(Z)
        out = {k: np.full(n, np.nan) for k in ("marg_gmm", "marg_gauss", "cond_ar", "cond_knn", "cond_trans", "trans_count")}
        out["marg_gmm"] = -self.gmm.score_samples(Z)
        D0 = Z - self.mu0; out["marg_gauss"] = 0.5 * np.einsum("ij,jk,ik->i", D0, self.iC0, D0) + 0.5 * self.ld0
        A = np.c_[H[ok], np.ones(ok.sum())]; R = Z[ok] - A @ self.B
        out["cond_ar"][ok] = 0.5 * np.einsum("ij,jk,ik->i", R, self.iCr, R) + 0.5 * self.ldr
        dist, idx = self.nn.kneighbors(H[ok])
        if exclude_self:
            idx = idx[:, 1:]
        else:
            idx = idx[:, :self.knn]
        S = self.Zs[idx]                               # (m, k, d) successor clouds
        mu = S.mean(1); Dv = Z[ok] - mu
        var = S.var(1) + 0.05 * np.diag(self.Cr)       # diag local variance, floored
        out["cond_knn"][ok] = 0.5 * (Dv ** 2 / var).sum(1) + 0.5 * np.log(var).sum(1)
        c = self.km.predict(Z)
        prev = np.full(n, -1); prev[2:] = c[:-2]
        m = prev >= 0
        out["cond_trans"][m] = -np.log(self.P[prev[m], c[m]])
        out["trans_count"][m] = self.Tcount[prev[m], c[m]]
        return out


def latent_pca(Ftr, Fte, d=20, seed=0):
    from sklearn.decomposition import PCA
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    pca = PCA(n_components=min(d, Ftr.shape[1]), random_state=seed).fit((Ftr - mu) / sd)
    return pca.transform((Ftr - mu) / sd), pca.transform((Fte - mu) / sd)


def latent_vade(Ftr, Fte, K=24, ld=10, seed=0):
    from models_vade import train_vade
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    v = train_vade(((Ftr - mu) / sd).astype(np.float32), n_clusters=K, latent_dim=ld, epochs=30, pretrain_epochs=20,
                   warmup=8, seed=seed, device="cpu")
    return v._encode_mean(((Ftr - mu) / sd).astype(np.float32)), v._encode_mean(((Fte - mu) / sd).astype(np.float32))


def calib_crossfit(Ztr, cm_factory):
    """Train-normal calibration scores by 2 contiguous folds (fit on one half, score the other)."""
    n = len(Ztr); h = n // 2
    parts = [(np.arange(0, h), np.arange(h, n)), (np.arange(h, n), np.arange(0, h))]
    ref = {}
    for fit_idx, sc_idx in parts:
        cm = cm_factory().fit(Ztr[fit_idx]); s = cm.score(Ztr[sc_idx])
        for k, v in s.items():
            ref.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in ref.items()}


def a10_eval(name, tag, Ztr, Zte, yw, hard, frontier=None, K=24, seed=0, hist_clean=None):
    keys = ("marg_gmm", "marg_gauss", "cond_ar", "cond_knn", "cond_trans")
    fac = lambda: CondModels(K=K, seed=seed)
    ref = calib_crossfit(Ztr, fac)
    cm = fac().fit(Ztr); te = cm.score(Zte)
    ok = np.isfinite(np.stack([te[k] for k in keys])).all(0)
    P = {k: pct_of(ref[k][np.isfinite(ref[k])], te[k]) for k in keys}
    row = dict(item="A10_main", dataset=name, latent=tag, K=K, ar_r2_train=cm.ar_r2,
               n_test=int(ok.sum()), n_anom=int((yw[ok] == 1).sum()), n_hard=int(hard[ok].sum()))
    msk_all = ok; msk_hard = ok & ((yw == 0) | hard)
    for k in keys:
        row[f"{k}_auroc_all"] = auroc(yw[msk_all], te[k][msk_all])
        row[f"{k}_auroc_hard"] = auroc(yw[msk_hard], te[k][msk_hard])
        row[f"{k}_tpr1_hard"] = float((P[k][msk_hard & (yw == 1)] > 0.99).mean()) if (msk_hard & (yw == 1)).any() else float("nan")
        row[f"{k}_fpr1_testnormal"] = float((P[k][ok & (yw == 0)] > 0.99).mean())
    # fused (max percentile of marginal and each conditional)
    for k in ("cond_ar", "cond_knn", "cond_trans"):
        fused = np.maximum(P["marg_gmm"], P[k])
        row[f"fused_gmm+{k}_auroc_hard"] = auroc(yw[msk_hard], fused[msk_hard])
        row[f"fused_gmm+{k}_auroc_all"] = auroc(yw[msk_all], fused[msk_all])
    # the A10 cell: marginally normal (gmm pct < .95) yet conditionally improbable (cond pct > .99)
    mnorm = ok & (P["marg_gmm"] < 0.95)
    cell = {}
    for k in ("cond_ar", "cond_knn", "cond_trans"):
        hi = P[k] > 0.99
        cell[k] = dict(anom_in_cell=int((mnorm & hi & (yw == 1)).sum()), anom_margnormal=int((mnorm & (yw == 1)).sum()),
                       hard_in_cell=int((mnorm & hi & hard).sum()), hard_margnormal=int((mnorm & hard).sum()),
                       normal_in_cell=int((mnorm & hi & (yw == 0)).sum()), normal_margnormal=int((mnorm & (yw == 0)).sum()))
        cell[k]["anom_rate"] = cell[k]["anom_in_cell"] / max(1, cell[k]["anom_margnormal"])
        cell[k]["normal_rate"] = cell[k]["normal_in_cell"] / max(1, cell[k]["normal_margnormal"])
        # AUROC restricted to the marginally-normal region (anomalies vs normals both with marg pct<.95)
        cell[k]["auroc_within_margnormal"] = auroc(yw[mnorm], te[k][mnorm])
        cell[k]["auroc_within_margnormal_marggmm"] = auroc(yw[mnorm], te["marg_gmm"][mnorm])
    row["a10_cell"] = cell
    # calibration-free cell: thresholds at TEST-NORMAL quantiles (marg < q95, cond > q99), comparable
    # between the unshuffled and shuffled fits; plus the same restricted to windows whose history
    # (the 120 samples before the anchor slot) is attack-free, so the flag cannot come from an
    # earlier anomaly contaminating the history.
    def cellT(te_, mask_extra=None):
        nrm = ok & (yw == 0)
        mn = ok & (te_["marg_gmm"] < np.quantile(te_["marg_gmm"][nrm], 0.95))
        if mask_extra is not None:
            mn = mn & mask_extra
        out = {}
        for k in ("cond_ar", "cond_knn", "cond_trans"):
            hi = te_[k] > np.quantile(te_[k][nrm], 0.99)
            out[k] = dict(anom=int((mn & hi & (yw == 1)).sum()), anom_margnormal=int((mn & (yw == 1)).sum()),
                          hard=int((mn & hi & hard).sum()), hard_margnormal=int((mn & hard).sum()),
                          normal=int((mn & hi & (yw == 0)).sum()), normal_margnormal=int((mn & (yw == 0)).sum()))
        return out
    row["a10_cellT"] = cellT(te)
    if hist_clean is not None:
        row["a10_cellT_cleanhist"] = cellT(te, hist_clean)
        mh = ok & ((yw == 0) | (hard & hist_clean))
        row["n_hard_cleanhist"] = int((hard & hist_clean & ok).sum())
        row["auroc_hard_cleanhist"] = {k: auroc(yw[mh], te[k][mh]) for k in keys}
        mn2 = ok & (P["marg_gmm"] < 0.95) & ((yw == 0) | hist_clean)
        row["auroc_within_margnormal_cleanhist"] = {k: auroc(yw[mn2], te[k][mn2]) for k in keys}
    # regime-transition legality: never-seen transitions
    tc = te["trans_count"]; okt = np.isfinite(tc)
    row["never_seen_transition"] = dict(anom_frac=float((tc[okt & (yw == 1)] == 0).mean()) if (okt & (yw == 1)).any() else float("nan"),
                                        hard_frac=float((tc[okt & hard] == 0).mean()) if (okt & hard).any() else float("nan"),
                                        normal_frac=float((tc[okt & (yw == 0)] == 0).mean()),
                                        train_n_transitions_seen=int((cm.Tcount > 0).sum()), K=K)
    # shuffle control (3 seeds)
    sh = []
    for s in range(3):
        order = np.random.default_rng(100 + s).permutation(len(Ztr))
        cms = CondModels(K=K, seed=seed).fit(Ztr, order=order); tes = cms.score(Zte)
        refs = calib_crossfit(Ztr[order], lambda: CondModels(K=K, seed=seed))
        Ps = {k: pct_of(refs[k][np.isfinite(refs[k])], tes[k]) for k in keys}
        r = dict(seed=s, ar_r2_train=cms.ar_r2, cellT=cellT(tes),
                 cellT_cleanhist=(cellT(tes, hist_clean) if hist_clean is not None else None))
        if hist_clean is not None:
            mh = ok & ((yw == 0) | (hard & hist_clean))
            r["auroc_hard_cleanhist"] = {k: auroc(yw[mh], tes[k][mh]) for k in ("cond_ar", "cond_knn", "cond_trans")}
        for k in ("cond_ar", "cond_knn", "cond_trans"):
            r[f"{k}_auroc_hard"] = auroc(yw[msk_hard], tes[k][msk_hard])
            r[f"{k}_auroc_all"] = auroc(yw[msk_all], tes[k][msk_all])
            hi = Ps[k] > 0.99
            r[f"{k}_cell_anom"] = int((mnorm & hi & (yw == 1)).sum()); r[f"{k}_cell_normal"] = int((mnorm & hi & (yw == 0)).sum())
        sh.append(r)
    row["shuffle_control"] = sh
    # frontier windows (WADI)
    if frontier is not None:
        fr = []
        for i in frontier:
            if i < len(yw):
                fr.append(dict(win=int(i), label=int(yw[i]), hard=bool(hard[i]),
                               **{f"{k}_pct": (float(P[k][i]) if np.isfinite(P[k][i]) else None) for k in keys},
                               trans_count=(float(tc[i]) if np.isfinite(tc[i]) else None)))
        row["frontier"] = fr
        fmask = np.zeros(len(yw), bool); fmask[[i for i in frontier if i < len(yw)]] = True
        mk = ok & ((yw == 0) | fmask)
        row["frontier_auroc"] = {k: auroc(fmask[mk].astype(int), te[k][mk]) for k in keys}
        row["frontier_n_above99"] = {k: int((P[k][fmask] > 0.99).sum()) for k in keys}
        row["frontier_n_above95"] = {k: int((P[k][fmask] > 0.95).sum()) for k in keys}
    emit(row)
    np.savez(os.path.join(HERE, f"a9a10_scatter_{name}_{tag}.npz"), y=yw, hard=hard, ok=ok, **{f"pct_{k}": P[k] for k in keys},
             **{f"raw_{k}": te[k] for k in keys}, trans_count=tc)
    return row


def run_a10(name, vade=False, seed=0):
    Xn, Xa, ya, ch = load_raw(name); C = len(ch)
    en, ea = grid_ends(len(Xn)), grid_ends(len(Xa))
    yw = labels_at(ya, ea)
    Fn, Fa = feats_at(Xn, en, W0), feats_at(Xa, ea, W0)
    triv_n, triv_a = np.abs(Fn[:, :C]).max(1), np.abs(Fa[:, :C]).max(1)
    hard = (yw == 1) & (triv_a <= np.quantile(triv_n, 0.99))
    frontier = [16, 17, 18, 19, 20, 21, 237, 238, 360, 361, 362, 544, 545, 546, 547] if name == "WADI_clean" else None
    if frontier is not None:
        assert len(ea) == 575, f"WADI grid mismatch {len(ea)} != 575 (frontier indices assume the paper's grid)"
    # history (the 120 samples before the anchor slot = the two preceding non-overlapping windows) attack-free
    hist_clean = np.array([e - W0 - 120 >= 0 and ya[e - W0 - 120:e - W0].sum() == 0 for e in ea])
    Ztr, Zte = latent_pca(Fn, Fa, d=20, seed=seed)
    a10_eval(name, "pca20", Ztr, Zte, yw, hard, frontier=frontier, seed=seed, hist_clean=hist_clean)
    if vade:
        Ztr, Zte = latent_vade(Fn, Fa, K=24, ld=10, seed=seed)
        a10_eval(name, "vade10", Ztr, Zte, yw, hard, frontier=frontier, seed=seed, hist_clean=hist_clean)


# ============================================================================= synthetic positive control
def synth_stream(n_cycles, illegal_every=0, seed=0, C=8, dwell=5, noise=0.3):
    """Raw stream with 3 regimes on a strict cycle A->B->C->A (dwell = 5 windows each). Each regime is a
    setpoint vector on C channels plus AR(1) noise. Anomaly (history-dependent): an ILLEGAL successor,
    the cycle runs backwards for one step (A->C, C->B or B->A): the window is a perfectly normal regime,
    only its history makes it wrong. Labels mark the first 2 windows after an illegal step."""
    rng = np.random.default_rng(seed)
    S = np.random.default_rng(12345).normal(0, 1.5, (3, C))   # regime setpoints SHARED by train and test
    seq, lab = [], []
    state = 0
    k = 0
    while k < n_cycles * 3:
        seq += [state] * dwell; lab += [0] * dwell
        k += 1
        if illegal_every and k % illegal_every == 0:
            state = (state - 1) % 3                   # backwards = illegal
            seq += [state] * dwell; lab += [1, 1] + [0] * (dwell - 2)
            k += 1
        state = (state + 1) % 3
    T = len(seq) * W0
    X = np.zeros((T, C)); e = np.zeros(C)
    for t in range(T):
        e = 0.8 * e + rng.normal(0, noise, C)
        X[t] = S[seq[t // W0]] + e
    return X.astype(np.float32), np.repeat(np.array(lab), W0).astype(int)


def run_synth():
    Xn, _ = synth_stream(120, 0, seed=1)
    Xa, ya = synth_stream(60, 4, seed=2)
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-8
    Xn, Xa = (Xn - mu) / sd, (Xa - mu) / sd
    en, ea = grid_ends(len(Xn)), grid_ends(len(Xa))
    yw = labels_at(ya, ea)
    Fn, Fa = feats_at(Xn, en, W0), feats_at(Xa, ea, W0)
    C = Xn.shape[1]
    hard = (yw == 1) & (np.abs(Fa[:, :C]).max(1) <= np.quantile(np.abs(Fn[:, :C]).max(1), 0.99))
    Ztr, Zte = latent_pca(Fn, Fa, d=8)
    emit(dict(item="synth_setup", n_train=len(en), n_test=len(ea), n_anom=int(yw.sum()), n_hard=int(hard.sum())))
    a10_eval("SYNTH_cycle", "pca8", Ztr, Zte, yw, hard, K=6)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "synth":
        run_synth()
    else:
        name = sys.argv[2]
        if what == "timescales":
            run_timescales(name)
        elif what == "wsweep":
            run_wsweep(name)
        elif what == "a10":
            run_a10(name, vade="--vade" in sys.argv)
