"""A8-MASKING breadth sweep over every dataset on disk.

A8 (correct formulation): two DISTINCT, non-overlapping regimes lie CLOSE in observation space.
Failure mode = FALSE NEGATIVE: an extreme value of regime A (an anomaly for A) drifts into the
neighbourhood of regime B and a nearest-component / mixture head scores it NORMAL ("is it near ANY
regime?"). Detecting it needs the EXPECTED regime from history (A8 -> A10).

Metric definitions are byte-identical across datasets (only the window grid is per dataset):
  features  : winfeat 'stats' (6/channel) on train-standardised raw streams -> feature-standardised
              on train -> PCA-10 (fit on train) -> full-cov GMM on train, K in {2,3,4,6,8,12,16} by BIC (seed s).
  s_k(x)    : per-component NLL  0.5*(d_k^2 + logdet S_k + D log 2pi)   (no mixing weight)
  s_NC(x)   : min_k s_k(x)      = the nearest-component head
  thresholds: q50, q99 of s_NC over train windows
  home(x)   : argmin_k s_k(x)
  (a) regime proximity : pairwise D_ij = sqrt(dmu' ((S_i+S_j)/2)^-1 dmu); LDA 5-fold CV error between
      members; valley ratio = min mixture density on the segment mu_i->mu_j / lower endpoint density.
      close-distinct pair := D_ij < 2R  AND  valley < 0.5  AND  LDA-CV error < 0.05  (R = q99.5 home
      Mahalanobis radius, ~5 in 10-D; "close" = B lies within twice the normal envelope of A).
  (b) susceptibility (label-free): for each component h, 200 directions u ~ N(0,S_h); probe point
      x = mu_h + R * u/|u|_{S_h}, R = q99.5 of train home-Mahalanobis distance. Probe is VALID iff
      s_h(x) > q99 (home alone would flag it). MASKED iff min_{k!=h} s_k(x) < q99 (another
      component absorbs it). susc_any = pi-weighted masked fraction among valid probes;
      susc_distinct = same but counting only absorbers k with valley(h,k) < 0.5 (a genuinely
      separate regime, not an adjacent slice of the same ridge).
      DIRECTED corridor (the headline label-free number): along the segment mu_h -> mu_k, the fraction
      of points where the home flags (s_h > q99) but k absorbs (s_k <= q99). corridor_distinct =
      pi-weighted mean over h of the largest corridor to a valley-separated k; mass_with_corridor_distinct
      = train mass in components that have at least one such masked corridor.
  (c) anomaly-masking rate (labelled): for each anomaly episode on the window grid, expected regime
      E = majority home of the m=5 label-0 windows preceding the episode. For each anomalous window:
      far_exp := s_E(x) > q99 ; close_other := min_{k!=E} s_k(x) < q50 ; missed_NC := s_NC(x) <= q99.
      masked := far_exp AND close_other  (implies missed_NC).  masking_rate = #masked / #anomalous.
      CONTROL: the same test with the label-free expectation (majority home of the previous 5 windows)
      on anomalous windows (masking_rate_labelfree_anom) and on test-NORMAL windows (switch_rate_normal =
      the natural regime-switch base rate); masking_excess = anomaly rate minus normal rate.
      LAG-MATCHED control for the oracle rate: normal windows compared with the regime a lag earlier, lag
      drawn from the anomalous windows' (t - episode start) distribution; masking_excess_lagmatched.
  (d) context-recovery gain (labelled): label-free expected regime E_t = majority home over the
      previous 5 windows of the same stream; s_ctx(x_t) = s_{E_t}(x_t).  Difficult subset = test
      windows with s_NC <= q99 (NC-normal-looking).  gain = AUROC(s_ctx) - AUROC(s_NC) on the subset.
      An oracle variant uses the pre-episode regime for anomalous windows; the LAG-MATCHED variant
      additionally gives every normal window a frozen expectation from a matched lag earlier (same
      lag distribution), so both classes are scored against an equally stale expectation.

Outputs: a8_masking_allsets.jsonl (one row per (dataset, seed), flushed; resumable), examples rows,
and a8_masking_allsets.log. Nothing in the paper or the shipped model is touched.
Usage (from poc/):  python _diagnostics/a8_masking_allsets.py [name ...]   (default: all)
"""
from __future__ import annotations
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
POC = os.path.dirname(HERE)
sys.path.insert(0, POC); sys.path.insert(0, HERE)
os.chdir(POC)
from winfeat import window_features
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

OUT = os.path.join(HERE, "a8_masking_allsets.jsonl")
CACHE = os.environ.get("A8_CACHE", r"E:\tmp\claude\E--Projects-Backlog-LatAD\6ae0c3f3-0eaf-4beb-8a09-88ce1e618d2a\scratchpad")
KS, NPC, SEEDS, M_HIST, NDIR = (2, 3, 4, 6, 8, 12, 16), 10, (0, 1, 2), 5, 200
LOG2PI = np.log(2 * np.pi)


def _js(o):
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def emit(row):
    row["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=_js) + "\n"); f.flush()
    print("EMIT", json.dumps(row, default=_js)[:300], flush=True)


def done_keys():
    s = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(line); s.add((r.get("kind"), r.get("dataset"), r.get("seed")))
            except Exception:
                pass
    return s


# =============================================================================== loaders
# each returns dict(train=[X_stream...], test=[(X_stream, y_stream)...] or [], W, ST, note, clip)
def _std(train, test, clip):
    Xn = np.concatenate(train, 0)
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-8
    f = lambda X: np.clip((X - mu) / sd, -clip, clip) if clip else (X - mu) / sd
    return [f(X).astype(np.float32) for X in train], [(f(X).astype(np.float32), y) for X, y in test]


def ld_cached(name, W=60, ST=30, clip=None, note=""):
    d = np.load(os.path.join(CACHE, f"raw_{name}.npz"), allow_pickle=True)   # already standardised+clipped
    return dict(train=[d["Xn"]], test=[(d["Xa"], d["ya"].astype(int))], W=W, ST=ST, note=note, nch=d["Xn"].shape[1])


def ld_wadi():  return ld_cached("WADI_clean", note="1 Hz ds10 (=0.1 Hz); W=60 (10 min); 2B_AIT_002_PV dropped; clip 10")
def ld_hai():   return ld_cached("HAI", note="hai-20.07, 1 Hz; train1+2 concat; test1+2 concat; W=60 s")
def ld_swat():  return ld_cached("SWaT_canon", note="Dec-2015 canonical interleaved attack file, ds10; W=60 (10 min)")


def ld_skab():
    import pandas as pd
    from glob import glob
    CH = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure", "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"]
    groups = [sorted(glob("datasets/SKAB/data/%s/*.csv" % g)) for g in ("valve1", "valve2", "other")]
    trf, tef = [], []
    for g in groups:
        trf += g[0::2]; tef += g[1::2]
    train, test = [], []
    for f in trf:
        d = pd.read_csv(f, sep=";"); X = d[CH].to_numpy(np.float64); y = d["anomaly"].to_numpy(float)
        # train = longest all-normal prefix runs (split the stream at anomalous rows)
        idx = np.where(y == 0)[0]
        cuts = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for c in cuts:
            if len(c) >= 40: train.append(X[c])
    for f in tef:
        d = pd.read_csv(f, sep=";"); test.append((d[CH].to_numpy(np.float64), d["anomaly"].to_numpy(int)))
    tr, te = _std(train, test, None)
    return dict(train=tr, test=te, W=20, ST=10, note="1 Hz; alternate files train/test; train = all-normal runs of train files; W=20 s", nch=8)


def ld_metropt():
    d = np.load("datasets/_new/MetroPT3/metropt_data.npz", allow_pickle=True)
    tr, te = _std([d["Xn"]], [(d["Xa"], d["ya"].astype(int))], None)
    return dict(train=tr, test=te, W=60, ST=30, note="1-min resample; W=60 (1 h); labels = 3 published failure windows (segment-level)", nch=15)


def ld_wind():
    d = np.load("datasets/_new/wind_scada/wind_t06.npz", allow_pickle=True)
    tr, te = _std([d["Xn"]], [(d["Xa"], d["ya"].astype(int))], 10.0)
    return dict(train=tr, test=te, W=12, ST=6, note="10-min SCADA T06; W=12 (2 h); anomaly = 72 h before 7 failures; clip 10", nch=79)


def ld_smd(m):
    base = "datasets/_new/OmniAnomaly/ServerMachineDataset"
    Xn = np.loadtxt(f"{base}/train/machine-{m}.txt", delimiter=","); Xa = np.loadtxt(f"{base}/test/machine-{m}.txt", delimiter=",")
    ya = np.loadtxt(f"{base}/test_label/machine-{m}.txt", delimiter=",").astype(int)
    tr, te = _std([Xn], [(Xa, ya)], 10.0)
    return dict(train=tr, test=te, W=60, ST=30, note=f"SMD machine-{m}, 1-min, 38 ch; W=60 (1 h); clip 10", nch=38)


def ld_cranfield():
    from scipy.io import loadmat
    base = "datasets/_new/Cranfield/CUcasestudy/CUcasestudy"
    T = loadmat(f"{base}/Training.mat"); train = [np.asarray(T[k], np.float64) for k in ("T1", "T2", "T3")]
    test = []
    for i in range(1, 7):
        D = loadmat(f"{base}/FaultyCase{i}.mat")
        for k in sorted(x for x in D if x.startswith("Set")):
            X = np.asarray(D[k], np.float64); ev = np.asarray(D["EvoFault" + k[3:]], float).ravel()
            test.append((X, (ev > 0).astype(int)))
    tr, te = _std(train, test, 10.0)
    return dict(train=tr, test=te, W=20, ST=10, note="1 Hz, 24 ch; train T1-T3; test FaultyCase1-6 (label EvoFault>0; cases 1-3 faulty throughout); clip 10", nch=24)


def ld_canbus():
    from canbus_a8_screen import load_canbus_normal
    trips = load_canbus_normal()
    tr, _ = _std(list(trips.values()), [], None)
    return dict(train=tr, test=[], W=6, ST=3, note="OBD-II 19 trips, 8 ch @ 4 s; W=6 (24 s); normal only", nch=8)


def ld_paderborn():
    import eda_real as E
    from a3_screen_smd_pu import _frame_bandpower
    recs = E._raw_paderborn()
    seqs = [(r["cond"], _frame_bandpower(r["sig"], L=2048, nbands=12)) for r in recs]
    tr, _ = _std([s for _, s in seqs], [], None)
    conds = [c for c, _ in seqs]
    return dict(train=tr, test=[], W=10, ST=5, note="healthy bearings K001-K005, 4 operating conditions x 20 runs; vibration_1 64 kHz -> 2048-sample log band-power frames (13 ch); W=10 frames; normal only", nch=13, regime_of_stream=conds)


def ld_ahu():
    import pandas as pd
    d = pd.read_csv("datasets/_new/_future/AHU_HVAC/office_ahu.csv", low_memory=False)
    sens = [c for c in d.columns if c not in ("AHU name", "Time", "labeling")]
    d[sens] = d[sens].apply(pd.to_numeric, errors="coerce")
    ahus = sorted(d["AHU name"].unique())
    train, test = [], []
    for i, a in enumerate(ahus):
        g = d[d["AHU name"] == a].sort_values("Time")
        X = g[sens].fillna(0.0).to_numpy(np.float64); y = (g["labeling"] != "Normal condition").to_numpy(int)
        if i % 2 == 0:
            idx = np.where(y == 0)[0]; cuts = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            for c in cuts:
                if len(c) >= 48: train.append(X[c])
        else:
            test.append((X, y))
    tr, te = _std(train, test, 10.0)
    return dict(train=tr, test=te, W=24, ST=12, note="office AHU x20, hourly, 18 ch (seasonal NaN->0); train = normal runs of even AHUs, test = odd AHUs; W=24 h; rule-derived fault labels", nch=18)


def ld_miim():
    d = np.load("datasets/miim/miim_unified_seed0.npz", allow_pickle=True)
    return dict(train=None, test=None, W=None, ST=None, prewin=(d["x_train"], d["x_test"], d["y_test"].astype(int), d["mode_train"]),
                note="paper's synthetic MIIM PoC (pre-windowed 144-d stats features, 24 ch); test treated as ONE time-ordered stream", nch=24)


def synth_control(kind, seed=0, d=24, n_steps=300000, dwell=1200, W=60, ST=30):
    """Positive control: regimes A (origin), B (close-but-distinct: raw offset 1.8 sd along a random
    all-channel direction u1 -> PCA-10 feature-space separation ~7 (< 2R; PCA+window mixing compresses
    the raw-space separation by ~2x, measured)), C (far). Anomalies = A segments whose mean drifts onto B's centre
    (absorbed by B) or, in the NEGATIVE control, drifts by the same amount along an orthogonal
    direction v where no regime lives (not absorbed). Half the anomaly episodes in the positive
    control are 'absorbed' type, half orthogonal, so masking_rate has a known expected value ~0.5."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(0, 1, (d, 3)))                  # regime offsets spread over ALL channels
    u1, v, u2 = Q[:, 0], Q[:, 1], Q[:, 2]                          # (as real plants shift many correlated channels)
    off = 1.8
    C = {"A": np.zeros(d), "B": off * u1, "C": 6.0 * u2}
    scale = rng.uniform(0.5, 1.5, d)
    def seg(c, L): return c + rng.normal(0, 1, (L, d)) * scale
    def stream(n, anomalies):
        X = np.zeros((n, d)); y = np.zeros(n, int); t = 0; nep = 0
        while t < n:
            L = int(rng.integers(dwell // 2, dwell * 2)); r = ["A", "B", "C"][int(rng.integers(3))]
            c = C[r]
            if anomalies and r == "A" and rng.random() < 0.35:
                nep += 1
                tgt = C["B"] if (kind == "pos" and nep % 2 == 0) else (off * u2)  # absorbed / toward C (empty)
                if kind == "neg": tgt = off * u2
                lam = np.linspace(0, 1, L)[:, None]
                s = seg(c, L) + lam * (tgt - c)                    # linear drift onto target
                yy = (lam.ravel() > 0.5).astype(int)              # anomalous once past halfway
            else:
                s = seg(c, L); yy = np.zeros(L, int)
            X[t:t + L] = s[:n - t]; y[t:t + L] = yy[:n - t]; t += L
        return X, y
    Xn, _ = stream(n_steps, False); Xa, ya = stream(n_steps, True)
    tr, te = _std([Xn], [(Xa, ya)], None)
    return dict(train=tr, test=te, W=W, ST=ST, note=f"synthetic {kind} control: A/B (raw offset {off} sd along an all-channel direction, close-distinct in feature space) + far C; anomalies = A drifting onto B (absorbed) or by the same amount toward far C (empty space, not absorbed)", nch=d)


LOADERS = {
    "WADI": ld_wadi, "HAI": ld_hai, "SWaT": ld_swat, "SKAB": ld_skab, "MetroPT": ld_metropt, "WindSCADA": ld_wind,
    "SMD-1-1": lambda: ld_smd("1-1"), "SMD-1-4": lambda: ld_smd("1-4"), "SMD-2-1": lambda: ld_smd("2-1"), "SMD-3-7": lambda: ld_smd("3-7"),
    "Cranfield": ld_cranfield, "CANbus": ld_canbus, "Paderborn": ld_paderborn, "AHU_office": ld_ahu, "MIIM_synth": ld_miim,
    "SYNTH_pos": lambda: synth_control("pos"), "SYNTH_neg": lambda: synth_control("neg"),
    "SYNTH_pos_slow": lambda: synth_control("pos", n_steps=1200000, dwell=12000),   # rare switching (every ~400 windows)
}


# =============================================================================== windowing
def windows(streams, W, ST, labelled):
    F, S, P, Y = [], [], [], []
    for si, item in enumerate(streams):
        X, y = (item if labelled else (item, None))
        for i in range(0, len(X) - W + 1, ST):
            F.append(window_features(X[i:i + W], "stats")); S.append(si); P.append(i)
            if y is not None:
                fr = y[i:i + W].mean(); Y.append(1 if fr > 0.5 else (0 if fr == 0 else -1))
    F = np.asarray(F, np.float32); F = np.nan_to_num(F)
    return F, np.asarray(S), np.asarray(P), (np.asarray(Y) if labelled else None)


# =============================================================================== model
class Head:
    def __init__(self, Ztr, seed):
        fits = [GaussianMixture(k, covariance_type="full", reg_covar=1e-3, random_state=seed, n_init=1, max_iter=300).fit(Ztr) for k in KS]
        bic = [g.bic(Ztr) for g in fits]; self.bic = dict(zip(KS, [round(float(b), 1) for b in bic]))
        self.g = fits[int(np.argmin(bic))]; self.K = self.g.n_components
        self.mu, self.S, self.pi = self.g.means_, self.g.covariances_, self.g.weights_
        self.Si = np.stack([np.linalg.inv(S) for S in self.S]); self.ld = np.array([np.linalg.slogdet(S)[1] for S in self.S])
    def d2(self, Z):
        D = Z[:, None, :] - self.mu[None]
        return np.einsum("nkd,kde,nke->nk", D, self.Si, D)
    def s(self, Z):                        # per-component NLL (n, K)
        return 0.5 * (self.d2(Z) + self.ld[None] + Z.shape[1] * LOG2PI)
    def mixlogp(self, Z): return self.g.score_samples(Z)


def majority(a):
    v, c = np.unique(a, return_counts=True); return int(v[np.argmax(c)])


def expected_labelfree(home, stream, m=M_HIST):
    """E_t = majority home over the previous m windows of the same stream (-1 if < 2 available)."""
    E = np.full(len(home), -1, int)
    for t in range(len(home)):
        lo = max(0, t - m); h = [home[j] for j in range(lo, t) if stream[j] == stream[t]]
        if len(h) >= 2: E[t] = majority(np.array(h))
    return E


def episodes(y, stream):
    """maximal runs of label != 0 (within a stream) containing at least one label 1 -> (start, end)."""
    ep = []; t = 0; n = len(y)
    while t < n:
        if y[t] != 0:
            j = t
            while j < n and y[j] != 0 and stream[j] == stream[t]: j += 1
            if (y[t:j] == 1).any(): ep.append((t, j))
            t = j
        else: t += 1
    return ep


def auroc(y, s):
    m = np.isfinite(s) & (y >= 0)
    if (y[m] == 1).sum() == 0 or (y[m] == 0).sum() == 0: return float("nan")
    return float(roc_auc_score(y[m], s[m]))


# =============================================================================== metrics
def metric_a(H, Ztr, home, R):
    """pairwise separation / LDA / valley; returns pair table + close-distinct count."""
    K = H.K; pairs = []; valley = np.ones((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            dm = H.mu[i] - H.mu[j]; Sp = 0.5 * (H.S[i] + H.S[j])
            D = float(np.sqrt(dm @ np.linalg.solve(Sp, dm)))
            lam = np.linspace(0, 1, 41)[:, None]; seg = H.mu[j][None] + lam * dm[None]
            lp = H.mixlogp(seg); val = float(np.exp(lp[1:-1].min() - min(lp[0], lp[-1])))
            valley[i, j] = valley[j, i] = val
            mi, mj = home == i, home == j; err = float("nan")
            if mi.sum() >= 20 and mj.sum() >= 20:
                Zp = np.concatenate([Ztr[mi], Ztr[mj]]); yp = np.r_[np.zeros(mi.sum()), np.ones(mj.sum())]
                cv = StratifiedKFold(5, shuffle=True, random_state=0)
                err = float(1 - cross_val_score(LinearDiscriminantAnalysis(), Zp, yp, cv=cv).mean())
            pairs.append(dict(i=i, j=j, D=round(D, 3), valley=round(val, 4), lda_err=round(err, 4) if err == err else None,
                              close_distinct=bool(D < 2 * R and val < 0.5 and err == err and err < 0.05)))
    cd = [p for p in pairs if p["close_distinct"]]
    comps_cd = set([p["i"] for p in cd] + [p["j"] for p in cd])
    return dict(n_close_distinct=len(cd), mass_close_distinct=float(sum(H.pi[k] for k in comps_cd)),
                minD_distinct=min([p["D"] for p in pairs if p["valley"] < 0.5] or [float("nan")]),
                nearest_D_mean=float(np.mean([min(p["D"] for p in pairs if i in (p["i"], p["j"])) for i in range(K)])),
                n_pairs_D_lt2R=sum(p["D"] < 2 * R for p in pairs), n_pairs_valley_lt05=sum(p["valley"] < 0.5 for p in pairs),
                pairs=pairs), valley


def metric_b(H, valley, q99, R, seed):
    rng = np.random.default_rng(1000 + seed); K = H.K
    per = []
    for h in range(K):
        L = np.linalg.cholesky(H.S[h]); U = rng.normal(0, 1, (NDIR, H.mu.shape[1]))
        U = U / np.linalg.norm(U, axis=1, keepdims=True)                     # unit in whitened space
        X = H.mu[h][None] + R * (U @ L.T)                                    # Mahalanobis-R shell of home
        s = H.s(X); valid = s[:, h] > q99
        others = np.delete(np.arange(K), h)
        so = s[:, others]; absorber = others[np.argmin(so, 1)]; masked_any = so.min(1) < q99
        dist_ok = valley[h][absorber] < 0.5
        # distinct: some absorber with valley<0.5 (search all k, not only the closest)
        masked_dist = np.array([(so[n][(valley[h][others] < 0.5)] < q99).any() if (valley[h][others] < 0.5).any() else False for n in range(NDIR)])
        per.append(dict(h=h, pi=float(H.pi[h]), valid=float(valid.mean()),
                        any=float(masked_any[valid].mean()) if valid.any() else float("nan"),
                        distinct=float(masked_dist[valid].mean()) if valid.any() else float("nan")))
    # directed corridor test: along the segment mu_h -> mu_k, fraction of points where the HOME flags
    # (s_h > q99) but component k absorbs (s_k <= q99). A masked corridor exists iff that fraction > 0.
    lam = np.linspace(0, 1, 101)[:, None]; corr = np.zeros((K, K)); corr_dist = np.zeros((K, K))
    for h in range(K):
        for k in range(K):
            if k == h: continue
            X = H.mu[h][None] + lam * (H.mu[k] - H.mu[h])[None]; s = H.s(X)
            corr[h, k] = float(((s[:, h] > q99) & (s[:, k] <= q99)).mean())
            corr_dist[h, k] = corr[h, k] if valley[h, k] < 0.5 else 0.0
    for p in per:
        h = p["h"]; p["corridor_any"] = float(corr[h].max()); p["corridor_distinct"] = float(corr_dist[h].max())
    w = np.array([p["pi"] for p in per]); va = np.array([p["valid"] for p in per])
    def wmean(key):
        x = np.array([p[key] for p in per]); m = np.isfinite(x); return float((x[m] * w[m]).sum() / w[m].sum()) if m.any() else float("nan")
    return dict(susc_any=wmean("any"), susc_distinct=wmean("distinct"), probe_valid=float((va * w).sum()), R=float(R),
                corridor_any=wmean("corridor_any"), corridor_distinct=wmean("corridor_distinct"),
                mass_with_corridor_distinct=float(w[np.array([p["corridor_distinct"] > 0 for p in per])].sum()), per=per)


def metric_c(H, s_te, y, stream, home_te, q50, q99):
    eps = episodes(y, stream)
    rows = []; n_anom = 0; n_masked = 0; n_missed = 0; n_missed_subtle = 0; n_first = 0; n_masked_first = 0; ex = []; lags = []
    for (a, b) in eps:
        pre = [j for j in range(max(0, a - M_HIST), a) if stream[j] == stream[a] and y[j] == 0]
        if len(pre) < 2: continue
        E = majority(home_te[pre])
        for t in range(a, b):
            if y[t] != 1: continue
            sE = s_te[t, E]; so = np.delete(s_te[t], E).min(); snc = s_te[t].min()
            far, close, missed = sE > q99, so < q50, snc <= q99
            masked = far and close
            n_anom += 1; n_masked += masked; n_missed += missed; n_missed_subtle += (missed and not far); lags.append(t - a)
            if t - a < 10: n_first += 1; n_masked_first += masked
            if masked and len(ex) < 8:
                ex.append(dict(t=int(t), stream=int(stream[t]), E=int(E), absorber=int(np.argmin(s_te[t])), s_E=round(float(sE), 2), s_other=round(float(so), 2), s_NC=round(float(snc), 2)))
    # LAG-MATCHED CONTROL for the oracle rate: the oracle expectation is frozen at episode start, so an
    # anomalous window t is compared with the regime (t - a) windows earlier. For each test-NORMAL window
    # draw a lag from the empirical (t - a) distribution of the evaluated anomalous windows and use the
    # majority home of the 5 windows ending that lag before t. This is the regime-switch base rate over
    # the same horizon; masking_excess_lagmatched = masking_rate - this rate.
    rng = np.random.default_rng(7); n_ctl = 0; n_ctl_hit = 0; E_lagnorm = np.full(len(y), -1, int)
    if lags:
        for t in np.where(y == 0)[0]:
            lag = int(rng.choice(lags)); pre = [j for j in range(max(0, t - lag - M_HIST), max(0, t - lag)) if stream[j] == stream[t]]
            if len(pre) < 2: continue
            E = majority(home_te[pre]); n_ctl += 1; E_lagnorm[t] = E
            n_ctl_hit += (s_te[t, E] > q99) and (np.delete(s_te[t], E).min() < q50)
    lag_ctl = n_ctl_hit / n_ctl if n_ctl else float("nan")
    # CONTROL: same far-from-expected AND inside-another-regime test with the LABEL-FREE expectation
    # (majority home of the previous 5 windows) on anomalous windows and on test-normal windows. The
    # normal-window rate is the natural regime-switch base rate; the anomaly excess over it is the signal.
    E_lf = expected_labelfree(home_te, stream); ok = E_lf >= 0
    sE = s_te[np.arange(len(E_lf)), np.maximum(E_lf, 0)]
    so = np.array([np.delete(s_te[t], max(E_lf[t], 0)).min() for t in range(len(E_lf))])
    lf = (sE > q99) & (so < q50) & ok
    lf_anom = float(lf[(y == 1) & ok].mean()) if ((y == 1) & ok).any() else float("nan")
    lf_norm = float(lf[(y == 0) & ok].mean()) if ((y == 0) & ok).any() else float("nan")
    return dict(n_episodes=len(eps), n_anom_eval=n_anom, masking_rate=(n_masked / n_anom if n_anom else float("nan")),
                masking_rate_labelfree_anom=lf_anom, switch_rate_normal=lf_norm, masking_excess=lf_anom - lf_norm,
                switch_rate_normal_lagmatched=lag_ctl, masking_excess_lagmatched=((n_masked / n_anom) - lag_ctl) if n_anom else float("nan"),
                median_lag_windows=(float(np.median(lags)) if lags else float("nan")), E_lagnorm=E_lagnorm,
                nc_miss_rate=(n_missed / n_anom if n_anom else float("nan")),
                masked_share_of_misses=(n_masked / n_missed if n_missed else float("nan")),
                subtle_share_of_misses=(n_missed_subtle / n_missed if n_missed else float("nan")),
                masking_rate_first10=(n_masked_first / n_first if n_first else float("nan")), examples=ex)


def metric_d(H, s_te, y, stream, home_te, q99, oracle_E=None, E_lagnorm=None):
    E = expected_labelfree(home_te, stream)
    ok = E >= 0
    s_nc = s_te.min(1); s_ctx = np.where(ok, s_te[np.arange(len(E)), np.maximum(E, 0)], np.nan)
    diff = (s_nc <= q99) & ok
    out = dict(auroc_nc=auroc(y[ok], s_nc[ok]), auroc_ctx=auroc(y[ok], s_ctx[ok]),
               auroc_nc_diff=auroc(y[diff], s_nc[diff]), auroc_ctx_diff=auroc(y[diff], s_ctx[diff]),
               n_diff=int(diff.sum()), n_diff_anom=int((y[diff] == 1).sum()))
    out["gain_diff"] = out["auroc_ctx_diff"] - out["auroc_nc_diff"]; out["gain_full"] = out["auroc_ctx"] - out["auroc_nc"]
    if oracle_E is not None:
        Eo = np.where(oracle_E >= 0, oracle_E, E); oko = Eo >= 0
        s_o = s_te[np.arange(len(Eo)), np.maximum(Eo, 0)]; d2 = (s_nc <= q99) & oko
        out["auroc_ctx_oracle_diff"] = auroc(y[d2], s_o[d2]); out["gain_diff_oracle"] = out["auroc_ctx_oracle_diff"] - auroc(y[d2], s_nc[d2])
    if E_lagnorm is not None and oracle_E is not None:
        El = np.where(y == 0, E_lagnorm, oracle_E); okl = El >= 0
        s_l = s_te[np.arange(len(El)), np.maximum(El, 0)]; d3 = (s_nc <= q99) & okl
        out["auroc_ctx_lagmatched_diff"] = auroc(y[d3], s_l[d3]); out["auroc_nc_lagmatched_diff"] = auroc(y[d3], s_nc[d3])
        out["gain_diff_lagmatched"] = out["auroc_ctx_lagmatched_diff"] - out["auroc_nc_lagmatched_diff"]
        out["gain_full_lagmatched"] = auroc(y[okl], s_l[okl]) - auroc(y[okl], s_nc[okl])
    return out


def oracle_expected(y, stream, home_te):
    Eo = np.full(len(y), -1, int)
    for (a, b) in episodes(y, stream):
        pre = [j for j in range(max(0, a - M_HIST), a) if stream[j] == stream[a] and y[j] == 0]
        if len(pre) >= 2: Eo[a:b] = majority(home_te[pre])
    return Eo


# =============================================================================== driver
def run(name):
    dk = done_keys(); t0 = time.time()
    if all(("main", name, s) in dk for s in SEEDS):
        print(f"[{name}] cached"); return
    D = LOADERS[name]()
    if D.get("prewin") is not None:
        Ftr, Fte, yte, mode_tr = D["prewin"]; Str, Ste = np.zeros(len(Ftr), int), np.zeros(len(Fte), int); labelled = True
    else:
        labelled = len(D["test"]) > 0
        Ftr, Str, Ptr, _ = windows(D["train"], D["W"], D["ST"], False)
        if labelled: Fte, Ste, Pte, yte = windows(D["test"], D["W"], D["ST"], True)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    Xtr = (Ftr - mu) / sd
    sub = np.random.default_rng(0).choice(len(Xtr), min(len(Xtr), 30000), replace=False)
    pca = PCA(NPC, random_state=0).fit(Xtr[sub]); Ztr = pca.transform(Xtr)
    Zte = pca.transform((Fte - mu) / sd) if labelled else None
    print(f"[{name}] train windows {len(Ztr)}" + (f", test {len(Zte)} (anom {int((yte==1).sum())}, ambiguous {int((yte==-1).sum())})" if labelled else ", normal-only") + f"  ({time.time()-t0:.0f}s)", flush=True)
    for seed in SEEDS:
        if ("main", name, seed) in dk: continue
        H = Head(Ztr[sub] if len(Ztr) > 30000 else Ztr, seed)
        s_tr = H.s(Ztr); home = s_tr.argmin(1); snc = s_tr.min(1)
        q50, q99 = np.quantile(snc, 0.5), np.quantile(snc, 0.99)
        dh = np.sqrt(H.d2(Ztr)[np.arange(len(Ztr)), home]); R = float(np.quantile(dh, 0.995))
        A, valley = metric_a(H, Ztr, home, R)
        B = metric_b(H, valley, q99, R, seed)
        row = dict(kind="main", dataset=name, seed=seed, K=H.K, bic=H.bic, labelled=labelled, n_train=int(len(Ztr)), nch=D["nch"], W=D["W"], ST=D["ST"], note=D["note"],
                   pca_var=float(pca.explained_variance_ratio_.sum()), q50=float(q50), q99=float(q99), R=R,
                   **{f"a_{k}": v for k, v in A.items() if k != "pairs"}, **{f"b_{k}": v for k, v in B.items() if k != "per"})
        if labelled:
            s_te = H.s(Zte); home_te = s_te.argmin(1)
            Cm = metric_c(H, s_te, yte, Ste, home_te, q50, q99)
            Dm = metric_d(H, s_te, yte, Ste, home_te, q99, oracle_expected(yte, Ste, home_te), Cm["E_lagnorm"])
            row.update(n_test=int(len(Zte)), n_anom=int((yte == 1).sum()), **{f"c_{k}": v for k, v in Cm.items() if k not in ("examples", "E_lagnorm")}, **{f"d_{k}": v for k, v in Dm.items()})
            emit(dict(kind="examples", dataset=name, seed=seed, masked_examples=Cm["examples"]))
        if "regime_of_stream" in D:   # labelled regimes (normal-only): true-regime separation
            reg = np.array(D["regime_of_stream"])[Str]; names = sorted(set(reg)); tr_pairs = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    Zi, Zj = Ztr[reg == names[i]], Ztr[reg == names[j]]; dm = Zi.mean(0) - Zj.mean(0); Sp = 0.5 * (np.cov(Zi.T) + np.cov(Zj.T))
                    Zp = np.concatenate([Zi, Zj]); yp = np.r_[np.zeros(len(Zi)), np.ones(len(Zj))]
                    err = float(1 - cross_val_score(LinearDiscriminantAnalysis(), Zp, yp, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean())
                    tr_pairs.append(dict(a=names[i], b=names[j], D=round(float(np.sqrt(dm @ np.linalg.solve(Sp, dm))), 3), lda_err=round(err, 4)))
            row["true_regime_pairs"] = tr_pairs
        emit(dict(kind="pairs", dataset=name, seed=seed, pairs=A["pairs"], per_component_susc=B["per"]))
        emit(row)
    print(f"[{name}] done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or list(LOADERS)
    for n in names:
        try:
            run(n)
        except Exception as e:
            import traceback; traceback.print_exc()
            emit(dict(kind="error", dataset=n, seed=None, error=repr(e)))
