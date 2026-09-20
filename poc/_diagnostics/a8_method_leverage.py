"""A8 method leverage: two candidate heads derived from the A8 findings, prototyped and tested on the
CLEAN pipeline (clean_recompute.md: FIX 1-3; WADI_clean difficult = 30 windows, HAI 167, SWaT_canon 85).

  Lever B (over-coverage)      : LOCAL-DENSITY / kNN head = distance to the k-th nearest TRAIN-normal window
                                 (VaDE latent, raw standardized feature space, PCA-20 of features). An isolated
                                 point has a large kNN distance whatever a wide mixture component says.
  Lever A (proximity masking)  : EXPECTED-REGIME (context) head = distance of the window to the regime the
                                 PREVIOUS window was in, in units of that regime's own train q99; plus the
                                 transition-weighted mixture NLL. Regimes = the VaDE's own components (model
                                 latent) and, as a model-free replicate, a PCA-10 / GMM-16 partition.
                                 LABEL-FREE rules only for the method (prev1 = previous window; selfgate = last
                                 window the detector itself scored normal); the label-using 'oracle' rule of the
                                 A8 study is kept ONLY as a reference upper bound and flagged as such.
  Controls                     : switch indicator (regime != previous regime), random-regime expectation,
                                 time-shuffled context. A context lift that these reproduce is the regime-switch
                                 base rate, not context.
  Gates (train-normal only)    : for every head the residual-head criterion (held-out 20% train q95 / in-sample
                                 q95 < 1.5, no new constant), plus diagnostics (over-coverage share, Spearman
                                 redundancy with the density head).

Stage `fit`  : python _diagnostics/a8_method_leverage.py fit <WADI_clean|HAI|SWaT_canon> [seeds]
               refits the shipped recipe (build_scores_table.py: K/latent per CFG, 40 epochs, warmup 8, FIX-2
               clip), checks the refit against the stored scores_<ds>.npz LatAD (corr, AUROC), computes every
               head on train and test, saves _diagnostics/a8_leverage_<ds>_seed<s>.npz (resumable: skips seeds
               whose npz exists) and a JSON line per seed in a8_method_leverage.jsonl.
Stage `eval` : python _diagnostics/a8_method_leverage.py eval [datasets]
               fuses each head (robust median/IQR z from TRAIN-normal, unit weight, fixed a priori) into
               LatAD, HC_coh and the HCcoh+LatAD headline (ensemble_final.py, experts_full), 5 seeds, clean
               Difficult / DoubleHard / Easy / All AUROC, paired episode bootstrap vs the headline,
               writes a8_method_leverage_eval.json.
Paper and shipped model files untouched.
"""
from __future__ import annotations
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
POC = os.path.dirname(HERE)
sys.path.insert(0, POC)
import eda_real as E
from models_vade import train_vade

CFG = {"WADI_clean": (20, 10), "HAI": (40, 16), "SWaT_canon": (40, 16)}
KS = [1, 5, 10, 20, 50]                     # kNN orders (k=10 is the pre-declared primary; others = sensitivity)
KNN_PRIMARY = 10
LOG = os.path.join(HERE, "a8_method_leverage.jsonl")
Q95 = 0.95


def _js(o):
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def emit(kind, **row):
    row = dict(kind=kind, ts=time.strftime("%H:%M:%S"), **row)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=_js) + "\n"); f.flush()
    print(f"[{kind}] " + json.dumps(row, default=_js)[:600], flush=True)


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float); m = np.isfinite(s)
    if (y[m] == 1).sum() == 0 or (y[m] == 0).sum() == 0: return float("nan")
    return float(roc_auc_score(y[m], s[m]))


# ----------------------------------------------------------------------------- kNN head
def knn_head(Ztr, Zte, ks=KS, excl=2):
    """Distance to the k-th nearest train window. Train reference is a TEMPORAL leave-out: neighbours within
    +-excl windows in time are excluded (overlapping windows would otherwise make every train window its own
    neighbour and under-state the normal kNN scale relative to test)."""
    kmax = max(ks)
    nn = NearestNeighbors(n_neighbors=kmax + 2 * excl + 1, algorithm="brute", n_jobs=4).fit(Ztr)
    dtr, itr = nn.kneighbors(Ztr)
    n = len(Ztr); rows = np.arange(n)[:, None]
    keep = np.abs(itr - rows) > excl
    out_tr = np.full((n, len(ks)), np.nan)
    for i in range(n):
        d = dtr[i][keep[i]]
        for j, k in enumerate(ks):
            out_tr[i, j] = d[k - 1] if len(d) >= k else d[-1]
    dte, _ = nn.kneighbors(Zte, n_neighbors=kmax)
    out_te = np.stack([dte[:, k - 1] for k in ks], 1)
    return out_tr, out_te


def knn_gate(Ztr, k=KNN_PRIMARY, excl=2):
    """Residual-head criterion transplanted: fit on the first 80% of train, score the held-out last 20%;
    ratio = q95(held-out) / q95(in-sample temporal leave-out). < 1.5 -> the head generalises to unseen normal."""
    nA = int(0.8 * len(Ztr)); A, B = Ztr[:nA], Ztr[nA:]
    trA, _ = knn_head(A, A[:1], ks=[k], excl=excl)
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(A)
    dB = nn.kneighbors(B)[0][:, k - 1]
    return float(np.quantile(dB, Q95) / (np.quantile(trA[:, 0], Q95) + 1e-9))


# ----------------------------------------------------------------------------- regimes
class DiagRegimes:
    """The VaDE's OWN mixture components (diagonal), scale-free via the q99 of each component's train members."""
    def __init__(self, mu, var, Ztr, lab_tr, min_n=20):
        self.mu, self.var = mu, var; self.K = len(mu)
        D = self.dist(Ztr); own = D[np.arange(len(Ztr)), lab_tr]
        pooled99, pooled50 = np.quantile(own, .99), np.quantile(own, .50)
        self.q99 = np.array([np.quantile(D[lab_tr == k, k], .99) if (lab_tr == k).sum() >= min_n else pooled99 for k in range(self.K)])
        self.q50 = np.array([np.quantile(D[lab_tr == k, k], .50) if (lab_tr == k).sum() >= min_n else pooled50 for k in range(self.K)])
        self.logdet = np.log(var).sum(1)

    def dist(self, Z):
        return np.sqrt(((Z[:, None, :] - self.mu[None]) ** 2 / self.var[None]).sum(2))

    def nll(self, Z):
        return 0.5 * self.dist(Z) ** 2 + 0.5 * self.logdet[None] + 0.5 * Z.shape[1] * np.log(2 * np.pi)


class FullRegimes:
    """Model-free replicate: Ledoit-Wolf full-covariance Gaussians per GMM label (as in a8_masking.py)."""
    def __init__(self, Z, lab, min_n=20):
        ks = [k for k in np.unique(lab) if (lab == k).sum() >= min_n]
        self.K = len(ks); self.mu, self.prec, self.logdet = [], [], []
        for k in ks:
            lw = LedoitWolf().fit(Z[lab == k]); self.mu.append(lw.location_); self.prec.append(lw.precision_)
            self.logdet.append(float(np.linalg.slogdet(lw.covariance_)[1]))
        self.mu = np.array(self.mu); self.logdet = np.array(self.logdet)
        D = self.dist(Z); own = D.argmin(1); self.lab_tr = own
        self.q99 = np.array([np.quantile(D[own == i, i], .99) if (own == i).sum() >= 5 else np.quantile(D[np.arange(len(Z)), own], .99) for i in range(self.K)])
        self.q50 = np.array([np.quantile(D[own == i, i], .50) if (own == i).sum() >= 5 else np.quantile(D[np.arange(len(Z)), own], .50) for i in range(self.K)])

    def dist(self, Z):
        D = np.empty((len(Z), self.K))
        for i in range(self.K):
            r = Z - self.mu[i]; D[:, i] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", r, self.prec[i], r), 0))
        return D

    def nll(self, Z):
        return 0.5 * self.dist(Z) ** 2 + 0.5 * self.logdet[None] + 0.5 * Z.shape[1] * np.log(2 * np.pi)


def transition_matrix(lab, K, alpha=0.5):
    T = np.full((K, K), alpha)
    for a, b in zip(lab[:-1], lab[1:]): T[a, b] += 1
    return T / T.sum(1, keepdims=True)


def context_scores(R, Ztr, Zte, lab_tr, lab_te, latad_te, latad_thr, y, seed):
    """All context scores on test (+ train calibration with the prev1 rule). Higher = more anomalous.
    Returns dict name -> (train_ref, test)."""
    Dtr, Dte = R.dist(Ztr), R.dist(Zte)
    rel_tr, rel_te = Dtr / R.q99[None], Dte / R.q99[None]
    ntr, nte = len(Ztr), len(Zte); atr, ate = np.arange(ntr), np.arange(nte)
    NLLtr, NLLte = R.nll(Ztr), R.nll(Zte)
    T = transition_matrix(lab_tr, R.K); logT = np.log(T)
    rng = np.random.default_rng(seed)
    # expectations on train: previous train window (contiguous stream)
    ptr = np.r_[lab_tr[0], lab_tr[:-1]]
    # expectations on test (label-free unless noted)
    prev1 = np.r_[lab_te[0], lab_te[:-1]]
    selfg = np.empty(nte, int); last = lab_te[0]
    for t in range(nte):                                   # last window the detector itself called normal
        selfg[t] = last
        if latad_te[t] <= latad_thr: last = lab_te[t]
    oracle = np.empty(nte, int); last = lab_te[0]          # REFERENCE ONLY: uses test labels
    for t in range(nte):
        oracle[t] = last
        if y[t] == 0: last = lab_te[t]
    rnd = rng.choice(R.K, size=nte, p=np.bincount(lab_tr, minlength=R.K) / ntr)      # control: random regime
    shuf = lab_te[rng.permutation(nte)]                                                 # control: shuffled context
    out = {}
    s_exp_tr = rel_tr[atr, ptr]; s_tr_tr = -logsumexp(logT[ptr] - NLLtr, 1)
    out["near"] = (rel_tr[atr, lab_tr], rel_te[ate, lab_te])
    for nm, ex in (("prev1", prev1), ("selfgate", selfg), ("oracle", oracle), ("ctl_rand", rnd), ("ctl_shuf", shuf)):
        out[f"exp_{nm}"] = (s_exp_tr, rel_te[ate, ex])
        out[f"trans_{nm}"] = (s_tr_tr, -logsumexp(logT[ex] - NLLte, 1))
        out[f"switch_{nm}"] = ((lab_tr != ptr).astype(float), (lab_te != ex).astype(float))
    # max(near, expected) as in the A8 study (percentile-fused) is a nonlinear rule; we keep the plain heads
    # and let the eval fuse them additively with the headline.
    stats = dict(K=int(R.K), normal_switch_rate_test=float((lab_te != prev1)[y == 0].mean()),
                 train_switch_rate=float((lab_tr != ptr).mean()),
                 anom_switch_rate_test=float((lab_te != prev1)[y == 1].mean()),
                 frac_train_beyond_own_q99=float((rel_tr[atr, lab_tr] > 1).mean()))
    return out, stats


def ctx_gate(R_cls, Ztr, lab_tr, seed, **kw):
    """Residual criterion for the expected-regime head: calibrate regimes on the first 80% of train, score the
    last 20% with the prev1 rule; ratio of q95s."""
    nA = int(0.8 * len(Ztr)); A, B = Ztr[:nA], Ztr[nA:]
    if R_cls is DiagRegimes:
        R = DiagRegimes(kw["mu"], kw["var"], A, lab_tr[:nA]); labA, labB = lab_tr[:nA], lab_tr[nA:]
    else:
        R = FullRegimes(A, lab_tr[:nA]); labA = R.lab_tr; labB = R.dist(B).argmin(1)
    relA, relB = R.dist(A) / R.q99[None], R.dist(B) / R.q99[None]
    sA = relA[np.arange(len(A)), np.r_[labA[0], labA[:-1]]]; sB = relB[np.arange(len(B)), np.r_[labB[0], labB[:-1]]]
    T = transition_matrix(labA, R.K); logT = np.log(T)
    tA = -logsumexp(logT[np.r_[labA[0], labA[:-1]]] - R.nll(A), 1); tB = -logsumexp(logT[np.r_[labB[0], labB[:-1]]] - R.nll(B), 1)
    r = lambda a, b: float(np.quantile(b, Q95) / (np.quantile(a, Q95) + 1e-9))
    return dict(exp_ratio=r(sA, sB), trans_ratio=r(tA, tB))


# ----------------------------------------------------------------------------- fit stage
def fit(name, seeds):
    t0 = time.time()
    D = E.load(name)
    Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"].astype(int)
    clipv = E.CLIP.get(name)
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr, Xte = (Xtr0 - mu) / sig, (Xte0 - mu) / sig
    if clipv: Xtr, Xte = np.clip(Xtr, -clipv, clipv), np.clip(Xte, -clipv, clipv)
    Xtr, Xte = Xtr.astype(np.float32), Xte.astype(np.float32)
    K, LD = CFG[name]; kd = min(80, max(20, len(Xtr0) // 10))
    shipped = np.load(os.path.join(HERE, f"scores_{name}.npz"))
    assert np.array_equal(shipped["label"].astype(int), y), "label mismatch vs scores npz"
    emit("setup", dataset=name, n_train=len(Xtr), n_test=len(Xte), n_anom=int(y.sum()), K=K, latent=LD, kd=kd, clip=clipv,
         load_secs=round(time.time() - t0, 1))
    # ---- seed-independent feature-space kNN heads (deterministic) ----
    ffile = os.path.join(HERE, f"a8_leverage_{name}_feat.npz")
    if not os.path.exists(ffile):
        t1 = time.time()
        kf_tr, kf_te = knn_head(Xtr, Xte)
        pca = PCA(20, random_state=0).fit(Xtr); Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)
        kp_tr, kp_te = knn_head(Ptr, Pte)
        gates = dict(knn_feat=knn_gate(Xtr), knn_pca20=knn_gate(Ptr))
        np.savez(ffile, knn_feat_tr=kf_tr, knn_feat_te=kf_te, knn_pca20_tr=kp_tr, knn_pca20_te=kp_te, ks=np.array(KS),
                 gate_knn_feat=gates["knn_feat"], gate_knn_pca20=gates["knn_pca20"])
        jk = KS.index(KNN_PRIMARY)
        emit("feat_knn", dataset=name, secs=round(time.time() - t1, 1), gates=gates,
             auroc_all_feat=auroc(y, kf_te[:, jk]), auroc_all_pca20=auroc(y, kp_te[:, jk]))
    for sd in seeds:
        out = os.path.join(HERE, f"a8_leverage_{name}_seed{sd}.npz")
        if os.path.exists(out):
            print(f"  seed {sd} exists, skip", flush=True); continue
        t1 = time.time()
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
        v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd); v.fit_resid_head(Xtr); v.fit_basin_head(Xtr)
        latad_te = np.asarray(v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto"))
        latad_tr = np.asarray(v.anomaly_score_hard(Xtr, use_resid="auto", use_basin="auto"))
        dens_te, near_te = v._hard_components(Xte); dens_tr, near_tr = v._hard_components(Xtr)
        Ztr, Zte = v._encode_mean(Xtr), v._encode_mean(Xte)
        Gtr, Gte = v._responsibilities(Xtr), v._responsibilities(Xte); lab_tr, lab_te = Gtr.argmax(1), Gte.argmax(1)
        with torch.no_grad():
            cmu = v.mu_c.cpu().numpy().astype(float); cvar = torch.exp(v._lvc()).cpu().numpy().astype(float)
        si = list(shipped["seeds"]).index(sd)
        s0 = shipped["LatAD"][si]
        chk = dict(corr_with_shipped=float(np.corrcoef(s0, latad_te)[0, 1]), spearman=float(spearmanr(s0, latad_te)[0]),
                   auroc_shipped=auroc(y, s0), auroc_refit=auroc(y, latad_te))
        # ---- lever B: kNN in the latent ----
        kl_tr, kl_te = knn_head(Ztr, Zte)
        g_knn_lat = knn_gate(Ztr)
        jk = KS.index(KNN_PRIMARY)
        thr_tr = np.quantile(latad_tr, Q95)
        # over-coverage diagnostics on TRAIN-normal: covered by the density head (below median NLL) yet isolated (kNN > q90)
        ovc = float(((dens_tr <= np.median(dens_tr)) & (kl_tr[:, jk] > np.quantile(kl_tr[:, jk], .90))).mean())
        rho = float(spearmanr(dens_tr, kl_tr[:, jk])[0])
        # ---- lever A: context heads on the VaDE's own components ----
        Rv = DiagRegimes(cmu, cvar, Ztr, lab_tr)
        ctx_v, st_v = context_scores(Rv, Ztr, Zte, lab_tr, lab_te, latad_te, thr_tr, y, sd)
        g_v = ctx_gate(DiagRegimes, Ztr, lab_tr, sd, mu=cmu, var=cvar)
        # ---- lever A replicate: model-free PCA-10 / GMM-16 partition (seeded like the VaDE) ----
        pca = PCA(10, random_state=sd).fit(Xtr); Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)
        gm = GaussianMixture(16, covariance_type="full", random_state=sd, reg_covar=1e-3, n_init=2).fit(Ptr)
        Ro = FullRegimes(Ptr, gm.predict(Ptr)); olab_tr = Ro.lab_tr; olab_te = Ro.dist(Pte).argmin(1)
        ctx_o, st_o = context_scores(Ro, Ptr, Pte, olab_tr, olab_te, latad_te, thr_tr, y, sd)
        g_o = ctx_gate(FullRegimes, Ptr, gm.predict(Ptr), sd)
        arrays = dict(y=y, latad_tr=latad_tr, latad_te=latad_te, dens_tr=dens_tr, dens_te=dens_te, near_tr=near_tr, near_te=near_te,
                      knn_lat_tr=kl_tr, knn_lat_te=kl_te, lab_tr=lab_tr, lab_te=lab_te, olab_tr=olab_tr, olab_te=olab_te)
        for pre, ctx in (("vade", ctx_v), ("obs", ctx_o)):
            for k_, (a, b) in ctx.items():
                arrays[f"{pre}_{k_}_tr"] = a; arrays[f"{pre}_{k_}_te"] = b
        np.savez(out, **arrays)
        emit("seed", dataset=name, seed=sd, secs=round(time.time() - t1, 1), resid_auto=bool(v._resid_auto), basin_lam=float(v._basin_lam),
             check_vs_shipped=chk, n_components_used=int(len(np.unique(lab_tr))),
             gates=dict(knn_lat=g_knn_lat, vade_exp=g_v["exp_ratio"], vade_trans=g_v["trans_ratio"], obs_exp=g_o["exp_ratio"], obs_trans=g_o["trans_ratio"],
                        resid_gen_ratio=float(v._resid_gen_ratio)),
             overcoverage_train=ovc, spearman_dens_knn_train=rho, ctx_vade=st_v, ctx_obs=st_o,
             auroc_all=dict(latad=auroc(y, latad_te), knn_lat10=auroc(y, kl_te[:, jk]), vade_exp_prev1=auroc(y, ctx_v["exp_prev1"][1]),
                            obs_exp_prev1=auroc(y, ctx_o["exp_prev1"][1])))
    emit("done", dataset=name, secs=round(time.time() - t0, 1))


# ----------------------------------------------------------------------------- eval stage
def rz(s, ref, clip=20.0):
    """Robust z: median / IQR of the TRAIN-normal reference, clipped. (A plain z-sum is degenerate on WADI: a few
    clipped-channel test windows give a std of 1e12.)"""
    med = np.median(ref); iqr = (np.quantile(ref, .75) - np.quantile(ref, .25)) / 1.349 + 1e-9
    return np.clip((s - med) / iqr, -clip, clip)


def evaluate(names, reps=1000):
    import ensemble_final as EF
    from onehot_filter import build_feats, loco_residual
    jf = os.path.join(HERE, os.environ.get("EVAL_JSON", "a8_method_leverage_eval.json"))
    RES = json.load(open(jf)) if os.path.exists(jf) else {}        # merge across separate eval runs
    for name in names:
        t0 = time.time()
        d = np.load(os.path.join(HERE, f"scores_{name}.npz"))
        Ex = np.load(os.path.join(POC, "sota_bundle", "experts_full", f"expert_{name}.npz"), allow_pickle=True)
        y = d["label"].astype(int); nseed = int(os.environ.get("NSEED", "5"))   # NSEED<5 only for smoke tests
        Cal, Tst = Ex["calib_surprise"], Ex["test_surprise"]; S = Tst.shape[1]
        w = Ex["comm_cohesion"].astype(float) * np.sqrt(Ex["comm_size"].astype(float))
        lat, lat_tr = d["LatAD"], d["LatAD_train"]
        # subsets exactly as ensemble_final.run
        sub_f = os.path.join(HERE, f"a8_leverage_subsets_{name}.npz")
        if os.path.exists(sub_f):
            sb = np.load(sub_f); diff, dhard, easy = sb["diff"], sb["dhard"], sb["easy"]
        else:
            fn, W, stride = E.RAW[name]; Dd = E.load(name)
            Fn, Fa, grp = build_feats(np.asarray(Dd["Xn_raw"], float), np.asarray(Dd["Xa_raw"], float), W, stride, onehot=True)
            r_tr, r_te = loco_residual(Fn, Fa[:len(y)], grp); lin_thr = float(np.quantile(r_tr, 0.99))
            mthr = float(d["maxz_thr"]); maxz = d["maxz"]
            easy = (y == 1) & (maxz > mthr); diff = (y == 1) & (maxz <= mthr); dhard = diff & (r_te <= lin_thr)
            np.savez(sub_f, diff=diff, dhard=dhard, easy=easy)
        subsets = {"All": y == 1, "Easy": easy, "Difficult": diff, "DoubleHard": dhard}
        print(f"{name}: subsets " + str({k: int(v.sum()) for k, v in subsets.items()}), flush=True)
        # headline components per seed (ensemble_final FIX-1 construction, train-normal calibrated)
        zh, zl, hcc = [], [], []
        for sd in range(nseed):
            P = np.stack([EF.pval(Cal[sd, g], Tst[sd, g]) for g in range(S)]); hc_coh = EF.HC(P, wt=w)
            Pc = np.stack([EF.pval(Cal[sd, g], Cal[sd, g]) for g in range(S)]); hccoh_c = EF.HC(Pc, wt=w)
            z = lambda s, r: (s - r.mean()) / (r.std() + 1e-9)
            nulltail = EF.surv(lat_tr[sd], lat[sd]); zl.append(z(nulltail, EF.surv(lat_tr[sd], lat_tr[sd])))
            zh.append(z(hc_coh, hccoh_c)); hcc.append(hc_coh)
        zh, zl, hcc = np.stack(zh), np.stack(zl), np.stack(hcc)
        headline = zh + zl
        base = {"LatAD": lat, "HC_coh": hcc, "HCcoh+LatAD": headline}
        # train-side reference of the headline (Cal slice = last 20% of train windows, as in modal_experts.py)
        ncal = Cal.shape[2]; assert ncal == len(lat_tr[0]) - len(lat_tr[0]) * 4 // 5, "Cal slice is not the last 20% of train"
        headline_tr, hcc_tr = [], []
        for sd in range(nseed):
            Pc = np.stack([EF.pval(Cal[sd, g], Cal[sd, g]) for g in range(S)]); hccoh_c = EF.HC(Pc, wt=w)
            st = EF.surv(lat_tr[sd], lat_tr[sd]); zl_tr = (st - st.mean()) / (st.std() + 1e-9)
            headline_tr.append((hccoh_c - hccoh_c.mean()) / (hccoh_c.std() + 1e-9) + zl_tr[-ncal:]); hcc_tr.append(hccoh_c)
        headline_tr, hcc_tr = np.stack(headline_tr), np.stack(hcc_tr)
        pct = lambda s, ref: np.searchsorted(np.sort(ref), s, side="right") / len(ref)
        # heads per seed
        ff = np.load(os.path.join(HERE, f"a8_leverage_{name}_feat.npz")); jk = KS.index(KNN_PRIMARY)
        heads = {}      # name -> (nseed, n) robust-z test scores
        refit_lat = []
        gates = {}
        for sd in range(nseed):
            a = np.load(os.path.join(HERE, f"a8_leverage_{name}_seed{sd}.npz"))
            refit_lat.append(a["latad_te"])
            def add(nm, tr, te):
                heads.setdefault(nm, []).append(rz(te, tr))
            for j, k in enumerate(KS):
                add(f"knn_lat_k{k}", a["knn_lat_tr"][:, j], a["knn_lat_te"][:, j])
                if sd == 0:
                    pass
            add("knn_feat_k10", ff["knn_feat_tr"][:, jk], ff["knn_feat_te"][:, jk])
            add("knn_pca20_k10", ff["knn_pca20_tr"][:, jk], ff["knn_pca20_te"][:, jk])
            for pre in ("vade", "obs"):
                for rule in ("near", "exp_prev1", "exp_selfgate", "exp_oracle", "exp_ctl_rand", "exp_ctl_shuf",
                             "trans_prev1", "trans_selfgate", "trans_oracle", "trans_ctl_rand", "trans_ctl_shuf",
                             "switch_prev1", "switch_oracle", "switch_ctl_rand"):
                    add(f"{pre}_{rule}", a[f"{pre}_{rule}_tr"], a[f"{pre}_{rule}_te"])
            # combined lever: kNN + context (both z), the "single architecture" candidate
        heads = {k: np.stack(v) for k, v in heads.items()}
        heads["knn_lat_k10+vade_exp_prev1"] = heads["knn_lat_k10"] + heads["vade_exp_prev1"]
        heads["knn_lat_k10+obs_exp_prev1"] = heads["knn_lat_k10"] + heads["obs_exp_prev1"]
        refit_lat = np.stack(refit_lat)
        # raw (un-z'd) train/test head arrays for the percentile-max fusion of the A8 study
        raw_heads = {}
        for sd in range(nseed):
            a = np.load(os.path.join(HERE, f"a8_leverage_{name}_seed{sd}.npz"))
            raw_heads.setdefault("knn_lat_k10", []).append((a["knn_lat_tr"][:, jk], a["knn_lat_te"][:, jk]))
            raw_heads.setdefault("knn_feat_k10", []).append((ff["knn_feat_tr"][:, jk], ff["knn_feat_te"][:, jk]))
            for pre in ("vade", "obs"):
                for rule in ("exp_prev1", "exp_selfgate", "exp_oracle", "trans_prev1", "exp_ctl_shuf"):
                    raw_heads.setdefault(f"{pre}_{rule}", []).append((a[f"{pre}_{rule}_tr"], a[f"{pre}_{rule}_te"]))
        # gate summaries from the jsonl
        gl = [json.loads(l) for l in open(LOG, encoding="utf-8") if l.strip()]
        gs = [r for r in gl if r["kind"] == "seed" and r["dataset"] == name]
        gate_keys = ["knn_lat", "vade_exp", "vade_trans", "obs_exp", "obs_trans", "resid_gen_ratio"]
        gates = {k: [r["gates"][k] for r in gs[-nseed:]] for k in gate_keys}
        gates["knn_feat"] = float(ff["gate_knn_feat"]); gates["knn_pca20"] = float(ff["gate_knn_pca20"])
        diag = {k: [r[k] for r in gs[-nseed:]] for k in ("overcoverage_train", "spearman_dens_knn_train")}
        diag["ctx_vade"] = [r["ctx_vade"] for r in gs[-nseed:]]; diag["ctx_obs"] = [r["ctx_obs"] for r in gs[-nseed:]]
        diag["refit_check"] = [r["check_vs_shipped"] for r in gs[-nseed:]]
        # ---- evaluation ----
        fn, W, stride = E.RAW[name]; L = int(np.ceil(W / stride)) + 1
        def au(arr, mk):
            keep = np.arange(len(y)) if mk is None else np.where((y == 0) | mk)[0]
            v = [roc_auc_score(y[keep], arr[i][keep]) for i in range(arr.shape[0])]
            return round(float(np.mean(v)), 4), round(float(np.std(v)), 4)
        rows = {}
        def rec(label, arr):
            rows[label] = {s: au(arr, None if s == "All" else mk) for s, mk in subsets.items()}
        for b, arr in base.items(): rec(b, arr)
        rec("LatAD_refit", refit_lat)
        for h, arr in heads.items():
            rec(f"head:{h}", arr)
            rec(f"LatAD+{h}", np.stack([rz(lat[i], lat_tr[i]) for i in range(nseed)]) + arr)
            rec(f"HCcoh+{h}", zh + arr)
            rec(f"HCcoh+LatAD+{h}", headline + arr)
        # nonlinear fusions of the A8 study (max of robust-z; max of train-percentiles), headline and HC_coh bases
        for h, lst in raw_heads.items():
            zmax_hl = np.stack([np.maximum(rz(headline[i], headline_tr[i]), rz(te, tr)) for i, (tr, te) in enumerate(lst)])
            pmax_hl = np.stack([np.maximum(pct(headline[i], headline_tr[i]), pct(te, tr)) for i, (tr, te) in enumerate(lst)])
            zmax_hc = np.stack([np.maximum(rz(hcc[i], hcc_tr[i]), rz(te, tr)) for i, (tr, te) in enumerate(lst)])
            pmax_hc = np.stack([np.maximum(pct(hcc[i], hcc_tr[i]), pct(te, tr)) for i, (tr, te) in enumerate(lst)])
            pmax_lat = np.stack([np.maximum(pct(lat[i], lat_tr[i]), pct(te, tr)) for i, (tr, te) in enumerate(lst)])
            rec(f"zmax(HCcoh+LatAD,{h})", zmax_hl); rec(f"pmax(HCcoh+LatAD,{h})", pmax_hl)
            rec(f"zmax(HC_coh,{h})", zmax_hc); rec(f"pmax(HC_coh,{h})", pmax_hc); rec(f"pmax(LatAD,{h})", pmax_lat)
        # per-window dump of the difficult subset (seed-mean robust-z of base and primary heads), for inspection
        rzm = lambda arr, ref: np.mean([rz(arr[i], ref[i]) for i in range(nseed)], 0)
        dump_cols = {"headline_z": rzm(headline, headline_tr), "hc_coh_z": rzm(hcc, hcc_tr), "latad_z": rzm(lat, lat_tr)}
        for h in ("knn_lat_k10", "knn_feat_k10", "vade_exp_prev1", "obs_exp_prev1", "vade_exp_oracle", "obs_exp_oracle", "obs_trans_prev1"):
            dump_cols[h] = heads[h].mean(0)
        # test-normal percentile of each column (rank among y==0 windows): 1.0 = above every normal window
        nrm = y == 0
        dump = []
        for i in np.where(diff)[0]:
            row = {"win": int(i), "double_hard": bool(dhard[i])}
            for c, v in dump_cols.items():
                row[c] = round(float(v[i]), 2); row[c + "_pctN"] = round(float((v[nrm] < v[i]).mean()), 3)
            dump.append(row)
        RES.setdefault("_dumps", {})[name] = dump
        # paired episode bootstrap of the fused headline vs the headline (and vs HC_coh-alone), difficult + double-hard,
        # for the pre-declared primary heads only (k=10 latent kNN; prev1 / selfgate context; combined)
        prim = ["knn_lat_k10", "knn_feat_k10", "knn_pca20_k10", "vade_exp_prev1", "vade_exp_selfgate", "vade_trans_prev1",
                "obs_exp_prev1", "obs_exp_selfgate", "obs_trans_prev1", "knn_lat_k10+vade_exp_prev1", "knn_lat_k10+obs_exp_prev1",
                "vade_exp_ctl_shuf", "obs_exp_ctl_shuf", "obs_switch_prev1", "vade_exp_oracle", "obs_exp_oracle"]
        boots = {}
        for h in prim:
            fused = headline + heads[h]; fused_h = zh + heads[h]
            boots[h] = {"HCcoh+LatAD+h vs HCcoh+LatAD": {s: EF.boot(y, fused, headline, mk, L, reps=reps) for s, mk in (("Difficult", diff), ("DoubleHard", dhard))},
                        "HCcoh+h vs HC_coh": {s: EF.boot(y, fused_h, hcc, mk, L, reps=reps) for s, mk in (("Difficult", diff), ("DoubleHard", dhard))}}
            print(f"  boot {h}: " + json.dumps(boots[h], default=_js)[:400], flush=True)
        RES[name] = dict(n_subset={k: int(v.sum()) for k, v in subsets.items()}, gates=gates, diag=diag, rows=rows, boots=boots,
                         secs=round(time.time() - t0, 1))
        json.dump(RES, open(jf, "w"), indent=1, default=_js)
        # console table for the primary heads
        print(f"\n=== {name} clean subsets {RES[name]['n_subset']} ===")
        show = list(base) + ["LatAD_refit"] + [p for h in prim for p in (f"head:{h}", f"LatAD+{h}", f"HCcoh+{h}", f"HCcoh+LatAD+{h}")]
        for r in show:
            c = rows[r]; print(f"{r:<46}" + "".join(f"{c[s][0]:.3f}±{c[s][1]:.3f}  " for s in ("All", "Easy", "Difficult", "DoubleHard")))
    return RES


if __name__ == "__main__":
    torch.set_num_threads(2)
    mode = sys.argv[1]
    if mode == "fit":
        seeds = tuple(int(s) for s in sys.argv[3].split(",")) if len(sys.argv) > 3 else (0, 1, 2, 3, 4)
        fit(sys.argv[2], seeds)
    elif mode == "eval":
        evaluate(sys.argv[2:] or ["WADI_clean", "HAI", "SWaT_canon"], reps=int(os.environ.get("BOOT_REPS", "1000")))
