"""A8 as PROXIMITY MASKING (false negatives), measured correctly.

A8 (correct reading): two DISTINCT, well-separated regimes lie CLOSE in observation space. An
extreme excursion of regime A (a genuine anomaly for A) drifts into the neighbourhood of regime B
and is scored as a normal B point by any head that scores a window against its CLOSEST component
(the s_near head of models_vade.py) or against the mixture density (high if ANY mode explains the
point). The anomaly is ABSORBED by the adjacent regime and MISSED. Detecting it needs the
EXPECTED regime from temporal context (the regime of the preceding normal windows), i.e. A10.

Measured here on WADI_clean, HAI, SWaT_canon (paper grid W=60/stride 30, "stats" features, the
shipped VaDE configuration, seed 0 + seed-robustness), plus a synthetic positive control.

  1. regime geometry      : pairwise Mahalanobis separation of regimes (VaDE latent and PCA-10
                            observation space), closest pairs, adjacency (extreme of A reaches B)
  2. masking test         : per anomaly window: missed by LatAD? nearest regime? expected regime
                            from context? far (>q99) from expected AND close (<q50) to another?
  3. context recovery     : AUROC of nearest-regime score vs expected-regime (context) score on
                            the difficult subset and on the masked subset; false-positive cost
  4. synthetic control    : two close-but-distinct regimes; excursion of A landing near B; the
                            nearest / density heads miss it, the context score catches it
  5. WADI frontier        : the 15 residual windows under the masking test

Outputs (incremental, flushed): a8_masking.<ds>.jsonl (one JSON line per item),
per-window tables a8_masking_<ds>_seed<s>.npz.  Paper and shipped model untouched.

Usage (from poc/):  python _diagnostics/a8_masking.py WADI_clean [seeds]   |   ... synth
"""
from __future__ import annotations
import os, sys, json, time, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
POC = os.path.dirname(HERE)
sys.path.insert(0, POC)
from models_vade import train_vade
from winfeat import window_features
import eda_real as E

CFG = {"WADI_clean": (20, 10), "HAI": (40, 16), "SWaT_canon": (40, 16)}   # (K, latent) as shipped
FRONTIER = [16, 17, 18, 19, 20, 21, 237, 238, 360, 361, 362, 544, 545, 546, 547]
OUT = None


def emit(kind, **row):
    row = dict(kind=kind, ts=time.strftime("%H:%M:%S"), **row)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=_js) + "\n"); f.flush()
    print(f"[{kind}] " + json.dumps(row, default=_js)[:700], flush=True)


def _js(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s); m = np.isfinite(s)
    if (y[m] == 1).sum() == 0 or (y[m] == 0).sum() == 0: return float("nan")
    return float(roc_auc_score(y[m], s[m]))


def pct_of(ref, x):
    r = np.sort(ref[np.isfinite(ref)]); return np.searchsorted(r, x, side="right") / len(r)


# ----------------------------------------------------------------------------- regimes
class Regimes:
    """Empirical per-regime Gaussians (Ledoit-Wolf, full cov) in a latent space, from train labels.
    d[:,k] = Mahalanobis distance of each point to regime k; per-regime train quantiles q50/q99 of
    the regime's OWN members give scale-free 'close' (< q50) and 'far' (> q99) tests."""

    def __init__(self, Z, lab, min_n=20):
        ks = [k for k in np.unique(lab) if (lab == k).sum() >= min_n]
        self.ks = np.array(ks); self.mu, self.prec, self.logdet = {}, {}, {}
        for k in ks:
            lw = LedoitWolf().fit(Z[lab == k])
            self.mu[k] = lw.location_; self.prec[k] = lw.precision_
            self.logdet[k] = float(np.linalg.slogdet(lw.covariance_)[1])
        D = self.dist(Z)
        own = np.argmin(D, 1)                                    # re-assign to nearest valid regime
        self.lab_tr = self.ks[own]
        # quantiles on the regime's ORIGINAL members (a regime can lose every window under nearest
        # re-assignment when a neighbour's LW covariance is wider; its own members still define its scale)
        self.q50 = np.array([np.quantile(D[lab == k, i], 0.50) for i, k in enumerate(ks)])
        self.q99 = np.array([np.quantile(D[lab == k, i], 0.99) for i, k in enumerate(ks)])
        self.size = np.array([(own == i).sum() for i in range(len(ks))])
        self.D_tr = D

    def dist(self, Z):
        D = np.empty((len(Z), len(self.ks)))
        for i, k in enumerate(self.ks):
            r = Z - self.mu[k]; D[:, i] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", r, self.prec[k], r), 0))
        return D

    def nll(self, Z):
        D = self.dist(Z); return 0.5 * D ** 2 + 0.5 * np.array([self.logdet[k] for k in self.ks])[None]

    def pairwise(self):
        """Centroid separation in POOLED Mahalanobis units, and the 'reach' test: does the q99
        envelope of A touch the q50 core of B along the centroid line (extreme of A lands inside B)?"""
        n = len(self.ks); rows = []
        for a in range(n):
            for b in range(a + 1, n):
                d = self.mu[self.ks[a]] - self.mu[self.ks[b]]
                Sa = np.linalg.inv(self.prec[self.ks[a]]); Sb = np.linalg.inv(self.prec[self.ks[b]])
                S = 0.5 * (Sa + Sb)
                sep = float(np.sqrt(d @ np.linalg.solve(S, d)))
                dA = float(np.sqrt(d @ self.prec[self.ks[a]] @ d))   # B's centre measured in A's metric
                dB = float(np.sqrt(d @ self.prec[self.ks[b]] @ d))   # A's centre in B's metric
                # B's centre is inside A's q99 envelope -> an A point at its q99 edge sits on B's centre
                rows.append(dict(a=int(self.ks[a]), b=int(self.ks[b]), sep_pooled=sep,
                                 centreB_in_A_units=dA / self.q99[a], centreA_in_B_units=dB / self.q99[b],
                                 nA=int(self.size[a]), nB=int(self.size[b])))
        return rows


def transition_matrix(lab_seq, K_index, alpha=0.5):
    """Row-stochastic P(next regime | current regime) from a train label sequence (index space)."""
    n = len(K_index); T = np.full((n, n), alpha)
    for a, b in zip(lab_seq[:-1], lab_seq[1:]):
        T[a, b] += 1
    return T / T.sum(1, keepdims=True)


def expected_regime(lab_idx, y, i, max_back=20):
    """Index of the regime of the last NORMAL window strictly before i (within max_back), else -1.
    Also returns the majority regime of the last 3 normal windows."""
    prev, hist = -1, []
    for j in range(i - 1, max(-1, i - 1 - max_back), -1):
        if y[j] == 0:
            if prev < 0: prev = lab_idx[j]
            hist.append(lab_idx[j])
            if len(hist) == 3: break
    maj = int(np.bincount(hist).argmax()) if hist else -1
    return prev, maj, len(hist)


# ----------------------------------------------------------------------------- core
def masking_analysis(name, tag, Ztr, Zte, lab_tr, y, hard, latad_te, latad_tr, dens_te, dens_tr, near_te, near_tr,
                     model_near_te, prev_is_test_contig=True, frontier=None, extra=None):
    """Everything downstream of a latent + train regime labels."""
    R = Regimes(Ztr, lab_tr)
    ks = R.ks; kidx = {k: i for i, k in enumerate(ks)}
    Dtr, Dte = R.D_tr, R.dist(Zte)
    near_tr_i, near_te_i = Dtr.argmin(1), Dte.argmin(1)
    # scale-free per-regime percentiles: distance to regime k relative to k's own q99
    rel_tr, rel_te = Dtr / R.q99[None], Dte / R.q99[None]
    T = transition_matrix(near_tr_i, ks)
    # ---- 1. regime geometry ------------------------------------------------------------------
    pw = R.pairwise(); seps = np.array([p["sep_pooled"] for p in pw])
    adj = [p for p in pw if min(p["centreB_in_A_units"], p["centreA_in_B_units"]) < 1.0 + R.q50.mean() / R.q99.mean()]
    # 'adjacent' = an A point at A's q99 edge reaches within B's q50 core (or vice versa)
    adj_strict = [p for p in pw if p["centreB_in_A_units"] < 1.0 or p["centreA_in_B_units"] < 1.0]
    chi_d = Ztr.shape[1]
    emit("regime_geometry", dataset=name, latent=tag, n_regimes=len(ks), dim=chi_d,
         sizes=R.size.tolist(), q50_mean=float(R.q50.mean()), q99_mean=float(R.q99.mean()),
         chi_ref_q50=float(np.sqrt(chi2.ppf(0.5, chi_d))), chi_ref_q99=float(np.sqrt(chi2.ppf(0.99, chi_d))),
         sep_min=float(seps.min()), sep_p10=float(np.quantile(seps, .1)), sep_p25=float(np.quantile(seps, .25)),
         sep_median=float(np.median(seps)), sep_p75=float(np.quantile(seps, .75)), sep_max=float(seps.max()),
         frac_pairs_sep_lt4=float((seps < 4).mean()), frac_pairs_sep_lt6=float((seps < 6).mean()),
         frac_pairs_sep_lt8=float((seps < 8).mean()),
         n_adjacent_pairs_reach=len(adj), n_adjacent_pairs_strict=len(adj_strict),
         frac_regimes_with_adjacent_partner=float(len(set([p["a"] for p in adj] + [p["b"] for p in adj])) / len(ks)),
         closest_pairs=sorted(pw, key=lambda p: p["sep_pooled"])[:8],
         # train-normal: share of windows whose SECOND-nearest regime is within its q50 core (sits in two cores)
         frac_train_in_two_cores=float((np.sort(rel_tr * R.q99[None] / R.q50[None], 1)[:, 1] < 1).mean()),
         frac_train_beyond_own_q99=float((rel_tr[np.arange(len(Dtr)), near_tr_i] > 1).mean()))
    # ---- 2. masking test -------------------------------------------------------------------------
    thr_tr = np.quantile(latad_tr, 0.95); thr_te = np.quantile(latad_te[y == 0], 0.95)
    dens_thr = np.quantile(dens_tr, 0.95); near_thr = np.quantile(near_tr, 0.95)
    n = len(y); rows = []
    ctx_prev = np.full(n, -1); ctx_maj = np.full(n, -1); ctx_n = np.zeros(n, int)
    for i in range(n):
        p, m, c = expected_regime(near_te_i, y, i); ctx_prev[i], ctx_maj[i], ctx_n[i] = p, m, c
    has_ctx = ctx_prev >= 0
    d_exp = np.where(has_ctx, rel_te[np.arange(n), np.maximum(ctx_prev, 0)], np.nan)        # /q99 of expected
    d_near_q50 = Dte[np.arange(n), near_te_i] / R.q50[near_te_i]                              # /q50 of nearest
    d_near_q99 = rel_te[np.arange(n), near_te_i]
    switched = has_ctx & (near_te_i != ctx_prev)
    far_exp = has_ctx & (d_exp > 1.0)
    close_other = switched & (d_near_q50 < 1.0)
    a8_pattern = far_exp & close_other
    a8_pattern_loose = far_exp & switched                                                     # far from expected, absorbed by another (any depth)
    missed_tr = latad_te <= thr_tr; missed_te = latad_te <= thr_te
    missed_near = near_te <= near_thr; missed_dens = dens_te <= dens_thr
    A = y == 1; N = y == 0
    def cnt(mask): return int(mask.sum())
    summ = dict(dataset=name, latent=tag, n_test=n, n_anom=cnt(A), n_hard=cnt(hard), n_normal=cnt(N),
                n_anom_with_context=cnt(A & has_ctx), n_hard_with_context=cnt(hard & has_ctx),
                latad_auroc_all=auroc(y, latad_te), latad_auroc_hard=auroc(y[N | hard], latad_te[N | hard]),
                thr_train_q95=float(thr_tr), thr_testnormal_q95=float(thr_te),
                fpr_testnormal_at_train_thr=float((~missed_tr[N]).mean()),
                # misses
                n_missed_anom_trthr=cnt(A & missed_tr), n_missed_hard_trthr=cnt(hard & missed_tr),
                n_missed_anom_tethr=cnt(A & missed_te), n_missed_hard_tethr=cnt(hard & missed_te),
                # base rates of the pattern on NORMAL test windows (the control: legal regime switches)
                normal_switch_rate=float(switched[N & has_ctx].mean()),
                normal_far_from_expected_rate=float(far_exp[N & has_ctx].mean()),
                normal_a8_pattern_rate=float(a8_pattern[N & has_ctx].mean()),
                normal_a8_pattern_loose_rate=float(a8_pattern_loose[N & has_ctx].mean()),
                # the pattern on anomalies
                anom_switch_rate=float(switched[A & has_ctx].mean()), hard_switch_rate=float(switched[hard & has_ctx].mean()) if cnt(hard & has_ctx) else None,
                anom_far_from_expected_rate=float(far_exp[A & has_ctx].mean()),
                hard_far_from_expected_rate=float(far_exp[hard & has_ctx].mean()) if cnt(hard & has_ctx) else None,
                anom_a8_pattern_rate=float(a8_pattern[A & has_ctx].mean()),
                hard_a8_pattern_rate=float(a8_pattern[hard & has_ctx].mean()) if cnt(hard & has_ctx) else None)
    for lbl, miss in (("trthr", missed_tr), ("tethr", missed_te), ("nearhead", missed_near), ("denshead", missed_dens)):
        mA = A & has_ctx & miss; mH = hard & has_ctx & miss
        summ[f"missed_{lbl}"] = dict(
            n_missed_anom=cnt(mA), n_missed_hard=cnt(mH),
            a8_masked_anom=cnt(mA & a8_pattern), a8_masked_hard=cnt(mH & a8_pattern),
            a8_masked_loose_anom=cnt(mA & a8_pattern_loose), a8_masked_loose_hard=cnt(mH & a8_pattern_loose),
            far_from_expected_anom=cnt(mA & far_exp), far_from_expected_hard=cnt(mH & far_exp),
            switched_anom=cnt(mA & switched), switched_hard=cnt(mH & switched),
            rate_a8_of_missed_anom=(cnt(mA & a8_pattern) / cnt(mA)) if cnt(mA) else None,
            rate_a8_of_missed_hard=(cnt(mH & a8_pattern) / cnt(mH)) if cnt(mH) else None,
            rate_loose_of_missed_anom=(cnt(mA & a8_pattern_loose) / cnt(mA)) if cnt(mA) else None,
            rate_loose_of_missed_hard=(cnt(mH & a8_pattern_loose) / cnt(mH)) if cnt(mH) else None,
            # missed anomalies that are NOT far from their expected regime (in-regime, truly in the bulk)
            in_expected_core_anom=cnt(mA & has_ctx & (d_exp < R.q50[np.maximum(ctx_prev, 0)] / R.q99[np.maximum(ctx_prev, 0)])),
            in_expected_core_hard=cnt(mH & has_ctx & (d_exp < R.q50[np.maximum(ctx_prev, 0)] / R.q99[np.maximum(ctx_prev, 0)])))
    # where do the missed anomalies sit relative to the expected regime? distribution of d_exp (in q99 units)
    for lbl, m in (("missed_anom", A & has_ctx & missed_tr), ("missed_hard", hard & has_ctx & missed_tr),
                   ("caught_anom", A & has_ctx & ~missed_tr), ("normal", N & has_ctx)):
        if m.sum():
            v = d_exp[m]; w = d_near_q50[m]
            summ[f"dexp_q99units_{lbl}"] = dict(n=int(m.sum()), p10=float(np.quantile(v, .1)), p50=float(np.median(v)),
                                                p90=float(np.quantile(v, .9)), frac_gt1=float((v > 1).mean()),
                                                dnear_q50units_p50=float(np.median(w)), frac_dnear_lt1=float((w < 1).mean()))
    emit("masking_summary", **summ)
    # concrete examples: missed anomalies fitting the A8 pattern (and the loose pattern), ranked by d_exp
    ex = np.where(A & has_ctx & missed_tr & a8_pattern_loose)[0]
    ex = ex[np.argsort(-d_exp[ex])][:12]
    for i in ex:
        emit("masked_example", dataset=name, latent=tag, idx=int(i), hard=bool(hard[i]),
             expected_regime=int(ks[ctx_prev[i]]), expected_regime_size=int(R.size[ctx_prev[i]]), ctx_len=int(ctx_n[i]),
             nearest_regime=int(ks[near_te_i[i]]), nearest_regime_size=int(R.size[near_te_i[i]]),
             model_nearest_component=int(model_near_te[i]) if model_near_te is not None else None,
             d_expected_over_q99=float(d_exp[i]), d_nearest_over_q50=float(d_near_q50[i]), d_nearest_over_q99=float(d_near_q99[i]),
             latad_pct_train=float(pct_of(latad_tr, latad_te[i])), dens_pct_train=float(pct_of(dens_tr, dens_te[i])),
             near_pct_train=float(pct_of(near_tr, near_te[i])), strict_a8=bool(a8_pattern[i]),
             pair_sep_pooled=next((p["sep_pooled"] for p in pw if {p["a"], p["b"]} == {int(ks[ctx_prev[i]]), int(ks[near_te_i[i]])}), None),
             trans_prob_expected_to_nearest=float(T[ctx_prev[i], near_te_i[i]]),
             prev_regimes=[int(ks[near_te_i[j]]) for j in range(max(0, i - 4), i)], prev_labels=[int(y[j]) for j in range(max(0, i - 4), i)],
             extra=(extra(i) if extra else None))
    # ---- 3. context recovery ----------------------------------------------------------------------
    # scores (all 'higher = more anomalous', scale-free via per-regime q99):
    s_near = d_near_q99                                                    # nearest-regime score
    s_exp = np.where(has_ctx, d_exp, d_near_q99)                           # expected-regime (strict 1-step context)
    NLL = R.nll(Zte)
    logT = np.log(T + 1e-12)
    s_trans = np.where(has_ctx, -logsumexp(logT[np.maximum(ctx_prev, 0)] - NLL, 1), -logsumexp(-NLL, 1))  # transition-weighted mixture NLL
    s_mix = -logsumexp(-NLL, 1)                                            # plain mixture NLL (density-like)
    # train calibration of the context scores (train is contiguous; expected = previous train window)
    ptr = np.r_[0, near_tr_i[:-1]]
    s_exp_tr = rel_tr[np.arange(len(Dtr)), ptr]
    NLLtr = R.nll(Ztr); s_trans_tr = -logsumexp(logT[ptr] - NLLtr, 1); s_near_tr = rel_tr[np.arange(len(Dtr)), near_tr_i]
    s_mix_tr = -logsumexp(-NLLtr, 1)
    P_near, P_exp, P_trans = pct_of(s_near_tr, s_near), pct_of(s_exp_tr, s_exp), pct_of(s_trans_tr, s_trans)
    s_fused = np.maximum(P_near, P_exp)
    rec = dict(dataset=name, latent=tag)
    subsets = {"all_anom": A, "hard": hard, "hard_with_ctx": hard & has_ctx,
               "missed_anom_trthr": A & missed_tr, "missed_hard_trthr": hard & missed_tr,
               "a8_masked_missed": A & missed_tr & a8_pattern, "a8_loose_missed": A & missed_tr & a8_pattern_loose,
               "frontier": (np.isin(np.arange(n), frontier) if frontier is not None else np.zeros(n, bool))}
    for sname, sm in subsets.items():
        if sm.sum() == 0: continue
        mk = N | sm; yy = sm[mk].astype(int)
        rec[f"auroc_{sname}"] = dict(n=int(sm.sum()), latad=auroc(yy, latad_te[mk]), nearest=auroc(yy, s_near[mk]),
                                     mixture=auroc(yy, s_mix[mk]), expected=auroc(yy, s_exp[mk]),
                                     transition=auroc(yy, s_trans[mk]), fused_near_exp=auroc(yy, s_fused[mk]),
                                     tpr5_latad=float((latad_te[sm] > thr_tr).mean()),
                                     tpr5_nearest=float((P_near[sm] > 0.95).mean()), tpr5_expected=float((P_exp[sm] > 0.95).mean()),
                                     tpr5_transition=float((P_trans[sm] > 0.95).mean()))
    rec["fpr_testnormal_at_train_q95"] = dict(latad=float((latad_te[N] > thr_tr).mean()), nearest=float((P_near[N] > 0.95).mean()),
                                              expected=float((P_exp[N] > 0.95).mean()), transition=float((P_trans[N] > 0.95).mean()))
    rec["fpr_testnormal_at_train_q99"] = dict(nearest=float((P_near[N] > 0.99).mean()), expected=float((P_exp[N] > 0.99).mean()),
                                              transition=float((P_trans[N] > 0.99).mean()))
    # how many missed anomalies does the context score recover at the SAME false-positive budget (test-normal q95)?
    for sc, lbl in ((s_exp, "expected"), (s_trans, "transition"), (s_fused, "fused")):
        t = np.quantile(sc[N], 0.95)
        rec[f"recovered_at_testnormal_fpr5_{lbl}"] = dict(missed_anom=int((A & missed_te).sum()), recovered_anom=int((A & missed_te & (sc > t)).sum()),
                                                          missed_hard=int((hard & missed_te).sum()), recovered_hard=int((hard & missed_te & (sc > t)).sum()),
                                                          lost_anom=int((A & ~missed_te & (sc <= t)).sum()))
    emit("context_recovery", **rec)
    # ---- 5. frontier ---------------------------------------------------------------------------------
    if frontier is not None:
        fr = []
        for i in frontier:
            fr.append(dict(win=int(i), label=int(y[i]), hard=bool(hard[i]), expected=int(ks[ctx_prev[i]]) if has_ctx[i] else None,
                           nearest=int(ks[near_te_i[i]]), switched=bool(switched[i]),
                           d_exp_q99=float(d_exp[i]) if has_ctx[i] else None, d_near_q50=float(d_near_q50[i]), d_near_q99=float(d_near_q99[i]),
                           latad_pct=float(pct_of(latad_tr, latad_te[i])), near_pct=float(P_near[i]), exp_pct=float(P_exp[i]), trans_pct=float(P_trans[i]),
                           a8_strict=bool(a8_pattern[i]), a8_loose=bool(a8_pattern_loose[i]),
                           trans_prob=float(T[ctx_prev[i], near_te_i[i]]) if has_ctx[i] else None))
        emit("frontier", dataset=name, latent=tag, windows=fr,
             n_a8_strict=int(sum(r["a8_strict"] for r in fr)), n_a8_loose=int(sum(r["a8_loose"] for r in fr)),
             n_exp_pct_gt95=int(sum(r["exp_pct"] > 0.95 for r in fr)), n_exp_pct_gt99=int(sum(r["exp_pct"] > 0.99 for r in fr)))
    return dict(near=near_te_i, exp=ctx_prev, d_exp=d_exp, d_near_q50=d_near_q50, a8=a8_pattern, a8_loose=a8_pattern_loose,
                s_near=s_near, s_exp=s_exp, s_trans=s_trans, P_near=P_near, P_exp=P_exp, P_trans=P_trans, ks=ks, missed_tr=missed_tr)


# ----------------------------------------------------------------------------- real datasets
def run_real(name, seeds=(0,)):
    global OUT
    OUT = os.path.join(HERE, f"a8_masking.{name}.jsonl")
    t0 = time.time()
    D = E.load(name)
    Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"].astype(int)
    C6 = Xte0.shape[1] // 6
    maxz_thr = float(np.quantile(np.abs(Xtr0[:, :C6]).max(1), 0.99))
    hard = (y == 1) & (np.abs(Xte0[:, :C6]).max(1) <= maxz_thr)
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr, Xte = ((Xtr0 - mu) / sig).astype(np.float32), ((Xte0 - mu) / sig).astype(np.float32)
    K, LD = CFG[name]; kd = min(80, max(20, len(Xtr) // 10))
    emit("setup", dataset=name, n_train=len(Xtr), n_test=len(Xte), n_anom=int(y.sum()), n_hard=int(hard.sum()), K=K, latent=LD, kd=kd,
         load_secs=round(time.time() - t0, 1))
    frontier = FRONTIER if name == "WADI_clean" else None
    if frontier is not None: assert len(y) == 575
    ch = D["ch"]; Xa_raw, idx = D["Xa_raw"], D["idx_a"]; W = D["W"]
    def describe(i):
        w = Xa_raw[idx[i]:idx[i] + W]; zm = w.mean(0); top = np.argsort(-np.abs(zm))[:4]
        return dict(t0=int(idx[i]), top_channels=[(str(ch[j]), round(float(zm[j]), 2)) for j in top])
    shipped = None
    sp = os.path.join(HERE, f"scores_{name}.npz")
    if os.path.exists(sp):
        shipped = np.load(sp)
    for seed in seeds:
        t1 = time.time()
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
        v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd); v.fit_resid_head(Xtr); v.fit_basin_head(Xtr)
        latad_te = np.asarray(v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto"))
        latad_tr = np.asarray(v.anomaly_score_hard(Xtr, use_resid="auto", use_basin="auto"))
        dens_te, near_te = v._hard_components(Xte); dens_tr, near_tr = v._hard_components(Xtr)
        Ztr, Zte = v._encode_mean(Xtr), v._encode_mean(Xte)
        G = v._responsibilities(Xtr); lab_tr = G.argmax(1)
        with torch.no_grad():
            model_near_te = v._log_pz_given_c(torch.as_tensor(Zte, dtype=torch.float32)).argmax(1).numpy()
        chk = None
        if shipped is not None and seed in list(shipped["seeds"]):
            s0 = shipped["LatAD"][list(shipped["seeds"]).index(seed)]
            chk = dict(corr_with_shipped=float(np.corrcoef(s0, latad_te)[0, 1]), auroc_shipped=auroc(y, s0), auroc_here=auroc(y, latad_te))
        emit("model", dataset=name, seed=seed, secs=round(time.time() - t1, 1), resid_auto=bool(v._resid_auto), basin_lam=float(v._basin_lam),
             n_components_used=int(len(np.unique(lab_tr))), check_vs_shipped=chk)
        res = masking_analysis(name, f"vade{LD}_seed{seed}", Ztr, Zte, lab_tr, y, hard, latad_te, latad_tr, dens_te, dens_tr, near_te, near_tr,
                               model_near_te, frontier=frontier, extra=describe)
        np.savez(os.path.join(HERE, f"a8_masking_{name}_seed{seed}.npz"), y=y, hard=hard, latad=latad_te, latad_tr_q95=np.quantile(latad_tr, .95),
                 **{k: np.asarray(v_) for k, v_ in res.items()})
        # observation-space replicate (PCA-10 of the standardised features, GMM K=16 labels), seed 0 only:
        # regimes defined WITHOUT the VaDE, so the geometry claim does not hinge on the learned latent
        if seed == seeds[0]:
            pca = PCA(10, random_state=0).fit(Xtr); Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)
            gm = GaussianMixture(16, covariance_type="full", random_state=0, reg_covar=1e-3, n_init=2).fit(Ptr)
            olab = gm.predict(Ptr)
            masking_analysis(name, "pca10_gmm16", Ptr, Pte, olab, y, hard, latad_te, latad_tr, dens_te, dens_tr, near_te, near_tr,
                             None, frontier=frontier, extra=describe)
    emit("done", dataset=name, secs=round(time.time() - t0, 1))


# ----------------------------------------------------------------------------- synthetic positive control
def run_synth(seed=0):
    """Two DISTINCT regimes A, B (centroids 6 pooled-sd apart along factor 1: LDA error ~0.1%, deep valley),
    low-rank 48-feature observation like real window stats. Stream alternates A/B with dwell 40 windows.
    Anomaly TOWARD-B: while in A, a 3-window excursion of factor 1 to +5 (an extreme of A that lands 1 sd
    from B's centre). Control anomaly AWAY: factor 1 to -5 (lands in empty space). Invariant stated in
    advance: nearest/mixture heads must catch AWAY and miss TOWARD-B; the expected-regime score must catch both."""
    global OUT
    OUT = os.path.join(HERE, "a8_masking.synth.jsonl")
    rng = np.random.default_rng(seed)
    d, r = 48, 3; A_map = rng.normal(size=(r, d)); DELTA = 6.0; dwell = 40
    def stream(n_windows, inject=None):
        S = rng.normal(size=(n_windows, r)); reg = ((np.arange(n_windows) // dwell) % 2).astype(int)
        S[:, 0] += DELTA * reg
        # AR(1) smoothing within regime to make consecutive windows correlated (like a real plant)
        for t in range(1, n_windows):
            if reg[t] == reg[t - 1]:
                m = np.array([DELTA * reg[t], 0.0, 0.0])
                S[t] = m + 0.6 * (S[t - 1] - m) + 0.8 * (S[t] - m)       # AR(1) around the regime mean
        y = np.zeros(n_windows, int); kind = np.full(n_windows, "", dtype=object)
        if inject:
            starts = [t for t in range(5, n_windows - 5) if reg[t] == 0 and reg[t - 5] == 0 and reg[t + 5] == 0]
            picks = rng.choice(starts, size=inject * 2, replace=False)
            for j, t in enumerate(picks):
                sgn = +1 if j < inject else -1
                for u in range(3):
                    S[t + u, 0] = sgn * 5.0 + 0.5 * rng.normal(); y[t + u] = 1; kind[t + u] = "toward_B" if sgn > 0 else "away"
        X = S @ A_map + 0.1 * rng.normal(size=(n_windows, d))
        return X.astype(np.float32), y, kind, reg, S
    Xtr, ytr, _, regtr, Str = stream(4000)
    Xte, yte, kind, regte, Ste = stream(3000, inject=25)
    mu, sig = Xtr.mean(0), Xtr.std(0) + 1e-8; Xtr, Xte = (Xtr - mu) / sig, (Xte - mu) / sig
    hard = yte == 1
    emit("setup", dataset="SYNTH_adjacent", n_train=len(Xtr), n_test=len(Xte), n_anom=int(yte.sum()),
         n_toward=int((kind == "toward_B").sum()), n_away=int((kind == "away").sum()), delta=DELTA, dwell=dwell)
    # ground-truth geometry check (true regime labels in the factor space): separation, Bayes error
    emit("synth_truth", sep_factor_units=DELTA, bayes_error=float(0.5 * (1 - math.erf(DELTA / 2 / math.sqrt(2)))),
         toward_B_distance_to_B_centre_sd=1.0, toward_B_distance_to_A_centre_sd=5.0)
    for K in (2, 8):
        v = train_vade(Xtr.astype(np.float32), n_clusters=K, latent_dim=8, epochs=40, warmup=8, seed=seed, device="cpu")
        v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=min(80, len(Xtr) // 10)); v.fit_resid_head(Xtr); v.fit_basin_head(Xtr)
        latad_te = np.asarray(v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto")); latad_tr = np.asarray(v.anomaly_score_hard(Xtr, use_resid="auto", use_basin="auto"))
        dens_te, near_te = v._hard_components(Xte); dens_tr, near_tr = v._hard_components(Xtr)
        Ztr, Zte = v._encode_mean(Xtr), v._encode_mean(Xte); lab_tr = v._responsibilities(Xtr).argmax(1)
        with torch.no_grad():
            mn = v._log_pz_given_c(torch.as_tensor(Zte, dtype=torch.float32)).argmax(1).numpy()
        thr = np.quantile(latad_tr, .95)
        tb, aw, N = kind == "toward_B", kind == "away", yte == 0
        emit("synth_shipped_heads", K=K, latent=8,
             latad_tpr5_toward=float((latad_te[tb] > thr).mean()), latad_tpr5_away=float((latad_te[aw] > thr).mean()),
             latad_auroc_toward=auroc(np.r_[np.zeros(N.sum()), np.ones(tb.sum())], np.r_[latad_te[N], latad_te[tb]]),
             latad_auroc_away=auroc(np.r_[np.zeros(N.sum()), np.ones(aw.sum())], np.r_[latad_te[N], latad_te[aw]]),
             dens_tpr5_toward=float((dens_te[tb] > np.quantile(dens_tr, .95)).mean()), dens_tpr5_away=float((dens_te[aw] > np.quantile(dens_tr, .95)).mean()),
             near_tpr5_toward=float((near_te[tb] > np.quantile(near_tr, .95)).mean()), near_tpr5_away=float((near_te[aw] > np.quantile(near_tr, .95)).mean()),
             latad_pct_toward_median=float(np.median(pct_of(latad_tr, latad_te[tb]))), latad_pct_away_median=float(np.median(pct_of(latad_tr, latad_te[aw]))),
             toward_assigned_to_B_component_frac=float(np.mean([lab_tr[regtr == 1].tolist().count(c) > lab_tr[regtr == 0].tolist().count(c) for c in mn[tb]])))
        res = masking_analysis("SYNTH_adjacent", f"vade8_K{K}", Ztr, Zte, lab_tr, yte, hard, latad_te, latad_tr, dens_te, dens_tr, near_te, near_tr, mn,
                               extra=lambda i: dict(kind=str(kind[i]), true_regime=int(regte[i]), factor1=round(float(Ste[i, 0]), 2)))
        # per-kind breakdown of the context recovery
        for lbl, m in (("toward_B", tb), ("away", aw)):
            mk = N | m; yy = m[mk].astype(int)
            emit("synth_by_kind", K=K, akind=lbl, n=int(m.sum()), auroc_latad=auroc(yy, latad_te[mk]), auroc_nearest=auroc(yy, res["s_near"][mk]),
                 auroc_expected=auroc(yy, res["s_exp"][mk]), auroc_transition=auroc(yy, res["s_trans"][mk]),
                 tpr5_nearest=float((res["P_near"][m] > .95).mean()), tpr5_expected=float((res["P_exp"][m] > .95).mean()),
                 tpr5_transition=float((res["P_trans"][m] > .95).mean()),
                 frac_a8_pattern=float(res["a8"][m].mean()), frac_a8_loose=float(res["a8_loose"][m].mean()),
                 frac_missed_latad=float((latad_te[m] <= thr).mean()))
    # observation-space version with TRUE regime labels (no learned latent at all)
    pca = PCA(10, random_state=0).fit(Xtr); Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)
    gm = GaussianMixture(2, covariance_type="full", random_state=0).fit(Ptr)
    res = masking_analysis("SYNTH_adjacent", "pca10_true2", Ptr, Pte, regtr, yte, hard, latad_te, latad_tr, dens_te, dens_tr, near_te, near_tr, None,
                           extra=lambda i: dict(kind=str(kind[i]), true_regime=int(regte[i]), factor1=round(float(Ste[i, 0]), 2)))
    tb, aw, N = kind == "toward_B", kind == "away", yte == 0
    for lbl, m in (("toward_B", tb), ("away", aw)):
        mk = N | m; yy = m[mk].astype(int)
        emit("synth_by_kind", K="true2_pca10", akind=lbl, n=int(m.sum()), auroc_nearest=auroc(yy, res["s_near"][mk]),
             auroc_expected=auroc(yy, res["s_exp"][mk]), auroc_transition=auroc(yy, res["s_trans"][mk]),
             tpr5_nearest=float((res["P_near"][m] > .95).mean()), tpr5_expected=float((res["P_exp"][m] > .95).mean()),
             frac_a8_pattern=float(res["a8"][m].mean()))
    emit("done", dataset="SYNTH_adjacent")


if __name__ == "__main__":
    torch.set_num_threads(2)
    which = sys.argv[1]
    if which == "synth":
        run_synth()
    else:
        seeds = tuple(int(s) for s in sys.argv[2].split(",")) if len(sys.argv) > 2 else (0,)
        run_real(which, seeds)
