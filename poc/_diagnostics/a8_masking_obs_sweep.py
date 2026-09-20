"""Robustness of the observation-space masking test to the regime partition (K, GMM seed, expected-
regime rule). Re-uses the seed-0 LatAD scores saved by a8_masking.py (a8_masking_<ds>_seed0.npz) so no
VaDE retraining; regimes = full-cov GMM on PCA-10 of the standardised stats features (train-normal).
Appends rows to a8_masking.<ds>.jsonl (kind = obs_sweep).  Paper and shipped model untouched.
Usage: python _diagnostics/a8_masking_obs_sweep.py WADI_clean
"""
from __future__ import annotations
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eda_real as E
import a8_masking as M

HERE = os.path.dirname(os.path.abspath(__file__))


def run(name, lseed=0):
    M.OUT = os.path.join(HERE, f"a8_masking.{name}.jsonl")
    D = E.load(name)
    Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"].astype(int)
    C6 = Xte0.shape[1] // 6
    hard = (y == 1) & (np.abs(Xte0[:, :C6]).max(1) <= np.quantile(np.abs(Xtr0[:, :C6]).max(1), 0.99))
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr, Xte = (Xtr0 - mu) / sig, (Xte0 - mu) / sig
    S = np.load(os.path.join(HERE, f"a8_masking_{name}_seed{lseed}.npz"))
    latad = S["latad"]; thr = float(S["latad_tr_q95"])
    N, A = y == 0, y == 1
    missed = A & (latad <= thr)
    frontier = M.FRONTIER if name == "WADI_clean" else None
    for dim in (10, 20):
        pca = PCA(dim, random_state=0).fit(Xtr); Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)
        for K in (8, 16, 24):
            for gs in (0, 1, 2):
                t0 = time.time()
                gm = GaussianMixture(K, covariance_type="full", random_state=gs, reg_covar=1e-3, n_init=1).fit(Ptr)
                lab = gm.predict(Ptr)
                R = M.Regimes(Ptr, lab); ks = R.ks
                Dtr, Dte = R.D_tr, R.dist(Pte)
                ntr, nte = Dtr.argmin(1), Dte.argmin(1)
                rel_te = Dte / R.q99[None]; rel_tr = Dtr / R.q99[None]
                T = M.transition_matrix(ntr, ks); logT = np.log(T + 1e-12)
                n = len(y)
                for rule in ("prev", "maj3"):
                    ctx = np.full(n, -1)
                    for i in range(n):
                        p, mj, c = M.expected_regime(nte, y, i)
                        ctx[i] = p if rule == "prev" else mj
                    has = ctx >= 0
                    d_exp = np.where(has, rel_te[np.arange(n), np.maximum(ctx, 0)], np.nan)
                    d_near50 = Dte[np.arange(n), nte] / R.q50[nte]
                    sw = has & (nte != ctx); far = has & (d_exp > 1); a8 = far & sw & (d_near50 < 1); loose = far & sw
                    s_near = rel_te[np.arange(n), nte]; s_exp = np.where(has, d_exp, s_near)
                    NLL = R.nll(Pte); s_trans = np.where(has, -logsumexp(logT[np.maximum(ctx, 0)] - NLL, 1), -logsumexp(-NLL, 1))
                    def au(sub, s):
                        mk = N | sub; return M.auroc(sub[mk].astype(int), s[mk])
                    # FUSION with LatAD (the A10-extension question): z-sum against TEST-normal stats (HAI test
                    # normals drift, so train z-scores would be dominated by the drift), and rank-max.
                    # ROBUST z (median / IQR of test normals, clipped at +-20): plain mean/std z is degenerate here
                    # because a handful of clipped-channel test normals give LatAD a std of ~1e12 on WADI.
                    def z(s):
                        q1, q2, q3 = np.percentile(s[N], [25, 50, 75]); return np.clip((s - q2) / (q3 - q1 + 1e-9), -20, 20)
                    fz_exp, fz_trans = z(latad) + z(s_exp), z(latad) + z(s_trans)
                    rk = lambda s: np.searchsorted(np.sort(s[N]), s, side="right") / N.sum()
                    fm_exp = np.maximum(rk(latad), rk(s_exp)); fr_exp = rk(latad) + rk(s_exp)
                    row = dict(dataset=name, latad_seed=lseed, dim=dim, K=K, gmm_seed=gs, rule=rule, n_regimes=len(ks),
                               normal_far_rate=float(far[N & has].mean()), normal_loose_rate=float(loose[N & has].mean()), normal_a8_rate=float(a8[N & has].mean()),
                               n_missed=int(missed.sum()), missed_far=int((missed & far).sum()), missed_loose=int((missed & loose).sum()), missed_a8=int((missed & a8).sum()),
                               missed_no_ctx=int((missed & ~has).sum()),
                               enrichment_loose=(float((missed & loose).sum() / max(1, (missed & has).sum())) / max(1e-9, float(loose[N & has].mean()))),
                               auroc_hard=dict(latad=au(hard, latad), nearest=au(hard, s_near), expected=au(hard, s_exp), transition=au(hard, s_trans)),
                               auroc_missed=dict(latad=au(missed, latad), nearest=au(missed, s_near), expected=au(missed, s_exp), transition=au(missed, s_trans)),
                               auroc_hard_fused=dict(zsum_exp=au(hard, fz_exp), zsum_trans=au(hard, fz_trans), rankmax_exp=au(hard, fm_exp), ranksum_exp=au(hard, fr_exp)),
                               auroc_all_fused=dict(latad=au(A, latad), zsum_exp=au(A, fz_exp), zsum_trans=au(A, fz_trans), rankmax_exp=au(A, fm_exp), ranksum_exp=au(A, fr_exp)),
                               secs=round(time.time() - t0, 1))
                    if frontier is not None:
                        fm = np.zeros(n, bool); fm[frontier] = True
                        row["auroc_frontier"] = dict(latad=au(fm, latad), nearest=au(fm, s_near), expected=au(fm, s_exp), transition=au(fm, s_trans))
                        row["frontier_loose"] = int((fm & loose).sum()); row["frontier_a8"] = int((fm & a8).sum())
                        row["frontier_exp_pct_gt95"] = int((M.pct_of(rel_tr[np.arange(len(Dtr)), np.r_[0, ntr[:-1]]], s_exp[fm]) > 0.95).sum())
                    M.emit("obs_sweep", **row)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
