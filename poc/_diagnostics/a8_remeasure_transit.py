"""Cranfield: are setpoint-step windows improbable under BOTH neighbouring setpoints (true between-regime
points), or merely assigned to the nearer one?  Per detected step: Gaussian (PCA-10) on the steady segment
before and after, Mahalanobis distance of each transition window to both, compared with the within-segment
99th percentile.  Also prints the window-by-window time course across a few steps (VaDE max-resp, obs
log-density percentile, distance to prev/next setpoint).  Appends to a8_remeasure.jsonl."""
from __future__ import annotations
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import a8_remeasure as A

torch.set_num_threads(4)
D = A.build_cranfield()
X = D["X"]; Xs = ((X - X.mean(0)) / (X.std(0) + 1e-8)).astype(np.float32)
v, G, z = A.vade_resp(Xs, K=16, LD=8, epochs=40, warmup=8, seed=0)
Z = PCA(10, random_state=0).fit_transform(Xs)
seg, trans, file, pos = D["seg"], D["trans"], D["file"], D["pos"]
steady = ~trans
gm = GaussianMixture(16, covariance_type="full", random_state=0, reg_covar=1e-3, n_init=2).fit(Z[steady])
ld = gm.score_samples(Z)
with torch.no_grad():
    nll = -v._log_pz_given_c(torch.as_tensor(z, dtype=torch.float32)).max(1).values.numpy()


def maha_stats(Zseg):
    mu = Zseg.mean(0); S = np.cov(Zseg.T) + 1e-3 * np.eye(10); P = np.linalg.inv(S)
    d = np.sqrt(np.einsum("ij,jk,ik->i", Zseg - mu, P, Zseg - mu))
    return mu, P, np.quantile(d, 0.99)


def maha(Zq, mu, P):
    return np.sqrt(np.einsum("ij,jk,ik->i", Zq - mu, P, Zq - mu))


# walk each file; a "step" = maximal run of transition windows with a steady segment on both sides
rows = []; courses = []
for fi in range(3):
    idx = np.where(file == fi)[0]
    i = 0
    while i < len(idx):
        if not trans[idx[i]]:
            i += 1; continue
        j = i
        while j < len(idx) and trans[idx[j]]:
            j += 1
        run = idx[i:j]
        prev_seg = seg[idx[i - 1]] if i > 0 else -1
        next_seg = seg[idx[j]] if j < len(idx) else -1
        if prev_seg >= 0 and next_seg >= 0 and prev_seg != next_seg:
            a = Z[(seg == prev_seg) & steady]; b = Z[(seg == next_seg) & steady]
            if len(a) >= 12 and len(b) >= 12:
                mua, Pa, qa = maha_stats(a); mub, Pb, qb = maha_stats(b)
                da, db = maha(Z[run], mua, Pa), maha(Z[run], mub, Pb)
                both = (da > qa) & (db > qb)
                rows.append(dict(file=int(fi), start_window=int(run[0]), n_windows=int(len(run)), prev_seg=int(prev_seg), next_seg=int(next_seg),
                                 n_prev=int(len(a)), n_next=int(len(b)),
                                 frac_improbable_under_both=float(both.mean()), n_improbable_under_both=int(both.sum()),
                                 min_maha_ratio_median=float(np.median(np.minimum(da / qa, db / qb))),
                                 vade_maxresp_mean=float(G[run].max(1).mean()),
                                 vade_maxresp_on_improbable=float(G[run][both].max(1).mean()) if both.any() else None,
                                 sep_prev_next_maha=float(maha(mub[None], mua, Pa)[0])))
                if len(courses) < 4 and both.sum() >= 2:
                    span = list(range(max(idx[0], run[0] - 2), min(idx[-1], run[-1] + 2) + 1))
                    courses.append(dict(file=int(fi), windows=[dict(idx=int(w), t0=D["info"][w]["t0"], trans=bool(trans[w]),
                                                                    ch12=D["info"][w]["ch12"], ch13=D["info"][w]["ch13"],
                                                                    vade_top=int(G[w].argmax()), vade_maxresp=round(float(G[w].max()), 3),
                                                                    obs_ld_pct=round(float((ld[steady] <= ld[w]).mean()), 3),
                                                                    vade_nll_pct=round(float((nll[steady] <= nll[w]).mean()), 3),
                                                                    d_prev_over_q99=round(float(maha(Z[w][None], mua, Pa)[0] / qa), 2),
                                                                    d_next_over_q99=round(float(maha(Z[w][None], mub, Pb)[0] / qb), 2)) for w in span]))
        i = j

n_tw = sum(r["n_windows"] for r in rows); n_both = sum(r["n_improbable_under_both"] for r in rows)
summ = dict(n_steps=len(rows), n_transition_windows_between_two_known_setpoints=n_tw, n_improbable_under_both=n_both,
            frac_improbable_under_both=round(n_both / max(1, n_tw), 4),
            frac_of_all_windows=round(n_both / len(X), 4),
            steps_with_any_improbable=int(sum(r["n_improbable_under_both"] > 0 for r in rows)),
            vade_maxresp_on_improbable_mean=round(float(np.mean([r["vade_maxresp_on_improbable"] for r in rows if r["vade_maxresp_on_improbable"] is not None])), 4),
            vade_maxresp_all_transition=round(float(np.mean([r["vade_maxresp_mean"] for r in rows])), 4),
            median_sep_prev_next_maha=round(float(np.median([r["sep_prev_next_maha"] for r in rows])), 2),
            # steady-state control: what fraction of STEADY windows are 'improbable under both' its own segment and its neighbour? (should be ~1%)
            )
ctrl = []
for r in rows:
    a = Z[(seg == r["prev_seg"]) & steady]; mua, Pa, qa = maha_stats(a)
    ctrl.append(float((maha(a, mua, Pa) > qa).mean()))
summ["control_frac_steady_beyond_own_q99"] = round(float(np.mean(ctrl)), 4)
A.emit("cranfield_between_setpoint_test", **summ)
for r in rows:
    A.emit("cranfield_step", **r)
for c in courses:
    A.emit("cranfield_step_course", **c)
