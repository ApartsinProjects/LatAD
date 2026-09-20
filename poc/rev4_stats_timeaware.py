"""E4 (reviewer items A-22/A-23/A-24): time-aware significance test.

We ALREADY use an episode + moving-block bootstrap (rev4_stats.py). This script
STRENGTHENS it by choosing the moving-block length from the SCORE AUTOCORRELATION
(the lag at which the normal-window score ACF first drops below 0.1), applied
uniformly to every dataset, and reports the difficult-subset paired difference
(LatAD vs strongest same-family competitor) under three block lengths:

  L = 1        -> i.i.d. resample  (invariant/sanity: reproduces the naive bootstrap)
  L = overlap  -> ceil(W/stride)+1 (what rev4_stats.py used)
  L = acf      -> autocorrelation-derived  (the time-aware test we report)

Definitions (A-23): CI = two-sided [2.5, 97.5] bootstrap percentile of the metric;
P = one-sided bootstrap Pr(paired diff <= 0). A result is a WIN only if diff>0 and
the diff CI excludes 0 (equivalently P small).

Reads _diagnostics/scores_<DS>.npz only. No re-inference. Writes
_diagnostics/rev4_timeaware.json. Does NOT touch the paper.
"""
from __future__ import annotations
import json, os, numpy as np
from sklearn.metrics import roc_auc_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}      # (W, stride)
COMPET = {"WADI": "IF", "HAI": "AE", "SWaT": "linres"}
B = 2000


def episodes(y):
    eps, i, n = [], 0, len(y)
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1:
                j += 1
            eps.append(np.arange(i, j)); i = j
        else:
            i += 1
    return eps


def moving_block(idx, L, rng):
    n = len(idx)
    if n == 0 or L <= 1:
        # L<=1 -> i.i.d. resample of the same length
        return idx if L <= 0 else rng.choice(idx, size=n, replace=True) if L == 1 else idx
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, max(1, n - L + 1), size=nb)
    return np.concatenate([idx[s:s + L] for s in starts])[:n]


def _mean_auroc(scoremat, yk, keep):
    if scoremat.ndim == 2:
        return float(np.nanmean([roc_auc_score(yk, scoremat[si][keep]) for si in range(scoremat.shape[0])]))
    return float(roc_auc_score(yk, scoremat[keep]))


def acf_block_len(s, max_lag=300, thr=0.1):
    """First lag k with ACF(k) < thr, on the (mean-removed) series s."""
    s = np.asarray(s, float)
    s = s - s.mean()
    denom = float((s * s).sum())
    if denom == 0 or len(s) < 3:
        return 1
    for k in range(1, min(max_lag, len(s) - 1)):
        r = float((s[:-k] * s[k:]).sum()) / denom
        if r < thr:
            return k
    return min(max_lag, len(s) - 1)


def boot(y, method, compet, hard, L, reps, rng):
    y = y.astype(int)
    norm_idx = np.where(y == 0)[0]
    anom_eps = [e for e in episodes(y) if hard[e].any()]
    hard_eps = [e[hard[e]] for e in anom_eps]
    keep0 = np.where((y == 0) | hard)[0]
    a_pt = _mean_auroc(method, y[keep0], keep0)
    c_pt = _mean_auroc(compet, y[keep0], keep0)
    a_bs, c_bs, diffs = [], [], []
    for _ in range(reps):
        sn = moving_block(norm_idx, L, rng)
        if hard_eps:
            pick = rng.integers(0, len(hard_eps), size=len(hard_eps))
            sh = np.concatenate([hard_eps[k] for k in pick])
        else:
            sh = np.array([], int)
        if len(sh) < 2:
            continue
        keep = np.concatenate([sn, sh]); yk = y[keep]
        if yk.sum() < 2 or (yk == 0).sum() < 2:
            continue
        try:
            a_bs.append(_mean_auroc(method, yk, keep))
            c_bs.append(_mean_auroc(compet, yk, keep))
            diffs.append(a_bs[-1] - c_bs[-1])
        except ValueError:
            continue
    diffs = np.array(diffs)
    q = lambda v: [round(float(np.quantile(v, 0.025)), 3), round(float(np.quantile(v, 0.975)), 3)]
    return dict(
        L=int(L), n_boot=len(diffs),
        auroc=round(a_pt, 3), auroc_ci=q(np.array(a_bs)),
        compet_auroc=round(c_pt, 3),
        diff=round(a_pt - c_pt, 3), diff_ci=q(diffs),
        p_diff_le_0=round(float((diffs <= 0).mean()), 4),
        win=bool((a_pt - c_pt) > 0 and q(diffs)[0] > 0),
    )


ALL = {}
for name in ["WADI", "HAI", "SWaT"]:
    W, stride = CFG[name]
    d = np.load(f"{OUT}/scores_{name}.npz")
    y = d["label"].astype(int)
    thr = float(d["maxz_thr"])
    easy = (y == 1) & (d["maxz"] > thr)
    hard = (y == 1) & ~easy
    method = d["LatAD"]; compet = d[COMPET[name]]
    # ACF block length from the seed-mean LatAD score on NORMAL windows (the null's dependence)
    lat_mean = method.mean(0) if method.ndim == 2 else method
    L_acf = acf_block_len(lat_mean[y == 0])
    L_overlap = int(np.ceil(W / stride)) + 1
    reps = 1000 if name == "HAI" else B
    rng = np.random.default_rng(0)
    res = {
        "competitor": COMPET[name],
        "n_difficult": int(hard.sum()),
        "n_episodes_difficult": int(len([e for e in episodes(y) if hard[e].any()])),
        "L_acf": int(L_acf), "L_overlap": int(L_overlap),
        "iid":     boot(y, method, compet, hard, 1,        reps, np.random.default_rng(0)),
        "overlap": boot(y, method, compet, hard, L_overlap, reps, np.random.default_rng(0)),
        "acf":     boot(y, method, compet, hard, L_acf,     reps, np.random.default_rng(0)),
    }
    ALL[name] = res
    print(f"\n=== {name} ===  competitor={res['competitor']}  n_diff={res['n_difficult']} "
          f"eps={res['n_episodes_difficult']}  L_acf={L_acf} L_overlap={L_overlap}")
    for tag in ["iid", "overlap", "acf"]:
        r = res[tag]
        print(f"  {tag:8s} L={r['L']:2d}: diff={r['diff']:+.3f} CI{r['diff_ci']} "
              f"P(diff<=0)={r['p_diff_le_0']:.4f}  WIN={r['win']}")

json.dump(ALL, open(f"{OUT}/rev4_timeaware.json", "w"), indent=1)
print(f"\nsaved -> {OUT}/rev4_timeaware.json")
