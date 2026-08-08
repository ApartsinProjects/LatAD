"""Rev4 Tier-2 items 7 & 9: episode counts, episode/block-bootstrap CIs + paired
significance, and deployable metrics (AUPRC, TPR@1%FPR, normal-quantile-threshold F1).

Reads the unified per-window scores tables (_diagnostics/scores_<DS>.npz) only -- no
re-inference. Difficult-subset AUROC protocol matches build_comparison_table.py exactly:
subset k = (y==0) | mask, mask = difficult anomalous windows; multi-seed -> per-seed
AUROC then mean. CIs via moving-block bootstrap of normal windows + episode-level
bootstrap of anomalous episodes (respects window overlap / serial dependence).

Wins-only: this SCRIPT computes; the paper reports only the results that survive. Output
-> _diagnostics/rev4_stats_<DS>.json and a combined _diagnostics/rev4_stats.json.
"""
from __future__ import annotations
import json, os, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
RNG = np.random.default_rng(0)
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}  # W, stride -> overlap block
B = 2000  # bootstrap reps (HAI reduced below)

# strongest same-family competitor per dataset for the paired test
COMPET = {"WADI": "IF", "HAI": "AE", "SWaT": "linres"}


def episodes(y):
    """Contiguous runs of y==1 (attacks) in window order -> list of (start,end) index arrays."""
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


def auroc_sub(y, s, keep):
    yy = y[keep]
    if yy.sum() < 2 or (yy == 0).sum() < 2:
        return float("nan")
    return roc_auc_score(yy, s[keep])


def moving_block(idx, L, rng):
    """Moving-block bootstrap resample of a 1-D index array to ~same length."""
    n = len(idx)
    if n == 0:
        return idx
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, max(1, n - L + 1), size=nb)
    out = np.concatenate([idx[s:s + L] for s in starts])[:n]
    return out


def _mean_auroc_on(scoremat, yk, keep):
    """Seed-averaged AUROC on the kept indices. scoremat: (n_seed,n_win) or (n_win,)."""
    if scoremat.ndim == 2:
        return float(np.nanmean([roc_auc_score(yk, scoremat[si][keep]) for si in range(scoremat.shape[0])]))
    return float(roc_auc_score(yk, scoremat[keep]))


def boot_ci(y, score_multi, hard_mask, compet, seeds_axis, reps, rng):
    """Bootstrap distribution of difficult-subset AUROC (seed-averaged) for the method,
    the competitor, and their paired difference. Both score_multi and compet may be
    (n_seed,n_win) or (n_win,). Returns dict of point + 95% CI + P(diff<=0)."""
    y = y.astype(int)
    norm_idx = np.where(y == 0)[0]
    anom_eps = [e for e in episodes(y) if hard_mask[e].any()]  # attacks with >=1 difficult win
    # restrict each episode's contributed windows to the difficult ones
    hard_eps = [e[hard_mask[e]] for e in anom_eps]
    L = CFG_L

    # point estimates on the true (unresampled) subset
    keep0 = np.where((y == 0) | hard_mask)[0]
    a_pt = _mean_auroc_on(score_multi, y[keep0], keep0)
    c_pt = _mean_auroc_on(compet, y[keep0], keep0)

    diffs, a_bs, c_bs = [], [], []
    for _ in range(reps):
        sn = moving_block(norm_idx, L, rng)
        # episode bootstrap: resample attacks with replacement, take their difficult windows
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
            a = _mean_auroc_on(score_multi, yk, keep)
            c = _mean_auroc_on(compet, yk, keep)
        except ValueError:
            continue
        a_bs.append(a); c_bs.append(c); diffs.append(a - c)
    a_bs, c_bs, diffs = map(np.array, (a_bs, c_bs, diffs))
    q = lambda v: [round(float(np.quantile(v, 0.025)), 3), round(float(np.quantile(v, 0.975)), 3)]
    return dict(
        auroc=round(float(a_pt), 3), auroc_ci=q(a_bs),
        compet_auroc=round(float(c_pt), 3), compet_ci=q(c_bs),
        diff=round(float(a_pt - c_pt), 3), diff_ci=q(diffs),
        p_diff_le_0=round(float((diffs <= 0).mean()), 4),
        n_boot=len(diffs))


def deployable(y, s):
    """AUPRC, TPR@1%FPR (threshold from normal windows only), and F1 at a
    normal-quantile threshold (99th pct of normal scores -> ~1% FPR by construction)."""
    y = y.astype(int)
    normal = s[y == 0]
    ap = float(average_precision_score(y, s))
    thr99 = float(np.quantile(normal, 0.99))  # calibrate on NORMAL only (no anomaly labels)
    pred = (s > thr99).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(auprc=round(ap, 3), tpr_at_1pct_fpr=round(rec, 3),
                prec_at_normal_q99=round(prec, 3), f1_normal_q99=round(f1, 3))


ALL = {}
for name in ["WADI", "HAI", "SWaT"]:
    global CFG_L
    W, stride = CFG[name]
    CFG_L = int(np.ceil(W / stride)) + 1
    reps = 1000 if name == "HAI" else B
    d = np.load(f"{OUT}/scores_{name}.npz"); y = d["label"].astype(int)
    thr = float(d["maxz_thr"])
    easy = (y == 1) & (d["maxz"] > thr); hard = (y == 1) & ~easy
    eps_all = episodes(y)
    eps_hard = [e for e in eps_all if hard[e].any()]
    compet_key = COMPET[name]
    compet = d[compet_key]
    # paired significance on the DIFFICULT subset: LatAD vs strongest same-family competitor
    ci = boot_ci(y, d["LatAD"], hard, compet, None, reps, RNG)
    # deployable metrics: multi-seed methods -> mean±std over seeds; single -> point
    def dep_multi(arr):
        if arr.ndim == 2:
            per = [deployable(y, arr[si]) for si in range(arr.shape[0])]
            keys = per[0].keys()
            out = {}
            for k in keys:
                vals = [p[k] for p in per]
                out[k] = round(float(np.mean(vals)), 3)
                out[k + "_sd"] = round(float(np.std(vals)), 3)
            return out
        return deployable(y, arr)
    dep = {m: dep_multi(d[m]) for m in ["LatAD", "IF", "AE", "USAD", "TranAD"] if m in d.files}
    res = dict(
        n_windows=int(len(y)), n_anom=int((y == 1).sum()),
        n_easy=int(easy.sum()), n_difficult=int(hard.sum()),
        n_attack_episodes_total=len(eps_all),
        n_attack_episodes_with_difficult=len(eps_hard),
        difficult_windows_per_episode=[int(hard[e].sum()) for e in eps_hard],
        competitor=compet_key,
        difficult_paired=ci,
        deployable=dep,
    )
    ALL[name] = res
    json.dump(res, open(f"{OUT}/rev4_stats_{name}.json", "w"), indent=1)
    print(f"\n=== {name} ===")
    print(f"  windows={res['n_windows']} anom={res['n_anom']} "
          f"(easy {res['n_easy']} + difficult {res['n_difficult']})")
    print(f"  attack episodes total={res['n_attack_episodes_total']}  "
          f"with>=1 difficult window={res['n_attack_episodes_with_difficult']}")
    print(f"  difficult windows/episode: {res['difficult_windows_per_episode']}")
    print(f"  DIFFICULT LatAD vs {compet_key}: "
          f"LatAD {ci['auroc']} CI{ci['auroc_ci']} | {compet_key} {ci['compet_auroc']} CI{ci['compet_ci']}")
    print(f"    paired diff {ci['diff']} CI{ci['diff_ci']}  P(diff<=0)={ci['p_diff_le_0']}  (n_boot={ci['n_boot']})")
    print(f"  deployable (seed0): " + "  ".join(
        f"{m}:AUPRC={v['auprc']},TPR@1%FPR={v['tpr_at_1pct_fpr']},F1n={v['f1_normal_q99']}"
        for m, v in dep.items()))

json.dump(ALL, open(f"{OUT}/rev4_stats.json", "w"), indent=1)
print(f"\nsaved -> {OUT}/rev4_stats.json")
