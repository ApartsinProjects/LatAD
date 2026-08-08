"""Rev4 new idea #7: DOUBLE-HARD subset, leakage-free.

A window is double-hard iff it is an anomaly AND separated by NEITHER filter, using
TRAIN-normal-calibrated 99th-percentile thresholds for both:
  - univariate: max|z| over the mean stat-block (maxz, threshold maxz_thr from scores_<DS>.npz);
  - linear cross-channel: one-hot leave-one-channel-out residual (LinRes), threshold =
    99th pct of the TRAIN-normal residual r_tr (from onehot_filter.loco_residual).
LatAD is never used to define its own subset. Reports difficult-AUROC per method (5-seed
mean+-std) + episode-block bootstrap of LatAD vs the strongest baseline. Output ->
_diagnostics/rev4_doublehard.json.
"""
from __future__ import annotations
import json, os, numpy as np
from sklearn.metrics import roc_auc_score
import eda_real as E
from onehot_filter import build_feats, loco_residual

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}   # W, stride for episode block len
COMPET = {"WADI": "IF", "HAI": "AE", "SWaT": "linres"}
RNG = np.random.default_rng(0)


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


def mean_auroc(arr, yk, keep):
    if arr.ndim == 2:
        return float(np.nanmean([roc_auc_score(yk, arr[i][keep]) for i in range(arr.shape[0])]))
    return float(roc_auc_score(yk, arr[keep]))


def boot(y, method, compet, hard, L, reps=2000):
    y = y.astype(int); norm = np.where(y == 0)[0]
    heps = [e[hard[e]] for e in episodes(y) if hard[e].any()]
    keep0 = np.where((y == 0) | hard)[0]
    a_pt = mean_auroc(method, y[keep0], keep0); c_pt = mean_auroc(compet, y[keep0], keep0)
    diffs = []
    for _ in range(reps):
        nb = int(np.ceil(len(norm) / L))
        st = RNG.integers(0, max(1, len(norm) - L + 1), size=nb)
        sn = np.concatenate([norm[s:s + L] for s in st])[:len(norm)]
        if not heps:
            break
        pick = RNG.integers(0, len(heps), size=len(heps))
        sh = np.concatenate([heps[k] for k in pick])
        if len(sh) < 2:
            continue
        keep = np.concatenate([sn, sh]); yk = y[keep]
        if yk.sum() < 2 or (yk == 0).sum() < 2:
            continue
        diffs.append(mean_auroc(method, yk, keep) - mean_auroc(compet, yk, keep))
    diffs = np.array(diffs)
    q = lambda v: [round(float(np.quantile(v, 0.025)), 3), round(float(np.quantile(v, 0.975)), 3)]
    return dict(diff=round(a_pt - c_pt, 3), diff_ci=q(diffs) if len(diffs) else None,
                p_le_0=round(float((diffs <= 0).mean()), 4) if len(diffs) else None)


ALL = {}
for name in ["WADI", "HAI", "SWaT"]:
    D = E.load(name); fn, W, stride = E.RAW[name]
    Xn_raw, Xa_raw = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
    d = np.load(f"{OUT}/scores_{name}.npz"); y = d["label"].astype(int)
    mthr = float(d["maxz_thr"]); maxz = d["maxz"]
    # leak-free LinRes: train-normal residual sets the threshold
    Fn, Fa, grp = build_feats(Xn_raw, Xa_raw, W, stride, onehot=True)
    r_tr, r_te = loco_residual(Fn, Fa[:len(y)], grp)
    lin_thr = float(np.quantile(r_tr, 0.99))
    dhard = (y == 1) & (maxz <= mthr) & (r_te <= lin_thr)
    canon = (y == 1) & (maxz <= mthr)
    eps_d = [e for e in episodes(y) if dhard[e].any()]
    L = int(np.ceil(W / stride)) + 1
    rows = {}
    for m in ["trivial max|z|", "IF", "AE", "linres", "USAD", "TranAD", "LatAD"]:
        key = {"trivial max|z|": "maxz"}.get(m, m)
        if key not in d.files:
            continue
        keep = np.where((y == 0) | dhard)[0]
        arr = d[key]
        if arr.ndim == 2:
            v = [roc_auc_score(y[keep], arr[i][keep]) for i in range(arr.shape[0])]
            rows[m] = dict(auroc=round(np.mean(v), 3), sd=round(np.std(v), 3))
        else:
            rows[m] = dict(auroc=round(float(roc_auc_score(y[keep], arr[keep])), 3))
    ck = COMPET[name]
    sig = boot(y, d["LatAD"], d[ck], dhard, L) if dhard.sum() >= 3 else None
    ALL[name] = dict(n_canonical=int(canon.sum()), n_double_hard=int(dhard.sum()),
                     n_episodes=len(eps_d), lin_thr=round(lin_thr, 4),
                     competitor=ck, significance=sig, rows=rows)
    print(f"\n=== {name}: canonical-difficult {int(canon.sum())} -> DOUBLE-HARD {int(dhard.sum())} "
          f"({len(eps_d)} episodes) ===")
    for m, r in rows.items():
        print(f"   {m:14} {r['auroc']}{'±'+str(r['sd']) if 'sd' in r else ''}")
    if sig:
        print(f"   LatAD vs {ck}: diff {sig['diff']} CI {sig['diff_ci']} P(<=0)={sig['p_le_0']}")

json.dump(ALL, open(f"{OUT}/rev4_doublehard.json", "w"), indent=1)
print(f"\nsaved -> {OUT}/rev4_doublehard.json")
