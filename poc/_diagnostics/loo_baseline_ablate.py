"""Ablation for loo_baseline_why: which feature family carries the LinRes score? Score = mean of
per-feature normalised LOO residuals restricted to (a) discrete one-hot state-fraction features,
(b) continuous window-mean features, (c) all (canonical). Also numeric (no one-hot) LinRes.
Prints All / difficult AUROC per dataset. Invariant: (c) == npz linres AUROC."""
from __future__ import annotations
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import eda_real as E
from onehot_filter import build_feats, DMAX

HERE = os.path.dirname(os.path.abspath(__file__))


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k]))


def loco(Fn, Fa, grp):
    Rtr = np.zeros_like(Fn); Rte = np.zeros_like(Fa)
    for j in range(Fn.shape[1]):
        cols = np.where(grp != grp[j])[0]
        m = LinearRegression().fit(Fn[:, cols], Fn[:, j])
        Rtr[:, j] = (m.predict(Fn[:, cols]) - Fn[:, j]) ** 2
        Rte[:, j] = (m.predict(Fa[:, cols]) - Fa[:, j]) ** 2
    return Rtr, Rte


out = {}
for name in (sys.argv[1:] or ["WADI_clean", "SWaT_canon", "HAI"]):
    D = E.load(name); fn, W, stride = E.RAW[name]
    Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
    Z = np.load(os.path.join(HERE, f"scores_{name}.npz")); y = Z["label"].astype(int)
    diff = (y == 1) & (Z["maxz"] <= float(Z["maxz_thr"]))
    Fn, Fa, grp = build_feats(Xn, Xa, W, stride, onehot=True); Fa = Fa[:len(y)]
    nstates = np.array([len(np.unique(Xn[:, c])) for c in range(Xn.shape[1])])
    is_disc = np.array([nstates[g] <= DMAX for g in grp])
    Rtr, Rte = loco(Fn, Fa, grp)
    r = {}
    r["canon_all"], r["canon_diff"] = au(y, Rte.mean(1), y == 1), au(y, Rte.mean(1), diff)
    r["npz_all"], r["npz_diff"] = au(y, Z["linres"], y == 1), au(y, Z["linres"], diff)
    for tag, m in [("discrete_only", is_disc), ("continuous_only", ~is_disc)]:
        if m.sum() == 0:
            continue
        s_raw = Rte[:, m].mean(1)
        r[f"{tag}_raw_all"], r[f"{tag}_raw_diff"] = au(y, s_raw, y == 1), au(y, s_raw, diff)
    r["n_feat_discrete"], r["n_feat_continuous"] = int(is_disc.sum()), int((~is_disc).sum())
    r["n_ch_discrete"] = int(len(np.unique(grp[is_disc]))) if is_disc.any() else 0
    # numeric LinRes (discrete channels as plain numeric means)
    Fn2, Fa2, grp2 = build_feats(Xn, Xa, W, stride, onehot=False); Fa2 = Fa2[:len(y)]
    _, Rte2 = loco(Fn2, Fa2, grp2)
    r["numeric_all"], r["numeric_diff"] = au(y, Rte2.mean(1), y == 1), au(y, Rte2.mean(1), diff)
    # continuous channels only, discrete channels removed from predictors too
    keep = ~is_disc
    _, Rte3 = loco(Fn[:, keep], Fa[:, keep], grp[keep])
    r["continuous_dropdisc_all"], r["continuous_dropdisc_diff"] = au(y, Rte3.mean(1), y == 1), au(y, Rte3.mean(1), diff)
    out[name] = r
    print(f"\n=== {name} ===", flush=True)
    for k, v in r.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)
    json.dump(out, open(os.path.join(HERE, "loo_baseline_ablate.json"), "w"), indent=1)
