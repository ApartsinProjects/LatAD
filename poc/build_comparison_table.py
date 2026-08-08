"""Phase A: final §6 comparison-table numbers, construct-matched from the unified per-window scores
tables (_diagnostics/scores_<DS>.npz). Per method x subset(All/Easy/Difficult): AUROC + best raw F1.
Multi-seed methods (LatAD, IF, AE) -> mean±std over seeds; single (USAD, TranAD, LinRes, trivial).
Canonical max|z| difficulty split (threshold stored in the table). SOTA are window-aligned (already
averaged onto the label windows in the tables). Emits _diagnostics/comparison_table.json + prints.
"""
from __future__ import annotations
import json, os, numpy as np
from sklearn.metrics import roc_auc_score, f1_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
ROWS = [("trivial max|z|", "maxz", "single"), ("Isolation Forest", "IF", "multi"),
        ("AutoEncoder", "AE", "multi"), ("LinRes (one-hot)", "linres", "single"),
        ("USAD", "USAD", "single"), ("TranAD", "TranAD", "single"),
        ("VaDE-hard+resid (ours)", "LatAD", "multi")]


def bestf1(y, s):
    qs = np.quantile(s, np.linspace(0.5, 0.999, 80))
    return float(max(f1_score(y, s > t, zero_division=0) for t in qs))


def metrics_on(y, s, mask):
    k = (y == 0) | mask
    if mask.sum() < 3:
        return (float("nan"), float("nan"))
    return (float(roc_auc_score(y[k], s[k])), bestf1(y[k].astype(int), s[k]))


def cell(d, key, kind, y, mask):
    if key not in d.files:
        return None
    if kind == "multi":
        arr = d[key]
        vals = [metrics_on(y, arr[i], mask) for i in range(arr.shape[0])]
        au = np.nanmean([v[0] for v in vals]); au_sd = np.nanstd([v[0] for v in vals])
        f1 = np.nanmean([v[1] for v in vals])
        return dict(auroc=round(au, 3), auroc_sd=round(au_sd, 3), f1=round(f1, 3))
    au, f1 = metrics_on(y, d[key], mask)
    return dict(auroc=round(au, 3), f1=round(f1, 3))


TAB = {}
for name in ["WADI", "HAI", "SWaT"]:
    d = np.load(f"{OUT}/scores_{name}.npz"); y = d["label"]
    thr = float(d["maxz_thr"])
    easy = (y == 1) & (d["maxz"] > thr); hard = (y == 1) & ~easy; allm = (y == 1)
    n = dict(anom=int(allm.sum()), easy=int(easy.sum()), difficult=int(hard.sum()))
    print(f"\n=== {name}  (anom {n['anom']} = {n['easy']} easy + {n['difficult']} difficult) ===")
    print(f"  {'method':22} | {'All AUROC':>12} {'F1':>5} | {'Easy AUROC':>12} {'F1':>5} | {'Diff AUROC':>12} {'F1':>5}")
    TAB[name] = {"n": n, "rows": {}}
    for label, key, kind in ROWS:
        c = {sub: cell(d, key, kind, y, m) for sub, m in [("all", allm), ("easy", easy), ("difficult", hard)]}
        if c["all"] is None:
            continue
        TAB[name]["rows"][label] = c

        def fmt(cc):
            if cc is None: return f"{'-':>12} {'-':>5}"
            a = f"{cc['auroc']:.3f}" + (f"±{cc['auroc_sd']:.3f}" if 'auroc_sd' in cc else "")
            return f"{a:>12} {cc['f1']:.3f}"
        print(f"  {label:22} | {fmt(c['all'])} | {fmt(c['easy'])} | {fmt(c['difficult'])}")

json.dump(TAB, open(f"{OUT}/comparison_table.json", "w"), indent=1)
print(f"\nsaved -> {OUT}/comparison_table.json")
