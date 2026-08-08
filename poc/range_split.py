"""Difficulty split by OBSERVED NORMAL RANGE instead of a quantile of max|z|.
easy  = some channel leaves its normal [min,max] envelope at any timestep in the window
        (out of range -> trivially detectable; includes unseen actuator states)
difficult = every channel stays within the exact range it occupied during normal operation.
0% train false positives by construction. A margin (fraction of the channel's normal range) can be
added to tolerate tiny drift. Reports subset sizes + each method's difficult-AUROC from the scores
tables, vs the canonical max|z| quantile split.
"""
from __future__ import annotations
import sys, numpy as np
from sklearn.metrics import roc_auc_score
import eda_real as E

MULTI = ("LatAD", "IF", "AE"); SINGLE = ("USAD", "TranAD"); EXTRA = ("linres",)


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k])) if m.sum() >= 3 else float("nan")


def range_hard(name, margin):
    D = E.load(name); fn, W, stride = E.RAW[name]
    Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
    y = D["ya_w"].astype(int)
    lo, hi = Xn.min(0), Xn.max(0)
    rng = (hi - lo); lo = lo - margin * rng; hi = hi + margin * rng
    starts = range(0, len(Xa) - W + 1, stride)
    oor = np.array([bool(((Xa[i:i + W] < lo) | (Xa[i:i + W] > hi)).any()) for i in starts])[:len(y)]
    hard = (y == 1) & ~oor
    return hard, y, int((~oor & (y == 1)).sum()), int((oor & (y == 1)).sum())


def report(name, margin):
    d = np.load(f"_diagnostics/scores_{name}.npz")
    y = d["label"]
    # canonical max|z| split for reference
    mz_hard = (y == 1) & ~(d["maxz"] > float(d["maxz_thr"]))
    r_hard, y2, n_hard, n_easy = range_hard(name, margin)
    assert len(y2) == len(y)
    print(f"\n=== {name}  margin={margin} ===")
    print(f"  max|z|-quantile difficult n={int(mz_hard.sum())}   |   range-based difficult n={n_hard} (easy {n_easy})")

    def line(mask, tag):
        row = f"  {tag:16}"
        for m in MULTI:
            if m in d.files:
                a = [au(y, d[m][i], mask) for i in range(d[m].shape[0])]
                row += f" {m}={np.nanmean(a):.3f}"
        for m in SINGLE:
            if m in d.files:
                row += f" {m}={au(y, d[m], mask):.3f}"
        if "linres" in d.files:
            row += f" LinRes={au(y, d['linres'], mask):.3f}"
        print(row)
    line(mz_hard, "maxz-split")
    line(r_hard, "range-split")


if __name__ == "__main__":
    margin = 0.0
    names = ["WADI", "HAI", "SWaT"]
    for a in sys.argv[1:]:
        if a.replace(".", "").isdigit(): margin = float(a)
        elif a in names: names = [a]
    for n in names:
        report(n, margin)
