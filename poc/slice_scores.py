"""Offline filter evaluator on the unified per-window tables (_diagnostics/scores_<DS>.npz).
Any difficulty filter is a mask; multi-seed methods report mean+/-std over seeds, SOTA single value.
Filters: 'maxz' (canonical), 'cascade' (maxz + linear-residual), 'abs<K>' (absolute-sigma on maxz).
Usage: python slice_scores.py [maxz|cascade|abs5] [WADI HAI SWaT]
"""
from __future__ import annotations
import sys, numpy as np
from sklearn.metrics import roc_auc_score

MULTI = ("LatAD", "IF", "AE"); SINGLE = ("USAD", "TranAD", "GDN")


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k])) if m.sum() >= 3 else float("nan")


def hard_mask(d, y, filt):
    mz = d["maxz"]; thr = float(d["maxz_thr"])
    easyZ = (y == 1) & (mz > thr)
    if filt == "maxz":
        return (y == 1) & ~easyZ
    if filt == "cascade":
        lr = d["linres"]; thrR = float(np.quantile(lr[y == 0], 0.99))
        return (y == 1) & ~easyZ & ~(lr > thrR)
    if filt.startswith("abs"):
        # absolute-sigma cut on standardized maxz proxy is not stored; reuse maxz>K*median-normal
        K = float(filt[3:]); base = np.median(mz[y == 0]) + 1e-9
        return (y == 1) & ~((y == 1) & (mz > K * base))
    raise ValueError(filt)


def report(filt, names):
    for n in names:
        d = np.load(f"_diagnostics/scores_{n}.npz"); y = d["label"]
        hard = hard_mask(d, y, filt)
        print(f"\n{n}  filter={filt}  anom={int((y==1).sum())}  hard={int(hard.sum())}")
        for m in MULTI:
            if m in d.files:
                a = [au(y, d[m][i], hard) for i in range(d[m].shape[0])]
                print(f"   {m:7} {np.nanmean(a):.3f} +/- {np.nanstd(a):.3f}")
        for m in SINGLE:
            if m in d.files:
                print(f"   {m:7} {au(y, d[m], hard):.3f}")
        # linear-residual detector (deterministic). Circular on the 'cascade' subset it defines,
        # so only report it as a competitor on non-cascade filters.
        if filt != "cascade":
            print(f"   {'LinRes':7} {au(y, d['linres'], hard):.3f}")
        else:
            print(f"   {'LinRes':7} (defines this filter; use maxz split for a fair comparison)")


if __name__ == "__main__":
    args = sys.argv[1:]
    filt = args[0] if args and args[0] in ("maxz", "cascade") or (args and args[0].startswith("abs")) else "cascade"
    names = [a for a in args if a in ("WADI", "HAI", "SWaT")] or ["WADI", "HAI", "SWaT"]
    report(filt, names)
