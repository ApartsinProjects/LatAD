"""Evaluate the difficulty prefilters as a DEPLOYED cascade detector on the FULL anomaly set,
including the prefilter's own false positives. Each method is scored two ways:
  standalone : the method alone
  cascade    : max(prefilter, method)  -- an OR-alarm; a window is anomalous if the simple
               prefilter OR the method scores it high (scores z-standardized on test-normal so the
               max is in comparable 'sigmas above normal' units).
Prefilters: 'lin' (well-calibrated linear-residual only) and 'full' (linear-residual OR max|z|,
which on WADI carries the drift-inflated max|z| FPs). Metrics that penalize FPs: AUROC, and TPR at
1% / 5% test-normal FPR. Multi-seed methods report mean over seeds. All from _diagnostics/scores_<DS>.npz.
"""
from __future__ import annotations
import sys, numpy as np
from sklearn.metrics import roc_auc_score

MULTI = ("LatAD", "IF", "AE"); SINGLE = ("USAD", "TranAD", "GDN")


def zstd(s, y):
    n = s[y == 0]
    return (s - np.median(n)) / (n.std() + 1e-9)


def tpr_at_fpr(y, s, f):
    thr = np.quantile(s[y == 0], 1 - f)
    return float((s[y == 1] > thr).mean())


def metrics(y, s):
    return (round(float(roc_auc_score(y, s)), 3),
            round(tpr_at_fpr(y, s, 0.01), 3), round(tpr_at_fpr(y, s, 0.05), 3))


def method_scores(d, m, y):
    """return list of z-standardized score vectors (per seed for multi, single-element for SOTA)."""
    if m in MULTI and m in d.files:
        return [zstd(d[m][i], y) for i in range(d[m].shape[0])]
    if m in SINGLE and m in d.files:
        return [zstd(d[m], y)]
    return None


def report(name, pref):
    d = np.load(f"_diagnostics/scores_{name}.npz"); y = d["label"]
    zmax = zstd(d["maxz"], y); zlin = zstd(d["linres"], y)
    s_pre = zlin if pref == "lin" else np.maximum(zlin, zmax)
    pa, p1, p5 = metrics(y, s_pre)
    print(f"\n=== {name}  prefilter='{pref}'  anom={int((y==1).sum())}/{len(y)} ===")
    print(f"  {'method':8} {'ALONE auroc/tpr@1/tpr@5':>26}   {'CASCADE auroc/tpr@1/tpr@5':>26}")
    print(f"  {'prefilter':8} {'-':>26}   {f'{pa:.3f} / {p1:.3f} / {p5:.3f}':>26}")
    for m in MULTI + SINGLE:
        ss = method_scores(d, m, y)
        if ss is None:
            continue
        alone = np.mean([metrics(y, s) for s in ss], 0)
        casc = np.mean([metrics(y, np.maximum(s_pre, s)) for s in ss], 0)
        print(f"  {m:8} {f'{alone[0]:.3f} / {alone[1]:.3f} / {alone[2]:.3f}':>26}   "
              f"{f'{casc[0]:.3f} / {casc[1]:.3f} / {casc[2]:.3f}':>26}")


if __name__ == "__main__":
    pref = "lin"; names = ["WADI", "HAI", "SWaT"]
    for a in sys.argv[1:]:
        if a in ("lin", "full"): pref = a
        elif a in names: names = [a]
    for n in names:
        report(n, pref)
