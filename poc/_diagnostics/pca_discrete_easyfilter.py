"""Extended difficulty stratification: add a PCA-per-axis and a discrete-state trivial
detector to the 'easy' family, so linearly/discretely-detectable anomalies are bucketed
EASY (uniformly, pre-stated), leaving the genuinely joint-nonlinear faults as 'difficult'.

Trivial family (a window is EASY if ANY exceeds its own train-normal 99th pct):
  T1 continuous : max|z| of window-mean over ALL continuous channels (current filter, all-channel)
  T2 discrete   : max|z| over one-hot state-fraction features of discrete channels (onehot_filter)
  T3 pca-axis   : max|z| over WHITENED principal-component axes of the window-feature matrix
                  (the user's idea: threshold each PCA axis like each original coordinate)

CIRCULARITY GUARD: linres/l2 (linear) and PCA DEFINE difficulty here, so they are NOT reported
as beaten baselines. LatAD is compared only against IF + deep (USAD/TranAD/AE) on the survivors.
Report-only. Prints, per dataset, difficult counts under each family and LatAD-vs-(IF/deep).
"""
from __future__ import annotations
import os, sys, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eda_real as E
from onehot_filter import build_feats

HERE = os.path.dirname(os.path.abspath(__file__))
DMAX = 6


def au(y, s, diff):
    k = (y == 0) | diff
    return float(roc_auc_score(y[k], s[k])) if 0 < y[k].sum() < k.sum() else float("nan")


def run(name):
    d = dict(np.load(os.path.join(HERE, f"scores_{name}.npz")))
    y = d["label"].astype(int)
    D = E.load(name); fn, W, stride = E.RAW[name]
    Xtr0 = np.asarray(D["Xn_w"], np.float32); Xte0 = np.asarray(D["Xa_w"], np.float32)
    Xn_raw = np.asarray(D["Xn_raw"], float); Xa_raw = np.asarray(D["Xa_raw"], float)
    n = len(y)

    # T1 continuous (all channels, not just first sixth)
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    ztr = np.abs((Xtr0 - mu) / sig).max(1); zte = np.abs((Xte0 - mu) / sig).max(1)
    thr1 = np.quantile(ztr, 0.99); T1 = zte > thr1

    # T3 PCA-per-axis (whitened), on standardized window features
    Xtr_s = (Xtr0 - mu) / sig; Xte_s = (Xte0 - mu) / sig
    p = PCA(min(Xtr_s.shape[1], len(Xtr_s) - 1), random_state=0).fit(Xtr_s)
    Ptr = p.transform(Xtr_s); Pte = p.transform(Xte_s)
    ps = Ptr.std(0) + 1e-8
    mtr = np.abs(Ptr / ps).max(1); mte = np.abs(Pte / ps).max(1)
    thr3 = np.quantile(mtr, 0.99); T3 = mte > thr3

    # T2 discrete one-hot state fractions
    Fn, Fa, grp = build_feats(Xn_raw, Xa_raw, W, stride, onehot=True)
    Fa = Fa[:n]
    # discrete = channels whose original had <= DMAX distinct train values AND expanded to >1 onehot col
    disc_cols = []
    for c in range(Xn_raw.shape[1]):
        k = len(np.unique(Xn_raw[:, c]))
        if 1 < k <= DMAX:
            disc_cols.extend(np.where(grp == c)[0].tolist())
    if disc_cols:
        Dn, Dte = Fn[:, disc_cols], Fa[:, disc_cols]
        dmu, dsig = Dn.mean(0), Dn.std(0) + 1e-8
        dztr = np.abs((Dn - dmu) / dsig).max(1); dzte = np.abs((Dte - dmu) / dsig).max(1)
        thr2 = np.quantile(dztr, 0.99); T2 = dzte > thr2
    else:
        T2 = np.zeros(n, bool)

    anom = y == 1
    def stratum(easy): return anom & ~easy
    fams = {"T1 only (current-ish)": T1, "+T3 pca": T1 | T3, "+T2 disc": T1 | T2, "+T2+T3 (all)": T1 | T2 | T3}
    print(f"\n== {name}: n={n} anom={int(anom.sum())} disc_channels={len(set(grp[disc_cols])) if disc_cols else 0}")
    for tag, easy in fams.items():
        diff = stratum(easy)
        print(f"  [{tag:16s}] difficult anomalies = {int(diff.sum())}/{int(anom.sum())}")
    # evaluate on the strongest stratification (+T2+T3), non-circular methods only
    diff = stratum(T1 | T2 | T3)
    if diff.sum() < 3:
        print("  (too few difficult anomalies to score)"); return
    print(f"  -- AUROC on '+T2+T3' difficult (LatAD vs IF/deep; linres/l2 excluded, circular) --")
    for k in ("LatAD", "IF", "AE", "USAD", "TranAD"):
        if k not in d: continue
        S = d[k]; S = S[None] if S.ndim == 1 else S
        a = np.mean([au(y, s, diff) for s in S])
        print(f"     {k:8s} {a:.3f}")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI_clean", "HAI", "SWaT"]):
        try:
            run(nm)
        except Exception as e:
            print(f"[{nm}] ERROR {type(e).__name__}: {e}")
