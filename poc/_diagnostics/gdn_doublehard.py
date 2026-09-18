"""GDN double-hard AUROC per dataset, construct-matched: dhard = anomaly separated by
NEITHER the trivial max|z| rule NOR linres (thresholds from train-normal 99th pct), exactly
as rev4_doublehard.py. GDN per-window scores from gdn_rescore_clean.py's inputs."""
import os, sys, numpy as np
from sklearn.metrics import roc_auc_score
POC = r"E:\Projects\Backlog\LatAD\poc"; os.chdir(POC); sys.path.insert(0, POC)
import eda_real as E
from ensemble_final import build_feats, loco_residual
D = r"E:\Projects\Backlog\LatAD\poc\_diagnostics"
JOBS = [("WADI_clean", "score_GDN_WADI_clean_s0.npy"), ("HAI", "gdn_fast_score_HAI_s0.npy"),
        ("SWaT_canon", "score_GDN_SWaT_canon_s0.npy")]
def win_starts(n, W, s): return list(range(0, n - W + 1, s))
def win_avg(ts, st, W):  return np.array([ts[i:i+W].mean() for i in st], np.float32)
for name, gf in JOBS:
    Dd = E.load(name); fn, W, stride = E.RAW[name]
    Xn_raw = np.asarray(Dd["Xn_raw"], float); Xa_raw = np.asarray(Dd["Xa_raw"], float)
    d = np.load(f"{D}/scores_{name}.npz"); y = d["label"].astype(int)
    mthr = float(d["maxz_thr"]); maxz = d["maxz"]
    Fn, Fa, grp = build_feats(Xn_raw, Xa_raw, W, stride, onehot=True)
    r_tr, r_te = loco_residual(Fn, Fa[:len(y)], grp); lin_thr = float(np.quantile(r_tr, 0.99))
    dhard = (y == 1) & (maxz <= mthr) & (r_te <= lin_thr)
    keep = (y == 0) | dhard
    starts = win_starts(len(Xa_raw), W, stride)
    ts = np.load(f"{D}/{gf}"); ts = ts.mean(1) if ts.ndim > 1 else ts
    g = win_avg(ts, starts, W)
    print(f"  [{name}] GDN double-hard AUROC = {roc_auc_score(y[keep], g[keep]):.3f}  (n_dhard={int(dhard.sum())})")
print("DONE")
