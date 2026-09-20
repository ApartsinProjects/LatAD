"""Re-score GDN per-timestep dumps onto the CLEAN window grid + clean difficulty subsets,
construct-matched with clean_recompute.md (mirrors rev4_sota_aggregate.win_avg exactly)."""
import os, sys, numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eda_real as E
D = os.path.dirname(os.path.abspath(__file__))

JOBS = [("WADI_clean", "score_GDN_WADI_clean_s0.npy"),
        ("HAI",        "gdn_fast_score_HAI_s0.npy"),
        ("SWaT_canon", "sota_pull_official/score_GDN_SWaT_canon_s0.npy")]  # clean; leaked sibling quarantined

def win_starts(n, W, stride): return list(range(0, n - W + 1, stride))
def win_avg(ts, starts, W):   return np.array([ts[i:i+W].mean() for i in starts], np.float32)

for name, gf in JOBS:
    Dd = E.load(name); fn, W, stride = E.RAW[name]
    Xa = np.asarray(Dd["Xa_raw"], float); starts = win_starts(len(Xa), W, stride)
    d = np.load(f"{D}/scores_{name}.npz"); y = d["label"].astype(int)
    if len(starts) != len(y):
        print(f"  [{name}] window/label mismatch {len(starts)} vs {len(y)} -> skip"); continue
    thr = float(d["maxz_thr"]); mz = d["maxz"]
    hard = (y == 1) & (mz <= thr); easy = (y == 1) & (mz > thr)
    keepH = (y == 0) | hard; keepE = (y == 0) | easy
    p = f"{D}/{gf}"
    if not os.path.exists(p): print(f"  [{name}] missing {gf} -> skip"); continue
    ts = np.load(p); ts = ts.mean(1) if ts.ndim > 1 else ts
    if len(ts) != len(Xa):
        print(f"  [{name}] GDN ts len {len(ts)} != raw {len(Xa)} -> skip"); continue
    g = win_avg(ts, starts, W)
    aA = roc_auc_score(y, g); aE = roc_auc_score(y[keepE], g[keepE]); aH = roc_auc_score(y[keepH], g[keepH])
    print(f"  [{name}] GDN  All {aA:.3f}  Easy {aE:.3f}  Difficult {aH:.3f}   (n_diff={int(hard.sum())}, n_win={len(y)})")
print("GDN_RESCORE_DONE")
