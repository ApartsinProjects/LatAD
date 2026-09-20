"""Fold downloaded Modal per-timestep SOTA dumps (score_<M>_SWaT_canon_s<seed>.npy) into:
 (1) scores_sota_ms_SWaT_canon.npz  -> multi-seed USAD/TranAD (nseed,nwin) + GDN (nwin)  [ensemble_final]
 (2) scores_SWaT_canon.npz          -> add USAD/TranAD (seed-mean single) + GDN (single)  [build/headline]
Window-averaged on the SAME grid as build_scores_table. Usage: python merge_sota_swat_canon.py <scoredir>"""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from sklearn.metrics import roc_auc_score
import eda_real as E
SD = sys.argv[1]; name = "SWaT_canon"; SEEDS = [0,1,2,3,4]
OUT = os.path.join(ROOT, "_diagnostics")
D = E.load(name); fn, W, stride = E.RAW[name]
Xa_raw = np.asarray(D["Xa_raw"], float)
starts = list(range(0, len(Xa_raw) - W + 1, stride))
d = dict(np.load(f"{OUT}/scores_{name}.npz", allow_pickle=True))
y = d["label"].astype(int)
assert len(starts) == len(y), f"{len(starts)} vs {len(y)}"
wavg = lambda ts: np.array([ts[i:i+W].mean() for i in starts], np.float32)
def load_ts(m, s):
    p = os.path.join(SD, f"score_{m}_{name}_s{s}.npy")
    if not os.path.exists(p): return None
    ts = np.load(p); ts = ts.mean(1) if ts.ndim > 1 else ts
    if len(ts) != len(Xa_raw):
        print(f"  {m} s{s}: len {len(ts)} != {len(Xa_raw)} skip"); return None
    return wavg(ts)
thr = float(d["maxz_thr"]); hard = (y==1) & ~((y==1) & (d["maxz"]>thr))
keepH = np.where((y==0)|hard)[0]
ms = {}
for m in ["USAD","TranAD"]:
    per = [load_ts(m,s) for s in SEEDS]; per=[p for p in per if p is not None]
    if per:
        arr = np.stack(per); ms[m]=arr
        d[m] = arr.mean(0).astype(np.float32)   # single fallback = seed-mean
        auH=[roc_auc_score(y[keepH],arr[i][keepH]) for i in range(len(arr))]
        print(f"{m}: n_seed={len(arr)} DIFFICULT {np.mean(auH):.3f}+-{np.std(auH):.3f}")
g = load_ts("GDN",0)
if g is not None:
    ms["GDN"]=g; d["GDN"]=g.astype(np.float32)
    print(f"GDN: DIFFICULT {roc_auc_score(y[keepH],g[keepH]):.3f}")
if ms:
    np.savez(f"{OUT}/scores_sota_ms_{name}.npz", label=y, hard=hard, **ms)
    print("saved scores_sota_ms")
np.savez(f"{OUT}/scores_{name}.npz", **d)
print("patched scores_SWaT_canon.npz with", [m for m in ("USAD","TranAD","GDN") if m in d])
