"""Rebuild sota_bundle/ens_bundle/bundle_SWaT_canon.npz from the CLEAN load + fresh scores npz.
Keys match the existing bundle (Xn_w, Xa_w, y, nch, maxz, maxz_thr, label). No model training."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import eda_real as E
name = "SWaT_canon"
D = E.load(name)
d = np.load(os.path.join(ROOT, "_diagnostics", f"scores_{name}.npz"))
Xn_w = D["Xn_w"].astype(np.float32); Xa_w = D["Xa_w"].astype(np.float32)
y = D["ya_w"].astype(np.int64); nch = len(D["ch"])
assert np.array_equal(d["label"].astype(int), y), "scores/label vs load mismatch"
out = os.path.join(ROOT, "sota_bundle", "ens_bundle", f"bundle_{name}.npz")
np.savez(out, Xn_w=Xn_w, Xa_w=Xa_w, y=y, nch=np.int64(nch),
         maxz=d["maxz"].astype(np.float32), maxz_thr=np.float32(float(d["maxz_thr"])), label=y)
print(f"saved {out}: Xn_w{Xn_w.shape} Xa_w{Xa_w.shape} nch={nch} anom={int(y.sum())}/{len(y)} maxz_thr={float(d['maxz_thr']):.3f}")
