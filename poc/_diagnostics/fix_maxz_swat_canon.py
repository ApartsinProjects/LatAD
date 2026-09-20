"""Correct the SWaT_canon difficulty axis after the leak fix. The clean train has train-constant
channels (unrecorded in the mirror's Normal_v0 + genuine status/setpoints); standardising them with
(std+1e-8) makes any test deviation a ~1e8-sigma divide-by-eps artifact that saturates max|z| and
sends every anomaly to 'Easy' (Difficult subset empties). Recompute max|z| over ACTIVE first-sixth
channels only (train-normal std > 1e-6) -- the exact `active` rule the community-expert construction
already uses (modal_experts.py) -- on the CLIP=10-bounded features. Overwrite maxz/maxz_thr in
scores_SWaT_canon.npz. SWaT_canon only; HAI/WADI untouched."""
import os, sys, numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
import eda_real as E
name="SWaT_canon"; clip=E.CLIP.get(name)
Dc=E.load(name, clip=clip); Xtr=Dc["Xn_w"].astype(np.float64); Xte=Dc["Xa_w"].astype(np.float64)
C6=Xte.shape[1]//6
sd=Xtr[:,:C6].std(0); act=np.where(sd>=1e-6)[0]
maxz=np.abs(Xte[:,act]).max(1).astype(np.float32)
maxz_thr=np.float32(float(np.quantile(np.abs(Xtr[:,act]).max(1),0.99)))
p=os.path.join(ROOT,"_diagnostics",f"scores_{name}.npz")
d=dict(np.load(p, allow_pickle=True))
old_thr=float(d["maxz_thr"]); old_diff=int(((d["label"].astype(int)==1)&(d["maxz"]<=old_thr)).sum())
d["maxz"]=maxz; d["maxz_thr"]=maxz_thr
np.savez(p, **d)
y=d["label"].astype(int); newdiff=int(((y==1)&(maxz<=float(maxz_thr))).sum())
print(f"active first-sixth channels kept {len(act)}/{C6} (dropped {C6-len(act)} train-constant); clip={clip}")
print(f"maxz_thr {old_thr:.3f} -> {float(maxz_thr):.3f}; Difficult anomalies (old-stored raw-maxz){old_diff} -> (active) {newdiff}; overwrote {p}")
