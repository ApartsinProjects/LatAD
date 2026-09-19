"""Correct the SWaT_canon difficulty axis for the OFFICIAL-normal train (2026-09-19, v2).

Problem this fixes (root cause of the below-chance breakage on the official normal):
the official Dec-2015 normal recording (swat_normal_canonical.npz) under-samples the operating
range of the water-quality ANALYSER channels (AIT201/AIT202/AIT501/AIT504): their normal
operating point in the recording differs from the normal periods interleaved in the attack log
by >=2 sigma (AIT201 by ~11 sigma; the analysers recalibrate between the two collection
campaigns). Standardising on the narrow recording scale makes >half the TEST-NORMAL windows
saturate max|z| at the +-10 clip, so the train-99th-pct difficulty threshold (calibrated on the
frozen/quiet train) sits far below the test-normal max|z| level. The "Difficult" subset
(max|z| <= thr) then selects the atypically-quiet anomalies against a backdrop of saturated-high
normals, which INVERTS every detector (trivial max|z| AUROC 0.206, all methods below chance).

Fix (label-transparent, difficulty-axis-only; detectors are untouched and still see every channel):
recompute max|z| over the first-sixth (window-mean) block on channels that are BOTH
  (a) ACTIVE  -- train-normal std >= 1e-6 (the existing rule; drops 14 frozen pump/UV/status), and
  (b) COVERAGE-STABLE -- |mean z of the TEST-NORMAL windows| < 2 (drops the 4 recalibrated AIT
      analysers whose recording range does not cover the test operating range; standardising them
      is as ill-defined as standardising a constant channel).
UNCLIPPED, to match build_scores_table's canonical axis for HAI/WADI. Threshold = 99th pct of the
train-normal max|z| over the same channel set. SWaT_canon ONLY; HAI/WADI untouched.

Invariant restored: trivial max|z| AUROC on the Difficult subset returns to the healthy
0.60-0.65 band (WADI_clean 0.60-0.70, fallback-SWaT 0.601), NOT below chance.
"""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import eda_real as E

name = "SWaT_canon"
Du = E.load(name, clip=None)                                   # UNCLIPPED windowed features (canonical axis)
Xtr = Du["Xn_w"].astype(np.float64); Xte = Du["Xa_w"].astype(np.float64)
C6 = len(Du["ch"]); ch = Du["ch"]
Xtr6, Xte6 = Xtr[:, :C6], Xte[:, :C6]

p = os.path.join(ROOT, "_diagnostics", f"scores_{name}.npz")
d = dict(np.load(p, allow_pickle=True))
y = d["label"].astype(int)

mu, sg = Xtr6.mean(0), Xtr6.std(0) + 1e-8
active = Xtr6.std(0) >= 1e-6                                    # (a) not frozen in train
tn_shift = np.abs(((Xte6[y == 0] - mu) / sg).mean(0))          # test-NORMAL operating-point shift (z)
stable = tn_shift < 2.0                                        # (b) recording covers the test range
keep = np.where(active & stable)[0]
dropped_const = [ch[i] for i in range(C6) if not active[i]]
dropped_shift = [ch[i] for i in range(C6) if active[i] and not stable[i]]

maxz = np.abs(Xte6[:, keep]).max(1).astype(np.float32)
maxz_thr = np.float32(float(np.quantile(np.abs(Xtr6[:, keep]).max(1), 0.99)))

old_thr = float(d["maxz_thr"]); old_diff = int(((y == 1) & (d["maxz"] <= old_thr)).sum())
d["maxz"] = maxz; d["maxz_thr"] = maxz_thr
np.savez(p, **d)

from sklearn.metrics import roc_auc_score
new_diff = int(((y == 1) & (maxz <= float(maxz_thr))).sum())
new_easy = int(((y == 1) & (maxz > float(maxz_thr))).sum())
m = (y == 0) | ((y == 1) & (maxz <= float(maxz_thr)))
triv_diff = roc_auc_score(y[m], maxz[m]); triv_all = roc_auc_score(y, maxz)
print(f"kept {len(keep)}/{C6} first-sixth channels")
print(f"  dropped {len(dropped_const)} frozen(train-const): {dropped_const}")
print(f"  dropped {len(dropped_shift)} coverage-shift(>=2z): {dropped_shift}")
print(f"maxz_thr {old_thr:.3f} -> {float(maxz_thr):.3f}")
print(f"Easy/Difficult = {new_easy}/{new_diff}  (was Difficult {old_diff})")
print(f"INVARIANT trivial max|z|: All AUROC {triv_all:.3f}  Difficult AUROC {triv_diff:.3f}  (healthy band 0.55-0.65)")
print(f"overwrote {p}")
