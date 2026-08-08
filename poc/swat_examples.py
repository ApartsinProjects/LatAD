"""Forensics: pick SWaT anomaly timesteps that evade the per-channel max|z| rule ('difficult')
but are large multivariate deviations, and SHOW what actually makes them anomalous -- which named
sensors deviate and by how much. Also test the SIMPLEST possible multivariate axis (no covariance):
breadth = how many channels are moderately off (|z|>2), and L1 = mean|z|.
"""
from __future__ import annotations
import numpy as np
import eda_real as E

Xn, Xa, ya, sens = E._raw_swat()
Xn = np.asarray(Xn, float); Xa = np.asarray(Xa, float); ya = np.asarray(ya, int)
sens = list(sens)
mu, sd = Xn.mean(0), Xn.std(0) + 1e-8
Za = (Xa - mu) / sd

maxz = np.abs(Za).max(1)
thr = np.quantile(np.abs((Xn - mu) / sd).max(1), 0.99)
breadth = (np.abs(Za) > 2).sum(1)                     # SIMPLE: count of channels moderately off
l1 = np.abs(Za).mean(1)                                # SIMPLE: mean |z|

diff = (ya == 1) & (maxz <= thr)                       # difficult = evades max|z|
easy = (ya == 1) & (maxz > thr)
norm = (ya == 0)
print(f"SWaT: {int(easy.sum())} easy / {int(diff.sum())} difficult anomaly timesteps; max|z| thr={thr:.1f}")
print(f"  breadth(#|z|>2): normal med={np.median(breadth[norm]):.0f}  difficult med={np.median(breadth[diff]):.0f}  easy med={np.median(breadth[easy]):.0f}")
print(f"  mean|z| (L1):    normal med={np.median(l1[norm]):.2f}  difficult med={np.median(l1[diff]):.2f}  easy med={np.median(l1[easy]):.2f}")
print(f"  max|z|:          normal med={np.median(maxz[norm]):.2f}  difficult med={np.median(maxz[diff]):.2f} (all <= {thr:.1f} by def)")

# a few specific difficult examples: highest-breadth difficult timesteps (clearly anomalous, yet no single spike)
idx = np.where(diff)[0]
order = idx[np.argsort(-breadth[idx])]
print("\n=== 4 difficult SWaT anomaly examples (evade max|z|, sorted by breadth) ===")
for t in order[:4]:
    z = Za[t]; top = np.argsort(-np.abs(z))[:6]
    print(f"\n t={t}: max|z|={maxz[t]:.1f} (<{thr:.1f}), #|z|>2={breadth[t]}, mean|z|={l1[t]:.2f}")
    for c in top:
        print(f"    {sens[c]:12} z={z[c]:+5.1f}  value={Xa[t,c]:9.3f}  (normal mean {mu[c]:.3f} sd {sd[c]:.3f})")
