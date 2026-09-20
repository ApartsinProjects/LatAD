"""H2 on the real datasets: A3 overlap read across representations of increasing richness.
Spaces: PCA-k (k in 5,20,50,200) and the FULL standardised 6-stat window features; VaDE latent LD in
{6,16,32}. K fixed at the paper's K (SKAB 16, WADI 20, HAI 40, SWaT 40). One seed. Train-normal only.
Benchmarks use the cached e2_fable_*.npz Xtr_s (same standardised features as the paper models);
SKAB is rebuilt via eda_real.load (same window_features('stats')) and standardised on train.

Invariant (stated before running): if H2 is right, benchmark overlap (gmm_H / knn_amb) RISES in the
rich spaces toward SKAB's level; if A3 is genuinely absent it stays low in every space. The synthetic
control (fable_a3_spaces_synth.py) fixes what 'low' and 'high' mean at each dim.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fable_a3_spaces_lib import overlap_in_space, vade_overlap, pca_spaces, JsonStore, HERE, ROOT

OUT = JsonStore(os.path.join(HERE, "fable_a3_spaces_real.json"))
KPAPER = {"SKAB": 16, "WADI": 20, "HAI": 40, "SWaT": 40}


def log(*a):
    print(*a, flush=True)


def load_train(name):
    if name == "SKAB":
        import eda_real as E
        X = np.asarray(E.load("SKAB")["Xn_w"], np.float32)
        return ((X - X.mean(0)) / (X.std(0) + 1e-8)).astype(np.float32)
    return np.load(os.path.join(HERE, f"e2_fable_{name}.npz"))["Xtr_s"].astype(np.float32)


if __name__ == "__main__":
    t0 = time.time()
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["SKAB", "WADI", "SWaT", "HAI"]
    for name in names:
        X = load_train(name); K = KPAPER[name]; R = OUT.d.setdefault(name, {})
        R["n"], R["d_full"] = int(X.shape[0]), int(X.shape[1])
        log(f"== {name} n={X.shape[0]} d={X.shape[1]} K={K}")
        ks = [k for k in (5, 20, 50, 200) if k < X.shape[1]] + [X.shape[1]]
        spaces, evr = pca_spaces(X, ks)
        R["pca_cum_evr"] = {str(k): float(evr[:k].sum()) for k in ks}
        for k in ks:
            key = f"pca{k}" if k < X.shape[1] else "full"
            if key in R: continue
            t1 = time.time(); R[key] = overlap_in_space(spaces[k], K, kmeans_too=True); R[key]["secs"] = round(time.time() - t1, 1)
            OUT.save(); log(f"  {key}: {R[key]}")
        for LD in (6, 16, 32):
            key = f"vade_LD{LD}"
            if key in R: continue
            t1 = time.time(); r, _, _ = vade_overlap(X, K, LD); r["secs"] = round(time.time() - t1, 1)
            R[key] = r; OUT.save(); log(f"  {key}: {r}")
    log(f"done {time.time()-t0:.0f}s")
