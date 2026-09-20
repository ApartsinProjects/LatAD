"""Synthetic INVARIANT for the A3 space/K check. Known-A3 = two OVERLAPPING Gaussian regimes (R1,R2,
separation Delta=2.5 sd along dim 0) plus a far third regime R3 (dim 1, +10). Noise dims 2-4 have sd 1.8
(> the R1/R2 discriminating variance, so PCA-1/2 DROP dim 0 = 'too coarse latent mixes the regimes');
dims 5..199 decay. Controls: no-A3 twin (Delta=10) and a single blob (over-segmentation floor).

v2 (v1 with isotropic high-dim noise was invalid: nothing recovered the regimes, ARI<=0.1 everywhere; kept as fable_a3_spaces_synth.json).
Invariants stated BEFORE running:
  S1 known-A3, rich space (PCA>=5 / full / VaDE LD>=6), K=3: gmm_H ~ true-posterior H_norm (+-0.05)
     and >> the no-A3 twin (S2) and >> single-blob floor.
  S2 no-A3 twin, any space with k>=5, K=3: gmm_H < 0.02, knn_amb < 0.02.
  S3 single blob, K=3 (and K=8): the over-segmentation floor. If it reaches the known-A3 level, entropy
     cannot separate overlap from over-splitting at that K.
  S4 (H2 mechanism) known-A3 in PCA-1/2 and VaDE LD=1/2: knn_amb_true HIGH (regimes mixed). If the
     MEASURED gmm_H/vade_H DROPS below the rich-space value -> coarseness HIDES A3 (H2 viable).
     If it stays or RISES -> coarseness fabricates/keeps overlap, H2 is not a hiding mechanism.
  S5 over-clustering (K=8 on known-A3 / no-A3 / blob): report how much K inflates the read.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fable_a3_spaces_lib import overlap_in_space, vade_overlap, pca_spaces, JsonStore, HERE

OUT = JsonStore(os.path.join(HERE, "fable_a3_spaces_synth2.json"))
D = 200


def gen(delta, n=(1200, 1200, 600), seed=0, blob=False, n_fac=10, noise=0.5):
    """v2: low-rank CORRELATED structure (like real stats windows). 10 latent factors with sd 3*0.8^j,
    random 200x10 loadings, iid feature noise sd 0.5. R3 sits at +12 on factor 0 (dominant); R1/R2 are
    separated by delta*sd on factor 5 (sd 0.98), whose variance stays BELOW factors 0-2 for delta=2.5,
    so PCA-1/2 drop it (coarse latent mixes R1/R2) while PCA>=5 keeps it. Truth posterior from factors."""
    rng = np.random.default_rng(seed)
    fsd = 3.0 * 0.8 ** np.arange(n_fac)
    A = rng.normal(0, 1, (D, n_fac)) / np.sqrt(n_fac)
    means = np.zeros((3, n_fac)); means[1, 5] = delta * fsd[5]; means[2, 0] = 12.0
    if blob:
        F = rng.normal(0, 1, (sum(n), n_fac)) * fsd
        X = F @ A.T + rng.normal(0, noise, (len(F), D))
        return X.astype(np.float32), np.zeros(len(X), int), None
    F, y = [], []
    for r, nr in enumerate(n):
        F.append(rng.normal(0, 1, (nr, n_fac)) * fsd + means[r]); y += [r] * nr
    F = np.vstack(F); y = np.asarray(y)
    X = F @ A.T + rng.normal(0, noise, (len(F), D))
    lp = np.stack([norm.logpdf(F[:, 0], means[r, 0], fsd[0]) + norm.logpdf(F[:, 5], means[r, 5], fsd[5])
                   + np.log(n[r] / sum(n)) for r in range(3)], 1)
    P = np.exp(lp - logsumexp(lp, 1, keepdims=True))
    Htrue = float((-(P * np.log(P + 1e-12)).sum(1) / np.log(3)).mean())
    rho_true = float((P.max(1) < 0.5).mean())
    return X.astype(np.float32), y, dict(H_true=Htrue, rho_true=rho_true, H_true_binary_R1R2=float(Htrue * np.log(3) / np.log(2)))


def log(*a):
    print(*a, flush=True)


if __name__ == "__main__":
    t0 = time.time()
    cases = {"A3_delta2.5": dict(delta=2.5), "noA3_delta10": dict(delta=10.0), "blob": dict(delta=0, blob=True)}
    for cname, kw in cases.items():
        X, y, truth = gen(**kw)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        R = OUT.d.setdefault(cname, {})
        if truth: R["truth"] = truth
        log(f"== {cname} truth={truth}")
        spaces, evr = pca_spaces(X, [1, 2, 5, 20, 50, 200])
        R["pca_evr_first5"] = [float(e) for e in evr[:5]]
        tl = None if kw.get("blob") else y
        for k, Z in spaces.items():
            for K in (3, 8):
                key = f"pca{k}_K{K}"
                if key in R: continue
                R[key] = overlap_in_space(Z, K, true_lab=tl, kmeans_too=False); OUT.save()
                log(f"  {key}: {R[key]}")
        for LD in (1, 2, 6, 16):
            for K in (3, 8):
                key = f"vade_LD{LD}_K{K}"
                if key in R: continue
                if K == 8 and LD not in (2, 6): continue
                r, _, _ = vade_overlap(X, K, LD, true_lab=tl); R[key] = r; OUT.save()
                log(f"  {key}: {r}")
    log(f"done {time.time()-t0:.0f}s")
