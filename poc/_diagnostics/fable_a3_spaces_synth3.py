"""Synthetic invariant v3 (v2 weakness: a K=3 diag GMM never found the low-variance R1/R2 split, ARI<=0.29,
so its entropy was 'wrong-cut' ambiguity). v3: FIVE regimes in a 10-factor low-rank space, four of them far
apart (+-12 on factors 0/1), one OVERLAPPING pair R1/R2 separated by delta*sd on factor 2 (sd 1.92) whose
variance stays below factors 0,1 (which carry the far regimes), so PCA-1/2 DROP the overlap direction.
Also a per-PC whitened variant (VaDE's KL prior roughly whitens its latent). Controls: no-A3 (delta=8)
and a 4-regime twin with NO pair (R1 only), K=5 -> forced over-segmentation of one clump.

Invariants (before running):
  S1 known-A3 (delta=2.5), rich space (PCA>=5, VaDE LD>=6), K=5: ARI(true) > 0.8 and gmm_H within
     +-0.05 of the truth posterior entropy; no-A3 twin: gmm_H < 0.02.
  S3 over-seg twin (4 regimes, K=5): reports what a forced split of one clump reads as.
  S4 (H2) known-A3 in PCA-1/2, VaDE LD=1/2: if measured H DROPS below rich-space H -> coarseness hides
     A3 (H2 viable); if it stays or rises -> not a hiding mechanism.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fable_a3_spaces_lib import overlap_in_space, vade_overlap, pca_spaces, JsonStore, HERE

OUT = JsonStore(os.path.join(HERE, "fable_a3_spaces_synth3.json"))
D, NF = 200, 10


def gen(delta, pair=True, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    fsd = 3.0 * 0.8 ** np.arange(NF)
    A = rng.normal(0, 1, (D, NF)) / np.sqrt(NF)
    means = np.zeros((5, NF)); means[1, 2] = delta * fsd[2]
    means[2, 0] = 12; means[3, 1] = 12; means[4, 0] = -12
    n = [900, 900, 400, 400, 400]
    if not pair:
        n[1] = 0
    F, y = [], []
    for r, nr in enumerate(n):
        if nr == 0: continue
        F.append(rng.normal(0, 1, (nr, NF)) * fsd + means[r]); y += [r] * nr
    F = np.vstack(F); y = np.asarray(y)
    X = F @ A.T + rng.normal(0, noise, (len(F), D))
    reg = [r for r in range(5) if n[r] > 0]
    lp = np.stack([sum(norm.logpdf(F[:, j], means[r, j], fsd[j]) for j in (0, 1, 2)) + np.log(n[r] / sum(n)) for r in reg], 1)
    P = np.exp(lp - logsumexp(lp, 1, keepdims=True))
    Htrue = float((-(P * np.log(P + 1e-12)).sum(1) / np.log(5)).mean())
    return X.astype(np.float32), y, dict(H_true_logK5=Htrue, rho_true=float((P.max(1) < 0.5).mean()), n_regimes=len(reg))


def log(*a):
    print(*a, flush=True)


if __name__ == "__main__":
    t0 = time.time()
    cases = {"A3_delta2.5": dict(delta=2.5), "noA3_delta8": dict(delta=8.0), "overseg_4regimes": dict(delta=0, pair=False)}
    for cname, kw in cases.items():
        X, y, truth = gen(**kw)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        R = OUT.d.setdefault(cname, {}); R["truth"] = truth
        log(f"== {cname} truth={truth}")
        spaces, evr = pca_spaces(X, [1, 2, 5, 20, 50, 200])
        R["pca_evr_first6"] = [float(e) for e in evr[:6]]
        for k, Z in spaces.items():
            for wh in ("", "w"):
                Zk = Z / Z.std(0) if wh else Z
                for K in (5, 10):
                    key = f"pca{k}{wh}_K{K}"
                    if key in R: continue
                    R[key] = overlap_in_space(Zk, K, true_lab=y, kmeans_too=False); OUT.save()
                    log(f"  {key}: H={R[key]['gmm_H']:.3f} rho={R[key]['gmm_rho']:.3f} knn_amb={R[key]['knn_amb']:.3f} knn_amb_true={R[key]['knn_amb_true']:.3f} ARI={R[key]['ari_gmm_true']:.2f}")
        for LD in (1, 2, 6, 16):
            for K in (5, 10):
                key = f"vade_LD{LD}_K{K}"
                if key in R: continue
                if K == 10 and LD not in (2, 6): continue
                r, _, _ = vade_overlap(X, K, LD, true_lab=y); R[key] = r; OUT.save()
                log(f"  {key}: H={r['vade_H']:.3f} rho={r['vade_rho']:.3f} knn_amb={r['knn_amb']:.3f} knn_amb_true={r['knn_amb_true']:.3f} ARI={r['ari_vade_true']:.2f}")
    log(f"done {time.time()-t0:.0f}s")
