"""
Component 1 - physically-guided, controllable anomaly generation.

Pipeline (per the idea):
  1. SHUFFLE channels between windows of DIFFERENT operating modes -> a candidate
     that keeps every channel's marginal realistic but breaks the cross-channel
     coupling (the physics/control structure). This is the "dependency-violation"
     fault a uniform-random outlier can never be.
  2. FILTER with an ORACLE that is INDEPENDENT of the detector under test, so the
     benchmark is not rigged. On synthetic data the oracle is the TRUE generative
     process (kNN distance to a true-normal reference); a shuffled candidate is
     kept only if it is genuinely off the true normal manifold. (Shuffling can
     land back on a valid state - those are discarded.)
  3. CONTROL difficulty by moving the candidate's latent a fraction alpha toward
     the nearest normal cluster centre, then DECODING. The decoder is a physics
     prior (its outputs stay plausible); alpha tunes the distance from normal, so
     one dial spans near-boundary (hard) to far (easy) faults.

Crucially the generator VAE (step 3) is only a physics prior; LABELS come from
the oracle (step 2), never from the model being evaluated.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------------------------------- #
#  Step 1 - shuffle channels across modes
# --------------------------------------------------------------------------- #
def shuffle_candidates(X, modes, n_cand, swap_frac=0.4, contiguous=True, rng=None):
    """Build candidates by copying a block of channels from a donor window of a
    DIFFERENT mode into a source window. contiguous=True swaps an adjacent block
    (a 'subsystem'), which is more physically meaningful than scattered channels."""
    rng = rng or np.random.default_rng(0)
    n, d = X.shape
    k = max(1, int(round(swap_frac * d)))
    src = rng.integers(0, n, n_cand)
    out = X[src].copy()
    for i in range(n_cand):
        m = modes[src[i]]
        donor = int(rng.integers(0, n))
        for _ in range(8):                      # try to find a different-mode donor
            if modes[donor] != m:
                break
            donor = int(rng.integers(0, n))
        if contiguous:
            start = int(rng.integers(0, d - k + 1))
            ch = np.arange(start, start + k)
        else:
            ch = rng.choice(d, size=k, replace=False)
        out[i, ch] = X[donor, ch]
    return out


# --------------------------------------------------------------------------- #
#  Step 2 - detector-independent oracle (synthetic: true generative process)
# --------------------------------------------------------------------------- #
class ModeDistance:
    """Distance from a point to the nearest operating mode, in Mahalanobis sigma.

    Answers 'how far from the clusters is this candidate'. Uses per-mode MEANS
    with a single POOLED within-mode covariance (Ledoit-Wolf shrunk), so a rare
    4-sample mode still has a usable metric via the shared covariance. Built from
    the TRUE modes + labels, never the detector, so band-mining stays non-circular.
    """

    def __init__(self, x_normal, modes):
        from sklearn.covariance import LedoitWolf
        self.uniq = np.unique(modes)
        idx = {m: i for i, m in enumerate(self.uniq)}
        self.means = np.stack([x_normal[modes == m].mean(0) for m in self.uniq])
        resid = x_normal - self.means[np.array([idx[m] for m in modes])]
        self.lw = LedoitWolf().fit(resid)          # pooled within-mode covariance

    def min_sigma(self, X):
        X = np.asarray(X)
        d2 = np.stack([self.lw.mahalanobis(X - mu) for mu in self.means], axis=1)
        return np.sqrt(d2.min(1)), d2.argmin(1)


def in_range_mask(X, x_normal, q=0.999):
    """True where every channel of X lies within the normal per-channel support
    [q_lo, q_hi] - the 'each variable has a real value' physical-validity check."""
    lo = np.quantile(x_normal, 1 - q, axis=0)
    hi = np.quantile(x_normal, q, axis=0)
    return np.all((X >= lo) & (X <= hi), axis=1)


class TrueNormalOracle:
    """Labels a point by its kNN distance to a large TRUE-normal reference set.
    This uses only the true data-generating process, never the learned detector,
    so anomalies it certifies are not defined in the detector's own terms."""

    def __init__(self, x_true_normal, k=5):
        self.nn = NearestNeighbors(n_neighbors=k).fit(x_true_normal)
        self.k = k
        # calibration: how far is a genuine normal point from the reference?
        # Query points are IN the fitted set, so drop the self-match (dist 0)
        # by taking k+1 neighbours and discarding the first column - otherwise
        # normal_dist is biased low and the anomaly threshold is too permissive.
        d, _ = self.nn.kneighbors(x_true_normal[:4000], n_neighbors=k + 1)
        self.normal_dist = d[:, 1:].mean(1)

    def distance(self, X):
        d, _ = self.nn.kneighbors(np.asarray(X))
        return d.mean(1)

    def threshold(self, q=0.99):
        return float(np.quantile(self.normal_dist, q))

    def is_anomaly(self, X, q=0.99):
        return self.distance(X) > self.threshold(q)


# --------------------------------------------------------------------------- #
#  Step 3 - controllable difficulty via latent interpolation to nearest mode
# --------------------------------------------------------------------------- #
@torch.no_grad()
def control_difficulty(generator, X, alpha, device="cpu"):
    """Encode X, move each latent a fraction alpha toward its nearest cluster
    centre, decode. alpha=1 -> raw candidate (far), alpha=0 -> on the centroid
    (normal). Requires a generator with .encode, .decode and mu_c centroids
    (a trained VaDE). The generator is a PHYSICS PRIOR only, not a labeller."""
    xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    z = generator.encode(xt)[0]                          # (N,d)
    mu_c = generator.mu_c.detach()                       # (K,d)
    dist = torch.cdist(z, mu_c)                          # (N,K)
    near = mu_c[dist.argmin(1)]                          # (N,d) nearest centroid
    z_a = near + alpha * (z - near)
    return generator.decode(z_a).cpu().numpy()


def control_difficulty_input(X, x_normal, modes, alpha):
    """No-decode difficulty control: move each candidate a fraction alpha from
    its nearest true-mode centroid (in INPUT space). Wider, cleaner difficulty
    range than latent+decode, at the cost of the decoder's plausibility prior.
    alpha=0 -> on the centroid (normal), alpha=1 -> raw candidate."""
    uniq = [m for m in np.unique(modes)]
    cent = np.stack([x_normal[modes == m].mean(0) for m in uniq])
    from scipy.spatial.distance import cdist
    near = cent[cdist(X, cent).argmin(1)]
    return near + alpha * (X - near)


def generate_controlled_set(generator, x_true_normal, modes, oracle,
                            alphas=(0.4, 0.6, 0.8, 1.0), n_cand=3000,
                            swap_frac=0.4, keep_q=0.99, seed=0, device="cpu"):
    """Full generator: shuffle -> (per alpha) control+decode -> oracle-verify.
    Returns, per alpha, the decoded anomalies that the oracle still certifies as
    off-manifold, plus their oracle distance (the realised difficulty)."""
    rng = np.random.default_rng(seed)
    cand = shuffle_candidates(x_true_normal, modes, n_cand, swap_frac, rng=rng)
    out = {}
    for a in alphas:
        x_a = control_difficulty(generator, cand, a, device=device)
        d = oracle.distance(x_a)
        keep = d > oracle.threshold(keep_q)
        out[a] = {"x": x_a[keep], "oracle_dist": d[keep],
                  "kept": int(keep.sum()), "n": n_cand}
    return out


# --------------------------------------------------------------------------- #
#  Demo: show alpha controls realised difficulty, and it is non-circular
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from data import make_miim
    from models_vade import train_vade
    from pyod.models.iforest import IForest

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, meta = make_miim(n_modes=40, seed=0)
    # oracle = fresh true-normal from the SAME modes/frame (non-circular label)
    oracle = TrueNormalOracle(meta["oracle_normal"], k=5)
    gen = train_vade(ds.x_train, n_clusters=40, latent_dim=10, epochs=60,
                     seed=0, device=device)              # physics prior only

    # a detector INDEPENDENT of the generator's labelling
    det = IForest(n_estimators=200, random_state=0).fit(ds.x_train)
    thr = np.quantile(det.decision_function(ds.x_train), 0.95)
    rng = np.random.default_rng(0)
    cand = shuffle_candidates(ds.x_train, ds.mode_train, 3000, swap_frac=0.4, rng=rng)

    def row(tag, x_a):
        d = oracle.distance(x_a)
        keep = d > oracle.threshold(0.99)
        rec = float((det.decision_function(x_a[keep]) > thr).mean()) if keep.any() else float("nan")
        print(f"{tag:>18}{d.mean():>13.3f}{keep.mean():>12.2f}{rec:>16.3f}")

    print(f"{'generator/alpha':>18}{'oracle_dist':>13}{'frac_anom':>12}{'IForest recall':>16}")
    print("-- latent + decode (physics prior; subtle boundary faults) --")
    for a in (0.3, 0.5, 0.7, 1.0):
        row(f"decode a={a}", control_difficulty(gen, cand, a, device=device))
    print("-- input space (wider difficulty ladder) --")
    for a in (0.2, 0.4, 0.7, 1.0):
        row(f"input a={a}", control_difficulty_input(cand, ds.x_train, ds.mode_train, a))
