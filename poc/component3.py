"""
Component 3 - the cluster-based ensemble (the proposal's false-negative fix).

A single reconstruction model trained on all modes can interpolate a smooth
"normal" bridge across the gap between two modes, so a fault sitting in that gap
reconstructs well and is missed (the over-interpolation false negative). Instead
we fit ONE reconstruction expert PER discovered mode (input-space PCA on that
mode's points only). Each expert knows only its own manifold, so none of them
can reconstruct a between-modes point.

Score = MIN reconstruction error over experts ("does ANY single mode explain
this point?"). Normal -> its own mode's expert explains it (min low). Pocket ->
no expert explains it (min high). The owner-gap (2nd-min minus min) adds pocket
sensitivity: a normal point has one clear owner (large gap); a pocket is equally
poorly explained by its two neighbours (small gap). Compared against a single
GLOBAL PCA expert to show the ensemble catches what one model interpolates over.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier

from models_vade import _as_tensor


def _assign(vade, x):
    with torch.no_grad():
        z = vade.encode(_as_tensor(x, vade))[0]
        return vade._log_pz_given_c(z).argmax(1).cpu().numpy()


class ClusterEnsemble:
    def __init__(self, vade, x_train, latent_dim=10, min_pts=10):
        with torch.no_grad():
            z = vade.encode(_as_tensor(x_train, vade))[0]
            assign = vade._log_pz_given_c(z).argmax(1).cpu().numpy()  # nearest mode
        self.assign = assign
        n_feat = x_train.shape[1]
        # one reconstruction expert per mode (rare modes -> mean-only expert)
        self.experts = []
        for c in np.unique(assign):
            members = x_train[assign == c]
            if len(members) > min_pts:
                ncomp = int(min(latent_dim, len(members) - 1, n_feat))
                self.experts.append(PCA(n_components=ncomp).fit(members))
            else:
                self.experts.append(("mean", members.mean(0)))
        # single global expert (the over-interpolating baseline)
        self.global_pca = PCA(n_components=int(min(latent_dim, n_feat))).fit(x_train)

    @staticmethod
    def _recon_err(expert, X):
        if isinstance(expert, tuple):                 # degenerate mean expert
            return ((X - expert[1]) ** 2).sum(1)
        Xr = expert.inverse_transform(expert.transform(X))
        return ((X - Xr) ** 2).sum(1)

    def expert_matrix(self, X):
        return np.stack([self._recon_err(e, X) for e in self.experts], axis=1)  # (N,K)

    def scores(self, X):
        """min reconstruction error over experts, and the owner-gap."""
        M = np.sort(self.expert_matrix(np.asarray(X)), axis=1)
        return M[:, 0], (M[:, 1] - M[:, 0])           # min_err, owner_gap

    def global_score(self, X):
        return self._recon_err(self.global_pca, np.asarray(X))


class ModeSubsetEnsemble:
    """Redesigned C3 (addresses 'one expert per mode is too many models').

    Each of M ensemble members is trained on the normal points of a random SUBSET
    of the discovered modes (the negatives) versus the C1-generated anomalies (the
    positives) - i.e. C1 turns detection into a SUPERVISED problem. Members differ
    in which modes they treat as normal, giving diversity; the ensemble averages
    their anomaly probabilities. M << K (number of modes), and the members are
    binary classifiers rather than reconstruction experts.

    Non-circularity is the CALLER's responsibility: members are trained on the C1
    anomalies passed as x_anom, so the test faults must be a DIFFERENT family (e.g.
    OOD / pockets) or a disjoint held-out C1 split. This class does not itself hold
    out a C1 split; do not evaluate it on the same generated pool it trained on.
    """

    def __init__(self, vade, x_train, x_anom, n_members=10, frac_modes=0.5,
                 max_per_class=6000, seed=0):
        x_anom = np.asarray(x_anom)
        if len(x_anom) < 10:
            raise ValueError("ModeSubsetEnsemble needs C1 anomalies to train on")
        assign = _assign(vade, x_train)
        modes = np.unique(assign)
        rng = np.random.default_rng(seed)
        self.members = []
        for j in range(n_members):
            k = max(1, int(round(frac_modes * len(modes))))
            S = rng.choice(modes, size=k, replace=False)
            neg = x_train[np.isin(assign, S)]
            # balance the two classes per member so probabilities are comparable
            n_bal = min(len(neg), len(x_anom), max_per_class)
            neg = _sub(neg, n_bal, rng)
            pos = _sub(x_anom, n_bal, rng)
            X = np.concatenate([neg, pos])
            y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
            clf = HistGradientBoostingClassifier(
                max_depth=4, max_iter=150, learning_rate=0.1,
                random_state=seed + j).fit(X, y)
            self.members.append(clf)

    def score(self, X):
        X = np.asarray(X)
        return np.mean([m.predict_proba(X)[:, 1] for m in self.members], axis=0)


def _sub(A, n, rng):
    return A if len(A) <= n else A[rng.choice(len(A), n, replace=False)]


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    from data import make_miim
    from models_vade import train_vade
    from generate import (TrueNormalOracle, shuffle_candidates,
                          control_difficulty_input, in_range_mask)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, meta = make_miim(n_modes=40, seed=0)
    vade = train_vade(ds.x_train, n_clusters=40, latent_dim=10, epochs=60,
                      seed=0, device=device)
    y = ds.y_test
    normal = y == 0
    pkt = ds.atype_test == "pocket"
    ood = ds.atype_test == "ood"

    def auroc(mask_pos, score):
        m = normal | mask_pos
        return roc_auc_score(y[m], score[m])

    # ---- per-mode reconstruction ensemble vs single global ----
    ens = ClusterEnsemble(vade, ds.x_train)
    min_s, _ = ens.scores(ds.x_test)
    glob = ens.global_score(ds.x_test)
    print("per-mode reconstruction ensemble vs single global model:")
    print(f"{'target':<10}{'single-global':>15}{'per-mode-ens':>15}")
    for tag, mask in [("pocket", pkt), ("ood", ood)]:
        print(f"  {tag:<8}{auroc(mask, glob):>15.3f}{auroc(mask, min_s):>15.3f}")

    # ---- supervised mode-subset ensemble: train on C1, test on OOD/pocket ----
    oracle = TrueNormalOracle(meta["oracle_normal"])
    rng = np.random.default_rng(0)
    cand = shuffle_candidates(ds.x_train, ds.mode_train, 6000, swap_frac=0.4, rng=rng)
    pool = np.concatenate([control_difficulty_input(cand, ds.x_train, ds.mode_train, a)
                           for a in (0.3, 0.5, 0.7, 1.0)])
    x_anom = pool[in_range_mask(pool, ds.x_train) & oracle.is_anomaly(pool)]
    sup = ModeSubsetEnsemble(vade, ds.x_train, x_anom, n_members=10, seed=0)
    ssc = sup.score(ds.x_test)
    print(f"\nsupervised mode-subset ensemble (10 members, trained on C1 anomalies,"
          f" tested on unseen fault families):")
    for tag, mask in [("pocket", pkt), ("ood", ood)]:
        print(f"  {tag:<8}{'AUROC':>8} {auroc(mask, ssc):.3f}")
