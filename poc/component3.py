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

from models_vade import _as_tensor


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


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    from data import make_miim
    from models_vade import train_vade

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, meta = make_miim(n_modes=40, seed=0)
    vade = train_vade(ds.x_train, n_clusters=40, latent_dim=10, epochs=60,
                      seed=0, device=device)
    ens = ClusterEnsemble(vade, ds.x_train)

    y = ds.y_test
    normal = y == 0
    pkt = ds.atype_test == "pocket"
    ood = ds.atype_test == "ood"
    min_s, gap = ens.scores(ds.x_test)
    glob = ens.global_score(ds.x_test)

    def auroc(mask_pos, score):
        m = normal | mask_pos
        return roc_auc_score(y[m], score[m])

    print(f"{'target':<10}{'single-global':>15}{'ensemble-min':>15}{'ens min+gap':>14}")
    ens_fused = (min_s - min_s[normal].mean()) / (min_s[normal].std() + 1e-9) \
        - (gap - gap[normal].mean()) / (gap[normal].std() + 1e-9)
    for tag, mask in [("pocket", pkt), ("ood", ood)]:
        print(f"{tag:<10}{auroc(mask, glob):>15.3f}{auroc(mask, min_s):>15.3f}"
              f"{auroc(mask, ens_fused):>14.3f}")
