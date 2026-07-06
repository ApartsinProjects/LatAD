"""
LDT Model A - the frozen mode encoder.

A thin wrapper over the existing VaDE (models_vade.py). A owns the mode
*vocabulary*: it is trained on normal windows only, then FROZEN. Per window it
emits the three quantities the spec names:

  gamma in Delta^K : responsibility vector (soft mode membership) - B's input token
  d in R^K         : mode-distance vector (Mahalanobis to each component) - C's geometry
  pi in Delta^K    : mode prior pi_k = E[gamma_k] over the training set - rarity term

Nothing here re-implements VaDE; A just standardises the read-out interface and
the freeze so B's vocabulary cannot drift.
"""

from __future__ import annotations

import numpy as np
import torch

from models_vade import train_vade, _as_tensor


class ModeEncoderA:
    """Frozen VaDE + the (gamma, d, pi) read-out interface."""

    def __init__(self, vade, pi):
        self.vade = vade
        self.K = vade.K
        self.pi = np.asarray(pi, dtype=np.float64)          # mode prior E[gamma]
        self.device = next(vade.parameters()).device

    # --------------------------------------------------------------------- #
    @classmethod
    def fit(cls, x_train, n_clusters=24, latent_dim=8, hidden=(128, 64),
            pretrain_epochs=15, epochs=30, warmup=8, seed=0, device="cpu",
            verbose=False):
        """Train VaDE on normal windows, fit the residual whitener, compute pi,
        then freeze."""
        vade = train_vade(x_train, n_clusters=n_clusters, latent_dim=latent_dim,
                          hidden=hidden, pretrain_epochs=pretrain_epochs,
                          epochs=epochs, warmup=warmup, seed=seed, device=device,
                          verbose=verbose)
        vade.fit_residual_whitener(x_train)
        # mode prior = mean responsibility over the training set
        enc = cls(vade, pi=np.ones(vade.K) / vade.K)
        gamma_tr, _ = enc.encode(x_train)
        enc.pi = gamma_tr.mean(0).astype(np.float64)
        enc.pi = enc.pi / enc.pi.sum()
        enc.freeze()
        return enc

    # --------------------------------------------------------------------- #
    def freeze(self):
        self.vade.eval()
        for p in self.vade.parameters():
            p.requires_grad_(False)
        return self

    @torch.no_grad()
    def encode(self, x, batch=8192):
        """Return (gamma, d) for a batch of windows.

        gamma : (N, K) responsibility softmax over -d (with the mixture prior).
        d     : (N, K) Mahalanobis (diagonal) distance to each component.
        """
        v = self.vade
        lv = v._lvc()                                     # (K, d) floored logvar
        log_pi = torch.log_softmax(v.pi_logit, dim=0)     # (K,)
        gammas, dists = [], []
        N = len(x)
        for s in range(0, N, batch):
            xb = _as_tensor(x[s:s + batch], v)
            mu = v.encode(xb)[0]                           # (B, d) latent mean
            z_e = mu.unsqueeze(1)                          # (B, 1, d)
            muc = v.mu_c.unsqueeze(0)                      # (1, K, d)
            inv = torch.exp(-lv).unsqueeze(0)              # (1, K, d)
            d = torch.sum((z_e - muc) ** 2 * inv, dim=2)   # (B, K) Mahalanobis
            log_p_cz = log_pi.unsqueeze(0) + v._log_pz_given_c(mu)
            gamma = torch.softmax(log_p_cz, dim=1)
            gammas.append(gamma.cpu().numpy())
            dists.append(d.cpu().numpy())
        return np.concatenate(gammas), np.concatenate(dists)

    @torch.no_grad()
    def centroids(self):
        """A's latent centroids mu_c (K, d) - the init for B's mode embeddings."""
        return self.vade.mu_c.detach().cpu().numpy()
