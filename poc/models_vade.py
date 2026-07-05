"""
Joint latent encoding + structure discovery (the proposal's foundational step),
implemented as VaDE (Variational Deep Embedding, Jiang et al. 2017): a VAE whose
latent prior is a Gaussian mixture, so the representation and the operating-mode
clusters are learned *together* rather than one after the other.

Also provides a plain VAE (standard N(0,I) prior) used for the SEQUENTIAL
ablation: encode first, then fit a GMM on the latent afterwards. Comparing the
two isolates the benefit of joint training, matching the 'joint or sequentially'
question in the brief.

Anomaly score (both variants, calibration-free): reconstruction error plus the
negative log-density of the latent under the (mixture) prior. The mixture-density
term is what flags a 'pocket' fault sitting between two modes even when its
reconstruction is fine, and what keeps a rare-but-valid mode from being alarmed.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

LOG2PI = math.log(2.0 * math.pi)


def _as_tensor(x, module):
    """Coerce numpy/tensor input to a float32 tensor on the module's device."""
    dev = next(module.parameters()).device
    return torch.as_tensor(x, dtype=torch.float32, device=dev)


def fit_residual_whitener(residuals):
    """Ledoit-Wolf shrinkage estimate of the reconstruction-residual covariance
    on NORMAL data. Its .mahalanobis(r) returns r^T Sigma^-1 r, i.e. the whitened
    residual energy - the proper Gaussian NLL of the residual, which respects the
    per-channel scale and cross-channel correlation that a plain sum-of-squares
    ignores."""
    from sklearn.covariance import LedoitWolf
    return LedoitWolf().fit(residuals)


def _recon_energy(x, x_hat, whitener):
    """Residual score: whitened (Mahalanobis) if a whitener is fitted, else the
    plain sum of squared residuals."""
    r = (x - x_hat).detach().cpu().numpy()
    if whitener is not None:
        return 0.5 * whitener.mahalanobis(r)      # 0.5 r^T Sigma^-1 r  (Gaussian NLL)
    return (r ** 2).sum(1)


def _mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class VaDE(nn.Module):
    def __init__(self, n_features, latent_dim=10, hidden=(128, 64), n_clusters=30):
        super().__init__()
        self.d = latent_dim
        self.K = n_clusters
        enc = [n_features, *hidden]
        self.encoder = _mlp(enc)
        self.fc_mu = nn.Linear(hidden[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden[-1], latent_dim)
        self.decoder = _mlp([latent_dim, *reversed(hidden), n_features])
        # Gaussian-mixture prior parameters (learned jointly with the networks).
        self.pi_logit = nn.Parameter(torch.zeros(n_clusters))
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim) * 0.5)
        self.logvar_c = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        # Variance floor: stops a component collapsing to a spike (and stops the
        # cluster-collapse where most components are abandoned). See train_vade.
        self.logvar_floor = math.log(0.05)
        self.res_whitener = None   # residual covariance model (set post-training)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder(z)

    def _lvc(self):
        """Component log-variances, floored."""
        return torch.clamp(self.logvar_c, min=self.logvar_floor)

    # ---- log N(z | mu_c, var_c) for every cluster: (B, K) ----
    def _log_pz_given_c(self, z):
        z_e = z.unsqueeze(1)                       # (B,1,d)
        mu = self.mu_c.unsqueeze(0)                # (1,K,d)
        lv = self._lvc().unsqueeze(0)              # (1,K,d)
        return -0.5 * (LOG2PI * self.d
                       + torch.sum(lv + (z_e - mu) ** 2 / torch.exp(lv), dim=2))

    def loss(self, x, beta=1.0, cov_reg=0.0):
        """VaDE negative ELBO (per-batch mean).

        beta    : warm-up weight on the clustering/KL terms (0->1). Keeping it
                  small early lets the encoder settle on the pretrained GMM init
                  before the mixture is pulled around, which is what prevents the
                  40-modes-collapse-to-5 failure.
        cov_reg : DAGMM-style penalty on tiny component variances (1/var), a
                  second guard against component collapse.
        """
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_hat = self.decode(z)
        recon = F.mse_loss(x_hat, x, reduction="none").sum(1)

        log_pi = F.log_softmax(self.pi_logit, dim=0)          # (K,)
        log_p_cz = log_pi.unsqueeze(0) + self._log_pz_given_c(z)  # (B,K)
        gamma = torch.softmax(log_p_cz, dim=1)                 # responsibilities

        mu_e, lv_e = mu.unsqueeze(1), logvar.unsqueeze(1)      # (B,1,d)
        muc, lvc = self.mu_c.unsqueeze(0), self._lvc().unsqueeze(0)
        inv = torch.exp(-lvc)
        term_a = 0.5 * torch.sum(gamma * torch.sum(
            lvc + torch.exp(lv_e) * inv + (mu_e - muc) ** 2 * inv, dim=2), dim=1)
        term_b = torch.sum(gamma * (torch.log(gamma + 1e-10) - log_pi.unsqueeze(0)), dim=1)
        term_c = -0.5 * torch.sum(1 + logvar, dim=1)
        cov_pen = cov_reg * torch.sum(torch.exp(-self._lvc()))
        return (recon + beta * (term_a + term_b + term_c)).mean() + cov_pen

    @torch.no_grad()
    def fit_residual_whitener(self, x):
        x = _as_tensor(x, self)
        r = (x - self.decode(self.encode(x)[0])).cpu().numpy()
        self.res_whitener = fit_residual_whitener(r)

    @torch.no_grad()
    def anomaly_score(self, x):
        """Nearest-mode score = whitened residual energy + nearest-component
        latent NLL. Both terms are negative log-likelihoods, so their sum is the
        joint NLL of the point. The latent term uses the CLOSEST component (not
        the pi-weighted mixture): a rare-but-valid mode must not be penalised
        merely for being rare - the proposal's central false-positive concern."""
        x = _as_tensor(x, self)
        mu, _ = self.encode(x)
        x_hat = self.decode(mu)
        recon = _recon_energy(x, x_hat, self.res_whitener)
        log_near = self._log_pz_given_c(mu).max(dim=1).values.cpu().numpy()
        return recon - log_near


class PlainVAE(nn.Module):
    """Standard VAE with an N(0, I) prior, for the sequential ablation."""

    def __init__(self, n_features, latent_dim=10, hidden=(128, 64)):
        super().__init__()
        self.d = latent_dim
        self.encoder = _mlp([n_features, *hidden])
        self.fc_mu = nn.Linear(hidden[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden[-1], latent_dim)
        self.decoder = _mlp([latent_dim, *reversed(hidden), n_features])
        self.gmm: GaussianMixture | None = None  # fit post-hoc on the latent
        self.res_whitener = None

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def loss(self, x, beta=1.0):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_hat = self.decoder(z)
        recon = F.mse_loss(x_hat, x, reduction="none").sum(1)
        kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)
        return (recon + beta * kl).mean()

    @torch.no_grad()
    def encode_mean(self, x):
        return self.encode(_as_tensor(x, self))[0].cpu().numpy()

    def fit_gmm(self, x, n_clusters, seed=0):
        """Sequential step: cluster the latent AFTER the VAE is trained."""
        self.eval()
        z = self.encode_mean(x)
        self.gmm = GaussianMixture(
            n_components=n_clusters, covariance_type="diag",
            reg_covar=1e-4, random_state=seed, n_init=1).fit(z)

    @torch.no_grad()
    def fit_residual_whitener(self, x):
        x = _as_tensor(x, self)
        r = (x - self.decoder(self.encode(x)[0])).cpu().numpy()
        self.res_whitener = fit_residual_whitener(r)

    @torch.no_grad()
    def anomaly_score(self, x):
        x = _as_tensor(x, self)
        mu, _ = self.encode(x)
        x_hat = self.decoder(mu)
        recon = _recon_energy(x, x_hat, self.res_whitener)
        z = mu.cpu().numpy()
        # Nearest-component log-density (unweighted), same rare-mode-safe rule
        # as VaDE, so the joint-vs-sequential comparison is on equal footing.
        means, covs = self.gmm.means_, self.gmm.covariances_       # (K,d) diag
        d = z.shape[1]
        diff2 = (z[:, None, :] - means[None]) ** 2                 # (N,K,d)
        log_comp = -0.5 * (d * LOG2PI
                           + np.sum(np.log(covs)[None] + diff2 / covs[None], axis=2))
        log_near = log_comp.max(axis=1)
        return recon - log_near


# --------------------------------------------------------------------------- #
#  Training helpers
# --------------------------------------------------------------------------- #
def _loader(x, batch=256, shuffle=True, device="cpu"):
    t = torch.as_tensor(x, dtype=torch.float32, device=device)
    idx = torch.randperm(len(t)) if shuffle else torch.arange(len(t))
    for i in range(0, len(t), batch):
        yield t[idx[i:i + batch]]


def train_vade(x, n_clusters, latent_dim=10, hidden=(128, 64),
               pretrain_epochs=30, epochs=60, lr=1e-3, seed=0, device="cpu",
               warmup=15, cov_reg=1e-3, mix_lr_scale=0.1, verbose=False):
    """Standard VaDE recipe: pretrain as a plain VAE, initialise the mixture
    from a GMM on the latent, then train the full joint objective.

    Anti-collapse measures (the fix for '40 modes collapse to 5 components'):
      warmup       : anneal the clustering/KL weight beta 0->1 over this many
                     epochs, so the encoder aligns to the GMM init first.
      cov_reg      : penalty on tiny component variances (DAGMM-style).
      mix_lr_scale : mixture params (pi, mu_c, logvar_c) learn slower than the
                     networks, so the good 40-cluster init is not destroyed.
    """
    torch.manual_seed(seed)
    model = VaDE(x.shape[1], latent_dim, hidden, n_clusters).to(device)

    # ---- pretrain (plain VAE) so the latent is meaningful before clustering ----
    opt = torch.optim.Adam(list(model.encoder.parameters())
                           + list(model.fc_mu.parameters())
                           + list(model.fc_logvar.parameters())
                           + list(model.decoder.parameters()), lr=lr)
    for ep in range(pretrain_epochs):
        model.train()
        for xb in _loader(x, device=device):
            mu, logvar = model.encode(xb)
            z = model.reparam(mu, logvar)
            recon = F.mse_loss(model.decode(z), xb, reduction="none").sum(1)
            kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)
            loss = (recon + kl).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    # ---- initialise the mixture prior from a GMM on encoded means ----
    model.eval()
    with torch.no_grad():
        zt = model.encode(torch.as_tensor(x, dtype=torch.float32, device=device))[0].cpu().numpy()
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="diag",
                          reg_covar=1e-4, random_state=seed, n_init=3).fit(zt)
    with torch.no_grad():
        model.pi_logit.copy_(torch.log(torch.as_tensor(gmm.weights_ + 1e-8, dtype=torch.float32)))
        model.mu_c.copy_(torch.as_tensor(gmm.means_, dtype=torch.float32))
        model.logvar_c.copy_(torch.log(torch.as_tensor(gmm.covariances_ + 1e-6, dtype=torch.float32)))

    # ---- joint training of the full VaDE objective ----
    # Identity-based split (tensor `in set` compares by __eq__ -> ambiguous).
    mix_params = [model.pi_logit, model.mu_c, model.logvar_c]
    mix_ids = {id(p) for p in mix_params}
    net_params = [p for p in model.parameters() if id(p) not in mix_ids]
    opt = torch.optim.Adam([
        {"params": net_params, "lr": lr},
        {"params": mix_params, "lr": lr * mix_lr_scale},
    ])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    for ep in range(epochs):
        beta = min(1.0, (ep + 1) / max(1, warmup))   # KL/cluster warm-up
        model.train()
        tot = 0.0
        for xb in _loader(x, device=device):
            loss = model.loss(xb, beta=beta, cov_reg=cov_reg)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  [vade] epoch {ep:3d}  beta {beta:.2f}  loss {tot:.1f}")
    model.eval()
    return model


def train_plain_vae(x, latent_dim=10, hidden=(128, 64), epochs=60, lr=1e-3,
                    seed=0, device="cpu"):
    torch.manual_seed(seed)
    model = PlainVAE(x.shape[1], latent_dim, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for xb in _loader(x, device=device):
            loss = model.loss(xb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model
