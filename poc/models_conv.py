"""
Path B: a 1-D convolutional VaDE for real time-series windows.

Replaces the MLP encoder/decoder with a 1-D CNN over time (dilated convolutions =
multiple receptive fields = multiple time scales, the architectural realization of
assumption A8). The Gaussian-mixture prior, ELBO, anti-collapse recipe, anomaly
score, and every component (C1-C4) are inherited from VaDE unchanged - only the
encoder/decoder differ. Input is the flattened window (N, W*C) the loaders already
produce; the model reshapes to (N, C, W) internally, so the framework and the
CPSDataset interface are untouched.

Applies to windowed real data (SKAB, HAI); the static synthetic benchmark keeps
the MLP VaDE (it has no time axis to convolve over).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

from models_vade import VaDE, _loader


class ConvVaDE(VaDE):
    def __init__(self, W, C, latent_dim=10, hidden_ch=64, n_clusters=20):
        nn.Module.__init__(self)
        self.d, self.K, self.W, self.C, self.ch = latent_dim, n_clusters, W, C, hidden_ch
        self.logvar_floor = math.log(0.05)
        self.res_whitener = None
        # Gaussian-mixture prior (same as VaDE)
        self.pi_logit = nn.Parameter(torch.zeros(n_clusters))
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim) * 0.5)
        self.logvar_c = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        # 1-D conv encoder over time; last layer dilated for multi-scale (A8)
        self.enc_conv = nn.Sequential(
            nn.Conv1d(C, hidden_ch, 5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden_ch, hidden_ch, 5, padding=2, stride=2), nn.ReLU(),
            nn.Conv1d(hidden_ch, hidden_ch, 3, padding=2, dilation=2), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_ch, latent_dim)
        self.fc_logvar = nn.Linear(hidden_ch, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, hidden_ch * W)
        self.dec_conv = nn.Sequential(
            nn.Conv1d(hidden_ch, hidden_ch, 5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden_ch, C, 5, padding=2),
        )

    def encode(self, x):
        x = x.view(-1, self.W, self.C).transpose(1, 2)     # (N, C, W)
        h = self.enc_conv(x).mean(-1)                      # global average pool -> (N, ch)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.dec_fc(z).view(-1, self.ch, self.W)
        x = self.dec_conv(h)                               # (N, C, W)
        return x.transpose(1, 2).reshape(x.size(0), -1)    # -> (N, W*C), matches input


def train_conv_vade(x, W, C, n_clusters=20, latent_dim=10, hidden_ch=64,
                    pretrain_epochs=30, epochs=60, lr=1e-3, warmup=15,
                    cov_reg=1e-3, mix_lr_scale=0.1, seed=0, device="cpu", verbose=False):
    torch.manual_seed(seed)
    model = ConvVaDE(W, C, latent_dim, hidden_ch, n_clusters).to(device)
    mix_ids = {id(model.pi_logit), id(model.mu_c), id(model.logvar_c)}
    net = [p for p in model.parameters() if id(p) not in mix_ids]

    # ---- pretrain as a plain VAE ----
    opt = torch.optim.Adam(net, lr=lr)
    for ep in range(pretrain_epochs):
        model.train()
        for xb in _loader(x, device=device):
            mu, logvar = model.encode(xb)
            z = model.reparam(mu, logvar)
            recon = F.mse_loss(model.decode(z), xb, reduction="none").sum(1)
            kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)
            loss = (recon + kl).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    # ---- GMM init of the mixture from encoded means ----
    model.eval()
    with torch.no_grad():
        zt = model.encode(torch.as_tensor(x, dtype=torch.float32, device=device))[0].cpu().numpy()
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="diag",
                          reg_covar=1e-4, random_state=seed, n_init=3).fit(zt)
    with torch.no_grad():
        model.pi_logit.copy_(torch.log(torch.as_tensor(gmm.weights_ + 1e-8, dtype=torch.float32)))
        model.mu_c.copy_(torch.as_tensor(gmm.means_, dtype=torch.float32))
        model.logvar_c.copy_(torch.log(torch.as_tensor(gmm.covariances_ + 1e-6, dtype=torch.float32)))

    # ---- joint VaDE training (inherited loss) ----
    opt = torch.optim.Adam([{"params": net, "lr": lr},
                            {"params": [model.pi_logit, model.mu_c, model.logvar_c],
                             "lr": lr * mix_lr_scale}])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    for ep in range(epochs):
        beta = min(1.0, (ep + 1) / max(1, warmup))
        model.train()
        for xb in _loader(x, device=device):
            loss = model.loss(xb, beta=beta, cov_reg=cov_reg)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if verbose and ep % 10 == 0:
            print(f"  [conv-vade] epoch {ep} beta {beta:.2f} loss {loss.item():.1f}")
    model.eval()
    return model
