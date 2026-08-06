"""
LDT Model C - the trajectory-conditional anomaly detector.

C is its OWN VaDE over the window, with B's trajectory context c_t injected by
FiLM (feature-wise linear modulation) inside the encoder:

    h' = gamma_F(c_t) (*) h + beta_F(c_t)

The FiLM generators are RESIDUAL / identity-initialised (gamma_F -> 1, beta_F ->
0) so at init conditioning is a no-op and cannot suppress a true off-manifold
score. To guard against the "explain-away" dual-of-dilution risk, the scale is a
softplus-based *non-negative* multiplier centred at 1 and the base score used for
detection is computed from the identity path; the context can raise but is kept
from fully suppressing the base off-manifold signal.

On top of the conditioned VaDE we run the scoring stack:
  Base : whitened residual + nearest-component latent NLL      (models_vade)
  C2   : basin-of-attraction agreement                         (component2)
  C3   : per-mode PCA min + supervised mode-subset member       (component3)
  C4   : per-mode TFAR thresholds                               (component4)
  C1   : channel-shuffle generated near-anomalies feed C3's member (generate)

Fusion -> one number: the supervised member consumes
[s_rec, s_lat, C2 features, per-mode PCA min] and emits a single score, which C4
calibrates per mode.

Ablation: passing c_t = 0 (context zeroed) gives the "C-alone" control.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier

from models_vade import LOG2PI, fit_residual_whitener, _recon_energy


# ------------------------------------------------------------------------- #
#  Conditioned VaDE (own weights; FiLM on c_t)
# ------------------------------------------------------------------------- #
class FiLMBlock(nn.Module):
    """Residual, identity-initialised FiLM. Scale is a non-negative multiplier
    centred at 1 (softplus around a zero-init pre-activation), shift starts at 0,
    so at init h' = h and context can modulate but not zero-out the signal."""

    def __init__(self, ctx_dim, feat_dim):
        super().__init__()
        self.to_scale = nn.Linear(ctx_dim, feat_dim)
        self.to_shift = nn.Linear(ctx_dim, feat_dim)
        nn.init.zeros_(self.to_scale.weight); nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight); nn.init.zeros_(self.to_shift.bias)
        self._sp0 = math.log(math.e - 1.0)   # softplus(x)=1 at this pre-activation

    def forward(self, h, c):
        # softplus(pre + sp0) = 1 at init -> identity scale
        scale = F.softplus(self.to_scale(c) + self._sp0)
        shift = self.to_shift(c)
        return scale * h + shift


class CondVaDE(nn.Module):
    """VaDE whose encoder hidden state is FiLM-modulated by c_t. Same mixture
    prior / anti-collapse structure as models_vade.VaDE, own weights."""

    def __init__(self, n_features, ctx_dim, latent_dim=8, hidden=(128, 64),
                 n_clusters=24):
        super().__init__()
        self.d = latent_dim
        self.K = n_clusters
        self.ctx_dim = ctx_dim
        h1, h2 = hidden
        self.enc1 = nn.Linear(n_features, h1)
        self.enc2 = nn.Linear(h1, h2)
        self.film1 = FiLMBlock(ctx_dim, h1)
        self.film2 = FiLMBlock(ctx_dim, h2)
        self.fc_mu = nn.Linear(h2, latent_dim)
        self.fc_logvar = nn.Linear(h2, latent_dim)
        dec = [latent_dim, h2, h1, n_features]
        layers = []
        for i in range(len(dec) - 1):
            layers.append(nn.Linear(dec[i], dec[i + 1]))
            if i < len(dec) - 2:
                layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*layers)
        self.pi_logit = nn.Parameter(torch.zeros(n_clusters))
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim) * 0.5)
        self.logvar_c = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self.logvar_floor = math.log(0.05)
        self.res_whitener = None

    def encode(self, x, c):
        h = torch.relu(self.film1(self.enc1(x), c))
        h = torch.relu(self.film2(self.enc2(h), c))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder(z)

    def _lvc(self):
        return torch.clamp(self.logvar_c, min=self.logvar_floor)

    def _log_pz_given_c(self, z):
        z_e = z.unsqueeze(1)
        mu = self.mu_c.unsqueeze(0)
        lv = self._lvc().unsqueeze(0)
        return -0.5 * (LOG2PI * self.d
                       + torch.sum(lv + (z_e - mu) ** 2 / torch.exp(lv), dim=2))

    def loss(self, x, c, beta=1.0, cov_reg=0.0):
        mu, logvar = self.encode(x, c)
        z = self.reparam(mu, logvar)
        x_hat = self.decode(z)
        recon = F.mse_loss(x_hat, x, reduction="none").sum(1)
        log_pi = F.log_softmax(self.pi_logit, dim=0)
        log_p_cz = log_pi.unsqueeze(0) + self._log_pz_given_c(z)
        gamma = torch.softmax(log_p_cz, dim=1)
        mu_e, lv_e = mu.unsqueeze(1), logvar.unsqueeze(1)
        muc, lvc = self.mu_c.unsqueeze(0), self._lvc().unsqueeze(0)
        inv = torch.exp(-lvc)
        term_a = 0.5 * torch.sum(gamma * torch.sum(
            lvc + torch.exp(lv_e) * inv + (mu_e - muc) ** 2 * inv, dim=2), dim=1)
        term_b = torch.sum(gamma * (torch.log(gamma + 1e-10) - log_pi.unsqueeze(0)), dim=1)
        term_c = -0.5 * torch.sum(1 + logvar, dim=1)
        cov_pen = cov_reg * torch.sum(torch.exp(-self._lvc()))
        return (recon + beta * (term_a + term_b + term_c)).mean() + cov_pen


def _loader_xc(x, c, batch=256, shuffle=True, device="cpu"):
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    ct = torch.as_tensor(c, dtype=torch.float32, device=device)
    idx = torch.randperm(len(xt)) if shuffle else torch.arange(len(xt))
    for i in range(0, len(xt), batch):
        j = idx[i:i + batch]
        yield xt[j], ct[j]


def train_cond_vade(x, c, n_clusters=24, latent_dim=8, hidden=(128, 64),
                    pretrain_epochs=12, epochs=25, lr=1e-3, seed=0, device="cpu",
                    warmup=8, cov_reg=1e-3, mix_lr_scale=0.1, verbose=False):
    """Train the conditioned VaDE: pretrain as a (conditioned) VAE, GMM-init the
    mixture on encoded means, then joint VaDE objective. c is the per-window
    context (N, ctx_dim)."""
    from sklearn.mixture import GaussianMixture
    torch.manual_seed(seed)
    model = CondVaDE(x.shape[1], c.shape[1], latent_dim, hidden, n_clusters).to(device)

    net_pre = (list(model.enc1.parameters()) + list(model.enc2.parameters())
               + list(model.film1.parameters()) + list(model.film2.parameters())
               + list(model.fc_mu.parameters()) + list(model.fc_logvar.parameters())
               + list(model.decoder.parameters()))
    opt = torch.optim.Adam(net_pre, lr=lr)
    for ep in range(pretrain_epochs):
        model.train()
        for xb, cb in _loader_xc(x, c, device=device):
            mu, logvar = model.encode(xb, cb)
            z = model.reparam(mu, logvar)
            recon = F.mse_loss(model.decode(z), xb, reduction="none").sum(1)
            kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)
            loss = (recon + kl).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        zt = model.encode(torch.as_tensor(x, dtype=torch.float32, device=device),
                          torch.as_tensor(c, dtype=torch.float32, device=device))[0].cpu().numpy()
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="diag",
                          reg_covar=1e-4, random_state=seed, n_init=3).fit(zt)
    with torch.no_grad():
        model.pi_logit.copy_(torch.log(torch.as_tensor(gmm.weights_ + 1e-8, dtype=torch.float32)))
        model.mu_c.copy_(torch.as_tensor(gmm.means_, dtype=torch.float32))
        model.logvar_c.copy_(torch.log(torch.as_tensor(gmm.covariances_ + 1e-6, dtype=torch.float32)))

    mix_params = [model.pi_logit, model.mu_c, model.logvar_c]
    mix_ids = {id(p) for p in mix_params}
    net_params = [p for p in model.parameters() if id(p) not in mix_ids]
    opt = torch.optim.Adam([
        {"params": net_params, "lr": lr},
        {"params": mix_params, "lr": lr * mix_lr_scale}])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    for ep in range(epochs):
        beta = min(1.0, (ep + 1) / max(1, warmup))
        model.train(); tot = 0.0
        for xb, cb in _loader_xc(x, c, device=device):
            loss = model.loss(xb, cb, beta=beta, cov_reg=cov_reg)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  [C] epoch {ep:2d}  beta {beta:.2f}  loss {tot:.1f}")
    model.eval()
    return model


# ------------------------------------------------------------------------- #
#  Read-outs on the conditioned VaDE (batched)
# ------------------------------------------------------------------------- #
@torch.no_grad()
def _encode_mu(model, x, c, batch=8192, device="cpu"):
    outs = []
    for s in range(0, len(x), batch):
        xb = torch.as_tensor(x[s:s + batch], dtype=torch.float32, device=device)
        cb = torch.as_tensor(c[s:s + batch], dtype=torch.float32, device=device)
        outs.append(model.encode(xb, cb)[0].cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def fit_whitener(model, x, c, device="cpu"):
    mu = _encode_mu(model, x, c, device=device)
    z = torch.as_tensor(mu, dtype=torch.float32, device=device)
    x_hat = model.decode(z).cpu().numpy()
    model.res_whitener = fit_residual_whitener(x - x_hat)


@torch.no_grad()
def base_scores(model, x, c, device="cpu", batch=8192):
    """Base = whitened residual energy (s_rec) + nearest-component latent NLL
    (s_lat = min_k d_k). Returns (s_rec, s_lat)."""
    s_rec, s_lat = [], []
    for s in range(0, len(x), batch):
        xb = torch.as_tensor(x[s:s + batch], dtype=torch.float32, device=device)
        cb = torch.as_tensor(c[s:s + batch], dtype=torch.float32, device=device)
        mu = model.encode(xb, cb)[0]
        x_hat = model.decode(mu)
        rec = _recon_energy(xb, x_hat, model.res_whitener)
        log_near = model._log_pz_given_c(mu).max(dim=1).values.cpu().numpy()
        s_rec.append(np.asarray(rec)); s_lat.append(-log_near)
    return np.concatenate(s_rec), np.concatenate(s_lat)


@torch.no_grad()
def assign_modes(model, x, c, device="cpu", batch=8192):
    out = []
    for s in range(0, len(x), batch):
        xb = torch.as_tensor(x[s:s + batch], dtype=torch.float32, device=device)
        cb = torch.as_tensor(c[s:s + batch], dtype=torch.float32, device=device)
        mu = model.encode(xb, cb)[0]
        out.append(model._log_pz_given_c(mu).argmax(1).cpu().numpy())
    return np.concatenate(out)


def basin_features_cond(model, x, c, restarts=4, steps=40, step_size=0.35,
                        pert=0.3, chunk=4096, device="cpu"):
    """C2 basin descent on the conditioned VaDE latent (context enters only via
    the encoder that produces z0; the potential uses the mixture prior)."""
    z0_all = torch.as_tensor(_encode_mu(model, x, c, device=device),
                             dtype=torch.float32, device=device)
    mu_c = model.mu_c.detach(); lvc = model._lvc().detach()
    N, d = z0_all.shape; K = mu_c.shape[0]
    agreement = np.zeros(N, np.float32)

    def potential(z):
        z_e = z.unsqueeze(1)
        logN = -0.5 * (LOG2PI * d + torch.sum(
            lvc.unsqueeze(0) + (z_e - mu_c.unsqueeze(0)) ** 2
            / torch.exp(lvc.unsqueeze(0)), dim=2))
        return -torch.logsumexp(logN, dim=1), logN

    for s in range(0, N, chunk):
        z0 = z0_all[s:s + chunk]; B = z0.shape[0]
        finals = torch.empty((restarts, B), dtype=torch.long, device=device)
        for r in range(restarts):
            z = (z0 + pert * torch.randn_like(z0)).detach().requires_grad_(True)
            for t in range(steps):
                U, _ = potential(z)
                g = torch.autograd.grad(U.sum(), z)[0]
                g = g / (g.norm(dim=1, keepdim=True) + 1e-8)
                lr = step_size * (1.0 - t / steps)
                z = (z - lr * g).detach().requires_grad_(True)
            with torch.no_grad():
                _, logN = potential(z); finals[r] = logN.argmax(1)
        fnp = finals.cpu().numpy()
        for i in range(B):
            agreement[s + i] = np.bincount(fnp[:, i], minlength=K).max() / restarts
    return agreement                                    # 1 - agreement = instability


# ------------------------------------------------------------------------- #
#  C3 per-mode PCA experts + supervised fuser; C4 lives in run_ldt via component4
# ------------------------------------------------------------------------- #
class PerModePCA:
    def __init__(self, x_train, assign, latent_dim=8, min_pts=10):
        self.experts = []
        n_feat = x_train.shape[1]
        self.modes = np.unique(assign)
        for c in self.modes:
            members = x_train[assign == c]
            if len(members) > min_pts:
                ncomp = int(min(latent_dim, len(members) - 1, n_feat))
                self.experts.append(PCA(n_components=ncomp).fit(members))
            else:
                self.experts.append(("mean", members.mean(0)))

    @staticmethod
    def _err(expert, X):
        if isinstance(expert, tuple):
            return ((X - expert[1]) ** 2).sum(1)
        Xr = expert.inverse_transform(expert.transform(X))
        return ((X - Xr) ** 2).sum(1)

    def min_score(self, X):
        M = np.stack([self._err(e, X) for e in self.experts], axis=1)
        return M.min(1)


def build_fuser(feats_normal, feats_anom, seed=0):
    """C3 supervised member = the fuser. Trains normal-vs-C1 on the fused feature
    matrix [s_rec, s_lat, 1-agreement, pca_min] and returns P(anomaly)."""
    X = np.concatenate([feats_normal, feats_anom])
    y = np.r_[np.zeros(len(feats_normal)), np.ones(len(feats_anom))]
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200,
                                         learning_rate=0.1, random_state=seed)
    clf.fit(X, y)
    return clf
