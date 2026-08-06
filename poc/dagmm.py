"""DAGMM prototype (Deep Autoencoding Gaussian Mixture Model, Zong et al. ICLR 2018).

A compression autoencoder produces a low-dim code z_c; two reconstruction-error features
(relative-euclidean, cosine) are appended -> z = [z_c, z_r]. An estimation network maps z to
soft GMM membership gamma; the GMM (phi, mu, Sigma) is computed from batch statistics, and the
anomaly score is the SAMPLE ENERGY E(z) = -log sum_k phi_k N(z; mu_k, Sigma_k). Training jointly
minimises  reconstruction + lambda1*mean(energy) + lambda2*cov_penalty, so the latent is SHAPED
into well-separated Gaussians (the 'shape the clusters to be Gaussian' idea) and the per-mode
Mahalanobis/energy is learned end-to-end rather than fit post-hoc.
"""
from __future__ import annotations
import math, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes, act=nn.Tanh, last_act=False):
    L = []
    for i in range(len(sizes) - 1):
        L.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            L.append(act())
    return nn.Sequential(*L)


class DAGMM(nn.Module):
    def __init__(self, n_features, zc_dim=4, n_gmm=10, enc=(128, 64), est=(32,)):
        super().__init__()
        self.encoder = _mlp([n_features, *enc, zc_dim])
        self.decoder = _mlp([zc_dim, *reversed(enc), n_features])
        self.zd = zc_dim + 2                                   # z_c + [rel-euclid, cosine]
        self.estim = nn.Sequential(_mlp([self.zd, *est], last_act=True),
                                   nn.Dropout(0.3), nn.Linear(est[-1], n_gmm), nn.Softmax(dim=1))
        self.K = n_gmm
        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, self.zd))
        self.register_buffer("cov", torch.zeros(n_gmm, self.zd, self.zd))

    def recon_features(self, x, xh):
        re = torch.norm(x - xh, dim=1) / (torch.norm(x, dim=1) + 1e-10)
        cos = F.cosine_similarity(x, xh, dim=1)
        return torch.stack([re, cos], dim=1)

    def forward(self, x):
        zc = self.encoder(x); xh = self.decoder(zc)
        z = torch.cat([zc, self.recon_features(x, xh)], dim=1)
        gamma = self.estim(z)
        return xh, z, gamma

    def gmm_params(self, z, gamma):
        N = gamma.sum(0)                                       # (K,)
        phi = N / z.shape[0]
        mu = (gamma.t() @ z) / (N.unsqueeze(1) + 1e-10)        # (K, zd)
        cov = torch.zeros(self.K, self.zd, self.zd, device=z.device)
        for k in range(self.K):
            d = z - mu[k]; w = gamma[:, k].unsqueeze(1)
            cov[k] = (w * d).t() @ d / (N[k] + 1e-10)
        return phi, mu, cov

    def energy(self, z, phi, mu, cov):
        eps = 1e-4 * torch.eye(self.zd, device=z.device)
        logs = []
        for k in range(self.K):
            Lk = torch.linalg.cholesky(cov[k] + eps)
            diff = (z - mu[k]).t()                              # (zd, N)
            sol = torch.linalg.solve_triangular(Lk, diff, upper=False)
            maha = (sol ** 2).sum(0)                            # (N,)
            logdet = 2 * torch.log(torch.diagonal(Lk)).sum()
            logs.append(torch.log(phi[k] + 1e-10) - 0.5 * (maha + self.zd * math.log(2 * math.pi) + logdet))
        return -torch.logsumexp(torch.stack(logs, dim=1), dim=1)   # (N,) sample energy

    def cov_penalty(self, cov):
        eps = 1e-4 * torch.eye(self.zd, device=cov.device)
        return sum(torch.diagonal(torch.linalg.inv(cov[k] + eps)).sum() for k in range(self.K))


def train_dagmm(X, zc_dim=4, n_gmm=4, epochs=200, lr=1e-3, l1=0.1, l2=0.005, l_ent=0.0,
                seed=0, device="cpu"):
    """l_ent>0 penalises collapse (entropy of the batch-average gamma). Input is z-scored per
    feature (essential: raw stats-feature scales saturate the Tanh AE)."""
    torch.manual_seed(seed)
    X = np.asarray(X, np.float32); _mu = X.mean(0); _sd = X.std(0) + 1e-8; X = (X - _mu) / _sd
    m = DAGMM(X.shape[1], zc_dim, n_gmm).to(device)
    m.in_mu = torch.as_tensor(_mu, device=device); m.in_sd = torch.as_tensor(_sd, device=device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    n = len(Xt); bs = 1024
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            xb = Xt[perm[i:i + bs]]
            xh, z, gamma = m(xb)
            phi, mu, cov = m.gmm_params(z, gamma)
            e = m.energy(z, phi, mu, cov)
            gm = gamma.mean(0); ent = -(gm * torch.log(gm + 1e-10)).sum()   # high = all comps used
            loss = F.mse_loss(xh, xb) + l1 * e.mean() + l2 * m.cov_penalty(cov) - l_ent * ent
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    # freeze GMM params on the full train set
    m.eval()
    with torch.no_grad():
        _, z, gamma = m(Xt); phi, mu, cov = m.gmm_params(z, gamma)
        m.phi.copy_(phi); m.mu.copy_(mu); m.cov.copy_(cov)
    return m


@torch.no_grad()
def dagmm_score(m, X, device="cpu"):
    X = (np.asarray(X, np.float32) - m.in_mu.cpu().numpy()) / m.in_sd.cpu().numpy()   # match train z-score
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(Xt), 4096):
        _, z, _ = m(Xt[i:i + 4096])
        out.append(m.energy(z, m.phi, m.mu, m.cov).cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    from winfeat import window_features
    import eda_real as E
    W, ST = 60, 30
    def win(Xx, yy=None):
        a, b = [], []
        for i in range(0, len(Xx) - W + 1, ST):
            a.append(window_features(Xx[i:i + W], "stats"))
            if yy is not None:
                b.append(int(yy[i:i + W].mean() > 0.05))
        return np.asarray(a, np.float32), (np.asarray(b, int) if yy is not None else None)
    D = E.load("WADI"); Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy
    m = train_dagmm(Xtr, zc_dim=4, n_gmm=10, epochs=100, seed=0)
    s = dagmm_score(m, Xte)
    def au(mask):
        k = (yw == 0) | mask; return roc_auc_score(yw[k], s[k])
    print(f"DAGMM WADI:  ALL {au(yw==1):.3f}   EASY {au(easy):.3f}   HARD {au(hard):.3f}")
