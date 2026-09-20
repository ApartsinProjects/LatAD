"""Side check surfaced by the space sweep: a post-hoc GMM refit on SKAB's OWN VaDE latent reads H=0.05 while
VaDE's own responsibilities read 0.29 on the same points. Test whether VaDE's component variance floor
(logvar_floor = log 0.05, sd 0.224) is what broadens SKAB's components: (a) fraction of component-dims
sitting at the floor, (b) VaDE's sd vs the empirical within-component sd, (c) H with VaDE's means/pi but
empirical variances, (d) refit GMM with the same floor imposed. WADI cached latent as the benchmark contrast.
Invariant: if the floor drives SKAB's read, (c) collapses to the refit value and (d) reproduces ~0.3.
"""
import sys, os, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from fable_a3_spaces_lib import H_norm
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
import eda_real as E
from models_vade import train_vade
LOG2PI = np.log(2 * np.pi)


def resp(z, mu, lv, lp):
    logN = -0.5 * (LOG2PI * z.shape[1] + (lv[None] + (z[:, None, :] - mu[None]) ** 2 / np.exp(lv[None])).sum(2))
    l = lp[None] + logN; return np.exp(l - logsumexp(l, 1, keepdims=True))


X = np.asarray(E.load("SKAB")["Xn_w"], np.float32); X = (X - X.mean(0)) / (X.std(0) + 1e-8)
for seed in (0, 1):
    v = train_vade(X, n_clusters=16, latent_dim=6, epochs=40, warmup=8, seed=seed, device="cpu")
    with torch.no_grad():
        z = v.encode(torch.as_tensor(X))[0].numpy().astype(np.float64)
    G = v._responsibilities(X)
    lvc = v._lvc().detach().numpy(); mu = v.mu_c.detach().numpy(); lp = torch.log_softmax(v.pi_logit, 0).detach().numpy()
    at_floor = (np.abs(lvc - np.log(0.05)) < 1e-6).mean()
    g = GaussianMixture(16, covariance_type="diag", reg_covar=1e-4, random_state=0).fit(z)
    Hg = H_norm(g.predict_proba(z)).mean()
    Hf = H_norm(resp(z, g.means_, np.log(np.maximum(g.covariances_, 0.05)), np.log(g.weights_))).mean()
    lab = G.argmax(1)
    var_hat = np.stack([z[lab == k].var(0) + 1e-4 if (lab == k).sum() > 1 else np.exp(lvc[k]) for k in range(16)])
    Hv = H_norm(resp(z, mu, np.log(var_hat), lp)).mean()
    print(f"SKAB seed{seed}: VaDE own H={H_norm(G).mean():.3f} | frac comp-dims at variance floor={at_floor:.2f} | "
          f"mean VaDE comp sd={np.exp(0.5 * lvc).mean():.3f} vs empirical within-comp sd={np.sqrt(var_hat).mean():.3f}", flush=True)
    print(f"   refit GMM on same latent H={Hg:.3f} | refit GMM with var floor 0.05 H={Hf:.3f} | VaDE means+pi with empirical vars H={Hv:.3f}", flush=True)
e = np.load(os.path.join(HERE, "e2_fable_WADI.npz")); z = e["ztr"].astype(np.float64)
G = resp(z, e["mu_c"], e["lvc"], e["logpi"]); lab = G.argmax(1)
var_hat = np.stack([z[lab == k].var(0) + 1e-4 if (lab == k).sum() > 1 else np.exp(e["lvc"][k]) for k in range(20)])
print(f"WADI: VaDE own H={H_norm(G).mean():.3f} | frac at floor={(np.abs(e['lvc'] - np.log(0.05)) < 1e-6).mean():.2f} | "
      f"comp sd={np.exp(0.5 * e['lvc']).mean():.3f} vs empirical {np.sqrt(var_hat).mean():.3f} | "
      f"empirical-var H={H_norm(resp(z, e['mu_c'], np.log(var_hat), e['logpi'])).mean():.3f}", flush=True)
