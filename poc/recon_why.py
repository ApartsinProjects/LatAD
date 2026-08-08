"""Why does reconstruction miss correlation-breaks that density catches? Test reachability-vs-
probability on WADI. (1) On the real difficult anomalies: AUROC of reconstruction error vs latent
density, plus whether the anomalies are ON the normal manifold (low PCA off-subspace residual) yet
BETWEEN modes (high GMM responsibility entropy). (2) Controlled probe: build synthetic BETWEEN-MODE
points by averaging two normal windows from DIFFERENT modes (an improbable-but-reachable combination)
and show reconstruction error stays LOW while density drops -- the decoder interpolates across modes.
"""
from __future__ import annotations
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy
from models_vade import train_vade
from compare_baselines import ae_scores
from winfeat import window_features
import eda_real as E

RNG = np.random.RandomState(0)


def winmat(X, W, s):
    return np.stack([window_features(X[i:i + W], "stats") for i in range(0, len(X) - W + 1, s)]).astype(np.float32)


D = E.load("WADI"); fn, W, stride = E.RAW["WADI"]
Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
y = D["ya_w"].astype(int)
Xtr0, Xte0 = winmat(Xn, W, stride), winmat(Xa, W, stride)[:len(y)]
C6 = Xte0.shape[1] // 6
maxz = np.abs(Xte0[:, :C6]).max(1); hard = (y == 1) & ~(maxz > np.quantile(np.abs(Xtr0[:, :C6]).max(1), 0.99))
mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
Xtr, Xte = ((Xtr0 - mu) / sig).astype(np.float32), ((Xte0 - mu) / sig).astype(np.float32)

v = train_vade(Xtr, n_clusters=20, latent_dim=10, epochs=40, warmup=8, seed=0, device="cpu")
kd = min(80, max(20, len(Xtr) // 10))
v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd)
Ztr, Zte = v._encode_mean(Xtr), v._encode_mean(Xte)
dens_te = -v.latent_gmm.score_samples(Zte)                       # higher = lower density = more anomalous
dens_tr = -v.latent_gmm.score_samples(Ztr)
recon = ae_scores(Xtr, Xte, seed=0)                             # AE reconstruction error


def au(s, m):
    k = (y == 0) | m; return roc_auc_score(y[k], s[k])


print(f"WADI difficult n={int(hard.sum())}")
print(f"  reconstruction error AUROC@difficult = {au(recon, hard):.3f}   (reachability: can the decoder make it?)")
print(f"  latent density      AUROC@difficult = {au(dens_te, hard):.3f}   (probability: is it likely?)")

# on-manifold? PCA off-subspace residual (95% var)
p = PCA(n_components=0.95, svd_solver="full").fit(Xtr)
resid = np.sqrt(((Xte - p.inverse_transform(p.transform(Xte))) ** 2).mean(1))
print(f"  PCA off-subspace residual AUROC@difficult = {au(resid, hard):.3f}   (low -> anomalies stay ON the manifold)")

# between modes? entropy of GMM responsibilities (real windows)
muc = v.mu_c.detach().cpu().numpy()
def resp_entropy(Z):
    d = ((Z[:, None] - muc[None]) ** 2).sum(-1); w = np.exp(-0.5 * (d - d.min(1, keepdims=True))); w /= w.sum(1, keepdims=True)
    return np.array([entropy(wi) for wi in w])
ent = resp_entropy(Zte); ent_tr = resp_entropy(Ztr)
print(f"  mode-responsibility entropy: normal med={np.median(ent_tr):.2f}  difficult med={np.median(ent[hard]):.2f}  "
      f"(higher on difficult -> between modes)")

# --- controlled probe: synthetic BETWEEN-MODE points (average two normals from different modes) ---
mtr = ((Ztr[:, None] - muc[None]) ** 2).sum(-1).argmin(1)
pairs = []
for _ in range(2000):
    i, j = RNG.randint(len(Xtr)), RNG.randint(len(Xtr))
    if mtr[i] != mtr[j]: pairs.append((i, j))
pairs = pairs[:1000]
mix = np.array([(Xtr[i] + Xtr[j]) / 2 for i, j in pairs], np.float32)     # improbable but reachable combos
recon_mix = ae_scores(Xtr, mix, seed=0)
dens_mix = -v.latent_gmm.score_samples(v._encode_mean(mix))
print("\n  --- controlled between-mode probe (avg of two normals from different modes) ---")
print(f"  reconstruction error: normal med={np.median(recon):.3f}  between-mode med={np.median(recon_mix):.3f}  "
      f"(similar -> decoder reconstructs the improbable combo)")
print(f"  latent density (neg): normal med={np.median(dens_tr):.2f}  between-mode med={np.median(dens_mix):.2f}  "
      f"(higher on between-mode -> density flags it)")
print(f"  AUROC normal-vs-between:  reconstruction={roc_auc_score(np.r_[np.zeros(len(recon)),np.ones(len(recon_mix))], np.r_[recon,recon_mix]):.3f}  "
      f"density={roc_auc_score(np.r_[np.zeros(len(dens_tr)),np.ones(len(dens_mix))], np.r_[dens_tr,dens_mix]):.3f}")
