"""Encode each window as (mode_id, within-cluster density) and test whether the SEQUENCE of the
within-cluster signal separates the 11 range-hard WADI windows -- the info pure mode IDs threw away.
within_z = how far the latent sits from its assigned cluster centre, standardized by that mode's
normal spread (peripheral-within-mode). Test: static, its trailing MEAN (sustained peripheral), and
its trailing STD (frozen within mode), plus deviation vs the mode's own normal within-z.
"""
from __future__ import annotations
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
from winfeat import window_features
import eda_real as E

K, LD, SEED = 20, 10, 0


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k])) if m.sum() >= 3 else float("nan")


def winmat(X, W, stride):
    return np.stack([window_features(X[i:i + W], "stats") for i in range(0, len(X) - W + 1, stride)]).astype(np.float32)


def trail(s, P, fn):
    return np.array([fn(s[max(0, i - P + 1):i + 1]) for i in range(len(s))])


D = E.load("WADI"); fn, W, stride = E.RAW["WADI"]
Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
y = D["ya_w"].astype(int)
starts = list(range(0, len(Xa) - W + 1, stride))[:len(y)]
lo, hi = Xn.min(0), Xn.max(0)
oor = np.array([bool(((Xa[i:i + W] < lo) | (Xa[i:i + W] > hi)).any()) for i in starts])
hard = (y == 1) & ~oor

Xtr0, Xte0 = winmat(Xn, W, stride), winmat(Xa, W, stride)[:len(y)]
mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
Xtr, Xte = ((Xtr0 - mu) / sig).astype(np.float32), ((Xte0 - mu) / sig).astype(np.float32)
v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=SEED, device="cpu")
muc = v.mu_c.detach().cpu().numpy()
Ztr, Zte = v._encode_mean(Xtr), v._encode_mean(Xte)
mtr = ((Ztr[:, None] - muc[None]) ** 2).sum(-1).argmin(1)
mte = ((Zte[:, None] - muc[None]) ** 2).sum(-1).argmin(1)
dtr = ((Ztr - muc[mtr]) ** 2).sum(1)                       # dist to own centre (normal)
dte = ((Zte - muc[mte]) ** 2).sum(1)
dmu = np.array([dtr[mtr == m].mean() if (mtr == m).any() else 0.0 for m in range(K)])
dsd = np.array([dtr[mtr == m].std() + 1e-9 for m in range(K)])
within_z = (dte - dmu[mte]) / dsd[mte]                     # peripheral-within-mode, standardized

sig_static = within_z
sig_tmean = trail(within_z, 4, np.mean)                    # sustained peripheral
sig_tstd = -trail(within_z, 4, np.std)                     # frozen within mode (low std -> high score)
sig_tstd_all = -trail(within_z, 8, np.std)

print(f"range-hard n={int(hard.sum())}")
print(f"  {'signal':22} {'AUROC@range-hard':>17} {'AUROC@all-anom':>15}")
for name, s in [("within_z (static)", sig_static), ("trailing-mean P4", sig_tmean),
                ("trailing-std(frozen) P4", sig_tstd), ("trailing-std(frozen) P8", sig_tstd_all)]:
    print(f"  {name:22} {au(y, s, hard):>17.3f} {au(y, s, (y == 1)):>15.3f}")

print("\n=== 11 windows: within_z and its 4-window trailing std (frozen if ~0) ===")
tstd4 = trail(within_z, 4, np.std)
for i in np.where(hard)[0]:
    print(f"  win {i:3}: mode={mte[i]:2}  within_z={within_z[i]:+6.2f}  trail_std={tstd4[i]:.2f}")
