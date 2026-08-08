"""Diagnostic: does the VaDE MODE sequence carry signal on the 11 range-hard WADI windows?
Assign each window a mode (nearest VaDE component). From NORMAL data build: mode occupancy P(m),
order-1 transition P(m'|m), and per-mode dwell-length distribution. Then for every test window score
three mode-level signals and check AUROC on the range-hard subset (and print the 11 explicitly):
  rare   = -log P(mode)                     (occupies a rare mode)
  trans  = -log P(mode | prev mode)         (illegal/rare incoming transition)
  dwell  = current run-length z vs normal    (held a mode abnormally long)
If none separates the 11, the mode sequence is blind to them and a sequence model cannot help.
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


def runlen(seq):
    r = np.ones(len(seq), int)
    for i in range(1, len(seq)):
        r[i] = r[i - 1] + 1 if seq[i] == seq[i - 1] else 1
    return r


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
mtr = ((v._encode_mean(Xtr)[:, None] - muc[None]) ** 2).sum(-1).argmin(1)   # normal mode sequence
mte = ((v._encode_mean(Xte)[:, None] - muc[None]) ** 2).sum(-1).argmin(1)   # test mode sequence

# normal statistics
occ = np.bincount(mtr, minlength=K) + 1.0; Pocc = occ / occ.sum()
T = np.ones((K, K)); [T.__setitem__((mtr[i - 1], mtr[i]), T[mtr[i - 1], mtr[i]] + 1) for i in range(1, len(mtr))]
Ptr = T / T.sum(1, keepdims=True)
rl_tr = runlen(mtr)
dwell_mu = np.array([rl_tr[mtr == m].mean() if (mtr == m).any() else 1.0 for m in range(K)])
dwell_sd = np.array([rl_tr[mtr == m].std() + 1e-6 for m in range(K)])

rl_te = runlen(mte)
rare = -np.log(Pocc[mte])
trans = np.array([-np.log(Ptr[mte[i - 1], mte[i]]) if i > 0 else 0.0 for i in range(len(mte))])
dwell = (rl_te - dwell_mu[mte]) / dwell_sd[mte]

print(f"range-hard n={int(hard.sum())}; distinct normal modes used={len(np.unique(mtr))}")
print(f"  {'signal':8} {'AUROC@range-hard':>17} {'AUROC@all-anom':>15}")
for name, s in [("rare", rare), ("trans", trans), ("dwell", dwell)]:
    print(f"  {name:8} {au(y, s, hard):>17.3f} {au(y, s, (y == 1)):>15.3f}")

print("\n=== the 11 range-hard windows (mode | prev->mode P | dwell) ===")
for i in np.where(hard)[0]:
    pm = mte[i - 1] if i > 0 else -1
    print(f"  win {i:3}: mode={mte[i]:2} (occ {Pocc[mte[i]]*100:4.1f}%)  "
          f"prev {pm:2}->{mte[i]:2} P={Ptr[pm, mte[i]]*100 if pm>=0 else 0:5.1f}%  "
          f"dwell={rl_te[i]:2} (normal {dwell_mu[mte[i]]:.1f}+/-{dwell_sd[mte[i]]:.1f})")
