"""Trajectory detector on the (mode_id, within-cluster-density) encoding.
within_z = robustly-standardized distance of the latent from its assigned cluster centre.
Signals:
  (A) combined hand-crafted = max(sustained-peripheral, frozen), each standardized on normal.
  (B) small sequence model  = a tiny MLP autoregressor p(within_z_t | history) fit on NORMAL;
      surprise = |residual|/std (catches peripheral MOVES); combined with the frozen term
      (variance-collapse, catches HOLDS the AR model is blind to).
Report multi-seed AUROC on the range-hard subset and all anomalies, vs the static LatAD baseline.
"""
from __future__ import annotations
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor
from models_vade import train_vade
from winfeat import window_features
import eda_real as E

K, LD, SEEDS, P = 20, 10, [0, 1, 2], 4


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k])) if m.sum() >= 3 else float("nan")


def winmat(X, W, stride):
    return np.stack([window_features(X[i:i + W], "stats") for i in range(0, len(X) - W + 1, stride)]).astype(np.float32)


def zn(s, ref):
    return (s - np.median(ref)) / (ref.std() + 1e-9)


def trail(s, p, fn):
    return np.array([fn(s[max(0, i - p + 1):i + 1]) for i in range(len(s))])


def within(Z, muc, mtr_ref, is_ref):
    m = ((Z[:, None] - muc[None]) ** 2).sum(-1).argmin(1)
    d = ((Z - muc[m]) ** 2).sum(1)
    return m, d


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

res = {k: [] for k in ["static", "combined_A", "seqmodel_B", "seq+frozen"]}
for sd in SEEDS:
    v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
    muc = v.mu_c.detach().cpu().numpy()
    Ztr, Zte = v._encode_mean(Xtr), v._encode_mean(Xte)
    mtr = ((Ztr[:, None] - muc[None]) ** 2).sum(-1).argmin(1); dtr = ((Ztr - muc[mtr]) ** 2).sum(1)
    mte = ((Zte[:, None] - muc[None]) ** 2).sum(-1).argmin(1); dte = ((Zte - muc[mte]) ** 2).sum(1)
    dmu = np.array([np.median(dtr[mtr == m]) if (mtr == m).any() else 0.0 for m in range(K)])
    glob = np.median(dtr) + 1e-9
    dsd = np.array([max(dtr[mtr == m].std(), 0.3 * glob) if (mtr == m).any() else glob for m in range(K)])
    wz_tr = np.clip((dtr - dmu[mtr]) / dsd[mtr], -8, 8)
    wz_te = np.clip((dte - dmu[mte]) / dsd[mte], -8, 8)

    # static baseline
    res["static"].append(au(y, wz_te, hard))
    # (A) combined hand-crafted: sustained-peripheral + frozen
    per_tr, per_te = trail(wz_tr, P, np.mean), trail(wz_te, P, np.mean)
    frz_tr, frz_te = -trail(wz_tr, P, np.std), -trail(wz_te, P, np.std)
    comboA = np.maximum(zn(per_te, per_tr), zn(frz_te, frz_tr))
    res["combined_A"].append(au(y, comboA, hard))
    # (B) small sequence model: MLP AR predicting wz_t from history, fit on NORMAL
    def feats(wz, m):
        lag1 = np.r_[0, wz[:-1]]; lag2 = np.r_[0, 0, wz[:-2]]; lag3 = np.r_[0, 0, 0, wz[:-3]]
        # dwell in current mode
        dw = np.ones(len(m))
        for i in range(1, len(m)):
            dw[i] = dw[i - 1] + 1 if m[i] == m[i - 1] else 1
        return np.c_[lag1, lag2, lag3, dw, m.astype(float)]
    Ftr, Fte = feats(wz_tr, mtr), feats(wz_te, mte)
    mlp = MLPRegressor(hidden_layer_sizes=(16,), max_iter=400, random_state=sd, alpha=1e-2).fit(Ftr, wz_tr)
    rtr = wz_tr - mlp.predict(Ftr); rte = wz_te - mlp.predict(Fte)
    surprise = np.abs(rte) / (rtr.std() + 1e-9)
    res["seqmodel_B"].append(au(y, surprise, hard))
    res["seq+frozen"].append(au(y, np.maximum(zn(surprise, np.abs(rtr) / (rtr.std() + 1e-9)), zn(frz_te, frz_tr)), hard))

print(f"range-hard n={int(hard.sum())}  (static LatAD full baseline ~0.485)")
for k, vv in res.items():
    print(f"  {k:12} range-hard = {np.nanmean(vv):.3f} +/- {np.nanstd(vv):.3f}")
