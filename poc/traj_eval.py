"""Confirm the trajectory detector: run seq+frozen on WADI and HAI range-hard subsets, with a
bootstrap CI over the (few) anomalies -- HAI's n=81 gives real power vs WADI's n=11. Compares the
trajectory score to the static LatAD baseline. Multi-seed detector score = mean over seeds; the
bootstrap resamples the hard-anomaly positives (negatives fixed) to reflect small-n uncertainty.
"""
from __future__ import annotations
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor
from models_vade import train_vade
from winfeat import window_features
import eda_real as E

CFG = {"WADI": (20, 10), "HAI": (40, 16)}
SEEDS, P, B = [0, 1, 2], 4, 2000
RNG = np.random.RandomState(0)


def winmat(X, W, stride):
    return np.stack([window_features(X[i:i + W], "stats") for i in range(0, len(X) - W + 1, stride)]).astype(np.float32)


def zn(s, ref): return (s - np.median(ref)) / (ref.std() + 1e-9)
def trail(s, p, fn): return np.array([fn(s[max(0, i - p + 1):i + 1]) for i in range(len(s))])


def dwell(m):
    dw = np.ones(len(m))
    for i in range(1, len(m)):
        dw[i] = dw[i - 1] + 1 if m[i] == m[i - 1] else 1
    return dw


def boot_ci(y, s, hard):
    neg = np.where(y == 0)[0]; pos = np.where(hard)[0]
    aucs = []
    for _ in range(B):
        pb = RNG.choice(pos, len(pos), replace=True)
        idx = np.r_[neg, pb]
        aucs.append(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pb))], s[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def run(name):
    D = E.load(name); fn, W, stride = E.RAW[name]
    Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
    y = D["ya_w"].astype(int)
    starts = list(range(0, len(Xa) - W + 1, stride))[:len(y)]
    lo, hi = Xn.min(0), Xn.max(0)
    oor = np.array([bool(((Xa[i:i + W] < lo) | (Xa[i:i + W] > hi)).any()) for i in starts])
    hard = (y == 1) & ~oor
    Xtr0, Xte0 = winmat(Xn, W, stride), winmat(Xa, W, stride)[:len(y)]
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr, Xte = ((Xtr0 - mu) / sig).astype(np.float32), ((Xte0 - mu) / sig).astype(np.float32)
    K, LD = CFG[name]
    traj, base = [], []
    for sd in SEEDS:
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
        kd = min(80, max(20, len(Xtr) // 10))
        v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd)
        v.fit_resid_head(Xtr); v.fit_basin_head(Xtr)
        muc = v.mu_c.detach().cpu().numpy()
        Ztr, Zte = v._encode_mean(Xtr), v._encode_mean(Xte)
        mtr = ((Ztr[:, None] - muc[None]) ** 2).sum(-1).argmin(1); dtr = ((Ztr - muc[mtr]) ** 2).sum(1)
        mte = ((Zte[:, None] - muc[None]) ** 2).sum(-1).argmin(1); dte = ((Zte - muc[mte]) ** 2).sum(1)
        dmu = np.array([np.median(dtr[mtr == m]) if (mtr == m).any() else 0.0 for m in range(K)])
        glob = np.median(dtr) + 1e-9
        dsd = np.array([max(dtr[mtr == m].std(), 0.3 * glob) if (mtr == m).any() else glob for m in range(K)])
        wz_tr = np.clip((dtr - dmu[mtr]) / dsd[mtr], -8, 8); wz_te = np.clip((dte - dmu[mte]) / dsd[mte], -8, 8)
        base.append(v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto"))
        frz_tr, frz_te = -trail(wz_tr, P, np.std), -trail(wz_te, P, np.std)
        Ftr = np.c_[np.r_[0, wz_tr[:-1]], np.r_[0, 0, wz_tr[:-2]], dwell(mtr), mtr.astype(float)]
        Fte = np.c_[np.r_[0, wz_te[:-1]], np.r_[0, 0, wz_te[:-2]], dwell(mte), mte.astype(float)]
        mlp = MLPRegressor(hidden_layer_sizes=(16,), max_iter=400, random_state=sd, alpha=1e-2).fit(Ftr, wz_tr)
        rtr, rte = wz_tr - mlp.predict(Ftr), wz_te - mlp.predict(Fte)
        sur_te = np.abs(rte) / (rtr.std() + 1e-9); sur_ref = np.abs(rtr) / (rtr.std() + 1e-9)
        traj.append(np.maximum(zn(sur_te, sur_ref), zn(frz_te, frz_tr)))
    s_traj = np.mean(traj, 0); s_base = np.mean([np.asarray(b) for b in base], 0)

    def au(s):
        k = (y == 0) | hard; return roc_auc_score(y[k], s[k])
    print(f"\n=== {name}  range-hard n={int(hard.sum())} ===")
    for tag, s in [("LatAD static", s_base), ("trajectory(seq+frozen)", s_traj)]:
        lo95, hi95 = boot_ci(y, s, hard)
        print(f"  {tag:24} AUROC={au(s):.3f}  95%CI[{lo95:.3f}, {hi95:.3f}]")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI", "HAI"]):
        run(nm)
