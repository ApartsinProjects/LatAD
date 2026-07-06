"""Profile the MISSED hard WADI anomalies (false negatives of the per-mode Mahalanobis
detector @5%FPR): how many modes they occupy, what is special about those modes, how their
residual compares to NORMAL residuals in the same mode, and the neighboring-mode structure."""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from winfeat import window_features
import eda_real as E

W, ST = 60, 30


def win(X, y=None):
    Xw, yl = [], []
    for i in range(0, len(X) - W + 1, ST):
        Xw.append(window_features(X[i:i + W], "stats"))
        if y is not None:
            yl.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(Xw, np.float32), (np.asarray(yl, int) if y is not None else None)


def main():
    D = E.load("WADI"); Xn, Xa, yp = D["Xn_raw"], D["Xa_raw"], D["ya_raw"]
    Xtr, _ = win(Xn); Xte, yw = win(Xa, yp)
    C6 = Xte.shape[1] // 6
    triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    hard = (yw == 1) & (triv <= np.quantile(trn, 0.99))

    P = PCA(20, random_state=0).fit(Xtr); Ztr = P.transform(Xtr).astype(np.float64); Zte = P.transform(Xte).astype(np.float64)
    g = GaussianMixture(20, covariance_type="full", random_state=0, reg_covar=1e-3).fit(Ztr)
    atr = g.predict(Ztr); ate = g.predict(Zte)
    lw = {k: LedoitWolf().fit(Ztr[atr == k]) for k in np.unique(atr) if (atr == k).sum() >= 40}
    glob = LedoitWolf().fit(Ztr)

    def mahal(Z, a):
        s = np.zeros(len(Z))
        for k in np.unique(a):
            s[a == k] = lw.get(k, glob).mahalanobis(Z[a == k])
        return s
    s_tr = mahal(Ztr, atr); s_te = mahal(Zte, ate)
    thr = np.quantile(s_te[yw == 0], 0.95)
    missed = hard & (s_te <= thr); mi = np.where(missed)[0]
    print(f"missed hard anomalies: {int(missed.sum())} / {int(hard.sum())}")

    # (1) how many modes; (2) mode sizes; (3) residual vs normal-in-mode; (4) nearest neighbor mode
    mass = np.bincount(atr, minlength=g.n_components) / len(atr)
    cent = g.means_
    nn_dist = np.array([np.sort(np.linalg.norm(cent - cent[k], axis=1))[1] for k in range(g.n_components)])
    modes_hit = ate[mi]
    print(f"(1) distinct modes occupied: {len(np.unique(modes_hit))}  -> {dict(zip(*np.unique(modes_hit, return_counts=True)))}")
    print(f"\n{'win':>5}{'mode':>5}{'modeMass':>10}{'anomMahal':>11}{'norm95_inMode':>14}{'pctOfNorm':>10}{'nnModeDist':>11}")
    for i in mi:
        k = ate[i]
        sn = s_tr[atr == k]                                # normal Mahalanobis in this mode
        n95 = np.quantile(sn, 0.95) if len(sn) else np.nan
        pct = float((sn < s_te[i]).mean()) if len(sn) else np.nan   # where the anomaly sits vs normal-in-mode
        print(f"{i:>5}{k:>5}{mass[k]:>10.3f}{s_te[i]:>11.1f}{n95:>14.1f}{pct*100:>9.0f}%{nn_dist[k]:>11.1f}")

    # summary: fraction of missed whose Mahalanobis is BELOW the mode's normal 95th pct (looks normal)
    below = []
    for i in mi:
        k = ate[i]; sn = s_tr[atr == k]
        below.append(s_te[i] < np.quantile(sn, 0.95) if len(sn) else True)
    print(f"\n(3) missed anomalies inside their mode's NORMAL 95%% Mahal range: {int(np.sum(below))}/{len(mi)}")
    print(f"(2) hosting-mode mass: mean {mass[modes_hit].mean():.3f} vs all-mode mean {mass.mean():.3f} "
          f"(hosting modes are {'LARGER/common' if mass[modes_hit].mean()>mass.mean() else 'smaller/rare'})")
    print(f"(4) hosting-mode nearest-neighbor dist: mean {nn_dist[modes_hit].mean():.1f} vs all-mode mean {nn_dist.mean():.1f}")


if __name__ == "__main__":
    main()
