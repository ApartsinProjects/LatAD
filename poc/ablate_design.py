"""Design ablation on the real datasets, answering three 'should we?' questions with
one harness (AUROC / TPR@5%FPR, same standardised+clipped raw channels as the EDA):

  overspecified clustering : does K >> #modes help? (joint VaDE at K vs 3K)
  sequential vs joint      : plain VAE + post-hoc GMM  vs  joint VaDE
  larger windows           : W vs 3*W (subtle in-mode faults may need more context)

Reuses eda_real's raw loaders so windows can be rebuilt at any W. stats features are
6*C regardless of W, so the VaDE input dim is unchanged across window sizes.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
from models_vade import train_vade, train_plain_vae
from winfeat import window_features
import eda_real as E

K = {"SKAB": 12, "HAI": 24, "WADI": 20, "SWaT": 16}     # ~#modes from the EDA
STRIDE = {"SKAB": 10, "HAI": 60, "WADI": 30, "SWaT": 30}


def windowize(Xn, Xa, ya, W, stride):
    def win(X, y=None):
        Xw, yl = [], []
        for i in range(0, len(X) - W + 1, stride):
            Xw.append(window_features(X[i:i + W], "stats"))
            if y is not None:
                yl.append(int(y[i:i + W].mean() > 0.05))
        return np.asarray(Xw, np.float32), (np.asarray(yl, int) if y is not None else None)
    a, _ = win(Xn); b, c = win(Xa, ya)
    return a, b, c


def evalu(y, s):
    au = roc_auc_score(y, s); thr = np.quantile(s[y == 0], 0.95)
    return au, float((s[y == 1] > thr).mean())


def joint(Xtr, Xte, k, seed=0):
    v = train_vade(Xtr, n_clusters=k, latent_dim=10, epochs=40, warmup=8, seed=seed, device="cpu")
    v.fit_residual_whitener(Xtr)
    return v.anomaly_score(Xte)


def seq(Xtr, Xte, k, seed=0):
    pv = train_plain_vae(Xtr, latent_dim=10, epochs=40, seed=seed, device="cpu")
    pv.fit_gmm(Xtr, k); pv.fit_residual_whitener(Xtr)
    return pv.anomaly_score(Xte)


def main(names):
    print(f"{'dataset':<8}{'config':<16}{'K':>4}{'W':>5}{'ntr':>7}{'AUROC':>8}{'TPR@5':>8}")
    print("-" * 56)
    for nm in names:
        D = E.load(nm); Xn, Xa, ya = D["Xn_raw"], D["Xa_raw"], D["ya_raw"]
        W, st, k = D["W"], STRIDE[nm], K[nm]
        Xtr, Xte, yte = windowize(Xn, Xa, ya, W, st)
        def line(tag, kk, ww, ntr, s, yy):
            au, tp = evalu(yy, s); print(f"{nm:<8}{tag:<16}{kk:>4}{ww:>5}{ntr:>7}{au:>8.3f}{tp:>8.3f}")
        line("joint K", k, W, len(Xtr), joint(Xtr, Xte, k), yte)
        line("overspec 3K", 3 * k, W, len(Xtr), joint(Xtr, Xte, 3 * k), yte)
        line("sequential K", k, W, len(Xtr), seq(Xtr, Xte, k), yte)
        Xtr3, Xte3, yte3 = windowize(Xn, Xa, ya, W * 3, st)
        line("largerW 3x", k, W * 3, len(Xtr3), joint(Xtr3, Xte3, k), yte3)
        print()


if __name__ == "__main__":
    import sys
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["SKAB", "HAI", "WADI", "SWaT"]
    main(names)
