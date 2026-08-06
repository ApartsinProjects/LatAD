"""Root-cause the 'in-common' misses: anomalies that sit INSIDE a normal mode and that
the trivial max|z| rule cannot separate (the hard subset). Which lever recovers them?

  H1 clusters too coarse   -> finer clustering (K x4)
  H2 model too weak        -> bigger VaDE (latent 24, hidden 256/128)
  H3 temporal-context      -> dynamics features (winfeat 'temporal': slopes, velocity, band-power)
  H4 window too short      -> 3x longer window

Metric = AUROC and TPR@5%FPR on (all normal + HARD anomalies only), i.e. how well each
config separates the subtle in-mode anomalies from normal. The lever with the biggest
lift over 'base' is the cause. Focus on the in-common-heavy sets: SKAB (78%) and WADI (93%).
"""
from __future__ import annotations
import sys, numpy as np
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
from winfeat import window_features
import eda_real as E

K = {"SKAB": 12, "WADI": 20, "HAI": 24, "SWaT": 16}
STRIDE = {"SKAB": 10, "WADI": 30, "HAI": 60, "SWaT": 30}


def windowize(Xn, Xa, ya, W, stride, rep="stats"):
    def win(X, y=None):
        Xw, yl = [], []
        for i in range(0, len(X) - W + 1, stride):
            Xw.append(window_features(X[i:i + W], rep))
            if y is not None:
                yl.append(int(y[i:i + W].mean() > 0.05))
        return np.asarray(Xw, np.float32), (np.asarray(yl, int) if y is not None else None)
    a, _ = win(Xn); b, c = win(Xa, ya)
    return a, b, c


def hard_eval(Xtr, Xte, yte, k, latent=10, hidden=(128, 64), seed=0):
    """Train VaDE; return (full AUROC, hard AUROC, hard TPR@5%). HARD = anomalies whose
    per-channel window-mean max|z| is below the 99th percentile of normal (not trivially
    separable). The trivial score is read straight off the standardised stats features."""
    C = None  # infer channel count from feature block (stats = 6 blocks)
    trivial = np.abs(Xte[:, :Xte.shape[1] // 6]).max(1)          # level block only
    trivial_n = np.abs(Xtr[:, :Xtr.shape[1] // 6]).max(1)
    v = train_vade(Xtr, n_clusters=k, latent_dim=latent, hidden=hidden, epochs=40,
                   warmup=8, seed=seed, device="cpu")
    v.fit_residual_whitener(Xtr); s = v.anomaly_score(Xte)
    full_au = roc_auc_score(yte, s)
    easy = trivial > np.quantile(trivial_n, 0.99)
    hardmask = (yte == 0) | ((yte == 1) & ~easy)               # normal + hard anomalies
    yh, sh = yte[hardmask], s[hardmask]
    if (yh == 1).sum() < 5:
        return full_au, float("nan"), float("nan"), int((yte == 1).sum()), 0
    hard_au = roc_auc_score(yh, sh)
    thr = np.quantile(sh[yh == 0], 0.95); hard_tpr = float((sh[yh == 1] > thr).mean())
    return full_au, hard_au, hard_tpr, int((yte == 1).sum()), int((yh == 1).sum())


def main(names):
    print(f"{'dataset':<7}{'lever':<16}{'K':>4}{'lat':>4}{'W':>5}{'rep':>9}"
          f"{'fullAU':>8}{'hardAU':>8}{'hardTPR':>9}{'nHard':>7}")
    print("-" * 77)
    for nm in names:
        D = E.load(nm); Xn, Xa, ya = D["Xn_raw"], D["Xa_raw"], D["ya_raw"]
        W, st, k = D["W"], STRIDE[nm], K[nm]
        base = windowize(Xn, Xa, ya, W, st)
        def row(lever, kk, lat, ww, rep, data):
            Xtr, Xte, yte = data
            fa, ha, ht, na, nh = hard_eval(Xtr, Xte, yte, kk, latent=lat,
                                           hidden=(256, 128) if lat > 10 else (128, 64))
            print(f"{nm:<7}{lever:<16}{kk:>4}{lat:>4}{ww:>5}{rep:>9}"
                  f"{fa:>8.3f}{ha:>8.3f}{ht:>9.3f}{nh:>7}")
        row("base", k, 10, W, "stats", base)
        row("H1 finer K x4", 4 * k, 10, W, "stats", base)
        row("H2 bigger model", k, 24, W, "stats", base)
        row("H3 temporal feats", k, 10, W, "temporal", windowize(Xn, Xa, ya, W, st, "temporal"))
        row("H4 window x3", k, 10, W * 3, "stats", windowize(Xn, Xa, ya, W * 3, st))
        print()


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["SKAB", "WADI"]
    main(names)
