"""Apples-to-apples SOTA vs ours on WADI. SOTA (USAD/TranAD/GDN) scores per-timestep;
ours scores per-window. We fetched the SOTA per-point arrays, so here we AGGREGATE them
onto our exact window grid (W=60, stride=30 over the x10-downsampled attack stream, the
same grid eda_real/wadi use) and report window-level ALL/EASY/HARD AUROC + F1 next to ours.
Also prints the raw point-wise F1 (the number comparable to published SOTA)."""
from __future__ import annotations
import glob, os, numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from models_vade import train_vade
from compare_baselines import ae_scores
from winfeat import window_features
import eda_real as E

W, ST = 60, 30
RES = os.path.join(os.path.dirname(__file__), "sota_bundle", "results")


def win_feats(X, y=None):
    Xw, yl = [], []
    for i in range(0, len(X) - W + 1, ST):
        Xw.append(window_features(X[i:i + W], "stats"))
        if y is not None:
            yl.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(Xw, np.float32), (np.asarray(yl, int) if y is not None else None)


def win_pointscore(s):
    return np.array([s[i:i + W].max() for i in range(0, len(s) - W + 1, ST)])   # window = max point score


def metr(y, s, mask):
    """AUROC, best-F1 over a threshold sweep, and the FPR (on normal) AT that best-F1 threshold."""
    keep = (y == 0) | mask
    yk, sk = y[keep], s[keep]
    if (yk == 1).sum() < 3:
        return (float("nan"), float("nan"), float("nan"))
    au = roc_auc_score(yk, sk)
    qs = np.quantile(sk, np.linspace(0.80, 0.999, 60))
    f1s = [(f1_score(yk, sk > t), t) for t in qs]
    f1, tbest = max(f1s, key=lambda p: p[0])
    fpr = float((sk[yk == 0] > tbest).mean())
    return round(au, 3), round(f1, 3), round(fpr, 3)


def raw_point_f1(name):
    sp, lp = f"{RES}/score_{name}.npy", f"{RES}/labels_{name}.npy"
    if not (os.path.exists(sp) and os.path.exists(lp)):
        return None
    s = np.load(sp); y = np.load(lp).astype(int); s = s.mean(1) if s.ndim > 1 else s
    qs = np.quantile(s, np.linspace(0.80, 0.999, 80))
    return round(max(f1_score(y, s > t) for t in qs), 3)


def main():
    D = E.load("WADI"); Xn, Xa, yp = D["Xn_raw"], D["Xa_raw"], D["ya_raw"]
    Xtr, _ = win_feats(Xn); Xte, yw = win_feats(Xa, yp)
    C6 = Xte.shape[1] // 6
    triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy
    print(f"WADI window grid: {len(Xte)} test windows, anom={int((yw==1).sum())} "
          f"(easy={int(easy.sum())}, hard={int(hard.sum())})")

    scores = {"trivial max|z|": triv}
    scores["IsolationForest"] = -IsolationForest(n_estimators=200, random_state=0).fit(Xtr).decision_function(Xte)
    scores["LOF"] = -LocalOutlierFactor(30, novelty=True).fit(Xtr).decision_function(Xte)
    scores["AutoEncoder"] = ae_scores(Xtr, Xte, device="cpu")
    v = train_vade(Xtr, n_clusters=20, latent_dim=10, epochs=40, warmup=8, seed=0, device="cpu")
    v.fit_residual_whitener(Xtr); scores["VaDE (ours)"] = v.anomaly_score(Xte)
    from permode_mahal import PerModeMahal
    scores["per-mode Mahal (ours)"] = PerModeMahal().fit(Xtr).score(Xte)
    from sklearn.decomposition import PCA
    Pk = PCA(20, random_state=0).fit(Xtr)                  # LOF in the reduced latent (best on HARD)
    scores["LOF-latent (ours)"] = -LocalOutlierFactor(20, novelty=True).fit(
        Pk.transform(Xtr)).decision_function(Pk.transform(Xte))

    # SOTA: window the fetched per-point arrays onto the same grid
    for name in ["USAD", "TranAD", "GDN"]:
        sp = f"{RES}/score_{name}.npy"
        if os.path.exists(sp):
            s = np.load(sp); s = s.mean(1) if s.ndim > 1 else s
            sw = win_pointscore(s)
            n = min(len(sw), len(yw)); scores[f"{name} (SOTA)"] = sw[:n]

    print(f"\n{'method':<18}{'ALL au/f1/fpr':>18}{'EASY au/f1/fpr':>18}{'HARD au/f1/fpr':>18}{'rawF1':>7}")
    print("-" * 79)
    for nm, s in scores.items():
        m = min(len(s), len(yw)); s2, y2 = s[:m], yw[:m]
        e2, h2 = easy[:m], hard[:m]
        a = metr(y2, s2, y2 == 1); ez = metr(y2, s2, e2); hd = metr(y2, s2, h2)
        rp = raw_point_f1(nm.split(" ")[0]) if "SOTA" in nm else ""
        cell = lambda t: f"{t[0]}/{t[1]}/{t[2]}"
        print(f"{nm:<18}{cell(a):>18}{cell(ez):>18}{cell(hd):>18}{str(rp):>7}")
    print("\nau/f1/fpr: AUROC, best-F1 (oracle threshold), and FPR-on-normal AT that best-F1 threshold. "
          "rawF1 = SOTA raw point-wise best-F1 (pub: WADI GDN 0.57). SOTA scores max-pooled to our grid.")


if __name__ == "__main__":
    main()
