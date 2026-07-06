"""Standard 3-way report for the real datasets: metrics on ALL / EASY / HARD anomalies,
computed on ONE pass so the numbers are construct-matched.

  EASY = anomalies a trivial single-channel rule separates (max|z| level > 99th pct normal)
         i.e. range/threshold/rule detectable.
  HARD = the rest (subtle, in-mode) -- the discriminative subset where a method must earn it.

Each metric block = {normal windows} + {that anomaly subset}. AUROC and TPR@5%FPR (threshold
set on the normal windows). Detectors: the trivial rule, IF, LOF, AutoEncoder, and our VaDE.
SOTA (published raw point-wise F1) is cited in the printed footer, not computed here.
"""
from __future__ import annotations
import sys, numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
from compare_baselines import ae_scores
from winfeat import window_features
import eda_real as E

K = {"SKAB": 12, "HAI": 24, "WADI": 20, "SWaT": 16}
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


def subset_metrics(y, s, mask):
    """AUROC + TPR@5%FPR on normal + the masked anomaly subset."""
    keep = (y == 0) | mask
    yk, sk = y[keep], s[keep]
    if (yk == 1).sum() < 3:
        return float("nan"), float("nan")
    au = roc_auc_score(yk, sk); thr = np.quantile(sk[yk == 0], 0.95)
    return au, float((sk[yk == 1] > thr).mean())


def run(name, seed=0):
    D = E.load(name); Xn, Xa, ya = D["Xn_raw"], D["Xa_raw"], D["ya_raw"]
    Xtr, Xte, y = windowize(Xn, Xa, ya, D["W"], STRIDE[name])
    C6 = Xte.shape[1] // 6
    triv = np.abs(Xte[:, :C6]).max(1); triv_n = np.abs(Xtr[:, :C6]).max(1)
    easy = (y == 1) & (triv > np.quantile(triv_n, 0.99))
    hard = (y == 1) & ~easy
    S = {"trivial max|z|": triv,
         "IsolationForest": -IsolationForest(n_estimators=200, random_state=seed).fit(Xtr).decision_function(Xte),
         "LOF": -LocalOutlierFactor(30, novelty=True).fit(Xtr).decision_function(Xte),
         "AutoEncoder": ae_scores(Xtr, Xte, device="cpu")}
    v = train_vade(Xtr, n_clusters=K[name], latent_dim=10, epochs=40, warmup=8, seed=seed, device="cpu")
    v.fit_residual_whitener(Xtr); S["VaDE (ours)"] = v.anomaly_score(Xte)

    print(f"\n########## {name}  test={len(Xte)}  anom={int((y==1).sum())} "
          f"(easy={int(easy.sum())}, hard={int(hard.sum())})  ##########")
    print(f"{'method':<18}" + "".join(f"{c:>16}" for c in ["ALL", "EASY", "HARD"]))
    print(f"{'':<18}" + "".join(f"{'AUROC  TPR':>16}" for _ in range(3)))
    print("-" * 66)
    for nm, s in S.items():
        cells = []
        for m in (y == 1, easy, hard):
            au, tp = subset_metrics(y, s, m); cells.append(f"{au:>6.3f} {tp:>5.3f}  ")
        print(f"{nm:<18}" + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["SKAB", "HAI", "WADI", "SWaT"]
    for nm in names:
        run(nm)
    print("\nSOTA (published RAW point-wise F1, for reference): SWaT ~0.81, WADI ~0.57 (GDN); "
          "SKAB ~0.78 (Conv-AE); HAI uses eTaPR. PA-F1 numbers (0.9+) are inflated (Kim 2022).")
