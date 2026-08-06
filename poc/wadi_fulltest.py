"""WADI robustness: the x10 downsample leaves only 56 anomaly windows. KEEP the same model
(trained on the x10 normal) but run inference over the FULL-resolution test at all 10 phase
offsets and pool -> ~560 anomaly windows, same window semantics, no retrain-on-more-data. Report
ours + parametric baselines. (SOTA would need a matched full-res Modal re-run.)
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score
from models_vade import train_vade
from compare_baselines import ae_scores
from winfeat import window_features
import eda_real as E

W, ST, DS = 60, 30, 10
def win(X, y=None):
    A, B = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], "stats"))
        if y is not None: B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)

def main(seed=0):
    Xn, Xa, ya, sens = E._raw_wadi(1)                       # FULL-res raw
    Xn0 = Xn[::DS]                                          # x10 normal = the training distribution
    mu, sd = Xn0.mean(0), Xn0.std(0) + 1e-8
    norm = lambda X: np.clip((X - mu) / sd, -10, 10).astype(np.float32)
    Xtr, _ = win(norm(Xn0))
    fmu, fsd = Xtr.mean(0), Xtr.std(0) + 1e-8
    std = lambda Z: ((Z - fmu) / fsd).astype(np.float32)
    C6 = Xtr.shape[1] // 6; thr99 = np.quantile(np.abs(Xtr[:, :C6]).max(1), 0.99)

    v = train_vade(std(Xtr), n_clusters=20, latent_dim=10, epochs=40, warmup=8, seed=seed)
    kd = min(80, max(20, len(Xtr) // 10))
    v.fit_residual_whitener(std(Xtr)); v.fit_latent_density(std(Xtr), k_density=kd)
    v.fit_resid_head(std(Xtr)); v.fit_basin_head(std(Xtr))
    ifm = IsolationForest(n_estimators=200, random_state=seed).fit(std(Xtr))

    SV, SIF, TRIV, Y, XTE = [], [], [], [], []
    for o in range(DS):                                    # all 10 phase offsets of the FULL test
        Xw, yw = win(norm(Xa[o::DS]), ya[o::DS])
        if Xw.size == 0: continue
        XTE.append(std(Xw)); TRIV.append(np.abs(Xw[:, :C6]).max(1))
        SV.append(v.anomaly_score_hard(std(Xw), use_resid="auto", use_basin="auto"))
        SIF.append(-ifm.decision_function(std(Xw))); Y.append(yw)
    sV = np.concatenate(SV); sIF = np.concatenate(SIF); triv = np.concatenate(TRIV); y = np.concatenate(Y)
    sAE = ae_scores(std(Xtr), np.concatenate(XTE), device="cpu")
    easy = (y == 1) & (triv > thr99); hard = (y == 1) & ~easy

    def au(s, m): k = (y == 0) | m; return roc_auc_score(y[k], s[k])
    print(f"WADI FULL test (pooled 10 offsets): {len(y)} windows, {int((y==1).sum())} anomalies "
          f"(easy {int(easy.sum())}, difficult {int(hard.sum())})  [was 56/37/19 at single offset]")
    print(f"{'method':<24}{'ALL':>7}{'EASY':>7}{'DIFF':>7}")
    for nm, s in [("trivial max|z|", triv), ("IsolationForest", sIF), ("AutoEncoder", sAE),
                  ("VaDE-hard+resid(auto)", sV)]:
        print(f"{nm:<24}{au(s,y==1):>7.3f}{au(s,easy):>7.3f}{au(s,hard):>7.3f}")

if __name__ == "__main__":
    main()
