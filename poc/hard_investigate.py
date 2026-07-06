"""Why do we (and everyone) fail on the HARD WADI anomalies? Per hard window, ask WHO
catches it (trivial / VaDE-stats / VaDE-temporal / USAD-SOTA) at a 5%-FPR threshold, and
characterise it: marginal extremeness, mode NLL, dominant channels, and a TEMPORAL
signature (does the window's dynamics differ from the preceding normal windows?).

Logic: HARD = trivial max|z| can't separate it (no single channel far out), so the fault
is multivariate (joint-structure break) or temporal (value fine, wrong for the sequence).
- caught by VaDE-stats but not trivial -> joint/structure break (correlation).
- caught by VaDE-temporal but not VaDE-stats -> temporal/dynamics.
- caught by USAD but not ours -> a structure our per-window features miss.
- caught by NOBODY -> likely needs raw-signal modelling or is ambiguous/mislabelled.
"""
from __future__ import annotations
import os, numpy as np
from models_vade import train_vade
from winfeat import window_features
import eda_real as E

W, ST = 60, 30
RES = os.path.join(os.path.dirname(__file__), "sota_bundle", "results")


def win(X, y=None, rep="stats"):
    Xw, yl = [], []
    for i in range(0, len(X) - W + 1, ST):
        Xw.append(window_features(X[i:i + W], rep))
        if y is not None:
            yl.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(Xw, np.float32), (np.asarray(yl, int) if y is not None else None)


def vade_score(Xtr, Xte, k=20, latent=10, hidden=(128, 64)):
    v = train_vade(Xtr, n_clusters=k, latent_dim=latent, hidden=hidden, epochs=40, warmup=8, seed=0, device="cpu")
    v.fit_residual_whitener(Xtr)
    return v.anomaly_score(Xte), v


def flags_at_5fpr(s_tr, s_te, y):
    thr = np.quantile(s_te[y == 0], 0.95)
    return s_te > thr


def main():
    D = E.load("WADI"); Xn, Xa, yp = D["Xn_raw"], D["Xa_raw"], D["ya_raw"]; ch = D["ch"]
    Xtr, _ = win(Xn); Xte, yw = win(Xa, yp)
    XtrT, _ = win(Xn, rep="temporal"); XteT, _ = win(Xa, yp, rep="temporal")
    C6 = Xte.shape[1] // 6
    triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy
    idx_hard = np.where(hard)[0]
    print(f"WADI: {len(Xte)} windows, {int((yw==1).sum())} anomalies, HARD={len(idx_hard)}")

    # detectors
    s_vade, v = vade_score(Xtr, Xte)
    s_temp, _ = vade_score(XtrT, XteT)
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    s_if = -IsolationForest(n_estimators=200, random_state=0).fit(Xtr).decision_function(Xte)
    s_lof = -LocalOutlierFactor(30, novelty=True).fit(Xtr).decision_function(Xte)
    dets = {"trivial": triv, "IF": s_if, "LOF": s_lof, "VaDE-stats": s_vade, "VaDE-temporal": s_temp}
    # USAD windowed (fetched per-point -> max per window)
    sp = f"{RES}/score_USAD.npy"
    if os.path.exists(sp):
        sPt = np.load(sp); sPt = sPt.mean(1) if sPt.ndim > 1 else sPt
        usw = np.array([sPt[i:i + W].max() for i in range(0, len(sPt) - W + 1, ST)])
        dets["USAD"] = usw[:len(yw)]
    caught = {n: flags_at_5fpr(s[yw == 0] if False else s, s, yw) for n, s in dets.items()}

    # mode NLL + assignment from VaDE
    from sklearn.decomposition import PCA
    pca = PCA(min(30, Xtr.shape[1]), random_state=0).fit(Xtr)
    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(20, covariance_type="diag", random_state=0, reg_covar=1e-2).fit(pca.transform(Xtr).astype(np.float64))
    Zte = pca.transform(Xte).astype(np.float64)
    nll_te = -g.score_samples(Zte); nll_tr = -g.score_samples(pca.transform(Xtr).astype(np.float64))
    assign = g.predict(Zte)

    # temporal-distinctiveness: how different is a window's dynamics vs the median normal window
    dyn = XteT[:, 4 * C6:5 * C6]                                   # within-window slope block
    dyn_n = XtrT[:, 4 * C6:5 * C6]; dmu = np.median(dyn_n, 0)
    dyn_dist = np.abs(dyn - dmu).max(1)

    print(f"\n{'win':>5}{'max|z|':>8}{'NLL%':>6}  caught_by            top-channels(window-mean z)")
    print("-" * 92)
    counts = {n: 0 for n in dets}
    none_ct = 0
    for i in idx_hard:
        cb = [n for n in dets if caught[n][i]]
        for n in cb:
            counts[n] += 1
        if not cb:
            none_ct += 1
        raw_w = Xa[i * ST:i * ST + W]; zmean = raw_w.mean(0)
        top = np.argsort(-np.abs(zmean))[:3]
        chans = ", ".join(f"{ch[j]}={zmean[j]:+.1f}" for j in top)
        nllpct = float((nll_te[i] > nll_tr).mean())
        print(f"{i:>5}{triv[i]:>8.1f}{nllpct*100:>5.0f}%  {','.join(cb) if cb else 'NONE':<20} {chans}")

    print("\n--- HARD-subset catch summary (of "
          f"{len(idx_hard)} hard windows) ---")
    for n in dets:
        print(f"  {n:<14} catches {counts[n]}/{len(idx_hard)}")
    print(f"  caught by NOBODY: {none_ct}/{len(idx_hard)}")
    # union
    union = np.zeros(len(idx_hard), bool)
    for k2, i in enumerate(idx_hard):
        union[k2] = any(caught[n][i] for n in dets)
    print(f"  caught by AT LEAST ONE: {int(union.sum())}/{len(idx_hard)}")


if __name__ == "__main__":
    main()
