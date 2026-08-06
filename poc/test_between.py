"""Hypothesis: the missed hard anomalies fall BETWEEN several clusters (pockets, A3). Our
score uses the NEAREST mode (min distance / max log-density), so a between-modes point looks
'close enough to some mode' and is not flagged, even though it sits in a low-density valley.

Measure 'betweenness' per window and compare normal vs hard-caught vs hard-missed:
  entropy    = H(responsibilities)             high -> spread across many modes
  d2/d1      = 2nd-nearest / nearest Mahal      ~1 -> two modes equally close (a pocket)
  margin     = logp(1st) - logp(2nd)            small -> ambiguous between top-2
Then test whether a betweenness-aware score (full MIXTURE density, or nearest+entropy) catches
the ones the nearest-mode score misses.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
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


def hauc(y, s, hard):
    keep = (y == 0) | hard
    return roc_auc_score(y[keep], s[keep])


def main():
    D = E.load("WADI"); Xn, Xa, yp = D["Xn_raw"], D["Xa_raw"], D["ya_raw"]
    Xtr, _ = win(Xn); Xte, yw = win(Xa, yp)
    C6 = Xte.shape[1] // 6
    triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    hard = (yw == 1) & (triv <= np.quantile(trn, 0.99))

    P = PCA(20, random_state=0).fit(Xtr); Ztr = P.transform(Xtr); Zte = P.transform(Xte)
    g = GaussianMixture(20, covariance_type="full", random_state=0, reg_covar=1e-3).fit(Ztr)

    # per-mode Mahalanobis to every mode
    def mahal_all(Z):
        out = np.zeros((len(Z), g.n_components))
        for k in range(g.n_components):
            d = Z - g.means_[k]; out[:, k] = np.einsum("ij,jk,ik->i", d, g.precisions_[k], d)
        return out
    Dte = mahal_all(Zte); Dte.sort(axis=1)                 # ascending
    nearest = Dte[:, 0]                                     # our detector (min Mahalanobis)
    d2d1 = Dte[:, 1] / (Dte[:, 0] + 1e-9)
    mixture_nll = -g.score_samples(Zte)                    # full mixture density (betweenness-aware)
    R = g.predict_proba(Zte); ent = -(R * np.log(R + 1e-12)).sum(1)
    logp = np.log(R + 1e-30); logp.sort(axis=1); margin = logp[:, -1] - logp[:, -2]

    # split hard anomalies into caught / missed by the nearest-mode detector @5%FPR
    thr = np.quantile(nearest[yw == 0], 0.95)
    caught = hard & (nearest > thr); missed = hard & (nearest <= thr)
    normal = yw == 0
    def stat(m, a): return f"{a[m].mean():.2f}"
    print(f"HARD={int(hard.sum())}  caught={int(caught.sum())}  missed={int(missed.sum())}")
    print(f"\n{'group':<16}{'entropy':>9}{'d2/d1':>8}{'margin':>9}{'nearestMahal':>14}")
    for name, m in [("normal", normal), ("hard-caught", caught), ("hard-missed", missed)]:
        if m.sum():
            print(f"{name:<16}{ent[m].mean():>9.2f}{d2d1[m].mean():>8.2f}{margin[m].mean():>9.2f}{nearest[m].mean():>14.1f}")

    z = lambda a: (a - a.mean()) / (a.std() + 1e-9)
    scores = {"nearest-mode Mahal (ours)": nearest, "full mixture NLL": mixture_nll,
              "responsibility entropy": ent, "neg margin": -margin,
              "nearest + entropy": z(nearest) + z(ent), "nearest + mixtureNLL": z(nearest) + z(mixture_nll)}
    print(f"\n{'detector':<28}{'HARD_AUROC':>11}")
    print("-" * 39)
    for n, s in scores.items():
        print(f"{n:<28}{hauc(yw, s, hard):>11.3f}")


if __name__ == "__main__":
    main()
