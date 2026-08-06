"""Router / 3rd-branch test. VaDE/IF are strong on EASY (out-of-envelope) but weak on HARD;
the latent mode-branches (per-mode Mahalanobis + density) are strong on HARD but weak on EASY.
Add an EASY/overall branch (IsolationForest, trained on normal) and route: treat easy first, defer
the 'looks-normal' residual to the latent hard-branches. Test combination strategies for ALL+EASY+HARD.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from winfeat import window_features
from latad_pipeline import LatentParamAD
import eda_real as E

W, ST = 60, 30


def win(X, y=None):
    Xw, yl = [], []
    for i in range(0, len(X) - W + 1, ST):
        Xw.append(window_features(X[i:i + W], "stats"))
        if y is not None:
            yl.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(Xw, np.float32), (np.asarray(yl, int) if y is not None else None)


def main(name="WADI"):
    D = E.load(name); Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy
    K = {"WADI": 20, "HAI": 24, "SKAB": 12}[name]

    # EASY branch: IsolationForest on features (catches out-of-envelope, low FPR)
    ifm = IsolationForest(n_estimators=200, random_state=0).fit(Xtr)
    sIF = -ifm.decision_function(Xte); sIF_tr = -ifm.decision_function(Xtr)
    # HARD branch: the latent mode pipeline (per-mode Mahal + density, fused)
    P = LatentParamAD(k_modes=K).fit(Xtr); sL = P.score(Xte); sL_tr = P.score(Xtr)

    zt = lambda s, ref: (s - ref.mean()) / (ref.std() + 1e-9)
    zIF, zL = zt(sIF, sIF_tr), zt(sL, sL_tr)
    # asymmetric-OR / cascade helpers: per-branch right-tail p-value from train-normal
    def pval(s_te, s_tr):
        order = np.sort(s_tr); return 1.0 - np.searchsorted(order, s_te, "right") / (len(order) + 1.0)
    pIF, pL = pval(sIF, sIF_tr), pval(sL, sL_tr)

    combos = {
        "IF only (easy)": sIF,
        "latent pipeline (hard)": sL,
        "z-sum (IF + latent)": zIF + zL,
        "max(zIF, zL)": np.maximum(zIF, zL),
        "OR min-p(IF, latent)": -np.log(np.minimum(pIF, pL) + 1e-9),
        "router: IF flag else latent": np.where(zIF > np.quantile(zt(sIF_tr, sIF_tr), 0.95),
                                                zIF + 5.0, zL),  # if IF fires, force high; else latent
    }
    def au(s, mask): k = (yw == 0) | mask; return roc_auc_score(yw[k], s[k])
    print(f"{name}  (easy={int(easy.sum())} hard={int(hard.sum())})")
    print(f"{'strategy':<30}{'ALL':>7}{'EASY':>7}{'HARD':>7}")
    for n, s in combos.items():
        print(f"{n:<30}{au(s, yw==1):>7.3f}{au(s, easy):>7.3f}{au(s, hard):>7.3f}")


if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI", "SKAB"]):
        main(nm); print()
