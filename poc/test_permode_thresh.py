"""Do we need per-mode THRESHOLDS for the latent distance, or does per-mode NORMALIZATION
+ a global threshold suffice? Mahalanobis is chi^2_r distributed where r=intrinsic dim varies
per mode, so a global threshold favours low-r modes. Test whether dividing the distance by the
mode's dof (or its median normal distance) makes a GLOBAL threshold match a mode-conditional one.
All thresholds calibrated to the SAME 5% overall test-normal FPR, so we compare recall fairly."""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
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


def main():
    D = E.load("WADI"); Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    hard = (yw == 1) & (triv <= np.quantile(trn, 0.99)); nh = int(hard.sum())
    P = PCA(20, random_state=0).fit(Xtr); Ztr = P.transform(Xtr).astype(float); Zte = P.transform(Xte).astype(float)
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
    # per-mode dof (intrinsic rank, 95% var) and median normal distance
    dof = {}; med = {}
    for k in np.unique(atr):
        Zk = Ztr[atr == k]
        if len(Zk) < 40:
            dof[k] = 20; med[k] = np.median(s_tr[atr == k]); continue
        ev = np.linalg.eigvalsh(np.cov(Zk.T)); ev = ev[::-1]
        dof[k] = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95) + 1)
        med[k] = np.median(s_tr[atr == k])
    dof_te = np.array([dof.get(k, 20) for k in ate]); med_te = np.array([med.get(k, 1.0) for k in ate])
    dof_tr = np.array([dof.get(k, 20) for k in atr]); med_tr = np.array([med.get(k, 1.0) for k in atr])

    def catch_at5(score_te, score_norm_te):
        thr = np.quantile(score_norm_te, 0.95)
        f = score_te > thr
        return int((f & hard).sum()), int((f & (yw == 1)).sum()), float(f[yw == 0].mean())
    def hauc(s):
        keep = (yw == 0) | hard; return roc_auc_score(yw[keep], s[keep])

    variants = {
        "raw Mahal (global thr)": s_te,
        "Mahal / dof (global thr)": s_te / dof_te,
        "Mahal / median_normal (global thr)": s_te / med_te,
    }
    print(f"WADI HARD n={nh}  (all thresholds -> 5% overall test-normal FPR)")
    print(f"{'variant':<34}{'HARD_AUROC':>11}{'hard@5%':>9}{'all@5%':>8}{'FPR':>7}")
    for n, s in variants.items():
        hc, ac, fpr = catch_at5(s, s[yw == 0])
        print(f"{n:<34}{hauc(s):>11.3f}{hc:>7}/{nh}{ac:>6}/56{fpr:>7.3f}")
    # mode-conditional threshold on raw Mahal, calibrated to 5% overall FPR
    #   per-mode threshold = per-mode quantile q; sweep q to hit 5% overall
    for q in [0.95, 0.97, 0.98]:
        mthr = {k: np.quantile(s_te[(yw == 0) & (ate == k)], q) if ((yw == 0) & (ate == k)).sum() > 5
                else np.quantile(s_te[yw == 0], q) for k in np.unique(ate)}
        f = np.array([s_te[i] > mthr[ate[i]] for i in range(len(s_te))])
        print(f"{'mode-cond thr q=' + str(q):<34}{'':>11}{int((f & hard).sum()):>7}/{nh}"
              f"{int((f & (yw == 1)).sum()):>6}/56{f[yw == 0].mean():>7.3f}")


if __name__ == "__main__":
    main()
