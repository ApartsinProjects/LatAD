"""Scaled-down detector: ONE VaDE (VAE + Gaussian-mixture clustering) per real dataset,
plus the C1-C4 scoring stack. No B / no FiLM / no trajectory (C5 is trajectory-only and
adds ~nothing on real data, per the EDA). This is the honest single-model configuration.

Stack on the plain VaDE:
  Base : whitened residual energy  +  nearest-component latent NLL
  C1   : channel-shuffle near-anomalies (discovered modes), oracle-filtered, hard band
  C2   : basin-of-attraction agreement (1 - agreement = instability)
  C3   : per-mode PCA experts (min reconstruction error)  ->  fused by a supervised member
         trained on normal-vs-C1, consuming [s_rec, s_lat, 1-agreement, pca_min]
  C4   : per-mode (mode-conditional) thresholds at 5% FPR

Reports IF / LOF / AE baselines, the base VaDE, and the fused C1-C4 score, per dataset.
"""
from __future__ import annotations
import sys, numpy as np, torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, f1_score
from models_vade import train_vade, _recon_energy
from compare_baselines import ae_scores
import component2, component4, ldt_c
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


def metrics(y, s):
    au = roc_auc_score(y, s); thr = np.quantile(s[y == 0], 0.95)
    return au, float((s[y == 1] > thr).mean()), f1_score(y, s > thr)


@torch.no_grad()
def base_and_assign(v, x, batch=8192):
    """s_rec (whitened residual), s_lat (nearest-component NLL), mode assignment."""
    dev = next(v.parameters()).device
    srec, slat, asg = [], [], []
    for s in range(0, len(x), batch):
        xb = torch.as_tensor(x[s:s + batch], dtype=torch.float32, device=dev)
        mu = v.encode(xb)[0]; xh = v.decode(mu)
        srec.append(np.asarray(_recon_energy(xb, xh, v.res_whitener)))
        lp = v._log_pz_given_c(mu)
        slat.append(-lp.max(1).values.cpu().numpy()); asg.append(lp.argmax(1).cpu().numpy())
    return np.concatenate(srec), np.concatenate(slat), np.concatenate(asg)


def c1_anomalies(x_train, seed=0, n=4000, swap_frac=0.4):
    """C1 on REAL data: DEPENDENCY-VIOLATION anomalies = normal windows with a
    fraction of their feature columns replaced by another normal window's. This
    breaks the JOINT cross-channel structure while leaving each marginal normal
    (the thesis anomaly). Robust on high-dim real data (the synthetic-only oracle /
    in-range / hard-band filters reject everything at 738 dims)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x_train), n)
    base = x_train[idx].copy()
    C = base.shape[1]; cols = rng.choice(C, size=max(1, int(swap_frac * C)), replace=False)
    donor = x_train[rng.integers(0, len(x_train), n)]
    base[:, cols] = donor[:, cols]                        # graft foreign columns -> broken dependencies
    return base.astype(np.float32)


def feats(v, x, pmpca):
    srec, slat, _ = base_and_assign(v, x)
    agr, _ = component2.basin_features(v, x, restarts=4, steps=40)
    return np.stack([srec, slat, 1.0 - agr, pmpca.min_score(x)], axis=1)


def run(name, seed=0):
    D = E.load(name); Xn, Xa, ya = D["Xn_raw"], D["Xa_raw"], D["ya_raw"]
    Xtr, Xte, yte = windowize(Xn, Xa, ya, D["W"], STRIDE[name])
    print(f"\n########## {name}  train={len(Xtr)} test={len(Xte)} feat={Xtr.shape[1]} "
          f"anom={yte.mean():.3f}  K={K[name]} ##########")
    S = {}
    S["IsolationForest"] = -IsolationForest(n_estimators=200, random_state=seed).fit(Xtr).decision_function(Xte)
    S["LOF"] = -LocalOutlierFactor(30, novelty=True).fit(Xtr).decision_function(Xte)
    S["AutoEncoder"] = ae_scores(Xtr, Xte, device="cpu")

    v = train_vade(Xtr, n_clusters=K[name], latent_dim=10, epochs=40, warmup=8, seed=seed, device="cpu")
    v.fit_residual_whitener(Xtr)
    S["VaDE base"] = v.anomaly_score(Xte)

    _, _, asg_tr = base_and_assign(v, Xtr)
    pmpca = ldt_c.PerModePCA(Xtr, asg_tr, latent_dim=10)
    x_anom = c1_anomalies(Xtr, seed=seed)
    fuser = ldt_c.build_fuser(feats(v, Xtr, pmpca), feats(v, x_anom, pmpca), seed=seed)
    fused_tr = fuser.predict_proba(feats(v, Xtr, pmpca))[:, 1]
    fused_te = fuser.predict_proba(feats(v, Xte, pmpca))[:, 1]
    S["VaDE + C1-C3 (fused)"] = fused_te

    # C4 mode-conditional operating point (audit only, not an AUROC)
    _, _, asg_te = base_and_assign(v, Xte)
    gthr = float(np.quantile(fused_tr, 0.95))
    thr = component4.mode_conditional_thresholds(fused_tr, asg_tr, target_fpr=0.05, global_thr=gthr)
    flag = component4.apply_mode_conditional(fused_te, asg_te, thr, gthr)
    c4_tpr = float(flag[yte == 1].mean()); c4_fpr = float(flag[yte == 0].mean())

    print(f"  C1 anomalies: {len(x_anom)}   modes used: {len(np.unique(asg_tr))}/{K[name]}")
    print(f"{'method':<24}{'AUROC':>8}{'TPR@5%':>9}{'F1':>7}")
    for nm, s in S.items():
        au, tp, f1 = metrics(yte, s); print(f"{nm:<24}{au:>8.3f}{tp:>9.3f}{f1:>7.3f}")
    print(f"{'C4 (mode-cond @5%)':<24}{'':>8}{c4_tpr:>9.3f}{'':>7}  (FPR {c4_fpr:.3f})")


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["SKAB", "HAI", "WADI", "SWaT"]
    for nm in names:
        run(nm)
