"""Root-cause: is VaDE-hard+resid worse than plain VaDE, and why? Decompose per subset and test
a MORE FLEXIBLE per-mode residual gate (apply the whitened residual only in modes where it
generalises to held-out normal, instead of one global on/off for the whole dataset).
"""
from __future__ import annotations
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from models_vade import train_vade, _as_tensor
from winfeat import window_features
import eda_real as E

W, ST = 60, 30
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SKAB": (16, 6)}
def win(X, y=None):
    A, B = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], "stats"))
        if y is not None: B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)
def au(y, s, mask): k = (y == 0) | mask; return roc_auc_score(y[k], s[k])
def z(s, r): return (s - r.mean()) / (r.std() + 1e-9)

def permode_resid_gate(v, Xtr_s, Xte_s, red_dim=30, min_mode=30, seed=0):
    """Per-mode residual: fit per-mode precision on 80% train, keep only modes whose held-out
    (last 20%) normal residual generalises (ratio<1.5); score = responsibility-weighted whitened
    residual over KEPT modes only, others contribute 0."""
    R_tr = v._residual(Xtr_s); R_te = v._residual(Xte_s)
    G_tr = v._responsibilities(Xtr_s); G_te = v._responsibilities(Xte_s); a = G_tr.argmax(1)
    rp = PCA(min(red_dim, R_tr.shape[1]), random_state=seed).fit(R_tr)
    Qtr, Qte = rp.transform(R_tr), rp.transform(R_te)
    nA = int(0.8 * len(Qtr)); aA = a[:nA]
    keep = {}
    rglob = LedoitWolf().fit(Qtr)
    for k in np.unique(a):
        idx = np.where(a == k)[0]
        if len(idx) < min_mode: continue
        est = LedoitWolf().fit(Qtr[idx])
        iA = idx[idx < nA]
        if len(iA) < min_mode: continue
        estA = LedoitWolf().fit(Qtr[iA]); iB = idx[idx >= nA]
        if len(iB) < 5: continue
        ratio = np.quantile(estA.mahalanobis(Qtr[iB]), 0.95) / (np.quantile(estA.mahalanobis(Qtr[iA]), 0.95) + 1e-9)
        if ratio < 1.5: keep[k] = est
    if not keep: return None, 0
    mk_te = np.stack([keep.get(k, rglob).mahalanobis(Qte) if k in keep else np.zeros(len(Qte))
                      for k in range(v.K)], 1)
    mk_tr = np.stack([keep.get(k, rglob).mahalanobis(Qtr) if k in keep else np.zeros(len(Qtr))
                      for k in range(v.K)], 1)
    return (G_te * mk_te).sum(1), (G_tr * mk_tr).sum(1), len(keep)

def main(name, seed=0):
    K, ld = CFG[name]
    D = E.load(name); Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = ((Xtr - m) / sd).astype(np.float32), ((Xte - m) / sd).astype(np.float32)
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy
    v = train_vade(Xtr_s, n_clusters=K, latent_dim=ld, epochs=40, warmup=8, seed=seed)
    v.fit_residual_whitener(Xtr_s); v.fit_latent_density(Xtr_s, k_density=80)
    v.fit_resid_head(Xtr_s); v.fit_basin_head(Xtr_s)

    vade = v.anomaly_score(Xte_s)
    noresid = v.anomaly_score_hard(Xte_s, use_resid=False, use_basin="auto")
    withresid = v.anomaly_score_hard(Xte_s, use_resid="auto", use_basin="auto")
    noresid_tr = v.anomaly_score_hard(Xtr_s, use_resid=False, use_basin="auto")
    pm = permode_resid_gate(v, Xtr_s, Xte_s, seed=seed)
    print(f"\n##### {name} K={K} ld={ld}  resid_auto={'ON' if v._resid_auto else 'off'}"
          f" (gen_ratio {v._resid_gen_ratio:.2f}) #####")
    print(f"{'variant':<28}{'ALL':>7}{'EASY':>7}{'DIFF':>7}")
    for nm, s in [("VaDE (recon+NLL)", vade), ("VaDE-hard (no resid)", noresid),
                  ("VaDE-hard +resid(global auto)", withresid)]:
        print(f"{nm:<28}{au(yw,s,yw==1):>7.3f}{au(yw,s,easy):>7.3f}{au(yw,s,hard):>7.3f}")
    if pm[0] is not None:
        fused = z(noresid, noresid_tr) + z(pm[0], pm[1])
        print(f"{'VaDE-hard +resid(PER-MODE)':<28}{au(yw,fused,yw==1):>7.3f}{au(yw,fused,easy):>7.3f}"
              f"{au(yw,fused,hard):>7.3f}   [{pm[2]}/{K} modes kept]")

if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI", "HAI", "SKAB"]):
        main(nm)
