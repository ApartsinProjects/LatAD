"""Why does USAD win SKAB-ALL, and can VaDE-hard overtake it? USAD is reconstruction-based; we
DROP reconstruction in VaDE-hard. Test whether adding recon back (globally / difficult-only),
window size, and per-mode thresholds close the ~0.03 AUC gap. Also align USAD per-window and see
WHICH windows it catches that we miss (easy vs difficult, density percentile).
"""
from __future__ import annotations
import os, numpy as np, torch
from sklearn.metrics import roc_auc_score
from models_vade import train_vade, _as_tensor
from winfeat import window_features
import eda_real as E

RES = os.path.join(os.path.dirname(__file__), "sota_bundle", "results")
def au(y, s, mask): k = (y == 0) | mask; return roc_auc_score(y[k], s[k])
def z(s, r): return (s - r.mean()) / (r.std() + 1e-9)

def winf(X, W, ST, rep="stats", y=None):
    A, B = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], rep))
        if y is not None: B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)

def fit(Xtr_s, K=16, ld=6, seed=0):
    v = train_vade(Xtr_s, n_clusters=K, latent_dim=ld, epochs=40, warmup=8, seed=seed)
    v.fit_residual_whitener(Xtr_s); v.fit_latent_density(Xtr_s, k_density=80)
    v.fit_resid_head(Xtr_s); v.fit_basin_head(Xtr_s)
    return v

def main():
    D = E.load("SKAB")
    # window sweep
    print("=== SKAB window sweep (VaDE-hard+resid+basin) ===")
    print(f"{'W':>4}{'ST':>4}{'ALL':>8}{'EASY':>8}{'DIFF':>8}")
    for W, ST in [(30, 15), (60, 30), (120, 60)]:
        Xtr, _ = winf(D["Xn_raw"], W, ST); Xte, yw = winf(D["Xa_raw"], W, ST, y=D["ya_raw"])
        mu, sg = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr_s, Xte_s = ((Xtr - mu) / sg).astype(np.float32), ((Xte - mu) / sg).astype(np.float32)
        C6 = Xte.shape[1] // 6; tv = np.abs(Xte[:, :C6]).max(1); tn = np.abs(Xtr[:, :C6]).max(1)
        ez = (yw == 1) & (tv > np.quantile(tn, 0.99)); hd = (yw == 1) & ~ez
        v = fit(Xtr_s)
        s = v.anomaly_score_hard(Xte_s, use_resid="auto", use_basin="auto")
        print(f"{W:>4}{ST:>4}{au(yw,s,yw==1):>8.3f}{au(yw,s,ez):>8.3f}{au(yw,s,hd):>8.3f}")

    # recon variants at W=60 + USAD alignment
    W, ST = 60, 30
    Xtr, _ = winf(D["Xn_raw"], W, ST); Xte, yw = winf(D["Xa_raw"], W, ST, y=D["ya_raw"])
    mu, sg = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = ((Xtr - mu) / sg).astype(np.float32), ((Xte - mu) / sg).astype(np.float32)
    C6 = Xte.shape[1] // 6; tv = np.abs(Xte[:, :C6]).max(1); tn = np.abs(Xtr[:, :C6]).max(1)
    ez = (yw == 1) & (tv > np.quantile(tn, 0.99)); hd = (yw == 1) & ~ez
    v = fit(Xtr_s)
    base = v.anomaly_score_hard(Xte_s, use_resid="auto", use_basin="auto"); base_tr = v.anomaly_score_hard(Xtr_s, use_resid="auto", use_basin="auto")
    recon = v.anomaly_score_hard(Xte_s, use_recon=True, use_resid="auto", use_basin="auto")
    vade = v.anomaly_score(Xte_s); vade_tr = v.anomaly_score(Xtr_s)
    fuse_vh = z(base, base_tr) + z(vade, vade_tr)
    print("\n=== SKAB recon variants (W=60) ===")
    print(f"{'variant':<26}{'ALL':>8}{'EASY':>8}{'DIFF':>8}")
    for nm, s in [("VaDE-hard (base)", base), ("VaDE-hard +recon", recon),
                  ("VaDE (recon+NLL)", vade), ("VaDE-hard + VaDE (z-sum)", fuse_vh)]:
        print(f"{nm:<26}{au(yw,s,yw==1):>8.3f}{au(yw,s,ez):>8.3f}{au(yw,s,hd):>8.3f}")

    # USAD alignment (SKAB full-res, window at W/ST like ours)
    p = f"{RES}/score_USAD_SKAB.npy"
    if os.path.exists(p):
        sp = np.load(p); sp = sp.mean(1) if sp.ndim > 1 else sp
        Nfull = len(D["Xa_raw"])
        if len(sp) < Nfull: sp = np.pad(sp, (0, Nfull - len(sp)), mode="edge")
        usad = np.array([sp[i:i + W].max() for i in range(0, Nfull - W + 1, ST)])[:len(yw)]
        m = min(len(usad), len(yw)); u, yy, e2, h2 = usad[:m], yw[:m], ez[:m], hd[:m]; b2 = base[:m]
        print(f"\n=== USAD vs ours (SKAB, {m} win) ===")
        print(f"{'USAD':<12}{au(yy,u,yy==1):>8.3f}{au(yy,u,e2):>8.3f}{au(yy,u,h2):>8.3f}")
        print(f"{'ours(base)':<12}{au(yy,b2,yy==1):>8.3f}{au(yy,b2,e2):>8.3f}{au(yy,b2,h2):>8.3f}")
        # where USAD wins: anomalies USAD ranks high but ours ranks low
        ru = u.argsort().argsort() / len(u); rb = b2.argsort().argsort() / len(b2)   # percentile ranks
        pos = yy == 1
        usad_only = pos & (ru > 0.9) & (rb < 0.6)     # USAD flags, ours doesn't
        ours_only = pos & (rb > 0.9) & (ru < 0.6)
        print(f"USAD-catches-ours-misses: {int(usad_only.sum())} ({int((usad_only&e2).sum())} easy, {int((usad_only&h2).sum())} diff)")
        print(f"ours-catches-USAD-misses: {int(ours_only.sum())} ({int((ours_only&e2).sum())} easy, {int((ours_only&h2).sum())} diff)")

if __name__ == "__main__":
    main()
