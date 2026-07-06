"""Two candidate improvements:
  (Q3) IsolationForest in the VaDE LATENT (vs on features) -- standalone and fused with VaDE-hard.
  (A8) TEMPORAL window features (feat_temporal: slope, velocity, spectral band power) instead of
       static 'stats' -- 'difficult' anomalies ARE dynamics/correlation faults, which stats discard.
Winning per-dataset (K, latent) configs. Target: lift the DIFFICULT column, esp. SKAB (>0.60).
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from models_vade import train_vade
from winfeat import window_features
import eda_real as E

W, ST = 60, 30
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SKAB": (16, 6)}

def win(X, rep, y=None):
    A, B = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], rep))
        if y is not None: B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)

def au(y, s, mask): k = (y == 0) | mask; return roc_auc_score(y[k], s[k])
def z(s, r): return (s - r.mean()) / (r.std() + 1e-9)

def fit_score(Xtr, Xte, K, ld, seed=0):
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = ((Xtr - m) / sd).astype(np.float32), ((Xte - m) / sd).astype(np.float32)
    v = train_vade(Xtr_s, n_clusters=K, latent_dim=ld, epochs=40, warmup=8, seed=seed)
    v.fit_residual_whitener(Xtr_s); v.fit_latent_density(Xtr_s, k_density=80)
    v.fit_resid_head(Xtr_s); v.fit_basin_head(Xtr_s)
    return v, Xtr_s, Xte_s

def main(name, seed=0):
    K, ld = CFG[name]
    D = E.load(name)
    # difficulty split from STATS triv (definition unchanged)
    Xtr_st, _ = win(D["Xn_raw"], "stats"); Xte_st, yw = win(D["Xa_raw"], "stats", D["ya_raw"])
    C6 = Xte_st.shape[1] // 6; triv = np.abs(Xte_st[:, :C6]).max(1); trn = np.abs(Xtr_st[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy
    print(f"\n##### {name} (easy={int(easy.sum())} hard={int(hard.sum())}) K={K} latent={ld} #####")
    print(f"{'variant':<26}{'ALL':>7}{'EASY':>7}{'DIFF':>7}")

    # --- stats features ---
    v, Xtr_s, Xte_s = fit_score(Xtr_st, Xte_st, K, ld, seed)
    base = v.anomaly_score_hard(Xte_s, use_resid="auto", use_basin="auto")
    base_tr = v.anomaly_score_hard(Xtr_s, use_resid="auto", use_basin="auto")
    Ztr, Zte = v._encode_mean(Xtr_s), v._encode_mean(Xte_s)
    ifl = IsolationForest(n_estimators=200, random_state=seed).fit(Ztr)
    sL, sL_tr = -ifl.decision_function(Zte), -ifl.decision_function(Ztr)
    fused = z(base, base_tr) + z(sL, sL_tr)
    for nm, s in [("VaDE-hard (stats)", base), ("IF-in-latent alone", sL), ("VaDE-hard + IF-latent", fused)]:
        print(f"{nm:<26}{au(yw,s,yw==1):>7.3f}{au(yw,s,easy):>7.3f}{au(yw,s,hard):>7.3f}")

    # --- temporal features (A8) ---
    Xtr_tp, _ = win(D["Xn_raw"], "temporal"); Xte_tp, _ = win(D["Xa_raw"], "temporal", D["ya_raw"])
    vt, Xtr_t, Xte_t = fit_score(Xtr_tp, Xte_tp, K, ld, seed)
    base_t = vt.anomaly_score_hard(Xte_t, use_resid="auto", use_basin="auto")
    print(f"{'VaDE-hard (temporal A8)':<26}{au(yw,base_t,yw==1):>7.3f}{au(yw,base_t,easy):>7.3f}{au(yw,base_t,hard):>7.3f}")


if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI", "HAI", "SKAB"]):
        main(nm)
