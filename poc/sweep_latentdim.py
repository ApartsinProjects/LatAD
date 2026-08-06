"""Selection-driven retrain: participation-ratio said latent_dim=10 is too small for WADI (PR 26.5)
and HAI (PR 16.1); BIC wanted K~40. Sweep latent_dim x n_clusters and measure VaDE-hard on the
DIFFICULT subset (the target). Everything train-normal only.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
from winfeat import window_features
import eda_real as E

W, ST = 60, 30
def win(X, y=None):
    A, B = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], "stats"))
        if y is not None: B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)

def au(y, s, mask): k = (y == 0) | mask; return roc_auc_score(y[k], s[k])

CONFIGS = {
    "WADI": [(10, 20), (26, 20), (26, 40), (10, 40)],
    "HAI":  [(10, 24), (16, 24), (16, 40), (10, 40)],
}

def main(name, seed=0):
    D = E.load(name); Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = ((Xtr - m) / sd).astype(np.float32), ((Xte - m) / sd).astype(np.float32)
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy
    print(f"\n##### {name} (easy={int(easy.sum())} hard={int(hard.sum())}) #####")
    print(f"{'latent_dim':>10}{'K':>5}{'ALL':>8}{'EASY':>8}{'HARD':>8}")
    for ld, K in CONFIGS[name]:
        v = train_vade(Xtr_s, n_clusters=K, latent_dim=ld, epochs=40, warmup=8, seed=seed)
        v.fit_residual_whitener(Xtr_s); v.fit_latent_density(Xtr_s, k_density=80); v.fit_resid_head(Xtr_s)
        s = v.anomaly_score_hard(Xte_s, use_resid="auto")
        tag = "  <- base" if (ld, K) == CONFIGS[name][0] else ""
        print(f"{ld:>10}{K:>5}{au(yw,s,yw==1):>8.3f}{au(yw,s,easy):>8.3f}{au(yw,s,hard):>8.3f}{tag}")


if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI", "HAI"]):
        main(nm)
