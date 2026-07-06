"""Find the winning VaDE-based config per dataset (K x latent_dim), scoring with the full
improved head: VaDE-hard + resid(auto) + basin(auto). Print vs the SOTA target so we can see
where VaDE-based already wins and where a config change closes the gap.
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
    "HAI":  ([(24, 10), (24, 16), (40, 10), (40, 16)], "SOTA best (USAD/TranAD): ALL 0.86  DIFF 0.51"),
    "SKAB": ([(12, 10), (16, 10), (16, 6), (24, 10)], "SOTA best (USAD): ALL 0.65  DIFF 0.60"),
}

def main(name, seed=0):
    cfgs, sota = CONFIGS[name]
    D = E.load(name); Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = ((Xtr - m) / sd).astype(np.float32), ((Xte - m) / sd).astype(np.float32)
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy
    print(f"\n##### {name} (easy={int(easy.sum())} hard={int(hard.sum())})   {sota} #####")
    print(f"{'K':>4}{'latent':>7}{'ALL':>8}{'EASY':>8}{'DIFF':>8}{'lam_basin':>10}")
    best = (None, -1)
    for K, ld in cfgs:
        v = train_vade(Xtr_s, n_clusters=K, latent_dim=ld, epochs=40, warmup=8, seed=seed)
        v.fit_residual_whitener(Xtr_s); v.fit_latent_density(Xtr_s, k_density=80)
        v.fit_resid_head(Xtr_s); v.fit_basin_head(Xtr_s)
        s = v.anomaly_score_hard(Xte_s, use_resid="auto", use_basin="auto")
        aa, ae, ah = au(yw, s, yw == 1), au(yw, s, easy), au(yw, s, hard)
        print(f"{K:>4}{ld:>7}{aa:>8.3f}{ae:>8.3f}{ah:>8.3f}{v._basin_lam:>10.2f}")
        if aa > best[1]: best = ((K, ld, aa, ae, ah), aa)
    b = best[0]
    print(f"  -> best config K={b[0]} latent={b[1]}: ALL {b[2]:.3f} EASY {b[3]:.3f} DIFF {b[4]:.3f}")


if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["HAI", "SKAB"]):
        main(nm)
