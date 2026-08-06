"""IsolationForest beats our VaDE on HARD-subset AUROC (0.695 vs 0.50) on WADI. Why, and
do the knobs (more clusters / less compression / C2 basin / C3 per-mode PCA / IF fusion)
close the gap? Measure HARD AUROC + catches@5%FPR for each variant, and inspect the
specific hard windows IF flags that our base VaDE misses."""
from __future__ import annotations
import os, numpy as np, torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from models_vade import train_vade, _recon_energy
import component2, ldt_c
from winfeat import window_features
import eda_real as E

W, ST = 60, 30


def win(X, y=None, rep="stats"):
    Xw, yl = [], []
    for i in range(0, len(X) - W + 1, ST):
        Xw.append(window_features(X[i:i + W], rep))
        if y is not None:
            yl.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(Xw, np.float32), (np.asarray(yl, int) if y is not None else None)


@torch.no_grad()
def parts(v, X):
    """recon-only (whitened residual), latent-NLL-only, full."""
    xt = torch.as_tensor(X, dtype=torch.float32)
    mu = v.encode(xt)[0]; xh = v.decode(mu)
    rec = np.asarray(_recon_energy(xt, xh, v.res_whitener))
    lat = -v._log_pz_given_c(mu).max(1).values.cpu().numpy()
    asg = v._log_pz_given_c(mu).argmax(1).cpu().numpy()
    return rec, lat, rec + lat, asg


def hard_auc(y, s, hard):
    keep = (y == 0) | hard
    return roc_auc_score(y[keep], s[keep])


def catches(s_tr_ref, s, y, hard):
    thr = np.quantile(s[y == 0], 0.95)
    return int(((s > thr) & hard).sum())


def main():
    D = E.load("WADI"); Xn, Xa, yp, ch = D["Xn_raw"], D["Xa_raw"], D["ya_raw"], D["ch"]
    Xtr, _ = win(Xn); Xte, yw = win(Xa, yp)
    C6 = Xte.shape[1] // 6
    triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    hard = (yw == 1) & (triv <= np.quantile(trn, 0.99))
    nh = int(hard.sum()); print(f"WADI HARD windows: {nh}")

    ifs = -IsolationForest(n_estimators=200, random_state=0).fit(Xtr).decision_function(Xte)
    v10 = train_vade(Xtr, 20, 10, epochs=40, warmup=8, seed=0, device="cpu"); v10.fit_residual_whitener(Xtr)
    rec, lat, full, asg = parts(v10, Xte)
    # IsolationForest on the VaDE LATENT (10-dim mu) instead of raw window features
    with torch.no_grad():
        mu_tr = v10.encode(torch.as_tensor(Xtr, dtype=torch.float32))[0].cpu().numpy()
        mu_te = v10.encode(torch.as_tensor(Xte, dtype=torch.float32))[0].cpu().numpy()
    if_lat = -IsolationForest(n_estimators=200, random_state=0).fit(mu_tr).decision_function(mu_te)
    v30 = train_vade(Xtr, 20, 30, epochs=40, warmup=8, seed=0, device="cpu"); v30.fit_residual_whitener(Xtr)
    _, _, full30, _ = parts(v30, Xte)
    vK = train_vade(Xtr, 80, 10, epochs=40, warmup=8, seed=0, device="cpu"); vK.fit_residual_whitener(Xtr)
    _, _, fullK, _ = parts(vK, Xte)
    # C2 basin instability
    agr, conv = component2.basin_features(v10, Xte, restarts=6, steps=60)
    basin = 1.0 - agr
    # C3 per-mode PCA min reconstruction
    _, _, _, asg_tr = parts(v10, Xtr)
    pm = ldt_c.PerModePCA(Xtr, asg_tr, latent_dim=10); pmpca = pm.min_score(Xte)
    # C3 ModeSubsetEnsemble: supervised members trained on random cluster-subsets vs C1 anomalies
    import consolidated
    from component3 import ModeSubsetEnsemble
    x_anom = consolidated.c1_anomalies(Xtr, seed=0)
    ens = ModeSubsetEnsemble(v10, Xtr, x_anom, n_members=10, frac_modes=0.5, seed=0)
    subens = ens.score(Xte)
    z = lambda a: (a - a.mean()) / (a.std() + 1e-9)

    variants = {
        "IsolationForest": ifs,
        "VaDE full (base)": full,
        "VaDE recon-only": rec,
        "VaDE latNLL-only": lat,
        "VaDE latent=30": full30,
        "VaDE K=80": fullK,
        "VaDE + C2 basin": z(full) + z(basin),
        "VaDE + C3 permodePCA": z(full) + z(pmpca),
        "C3 permodePCA only": pmpca,
        "C3 subset-ensemble only": subens,
        "latNLL + C3 subset-ens": z(lat) + z(subens),
        "IF-raw + VaDE": z(ifs) + z(full),
        "IF on VaDE-latent": if_lat,
        "IF-latent + latNLL": z(if_lat) + z(lat),
        "IF-latent + subset-ens": z(if_lat) + z(subens),
        "IF-raw + latNLL + subens": z(ifs) + z(lat) + z(subens),
    }
    print(f"\n{'variant':<24}{'HARD_AUROC':>11}{'hard@5%FPR':>12}")
    print("-" * 47)
    for n, s in variants.items():
        print(f"{n:<24}{hard_auc(yw, s, hard):>11.3f}{catches(None, s, yw, hard):>9}/{nh}")

    # inspect: hard windows IF flags but base VaDE misses (@5%FPR)
    thr_if = np.quantile(ifs[yw == 0], 0.95); thr_v = np.quantile(full[yw == 0], 0.95)
    only_if = np.where(hard & (ifs > thr_if) & (full <= thr_v))[0]
    print(f"\nhard windows IF catches but base VaDE misses: {list(only_if)}")
    for i in only_if:
        zmean = Xa[i * ST:i * ST + W].mean(0); top = np.argsort(-np.abs(zmean))[:4]
        print(f"  win {i}: max|z|={triv[i]:.1f}  " + ", ".join(f"{ch[j]}={zmean[j]:+.1f}" for j in top))


if __name__ == "__main__":
    main()
