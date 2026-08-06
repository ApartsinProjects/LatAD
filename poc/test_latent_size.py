"""Does a LESS-compressive VaDE preserve the hard-anomaly signal? Train VaDE at several
latent sizes and, for each, measure HARD-AUROC of (a) IF ON THE LATENT (how much isolation
signal the latent retains) and (b) the full recon+NLL score and (c) latent-NLL only.
Reference: IF on RAW 738-dim features = 0.695. If IF-on-latent -> 0.695 as latent grows,
'less compression' works; if the full/latNLL score stays flat, the problem is scoring."""
from __future__ import annotations
import numpy as np, torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from models_vade import train_vade, _recon_energy
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


@torch.no_grad()
def latent_and_scores(v, X):
    xt = torch.as_tensor(X, dtype=torch.float32)
    mu = v.encode(xt)[0]
    rec = np.asarray(_recon_energy(xt, v.decode(mu), v.res_whitener))
    lat = -v._log_pz_given_c(mu).max(1).values.cpu().numpy()
    return mu.cpu().numpy(), rec + lat, lat


def hauc(y, s, hard):
    keep = (y == 0) | hard
    return roc_auc_score(y[keep], s[keep])


def main():
    D = E.load("WADI"); Xn, Xa, yp = D["Xn_raw"], D["Xa_raw"], D["ya_raw"]
    Xtr, _ = win(Xn); Xte, yw = win(Xa, yp)
    C6 = Xte.shape[1] // 6
    triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    hard = (yw == 1) & (triv <= np.quantile(trn, 0.99))
    if_raw = hauc(yw, -IsolationForest(n_estimators=200, random_state=0).fit(Xtr).decision_function(Xte), hard)
    print(f"reference: IF on RAW {Xtr.shape[1]}-dim feats = {if_raw:.3f}   (HARD n={int(hard.sum())})\n")
    print(f"{'latent':>7}{'IF-on-latent':>14}{'full recon+NLL':>16}{'latNLL-only':>13}")
    print("-" * 50)
    for ld in [8, 10, 20, 40, 80]:
        v = train_vade(Xtr, 20, ld, epochs=40, warmup=8, seed=0, device="cpu"); v.fit_residual_whitener(Xtr)
        mu_tr, _, _ = latent_and_scores(v, Xtr)
        mu_te, full, lat = latent_and_scores(v, Xte)
        if_lat = hauc(yw, -IsolationForest(n_estimators=200, random_state=0).fit(mu_tr).decision_function(mu_te), hard)
        print(f"{ld:>7}{if_lat:>14.3f}{hauc(yw, full, hard):>16.3f}{hauc(yw, lat, hard):>13.3f}")


if __name__ == "__main__":
    main()
