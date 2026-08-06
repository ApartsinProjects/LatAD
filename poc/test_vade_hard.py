"""Where does VaDE fail on HARD anomalies -- reconstruction term or latent-NLL term?
VaDE score = recon_energy(whitened residual)  +  (-log p(z | nearest mode)).
Decompose the two terms, score each ALONE on ALL/EASY/HARD, and inspect where hard
windows sit relative to the 95%-normal threshold of each term.
"""
from __future__ import annotations
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from models_vade import train_vade, _recon_energy, _as_tensor
from winfeat import window_features
import eda_real as E

W, ST = 60, 30
def win(X, y=None):
    A, B = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], "stats"))
        if y is not None:
            B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)


@torch.no_grad()
def two_terms(v, Xs):
    """Return (recon_term, latentNLL_term) separately."""
    x = _as_tensor(Xs, v)
    mu, _ = v.encode(x)
    x_hat = v.decode(mu)
    recon = _recon_energy(x, x_hat, v.res_whitener)             # whitened residual energy
    log_near = v._log_pz_given_c(mu).max(dim=1).values.cpu().numpy()
    return np.asarray(recon), -log_near                        # (recon, latentNLL)


def au(y, s, mask):
    k = (y == 0) | mask
    return roc_auc_score(y[k], s[k])


def main(name="WADI", K=20, seed=0):
    D = E.load(name)
    Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = ((Xtr - m) / sd).astype(np.float32), ((Xte - m) / sd).astype(np.float32)
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy

    v = train_vade(Xtr_s, n_clusters=K, latent_dim=10, epochs=40, warmup=8, seed=seed)
    v.fit_residual_whitener(Xtr_s)
    rc_te, nl_te = two_terms(v, Xte_s); rc_tr, nl_tr = two_terms(v, Xtr_s)
    combined = (rc_te - rc_te.mean()) / rc_te.std() + (nl_te - nl_te.mean()) / nl_te.std()

    print(f"\n===== {name}  (easy={int(easy.sum())}  hard={int(hard.sum())}) =====")
    print(f"{'term':<22}{'ALL':>7}{'EASY':>7}{'HARD':>7}")
    for nm, s in [("recon residual", rc_te), ("latent NLL", nl_te),
                  ("combined (VaDE)", combined)]:
        print(f"{nm:<22}{au(yw, s, yw==1):>7.3f}{au(yw, s, easy):>7.3f}{au(yw, s, hard):>7.3f}")

    # where do HARD windows sit? threshold each term at 95% of NORMAL(train), count catches
    print(f"\n{'term':<22}{'thr(95%tr)':>11}{'hard>thr':>10}{'normFPR':>9}"
          f"{'hard med':>10}{'norm med':>10}")
    for nm, ste, str_ in [("recon residual", rc_te, rc_tr), ("latent NLL", nl_te, nl_tr)]:
        thr = np.quantile(str_, 0.95)
        catch = int((ste[hard] > thr).sum()); fpr = float((ste[yw == 0] > thr).mean())
        print(f"{nm:<22}{thr:>11.2f}{catch:>4}/{int(hard.sum()):<5}{fpr:>9.3f}"
              f"{np.median(ste[hard]):>10.2f}{np.median(ste[yw==0]):>10.2f}")

    # diagnosis: for the hard windows, is recon LOW (reconstructs fine) and is the
    # nearest-mode density HIGH (looks in-distribution)?  compare hard vs normal percentiles
    def pct_rank(vals, ref):        # median percentile of vals within ref
        return float(np.mean([(ref < x).mean() for x in vals]))
    print(f"\nhard-window median percentile within NORMAL:")
    print(f"  recon residual : {pct_rank(rc_te[hard], rc_tr):.2f}  "
          f"(1.0 = far above normal = detectable; ~0.5 = looks normal)")
    print(f"  latent NLL     : {pct_rank(nl_te[hard], nl_tr):.2f}  "
          f"(1.0 = far from every mode; ~0.5 = sits inside a mode)")


if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI"]):
        K = {"WADI": 20, "HAI": 24, "SKAB": 16}.get(nm, 20)
        main(nm, K=K)
