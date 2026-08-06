"""Inspect WADI false positives (normal flagged) and false negatives (hard anomaly missed) of
the VaDE-hard score, with per-point diagnostics, to motivate C2/C3 variants.
  C2 = basin test (FP fix: rare-valid vs fault via latent-potential basin geometry)
  C3 = per-mode reconstruction ensemble (FN fix: min-over-experts recon error + owner gap)
"""
from __future__ import annotations
import numpy as np, torch
from scipy.special import logsumexp
from models_vade import train_vade, _as_tensor
from winfeat import window_features
import eda_real as E
import component3 as C3

W, ST = 60, 30
def win(X, y=None):
    A, B = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], "stats"))
        if y is not None: B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)

@torch.no_grad()
def resp(v, Xs):
    mu = v.encode(_as_tensor(Xs, v))[0]
    lN = v._log_pz_given_c(mu).cpu().numpy()
    lp = torch.log_softmax(v.pi_logit, 0).cpu().numpy()[None] + lN
    return np.exp(lp - logsumexp(lp, 1, keepdims=True))

def main(name="WADI", K=20, seed=0):
    D = E.load(name); Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = ((Xtr - m) / sd).astype(np.float32), ((Xte - m) / sd).astype(np.float32)
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy

    v = train_vade(Xtr_s, n_clusters=K, latent_dim=10, epochs=40, warmup=8, seed=seed)
    v.fit_residual_whitener(Xtr_s); v.fit_latent_density(Xtr_s, k_density=80); v.fit_resid_head(Xtr_s)
    s = v.anomaly_score_hard(Xte_s, use_resid="auto")
    s_tr = v.anomaly_score_hard(Xtr_s, use_resid="auto")
    thr = np.quantile(s_tr, 0.95)

    G = resp(v, Xte_s); a = G.argmax(1); mx = G.max(1)
    dens, diag = v._hard_components(Xte_s)                                 # density NLL, diag NLL
    ens = C3.ClusterEnsemble(v, Xtr_s, latent_dim=10)                        # C3 experts
    minerr, gap = ens.scores(Xte_s); gerr = ens.global_score(Xte_s)
    def pct(arr, x): return float((arr < x).mean())
    dens_tr, diag_tr = v._hard_components(Xtr_s); minerr_tr, gap_tr = ens.scores(Xtr_s)

    print(f"\n===== {name}: VaDE-hard FP/FN inspection (thr={thr:.2f}) =====")
    fp = np.where((yw == 0) & (s > thr))[0]; fp = fp[np.argsort(-s[fp])][:6]
    fn = np.where(hard & (s <= thr))[0]; fn = fn[np.argsort(s[fn])][:6]
    hdr = f"{'idx':>5}{'score':>7}{'mode':>5}{'maxg':>6}{'triv':>6}{'densP':>7}{'diagP':>7}{'C3minP':>8}{'ownGap':>7}{'glob>ens':>9}"
    def line(i):
        return (f"{i:>5}{s[i]:>7.2f}{a[i]:>5}{mx[i]:>6.2f}{triv[i]:>6.1f}"
                f"{pct(dens_tr, dens[i]):>7.2f}{pct(diag_tr, diag[i]):>7.2f}"
                f"{pct(minerr_tr, minerr[i]):>8.2f}{gap[i]/(gap_tr.mean()+1e-9):>7.2f}"
                f"{gerr[i]/(minerr[i]+1e-9):>9.2f}")
    print(f"--- FALSE POSITIVES (normal, score>thr): {len(np.where((yw==0)&(s>thr))[0])} total ---")
    print(hdr)
    for i in fp: print(line(i))
    print(f"--- FALSE NEGATIVES (DIFFICULT anomaly, missed): {int((hard&(s<=thr)).sum())}/{int(hard.sum())} ---")
    print(hdr)
    for i in fn: print(line(i))
    fe = np.where(easy & (s <= thr))[0]; fe = fe[np.argsort(s[fe])][:6]
    print(f"--- FALSE NEGATIVES (EASY anomaly, missed): {int((easy&(s<=thr)).sum())}/{int(easy.sum())} ---")
    print(hdr)
    for i in fe: print(line(i))
    print("\nlegend: maxg=max responsibility (low=between modes); densP/diagP/C3minP = percentile of that")
    print("score within TRAIN-normal (1.0=extreme); ownGap=owner-gap / train-mean; glob>ens=global-PCA err /")
    print("ensemble-min err (>1 => a single mode explains it better than the global model = C3's target).")


if __name__ == "__main__":
    import sys
    nm = sys.argv[1] if len(sys.argv) > 1 else "WADI"
    main(nm, {"WADI": 20, "HAI": 24, "SKAB": 16}.get(nm, 20))
