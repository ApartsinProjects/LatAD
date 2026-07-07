"""What is special about SKAB's missed difficult anomalies and false positives? Characterise each
by mode assignment, responsibility (between-mode?), density percentile, and WITHIN-WINDOW DYNAMICS
(velocity, roughness, drift) vs the normal windows of the SAME mode. Is the fault's dynamics unusual?
"""
from __future__ import annotations
import numpy as np, torch
from scipy.special import logsumexp
from models_vade import train_vade, _as_tensor
from winfeat import window_features
import eda_real as E

W, ST = 60, 30
def win(X, y=None, keepraw=False):
    A, B, RAW = [], [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], "stats"))
        if keepraw: RAW.append(X[i:i + W])
        if y is not None: B.append(int(y[i:i + W].mean() > 0.05))
    return (np.asarray(A, np.float32), np.asarray(B, int) if y is not None else None,
            np.asarray(RAW, np.float32) if keepraw else None)

@torch.no_grad()
def resp(v, Xs):
    mu = v.encode(_as_tensor(Xs, v))[0]
    lN = v._log_pz_given_c(mu).cpu().numpy()
    lp = torch.log_softmax(v.pi_logit, 0).cpu().numpy()[None] + lN
    return np.exp(lp - logsumexp(lp, 1, keepdims=True))

def dyn(RAW):
    """per-window dynamics summaries (mean over channels): velocity |dx|, roughness std(dx), drift."""
    d = np.diff(RAW, axis=1)
    vel = np.abs(d).mean((1, 2)); rough = d.std((1, 2)); drift = np.abs(RAW[:, -1] - RAW[:, 0]).mean(1)
    return vel, rough, drift

def main(seed=0):
    D = E.load("SKAB"); Xtr, _, RAWtr = win(D["Xn_raw"], keepraw=True)
    Xte, yw, RAWte = win(D["Xa_raw"], D["ya_raw"], keepraw=True)
    mu, sg = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = ((Xtr - mu) / sg).astype(np.float32), ((Xte - mu) / sg).astype(np.float32)
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy
    v = train_vade(Xtr_s, n_clusters=16, latent_dim=6, epochs=40, warmup=8, seed=seed)
    v.fit_residual_whitener(Xtr_s); v.fit_latent_density(Xtr_s, k_density=40)
    v.fit_resid_head(Xtr_s); v.fit_basin_head(Xtr_s)
    s = v.anomaly_score_hard(Xte_s, use_resid="auto", use_basin="auto")
    s_tr = v.anomaly_score_hard(Xtr_s, use_resid="auto", use_basin="auto"); thr = np.quantile(s_tr, 0.95)

    Gtr, Gte = resp(v, Xtr_s), resp(v, Xte_s); a_tr, a_te = Gtr.argmax(1), Gte.argmax(1)
    dens_tr, _ = v._hard_components(Xtr_s); dens_te, _ = v._hard_components(Xte_s)
    velT, rghT, drfT = dyn(RAWtr); velE, rghE, drfE = dyn(RAWte)
    ent = lambda G: -(G * np.log(G + 1e-12)).sum(1)

    # per-mode normal reference for dynamics + density
    def mode_ref(mode, arr, a=a_tr):
        m = a == mode
        return (arr[m].mean(), arr[m].std() + 1e-9) if m.sum() else (arr.mean(), arr.std() + 1e-9)
    def pctl(mode, val, arr, a=a_tr):
        m = a == mode; ref = arr[m] if m.sum() > 5 else arr
        return float((ref < val).mean())

    fn = np.where(hard & (s <= thr))[0]; fn = fn[np.argsort(s[fn])][:8]     # worst-missed difficult
    fp = np.where((yw == 0) & (s > thr))[0]; fp = fp[np.argsort(-s[fp])][:8]  # top false positives
    print(f"SKAB: difficult {int(hard.sum())} (missed {int((hard&(s<=thr)).sum())}), FP {int(((yw==0)&(s>thr)).sum())}/{int((yw==0).sum())}")
    print("cols: idx | mode | maxg(between?) | entropy | densPctl(in-mode) | vel% | rough% | drift%  (%=percentile within that mode's NORMAL)")
    def line(i):
        m = a_te[i]
        return (f"{i:5d} m{m:<2d} maxg {Gte[i].max():.2f}  ent {ent(Gte[i:i+1])[0]:.2f}  "
                f"densP {pctl(m, dens_te[i], dens_tr):.2f}  "
                f"vel {pctl(m, velE[i], velT):.2f}  rough {pctl(m, rghE[i], rghT):.2f}  drift {pctl(m, drfE[i], drfT):.2f}")
    print("\n--- MISSED difficult anomalies (false negatives) ---")
    for i in fn: print(line(i))
    print("\n--- FALSE POSITIVES (normal flagged) ---")
    for i in fp: print(line(i))

    # aggregate: are difficult anomalies unusual in dynamics vs normal, or do they hide?
    print("\n--- aggregate: difficult vs normal (median percentile within assigned mode) ---")
    def agg(mask):
        idx = np.where(mask)[0]
        return (np.median([pctl(a_te[i], dens_te[i], dens_tr) for i in idx]),
                np.median([pctl(a_te[i], velE[i], velT) for i in idx]),
                np.median([pctl(a_te[i], rghE[i], rghT) for i in idx]),
                np.median([Gte[i].max() for i in idx]))
    for nm, mk in [("difficult", hard), ("easy", easy), ("normal(test)", yw == 0)]:
        dP, vP, rP, mg = agg(mk)
        print(f"  {nm:14} densP {dP:.2f}  velP {vP:.2f}  roughP {rP:.2f}  maxg {mg:.2f}")

if __name__ == "__main__":
    main()
