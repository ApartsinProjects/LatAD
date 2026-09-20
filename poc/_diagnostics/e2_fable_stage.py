"""E2 adversarial audit, stage 1: retrain the E2 VaDE (identical config/seed to e2_nearest.py),
cache the model + per-window per-component log N(z|c) + log pi + labels so all later
analysis is offline. Output: _diagnostics/e2_fable_<name>.npz and e2_fable_<name>.pt
"""
from __future__ import annotations
import os, sys, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.dirname(HERE)
sys.path.insert(0, POC); os.chdir(POC)
import numpy as np, torch
from models_vade import train_vade, _as_tensor
from winfeat import window_features
import eda_real as E

W, ST = 60, 30
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}


def win(X, y=None):
    A, B, I = [], [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], "stats")); I.append(i)
        if y is not None:
            B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None), np.asarray(I)


def logN_of(v, X):
    with torch.no_grad():
        mu = v.encode(_as_tensor(X, v))[0]
        logN = v._log_pz_given_c(mu).cpu().numpy().astype(np.float64)
        logpi = torch.log_softmax(v.pi_logit, 0).cpu().numpy().astype(np.float64)
        z = mu.cpu().numpy()
    return logN, logpi, z


def stage(name, seed=0):
    K, LD = CFG[name]
    D = E.load(name)
    Xtr, _, itr = win(D["Xn_raw"])
    Xte, yw, ite = win(D["Xa_raw"], D["ya_raw"])
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s = ((Xtr - m) / sd).astype(np.float32)
    Xte_s = ((Xte - m) / sd).astype(np.float32)
    C6 = Xte.shape[1] // 6
    triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy
    print(f"[{name}] train windows {Xtr.shape}, test {Xte.shape}, anomalies {yw.sum()}, hard {hard.sum()}", flush=True)
    t0 = time.time()
    v = train_vade(Xtr_s, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
    print(f"[{name}] trained in {time.time()-t0:.0f}s", flush=True)
    logN_tr, logpi, ztr = logN_of(v, Xtr_s)
    logN_te, _, zte = logN_of(v, Xte_s)
    lvc = torch.clamp(v.logvar_c, min=v.logvar_floor).detach().cpu().numpy()
    np.savez(os.path.join(HERE, f"e2_fable_{name}.npz"), logN_tr=logN_tr, logN_te=logN_te, logpi=logpi,
             ztr=ztr, zte=zte, yw=yw, hard=hard, easy=easy, itr=itr, ite=ite, mu_c=v.mu_c.detach().cpu().numpy(),
             lvc=lvc, Xtr_s=Xtr_s, Xte_s=Xte_s, m=m, sd=sd)
    torch.save(v.state_dict(), os.path.join(HERE, f"e2_fable_{name}.pt"))
    print(f"[{name}] saved", flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["HAI"]):
        stage(nm)
