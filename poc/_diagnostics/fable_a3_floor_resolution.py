"""Variance-floor resolution for the SKAB A3 witness. Retrain VaDE (SKAB, K16/LD6/ep40, the reported
config) at logvar_floor in {log 0.01, log 0.02, log 0.05 (default)}, seeds 0-2, and report per fit:
  rho / H_norm with the model's floored variances AND with empirical within-component variances;
  basin-head witness: difficult-subset AUROC OFF vs FORCED-ON(lam=1,2), exactly as a3_seed_robustness.
The floor is a model attribute set in VaDE.__init__; it is changed here by a LOCAL monkeypatch of
VaDE.__init__ (models_vade.py is not edited).
Invariant (before running): at the default floor, seed 0 must reproduce rho=0.58, entropy 0.291,
OFF 0.500 / ON(1) 0.605 (lift +0.105); seeds 0-2 mean lift ~ +0.07 (a3_seed_robustness.json).
Benchmark check: WADI (K20/LD10) at floor 0.01 vs 0.05, rho/H stay low.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, math
import numpy as np, torch
from scipy.special import logsumexp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import eda_real as E
import models_vade as MV
from a3_witness import win, au, CFG

FLOOR = {"floor": 0.05}
_orig_init = MV.VaDE.__init__


def _patched_init(self, *a, **k):
    _orig_init(self, *a, **k)
    self.logvar_floor = math.log(FLOOR["floor"])


MV.VaDE.__init__ = _patched_init
LOG2PI = np.log(2 * np.pi)


def resp(z, mu, lv, lp):
    logN = -0.5 * (LOG2PI * z.shape[1] + (lv[None] + (z[:, None, :] - mu[None]) ** 2 / np.exp(lv[None])).sum(2))
    l = lp[None] + logN; return np.exp(l - logsumexp(l, 1, keepdims=True))


def stats(v, X):
    G = v._responsibilities(X); K = G.shape[1]
    Hn = lambda G_: float(np.mean(-(G_ * np.log(G_ + 1e-12)).sum(1) / np.log(K)))
    with torch.no_grad():
        z = v.encode(torch.as_tensor(X))[0].numpy().astype(np.float64)
    lvc = v._lvc().detach().numpy(); mu = v.mu_c.detach().numpy(); lp = torch.log_softmax(v.pi_logit, 0).detach().numpy()
    lab = G.argmax(1)
    var_hat = np.stack([z[lab == k].var(0) + 1e-4 if (lab == k).sum() > 1 else np.exp(lvc[k]) for k in range(K)])
    Ge = resp(z, mu, np.log(var_hat), lp)
    return dict(rho=float((G.max(1) < 0.5).mean()), H=Hn(G), rho_emp=float((Ge.max(1) < 0.5).mean()), H_emp=Hn(Ge),
                frac_at_floor=float((np.abs(lvc - v.logvar_floor) < 1e-6).mean()),
                comp_sd=float(np.exp(0.5 * lvc).mean()), emp_sd=float(np.sqrt(var_hat).mean()))


name = "SKAB"; K, LD = CFG[name]
D = E.load(name)
Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
Xtr_s = ((Xtr - m) / sd).astype(np.float32); Xte_s = ((Xte - m) / sd).astype(np.float32)
C6 = Xte.shape[1] // 6
triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy
kd = min(80, max(20, len(Xtr_s) // 10))
print(f"SKAB train {len(Xtr_s)} test {len(Xte_s)} hard {int(hard.sum())} easy {int(easy.sum())}", flush=True)

OUT = os.path.join(HERE, "fable_a3_floor_resolution.json")
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
for floor in (0.05, 0.02, 0.01):
    FLOOR["floor"] = floor
    for seed in range(3):
        key = f"SKAB_floor{floor}_s{seed}"
        if key in R: print("cached", key, R[key]); continue
        v = MV.train_vade(Xtr_s, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
        assert abs(v.logvar_floor - math.log(floor)) < 1e-9
        v.fit_residual_whitener(Xtr_s); v.fit_latent_density(Xtr_s, k_density=kd)
        v.fit_resid_head(Xtr_s); v.fit_basin_head(Xtr_s)
        st = stats(v, Xtr_s)
        off_te = np.asarray(v.anomaly_score_hard(Xte_s, use_resid="auto", use_basin=False))
        ag_te = v._noise_agreement(Xte_s); am, asd = v._basin_ref
        on = lambda lam: off_te - lam * (ag_te - am) / asd
        a_off, a1, a2 = au(yw, off_te, hard), au(yw, on(1.0), hard), au(yw, on(2.0), hard)
        a_off_all, a1_all = au(yw, off_te, yw == 1), au(yw, on(1.0), yw == 1)
        r = dict(floor=floor, seed=seed, **{k: round(x, 3) for k, x in st.items()}, basin_lam=round(float(v._basin_lam), 3),
                 diff_OFF=round(a_off, 3), diff_ON1=round(a1, 3), diff_ON2=round(a2, 3),
                 lift1=round(a1 - a_off, 3), lift2=round(a2 - a_off, 3), all_OFF=round(a_off_all, 3), all_ON1=round(a1_all, 3))
        R[key] = r; json.dump(R, open(OUT, "w"), indent=1)
        print(f"floor {floor} seed {seed}: rho={r['rho']} H={r['H']} | emp rho={r['rho_emp']} H={r['H_emp']} | at_floor={r['frac_at_floor']} sd {r['comp_sd']} vs emp {r['emp_sd']} "
              f"| lam={r['basin_lam']} OFF={r['diff_OFF']} ON1={r['diff_ON1']} lift1={r['lift1']:+.3f} ON2={r['diff_ON2']} lift2={r['lift2']:+.3f} | all OFF {r['all_OFF']} ON1 {r['all_ON1']}", flush=True)

# benchmark check: WADI at floor 0.01 vs 0.05 (paper config K20/LD10)
Xw = np.load(os.path.join(HERE, "e2_fable_WADI.npz"))["Xtr_s"].astype(np.float32)
for floor in (0.05, 0.01):
    key = f"WADI_floor{floor}_s0"
    if key in R: print("cached", key, R[key]); continue
    FLOOR["floor"] = floor
    v = MV.train_vade(Xw, n_clusters=20, latent_dim=10, epochs=40, warmup=8, seed=0, device="cpu")
    v.fit_basin_head(Xw)
    st = stats(v, Xw); st["basin_lam"] = float(v._basin_lam)
    R[key] = {k: round(x, 3) for k, x in st.items()}; json.dump(R, open(OUT, "w"), indent=1)
    print(f"WADI floor {floor}: {R[key]}", flush=True)
print("done", flush=True)
