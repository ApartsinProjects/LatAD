"""A8 (between-regime overlap) screen on Cranfield Multiphase Flow Facility.

Loads the three normal-operation regimes (Training.mat T1/T2/T3, 24 process
vars, 1 Hz), pools them as train-normal (same treatment as the WADI/HAI/SWaT
A8 screen: overlap is read off the VaDE's own discovered-component
responsibilities, not off externally-labeled regime IDs), windows with
window_features(..., "stats"), and sweeps K = 8..128 reporting:
  mean_maxresp  : mean of the top responsibility per window (1 = fully
                  assigned to one component, lower = split across components)
  H_norm        : responsibility entropy normalized by log(K) in [0,1]
  rho           : fraction of windows with max-responsibility < 0.5 (thin
                  between-component pocket, the A3/A8 basin-head signature)
Reference (established, from a3_rho_screen.json / a3_ksweep.json):
  SKAB   entropy ~0.06 (K8) rising to ~0.53 (K64)  -- POSITIVE (A8 present, but
         later shown to be a variance-floor artifact; see fable_a3_floor_resolution)
  WADI   ~0.03-0.07 across K   -- NO overlap
  HAI    ~0.08-0.15 across K   -- NO overlap
  SWaT   ~0.01-0.02 across K   -- NO overlap

Variance-floor sanity check (mandatory before crediting any high entropy):
refit responsibilities at the SAME latent z using EMPIRICALLY estimated
per-component variance (assign-then-refit) instead of the model's floored
logvar_c (exp(logvar_floor) = 0.05). If entropy collapses under the empirical
refit, the apparent overlap is a floor artifact (this is exactly what
happened to the SKAB "A3 witness"; see fable_a3_floor_resolution.py).

Persists incrementally to cranfield_a8_screen.json (one row per K, flushed).
"""
from __future__ import annotations
import os, sys, json, warnings, time, math
warnings.filterwarnings("ignore")
import numpy as np
import torch
from scipy.io import loadmat
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
POC = os.path.dirname(HERE)
sys.path.insert(0, POC)
from models_vade import train_vade
import models_vade as MV
from winfeat import window_features

MAT = os.path.join(POC, "datasets", "_new", "Cranfield", "CUcasestudy", "CUcasestudy", "Training.mat")
OUT = os.path.join(HERE, "cranfield_a8_screen.json")
W, ST = 20, 10          # 1 Hz sampling: 20 s windows / 10 s stride (matches WADI-style W/2 stride)
LD = 8                   # latent dim, consistent with the a3_rho_screen CAND config
KS = [8, 16, 32, 64, 128]
LOG2PI = np.log(2 * np.pi)


def load_cranfield_normal():
    D = loadmat(MAT)
    regs = {}
    for k in ("T1", "T2", "T3"):
        assert k in D, f"{k} missing from Training.mat, keys={[x for x in D if not x.startswith('__')]}"
        regs[k] = np.asarray(D[k], np.float64)
    return regs


def window(X, regime_id):
    A, R = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], "stats"))
        R.append(regime_id)
    return np.asarray(A, np.float32), np.asarray(R, int)


def entropy_norm(G):
    K = G.shape[1]
    return float(np.mean(-(G * np.log(G + 1e-12)).sum(1) / np.log(K)))


def resp_from_var(z, mu, var, logpi):
    lv = np.log(var)
    logN = -0.5 * (LOG2PI * z.shape[1] + (lv[None] + (z[:, None, :] - mu[None]) ** 2 / var[None]).sum(2))
    l = logpi[None] + logN
    return np.exp(l - logsumexp(l, 1, keepdims=True))


def empirical_refit(v, X):
    """Re-derive responsibilities at the SAME encoder latent z, replacing the
    model's floored per-component variance with the empirically observed
    within-assigned-cluster variance (var-floor sanity check)."""
    with torch.no_grad():
        z = v.encode(torch.as_tensor(X))[0].numpy().astype(np.float64)
    G0 = v._responsibilities(X)
    lab = G0.argmax(1)
    mu = v.mu_c.detach().numpy().astype(np.float64)
    lvc = v._lvc().detach().numpy().astype(np.float64)
    logpi = torch.log_softmax(v.pi_logit, 0).detach().numpy().astype(np.float64)
    K = mu.shape[0]
    var_hat = np.stack([
        z[lab == k].var(0) + 1e-4 if (lab == k).sum() > 1 else np.exp(lvc[k])
        for k in range(K)
    ])
    Ge = resp_from_var(z, mu, var_hat, logpi)
    frac_at_floor = float((np.abs(lvc - v.logvar_floor) < 1e-6).mean())
    return Ge, frac_at_floor, float(np.exp(0.5 * lvc).mean()), float(np.sqrt(var_hat).mean())


def main():
    print("Loading Cranfield normal regimes (T1/T2/T3)...", flush=True)
    regs = load_cranfield_normal()
    for k, v in regs.items():
        print(f"  {k}: {v.shape}", flush=True)

    Xw_list, Rw_list = [], []
    for i, k in enumerate(("T1", "T2", "T3")):
        Xw, Rw = window(regs[k], i)
        Xw_list.append(Xw); Rw_list.append(Rw)
    Xw = np.concatenate(Xw_list, 0)
    Rw = np.concatenate(Rw_list, 0)
    print(f"Pooled windows: {Xw.shape}, per-regime counts {np.bincount(Rw)}", flush=True)

    mu, sig = Xw.mean(0), Xw.std(0) + 1e-8
    Xs = ((Xw - mu) / sig).astype(np.float32)

    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    t0 = time.time()
    for K in KS:
        key = f"K{K}"
        if key in R and "H_norm" in R[key]:
            print(f"cached {key}: {R[key]}", flush=True)
            continue
        v = train_vade(Xs, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=0, device="cpu")
        G = v._responsibilities(Xs)
        mr = G.max(1)
        row = dict(K=K, n_windows=int(len(Xs)),
                   mean_maxresp=round(float(mr.mean()), 4),
                   H_norm=round(entropy_norm(G), 4),
                   rho_lt0p5=round(float((mr < 0.5).mean()), 4),
                   rho_lt0p6=round(float((mr < 0.6).mean()), 4))
        R[key] = row
        json.dump(R, open(OUT, "w"), indent=1)
        print(f"[K={K:3d}] mean_maxresp={row['mean_maxresp']:.3f} H_norm={row['H_norm']:.3f} "
              f"rho(<0.5)={row['rho_lt0p5']:.3f} rho(<0.6)={row['rho_lt0p6']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # pick the K with the HIGHEST entropy for the variance-floor sanity check
    best_key = max((k for k in R if k.startswith("K") and "H_norm" in R[k]), key=lambda k: R[k]["H_norm"])
    best_K = R[best_key]["K"]
    print(f"\nHighest-entropy K = {best_K} (H_norm={R[best_key]['H_norm']}); "
          f"running variance-floor sanity check across seeds 0-2 ...", flush=True)

    if "variance_floor_check" not in R:
        R["variance_floor_check"] = {}
    for seed in range(3):
        skey = f"K{best_K}_seed{seed}"
        if skey in R["variance_floor_check"]:
            print(f"cached {skey}: {R['variance_floor_check'][skey]}", flush=True)
            continue
        v = train_vade(Xs, n_clusters=best_K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
        G = v._responsibilities(Xs)
        H_floor = entropy_norm(G)
        rho_floor = float((G.max(1) < 0.5).mean())
        Ge, frac_at_floor, comp_sd, emp_sd = empirical_refit(v, Xs)
        H_emp = entropy_norm(Ge)
        rho_emp = float((Ge.max(1) < 0.5).mean())
        row = dict(K=best_K, seed=seed, H_floor=round(H_floor, 4), rho_floor=round(rho_floor, 4),
                   H_emp=round(H_emp, 4), rho_emp=round(rho_emp, 4),
                   frac_components_at_floor=round(frac_at_floor, 4),
                   floored_comp_sd=round(comp_sd, 4), empirical_comp_sd=round(emp_sd, 4),
                   collapse=bool(H_emp < 0.5 * H_floor))
        R["variance_floor_check"][skey] = row
        json.dump(R, open(OUT, "w"), indent=1)
        print(f"[floor-check K={best_K} seed={seed}] H_floor={row['H_floor']:.3f} -> H_emp={row['H_emp']:.3f} "
              f"(rho {row['rho_floor']:.3f} -> {row['rho_emp']:.3f})  frac_at_floor={row['frac_components_at_floor']:.2f} "
              f"comp_sd(floored)={row['floored_comp_sd']:.3f} vs emp_sd={row['empirical_comp_sd']:.3f}  "
              f"collapse={row['collapse']}", flush=True)

    print(f"\ndone ({time.time()-t0:.0f}s total). See {OUT}", flush=True)


if __name__ == "__main__":
    main()
