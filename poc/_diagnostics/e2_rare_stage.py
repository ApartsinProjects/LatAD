"""E2 rare-regime prototypes, stage 0: extra seeds of the E2 VaDE (same config as
e2_fable_stage.py / e2_nearest.py) so every variant can be evaluated per-seed.
Seed 0 is NOT retrained: the existing e2_fable_<name>.npz is symlinked-by-copy of fields.
Output: _diagnostics/e2_rare_<name>_s<seed>.npz (latents + component params only).
"""
from __future__ import annotations
import os, sys, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.dirname(HERE)
sys.path.insert(0, POC); os.chdir(POC)
import numpy as np, torch
from models_vade import train_vade, _as_tensor

CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}
KEEP = ["logN_tr", "logN_te", "logpi", "ztr", "zte", "yw", "hard", "easy", "mu_c", "lvc"]


def out_path(name, seed):
    return os.path.join(HERE, f"e2_rare_{name}_s{seed}.npz")


def stage(name, seed):
    if os.path.exists(out_path(name, seed)):
        print(f"[{name} s{seed}] exists, skip", flush=True); return
    ref = np.load(os.path.join(HERE, f"e2_fable_{name}.npz"))
    if seed == 0:
        np.savez(out_path(name, seed), **{k: ref[k] for k in KEEP})
        print(f"[{name} s0] copied from e2_fable_{name}.npz", flush=True); return
    K, LD = CFG[name]
    Xtr_s, Xte_s = ref["Xtr_s"], ref["Xte_s"]          # identical standardised features as seed 0
    t0 = time.time()
    v = train_vade(Xtr_s, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
    with torch.no_grad():
        ztr = v.encode(_as_tensor(Xtr_s, v))[0]; zte = v.encode(_as_tensor(Xte_s, v))[0]
        logN_tr = v._log_pz_given_c(ztr).cpu().numpy().astype(np.float64)
        logN_te = v._log_pz_given_c(zte).cpu().numpy().astype(np.float64)
        logpi = torch.log_softmax(v.pi_logit, 0).cpu().numpy().astype(np.float64)
        lvc = torch.clamp(v.logvar_c, min=v.logvar_floor).cpu().numpy()
    np.savez(out_path(name, seed), logN_tr=logN_tr, logN_te=logN_te, logpi=logpi,
             ztr=ztr.cpu().numpy(), zte=zte.cpu().numpy(), yw=ref["yw"], hard=ref["hard"], easy=ref["easy"],
             mu_c=v.mu_c.detach().cpu().numpy(), lvc=lvc)
    print(f"[{name} s{seed}] trained+saved in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["HAI"]
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    for nm in names:
        for sd in seeds:
            stage(nm, sd)
