"""A3 support screen: rho = fraction of train-normal windows in thin between-regime pockets
(max component-responsibility < 0.5, the basin-head A3 signature). SKAB (rho=0.58) is the positive
reference; WADI/HAI/SWaT (~0.01-0.02) the negative. A candidate 'supports A3' if rho is comparably
high. Consistent K=16 across datasets so rho is comparable. Fast: one VaDE fit per dataset, no scoring.
"""
from __future__ import annotations
import os, sys, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eda_real as E
from models_vade import train_vade
HERE = os.path.dirname(os.path.abspath(__file__))
K, LD = 16, 8

REFS = ["SKAB", "WADI", "HAI", "SWaT"]                    # SKAB+ , others -
CANDS = ["MetroPT", "TEP", "BATADAL", "PSM"]              # untested, loaders exist


def rho_of(name):
    D = E.load(name)
    Xtr0 = np.asarray(D["Xn_w"], np.float32)
    n_anom = int(np.asarray(D.get("ya_w", np.zeros(1)), int).sum())
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32)
    v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=30, warmup=6, seed=0, device="cpu")
    G = v._responsibilities(Xtr)                          # (n, K)
    mr = G.max(1)
    return dict(n_train=len(Xtr), n_test_anom=n_anom, mean_maxresp=round(float(mr.mean()), 3),
                rho=round(float((mr < 0.5).mean()), 3), rho_0p6=round(float((mr < 0.6).mean()), 3))


if __name__ == "__main__":
    t0 = time.time(); out = {}
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else REFS + CANDS
    for nm in names:
        try:
            r = rho_of(nm); out[nm] = r
            tag = "REF" if nm in REFS else "CAND"
            a3 = "A3-SUPPORT" if r["rho"] >= 0.30 else ("weak" if r["rho"] >= 0.10 else "no-A3")
            print(f"  [{tag:4s}] {nm:10s} rho(<0.5)={r['rho']:.3f}  rho(<0.6)={r['rho_0p6']:.3f}  "
                  f"mean_maxresp={r['mean_maxresp']:.3f}  n_tr={r['n_train']}  -> {a3}", flush=True)
        except Exception as e:
            print(f"  [{nm}] ERR {type(e).__name__}: {e}", flush=True); out[nm] = {"error": str(e)}
    json.dump(out, open(os.path.join(HERE, "a3_rho_screen.json"), "w"), indent=1)
    print(f"\nreference: SKAB rho~0.58 (A3 holds); WADI/HAI/SWaT ~0.01-0.02 (no A3). done ({time.time()-t0:.0f}s)")
