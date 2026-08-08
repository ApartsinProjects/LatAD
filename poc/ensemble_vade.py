"""Ensemble of VaDE latent+clustering models, each trained on a RANDOM SENSOR SUBSET
(user idea). Each member is a full VaDE (jointly-learned latent + Gaussian-mixture clustering)
on the 6 per-channel window statistics of its subset; anomaly score = latent-only density
(mixture NLL + nearest-component, the LatAD base). A subset containing a sparse coherent
block sees it undiluted, and VaDE's multimodal high-K density (A1/A2/A4) can flag a joint
low-level that the full 123-channel model averages away.

Aggregate over members (max and top-3 mean of standardized scores). Reports difficult-subset
AUROC vs LatAD/IF, and the miss-rescue-wrapped fusion. Exploratory (test-informed groups /
test-normal calibration). Grounded in A1/A7 (per-subset mode discovery) + A5 (few levers).
"""
from __future__ import annotations
import json, os, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
import eda_real as E
import cycle_lib as CL

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
RNG = np.random.default_rng(0)
K = 24                        # ensemble members
NCL, LD, EP = 20, 8, 20       # per-member VaDE (small/fast)
M = {"WADI": 24, "HAI": 20, "SWaT": 16}


def subset_feats(Xw, nch, S):
    cols = [b * nch + c for b in range(6) for c in S]
    return np.ascontiguousarray(Xw[:, cols])


ALL = {}
for name in (sys.argv[1:] or ["WADI", "HAI", "SWaT"]):
    D = E.load(name); nch = len(D["ch"])
    Xn = np.asarray(D["Xn_w"], float); Xa = np.asarray(D["Xa_w"], float)
    y = np.asarray(D["ya_w"], int)
    d = np.load(f"{OUT}/scores_{name}.npz"); thr = float(d["maxz_thr"])
    hard = (y == 1) & ~((y == 1) & (d["maxz"] > thr)); keep = np.where((y == 0) | hard)[0]
    latad = d["LatAD"].mean(0); IFm = d["IF"].mean(0)
    nfit = len(Xn) * 4 // 5
    cols = []
    for j in range(K):
        S = np.sort(RNG.choice(nch, size=M[name], replace=False))
        Ftr = subset_feats(Xn, nch, S); Fte = subset_feats(Xa, nch, S)
        mu, sg = Ftr[:nfit].mean(0), Ftr[:nfit].std(0) + 1e-9
        Ztr = ((Ftr - mu) / sg).astype(np.float32); Zte = ((Fte - mu) / sg).astype(np.float32)
        kd = min(60, max(15, nfit // 10))
        v = train_vade(Ztr[:nfit], n_clusters=NCL, latent_dim=LD, epochs=EP, warmup=5, seed=0, device="cpu")
        v.fit_latent_density(Ztr[:nfit], k_density=kd)
        calib = np.asarray(v.anomaly_score_hard(Ztr[nfit:], use_resid=False, use_basin=False))
        sc = np.asarray(v.anomaly_score_hard(Zte, use_resid=False, use_basin=False))
        col = (sc - calib.mean()) / (calib.std() + 1e-9)
        if not np.all(np.isfinite(col)):
            col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)  # a degenerate member contributes nothing
            print(f"  {name} member {j+1}/{K} had non-finite scores -> zeroed", flush=True)
        cols.append(col)
        print(f"  {name} member {j+1}/{K} done", flush=True)
    Zc = np.stack(cols)
    ens_max = Zc.max(0); ens_top3 = np.sort(Zc, axis=0)[-3:].mean(0)
    np.savez(f"{OUT}/ens_scores_{name}.npz", ens_max=ens_max, ens_top3=ens_top3, label=y, hard=hard)
    def au(s): return round(float(roc_auc_score(y[keep], s[keep])), 3)
    nm = y == 0
    resc, nf, nfn = CL.rescue_wrap(latad, ens_max, y)
    ALL[name] = dict(LatAD=au(latad), IF=au(IFm), ens_max=au(ens_max), ens_top3=au(ens_top3),
                     rescued=au(resc), fires=nf, fires_normal=nfn)
    print(f"\n=== {name} (difficult n={int(hard.sum())}) ===")
    print(f"   LatAD {ALL[name]['LatAD']}  IF {ALL[name]['IF']}  ens_max {ALL[name]['ens_max']}  "
          f"ens_top3 {ALL[name]['ens_top3']}  rescued {ALL[name]['rescued']} (fires {nf}, {nfn} normal)")
    json.dump(ALL, open(f"{OUT}/ensemble_vade.json", "w"), indent=1)
print(f"\nsaved -> {OUT}/ensemble_vade.json")
