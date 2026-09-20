"""H5: config sweep (K x latent_dim) on cleaned WADI (clip=10) with a LABEL-FREE selection criterion.
Selection: time-ordered split of TRAIN-normal windows (first 80% fit / last 20% held-out). For each
config and seed: train VaDE + density head on the 80%, score the 20% -> (a) mean held-out density NLL
(lower = better generalising density model), (b) tail ratio q99(held-out)/q99(fit) (closer to 1 = better
calibrated; the same statistic the residual gate uses). Chosen config = lowest held-out NLL averaged
over seeds. Test difficult-AUROC is ALSO computed for every config (full-train model, canonical subset),
purely to report whether the train-selected config would have changed the outcome. Results appended
per (config, seed) to fable_cleanwadi_sweep.jsonl (resumable).
"""
from __future__ import annotations
import os, sys, json, itertools, numpy as np, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.dirname(HERE); sys.path.insert(0, POC); os.chdir(POC)
from sklearn.metrics import roc_auc_score as auc
from models_vade import train_vade
import eda_real as E

OUT = os.path.join(HERE, "fable_cleanwadi_sweep.jsonl")
KS = [int(k) for k in os.environ.get("KS", "12,20,30").split(",")]
LDS = [int(l) for l in os.environ.get("LDS", "8,10,16").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
done = set()
if os.path.exists(OUT):
    for ln in open(OUT):
        r = json.loads(ln); done.add((r["K"], r["LD"], r["seed"]))

D = E.load("WADI_clean", clip=10.0)
Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"].astype(int)
C6 = Xte0.shape[1] // 6
maxz = np.abs(Xte0[:, :C6]).max(1); thr = float(np.quantile(np.abs(Xtr0[:, :C6]).max(1), 0.99))
hard = (y == 1) & (maxz <= thr); KH = (y == 0) | hard
mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
nA = int(0.8 * len(Xtr)); XA, XB = Xtr[:nA], Xtr[nA:]
kd_full = min(80, max(20, len(Xtr) // 10)); kd_A = min(80, max(20, nA // 10))
print(f"n_tr={len(Xtr)} fit={nA} heldout={len(XB)} difficult={hard.sum()} configs={len(KS)*len(LDS)} seeds={SEEDS}", flush=True)

def fit(X, K, LD, sd, kd):
    v = train_vade(X, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
    v.fit_residual_whitener(X); v.fit_latent_density(X, k_density=kd); v.fit_resid_head(X); v.fit_basin_head(X)
    return v

for K, LD, sd in itertools.product(KS, LDS, SEEDS):
    if (K, LD, sd) in done: continue
    # --- label-free selection statistics (80/20 time-ordered split of train) ---
    vA = fit(XA, K, LD, sd, kd_A)
    dA, _ = vA._hard_components(XA); dB, _ = vA._hard_components(XB)
    sA = vA.anomaly_score_hard(XA, use_resid="auto", use_basin="auto"); sB = vA.anomaly_score_hard(XB, use_resid="auto", use_basin="auto")
    sel = dict(heldout_dens_nll=float(dB.mean()), fit_dens_nll=float(dA.mean()),
               heldout_score_q99_ratio=float(np.quantile(sB, .99) / (np.quantile(sA, .99) + 1e-9)),
               heldout_score_mean=float(sB.mean()), resid_gen_ratio=float(vA._resid_gen_ratio))
    # --- test number for the FULL-train model (report-only, not used for selection) ---
    v = fit(Xtr, K, LD, sd, kd_full)
    s = v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto")
    s_res = v.anomaly_score_hard(Xte, use_resid=True, use_basin=False)
    r = dict(K=K, LD=LD, seed=sd, **sel, test_difficult_auroc=float(auc(y[KH], s[KH])), test_all_auroc=float(auc(y, s)),
             test_difficult_auroc_resid_forced=float(auc(y[KH], s_res[KH])), resid_auto_full=bool(v._resid_auto))
    with open(OUT, "a") as f: f.write(json.dumps(r) + "\n")
    print(f"K={K:2d} LD={LD:2d} seed={sd}: heldout NLL {sel['heldout_dens_nll']:.2f} (fit {sel['fit_dens_nll']:.2f}) "
          f"q99ratio {sel['heldout_score_q99_ratio']:.2f} | test difficult {r['test_difficult_auroc']:.3f} all {r['test_all_auroc']:.3f}", flush=True)
print("done", flush=True)
