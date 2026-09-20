"""Non-win C: would a TRAIN-ONLY generalization gate admit the PLAIN whitened reconstruction residual
(Table 6 row 1, the strongest single head on WADI/SWaT) where it helps, and reject it where it hurts?

The reported model gates a different head (responsibility-weighted, PCA-30, per-mode LedoitWolf
residual; fit_resid_head) with ratio = q95(held-out 20% of train) / q95(fit 80%) < 1.5. Here the
identical ratio is computed for the plain global whitened residual 0.5 r^T Sigma^-1 r, on a model
fit on the first 80% of train windows, with the last 20% held out. Reported per dataset/seed:
  ratio_plain, ratio_permode (the existing gate), and test difficult-AUROC of
  base, plain recon alone, base+plain (z on fit), base+permode(auto) -- on the SAME 80% model.
Selection must use only the ratios; the test AUROCs are for reporting.
Invariant: on HAI the plain residual is the weakest head (Table 6: 0.695 < 0.802), so if the gate
admits it there and it hurts, the gate is not a safe selector. Appends per (dataset, seed) to
fable_recon_gate.jsonl (resumable).
"""
from __future__ import annotations
import json, os, sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import roc_auc_score
import eda_real as E
from models_vade import train_vade, _as_tensor, _recon_energy

HERE = os.path.dirname(os.path.abspath(__file__))
OUTF = os.path.join(HERE, "fable_recon_gate.jsonl")
CFG = {"WADI_clean": (20, 10), "SWaT_canon": (40, 16), "HAI": (40, 16)}
done = set()
if os.path.exists(OUTF):
    for l in open(OUTF): r = json.loads(l); done.add((r["dataset"], r["seed"]))

for name in (sys.argv[1:] or ["WADI_clean", "SWaT_canon", "HAI"]):
    D = E.load(name); Xtr0 = D["Xn_w"].astype(np.float32); Xte0 = D["Xa_w"].astype(np.float32); y = D["ya_w"].astype(int)
    d = np.load(f"{HERE}/scores_{name}.npz"); thr = float(d["maxz_thr"]); diff = (y == 1) & (d["maxz"] <= thr)
    mu0, sg0 = Xtr0.mean(0), Xtr0.std(0) + 1e-8; Z = (Xte0 - mu0) / sg0
    const = np.where(Xtr0.std(0) < 1e-6)[0]; leak = (np.abs(Z[:, const]).max(1) > 100) if len(const) else np.zeros(len(y), bool)
    nfit = int(0.8 * len(Xtr0)); Xf0, Xh0 = Xtr0[:nfit], Xtr0[nfit:]
    mu, sig = Xf0.mean(0), Xf0.std(0) + 1e-8
    st = lambda A: ((A - mu) / sig).astype(np.float32)
    Xf, Xh, Xte = st(Xf0), st(Xh0), st(Xte0)
    K, LD = CFG[name]; kd = min(80, max(20, nfit // 10))
    for sd in (0, 1):
        if (name, sd) in done: continue
        t0 = time.time()
        v = train_vade(Xf, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
        v.fit_residual_whitener(Xf); v.fit_latent_density(Xf, k_density=kd); v.fit_resid_head(Xf); v.fit_basin_head(Xf)
        rec = lambda A: np.asarray(_recon_energy(_as_tensor(A, v), v.decode(v.encode(_as_tensor(A, v))[0]), v.res_whitener))
        rf, rh, rt = rec(Xf), rec(Xh), rec(Xte)
        ratio_plain = float(np.quantile(rh, 0.95) / (np.quantile(rf, 0.95) + 1e-9))
        base = np.asarray(v.anomaly_score_hard(Xte, use_near=True))
        base_pm = np.asarray(v.anomaly_score_hard(Xte, use_near=True, use_resid="auto"))
        zrec = (rt - rf.mean()) / (rf.std() + 1e-9)
        def au(s, mk):
            keep = (y == 0) | mk; return round(float(roc_auc_score(y[keep], s[keep])), 3)
        row = dict(dataset=name, seed=sd, ratio_plain=round(ratio_plain, 2), ratio_permode=round(v._resid_gen_ratio, 2),
                   permode_gate_on=bool(v._resid_auto), plain_gate_on_at_1p5=bool(ratio_plain < 1.5),
                   diff43_base=au(base, diff), diff43_plain=au(zrec, diff), diff43_base_plus_plain=au(base + zrec, diff),
                   diff43_base_plus_permode_auto=au(base_pm, diff),
                   diffNoLeak_base=au(base, diff & ~leak), diffNoLeak_plain=au(zrec, diff & ~leak),
                   diffNoLeak_base_plus_plain=au(base + zrec, diff & ~leak), n_diff_noleak=int((diff & ~leak).sum()),
                   secs=round(time.time() - t0))
        with open(OUTF, "a") as f: f.write(json.dumps(row) + "\n")
        print(row, flush=True)
print("DONE", flush=True)
