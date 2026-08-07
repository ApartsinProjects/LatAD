"""Referee-response local compute (free, CPU): per dataset x seed{0..4} produce
  (N1) full-model AUROC all/easy/difficult  -> multi-seed mean+-std CIs
  (R1) head ablation difficult-AUROC: density -> +nearest(base) -> +residual -> +basin -> full(auto)
  (R2) DEPLOYABLE operating point: threshold at train-normal 99th pct (1% train-normal FPR),
       then test F1/FPR at that fixed, label-free threshold.
Reuses the exact reported construction (report_table CFG, seed varies).
Run one process per dataset for parallelism:  python improve_multiseed.py WADI
"""
from __future__ import annotations
import os, sys, json, numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score
from models_vade import train_vade
from compare_baselines import ae_scores
import eda_real as E

CFG = {"WADI": (20, 10), "HAI": (40, 16), "SKAB": (16, 6), "SWaT": (40, 16)}
SEEDS = [0, 1, 2, 3, 4]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_diagnostics"); os.makedirs(OUT, exist_ok=True)


def au(y, s, mask):
    k = (y == 0) | mask
    return float(roc_auc_score(y[k], s[k])) if mask.sum() >= 3 else None


def deploy_f1_fpr(y, s_te, s_tr):
    """Threshold at 99th percentile of TRAIN-normal score (target 1% train FPR),
    label-free. Report test F1 (all test) and test FPR at that threshold."""
    thr = float(np.quantile(s_tr, 0.99))
    pred = (s_te >= thr).astype(int)
    f1 = float(f1_score(y, pred)) if pred.sum() > 0 else 0.0
    fpr = float((pred[y == 0]).mean())
    return f1, fpr


def run(name):
    D = E.load(name)
    Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"]
    C6 = Xte0.shape[1] // 6
    triv = np.abs(Xte0[:, :C6]).max(1); trn = np.abs(Xtr0[:, :C6]).max(1)
    easy = (y == 1) & (triv > np.quantile(trn, 0.99)); hard = (y == 1) & ~easy
    K, LD = CFG[name]
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)

    rows = []
    for seed in SEEDS:
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
        kd = min(80, max(20, len(Xtr) // 10))
        v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd)
        v.fit_resid_head(Xtr); v.fit_basin_head(Xtr)

        # --- ablation component scores (test) ---
        dens_te, near_te = v._hard_components(Xte); dm, ds, nm, ns = v._hd_ref
        s_dens = (dens_te - dm) / ds
        s_base = v.anomaly_score_hard(Xte, use_resid=False, use_basin=False)          # density+nearest
        s_res = v.anomaly_score_hard(Xte, use_resid=True,  use_basin=False)           # +residual (forced)
        s_bas = v.anomaly_score_hard(Xte, use_resid=False, use_basin=True)            # +basin (forced)
        s_full = v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto")        # reported model
        s_full_tr = v.anomaly_score_hard(Xtr, use_resid="auto", use_basin="auto")

        rows.append({
            "seed": seed,
            "full_auroc": {"all": au(y, s_full, y == 1), "easy": au(y, s_full, easy), "hard": au(y, s_full, hard)},
            "ablation_hard_auroc": {
                "density":        au(y, s_dens, hard),
                "base(dens+near)": au(y, s_base, hard),
                "base+residual":  au(y, s_res,  hard),
                "base+basin":     au(y, s_bas,  hard),
                "full(auto)":     au(y, s_full, hard),
            },
            "deploy_op": dict(zip(["f1", "fpr"], deploy_f1_fpr(y, s_full, s_full_tr))),
            "gates": {"resid_auto_on": bool(getattr(v, "_resid_auto", False)),
                      "basin_lambda": float(getattr(v, "_basin_lam", 0.0))},
        })
        print(f"[{name} seed{seed}] full hard AUROC={rows[-1]['full_auroc']['hard']:.3f} "
              f"deploy F1={rows[-1]['deploy_op']['f1']:.3f} FPR={rows[-1]['deploy_op']['fpr']:.3f}", flush=True)

    # aggregate mean/std
    def agg(getter):
        vals = [getter(r) for r in rows if getter(r) is not None]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)} if vals else None
    summary = {
        "dataset": name, "seeds": SEEDS,
        "full_auroc": {k: agg(lambda r, k=k: r["full_auroc"][k]) for k in ("all", "easy", "hard")},
        "ablation_hard_auroc": {k: agg(lambda r, k=k: r["ablation_hard_auroc"][k])
                                for k in rows[0]["ablation_hard_auroc"]},
        "deploy_op": {k: agg(lambda r, k=k: r["deploy_op"][k]) for k in ("f1", "fpr")},
        "per_seed": rows,
    }
    p = os.path.join(OUT, f"multiseed_{name}.json")
    json.dump(summary, open(p, "w"), indent=1)
    h = summary["full_auroc"]["hard"]
    print(f"=== {name} DONE: difficult AUROC {h['mean']:.3f} +- {h['std']:.3f} (n={h['n']}) -> {p}", flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI", "HAI", "SKAB"]):
        run(nm)
