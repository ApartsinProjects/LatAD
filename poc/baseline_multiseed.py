"""Multi-seed difficult-AUROC for the local baselines (Isolation Forest, AutoEncoder),
so the comparison against our multi-seed model has CIs on BOTH sides. Same data/split as
report_table. Local CPU, free.  python baseline_multiseed.py WADI
"""
from __future__ import annotations
import os, sys, json, numpy as np, torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from compare_baselines import ae_scores
import eda_real as E

SEEDS = [0, 1, 2, 3, 4]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics"); os.makedirs(OUT, exist_ok=True)


def au(y, s, mask):
    k = (y == 0) | mask
    return float(roc_auc_score(y[k], s[k]))


def run(name):
    D = E.load(name)
    Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"]
    C6 = Xte0.shape[1] // 6
    triv = np.abs(Xte0[:, :C6]).max(1); trn = np.abs(Xtr0[:, :C6]).max(1)
    hard = (y == 1) & ~(triv > np.quantile(trn, 0.99))
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)

    res = {"IsolationForest": [], "AutoEncoder": []}
    for seed in SEEDS:
        sIF = -IsolationForest(n_estimators=200, random_state=seed).fit(Xtr).decision_function(Xte)
        res["IsolationForest"].append(au(y, sIF, hard))
        np.random.seed(seed); torch.manual_seed(seed)
        sAE = ae_scores(Xtr, Xte, device="cpu")
        res["AutoEncoder"].append(au(y, sAE, hard))
        print(f"[{name} seed{seed}] IF={res['IsolationForest'][-1]:.3f} AE={res['AutoEncoder'][-1]:.3f}", flush=True)

    summary = {"dataset": name, "seeds": SEEDS,
               "difficult_auroc": {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": [round(x, 3) for x in v]}
                                   for m, v in res.items()}}
    json.dump(summary, open(os.path.join(OUT, f"baseline_multiseed_{name}.json"), "w"), indent=1)
    print(f"=== {name}: IF {np.mean(res['IsolationForest']):.3f}+-{np.std(res['IsolationForest']):.3f} "
          f"AE {np.mean(res['AutoEncoder']):.3f}+-{np.std(res['AutoEncoder']):.3f}", flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI", "HAI", "SKAB"]):
        run(nm)
