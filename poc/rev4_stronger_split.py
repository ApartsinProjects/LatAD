"""Rev4 Tier-2 item 6: stronger univariate difficulty split.

The canonical split flags an anomalous window 'easy' iff max|z| over the FIRST stat-block
(per-channel window MEAN) exceeds the 99th train-normal percentile. A window can thus be
'difficult' yet be trivially caught by another univariate statistic the model already
ingests (std, min, max, first-last diff, range). This script redefines 'easy' as
detectable by the max over ALL SIX standardized per-channel statistics, shrinking the
difficult subset to windows no simple univariate statistic separates -- the strong test
of the joint-structure claim (and the direct answer to Pinet 2026).

No re-inference: reuses every method's per-window score from _diagnostics/scores_<DS>.npz;
only the difficulty MASK changes. Difficult-AUROC protocol matches build_comparison_table.
Output -> _diagnostics/rev4_stronger_split.json.
"""
from __future__ import annotations
import json, os, numpy as np
from sklearn.metrics import roc_auc_score
import eda_real as E

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
ROWS = [("trivial max|z|", "maxz"), ("Isolation Forest", "IF"), ("AutoEncoder", "AE"),
        ("LinRes (one-hot)", "linres"), ("USAD", "USAD"), ("TranAD", "TranAD"),
        ("VaDE-hard+resid (ours)", "LatAD")]


def auroc_multi(y, arr, keep):
    if arr.ndim == 2:
        return (round(float(np.nanmean([roc_auc_score(y[keep], arr[i][keep]) for i in range(arr.shape[0])])), 3),
                round(float(np.nanstd([roc_auc_score(y[keep], arr[i][keep]) for i in range(arr.shape[0])])), 3))
    return (round(float(roc_auc_score(y[keep], arr[keep])), 3), None)


ALL = {}
for name in ["WADI", "HAI", "SWaT"]:
    D = E.load(name)
    Xtr0 = np.asarray(D["Xn_w"], float); Xte0 = np.asarray(D["Xa_w"], float)
    y = np.asarray(D["ya_w"], int)
    d = np.load(f"{OUT}/scores_{name}.npz")
    assert len(y) == len(d["label"]) and (y == d["label"]).all(), f"{name}: label mismatch vs npz"

    # canonical (first stat-block only, unstandardized abs) -- reproduce for reference
    C6 = Xte0.shape[1] // 6
    maxz1 = np.abs(Xte0[:, :C6]).max(1)
    thr1 = float(np.quantile(np.abs(Xtr0[:, :C6]).max(1), 0.99))
    easy1 = (y == 1) & (maxz1 > thr1); hard1 = (y == 1) & ~easy1

    # STRONGER: standardize every feature column on train-normal, max|z| over ALL 6 blocks
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    ztr = np.abs((Xtr0 - mu) / sig).max(1)
    zte = np.abs((Xte0 - mu) / sig).max(1)
    thrA = float(np.quantile(ztr, 0.99))
    easyA = (y == 1) & (zte > thrA); hardA = (y == 1) & ~easyA

    keepA = np.where((y == 0) | hardA)[0]
    res = {"n_anom": int((y == 1).sum()),
           "canonical_difficult": int(hard1.sum()),
           "stronger_difficult": int(hardA.sum()),
           "removed_by_stronger": int((hard1 & ~hardA).sum()),
           "rows": {}}
    for label, key in ROWS:
        if key not in d.files:
            continue
        if hardA.sum() >= 3:
            au, sd = auroc_multi(y, d[key], keepA)
        else:
            au, sd = float("nan"), None
        res["rows"][label] = {"auroc": au, **({"auroc_sd": sd} if sd is not None else {})}
    ALL[name] = res
    print(f"\n=== {name} ===  anom={res['n_anom']}  "
          f"canonical-difficult={res['canonical_difficult']} -> "
          f"stronger-difficult={res['stronger_difficult']} "
          f"(removed {res['removed_by_stronger']})")
    for label, key in ROWS:
        if label in res["rows"]:
            c = res["rows"][label]
            s = f"±{c['auroc_sd']:.3f}" if "auroc_sd" in c else ""
            print(f"    {label:24} difficult-AUROC {c['auroc']}{s}")

json.dump(ALL, open(f"{OUT}/rev4_stronger_split.json", "w"), indent=1)
print(f"\nsaved -> {OUT}/rev4_stronger_split.json")
