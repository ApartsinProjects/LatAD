"""HAI detection-set complementarity on the FULL-HEAD regime-community model (HCcoh+LatAD),
re-verifying the earlier global-model result (49 anomalies caught by LatAD alone). Equal alarm
budget; canonical difficult subset. Reports the Venn vs linres/IF/AE/TranAD.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
os.environ["EXPERTS_DIR"] = "sota_bundle/experts_full"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ensemble_final import ensemble_scores
HERE = os.path.dirname(os.path.abspath(__file__))

ens, y, d, nseed = ensemble_scores("HAI")
triv = d["maxz"]; thr = float(d["maxz_thr"])
hard = (y == 1) & ~((y == 1) & (triv > thr))
idxA = np.where(hard)[0]
nN = int((y == 0).sum()); nA = int(hard.sum())
budget = int(0.05 * nN) + nA

def sc(a): return a.mean(0) if a.ndim > 1 else a
methods = {"HCcoh+LatAD(full)": sc(ens["HCcoh+LatAD"]), "linres": sc(d["linres"]),
           "IF": sc(d["IF"]), "AE": sc(d["AE"]), "TranAD": sc(d["TranAD"])}
det = {k: set(i for i in idxA.tolist() if i in set(np.argsort(-s)[:budget].tolist())) for k, s in methods.items()}
L = det["HCcoh+LatAD(full)"]; others = det["linres"] | det["IF"] | det["AE"] | det["TranAD"]
out = {"n_difficult": nA, "budget": budget,
       "caught": {k: len(v) for k, v in det.items()},
       "only_LatAD": len(L - others),
       "LatAD_not_TranAD": len(L - det["TranAD"]), "TranAD_not_LatAD": len(det["TranAD"] - L),
       "LatAD_not_linres": len(L - det["linres"]), "linres_not_LatAD": len(det["linres"] - L),
       "jaccard_LatAD_TranAD": round(len(L & det["TranAD"]) / max(1, len(L | det["TranAD"])), 3),
       "jaccard_LatAD_linres": round(len(L & det["linres"]) / max(1, len(L | det["linres"])), 3)}
json.dump(out, open(os.path.join(HERE, "complementarity_fullhead.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
