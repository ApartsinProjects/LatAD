"""Common-mode-block-level head across all 3 datasets + clean fusions with LatAD.
Groups auto-discovered from train-normal correlation (leak-free). Head = calibrated max over
groups of the two-sided common-mode tail. Reports difficult-subset AUROC of: head alone, LatAD,
and fusions max(z_latad,z_head) and (z_latad+z_head). Goal: does the head lift WADI without
hurting HAI/SWaT (i.e. is it a safe auto-gated addition)?
"""
from __future__ import annotations
import json, os, numpy as np
from numpy.linalg import svd
from sklearn.metrics import roc_auc_score
import eda_real as E

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
THR, MINSZ = 0.7, 3


def groups_head(Zn, Za):
    nA = len(Zn) // 2; A, B = Zn[:nA], Zn[nA:]
    C = np.corrcoef(A.T); np.fill_diagonal(C, 0.0); adj = np.abs(C) > THR
    seen = np.zeros(Zn.shape[1], bool); groups = []
    for i in range(Zn.shape[1]):
        if seen[i]:
            continue
        st = [i]; comp = []
        while st:
            v = st.pop()
            if seen[v]:
                continue
            seen[v] = True; comp.append(v); st += list(np.where(adj[v])[0])
        if len(comp) >= MINSZ:
            groups.append(sorted(comp))
    if not groups:
        return None, 0
    def fs(Z, G):
        U, S, Vt = svd(A[:, G] - A[:, G].mean(0), full_matrices=False)
        u = Vt[0] * np.sign(Vt[0].sum()); g = Z[:, G] @ u
        ga = A[:, G] @ u; return np.abs((g - ga.mean()) / (ga.std() + 1e-9))
    hB = np.max([fs(B, G) for G in groups], axis=0); mB, sB = hB.mean(), hB.std() + 1e-9
    return (np.max([fs(Za, G) for G in groups], axis=0) - mB) / sB, len(groups)


ALL = {}
for name in ["WADI", "HAI", "SWaT"]:
    D = E.load(name); ch = len([str(c) for c in D["ch"]])
    Mn = np.asarray(D["Xn_w"], float)[:, :ch]; Ma = np.asarray(D["Xa_w"], float)[:, :ch]
    y = np.asarray(D["ya_w"], int)
    d = np.load(f"{OUT}/scores_{name}.npz"); thr = float(d["maxz_thr"])
    hard = (y == 1) & ~((y == 1) & (d["maxz"] > thr))
    latad = d["LatAD"].mean(0)
    mu, sg = Mn.mean(0), Mn.std(0) + 1e-9
    head_a, ng = groups_head((Mn - mu) / sg, (Ma - mu) / sg)
    keep = (y == 0) | hard
    A = lambda s: round(float(roc_auc_score(y[keep], s[keep])), 3)
    zl = (latad - latad.mean()) / latad.std()
    zh = (head_a - head_a.mean()) / head_a.std()
    ALL[name] = dict(n_groups=ng, head=A(head_a), LatAD=A(latad),
                     fuse_max=A(np.maximum(zl, zh)), fuse_sum=A(zl + zh))
    print(f"{name:5} (groups {ng}):  head {ALL[name]['head']}  LatAD {ALL[name]['LatAD']}  "
          f"max(zL,zH) {ALL[name]['fuse_max']}  (zL+zH) {ALL[name]['fuse_sum']}")
json.dump(ALL, open(f"{OUT}/mine_diag3.json", "w"), indent=1)
print(f"saved -> {OUT}/mine_diag3.json")
