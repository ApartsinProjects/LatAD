"""Honest, leak-free test of the common-mode-block-level head.

Groups are auto-discovered from TRAIN-NORMAL correlation only (connected components of
|corr|>THR on per-channel window means, size>=3). For each group, the common factor is its
train-normal PC1; a window's group score is the two-sided common-mode tail |z_common|,
z-normalised on train-normal. The head score is the calibrated max over groups (multiple-
testing handled by calibrating the max on held-out train-normal). No test data or LatAD used
to pick groups. Reports WADI difficult-subset AUROC of the head alone, LatAD, IF, and the
rescue-fused LatAD + max(0, head - tau). Also whether the mined analyzer community is among
the auto-discovered groups (to rebut cherry-picking).
"""
from __future__ import annotations
import json, os, numpy as np
from numpy.linalg import svd
from sklearn.metrics import roc_auc_score
import eda_real as E

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
NAME = "WADI"; THR = 0.7; MINSZ = 3

D = E.load(NAME); ch = [str(c) for c in D["ch"]]
Mn = np.asarray(D["Xn_w"], float)[:, :len(ch)]; Ma = np.asarray(D["Xa_w"], float)[:, :len(ch)]
y = np.asarray(D["ya_w"], int)
d = np.load(f"{OUT}/scores_{NAME}.npz"); thr = float(d["maxz_thr"])
hard = (y == 1) & ~((y == 1) & (d["maxz"] > thr))
latad = d["LatAD"].mean(0); IFs = d["IF"].mean(0)

# split train-normal in half: fit groups+factors on A, calibrate the max on held-out B
mu, sg = Mn.mean(0), Mn.std(0) + 1e-9
Zn = (Mn - mu) / sg; Za = (Ma - mu) / sg
nA = len(Zn) // 2; A, B = Zn[:nA], Zn[nA:]

# auto-discover correlation communities on A (connected components of |corr|>THR)
C = np.corrcoef(A.T); np.fill_diagonal(C, 0.0)
adj = np.abs(C) > THR
seen = np.zeros(len(ch), bool); groups = []
for i in range(len(ch)):
    if seen[i]:
        continue
    stack = [i]; comp = []
    while stack:
        v = stack.pop()
        if seen[v]:
            continue
        seen[v] = True; comp.append(v)
        stack += list(np.where(adj[v])[0])
    if len(comp) >= MINSZ:
        groups.append(sorted(comp))

# per group: common factor (PC1 on A), z_common on train-normal (A) scale
def factor_scores(Z, G):
    Xg = Z[:, G]
    U, S, Vt = svd(A[:, G] - A[:, G].mean(0), full_matrices=False)
    u = Vt[0] * np.sign(Vt[0].sum())
    g = Xg @ u
    gmu, gsd = (A[:, G] @ u).mean(), (A[:, G] @ u).std() + 1e-9
    return np.abs((g - gmu) / gsd)               # two-sided common-mode tail

# head score = max over groups; calibrate the max distribution on held-out normal B
head_B = np.max([factor_scores(B, G) for G in groups], axis=0)
mB, sB = head_B.mean(), head_B.std() + 1e-9
def head(Z):
    return (np.max([factor_scores(Z, G) for G in groups], axis=0) - mB) / sB
head_a = head(Za)

def auroc(score):
    keep = (y == 0) | hard; return round(float(roc_auc_score(y[keep], score[keep])), 3)

zl = (latad - latad.mean()) / latad.std()
tau = 2.0                                          # rescue threshold (train-normal sigma units)
fused = zl + np.maximum(0.0, head_a - tau)

# is the mined analyzer community recovered?
name2i = {c: i for i, c in enumerate(ch)}
aiti = name2i.get("2B_AIT_004_PV")
mined_group = next(([ch[j] for j in G] for G in groups if aiti in G), None)

rep = dict(
    n_groups=len(groups), group_sizes=sorted(len(G) for G in groups)[-8:],
    mined_analyzer_group=(mined_group[:12] if mined_group else None),
    mined_group_size=(len(mined_group) if mined_group else 0),
    difficult_AUROC=dict(
        head_common_mode_scan=auroc(head_a),
        LatAD=auroc(latad), IF=auroc(IFs),
        LatAD_plus_rescue=auroc(fused)),
)
json.dump(rep, open(f"{OUT}/mine_diag2_WADI.json", "w"), indent=1)
print(f"auto-discovered {rep['n_groups']} train-normal correlation groups (|r|>{THR}, size>={MINSZ}); "
      f"largest sizes {rep['group_sizes']}")
print(f"analyzer community containing 2B_AIT_004_PV (auto-discovered, size {rep['mined_group_size']}):")
print(f"   {rep['mined_analyzer_group']}")
print(f"\nWADI DIFFICULT-subset AUROC (leak-free groups):")
for k, v in rep["difficult_AUROC"].items():
    print(f"   {k}: {v}")
print(f"\nsaved -> {OUT}/mine_diag2_WADI.json")
