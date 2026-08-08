"""Falsifiable diagnostic for the 'coherent block translation' hypothesis (ChatGPT extension #1).

For the correlated analyzer group G, on train-normal-standardized per-channel window means:
  common factor  g_G = u_G^T x_G           (u_G = PC1 of G on train-normal)
  z_common       = (g_G - E_tr g_G)/sd_tr g_G           expect: large NEGATIVE for the WADI misses
  z_external     = leave-GROUP-out residual: g_G - ridge(g_G | x_notG), /sd_tr        expect: large |.|
  z_contrast     = ||(I-uu^T)(x_G-mu_G)|| standardized                              expect: ORDINARY
And the actual extension score S_block=|z_external|: does it SEPARATE WADI difficult from normal
(AUROC), and is it complementary to LatAD? Group G is ALSO checked to emerge from a train-normal
correlation community (not hand-picked) to preempt a test-engineering objection.
"""
from __future__ import annotations
import json, os, numpy as np
from numpy.linalg import svd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
import eda_real as E

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
NAME = "WADI"
SEED_GROUP = ["2B_AIT_004_PV", "2A_AIT_004_PV", "1_AIT_004_PV", "2B_AIT_003_PV", "1_AIT_003_PV"]

D = E.load(NAME); ch = [str(c) for c in D["ch"]]
Mn = np.asarray(D["Xn_w"], float)[:, :len(ch)]     # train-normal per-channel window means
Ma = np.asarray(D["Xa_w"], float)[:, :len(ch)]     # test per-channel window means
y = np.asarray(D["ya_w"], int)
d = np.load(f"{OUT}/scores_{NAME}.npz"); thr = float(d["maxz_thr"])
hard = (y == 1) & ~((y == 1) & (d["maxz"] > thr))
latad = d["LatAD"].mean(0); IFs = d["IF"].mean(0)

mu, sg = Mn.mean(0), Mn.std(0) + 1e-9
Zn = (Mn - mu) / sg; Za = (Ma - mu) / sg
name2i = {c: i for i, c in enumerate(ch)}
G = [name2i[c] for c in SEED_GROUP if c in name2i]
notG = [i for i in range(len(ch)) if i not in G]

# --- does G emerge from a train-normal correlation community around 2B_AIT_004_PV? ---
C = np.corrcoef(Zn.T)
anchor = name2i["2B_AIT_004_PV"]
nbrs = np.argsort(-C[anchor])[:8]
community = [(ch[i], round(float(C[anchor, i]), 2)) for i in nbrs]

# --- common factor u_G = PC1 of group on train-normal ---
U, S, Vt = svd(Zn[:, G] - Zn[:, G].mean(0), full_matrices=False)
u = Vt[0];  u = u * np.sign(u.sum())               # orient positive
gt_n = Zn[:, G] @ u; gt_a = Za[:, G] @ u
gmu, gsd = gt_n.mean(), gt_n.std() + 1e-9

# --- z_external: ridge predict g from channels OUTSIDE G (leave-group-out) ---
r = Ridge(alpha=10.0).fit(Zn[:, notG], gt_n)
res_n = gt_n - r.predict(Zn[:, notG]); res_a = gt_a - r.predict(Za[:, notG])
rsd = res_n.std() + 1e-9

# --- z_contrast: within-group shape after removing the common factor ---
def contrast(Z):
    P = Z[:, G] - np.outer(Z[:, G] @ u, u)          # remove common-mode
    return np.sqrt((P ** 2).sum(1))
cn = contrast(Zn); ca = contrast(Za); cmu, csd = cn.mean(), cn.std() + 1e-9

z_common = (gt_a - gmu) / gsd
z_external = res_a / rsd
z_contrast = (ca - cmu) / csd

def auroc(score):
    keep = (y == 0) | hard; return round(float(roc_auc_score(y[keep], score[keep])), 3)

S_block = np.abs(z_external)
# worst WADI difficult windows (lowest LatAD percentile)
hidx = np.where(hard)[0]
lp = np.array([(latad <= latad[i]).mean() for i in hidx])
worst = hidx[np.argsort(lp)][:8]

rep = {
    "group": SEED_GROUP,
    "train_normal_community_around_2B_AIT_004_PV": community,
    "pattern_on_worst_windows": [
        dict(window=int(w), z_common=round(float(z_common[w]), 2),
             z_external=round(float(z_external[w]), 2), z_contrast=round(float(z_contrast[w]), 2),
             latad_pct=round(float((latad <= latad[w]).mean()), 3))
        for w in worst],
    "difficult_medians": dict(
        z_common=round(float(np.median(z_common[hard])), 2),
        z_external=round(float(np.median(np.abs(z_external[hard]))), 2),
        z_contrast=round(float(np.median(z_contrast[hard])), 2)),
    "normal_medians": dict(
        z_common=round(float(np.median(z_common[y == 0])), 2),
        abs_z_external=round(float(np.median(np.abs(z_external[y == 0]))), 2),
        z_contrast=round(float(np.median(z_contrast[y == 0])), 2)),
    "difficult_AUROC": dict(
        S_block_leave_group_out=auroc(S_block),
        z_common_only=auroc(-z_common),          # low common level = anomalous -> use -z_common
        LatAD=auroc(latad), IF=auroc(IFs),
        LatAD_plus_Sblock=auroc((latad - latad.mean()) / latad.std() + (S_block - S_block.mean()) / S_block.std())),
}
json.dump(rep, open(f"{OUT}/mine_diag_WADI.json", "w"), indent=1)
print("Train-normal community around 2B_AIT_004_PV:")
for c, v in community: print(f"   {c}: r={v}")
print(f"\nPattern on 8 worst WADI difficult windows (hypothesis: z_common<<0, |z_external| large, z_contrast ordinary):")
for r_ in rep["pattern_on_worst_windows"]:
    print(f"   win{r_['window']}: z_common={r_['z_common']}  z_external={r_['z_external']}  z_contrast={r_['z_contrast']}  (latad_pct={r_['latad_pct']})")
print(f"\nmedians  difficult: {rep['difficult_medians']}")
print(f"medians  normal:    {rep['normal_medians']}")
print(f"\nDIFFICULT-subset AUROC:")
for k, v in rep["difficult_AUROC"].items(): print(f"   {k}: {v}")
print(f"\nsaved -> {OUT}/mine_diag_WADI.json")
