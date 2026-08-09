"""SKAB falsification test for the new method (guided community-density ensemble + aggregators).
SKAB is low-dimensional (8 channels), so the theory predicts NO gain from the subspace/
factorization (there is no high-dimensional dilution to undo). Registry/diagnostic only; NOT
added to the paper. Self-contained + local (SKAB is tiny).
"""
from __future__ import annotations
import json, os, numpy as np, warnings
warnings.filterwarnings("ignore")
from numpy.linalg import svd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
from compare_baselines import ae_scores
import eda_real as E

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
SEEDS = [0, 1, 2, 3, 4]; K_LAT, LD_LAT = 10, 6


def surv(ref, v):
    o = np.sort(ref); r = np.searchsorted(o, v, side="right") / len(o); return -np.log(np.clip(1 - r, 1e-4, 1.0))
def pv(ref, v):
    o = np.sort(ref); r = np.searchsorted(o, v, side="left") / len(o); return np.clip(1 - r, 1e-4, 1.0)
def HC(P, wt=None):
    S, n = P.shape
    if wt is not None: P = np.clip(P ** (wt[:, None] / (wt.mean() + 1e-9)), 1e-4, 1.0)
    Ps = np.sort(P, 0); i = (np.arange(1, S + 1) / S)[:, None]
    hc = np.sqrt(S) * (i - Ps) / np.sqrt(np.clip(Ps * (1 - Ps), 1e-6, None)); hc[Ps >= 0.5] = -np.inf
    return np.nan_to_num(hc.max(0), neginf=0.0, posinf=1e6)


D = E.load("SKAB"); fn, W, stride = E.RAW["SKAB"]
Xn_raw, Xa_raw = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
Xn_w, Xa_w = np.asarray(D["Xn_w"], float), np.asarray(D["Xa_w"], float)
y = np.asarray(D["ya_w"], int); nch = len(D["ch"])
C6 = Xa_w.shape[1] // 6
maxz = np.abs(Xa_w[:, :C6]).max(1); thr = float(np.quantile(np.abs(Xn_w[:, :C6]).max(1), 0.99))
hard = (y == 1) & ~((y == 1) & (maxz > thr)); keep = np.where((y == 0) | hard)[0]
print(f"SKAB: {nch} ch, {len(y)} test win, anom {int(y.sum())}, difficult {int(hard.sum())} (easy {int(((y==1)&(maxz>thr)).sum())})")

mu, sg = Xn_w.mean(0), Xn_w.std(0) + 1e-9
Ztr, Zte = ((Xn_w - mu) / sg).astype(np.float32), ((Xa_w - mu) / sg).astype(np.float32)
# LatAD 5-seed + IF + AE
lat, IFs, AEs = [], [], []
for sd in SEEDS:
    v = train_vade(Ztr, n_clusters=K_LAT, latent_dim=LD_LAT, epochs=40, warmup=8, seed=sd, device="cpu")
    v.fit_residual_whitener(Ztr); v.fit_latent_density(Ztr, k_density=min(40, max(10, len(Ztr) // 10)))
    v.fit_resid_head(Ztr); v.fit_basin_head(Ztr)
    lat.append(np.asarray(v.anomaly_score_hard(Zte, use_resid="auto", use_basin="auto")))
    IFs.append(-IsolationForest(n_estimators=200, random_state=sd).fit(Ztr).score_samples(Zte))
    AEs.append(ae_scores(Ztr, Zte, seed=sd))
lat, IFs, AEs = np.stack(lat), np.stack(IFs), np.stack(AEs)
# linres (leave-one-channel-out linear on window means)
Mn, Ma = Xn_w[:, :nch], Xa_w[:, :nch]
mm, ms = Mn.mean(0), Mn.std(0) + 1e-9; Mnz, Maz = (Mn - mm) / ms, (Ma - mm) / ms
lin = np.zeros(len(Ma))
for c in range(nch):
    cols = [j for j in range(nch) if j != c]
    lr = LinearRegression().fit(Mnz[:, cols], Mnz[:, c]); lin += (lr.predict(Maz[:, cols]) - Maz[:, c]) ** 2
lin /= nch

def au(s): return float(roc_auc_score(y[keep], s[keep]))
base = {"LatAD": round(float(np.mean([au(lat[i]) for i in range(5)])), 3),
        "IF": round(float(np.mean([au(IFs[i]) for i in range(5)])), 3),
        "AE": round(float(np.mean([au(AEs[i]) for i in range(5)])), 3),
        "linres": round(au(lin), 3), "trivial_maxz": round(au(maxz), 3)}
print("baselines difficult-AUROC:", base)

# --- community experts (HAC on 8 channels) ---
Za = Mnz; A = Za[:len(Za) * 4 // 5]  # not used; fit communities on full train-normal
Cabs = np.abs(np.nan_to_num(np.corrcoef(Mnz.T))); np.fill_diagonal(Cabs, 0.0)
dist = 1 - Cabs; dist = (dist + dist.T) / 2
L = linkage(squareform(dist, checks=False), method="average"); _, nodes = to_tree(L, rd=True)
def leaves(n): return [n.id] if n.is_leaf() else leaves(n.left) + leaves(n.right)
seen, comms, coh = set(), [], []
for n in nodes:
    if not n.is_leaf():
        lv = sorted(leaves(n))
        if 3 <= len(lv) <= nch and tuple(lv) not in seen:
            seen.add(tuple(lv)); comms.append(lv)
            sub = Cabs[np.ix_(lv, lv)]; coh.append(float(sub.sum() / (len(lv) * (len(lv) - 1))))
print(f"HAC communities (size 3-{nch}): {len(comms)}  (low-dim -> few/trivial groups expected)")

def sf(Xw, S): return np.nan_to_num(np.ascontiguousarray(Xw[:, [b * nch + c for b in range(6) for c in S]]))
nfit = len(Xn_w) * 4 // 5
w = np.array(coh) * np.sqrt([len(g) for g in comms]) if comms else np.array([])
agg = {"commOR": [], "factor_sum": [], "HC": [], "HC_coh": [], "null+HC_coh": []}
for sd in SEEDS:
    if comms:
        Zc, Zt = [], []
        for G in comms:
            Ff, Fc, Ft = sf(Xn_w[:nfit], G), sf(Xn_w[nfit:], G), sf(Xa_w, G)
            m2, s2 = Ff.mean(0), Ff.std(0) + 1e-9
            Ztr2 = np.nan_to_num((Ff - m2) / s2).astype(np.float32)
            v = train_vade(Ztr2, n_clusters=min(15, max(6, len(G))), latent_dim=min(6, max(3, len(G) // 2)),
                           epochs=15, warmup=4, seed=sd, device="cpu")
            v.fit_latent_density(Ztr2, k_density=min(30, max(8, nfit // 12)))
            cc = np.asarray(v.anomaly_score_hard(np.nan_to_num((Fc - m2) / s2).astype(np.float32), use_resid=False, use_basin=False))
            tt = np.asarray(v.anomaly_score_hard(np.nan_to_num((Ft - m2) / s2).astype(np.float32), use_resid=False, use_basin=False))
            cm, cs = cc.mean(), cc.std() + 1e-9
            Zc.append(np.nan_to_num((cc - cm) / cs)); Zt.append(np.nan_to_num((tt - cm) / cs))
        Cal, Tst = np.stack(Zc), np.stack(Zt)
        tails = np.stack([surv(Cal[g], Tst[g]) for g in range(len(comms))])
        P = np.stack([pv(Cal[g], Tst[g]) for g in range(len(comms))])
        wn = w / (w.max() + 1e-9)
        commOR = (wn[:, None] * tails).max(0); fsum = tails.sum(0); hc = HC(P); hc_coh = HC(P, wt=w)
    else:
        commOR = fsum = hc = hc_coh = np.zeros(len(Xa_w))
    nm = y == 0; nulltail = surv(lat[sd][nm], lat[sd])
    zz = lambda s: (s - s[nm].mean()) / (s[nm].std() + 1e-9)
    nhc = np.maximum(zz(hc_coh), zz(nulltail))
    for k, s in [("commOR", commOR), ("factor_sum", fsum), ("HC", hc), ("HC_coh", hc_coh), ("null+HC_coh", nhc)]:
        agg[k].append(round(au(s), 3))
res = {"baselines": base, "n_communities": len(comms),
       "aggregators": {k: [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)] for k, v in agg.items()}}
print("\nNEW-METHOD aggregators difficult-AUROC:")
for k, v in res["aggregators"].items(): print(f"   {k:14} {v}")
best_new = max(res["aggregators"].items(), key=lambda kv: kv[1][0])
print(f"\nbest new-method: {best_new[0]} {best_new[1]}  vs LatAD {base['LatAD']} / best-baseline {max(base.values())}")
print(f"VERDICT: {'gain' if best_new[1][0] > max(base.values())+0.01 else 'NO gain (theory-consistent: low-dim, no dilution to undo)'}")
json.dump(res, open(f"{OUT}/skab_newmethod.json", "w"), indent=1)
print(f"saved -> {OUT}/skab_newmethod.json")
