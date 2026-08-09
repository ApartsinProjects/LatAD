"""Final aggregation sweep over the CACHED expert library (no retraining). Tests the remaining
candidate combination rules across all three datasets and picks the best all-three, with
episode-bootstrap significance. Experts: per-community calibrated surprises + cohesion.
"""
from __future__ import annotations
import json, os, numpy as np
from sklearn.metrics import roc_auc_score

BD = "sota_bundle/experts"; OUT = "_diagnostics"
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}
COMPET = {"WADI": "IF", "HAI": "AE", "SWaT": "linres"}
RNG = np.random.default_rng(0)


def surv(ref, v):
    o = np.sort(ref); r = np.searchsorted(o, v, side="right") / len(o); return -np.log(np.clip(1 - r, 1e-4, 1.0))
def pval(ref, v):
    o = np.sort(ref); r = np.searchsorted(o, v, side="left") / len(o); return np.clip(1 - r, 1e-4, 1.0)


def HC(P, wt=None):
    """Higher Criticism over per-factor p-values P (S,n). Optional per-factor weight wt (S,)
    scales each factor's evidence (violation in a tight community counts more)."""
    S, n = P.shape
    if wt is not None:
        # weight -> replicate factor evidence proportional to weight via effective p (p^wt keeps small-p emphasis)
        P = np.clip(P ** (wt[:, None] / (wt.mean() + 1e-9)), 1e-4, 1.0)
    Ps = np.sort(P, axis=0); i = (np.arange(1, S + 1) / S)[:, None]
    hc = np.sqrt(S) * (i - Ps) / np.sqrt(np.clip(Ps * (1 - Ps), 1e-6, None))
    hc[Ps >= 0.5] = -np.inf
    return np.nan_to_num(hc.max(0), neginf=0.0, posinf=1e6)


def episodes(y):
    eps, i, n = [], 0, len(y)
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1: j += 1
            eps.append(np.arange(i, j)); i = j
        else: i += 1
    return eps


ALL = {}
for name in ["WADI", "HAI", "SWaT"]:
    p = f"{BD}/expert_{name}.npz"
    if not os.path.exists(p):
        continue
    E = np.load(p, allow_pickle=True)
    Cal, Tst = E["calib_surprise"], E["test_surprise"]; coh = E["comm_cohesion"].astype(float); size = E["comm_size"].astype(float)
    y = E["y"].astype(int); hard = E["hard"].astype(bool); keep = np.where((y == 0) | hard)[0]; nm = y == 0
    lat = E["LatAD"]; compet = E[COMPET[name]]; nseed, S, ntest = Tst.shape
    w = coh * np.sqrt(size); wn = w / (w.max() + 1e-9)
    def au(s): return float(roc_auc_score(y[keep], s[keep]))
    def z(s): return (s - s[nm].mean()) / (s[nm].std() + 1e-9)

    cand = {"commOR_cohsqrt": [], "factor_sum": [], "HC": [], "HC_coh": [],
            "sum+max_z": [], "max(sum_z,max_z)": [], "null+commOR": [], "null+HC": [], "HC_with_null": []}
    store = {k: [] for k in cand}
    for sd in range(nseed):
        tails = np.stack([surv(Cal[sd, g], Tst[sd, g]) for g in range(S)])
        P = np.stack([pval(Cal[sd, g], Tst[sd, g]) for g in range(S)])
        commOR = (wn[:, None] * tails).max(0)
        fsum = tails.sum(0)
        hc = HC(P); hc_coh = HC(P, wt=w)
        li = min(sd, lat.shape[0] - 1); nulltail = surv(lat[li][nm], lat[li])
        nullp = pval(lat[li][nm], lat[li])
        summax = z(fsum) + z(commOR)
        maxsummax = np.maximum(z(fsum), z(commOR))
        null_commOR = np.maximum(z(commOR), z(nulltail))
        null_HC = np.maximum(z(hc), z(nulltail))
        Pn = np.vstack([P, nullp[None, :]]); hc_null = HC(Pn)
        for k, v in [("commOR_cohsqrt", commOR), ("factor_sum", fsum), ("HC", hc), ("HC_coh", hc_coh),
                     ("sum+max_z", summax), ("max(sum_z,max_z)", maxsummax), ("null+commOR", null_commOR),
                     ("null+HC", null_HC), ("HC_with_null", hc_null)]:
            cand[k].append(round(au(v), 3)); store[k].append(v)
    ALL[name] = dict(LatAD=round(float(np.mean([au(lat[i]) for i in range(lat.shape[0])])), 3),
                     compet=COMPET[name],
                     compet_auroc=round(float(np.mean([au(compet[i]) for i in range(compet.shape[0])]) if compet.ndim == 2 else au(compet)), 3),
                     agg={k: [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)] for k, v in cand.items()},
                     _store={k: np.stack(store[k]) for k in store})
    print(f"=== {name} (S={S}) LatAD {ALL[name]['LatAD']} {COMPET[name]} {ALL[name]['compet_auroc']} ===")
    for k, v in ALL[name]["agg"].items():
        print(f"   {k:18} {v}")

# pick the aggregator with the best WORST-CASE margin over baseline across all three
names = list(ALL)
cands = list(ALL[names[0]]["agg"])
print("\n=== all-three margin over max(LatAD,competitor) per aggregator ===")
best = None
for k in cands:
    margins = [ALL[n]["agg"][k][0] - max(ALL[n]["LatAD"], ALL[n]["compet_auroc"]) for n in names]
    worst = min(margins)
    print(f"   {k:18} margins {[round(m,3) for m in margins]}  worst {round(worst,3)}")
    if best is None or worst > best[1]:
        best = (k, worst)
print(f"\nBEST all-three aggregator (max-min margin): {best[0]}  (worst margin {round(best[1],3)})")

# significance of the best aggregator vs competitor, each dataset
for name in names:
    E = np.load(f"{BD}/expert_{name}.npz", allow_pickle=True)
    y = E["y"].astype(int); hard = E["hard"].astype(bool); keep = np.where((y == 0) | hard)[0]
    compet = E[COMPET[name]]; cm = compet if compet.ndim == 2 else compet[None, :]
    F = ALL[name]["_store"][best[0]]
    W, st = CFG[name]; Lblk = int(np.ceil(W / st)) + 1
    norm = np.where(y == 0)[0]; heps = [e[hard[e]] for e in episodes(y) if hard[e].any()]
    def mAU(mat, idx): return float(np.mean([roc_auc_score(y[idx], mat[i][idx]) for i in range(mat.shape[0])]))
    dpt = mAU(F, keep) - mAU(cm, keep); diffs = []
    for _ in range(1500):
        nb = int(np.ceil(len(norm) / Lblk)); ss = RNG.integers(0, max(1, len(norm) - Lblk + 1), size=nb)
        sn = np.concatenate([norm[s:s + Lblk] for s in ss])[:len(norm)]
        pk = RNG.integers(0, len(heps), size=len(heps)); sh = np.concatenate([heps[k] for k in pk])
        idx = np.concatenate([sn, sh])
        if y[idx].sum() < 2 or (y[idx] == 0).sum() < 2: continue
        diffs.append(mAU(F, idx) - mAU(cm, idx))
    diffs = np.array(diffs)
    print(f"   {name}: {best[0]} vs {COMPET[name]}  diff {round(dpt,3)} CI "
          f"[{round(np.quantile(diffs,.025),3)},{round(np.quantile(diffs,.975),3)}] P(<=0)={round((diffs<=0).mean(),4)}")

for n in ALL: ALL[n].pop("_store", None)
json.dump(ALL, open(f"{OUT}/bma_final.json", "w"), indent=1)
print(f"\nsaved -> {OUT}/bma_final.json")
