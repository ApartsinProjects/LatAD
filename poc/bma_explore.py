"""Explore aggregations over the CACHED expert library (sota_bundle/experts/expert_<DS>.npz)
with NO retraining. Each community expert has a per-window surprise (calibrated on held-out
train-normal) and a COHESION weight (mean pairwise |rho| within the community).

Key object (per user): per-community calibrated tail, aggregated by quality-WEIGHTED min-
probability (max), where a violation in a tightly-correlated community counts more:
    tail_G(x) = -log P_calib(surprise_G >= surprise_G(x))          # leak-free, on held-out normal
    score(x)  = max_G [ w_G * tail_G(x) ]                          # weighted OR, isolates violated G
Weight variants explored: cohesion, cohesion*sqrt(size) (sqrt(m) amplification), cohesion^2.
The NULL (LatAD) is included as another expert (test-normal tail proxy) with prior w0.

Reports difficult-subset AUROC (mean+-std over seeds) + episode bootstrap vs the strongest
baseline, for each weighting. Instant; rerun freely as new weighting ideas arise.
"""
from __future__ import annotations
import json, os, sys, numpy as np
from sklearn.metrics import roc_auc_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
BD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sota_bundle", "experts")
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}
COMPET = {"WADI": "IF", "HAI": "AE", "SWaT": "linres"}
RNG = np.random.default_rng(0)


def surv(ref, v):
    o = np.sort(ref); r = np.searchsorted(o, v, side="right") / len(o); return -np.log(np.clip(1 - r, 1e-4, 1.0))


def pval(ref, v):
    """upper-tail p-value P_calib(surprise >= v); small = anomalous."""
    o = np.sort(ref); r = np.searchsorted(o, v, side="left") / len(o); return np.clip(1 - r, 1e-4, 1.0)


def higher_criticism(P):
    """HC over per-factor p-values P (S, n) -> (n,). Sparse-optimal combination (Donoho-Jin).
    HC(x) = max_{i: p_(i) < 0.5} sqrt(S) (i/S - p_(i)) / sqrt(p_(i)(1-p_(i)))."""
    S, n = P.shape
    Ps = np.sort(P, axis=0)                       # ascending per window
    i = (np.arange(1, S + 1) / S)[:, None]        # (S,1)
    denom = np.sqrt(np.clip(Ps * (1 - Ps), 1e-6, None))
    hc = np.sqrt(S) * (i - Ps) / denom
    hc[Ps >= 0.5] = -np.inf                       # standard HC restriction to the small-p half
    out = hc.max(0)
    return np.nan_to_num(out, neginf=0.0, posinf=1e6)   # windows with no small-p factor -> 0


def episodes(y):
    eps, i, n = [], 0, len(y)
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1: j += 1
            eps.append(np.arange(i, j)); i = j
        else: i += 1
    return eps


def main(names):
    ALL = {}
    for name in names:
        p = f"{BD}/expert_{name}.npz"
        if not os.path.exists(p):
            print(f"[{name}] no expert artifact yet ({p})"); continue
        E = np.load(p, allow_pickle=True)
        Cal, Tst = E["calib_surprise"], E["test_surprise"]      # (nseed, S, n)
        size = E["comm_size"].astype(float); coh = E["comm_cohesion"].astype(float)
        y = E["y"].astype(int); hard = E["hard"].astype(bool); keep = np.where((y == 0) | hard)[0]
        lat = E["LatAD"]; compet = E[COMPET[name]]; nm = y == 0
        nseed, S, ntest = Tst.shape
        weightings = {"cohesion": coh, "coh*sqrt(size)": coh * np.sqrt(size),
                      "coh^2": coh ** 2, "uniform": np.ones(S)}
        def au(s): return float(roc_auc_score(y[keep], s[keep]))

        res = {"S": int(S), "LatAD": round(float(np.mean([au(lat[i]) for i in range(lat.shape[0])])), 3),
               "compet": COMPET[name],
               "compet_auroc": round(float(np.mean([au(compet[i]) for i in range(compet.shape[0])]) if compet.ndim == 2 else au(compet)), 3),
               "weightings": {}}
        best_key, best_seedscores = None, None
        for wk, w in weightings.items():
            wn = w / (w.max() + 1e-9)
            per, seedscores = [], []
            for sd in range(nseed):
                # per-community calibrated tails (calib=held-out normal for this seed)
                tails = np.stack([surv(Cal[sd, g], Tst[sd, g]) for g in range(S)])   # (S, ntest)
                comm_or = (wn[:, None] * tails).max(0)                               # weighted OR (factor-max)
                # include null (LatAD) as an expert; test-normal tail proxy
                li = min(sd, lat.shape[0] - 1)
                nulltail = surv(lat[li][nm], lat[li])
                w0 = float(np.median(wn))                                            # null prior
                score = np.maximum(comm_or, w0 * nulltail)
                per.append(round(au(score), 3)); seedscores.append(score)
            res["weightings"][wk] = [round(float(np.mean(per)), 3), round(float(np.std(per)), 3)]
            if best_key is None or np.mean(per) > np.mean([res["weightings"][best_key][0]]):
                best_key, best_seedscores = wk, np.stack(seedscores)
        # extra aggregators over the COMMUNITY factors: HC (sparse-optimal), sum (dense/factorized), mean
        extra = {"HC": [], "factor_sum": [], "factor_mean": [], "cohes_wmax": []}
        for sd in range(nseed):
            tails = np.stack([surv(Cal[sd, g], Tst[sd, g]) for g in range(S)])
            P = np.stack([pval(Cal[sd, g], Tst[sd, g]) for g in range(S)])
            wn = coh / (coh.max() + 1e-9)
            extra["HC"].append(round(au(higher_criticism(P)), 3))
            extra["factor_sum"].append(round(au(tails.sum(0)), 3))       # ~ user's level-sum, over communities
            extra["factor_mean"].append(round(au(tails.mean(0)), 3))
            extra["cohes_wmax"].append(round(au((wn[:, None] * tails).max(0)), 3))
        res["aggregators"] = {k: [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)] for k, v in extra.items()}
        # significance for the best weighting vs competitor
        W, st = CFG[name]; Lblk = int(np.ceil(W / st)) + 1
        norm = np.where(y == 0)[0]; heps = [e[hard[e]] for e in episodes(y) if hard[e].any()]
        cm = compet if compet.ndim == 2 else compet[None, :]
        def mAU(mat, idx): return float(np.mean([roc_auc_score(y[idx], mat[i][idx]) for i in range(mat.shape[0])]))
        dpt = mAU(best_seedscores, keep) - mAU(cm, keep); diffs = []
        for _ in range(1500):
            nb = int(np.ceil(len(norm) / Lblk)); ss = RNG.integers(0, max(1, len(norm) - Lblk + 1), size=nb)
            sn = np.concatenate([norm[s:s + Lblk] for s in ss])[:len(norm)]
            pk = RNG.integers(0, len(heps), size=len(heps)); sh = np.concatenate([heps[k] for k in pk])
            idx = np.concatenate([sn, sh])
            if y[idx].sum() < 2 or (y[idx] == 0).sum() < 2: continue
            diffs.append(mAU(best_seedscores, idx) - mAU(cm, idx))
        diffs = np.array(diffs)
        res["best_weighting"] = best_key
        res["best_sig_vs_compet"] = dict(diff=round(float(dpt), 3),
            ci=[round(float(np.quantile(diffs, 0.025)), 3), round(float(np.quantile(diffs, 0.975)), 3)],
            p_le_0=round(float((diffs <= 0).mean()), 4))
        ALL[name] = res
        print(f"=== {name} (S={S}) LatAD {res['LatAD']} vs {res['compet']} {res['compet_auroc']} ===")
        for wk, v in res["weightings"].items():
            print(f"   weighted-OR [{wk:14}] {v}")
        print("   aggregators: " + "  ".join(f"{k} {v}" for k, v in res["aggregators"].items())
              + "   (factor_sum ~ your level-sum; HC = Higher Criticism)")
        print(f"   best={best_key}  sig vs {res['compet']}: {res['best_sig_vs_compet']}")
    json.dump(ALL, open(f"{OUT}/bma_explore.json", "w"), indent=1)
    print(f"\nsaved -> {OUT}/bma_explore.json")


if __name__ == "__main__":
    main(sys.argv[1:] or ["WADI", "HAI", "SWaT"])
