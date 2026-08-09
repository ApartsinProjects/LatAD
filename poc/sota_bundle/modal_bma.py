"""Quality-weighted Bayesian-model-averaging over factorizations, in SURPRISE space (Modal).

Library of models:
  M_0 = NULL (no factorization) = LatAD (from bundle).
  M_1..M_S = PARTITIONS at multiple scales: cut the HAC dendrogram to maxclust in {5,10,20,40}
             -> disjoint blocks. Each partition is a valid factorization (no double-counting):
             surprise s_s(x) = SUM over its blocks of standardized block-density NLL.
Each block's density is a small VaDE latent-density (size>=3) or a Gaussian NLL (size<3), trained
on train-normal, standardized on held-out train-normal. Unique blocks are trained once and reused
across partitions/scales.

Quality prior w_s = weighted MODULARITY of partition s on the |rho| graph (train-normal) -> a good
factorization (blocks internally correlated, between-block independent) gets high weight; a bad one
(splitting correlated channels) gets low weight. The null gets a tunable prior w0.

Aggregate in SURPRISE space (union bound / product-of-experts), NOT a density mixture:
  weighted-max :  max_m  w_m * tail_m(x)            (quality-weighted OR)
  weighted-mean:  sum_m  w_m * tail_m(x) / sum w_m  (quality-weighted PoE)
where tail_m = -log P_heldout-normal(s_m >= .). Reported vs the null (LatAD) and baselines, with
episode-bootstrap significance. Run: modal run modal_bma.py --datasets "WADI,HAI,SWaT" --seeds 2
"""
from __future__ import annotations
import json
from pathlib import Path
import modal

HERE = Path(__file__).parent.resolve()
image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python="3.11")
    .pip_install("numpy<2", "scikit-learn", "scipy")
    .add_local_file(str(HERE.parent / "models_vade.py"), "/app/models_vade.py")
)
for ds in ["WADI", "HAI", "SWaT"]:
    f = HERE / "ens_bundle" / f"bundle_{ds}.npz"
    if f.exists():
        image = image.add_local_file(str(f), f"/app/bundle_{ds}.npz")
app = modal.App("latad-bma", image=image)
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}
COMPET = {"WADI": "IF", "HAI": "AE", "SWaT": "linres"}
MAXCLUST = [5, 10, 20, 40]


@app.function(cpu=8.0, memory=49152, timeout=2 * 60 * 60)
def run_ds(name: str, nseeds: int = 2) -> dict:
    import sys, numpy as np, warnings
    warnings.filterwarnings("ignore"); sys.path.insert(0, "/app")
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.metrics import roc_auc_score
    from models_vade import train_vade

    B = np.load(f"/app/bundle_{name}.npz")
    Xn, Xa = B["Xn_w"].astype(float), B["Xa_w"].astype(float)
    y = B["y"].astype(int); nch = int(B["nch"]); thr = float(B["maxz_thr"])
    hard = (y == 1) & ~((y == 1) & (B["maxz"] > thr)); keep = np.where((y == 0) | hard)[0]
    lat = B["LatAD"]; compet = B[COMPET[name]]; SEEDS = list(range(nseeds))
    nfit = len(Xn) * 4 // 5; nm = y == 0

    means = Xn[:, :nch]; sdc = means.std(0); active = np.where(sdc > 1e-6)[0]
    Zact = (means[:, active] - means[:, active].mean(0)) / sdc[active]
    Cabs = np.abs(np.nan_to_num(np.corrcoef(Zact.T))); np.fill_diagonal(Cabs, 0.0)
    dist = 1 - Cabs; dist = (dist + dist.T) / 2
    L = linkage(squareform(dist, checks=False), method="average")

    # partitions at multiple scales (labels over ACTIVE channels)
    partitions = []                      # list of (blocks[list of orig-ch-idx], modularity)
    k2m = Cabs.sum() / 2.0; deg = Cabs.sum(1)
    def modularity(lab):
        Q = 0.0
        for c in np.unique(lab):
            idx = np.where(lab == c)[0]
            Q += Cabs[np.ix_(idx, idx)].sum() / (2 * k2m) - (deg[idx].sum() / (2 * k2m)) ** 2
        return float(Q)
    for kc in MAXCLUST:
        lab = fcluster(L, t=min(kc, len(active)), criterion="maxclust")
        blocks = [[int(active[i]) for i in np.where(lab == c)[0]] for c in np.unique(lab)]
        partitions.append((blocks, max(0.0, modularity(lab))))

    def sf(Xw, S): return np.ascontiguousarray(Xw[:, [b * nch + c for b in range(6) for c in S]])
    def surv(ref, v):
        o = np.sort(ref); r = np.searchsorted(o, v, side="right") / len(o); return -np.log(np.clip(1 - r, 1e-4, 1))
    def block_nll(G, seed):
        Ff, Fc, Ft = sf(Xn[:nfit], G), sf(Xn[nfit:], G), sf(Xa, G)
        m2, s2 = Ff.mean(0), Ff.std(0) + 1e-9
        Ztr, Zc, Zt = (Ff - m2) / s2, (Fc - m2) / s2, (Ft - m2) / s2
        if len(G) >= 3:
            v = train_vade(Ztr.astype(np.float32), n_clusters=min(20, max(6, len(G))),
                           latent_dim=min(8, max(3, len(G) // 2)), epochs=15, warmup=4, seed=seed, device="cpu")
            v.fit_latent_density(Ztr.astype(np.float32), k_density=min(50, max(12, nfit // 12)))
            cc = np.asarray(v.anomaly_score_hard(Zc.astype(np.float32), use_resid=False, use_basin=False))
            tt = np.asarray(v.anomaly_score_hard(Zt.astype(np.float32), use_resid=False, use_basin=False)); del v
        else:                                   # tiny block: Gaussian NLL proxy (sum z^2 over block mean-feats)
            cc = (Zc[:, :len(G)] ** 2).sum(1); tt = (Zt[:, :len(G)] ** 2).sum(1)
        cm, cs = cc.mean(), cc.std() + 1e-9
        return np.nan_to_num((cc - cm) / cs), np.nan_to_num((tt - cm) / cs)

    def au(s, idx=keep): return float(roc_auc_score(y[idx], s[idx]))

    # accumulate over seeds
    agg_names = ["null_LatAD", "wmax", "wmean", "umax", "wmax_noNull"]
    per = {k: [] for k in agg_names}
    wmax_te = []                                  # for significance (seed-stacked)
    for sd in SEEDS:
        # unique blocks across partitions -> train once
        cache = {}
        def get(G):
            k = tuple(sorted(G))
            if k not in cache: cache[k] = block_nll(list(k), sd)
            return cache[k]
        part_tail_c, part_tail_t, weights = [], [], []
        for blocks, w in partitions:
            sc = np.zeros(len(Xn) - nfit); st = np.zeros(len(Xa))
            for G in blocks:
                bc, bt = get(G); sc += bc; st += bt          # SUM = within-partition factorized NLL
            part_tail_c.append(surv(sc, sc)); part_tail_t.append(surv(sc, st)); weights.append(w)
        weights = np.array(weights); wn = weights / (weights.sum() + 1e-9)
        # null = LatAD (calibrate tail on test-normal proxy, per seed)
        li = min(sd, lat.shape[0] - 1)
        null_t = surv(lat[li][nm], lat[li]); null_c = surv(lat[li][nm], lat[li][nm] if False else lat[li])
        PT = np.stack(part_tail_t)                                   # (S, n_test)
        w0 = float(weights.mean())                                  # null prior = mean partition quality
        # aggregators (surprise space)
        wmax = np.maximum((wn[:, None] * PT).max(0), (w0 / (w0 + weights.sum())) * null_t)  # quality-weighted OR incl null
        wmax_noNull = (wn[:, None] * PT).max(0)
        wmean = (wn[:, None] * PT).sum(0) * (weights.sum() / (weights.sum() + w0)) + null_t * (w0 / (weights.sum() + w0))
        umax = PT.max(0)
        per["null_LatAD"].append(round(au(null_t), 3))
        per["wmax"].append(round(au(wmax), 3)); per["wmean"].append(round(au(wmean), 3))
        per["umax"].append(round(au(umax), 3)); per["wmax_noNull"].append(round(au(wmax_noNull), 3))
        wmax_te.append(wmax)
        print(f"[{name}] seed {sd} mod={[round(w,3) for w in weights]} : " +
              "  ".join(f"{k} {per[k][-1]}" for k in agg_names), flush=True)

    latau = round(float(np.mean([au(lat[i]) for i in range(lat.shape[0])])), 3)
    cau = round(float(np.mean([au(compet[i]) for i in range(compet.shape[0])]) if compet.ndim == 2 else au(compet)), 3)

    # significance: wmax vs competitor, episode bootstrap
    def episodes(yy):
        eps, i, m = [], 0, len(yy)
        while i < m:
            if yy[i] == 1:
                j = i
                while j < m and yy[j] == 1: j += 1
                eps.append(np.arange(i, j)); i = j
            else: i += 1
        return eps
    W, st = CFG[name]; Lblk = int(np.ceil(W / st)) + 1
    norm = np.where(y == 0)[0]; heps = [e[hard[e]] for e in episodes(y) if hard[e].any()]
    F = np.stack(wmax_te); cm = compet if compet.ndim == 2 else compet[None, :]
    def mAU(mat, idx): return float(np.mean([roc_auc_score(y[idx], mat[i][idx]) for i in range(mat.shape[0])]))
    rng = np.random.default_rng(0); dpt = mAU(F, keep) - mAU(cm, keep); diffs = []
    for _ in range(1500):
        nb = int(np.ceil(len(norm) / Lblk)); ss = rng.integers(0, max(1, len(norm) - Lblk + 1), size=nb)
        sn = np.concatenate([norm[s:s + Lblk] for s in ss])[:len(norm)]
        pk = rng.integers(0, len(heps), size=len(heps)); sh = np.concatenate([heps[k] for k in pk])
        idx = np.concatenate([sn, sh])
        if y[idx].sum() < 2 or (y[idx] == 0).sum() < 2: continue
        diffs.append(mAU(F, idx) - mAU(cm, idx))
    diffs = np.array(diffs)
    res = dict(dataset=name, LatAD=latau, compet=COMPET[name], compet_auroc=cau,
               partition_modularity=[round(w, 3) for _, w in partitions],
               agg={k: [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)] for k, v in per.items()},
               wmax_sig_vs_compet=dict(diff=round(float(dpt), 3),
                   ci=[round(float(np.quantile(diffs, 0.025)), 3), round(float(np.quantile(diffs, 0.975)), 3)],
                   p_le_0=round(float((diffs <= 0).mean()), 4)))
    print(f"[{name}] DONE: {json.dumps(res)}", flush=True)
    return res


@app.local_entrypoint()
def main(datasets: str = "WADI,HAI,SWaT", seeds: int = 2):
    ds = datasets.split(",")
    results = list(run_ds.starmap([(d, seeds) for d in ds]))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "bma_modal.json").write_text(json.dumps(results, indent=2))
    print("\n==================== BMA SUMMARY ====================")
    for r in results:
        a = r["agg"]
        print(f"{r['dataset']:5} LatAD {r['LatAD']} vs {r['compet']} {r['compet_auroc']} | mod {r['partition_modularity']}"
              f" | wmax {a['wmax']} wmean {a['wmean']} wmax_noNull {a['wmax_noNull']} umax {a['umax']}"
              f" | wmax vs {r['compet']}: {r['wmax_sig_vs_compet']}")
