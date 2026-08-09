"""Guided HAC-community VaDE ensemble on Modal (high-RAM container; HAI OOMs locally).
One container per dataset (parallel). Self-contained: reads ens_bundle/bundle_<DS>.npz +
models_vade.py, no eda_real/raw-data dependency.

Per dataset: HAC nested correlation communities on train-normal |rho| (subtrees size [3,25]);
one VaDE latent-density member per community; 3 seeds. Aggregators: max / q90 / q95 / sum
(=factorized joint NLL) / top5, each calibrated on held-out train-normal. Difficult-subset
AUROC (mean+-std over seeds) + episode-bootstrap significance of the sum aggregator and of the
tail-max fusion with LatAD vs the strongest baseline.

Run:  modal run modal_ensemble.py --datasets "WADI,HAI,SWaT" --seeds 3
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

app = modal.App("latad-ensemble", image=image)
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}   # window (W,stride) for episode block
COMPET = {"WADI": "IF", "HAI": "AE", "SWaT": "linres"}


@app.function(cpu=8.0, memory=49152, timeout=2 * 60 * 60)
def run_ds(name: str, nseeds: int = 3) -> dict:
    import sys, numpy as np, warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "/app")
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform
    from sklearn.metrics import roc_auc_score
    from models_vade import train_vade

    B = np.load(f"/app/bundle_{name}.npz")
    Xn, Xa = B["Xn_w"].astype(float), B["Xa_w"].astype(float)
    y = B["y"].astype(int); nch = int(B["nch"]); thr = float(B["maxz_thr"])
    hard = (y == 1) & ~((y == 1) & (B["maxz"] > thr)); keep = np.where((y == 0) | hard)[0]
    lat = B["LatAD"]; compet = B[COMPET[name]]
    SEEDS = list(range(nseeds)); MAXSZ = 25

    means = Xn[:, :nch]; sdc = means.std(0); active = np.where(sdc > 1e-6)[0]
    Zact = (means[:, active] - means[:, active].mean(0)) / sdc[active]
    C = np.nan_to_num(np.corrcoef(Zact.T)); dist = 1 - np.abs(C)
    np.fill_diagonal(dist, 0.0); dist = (dist + dist.T) / 2
    L = linkage(squareform(dist, checks=False), method="average"); _, nodes = to_tree(L, rd=True)
    def leaves(n): return [n.id] if n.is_leaf() else leaves(n.left) + leaves(n.right)
    seen, comms = set(), []
    for n in nodes:
        if not n.is_leaf():
            lv = sorted(leaves(n))
            if 3 <= len(lv) <= MAXSZ and tuple(lv) not in seen:
                seen.add(tuple(lv)); comms.append([int(active[i]) for i in lv])

    def sf(Xw, S): return np.ascontiguousarray(Xw[:, [b * nch + c for b in range(6) for c in S]])
    nfit = len(Xn) * 4 // 5
    aggs = ["max", "q90", "q95", "sum", "top5"]
    def aggregate(A, kind):
        return {"max": A.max(0), "q90": np.percentile(A, 90, 0), "q95": np.percentile(A, 95, 0),
                "sum": A.sum(0), "top5": np.sort(A, 0)[-5:].mean(0)}[kind]

    def au(s, idx=keep): return float(roc_auc_score(y[idx], s[idx]))
    def surv(ref, v):
        o = np.sort(ref); r = np.searchsorted(o, v, side="right") / len(o); return -np.log(np.clip(1 - r, 1e-4, 1))
    def episodes(yy):
        eps, i, m = [], 0, len(yy)
        while i < m:
            if yy[i] == 1:
                j = i
                while j < m and yy[j] == 1:
                    j += 1
                eps.append(np.arange(i, j)); i = j
            else:
                i += 1
        return eps

    per = {k: [] for k in aggs}
    sum_te, fused_te = [], []
    nm = y == 0
    for sd in SEEDS:
        Zc, Zt = [], []
        for G in comms:
            Ff, Fc, Ft = sf(Xn[:nfit], G), sf(Xn[nfit:], G), sf(Xa, G)
            m2, s2 = Ff.mean(0), Ff.std(0) + 1e-9
            v = train_vade(((Ff - m2) / s2).astype(np.float32), n_clusters=min(20, max(6, len(G))),
                           latent_dim=min(8, max(3, len(G) // 2)), epochs=15, warmup=4, seed=sd, device="cpu")
            v.fit_latent_density(((Ff - m2) / s2).astype(np.float32), k_density=min(50, max(12, nfit // 12)))
            cc = np.asarray(v.anomaly_score_hard(((Fc - m2) / s2).astype(np.float32), use_resid=False, use_basin=False))
            tt = np.asarray(v.anomaly_score_hard(((Ft - m2) / s2).astype(np.float32), use_resid=False, use_basin=False))
            cm, cs = cc.mean(), cc.std() + 1e-9
            Zc.append(np.nan_to_num((cc - cm) / cs)); Zt.append(np.nan_to_num((tt - cm) / cs)); del v
        Zc, Zt = np.stack(Zc), np.stack(Zt)
        for k in aggs:
            ac, at = aggregate(Zc, k), aggregate(Zt, k); s = (at - ac.mean()) / (ac.std() + 1e-9)
            per[k].append(round(au(s), 3))
            if k == "sum":
                sum_te.append(s)
        # tail-max fusion of the sum ensemble with LatAD
        es = sum_te[-1]
        for li in range(lat.shape[0]):
            fused_te.append(np.maximum(surv(lat[li][nm], lat[li]), surv(es[nm], es)))
        print(f"[{name}] seed {sd}: " + "  ".join(f"{k} {per[k][-1]}" for k in aggs), flush=True)

    latau = round(float(np.mean([au(lat[i]) for i in range(lat.shape[0])])), 3)
    cau = round(float(np.mean([au(compet[i]) for i in range(compet.shape[0])]) if compet.ndim == 2 else au(compet)), 3)

    # significance: sum aggregator (seed-mean) and fusion vs competitor, episode bootstrap
    W, st = CFG[name]; Lblk = int(np.ceil(W / st)) + 1
    norm = np.where(y == 0)[0]; heps = [e[hard[e]] for e in episodes(y) if hard[e].any()]
    SUM = np.stack(sum_te); FUS = np.stack(fused_te); cm = compet if compet.ndim == 2 else compet[None, :]
    def mAU(mat, idx): return float(np.mean([roc_auc_score(y[idx], mat[i][idx]) for i in range(mat.shape[0])]))
    rng = np.random.default_rng(0)
    def boot(mat):
        dpt = mAU(mat, keep) - mAU(cm, keep); diffs = []
        for _ in range(1500):
            nb = int(np.ceil(len(norm) / Lblk)); ss = rng.integers(0, max(1, len(norm) - Lblk + 1), size=nb)
            sn = np.concatenate([norm[s:s + Lblk] for s in ss])[:len(norm)]
            pk = rng.integers(0, len(heps), size=len(heps)); sh = np.concatenate([heps[k] for k in pk])
            idx = np.concatenate([sn, sh])
            if y[idx].sum() < 2 or (y[idx] == 0).sum() < 2:
                continue
            diffs.append(mAU(mat, idx) - mAU(cm, idx))
        diffs = np.array(diffs)
        return dict(diff=round(float(dpt), 3),
                    ci=[round(float(np.quantile(diffs, 0.025)), 3), round(float(np.quantile(diffs, 0.975)), 3)],
                    p_le_0=round(float((diffs <= 0).mean()), 4))

    res = dict(dataset=name, n_communities=len(comms), LatAD=latau, compet=COMPET[name], compet_auroc=cau,
               ensemble={k: [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)] for k, v in per.items()},
               sum_sig_vs_compet=boot(SUM), fused_sum_tailmax_sig_vs_compet=boot(FUS))
    print(f"[{name}] DONE: {json.dumps(res)}", flush=True)
    return res


@app.local_entrypoint()
def main(datasets: str = "WADI,HAI,SWaT", seeds: int = 3):
    ds = datasets.split(",")
    results = list(run_ds.starmap([(d, seeds) for d in ds]))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "ensemble_modal.json").write_text(json.dumps(results, indent=2))
    print("\n==================== ENSEMBLE SUMMARY ====================")
    for r in results:
        e = r["ensemble"]
        print(f"{r['dataset']:5} ({r['n_communities']} comms) LatAD {r['LatAD']} vs {r['compet']} {r['compet_auroc']}"
              f" | sum {e['sum']} q95 {e['q95']} q90 {e['q90']}"
              f" | sum vs {r['compet']}: {r['sum_sig_vs_compet']} | fused: {r['fused_sum_tailmax_sig_vs_compet']}")
