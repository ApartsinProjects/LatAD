"""Train the per-community EXPERT LIBRARY once and cache each expert's per-window surprise,
so aggregation/weighting ideas can be explored locally with NO retraining.

Experts = HAC nested correlation communities (dendrogram subtrees size [3,25]) on train-normal.
For each community G and seed: a small VaDE latent-density; dump its standardized NLL on the
held-out train-normal (CALIB) and on TEST. Also record community metadata: channels, size, and
COHESION = mean pairwise |rho| within G (train-normal) -> the informativeness weight (a density
violation in a tightly-correlated community is more diagnostic; user's point).

Returns per dataset a compact artifact (saved to sota_bundle/experts/expert_<DS>.npz):
  test_surprise (nseed,S,n_test), calib_surprise (nseed,S,n_calib), comm_size, comm_cohesion,
  comm_channels(padded), y, hard, maxz, maxz_thr, LatAD(test), IF/AE/linres(test).
Aggregation (weighted min-prob incl null LatAD, etc.) is then a cheap local script over this.

Run: modal run modal_experts.py --datasets "WADI,HAI,SWaT" --seeds 3
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
for ds in ["WADI", "HAI", "SWaT", "WADI_clean", "SWaT_canon"]:
    f = HERE / "ens_bundle" / f"bundle_{ds}.npz"
    if f.exists():
        image = image.add_local_file(str(f), f"/app/bundle_{ds}.npz")
app = modal.App("latad-experts", image=image)
MAXSZ = 25


@app.function(cpu=8.0, memory=65536, timeout=5 * 60 * 60)
def experts(name: str, nseeds: int = 3, seed_base: int = 0, use_full: int = 0) -> dict:
    import sys, gc, numpy as np, warnings
    warnings.filterwarnings("ignore"); sys.path.insert(0, "/app")
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform
    from models_vade import train_vade

    B = np.load(f"/app/bundle_{name}.npz")
    Xn, Xa = B["Xn_w"].astype(float), B["Xa_w"].astype(float)
    nch = int(B["nch"]); nfit = len(Xn) * 4 // 5
    means = Xn[:, :nch]; sdc = means.std(0); active = np.where(sdc > 1e-6)[0]
    Zact = (means[:, active] - means[:, active].mean(0)) / sdc[active]
    Cabs = np.abs(np.nan_to_num(np.corrcoef(Zact.T))); np.fill_diagonal(Cabs, 0.0)
    dist = 1 - Cabs; dist = (dist + dist.T) / 2
    L = linkage(squareform(dist, checks=False), method="average"); _, nodes = to_tree(L, rd=True)
    def leaves(n): return [n.id] if n.is_leaf() else leaves(n.left) + leaves(n.right)
    seen, comms, coh = set(), [], []
    a2p = {int(a): i for i, a in enumerate(active)}                # orig-ch -> active-pos
    for n in nodes:
        if not n.is_leaf():
            lv = sorted(leaves(n))
            if 3 <= len(lv) <= MAXSZ and tuple(lv) not in seen:
                seen.add(tuple(lv))
                G = [int(active[i]) for i in lv]
                comms.append(G)
                pos = [a2p[c] for c in G]
                sub = Cabs[np.ix_(pos, pos)]
                coh.append(float(sub.sum() / (len(pos) * (len(pos) - 1))))   # mean pairwise |rho|

    def sf(Xw, S): return np.nan_to_num(np.ascontiguousarray(Xw[:, [b * nch + c for b in range(6) for c in S]]))
    S = len(comms); ncal = len(Xn) - nfit; ntest = len(Xa)
    Cal = np.zeros((nseeds, S, ncal), np.float32); Tst = np.zeros((nseeds, S, ntest), np.float32)
    for si in range(nseeds):
        sd = seed_base + si
        nfail = 0
        for gi, G in enumerate(comms):
            try:
                Ff, Fc, Ft = sf(Xn[:nfit], G), sf(Xn[nfit:], G), sf(Xa, G)
                m2, s2 = Ff.mean(0), Ff.std(0) + 1e-9
                cl = lambda A: np.clip(np.nan_to_num(((A - m2) / s2), posinf=10.0, neginf=-10.0), -10, 10).astype(np.float32)
                Ztr, Zc2, Zt2 = cl(Ff), cl(Fc), cl(Ft)
                v = train_vade(Ztr, n_clusters=min(20, max(6, len(G))),
                               latent_dim=min(8, max(3, len(G) // 2)), epochs=15, warmup=4, seed=sd, device="cpu")
                v.fit_latent_density(Ztr, k_density=min(50, max(12, nfit // 12)))
                if use_full:   # apply A8 (per-community whitened residual) + A3 (basin) inside each expert
                    v.fit_residual_whitener(Ztr); v.fit_resid_head(Ztr); v.fit_basin_head(Ztr)
                ur = "auto" if use_full else False; ub = "auto" if use_full else False
                cc = np.asarray(v.anomaly_score_hard(Zc2, use_resid=ur, use_basin=ub))
                tt = np.asarray(v.anomaly_score_hard(Zt2, use_resid=ur, use_basin=ub)); del v
                cm, cs = cc.mean(), cc.std() + 1e-9
                if not (np.isfinite(cm) and cs > 0):
                    raise ValueError("non-finite expert scores")
                Cal[si, gi] = np.nan_to_num((cc - cm) / cs); Tst[si, gi] = np.nan_to_num((tt - cm) / cs)
                del Ff, Fc, Ft, Ztr, Zc2, Zt2, cc, tt
            except Exception as e:
                Cal[si, gi] = 0.0; Tst[si, gi] = 0.0; nfail += 1
                print(f"[{name}] seed {sd} comm {gi} (|G|={len(G)}) skipped: {e}", flush=True)
            gc.collect()
        print(f"[{name}] seed {sd}: {S} experts trained ({nfail} skipped)", flush=True)

    if not np.all(np.isfinite(Cal)) or not np.all(np.isfinite(Tst)):
        Cal = np.nan_to_num(Cal); Tst = np.nan_to_num(Tst)
    mx = max(len(g) for g in comms)
    chpad = np.full((S, mx), -1, int)
    for i, g in enumerate(comms): chpad[i, :len(g)] = g
    out = dict(name=name, S=S, comm_size=[len(g) for g in comms], comm_cohesion=coh,
               comm_channels=chpad.tolist(),
               calib_surprise=Cal, test_surprise=Tst,
               y=B["y"], hard=(B["y"].astype(int) == 1) & ~((B["y"].astype(int) == 1) & (B["maxz"] > float(B["maxz_thr"]))))
    print(f"[{name}] DONE experts={S}", flush=True)
    return out


@app.local_entrypoint()
def main(datasets: str = "WADI,HAI,SWaT", seeds: int = 3, full: int = 0):
    import numpy as np
    ds = datasets.split(",")
    outdir = HERE / ("experts_full" if full else "experts"); outdir.mkdir(exist_ok=True)
    for d in ds:
        # seeds are independent -> run them as parallel containers (wall-clock ~= one seed)
        cals, tsts, meta = [], [], None
        for s, r in enumerate(experts.starmap([(d, 1, s, full) for s in range(seeds)], return_exceptions=True)):
            if isinstance(r, Exception):
                print(f"[{d}] seed {s} errored: {r}", flush=True); continue
            cals.append(np.asarray(r["calib_surprise"])[0]); tsts.append(np.asarray(r["test_surprise"])[0]); meta = r
        if meta is None:
            print(f"[{d}] ALL seeds failed", flush=True); continue
        np.savez(outdir / f"expert_{d}.npz",
                 comm_size=np.array(meta["comm_size"]), comm_cohesion=np.array(meta["comm_cohesion"]),
                 comm_channels=np.array(meta["comm_channels"]),
                 calib_surprise=np.stack(cals), test_surprise=np.stack(tsts),
                 y=meta["y"], hard=meta["hard"])
        print(f"[{d}] saved {len(cals)}/{seeds} seeds (parallel, S={meta['S']})", flush=True)
