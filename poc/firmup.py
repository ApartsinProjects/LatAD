"""Firmed-up, leak-free evaluation of the LatAD + subspace-VaDE-ensemble tail-max fusion.

Rigor gains over the exploratory run:
  * Calibration is on HELD-OUT TRAIN-NORMAL (CALIB split), not test-normal. LatAD is retrained
    to dump train-normal (CALIB) AND test scores consistently; each ensemble member likewise
    scores CALIB + test. Survival functions for the tail-max fusion are fit on CALIB only.
  * The ensemble is MULTI-REPLICA (R independent random-subset draws + VaDE inits) -> mean+-std.
  * Difficulty split = the paper's train-normal max|z| threshold (from scores_<DS>.npz).

Fusion: fused = max( -log P_CALIB(LatAD>=s), -log P_CALIB(ens>=s) )  (union-bound over experts).
Reports difficult-subset AUROC (mean+-std over LatAD-seed x ensemble-replica) + episode
bootstrap vs the strongest baseline. Output -> _diagnostics/firmup.json.
"""
from __future__ import annotations
import json, os, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
import eda_real as E

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
LATCFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}    # LatAD (n_clusters, latent_dim)
BLK = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}       # window (W, stride) for episodes
SEEDS = [0, 1, 2, 3, 4]
R, KE, MSUB = 3, 16, {"WADI": 24, "HAI": 20, "SWaT": 16}          # ensemble replicas / members / subset size
NCL_E, LD_E, EP_E = 20, 8, 20
COMPET = {"WADI": "IF", "HAI": "AE", "SWaT": "linres"}


def subset_feats(Xw, nch, S):
    return np.ascontiguousarray(Xw[:, [b * nch + c for b in range(6) for c in S]])


def surv(ref, v):
    o = np.sort(ref); r = np.searchsorted(o, v, side="right") / len(o)
    return -np.log(np.clip(1.0 - r, 1e-4, 1.0))


def episodes(y):
    eps, i, n = [], 0, len(y)
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1:
                j += 1
            eps.append(np.arange(i, j)); i = j
        else:
            i += 1
    return eps


ALL = {}
for name in ["WADI", "HAI", "SWaT"]:
    D = E.load(name); nch = len(D["ch"])
    Xn = np.asarray(D["Xn_w"], float); Xa = np.asarray(D["Xa_w"], float)
    y = np.asarray(D["ya_w"], int)
    d = np.load(f"{OUT}/scores_{name}.npz"); thr = float(d["maxz_thr"])
    hard = (y == 1) & ~((y == 1) & (d["maxz"] > thr)); keep = np.where((y == 0) | hard)[0]
    compet = d[COMPET[name]]
    mu, sg = Xn.mean(0), Xn.std(0) + 1e-9
    Ztr = ((Xn - mu) / sg).astype(np.float32); Zte = ((Xa - mu) / sg).astype(np.float32)
    nfit = len(Ztr) * 4 // 5; FIT, CAL = Ztr[:nfit], Ztr[nfit:]
    K, LD = LATCFG[name]

    # --- LatAD retrained per seed: CALIB + TEST scores ---
    lat_cal, lat_te = [], []
    for sd in SEEDS:
        v = train_vade(FIT, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
        v.fit_residual_whitener(FIT); v.fit_latent_density(FIT, k_density=min(80, max(20, nfit // 10)))
        v.fit_resid_head(FIT); v.fit_basin_head(FIT)
        lat_cal.append(np.asarray(v.anomaly_score_hard(CAL, use_resid="auto", use_basin="auto")))
        lat_te.append(np.asarray(v.anomaly_score_hard(Zte, use_resid="auto", use_basin="auto")))
        print(f"  {name} LatAD seed {sd} done", flush=True)

    # --- ensemble: R replicas, each KE random-subset VaDEs; CALIB + TEST top-3 ---
    ens_cal, ens_te = [], []
    for rep in range(R):
        rng = np.random.default_rng(100 + rep)
        cc, ct = [], []
        for j in range(KE):
            S = np.sort(rng.choice(nch, size=MSUB[name], replace=False))
            Ff, Fc, Ft = subset_feats(FIT, nch, S), subset_feats(CAL, nch, S), subset_feats(Zte, nch, S)
            m2, s2 = Ff.mean(0), Ff.std(0) + 1e-9
            vf = train_vade(((Ff - m2) / s2).astype(np.float32), n_clusters=NCL_E, latent_dim=LD_E,
                            epochs=EP_E, warmup=5, seed=rep, device="cpu")
            vf.fit_latent_density(((Ff - m2) / s2).astype(np.float32), k_density=min(60, max(15, nfit // 10)))
            sc_c = np.asarray(vf.anomaly_score_hard(((Fc - m2) / s2).astype(np.float32), use_resid=False, use_basin=False))
            sc_t = np.asarray(vf.anomaly_score_hard(((Ft - m2) / s2).astype(np.float32), use_resid=False, use_basin=False))
            ref = sc_c
            cc.append(np.nan_to_num((sc_c - ref.mean()) / (ref.std() + 1e-9)))
            ct.append(np.nan_to_num((sc_t - ref.mean()) / (ref.std() + 1e-9)))
        ens_cal.append(np.sort(np.stack(cc), axis=0)[-3:].mean(0))
        ens_te.append(np.sort(np.stack(ct), axis=0)[-3:].mean(0))
        print(f"  {name} ensemble replica {rep+1}/{R} done", flush=True)

    def au(s): return float(roc_auc_score(y[keep], s[keep]))
    lat_au = [au(t) for t in lat_te]; ens_au = [au(t) for t in ens_te]
    fused_te, fused_au = [], []
    for i in range(len(SEEDS)):
        for r in range(R):
            f = np.maximum(surv(lat_cal[i], lat_te[i]), surv(ens_cal[r], ens_te[r]))
            fused_te.append(f); fused_au.append(au(f))
    ALL[name] = dict(
        LatAD=[round(np.mean(lat_au), 3), round(np.std(lat_au), 3)],
        ensemble=[round(np.mean(ens_au), 3), round(np.std(ens_au), 3)],
        fused_tailmax=[round(np.mean(fused_au), 3), round(np.std(fused_au), 3)],
        compet=COMPET[name], compet_auroc=round(float(np.mean([au(compet[i]) for i in range(compet.shape[0])]) if compet.ndim == 2 else au(compet)), 3))
    # significance: fused (all combos) vs competitor, episode bootstrap
    W, st = BLK[name]; L = int(np.ceil(W / st)) + 1
    norm = np.where(y == 0)[0]; heps = [e[hard[e]] for e in episodes(y) if hard[e].any()]
    F = np.stack(fused_te); cm = compet if compet.ndim == 2 else compet[None, :]
    def mAU(m, idx): return float(np.mean([roc_auc_score(y[idx], m[i][idx]) for i in range(m.shape[0])]))
    rng = np.random.default_rng(0); dpt = mAU(F, keep) - mAU(cm, keep); diffs = []
    for _ in range(1500):
        nb = int(np.ceil(len(norm) / L)); ssi = rng.integers(0, max(1, len(norm) - L + 1), size=nb)
        sn = np.concatenate([norm[s:s + L] for s in ssi])[:len(norm)]
        pk = rng.integers(0, len(heps), size=len(heps)); sh = np.concatenate([heps[k] for k in pk])
        idx = np.concatenate([sn, sh])
        if y[idx].sum() < 2 or (y[idx] == 0).sum() < 2:
            continue
        diffs.append(mAU(F, idx) - mAU(cm, idx))
    diffs = np.array(diffs)
    ALL[name]["sig_vs_compet"] = dict(diff=round(float(dpt), 3),
        ci=[round(float(np.quantile(diffs, 0.025)), 3), round(float(np.quantile(diffs, 0.975)), 3)],
        p_le_0=round(float((diffs <= 0).mean()), 4))
    print(f"=== {name}: LatAD {ALL[name]['LatAD']}  ens {ALL[name]['ensemble']}  "
          f"FUSED {ALL[name]['fused_tailmax']}  vs {COMPET[name]} {ALL[name]['compet_auroc']}: "
          f"{ALL[name]['sig_vs_compet']}", flush=True)
    json.dump(ALL, open(f"{OUT}/firmup.json", "w"), indent=1)
print(f"\nsaved -> {OUT}/firmup.json")
