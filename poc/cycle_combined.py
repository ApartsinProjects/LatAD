"""Synthesis: LatAD + miss-rescue wrapper over TWO complementary auxiliary heads:
  - common-mode block level (A5)      -> targets WADI coherent-block translations
  - random-subspace covariance density (A5/A2) -> targets HAI cross-channel breaks
Family-wise calibrated so the joint rescue rate on normal stays ~0.5%. The wrapper never
lowers LatAD and skips points LatAD already ranks extreme -> the SWaT ceiling is protected.

Per seed: difficult-subset AUROC of LatAD vs the rescued score; WADI + HAI episode-bootstrap
significance vs their strongest baseline. Exploratory calibration (test-normal reference).
"""
from __future__ import annotations
import json, os, numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
import cycle_lib as CL

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
RNG = np.random.default_rng(0)
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}
COMPET = {"WADI": "IF", "HAI": "AE", "SWaT": "linres"}


def subspace_max(A, B, Za, m=8, K=400):
    nch = A.shape[1]; cols = []
    for _ in range(K):
        S = RNG.choice(nch, size=m, replace=False)
        lw = LedoitWolf().fit(A[:, S]); mu = A[:, S].mean(0); P = lw.precision_
        def maha(X):
            dd = X[:, S] - mu; return np.einsum('ij,jk,ik->i', dd, P, dd)
        dB = maha(B); cols.append((maha(Za) - dB.mean()) / (dB.std() + 1e-9))
    return np.max(cols, axis=0)


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


def famwise_rescue(latad, aux_pcts, y, u0_hi=0.99, fw=0.995):
    nm = y == 0
    u0 = CL.pct(latad[nm], latad)
    aux = np.max(aux_pcts, axis=0)                 # family-wise: strongest aux evidence
    fire = (u0 < u0_hi) & (aux > fw)
    out = u0.copy(); out[fire] = np.maximum(u0[fire], aux[fire])
    return out, int(fire.sum()), int((fire & nm).sum())


ALL = {}
for name in ["WADI", "HAI", "SWaT"]:
    bl = CL.load_blocks(name); y = bl["y"]; Zn, Za = bl["Zn"], bl["Za"]
    nA = len(Zn) // 2; A, B = Zn[:nA], Zn[nA:]
    d = np.load(f"{OUT}/scores_{name}.npz"); thr = float(d["maxz_thr"])
    hard = (y == 1) & ~((y == 1) & (d["maxz"] > thr)); keep = np.where((y == 0) | hard)[0]
    cmh = CL.common_mode_head(bl); sub = subspace_max(A, B, Za)
    nm = y == 0
    aux_pcts = np.stack([CL.pct(cmh[nm], cmh), CL.pct(sub[nm], sub)])

    lat = d["LatAD"]; ifm = d["IF"]; compet = d[COMPET[name]]
    def au(s, idx=keep): return float(roc_auc_score(y[idx], s[idx]))
    per_lat, per_res = [], []
    resc_last = None
    for i in range(lat.shape[0]):
        per_lat.append(au(lat[i]))
        resc, nf, nfn = famwise_rescue(lat[i], aux_pcts, y)
        per_res.append(au(resc)); resc_last = (resc, nf, nfn)
    ALL[name] = dict(LatAD=(round(np.mean(per_lat), 3), round(np.std(per_lat), 3)),
                     rescued=(round(np.mean(per_res), 3), round(np.std(per_res), 3)),
                     compet=COMPET[name], compet_auroc=round(float(np.mean([au(compet if compet.ndim == 1 else compet[0])] if compet.ndim == 1 else [au(compet[i]) for i in range(compet.shape[0])])), 3),
                     fires=resc_last[1], fires_normal=resc_last[2])
    print(f"{name:5}: LatAD {ALL[name]['LatAD']}  ->  RESCUED {ALL[name]['rescued']}  "
          f"(compet {ALL[name]['compet']} {ALL[name]['compet_auroc']}; fires {resc_last[1]}, {resc_last[2]} normal)")

    # significance of rescued vs competitor (episode bootstrap), per-seed-averaged
    W, st = CFG[name]; L = int(np.ceil(W / st)) + 1
    norm = np.where(y == 0)[0]; heps = [e[hard[e]] for e in episodes(y) if hard[e].any()]
    resc_seed = np.stack([famwise_rescue(lat[i], aux_pcts, y)[0] for i in range(lat.shape[0])])
    cm = compet if compet.ndim == 2 else compet[None, :]
    def mAU(mat, idx): return float(np.mean([roc_auc_score(y[idx], mat[i][idx]) for i in range(mat.shape[0])]))
    dpt = mAU(resc_seed, keep) - mAU(cm, keep); diffs = []
    for _ in range(2000):
        nb = int(np.ceil(len(norm) / L)); ss = RNG.integers(0, max(1, len(norm) - L + 1), size=nb)
        sn = np.concatenate([norm[s:s + L] for s in ss])[:len(norm)]
        pk = RNG.integers(0, len(heps), size=len(heps)); sh = np.concatenate([heps[k] for k in pk])
        idx = np.concatenate([sn, sh])
        if y[idx].sum() < 2 or (y[idx] == 0).sum() < 2:
            continue
        diffs.append(mAU(resc_seed, idx) - mAU(cm, idx))
    diffs = np.array(diffs)
    ALL[name]["sig_vs_compet"] = dict(diff=round(float(dpt), 3),
        ci=[round(float(np.quantile(diffs, 0.025)), 3), round(float(np.quantile(diffs, 0.975)), 3)],
        p_le_0=round(float((diffs <= 0).mean()), 4))
    print(f"       rescued vs {COMPET[name]}: diff {ALL[name]['sig_vs_compet']['diff']} "
          f"CI {ALL[name]['sig_vs_compet']['ci']} P(<=0)={ALL[name]['sig_vs_compet']['p_le_0']}")

json.dump(ALL, open(f"{OUT}/cycle_combined.json", "w"), indent=1)
print(f"\nsaved -> {OUT}/cycle_combined.json")
