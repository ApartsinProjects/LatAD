"""Auto-gated common-mode-block-level head, multi-seed + significance.

Head: correlation communities auto-discovered from train-normal (|r|>0.7, size>=3, on the
first half A of train-normal window-means); per-community common factor = PC1 on A; window
score = calibrated max over communities of the two-sided common-mode tail |z_common|.

Rescue fusion (train-normal-only calibration; a no-op where the head does not fire, so it
cannot reshuffle LatAD's good ranking):
    S = z(LatAD) + max(0, head - tau),   tau = q-th percentile of the head on HELD-OUT
train-normal (the second half B). tau is pre-registered at the 99.5th percentile; a sweep
{99, 99.5, 99.9} is reported for transparency. lambda = 1 (heads are on the same z scale).

Per seed: difficult-subset AUROC of LatAD and the fused score -> mean+-std over 5 seeds.
WADI: episode-block bootstrap of the paired (fused - IF) difference at the pre-registered tau.
Output -> _diagnostics/mine_head.json.
"""
from __future__ import annotations
import json, os, numpy as np
from numpy.linalg import svd
from sklearn.metrics import roc_auc_score
import eda_real as E

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
THR, MINSZ = 0.7, 3
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}
RNG = np.random.default_rng(0)
TAUS = {"q99": 99.0, "q99.5": 99.5, "q99.9": 99.9}
PRE = "q99.5"     # pre-registered threshold


def build_head(Mn, Ma):
    mu, sg = Mn.mean(0), Mn.std(0) + 1e-9
    Zn, Za = (Mn - mu) / sg, (Ma - mu) / sg
    nA = len(Zn) // 2; A, B = Zn[:nA], Zn[nA:]
    C = np.corrcoef(A.T); np.fill_diagonal(C, 0.0); adj = np.abs(C) > THR
    seen = np.zeros(Zn.shape[1], bool); groups = []
    for i in range(Zn.shape[1]):
        if seen[i]:
            continue
        st = [i]; comp = []
        while st:
            v = st.pop()
            if seen[v]:
                continue
            seen[v] = True; comp.append(v); st += list(np.where(adj[v])[0])
        if len(comp) >= MINSZ:
            groups.append(sorted(comp))
    if not groups:
        return None, None, 0
    us = []
    for G in groups:
        U, S, Vt = svd(A[:, G] - A[:, G].mean(0), full_matrices=False)
        u = Vt[0] * np.sign(Vt[0].sum()); ga = A[:, G] @ u
        us.append((u, ga.mean(), ga.std() + 1e-9))
    def score(Z):
        return np.max([np.abs((Z[:, G] @ u - m) / s) for G, (u, m, s) in zip(groups, us)], axis=0)
    head_B = score(B)                         # held-out-normal head, for tau + z-calibration
    hm, hs = head_B.mean(), head_B.std() + 1e-9
    head_a = (score(Za) - hm) / hs
    tau = {k: float((np.percentile(head_B, p) - hm) / hs) for k, p in TAUS.items()}
    return head_a, tau, len(groups)


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
    W, stride = CFG[name]
    D = E.load(name); nch = len(D["ch"])
    Mn = np.asarray(D["Xn_w"], float)[:, :nch]; Ma = np.asarray(D["Xa_w"], float)[:, :nch]
    y = np.asarray(D["ya_w"], int)
    d = np.load(f"{OUT}/scores_{name}.npz"); thr = float(d["maxz_thr"])
    hard = (y == 1) & ~((y == 1) & (d["maxz"] > thr))
    keep = np.where((y == 0) | hard)[0]
    IFs = d["IF"].mean(0)
    head_a, tau, ng = build_head(Mn, Ma)

    def auroc(s):
        return float(roc_auc_score(y[keep], s[keep]))

    lat = d["LatAD"]                                    # (5, n)
    per_lat = [auroc(lat[i]) for i in range(lat.shape[0])]
    fused_by_tau = {}
    for tk, tv in tau.items():
        rescue = np.maximum(0.0, head_a - tv)
        per = []
        for i in range(lat.shape[0]):
            zl = (lat[i] - lat[i].mean()) / lat[i].std()
            per.append(auroc(zl + rescue))
        fused_by_tau[tk] = (round(float(np.mean(per)), 3), round(float(np.std(per)), 3))
    ALL[name] = dict(n_groups=ng,
                     LatAD=(round(float(np.mean(per_lat)), 3), round(float(np.std(per_lat)), 3)),
                     IF=round(float(np.mean([auroc(d["IF"][i]) for i in range(d["IF"].shape[0])])), 3),
                     fused=fused_by_tau)
    print(f"{name:5} (groups {ng}): LatAD {ALL[name]['LatAD']}  IF {ALL[name]['IF']}  "
          + "  ".join(f"fused[{k}] {v}" for k, v in fused_by_tau.items()))

    # WADI significance at the pre-registered tau: paired episode bootstrap (fused - IF)
    if name == "WADI":
        rescue = np.maximum(0.0, head_a - tau[PRE])
        norm = np.where(y == 0)[0]; heps = [e[hard[e]] for e in episodes(y) if hard[e].any()]
        L = int(np.ceil(W / stride)) + 1
        # point estimate: mean over seeds of fused AUROC and IF AUROC
        fused_seed = []
        for i in range(lat.shape[0]):
            zl = (lat[i] - lat[i].mean()) / lat[i].std(); fused_seed.append(zl + rescue)
        fused_seed = np.stack(fused_seed)
        ifm = d["IF"]
        def mAU(mat, idx):
            return float(np.mean([roc_auc_score(y[idx], mat[i][idx]) for i in range(mat.shape[0])]))
        diff_pt = mAU(fused_seed, keep) - mAU(ifm, keep)
        diffs = []
        for _ in range(2000):
            nb = int(np.ceil(len(norm) / L))
            st = RNG.integers(0, max(1, len(norm) - L + 1), size=nb)
            sn = np.concatenate([norm[s:s + L] for s in st])[:len(norm)]
            pick = RNG.integers(0, len(heps), size=len(heps)); sh = np.concatenate([heps[k] for k in pick])
            idx = np.concatenate([sn, sh])
            if y[idx].sum() < 2 or (y[idx] == 0).sum() < 2:
                continue
            diffs.append(mAU(fused_seed, idx) - mAU(ifm, idx))
        diffs = np.array(diffs)
        ALL[name]["wadi_sig_vs_IF"] = dict(
            tau=PRE, diff=round(float(diff_pt), 3),
            ci=[round(float(np.quantile(diffs, 0.025)), 3), round(float(np.quantile(diffs, 0.975)), 3)],
            p_le_0=round(float((diffs <= 0).mean()), 4))
        s = ALL[name]["wadi_sig_vs_IF"]
        print(f"      WADI fused vs IF @ {PRE}: diff {s['diff']} CI {s['ci']} P(<=0)={s['p_le_0']}")

json.dump(ALL, open(f"{OUT}/mine_head.json", "w"), indent=1)
print(f"\nsaved -> {OUT}/mine_head.json")
