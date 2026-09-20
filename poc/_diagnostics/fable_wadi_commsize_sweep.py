"""Non-win D: is there a WADI-specific community configuration, selected by a TRAIN-ONLY criterion,
under which the factorized (community) model gains more on the difficult subset?

Sweeps the community-size cap MAXSZ (the expert library's only structural knob; 25 in the paper) and
rebuilds the full-head expert library locally (same recipe as sota_bundle/modal_experts.py: HAC
average-linkage on |rho| of train-normal channel means, dendrogram subtrees of size [3, MAXSZ],
per-community VaDE K=min(20,max(6,|G|)), LD=min(8,max(3,|G|//2)), 15 epochs, clip +-10, full heads).
Train-only selection criterion (label-free): split the held-out calibration slice in two halves A/B;
the fraction of B windows whose HC_coh (referenced on A) exceeds the A 99th percentile. A value near
0.01 means the aggregate is calibrated on unseen normal; larger means the library over-fires.
Test difficult-AUROC (43 windows, and the 31 windows without a train-constant-channel blow-up) is
computed for reporting only and never used for selection.
Persists per (MAXSZ, seed) to fable_wadi_commsize_sweep.jsonl (resumable); summary to
fable_wadi_commsize_sweep.json.
"""
from __future__ import annotations
import json, os, sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score
import eda_real as E
from models_vade import train_vade
from ensemble_final import surv, pval, HC

HERE = os.path.dirname(os.path.abspath(__file__))
OUTF = os.path.join(HERE, "fable_wadi_commsize_sweep.jsonl")
NAME = "WADI_clean"
SEEDS = [0, 1, 2]
SIZES = [int(s) for s in os.environ.get("SIZES", "12,25,40").split(",")]


def communities(Xn, nch, maxsz, minsz=3):
    means = Xn[:, :nch]; sdc = means.std(0); active = np.where(sdc > 1e-6)[0]
    Zact = (means[:, active] - means[:, active].mean(0)) / sdc[active]
    Cabs = np.abs(np.nan_to_num(np.corrcoef(Zact.T))); np.fill_diagonal(Cabs, 0.0)
    dist = 1 - Cabs; dist = (dist + dist.T) / 2
    L = linkage(squareform(dist, checks=False), method="average"); _, nodes = to_tree(L, rd=True)
    def leaves(n): return [n.id] if n.is_leaf() else leaves(n.left) + leaves(n.right)
    seen, comms, coh = set(), [], []
    a2p = {int(a): i for i, a in enumerate(active)}
    for n in nodes:
        if not n.is_leaf():
            lv = sorted(leaves(n))
            if minsz <= len(lv) <= maxsz and tuple(lv) not in seen:
                seen.add(tuple(lv)); G = [int(active[i]) for i in lv]; comms.append(G)
                pos = [a2p[c] for c in G]; sub = Cabs[np.ix_(pos, pos)]
                coh.append(float(sub.sum() / (len(pos) * (len(pos) - 1))))
    return comms, np.array(coh)


def main():
    D = E.load(NAME); Xn = np.asarray(D["Xn_w"], float); Xa = np.asarray(D["Xa_w"], float); y = D["ya_w"].astype(int)
    nch = len(D["ch"]); nfit = len(Xn) * 4 // 5
    d = np.load(f"{HERE}/scores_{NAME}.npz"); thr = float(d["maxz_thr"]); maxz = d["maxz"]
    diff = (y == 1) & (maxz <= thr)
    mu0, sg0 = Xn.mean(0), Xn.std(0) + 1e-8; Z = (Xa - mu0) / sg0
    const = np.where(Xn.std(0) < 1e-6)[0]; leak = np.abs(Z[:, const]).max(1) > 100
    lat = d["LatAD"]; linres = d["linres"]
    done = set()
    if os.path.exists(OUTF):
        for line in open(OUTF): r = json.loads(line); done.add((r["maxsz"], r["seed"]))
    sf = lambda Xw, S: np.nan_to_num(np.ascontiguousarray(Xw[:, [b * nch + c for b in range(6) for c in S]]))
    for maxsz in SIZES:
        comms, coh = communities(Xn, nch, maxsz); S = len(comms); w = coh * np.sqrt([len(g) for g in comms])
        for sd in SEEDS:
            if (maxsz, sd) in done: continue
            t0 = time.time(); Cal = np.zeros((S, len(Xn) - nfit)); Tst = np.zeros((S, len(Xa)))
            for gi, G in enumerate(comms):
                Ff, Fc, Ft = sf(Xn[:nfit], G), sf(Xn[nfit:], G), sf(Xa, G)
                m2, s2 = Ff.mean(0), Ff.std(0) + 1e-9
                cl = lambda A: np.clip(np.nan_to_num((A - m2) / s2, posinf=10.0, neginf=-10.0), -10, 10).astype(np.float32)
                Ztr, Zc, Zt = cl(Ff), cl(Fc), cl(Ft)
                try:
                    v = train_vade(Ztr, n_clusters=min(20, max(6, len(G))), latent_dim=min(8, max(3, len(G) // 2)),
                                   epochs=15, warmup=4, seed=sd, device="cpu")
                    v.fit_latent_density(Ztr, k_density=min(50, max(12, nfit // 12)))
                    v.fit_residual_whitener(Ztr); v.fit_resid_head(Ztr); v.fit_basin_head(Ztr)
                    cc = np.asarray(v.anomaly_score_hard(Zc, use_resid="auto", use_basin="auto"))
                    tt = np.asarray(v.anomaly_score_hard(Zt, use_resid="auto", use_basin="auto"))
                    cm, cs = cc.mean(), cc.std() + 1e-9
                    Cal[gi] = np.nan_to_num((cc - cm) / cs); Tst[gi] = np.nan_to_num((tt - cm) / cs)
                except Exception as e:
                    print(f"  comm {gi} skipped: {e}", flush=True)
            # aggregates on test, calibrated on the calib slice (train-only)
            P = np.stack([pval(Cal[g], Tst[g]) for g in range(S)]); hccoh = HC(P, wt=w)
            Pc = np.stack([pval(Cal[g], Cal[g]) for g in range(S)]); hccoh_c = HC(Pc, wt=w)
            # label-free criterion: A/B halves of the calib slice
            half = Cal.shape[1] // 2
            PA = np.stack([pval(Cal[g, :half], Cal[g, :half]) for g in range(S)]); hA = HC(PA, wt=w)
            PB = np.stack([pval(Cal[g, :half], Cal[g, half:]) for g in range(S)]); hB = HC(PB, wt=w)
            excess = float((hB > np.quantile(hA, 0.99)).mean())
            zc = lambda s, r: (s - r.mean()) / (r.std() + 1e-9)
            ltr = np.load(f"{HERE}/fable_leak_trainscores_{NAME}.npz")[f"train_{sd}"]
            tail = surv(ltr, lat[sd]); tail_ref = surv(ltr, ltr)
            fused = zc(hccoh, hccoh_c) + zc(tail, tail_ref)
            def au(s, mk):
                keep = (y == 0) | mk; return round(float(roc_auc_score(y[keep], s[keep])), 3)
            row = dict(maxsz=maxsz, seed=sd, n_comm=S, mean_size=round(float(np.mean([len(g) for g in comms])), 1),
                       criterion_excess_fpr_B_at_A99=round(excess, 4),
                       HC_coh_diff43=au(hccoh, diff), HC_coh_diff31=au(hccoh, diff & ~leak),
                       fused_diff43=au(fused, diff), fused_diff31=au(fused, diff & ~leak),
                       linres_diff43=au(linres, diff), linres_diff31=au(linres, diff & ~leak), secs=round(time.time() - t0))
            with open(OUTF, "a") as f: f.write(json.dumps(row) + "\n")
            print(row, flush=True)
    rows = [json.loads(l) for l in open(OUTF)]
    summ = {}
    for maxsz in sorted({r["maxsz"] for r in rows}):
        rr = [r for r in rows if r["maxsz"] == maxsz]
        summ[maxsz] = {k: [round(float(np.mean([r[k] for r in rr])), 3), round(float(np.std([r[k] for r in rr])), 3)]
                       for k in ("criterion_excess_fpr_B_at_A99", "HC_coh_diff43", "HC_coh_diff31", "fused_diff43", "fused_diff31")}
        summ[maxsz]["n_comm"] = rr[0]["n_comm"]
    json.dump(summ, open(os.path.join(HERE, "fable_wadi_commsize_sweep.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1)); print("DONE", flush=True)


if __name__ == "__main__":
    main()
