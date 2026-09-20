"""Community-level train-only stationarity gate (the Route-2 gate applied to the community experts).
For each community and seed: FP = fraction of the chronologically-held-out CALIBRATION slice (last 20% of train)
whose expert surprise exceeds the in-sample FIT-slice 99th percentile (both standardised on the calibration
slice, same transform). Gate: drop a community when its seed-mean FP exceeds the a-priori budget (5%, the same
budget as the boosted-expert gate; 1% and 10% reported too). No test information. Evaluates HCcoh+LatAD with all
vs gated communities on Difficult / DoubleHard, with paired bootstraps (gated minus ungated, gated minus boosted).
Usage: python comm_gate_eval.py <ds> [tag=cur_fit]  ->  prints + appends to comm_gate_eval.json"""
from __future__ import annotations
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from sklearn.metrics import roc_auc_score
import eda_real as E
from onehot_filter import build_feats, loco_residual
import ensemble_final as EF

HERE = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "cur_fit"
Ex = np.load(os.path.join(ROOT, "sota_bundle", "experts_variants", tag, f"expert_{name}.npz"), allow_pickle=True)
Cal, Tst, Fit = Ex["calib_surprise"], Ex["test_surprise"], Ex["fit_surprise"]; nseed, S, _ = Tst.shape
d = np.load(f"{EF.OUT}/scores_{name}.npz"); y = d["label"].astype(int)
D = E.load(name); ch = D["ch"]; fn, W, stride = E.RAW[name]
Fn, Fa, grp = build_feats(np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float), W, stride, onehot=True); Fa = Fa[:len(y)]
r_tr, r_te = loco_residual(Fn, Fa, grp); diff = (y == 1) & (d["maxz"] <= float(d["maxz_thr"])); dhard = diff & (r_te <= np.quantile(r_tr, 0.99)); nrm = y == 0
comm = [[int(c) for c in row if c >= 0] for row in Ex["comm_channels"]]
coh = Ex["comm_cohesion"].astype(float); size = Ex["comm_size"].astype(float); w = coh * np.sqrt(size)
lat, lat_tr = d["LatAD"], d["LatAD_train"]; z = lambda s, r: (s - r.mean()) / (r.std() + 1e-9)
FP = np.zeros((nseed, S))
for sd in range(nseed):
    for g in range(S):
        q = np.quantile(Fit[sd, g], 0.99); FP[sd, g] = (Cal[sd, g] > q).mean()
fp = FP.mean(0)
degen = (Cal.std(2) < 1e-3).any(0)
# also the actual test-normal floor rate per community (diagnostic only, NOT used by the gate)
floor_n = np.array([np.mean([(EF.pval(Cal[sd, g], Tst[sd, g])[nrm] <= 1e-4 + 1e-9).mean() for sd in range(nseed)]) for g in range(S)])


def fuse(ks):
    acc = []
    for sd in range(nseed):
        P = np.stack([EF.pval(Cal[sd, g], Tst[sd, g]) for g in ks]); Pc = np.stack([EF.pval(Cal[sd, g], Cal[sd, g]) for g in ks])
        zl = z(EF.surv(lat_tr[sd], lat[sd]), EF.surv(lat_tr[sd], lat_tr[sd]))
        acc.append(z(EF.HC(P, wt=w[ks]), EF.HC(Pc, wt=w[ks])) + zl)
    return np.stack(acc)


def au_ms(arr, m):
    k = (y == 0) | m
    return float(np.mean([roc_auc_score(y[k], arr[i][k]) for i in range(arr.shape[0])])) if arr.ndim == 2 else float(roc_auc_score(y[k], arr[k]))


L = int(np.ceil(W / stride)) + 1
res = dict(name=name, S=int(S), fp_per_comm=[(int(g), [ch[c] for c in comm[g]][:6], round(float(fp[g]), 3), round(float(floor_n[g]), 3), bool(degen[g])) for g in np.argsort(-fp)])
print(f"=== {name} ({tag}): S={S}; held-out-train FP at fit-p99 per community (top 8), with test-normal floor rate (diagnostic) ===")
for r in res["fp_per_comm"][:8]:
    print("  ", r)
print("  corr(train-holdout FP, test-normal floor rate) =", round(float(np.corrcoef(fp, floor_n)[0, 1]), 3))
res["corr_fp_floor"] = round(float(np.corrcoef(fp, floor_n)[0, 1]), 3)
full = fuse(np.arange(S)); res["all"] = dict(Difficult=round(au_ms(full, diff), 3), DoubleHard=round(au_ms(full, dhard), 3))
print("  all communities:", res["all"])
bpath = os.path.join(HERE, f"boosted_loo_{name}.npz"); b_te = np.load(bpath)["b_te"] if os.path.exists(bpath) else None
for budget in (0.01, 0.05, 0.10):
    keep = np.where(fp <= budget)[0]
    if len(keep) < 3:
        res[f"budget_{budget}"] = dict(n_keep=int(len(keep)), note="fewer than 3 communities kept"); print(f"  budget {budget}: keeps {len(keep)}, skipped"); continue
    g = fuse(keep)
    r = dict(n_keep=int(len(keep)), dropped=[[ch[c] for c in comm[k]][:4] for k in range(S) if fp[k] > budget],
             Difficult=round(au_ms(g, diff), 3), DoubleHard=round(au_ms(g, dhard), 3))
    EF.RNG = np.random.default_rng(0); r["boot_gated_minus_all_Difficult"] = EF.boot(y, g, full, diff, L, reps=2000)
    EF.RNG = np.random.default_rng(0); r["boot_gated_minus_all_DoubleHard"] = EF.boot(y, g, full, dhard, L, reps=2000)
    if b_te is not None:
        EF.RNG = np.random.default_rng(0); r["boot_gated_minus_boosted_Difficult"] = EF.boot(y, g, b_te, diff, L, reps=2000)
        EF.RNG = np.random.default_rng(0); r["boot_gated_minus_boosted_DoubleHard"] = EF.boot(y, g, b_te, dhard, L, reps=2000)
    res[f"budget_{budget}"] = r
    print(f"  budget {budget}: keep {len(keep)}/{S}, dropped {r['dropped']}; Difficult {r['Difficult']} DoubleHard {r['DoubleHard']}")
    print(f"     gated-all: diff {r['boot_gated_minus_all_Difficult']}  dhard {r['boot_gated_minus_all_DoubleHard']}")
    if b_te is not None:
        print(f"     gated-boosted: diff {r['boot_gated_minus_boosted_Difficult']}  dhard {r['boot_gated_minus_boosted_DoubleHard']}")
outp = os.path.join(HERE, "comm_gate_eval.json"); allr = json.load(open(outp)) if os.path.exists(outp) else {}
allr[name] = res; json.dump(allr, open(outp, "w"), indent=1)
