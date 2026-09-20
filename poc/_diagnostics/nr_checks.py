"""Validation checks for nr_discovery: lost anomalies, non-block members, absorption test, recurrent
regime-19 candidate, rule-threshold sensitivity. Appends results to nr_discovery.json under 'checks'."""
import json, warnings, numpy as np
warnings.filterwarnings("ignore")
ROOT = "E:/Projects/Backlog/LatAD/poc"
J = json.load(open(f"{ROOT}/_diagnostics/nr_discovery.json"))
M = np.load(f"{ROOT}/_diagnostics/nr_members.npz")
CH = {}


def runs(idx, gap=2):
    idx = np.sort(idx)
    if len(idx) == 0: return []
    br = np.flatnonzero(np.diff(idx) > gap + 1)
    s = np.r_[idx[0], idx[br + 1]]; e = np.r_[idx[br], idx[-1]]
    return [(int(a), int(b)) for a, b in zip(s, e)]


def attack_runs(y):
    return runs(np.flatnonzero(y == 1), gap=0)


ds = "HAI"
S = np.load(f"{ROOT}/_diagnostics/scores_{ds}.npz"); F = np.load(f"{ROOT}/_diagnostics/e2_fable_{ds}.npz")
y = S["label"].astype(int); zte, ztr = F["zte"].astype(float), F["ztr"].astype(float)
lat = S["LatAD"].mean(0)
ar = attack_runs(y)
print(f"HAI: {len(ar)} attack runs; attack-run bounds near block: {[r for r in ar if 6300 <= r[0] <= 7200 or 6300 <= r[1] <= 7200]}")

for op in ["train99_mix", "latad_fpr5"]:
    fi, lab = M[f"{ds}_{op}_fi"], M[f"{ds}_{op}_lab"]
    nn_cl = [c["cluster"] for c in J[ds]["operating_points"][op]["clusters"] if c["verdict"] == "NEW-NORMAL"]
    idx = np.sort(fi[np.isin(lab, nn_cl)])
    lost = idx[y[idx] == 1]
    # each lost anomaly: its attack run, and distance (windows) to nearest y0 window
    info = []
    for t in lost:
        r = [a for a in ar if a[0] <= t <= a[1]][0]
        info.append(dict(t=int(t), run=r, run_len=r[1] - r[0] + 1, pos_in_run=int(t - r[0]), latad=float(lat[t])))
    print(f"\n[{op}] NEW-NORMAL cluster(s) {nn_cl}: {len(idx)} windows, blocks={runs(idx)}")
    print(f"  lost anomalies ({len(lost)}): " + "; ".join(f"t={d['t']} run{d['run']} len{d['run_len']} pos{d['pos_in_run']}" for d in info))
    n_boundary = sum(1 for d in info if d["pos_in_run"] == 0 or d["pos_in_run"] == d["run_len"] - 1)
    inside_block = int(((lost >= 6436) & (lost <= 7081)).sum())
    print(f"  lost at attack-run boundary (first/last window): {n_boundary}/{len(lost)}; lost inside 6436..7081: {inside_block}")
    # non-block members
    nb = idx[(idx < 6436) | (idx > 7081)]
    print(f"  non-block members: {len(nb)} (y1={int(y[nb].sum())}); blocks={runs(nb)}")
    # absorption test: diag Gaussian on the cluster's y0 members' latent (label-free: use all members)
    mu, sd = zte[idx].mean(0), zte[idx].std(0) + 1e-6
    maha = np.sqrt((((zte - mu) / sd) ** 2).sum(1))
    r99 = np.quantile(maha[idx], 0.99)
    absorbed = np.flatnonzero(maha <= r99)
    abs_out = np.setdiff1d(absorbed, idx)
    print(f"  absorption (maha<=cluster p99={r99:.2f}): {len(absorbed)} test windows total; outside cluster {len(abs_out)} (y1={int(y[abs_out].sum())} of {int((y==1).sum())} anomalies; y0={int((y[abs_out]==0).sum())})")
    # what LatAD score did absorbed-outside anomalies have (were they detected at all?)
    a1 = abs_out[y[abs_out] == 1]
    thr = J[ds]["operating_points"][op]["threshold"]
    print(f"  absorbed outside-cluster anomalies flagged at this op: {int((lat[a1] > thr).sum()) if op != 'train99_mix' else 'n/a(mix score)'}")
    CH[f"{ds}_{op}"] = dict(newnormal_clusters=nn_cl, n=int(len(idx)), blocks=runs(idx), lost=info, lost_boundary=n_boundary,
                            lost_inside_block=inside_block, nonblock_n=int(len(nb)), nonblock_y1=int(y[nb].sum()), nonblock_blocks=runs(nb),
                            absorb_r99=float(r99), absorbed_total=int(len(absorbed)), absorbed_outside=int(len(abs_out)),
                            absorbed_outside_y1=int(y[abs_out].sum()), absorbed_outside_y0=int((y[abs_out] == 0).sum()))

# recurrent regime-19 candidate at train99 (cluster with regime 19, ~186 windows)
fi, lab = M["HAI_train99_mix_fi"], M["HAI_train99_mix_lab"]
cl = J["HAI"]["operating_points"]["train99_mix"]["clusters"]
c19 = [c for c in cl if c["regime_argmax_top"] == 19 and c["size"] > 100][0]
idx19 = np.sort(fi[lab == c19["cluster"]])
rr = runs(idx19)
print(f"\nrecurrent candidate (cluster {c19['cluster']}, regime 19): n={len(idx19)} y1={int(y[idx19].sum())} blocks={len(rr)} block lens={sorted([b-a+1 for a,b in rr], reverse=True)[:12]}")
gaps = np.diff([a for a, _ in rr])
print(f"  inter-block start gaps: median={np.median(gaps):.0f} windows, IQR={np.quantile(gaps,.25):.0f}..{np.quantile(gaps,.75):.0f}; regime-19 train share={np.mean(np.argmax(F['logpi'][None]+F['logN_tr'],1)==19):.4f}")
nn_idx = np.sort(fi[np.isin(lab, [c["cluster"] for c in cl if c["verdict"] == "NEW-NORMAL"])])
thr = J["HAI"]["operating_points"]["train99_mix"]["threshold"]
from scipy.special import logsumexp
s_te = -logsumexp(F["logpi"][None] + F["logN_te"], 1)
flag = s_te > thr; f2 = flag.copy(); f2[nn_idx] = False; f3 = f2.copy(); f3[idx19] = False
print(f"  FPR train99: base {((y==0)&flag).sum()/(y==0).sum():.4f} -> fold block {((y==0)&f2).sum()/(y==0).sum():.4f} -> also fold regime-19 cluster {((y==0)&f3).sum()/(y==0).sum():.4f}; TPR {((y==1)&flag).sum()/(y==1).sum():.3f}/{((y==1)&f2).sum()/(y==1).sum():.3f}/{((y==1)&f3).sum()/(y==1).sum():.3f}")
CH["HAI_regime19_recurrent"] = dict(cluster=c19["cluster"], n=int(len(idx19)), y1=int(y[idx19].sum()), n_blocks=len(rr),
                                    fpr_after_also_folding=float(((y == 0) & f3).sum() / (y == 0).sum()), tpr_after=float(((y == 1) & f3).sum() / (y == 1).sum()))

# residual false alarms after folding: where are they?
res = np.flatnonzero(f2 & (y == 0))
rr = runs(res)
print(f"  residual y0 flags after fold: {len(res)} in {len(rr)} blocks; longest {max(b-a+1 for a,b in rr)}; blocks>=10: {sum(b-a+1>=10 for a,b in rr)}; in regime-19 cluster {int(np.isin(res, idx19).sum())}; in noise {int(np.isin(res, fi[lab==-1]).sum())}")

# rule sensitivity: verdict flips over threshold grid, all datasets/ops (label-free features from JSON)
print("\nrule sensitivity (clusters whose verdict changes vs base rule RUN>=30, COH<=2, SPE<=3):")
sens = {}
for RUN in (15, 20, 30, 50, 100):
    for COH in (1.5, 2.0, 3.0):
        for SPE in (2.0, 3.0, 5.0):
            flips = []
            for d in ("HAI", "WADI", "SWaT"):
                for op, O in J[d]["operating_points"].items():
                    for c in O.get("clusters", []):
                        if c["cluster"] == -1: continue
                        v = "NEW-NORMAL" if (c["longest_run"] >= RUN and c["cohesion_ratio"] <= COH and c["spe_ratio"] <= SPE) else "ANOMALY"
                        if v != c["verdict"]:
                            flips.append(f"{d}/{op}/c{c['cluster']}(n={c['size']},y1={100*c['frac_y1']:.0f}%)->{v}")
            sens[f"RUN{RUN}_COH{COH}_SPE{SPE}"] = flips
            if flips: print(f"  RUN>={RUN} COH<={COH} SPE<={SPE}: {flips}")
print("  (no line printed for a setting = identical verdicts)")
CH["rule_sensitivity"] = sens

# SPE-only and temporal-only single-criterion views on all non-noise clusters (label check)
print("\nsingle-criterion label check over all non-noise clusters (n>=20):")
rows = [(d, op, c) for d in ("HAI", "WADI", "SWaT") for op, O in J[d]["operating_points"].items() for c in O.get("clusters", []) if c["cluster"] != -1 and c["size"] >= 20]
for name, fn in [("SPE<=3", lambda c: c["spe_ratio"] <= 3), ("run>=30", lambda c: c["longest_run"] >= 30), ("coh<=2", lambda c: c["cohesion_ratio"] <= 2),
                 ("SPE<=3&run>=30", lambda c: c["spe_ratio"] <= 3 and c["longest_run"] >= 30)]:
    pos = [c for _, _, c in rows if fn(c)]; neg = [c for _, _, c in rows if not fn(c)]
    print(f"  {name:>15}: passes {len(pos)} clusters (mean y1 {np.mean([c['frac_y1'] for c in pos]) if pos else float('nan'):.2f}), fails {len(neg)} (mean y1 {np.mean([c['frac_y1'] for c in neg]) if neg else float('nan'):.2f})")

J["checks"] = CH
json.dump(J, open(f"{ROOT}/_diagnostics/nr_discovery.json", "w"), indent=1, default=lambda o: None if isinstance(o, float) and np.isnan(o) else (o.tolist() if hasattr(o, "tolist") else str(o)))
print("\nchecks appended to nr_discovery.json")
