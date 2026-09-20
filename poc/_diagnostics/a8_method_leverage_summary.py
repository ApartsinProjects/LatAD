"""Markdown tables from a8_method_leverage_eval*.json (read-only summariser)."""
import json, os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for f in sorted(glob.glob(os.path.join(HERE, "a8_method_leverage_eval*.json"))):
    j = json.load(open(f)); R.update({k: v for k, v in j.items() if k != "_dumps"}); R.setdefault("_dumps", {}).update(j.get("_dumps", {}))
DS = [d for d in ("WADI_clean", "HAI", "SWaT_canon") if d in R]
SUB = ["Difficult", "DoubleHard", "Easy", "All"]
def cell(r, d, s):
    v = R[d]["rows"].get(r); return "n/a" if v is None else f"{v[s][0]:.3f}±{v[s][1]:.3f}"
def table(rows, subs=("Difficult", "DoubleHard"), title=None):
    if title: print(f"\n**{title}**\n")
    print("| score | " + " | ".join(f"{d} {s}" for d in DS for s in subs) + " |")
    print("|---|" + "---|" * (len(DS) * len(subs)))
    for r in rows:
        print(f"| {r} | " + " | ".join(cell(r, d, s) for d in DS for s in subs) + " |")
base = ["LatAD", "LatAD_refit", "HC_coh", "HCcoh+LatAD"]
table(base, SUB, "Bases (clean subsets, 5 seeds)")
prim_B = ["knn_lat_k10", "knn_feat_k10", "knn_pca20_k10"]
prim_A = ["vade_exp_prev1", "vade_exp_selfgate", "vade_trans_prev1", "obs_exp_prev1", "obs_exp_selfgate", "obs_trans_prev1"]
ctl = ["vade_exp_ctl_rand", "vade_exp_ctl_shuf", "obs_exp_ctl_rand", "obs_exp_ctl_shuf", "vade_switch_prev1", "obs_switch_prev1", "vade_exp_oracle", "obs_exp_oracle", "vade_trans_oracle", "obs_trans_oracle"]
comb = ["knn_lat_k10+vade_exp_prev1", "knn_lat_k10+obs_exp_prev1"]
for grp, nm in ((prim_B, "Lever B: kNN heads"), (prim_A, "Lever A: context heads (label-free)"), (ctl, "Controls and label-using oracle (reference only)"), (comb, "Combined heads")):
    rows = []
    for h in grp:
        rows += [f"head:{h}", f"LatAD+{h}", f"HCcoh+{h}", f"HCcoh+LatAD+{h}"]
    table(rows, ("Difficult", "DoubleHard"), nm)
table([f"head:knn_lat_k{k}" for k in (1, 5, 10, 20, 50)] + [f"HCcoh+knn_lat_k{k}" for k in (1, 5, 10, 20, 50)], ("Difficult", "DoubleHard"), "kNN order sensitivity (latent)")
mx = []
for h in ("knn_lat_k10", "knn_feat_k10", "vade_exp_prev1", "obs_exp_prev1", "vade_trans_prev1", "obs_trans_prev1", "obs_exp_oracle", "obs_exp_ctl_shuf"):
    mx += [f"zmax(HCcoh+LatAD,{h})", f"pmax(HCcoh+LatAD,{h})", f"zmax(HC_coh,{h})", f"pmax(HC_coh,{h})", f"pmax(LatAD,{h})"]
table(mx, ("Difficult", "DoubleHard"), "Nonlinear (max) fusions as in the A8 study")
table([f"head:{h}" for h in prim_B + prim_A], ("Easy", "All"), "Head-alone on Easy / All (invariant: kNN must catch isolated anomalies)")
print("\n**Paired episode bootstrap, fused vs base (diff, 95% CI, P(diff<=0))**\n")
print("| head | " + " | ".join(f"{d} {c} {s}" for d in DS for c in ("+headline", "+HC_coh") for s in ("Diff", "DH")) + " |")
print("|---|" + "---|" * (len(DS) * 4))
for h in R[DS[0]]["boots"]:
    row = []
    for d in DS:
        b = R[d]["boots"].get(h, {})
        for c in ("HCcoh+LatAD+h vs HCcoh+LatAD", "HCcoh+h vs HC_coh"):
            for s in ("Difficult", "DoubleHard"):
                v = b.get(c, {}).get(s)
                row.append("n/a" if not v else f"{v['diff']:+.3f} [{v['diff_ci'][0]:+.2f},{v['diff_ci'][1]:+.2f}] P={v['p_le_0']:.2f}")
    print(f"| {h} | " + " | ".join(row) + " |")
print("\n**Train-normal gates (held-out-20% q95 / in-sample q95; residual-head rule: ON if < 1.5) and diagnostics**\n")
print("| dataset | knn_lat | knn_feat | knn_pca20 | vade_exp | vade_trans | obs_exp | obs_trans | resid (shipped gate) | over-coverage share (train) | Spearman(dens,kNN) train | normal switch rate test vade/obs | anomaly switch rate vade/obs |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in DS:
    g = R[d]["gates"]; dg = R[d]["diag"]
    f = lambda k: f"{np.mean(g[k]):.2f}±{np.std(g[k]):.2f}" if isinstance(g[k], list) else f"{g[k]:.2f}"
    cv = dg["ctx_vade"]; co = dg["ctx_obs"]
    print(f"| {d} | {f('knn_lat')} | {f('knn_feat')} | {f('knn_pca20')} | {f('vade_exp')} | {f('vade_trans')} | {f('obs_exp')} | {f('obs_trans')} | {f('resid_gen_ratio')} | "
          f"{np.mean(dg['overcoverage_train']):.3f} | {np.mean(dg['spearman_dens_knn_train']):.2f} | "
          f"{np.mean([c['normal_switch_rate_test'] for c in cv]):.2f}/{np.mean([c['normal_switch_rate_test'] for c in co]):.2f} | "
          f"{np.mean([c['anom_switch_rate_test'] for c in cv]):.2f}/{np.mean([c['anom_switch_rate_test'] for c in co]):.2f} |")
print("\n**Refit check vs stored scores (corr / AUROC all, per seed)**\n")
for d in DS:
    print(f"- {d}: " + "; ".join(f"s{i} r={c['corr_with_shipped']:.3f} {c['auroc_shipped']:.3f}->{c['auroc_refit']:.3f}" for i, c in enumerate(R[d]["diag"]["refit_check"])))
if "WADI_clean" in R.get("_dumps", {}):
    print("\n**WADI clean difficult windows: seed-mean robust-z (train-normal scale) and its percentile among test-normal windows**\n")
    cols = ["headline_z", "hc_coh_z", "latad_z", "knn_lat_k10", "knn_feat_k10", "vade_exp_prev1", "obs_exp_prev1", "obs_exp_oracle", "obs_trans_prev1"]
    print("| win | DH | " + " | ".join(cols) + " |"); print("|---|---|" + "---|" * len(cols))
    for r in R["_dumps"]["WADI_clean"]:
        print(f"| {r['win']} | {'y' if r['double_hard'] else ''} | " + " | ".join(f"{r[c]:+.1f} ({r[c+'_pctN']:.2f})" for c in cols) + " |")
