"""Summarise a8_valley_real.jsonl: per-partition detection, interaction, enrichment, magnitude-matched, valley normals."""
import json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "a8_valley_real.jsonl"), encoding="utf-8")]
rows = [r for r in rows if r["kind"] == "main"]
DETS = ["LatAD", "USAD", "TranAD", "AE", "IF", "linres", "maxz"]
PARTS = ["in_cluster", "valley", "valley_distinct", "beyond", "off", "all"]
f = lambda x: "  nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:5.2f}"
for ds in ["SWaT_canon", "WADI_clean", "HAI"]:
    rs = [r for r in rows if r["dataset"] == ds]
    if not rs: continue
    print(f"\n===== {ds}: seeds {[r['seed'] for r in rs]}  K={[r['K'] for r in rs]}  R={[round(r['R'],2) for r in rs]}  n_norm={rs[0]['n_norm']} n_anom={rs[0]['n_anom']}")
    print("  share of normals by part (seed mean):", {p: round(np.mean([r['share_norm'][p] for r in rs]), 4) for p in ["in_cluster", "valley", "beyond", "off"]})
    print("  share of anomalies by part (seed mean):", {p: round(np.mean([r['share_anom'][p] for r in rs]), 4) for p in ["in_cluster", "valley", "valley_distinct", "beyond", "off"]})
    print(f"  {'part':16s} {'n(seeds)':>14s} {'dh_med':>7s} | " + " ".join(f"{d:>12s}" for d in DETS) + "   [AUROC vs all normals / recall@1%FPR]")
    for p in PARTS:
        ns = [r["detection"][p]["n"] for r in rs]
        line = f"  {p:16s} {str(ns):>14s} {np.nanmean([r['detection'][p]['dh_median'] for r in rs]):7.2f} | "
        for d in DETS:
            au = np.nanmean([r["detection"][p][d]["auroc"] for r in rs]); rc = np.nanmean([r["detection"][p][d]["rec99"] for r in rs])
            line += f" {f(au)}/{f(rc)}"
        print(line)
    print("  interaction (LatAD-deep)[valley] - (LatAD-deep)[off], per seed:")
    for deep in ["USAD", "TranAD", "AE"]:
        for r in rs:
            i = r["interaction"][deep]
            if "obs" in i: print(f"    {deep:7s} s{r['seed']} n_v={i['n_valley']:3d} n_o={i['n_off']:3d} obs={i['obs']:+.3f} ci=[{i['ci'][0]:+.3f},{i['ci'][1]:+.3f}] perm-null sd={i['null_sd']:.3f} p={i['p_perm']:.3f}")
            else: print(f"    {deep:7s} s{r['seed']} n_v={i['n_valley']} n_o={i['n_off']} (too few)")
    print("  miss enrichment: G = deep misses (<= q99 normal) AND LatAD catches (> q99 normal), among anomalies")
    for deep in ["USAD", "TranAD", "AE"]:
        for r in rs:
            e = r["enrichment"][deep]
            v = e["valley"]; o = e["off"]; ic = e["in_cluster"]
            print(f"    {deep:7s} s{r['seed']} |G|={e['n_G']:3d} |notG|={e['n_notG']:3d} |reverse|={e['n_reverse']:3d}  valley share G/out/all={v['share_in_G']:.3f}/{v['share_outside_G']:.3f}/{v['share_all_anom']:.3f} enr={v['enrichment']:.2f} p={v['fisher_p']:.3f} | off share G/out={o['share_in_G']:.3f}/{o['share_outside_G']:.3f} p={o['fisher_p']:.3f} | in_cluster G/out={ic['share_in_G']:.3f}/{ic['share_outside_G']:.3f} p={ic['fisher_p']:.3f}")
    print("  magnitude-matched (terciles of dh among out-of-envelope anomalies): AUROC LatAD/USAD/TranAD")
    for r in rs:
        for k, b in r["magnitude_matched"].items():
            s = f"    s{r['seed']} {k} dh in [{b['dh_range'][0]:.1f},{b['dh_range'][1]:.1f}]: "
            for p in ["valley", "off", "beyond"]:
                s += f" {p} n={b[p]['n']:3d} {f(b[p]['LatAD'])}/{f(b[p]['USAD'])}/{f(b[p]['TranAD'])} |"
            print(s)
    print("  FPR at global q99 on NORMAL windows by part (seed mean): ")
    for d in DETS:
        print(f"    {d:7s} " + " ".join(f"{p}={np.nanmean([r['fpr_normals_by_part'][d][p] for r in rs]):.3f}" for p in ["in_cluster", "valley", "beyond", "off"]) + f"   n_norm_valley={[r['n_norm_valley'] for r in rs]}")
