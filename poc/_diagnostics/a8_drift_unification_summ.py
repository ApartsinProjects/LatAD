"""Seed-mean (min-max) tables from a8_drift_unification.json for the .md."""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "a8_drift_unification.json")))
PARTS = ["in_cluster", "valley", "beyond", "off"]


def agg(vals, fmt="{:.3f}"):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v: return "-"
    if len(v) == 1: return fmt.format(v[0])
    return (fmt + " [" + fmt + ", " + fmt + "]").format(np.mean(v), min(v), max(v))


def get(seeds, path):
    out = []
    for s in seeds.values():
        x = s
        try:
            for p in path: x = x[p]
        except (KeyError, TypeError, IndexError): x = None
        out.append(x)
    return out


for ds, r in J.items():
    S = r["seeds"]; print(f"\n===== {ds}: n={r['n']} attacks={r['n_attacks']} difficult={r['n_difficult']} test-normal drifted={r['test_normal_drifted']:.3f} drifting comms={r['drifting_communities']} seeds={list(S)}")
    print("Q1 shares (seed mean [min,max]):")
    for who in ("normals", "attacks", "difficult"):
        print(f"  {who:9s} " + " | ".join(f"{p} {agg(get(S, ['share', who, p]))}" for p in PARTS))
    print("  normals by third (valley / off / in_cluster / drifted):")
    for k in range(3):
        print(f"    T{k+1}: " + " / ".join(agg(get(S, ["share_normals_by_third", k, p])) for p in ["valley", "off", "in_cluster"]) + " / " + agg(get(S, ["drifted_normals_by_third", k])))
    print("Q2a normals partition x DRIFTED: part: n, drifted share, DR pct median")
    for p in PARTS:
        print(f"  {p:10s} n {agg(get(S, ['normals_part_x_drifted', p, 'n']), '{:.0f}')}  drifted {agg(get(S, ['normals_part_x_drifted', p, 'drifted']))}  DRpct {agg(get(S, ['normals_part_x_drifted', p, 'DR_pct_med']))}")
    print("  drifted normals fall in: " + " ".join(f"{p} {agg(get(S, ['drifted_normals_partition', p]))}" for p in PARTS))
    print("  non-drifted normals fall in: " + " ".join(f"{p} {agg(get(S, ['nondrifted_normals_partition', p]))}" for p in PARTS))
    print("Q2b Spearman (out-of-envelope windows):")
    for who in ("normals", "attacks"):
        print(f"  {who:8s} n {agg(get(S, ['correlations', who, 'n_out']), '{:.0f}')}: betw~DR {agg(get(S, ['correlations', who, 'betw_vs_DR', 0]), '{:+.3f}')}; valley-ind~DR {agg(get(S, ['correlations', who, 'valley_ind_vs_DR', 0]), '{:+.3f}')}; off-ind~DR {agg(get(S, ['correlations', who, 'off_ind_vs_DR', 0]), '{:+.3f}')}; dh~DR {agg(get(S, ['correlations', who, 'dh_vs_DR', 0]), '{:+.3f}')}; perp~DR {agg(get(S, ['correlations', who, 'perp_vs_DR', 0]), '{:+.3f}')}; I4 null {agg(get(S, ['correlations', who, 'I4_null_betw_vs_shuffled_drifted', 0]), '{:+.3f}')}")
    print("Q2d direction:")
    print(f"  out-of-env normals n {agg(get(S, ['direction', 'n_out_normals']), '{:.0f}')}: cos(axis h->j) med {agg(get(S, ['direction', 'cos_disp_vs_axis_hj_med']))} (>0.7 {agg(get(S, ['direction', 'frac_cos_axis_gt_0_7']))}); cos(drift dir) med {agg(get(S, ['direction', 'cos_disp_vs_late_drift_dir_med']))} (>0.7 {agg(get(S, ['direction', 'frac_cos_drift_gt_0_7']))})")
    print(f"  DRIFTED out-of-env normals n {agg(get(S, ['direction_drifted_out_normals', 'n']), '{:.0f}')}: cos(axis) med {agg(get(S, ['direction_drifted_out_normals', 'cos_axis_med']))} (>0.7 {agg(get(S, ['direction_drifted_out_normals', 'frac_cos_axis_gt_0_7']))}); cos(drift) med {agg(get(S, ['direction_drifted_out_normals', 'cos_drift_med']))} (>0.7 {agg(get(S, ['direction_drifted_out_normals', 'frac_cos_drift_gt_0_7']))}); t {agg(get(S, ['direction_drifted_out_normals', 't_med']))}; perp/R {agg(get(S, ['direction_drifted_out_normals', 'perp_over_R_med']))}")
    print(f"  late-third normals: nearest-regime dist {agg(get(S, ['direction', 'late_normals_nearest_dist_med']))} vs R {agg(get(S, ['direction', 'R']))} (early third {agg(get(S, ['direction', 'early_normals_nearest_dist_med']))}); inside any regime {agg(get(S, ['direction', 'late_normals_inside_any_regime']))}; perp/R {agg(get(S, ['direction', 'late_normals_perp_over_R_med']))}; t {agg(get(S, ['direction', 'late_normals_t_med']))}")
    print("Q2c subspace:")
    for lab in ("drifting_comm_channels", "other_channels"):
        if not any(get(S, ["subspace", lab])): continue
        print(f"  {lab} ({agg(get(S, ['subspace', lab, 'n_channels']), '{:.0f}')} ch, K {agg(get(S, ['subspace', lab, 'K']), '{:.0f}')}, R {agg(get(S, ['subspace', lab, 'R']), '{:.1f}')}):")
        print(f"    normals " + " ".join(f"{p} {agg(get(S, ['subspace', lab, 'normals', p]))}" for p in PARTS) + " | attacks " + " ".join(f"{p} {agg(get(S, ['subspace', lab, 'attacks', p]))}" for p in PARTS))
        for k in range(3):
            print(f"    T{k+1} normals: valley {agg(get(S, ['subspace', lab, 'normals_by_third', k, 'valley']))} off {agg(get(S, ['subspace', lab, 'normals_by_third', k, 'off']))} in {agg(get(S, ['subspace', lab, 'normals_by_third', k, 'in_cluster']))}")
        print(f"    drifted normals: " + " ".join(f"{p} {agg(get(S, ['subspace', lab, 'drifted_normals', p]))}" for p in PARTS) + f" | non-drifted: " + " ".join(f"{p} {agg(get(S, ['subspace', lab, 'nondrifted_normals', p]))}" for p in PARTS))
        print(f"    rho(betw, DR) normals {agg(get(S, ['subspace', lab, 'rho_betw_DR_normals', 0]), '{:+.3f}')}; rho(dh, DR) normals {agg(get(S, ['subspace', lab, 'rho_dh_DR_normals', 0]), '{:+.3f}')}; late inside-regime {agg(get(S, ['subspace', lab, 'late_normals_inside_any_regime']))}; late perp/R {agg(get(S, ['subspace', lab, 'late_normals_perp_over_R_med']))}; close pairs D<2R {agg(get(S, ['subspace', lab, 'n_close_pairs_D_lt_2R']), '{:.0f}')}")
    print("Q3 attack typing per partition: n | flagged head/cp | typed CP/DRIFT/SUB of head-flagged | drifted | AUROC head/cp/dr | global pct head/cp | era-local head/cp")
    for p in PARTS + ["all"]:
        g = lambda k: get(S, ["attack_typing", p] + k)
        print(f"  {p:10s} n {agg(g(['n']), '{:.0f}')} (diff {agg(g(['n_difficult']), '{:.0f}')}) | {agg(g(['flagged_head']))} / {agg(g(['flagged_cp']))} | {agg(g(['typed_of_head_flagged', 'changepoint']))} / {agg(g(['typed_of_head_flagged', 'drift']))} / {agg(g(['typed_of_head_flagged', 'subthreshold']))} | {agg(g(['drifted_frac']))} | {agg(g(['auroc_head']))} / {agg(g(['auroc_cp']))} / {agg(g(['auroc_dr']))} | {agg(g(['global_pct_med', 'head']))} / {agg(g(['global_pct_med', 'cp']))} | {agg(g(['local_pct_med', 'head']))} / {agg(g(['local_pct_med', 'cp']))}")
