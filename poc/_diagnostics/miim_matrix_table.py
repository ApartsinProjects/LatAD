"""Summarise _diagnostics/miim_matrix.json into the A1-A10 x dataset verdict table and
write the verdicts back into the JSON (key 'verdicts')."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
J = os.path.join(HERE, "miim_matrix.json")
R = json.load(open(J))
DS = [d for d in ["WADI", "HAI", "SWaT", "SKAB"] if d in R]
V = {}


def put(a, d, verdict, txt):
    V.setdefault(a, {})[d] = dict(verdict=verdict, evidence=txt)


for d in DS:
    r = R[d]
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = (r[k] for k in ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"])
    # A1 multimodality: BIC K and held-out-median K
    kb, kh, kmax = a1["K_bic"], a1["K_heldout_ll_median"], a1["K_max"]
    v = "HOLDS" if kb >= 8 and kh >= 8 else ("PARTIAL" if kb >= 4 else "ABSENT")
    put("A1", d, v, f"BIC K*={kb}, held-out K*={kh} (Kmax {kmax}); cov PR={a1['cov_participation_ratio']:.1f}")
    # A2 regime growth: BIC still improving at Kmax and no 1% saturation
    imp = a2["bic_still_improving_at_Kmax"]
    # saturation: step gain as a fraction of the TOTAL BIC drop K=1 -> argmin (robust to BIC sign)
    bc = a1["bic_curve"]; b = [x["bic"] for x in bc]; kk = [x["K"] for x in bc]
    tot = b[0] - min(b); steps = [(b[i - 1] - b[i]) / tot for i in range(1, len(b))]
    ks = next((kk[i + 1] for i, g in enumerate(steps) if g < 0.01), None)
    a2["step_gain_frac_of_total_drop"] = [round(g, 3) for g in steps]; a2["K_sat_1pct_of_total"] = ks
    hocurve = a2["ll_ho_curve_median"]; ho_imp = hocurve[-1][1] > hocurve[-2][1]
    v = "HOLDS" if (imp and ho_imp) else ("PARTIAL" if (imp or ho_imp) else "ABSENT")
    put("A2", d, v, f"BIC argmin {kb}, BIC improving at Kmax={kmax}: {imp}; held-out median still rising at Kmax: {ho_imp}; K_sat(step<1% of total BIC drop)={ks}; step gains {a2['step_gain_frac_of_total_drop']}")
    # A3 thin pockets
    rho = a3["rho"]
    v = "HOLDS" if rho >= 0.3 else ("PARTIAL" if rho >= 0.1 else "ABSENT")
    put("A3", d, v, f"rho={rho:.3f} (train max-resp<0.5), basin gate {'fires' if a3['basin_lam'] > 0 else 'off'}")
    # A4 fringes
    g = a4["heldout_ll_gain_highK_minus_paperK_median"]; f = a4["frac_heldout_windows_improved_vs_paperK"]
    k = a4["median_within_mode_excess_kurtosis"]
    v = "HOLDS" if (g > 5 and f > 0.6) else ("PARTIAL" if (g > 0 or k > 1) else "ABSENT")
    put("A4", d, v, f"K=80 vs K_paper={a4['K_paper']}: held-out median ll gain {g:+.1f} nats/window, {f:.0%} windows improved; within-mode excess kurtosis {k:.2f}")
    # A5 few levers
    fr = a5["PR_frac_Deff"]; n90 = a5["n90"]
    v = "HOLDS" if fr < 0.1 else ("PARTIAL" if fr < 0.2 else "ABSENT")
    put("A5", d, v, f"PCA PR={a5['PR']:.1f} of D_eff={a5['D_eff']} ({fr:.1%}); n90={n90} ({a5['n90_frac_Deff']:.1%}), n95={a5['n95']}")
    # A6 heavy tail
    sl = a6["zipf_slope"]; rt = a6["max_min_ratio"]; gi = a6["gini"]
    v = "HOLDS" if (gi > 0.45 and rt > 20) else ("PARTIAL" if gi > 0.3 else "ABSENT")
    put("A6", d, v, f"K={a6['K']}: Zipf slope {sl:.2f}, max/min occupancy {rt:.0f}x, Gini {gi:.2f}, top mode {a6['top1']:.0%}")
    # A7 hidden regimes: low silhouette => implicit
    sf = a7["sil_feature_kmeans"]; sl_ = a7.get("sil_latent_vade", float("nan"))
    v = "HOLDS" if (sf < 0.15) else ("PARTIAL" if sf < 0.3 else "ABSENT")
    put("A7", d, v, f"silhouette feature-space {sf:.2f} (KMeans K={a7['K_mode']}); VaDE latent {sl_:.2f} (K={a7.get('vade_K')}); train between-mode {a7.get('vade_frac_maxresp_lt_0.5_train', float('nan')):.2f}")
    # A8 typed channels + locality
    nc = a8["E3_hac_nested_communities_3to25"]; q = a8["thr0.5_modularity"]; fd = a8["frac_discrete_of_live"]
    v = "HOLDS" if (q > 0.3 and nc >= 10) else ("PARTIAL" if q > 0.2 or nc >= 5 else "ABSENT")
    put("A8", d, v, f"{nc} HAC communities (E3 rule), {a8['thr0.5_greedy_modularity_communities']} greedy communities Q={q:.2f} at |r|>0.5; discrete channels {a8['n_discrete_le10vals']}/{a8['n_live']} ({fd:.0%}), const {a8['n_const']}")
    # A9 multiscale: exercised only if anomalies are NOT snapshot-separable
    fe = a9["frac_anom_easy"]; ad = a9["triv_auroc_difficult"]; ae = a9["triv_auroc_easy"]
    v = "NOT-EXERCISED" if fe > 0.5 else ("PARTIAL" if fe > 0.3 else "HOLDS")
    shift = f"; CAVEAT {a9['frac_test_normal_over_thr']:.0%} of test-NORMAL windows also exceed it (test-side shift)" if a9["frac_test_normal_over_thr"] > 0.1 else ""
    put("A9", d, v, f"{fe:.0%} of anomaly windows exceed the train-q99 single-window max|z| rule; trivial-rule AUROC easy {ae:.2f} / difficult {ad:.2f} / all {a9['triv_auroc_all']:.2f}{shift}")
    # A10 path dependence
    tb = a10.get("test_anom_between_mode_frac")
    v = "NOT-EXERCISED" if fe > 0.5 else "PARTIAL"
    ext = f"; test anomalies between modes {tb:.0%}, train rho {rho:.2f}" if tb is not None else ""
    put("A10", d, v, f"no labelled history-conditioned fault; {fe:.0%} snapshot-separable, remainder bounded by difficult AUROC {ad:.2f}{ext}")

R["verdicts"] = V
json.dump(R, open(J, "w"), indent=1)
names = {"A1": "A1 Regime mixture", "A2": "A2 Regime explosion", "A3": "A3 Hard envelopes", "A4": "A4 Thin fringes",
         "A5": "A5 Few levers", "A6": "A6 Heavy tail", "A7": "A7 Hidden regimes", "A8": "A8 Mixed signals",
         "A9": "A9 Many clocks", "A10": "A10 Path dependence"}
print("| Assumption | " + " | ".join(DS) + " |")
print("|---|" + "---|" * len(DS))
for a in names:
    print(f"| {names[a]} | " + " | ".join(f"**{V[a][d]['verdict']}** ({V[a][d]['evidence']})" for d in DS) + " |")
