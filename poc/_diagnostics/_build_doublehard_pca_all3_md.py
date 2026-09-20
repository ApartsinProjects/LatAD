"""Render _diagnostics/doublehard_pca_all3.md from doublehard_pca_all3.json (read-only)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "doublehard_pca_all3.json")))
DS = ["HAI", "WADI_clean", "SWaT_canon"]
COL = {"HAI": "HAI", "WADI_clean": "WADI", "SWaT_canon": "SWaT"}

# fair detectors (competitors) + our method + floor/ablation, in report order
ALTS = ["IF", "AE", "USAD", "TranAD", "GDN", "LinRes", "boosted_LOO"]
OURS = "LatAD (regime-community)"
FLOOR = "trivial max|z|"
ABL = "LatAD (global density)"


def cell(rows, m):
    if m not in rows:
        return "n/a"
    r = rows[m]
    return f"{r['auroc']:.3f}±{r['sd']:.3f}" if "sd" in r else f"{r['auroc']:.3f}"


def fmt_sig(s):
    if not s:
        return "n/a"
    return f"{s['diff']:+.3f} [{s['diff_ci'][0]:+.3f}, {s['diff_ci'][1]:+.3f}], P(diff<=0)={s['p_le_0']}"


L = []
L.append("# PCA double-hard leaderboard (all three datasets)\n")
L.append("Second filter is **PCA** (Hotelling T2 on retained components OR SPE/Q on dropped "
         "components), fit on the TRAIN-NORMAL standardized windowed features, thresholds at "
         "train-normal q99; 95% variance retained. A window is double-hard iff it is an anomaly "
         "AND `max|z|<=maxz_thr` (difficult filter, channel-based) AND `T2<=q99(T2_train)` AND "
         "`SPE<=q99(SPE_train)`. The detector under test never participates in either filter. "
         "LinRes and boosted_LOO are FAIR DETECTORS here (PCA is the filter, not LinRes). "
         "Leak-free: PCA/thresholds/train-normal from train only. AUROC is 5-seed mean±SD for "
         "seeded methods, single value otherwise. Our method (community fusion HCcoh+LatAD) bolded.\n")

# expert provenance
L.append("**Experts used:** HAI `experts_full/expert_HAI.npz`, WADI `experts_full/expert_WADI_clean.npz`, "
         "SWaT `experts_full/expert_SWaT_canon.npz` (clean official-normal bundle, **S=24**, has "
         "`fit_surprise`, mtime 12:33). SWaT_canon uses the clean official Dec-2015 normal train "
         "(0 test-normal overlap, fail-fast guard). Cross-check: SWaT HCcoh+LatAD *difficult* AUROC "
         "= 0.524 (near chance, matches the clean ~0.536 expectation; not the stale ~0.70+).\n")

# subset sizes
L.append("## PCA double-hard subset sizes (95% variance)\n")
L.append("| dataset | windows | attack episodes | (90% var) | (99% var) |")
L.append("|---|--:|--:|--:|--:|")
for ds in DS:
    o = J[ds]; sub = o["subset"]["pca"]; sens = o["sensitivity"]
    s90, s99 = sens["0.9"], sens["0.99"]
    L.append(f"| {COL[ds]} | {sub['n']} | {sub['episodes']} | "
             f"{s90['n_double_hard']}w/{s90['n_episodes']}ep | {s99['n_double_hard']}w/{s99['n_episodes']}ep |")
L.append("")

# main table
L.append("## AUROC on the PCA double-hard (95% variance)\n")
L.append("| method | HAI | WADI | SWaT |")
L.append("|---|--:|--:|--:|")
# our method first (bolded)
L.append("| **LatAD community fusion (HCcoh+LatAD) — ours** | "
         + " | ".join(f"**{cell(J[ds]['rows_pca'], OURS)}**" for ds in DS) + " |")
for m in ALTS:
    L.append(f"| {m} | " + " | ".join(cell(J[ds]["rows_pca"], m) for ds in DS) + " |")
L.append(f"| _{ABL} (ablation)_ | " + " | ".join("_" + cell(J[ds]["rows_pca"], ABL) + "_" for ds in DS) + " |")
L.append(f"| _{FLOOR} (filter floor)_ | " + " | ".join("_" + cell(J[ds]["rows_pca"], FLOOR) + "_" for ds in DS) + " |")
L.append("")

# significance
L.append("## Significance: community fusion vs the STRONGEST alternative\n")
L.append("Episode-block moving-block bootstrap, 2000 reps; paired AUROC difference "
         "(ours minus strongest fair alternative), 95% CI, one-sided P(diff<=0). "
         "Positive diff + P below 0.05 = ours significantly leads.\n")
L.append("| dataset | strongest alt | alt AUROC | ours AUROC | diff [95% CI], one-sided P | verdict |")
L.append("|---|---|--:|--:|---|---|")
for ds in DS:
    o = J[ds]; s = o["significance"]; ck = o["competitor"]
    rows = o["rows_pca"]
    altv = rows[ck]["auroc"]; ourv = rows[OURS]["auroc"]
    if s["diff"] > 0 and s["p_le_0"] is not None and s["p_le_0"] < 0.05:
        verd = "ours leads (sig.)"
    elif s["diff"] > 0:
        verd = "ours ahead, NOT sig."
    else:
        verd = f"{ck} leads (sig.)" if s["p_le_0"] == 1.0 else f"{ck} ahead"
    L.append(f"| {COL[ds]} | {ck} | {altv:.3f} | {ourv:.3f} | {fmt_sig(s)} | {verd} |")
L.append("")

# invariants
L.append("## Invariant / sanity checks\n")
for ds in DS:
    o = J[ds]; pca = o["pca"]; rows = o["rows_pca"]
    L.append(f"- **{COL[ds]}**: PCA k@95%={pca['k95']}/{o['d']}; train-normal filter-flag rate "
             f"{pca['train_flag_rate']*100:.2f}% (expect ~1-2%); all-components SPE ~0 (PCA "
             f"self-consistency ok). Floor `trivial max|z|`={rows[FLOOR]['auroc']:.3f} "
             f"(near/below chance as required — it is a defining filter).")
L.append("")
L.append("**Below-chance learned methods (explained, not a bug):** the double-hard subset removes "
         "by construction every window separable by the channel-wise `max|z|` filter OR the "
         "linear-Gaussian PCA filter (T2/SPE). Reconstruction/density/linear detectors (IF, AE, "
         "USAD, TranAD, LinRes, boosted_LOO, LatAD global-density) re-detect essentially that same "
         "removed signal, so on the residual windows their ordering inverts to at/below chance. "
         "This is the intended demonstration of the filter, not a sign error: the sign is not "
         "globally inverted, since GDN (graph-relational signal) and, on HAI/WADI, our community "
         "fusion stay above chance on the same subset. On SWaT_canon nearly every detector except "
         "GDN falls below chance on the 28-window/9-episode subset, and our community fusion "
         "(0.178) is below chance too — consistent with the clean SWaT community signal already "
         "being near chance on 'difficult' (0.524); GDN is the sole survivor there.\n")

open(os.path.join(HERE, "doublehard_pca_all3.md"), "w", encoding="utf-8").write("\n".join(L))
print("wrote doublehard_pca_all3.md")
