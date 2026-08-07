"""Headline figure for the revised claim: the detector's difficult-subset advantage over
the best baseline tracks the measured number of operating modes K* (all multi-seeded).
Panel A: ours vs IF vs AE difficult-AUROC per dataset (ordered by K*).
Panel B: advantage over best baseline vs K* (the thesis, as a measured relationship).
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
NAMES = ["SKAB", "WADI", "HAI", "SWaT"]   # ascending K*
def load(n):
    o = json.load(open(f"{D}/multiseed_{n}.json"))["full_auroc"]["hard"]
    b = json.load(open(f"{D}/baseline_multiseed_{n}.json"))["difficult_auroc"]
    k = json.load(open(f"{D}/multimodality.json"))[n]["Kstar_BIC"]
    return dict(name=n, ours=o["mean"], ours_sd=o["std"],
                IF=b["IsolationForest"]["mean"], AE=b["AutoEncoder"]["mean"], K=k)
R = sorted([load(n) for n in NAMES], key=lambda r: r["K"])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [1.25, 1]})

# Panel A: grouped bars
x = np.arange(len(R)); w = 0.26
labels = [f"{r['name']}\n$K^*$={r['K']}" for r in R]
axA.bar(x - w, [r["IF"] for r in R], w, label="Isolation Forest", color="#9fb8cf")
axA.bar(x,     [r["AE"] for r in R], w, label="AutoEncoder",      color="#c9c9c9")
axA.bar(x + w, [r["ours"] for r in R], w, yerr=[r["ours_sd"] for r in R], capsize=3,
        label="Ours (VaDE-hard)", color="#1f4e79")
axA.set_xticks(x); axA.set_xticklabels(labels)
axA.set_ylabel("difficult-subset AUROC"); axA.set_ylim(0.35, 1.0)
axA.axhline(0.5, ls=":", c="#999", lw=1)
axA.set_title("Difficult-subset detection (5-seed mean $\\pm$ std)")
axA.legend(frameon=False, fontsize=9, loc="upper left"); axA.grid(alpha=.2, axis="y")

# Panel B: advantage over best baseline vs K*
adv = [r["ours"] - max(r["IF"], r["AE"]) for r in R]
Ks = [r["K"] for r in R]
axB.axhline(0, c="#c0392b", lw=1)
axB.plot(Ks, adv, "-", c="#1f4e79", alpha=.5, zorder=1)
axB.scatter(Ks, adv, s=90, c="#1f4e79", zorder=3, edgecolor="white", linewidth=1)
for r, a in zip(R, adv):
    axB.annotate(r["name"], (r["K"], a), textcoords="offset points", xytext=(6, 6), fontsize=9)
axB.set_xlabel("measured operating modes  $K^*$ (BIC)")
axB.set_ylabel("advantage over best baseline (AUROC)")
axB.set_title("Advantage tracks multimodality")
axB.grid(alpha=.2)

fig.tight_layout()
fig.savefig(f"{D}/thesis_advantage_vs_K.png", dpi=150, bbox_inches="tight")
fig.savefig(f"{D}/thesis_advantage_vs_K.svg", bbox_inches="tight")
print("wrote thesis_advantage_vs_K.png/.svg")
print(f"{'ds':5}{'K*':>4}{'ours':>8}{'bestbase':>10}{'adv':>8}")
for r, a in zip(R, adv):
    print(f"{r['name']:5}{r['K']:>4}{r['ours']:>8.3f}{max(r['IF'],r['AE']):>10.3f}{a:>+8.3f}")
