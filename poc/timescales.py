"""Quantify the 'many clocks' assumption (A9): each sensor evolves on its own
characteristic time scale. We measure, per channel, the autocorrelation decay time
tau (first lag where the normal-data ACF drops below 1/e) -- the channel's 'clock'.
A dataset is 'multi-clock' when tau spans a wide range across channels.

Outputs _diagnostics/timescales.json + a figure (per-channel tau distribution per dataset).
Also reports how much of the fault energy sits in the FAST band, to explain why
window-statistic features (which see level/spread, not oscillation) miss SKAB.
"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import eda_real as E

DATASETS = ["WADI", "HAI", "SKAB", "SWaT"]
COL = {"WADI": "#1f4e79", "HAI": "#0f7d84", "SKAB": "#c0392b", "SWaT": "#9a6a12"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics"); os.makedirs(OUT, exist_ok=True)
LMAX = 200


def acf_decay(x, lmax=LMAX):
    """First lag where autocorrelation < 1/e. x: 1-D normal series (already ~standardized)."""
    x = x - x.mean()
    v = np.dot(x, x)
    if v < 1e-9:
        return np.nan                      # constant channel: no clock
    lmax = min(lmax, len(x) - 1)
    for k in range(1, lmax + 1):
        r = np.dot(x[:-k], x[k:]) / v
        if r < np.exp(-1):
            return float(k)
    return float(lmax)                      # slower than the window we looked at


def analyze(name):
    D = E.load(name)
    Xn = np.asarray(D["Xn_raw"], float)     # normal stream, (T, C), standardized
    taus = []
    for c in range(Xn.shape[1]):
        t = acf_decay(Xn[:, c])
        if not np.isnan(t):
            taus.append(t)
    taus = np.array(taus)
    # 'clock spread' = ratio of slow to fast characteristic times across channels
    p95, p05 = np.percentile(taus, 95), np.percentile(taus, 5)
    spread = float(p95 / max(p05, 1.0))
    return {"dataset": name, "n_channels": int(Xn.shape[1]), "n_active": int(len(taus)),
            "tau_median": float(np.median(taus)), "tau_p05": float(p05), "tau_p95": float(p95),
            "clock_spread_p95_p05": round(spread, 1),
            "log10_tau_std": round(float(np.std(np.log10(taus + 1))), 3),
            "taus": [round(float(t), 1) for t in taus]}


def main():
    res = {n: analyze(n) for n in DATASETS}
    json.dump(res, open(f"{OUT}/timescales.json", "w"), indent=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.7, 1]})
    # per-channel tau strip (log), one column per dataset
    for i, n in enumerate(DATASETS):
        t = np.array(res[n]["taus"])
        xj = i + (np.random.default_rng(0).random(len(t)) - 0.5) * 0.5
        ax1.scatter(xj, t, s=14, alpha=.5, color=COL[n])
        ax1.plot([i - 0.32, i + 0.32], [res[n]["tau_median"]] * 2, color="black", lw=2, zorder=5)
    ax1.set_yscale("log"); ax1.set_xticks(range(len(DATASETS))); ax1.set_xticklabels(DATASETS)
    ax1.set_ylabel("sensor time scale  $\\tau$  (ACF 1/e decay, samples)")
    ax1.set_title("Each dot is one sensor's clock; wide spread = 'many clocks' (A9)")
    ax1.grid(alpha=.2, axis="y")
    # clock-spread bar
    sp = [res[n]["clock_spread_p95_p05"] for n in DATASETS]
    ax2.bar(DATASETS, sp, color=[COL[n] for n in DATASETS])
    for i, s in enumerate(sp): ax2.text(i, s, f"{s:.0f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylabel("clock spread  ($\\tau_{95}/\\tau_{5}$)"); ax2.set_title("Range of time scales across sensors")
    ax2.grid(alpha=.2, axis="y")
    fig.tight_layout()
    fig.savefig(f"{OUT}/timescales.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT}/timescales.svg", bbox_inches="tight")

    print(f"{'ds':5}{'chan':>6}{'tau_med':>9}{'tau_p05':>9}{'tau_p95':>9}{'spread':>8}{'logtau_std':>11}")
    for n in DATASETS:
        r = res[n]
        print(f"{n:5}{r['n_channels']:>6}{r['tau_median']:>9.1f}{r['tau_p05']:>9.1f}"
              f"{r['tau_p95']:>9.1f}{r['clock_spread_p95_p05']:>7.0f}x{r['log10_tau_std']:>11.3f}")


if __name__ == "__main__":
    main()
