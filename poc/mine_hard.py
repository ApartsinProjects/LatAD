"""Mine the hardest-to-catch anomalies: difficult-subset windows that LatAD scores most
normal (worst-caught), and characterize each concretely so we can target a method extension.

For each dataset: rank difficult (or double-hard) anomaly windows by LatAD score percentile
among ALL windows (low = missed). For the worst K, report:
  - LatAD / IF / AE / density / nearest / linres score PERCENTILES (who catches it?);
  - top deviating channels (name, window-mean z vs train-normal), and the temporal SHAPE
    of each (level-shift vs transient vs dip) via first-half/second-half means and slope;
  - cross-channel break magnitude (LinRes residual percentile);
  - attack episode id.
Aggregates the recurring pattern. Output -> _diagnostics/mine_hard_<DS>.json + printed report.
"""
from __future__ import annotations
import json, os, sys, numpy as np
import eda_real as E

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
K = 8  # worst-caught windows to characterize


def pct_of(all_scores, v):
    return float((all_scores <= v).mean())


def episodes(y):
    eps, i, n = [], 0, len(y); lab = np.zeros(n, int)
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1:
                j += 1
            for k in range(i, j):
                lab[k] = len(eps) + 1
            eps.append((i, j)); i = j
        else:
            i += 1
    return lab


def characterize(name, subset="difficult"):
    D = E.load(name); fn, W, stride = E.RAW[name]
    Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
    ch = [str(c) for c in D["ch"]]
    idx = np.asarray(D["idx_a"], int)
    d = np.load(f"{OUT}/scores_{name}.npz"); y = d["label"].astype(int); thr = float(d["maxz_thr"])
    mu, sg = Xn.mean(0), Xn.std(0) + 1e-9
    latad = d["LatAD"].mean(0); IFs = d["IF"].mean(0); AEs = d["AE"].mean(0)
    lin = d["linres"]
    h = np.load(f"{OUT}/heads_{name}.npz")
    dens = h["density"].mean(0); near = h["nearest"].mean(0)
    easy = (y == 1) & (d["maxz"] > thr); hard = (y == 1) & ~easy
    if subset == "double":
        # add LinRes filter with train-normal threshold proxy (test-normal q99 here, quick)
        lthr = float(np.quantile(lin[y == 0], 0.99)); hard = hard & (lin <= lthr)
    epi = episodes(y)
    hidx = np.where(hard)[0]
    # rank difficult windows by LatAD percentile among ALL windows (low = worst-caught)
    lp = np.array([pct_of(latad, latad[i]) for i in hidx])
    worst = hidx[np.argsort(lp)][:K]

    recs = []
    for wi in worst:
        s, e = idx[wi], idx[wi] + W
        win = Xa[s:e]                                   # (W, C)
        z = (win.mean(0) - mu) / sg                     # per-channel window-mean z
        top = np.argsort(-np.abs(z))[:6]
        chans = []
        for c in top:
            col = (win[:, c] - mu[c]) / sg[c]           # standardized channel trace in-window
            fh, sh = col[:W // 2].mean(), col[W // 2:].mean()
            within = col.std()
            shape = ("level-shift" if abs(fh - sh) > 1.0 and within < abs(z[c]) * 0.8
                     else "transient" if within > max(1.0, abs(z[c])) else "flat-offset")
            chans.append(dict(ch=ch[c], z=round(float(z[c]), 2),
                              first_half_z=round(float(fh), 2), second_half_z=round(float(sh), 2),
                              within_std=round(float(within), 2), shape=shape))
        recs.append(dict(
            window=int(wi), episode=int(epi[wi]),
            latad_pct=round(pct_of(latad, latad[wi]), 3),
            if_pct=round(pct_of(IFs, IFs[wi]), 3),
            ae_pct=round(pct_of(AEs, AEs[wi]), 3),
            density_pct=round(pct_of(dens, dens[wi]), 3),
            nearest_pct=round(pct_of(near, near[wi]), 3),
            linres_pct=round(pct_of(lin, lin[wi]), 3),
            maxz=round(float(d["maxz"][wi]), 3),
            top_channels=chans))
    # aggregate: which channels recur, common shape, does IF/linres catch what LatAD misses?
    from collections import Counter
    chan_counter = Counter(c["ch"] for r in recs for c in r["top_channels"])
    shape_counter = Counter(c["shape"] for r in recs for c in r["top_channels"])
    agg = dict(
        n_worst=len(recs),
        median_latad_pct=round(float(np.median([r["latad_pct"] for r in recs])), 3),
        median_if_pct=round(float(np.median([r["if_pct"] for r in recs])), 3),
        median_ae_pct=round(float(np.median([r["ae_pct"] for r in recs])), 3),
        median_linres_pct=round(float(np.median([r["linres_pct"] for r in recs])), 3),
        recurring_channels=chan_counter.most_common(8),
        shape_mix=shape_counter.most_common())
    return dict(dataset=name, subset=subset, n_difficult=int(hard.sum()), aggregate=agg, worst=recs)


if __name__ == "__main__":
    names = sys.argv[1:] or ["WADI", "HAI", "SWaT"]
    ALL = {}
    for nm in names:
        r = characterize(nm, subset="difficult")
        ALL[nm] = r
        json.dump(r, open(f"{OUT}/mine_hard_{nm}.json", "w"), indent=1)
        a = r["aggregate"]
        print(f"\n===== {nm}: {r['n_difficult']} difficult; {a['n_worst']} worst-caught =====")
        print(f"  median score-percentile among ALL windows (low = missed):")
        print(f"    LatAD {a['median_latad_pct']}  IF {a['median_if_pct']}  AE {a['median_ae_pct']}  LinRes {a['median_linres_pct']}")
        print(f"  recurring top channels: {a['recurring_channels']}")
        print(f"  temporal shapes: {a['shape_mix']}")
        for w in r["worst"][:4]:
            tc = ", ".join(f"{c['ch']}(z={c['z']},{c['shape']})" for c in w["top_channels"][:3])
            print(f"    win{w['window']} ep{w['episode']}: LatAD_pct={w['latad_pct']} IF_pct={w['if_pct']} "
                  f"linres_pct={w['linres_pct']} | {tc}")
    json.dump(ALL, open(f"{OUT}/mine_hard.json", "w"), indent=1)
    print(f"\nsaved -> {OUT}/mine_hard.json")
