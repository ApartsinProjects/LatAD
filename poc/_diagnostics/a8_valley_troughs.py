"""How deep are the inter-regime valleys on the real datasets, in the same PCA-10 + BIC-GMM geometry as a8_valley_real.py?

For every pair of components (i, j) with a density valley (min mixture density along mu_i -> mu_j below 0.5 of the
lower endpoint), record the mixture log-density at the deepest point of the segment as a percentile of the TRAIN
windows' log-density, and the Mahalanobis distance from the deepest point to its nearest component. A valley whose
floor sits above the train 1st percentile cannot be flagged at 1% FPR by ANY density head fitted to this geometry;
a floor above the train 50th percentile is typical normal operation. Also reports, over the close pairs
(D < 2R), the median floor percentile. Output: a8_valley_troughs.json + stdout.
"""
import os, sys, json, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE)); os.chdir(os.path.dirname(HERE))
from a8_valley_real import feats, Geo, PCA, NPC, SEEDS
out = {}
for ds in ["SWaT_canon", "WADI_clean", "HAI"]:
    Ftr, Fte, yfrac = feats(ds); mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    pca = PCA(NPC, random_state=0).fit((Ftr - mu) / sd); Ztr = pca.transform((Ftr - mu) / sd)
    out[ds] = []
    for seed in SEEDS:
        G = Geo(Ztr, seed); R = float(np.quantile(G.dist(Ztr).min(1), 0.995))
        lp_tr = G.g.score_samples(Ztr); q = lambda v: float((lp_tr < v).mean())
        ts = np.linspace(0, 1, 41); rows = []
        for i, j in itertools.combinations(range(G.K), 2):
            seg = G.mu[i][None] + ts[:, None] * (G.mu[j] - G.mu[i])[None]; lp = G.g.score_samples(seg); k = int(np.argmin(lp))
            dmin = float(G.dist(seg[k:k + 1]).min())
            rows.append(dict(i=i, j=j, D=float(G.D[i, j]), valley=float(G.valley[i, j]), floor_pct=q(lp[k]), floor_t=float(ts[k]), floor_dnear=dmin,
                             pi=float(min(G.g.weights_[i], G.g.weights_[j]))))
        vr = [r for r in rows if r["valley"] < 0.5]; close = [r for r in vr if r["D"] < 2 * R]
        summ = dict(seed=seed, K=G.K, R=R, n_pairs=len(rows), n_valley=len(vr), n_close_valley=len(close),
                    floor_pct_median_close=float(np.median([r["floor_pct"] for r in close])) if close else None,
                    floor_pct_min_close=float(np.min([r["floor_pct"] for r in close])) if close else None,
                    share_close_floor_below_q1=float(np.mean([r["floor_pct"] < 0.01 for r in close])) if close else None,
                    share_close_floor_below_q5=float(np.mean([r["floor_pct"] < 0.05 for r in close])) if close else None,
                    share_close_floor_dnear_gt_R=float(np.mean([r["floor_dnear"] > R for r in close])) if close else None,
                    floor_dnear_median_close=float(np.median([r["floor_dnear"] for r in close])) if close else None,
                    deepest_close=sorted(close, key=lambda r: r["floor_pct"])[:5])
        out[ds].append(summ)
        print(f"[{ds} s{seed}] K={G.K} R={R:.2f} pairs={len(rows)} valley-separated={len(vr)} close(D<2R)&valley={len(close)} | floor pct over close pairs: median={summ['floor_pct_median_close']:.3f} min={summ['floor_pct_min_close']:.4f} share<q1={summ['share_close_floor_below_q1']:.2f} share<q5={summ['share_close_floor_below_q5']:.2f} | floor nearest-Mahalanobis median={summ['floor_dnear_median_close']:.2f} share>R={summ['share_close_floor_dnear_gt_R']:.2f}", flush=True)
json.dump(out, open(os.path.join(HERE, "a8_valley_troughs.json"), "w"), indent=1)
