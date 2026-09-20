"""Pilot part 3: latent-space gap DETECTOR on the benchmarks' hard subset vs cached base head (seed 0)."""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from scipy.special import logsumexp
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "fable_a3_pilot.json")
RES = json.load(open(OUT)); RES.setdefault("latent_detector", {})
def au(y, s, m):
    k = (y == 0) | m; return float(roc_auc_score(y[k], s[k]))
for name in ["WADI", "HAI", "SWaT"]:
    d = np.load(os.path.join(HERE, f"e2_fable_{name}.npz")); h = np.load(os.path.join(HERE, f"heads_{name}.npz"))
    z, zt, C = d["ztr"].astype(float), d["zte"].astype(float), d["mu_c"].astype(float)
    yw, hard = d["yw"], d["hard"]
    assert len(h["label"]) == len(yw) and (h["label"] == yw).all(), "split mismatch"
    lp = d["logpi"][None] + d["logN_tr"]; lab = np.exp(lp - logsumexp(lp, 1, keepdims=True)).argmax(1)
    nn = NearestNeighbors(n_neighbors=11).fit(z); rad_te = nn.kneighbors(zt, n_neighbors=10)[0][:, -1]
    dc_tr = ((z[:, None, :] - C[None]) ** 2).sum(-1); r_own = np.sqrt(dc_tr[np.arange(len(z)), lab])
    med_r = np.array([np.median(r_own[lab == c]) if (lab == c).sum() > 0 else np.nan for c in range(len(C))])
    med_r = np.nan_to_num(med_r, nan=np.nanmedian(med_r))
    dc = ((zt[:, None, :] - C[None]) ** 2).sum(-1); o = np.argsort(dc, 1); a, b = o[:, 0], o[:, 1]
    v = C[b] - C[a]; L2 = (v ** 2).sum(1) + 1e-12; t = ((zt - C[a]) * v).sum(1) / L2
    proj = C[a] + np.clip(t, 0, 1)[:, None] * v; perp = np.linalg.norm(zt - proj, axis=1) / (med_r[a] + 1e-9)
    interior = ((t > 0.2) & (t < 0.8)).astype(float); chord = interior * np.exp(-perp)
    gap = rankdata(rad_te) + rankdata(chord)
    base = h["base"][0]
    r = dict(base_hard=au(yw, base, hard), knn_z_hard=au(yw, rad_te, hard), chord_hard=au(yw, chord, hard),
             gap_hard=au(yw, gap, hard), base_plus_gap_hard=au(yw, rankdata(base) + 0.5 * gap, hard),
             n_hard=int(hard.sum()), frac_hard_interior=float(interior[hard].mean()),
             frac_normal_interior=float(interior[yw == 0].mean()))
    RES["latent_detector"][name] = r; print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(RES, open(OUT, "w"), indent=1)
