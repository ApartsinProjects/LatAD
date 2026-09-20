"""Datapoint-level companion to a8_drift_unification.py (SWaT_canon by default).
(1) every VALLEY attack window (geometry seed 0): time index, third, episode, dh / t / perp, DRIFTED, typing,
    head / CP global and era-local percentiles, and the communities carrying the largest fast residual;
(2) magnitude-matched check on NORMALS: within terciles of dh (out-of-envelope normals), DRIFTED share and DR
    percentile for valley vs off, so the valley-vs-off contrast is not a magnitude artefact;
(3) episode view: per attack episode, share of windows in each partition and head -> CP era-local change.
Output: a8_drift_unification_inspect.log
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ["EXPERTS_DIR"] = os.path.join(ROOT, "sota_bundle", "experts_full")
import ensemble_final as EF
from a8_valley_real import CACHE, PARTS
import importlib.util
# reuse the decomposition code from the main script without executing its main loop
src = open(os.path.join(HERE, "a8_drift_unification.py"), encoding="utf-8").read().split("# ------------------------------------------------------------------ main")[0]
ns = {"__file__": os.path.join(HERE, "a8_drift_unification.py")}; exec(compile(src, "a8_drift_unification_funcs", "exec"), ns)
scores, pct_vs, local_pct = ns["scores"], ns["pct_vs"], ns["local_pct"]
STEP_S = {"SWaT_canon": 300, "WADI_clean": 300, "HAI": 30}
LOG = open(os.path.join(HERE, "a8_drift_unification_inspect.log"), "a")


def log(s=""):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


for name in (sys.argv[1:] or ["SWaT_canon"]):
    seed = 0
    d = np.load(f"{EF.OUT}/scores_{name}.npz"); y = d["label"].astype(int); n = len(y); nrm = y == 0; an = y == 1
    Ex = np.load(f"{os.environ['EXPERTS_DIR']}/expert_{name}.npz", allow_pickle=True)
    step = STEP_S[name]; K24 = int(24 * 3600 / step); third = np.minimum((np.arange(n) * 3) // n, 2)
    diff = an & (d["maxz"] <= float(d["maxz_thr"]))
    SP = scores(Ex, d, K24, scale=True)
    hm, cm, dm = SP["head"].mean(0), SP["cp"].mean(0), SP["dr"].mean(0)
    thr_h, thr_c = (float(np.quantile(SP[k].mean(0), 0.99)) for k in ("head_cal", "cp_cal"))
    slowT = SP["slowT"].mean(0); fastT = SP["fastT"].mean(0); Cal = Ex["calib_surprise"].astype(float).mean(0)
    q99g = np.quantile(Cal, 0.99, axis=1); fd = (slowT > q99g[:, None]).any(0); fh, fc = hm > thr_h, cm > thr_c
    typ = np.where(fc, "CP", np.where(fd, "DRIFT", "SUB"))
    names = [str(c) for c in np.load(os.path.join(CACHE, f"raw_{name}.npz"), allow_pickle=True)["ch"]]
    comm = [[names[int(c)] for c in Ex["comm_channels"][g] if c >= 0] for g in range(slowT.shape[0])]
    V = pd.read_csv(os.path.join(HERE, f"a8_valley_windows_{name}_seed{seed}.csv")); part = V["part"].values.astype(str)
    pg = {k: pct_vs(v, nrm) for k, v in (("head", hm), ("cp", cm), ("dr", dm))}
    eps = EF.episodes(y); ep_of = np.full(n, -1)
    for k, e in enumerate(eps): ep_of[e] = k
    log(f"\n================ {name} (geometry seed {seed}) ================")
    # (1) valley attacks
    idx = np.where(an & (part == "valley"))[0]
    lh, lc = local_pct(hm, idx, nrm, K24), local_pct(cm, idx, nrm, K24)
    log(f"(1) VALLEY attack windows n={len(idx)} (difficult {int(diff[idx].sum())}), thirds {np.bincount(third[idx], minlength=3).tolist()}, episodes {sorted(set(int(ep_of[i]) for i in idx))}")
    log(f"  {'t':>5s} {'T':>1s} {'ep':>3s} {'start':>5s} {'len':>3s} {'diff':>4s} {'dh':>5s} {'t_ax':>5s} {'perp':>5s} {'h->j':>5s} {'drif':>4s} {'type':>5s} {'gH':>5s} {'gCP':>5s} {'lH':>5s} {'lCP':>5s}  top fast-residual communities")
    for q, i in enumerate(idx):
        top = np.argsort(-fastT[:, i])[:2]
        log(f"  {i:5d} {third[i]+1:1d} {ep_of[i]:3d} {eps[ep_of[i]][0]:5d} {len(eps[ep_of[i]]):3d} {int(diff[i]):4d} {V['dh'][i]:5.1f} {V['t'][i]:5.2f} {V['perp'][i]:5.1f} {V['h'][i]:2d}>{V['j'][i]:<2d} {int(fd[i]):4d} {typ[i]:>5s} {pg['head'][i]:5.2f} {pg['cp'][i]:5.2f} {lh[q]:5.2f} {lc[q]:5.2f}  "
            + "; ".join(f"g{g} {fastT[g, i]:+.1f} {comm[g][:4]}" for g in top))
    # same summary for OFF and IN_CLUSTER attacks, by third
    for p in ("off", "in_cluster", "beyond"):
        ii = np.where(an & (part == p))[0]
        if len(ii) == 0: continue
        log(f"  {p}: n={len(ii)} thirds {np.bincount(third[ii], minlength=3).tolist()} difficult {int(diff[ii].sum())} drifted {fd[ii].mean():.2f} global head/cp pct med {np.median(pg['head'][ii]):.3f}/{np.median(pg['cp'][ii]):.3f} era-local {np.median(local_pct(hm, ii, nrm, K24)):.3f}/{np.median(local_pct(cm, ii, nrm, K24)):.3f}")
    # (2) magnitude-matched normals
    mo = nrm & (part != "in_cluster"); q = np.quantile(V["dh"][mo], [1 / 3, 2 / 3])
    log(f"(2) magnitude-matched NORMALS out of envelope n={mo.sum()}: dh terciles at {q.round(1).tolist()}")
    for b, (lo, hi) in enumerate([(-np.inf, q[0]), (q[0], q[1]), (q[1], np.inf)]):
        m = mo & (V["dh"].values > lo) & (V["dh"].values <= hi)
        s = f"  tercile {b} dh in ({lo:.1f},{hi:.1f}]:"
        for p in ("valley", "off", "beyond"):
            mm = m & (part == p)
            s += f"  {p} n={mm.sum():3d} drifted {fd[mm].mean() if mm.any() else float('nan'):.2f} DRpct {np.median(pg['dr'][mm]) if mm.any() else float('nan'):.2f} T3-share {(third[mm] == 2).mean() if mm.any() else float('nan'):.2f} |"
        log(s)
    # (3) episodes
    log(f"(3) episodes ({len(eps)}): start len diff | partition shares in/valley/beyond/off | era-local pct head -> CP (median)")
    for k, e in enumerate(eps):
        sh = [float((part[e] == p).mean()) for p in PARTS]
        log(f"  ep{k:2d} @{e[0]:5d} n={len(e):3d} diff={int(diff[e].sum()):3d} | {sh[0]:.2f}/{sh[1]:.2f}/{sh[2]:.2f}/{sh[3]:.2f} | {np.median(local_pct(hm, e, nrm, K24)):.3f} -> {np.median(local_pct(cm, e, nrm, K24)):.3f}  drifted {fd[e].mean():.2f}")
