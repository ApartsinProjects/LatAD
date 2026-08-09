"""Construct-matched community-density ENSEMBLE results, one artifact.

Combines the per-community VaDE experts (expert_<DS>.npz, 5-seed) with the canonical baselines
and the LatAD null expert from scores_<DS>.npz, on the SAME subset definitions as the main tables:
  - Easy / Difficult:  maxz vs train-99th maxz_thr           (build_scores_table)
  - Double-hard:       anomaly & maxz<=thr & loco_residual<=train-99th   (rev4_doublehard)
Aggregators reported (all train-normal-only; combination rule fixed a priori):
  - HC        : Higher Criticism over per-community upper-tail p-values (sparsity-adaptive).
  - HC_coh    : HC weighted by community cohesion*sqrt(size) (a violation in a tight regime counts more).
  - null+HC   : max( z(HC), z(LatAD null-expert tail) ) -- the global unfactorized expert included so
                a dense whole-system fault is never missed. HEADLINE ensemble.
Every method is scored on All/Easy/Difficult/Double-hard with AUROC (5-seed mean+-std) and best raw
point-wise F1 (0.80-0.999 quantile grid, per-seed then mean), plus an episode-block bootstrap of the
headline ensemble vs the strongest baseline on Difficult and Double-hard. Saves _diagnostics/ensemble_final.json.
"""
from __future__ import annotations
import json, os, numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import eda_real as E
from onehot_filter import build_feats, loco_residual

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}   # (K, latent); W/stride from eda_real
COMPET = {"WADI": "IF", "HAI": "AE", "SWaT": "linres"}          # strongest classical baseline per dataset
RNG = np.random.default_rng(0)
REPS = int(os.environ.get("BOOT_REPS", "2000"))                 # set 0 to skip the slow bootstrap
OUTJSON = os.environ.get("OUT_JSON", "ensemble_final.json")
HEAD = os.environ.get("HEAD", "null+HC")                        # which aggregator to bootstrap for significance
BASELINES = ["trivial max|z|", "IF", "AE", "linres", "USAD", "TranAD", "GDN"]
KEYMAP = {"trivial max|z|": "maxz"}


def surv(ref, v):
    o = np.sort(ref); r = np.searchsorted(o, v, side="right") / len(o)
    return -np.log(np.clip(1 - r, 1e-4, 1.0))


def pval(ref, v):
    o = np.sort(ref); r = np.searchsorted(o, v, side="left") / len(o)
    return np.clip(1 - r, 1e-4, 1.0)


def HC(P, wt=None):
    S, n = P.shape
    if wt is not None:
        P = np.clip(P ** (wt[:, None] / (wt.mean() + 1e-9)), 1e-4, 1.0)
    Ps = np.sort(P, axis=0); i = (np.arange(1, S + 1) / S)[:, None]
    hc = np.sqrt(S) * (i - Ps) / np.sqrt(np.clip(Ps * (1 - Ps), 1e-6, None))
    hc[Ps >= 0.5] = -np.inf
    return np.nan_to_num(hc.max(0), neginf=0.0, posinf=1e6)


def episodes(y):
    eps, i, n = [], 0, len(y)
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1:
                j += 1
            eps.append(np.arange(i, j)); i = j
        else:
            i += 1
    return eps


def bestf1(yk, sk):
    if (yk == 1).sum() < 1 or (yk == 0).sum() < 1:
        return float("nan")
    qs = np.quantile(sk, np.linspace(0.80, 0.999, 60))
    return max(f1_score(yk, sk > t) for t in qs)


def au_multi(arr, y, keep):
    """arr (nseed,n) or (n,) -> (mean AUROC over seeds, std, mean bestF1 over seeds) on keep."""
    if arr.ndim == 2:
        aus = [roc_auc_score(y[keep], arr[i][keep]) for i in range(arr.shape[0])]
        f1s = [bestf1(y[keep], arr[i][keep]) for i in range(arr.shape[0])]
        return round(float(np.mean(aus)), 3), round(float(np.std(aus)), 3), round(float(np.nanmean(f1s)), 3)
    a = float(roc_auc_score(y[keep], arr[keep])); f = bestf1(y[keep], arr[keep])
    return round(a, 3), 0.0, round(float(f), 3)


def boot(y, method, compet, mask, L, reps=2000):
    y = y.astype(int); norm = np.where(y == 0)[0]
    heps = [e[mask[e]] for e in episodes(y) if mask[e].any()]
    keep0 = np.where((y == 0) | mask)[0]
    mau = lambda arr, idx: (float(np.mean([roc_auc_score(y[idx], arr[i][idx]) for i in range(arr.shape[0])]))
                            if arr.ndim == 2 else float(roc_auc_score(y[idx], arr[idx])))
    dpt = mau(method, keep0) - mau(compet, keep0); diffs = []
    if not heps or reps <= 0:
        return dict(diff=round(dpt, 3), diff_ci=None, p_le_0=None, n_episodes=len(heps))
    for _ in range(reps):
        nb = int(np.ceil(len(norm) / L)); st = RNG.integers(0, max(1, len(norm) - L + 1), size=nb)
        sn = np.concatenate([norm[s:s + L] for s in st])[:len(norm)]
        pick = RNG.integers(0, len(heps), size=len(heps)); sh = np.concatenate([heps[k] for k in pick])
        if len(sh) < 2:
            continue
        idx = np.concatenate([sn, sh])
        if y[idx].sum() < 2 or (y[idx] == 0).sum() < 2:
            continue
        diffs.append(mau(method, idx) - mau(compet, idx))
    diffs = np.array(diffs)
    q = lambda v: [round(float(np.quantile(v, 0.025)), 3), round(float(np.quantile(v, 0.975)), 3)]
    return dict(diff=round(dpt, 3), diff_ci=q(diffs) if len(diffs) else None,
                p_le_0=round(float((diffs <= 0).mean()), 4) if len(diffs) else None, n_episodes=len(heps))


def ensemble_scores(name):
    """Return dict of ensemble score arrays (nseed,n): HC, HC_coh, null+HC, plus context."""
    d = np.load(f"{OUT}/scores_{name}.npz")
    Ex = np.load(f"{os.environ.get('EXPERTS_DIR', 'sota_bundle/experts')}/expert_{name}.npz", allow_pickle=True)
    y = d["label"].astype(int)
    assert len(Ex["y"]) == len(y) and int(np.abs(Ex["y"].astype(int) - y).sum()) == 0, f"{name}: expert/scores label mismatch"
    Cal, Tst = Ex["calib_surprise"], Ex["test_surprise"]          # (nseed,S,ncal/ntest)
    coh = Ex["comm_cohesion"].astype(float); size = Ex["comm_size"].astype(float)
    w = coh * np.sqrt(size)
    lat = d["LatAD"]; nseed = min(Tst.shape[0], lat.shape[0]); nm = y == 0; S = Tst.shape[1]
    wn = w / (w.max() + 1e-9)
    keys = ["HC", "HC_coh", "null+HC",                 # p-value-level aggregation (current)
            "sum+LatAD", "max+LatAD", "cohmax+LatAD", "HC+LatAD", "HCcoh+LatAD", "q95+LatAD"]  # density then LatAD post-proc
    acc = {k: [] for k in keys}
    for sd in range(nseed):
        tails = np.stack([surv(Cal[sd, g], Tst[sd, g]) for g in range(S)])   # per-community surprise
        P = np.stack([pval(Cal[sd, g], Tst[sd, g]) for g in range(S)])
        hc = HC(P); hc_coh = HC(P, wt=w)
        nulltail = surv(lat[sd][nm], lat[sd])
        z = lambda s: (s - s[nm].mean()) / (s[nm].std() + 1e-9)
        zl = z(nulltail)                                # LatAD post-processing signal (z vs normal)
        acc["HC"].append(hc); acc["HC_coh"].append(hc_coh)
        acc["null+HC"].append(np.maximum(z(hc), zl))
        # aggregate the community densities into ONE score, then fuse with LatAD (z-summed post-proc)
        acc["sum+LatAD"].append(z(tails.sum(0)) + zl)
        acc["max+LatAD"].append(z(tails.max(0)) + zl)
        acc["cohmax+LatAD"].append(z((wn[:, None] * tails).max(0)) + zl)
        acc["HC+LatAD"].append(z(hc) + zl)
        acc["HCcoh+LatAD"].append(z(hc_coh) + zl)
        acc["q95+LatAD"].append(z(np.quantile(tails, 0.95, axis=0)) + zl)
    return {k: np.stack(v) for k, v in acc.items()}, y, d, nseed


def run(name):
    ens, y, d, nseed = ensemble_scores(name)
    fn, W, stride = E.RAW[name]
    Dd = E.load(name); Xn_raw, Xa_raw = np.asarray(Dd["Xn_raw"], float), np.asarray(Dd["Xa_raw"], float)
    mthr = float(d["maxz_thr"]); maxz = d["maxz"]
    Fn, Fa, grp = build_feats(Xn_raw, Xa_raw, W, stride, onehot=True)
    r_tr, r_te = loco_residual(Fn, Fa[:len(y)], grp); lin_thr = float(np.quantile(r_tr, 0.99))
    easy = (y == 1) & (maxz > mthr); diff = (y == 1) & (maxz <= mthr)
    dhard = (y == 1) & (maxz <= mthr) & (r_te <= lin_thr)
    subsets = {"All": (y == 1), "Easy": easy, "Difficult": diff, "DoubleHard": dhard}

    methods = {}
    for m in BASELINES:
        k = KEYMAP.get(m, m)
        if k in d.files:
            methods[m] = d[k]
    methods["LatAD"] = d["LatAD"]
    # USAD/TranAD: use the multi-seed SOTA arrays (co-computed on the identical window grid) so their
    # AUROC is a five-seed mean+-std, consistent with the learned-detector protocol used for IF/AE/LatAD.
    msf = f"{OUT}/scores_sota_ms_{name}.npz"
    if os.path.exists(msf):
        ms = np.load(msf, allow_pickle=True)
        assert np.array_equal(ms["label"].astype(int), y), f"{name}: ms/scores label mismatch"
        for m in ("USAD", "TranAD"):
            if m in ms.files and m in methods:
                methods[m] = ms[m]
    for k, v in ens.items():
        methods[k] = v

    res = {"n_windows": int(len(y)), "n_anom": int(y.sum()), "maxz_thr": round(mthr, 3),
           "lin_thr": round(lin_thr, 4), "nseed": nseed,
           "n_subset": {s: int(((y == 0) | mk).sum()) if s == "All" else int(mk.sum()) for s, mk in subsets.items()},
           "n_episodes": {s: len([e for e in episodes(y) if mk[e].any()]) for s, mk in subsets.items()},
           "results": {}}
    for m, arr in methods.items():
        res["results"][m] = {}
        for s, mk in subsets.items():
            keep = np.arange(len(y)) if s == "All" else np.where((y == 0) | mk)[0]
            if (y[keep] == 1).sum() < 2:
                res["results"][m][s] = None; continue
            au, sd, f1 = au_multi(arr, y, keep)
            res["results"][m][s] = dict(auroc=au, sd=sd, f1=f1)
    # significance: headline null+HC vs strongest baseline, on Difficult and DoubleHard
    L = int(np.ceil(W / stride)) + 1; ck = COMPET[name]
    res["significance"] = {
        "vs": ck,
        "Difficult": boot(y, methods[HEAD], methods[ck], diff, L, reps=REPS) if diff.sum() >= 3 else None,
        "DoubleHard": boot(y, methods[HEAD], methods[ck], dhard, L, reps=REPS) if dhard.sum() >= 3 else None,
    }
    return res


if __name__ == "__main__":
    import sys
    ALL = {}
    for nm in (sys.argv[1:] or ["WADI", "HAI", "SWaT"]):
        ALL[nm] = run(nm)
        r = ALL[nm]
        print(f"\n=== {nm}  (nseed={r['nseed']}, subsets {r['n_subset']}) ===")
        hdr = f"{'method':<16}{'All':>14}{'Easy':>14}{'Difficult':>14}{'DoubleHard':>14}"
        print(hdr)
        for m, cols in r["results"].items():
            def cell(c):
                return "  n/a" if c is None else f"{c['auroc']:.3f}±{c['sd']:.3f}"
            print(f"{m:<16}" + "".join(f"{cell(cols[s]):>14}" for s in ["All", "Easy", "Difficult", "DoubleHard"]))
        sig = r["significance"]
        for sub in ("Difficult", "DoubleHard"):
            s = sig[sub]
            if s:
                print(f"   null+HC vs {sig['vs']} [{sub}]: diff {s['diff']} CI {s['diff_ci']} "
                      f"P(<=0)={s['p_le_0']} (episodes {s['n_episodes']})")
    json.dump(ALL, open(f"{OUT}/{OUTJSON}", "w"), indent=1)
    print(f"\nsaved -> {OUT}/{OUTJSON}")
