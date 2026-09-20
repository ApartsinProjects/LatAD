"""5-seed USAD/TranAD aggregation for WADI_clean and SWaT_canon (paper Table 3/4 SD backfill).

Adapts rev4_sota_aggregate.py (window-averaging) + ensemble_final.py's DoubleHard definition
to the canonical "_clean"/"_canon" datasets. Reads per-timestep score dumps
score_<MODEL>_<DS>_s<seed>.npy pulled from the latad-sota-results Modal volume, window-averages
onto the SAME grid as scores_<DS>.npz (construct-matched), and reports 5-seed mean+-SD AUROC on
All / Easy / Difficult / DoubleHard, using the CLEAN difficulty/double-hard masks already stored
in scores_WADI_clean.npz / scores_SWaT_canon.npz (+ onehot_filter LinRes for DoubleHard).

Usage: python sota_5seed_aggregate.py --scoredir <dir with score_<M>_<DS>_s<seed>.npy>
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eda_real as E
from onehot_filter import build_feats, loco_residual

OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3, 4]
DATASETS = ["WADI_clean", "SWaT_canon"]

# point estimates currently in Table 3 (single-run), for the sanity check
CURRENT = {
    "WADI_clean": {"USAD": {"All": 0.757, "Difficult": 0.579}, "TranAD": {"All": 0.786, "Difficult": 0.613}},
    "SWaT_canon": {"USAD": {"All": 0.873, "Difficult": 0.658}, "TranAD": {"All": 0.873, "Difficult": 0.655}},
}


def win_starts(n, W, stride):
    return list(range(0, n - W + 1, stride))


def win_avg(ts, starts, W):
    return np.array([ts[i:i + W].mean() for i in starts], np.float32)


def auroc_on(y, s, keep):
    return float(roc_auc_score(y[keep], s[keep]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoredir", required=True)
    args = ap.parse_args()
    SD = args.scoredir

    ALL = {}
    for name in DATASETS:
        D = E.load(name)
        fn, W, stride = E.RAW[name]
        Xn_raw = np.asarray(D["Xn_raw"], float)
        Xa_raw = np.asarray(D["Xa_raw"], float)
        starts = win_starts(len(Xa_raw), W, stride)

        d = np.load(f"{OUT}/scores_{name}.npz")
        y = d["label"].astype(int)
        assert len(starts) == len(y), f"{name}: {len(starts)} window starts vs {len(y)} labels"
        mthr = float(d["maxz_thr"]); maxz = d["maxz"]

        # DoubleHard: same definition as ensemble_final.py / rev4_doublehard.py
        Fn, Fa, grp = build_feats(Xn_raw, Xa_raw, W, stride, onehot=True)
        r_tr, r_te = loco_residual(Fn, Fa[:len(y)], grp)
        lin_thr = float(np.quantile(r_tr, 0.99))

        easy = (y == 1) & (maxz > mthr)
        diff = (y == 1) & (maxz <= mthr)
        dhard = diff & (r_te <= lin_thr)
        subsets = {"All": (y == 1), "Easy": easy, "Difficult": diff, "DoubleHard": dhard}
        n_sub = {s: int(((y == 0) | mk).sum()) if s != "All" else int(len(y)) for s, mk in subsets.items()}

        def load_ts(model, seed):
            p = os.path.join(SD, f"score_{model}_{name}_s{seed}.npy")
            if not os.path.exists(p):
                print(f"  [{name}] MISSING {p}")
                return None
            ts = np.load(p)
            ts = ts.mean(1) if ts.ndim > 1 else ts
            if len(ts) != len(Xa_raw):
                print(f"  [{name}] len mismatch ts={len(ts)} raw={len(Xa_raw)} -> skip")
                return None
            return win_avg(ts, starts, W)

        report = {}
        per_seed_dump = {}
        for model in ["USAD", "TranAD"]:
            per = [load_ts(model, s) for s in SEEDS]
            got_seeds = [s for s, p in zip(SEEDS, per) if p is not None]
            per = [p for p in per if p is not None]
            if not per:
                continue
            arr = np.stack(per)  # (n_seed, n_window)
            rep_sub = {}
            per_seed_sub = {}
            for sname, mk in subsets.items():
                keep = np.arange(len(y)) if sname == "All" else np.where((y == 0) | mk)[0]
                vals = [auroc_on(y, arr[i], keep) for i in range(len(arr))]
                rep_sub[sname] = dict(mean=round(float(np.mean(vals)), 4),
                                       sd=round(float(np.std(vals)), 4), n=n_sub[sname])
                per_seed_sub[sname] = {seed: round(v, 4) for seed, v in zip(got_seeds, vals)}
            report[model] = dict(n_seed=len(arr), seeds=got_seeds, subsets=rep_sub, per_seed=per_seed_sub)
        ALL[name] = dict(n_windows=int(len(y)), n_anom=int(y.sum()),
                          n_easy=int(easy.sum()), n_difficult=int(diff.sum()), n_doublehard=int(dhard.sum()),
                          results=report)
        print(f"\n=== {name} (n_windows={len(y)}, anom={int(y.sum())}, easy={int(easy.sum())}, "
              f"difficult={int(diff.sum())}, doublehard={int(dhard.sum())}) ===")
        for m, r in report.items():
            line = f"  {m:7} n_seed={r['n_seed']} "
            for s in ["All", "Easy", "Difficult", "DoubleHard"]:
                x = r["subsets"][s]
                line += f"| {s} {x['mean']:.3f}+-{x['sd']:.3f} (n={x['n']}) "
            print(line)
            cur = CURRENT.get(name, {}).get(m, {})
            for s, cv in cur.items():
                mv = r["subsets"][s]["mean"]
                flag = " <<< FLAG >2% off" if abs(mv - cv) > 0.02 else ""
                print(f"      sanity {s}: 5seed-mean={mv:.3f} vs current-point-estimate={cv:.3f} "
                      f"(delta={mv-cv:+.3f}){flag}")

    json.dump(ALL, open(f"{OUT}/sota_5seed_wadi_swat.json", "w"), indent=2)
    print(f"\nsaved -> {OUT}/sota_5seed_wadi_swat.json")


if __name__ == "__main__":
    main()
