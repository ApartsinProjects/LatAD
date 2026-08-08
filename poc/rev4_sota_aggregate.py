"""Rev4 item 8b + 11: fold multi-seed USAD/TranAD and single-seed GDN (from the Modal
matrix) into per-window scores, and report difficult-subset AUROC.

Reads per-timestep score dumps score_<MODEL>_<DS>_s<SEED>.npy downloaded from the Modal
results volume (pass --scoredir). Window-averages each onto the SAME label grid as
build_scores_table (mean over each window's timesteps), per seed. USAD/TranAD -> (n_seed,
n_window); GDN -> (n_window,). Difficulty mask = canonical max|z| from scores_<DS>.npz, so
numbers are construct-matched with Table 3. Saves _diagnostics/scores_sota_ms_<DS>.npz and
prints difficult/all AUROC (multi-seed mean+-std). Wins-only: report only what survives.
"""
from __future__ import annotations
import argparse, os, json, numpy as np
from sklearn.metrics import roc_auc_score
import eda_real as E

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
SEEDS = [0, 1, 2, 3, 4]


def win_starts(n, W, stride):
    return list(range(0, n - W + 1, stride))


def win_avg(ts, starts, W):
    return np.array([ts[i:i + W].mean() for i in starts], np.float32)


def auroc_on(y, s, keep):
    return float(roc_auc_score(y[keep], s[keep]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoredir", required=True, help="local dir with score_<M>_<DS>_s<seed>.npy")
    args = ap.parse_args()
    SD = args.scoredir
    ALL = {}
    for name in ["WADI", "HAI", "SWaT"]:
        D = E.load(name); fn, W, stride = E.RAW[name]
        Xa_raw = np.asarray(D["Xa_raw"], float)
        starts = win_starts(len(Xa_raw), W, stride)
        d = np.load(f"{OUT}/scores_{name}.npz"); y = d["label"].astype(int)
        assert len(starts) == len(y), f"{name}: {len(starts)} windows vs {len(y)} labels"
        thr = float(d["maxz_thr"]); hard = (y == 1) & ~((y == 1) & (d["maxz"] > thr))
        keepH = np.where((y == 0) | hard)[0]; allm = np.where(y == y)[0]

        cols = {}
        def load_ts(model, seed):
            p = os.path.join(SD, f"score_{model}_{name}_s{seed}.npy")
            if not os.path.exists(p):
                return None
            ts = np.load(p); ts = ts.mean(1) if ts.ndim > 1 else ts
            if len(ts) != len(Xa_raw):
                print(f"  [{name}] {model} s{seed}: len {len(ts)} != raw {len(Xa_raw)} -> skip")
                return None
            return win_avg(ts, starts, W)

        report = {}
        for model in ["USAD", "TranAD"]:
            per = [load_ts(model, s) for s in SEEDS]
            per = [p for p in per if p is not None]
            if per:
                arr = np.stack(per); cols[model] = arr
                auH = [auroc_on(y, arr[i], keepH) for i in range(len(arr))]
                auA = [auroc_on(y, arr[i], allm) for i in range(len(arr))]
                report[model] = dict(n_seed=len(arr),
                                     all=[round(np.mean(auA), 3), round(np.std(auA), 3)],
                                     difficult=[round(np.mean(auH), 3), round(np.std(auH), 3)])
        g = load_ts("GDN", 0)
        if g is not None:
            cols["GDN"] = g
            report["GDN"] = dict(n_seed=1, all=[round(auroc_on(y, g, allm), 3), None],
                                 difficult=[round(auroc_on(y, g, keepH), 3), None])
        if cols:
            np.savez(f"{OUT}/scores_sota_ms_{name}.npz", label=y, hard=hard, **cols)
        ALL[name] = report
        print(f"\n=== {name} (difficult n={int(hard.sum())}) ===")
        for m, r in report.items():
            sd = lambda v: f"±{v[1]:.3f}" if v[1] is not None else ""
            print(f"  {m:7} (n_seed={r['n_seed']})  ALL {r['all'][0]}{sd(r['all'])}  "
                  f"DIFFICULT {r['difficult'][0]}{sd(r['difficult'])}")
    json.dump(ALL, open(f"{OUT}/rev4_sota_ms.json", "w"), indent=1)
    print(f"\nsaved -> {OUT}/rev4_sota_ms.json")


if __name__ == "__main__":
    main()
