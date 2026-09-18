"""Review-2 recompute: (1) Table A2 WADI on the CLEAN 30-window subset, (2) WADI seed SDs
for Table 3/4, (3) WADI difficult-subset significance of HCcoh+LatAD vs each learned detector.
CLEAN pipeline (FIX 1-3): WADI = WADI_clean, difficulty mask from scores_WADI_clean.npz
(maxz <= maxz_thr, 30 windows), experts_full. Persists -> _diagnostics/review2_recompute.json.
"""
from __future__ import annotations
import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("BOOT_REPS", "0")            # ensemble_final's own boot: skip (we do our own)
import numpy as np
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                        # = poc/  (E.load needs ../datasets, EF needs EXPERTS_DIR relative to poc)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.chdir(ROOT)
import eda_real as E
from winfeat import window_features
from models_vade import train_vade

NAME = "WADI_clean"
SEEDS = [0, 1, 2, 3, 4]
d = np.load(f"{HERE}/scores_{NAME}.npz", allow_pickle=True)
y = d["label"].astype(int)
mz = d["maxz"]; thr = float(d["maxz_thr"])
HARD = (y == 1) & (mz <= thr)                       # clean 30-window difficult mask
EASY = (y == 1) & (mz > thr)
fn, W, stride = E.RAW[NAME]
OUT = {"dataset": NAME, "n_windows": int(len(y)), "n_anom": int(y.sum()),
       "n_difficult": int(HARD.sum()), "maxz_thr": round(thr, 4)}
print(f"[setup] n_win={len(y)} n_anom={int(y.sum())} difficult={int(HARD.sum())} thr={thr:.4f}", flush=True)

# ---------------------------------------------------------------- Task 1: Table A2 WADI
def winfeats(X, rep):
    F = np.stack([window_features(X[i:i + W], rep) for i in range(0, len(X) - W + 1, stride)]).astype(np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

def au_hard(s):
    keep = (y == 0) | HARD
    return float(roc_auc_score(y[keep], s[keep]))

def task1_repr_ablation():
    K, LD = 20, 10                                  # repr_ablation CFG for WADI_clean
    Dd = E.load(NAME)
    Xn = np.asarray(Dd["Xn_raw"], np.float32); Xa = np.asarray(Dd["Xa_raw"], np.float32)
    assert len(range(0, len(Xa) - W + 1, stride)) == len(y), "window/label mismatch"
    res = {}
    for rep in ("stats", "temporal"):
        Xtr0 = winfeats(Xn, rep); Xte0 = winfeats(Xa, rep)
        mu, sg = Xtr0.mean(0), Xtr0.std(0) + 1e-8
        clp = lambda A: np.clip(np.nan_to_num((A - mu) / sg, nan=0.0, posinf=0.0, neginf=0.0), -10, 10).astype(np.float32)
        Xtr, Xte = clp(Xtr0), clp(Xte0)
        kd = min(80, max(20, len(Xtr0) // 10))
        aus = []
        for sd in SEEDS:
            v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
            v.fit_latent_density(Xtr, k_density=kd)
            s = np.asarray(v.anomaly_score_hard(Xte, use_resid=False, use_basin=False))
            aus.append(au_hard(s))
        res[rep] = dict(dim=int(Xtr0.shape[1]), diff=round(float(np.mean(aus)), 4),
                        sd=round(float(np.std(aus)), 4), per_seed=[round(a, 4) for a in aus])
        print(f"[A2] {rep:9s} dim={Xtr0.shape[1]:3d} difficult={res[rep]['diff']:.4f}+-{res[rep]['sd']:.4f}", flush=True)
    res["temporal_minus_stats"] = round(res["temporal"]["diff"] - res["stats"]["diff"], 4)
    print(f"[A2] temporal - stats = {res['temporal_minus_stats']:+.4f}", flush=True)
    return res

# ---------------------------------------------------------------- ensemble scores (HCcoh+LatAD)
os.environ["EXPERTS_DIR"] = "sota_bundle/experts_full"
import ensemble_final as EF                          # resolves EXPERTS_DIR relative to cwd (=poc)
ens, y2, d2, nseed = EF.ensemble_scores(NAME)
assert np.array_equal(y2, y)
HEADKEY = "HCcoh+LatAD"
regime = ens[HEADKEY]                               # (nseed, n)

# ---------------------------------------------------------------- Task 2: WADI seed SDs
def au_multi(arr, keep):
    if arr.ndim == 2:
        aus = [roc_auc_score(y[keep], arr[i][keep]) for i in range(arr.shape[0])]
        return round(float(np.mean(aus)), 3), round(float(np.std(aus)), 3)
    return round(float(roc_auc_score(y[keep], arr[keep])), 3), None

def task2_sds():
    r_tr = None
    # double-hard needs loco residual threshold (same as ensemble_final.run)
    Dd = E.load(NAME); Xn_raw = np.asarray(Dd["Xn_raw"], float); Xa_raw = np.asarray(Dd["Xa_raw"], float)
    from onehot_filter import build_feats, loco_residual
    Fn, Fa, grp = build_feats(Xn_raw, Xa_raw, W, stride, onehot=True)
    r_tr, r_te = loco_residual(Fn, Fa[:len(y)], grp); lin_thr = float(np.quantile(r_tr, 0.99))
    dhard = (y == 1) & (mz <= thr) & (r_te <= lin_thr)
    subsets = {"All": (y == 1), "Easy": EASY, "Difficult": HARD, "DoubleHard": dhard}
    methods = {"IsolationForest": d["IF"], "AutoEncoder": d["AE"],
               "LatAD-global": d["LatAD"], "LatAD regime-community (HCcoh+LatAD)": regime}
    out = {"n_subset": {s: int(((y == 0) | mk).sum()) if s == "All" else int(mk.sum()) for s, mk in subsets.items()}}
    tbl = {}
    for m, arr in methods.items():
        tbl[m] = {}
        for s, mk in subsets.items():
            keep = np.arange(len(y)) if s == "All" else np.where((y == 0) | mk)[0]
            au, sdv = au_multi(arr, keep)
            tbl[m][s] = dict(auroc=au, sd=sdv)
        print(f"[SD] {m:38s} " + "  ".join(
            f"{s}={tbl[m][s]['auroc']:.3f}+-{tbl[m][s]['sd']:.3f}" for s in subsets), flush=True)
    out["table"] = tbl
    return out

# ---------------------------------------------------------------- Task 3: significance vs learned
def episodes(yy):
    eps, i, n = [], 0, len(yy)
    while i < n:
        if yy[i] == 1:
            j = i
            while j < n and yy[j] == 1:
                j += 1
            eps.append(np.arange(i, j)); i = j
        else:
            i += 1
    return eps

def mean_auroc(arr, yk, keep):
    if arr.ndim == 2:
        return float(np.mean([roc_auc_score(yk, arr[i][keep]) for i in range(arr.shape[0])]))
    return float(roc_auc_score(yk, arr[keep]))

def paired_boot(method, compet, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    L = int(np.ceil(W / stride)) + 1
    norm = np.where(y == 0)[0]
    hard_eps = [e[HARD[e]] for e in episodes(y) if HARD[e].any()]
    keep0 = np.where((y == 0) | HARD)[0]
    a_pt = mean_auroc(method, y[keep0], keep0); c_pt = mean_auroc(compet, y[keep0], keep0)
    diffs = []
    for _ in range(reps):
        nb = int(np.ceil(len(norm) / L)); st = rng.integers(0, max(1, len(norm) - L + 1), size=nb)
        sn = np.concatenate([norm[s:s + L] for s in st])[:len(norm)]
        pick = rng.integers(0, len(hard_eps), size=len(hard_eps)); sh = np.concatenate([hard_eps[k] for k in pick])
        if len(sh) < 2:
            continue
        idx = np.concatenate([sn, sh]); yk = y[idx]
        if yk.sum() < 2 or (yk == 0).sum() < 2:
            continue
        diffs.append(mean_auroc(method, yk, idx) - mean_auroc(compet, yk, idx))
    diffs = np.array(diffs)
    q = lambda v: [round(float(np.quantile(v, 0.025)), 3), round(float(np.quantile(v, 0.975)), 3)]
    return dict(method_auroc=round(a_pt, 3), compet_auroc=round(c_pt, 3),
                diff=round(a_pt - c_pt, 3), diff_ci=q(diffs),
                p_le_0=round(float((diffs <= 0).mean()), 4),
                n_episodes=len(hard_eps), n_boot=len(diffs))

def task3_significance():
    # GDN per-window (window-averaged per-timestep, seed 0), aligned to y
    Dd = E.load(NAME); Xa = np.asarray(Dd["Xa_raw"], float)
    starts = list(range(0, len(Xa) - W + 1, stride))
    ts = np.load(f"{HERE}/score_GDN_{NAME}_s0.npy"); ts = ts.mean(1) if ts.ndim > 1 else ts
    gdn = np.array([ts[i:i + W].mean() for i in starts], np.float32)
    learned = {"AutoEncoder": d["AE"], "USAD": d["USAD"], "TranAD": d["TranAD"],
               "IsolationForest": d["IF"], "GDN": gdn}
    out = {}
    for m, arr in learned.items():
        out[m] = paired_boot(regime, arr, reps=2000)
        r = out[m]
        print(f"[SIG] HCcoh+LatAD ({r['method_auroc']}) vs {m:16s} ({r['compet_auroc']}): "
              f"diff {r['diff']:+.3f} CI{r['diff_ci']} P(<=0)={r['p_le_0']} eps={r['n_episodes']}", flush=True)
    # verdict: significantly ahead of EVERY learned detector at alpha=0.05 (one-sided)?
    all_sig = all(out[m]["p_le_0"] < 0.05 for m in learned)
    out["_verdict_ahead_of_every_learned_at_0.05"] = bool(all_sig)
    print(f"[SIG] VERDICT significantly ahead of every learned detector (p<0.05): {all_sig}", flush=True)
    return out

if __name__ == "__main__":
    OUT["task1_tableA2_WADI"] = task1_repr_ablation()
    OUT["task2_seed_sds"] = task2_sds()
    OUT["task3_significance_vs_learned"] = task3_significance()
    json.dump(OUT, open(f"{HERE}/review2_recompute.json", "w"), indent=1)
    print(f"\nsaved -> {HERE}/review2_recompute.json", flush=True)
