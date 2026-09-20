"""Non-wins A (WADI significance) and B (WADI F1): bug hunt + power analysis, offline on the stored
score tables (no re-inference). Output -> fable_wadi_nonwin_audit.json (written after each block).

Blocks
  1. LEAK AUDIT. build_scores_table.py re-standardizes the window features with train std+1e-8 and no
     clip; features of train-CONSTANT channels reach |z| ~ 1e9 in test. Which difficult windows carry
     such a blow-up, and how do every method's Difficult AUROC / F1 / bootstrap change when those
     windows are (a) kept (paper), (b) removed, (c) scored alone?
  2. PER-EPISODE DECOMPOSITION. Difficult AUROC of headline vs linres per attack episode and
     leave-one-episode-out: which episodes carry the margin, which lose it.
  3. ACHIEVABLE SIGNIFICANCE. Shift the headline's difficult-window scores upward by delta (rank
     inflation), rerun the episode-block bootstrap: what Difficult AUROC would clear P<0.05 at n=11
     episodes? And: with the observed per-episode heterogeneity, how many episodes would the current
     +0.037 need (bootstrap with n_ep in {11,22,44,88} resampled episodes).
  4. F1 AUDIT. Best-F1 under the 60-point 0.80-0.999 quantile grid (paper) vs an exhaustive threshold
     sweep; per-seed-oracle-then-mean vs oracle on the seed-mean score; episode-bootstrap CI of the F1
     difference LatAD - linres; normal-quantile (99th) F1 (deployable, no oracle).
"""
from __future__ import annotations
import json, os, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import roc_auc_score, f1_score
import eda_real as E
from ensemble_final import ensemble_scores, episodes, boot, bestf1, OUT

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(0)
NAME = "WADI_clean"
OUTJ = os.path.join(HERE, "fable_wadi_nonwin_audit.json")
RES = {}


def save():
    json.dump(RES, open(OUTJ, "w"), indent=1)


def au(arr, y, keep):
    if arr.ndim == 2:
        a = [roc_auc_score(y[keep], arr[i][keep]) for i in range(arr.shape[0])]
        return round(float(np.mean(a)), 3), round(float(np.std(a)), 3)
    return round(float(roc_auc_score(y[keep], arr[keep])), 3), 0.0


ens, y, d, nseed = ensemble_scores(NAME)
mthr = float(d["maxz_thr"]); maxz = d["maxz"]
easy = (y == 1) & (maxz > mthr); diff = (y == 1) & (maxz <= mthr)
methods = {"maxz": d["maxz"], "IF": d["IF"], "AE": d["AE"], "linres": d["linres"], "USAD": d["USAD"],
           "TranAD": d["TranAD"], "LatAD_global": d["LatAD"], "HC_coh": ens["HC_coh"],
           "null+HC": ens["null+HC"], "HCcoh+LatAD": ens["HCcoh+LatAD"]}
fn, W, stride = E.RAW[NAME]; L = int(np.ceil(W / stride)) + 1

# ---------------- 1. leak audit ----------------
D = E.load(NAME); Xn, Xa, ch = D["Xn_w"].astype(np.float64), D["Xa_w"].astype(np.float64), D["ch"]; nch = len(ch)
mu, sg = Xn.mean(0), Xn.std(0) + 1e-8
Z = (Xa - mu) / sg
const_feat = np.where(Xn.std(0) < 1e-6)[0]                      # train-constant window features
blow = np.abs(Z[:, const_feat]).max(1) if len(const_feat) else np.zeros(len(y))
leak = blow > 100                                                # |z| > 100 only possible via the 1e-8 floor
trivial_full = np.abs(Z).max(1)                                  # max|z| over ALL six feature blocks (re-standardized)
trivial_full_thr = float(np.quantile(np.abs((Xn - mu) / sg).max(1), 0.99))
blocks = ["mean", "std", "min", "max", "trend", "range"]
leak_ch = {}
for i in np.where(leak & diff)[0]:
    f = const_feat[np.argmax(np.abs(Z[i, const_feat]))]
    leak_ch[int(i)] = (ch[f % nch], blocks[f // nch], float(np.abs(Z[i, f])))
RES["leak"] = dict(
    n_const_features=int(len(const_feat)), n_const_channels=int((Xn.std(0)[:nch] < 1e-6).sum()),
    n_leak_windows=int(leak.sum()), n_leak_difficult=int((leak & diff).sum()), n_leak_easy=int((leak & easy).sum()),
    n_leak_normal=int((leak & (y == 0)).sum()),
    leak_difficult_windows={k: v for k, v in leak_ch.items()},
    leak_normal_windows=[int(i) for i in np.where(leak & (y == 0))[0]],
    trivial_full_thr=trivial_full_thr,
    n_difficult_caught_by_full_trivial=int((diff & (trivial_full > trivial_full_thr)).sum()),
    n_normal_caught_by_full_trivial=int(((y == 0) & (trivial_full > trivial_full_thr)).sum()),
)
sub = {"difficult_all43": diff, "difficult_minus_leak": diff & ~leak, "difficult_leak_only": diff & leak,
       "difficult_minus_fulltrivial": diff & ~(trivial_full > trivial_full_thr)}
RES["leak"]["auroc"] = {}
for s, mk in sub.items():
    keep = np.where((y == 0) | mk)[0]
    RES["leak"]["auroc"][s] = {"n_anom": int(mk.sum()), **{m: au(a, y, keep) for m, a in methods.items()}}
# rank of leak windows under each method (percentile vs test normals, seed-mean)
def pct(arr, idx):
    a = arr if arr.ndim == 1 else arr.mean(0) if arr.max() < 1e6 else np.median(np.argsort(np.argsort(arr, 1), 1), 0)
    norm = a[y == 0]
    return [round(float((norm < a[i]).mean()), 3) for i in idx]
li = np.where(leak & diff)[0]
RES["leak"]["leak_window_percentile_vs_normals"] = {m: pct(a, li) for m, a in methods.items()}
# bootstrap headline vs linres on difficult-minus-leak
RES["leak"]["boot_minus_leak"] = {m: boot(y, methods[m], d["linres"], diff & ~leak, L, reps=2000)
                                  for m in ("HCcoh+LatAD", "null+HC", "LatAD_global", "AE")}
RES["leak"]["boot_all43"] = {m: boot(y, methods[m], d["linres"], diff, L, reps=2000)
                             for m in ("HCcoh+LatAD", "null+HC")}
save(); print("block 1 done", flush=True)

# ---------------- 2. per-episode decomposition ----------------
eps = [e for e in episodes(y) if diff[e].any()]
head = ens["HCcoh+LatAD"]; lin = d["linres"]
norm_idx = np.where(y == 0)[0]
per = []
for k, e in enumerate(eps):
    he = e[diff[e]]
    keep = np.concatenate([norm_idx, he])
    a_h = np.mean([roc_auc_score(y[keep], head[i][keep]) for i in range(nseed)])
    a_l = roc_auc_score(y[keep], lin[keep])
    # leave-one-episode-out
    rest = np.concatenate([x[diff[x]] for j, x in enumerate(eps) if j != k])
    keep2 = np.concatenate([norm_idx, rest])
    loo_h = np.mean([roc_auc_score(y[keep2], head[i][keep2]) for i in range(nseed)])
    loo_l = roc_auc_score(y[keep2], lin[keep2])
    per.append(dict(episode=k, start=int(e[0]), len=int(len(e)), n_difficult=int(len(he)),
                    n_leak=int(leak[he].sum()), headline=round(float(a_h), 3), linres=round(float(a_l), 3),
                    diff=round(float(a_h - a_l), 3), loo_diff=round(float(loo_h - loo_l), 3)))
RES["per_episode"] = per
save(); print("block 2 done", flush=True)

# ---------------- 3. achievable significance ----------------
def shifted(arr, delta):
    out = arr.copy()
    for i in range(arr.shape[0]):
        s = out[i]; sd = s[y == 0].std() + 1e-9
        s[diff] = s[diff] + delta * sd                       # shift difficult windows up by delta normal-SDs
    return out
keep = np.where((y == 0) | diff)[0]
sweep = []
for delta in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
    arr = shifted(head, delta)
    a = au(arr, y, keep)
    b = boot(y, arr, lin, diff, L, reps=2000)
    sweep.append(dict(delta_normalSD=delta, headline_difficult=a[0], diff_vs_linres=b["diff"], ci=b["diff_ci"], p=b["p_le_0"]))
    print(f"  delta {delta}: AUROC {a[0]} diff {b['diff']} P {b['p_le_0']}", flush=True)
RES["achievable"] = {"shift_sweep": sweep}
# how many episodes would the CURRENT margin need? bootstrap resampling n_ep episodes (with replacement)
heps = [e[diff[e]] for e in eps]
need = []
for n_ep in [11, 22, 44, 88, 176]:
    diffs = []
    for _ in range(1500):
        nb = int(np.ceil(len(norm_idx) / L)); st = RNG.integers(0, max(1, len(norm_idx) - L + 1), size=nb)
        sn = np.concatenate([norm_idx[s:s + L] for s in st])[:len(norm_idx)]
        pick = RNG.integers(0, len(heps), size=n_ep); sh = np.concatenate([heps[k] for k in pick])
        idx = np.concatenate([sn, sh])
        mh = np.mean([roc_auc_score(y[idx], head[i][idx]) for i in range(nseed)])
        diffs.append(mh - roc_auc_score(y[idx], lin[idx]))
    diffs = np.array(diffs)
    need.append(dict(n_episodes=n_ep, diff_mean=round(float(diffs.mean()), 3),
                     ci=[round(float(np.quantile(diffs, .025)), 3), round(float(np.quantile(diffs, .975)), 3)],
                     p_le_0=round(float((diffs <= 0).mean()), 4)))
    print(f"  n_ep {n_ep}: {need[-1]}", flush=True)
RES["achievable"]["episodes_needed_at_current_margin"] = need
save(); print("block 3 done", flush=True)

# ---------------- 4. F1 audit ----------------
def bestf1_exh(yk, sk):
    o = np.unique(sk)
    return max(f1_score(yk, sk >= t) for t in o) if len(o) < 3000 else bestf1(yk, sk)
def f1q99(yk, sk, ref):
    return f1_score(yk, sk > np.quantile(ref, 0.99))
f1res = {}
for s, mk in {"All": (y == 1), "Difficult": diff, "difficult_minus_leak": diff & ~leak}.items():
    kp = np.arange(len(y)) if s == "All" else np.where((y == 0) | mk)[0]
    row = {}
    for m in ("linres", "AE", "LatAD_global", "HCcoh+LatAD", "null+HC", "HC_coh"):
        arr = methods[m]
        if arr.ndim == 2:
            grid = [bestf1(y[kp], arr[i][kp]) for i in range(nseed)]
            exh = [bestf1_exh(y[kp], arr[i][kp]) for i in range(nseed)]
            rk = np.mean([np.argsort(np.argsort(arr[i])) for i in range(nseed)], 0)   # seed-mean rank score
            row[m] = dict(grid_perseed_mean=round(float(np.mean(grid)), 3), grid_perseed_sd=round(float(np.std(grid)), 3),
                          exhaustive_perseed_mean=round(float(np.mean(exh)), 3),
                          oracle_on_seedmean_rank=round(float(bestf1_exh(y[kp], rk[kp])), 3),
                          f1_at_normal_q99=round(float(np.mean([f1q99(y[kp], arr[i][kp], arr[i][y == 0]) for i in range(nseed)])), 3))
        else:
            row[m] = dict(grid=round(float(bestf1(y[kp], arr[kp])), 3), exhaustive=round(float(bestf1_exh(y[kp], arr[kp])), 3),
                          f1_at_normal_q99=round(float(f1q99(y[kp], arr[kp], arr[y == 0])), 3))
    f1res[s] = row
# episode-bootstrap CI of best-F1(HCcoh+LatAD) - best-F1(linres) on Difficult
heps = [e[diff[e]] for e in eps]; dl = []
for _ in range(1000):
    nb = int(np.ceil(len(norm_idx) / L)); st = RNG.integers(0, max(1, len(norm_idx) - L + 1), size=nb)
    sn = np.concatenate([norm_idx[s:s + L] for s in st])[:len(norm_idx)]
    pick = RNG.integers(0, len(heps), size=len(heps)); sh = np.concatenate([heps[k] for k in pick])
    idx = np.concatenate([sn, sh])
    fh = np.mean([bestf1(y[idx], head[i][idx]) for i in range(nseed)]); fl = bestf1(y[idx], lin[idx])
    dl.append(fh - fl)
dl = np.array(dl)
f1res["boot_bestF1_diff_HCcoh_vs_linres_Difficult"] = dict(mean=round(float(dl.mean()), 3),
    ci=[round(float(np.quantile(dl, .025)), 3), round(float(np.quantile(dl, .975)), 3)], p_le_0=round(float((dl <= 0).mean()), 3))
RES["f1"] = f1res
save(); print("block 4 done; saved", OUTJ, flush=True)
