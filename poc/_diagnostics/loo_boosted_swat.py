"""Try to break the boosted channel-wise LOO result on SWaT_canon (difficult AUROC 0.881 > LatAD headline).
(1) episode-block bootstrap (ensemble_final.boot, identical construction) of boosted-LOO minus the headline
    ensembles (HCcoh+LatAD, null+HC) and minus global LatAD, on Difficult and DoubleHard;
(2) DoubleHard re-score of the boosted LOO;
(3) PIT502: drop it (as target and predictor / as target only), score it alone (|z| of its window mean,
    its own boosted self-prediction residual), per-episode attribution;
(4) 5-fold cross-fitted train residuals so the boosted LOO has a fair train-p99 (in-sample HGB residuals are
    optimistic). Invariants: I1 linear LOO reproduces npz linres; I2 headline AUROCs reproduce
    headline_clean_results.md (HCcoh+LatAD diff 0.840, dhard 0.775 on the n=85/59 subsets or the current
    scores_SWaT_canon subsets); I3 boosted LOO reproduces loo_baseline_why (0.881 difficult).
Output: _diagnostics/loo_boosted_swat.json"""
from __future__ import annotations
import os, sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("EXPERTS_DIR", os.path.join(ROOT, "sota_bundle", "experts_full"))
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import eda_real as E
from onehot_filter import build_feats, loco_residual
import ensemble_final as EF

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "SWaT_canon"; REPS = int(os.environ.get("BOOT_REPS", "2000"))


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k]))


def au_ms(y, arr, m):
    k = (y == 0) | m
    return float(np.mean([roc_auc_score(y[k], arr[i][k]) for i in range(arr.shape[0])]))


def hgb():
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=0)


def boosted_loco(Fn, Fa, grp, crossfit=True, targets=None):
    """per-feature squared residual on test (Rte) and cross-fitted train (Rtr_cv). targets: feature idx subset."""
    tg = range(Fn.shape[1]) if targets is None else targets
    Rte = np.zeros((len(Fa), Fn.shape[1])); Rtr = np.zeros_like(Fn)
    kf = KFold(5, shuffle=True, random_state=0)
    for j in tg:
        cols = np.where(grp != grp[j])[0]
        m = hgb().fit(Fn[:, cols], Fn[:, j])
        Rte[:, j] = (m.predict(Fa[:, cols]) - Fa[:, j]) ** 2
        if crossfit:
            for tr, va in kf.split(Fn):
                mm = hgb().fit(Fn[tr][:, cols], Fn[tr, j])
                Rtr[va, j] = (mm.predict(Fn[va][:, cols]) - Fn[va, j]) ** 2
    return Rtr, Rte


t0 = time.time()
ens, y, d, nseed = EF.ensemble_scores(NAME)
fn, W, stride = E.RAW[NAME]; D = E.load(NAME); ch = D["ch"]
Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
maxz, mthr = d["maxz"], float(d["maxz_thr"])
Fn, Fa, grp = build_feats(Xn, Xa, W, stride, onehot=True); Fa = Fa[:len(y)]
r_tr, r_te = loco_residual(Fn, Fa, grp); lin_thr = float(np.quantile(r_tr, 0.99))
diff = (y == 1) & (maxz <= mthr); dhard = diff & (r_te <= lin_thr); nrm = y == 0
res = dict(n_diff=int(diff.sum()), n_dhard=int(dhard.sum()),
           I1_linres_relerr=float(np.max(np.abs(r_te - d["linres"]) / (np.abs(d["linres"]) + 1e-6))))
print(f"subsets: diff {diff.sum()} dhard {dhard.sum()}  I1={res['I1_linres_relerr']:.1e}", flush=True)

# ---- boosted LOO (all features), cross-fitted train residuals ----
Rtr_h, Rte_h = boosted_loco(Fn, Fa, grp)
sh = Rte_h.mean(1); sh_tr = Rtr_h.mean(1); hthr = float(np.quantile(sh_tr, 0.99))
res["boosted_diff"], res["boosted_dhard"], res["boosted_all"] = au(y, sh, diff), au(y, sh, dhard), au(y, sh, y == 1)
res["boosted_testnormal_FP_at_cv_train_p99"] = float((sh[nrm] > hthr).mean())
res["boosted_diff_TP_at_cv_train_p99"] = float((sh[diff] > hthr).mean())
res["linres_diff"], res["linres_dhard"] = au(y, r_te, diff), au(y, r_te, dhard)
print(f"boosted: diff {res['boosted_diff']:.3f} dhard {res['boosted_dhard']:.3f} all {res['boosted_all']:.3f} "
      f"FP@cvp99 {res['boosted_testnormal_FP_at_cv_train_p99']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
# nonlinear double-hard: also below the boosted cv-train p99
nldh = diff & (sh <= hthr)
res["n_nonlinear_dhard"] = int(nldh.sum())

# ---- headline / competitors on the same subsets ----
comp = {"LatAD_global": d["LatAD"], "HCcoh+LatAD": ens["HCcoh+LatAD"], "null+HC": ens["null+HC"], "cohmax+LatAD": ens["cohmax+LatAD"]}
for k, v in comp.items():
    res[f"{k}_diff"], res[f"{k}_dhard"] = au_ms(y, v, diff), au_ms(y, v, dhard)
    res[f"{k}_nonlinear_dhard"] = au_ms(y, v, nldh) if nldh.sum() >= 3 else None
    print(f"{k}: diff {res[f'{k}_diff']:.3f} dhard {res[f'{k}_dhard']:.3f}", flush=True)
res["boosted_nonlinear_dhard"] = au(y, sh, nldh) if nldh.sum() >= 3 else None
res["linres_nonlinear_dhard"] = au(y, r_te, nldh) if nldh.sum() >= 3 else None

# ---- (1) episode-block bootstrap: boosted minus headline ----
L = int(np.ceil(W / stride)) + 1
res["boot"] = {}
for k in ("HCcoh+LatAD", "null+HC", "LatAD_global"):
    for sub, mk in (("Difficult", diff), ("DoubleHard", dhard)):
        EF.RNG = np.random.default_rng(0)
        b = EF.boot(y, sh, comp[k], mk, L, reps=REPS)
        res["boot"][f"boosted_minus_{k}_{sub}"] = b
        print(f"  boot boosted - {k} [{sub}]: diff {b['diff']} CI {b['diff_ci']} P(<=0)={b['p_le_0']} eps {b['n_episodes']}", flush=True)
# reverse direction one-sided P for the headline leading on DoubleHard
for k in ("HCcoh+LatAD", "null+HC"):
    EF.RNG = np.random.default_rng(0)
    b = EF.boot(y, comp[k], sh, dhard, L, reps=REPS)
    res["boot"][f"{k}_minus_boosted_DoubleHard"] = b
    print(f"  boot {k} - boosted [DoubleHard]: diff {b['diff']} CI {b['diff_ci']} P(<=0)={b['p_le_0']}", flush=True)

# ---- (3) PIT502 ----
nstates = np.array([len(np.unique(Xn[:, c])) for c in range(Xn.shape[1])])
gname = {int(g): ch[int(g)] for g in np.unique(grp)}
pit = [g for g, n in gname.items() if n == "PIT502"]
assert len(pit) == 1, f"PIT502 not found among used channels: {sorted(gname.values())}"
pit = pit[0]; jp = np.where(grp == pit)[0]; assert len(jp) == 1
jp = int(jp[0])
# raw-unit share of PIT502 in the boosted score on difficult windows
share = Rte_h[:, jp] / (Rte_h.sum(1) + 1e-12)
res["PIT502_share_boosted_diff_med"] = float(np.median(share[diff]))
res["PIT502_share_boosted_normal_med"] = float(np.median(share[nrm]))
res["PIT502_is_top1_boosted_diff_count"] = int((Rte_h[diff].argmax(1) == jp).sum())
# (a) drop PIT502 as target only (keep as predictor)
keep_t = [j for j in range(Fn.shape[1]) if j != jp]
s_not = Rte_h[:, keep_t].mean(1)
res["boosted_dropPIT502_target_diff"], res["boosted_dropPIT502_target_dhard"] = au(y, s_not, diff), au(y, s_not, dhard)
# (a') drop PIT502 entirely (target and predictor): refit
mk = grp != pit
_, Rte_np = boosted_loco(Fn[:, mk], Fa[:, mk], grp[mk], crossfit=False)
s_np = Rte_np.mean(1)
res["boosted_dropPIT502_all_diff"], res["boosted_dropPIT502_all_dhard"] = au(y, s_np, diff), au(y, s_np, dhard)
# (b) PIT502 alone
z_pit = np.abs(Fa[:, jp])                                  # train-standardised window mean
res["PIT502_absz_diff"], res["PIT502_absz_dhard"] = au(y, z_pit, diff), au(y, z_pit, dhard)
res["PIT502_boosted_selfresid_diff"], res["PIT502_boosted_selfresid_dhard"] = au(y, Rte_h[:, jp], diff), au(y, Rte_h[:, jp], dhard)
res["PIT502_linear_selfresid_diff"] = au(y, ((LinearRegression().fit(Fn[:, grp != pit], Fn[:, jp]).predict(Fa[:, grp != pit]) - Fa[:, jp]) ** 2), diff)
# PIT502 in the difficult windows: raw value stats vs train
res["PIT502_train_mean_std_raw"] = [float(Xn[:, pit].mean()), float(Xn[:, pit].std())]
res["PIT502_testnormal_z_med"], res["PIT502_diff_z_med"] = float(np.median(Fa[nrm, jp])), float(np.median(Fa[diff, jp]))
print(f"PIT502: share diff {res['PIT502_share_boosted_diff_med']:.2f} normal {res['PIT502_share_boosted_normal_med']:.2f}; "
      f"drop-target diff {res['boosted_dropPIT502_target_diff']:.3f}; drop-all diff {res['boosted_dropPIT502_all_diff']:.3f}; "
      f"|z| alone diff {res['PIT502_absz_diff']:.3f}; self-resid alone diff {res['PIT502_boosted_selfresid_diff']:.3f}", flush=True)
# bootstrap of the drop-PIT502 boosted score vs headline
for sub, mk2 in (("Difficult", diff), ("DoubleHard", dhard)):
    EF.RNG = np.random.default_rng(0)
    b = EF.boot(y, s_np, comp["HCcoh+LatAD"], mk2, L, reps=REPS)
    res["boot"][f"boosted_dropPIT502_minus_HCcoh+LatAD_{sub}"] = b
    print(f"  boot boosted(no PIT502) - HCcoh+LatAD [{sub}]: diff {b['diff']} CI {b['diff_ci']} P(<=0)={b['p_le_0']}", flush=True)

# ---- per-episode inspection on difficult windows ----
pr = lambda s: np.searchsorted(np.sort(s[nrm]), s, side="left") / nrm.sum()
p_b, p_h, p_pit = pr(sh), pr(ens["HCcoh+LatAD"].mean(0)), pr(Rte_h[:, jp])
idx_a = D["idx_a"]
ep_rows = []
for e in EF.episodes(y):
    ed = e[diff[e]]
    if len(ed) == 0:
        continue
    top = [gname[int(grp[j])] for j in Rte_h[ed].argmax(1)]
    ep_rows.append(dict(start_win=int(e[0]), start_row=int(idx_a[e[0]]), n_ep=int(len(e)), n_diff=int(len(ed)),
                        n_dhard=int(dhard[e].sum()), boosted_pct=round(float(np.median(p_b[ed])), 3),
                        headline_pct=round(float(np.median(p_h[ed])), 3), PIT502_pct=round(float(np.median(p_pit[ed])), 3),
                        PIT502_share=round(float(np.median(share[ed])), 2), PIT502_z=round(float(np.median(Fa[ed, jp])), 2),
                        top1_channels=sorted(set(top))))
res["episodes"] = ep_rows
for r in ep_rows:
    print("  ", r, flush=True)
res["elapsed_s"] = round(time.time() - t0, 1)
json.dump(res, open(os.path.join(HERE, "loo_boosted_swat.json"), "w"), indent=1)
print("saved", flush=True)
