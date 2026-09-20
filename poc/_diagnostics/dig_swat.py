"""DIG into the SWaT tie (boosted channel-wise LOO 0.881 vs LatAD community fusion 0.837), all on cached artifacts.
 (A) AGGREGATION: fixed poolings over the SAME per-community experts (experts_full), all three datasets.
 (B) COVERAGE: for each SWaT difficult episode, do the attacked channels sit in one community? per-community
     upper-tail p on the episode windows; which community leads; boosted vs headline percentile.
 (C) ACTUATOR ABLATION of the channel-wise baselines: drop all discrete (one-hot) channels / drop only attacked
     actuators, re-score boosted and linear LOO on difficult / double-hard.
 (D) PER-WINDOW dump on SWaT difficult windows where boosted percentile - headline percentile > 0.2 (and < -0.2).
Output: _diagnostics/dig_swat.json. Invariants: HCcoh+LatAD / cohmax+LatAD reproduce ensemble_final (SWaT 0.837/0.829,
HAI 0.845, WADI 0.771 difficult); boosted SWaT difficult 0.881."""
from __future__ import annotations
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("EXPERTS_DIR", os.path.join(ROOT, "sota_bundle", "experts_full"))
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
import eda_real as E
from onehot_filter import build_feats, loco_residual, DMAX
import ensemble_final as EF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}


def au_ms(y, arr, m):
    k = (y == 0) | m
    if arr.ndim == 1:
        return float(roc_auc_score(y[k], arr[k]))
    return float(np.mean([roc_auc_score(y[k], arr[i][k]) for i in range(arr.shape[0])]))


def HC_alpha(P, alpha0, wt=None):
    S, n = P.shape
    if wt is not None:
        P = np.clip(P ** (wt[:, None] / (wt.mean() + 1e-9)), 1e-4, 1.0)
    Ps = np.sort(P, axis=0); i = (np.arange(1, S + 1) / S)[:, None]
    hc = np.sqrt(S) * (i - Ps) / np.sqrt(np.clip(Ps * (1 - Ps), 1e-6, None))
    hc[Ps >= alpha0] = -np.inf
    return np.nan_to_num(hc.max(0), neginf=0.0, posinf=1e6)


def poolings(name):
    d = np.load(f"{EF.OUT}/scores_{name}.npz")
    Ex = np.load(f"{os.environ['EXPERTS_DIR']}/expert_{name}.npz", allow_pickle=True)
    y = d["label"].astype(int); Cal, Tst = Ex["calib_surprise"], Ex["test_surprise"]
    coh = Ex["comm_cohesion"].astype(float); size = Ex["comm_size"].astype(float); w = coh * np.sqrt(size)
    lat, lat_tr = d["LatAD"], d["LatAD_train"]; nseed = min(Tst.shape[0], lat.shape[0]); S = Tst.shape[1]
    wn = w / (w.max() + 1e-9); cn = coh / (coh.max() + 1e-9)
    z = lambda s, r: (s - r.mean()) / (r.std() + 1e-9)
    acc = {}
    for sd in range(nseed):
        tails = np.stack([EF.surv(Cal[sd, g], Tst[sd, g]) for g in range(S)]); tails_c = np.stack([EF.surv(Cal[sd, g], Cal[sd, g]) for g in range(S)])
        P = np.stack([EF.pval(Cal[sd, g], Tst[sd, g]) for g in range(S)]); Pc = np.stack([EF.pval(Cal[sd, g], Cal[sd, g]) for g in range(S)])
        zl = z(EF.surv(lat_tr[sd], lat[sd]), EF.surv(lat_tr[sd], lat_tr[sd]))
        rules = {}
        rules["HCcoh+LatAD (current)"] = z(EF.HC(P, wt=w), EF.HC(Pc, wt=w)) + zl
        rules["cohmax+LatAD"] = z((wn[:, None] * tails).max(0), (wn[:, None] * tails_c).max(0)) + zl
        rules["max+LatAD (Tippett)"] = z(tails.max(0), tails_c.max(0)) + zl
        rules["sum+LatAD (Fisher)"] = z(tails.sum(0), tails_c.sum(0)) + zl
        rules["cohsum+LatAD (coh-weighted Fisher)"] = z((cn[:, None] * tails).sum(0), (cn[:, None] * tails_c).sum(0)) + zl
        rules["wsum+LatAD (coh*sqrt(size) Fisher)"] = z((wn[:, None] * tails).sum(0), (wn[:, None] * tails_c).sum(0)) + zl
        for k in (2, 3, 5):
            tk = np.sort(tails, 0)[-k:].sum(0); tkc = np.sort(tails_c, 0)[-k:].sum(0)
            rules[f"top{k}sum+LatAD"] = z(tk, tkc) + zl
        for a0 in (0.1, 0.2, 0.5):
            rules[f"HCcoh(a0={a0})+LatAD"] = z(HC_alpha(P, a0, w), HC_alpha(Pc, a0, w)) + zl
            rules[f"HC(a0={a0})+LatAD"] = z(HC_alpha(P, a0), HC_alpha(Pc, a0)) + zl
        # community-only (no LatAD null term) for reference
        rules["HCcoh only"] = z(EF.HC(P, wt=w), EF.HC(Pc, wt=w)); rules["cohmax only"] = z((wn[:, None] * tails).max(0), (wn[:, None] * tails_c).max(0))
        rules["LatAD null only"] = zl
        # Stouffer over z-scored tails
        zt = (tails - tails_c.mean(1, keepdims=True)) / (tails_c.std(1, keepdims=True) + 1e-9)
        ztc = (tails_c - tails_c.mean(1, keepdims=True)) / (tails_c.std(1, keepdims=True) + 1e-9)
        rules["stouffer+LatAD"] = z(zt.sum(0) / np.sqrt(S), ztc.sum(0) / np.sqrt(S)) + zl
        rules["cohstouffer+LatAD"] = z((cn[:, None] * zt).sum(0), (cn[:, None] * ztc).sum(0)) + zl
        for k, v in rules.items():
            acc.setdefault(k, []).append(v)
    return {k: np.stack(v) for k, v in acc.items()}, y, d, Ex


def subsets(name, y, d):
    D = E.load(name); fn, W, stride = E.RAW[name]
    Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
    Fn, Fa, grp = build_feats(Xn, Xa, W, stride, onehot=True); Fa = Fa[:len(y)]
    r_tr, r_te = loco_residual(Fn, Fa, grp); lin_thr = float(np.quantile(r_tr, 0.99))
    diff = (y == 1) & (d["maxz"] <= float(d["maxz_thr"])); dhard = diff & (r_te <= lin_thr)
    return D, Xn, Xa, Fn, Fa, grp, r_te, diff, dhard, W, stride


# ---------------- (A) aggregation on all three datasets ----------------
OUT["aggregation"] = {}
CTX = {}
for name in ["SWaT_canon", "WADI_clean", "HAI"]:
    rules, y, d, Ex = poolings(name)
    D, Xn, Xa, Fn, Fa, grp, r_te, diff, dhard, W, stride = subsets(name, y, d)
    CTX[name] = (rules, y, d, Ex, D, Xn, Xa, Fn, Fa, grp, r_te, diff, dhard, W, stride)
    tab = {k: dict(Difficult=round(au_ms(y, v, diff), 3), DoubleHard=round(au_ms(y, v, dhard), 3)) for k, v in rules.items()}
    OUT["aggregation"][name] = tab
    print(f"\n=== (A) aggregation {name} (diff {diff.sum()}, dhard {dhard.sum()}) ===", flush=True)
    for k, v in tab.items():
        print(f"  {k:40s} diff {v['Difficult']:.3f}  dhard {v['DoubleHard']:.3f}", flush=True)
json.dump(OUT, open(os.path.join(HERE, "dig_swat.json"), "w"), indent=1)

# ---------------- (B) coverage on SWaT ----------------
name = "SWaT_canon"
rules, y, d, Ex, D, Xn, Xa, Fn, Fa, grp, r_te, diff, dhard, W, stride = CTX[name]
ch = D["ch"]; nrm = y == 0
comm = [[int(c) for c in row if c >= 0] for row in Ex["comm_channels"]]
coh = Ex["comm_cohesion"].astype(float)
Cal, Tst = Ex["calib_surprise"], Ex["test_surprise"]; S = Tst.shape[1]; nseed = Tst.shape[0]
# seed-mean per-community upper-tail p-value on test
Pm = np.mean([np.stack([EF.pval(Cal[sd, g], Tst[sd, g]) for g in range(S)]) for sd in range(nseed)], 0)  # (S,n)
bz = np.load(os.path.join(HERE, "boosted_loo_SWaT_canon.npz")); b_te = bz["b_te"]
head = rules["HCcoh+LatAD (current)"].mean(0)
pr = lambda s: np.searchsorted(np.sort(s[nrm]), s, side="left") / nrm.sum()
p_b, p_h = pr(b_te), pr(head)
eps_map = [json.loads(l) for l in open(os.path.join(HERE, "framing_loc", "loc2_episodes_SWaT_canon.jsonl"))]
eps = EF.episodes(y); assert len(eps) == len(eps_map), (len(eps), len(eps_map))
ci = {c: i for i, c in enumerate(ch)}
# boosted per-feature residuals for top channels (refit quickly, no crossfit)
Rte_h = np.zeros_like(Fa)
for j in range(Fn.shape[1]):
    cols = np.where(grp != grp[j])[0]
    m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=0).fit(Fn[:, cols], Fn[:, j])
    Rte_h[:, j] = (m.predict(Fa[:, cols]) - Fa[:, j]) ** 2
cov_rows = []
for e, meta in zip(eps, eps_map):
    ed = e[diff[e]]
    if len(ed) == 0:
        continue
    pts = [p for p in meta["points"] if p in ci]; idx = [ci[p] for p in pts]
    containing = [g for g, G in enumerate(comm) if any(c in G for c in idx)]
    together = [g for g, G in enumerate(comm) if all(c in G for c in idx)] if idx else []
    # per-community p on the difficult windows of this episode (seed-mean), best community, and whether it is a containing one
    pmin = Pm[:, ed].min(1); best = int(np.argmin(pmin))
    cont_best = float(pmin[containing].min()) if containing else None
    top_b = [ch[int(grp[j])] for j in Rte_h[ed].argmax(1)]
    cov_rows.append(dict(attacks=meta["attacks"], points=pts, n_diff=int(len(ed)), n_dhard=int(dhard[e].sum()),
                         n_comm_containing=len(containing), n_comm_holding_all=len(together),
                         smallest_comm_holding_all=int(min(len(comm[g]) for g in together)) if together else None,
                         best_comm=best, best_comm_p=round(float(pmin[best]), 4), best_comm_contains_target=best in containing,
                         best_containing_comm_p=cont_best, best_comm_channels=[ch[c] for c in comm[best]][:8],
                         boosted_pct=round(float(np.median(p_b[ed])), 3), headline_pct=round(float(np.median(p_h[ed])), 3),
                         boosted_top1=sorted(set(top_b))))
OUT["coverage_SWaT"] = cov_rows
print("\n=== (B) coverage SWaT difficult episodes ===", flush=True)
for r in cov_rows:
    print("  ", r, flush=True)
n_split = sum(1 for r in cov_rows if r["points"] and r["n_comm_holding_all"] == 0)
OUT["coverage_summary"] = dict(n_episodes=len(cov_rows), n_target_split_across_communities=n_split,
                               n_best_comm_contains_target=sum(1 for r in cov_rows if r["best_comm_contains_target"]))
print("  summary:", OUT["coverage_summary"], flush=True)
json.dump(OUT, open(os.path.join(HERE, "dig_swat.json"), "w"), indent=1)

# ---------------- (C) actuator ablation of the channel-wise baselines on SWaT ----------------
nstates = np.array([len(np.unique(Xn[:, c])) for c in range(Xn.shape[1])])
is_disc_feat = np.array([nstates[g] <= DMAX for g in grp])
attacked = set()
for meta in eps_map:
    attacked |= {ci[p] for p in meta["points"] if p in ci}
att_disc = sorted(c for c in attacked if nstates[c] <= DMAX)
att_feat = np.array([int(g) in attacked for g in grp])


def rescore(mask, tag):
    Fn2, Fa2, g2 = Fn[:, mask], Fa[:, mask], grp[mask]
    _, r2 = loco_residual(Fn2, Fa2, g2)
    Rb = np.zeros_like(Fa2)
    for j in range(Fn2.shape[1]):
        cols = np.where(g2 != g2[j])[0]
        m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=0).fit(Fn2[:, cols], Fn2[:, j])
        Rb[:, j] = (m.predict(Fa2[:, cols]) - Fa2[:, j]) ** 2
    sb = Rb.mean(1)
    r = dict(n_feat=int(mask.sum()), linres_diff=round(au_ms(y, r2, diff), 3), linres_dhard=round(au_ms(y, r2, dhard), 3),
             boosted_diff=round(au_ms(y, sb, diff), 3), boosted_dhard=round(au_ms(y, sb, dhard), 3))
    print(f"  {tag:45s} {r}", flush=True); return r


print("\n=== (C) actuator ablation SWaT ===", flush=True)
OUT["actuator_ablation_SWaT"] = {
    "attacked_discrete_channels": [ch[c] for c in att_disc],
    "all features (canonical)": rescore(np.ones(len(grp), bool), "all features (canonical)"),
    "drop ALL discrete/actuator channels": rescore(~is_disc_feat, "drop ALL discrete/actuator channels"),
    "drop attacked actuator channels only": rescore(~np.array([int(g) in att_disc for g in grp]), "drop attacked actuator channels only"),
    "drop ALL attacked channels (any type)": rescore(~att_feat, "drop ALL attacked channels (any type)"),
}
json.dump(OUT, open(os.path.join(HERE, "dig_swat.json"), "w"), indent=1)

# ---------------- (D) per-window dump ----------------
rows = []
for w in np.where(diff)[0]:
    gap = p_b[w] - p_h[w]
    if abs(gap) < 0.2:
        continue
    e_i = next(i for i, e in enumerate(eps) if w in e); meta = eps_map[e_i]
    j = int(Rte_h[w].argmax()); shares = Rte_h[w] / Rte_h[w].sum()
    order = np.argsort(-Rte_h[w])[:3]
    best = int(np.argmin(Pm[:, w]))
    pts = [p for p in meta["points"] if p in ci]; idx = [ci[p] for p in pts]
    rows.append(dict(w=int(w), attacks=meta["attacks"], points=pts, gap=round(float(gap), 3), boosted_pct=round(float(p_b[w]), 3),
                     headline_pct=round(float(p_h[w]), 3), dhard=bool(dhard[w]),
                     boosted_top3=[(ch[int(grp[k])], round(float(shares[k]), 2)) for k in order],
                     best_comm=best, best_comm_p=round(float(Pm[best, w]), 4),
                     best_comm_has_target=any(c in comm[best] for c in idx),
                     target_comm_min_p=round(float(min(Pm[g, w] for g, G in enumerate(comm) if any(c in G for c in idx))), 4) if idx else None,
                     n_comm_p_lt_001=int((Pm[:, w] < 0.01).sum())))
OUT["perwindow_SWaT"] = rows
print("\n=== (D) per-window (|gap|>0.2) ===", flush=True)
for r in rows:
    print("  ", r, flush=True)
json.dump(OUT, open(os.path.join(HERE, "dig_swat.json"), "w"), indent=1)
print("saved", flush=True)
