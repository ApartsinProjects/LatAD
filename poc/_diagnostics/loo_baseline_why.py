"""Why is the leave-one-channel-out linear baseline (LinRes one-hot) so strong on WADI_clean and
weaker on HAI? Per-window / per-channel decomposition of the LOO residual, concentration, effective
rank, linear-fit tightness, and a nonlinear / per-regime LOO ablation. Writes
_diagnostics/loo_baseline_why.json + prints. Invariants: (I1) recomputed linres == npz linres;
(I2) recomputed AUROCs == headline_clean_results.md (WADI_clean All 0.834 / difficult 0.787)."""
from __future__ import annotations
import os, sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
import eda_real as E
from onehot_filter import build_feats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k]))


def pct_rank(s_norm, s):
    """fraction of test-normal scores strictly below s (per-window detectability)."""
    sn = np.sort(s_norm)
    return np.searchsorted(sn, s, side="left") / len(sn)


def loco_matrix(Fn, Fa, grp, model="lin"):
    Rtr = np.zeros_like(Fn); Rte = np.zeros_like(Fa); R2 = np.zeros(Fn.shape[1])
    for j in range(Fn.shape[1]):
        cols = np.where(grp != grp[j])[0]
        if model == "lin":
            m = LinearRegression().fit(Fn[:, cols], Fn[:, j])
        else:
            m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=0).fit(Fn[:, cols], Fn[:, j])
        ptr = m.predict(Fn[:, cols]); pte = m.predict(Fa[:, cols])
        Rtr[:, j] = (ptr - Fn[:, j]) ** 2; Rte[:, j] = (pte - Fa[:, j]) ** 2
        v = Fn[:, j].var(); R2[j] = 1 - Rtr[:, j].mean() / v if v > 0 else np.nan
    return Rtr, Rte, R2


def episodes(mask):
    idx = np.where(mask)[0]; eps = []
    for i in idx:
        if eps and i == eps[-1][-1] + 1:
            eps[-1].append(int(i))
        else:
            eps.append([int(i)])
    return eps


def run(name, do_nonlin=True):
    t0 = time.time()
    D = E.load(name); fn, W, stride = E.RAW[name]; ch = D["ch"]
    Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
    Z = np.load(os.path.join(HERE, f"scores_{name}.npz"), allow_pickle=True)
    y = Z["label"].astype(int); maxz, thr = Z["maxz"], float(Z["maxz_thr"])
    diff = (y == 1) & (maxz <= thr); easy = (y == 1) & (maxz > thr); nrm = y == 0
    Fn, Fa, grp = build_feats(Xn, Xa, W, stride, onehot=True); Fa = Fa[:len(y)]
    Rtr, Rte, R2 = loco_matrix(Fn, Fa, grp)
    linres = Rte.mean(1)
    i1 = float(np.max(np.abs(linres - Z["linres"]) / (np.abs(Z["linres"]) + 1e-6)))
    res = dict(name=name, n_win=int(len(y)), n_anom=int(y.sum()), n_diff=int(diff.sum()), n_easy=int(easy.sum()),
               n_ch_raw=len(ch), n_feat=int(Fn.shape[1]), n_ch_used=int(len(np.unique(grp))),
               I1_max_rel_err_linres=i1)
    # ---- AUROCs (I2) ----
    lat = Z["LatAD"].mean(0)
    for nm, s in [("linres", linres), ("LatAD_meanseed", lat), ("AE_meanseed", Z["AE"].mean(0)),
                  ("USAD", Z["USAD"]), ("TranAD", Z["TranAD"]), ("maxz", maxz)]:
        res[f"au_all_{nm}"] = au(y, s, y == 1); res[f"au_diff_{nm}"] = au(y, s, diff)
    res["au_diff_LatAD_perseed"] = [au(y, Z["LatAD"][k], diff) for k in range(Z["LatAD"].shape[0])]
    # ---- per-feature residual normalised by its train RMS -> channel attribution ----
    scale = Rtr.mean(0) + 1e-12
    Rn = Rte / scale                                   # each col ~1 on train-normal

    def chan_share(rows):
        out = []
        for r in rows:
            cs = {}
            for j, g in enumerate(grp):
                cs[g] = cs.get(g, 0.0) + Rn[r, j]
            items = sorted(cs.items(), key=lambda kv: -kv[1])
            v = np.array([x for _, x in items]); p = v / v.sum()
            H = -(p[p > 0] * np.log(p[p > 0])).sum()
            out.append(dict(top1=float(p[0]), top3=float(p[:3].sum()), n_eff=float(np.exp(H)),
                            n_above3=int((v > 3).sum()), n_above10=int((v > 10).sum()),
                            top_ch=[ch[k] for k, _ in items[:3]],
                            top_vals=[round(float(x), 1) for x in v[:3]]))
        return out

    for tag, m in [("diff", diff), ("easy", easy), ("normal", nrm)]:
        rows = np.where(m)[0]
        if tag == "normal":
            rows = rows[np.linspace(0, len(rows) - 1, min(400, len(rows))).astype(int)]
        cs = chan_share(rows)
        res[f"conc_{tag}"] = dict(n=len(cs),
                                 top1_med=float(np.median([c["top1"] for c in cs])),
                                 top3_med=float(np.median([c["top3"] for c in cs])),
                                 n_eff_med=float(np.median([c["n_eff"] for c in cs])),
                                 n_above3_med=float(np.median([c["n_above3"] for c in cs])),
                                 n_above10_med=float(np.median([c["n_above10"] for c in cs])))
    # ---- per difficult window: who wins, LinRes or LatAD, and which channel drives LinRes ----
    pr_lin = pct_rank(linres[nrm], linres); pr_lat = pct_rank(lat[nrm], lat)
    thr_lin = np.quantile(Rtr.mean(1), 0.99)
    rows = np.where(diff)[0]; cs = chan_share(rows)
    per = []
    for r, c in zip(rows, cs):
        per.append(dict(w=int(r), lin_pct=round(float(pr_lin[r]), 3), lat_pct=round(float(pr_lat[r]), 3),
                        lin_caught_p99=bool(linres[r] > thr_lin), **c))
    res["diff_windows"] = per
    res["diff_lin_caught_p99"] = int(sum(p["lin_caught_p99"] for p in per))
    res["diff_lin_pct_med"] = float(np.median([p["lin_pct"] for p in per]))
    res["diff_lat_pct_med"] = float(np.median([p["lat_pct"] for p in per]))
    res["diff_lin_wins"] = int(sum(p["lin_pct"] > p["lat_pct"] + 0.05 for p in per))
    res["diff_lat_wins"] = int(sum(p["lat_pct"] > p["lin_pct"] + 0.05 for p in per))
    st = [p for p in per if p["lin_pct"] >= 0.95]; wk = [p for p in per if p["lin_pct"] < 0.8]
    res["diff_conc_lin_strong"] = dict(n=len(st), top1_med=float(np.median([p["top1"] for p in st])) if st else None,
                                       n_eff_med=float(np.median([p["n_eff"] for p in st])) if st else None)
    res["diff_conc_lin_weak"] = dict(n=len(wk), top1_med=float(np.median([p["top1"] for p in wk])) if wk else None,
                                     n_eff_med=float(np.median([p["n_eff"] for p in wk])) if wk else None)
    # ---- tightness of the violated linear invariant: R2 of the top channel's LOO fit ----
    chR2 = {}
    for j, g in enumerate(grp):
        chR2.setdefault(int(g), []).append(R2[j])
    chR2 = {g: float(np.nanmean(v)) for g, v in chR2.items()}
    name2g = {ch[g]: g for g in chR2}
    res["R2_top_channel_diff_med"] = float(np.median([chR2[name2g[p["top_ch"][0]]] for p in per]))
    res["R2_all_channels_med"] = float(np.median(list(chR2.values())))
    res["R2_all_channels_frac_gt0.9"] = float(np.mean([v > 0.9 for v in chR2.values()]))
    # ---- effective rank (participation ratio) of the difficult-window deviations ----
    mu, sd = Fn.mean(0), Fn.std(0) + 1e-9

    def prank(rows):
        if len(rows) < 3:
            return None
        Dv = (Fa[rows] - mu) / sd
        ev = np.linalg.svd(Dv - Dv.mean(0), compute_uv=False) ** 2
        return float(ev.sum() ** 2 / (ev ** 2).sum())

    res["prank_diff"] = prank(np.where(diff)[0]); res["prank_easy"] = prank(np.where(easy)[0])
    rn = np.where(nrm)[0]
    res["prank_normal_matched"] = prank(rn[np.random.RandomState(0).choice(len(rn), min(len(rn), max(3, int(diff.sum()))), replace=False)])
    # ---- per-episode summary ----
    eps = episodes(diff); ep_rows = []
    for e in eps:
        ps = [p for p in per if p["w"] in e]
        tops = {}
        for p in ps:
            for k, v in zip(p["top_ch"], p["top_vals"]):
                tops[k] = tops.get(k, 0) + v
        ep_rows.append(dict(start=e[0], n=len(e), lin_pct_med=round(float(np.median([p["lin_pct"] for p in ps])), 3),
                            lat_pct_med=round(float(np.median([p["lat_pct"] for p in ps])), 3),
                            n_eff_med=round(float(np.median([p["n_eff"] for p in ps])), 2),
                            top_ch=[k for k, _ in sorted(tops.items(), key=lambda kv: -kv[1])[:3]]))
    res["episodes"] = ep_rows
    # ---- ablations: per-regime linear LOO, nonlinear (HGB) LOO ----
    km = KMeans(n_clusters=8, n_init=3, random_state=0).fit(Fn)
    ctr, cte = km.labels_, km.predict(Fa)
    r_reg = np.zeros(len(Fa)); n_fallback = 0
    for k in range(km.n_clusters):
        mtr, mte = ctr == k, cte == k
        if mtr.sum() < 3 * Fn.shape[1] or mte.sum() == 0:
            r_reg[mte] = linres[mte]; n_fallback += int(mte.sum()); continue
        _, Rk, _ = loco_matrix(Fn[mtr], Fa[mte], grp)
        r_reg[mte] = Rk.mean(1)
    res["per_regime_fallback_windows"] = n_fallback
    res["au_diff_linres_per_regime"] = au(y, r_reg, diff); res["au_all_linres_per_regime"] = au(y, r_reg, y == 1)
    if do_nonlin:
        _, Rh, _ = loco_matrix(Fn, Fa, grp, model="hgb"); rh = Rh.mean(1)
        res["au_diff_linres_hgb"] = au(y, rh, diff); res["au_all_linres_hgb"] = au(y, rh, y == 1)
    res["elapsed_s"] = round(time.time() - t0, 1)
    OUT[name] = res
    keys = [k for k in res if k not in ("diff_windows", "episodes")]
    print(f"\n=== {name} ===", flush=True)
    for k in keys:
        print(f"  {k}: {res[k]}", flush=True)
    print("  episodes:", flush=True)
    for e in ep_rows:
        print("   ", e, flush=True)
    json.dump(OUT, open(os.path.join(HERE, "loo_baseline_why.json"), "w"), indent=1)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI_clean", "SWaT_canon", "HAI"]):
        run(nm)
