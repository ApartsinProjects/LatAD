"""Per-window root cause on SWaT difficult windows where the LatAD headline (HCcoh+LatAD, experts_full) scores
BELOW the boosted channel-wise LOO (percentile vs test normals). For each: attacked points (iTrust), community
structure (containing communities, target-community min p, best community), boosted top-3 drivers, whether the
top-1 / any top-3 driver is a directly attacked point, a physically adjacent channel (same or next stage) or an
unrelated channel, a rule-based root cause and avoidance lever. Then the right-channel vs wrong-reason fraction of
boosted's edge and bootstraps restricted to the honest-attack and wrong-reason subsets. Also per-seed spread of
the headline for the Modal artifact vs the local rebuild. Output: dig_swat_perwindow.json"""
from __future__ import annotations
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ["EXPERTS_DIR"] = os.path.join(ROOT, "sota_bundle", "experts_full")
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor
import eda_real as E
from onehot_filter import build_feats, loco_residual
import ensemble_final as EF

HERE = os.path.dirname(os.path.abspath(__file__)); name = "SWaT_canon"
ens, y, d, nseed = EF.ensemble_scores(name)
D = E.load(name); ch = D["ch"]; fn, W, stride = E.RAW[name]
Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
Fn, Fa, grp = build_feats(Xn, Xa, W, stride, onehot=True); Fa = Fa[:len(y)]
r_tr, r_te = loco_residual(Fn, Fa, grp); lin_thr = float(np.quantile(r_tr, 0.99))
diff = (y == 1) & (d["maxz"] <= float(d["maxz_thr"])); dhard = diff & (r_te <= lin_thr); nrm = y == 0
Ex = np.load(f"{os.environ['EXPERTS_DIR']}/expert_{name}.npz", allow_pickle=True)
comm = [[int(c) for c in row if c >= 0] for row in Ex["comm_channels"]]
Cal, Tst = Ex["calib_surprise"], Ex["test_surprise"]; S = Tst.shape[1]
calsd = Cal.std(2); degen = (calsd < 1e-3).any(0)
Pm = np.mean([np.stack([EF.pval(Cal[sd, g], Tst[sd, g]) for g in range(S)]) for sd in range(nseed)], 0)
b_te = np.load(os.path.join(HERE, "boosted_loo_SWaT_canon.npz"))["b_te"]
head = ens["HCcoh+LatAD"]; head_m = head.mean(0)
pr = lambda s: np.searchsorted(np.sort(s[nrm]), s, side="left") / nrm.sum()
p_b, p_h = pr(b_te), pr(head_m)
Rte = np.zeros_like(Fa)
for j in range(Fn.shape[1]):
    cols = np.where(grp != grp[j])[0]
    m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=0).fit(Fn[:, cols], Fn[:, j])
    Rte[:, j] = (m.predict(Fa[:, cols]) - Fa[:, j]) ** 2
# channel-level residual (sum over a channel's features)
chan_of_feat = grp; chans = sorted(set(int(g) for g in grp))
Rch = np.stack([Rte[:, grp == c].sum(1) for c in chans], 1)      # (n, nchan_used)
eps_map = [json.loads(l) for l in open(os.path.join(HERE, "framing_loc", "loc2_episodes_SWaT_canon.jsonl"))]
eps = EF.episodes(y); assert len(eps) == len(eps_map)
ci = {c: i for i, c in enumerate(ch)}
stage = lambda c: int(''.join(x for x in c if x.isdigit())[0]) if any(x.isdigit() for x in c) else -1
rows = []
for e, meta in zip(eps, eps_map):
    pts = [p for p in meta["points"] if p in ci]; idx = [ci[p] for p in pts]; st = {stage(p) for p in pts}
    containing = [g for g in range(S) if any(c in comm[g] for c in idx) and not degen[g]]
    holding_all = [g for g in range(S) if all(c in comm[g] for c in idx) and not degen[g]] if idx else []
    for w in e:
        if not diff[w] or p_h[w] >= p_b[w]:
            continue
        sh = Rch[w] / Rch[w].sum(); order = np.argsort(-sh)[:3]
        drivers = [(ch[chans[k]], round(float(sh[k]), 2)) for k in order]
        def cls(c):
            if c in pts: return "direct"
            if stage(c) in st or (stage(c) - 1) in st or (stage(c) + 1) in st: return "adjacent"
            return "unrelated"
        c1 = cls(drivers[0][0]); c3 = [cls(c) for c, _ in drivers]
        pnd = Pm[:, w].copy(); pnd[degen] = 1.0; best = int(np.argmin(pnd))
        tgt_p = float(min(Pm[g, w] for g in containing)) if containing else None
        n_fire = int((pnd < 0.01).sum())
        # root cause / lever
        if not pts:
            rc, lever = "unlabelled label run (no iTrust attack row); both detectors low", "d"
        elif tgt_p is not None and tgt_p <= 0.001 and n_fire >= 2:
            rc, lever = "target community fires (p<=0.001) but the HC fusion rank is below boosted: fusion-scale, not detection", "b"
        elif tgt_p is not None and tgt_p <= 0.01:
            rc, lever = "target community fires weakly (p<=0.01); one-community violation ranks low under HC", "b"
        elif not holding_all and len(pts) > 1:
            rc, lever = "multi-point attack split across communities; no community sees the joint violation", "a"
        elif c1 == "direct" or c1 == "adjacent":
            rc, lever = "target community quiet; boosted reads a nonlinear channel-wise relation of the attacked/adjacent channel", "c"
        else:
            rc, lever = "target community quiet; boosted fires on an unrelated channel (offset), not on the attack", "wrong-reason"
        rows.append(dict(w=int(w), attacks=meta["attacks"], points=pts, dhard=bool(dhard[w]),
                         boosted_pct=round(float(p_b[w]), 3), headline_pct=round(float(p_h[w]), 3), gap=round(float(p_b[w] - p_h[w]), 3),
                         n_comm_containing=len(containing), holds_all=bool(holding_all), target_comm_min_p=None if tgt_p is None else round(tgt_p, 4),
                         best_comm=best, best_comm_channels=[ch[c] for c in comm[best]][:6], best_comm_p=round(float(pnd[best]), 4),
                         n_comm_fire=n_fire, drivers=drivers, top1_class=c1, top3_classes=c3,
                         any_top3_direct=("direct" in c3), root_cause=rc, lever=lever))
n = len(rows)
lab = [r for r in rows if r["points"]]
frac = dict(n_losing=n, n_labelled=len(lab),
            top1_direct=sum(r["top1_class"] == "direct" for r in lab) / len(lab),
            top1_adjacent=sum(r["top1_class"] == "adjacent" for r in lab) / len(lab),
            top1_unrelated=sum(r["top1_class"] == "unrelated" for r in lab) / len(lab),
            any_top3_direct=sum(r["any_top3_direct"] for r in lab) / len(lab),
            gap_weighted_top1_direct_or_adjacent=sum(r["gap"] for r in lab if r["top1_class"] != "unrelated") / sum(r["gap"] for r in lab),
            levers={k: sum(r["lever"] == k for r in rows) for k in ("a", "b", "c", "d", "wrong-reason")})
# restricted bootstraps (headline minus boosted)
L = int(np.ceil(W / stride)) + 1
honest = np.zeros(len(y), bool); wrong = np.zeros(len(y), bool)
for r in rows:
    (honest if r["top1_class"] != "unrelated" else wrong)[r["w"]] = True
# honest-attack subset over ALL difficult windows: boosted's top-1 driver is direct or adjacent (or LatAD wins)
allhon = np.zeros(len(y), bool)
for e, meta in zip(eps, eps_map):
    pts = [p for p in meta["points"] if p in ci]; st = {stage(p) for p in pts}
    for w in e:
        if diff[w] and pts:
            c = ch[chans[int(np.argmax(Rch[w]))]]
            if c in pts or stage(c) in st or (stage(c) - 1) in st or (stage(c) + 1) in st:
                allhon[w] = True
boots = {}
for tag, mk in [("Difficult (all)", diff), ("DoubleHard (all)", dhard), ("Difficult, boosted top-1 direct/adjacent", diff & allhon),
                ("DoubleHard, boosted top-1 direct/adjacent", dhard & allhon), ("Difficult, losing windows only", diff & (honest | wrong)),
                ("Difficult, losing & honest", diff & honest), ("Difficult, losing & wrong-reason", diff & wrong)]:
    if mk.sum() < 3:
        boots[tag] = None; continue
    EF.RNG = np.random.default_rng(0); b = EF.boot(y, head, b_te, mk, L, reps=2000); b["n"] = int(mk.sum()); boots[tag] = b
# per-seed spread, Modal vs local rebuild
def au(y, s, m):
    k = (y == 0) | m; return float(roc_auc_score(y[k], s[k]))
seeds = {"modal_full": [round(au(y, head[i], diff), 3) for i in range(head.shape[0])]}
os.environ["EXPERTS_DIR"] = os.path.join(ROOT, "sota_bundle", "experts_variants", "cur_avg25")
ensl, _, _, _ = EF.ensemble_scores(name); hl = ensl["HCcoh+LatAD"]
seeds["local_cur_avg25"] = [round(au(y, hl[i], diff), 3) for i in range(hl.shape[0])]
seeds["pooled_10seed_mean"] = round(float(np.mean(seeds["modal_full"] + seeds["local_cur_avg25"])), 3)
out = dict(rows=rows, fractions=frac, boots=boots, seed_spread=seeds)
json.dump(out, open(os.path.join(HERE, "dig_swat_perwindow.json"), "w"), indent=1)
for r in rows:
    print(r)
print("\nfractions:", json.dumps(frac, indent=1))
print("\nboots (headline - boosted):")
for k, v in boots.items():
    print("  ", k, v)
print("\nseed spread:", seeds)
