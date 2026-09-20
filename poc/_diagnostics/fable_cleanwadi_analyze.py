"""Error analysis on a cleaned-WADI score table (default: the clip=10 rebuild).
  1. canonical difficult-subset AUROC table (paper construction: difficult anomalies vs ALL test normals),
     per seed + paired bootstrap CI on LatAD-AE; also the mis-constructed 'maxz>thr window' subset.
  2. head ablation from stored components (auto-gated = reported; forced variants shown, flagged).
  3. coverage audit on the CLEAN clip=10 loader: 1-NN distance to train-normal (train-LOO p99 = OOS);
     classify LatAD's difficult-subset false alarms; OOS-normal exclusion test (0 anomalies removed).
  4. duplicate / mislabel audit of the difficult anomalies.
  5. per-window table: difficult anomalies AE catches & LatAD misses (and vice versa), with attack
     segment and top deviating channels.
Usage: python fable_cleanwadi_analyze.py [scores.npz]
"""
from __future__ import annotations
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.dirname(HERE); sys.path.insert(0, POC); os.chdir(POC)
from sklearn.metrics import roc_auc_score as auc
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import eda_real as E

P = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "fable_cleanwadi_scores_clip10_default.npz")
S = np.load(P); y = S["label"].astype(int); maxz = S["maxz"]; thr = float(S["maxz_thr"])
clip = float(S["clip"]) if "clip" in S.files else None
print(f"table: {P}\n  clip={clip} K={S['K'] if 'K' in S.files else '?'} LD={S['LD'] if 'LD' in S.files else '?'}")
hard = (y == 1) & (maxz <= thr); easy = (y == 1) & (maxz > thr); norm = y == 0
KH, KE = norm | hard, norm | easy
print(f"  n_win={len(y)} normals={norm.sum()} anomalies={y.sum()} = easy {easy.sum()} + difficult {hard.sum()}  thr={thr:.3f}")
REPORT = dict(table=P, clip=clip, n_normal=int(norm.sum()), n_hard=int(hard.sum()), n_easy=int(easy.sum()))
rng = np.random.default_rng(0)

def M(k):
    a = S[k]; return a if a.ndim == 2 else a[None]
methods = [k for k in ("LatAD", "AE", "IF", "linres", "l2", "USAD", "TranAD") if k in S.files]

def table(mask, title):
    out = {}
    print(f"\n== {title}: {int((y[mask]==1).sum())} anomalies vs {int((y[mask]==0).sum())} normals")
    for k in methods:
        per = [auc(y[mask], s[mask]) for s in M(k)]
        out[k] = dict(mean=float(np.mean(per)), std=float(np.std(per)), per_seed=[float(p) for p in per])
        print(f"  {k:7s} {np.mean(per):.3f} +- {np.std(per):.3f}  {np.round(per,3)}")
    return out

def boot(mask, a="LatAD", b="AE", B=3000):
    idx = np.flatnonzero(mask); A, Bm = M(a), M(b); d = []
    for _ in range(B):
        bi = rng.choice(idx, len(idx), replace=True)
        if y[bi].min() == y[bi].max(): continue
        d.append(np.mean([auc(y[bi], s[bi]) for s in A]) - np.mean([auc(y[bi], s[bi]) for s in Bm]))
    d = np.array(d); lo, hi = np.quantile(d, [.025, .975])
    print(f"  paired bootstrap {a}-{b}: {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(<=0)={np.mean(d<=0):.3f}")
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), p_le0=float(np.mean(d <= 0)))

# ---------- 1. construct-matched tables ----------
REPORT["canonical_difficult"] = table(KH, "CANONICAL difficult (paper: difficult anomalies vs all normals)")
REPORT["canonical_difficult_boot_LatAD_AE"] = boot(KH)
for b in ("IF", "linres", "l2"):
    if b in S.files: REPORT[f"canonical_difficult_boot_LatAD_{b}"] = boot(KH, "LatAD", b)
REPORT["canonical_easy"] = table(KE, "CANONICAL easy")
REPORT["all"] = table(np.ones_like(y, bool), "ALL")
mis = maxz > thr
REPORT["misconstructed_maxz_gt_thr"] = table(mis, "MIS-CONSTRUCTED 'maxz>thr over all windows' (= easy anomalies + few normals)")

# ---------- 2. head ablation (offline, from components) ----------
if "dens" in S.files:
    g = json.loads(str(S["gates"])); print("\n== gates per seed:", g)
    dens, near, res, bas, full = S["dens"], S["near"], S["resid"], S["basin"], S["LatAD"]
    variants = {"density only": dens, "dens+near (base)": dens + near, "base+resid(forced)": dens + near + res,
                "base+basin(forced)": dens + near + bas, "full(auto) = reported": full, "dens+resid(forced)": dens + res}
    REPORT["head_ablation_difficult"] = {}
    for nm, sc in variants.items():
        per = [auc(y[KH], s[KH]) for s in sc]
        REPORT["head_ablation_difficult"][nm] = dict(mean=float(np.mean(per)), std=float(np.std(per)))
        print(f"  {nm:24s} difficult {np.mean(per):.3f} +- {np.std(per):.3f}   all {np.mean([auc(y, s) for s in sc]):.3f}")
    # sanity invariant: full(auto) must equal base when both gates are off
    if all((not gg["resid_auto"]) and gg["basin_lam"] == 0 for gg in g):
        print("  [invariant] gates off -> full == base:", np.allclose(full, dens + near, atol=1e-4))
    REPORT["gates"] = g

# ---------- 3. coverage audit on the clean clip=10 loader ----------
D = E.load("WADI_clean", clip=10.0 if clip is None else clip)
Xn, Xa, ch, idx_a, W = D["Xn_w"].astype(float), D["Xa_w"].astype(float), D["ch"], D["idx_a"], D["W"]
assert np.array_equal(D["ya_w"].astype(int), y)
C = len(ch)
mu, sd = Xn.mean(0), Xn.std(0); sd = np.maximum(sd, 0.01 * np.median(sd))
Zn, Za = (Xn - mu) / sd, (Xa - mu) / sd
nn = NearestNeighbors(n_neighbors=2).fit(Zn)
d_loo = nn.kneighbors(Zn)[0][:, 1]; d_te = nn.kneighbors(Za, n_neighbors=1)[0][:, 0]
oos_thr = float(np.quantile(d_loo, 0.99)); oos = d_te > oos_thr
print(f"\n== coverage: 1-NN dist to train, OOS thr (train LOO p99) = {oos_thr:.2f}")
print(f"  OOS rate: test normals {oos[norm].mean():.3f} ({oos[norm].sum()}/{norm.sum()}), all anomalies {oos[y==1].mean():.3f}, "
      f"difficult anomalies {oos[hard].mean():.3f} ({oos[hard].sum()}/{hard.sum()}), easy {oos[easy].mean():.3f}")
# also time-ordered held-out check: is the OOS threshold itself sane? (last 20% of train vs first 80%)
nA = int(0.8 * len(Zn)); nnA = NearestNeighbors(n_neighbors=1).fit(Zn[:nA])
dB = nnA.kneighbors(Zn[nA:])[0][:, 0]; print(f"  sanity: last-20%-of-train OOS rate vs first-80% = {(dB > oos_thr).mean():.3f} (temporal drift inside train)")
REPORT["coverage"] = dict(oos_thr=oos_thr, oos_rate_normals=float(oos[norm].mean()), n_oos_normals=int(oos[norm].sum()),
                          oos_rate_anoms=float(oos[y == 1].mean()), oos_rate_hard=float(oos[hard].mean()),
                          train_tail_oos_rate=float((dB > oos_thr).mean()))

lat = M("LatAD").mean(0); ae = M("AE").mean(0)
# percentile rank of each window against TEST NORMALS (higher = more anomalous)
def prank(s): return np.array([(s[norm] < v).mean() for v in s])
pl, pa = prank(lat), prank(ae)
# LatAD false alarms in the difficult-subset ranking: normals ranked above the median difficult anomaly
def false_alarms(s, q=0.99):
    t = np.quantile(s[norm], q); return norm & (s > t)
for q in (0.95, 0.99):
    fl, fa = false_alarms(lat, q), false_alarms(ae, q)
    print(f"  false alarms @ {int((1-q)*100)}% normal-FPR: LatAD {fl.sum()} of which OOS {oos[fl].sum()} ({oos[fl].mean():.2f}); "
          f"AE {fa.sum()} of which OOS {oos[fa].sum()} ({oos[fa].mean():.2f})")
    REPORT["coverage"][f"latad_fa_oos_frac_q{q}"] = float(oos[fl].mean()); REPORT["coverage"][f"ae_fa_oos_frac_q{q}"] = float(oos[fa].mean())
# where do OOS normals sit in each method's ranking (mean percentile)?
print(f"  mean percentile of OOS normals: LatAD {pl[norm & oos].mean():.3f}  AE {pa[norm & oos].mean():.3f}; in-support normals: LatAD {pl[norm & ~oos].mean():.3f} AE {pa[norm & ~oos].mean():.3f}")
# contiguous OOS-normal blocks
def runs(mask):
    i = np.flatnonzero(mask)
    if len(i) == 0: return []
    br = np.flatnonzero(np.diff(i) > 1); st = np.r_[i[0], i[br + 1]]; en = np.r_[i[br], i[-1]]
    return [(int(s), int(e - s + 1)) for s, e in zip(st, en)]
rr = runs(oos & norm); print(f"  OOS-normal runs: {len(rr)}; lengths {[l for _, l in rr]}")
REPORT["coverage"]["oos_normal_runs"] = rr

# exclusion test: drop OOS normals (removes 0 anomalies by construction), all methods on the same windows
keep = ~(oos & norm)
print(f"\n== EXCLUSION TEST: drop {int((oos & norm).sum())} OOS test normals (anomalies removed: {int((~keep & (y==1)).sum())})")
REPORT["exclusion_oos_normals"] = table(KH & keep, "difficult, OOS normals excluded")
REPORT["exclusion_oos_normals_boot_LatAD_AE"] = boot(KH & keep)
# control: drop the same NUMBER of random in-support normals (should NOT move LatAD much if the effect is coverage)
ctrl = []
for r in range(20):
    rk = norm.copy(); drop = rng.choice(np.flatnonzero(norm & ~oos), int((oos & norm).sum()), replace=False); rk[drop] = False
    m2 = hard | rk
    ctrl.append(np.mean([auc(y[m2], s[m2]) for s in M("LatAD")]) - np.mean([auc(y[m2], s[m2]) for s in M("AE")]))
print(f"  control (drop same # of random in-support normals): LatAD-AE = {np.mean(ctrl):+.3f} +- {np.std(ctrl):.3f}")
REPORT["exclusion_control_random_LatAD_minus_AE"] = dict(mean=float(np.mean(ctrl)), std=float(np.std(ctrl)))

# ---------- 4. duplicates / mislabels among difficult anomalies ----------
di = np.flatnonzero(hard); Zd = Za[di]
Dm = squareform(pdist(Zd)); eps10, eps50 = float(np.quantile(d_loo, 0.10)), float(np.quantile(d_loo, 0.50))
cl10 = fcluster(linkage(pdist(Zd), "single"), eps10, "distance"); cl50 = fcluster(linkage(pdist(Zd), "single"), eps50, "distance")
ep = runs(hard)
np.fill_diagonal(Dm, np.inf)
print(f"\n== duplicates among {len(di)} difficult anomalies: episodes={len(ep)} lens={[l for _,l in ep]}; "
      f"near-dup clusters eps=p10 trainLOO({eps10:.2f}) -> {cl10.max()}, eps=p50({eps50:.2f}) -> {cl50.max()}; "
      f"min pairwise {Dm.min():.2f}, median 1NN within-difficult {np.median(Dm.min(1)):.2f} vs to-train {np.median(d_te[di]):.2f}")
nn0 = NearestNeighbors(n_neighbors=1).fit(Za[norm]); d_a2n = nn0.kneighbors(Za[y == 1])[0][:, 0]
inl = np.minimum(d_a2n, d_te[y == 1]) < eps50
print(f"  anomalies within train-LOO-median ({eps50:.2f}) of some normal (possible mislabel): {inl.sum()}/{(y==1).sum()}, difficult among them {(inl & hard[y==1]).sum()}")
REPORT["duplicates"] = dict(n_episodes=len(ep), episode_lengths=[l for _, l in ep], clusters_p10=int(cl10.max()), clusters_p50=int(cl50.max()),
                            min_pairwise=float(Dm.min()), n_possible_mislabel=int(inl.sum()), n_possible_mislabel_difficult=int((inl & hard[y == 1]).sum()))

# ---------- 5. per-window table ----------
seg = runs(D["ya_raw"] == 1)
def segof(w):
    st = idx_a[w]; return [k for k, (a, l) in enumerate(seg) if st < a + l and st + W > a]
rows = []
for w in di:
    m = Za[w, :C]; top = np.argsort(-np.abs(m))[:3]
    rows.append(dict(win=int(w), seg=segof(w), maxz=round(float(maxz[w]), 2), oos=bool(oos[w]), d1nn=round(float(d_te[w]), 1),
                     LatAD_pct=round(float(pl[w]), 3), AE_pct=round(float(pa[w]), 3), linres_pct=round(float(prank(S["linres"])[w]), 3) if "linres" in S.files else None,
                     top=[(ch[t], round(float(m[t]), 2)) for t in top]))
ae_wins = [r for r in rows if r["AE_pct"] - r["LatAD_pct"] > 0.10]; lat_wins = [r for r in rows if r["LatAD_pct"] - r["AE_pct"] > 0.10]
print(f"\n== difficult anomalies AE ranks >10pct higher than LatAD: {len(ae_wins)}; LatAD > AE: {len(lat_wins)}; within 10pct: {len(rows)-len(ae_wins)-len(lat_wins)}")
print("  -- AE catches / LatAD misses:")
for r in ae_wins: print("   ", r)
print("  -- LatAD catches / AE misses:")
for r in lat_wins: print("   ", r)
print("  -- all difficult (seg, maxz, oos, LatAD_pct, AE_pct):")
for r in rows: print(f"    w{r['win']:3d} seg{r['seg']} maxz {r['maxz']:5.2f} oos={int(r['oos'])} d1nn={r['d1nn']:6.1f} LatAD {r['LatAD_pct']:.3f} AE {r['AE_pct']:.3f} lin {r['linres_pct']} top {r['top']}")
REPORT["difficult_windows"] = rows
# per-segment summary: mean percentile per method per attack segment (difficult windows only)
segsum = {}
for r in rows:
    for s in r["seg"]: segsum.setdefault(s, []).append(r)
print("\n== per attack segment (difficult windows): n, mean LatAD pct, mean AE pct, frac OOS, dominant channel")
for s in sorted(segsum):
    rs = segsum[s]; chs = {}
    for r in rs: chs[r["top"][0][0]] = chs.get(r["top"][0][0], 0) + 1
    print(f"  seg {s:2d} (rows {seg[s][0]}..{seg[s][0]+seg[s][1]}): n={len(rs)} LatAD {np.mean([r['LatAD_pct'] for r in rs]):.3f} AE {np.mean([r['AE_pct'] for r in rs]):.3f} "
          f"oos {np.mean([r['oos'] for r in rs]):.2f}  {max(chs, key=chs.get)}")
# the LatAD top false alarms: which normals, OOS?, contiguous?, dominant channel
fl = np.flatnonzero(false_alarms(lat, 0.95)); print(f"\n== LatAD false alarms @5% FPR (n={len(fl)}): idx {fl.tolist()}")
for w in fl:
    m = Za[w, :C]; top = np.argsort(-np.abs(m))[:3]
    print(f"    w{w:3d} oos={int(oos[w])} d1nn={d_te[w]:6.1f} LatAD_pct {pl[w]:.3f} AE_pct {pa[w]:.3f} near_attack={int(min(abs(w-a) for a in np.flatnonzero(y==1)))} top {[(ch[t], round(float(m[t]),2)) for t in top]}")
json.dump(REPORT, open(os.path.join(HERE, "fable_cleanwadi_analysis.json"), "w"), indent=1, default=float)
print("\nsaved fable_cleanwadi_analysis.json")
