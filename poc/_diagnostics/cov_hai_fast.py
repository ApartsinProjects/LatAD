"""HAI, foreground, subsampled where needed (3000/side for energy + C2ST; full windows for occupancy/exclusion;
even-in-time 6000-window subsample for rarefaction). Writes threshold_free / rarefaction / exclusion keys for HAI into
cov_coverage.json and cov_rarefaction.json. Model-free; reuses bundle arrays and scores_HAI.npz."""
import json, os, sys, time, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cov_c2st import two_sample, occupancy, block_stats, even, CHUNKS
from cov_rarefaction import leader_cluster, count_leaders, gmm_counts, fit_saturation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COV = os.path.join(ROOT, "_diagnostics", "cov_coverage.json"); RAR = os.path.join(ROOT, "_diagnostics", "cov_rarefaction.json")
ds = "HAI"; t0 = time.time()
b = np.load(os.path.join(ROOT, "sota_bundle", "ens_bundle", f"bundle_{ds}.npz")); e = np.load(os.path.join(ROOT, "_diagnostics", f"e2_fable_{ds}.npz"))
Xn, Xa, y = b["Xn_w"].astype(np.float64), b["Xa_w"].astype(np.float64), b["y"]
mu, sd = Xn.mean(0), Xn.std(0); sd = np.maximum(sd, 0.01 * np.median(sd))
Zn, Za = (Xn - mu) / sd, np.clip((Xa - mu) / sd, -50, 50); nrm = np.where(y == 0)[0]; Zt = Za[nrm]; n = len(Zn); h = n // 2
T = {}
def P(*a): print(*a, flush=True)

# ---- (1) energy distance, 3000/side ----
ts = {"even_seed0": two_sample(Zn, Zt, seed=0, sub="even", n_sub=3000, n_perm=1000)}
for s_ in (0, 1, 2): ts[f"random_seed{s_}"] = two_sample(Zn, Zt, seed=s_, sub="random", n_sub=3000, n_perm=1000)
ts["null_ref_train_first_vs_second_half"] = two_sample(Zn[:h], Zn[h:], seed=0, sub="even", n_sub=3000, n_perm=300)
fifths = []
for j in range(5):
    lo, hi = (j * n) // 5, ((j + 1) * n) // 5; m = np.zeros(n, bool); m[lo:hi] = True
    r5 = two_sample(Zn[m], Zn[~m], seed=0, sub="even", n_sub=3000, n_perm=100); fifths.append(dict(fifth=j, energy=r5["energy"], H=r5["energy_coef_H"], p_block=r5["p_energy_blockperm"]))
ts["within_train_fifth_vs_rest"] = fifths; ref_e = [f["energy"] for f in fifths]
ts["test_vs_train_over_within_train"] = dict(energy_ratio_to_max_fifth=float(ts["even_seed0"]["energy"] / max(ref_e)), energy_ratio_to_mean_fifth=float(ts["even_seed0"]["energy"] / np.mean(ref_e)),
                                             within_train_energy=dict(mean=float(np.mean(ref_e)), max=float(max(ref_e)), last_fifth=float(ref_e[-1])))
T["two_sample"] = ts
for k, v in ts.items():
    if isinstance(v, dict) and "n_per_side" in v: P(f"  {k}: n={v['n_per_side']} E={v['energy']:.3f} H={v['energy_coef_H']:.4f} nullmax={v['energy_null_block']['max']:.3f} z={v['energy_z_vs_block_null']:.1f} p_blk={v['p_energy_blockperm']:.4f} mmd2={v['mmd2']:.4f} p={v['p_mmd_blockperm']:.4f}")
P("  fifths:", [round(x, 3) for x in ref_e], ts["test_vs_train_over_within_train"], f"[{time.time()-t0:.0f}s]")

# ---- (2) occupancy ----
occ = {}
itr = np.argmax(e["logN_tr"] + e["logpi"], 1); ite = np.argmax(e["logN_te"] + e["logpi"], 1)[nrm]; Kv = int(len(e["logpi"]))
occ["vade_regimes"], p, q = occupancy(itr, ite, Kv)
occ["vade_regimes"]["regime_21"] = dict(train=float(p[21]), test=float(q[21]), share_of_tv=float(abs(p[21] - q[21]) / np.abs(p - q).sum()))
occ["vade_regimes"]["null_train_halves"] = {k: v for k, v in occupancy(itr[:h], itr[h:], Kv)[0].items() if k in ("tv", "kl_test_from_train")}
pooled = np.vstack([Zn, Zt]); pca = PCA(20, random_state=0).fit(pooled); Pp = pca.transform(pooled)
for k in (8, 16, 32, 64):
    km = KMeans(k, n_init=3, random_state=0).fit(Pp); occ[f"kmeans{k}"], pk, qk = occupancy(km.labels_[:n], km.labels_[n:], k)
    kmn = KMeans(k, n_init=2, random_state=0).fit(Pp[:n]); occ[f"kmeans{k}"]["null_train_halves"] = {kk: vv for kk, vv in occupancy(kmn.labels_[:h], kmn.labels_[h:], k)[0].items() if kk in ("tv", "kl_test_from_train")}
    # which kmeans mode does the 6436..7081 block occupy, and its train share
    blk = (nrm >= 6436) & (nrm <= 7081); lb = km.labels_[n:][blk]; top = np.bincount(lb, minlength=k).argmax()
    occ[f"kmeans{k}"]["hai_block_mode"] = dict(mode=int(top), frac_block_in_mode=float((lb == top).mean()), train_share=float(pk[top]), test_share=float(qk[top]))
T["occupancy"] = occ
for k, v in occ.items(): P(f"  occ {k}: TV={v['tv']:.3f} KL={v['kl_test_from_train']:.3f} mass<1%={v['mass_test_in_modes_with_train_lt_1pct']:.3f} top={v['top_mode_share_of_tv']:.2f} nullTV={v['null_train_halves']['tv']:.3f}", v.get("regime_21", ""), v.get("hai_block_mode", ""))
blk = (nrm >= 6436) & (nrm <= 7081); P("  VaDE regime of block windows:", np.bincount(ite[blk], minlength=Kv).argsort()[::-1][:3], "shares", np.sort(np.bincount(ite[blk]) / blk.sum())[::-1][:3].round(3), f"[{time.time()-t0:.0f}s]")

# ---- C2ST 3000/side blocked LR, score ALL test-normal windows out-of-fold; block-permuted null (5) ----
m3 = 3000; A, ia = even(Zn, m3); B, ib = even(Zt, m3)
X = np.vstack([A, B]); yy = np.r_[np.zeros(m3), np.ones(m3)]; chunk = np.r_[(np.arange(m3) * CHUNKS) // m3, CHUNKS + (np.arange(m3) * CHUNKS) // m3]
chunk_all_test = CHUNKS + (np.arange(len(Zt)) * CHUNKS) // len(Zt)      # chunk id of every test-normal window (same time partition)
def fit(lbl, score_all=False):
    oof = np.full(len(lbl), np.nan); oof_all = np.full(len(Zt), np.nan)
    for tr, te in GroupKFold(10).split(X, lbl, chunk):
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(X[tr], lbl[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
        if score_all:
            held = np.unique(chunk[te]); mask = np.isin(chunk_all_test, held); oof_all[mask] = clf.predict_proba(Zt[mask])[:, 1]
    return roc_auc_score(lbl, oof), oof_all
auc, oof_all = fit(yy, True); assert not np.isnan(oof_all).any()
r = np.random.default_rng(0); null = []
for _ in range(5):
    cl = np.zeros(2 * CHUNKS); cl[r.permutation(2 * CHUNKS)[CHUNKS:]] = 1; null.append(fit(cl[chunk])[0])
T["c2st"] = dict(auc=float(auc), n_per_class=m3, null_auc_mean=float(np.mean(null)), null_auc_max=float(np.max(null)), n_null=5, p_vs_block_null=float((np.sum(np.array(null) >= auc) + 1) / 6))
# train-halves reference
Xh = np.vstack([even(Zn[:h], m3)[0], even(Zn[h:], m3)[0]]); oofh = np.full(2 * m3, np.nan)
for tr, te in GroupKFold(10).split(Xh, yy, chunk):
    oofh[te] = LogisticRegression(C=0.1, max_iter=2000).fit(Xh[tr], yy[tr]).predict_proba(Xh[te])[:, 1]
T["c2st"]["null_ref_train_halves_auc"] = float(roc_auc_score(yy, oofh))
P("  C2ST:", T["c2st"], f"[{time.time()-t0:.0f}s]")
# ---- (3) block structure over ALL test-normal windows ----
struct = {}
for pth in (0.9, 0.95):
    struct[f"p_gt_{pth}"] = block_stats(nrm[oof_all > pth]); struct[f"p_gt_{pth}"]["frac_of_test_normal"] = float((oof_all > pth).mean())
struct["oof_quantiles_test_normal"] = {str(qq): float(np.quantile(oof_all, qq)) for qq in (0.1, 0.5, 0.9, 0.99)}
struct["hai_ref_block_6436_7081"] = dict(n_normal_in_block=int(blk.sum()), mean_oof_in_block=float(oof_all[blk].mean()), frac_gt_0p9_in_block=float((oof_all[blk] > 0.9).mean()),
                                         mean_oof_outside=float(oof_all[~blk].mean()), frac_gt_0p9_outside=float((oof_all[~blk] > 0.9).mean()))
T["c2st_structure"] = struct; np.savez(os.path.join(ROOT, "_diagnostics", f"cov_c2st_oof_{ds}.npz"), oof=oof_all, pos=nrm)
P("  structure:", struct)

# ---- relative OOS (footnote) ----
mm = int(0.1 * n); rel = {}
for name, hidx in [("random10", np.random.default_rng(0).choice(n, mm, replace=False)), ("last10_temporal", np.arange(n - mm, n))]:
    mask = np.ones(n, bool); mask[hidx] = False; nn = NearestNeighbors(n_neighbors=2, algorithm="brute", n_jobs=-1).fit(Zn[mask])
    dl = nn.kneighbors(Zn[mask])[0][:, 1]; d_ho = nn.kneighbors(Zn[hidx])[0][:, 0]; d_te = nn.kneighbors(Zt)[0][:, 0]
    rel[name] = {f"q{qn}": dict(heldout_train_rate=float((d_ho > np.quantile(dl, qn)).mean()), test_normal_rate=float((d_te > np.quantile(dl, qn)).mean()),
                                ratio=float((d_te > np.quantile(dl, qn)).mean() / max((d_ho > np.quantile(dl, qn)).mean(), 1e-9))) for qn in (0.9, 0.95, 0.99)}
    rel[name]["median_d1_ratio_test_over_heldout"] = float(np.median(d_te) / np.median(d_ho))
T["relative_oos"] = rel; P("  relOOS:", rel, f"[{time.time()-t0:.0f}s]")

# ---- size curve (prefix + random) energy at 2000/side ----
curve = []
for f in (0.1, 0.25, 0.5, 0.75, 1.0):
    nf = int(round(f * n)); subs = [("prefix", np.arange(nf))] + ([("random", np.sort(np.random.default_rng(s_).choice(n, nf, replace=False))) for s_ in (0, 1, 2)] if f < 1 else [("random", np.arange(n))])
    for kind, idx in subs:
        tsr = two_sample(Zn[idx], Zt, seed=0, sub="even", n_sub=2000, n_perm=100); curve.append(dict(frac=f, n_sub=nf, kind=kind, energy=tsr["energy"], H=tsr["energy_coef_H"], p_energy_blockperm=tsr["p_energy_blockperm"]))
T["size_curve"] = curve
T["size_curve_agg"] = dict(energy={str(f): dict(random_mean=float(np.mean([c["energy"] for c in curve if c["frac"] == f and c["kind"] == "random"])), prefix=float([c["energy"] for c in curve if c["frac"] == f and c["kind"] == "prefix"][0])) for f in (0.1, 0.25, 0.5, 0.75, 1.0)})
P("  size curve energy:", T["size_curve_agg"]["energy"], f"[{time.time()-t0:.0f}s]")
cov = json.load(open(COV)); cov[ds]["threshold_free"] = T; json.dump(cov, open(COV, "w"), indent=1)

# ---- (4) rarefaction on even-in-time 6000-window subsamples ----
stream = np.vstack([Zn, Zt]); d_loo = NearestNeighbors(n_neighbors=2, algorithm="brute", n_jobs=-1).fit(Zn).kneighbors(Zn)[0][:, 1]; eps0 = float(np.median(d_loo))
R = dict(n_train=n, n_test_normal=int(len(Zt)), eps_median_train_loo=eps0, rarefaction_subsample=6000)
FR = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]; curves = {}
for src_name, Zfull in (("train_only", Zn), ("full_normal_stream", stream)):
    Z, sidx = even(Zfull, 6000); N = len(Z); Pz = pca.transform(Z); rows = []
    for f in FR:
        nf = max(int(round(f * N)), 40)
        subs = [("prefix", np.arange(nf))] + [("random", np.sort(np.random.default_rng(s_).choice(N, nf, replace=False))) for s_ in (0, 1, 2)]
        for kind, idx in subs:
            row = dict(frac=f, n=int(nf), kind=kind); order = np.arange(nf) if kind == "prefix" else np.random.default_rng(int(idx[0]) + nf).permutation(nf)
            for mlt in (2, 4, 8):
                tot, big = count_leaders(Z[idx], mlt * eps0, order); row[f"leaders_x{mlt}"] = tot; row[f"leaders_x{mlt}_ge5"] = big
            row.update({f"gmm_{k}": v for k, v in gmm_counts(Pz[idx]).items()}); rows.append(row)
    keys = [k for k in rows[0] if k not in ("frac", "n", "kind")]
    agg = {k: {str(f): dict(random_mean=float(np.mean([r_[k] for r_ in rows if r_["frac"] == f and r_["kind"] == "random"])), random_sd=float(np.std([r_[k] for r_ in rows if r_["frac"] == f and r_["kind"] == "random"])),
                             prefix=int([r_[k] for r_ in rows if r_["frac"] == f and r_["kind"] == "prefix"][0])) for f in FR} for k in keys}
    sat = {}
    for k in keys:
        ns = [int(round(f * N)) for f in FR]; S = [agg[k][str(f)]["random_mean"] for f in FR]; Sp = [agg[k][str(f)]["prefix"] for f in FR]
        s = fit_saturation(ns, S); s["rel_rise_0p5_to_1"] = float((S[-1] - S[4]) / max(S[4], 1e-9)); s["rel_rise_0p7_to_1"] = float((S[-1] - S[5]) / max(S[5], 1e-9))
        s["prefix_rel_rise_0p5_to_1"] = float((Sp[-1] - Sp[4]) / max(Sp[4], 1e-9)); s["prefix_rel_rise_0p7_to_1"] = float((Sp[-1] - Sp[5]) / max(Sp[5], 1e-9)); sat[k] = s
    curves[src_name] = dict(rows=rows, agg=agg, saturation=sat)
    P(f"  [{src_name}] N={N}")
    for k in keys:
        if "capped" in k or "grid" in k: continue
        P(f"    {k:18s} random:", [round(agg[k][str(f)]['random_mean'], 1) for f in FR], " prefix:", [agg[k][str(f)]['prefix'] for f in FR], f" Smax={sat[k]['Smax']:.1f} at_full={sat[k]['frac_of_asymptote_at_full']:.2f} rise.5-1={sat[k]['rel_rise_0p5_to_1']:.2f} prefix rise.5-1={sat[k]['prefix_rel_rise_0p5_to_1']:.2f}")
    P(f"  [{time.time()-t0:.0f}s]")
R["rarefaction"] = curves
# first appearance on the 6000 even subsample of the full stream
Z, sidx = even(stream, 6000); fa = {}
for mlt in (2, 4, 8):
    lab, leaders = leader_cluster(Z, mlt * eps0, np.arange(len(Z))); cnt = np.bincount(lab); big = np.where(cnt >= 5)[0]
    born_test = sidx[leaders[big]] >= n; test_rows = sidx >= n
    info = dict(n_regimes_ge5=int(len(big)), n_born_in_train=int((~born_test).sum()), n_born_in_test=int(born_test.sum()),
                test_normal_mass_in_test_born_regimes=float(np.isin(lab[test_rows], big[born_test]).mean()))
    tpos = np.where(test_rows, nrm[np.clip(sidx - n, 0, len(nrm) - 1)], -1)          # test-window position of each stream row
    inblk = (tpos >= 6436) & (tpos <= 7081); regs_blk = np.unique(lab[inblk])
    born_blk = [j for j in regs_blk if inblk[leaders[j]]]
    info["hai_block"] = dict(n_rows_in_block=int(inblk.sum()), n_regimes_in_block=int(len(regs_blk)), n_regimes_born_in_block=len(born_blk),
                             frac_block_rows_in_block_born_regimes=float(np.isin(lab[inblk], born_blk).mean()) if inblk.any() else None,
                             block_born_members_outside_block=int(np.isin(lab, born_blk).sum() - np.isin(lab[inblk], born_blk).sum()),
                             first_birth_test_pos=int(min(tpos[leaders[j]] for j in born_blk)) if born_blk else None,
                             extra_test_normal_windows_needed=int(min(np.searchsorted(nrm, tpos[leaders[j]]) for j in born_blk) + 1) if born_blk else None)
    if born_blk: info["hai_block"]["extra_as_frac_of_train"] = float(info["hai_block"]["extra_test_normal_windows_needed"] / n)
    fa[f"x{mlt}"] = info; P(f"  first-appearance x{mlt}:", info)
R["first_appearance"] = fa

# ---- (5) exclusion decomposition ----
s = np.load(os.path.join(ROOT, "_diagnostics", f"scores_{ds}.npz")); assert (s["label"] == y).all(); hard = e["hard"]
def runs(idx, bridge=2):
    idx = np.sort(idx); out = []
    if len(idx) == 0: return out
    st, pv = idx[0], idx[0]
    for i in idx[1:]:
        if i - pv > bridge + 1: out.append((int(st), int(pv))); st = i
        pv = i
    out.append((int(st), int(pv))); return out
blocks = [(a, bb) for a, bb in runs(nrm[oof_all > 0.9]) if bb - a + 1 >= 30]
excl = np.zeros(len(y), bool)
for a, bb in blocks: excl[a:bb + 1] = True
n_anom_in_span = int((excl & (y == 1)).sum()); excl &= (y == 0); assert (y[excl] == 0).all(); keep = ~excl
def metrics(scr, mask, thr_fixed):
    yy_, ss = y[mask], scr[mask]; nn_ = ss[yy_ == 0]; aa = ss[yy_ == 1]; out = dict(n_normal=int((yy_ == 0).sum()), n_anom=int((yy_ == 1).sum()), auroc=float(roc_auc_score(yy_, ss)))
    hm = hard[mask]; sub = (yy_ == 0) | ((yy_ == 1) & hm); out["auroc_difficult"] = float(roc_auc_score(yy_[sub], ss[sub]))
    for qv, name in ((0.99, "fpr1"), (0.95, "fpr5")): out[f"tpr_at_{name}_own"] = float((aa > np.quantile(nn_, qv)).mean())
    for name, t in thr_fixed.items(): out[f"fpr_at_{name}_fixedthr"] = float((nn_ > t).mean()); out[f"tpr_at_{name}_fixedthr"] = float((aa > t).mean())
    return out
dec = dict(exclusion_blocks=[(a, bb, bb - a + 1) for a, bb in blocks], n_normal_excluded=int(excl.sum()), n_anom_excluded=0, n_anom_inside_block_spans_kept=n_anom_in_span, frac_test_normal_excluded=float(excl.sum() / (y == 0).sum()))
det = {"LatAD": s["LatAD"].mean(0), "maxz": s["maxz"], "IF": s["IF"].mean(0), "AE": s["AE"].mean(0), "USAD": s["USAD"], "TranAD": s["TranAD"]}
for name, scr in det.items():
    nn_full = scr[y == 0]; thr = dict(fpr1=float(np.quantile(nn_full, 0.99)), fpr5=float(np.quantile(nn_full, 0.95)))
    if name == "maxz": thr["train99"] = float(s["maxz_thr"])
    full = metrics(scr, np.ones(len(y), bool), thr); wo = metrics(scr, keep, thr)
    dec[name] = dict(full=full, without_block=wo, delta={k: wo[k] - full[k] for k in ("auroc", "auroc_difficult", "fpr_at_fpr1_fixedthr", "fpr_at_fpr5_fixedthr", "tpr_at_fpr1_own", "tpr_at_fpr5_own")},
                     share_of_flagged_normals_in_block={k: float(excl[(y == 0) & (scr > t)].mean()) if ((y == 0) & (scr > t)).any() else None for k, t in thr.items()})
R["exclusion"] = dec
P(f"  exclusion: blocks {dec['exclusion_blocks']} n_normal_excl {dec['n_normal_excluded']} ({dec['frac_test_normal_excluded']:.3%}) anom_excl 0 (anomaly windows inside spans, kept: {n_anom_in_span})")
for name in det:
    d = dec[name]; P(f"    {name:7s} AUROC {d['full']['auroc']:.4f} -> {d['without_block']['auroc']:.4f} | diff {d['full']['auroc_difficult']:.4f} -> {d['without_block']['auroc_difficult']:.4f} | FPR@fixed1% {d['full']['fpr_at_fpr1_fixedthr']:.4f} -> {d['without_block']['fpr_at_fpr1_fixedthr']:.4f} | FPR@fixed5% {d['full']['fpr_at_fpr5_fixedthr']:.4f} -> {d['without_block']['fpr_at_fpr5_fixedthr']:.4f} | TPR@1%own {d['full']['tpr_at_fpr1_own']:.3f} -> {d['without_block']['tpr_at_fpr1_own']:.3f} | TPR@5%own {d['full']['tpr_at_fpr5_own']:.3f} -> {d['without_block']['tpr_at_fpr5_own']:.3f} | flagged-in-block share: {d['share_of_flagged_normals_in_block']}")
rar = json.load(open(RAR)) if os.path.exists(RAR) else {}; rar[ds] = R; json.dump(rar, open(RAR, "w"), indent=1)
cov = json.load(open(COV)); cov[ds]["rarefaction_summary"] = {src: {k: dict(agg=curves[src]["agg"][k], sat=curves[src]["saturation"][k]) for k in curves[src]["agg"]} for src in curves}
cov[ds]["first_appearance"] = fa; cov[ds]["exclusion"] = dec; json.dump(cov, open(COV, "w"), indent=1)
P(f"done [{time.time()-t0:.0f}s]")
