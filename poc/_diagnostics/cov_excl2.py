"""Label-free uncovered-regime identification via TRAIN-EMPTY MODES of a pooled KMeans (PCA-20 of standardized features),
robust over K in {16,32,64}; block structure of that mass; and the with/without-exclusion performance decomposition
using that set (K=32 primary). WADI is run in both the baseline space and with the artifact sensors removed
(2B_AIT_002_PV, 1_AIT_004_PV, 2B_AIT_004_PV; features derived from raw channels 102, 3, 104). Writes cov_excl2.json and
merges into cov_coverage.json under 'exclusion_trainempty'."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cov_c2st import block_stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COV = os.path.join(ROOT, "_diagnostics", "cov_coverage.json"); OUT = os.path.join(ROOT, "_diagnostics", "cov_excl2.json")
TRAIN_EMPTY = 0.001     # mode counts as train-empty if train share < 0.1 %
ARTIFACT_WADI = (102, 3, 104)   # raw channel indices; feature j -> channel j % nch


def analyse(ds, Zn, Zt, y, nrm, s, hard, tag):
    n = len(Zn); pooled = np.vstack([Zn, Zt]); Pp = PCA(20, random_state=0).fit_transform(pooled); R = {}
    for K in (16, 32, 64):
        km = KMeans(K, n_init=3, random_state=0).fit(Pp); lt, le = km.labels_[:n], km.labels_[n:]
        p = np.bincount(lt, minlength=K) / n; q = np.bincount(le, minlength=K) / len(le)
        empty = np.where(p < TRAIN_EMPTY)[0]; sel = np.isin(le, empty); pos = nrm[sel]
        bs = block_stats(pos); info = dict(n_train_empty_modes=int(len(empty)), test_mass=float(sel.mean()), n_windows=int(sel.sum()),
                                           modes=[dict(mode=int(k), train=float(p[k]), test=float(q[k])) for k in empty if q[k] > 0], blocks=bs)
        if ds == "HAI":
            blk = (nrm >= 6436) & (nrm <= 7081); info["ref_block_6436_7081"] = dict(frac_in_train_empty=float(sel[blk].mean()), share_of_train_empty_mass=float((sel & blk).sum() / max(sel.sum(), 1)))
        R[f"K{K}"] = info
        print(f"  [{tag}] K={K}: train-empty modes {len(empty)}, test mass {sel.mean():.3%} ({sel.sum()} win), runs>=30: {bs['n_runs_ge30']} longest {bs['longest_run']} frac_in_ge30 {bs['frac_in_runs_ge30']:.2f}", info.get("ref_block_6436_7081", ""), bs["runs_ge30"][:8], flush=True)
    # exclusion decomposition with K=32 set, contiguous runs>=30 only (label-free), y==0 only
    km = KMeans(32, n_init=3, random_state=0).fit(Pp); lt, le = km.labels_[:n], km.labels_[n:]
    p = np.bincount(lt, minlength=32) / n; empty = np.where(p < TRAIN_EMPTY)[0]; pos = nrm[np.isin(le, empty)]
    bl = [(a, b) for a, b in [(r[0], r[1]) for r in block_stats(pos)["runs_ge30"]]]
    excl = np.zeros(len(y), bool)
    for a, b in bl: excl[a:b + 1] = True
    excl &= np.isin(np.arange(len(y)), pos)          # only the train-empty windows inside those runs (all y==0 by construction)
    assert (y[excl] == 0).all(); keep = ~excl
    def metrics(scr, mask, thr):
        yy, ss = y[mask], scr[mask]; nn_ = ss[yy == 0]; aa = ss[yy == 1]; hm = hard[mask]; sub = (yy == 0) | ((yy == 1) & hm)
        o = dict(n_normal=int((yy == 0).sum()), n_anom=int((yy == 1).sum()), auroc=float(roc_auc_score(yy, ss)), auroc_difficult=float(roc_auc_score(yy[sub], ss[sub])))
        for qv, nm in ((0.99, "fpr1"), (0.95, "fpr5")): o[f"tpr_at_{nm}_own"] = float((aa > np.quantile(nn_, qv)).mean())
        for nm, t in thr.items(): o[f"fpr_at_{nm}_fixedthr"] = float((nn_ > t).mean()); o[f"tpr_at_{nm}_fixedthr"] = float((aa > t).mean())
        return o
    dec = dict(criterion="pooled KMeans(32) modes with train share < 0.1 %, contiguous runs >= 30, y==0 only", blocks=[(a, b, b - a + 1) for a, b in bl],
               n_normal_excluded=int(excl.sum()), n_anom_excluded=0, frac_test_normal_excluded=float(excl.sum() / (y == 0).sum()))
    det = {"LatAD": s["LatAD"].mean(0), "maxz": s["maxz"], "IF": s["IF"].mean(0), "AE": s["AE"].mean(0), "USAD": s["USAD"], "TranAD": s["TranAD"]}
    for name, scr in det.items():
        nf = scr[y == 0]; thr = dict(fpr1=float(np.quantile(nf, 0.99)), fpr5=float(np.quantile(nf, 0.95)))
        if name == "maxz": thr["train99"] = float(s["maxz_thr"])
        full = metrics(scr, np.ones(len(y), bool), thr); wo = metrics(scr, keep, thr)
        dec[name] = dict(full=full, without_block=wo, delta={k: wo[k] - full[k] for k in ("auroc", "auroc_difficult", "fpr_at_fpr1_fixedthr", "fpr_at_fpr5_fixedthr", "tpr_at_fpr1_own", "tpr_at_fpr5_own")},
                         share_of_flagged_normals_in_block={k: (float(excl[(y == 0) & (scr > t)].mean()) if ((y == 0) & (scr > t)).any() else None) for k, t in thr.items()})
        d = dec[name]; print(f"    {name:7s} AUROC {full['auroc']:.4f} -> {wo['auroc']:.4f} | diff {full['auroc_difficult']:.4f} -> {wo['auroc_difficult']:.4f} | FPR@fixed1% {full['fpr_at_fpr1_fixedthr']:.4f} -> {wo['fpr_at_fpr1_fixedthr']:.4f} | FPR@fixed5% {full['fpr_at_fpr5_fixedthr']:.4f} -> {wo['fpr_at_fpr5_fixedthr']:.4f} | TPR@1%own {full['tpr_at_fpr1_own']:.3f} -> {wo['tpr_at_fpr1_own']:.3f} | TPR@5%own {full['tpr_at_fpr5_own']:.3f} -> {wo['tpr_at_fpr5_own']:.3f} | flagged-in-block {d['share_of_flagged_normals_in_block']}", flush=True)
    print(f"  [{tag}] exclusion blocks {dec['blocks']} -> {dec['n_normal_excluded']} normals ({dec['frac_test_normal_excluded']:.2%}), 0 anomalies", flush=True)
    R["exclusion_K32"] = dec; return R


out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in (sys.argv[1:] or ["WADI", "HAI", "SWaT"]):
    b = np.load(os.path.join(ROOT, "sota_bundle", "ens_bundle", f"bundle_{ds}.npz")); e = np.load(os.path.join(ROOT, "_diagnostics", f"e2_fable_{ds}.npz")); s = np.load(os.path.join(ROOT, "_diagnostics", f"scores_{ds}.npz"))
    Xn, Xa, y = b["Xn_w"].astype(np.float64), b["Xa_w"].astype(np.float64), b["y"]; assert (s["label"] == y).all()
    mu, sd = Xn.mean(0), Xn.std(0); sd = np.maximum(sd, 0.01 * np.median(sd)); Zn, Za = (Xn - mu) / sd, np.clip((Xa - mu) / sd, -50, 50); nrm = np.where(y == 0)[0]
    print(f"== {ds}", flush=True)
    res = {"baseline": analyse(ds, Zn, Za[nrm], y, nrm, s, e["hard"], "baseline")}
    if ds == "WADI":
        nch = int(b["nch"]); keep = np.array([(j % nch) not in ARTIFACT_WADI for j in range(Zn.shape[1])])
        res["artifact_sensors_removed"] = analyse(ds, Zn[:, keep], Za[nrm][:, keep], y, nrm, s, e["hard"], "no-artifact"); res["artifact_sensors_removed"]["n_features_kept"] = int(keep.sum())
    out[ds] = res; json.dump(out, open(OUT, "w"), indent=1)
    cov = json.load(open(COV)); cov[ds]["exclusion_trainempty"] = res; json.dump(cov, open(COV, "w"), indent=1)
print("saved", OUT)
