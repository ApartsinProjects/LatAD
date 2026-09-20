"""E2 adversarial audit, stage 2 (offline, from e2_fable_<name>.npz):
  1. bug audit invariants   2. per-point gap EDA on rare-regime normals
  3. fairer metrics          4. is-the-condition-present census
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy.special import logsumexp
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
RARE_PI = 0.02


def au(y, s, mask):
    k = (y == 0) | mask; yy = y[k]
    if yy.sum() < 2 or (yy == 0).sum() < 2: return float("nan")
    return float(roc_auc_score(yy, s[k]))


def analyze(name):
    d = np.load(os.path.join(HERE, f"e2_fable_{name}.npz"))
    logN_tr, logN_te, logpi, yw, hard = d["logN_tr"], d["logN_te"], d["logpi"], d["yw"], d["hard"]
    lvc, mu_c = d["lvc"], d["mu_c"]
    K = logN_tr.shape[1]; LD = mu_c.shape[1]
    out = {"dataset": name}
    P = lambda *a: print(*a, flush=True)
    P(f"\n{'='*80}\n{name}: K={K} latent={LD} train={len(logN_tr)} test={len(logN_te)} test-normal={(yw==0).sum()}")

    mix = lambda L: -logsumexp(logpi[None] + L, axis=1)
    near = lambda L: -L.max(axis=1)
    s_tr = dict(mixture=mix(logN_tr), nearest=near(logN_tr))
    s_te = dict(mixture=mix(logN_te), nearest=near(logN_te))
    a_tr, a_te = logN_tr.argmax(1), logN_te.argmax(1)
    # responsibility-based (pi-weighted) assignment
    r_tr = (logpi[None] + logN_tr).argmax(1); r_te = (logpi[None] + logN_te).argmax(1)
    occ = np.bincount(a_tr, minlength=K) / len(a_tr)
    occ_r = np.bincount(r_tr, minlength=K) / len(r_tr)
    rare = np.where(occ < RARE_PI)[0]
    norm = yw == 0
    rare_norm = norm & np.isin(a_te, rare)

    # ---------------- 1. BUG AUDIT ----------------
    P("\n--- 1. bug audit invariants ---")
    gap_tr = s_tr["mixture"] - s_tr["nearest"]; gap_te = s_te["mixture"] - s_te["nearest"]
    bound_te = -logpi[a_te]
    P(f"mix >= near on all points: train {bool((gap_tr >= -1e-9).all())}, test {bool((gap_te >= -1e-9).all())}")
    P(f"gap <= -log pi_argmax on all test points: {bool((gap_te <= bound_te + 1e-9).all())}")
    P(f"learned pi (softmax logit) vs train occupancy (argmax-nearest): corr={np.corrcoef(np.exp(logpi), occ)[0,1]:.3f}, "
      f"max |pi-occ|={np.abs(np.exp(logpi)-occ).max():.4f}")
    P(f"rare regimes (occ<2%): {len(rare)} ids={rare.tolist()}")
    P(f"  their learned pi: {np.round(np.exp(logpi[rare]),4).tolist()}")
    P(f"  their -log pi (max possible gap): {np.round(-logpi[rare],2).tolist()}")
    P(f"  their mean log-var (component width, floor {np.log(0.05):.2f}): {np.round(lvc[rare].mean(1),2).tolist()}")
    P(f"  all-components mean log-var: median {np.median(lvc.mean(1)):.2f}, rare median {np.median(lvc[rare].mean(1)):.2f}, common median {np.median(np.delete(lvc.mean(1), rare)):.2f}")
    P(f"assignment nearest vs responsibility disagree: train {np.mean(a_tr!=r_tr):.3f}, test {np.mean(a_te!=r_te):.3f}")
    P(f"rare-regime normal test windows: nearest-assign {rare_norm.sum()}, resp-assign {(norm & np.isin(r_te, rare)).sum()}")
    P(f"train share in rare regimes: nearest {np.isin(a_tr, rare).mean():.3f}, resp {np.isin(r_tr, rare).mean():.3f}; test-normal share {np.isin(a_te[norm], rare).mean():.3f}")
    # reproduce E2 numbers
    for key in s_tr:
        thr = np.quantile(s_tr[key], 0.99); pred = s_te[key] > thr
        P(f"  E2 repro {key:8s}: thr={thr:.2f} FPR_all={np.mean(pred[norm]):.4f} FPR_rare={np.mean(pred[rare_norm]):.4f} diffAUROC={au(yw, s_te[key], hard):.3f}")
    # who sets the 99th pct train threshold? share of top-1% train scores in rare regimes
    for key in s_tr:
        top = s_tr[key] >= np.quantile(s_tr[key], 0.99)
        P(f"  top-1% TRAIN {key:8s}: share in rare regimes {np.isin(a_tr[top], rare).mean():.3f} (base rate {np.isin(a_tr, rare).mean():.3f})")

    # ---------------- 2. EDA on rare-regime normals ----------------
    P("\n--- 2. per-point gap on rare-regime NORMAL test windows ---")
    g = gap_te[rare_norm]
    if len(g):
        q = np.quantile(g, [0, .1, .25, .5, .75, .9, .99, 1])
        P(f"gap = mix-near (nats): n={len(g)} mean={g.mean():.2f} quantiles(0,10,25,50,75,90,99,100)={np.round(q,2).tolist()}")
        gc = gap_te[norm & ~np.isin(a_te, rare)]
        P(f"gap on COMMON-regime normals: n={len(gc)} mean={gc.mean():.2f} median={np.median(gc):.2f} p90={np.quantile(gc,.9):.2f}")
        sn = s_te["nearest"]
        P(f"nearest NLL: rare-normals median {np.median(sn[rare_norm]):.2f}  common-normals median {np.median(sn[norm & ~np.isin(a_te, rare)]):.2f}  train median {np.median(s_tr['nearest']):.2f} train p99 {np.quantile(s_tr['nearest'],.99):.2f}")
        # per-regime: occupancy, #test-normal, gap, nearest-NLL percentile within that regime's train points
        P("per rare regime: id occ pi  n_tr n_te_norm  mean_gap  med_nearNLL_te  med_nearNLL_tr  frac_te_above_tr_p99(near)  frac_flag_mix frac_flag_near")
        thr_m = np.quantile(s_tr["mixture"], .99); thr_n = np.quantile(s_tr["nearest"], .99)
        rows = []
        for c in rare:
            mt = rare_norm & (a_te == c); tr_c = a_tr == c
            if mt.sum() == 0: continue
            trp99 = np.quantile(s_tr["nearest"][tr_c], .99) if tr_c.sum() >= 5 else np.nan
            rows.append((int(c), occ[c], np.exp(logpi[c]), int(tr_c.sum()), int(mt.sum()), gap_te[mt].mean(),
                         np.median(sn[mt]), np.median(s_tr["nearest"][tr_c]) if tr_c.sum() else np.nan,
                         np.mean(sn[mt] > trp99) if tr_c.sum() >= 5 else np.nan,
                         np.mean(s_te["mixture"][mt] > thr_m), np.mean(sn[mt] > thr_n)))
        for r in sorted(rows, key=lambda r: -r[4]):
            P("  %3d %.4f %.4f %5d %5d  %6.2f  %7.2f  %7.2f  %s  %.3f %.3f" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
              ("%.3f" % r[8]) if not np.isnan(r[8]) else "  nan", r[9], r[10]))
        # "valid" rare normals: nearest NLL within that regime's train p99 (point is well inside its own component)
        valid = np.zeros(len(yw), bool)
        for c in rare:
            tr_c = a_tr == c
            if tr_c.sum() >= 5:
                valid |= rare_norm & (a_te == c) & (sn <= np.quantile(s_tr["nearest"][tr_c], .99))
        P(f"rare-regime normals that are VALID (near-NLL <= own-regime train p99): {valid.sum()}/{rare_norm.sum()}")
        if valid.sum():
            P(f"  on VALID rare normals: gap mean {gap_te[valid].mean():.2f} median {np.median(gap_te[valid]):.2f}; "
              f"flagged mix {np.mean(s_te['mixture'][valid] > thr_m):.3f} near {np.mean(sn[valid] > thr_n):.3f}")
        # top-20 / bottom-20 by gap
        idx = np.where(rare_norm)[0]
        order = idx[np.argsort(-gap_te[idx])]
        def show(ii, label):
            P(f"  {label}: idx  comp  occ    pi      near   mix    gap   -logpi  z-dist(sd)  flag_mix flag_near")
            for i in ii:
                c = a_te[i]; zd = np.sqrt((((d['zte'][i] - mu_c[c])**2) / np.exp(lvc[c])).mean())
                P(f"    {i:6d} {c:3d} {occ[c]:.4f} {np.exp(logpi[c]):.4f} {sn[i]:7.2f} {s_te['mixture'][i]:7.2f} {gap_te[i]:5.2f} {-logpi[c]:6.2f}  {zd:5.2f}   {int(s_te['mixture'][i]>thr_m)} {int(sn[i]>thr_n)}")
        show(order[:20], "TOP-20 gap (mixture most exceeds nearest)")
        show(order[-20:], "BOTTOM-20 gap")

    # ---------------- 3. fairer metrics ----------------
    P("\n--- 3. fairer metrics ---")
    out["fair"] = {}
    sm, sn = s_te["mixture"], s_te["nearest"]
    common_norm = norm & ~np.isin(a_te, rare)
    if rare_norm.sum() >= 5:
        # (b) FPR on rare normals at matched OVERALL test-normal FPR
        for f in [0.01, 0.02, 0.05, 0.10]:
            tm = np.quantile(sm[norm], 1 - f); tn = np.quantile(sn[norm], 1 - f)
            P(f"  matched overall test-normal FPR={f:.2f}: FPR_rare mix={np.mean(sm[rare_norm]>tm):.4f} near={np.mean(sn[rare_norm]>tn):.4f} | FPR_common mix={np.mean(sm[common_norm]>tm):.4f} near={np.mean(sn[common_norm]>tn):.4f}")
        # matched recall on (difficult) anomalies
        for rec in [0.5, 0.8]:
            for lab, msk in [("all-anom", yw == 1), ("hard", hard)]:
                if msk.sum() < 5: continue
                tm = np.quantile(sm[msk], 1 - rec); tn = np.quantile(sn[msk], 1 - rec)
                P(f"  matched {lab} recall={rec}: FPR_rare mix={np.mean(sm[rare_norm]>tm):.4f} near={np.mean(sn[rare_norm]>tn):.4f} | FPR_all mix={np.mean(sm[norm]>tm):.4f} near={np.mean(sn[norm]>tn):.4f}")
        # (c) share of top-1%/5% highest-scoring NORMAL windows in rare regimes
        for f in [0.01, 0.05]:
            tm = np.quantile(sm[norm], 1 - f); tn = np.quantile(sn[norm], 1 - f)
            P(f"  top-{int(f*100)}% scoring normals, share in rare regimes: mix={np.isin(a_te[norm & (sm>tm)], rare).mean():.3f} near={np.isin(a_te[norm & (sn>tn)], rare).mean():.3f} (base {np.isin(a_te[norm], rare).mean():.3f})")
        # (d) rank statistic: AUROC of rare-normal vs common-normal (1.0 = score fully penalizes rarity; 0.5 = blind to it)
        yy = rare_norm[norm].astype(int)
        P(f"  AUROC(rare-normal vs common-normal): mix={roc_auc_score(yy, sm[norm]):.3f} near={roc_auc_score(yy, sn[norm]):.3f}")
        # rank shift: median percentile rank (among all test normals) of rare normals
        def prank(s):
            r = np.argsort(np.argsort(s[norm])) / norm.sum(); return np.median(r[rare_norm[norm]])
        P(f"  median percentile-rank of rare normals among all normals: mix={prank(sm):.3f} near={prank(sn):.3f}")
        # paired: per rare-normal point, does mixture rank it higher than nearest (rank among all test normals)?
        rm = np.argsort(np.argsort(sm[norm])); rn = np.argsort(np.argsort(sn[norm]))
        dr = (rm - rn)[rare_norm[norm]]
        P(f"  paired rank shift (mix rank - near rank) on rare normals: mean={dr.mean():.1f} median={np.median(dr):.1f} frac>0={np.mean(dr>0):.3f} frac<0={np.mean(dr<0):.3f} (n={len(dr)})")
        # overall test-normal AUROC-style: normal vs anomaly
        P(f"  AUROC all anomalies: mix={roc_auc_score(yw, sm):.3f} near={roc_auc_score(yw, sn):.3f}; hard: mix={au(yw,sm,hard):.3f} near={au(yw,sn,hard):.3f}")

    # ---------------- 4. condition census ----------------
    P("\n--- 4. is the condition present? occupancy census ---")
    so = np.sort(occ)[::-1]
    P(f"occupancy sorted: {np.round(so,4).tolist()}")
    P(f"max/min occ ratio: {so[0]/max(so[-1],1e-6):.0f}; #regimes <2%: {(occ<.02).sum()}, <1%: {(occ<.01).sum()}, <0.5%: {(occ<.005).sum()}, <0.1%: {(occ<.001).sum()}")
    P(f"mass in regimes <2%: {occ[occ<.02].sum():.3f}, <1%: {occ[occ<.01].sum():.3f}, <0.5%: {occ[occ<.005].sum():.3f}")
    # 'genuinely rare-but-valid' = test normal, nearest comp rare, near-NLL within own-regime train p99, AND component is not a catch-all (logvar not wide)
    # Zipf fit: log occ vs log rank slope
    rk = np.arange(1, K+1); m = so > 0
    slope = np.polyfit(np.log(rk[m]), np.log(so[m]), 1)[0]
    P(f"Zipf slope (log occ vs log rank): {slope:.2f}  (MIIM generator uses beta=2.1; flat=0)")
    # test-normal occupancy by regime vs train occupancy (drift)
    occ_te = np.bincount(a_te[norm], minlength=K) / norm.sum()
    P(f"train-vs-testnormal occupancy: corr={np.corrcoef(occ, occ_te)[0,1]:.3f}; regimes with test share > 3x train share: {[(int(c), round(float(occ[c]),4), round(float(occ_te[c]),4)) for c in range(K) if occ_te[c] > 3*occ[c] and occ_te[c] > .005]}")


if __name__ == "__main__":
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["HAI"]):
        analyze(nm)
