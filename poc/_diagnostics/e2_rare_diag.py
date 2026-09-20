"""Diagnostics behind e2_rare_variants.py results (HAI seed 0 unless told otherwise):
 A. where does my logN differ from the model's stored logN (repro error)?
 B. tiny-regime test normals: are they inside the train support at all? (kNN percentile, own-regime
    train spread, NLL under the refit component vs threshold)
 C. regthr AUROC collapse: which regimes do the (hard) anomalies get assigned to, and what are the
    per-regime thresholds there?
 D. regime 21 (78 train windows, 923 test normals, 100% FPR): fixable by refit?
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2_rare_variants import Ctx, logN, HERE, FLOOR

name, seed = (sys.argv[1] if len(sys.argv) > 1 else "HAI"), int(sys.argv[2]) if len(sys.argv) > 2 else 0
c = Ctx(name, seed); d = np.load(os.path.join(HERE, f"e2_rare_{name}_s{seed}.npz"))
P = lambda *a: print(*a, flush=True)

P("--- A. logN repro ---")
E = np.abs(c.Ltr - d["logN_tr"]); i, k = np.unravel_index(E.argmax(), E.shape)
P(f"max err at train row {i} comp {k}: mine={c.Ltr[i,k]:.4g} stored={d['logN_tr'][i,k]:.4g}; lvc[k] min={d['lvc'][k].min():.3f} max={d['lvc'][k].max():.3f}; n_c[k]={c.n_c[k]}")
P(f"rows/comps with err>1e-2: {(E>1e-2).any(1).sum()} rows, comps {np.where((E>1e-2).any(0))[0].tolist()}")
P(f"err on the ARGMAX entries only: {np.abs(c.Ltr[np.arange(len(c.Ltr)), c.a_tr] - d['logN_tr'][np.arange(len(c.Ltr)), d['logN_tr'].argmax(1)]).max():.3e}; argmax agree: {(c.a_tr == d['logN_tr'].argmax(1)).mean():.4f}")
rel = E / (np.abs(d["logN_tr"]) + 1); P(f"max RELATIVE err {rel.max():.3e}")

P("\n--- B. tiny-regime test normals: in-support? ---")
thr_near = np.quantile(-c.Ltr.max(1), .99)
ptile = np.searchsorted(np.sort(c.knn_tr), c.knn_te) / len(c.knn_tr)
sd = c.ztr.std(0) + 1e-8
for g in ("tiny", "rare"):
    m = c.groups[g]; P(f"{g}: n={m.sum()} kNN-dist percentile among train LOO: median {np.median(ptile[m]):.3f} p10 {np.quantile(ptile[m],.1):.3f}; frac >0.99: {np.mean(ptile[m]>.99):.3f}; frac>0.999: {np.mean(ptile[m]>.999):.3f}")
m = c.groups["all"] & ~c.groups["tiny"] & ~c.groups["rare"]
P(f"common normals: kNN percentile median {np.median(ptile[m]):.3f} frac>0.99 {np.mean(ptile[m]>.99):.3f}")
P(f"hard anomalies: kNN percentile median {np.median(ptile[c.hard]):.3f} frac>0.99 {np.mean(ptile[c.hard]>.99):.3f}")
for cc in c.tiny:
    mt = c.groups["tiny"] & (c.a_te == cc); mtr = c.a_tr == cc
    if mt.sum() == 0: continue
    ztr_c = c.ztr[mtr]
    # distance (whitened) from each tiny test window to nearest train point of ITS OWN regime, vs within-regime train NN spread
    dte = np.sqrt((((c.zte[mt][:, None] - ztr_c[None]) / sd) ** 2).sum(2)).min(1) if mtr.sum() else np.full(mt.sum(), np.nan)
    dd = np.sqrt((((ztr_c[:, None] - ztr_c[None]) / sd) ** 2).sum(2)); np.fill_diagonal(dd, np.inf); dtr = dd.min(1) if mtr.sum() > 1 else np.array([np.nan])
    # refit component NLL
    mu = ztr_c.mean(0); var = np.maximum((mtr.sum() * ztr_c.var(0) + 5 * c.var_pool) / (mtr.sum() + 5), FLOOR)
    nll_refit = -logN(c.zte[mt], mu[None], var[None])[:, 0]; nll_refit_tr = -logN(ztr_c, mu[None], var[None])[:, 0]
    P(f"  comp {cc}: n_tr={mtr.sum()} n_te={mt.sum()} | own-regime NN dist: test median {np.median(dte):.2f} vs train-LOO median {np.median(dtr):.2f} max {np.max(dtr):.2f} "
      f"| global kNN pct median {np.median(ptile[mt]):.3f} | base near-NLL med {np.median(-c.Lte[mt].max(1)):.1f} thr {thr_near:.1f} "
      f"| refit-comp NLL test med {np.median(nll_refit):.1f} train med {np.median(nll_refit_tr):.1f} | mean lvc {d['lvc'][cc].mean():.2f} refit mean logvar {np.log(var).mean():.2f}")

P("\n--- C. regthr collapse: where do anomalies land? ---")
s_tr, s_te = -c.Ltr.max(1), -c.Lte.max(1)
g99 = np.quantile(s_tr, .99)
p99_c = np.array([np.quantile(s_tr[c.a_tr == k], .99) if (c.a_tr == k).sum() >= 2 else g99 for k in range(c.K)])
P(f"global p99 {g99:.1f}; per-regime p99: min {p99_c.min():.1f} median {np.median(p99_c):.1f} max {p99_c.max():.1f}")
P("regimes with own p99 > global+10: " + ", ".join(f"{k}(n={c.n_c[k]},p99={p99_c[k]:.0f},te_norm={(c.groups['all']&(c.a_te==k)).sum()},anom={((c.yw==1)&(c.a_te==k)).sum()},hard={(c.hard&(c.a_te==k)).sum()})" for k in range(c.K) if p99_c[k] > g99 + 10))
ah = np.bincount(c.a_te[c.hard], minlength=c.K); aa = np.bincount(c.a_te[c.yw == 1], minlength=c.K)
top = np.argsort(-aa)[:8]
P("top regimes by anomaly count: " + ", ".join(f"{k}(anom={aa[k]},hard={ah[k]},n_tr={c.n_c[k]},p99_c={p99_c[k]:.0f},occ={c.occ[k]:.4f})" for k in top))
P(f"share of ALL anomalies assigned to regimes with own-p99 > global: {np.isin(c.a_te[c.yw==1], np.where(p99_c > g99)[0]).mean():.3f}; hard: {np.isin(c.a_te[c.hard], np.where(p99_c > g99)[0]).mean():.3f}; normals: {np.isin(c.a_te[c.yw==0], np.where(p99_c > g99)[0]).mean():.3f}")
P(f"share of anomalies assigned to rare(<2%) regimes: all {np.isin(c.a_te[c.yw==1], np.concatenate([c.rare, c.tiny])).mean():.3f} hard {np.isin(c.a_te[c.hard], np.concatenate([c.rare, c.tiny])).mean():.3f} (normals {np.isin(c.a_te[c.yw==0], np.concatenate([c.rare, c.tiny])).mean():.3f})")
ex = s_te - p99_c[c.a_te]
P(f"excess score: normals median {np.median(ex[c.yw==0]):.1f}, anomalies median {np.median(ex[c.yw==1]):.1f}, hard median {np.median(ex[c.hard]):.1f}")

if name == "HAI" and seed == 0:
    P("\n--- D. regime 21 ---")
    k = 21; mtr = c.a_tr == k; mt = c.groups["all"] & (c.a_te == k)
    P(f"n_tr={mtr.sum()} test-normal={mt.sum()} anomalies={((c.yw==1)&(c.a_te==k)).sum()} hard={(c.hard&(c.a_te==k)).sum()}; model mean lvc {d['lvc'][k].mean():.2f}")
    P(f"base near-NLL: train median {np.median(s_tr[mtr]):.1f} p99 {np.quantile(s_tr[mtr],.99):.1f}; test-normal median {np.median(s_te[mt]):.1f}; global thr {g99:.1f}")
    ztr_c = c.ztr[mtr]; mu = ztr_c.mean(0); var = np.maximum(ztr_c.var(0), FLOOR)
    P(f"refit (moment) NLL: train median {np.median(-logN(ztr_c, mu[None], var[None])[:,0]):.1f}; test-normal median {np.median(-logN(c.zte[mt], mu[None], var[None])[:,0]):.1f}; refit mean logvar {np.log(var).mean():.2f}")
    P(f"kNN percentile of regime-21 test normals: median {np.median(ptile[mt]):.3f} frac>0.99 {np.mean(ptile[mt]>.99):.3f}")
    dte = np.sqrt((((c.zte[mt][:, None] - ztr_c[None]) / sd) ** 2).sum(2)).min(1)
    dd = np.sqrt((((ztr_c[:, None] - ztr_c[None]) / sd) ** 2).sum(2)); np.fill_diagonal(dd, np.inf)
    P(f"own-regime NN dist: test median {np.median(dte):.2f} p90 {np.quantile(dte,.9):.2f} vs train-LOO median {np.median(dd.min(1)):.2f} max {dd.min(1).max():.2f}")
    # per-dim: which latent dims carry the shift?
    sh = (c.zte[mt].mean(0) - mu) / np.sqrt(var)
    P(f"test-vs-train mean shift in own-regime sd units per latent dim: {np.round(sh, 1).tolist()}")
