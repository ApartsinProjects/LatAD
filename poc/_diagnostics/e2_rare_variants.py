"""E2 rare-regime prototypes: scoring variants on the SAME trained VaDE latent (no retraining).
Input: _diagnostics/e2_rare_<name>_s<seed>.npz (from e2_rare_stage.py).
Variants (all reduce to the baseline at a stated parameter value; checked as an invariant):
  V1 shrink(lam)   : var_c' = (1-w) var_c + w var_pool, w = lam/(n_c+lam)         (lam=0 -> base)
  V1b refit(Nmin)  : components with n_c < Nmin get mean/var re-estimated from their own
                     assigned train points, var shrunk to pooled with weight lam/(n+lam)   (Nmin=0 -> base)
  V2 inflate(alpha): var_c' = var_c * (1 + alpha/sqrt(n_c))                          (alpha=0 -> base)
  V3 regthr(kappa) : per-regime threshold thr_c = w p99_c + (1-w) p99_global, w = n_c/(n_c+kappa);
                     score' = s - thr_c(assigned)                                      (kappa=inf -> base)
  V4 knn(Nmin)     : windows whose nearest component has n_c < Nmin are scored by k-NN distance in the
                     whitened latent, quantile-mapped onto the parametric train-score scale   (Nmin=0 -> base)
  V5 merge(Nmin)   : components with n_c < Nmin are deleted; their mass goes to the receiving
                     components (drop only, or drop + moment-refit of receivers)      (Nmin=0 -> base)
Metrics per variant, for BOTH the pi-weighted mixture and the nearest-component score, threshold = train p99:
  FPR on tiny-regime normals (n_c<20), on rare-but-not-tiny normals (occ<2%, n_c>=20), on all normals,
  difficult-subset AUROC, all-anomaly AUROC, and the E2 gap = FPR_mixture - FPR_nearest on tiny / rare.
Groups are fixed by the BASELINE assignment so every variant is scored on the same windows.
Writes e2_rare_results.json (all rows) and prints a per-dataset table (seed-mean).
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
HERE = os.path.dirname(os.path.abspath(__file__))
LOG2PI = math.log(2 * math.pi)
RARE_PI, TINY_N, FLOOR = 0.02, 20, 0.05


def logN(z, mu, var):
    """diag-Gaussian log N(z | mu_c, var_c) for all c: (n, K)"""
    d = z.shape[1]
    return -0.5 * (LOG2PI * d + np.log(var).sum(1)[None] + (((z[:, None, :] - mu[None]) ** 2) / var[None]).sum(2))


def au(y, s, mask):
    k = (y == 0) | mask; yy = y[k]
    return float(roc_auc_score(yy, s[k])) if yy.sum() >= 2 else float("nan")


class Ctx:
    def __init__(self, name, seed):
        d = np.load(os.path.join(HERE, f"e2_rare_{name}_s{seed}.npz"))
        self.name, self.seed = name, seed
        self.ztr, self.zte = d["ztr"].astype(np.float64), d["zte"].astype(np.float64)
        self.mu, self.var = d["mu_c"].astype(np.float64), np.exp(d["lvc"].astype(np.float64))
        self.logpi, self.yw, self.hard = d["logpi"], d["yw"], d["hard"]
        self.K = len(self.logpi)
        self.Ltr, self.Lte = logN(self.ztr, self.mu, self.var), logN(self.zte, self.mu, self.var)
        # invariant 0: my logN reproduces the model's stored logN
        self.err_logN = float(max(np.abs(self.Ltr - d["logN_tr"]).max(), np.abs(self.Lte - d["logN_te"]).max()))
        self.a_tr, self.a_te = self.Ltr.argmax(1), self.Lte.argmax(1)
        self.n_c = np.bincount(self.a_tr, minlength=self.K)
        self.occ = self.n_c / len(self.a_tr)
        self.tiny = np.where(self.n_c < TINY_N)[0]
        self.rare = np.where((self.occ < RARE_PI) & (self.n_c >= TINY_N))[0]
        norm = self.yw == 0
        self.groups = {"tiny": norm & np.isin(self.a_te, self.tiny), "rare": norm & np.isin(self.a_te, self.rare), "all": norm}
        if name == "HAI" and seed == 0:
            self.groups["r8_22"] = norm & np.isin(self.a_te, [8, 22]); self.groups["r21"] = norm & (self.a_te == 21)
        self.var_pool = self.ztr.var(0)
        # k-NN distance in whitened latent (train: leave-one-out)
        sd = self.ztr.std(0) + 1e-8
        nn_ = NearestNeighbors(n_neighbors=11).fit(self.ztr / sd)
        dtr, _ = nn_.kneighbors(self.ztr / sd); dte, _ = nn_.kneighbors(self.zte / sd, n_neighbors=10)
        self.knn_tr, self.knn_te = dtr[:, 1:].mean(1), dte.mean(1)

    def scores(self, Ltr, Lte, logpi=None):
        lp = self.logpi if logpi is None else logpi
        return dict(mixture=(-logsumexp(lp[None] + Ltr, 1), -logsumexp(lp[None] + Lte, 1)),
                    nearest=(-Ltr.max(1), -Lte.max(1)))

    def evaluate(self, sc, label, params, thr_fn=None):
        """sc: {key: (train_scores, test_scores)}; thr_fn(key, s_tr, s_te) -> (flag_te, ranking_te) or None"""
        row = dict(dataset=self.name, seed=self.seed, variant=label, **params)
        for key, (s_tr, s_te) in sc.items():
            if thr_fn is None:
                thr = np.quantile(s_tr, .99); flag, rank = s_te > thr, s_te
            else:
                flag, rank = thr_fn(key, s_tr, s_te)
            m = {f"fpr_{g}": (float(flag[msk].mean()) if msk.sum() else float("nan")) for g, msk in self.groups.items()}
            m["auroc_diff"] = au(self.yw, rank, self.hard); m["auroc_all"] = float(roc_auc_score(self.yw, rank))
            row[key] = m
        for g in ("tiny", "rare"):
            row[f"gap_{g}"] = row["mixture"][f"fpr_{g}"] - row["nearest"][f"fpr_{g}"]
        return row

    # ---------------- variants ----------------
    def v_base(self):
        return self.evaluate(self.scores(self.Ltr, self.Lte), "base", {})

    def v_shrink(self, lam):
        w = lam / (self.n_c + lam) if lam > 0 else np.zeros(self.K)
        var = (1 - w)[:, None] * self.var + w[:, None] * self.var_pool[None]
        return self.evaluate(self.scores(logN(self.ztr, self.mu, var), logN(self.zte, self.mu, var)), "shrink", dict(lam=lam))

    def v_refit(self, nmin, lam=5.0):
        mu, var = self.mu.copy(), self.var.copy()
        for c in range(self.K):
            m = self.a_tr == c; n = m.sum()
            if n == 0 or n >= nmin: continue
            zc = self.ztr[m]; mu[c] = zc.mean(0)
            vh = zc.var(0) if n > 1 else np.zeros_like(self.var_pool)
            var[c] = np.maximum((n * vh + lam * self.var_pool) / (n + lam), FLOOR)
        return self.evaluate(self.scores(logN(self.ztr, mu, var), logN(self.zte, mu, var)), "refit", dict(nmin=nmin, lam=lam))

    def v_inflate(self, alpha):
        var = self.var * (1 + alpha / np.sqrt(np.maximum(self.n_c, 1)))[:, None]
        return self.evaluate(self.scores(logN(self.ztr, self.mu, var), logN(self.zte, self.mu, var)), "inflate", dict(alpha=alpha))

    def v_regthr(self, kappa):
        def thr_fn(key, s_tr, s_te):
            g = np.quantile(s_tr, .99); thr_c = np.full(self.K, g)
            for c in range(self.K):
                m = self.a_tr == c; n = m.sum()
                if n < 2: continue
                w = 1.0 if kappa == 0 else n / (n + kappa)
                thr_c[c] = w * np.quantile(s_tr[m], .99) + (1 - w) * g
            ex = s_te - thr_c[self.a_te]
            return ex > 0, ex
        return self.evaluate(self.scores(self.Ltr, self.Lte), "regthr", dict(kappa=kappa), thr_fn)

    def v_knn(self, nmin, mode="replace"):
        sc = self.scores(self.Ltr, self.Lte); out = {}
        tiny_c = self.n_c < nmin
        for key, (s_tr, s_te) in sc.items():
            # monotone quantile map: kNN distance -> parametric train-score scale
            qs = np.linspace(0, 1, 2001); xk = np.quantile(self.knn_tr, qs); yk = np.quantile(s_tr, qs)
            ktr = np.interp(self.knn_tr, xk, yk); kte = np.interp(self.knn_te, xk, yk)
            if mode == "replace":
                ntr, nte = np.where(tiny_c[self.a_tr], ktr, s_tr), np.where(tiny_c[self.a_te], kte, s_te)
            elif mode == "fuse":
                ntr, nte = np.where(tiny_c[self.a_tr], .5 * (ktr + s_tr), s_tr), np.where(tiny_c[self.a_te], .5 * (kte + s_te), s_te)
            elif mode == "min":
                ntr, nte = np.where(tiny_c[self.a_tr], np.minimum(ktr, s_tr), s_tr), np.where(tiny_c[self.a_te], np.minimum(kte, s_te), s_te)
            else:  # pure kNN everywhere (reference)
                ntr, nte = ktr, kte
            out[key] = (ntr, nte)
        return self.evaluate(out, "knn", dict(nmin=nmin, mode=mode))

    def v_merge(self, nmin, refit=False):
        keep = self.n_c >= nmin
        if keep.sum() == 0: keep[:] = True
        mu, var, lp = self.mu[keep], self.var[keep], self.logpi.copy()
        # mass of dropped comps goes to the receiver of their train points
        Ltr_k = logN(self.ztr, mu, var); a_new = Ltr_k.argmax(1)
        p = np.exp(lp); p_keep = p[keep].copy()
        for c in np.where(~keep)[0]:
            m = self.a_tr == c
            if m.sum(): p_keep[np.bincount(a_new[m], minlength=keep.sum()).argmax()] += p[c]
            else: p_keep += p[c] / keep.sum()
        lp_keep = np.log(p_keep / p_keep.sum())
        if refit:
            for j in range(keep.sum()):
                m = a_new == j
                if m.sum() >= 2:
                    mu[j] = self.ztr[m].mean(0); var[j] = np.maximum(self.ztr[m].var(0), FLOOR)
        return self.evaluate(self.scores(logN(self.ztr, mu, var), logN(self.zte, mu, var), lp_keep), "merge", dict(nmin=nmin, refit=refit))


def run(name, seeds):
    rows = []
    for sd in seeds:
        c = Ctx(name, sd)
        print(f"[{name} s{sd}] K={c.K} tiny={c.tiny.tolist()} n={c.n_c[c.tiny].tolist()} rare={c.rare.tolist()} "
              f"| test-normal tiny={c.groups['tiny'].sum()} rare={c.groups['rare'].sum()} | logN repro err={c.err_logN:.2e}", flush=True)
        base = c.v_base(); rows.append(base)
        # invariants: parameter-zero variants must equal base exactly
        for lab, r in [("shrink0", c.v_shrink(0.0)), ("refit0", c.v_refit(0)), ("inflate0", c.v_inflate(0.0)),
                       ("regthr_inf", c.v_regthr(float("inf"))), ("knn0", c.v_knn(0)), ("merge0", c.v_merge(0))]:
            ok = all(abs(r[k][m] - base[k][m]) < 1e-9 for k in ("mixture", "nearest") for m in r[k] if not (np.isnan(r[k][m]) and np.isnan(base[k][m])))
            print(f"   invariant {lab:11s} == base: {ok}", flush=True)
            assert ok, lab
        for lam in (5, 20, 100, 1000): rows.append(c.v_shrink(float(lam)))
        for nmin in (20, 50, 100, 10**6): rows.append(c.v_refit(nmin, 5.0))
        rows.append(c.v_refit(20, 1.0)); rows.append(c.v_refit(20, 20.0))
        for al in (1, 3, 10, 30): rows.append(c.v_inflate(float(al)))
        for kp in (0, 5, 20, 100): rows.append(c.v_regthr(float(kp)))
        for nmin in (20, 50, 100): rows.append(c.v_knn(nmin, "replace"))
        rows.append(c.v_knn(20, "fuse")); rows.append(c.v_knn(20, "min")); rows.append(c.v_knn(10**6, "pure"))
        for nmin in (20, 50): rows.append(c.v_merge(nmin, False)); rows.append(c.v_merge(nmin, True))
    return rows


def key_of(r):
    return r["variant"] + "(" + ",".join(f"{k}={v}" for k, v in r.items() if k not in ("dataset", "seed", "variant", "mixture", "nearest", "gap_tiny", "gap_rare")) + ")"


def table(rows, name):
    keys = []; by = {}
    for r in rows:
        if r["dataset"] != name: continue
        k = key_of(r)
        if k not in keys: keys.append(k)
        by.setdefault(k, []).append(r)
    P = lambda *a: print(*a, flush=True)
    P(f"\n=== {name}: seed-mean over {len(by[keys[0]])} seeds (mix | near) ===")
    P(f"{'variant':28s} {'FPR tiny':>17s} {'FPR rare':>17s} {'FPR all':>17s} {'AUROC diff':>17s} {'AUROC all':>15s} {'gap tiny':>9s} {'gap rare':>9s}")
    for k in keys:
        rs = by[k]
        def mm(key, m): return np.nanmean([r[key][m] for r in rs])
        P(f"{k:28s} {mm('mixture','fpr_tiny'):8.3f}|{mm('nearest','fpr_tiny'):8.3f} {mm('mixture','fpr_rare'):8.3f}|{mm('nearest','fpr_rare'):8.3f} "
          f"{mm('mixture','fpr_all'):8.4f}|{mm('nearest','fpr_all'):8.4f} {mm('mixture','auroc_diff'):8.3f}|{mm('nearest','auroc_diff'):8.3f} "
          f"{mm('mixture','auroc_all'):7.3f}|{mm('nearest','auroc_all'):7.3f} {np.nanmean([r['gap_tiny'] for r in rs]):9.3f} {np.nanmean([r['gap_rare'] for r in rs]):9.3f}")


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["HAI"]
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    allrows = []
    for nm in names:
        allrows += run(nm, seeds); table(allrows, nm)
    out = os.path.join(HERE, "e2_rare_results.json")
    prev = json.load(open(out)) if os.path.exists(out) else []
    prev = [r for r in prev if r["dataset"] not in names] + allrows
    json.dump(prev, open(out, "w"), indent=1)
    print(f"\nsaved {len(prev)} rows -> {out}")
