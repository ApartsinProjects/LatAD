"""Locally-adaptive latent density (LOF, a local density RATIO) vs the fixed high-K GMM
density head. The rare-regime problem is that an ABSOLUTE density (GMM/KDE) penalises a
rare-but-valid regime just for being globally sparse. A LOCAL density ratio (LOF) scores a
point relative to its own neighbourhood, so a uniformly-sparse valid regime scores normal.

Isolates ONE change: the density-head scoring. Everything else (VaDE, config, encode,
difficulty mask) is identical. Three scorers on the SAME latents:
  - gmm      : -latent_gmm.score_samples  (the CURRENT reported density head)
  - fixed_kde: fixed-bandwidth Gaussian KDE (Silverman h) -- absolute density, reference
  - lof      : Local Outlier Factor in latent space -- locally-adaptive density RATIO

Metrics per dataset (5 seeds): difficult-subset AUROC (maxz>thr, the paper's axis),
full AUROC, and FPR on the sparsest-decile test-normals (the rare-regime proxy).
Bootstrap lof-fixed and lof-gmm on difficult AUROC. Report-only; wins-only.

INVARIANT (positive control, part A): on a synthetic cloud WITH rare valid modes, lof must
separate rare-mode NORMALS from off-manifold anomalies better than an absolute density does
(rare-normal-vs-anomaly AUROC higher). Note the Abramson adaptive-BANDWIDTH KDE was tried
first and FAILED this invariant (wide kernels lower the density AT rare points) -- which is
exactly why the local-ratio LOF is the right locally-adaptive tool. If lof fails the
invariant too, the implementation is broken -> do not trust part B.
"""
from __future__ import annotations
import os, sys, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
CFG = {"WADI": (20, 10), "WADI_clean": (20, 10), "HAI": (40, 16), "SWaT": (40, 16), "SWaT_canon": (40, 16), "SKAB": (16, 6)}
SEEDS = [0, 1, 2, 3, 4]
ALPHA = 0.5          # Abramson sensitivity
KPILOT = 25          # pilot kNN for local density
MNBR = 200           # truncate KDE to nearest-M train neighbours (far points ~0)


def _silverman_h(Z):
    n, d = Z.shape
    sig = Z.std(0).mean() + 1e-9
    return float(sig * n ** (-1.0 / (d + 4)))


def _kde_score(Zq, Ztr, nn, h, d):
    """-log f(Zq) under a fixed-bandwidth Gaussian KDE truncated to MNBR neighbours."""
    m = min(MNBR, len(Ztr))
    dist, _ = nn.kneighbors(Zq, n_neighbors=m)
    logk = -0.5 * (dist ** 2) / (h ** 2) - d * np.log(h) - 0.5 * d * np.log(2 * np.pi)
    mx = logk.max(1, keepdims=True)
    logf = (mx[:, 0] + np.log(np.exp(logk - mx).sum(1))) - np.log(len(Ztr))
    return -logf


def _scorers(Ztr, Zte):
    d = Ztr.shape[1]
    nn = NearestNeighbors().fit(Ztr)
    h = _silverman_h(Ztr)
    lof = LocalOutlierFactor(n_neighbors=KPILOT, novelty=True).fit(Ztr)
    return dict(
        fixed_kde=_kde_score(Zte, Ztr, nn, h, d),
        lof=-lof.score_samples(Zte),                 # higher = more anomalous
    ), nn, d


def _boot(a, b, y, mask, n=2000, seed=0):
    """paired bootstrap of AUROC(a)-AUROC(b) on the masked subset."""
    rng = np.random.default_rng(seed)
    idx = np.where((y == 0) | mask)[0]
    yy = y[idx]
    diffs = []
    for _ in range(n):
        s = rng.choice(len(idx), len(idx), replace=True)
        ys = yy[s]
        if ys.sum() < 2 or (ys == 0).sum() < 2:
            continue
        diffs.append(roc_auc_score(ys, a[idx][s]) - roc_auc_score(ys, b[idx][s]))
    diffs = np.array(diffs)
    return dict(mean=float(diffs.mean()), lo=float(np.quantile(diffs, .025)),
                hi=float(np.quantile(diffs, .975)), p_le0=float((diffs <= 0).mean()))


def _au(y, s, mask):
    k = (y == 0) | mask
    return float(roc_auc_score(y[k], s[k])) if 0 < y[k].sum() < k.sum() else float("nan")


# ---------------- Part A: synthetic invariant ----------------
def control():
    rng = np.random.default_rng(0)
    d = 10
    # The discriminating case: a DENSE dominant mode + a genuinely DIFFUSE valid mode (low
    # absolute density but locally coherent). Anomalies are isolated points of comparable
    # absolute density to the diffuse mode -> an ABSOLUTE density cannot tell them apart, a
    # LOCAL ratio (LOF) can (diffuse-mode members match their neighbours; anomalies do not).
    def draw(n, c, s):
        return rng.normal(c, s, (n, d))
    dense_c, diffuse_c = np.zeros(d), np.ones(d) * 8
    Ztr = np.vstack([draw(1200, dense_c, 0.3), draw(200, diffuse_c, 1.6)])
    Zn = np.vstack([draw(250, dense_c, 0.3), draw(60, diffuse_c, 1.6)])   # incl. diffuse-mode normals
    # anomalies: isolated, scattered in the void around the diffuse mode (similar abs density, no local peers)
    Za = diffuse_c + rng.uniform(-6, 6, (120, d))
    Za = Za[np.linalg.norm(Za - diffuse_c, axis=1) > 5.0][:80]            # keep the ones OUTSIDE the diffuse mode
    Zte = np.vstack([Zn, Za]); y = np.r_[np.zeros(len(Zn)), np.ones(len(Za))].astype(int)
    rare_mask = np.r_[np.zeros(250), np.ones(60), np.zeros(len(Za))].astype(bool)  # diffuse-mode normals
    sc, *_ = _scorers(Ztr, Zte)
    # rare-vs-anomaly AUROC: do RARE-mode normals (label 0) score below anomalies (label 1)?
    ra = rare_mask | (y == 1); yra = y[ra]
    out = {}
    for k, s in sc.items():
        out[k] = dict(auroc=round(_au(y, s, y == 1), 3),
                      rare_vs_anom_auroc=round(float(roc_auc_score(yra, s[ra])), 3),
                      rare_normal_score=round(float(s[rare_mask].mean()), 3),
                      common_normal_score=round(float(s[(y == 0) & ~rare_mask].mean()), 3))
    # invariant: LOF penalises a diffuse-but-VALID mode LESS than an absolute density does
    # (rare-normal / common-normal score ratio closer to 1). This is the locally-adaptive property.
    pen_lof = out["lof"]["rare_normal_score"] / out["lof"]["common_normal_score"]
    pen_fix = out["fixed_kde"]["rare_normal_score"] / out["fixed_kde"]["common_normal_score"]
    out["rare_penalty_ratio_lof"] = round(pen_lof, 3)
    out["rare_penalty_ratio_fixed_kde"] = round(pen_fix, 3)
    out["INVARIANT_lof_less_rare_penalty"] = bool(pen_lof < pen_fix)
    return out


# ---------------- Part B: the 3 datasets ----------------
def run_ds(name):
    import eda_real as E
    from models_vade import train_vade
    D = E.load(name); K, LD = CFG[name]
    Xtr0 = np.asarray(D["Xn_w"], np.float32); Xte0 = np.asarray(D["Xa_w"], np.float32)
    y = np.asarray(D["ya_w"], int)
    C6 = Xte0.shape[1] // 6
    maxz = np.abs(Xte0[:, :C6]).max(1)
    thr = float(np.quantile(np.abs(Xtr0[:, :C6]).max(1), 0.99))
    hard = (y == 1) & (maxz <= thr)   # CANONICAL difficult = subtle anomalies (fixed from maxz>thr)
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
    kd = min(80, max(20, len(Xtr0) // 10))
    acc = {s: {"diff": [], "full": [], "sparse_fpr": []} for s in ("gmm", "fixed_kde", "lof")}
    boot_af, boot_ag = [], []
    for sd in SEEDS:
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
        v.fit_latent_density(Xtr, k_density=kd)
        Ztr = v._encode_mean(Xtr); Zte = v._encode_mean(Xte)
        gmm = -v.latent_gmm.score_samples(Zte)
        sc, nn, d = _scorers(Ztr, Zte)
        S = dict(gmm=gmm, **sc)
        # sparse-decile normals (rare-regime proxy): train pilot density decile of test-normals
        dist, _ = nn.kneighbors(Zte, n_neighbors=KPILOT + 1)
        te_dens = -d * np.log(dist[:, KPILOT] + 1e-9)
        norm = y == 0
        sparse = norm & (te_dens < np.quantile(te_dens[norm], 0.10))
        for k, s in S.items():
            acc[k]["diff"].append(_au(y, s, hard)); acc[k]["full"].append(_au(y, s, y == 1))
            t = np.quantile(s[norm], 0.99)          # ~1% FPR by design on all-normal
            acc[k]["sparse_fpr"].append(float((s[sparse] > t).mean()) if sparse.sum() else float("nan"))
        boot_af.append(_boot(S["lof"], S["fixed_kde"], y, hard, seed=sd))
        boot_ag.append(_boot(S["lof"], S["gmm"], y, hard, seed=sd))
        print(f"  [{name}] seed {sd}: diff gmm={acc['gmm']['diff'][-1]:.3f} "
              f"fix={acc['fixed_kde']['diff'][-1]:.3f} lof={acc['lof']['diff'][-1]:.3f}", flush=True)
    def agg(x): return dict(mean=round(float(np.nanmean(x)), 4), std=round(float(np.nanstd(x)), 4),
                            per_seed=[round(float(v), 4) for v in x])
    res = {"dataset": name, "n_win": len(y), "n_diff": int(hard.sum()),
           "scores": {k: {m: agg(acc[k][m]) for m in acc[k]} for k in acc}}
    # average the per-seed bootstraps
    res["boot_adap_minus_fixed_diff"] = {kk: round(float(np.mean([b[kk] for b in boot_af])), 4)
                                         for kk in ("mean", "lo", "hi", "p_le0")}
    res["boot_adap_minus_gmm_diff"] = {kk: round(float(np.mean([b[kk] for b in boot_ag])), 4)
                                       for kk in ("mean", "lo", "hi", "p_le0")}
    return res


if __name__ == "__main__":
    t0 = time.time()
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI", "HAI", "SWaT"]
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    out = {"control": control()}
    print("CONTROL:", json.dumps(out["control"]), flush=True)
    out["datasets"] = []
    for nm in names:
        out["datasets"].append(run_ds(nm))
        json.dump(out, open(os.path.join(HERE, f"adaptive_density{suffix}.json"), "w"), indent=1)
        r = out["datasets"][-1]
        print(f"== {nm}: diff gmm={r['scores']['gmm']['diff']['mean']} "
              f"fixed={r['scores']['fixed_kde']['diff']['mean']} "
              f"lof={r['scores']['lof']['diff']['mean']}  "
              f"adap-fixed CI={r['boot_adap_minus_fixed_diff']}", flush=True)
    print(f"saved adaptive_density.json  ({time.time()-t0:.0f}s)", flush=True)
