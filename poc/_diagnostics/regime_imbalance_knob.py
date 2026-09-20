"""Single-knob regime-imbalance stability. Cluster train-normal into regimes (external KMeans);
downsample each regime to retention r_i = (n_i/n_max)^beta (beta=0 -> unchanged; larger beta ->
rarer regimes shrink faster). Retrain LatAD(global)/AE/IF; measure difficult-subset AUROC and the
FPR on test-normals of the RARE regimes, vs beta. A UNIFORM-subsample control at the SAME total N
isolates imbalance from data-quantity. Report-only; wins-only.

Audited 2026-09-17 (see fable_sampler_audit.md). Changes relative to the first version:
  * the rare-regime set R* is FIXED across beta and shared by both arms (regimes whose retention at
    BETA_MAX is < 0.5); previously it was recomputed per (beta, arm), empty at beta=0 (no baseline)
    and empty for the uniform arm (NaN), so the two arms were never measured on the same windows.
  * the FPR threshold is the 99th pct of the NON-rare test normals (matched threshold); previously
    it was the 99th pct of ALL test normals, which includes the rare windows themselves (circular)
    and caps rare FPR at 0.01 * N_normal / n_rare.
  * k_density is frozen at its full-N value (design B5); per-regime nested permutation prefixes
    (design B2) with the sampling rng crossed with the model seed; no-duplicate assert.
  * per-seed records, AUDC (trapezoid over beta) per (method, arm, seed), the imbalanced-minus-uniform
    contrast, the LatAD-minus-AE paired delta, a rare-vs-hard AUROC guard, and a kNN-coverage
    negative control (COV) that defines the dataset's dynamic range.

Invariants (expected outcomes stated in advance): beta=0 keeps every row in both arms and reproduces
the full-data scores bitwise; uniform-control rows are a superset-free plain subsample with the same
N as the imbalanced arm; COV rare FPR is non-decreasing in beta on the imbalanced arm and flat on the
uniform arm (within seed noise); the win would be LatAD's AUDC contrast below AE's.
"""
from __future__ import annotations
import os, sys, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eda_real as E
from models_vade import train_vade
from compare_baselines import ae_scores
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {"SKAB": (16, 6), "SWaT_canon": (40, 16), "HAI": (40, 16), "WADI_clean": (20, 10)}
BETAS = [0.0, 1.0, 2.0, 4.0]
KREG = 16          # external regime clustering
METHODS = ("LatAD", "IF", "AE", "COV")


def au(y, s, hard):
    k = (y == 0) | hard
    return float(roc_auc_score(y[k], s[k])) if 0 < y[k].sum() < k.sum() else float("nan")


def au_sub(pos_mask, neg_mask, s):
    """AUROC of pos_mask (label 1) vs neg_mask (label 0); nan if either side is empty."""
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return float("nan")
    yy = np.r_[np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())]
    return float(roc_auc_score(yy, np.r_[s[pos_mask], s[neg_mask]]))


def regime_perms(lab, rng):
    """one seeded permutation per regime; prefixes give nested subsamples across beta (design B2)."""
    return {r: rng.permutation(np.where(lab == r)[0]) for r in np.unique(lab)}


def retention(lab, beta):
    sizes = np.bincount(lab); nmax = sizes.max()
    return {r: (sizes[r] / nmax) ** beta for r in np.unique(lab)}


def sample_idx(lab, beta, perms):
    """indices kept under retention (n_i/n_max)^beta per regime (prefix of the per-regime permutation)."""
    ret = retention(lab, beta)
    keep = [perms[r][:max(1, int(round(len(perms[r]) * ret[r])))] for r in perms]
    idx = np.sort(np.concatenate(keep))
    assert len(np.unique(idx)) == len(idx), "duplicate rows in subsample"
    return idx


def score_all(Xtr, Xte, K, LD, seed, kd):
    v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
    v.fit_latent_density(Xtr, k_density=kd); v.fit_resid_head(Xtr); v.fit_basin_head(Xtr)
    lat = np.asarray(v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto"))
    ifs = -IsolationForest(n_estimators=200, random_state=seed).fit(Xtr).score_samples(Xte)
    ae = ae_scores(Xtr, Xte, seed=seed)
    cov = NearestNeighbors(n_neighbors=10).fit(Xtr).kneighbors(Xte)[0].mean(1)   # coverage negative control
    return dict(LatAD=lat, IF=ifs, AE=ae, COV=cov)


def metrics(s, y, hard, rare_norm, other_norm):
    t = np.quantile(s[other_norm], 0.99)                       # matched threshold on NON-rare test normals
    return dict(diff=au(y, s, hard),
                rarefpr=float((s[rare_norm] > t).mean()) if rare_norm.sum() else float("nan"),
                otherfpr=float((s[other_norm] > t).mean()),
                hard_tpr=float((s[hard] > t).mean()),         # detection at the SAME operating point (a low rare FPR
                                                                # from a flat score also gives a low TPR: the E2 trap)
                rare_vs_hard=au_sub(hard, rare_norm, s))       # guard: rare normals must still rank below hard anomalies


def summarise(per_seed, seeds):
    """AUDC per (method, arm, seed); contrast = imbalanced - uniform; delta = LatAD - AE, paired by seed."""
    b = np.asarray(BETAS); out = {}
    for m in METHODS:
        out[m] = {}
        for arm in ("imbalanced", "uniform_ctrl"):
            curves = [[per_seed[f"{arm}_b{beta}"][m]["rarefpr"][i] for beta in BETAS] for i in range(len(seeds))]
            trap = lambda c: float(np.sum((np.asarray(c)[1:] + np.asarray(c)[:-1]) / 2 * np.diff(b)) / (b[-1] - b[0]))
            audc = [trap(c) for c in curves]
            rise = [float(c[-1] - c[0]) for c in curves]
            out[m][arm] = dict(audc=audc, rise_end_minus_0=rise, curve_seedmean=[float(np.nanmean(x)) for x in zip(*curves)])
        out[m]["contrast_audc"] = [a - u for a, u in zip(out[m]["imbalanced"]["audc"], out[m]["uniform_ctrl"]["audc"])]
    out["delta_LatAD_minus_AE_contrast"] = [l - a for l, a in zip(out["LatAD"]["contrast_audc"], out["AE"]["contrast_audc"])]
    out["delta_LatAD_minus_IF_contrast"] = [l - a for l, a in zip(out["LatAD"]["contrast_audc"], out["IF"]["contrast_audc"])]
    out["dynamic_range_COV"] = float(np.mean(out["COV"]["imbalanced"]["rise_end_minus_0"]))
    return out


def run(name, seeds=(0, 1), betas=None):
    betas = BETAS if betas is None else betas
    D = E.load(name); K, LD = CFG[name]
    Xtr0 = np.asarray(D["Xn_w"], np.float32); Xte0 = np.asarray(D["Xa_w"], np.float32)
    y = np.asarray(D["ya_w"], int)
    C6 = Xte0.shape[1] // 6                                  # canonical difficulty axis (self-contained)
    triv = np.abs(Xte0[:, :C6]).max(1); thr = float(np.quantile(np.abs(Xtr0[:, :C6]).max(1), 0.99))
    hard = (y == 1) & ~((y == 1) & (triv > thr))            # frozen: computed once from the FULL train/test
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8               # frozen standardisation (full train)
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
    kd = min(80, max(20, len(Xtr) // 10))                    # frozen at the full-N value (design B5)
    reg = KMeans(KREG, n_init=4, random_state=0).fit(Xtr)
    lab = reg.labels_
    lab_te = reg.predict(Xte)
    # FIXED rare-regime set R*: regimes cut below 50% retention at the largest beta; same set for every
    # beta and both arms, so the rare FPR curve is measured on one window set with a beta=0 baseline.
    ret_max = retention(lab, max(betas))
    rare_set = sorted(int(r) for r in ret_max if ret_max[r] < 0.5)
    rare_norm = (y == 0) & np.isin(lab_te, rare_set)
    other_norm = (y == 0) & ~rare_norm
    sizes = np.bincount(lab)
    out = {"dataset": name, "n_train": len(Xtr), "n_regimes": KREG, "betas": betas, "seeds": list(seeds), "k_density": kd,
           "regime_sizes_train": sizes.tolist(), "regime_sizes_test_normal": np.bincount(lab_te[y == 0], minlength=KREG).tolist(),
           "rare_regimes": rare_set, "n_rare_train": int(sizes[rare_set].sum()) if rare_set else 0,
           "n_rare_normal_test": int(rare_norm.sum()), "n_other_normal_test": int(other_norm.sum()),
           "n_hard": int(hard.sum()), "arms": {}}
    print(f"[{name}] N={len(Xtr)} regimes={sizes.tolist()} rare={rare_set} rare_test_normals={int(rare_norm.sum())} "
          f"hard={int(hard.sum())} kd={kd}", flush=True)
    if not rare_set or rare_norm.sum() < 20:
        print(f"[{name}] WARNING: rare set empty or < 20 rare test normals; FPR granularity too coarse", flush=True)
    per_seed = {}
    for beta in betas:
        for arm in ("imbalanced", "uniform_ctrl"):
            acc = {m: {"diff": [], "rarefpr": [], "otherfpr": [], "hard_tpr": [], "rare_vs_hard": []} for m in METHODS}
            kept = []; kept_rare = []
            for sd in seeds:
                rng = np.random.default_rng(1000 + sd)        # sampling rng crossed with the model seed
                idx_imb = sample_idx(lab, beta, regime_perms(lab, rng))
                if arm == "imbalanced":
                    idx = idx_imb
                else:                                          # same N as the imbalanced arm, uniform rows
                    idx = np.sort(rng.choice(len(Xtr), len(idx_imb), replace=False))
                assert len(idx) == len(idx_imb)
                if beta == 0.0:
                    assert np.array_equal(idx, np.arange(len(Xtr))), "beta=0 must keep every row"
                kept.append(int(len(idx))); kept_rare.append(int(np.isin(lab[idx], rare_set).sum()))
                S = score_all(Xtr[idx], Xte, K, LD, sd, kd)
                for m, s in S.items():
                    for k_, v_ in metrics(s, y, hard, rare_norm, other_norm).items():
                        acc[m][k_].append(v_)
            key = f"{arm}_b{beta}"; per_seed[key] = acc
            out["arms"][key] = dict(
                n_kept=kept, n_kept_rare=kept_rare,
                **{m: {k_: round(float(np.nanmean(v_)), 4) for k_, v_ in acc[m].items()} for m in METHODS},
                per_seed=acc)
            r = out["arms"][key]
            print(f"  [{name}] {arm:12s} b={beta}: kept={kept[0]} kept_rare={kept_rare[0]} "
                  f"| diff L={r['LatAD']['diff']} AE={r['AE']['diff']} IF={r['IF']['diff']} "
                  f"| rareFPR L={r['LatAD']['rarefpr']} AE={r['AE']['rarefpr']} IF={r['IF']['rarefpr']} COV={r['COV']['rarefpr']} "
                  f"| hardTPR L={r['LatAD']['hard_tpr']} AE={r['AE']['hard_tpr']} "
                  f"| rare_vs_hard L={r['LatAD']['rare_vs_hard']} AE={r['AE']['rare_vs_hard']}", flush=True)
        json.dump(out, open(os.path.join(HERE, f"regime_imbalance_{name}.json"), "w"), indent=1)   # incremental
    if len(betas) > 1:
        out["summary"] = summarise(per_seed, seeds)
        sm = out["summary"]
        print(f"  [{name}] SUMMARY contrast AUDC (imb-uni) per seed: " +
              " ".join(f"{m}={np.round(sm[m]['contrast_audc'], 3).tolist()}" for m in METHODS) +
              f" | delta LatAD-AE={np.round(sm['delta_LatAD_minus_AE_contrast'], 3).tolist()}"
              f" | COV dynamic range={sm['dynamic_range_COV']:.3f}", flush=True)
    json.dump(out, open(os.path.join(HERE, f"regime_imbalance_{name}.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    t0 = time.time()
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["SKAB", "SWaT_canon"]):
        run(nm)
    print(f"done ({time.time()-t0:.0f}s)", flush=True)
