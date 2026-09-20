"""A3 cross-validation pilot: VaDE-FREE measures of 'thin between-regime pockets' on train-normal.

Invariants (stated BEFORE running):
  I0  synthetic control: two well-separated Gaussians -> between/gap fractions ~0 (<0.02);
      two Gaussians + 20% uniform bridge mass -> between fraction ~0.20 (0.12-0.30), gap fraction > 0.05.
  I1  if the measures are valid AND agree with rho: SKAB HIGH, WADI/HAI/SWaT LOW (same ordering as rho).
  I2  measure must be stable to K (dataset K vs K=8): ordering of datasets preserved.
  I3  rho re-computed on SKAB at the witness config (LD6/ep40) must reproduce ~0.58; at the screen
      config (LD8/ep30) ~0.10 (reproduces the known config-dependence; if both give the same, the
      earlier discrepancy was something else).

Measures (all on PCA-20 of the standardised window features, KMeans(K) labels, LOO kNN k=10):
  beta_knn : frac(points whose 10-NN majority-label share < 0.5)      -- kNN analog of rho
  between  : frac(points on the INTERIOR of the chord between their two nearest centroids
             (t in [0.2,0.8]) AND within one typical in-cluster radius of that chord) -- 'reachable'
  gap      : between AND LOO kNN radius > q90 of core points' radii   -- reachable but improbable
  H_norm   : (VaDE) mean normalised responsibility entropy, plus rho at 0.5/0.6/0.7 (gradedness)

Detector pilot (axis ii): kNN-radius, chord-proximity, and their rank-sum, AUROC on SKAB difficult
subset (witness split) and on miim pocket-vs-normal test windows.

Writes _diagnostics/fable_a3_pilot.json incrementally (flushed after each block). Read-only w.r.t.
paper/model/checkpoints.
"""
from __future__ import annotations
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from scipy.special import logsumexp
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "fable_a3_pilot.json")
RES = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    with open(OUT, "w") as f:
        json.dump(RES, f, indent=1); f.flush(); os.fsync(f.fileno())


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- geometry
def geometry(Z, K, seed=0, k=10):
    """Z: (n,d) already reduced. Returns per-point dict of arrays + KMeans."""
    km = KMeans(K, n_init=4, random_state=seed).fit(Z)
    lab = km.labels_; C = km.cluster_centers_
    # LOO kNN
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    dist, idx = nn.kneighbors(Z)
    dist, idx = dist[:, 1:], idx[:, 1:]
    rad = dist[:, -1]
    share = (lab[idx] == lab[:, None]).mean(1)
    # two nearest centroids and chord projection
    dc = ((Z[:, None, :] - C[None]) ** 2).sum(-1)
    o = np.argsort(dc, 1); a, b = o[:, 0], o[:, 1]
    ca, cb = C[a], C[b]
    v = cb - ca; L2 = (v ** 2).sum(1) + 1e-12
    t = ((Z - ca) * v).sum(1) / L2
    proj = ca + np.clip(t, 0, 1)[:, None] * v
    perp = np.linalg.norm(Z - proj, axis=1)
    # typical in-cluster radius = median distance-to-own-centroid of that cluster
    r_own = np.sqrt(dc[np.arange(len(Z)), lab])
    med_r = np.array([np.median(r_own[lab == c]) if (lab == c).any() else np.nan for c in range(K)])
    thr_perp = med_r[a]
    interior = (t > 0.2) & (t < 0.8)
    between = interior & (perp <= thr_perp)
    core = ~interior
    q90 = np.quantile(rad[core], 0.90) if core.sum() > 20 else np.quantile(rad, 0.90)
    gap = between & (rad > q90)
    return dict(lab=lab, C=C, rad=rad, share=share, t=t, perp=perp, thr_perp=thr_perp,
                between=between, gap=gap, km=km, q90=q90, med_r=med_r)


def summarise(g, n):
    return dict(n=int(n), beta_knn=float((g["share"] < 0.5).mean()),
                between=float(g["between"].mean()), gap=float(g["gap"].mean()),
                interior=float(((g["t"] > 0.2) & (g["t"] < 0.8)).mean()),
                mean_share=float(g["share"].mean()))


def measures_for(name, Xtr_s, K_list, seed=0):
    pca = PCA(min(20, Xtr_s.shape[1]), random_state=seed).fit(Xtr_s)
    Z = pca.transform(Xtr_s).astype(np.float64)
    out = {}
    for K in K_list:
        g = geometry(Z, K, seed)
        out[f"K{K}"] = summarise(g, len(Z))
        log(f"  [{name}] K={K:>2} beta_knn={out[f'K{K}']['beta_knn']:.3f} between={out[f'K{K}']['between']:.3f} "
            f"gap={out[f'K{K}']['gap']:.3f} interior={out[f'K{K}']['interior']:.3f}")
    return out, pca, Z


# ----------------------------------------------------------------------------- I0 synthetic control
def synthetic_control():
    if "control" in RES:
        return
    rng = np.random.default_rng(0)
    d = 20; n = 1500
    A = rng.normal(0, 1, (n, d)); B = rng.normal(0, 1, (n, d)); B[:, 0] += 12.0
    sep = np.vstack([A, B])
    # bridge: 20% of total mass uniform along the chord between the two means, with in-cluster-like spread
    nb = int(0.2 * 2 * n / 0.8)
    tt = rng.uniform(0.15, 0.85, nb)
    bridge = np.zeros((nb, d)); bridge[:, 0] = 12.0 * tt; bridge += rng.normal(0, 1, (nb, d)) * 0.7
    brg = np.vstack([A, B, bridge])
    RES["control"] = {}
    for nm, X, K in [("separated", sep, 2), ("bridged", brg, 2), ("separated_K8", sep, 8), ("bridged_K8", brg, 8)]:
        m = measures_for(f"ctrl:{nm}", X.astype(np.float32), [K])[0][f"K{K}"]
        m["true_bridge_mass"] = 0.0 if nm.startswith("sep") else float(nb / len(brg))
        RES["control"][nm] = m
    save()


# ----------------------------------------------------------------------------- real datasets
CFG = {"SKAB": (16, 6), "WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}


def load_skab():
    import eda_real as E
    from winfeat import window_features
    W, ST = 60, 30
    D = E.load("SKAB")

    def win(X, y=None):
        A, B = [], []
        for i in range(0, len(X) - W + 1, ST):
            A.append(window_features(X[i:i + W], "stats"))
            if y is not None:
                B.append(int(y[i:i + W].mean() > 0.05))
        return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)
    Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    C6 = Xte.shape[1] // 6
    triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy
    return ((Xtr - m) / sd).astype(np.float32), ((Xte - m) / sd).astype(np.float32), yw, hard, easy


def real_measures():
    RES.setdefault("real", {})
    data = {}
    for name in ["SKAB", "WADI", "HAI", "SWaT"]:
        K, LD = CFG[name]
        if name == "SKAB":
            Xtr_s, Xte_s, yw, hard, easy = load_skab()
            data[name] = (Xtr_s, Xte_s, yw, hard, easy)
        else:
            d = np.load(os.path.join(HERE, f"e2_fable_{name}.npz"))
            Xtr_s, Xte_s, yw, hard, easy = d["Xtr_s"], d["Xte_s"], d["yw"], d["hard"], d["easy"]
            data[name] = (Xtr_s, Xte_s, yw, hard, easy, d["logN_tr"], d["logpi"])
        if name in RES["real"] and "K8" in RES["real"][name]:
            log(f"  [{name}] cached"); continue
        t0 = time.time()
        Ks = sorted({K, 8})
        out, _, _ = measures_for(name, Xtr_s, Ks)
        out["K_reported"] = K; out["secs"] = round(time.time() - t0, 1)
        RES["real"][name] = out; save()
    return data


# ----------------------------------------------------------------------------- VaDE graded (entropy, rho thresholds)
def graded_from_logN(logN, logpi):
    lp = logpi[None] + logN
    G = np.exp(lp - logsumexp(lp, 1, keepdims=True))
    mr = G.max(1); K = G.shape[1]
    H = -(G * np.log(G + 1e-12)).sum(1) / np.log(K)
    return dict(K=int(K), rho_0p5=float((mr < 0.5).mean()), rho_0p6=float((mr < 0.6).mean()),
                rho_0p7=float((mr < 0.7).mean()), mean_maxresp=float(mr.mean()),
                H_norm_mean=float(H.mean()), H_norm_q90=float(np.quantile(H, 0.9)),
                frac_H_gt_0p3=float((H > 0.3).mean()))


def vade_graded(data):
    RES.setdefault("vade", {})
    for name in ["WADI", "HAI", "SWaT"]:
        if name in RES["vade"]:
            continue
        logN, logpi = data[name][5], data[name][6]
        RES["vade"][name] = {"cached_witness_cfg": graded_from_logN(logN, logpi)}
        log(f"  [{name}] vade graded: {RES['vade'][name]}"); save()
    # SKAB: two configs x two seeds (I3)
    from models_vade import train_vade
    Xtr_s = data["SKAB"][0]
    RES["vade"].setdefault("SKAB", {})
    for tag, (LD, ep, wu) in {"LD6_ep40": (6, 40, 8), "LD8_ep30": (8, 30, 6)}.items():
        for seed in (0, 1):
            key = f"{tag}_s{seed}"
            if key in RES["vade"]["SKAB"]:
                continue
            v = train_vade(Xtr_s, n_clusters=16, latent_dim=LD, epochs=ep, warmup=wu, seed=seed, device="cpu")
            import torch
            with torch.no_grad():
                mu = v.encode(torch.as_tensor(Xtr_s))[0]
                logN = v._log_pz_given_c(mu).cpu().numpy()
                logpi = torch.log_softmax(v.pi_logit, 0).cpu().numpy()
            RES["vade"]["SKAB"][key] = graded_from_logN(logN, logpi)
            log(f"  [SKAB] {key}: {RES['vade']['SKAB'][key]}"); save()


# ----------------------------------------------------------------------------- detector pilot (axis ii)
def detector_scores(Ztr, Zte, K, seed=0, k=10):
    g = geometry(Ztr, K, seed, k)
    nn = NearestNeighbors(n_neighbors=k).fit(Ztr)
    rad_te = nn.kneighbors(Zte)[0][:, -1]
    C = g["C"]; dc = ((Zte[:, None, :] - C[None]) ** 2).sum(-1)
    o = np.argsort(dc, 1); a, b = o[:, 0], o[:, 1]
    ca, cb = C[a], C[b]; v = cb - ca; L2 = (v ** 2).sum(1) + 1e-12
    t = ((Zte - ca) * v).sum(1) / L2
    proj = ca + np.clip(t, 0, 1)[:, None] * v
    perp = np.linalg.norm(Zte - proj, axis=1) / (g["med_r"][a] + 1e-9)
    interior = ((t > 0.2) & (t < 0.8)).astype(float)
    s_knn = rad_te
    s_chord = interior * np.exp(-perp)                      # high = sits on a between-mode chord
    s_gap = rankdata(s_knn) + rankdata(s_chord)             # reachable AND improbable
    return dict(knn=s_knn, chord=s_chord, gap=s_gap)


def detector_pilot(data):
    RES.setdefault("detector", {})
    # SKAB difficult subset
    if "SKAB" not in RES["detector"]:
        Xtr_s, Xte_s, yw, hard, easy = data["SKAB"]
        pca = PCA(20, random_state=0).fit(Xtr_s)
        Ztr, Zte = pca.transform(Xtr_s), pca.transform(Xte_s)
        S = detector_scores(Ztr, Zte, 16)
        r = {}
        for nm, s in S.items():
            kh = (yw == 0) | hard; ka = np.ones_like(yw, bool)
            r[nm] = dict(diff_auroc=float(roc_auc_score(yw[kh], s[kh])), all_auroc=float(roc_auc_score(yw, s)))
        r["reference"] = "witness: base OFF diff=0.500, basin forced lam1 0.605, lam2 0.645"
        RES["detector"]["SKAB"] = r; log(f"  [SKAB detector] {r}"); save()
    # miim pocket faults
    if "miim" not in RES["detector"]:
        d = np.load(os.path.join(ROOT, "datasets", "miim", "miim_unified_seed0.npz"), allow_pickle=True)
        Xtr = d["x_train"][d["y_train"] == 0]; Xte = d["x_test"]; yte = d["y_test"]; ty = d["atype_test"]
        m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr_s = (Xtr - m) / sd; Xte_s = (Xte - m) / sd
        pca = PCA(20, random_state=0).fit(Xtr_s)
        Ztr, Zte = pca.transform(Xtr_s), pca.transform(Xte_s)
        # also train-normal measures on miim (extra reference row)
        RES["real"]["miim_normal"] = measures_for("miim", Xtr_s.astype(np.float32), [8, 16])[0]
        S = detector_scores(Ztr, Zte, 8)
        r = {}
        norm = (yte == 0)
        for nm, s in S.items():
            r[nm] = {}
            for at in ["pocket", "near_boundary", "wrong_for_regime", "ood", "drift"]:
                kk = norm | (ty == at)
                r[nm][at] = float(roc_auc_score((ty == at)[kk].astype(int), s[kk]))
        RES["detector"]["miim"] = r; log(f"  [miim detector] {r}"); save()


if __name__ == "__main__":
    t0 = time.time()
    log("== I0 synthetic control"); synthetic_control()
    log("== real datasets (VaDE-free)"); data = real_measures()
    log("== VaDE graded"); vade_graded(data)
    log("== detector pilot"); detector_pilot(data)
    log(f"done in {time.time()-t0:.0f}s -> {OUT}")
