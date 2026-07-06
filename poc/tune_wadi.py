"""WADI tuning harness. We underperform IsolationForest (AUROC .629 vs .748).
Hypotheses to test cheaply, one knob at a time:
  H1  30/123 channels are near-constant in normal -> drop them (remove 180 noise feats).
  H2  window/downsample too coarse (600s windows dilute long-but-localised attacks).
  H3  feature rep: stats vs temporal (dynamics) vs robust-scaled.
  H4  VaDE capacity: K, latent_dim.
  H5  scaling: standard z-score vs robust (median/IQR) -- WADI has spiky actuators.

Strategy: first find the representation ceiling with the fast detectors (IF/LOF) under
each preprocessing, THEN train VaDE only on the promising ones. AUROC is the headline
(prevalence-invariant); also report TPR@5%FPR.
"""
from __future__ import annotations
import os, sys, time, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
from winfeat import window_features

DIR = "../datasets/itrust/WADI/WADI.A2_19 Nov 2019"


def load_raw():
    nrm = pd.read_csv(f"{DIR}/WADI_14days_new.csv")
    atk = pd.read_csv(f"{DIR}/WADI_attackdataLABLE.csv", skiprows=1)
    nrm.columns = [c.strip() for c in nrm.columns]; atk.columns = [c.strip() for c in atk.columns]
    lc = atk.columns[-1]
    meta = set(list(atk.columns[:3]) + list(nrm.columns[:3]) + [lc])
    sensors = [c for c in atk.columns if c not in meta and c in nrm.columns]
    sensors = [c for c in sensors if nrm[c].isna().mean() < 0.5]
    prep = lambda df: df[sensors].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0).values.astype(np.float32)
    Xn, Xa = prep(nrm), prep(atk)
    ya = (pd.to_numeric(atk[lc], errors="coerce").values == -1).astype(int)
    return Xn, Xa, ya, sensors


def build(Xn, Xa, ya, W, stride, downsample, rep, drop_const, scale):
    Xn, Xa = Xn[::downsample], Xa[::downsample]
    ya = ya[::downsample]
    if drop_const:
        keep = Xn.std(0) > 1e-6
        Xn, Xa = Xn[:, keep], Xa[:, keep]
    if scale == "z":
        mu, sd = Xn.mean(0), Xn.std(0) + 1e-8
    else:  # robust
        mu = np.median(Xn, 0)
        sd = (np.quantile(Xn, 0.75, 0) - np.quantile(Xn, 0.25, 0)) + 1e-6
    Xn, Xa = (Xn - mu) / sd, (Xa - mu) / sd

    def win(X, y=None):
        Xw, yl = [], []
        for i in range(0, len(X) - W + 1, stride):
            Xw.append(window_features(X[i:i + W], rep))
            if y is not None:
                yl.append(int(y[i:i + W].mean() > 0.05))
        return np.asarray(Xw, np.float32), (np.asarray(yl, int) if y is not None else None)
    Xtr, _ = win(Xn); Xte, yte = win(Xa, ya)
    return Xtr, Xte, yte


def evalu(y, s):
    au = roc_auc_score(y, s)
    thr = np.quantile(s[y == 0], 0.95)
    return au, float((s[y == 1] > thr).mean())


def fast_scores(Xtr, Xte, seed=0):
    S = {}
    S["IF"] = -IsolationForest(n_estimators=200, random_state=seed).fit(Xtr).decision_function(Xte)
    S["LOF"] = -LocalOutlierFactor(30, novelty=True).fit(Xtr).decision_function(Xte)
    return S


def sweep_fast():
    Xn, Xa, ya, sensors = load_raw()
    print(f"raw: normal={len(Xn)} attack={len(Xa)} sensors={len(sensors)} atkfrac={ya.mean():.3f}\n")
    grid = []
    # (W, stride, downsample, rep, drop_const, scale)
    base = dict(W=60, stride=30, downsample=10, rep="stats", drop_const=False, scale="z")
    grid.append(("baseline", base))
    grid.append(("drop_const", {**base, "drop_const": True}))
    grid.append(("robust_scale", {**base, "drop_const": True, "scale": "robust"}))
    grid.append(("temporal", {**base, "drop_const": True, "rep": "temporal"}))
    grid.append(("finer_win", {**base, "drop_const": True, "W": 30, "stride": 15}))
    grid.append(("coarser_ds", {**base, "drop_const": True, "downsample": 5, "W": 60, "stride": 30}))
    grid.append(("temporal_finer", {**base, "drop_const": True, "rep": "temporal", "W": 30, "stride": 15}))
    print(f"{'config':<18}{'ntr':>7}{'nte':>6}{'feat':>6}{'atk%':>6}"
          f"{'IF_AU':>8}{'IF_TPR':>8}{'LOF_AU':>8}{'LOF_TPR':>8}")
    print("-" * 78)
    results = {}
    for name, cfg in grid:
        t = time.time()
        Xtr, Xte, yte = build(Xn, Xa, ya, **cfg)
        S = fast_scores(Xtr, Xte)
        (ifa, ift), (lfa, lft) = evalu(yte, S["IF"]), evalu(yte, S["LOF"])
        results[name] = (cfg, ifa, lfa)
        print(f"{name:<18}{len(Xtr):>7}{len(Xte):>6}{Xtr.shape[1]:>6}{yte.mean()*100:>5.1f}%"
              f"{ifa:>8.3f}{ift:>8.3f}{lfa:>8.3f}{lft:>8.3f}   ({time.time()-t:.0f}s)")
    return results


def sweep_vade():
    """Train VaDE on the best fast representation (drop_const stats z-score),
    varying capacity, training-window count (via stride), and score decomposition.
    Goal: beat IF's 0.742. Report recon-only vs full (recon+latent-NLL) too."""
    import torch
    from models_vade import train_vade
    from compare_baselines import ae_scores
    Xn, Xa, ya, sensors = load_raw()
    base = dict(W=60, downsample=10, rep="stats", drop_const=True, scale="z")
    print(f"{'config':<34}{'ntr':>7}{'feat':>6}{'AUROC':>8}{'TPR@5':>8}")
    print("-" * 63)
    # AE reference on the best rep
    Xtr, Xte, yte = build(Xn, Xa, ya, stride=30, **base)
    ae = ae_scores(Xtr, Xte, device="cpu")
    au, tp = evalu(yte, ae); print(f"{'AutoEncoder (ref)':<34}{len(Xtr):>7}{Xtr.shape[1]:>6}{au:>8.3f}{tp:>8.3f}")
    au, tp = evalu(yte, -IsolationForest(n_estimators=200, random_state=0).fit(Xtr).decision_function(Xte))
    print(f"{'IsolationForest (ref)':<34}{len(Xtr):>7}{Xtr.shape[1]:>6}{au:>8.3f}{tp:>8.3f}")

    cfgs = [
        ("K12 L10 stride30", dict(stride=30), 12, 10),
        ("K20 L10 stride30", dict(stride=30), 20, 10),
        ("K12 L20 stride30", dict(stride=30), 12, 20),
        ("K12 L10 stride10 (3x data)", dict(stride=10), 12, 10),
        ("K20 L20 stride10 (3x data)", dict(stride=10), 20, 20),
        ("K12 L30 stride10 (3x data)", dict(stride=10), 12, 30),
    ]
    for name, ov, K, L in cfgs:
        Xtr, Xte, yte = build(Xn, Xa, ya, **{**base, **ov})
        v = train_vade(Xtr, n_clusters=K, latent_dim=L, epochs=40, warmup=8, seed=0, device="cpu")
        v.fit_residual_whitener(Xtr)
        s_full = v.anomaly_score(Xte)
        au, tp = evalu(yte, s_full)
        print(f"{name:<34}{len(Xtr):>7}{Xtr.shape[1]:>6}{au:>8.3f}{tp:>8.3f}")


def sweep_decomp():
    """Why do reconstruction methods collapse to chance on WADI while IF wins?
    Decompose the score into recon (raw SSE / whitened) vs latent-NLL, and test a
    channel-extremeness detector (max |z| per window) that mimics IF's mechanism.
    Also reconcile the earlier VaDE 0.629 (no drop_const, K20)."""
    import torch
    from models_vade import train_vade
    Xn, Xa, ya, sensors = load_raw()

    def run(tag, drop_const, K, L):
        Xtr, Xte, yte = build(Xn, Xa, ya, W=60, stride=30, downsample=10,
                              rep="stats", drop_const=drop_const, scale="z")
        # channel-extremeness (IF-like): how many sigma out is the most extreme feature
        ext = np.abs(Xte).max(1); au_e, tp_e = evalu(yte, ext)
        ext_top = np.sort(np.abs(Xte), 1)[:, -10:].mean(1); au_t, tp_t = evalu(yte, ext_top)
        v = train_vade(Xtr, n_clusters=K, latent_dim=L, epochs=40, warmup=8, seed=0, device="cpu")
        # decompose: raw recon SSE (no whitener) vs latent-NLL vs whitened-full
        with torch.no_grad():
            xt = torch.as_tensor(Xte)
            mu_z, _ = v.encode(xt); xhat = v.decode(mu_z).numpy()
        sse = ((Xte - xhat) ** 2).sum(1); au_s, tp_s = evalu(yte, sse)
        v.fit_residual_whitener(Xtr); full = v.anomaly_score(Xte); au_f, tp_f = evalu(yte, full)
        print(f"{tag:<22}{'':>4}  ext|z|max {au_e:.3f}/{tp_e:.3f}   ext-top10 {au_t:.3f}/{tp_t:.3f}"
              f"   reconSSE {au_s:.3f}/{tp_s:.3f}   full {au_f:.3f}/{tp_f:.3f}")

    print("detector decomposition (AUROC/TPR@5%):")
    run("keep-const K20 L10", False, 20, 10)
    run("drop-const K12 L10", True, 12, 10)


def sweep_fuse():
    """WADI attacks live in standardized channel excursions (esp. near-constant-in-
    normal actuators). IF ~0.742. Test: (1) a winsorized top-k extremeness branch,
    (2) fusing IF + VaDE, (3) fusing extremeness + VaDE. Keep ALL channels."""
    import torch
    from models_vade import train_vade
    Xn, Xa, ya, sensors = load_raw()
    Xtr, Xte, yte = build(Xn, Xa, ya, W=60, stride=30, downsample=10,
                          rep="stats", drop_const=False, scale="z")
    z = lambda a, r: (a - r.mean()) / (r.std() + 1e-9)

    ifs_tr = -IsolationForest(n_estimators=200, random_state=0).fit(Xtr).decision_function(Xtr)
    ifs = -IsolationForest(n_estimators=200, random_state=0).fit(Xtr).decision_function(Xte)
    v = train_vade(Xtr, n_clusters=20, latent_dim=10, epochs=40, warmup=8, seed=0, device="cpu")
    v.fit_residual_whitener(Xtr); vade_tr, vade = v.anomaly_score(Xtr), v.anomaly_score(Xte)

    def winsor_topk(X, k=10, clip=8.0):
        A = np.clip(np.abs(X), 0, clip)                    # cap numerical blow-ups
        return np.sort(A, 1)[:, -k:].mean(1)
    ext_tr, ext = winsor_topk(Xtr), winsor_topk(Xte)

    # empirical right-tail p-value from train-normal (small p = more anomalous)
    def pval(s_te, s_tr):
        order = np.sort(s_tr)
        rank = np.searchsorted(order, s_te, side="right")
        return 1.0 - rank / (len(order) + 1.0)
    p_if, p_v, p_e = pval(ifs, ifs_tr), pval(vade, vade_tr), pval(ext, ext_tr)
    or_minp = -np.log(np.minimum(p_if, p_v) + 1e-9)        # OR = most-surprised branch
    or_3 = -np.log(np.minimum(np.minimum(p_if, p_v), p_e) + 1e-9)

    cand = {
        "IF": ifs,
        "VaDE full": vade,
        "winsor-top10 |z|": ext,
        "IF + VaDE (z-sum)": z(ifs, ifs_tr) + z(vade, vade_tr),
        "OR(IF,VaDE) min-p": or_minp,
        "OR(IF,VaDE,ext) min-p": or_3,
    }
    print(f"{'detector':<24}{'AUROC':>8}{'TPR@5%':>9}")
    print("-" * 41)
    for n, s in cand.items():
        au, tp = evalu(yte, s); print(f"{n:<24}{au:>8.3f}{tp:>9.3f}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "fast"
    if what == "fast":
        sweep_fast()
    elif what == "vade":
        sweep_vade()
    elif what == "decomp":
        sweep_decomp()
    elif what == "fuse":
        sweep_fuse()
