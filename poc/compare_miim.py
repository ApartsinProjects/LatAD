"""Definitive per-category baseline table on the labelled miim_gen data.
Baselines (LOF/IF/AE) are fair on SNAPSHOT categories + hard-fringe FPR (E1);
ours (VaDE + trajectory context) is additionally scored on the trajectory fault.
"""
from __future__ import annotations
import os, numpy as np, torch
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
from miim_gen import make_dataset
import ldt_a, ldt_b, explore_c as E
from compare_baselines import ae_scores

CACHE = "datasets/miim/_cmp_miim.npz"
SNAP = ["pocket", "near_boundary", "drift", "ood"]
TRAJ = ["bad_transition"]
COLS = SNAP + TRAJ


def build(seed=0, K=64, latent=8, device="cpu"):
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True); return {k: z[k] for k in z.files}
    d = make_dataset(seed=seed, n_train=300000, n_test=300000, W=40, stride=20)
    xtr, xte, at = d["x_train"], d["x_test"], d["atype_test"]
    print("[data] " + " ".join(f"{t}={int((at==t).sum())}" for t in ["fringe"] + COLS))
    A = ldt_a.ModeEncoderA.fit(xtr, n_clusters=K, latent_dim=latent, pretrain_epochs=15,
                               epochs=30, warmup=8, seed=seed, device=device, verbose=False)
    g_tr, _ = A.encode(xtr); g_te, _ = A.encode(xte)
    B = ldt_b.TrajectoryEncoderB(K=K, emb_dim=64, ctx_dim=96, n_layers=2,
                                 centroids=A.centroids(), backbone="gru")
    ldt_b.train_B(B, g_tr, A.pi, epochs=60, seg_len=512, stride=128, batch_segs=16,
                  lr=3e-3, max_train_offset=64, device=device, seed=seed)
    out = dict(xtr=xtr, xte=xte, yte=d["y_test"], atype=at.astype(str),
               c_tr=ldt_b.emit_context(B, g_tr, device=device),
               c_te=ldt_b.emit_context(B, g_te, device=device))
    os.makedirs("datasets/miim", exist_ok=True); np.savez(CACHE, **out); return out


def tpr(sn, sp, f=0.05):
    return float("nan") if len(sp) == 0 else float((sp > np.quantile(sn, 1 - f)).mean())


def main():
    device = "cpu"
    C = build(device=device)
    xtr, xte, atype, yte = C["xtr"], C["xte"], C["atype"], C["yte"]
    c_tr, c_te = C["c_tr"], C["c_te"]
    normal = yte == 0; fringe = atype == "fringe"
    std = lambda a, r: ((a - r.mean(0)) / (r.std(0) + 1e-8)).astype(np.float32)
    xtr_s, xte_s = std(xtr, xtr), std(xte, xtr)

    S = {}
    S["LOF"] = -LocalOutlierFactor(30, novelty=True).fit(xtr_s).decision_function(xte_s)
    S["IsolationForest"] = -IsolationForest(n_estimators=200, random_state=0).fit(xtr_s).decision_function(xte_s)
    S["AutoEncoder"] = ae_scores(xtr_s, xte_s, device=device)
    v = train_vade(xtr_s, n_clusters=64, latent_dim=8, epochs=40, warmup=8, seed=0, device=device)
    v.fit_residual_whitener(xtr_s)
    sw_te, sw_tr = v.anomaly_score(xte_s), v.anomaly_score(xtr_s)
    S["VaDE (ours, window)"] = sw_te
    pca = PCA(24, random_state=0).fit(c_tr)
    g_tr, g_te = E.soft_gamma(v, xtr_s), E.soft_gamma(v, xte_s)
    sc_tr, sc_te = E.soft_ctx_score(pca.transform(c_tr), g_tr, np.ones(len(g_tr)),
                                    pca.transform(c_te), g_te)
    zz = lambda a, r: (a - r.mean()) / (r.std() + 1e-9)
    S["VaDE + context (ours+B)"] = zz(sw_te, sw_tr) + np.maximum(0.0, zz(sc_te, sc_tr))

    print(f"\n{'method':<24}" + "".join(f"{c:>13}" for c in COLS) + f"{'fringeFPR':>10}{'AUROC':>7}")
    print("-" * 100)
    for name, s in S.items():
        sn = s[normal]
        cells = "".join(f"{tpr(sn, s[atype == t]):>13.3f}" for t in COLS)
        ffpr = tpr(sn, s[fringe])                       # false-alarm rate on hard-valid fringe
        au = roc_auc_score((yte == 1) & ~np.isin(atype, TRAJ) if name in
                           ("LOF", "IsolationForest", "AutoEncoder") else yte == 1,
                           s) if True else 0
        print(f"{name:<24}" + cells + f"{ffpr:>10.3f}{au:>7.3f}")
    print("\nsnapshot cols fair for baselines; bad_transition needs trajectory (ours). "
          "fringeFPR = false alarms on hard-VALID points (lower better). AUROC excludes "
          "bad_transition for snapshot baselines.")


if __name__ == "__main__":
    main()
