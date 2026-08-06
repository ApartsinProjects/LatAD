"""Comprehensive comparison across 3 real datasets (WADI, HAI, SKAB): baselines + ours +
SOTA, on ALL / EASY / HARD subsets, each with AUROC / best-F1 / FPR-at-best-F1. Ours =
VaDE (recon+NLL), per-mode Mahalanobis (latent), and the consolidated LatentParamAD pipeline.
Baselines = trivial max|z|, IsolationForest, LOF, AutoEncoder. SOTA (USAD/TranAD/GDN) read from
sota_bundle/results/score_<MODEL>_<DATASET>.npy when a Modal run has produced them.

Everything for ours+baselines is computed locally on each dataset's window features (eda_real).
"""
from __future__ import annotations
import os, sys, numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, f1_score
from models_vade import train_vade
from compare_baselines import ae_scores
from permode_mahal import PerModeMahal
from latad_pipeline import LatentParamAD
import eda_real as E

RES = os.path.join(os.path.dirname(__file__), "sota_bundle", "results")
STRIDE = {"WADI": 30, "HAI": 60, "SKAB": 10}


def metr(y, s, mask):
    keep = (y == 0) | mask; yk, sk = y[keep], s[keep]
    if (yk == 1).sum() < 3:
        return (np.nan, np.nan, np.nan)
    au = roc_auc_score(yk, sk)
    qs = np.quantile(sk, np.linspace(0.80, 0.999, 60))
    f1s = [(f1_score(yk, sk > t), t) for t in qs]; f1, tb = max(f1s, key=lambda p: p[0])
    return au, f1, float((sk[yk == 0] > tb).mean())


def win_pointscore(s, W, st):
    return np.array([s[i:i + W].max() for i in range(0, len(s) - W + 1, st)])


def run(name):
    D = E.load(name); Xtr, Xte, y = D["Xn_w"], D["Xa_w"], D["ya_w"]
    C6 = Xte.shape[1] // 6
    triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (y == 1) & (triv > np.quantile(trn, 0.99)); hard = (y == 1) & ~easy
    K = {"WADI": 20, "HAI": 24, "SKAB": 12}[name]
    S = {"trivial max|z|": triv,
         "IsolationForest": -IsolationForest(n_estimators=200, random_state=0).fit(Xtr).decision_function(Xte),
         "LOF (raw)": -LocalOutlierFactor(30, novelty=True).fit(Xtr).decision_function(Xte),
         "AutoEncoder": ae_scores(Xtr, Xte, device="cpu")}
    v = train_vade(Xtr, n_clusters=K, latent_dim=10, epochs=40, warmup=8, seed=0, device="cpu")
    v.fit_residual_whitener(Xtr); S["VaDE (ours)"] = v.anomaly_score(Xte)
    v.fit_latent_density(Xtr, k_density=80); S["VaDE-hard (ours)"] = v.anomaly_score_hard(Xte)
    S["per-mode Mahal (ours)"] = PerModeMahal(n_modes=K).fit(Xtr).score(Xte)
    S["LatAD pipeline (ours)"] = LatentParamAD(k_modes=K).fit(Xtr).score(Xte)
    # SOTA (from Modal), max-pooled to this dataset's window grid. Some SOTA arrays were
    # produced on a DOWNSAMPLED copy (HAI x10) while ours are full-res -> upsample the SOTA
    # per-timestep score by the integer ratio so both land on the SAME window grid before
    # win_pointscore. No-op when lengths already match (WADI/SKAB).
    W, st = D["W"], STRIDE[name]
    Nfull = len(D["Xa_raw"])
    for mdl in ["USAD", "TranAD", "GDN"]:
        p = f"{RES}/score_{mdl}_{name}.npy"
        if os.path.exists(p):
            sp = np.load(p); sp = sp.mean(1) if sp.ndim > 1 else sp
            if len(sp) < 0.9 * Nfull:                      # SOTA ran downsampled -> upsample
                sp = np.repeat(sp, int(round(Nfull / len(sp))))
            if len(sp) < Nfull:
                sp = np.pad(sp, (0, Nfull - len(sp)), mode="edge")
            sp = sp[:Nfull]
            sw = win_pointscore(sp, W, st); S[f"{mdl} (SOTA)"] = sw[:len(y)]

    print(f"\n########## {name}  test={len(Xte)}  anom={int((y==1).sum())} "
          f"(easy={int(easy.sum())}, hard={int(hard.sum())}) ##########")
    print(f"{'method':<22}{'ALL au/f1/fpr':>19}{'EASY au/f1/fpr':>19}{'HARD au/f1/fpr':>19}")
    for nm, s in S.items():
        m = min(len(s), len(y)); s2, y2, e2, h2 = s[:m], y[:m], easy[:m], hard[:m]
        cells = []
        for msk in (y2 == 1, e2, h2):
            au, f1, fpr = metr(y2, s2, msk); cells.append(f"{au:.2f}/{f1:.2f}/{fpr:.2f}")
        print(f"{nm:<22}" + "".join(f"{c:>19}" for c in cells))


if __name__ == "__main__":
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI", "HAI", "SKAB"]):
        run(nm)
