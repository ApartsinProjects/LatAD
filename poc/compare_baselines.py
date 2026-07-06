"""Baselines vs ours on the temporal synthetic benchmark: LOF, Isolation Forest,
AutoEncoder, our window-VaDE, and our VaDE+trajectory-context. Per-type TPR@5%FPR + AUROC.

All baselines are SNAPSHOT detectors (window only) -> they should fail on bad_transition
(history-dependent); only our trajectory context catches it. Reuses the warm A+B cache.
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
import explore_c as E

TYPES = ["pocket", "drift", "bad_transition"]


def tpr_at(sn, sp, fpr=0.05):
    return float("nan") if len(sp) == 0 else float((sp > np.quantile(sn, 1 - fpr)).mean())


def ae_scores(Xtr, Xte, epochs=40, seed=0, device="cpu"):
    torch.manual_seed(seed); d = Xtr.shape[1]
    ae = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(),
                       nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d)).to(device)
    opt = torch.optim.Adam(ae.parameters(), 1e-3)
    xt = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    for _ in range(epochs):
        opt.zero_grad(); (((ae(xt) - xt) ** 2).mean()).backward(); opt.step()
    with torch.no_grad():
        zt = torch.as_tensor(Xte, dtype=torch.float32, device=device)
        return ((ae(zt) - zt) ** 2).sum(1).cpu().numpy()


def main():
    device = "cpu"
    C = np.load(E.CACHE, allow_pickle=True)
    xtr, xte, atype, yte = C["xtr"], C["xte"], C["atype"], C["yte"]
    c_tr, c_te = C["c_tr"], C["c_te"]; normal = yte == 0
    std = lambda a, r: ((a - r.mean(0)) / (r.std(0) + 1e-8)).astype(np.float32)
    xtr_s, xte_s = std(xtr, xtr), std(xte, xtr)

    scores = {}
    print("baselines...")
    scores["LOF"] = -LocalOutlierFactor(n_neighbors=30, novelty=True).fit(xtr_s).decision_function(xte_s)
    scores["IsolationForest"] = -IsolationForest(n_estimators=200, random_state=0).fit(xtr_s).decision_function(xte_s)
    scores["AutoEncoder"] = ae_scores(xtr_s, xte_s, device=device)

    print("ours...")
    v = train_vade(xtr_s, n_clusters=20, latent_dim=8, epochs=40, warmup=8, seed=0, device=device)
    v.fit_residual_whitener(xtr_s)
    sw_te, sw_tr = v.anomaly_score(xte_s), v.anomaly_score(xtr_s)
    scores["VaDE (ours, window)"] = sw_te
    pca = PCA(24, random_state=0).fit(c_tr)
    g_tr, g_te = E.soft_gamma(v, xtr_s), E.soft_gamma(v, xte_s)
    sc_tr, sc_te = E.soft_ctx_score(pca.transform(c_tr), g_tr, np.ones(len(g_tr)),
                                    pca.transform(c_te), g_te)
    z = lambda a, r: (a - r.mean()) / (r.std() + 1e-9)
    scores["VaDE + context (ours+B)"] = z(sw_te, sw_tr) + np.maximum(0.0, z(sc_te, sc_tr))

    print(f"\n{'method':<26}" + "".join(f"{t:>15}" for t in TYPES) + f"{'AUROC':>8}")
    print("-" * 81)
    for name, s in scores.items():
        sn = s[normal]
        print(f"{name:<26}" + "".join(f"{tpr_at(sn, s[atype == t]):>15.3f}" for t in TYPES)
              + f"{roc_auc_score(yte, s):>8.3f}")
    print("\n(per-type TPR @ 5% FPR; bad_transition snapshot floor = 0.057)")


if __name__ == "__main__":
    main()
