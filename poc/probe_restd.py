"""Does run_real_abc's per-feature re-standardization help or hurt? The real loaders
already standardize raw channels; re-standardizing the window-feature vector rescales
the near-constant-channel extremeness that carries WADI attacks. Test VaDE + IF with
and without it, same config, one pass per dataset (construct-matched)."""
from __future__ import annotations
import sys, numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score
from models_vade import train_vade


def metrics(y, s):
    au = roc_auc_score(y, s); thr = np.quantile(s[y == 0], 0.95)
    return au, float((s[y == 1] > thr).mean()), f1_score(y, s > thr)


def run(name, ds, K):
    xtr, xte, yte = ds.x_train, ds.x_test, ds.y_test.astype(int)
    print(f"\n### {name} train={len(xtr)} test={len(xte)} feat={xtr.shape[1]} anom={yte.mean():.3f}")
    print(f"{'variant':<16}{'IF_AU':>8}{'IF_TP':>8}{'V_AU':>8}{'V_TP':>8}{'V_F1':>8}")
    for tag, restd in [("re-std (old)", True), ("no re-std", False)]:
        if restd:
            m, sd = xtr.mean(0), xtr.std(0) + 1e-8
            a, b = ((xtr - m) / sd).astype(np.float32), ((xte - m) / sd).astype(np.float32)
        else:
            a, b = xtr.astype(np.float32), xte.astype(np.float32)
        ifs = -IsolationForest(n_estimators=200, random_state=0).fit(a).decision_function(b)
        v = train_vade(a, n_clusters=K, latent_dim=10, epochs=40, warmup=8, seed=0, device="cpu")
        v.fit_residual_whitener(a); vs = v.anomaly_score(b)
        ia, it, _ = metrics(yte, ifs); va, vt, vf = metrics(yte, vs)
        print(f"{tag:<16}{ia:>8.3f}{it:>8.3f}{va:>8.3f}{vt:>8.3f}{vf:>8.3f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "wadi,hai,skab"
    if "wadi" in which:
        import wadi; run("WADI", wadi.load_wadi(), 20)
    if "hai" in which:
        import hai; run("HAI", hai.load_hai(), 24)
    if "skab" in which:
        import skab; run("SKAB", skab.load_skab(), 16)
