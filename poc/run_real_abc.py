"""Run the A-B-C pipeline + baselines on the real CPS datasets (SKAB, HAI, WADI, SWaT).
Real data has binary labels only -> AUROC + TPR@5%FPR + F1. Baselines are LOF/IF/AE;
ours = window VaDE (+ event transition branch + optional neural c_t context via A->B).

Honest expectation: HAI/SKAB anomalies are snapshot-detectable, so the trajectory
branches should add little there; WADI/SWaT unknown.

CAVEAT (audit): the event + context (B) branches assume the test windows form ONE
time-ordered walk. That holds for WADI (single attack file) but NOT for SWaT (test =
held-out-normal ++ attack, one seam at the label flip) or HAI (windows concatenated
across files). On those, any transition-surprise / GRU-context score AT a seam is an
artifact, not real trajectory signal (on SWaT the seam aligns with the normal->attack
flip, so it spuriously flatters +event/+context). Trust only the window-VaDE column on
SWaT/HAI; the trajectory branches there are reported for completeness, not as results.
"""
from __future__ import annotations
import sys, numpy as np, torch
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score
from models_vade import train_vade
import explore_c as E
from compare_baselines import ae_scores
from event_detector import segment


def metrics(y, s):
    au = roc_auc_score(y, s)
    thr = np.quantile(s[y == 0], 0.95)                 # 5% FPR on normal
    pred = s > thr
    return au, float((s[y == 1] > thr).mean()), f1_score(y, pred)


def run(name, ds, K=24, do_B=True, device="cpu", seed=0):
    xtr, xte, yte = ds.x_train, ds.x_test, ds.y_test.astype(int)
    m, sd = xtr.mean(0), xtr.std(0) + 1e-8
    xtr_s, xte_s = ((xtr - m) / sd).astype(np.float32), ((xte - m) / sd).astype(np.float32)
    print(f"\n########## {name}  train={len(xtr)} test={len(xte)} feat={xtr.shape[1]} "
          f"anom={yte.mean():.3f}  K={K} ##########")
    S = {}
    S["LOF"] = -LocalOutlierFactor(30, novelty=True).fit(xtr_s).decision_function(xte_s)
    S["IsolationForest"] = -IsolationForest(n_estimators=200, random_state=seed).fit(xtr_s).decision_function(xte_s)
    S["AutoEncoder"] = ae_scores(xtr_s, xte_s, device=device)

    v = train_vade(xtr_s, n_clusters=K, latent_dim=10, epochs=40, warmup=8, seed=seed, device=device)
    v.fit_residual_whitener(xtr_s)
    sw_te, sw_tr = v.anomaly_score(xte_s), v.anomaly_score(xtr_s)
    S["VaDE (ours, window)"] = sw_te

    # event transition-surprise branch (free), from the VaDE's soft gamma in time order
    g_tr, g_te = E.soft_gamma(v, xtr_s), E.soft_gamma(v, xte_s)
    ev = segment(g_tr); Tc = np.ones((K, K)) * 0.5
    for a, b in zip(ev[:-1], ev[1:]):
        Tc[a[0], b[0]] += 1
    P = Tc / Tc.sum(1, keepdims=True)
    def evsurp(g):
        e = segment(g); out = np.zeros(len(g))
        for i in range(1, len(e)):
            out[e[i][1]:e[i][1] + e[i][2]] = -np.log(P[e[i - 1][0], e[i][0]] + 1e-12)
        return out
    st_te, st_tr = evsurp(g_te), evsurp(g_tr)
    z = lambda a, r: (a - r.mean()) / (r.std() + 1e-9)
    S["VaDE + event (ours)"] = z(sw_te, sw_tr) + np.maximum(0.0, z(st_te, st_tr))

    if do_B:
        import ldt_b
        from sklearn.decomposition import PCA
        B = ldt_b.TrajectoryEncoderB(K=K, emb_dim=48, ctx_dim=96, n_layers=2,
                                     centroids=v.mu_c.detach().cpu().numpy(), backbone="gru")
        ldt_b.train_B(B, g_tr, np.maximum(g_tr.mean(0), 1e-4), epochs=40, seg_len=512, stride=128,
                      batch_segs=16, lr=3e-3, max_train_offset=64, device=device, seed=seed)
        c_tr = ldt_b.emit_context(B, g_tr, device=device); c_te = ldt_b.emit_context(B, g_te, device=device)
        pca = PCA(min(24, c_tr.shape[1]), random_state=0).fit(c_tr)
        atr = g_tr.argmax(1); ate = g_te.argmax(1)
        sc_tr, sc_te = E.mode_ctx_score(pca.transform(c_tr), atr, pca.transform(c_te), ate, K)
        S["VaDE + context (ours+B)"] = z(sw_te, sw_tr) + np.maximum(0.0, z(sc_te, sc_tr))

    print(f"{'method':<26}{'AUROC':>8}{'TPR@5%':>9}{'F1':>7}")
    for name_, s in S.items():
        au, tp, f1 = metrics(yte, s)
        print(f"{name_:<26}{au:>8.3f}{tp:>9.3f}{f1:>7.3f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "skab,hai"
    do_B = "--noB" not in sys.argv
    if "skab" in which:
        import skab; run("SKAB", skab.load_skab(), K=16, do_B=do_B)
    if "hai" in which:
        import hai; run("HAI", hai.load_hai(), K=24, do_B=do_B)
    if "wadi" in which:
        import wadi; run("WADI", wadi.load_wadi(), K=20, do_B=do_B)
    if "swat" in which:
        import swat; run("SWaT", swat.load_swat(), K=20, do_B=do_B)   # needs Dec-2015 files
