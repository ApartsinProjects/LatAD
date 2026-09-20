"""Train-to-test drift of the LOO linear relations. For each dataset: fraction of TEST-NORMAL windows
whose LinRes exceeds the TRAIN-normal p99 (false-alarm rate), median LinRes ratio test-normal/train,
top drifting channels (largest median normalised residual on test-normal), difficult-vs-TRAIN-normal
AUROC (do the anomalies break the train relations?), and difficult-vs-test-normal AUROC (canonical).
Same for the LatAD score (LatAD_train in the npz) for a like-for-like drift comparison."""
from __future__ import annotations
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
import eda_real as E
from onehot_filter import build_feats

HERE = os.path.dirname(os.path.abspath(__file__))


def loco(Fn, Fa, grp):
    Rtr = np.zeros_like(Fn); Rte = np.zeros_like(Fa)
    for j in range(Fn.shape[1]):
        cols = np.where(grp != grp[j])[0]
        m = LinearRegression().fit(Fn[:, cols], Fn[:, j])
        Rtr[:, j] = (m.predict(Fn[:, cols]) - Fn[:, j]) ** 2
        Rte[:, j] = (m.predict(Fa[:, cols]) - Fa[:, j]) ** 2
    return Rtr, Rte


out = {}
for name in (sys.argv[1:] or ["WADI_clean", "SWaT_canon", "HAI"]):
    D = E.load(name); fn, W, stride = E.RAW[name]; ch = D["ch"]
    Xn, Xa = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
    Z = np.load(os.path.join(HERE, f"scores_{name}.npz")); y = Z["label"].astype(int)
    diff = (y == 1) & (Z["maxz"] <= float(Z["maxz_thr"])); nrm = y == 0
    Fn, Fa, grp = build_feats(Xn, Xa, W, stride, onehot=True); Fa = Fa[:len(y)]
    Rtr, Rte = loco(Fn, Fa, grp)
    str_, ste = Rtr.mean(1), Rte.mean(1)
    p99 = np.quantile(str_, 0.99)
    r = dict(n_train=int(len(str_)), n_test_normal=int(nrm.sum()), n_diff=int(diff.sum()))
    r["lin_testnormal_FP_at_train_p99"] = float((ste[nrm] > p99).mean())
    r["lin_diff_TP_at_train_p99"] = float((ste[diff] > p99).mean())
    r["lin_median_ratio_testnormal_over_train"] = float(np.median(ste[nrm]) / np.median(str_))
    r["lin_au_diff_vs_TRAINnormal"] = float(roc_auc_score(np.r_[np.zeros(len(str_)), np.ones(diff.sum())], np.r_[str_, ste[diff]]))
    r["lin_au_diff_vs_testnormal"] = float(roc_auc_score(y[nrm | diff], ste[nrm | diff]))
    r["lin_au_testnormal_vs_train"] = float(roc_auc_score(np.r_[np.zeros(len(str_)), np.ones(nrm.sum())], np.r_[str_, ste[nrm]]))
    # drift channels: median normalised residual on test-normal per channel
    scale = Rtr.mean(0) + 1e-12; Rn = Rte / scale
    chan = {}
    for j, g in enumerate(grp):
        chan.setdefault(int(g), []).append(j)
    med = {ch[g]: float(np.median(Rn[nrm][:, js].sum(1))) for g, js in chan.items()}
    top = sorted(med.items(), key=lambda kv: -kv[1])[:8]
    r["drift_channels_top8_median_norm_resid_testnormal"] = [(k, round(v, 1)) for k, v in top]
    r["n_channels_median_norm_resid_gt3_testnormal"] = int(sum(v > 3 for v in med.values()))
    # raw-unit (score-driver) attribution on difficult windows: share of the canonical mean residual
    raw_share, raw_top = [], {}
    for w in np.where(diff)[0]:
        cs = {ch[g]: float(Rte[w, js].sum()) for g, js in chan.items()}
        items = sorted(cs.items(), key=lambda kv: -kv[1]); v = np.array([x for _, x in items]); p = v / v.sum()
        H = -(p[p > 0] * np.log(p[p > 0])).sum()
        raw_share.append((float(p[0]), float(p[:3].sum()), float(np.exp(H))))
        raw_top[items[0][0]] = raw_top.get(items[0][0], 0) + 1
    rs = np.array(raw_share)
    r["raw_share_diff_top1_med"], r["raw_share_diff_top3_med"], r["raw_share_diff_neff_med"] = [float(np.median(rs[:, i])) for i in range(3)]
    r["raw_top1_channel_counts_diff"] = sorted(raw_top.items(), key=lambda kv: -kv[1])[:8]
    # LatAD like-for-like (mean over seeds)
    lt, la = Z["LatAD_train"].mean(0), Z["LatAD"].mean(0)
    q = np.quantile(lt, 0.99)
    r["latad_testnormal_FP_at_train_p99"] = float((la[nrm] > q).mean())
    r["latad_diff_TP_at_train_p99"] = float((la[diff] > q).mean())
    r["latad_au_testnormal_vs_train"] = float(roc_auc_score(np.r_[np.zeros(len(lt)), np.ones(nrm.sum())], np.r_[lt, la[nrm]]))
    r["latad_au_diff_vs_testnormal"] = float(roc_auc_score(y[nrm | diff], la[nrm | diff]))
    out[name] = r
    print(f"\n=== {name} ===", flush=True)
    for k, v in r.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}", flush=True)
    json.dump(out, open(os.path.join(HERE, "loo_baseline_drift.json"), "w"), indent=1)
