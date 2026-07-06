"""
LDT — simple EMA version (latent-digital-twin branch, first prototype).

The twin encodes the PAST as a multi-speed summary of which operating modes the
system has been in. No training, no forecasting, no anomaly logic:

  1. cluster normal windows into K modes (GMM) -> per-window soft assignment gamma;
  2. per run, in time order, keep three EMAs of gamma (short/med/long occupancy),
     plus dwell time, previous mode, and a window-sequence timestamp from run start;
  3. context c_t = [c_short, c_med, c_long, prev_mode(1-hot), log(1+dwell), log(1+t)].

Then CVaDE = the usual VaDE detector, run on [window features (+) c_t]. This
script trains CVaDE with and without c_t on HAI and reports the difference.

Run:  C:/Python314/python.exe ldt_ema.py
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture

import component2
import hai as H
from models_vade import train_vade

W, STRIDE, REP, K = 60, 60, "stats", 20
ALPHAS = (0.30, 0.05, 0.005)                      # short / med / long EMA rates
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_runs(files, sensors):
    """List of (window_features, labels) per file, in temporal order (= one run)."""
    runs = []
    for f in files:
        w, l = H._windows(H._read(f), sensors, W, STRIDE, REP)
        if len(w):
            runs.append((w.astype(np.float64), l))
    return runs


def ema_context(gamma):
    """Multi-speed summary of the past mode sequence for ONE run (time-ordered)."""
    T, k = gamma.shape
    cs = [gamma[0].copy() for _ in ALPHAS]
    prev = cur = int(gamma[0].argmax()); dwell = 0
    rows = []
    for t in range(T):
        g = gamma[t]
        if t > 0:
            for i, a in enumerate(ALPHAS):
                cs[i] = a * g + (1 - a) * cs[i]
            m = int(g.argmax())
            if m == cur:
                dwell += 1
            else:
                prev, cur, dwell = cur, m, 0
        prev_oh = np.zeros(k); prev_oh[prev] = 1.0
        rows.append(np.concatenate([cs[0], cs[1], cs[2], prev_oh,
                                    [np.log1p(dwell), np.log1p(t)]]))   # t = window seq from run start
    return np.asarray(rows)


def main():
    root = Path("datasets/hai/hai-20.07")
    trf = sorted(glob(str(root / "train*.csv.gz")))
    tef = sorted(glob(str(root / "test*.csv.gz")))
    sensors = [c for c in H._read(trf[0]).columns if c not in H.LABELS]
    tr_runs, te_runs = load_runs(trf, sensors), load_runs(tef, sensors)

    # standardize window features on train; fit K modes (GMM) on train normal
    Xtr_all = np.concatenate([r[0] for r in tr_runs])
    mu, sd = Xtr_all.mean(0), Xtr_all.std(0) + 1e-8
    sw = lambda X: ((X - mu) / sd).astype(np.float32)
    gmm = GaussianMixture(K, covariance_type="diag", reg_covar=1e-4,
                          random_state=0, n_init=3).fit(sw(Xtr_all))

    def assemble(runs):
        Xw = np.concatenate([sw(r[0]) for r in runs])
        C = np.concatenate([ema_context(gmm.predict_proba(sw(r[0]))) for r in runs])
        y = np.concatenate([r[1] for r in runs])
        return Xw, C, y

    Xw_tr, C_tr, y_tr = assemble(tr_runs)
    Xw_te, C_te, y_te = assemble(te_runs)
    cmu, csd = C_tr.mean(0), C_tr.std(0) + 1e-8
    C_tr = ((C_tr - cmu) / csd).astype(np.float32)
    C_te = ((C_te - cmu) / csd).astype(np.float32)

    rng = np.random.default_rng(0)
    cal = rng.permutation(np.where(y_te == 0)[0])[:np.sum(y_te == 0) // 2]

    def evaluate(xitr, xite, tag):
        np.random.seed(0); torch.manual_seed(0)
        v = train_vade(xitr, n_clusters=20, latent_dim=10, epochs=60, seed=0, device=device)
        v.fit_residual_whitener(xitr)
        te = component2.vade_scores(v, xitr, xite)[0]["VaDE + basin (full, ours)"][1]
        thr = np.quantile(te[cal], 0.95)
        print(f"  {tag:<18} AUROC={roc_auc_score(y_te, te):.3f} "
              f"AUPRC={average_precision_score(y_te, te):.3f} "
              f"TPR@5%={float((te[y_te == 1] > thr).mean()):.3f}")

    print(f"HAI: train={len(y_tr)} test={len(y_te)} winfeat={Xw_tr.shape[1]} "
          f"ctx={C_tr.shape[1]} K={K} anomaly-frac={y_te.mean():.3f}")
    evaluate(Xw_tr, Xw_te, "window only")
    evaluate(np.concatenate([Xw_tr, C_tr], 1), np.concatenate([Xw_te, C_te], 1),
             "window + LDT-EMA")


if __name__ == "__main__":
    main()
