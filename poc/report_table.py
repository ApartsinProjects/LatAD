"""Standardized results table: one JSON with 3 datasets x {easy,hard,all} x {AUC,F1,FPR},
rows = methods (baselines + SOTA + ours), plus per-dataset counts. Every fit uses train-normal
only; F1/FPR use the ORACLE best-F1 threshold swept on test (field-standard). For every subset
the evaluation set = ALL test-normal windows + that subset's anomalies, so the normal count is
constant across easy/hard/all; only the positive set changes.
"""
from __future__ import annotations
import os, sys, json, numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score
from models_vade import train_vade
from compare_baselines import ae_scores
import eda_real as E

RES = os.path.join(os.path.dirname(__file__), "sota_bundle", "results")
STRIDE = {"WADI": 30, "HAI": 30, "SKAB": 30}   # match eda_real's unified (W=60, stride=30) grid
# winning VaDE-based (K, latent_dim) per dataset, from the BIC/participation-ratio sweep
# (sweep_config.py): HAI 40/16 and SKAB 16/6 match their selection metrics; WADI stays 20/10.
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SKAB": (16, 6)}
GROUP = {"trivial max|z|": "Baseline", "IsolationForest": "Baseline", "AutoEncoder": "Baseline",
         "USAD": "SOTA", "TranAD": "SOTA", "GDN": "SOTA",
         "VaDE": "Ours", "VaDE-hard+resid(auto)": "Ours"}


def _thresholds(ss):
    """Distinct candidate thresholds (dedup ties so a saturated/plateaued score doesn't collapse
    the sweep to one value -> F1=0). Inclusive '>=' compare pairs with these."""
    qs = np.unique(np.quantile(ss, np.linspace(0.50, 1.0, 120)))
    return qs if len(qs) >= 2 else np.unique(ss)


def _bestf1(ys, ss):
    return max((f1_score(ys, ss >= t) for t in _thresholds(ss)), default=0.0)


def method_metrics(y, s, easy, hard):
    """Comparable metrics across subsets:
      AUROC  : prevalence-independent, all-normal + subset positives.
      F1     : PREVALENCE-MATCHED -- subsample normals so each subset has the SAME
               anomaly:normal ratio as 'all' (mean over 5 seeds), so F1 is comparable.
      FPR    : at a FIXED operating point (best-F1 threshold on the ALL subset),
               measured on all normals -> identical across subsets, prevalence-free.
    """
    yneg = np.where(y == 0)[0]
    ratio = (y == 1).sum() / max(1, len(yneg))                 # all-subset anom:normal
    _, thr_fix = max(((f1_score(y, s >= t), t) for t in _thresholds(s)), key=lambda p: p[0])
    fpr_fix = round(float((s[y == 0] >= thr_fix).mean()), 3)
    rng = np.random.default_rng(0)
    out = {}
    for key, mask in [("all", y == 1), ("easy", easy), ("hard", hard)]:
        npos = int(mask.sum())
        if npos < 3:
            out[key] = [None, None, None]; continue
        au = roc_auc_score(y[(y == 0) | mask], s[(y == 0) | mask])
        ntgt = min(len(yneg), max(npos, int(round(npos / ratio))))   # normals to match 'all' prevalence
        f1s = []
        for _ in range(5):
            sub = rng.choice(yneg, ntgt, replace=False)
            ys = np.r_[np.ones(npos), np.zeros(ntgt)]; ss = np.r_[s[mask], s[sub]]
            f1s.append(_bestf1(ys, ss))
        out[key] = [round(float(au), 3), round(float(np.mean(f1s)), 3), fpr_fix]
    return out


def win_pointscore(s, W, st):
    return np.array([s[i:i + W].max() for i in range(0, len(s) - W + 1, st)])


def run(name):
    D = E.load(name); Xtr, Xte, y = D["Xn_w"], D["Xa_w"], D["ya_w"]
    # difficulty split on RAW window features (definition unchanged)
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (y == 1) & (triv > np.quantile(trn, 0.99)); hard = (y == 1) & ~easy
    K, LD = CFG[name]
    # per-feature standardization on train-normal for the MODELS (VaDE/IF/AE all benefit; window
    # stats features have heterogeneous scales -- mean vs std vs range). Was missing -> depressed
    # the VaDE rows and made VaDE-hard look worse than VaDE (a table artifact, not the method).
    mu, sig = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = ((Xtr - mu) / sig).astype(np.float32); Xte = ((Xte - mu) / sig).astype(np.float32)
    S = {"trivial max|z|": triv,   # the difficulty-defining detector: strong on easy, ~chance on difficult
         "IsolationForest": -IsolationForest(n_estimators=200, random_state=0).fit(Xtr).decision_function(Xte),
         "AutoEncoder": ae_scores(Xtr, Xte, device="cpu")}
    v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=0, device="cpu")
    kd = min(80, max(20, len(Xtr) // 10))    # scale density components with train size (SKAB is small)
    v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd)
    v.fit_resid_head(Xtr); v.fit_basin_head(Xtr)
    sV_te, sV_tr = v.anomaly_score(Xte), v.anomaly_score(Xtr)
    sH_te = v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto")
    sH_tr = v.anomaly_score_hard(Xtr, use_resid="auto", use_basin="auto")
    S["VaDE"] = sV_te
    S["VaDE-hard+resid(auto)"] = sH_te
    W, st = D["W"], STRIDE[name]; Nfull = len(D["Xa_raw"])
    for mdl in ["USAD", "TranAD", "GDN"]:
        p = f"{RES}/score_{mdl}_{name}.npy"
        if os.path.exists(p):
            sp = np.load(p); sp = sp.mean(1) if sp.ndim > 1 else sp
            if len(sp) < 0.9 * Nfull: sp = np.repeat(sp, int(round(Nfull / len(sp))))
            if len(sp) < Nfull: sp = np.pad(sp, (0, Nfull - len(sp)), mode="edge")
            sw = win_pointscore(sp[:Nfull], W, st); S[mdl] = sw[:len(y)]

    n_norm = int((y == 0).sum()); n_an = int((y == 1).sum())
    def nn(npos):                              # normals actually used for the prevalence-matched F1
        return int(min(n_norm, max(npos, round(npos * n_norm / max(1, n_an)))))
    out = {"counts": {"n_test": int(len(y)), "n_normal": n_norm, "n_anom": n_an,
                      "n_easy": int(easy.sum()), "n_hard": int(hard.sum()),
                      "nn_easy": nn(int(easy.sum())), "nn_hard": nn(int(hard.sum())),
                      "nn_all": n_norm}, "methods": {}}
    for nm, s in S.items():
        m = min(len(s), len(y)); s2, y2, e2, h2 = s[:m], y[:m], easy[:m], hard[:m]
        mm = method_metrics(y2, s2, e2, h2)
        out["methods"][nm] = {"group": GROUP.get(nm, "Ours"), **mm}
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "latad_table.json")
    html = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "latad_results.html")
    res = {nm: run(nm) for nm in ["WADI", "HAI", "SKAB"]}
    json.dump(res, open(dst, "w"), indent=1)
    print("wrote", dst)
    # ALWAYS also render + save the HTML table to disk on every update
    import subprocess
    subprocess.run([sys.executable, os.path.join(here, "render_table.py"), dst, html])
    print("wrote", html)
