"""Build ONE unified per-window scores table per dataset so any difficulty filter can be tried
offline (no re-inference), with proper multi-seed CIs.

Canonical difficulty axis (matches improve_multiseed): maxz = |raw windowed feature|.max over the
first sixth of channels; threshold = 99th percentile of TRAIN-normal maxz (stored as maxz_thr).
Multi-seed methods (LatAD, IF, AE) are stored as (n_seed, n_window) so difficult-AUROC = per-seed
AUROC then mean/std. SOTA (USAD/TranAD/GDN) are single per-window vectors (window-averaged from the
Modal per-timestep dumps). Saves _diagnostics/scores_<DS>.npz.
"""
from __future__ import annotations
import os, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from models_vade import train_vade
from compare_baselines import ae_scores
import eda_real as E

CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}
SEEDS = [0, 1, 2, 3, 4]
SB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sota_bundle", "sota_scores")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")


def win_starts(n, W, stride):
    return list(range(0, n - W + 1, stride))


def loo_residual(Mn, Mte):
    C = Mn.shape[1]; r = np.zeros(len(Mte))
    for c in range(C):
        cols = [j for j in range(C) if j != c]
        lr = LinearRegression().fit(Mn[:, cols], Mn[:, c])
        r += (lr.predict(Mte[:, cols]) - Mte[:, c]) ** 2
    return r / C


def build(name):
    D = E.load(name); fn, W, stride = E.RAW[name]
    Xn_raw, Xa_raw = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
    Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"].astype(int)
    starts = win_starts(len(Xa_raw), W, stride)
    assert len(starts) == len(y), f"{name}: window/label mismatch {len(starts)} vs {len(y)}"

    # --- CANONICAL difficulty axis (improve_multiseed): raw features, first sixth, train 99th pct ---
    C6 = Xte0.shape[1] // 6
    maxz = np.abs(Xte0[:, :C6]).max(1).astype(np.float32)
    maxz_thr = float(np.quantile(np.abs(Xtr0[:, :C6]).max(1), 0.99))

    # --- extra filter stats ---
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xte = ((Xte0 - mu) / sig).astype(np.float32)
    l2 = np.sqrt((Xte ** 2).mean(1)).astype(np.float32)
    # canonical linear-residual filter: discrete-aware (one-hot state-fractions), leave-one-channel-out
    from onehot_filter import build_feats, loco_residual as loco_oh
    Fn, Fa, grp = build_feats(Xn_raw, Xa_raw, W, stride, onehot=True)
    _, linres = loco_oh(Fn, Fa[:len(y)], grp)
    linres = linres.astype(np.float32)

    # --- multi-seed method scores: (n_seed, n_window) ---
    K, LD = CFG[name]; Xtr = ((Xtr0 - mu) / sig).astype(np.float32)
    kd = min(80, max(20, len(Xtr0) // 10))
    latad, s_if, s_ae = [], [], []
    for sd in SEEDS:
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
        v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd)
        v.fit_resid_head(Xtr); v.fit_basin_head(Xtr)
        latad.append(np.asarray(v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto")))
        s_if.append(-IsolationForest(n_estimators=200, random_state=sd).fit(Xtr).score_samples(Xte))
        s_ae.append(ae_scores(Xtr, Xte, seed=sd))
        print(f"    {name} seed {sd} done", flush=True)
    cols = dict(label=y, maxz=maxz, maxz_thr=np.float32(maxz_thr), linres=linres, l2=l2,
                LatAD=np.stack(latad).astype(np.float32),
                IF=np.stack(s_if).astype(np.float32), AE=np.stack(s_ae).astype(np.float32),
                seeds=np.array(SEEDS))

    # --- SOTA per-timestep -> per-window (mean over each window); single vector ---
    for m in ("USAD", "TranAD", "GDN"):
        p = os.path.join(SB, f"score_{m}_{name}.npy")
        if os.path.exists(p):
            ts = np.load(p); ts = ts.mean(1) if ts.ndim > 1 else ts
            if len(ts) == len(Xa_raw):
                cols[m] = np.array([ts[i:i + W].mean() for i in starts], np.float32)
            else:
                print(f"  [{name}] {m}: length {len(ts)} != {len(Xa_raw)}, skipped")
    np.savez(os.path.join(OUT, f"scores_{name}.npz"), **cols)
    multi = [k for k in ("LatAD", "IF", "AE") if k in cols]
    single = [k for k in ("USAD", "TranAD", "GDN") if k in cols]
    print(f"  {name}: saved multiseed{multi} single{single}  n_win={len(y)} anom={int(y.sum())} "
          f"maxz_thr={maxz_thr:.2f}", flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI", "HAI", "SWaT"]):
        build(nm)
