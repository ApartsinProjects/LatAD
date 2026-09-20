"""A3 basin-agreement head: real-data witness + auto-gate validation.

For each dataset (SKAB, WADI, HAI, SWaT) with its REPORTED (K, latent) config:
  - train-only gate signals: rho = frac(train-normal max-resp < 0.5), mean max-resp, basin_lam
  - gate fires? (basin_lam > 0)
  - difficult-subset AUROC: basin head OFF vs FORCED ON (lam=1.0; SKAB also lam=2.0)
  - hard-catch count at the 95%-train threshold
  - invariant I2: when basin_lam==0, use_basin='auto' == use_basin=False (true no-op)

Writes one JSON row per (dataset, arm) to _diagnostics/a3_skab_witness.json incrementally,
flushed, resumable (skips dataset already fully present).
"""
from __future__ import annotations
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
from winfeat import window_features
import eda_real as E

W, ST = 60, 30
CFG = {"SKAB": (16, 6), "WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics", "a3_skab_witness.json")


def win(X, y=None):
    A, B = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], "stats"))
        if y is not None:
            B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)


def au(y, s, mask):
    k = (y == 0) | mask
    return float(roc_auc_score(y[k], s[k]))


def load_rows():
    if os.path.exists(OUT):
        try:
            return json.load(open(OUT))
        except Exception:
            return []
    return []


def save_rows(rows):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def run(name, seed=0):
    rows = load_rows()
    if any(r["dataset"] == name for r in rows):
        print(f"[{name}] already present, skipping", flush=True)
        return
    K, LD = CFG[name]
    D = E.load(name)
    Xtr, _ = win(D["Xn_raw"])
    Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s = ((Xtr - m) / sd).astype(np.float32)
    Xte_s = ((Xte - m) / sd).astype(np.float32)
    # canonical difficulty split: first sixth (window-mean block), 99th pct of train
    C6 = Xte.shape[1] // 6
    triv = np.abs(Xte[:, :C6]).max(1)
    trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99))
    hard = (yw == 1) & ~easy
    n_hard = int(hard.sum())

    kd = min(80, max(20, len(Xtr_s) // 10))
    v = train_vade(Xtr_s, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
    v.fit_residual_whitener(Xtr_s)
    v.fit_latent_density(Xtr_s, k_density=kd)
    v.fit_resid_head(Xtr_s)
    v.fit_basin_head(Xtr_s)

    maxr = v._responsibilities(Xtr_s).max(1)
    rho = float((maxr < 0.5).mean())
    mean_maxr = float(maxr.mean())
    basin_lam = float(getattr(v, "_basin_lam", 0.0))
    gate_fires = basin_lam > 0.0

    # base (OFF) score
    off_te = np.asarray(v.anomaly_score_hard(Xte_s, use_resid="auto", use_basin=False))
    off_tr = np.asarray(v.anomaly_score_hard(Xtr_s, use_resid="auto", use_basin=False))
    # auto arm (baked gate)
    auto_te = np.asarray(v.anomaly_score_hard(Xte_s, use_resid="auto", use_basin="auto"))
    noop_gap = float(np.max(np.abs(auto_te - off_te)))  # I2: ==0 when gate off

    # forced-ON: subtract lam*(agreement - train_mean)/train_std, exactly as the baked head does
    ag_te = v._noise_agreement(Xte_s)
    am, asd = v._basin_ref

    def on_score(lam):
        return off_te - lam * (ag_te - am) / asd

    thr = float(np.quantile(off_tr, 0.95))

    def catch(s):
        t = float(np.quantile(off_tr, 0.95)) if s is off_te else thr
        return int((hard & (s > t)).sum())

    au_off = au(yw, off_te, hard)
    hc_off = int((hard & (off_te > thr)).sum())

    new = []
    new.append(dict(dataset=name, arm="OFF", K=K, latent=LD, seed=seed,
                    rho=rho, mean_maxresp=mean_maxr, basin_lam=basin_lam, gate_fires=gate_fires,
                    diff_auroc=au_off, all_auroc=au(yw, off_te, yw == 1),
                    easy_auroc=au(yw, off_te, easy),
                    hard_catch=hc_off, n_hard=n_hard, noop_gap=noop_gap,
                    kd=kd, n_train_win=int(len(Xtr_s)), n_test_win=int(len(Xte_s))))
    lams = [1.0, 2.0] if name == "SKAB" else [1.0]
    for lam in lams:
        s = on_score(lam)
        new.append(dict(dataset=name, arm=f"FORCED_ON_lam{lam}", K=K, latent=LD, seed=seed,
                        rho=rho, mean_maxresp=mean_maxr, basin_lam=basin_lam, gate_fires=gate_fires,
                        diff_auroc=au(yw, s, hard), all_auroc=au(yw, s, yw == 1),
                        easy_auroc=au(yw, s, easy),
                        hard_catch=int((hard & (s > thr)).sum()), n_hard=n_hard,
                        forced_lam=lam))
    rows.extend(new)
    save_rows(rows)
    for r in new:
        print(f"[{name}] {r['arm']:<18} diffAUROC={r['diff_auroc']:.3f} "
              f"hardCatch={r['hard_catch']}/{n_hard} lam={basin_lam:.2f} "
              f"rho={rho:.2f} meanMaxR={mean_maxr:.2f} gate={gate_fires} noop_gap={new[0]['noop_gap']:.2e}",
              flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["SKAB", "WADI", "HAI", "SWaT"]):
        run(nm)
