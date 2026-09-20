"""SECOND A8 failure mode: ISOLATED anomaly ABSORBED by a large, wide component (density over-smoothing).

An anomaly with NO real normal neighbours (large kNN distance to train-normal windows) that still gets a
NORMAL score because a wide Gaussian's support reaches it. Computed alongside a8_masking_allsets.py on
the SAME windows / PCA-10 space (loaders and windowing imported from there; PCA is deterministic).

Definitions (byte-identical across datasets):
  knn(x)     : mean Euclidean distance in PCA-10 to the k=10 nearest train-normal windows (train: leave-one-out)
  isolated   : knn(x) > q99 of train knn
  heads      : NC_BIC  = nearest-component NLL of the BIC-selected full-cov GMM (the masking sweep's head)
               WIDE    = mixture NLL of a K=2 full-cov GMM (few wide components)
               NARROW  = mixture NLL of a diagonal GMM with kd = min(80, max(20, n_train//10)) components
                         (the paper's density-head recipe, in PCA-10 instead of the VaDE latent)
  absorbed(head) : isolated AND NLL_head < q50(train NLL_head)      (the coordinator's definition)
  absorbed99(head): isolated AND NLL_head <= q99(train NLL_head)    (isolated yet not flagged at the 1% FPR threshold)
  rank gap   : Spearman rho between knn and NLL_head on anomalies (and on test-normals / train as controls);
               gap = rho(train) - rho(anomalies)
Synthetic positive control SYNTH_ring: a regime whose window-mean traces a hollow ring (eight channels
oscillating at phases i*pi/4, period 600 s, amplitude 3 sd); anomaly = oscillation frozen at the ring's
centre (inside every wide Gaussian's 1-sigma, no neighbours). Expected: WIDE and NC absorb it, kNN flags it,
NARROW absorbs it less.
Outputs: a8_absorption_allsets.jsonl (one row per (dataset, seed), flushed, resumable).
"""
from __future__ import annotations
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import a8_masking_allsets as M
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

OUT = os.path.join(HERE, "a8_absorption_allsets.jsonl")
KNN = 10


def emit(row):
    row["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=M._js) + "\n"); f.flush()
    print("EMIT", json.dumps(row, default=M._js)[:300], flush=True)


def done_keys():
    s = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(line); s.add((r.get("dataset"), r.get("seed")))
            except Exception:
                pass
    return s


def synth_ring(seed=0, d=24, n_steps=300000, dwell=1200, W=60, ST=30):
    """A (origin), C (far), RING: channels 3..10 oscillate at phases i*pi/4 with period 600 and amplitude 3 sd
    (window mean traces a hollow ring). Anomaly = RING segment with the oscillation frozen at the centre."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(0, 1, (d, 3))); u2, u3 = Q[:, 1], Q[:, 2]
    C = {"A": np.zeros(d), "C": 6.0 * u2, "R": 6.0 * u3}; scale = rng.uniform(0.5, 1.5, d)   # ring centre away from A
    def seg(c, L): return c + rng.normal(0, 1, (L, d)) * scale
    def ring(L, t0, amp):
        s = seg(C["R"], L); ph = 2 * np.pi * (t0 + np.arange(L)) / 600.0
        for i in range(8):                                        # ring carried by 8 channels (phases i*pi/4)
            s[:, 3 + i] += amp * np.cos(ph + i * np.pi / 4)
        return s
    def stream(n, anomalies):
        X = np.zeros((n, d)); y = np.zeros(n, int); t = 0
        while t < n:
            L = int(rng.integers(dwell // 2, dwell * 2)); r = ["A", "C", "R"][int(rng.integers(3))]
            if r == "R":
                if anomalies and rng.random() < 0.4:
                    s = ring(L, t, 0.0); yy = np.ones(L, int)          # frozen at the centre
                else:
                    s = ring(L, t, 3.0); yy = np.zeros(L, int)
            else:
                s = seg(C[r], L); yy = np.zeros(L, int)
            X[t:t + L] = s[:n - t]; y[t:t + L] = yy[:n - t]; t += L
        return X, y
    Xn, _ = stream(n_steps, False); Xa, ya = stream(n_steps, True)
    tr, te = M._std([Xn], [(Xa, ya)], None)
    return dict(train=tr, test=te, W=W, ST=ST, note="synthetic over-smoothing control: hollow-ring regime; anomaly = oscillation frozen at the ring centre", nch=d)


LOADERS = dict(M.LOADERS); LOADERS["SYNTH_ring"] = synth_ring


def rho(a, b):
    if len(a) < 5: return float("nan")
    r = spearmanr(a, b).correlation; return float(r) if r == r else float("nan")


def run(name):
    dk = done_keys(); t0 = time.time()
    if all((name, s) in dk for s in M.SEEDS):
        print(f"[{name}] cached"); return
    D = LOADERS[name]()
    if D.get("prewin") is not None:
        Ftr, Fte, yte, _ = D["prewin"]; labelled = True
    else:
        labelled = len(D["test"]) > 0
        Ftr, Str, Ptr, _ = M.windows(D["train"], D["W"], D["ST"], False)
        if labelled: Fte, Ste, Pte, yte = M.windows(D["test"], D["W"], D["ST"], True)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    Xtr = (Ftr - mu) / sd
    sub = np.random.default_rng(0).choice(len(Xtr), min(len(Xtr), 30000), replace=False)
    pca = PCA(M.NPC, random_state=0).fit(Xtr[sub]); Ztr = pca.transform(Xtr)
    Zte = pca.transform((Fte - mu) / sd) if labelled else None
    Zfit = Ztr[sub] if len(Ztr) > 30000 else Ztr
    # kNN distances (train: leave-one-out)
    nn = NearestNeighbors(n_neighbors=KNN + 1).fit(Ztr)
    dtr = nn.kneighbors(Ztr)[0][:, 1:].mean(1)
    q99_knn = float(np.quantile(dtr, 0.99))
    dte = nn.kneighbors(Zte, n_neighbors=KNN)[0].mean(1) if labelled else None
    print(f"[{name}] train {len(Ztr)}" + (f" test {len(Zte)} anom {int((yte==1).sum())}" if labelled else " normal-only") + f" knn q99 {q99_knn:.3f} ({time.time()-t0:.0f}s)", flush=True)
    for seed in M.SEEDS:
        if (name, seed) in dk: continue
        H = M.Head(Zfit, seed)
        kd = min(80, max(20, len(Zfit) // 10))
        wide = GaussianMixture(2, covariance_type="full", reg_covar=1e-3, random_state=seed).fit(Zfit)
        narrow = GaussianMixture(kd, covariance_type="diag", reg_covar=1e-3, random_state=seed).fit(Zfit)
        heads = {"NC_BIC": lambda Z: H.s(Z).min(1), "WIDE": lambda Z: -wide.score_samples(Z), "NARROW": lambda Z: -narrow.score_samples(Z)}
        row = dict(dataset=name, seed=seed, labelled=labelled, K_bic=H.K, kd_narrow=kd, n_train=int(len(Ztr)), knn_q99=q99_knn,
                   isolated_frac_train=float((dtr > q99_knn).mean()))
        if labelled:
            iso = dte > q99_knn; an = yte == 1; no = yte == 0
            row.update(n_anom=int(an.sum()), isolated_frac_anom=float(iso[an].mean()), isolated_frac_testnormal=float(iso[no].mean()))
        for hn, fn in heads.items():
            s_tr = fn(Ztr); q50, q99 = np.quantile(s_tr, 0.5), np.quantile(s_tr, 0.99)
            row[f"{hn}_rho_train"] = rho(dtr, s_tr)
            if labelled:
                s_te = fn(Zte)
                for lab, m in (("anom", an), ("testnormal", no)):
                    row[f"{hn}_absorbed_{lab}"] = float((iso[m] & (s_te[m] < q50)).mean()) if m.any() else float("nan")
                    row[f"{hn}_absorbed99_{lab}"] = float((iso[m] & (s_te[m] <= q99)).mean()) if m.any() else float("nan")
                    row[f"{hn}_rho_{lab}"] = rho(dte[m], s_te[m])
                row[f"{hn}_miss_rate_anom"] = float((s_te[an] <= q99).mean())
                row[f"{hn}_iso_miss_rate_anom"] = float((s_te[an & iso] <= q99).mean()) if (an & iso).any() else float("nan")
                row[f"{hn}_rank_gap"] = row[f"{hn}_rho_train"] - row[f"{hn}_rho_anom"]
                if hn == "NC_BIC":
                    ex = np.where(an & iso & (s_te < q50))[0][:5]
                    row["examples_absorbed"] = [dict(t=int(t), knn=round(float(dte[t]), 3), nll=round(float(s_te[t]), 2), q50=round(float(q50), 2)) for t in ex]
        emit(row)
    print(f"[{name}] done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or list(LOADERS)
    for n in names:
        try:
            run(n)
        except Exception as e:
            import traceback; traceback.print_exc(); emit(dict(dataset=n, seed=None, error=repr(e)))
