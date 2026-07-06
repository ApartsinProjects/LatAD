"""
Temporal MIIM generator — a time-series benchmark that exercises EVERY phenomenon
the framework targets, so the two detection channels (snapshot vs trajectory) can
each be validated in a controlled setting (the static make_miim cannot, and the
real testbeds' anomalies turned out to be snapshot-only).

Normal operation = a stochastic walk over K operating modes:
  - MASSIVE IMBALANCED MULTIMODALITY: K modes, Zipf-imbalanced stationary frequency;
  - BOUNDED modes: within a mode the regime coordinate is an OU process, clipped;
  - MODE TRANSITION STATE MACHINE: sparse allowed-successor graph (forbidden edges);
  - per-mode DWELL times, log-uniform (some modes brief, some long) -> temporal scale mix;
  - TEMPORAL MULTISCALE: per-channel smoothing time constants (slow/inertial vs fast);
  - HETEROGENEOUS + QUANTIZED sensing (per-channel scale + resolution).

Anomalies (test stream only), spanning snapshot AND history-dependent:
  - pocket   (snapshot): a short between-two-modes segment;
  - drift    (history) : the regime coordinate slowly pushed out of bounds over a segment;
  - bad_transition (history): a FORBIDDEN mode->mode jump (window values legal, path invalid).

Returns windowed train (normal) / test (labelled) plus per-window anomaly TYPE and the
within-run window index, so a snapshot detector and a trajectory detector can be compared.
"""

from __future__ import annotations

import numpy as np

from data import _mode_manifold, _zipf_weights
from winfeat import window_features


def _graph(K, rng, n_succ=3):
    P = np.zeros((K, K)); succ = {}
    for i in range(K):
        cand = [j for j in range(K) if j != i]
        s = rng.choice(cand, size=min(n_succ, len(cand)), replace=False)
        P[i, s] = rng.dirichlet(np.ones(len(s)))
        succ[i] = set(int(j) for j in s)
    return P, succ


def make_temporal_miim(n_modes=20, n_features=24, latent_dim=3, seed=0,
                       n_train=60000, n_test=40000, W=40, stride=20,
                       p_anom=0.04, quantize=True):
    rng = np.random.default_rng(seed)
    weights = _zipf_weights(n_modes, 2.0, rng)                       # imbalanced mode freq
    mode_scale = rng.uniform(0.25, 0.7, n_modes)
    dwell_mean = np.exp(rng.uniform(np.log(15), np.log(220), n_modes))  # log-uniform dwell
    chan_alpha = rng.uniform(0.05, 0.95, n_features)                 # per-channel time constant
    chan_scale = rng.uniform(0.4, 2.4, n_features)
    q_frac = rng.choice([0.02, 0.05, 0.10, 0.30, 0.80], size=n_features, p=[.4, .25, .2, .1, .05])
    P, succ = _graph(n_modes, rng)

    def ou_segment(mode, dwell, drift):
        u = rng.normal(0, mode_scale[mode], latent_dim)
        out = []
        for k in range(dwell):
            u = 0.85 * u + 0.15 * rng.normal(0, mode_scale[mode], latent_dim)
            if drift is not None:
                u = u + drift * (k / max(1, dwell))                 # slide out of bounds
            uc = u if drift is not None else np.clip(u, -2 * mode_scale[mode], 2 * mode_scale[mode])
            out.append(_mode_manifold(uc[None], mode, n_features, rng)[0])
        return np.asarray(out)

    def gen(T, inject):
        X, M, A, TY = [], [], [], []
        cur = int(rng.choice(n_modes, p=weights)); pending_bad = False
        while len(X) < T:
            drift = None; atype = "none"; anom = 0
            if inject and rng.random() < p_anom:                    # drift anomaly
                drift = rng.normal(0, 1, latent_dim) * mode_scale[cur] * 3.0
                atype, anom = "drift", 1
            dwell = max(4, int(rng.exponential(dwell_mean[cur])))
            seg = ou_segment(cur, dwell, drift)
            lab = [cur] * len(seg); an = [anom] * len(seg); ty = [atype] * len(seg)
            if pending_bad:
                # Mark windows SETTLED inside the (valid) target mode, AFTER the
                # straddle — these are snapshot-normal; only the PATH here (a
                # forbidden predecessor) is invalid, so only the trajectory
                # channel can catch them. (Straddle windows are left unlabelled:
                # a mode-mix looks like any transition to a snapshot detector.)
                for k in range(W, min(dwell, W + 2 * stride)):
                    an[k], ty[k] = 1, "bad_transition"
                pending_bad = False
            X.extend(seg); M.extend(lab); A.extend(an); TY.extend(ty)
            if inject and rng.random() < p_anom:                    # pocket (snapshot) anomaly
                a, b = rng.choice(n_modes, size=2, replace=False)
                mid = 0.5 * (_mode_manifold(np.zeros((1, latent_dim)), a, n_features, rng)[0]
                             + _mode_manifold(np.zeros((1, latent_dim)), b, n_features, rng)[0])
                pk = mid + rng.normal(0, 0.05, (W, n_features))
                X.extend(pk); M.extend([-1] * W); A.extend([1] * W); TY.extend(["pocket"] * W)
            # next mode
            if inject and rng.random() < p_anom:                    # forbidden transition
                forb = [j for j in range(n_modes) if j not in succ[cur] and j != cur]
                cur = int(rng.choice(forb)) if forb else cur; pending_bad = True
            else:
                p = P[cur]; cur = int(rng.choice(n_modes, p=p / p.sum())) if p.sum() > 0 \
                    else int(rng.choice(n_modes, p=weights))
        X = np.asarray(X[:T]); M = np.asarray(M[:T]); A = np.asarray(A[:T]); TY = np.asarray(TY[:T], object)
        # temporal multiscale: per-channel smoothing (different time constants)
        S = np.empty_like(X); s = X[0].copy()
        for t in range(len(X)):
            s = chan_alpha * X[t] + (1 - chan_alpha) * s; S[t] = s
        # heterogeneous noise
        S = S + rng.normal(0, 1, S.shape) * chan_scale * 0.12
        return S, M, A, TY

    Xtr, Mtr, _, _ = gen(n_train, inject=False)                     # normal only
    Xte, Mte, Ate, TYte = gen(n_test, inject=True)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    if quantize:
        step = q_frac * sd
        Xtr = np.round(Xtr / step) * step
        Xte = np.round(Xte / step) * step
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

    def windowize(Z, M, A, TY, rep="stats"):
        Xw, y, atype, mode = [], [], [], []
        for i in range(0, len(Z) - W + 1, stride):
            Xw.append(window_features(Z[i:i + W], rep))
            aw = A[i:i + W]
            y.append(int(aw.mean() > 0.5))
            # dominant anomaly type in the window (else 'none')
            tw = TY[i:i + W][aw == 1]
            atype.append(tw[0] if len(tw) else "none")
            mode.append(int(np.bincount(M[i:i + W][M[i:i + W] >= 0], minlength=n_modes).argmax())
                        if (M[i:i + W] >= 0).any() else -1)
        return np.asarray(Xw, np.float32), np.asarray(y, int), np.asarray(atype, object), np.asarray(mode, int)

    Xw_tr, y_tr, _, mode_tr = windowize(Ztr, Mtr, np.zeros(len(Ztr), int), np.full(len(Ztr), "none", object))
    Xw_te, y_te, atype_te, mode_te = windowize(Zte, Mte, Ate, TYte)
    return {
        "x_train": Xw_tr, "x_test": Xw_te, "y_test": y_te,
        "atype_test": atype_te, "mode_train": mode_tr, "mode_test": mode_te,
        "n_features": Xw_tr.shape[1], "n_modes": n_modes, "W": W,
        "graph": succ, "weights": weights,
    }


if __name__ == "__main__":
    d = make_temporal_miim(seed=0)
    print(f"temporal-MIIM: train={len(d['x_train'])} test={len(d['x_test'])} "
          f"winfeat={d['n_features']} modes={d['n_modes']} anomaly-frac={d['y_test'].mean():.3f}")
    at = d["atype_test"]
    for t in ["none", "pocket", "drift", "bad_transition"]:
        print(f"  {t:<16} {int((at == t).sum())} windows")
    print(f"  distinct modes seen (train): {len(np.unique(d['mode_train']))}/{d['n_modes']}")
