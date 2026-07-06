"""
Unified MIIM generator (A1-A10), implementing poc/synthetic_data.html.

Phase A instantiates the system (modes, two-tier graph, geometry, typed channels,
per-mode ablation flags); Phase B walks it (semi-Markov, density-scaled dwell, OU
within a mode) and injects the six labelled anomaly families into the test stream.

Run:  C:/Python314/python.exe miim_gen.py
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from itertools import product

# ----------------------------- Phase A: instantiate the system -----------------------------

@dataclass
class System:
    K: int; d: int; r: int
    pi: np.ndarray                      # (K,) Zipf occupancy
    succ: dict; P: dict; forbidden: dict  # graph
    dwell: np.ndarray                   # (K,2) [D-,D+] per mode
    mu: np.ndarray; B: np.ndarray; sigma: np.ndarray; kappa: np.ndarray  # geometry
    ctype: np.ndarray                   # (d,) channel type id
    We: np.ndarray; s: np.ndarray; delta: np.ndarray; theta: np.ndarray  # sensing
    eta: np.ndarray                     # (K,) non-stationary noise multiplier per mode
    flag_hist: np.ndarray; acc_load: np.ndarray  # A10 accumulator
    acc_chan: int                       # index of the accumulator channel (or -1)
    centroid: np.ndarray = field(default=None)   # (K,d) signal-space centroids

TYPES = {"sensor": 0, "actuator": 1, "state": 2, "setpoint": 3, "accumulator": 4}


def _zipf(K, beta, rng):
    w = 1.0 / np.arange(1, K + 1) ** beta
    rng.shuffle(w)
    return w / w.sum()


def sample_system(n_sub=4, states=(3, 4, 5, 3), d=24, r=3, beta=2.1, p_forbid=0.5,
                  m_core=8, p_curv=0.3, q_noise=6, seed=0, K_cap=90) -> System:
    rng = np.random.default_rng(seed)
    # A1/A2: compositional modes, pruned by a compatibility relation.
    subs = [list(range(s)) for s in states[:n_sub]]
    tuples = list(product(*subs))
    # forbid each cross-subsystem sub-state pair independently; adapt p_forbid so a healthy
    # subset (target ~half of candidates) survives -- with many pairs, small p per pair suffices.
    def survivors(p):
        fp = {(j1, a, j2, b): rng.random() < p
              for j1 in range(n_sub) for j2 in range(j1 + 1, n_sub)
              for a in subs[j1] for b in subs[j2]}
        out = []
        for tup in tuples:
            if not any(fp[(j1, tup[j1], j2, tup[j2])]
                       for j1 in range(n_sub) for j2 in range(j1 + 1, n_sub)):
                out.append(tup)
        return out
    p = p_forbid; modes = survivors(p)
    while len(modes) < max(40, K_cap // 2) and p > 0.02:      # relax if over-pruned
        p *= 0.6; modes = survivors(p)
    rng.shuffle(modes)
    modes = modes[:K_cap]
    K = len(modes)
    pi = _zipf(K, beta, rng)

    # A9: two-tier graph. core = top-m by pi (a cycle + a few extra core edges); leaves attach to core.
    order = np.argsort(-pi)
    core = list(order[:min(m_core, K)]); leaves = list(order[min(m_core, K):])
    succ = {k: set() for k in range(K)}
    for i, k in enumerate(core):                       # core cycle + one chord
        succ[k].add(core[(i + 1) % len(core)])
        succ[k].add(core[(i + 2) % len(core)])
    for lf in leaves:                                  # each leaf: entered from a core node, returns to core
        host = int(rng.choice(core)); succ[host].add(lf)
        succ[lf].add(int(rng.choice(core)))
    P = {}
    for k in range(K):
        sc = sorted(succ[k]) or [int(rng.choice(core))]
        w = np.ones(len(sc))
        for i, t in enumerate(sc):                     # heavy weight on core successors
            w[i] = 4.0 if t in core else 1.0
        P[k] = (np.array(sc), w / w.sum())
    forbidden = {k: set(range(K)) - succ[k] - {k} for k in range(K)}

    # density-scaled dwell: common modes long, rare short
    rank = np.empty(K, int); rank[order] = np.arange(K)
    frac = 1 - rank / max(K - 1, 1)                     # 1 for most common .. 0 for rarest
    Dmin = 15 + frac * 40; Dmax = 40 + frac * 180
    dwell = np.stack([Dmin, Dmax], 1)

    # A3/A4/A5 geometry
    mu = rng.normal(0, 3.0, (K, d))
    B = np.zeros((K, d, r))
    for k in range(K):
        Q, _ = np.linalg.qr(rng.normal(0, 1, (d, r)))
        lam = np.sort(rng.uniform(0.6, 1.6, r))[::-1]   # one dominant lever
        B[k] = Q * lam
    sigma = rng.uniform(0.25, 0.7, K)
    kappa = np.where(rng.random(K) < p_curv, rng.uniform(0.1, 0.35, K), 0.0)

    # A8 typed channels + sensing params
    ctype = np.array([TYPES["sensor"]] * 12 + [TYPES["actuator"]] * 4 +
                     [TYPES["state"]] * 3 + [TYPES["setpoint"]] * 2 + [TYPES["accumulator"]] * 3)
    ctype = ctype[:d] if len(ctype) >= d else np.pad(ctype, (0, d - len(ctype)))
    We = rng.normal(0, 1, (d, q_noise)) / np.sqrt(q_noise)
    s = rng.uniform(0.4, 2.4, d)
    delta = np.where(ctype == TYPES["sensor"], 0.02, np.where(ctype == TYPES["actuator"], 0.8, 0.1))
    theta = np.zeros(d)                                 # thresholds set later at channel median
    eta = rng.uniform(0.5, 2.0, K)                      # A8 non-stationary noise multiplier per mode

    # A10 accumulator
    flag_hist = rng.random(K) < 0.5
    acc_load = rng.uniform(0.01, 0.1, K)
    acc_chan = int(np.where(ctype == TYPES["accumulator"])[0][0]) if (ctype == TYPES["accumulator"]).any() else -1

    return System(K, d, r, pi, succ, P, forbidden, dwell, mu, B, sigma, kappa,
                  ctype, We, s, delta, theta, eta, flag_hist, acc_load, acc_chan)


def g_k(sys, k, U):
    """Vectorised mode signature for U:(n,r) -> (n,d)."""
    base = sys.mu[k] + U @ sys.B[k].T
    if sys.kappa[k] > 0:
        base = base + sys.kappa[k] * np.tanh(1.3 * base[:, ::-1])
    return base


# ----------------------------- Phase B: walk + render + inject -----------------------------

def _ou_segment(sys, k, D, rng, drift=None):
    phi = 0.85
    U = np.empty((D, sys.r)); u = rng.normal(0, sys.sigma[k], sys.r)
    for t in range(D):
        u = phi * u + np.sqrt(1 - phi ** 2) * sys.sigma[k] * rng.normal(0, 1, sys.r)
        if drift is not None:
            u = u + drift * (t / max(1, D))            # un-clipped ramp (drift anomaly)
        U[t] = u if drift is not None else np.clip(u, -2 * sys.sigma[k], 2 * sys.sigma[k])
    return U


def _fringe_segment(sys, k, D, rng):
    dirs = rng.normal(0, 1, (D, sys.r)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rad = rng.uniform(1.7, 1.98, (D, 1)) * sys.sigma[k]
    return dirs * rad


def _noise(sys, k, D, rng):
    z1 = rng.normal(0, 1, (D, sys.We.shape[1])); z2 = rng.normal(0, 1, (D, sys.d))
    return sys.eta[k] * (z1 @ sys.We.T + z2 * sys.s)


def generate_stream(sys, T, inject, rng, quotas=None):
    """Returns X:(T,d) continuous pre-render, plus per-step mode/anom/atype and segment list."""
    segs = []                       # (mode, start, len, atype)
    X = []; M = []; A = []; TY = []
    visit = np.zeros(sys.K, int)
    cur = int(np.argmax(sys.pi))
    pending_bad = 0
    t = 0
    def emit(k, U, atype, anom):
        base = g_k(sys, k, U) + _noise(sys, k, len(U), rng)
        X.append(base); M.extend([k] * len(U)); A.extend([anom] * len(U)); TY.extend([atype] * len(U))
        segs.append((k, sum(len(x) for x in X[:-1]), len(U), atype))
    while t < T:
        D = int(rng.uniform(*sys.dwell[cur]))
        atype, anom, drift = "none", 0, None
        # ---- test-only anomalies ----
        if inject and rng.random() < 0.05:
            drift = rng.normal(0, 1, sys.r) * sys.sigma[cur] * 3.0; atype, anom = "drift", 1
        # fringe (normal) enrichment
        if not inject and rng.random() < 0.06:
            U = _fringe_segment(sys, cur, D, rng); emit(cur, U, "none", 0)
        else:
            U = _ou_segment(sys, cur, D, rng, drift); emit(cur, U, atype, anom)
        # settled bad_transition labelling (history anomaly)
        if pending_bad and inject:
            n = len(X[-1]); lab = A[-n:]; ty = TY[-n:]
            W = 40
            for i in range(W, min(n, W + 40)):
                A[-n + i] = 1; TY[-n + i] = "bad_transition"
            pending_bad = 0
        visit[cur] += 1
        t += D
        # ---- injected snapshot anomalies (test) ----
        if inject and rng.random() < 0.05:            # pocket
            a, b = rng.choice(sys.K, 2, replace=False)
            ca = g_k(sys, a, np.zeros((1, sys.r)))[0]; cb = g_k(sys, b, np.zeros((1, sys.r)))[0]
            tt = rng.uniform(0.35, 0.65); mid = (1 - tt) * ca + tt * cb
            pk = mid + rng.normal(0, 0.05, (40, sys.d))
            X.append(pk); M.extend([-1] * 40); A.extend([1] * 40); TY.extend(["pocket"] * 40); t += 40
        if inject and rng.random() < 0.04:            # OOD
            span = 6.0; x0 = rng.uniform(-1, 1, (40, sys.d)) * span
            X.append(sys.mu[cur] + x0 * 1.8); M.extend([-1] * 40); A.extend([1] * 40); TY.extend(["ood"] * 40); t += 40
        if inject and rng.random() < 0.04:            # wrong-for-regime (contextual): values from another mode
            other = int(rng.choice(sys.K)); U = _ou_segment(sys, cur, 40, rng)
            X.append(g_k(sys, other, U) + _noise(sys, cur, 40, rng))
            M.extend([cur] * 40); A.extend([1] * 40); TY.extend(["wrong_for_regime"] * 40); t += 40
        # ---- next mode ----
        if inject and rng.random() < 0.05 and sys.forbidden[cur]:   # forbidden jump
            cur = int(rng.choice(sorted(sys.forbidden[cur]))); pending_bad = 1
        else:
            sc, w = sys.P[cur]; cur = int(rng.choice(sc, p=w))
    X = np.concatenate(X)[:T]; M = np.array(M[:T]); A = np.array(A[:T]); TY = np.array(TY[:T], object)
    # coverage: ensure rare modes hit a quota (append forced normal visits)
    if quotas and not inject:
        need = [k for k in range(sys.K) if visit[k] < quotas]
        for k in need[:40]:
            U = _ou_segment(sys, k, 60, rng)
            X = np.vstack([X, g_k(sys, k, U) + _noise(sys, k, 60, rng)])
            M = np.concatenate([M, [k] * 60]); A = np.concatenate([A, [0] * 60])
            TY = np.concatenate([TY, np.array(["none"] * 60, object)])
    return X, M, A, TY


def _render(sys, X, M):
    """Apply per-channel type rendering + smoothing + quantization to continuous X:(T,d)."""
    T = len(X); Xr = X.copy()
    # per-channel smoothing (sensor inertia) on sensor/actuator channels
    alpha = np.where(np.isin(sys.ctype, [TYPES["sensor"], TYPES["actuator"]]),
                     np.linspace(0.1, 0.9, sys.d), 1.0)
    sm = Xr[0].copy()
    for t in range(T):
        sm = alpha * Xr[t] + (1 - alpha) * sm; Xr[t] = sm
    # state thresholds at channel median
    med = np.median(Xr, 0)
    for c in range(sys.d):
        ct = sys.ctype[c]
        if ct == TYPES["state"]:
            Xr[:, c] = (Xr[:, c] > med[c]).astype(float)
        elif ct == TYPES["setpoint"]:                 # hold value at each mode-entry
            entry = np.r_[0, np.where(np.diff(M) != 0)[0] + 1]
            for i, e in enumerate(entry):
                nxt = entry[i + 1] if i + 1 < len(entry) else T
                Xr[e:nxt, c] = Xr[e, c]
    # accumulator channel: integrator over mode load
    if sys.acc_chan >= 0:
        a = 1.0; col = np.empty(T)
        for t in range(T):
            k = M[t]
            load = sys.acc_load[k] if (k >= 0 and sys.flag_hist[k]) else 0.005
            a = np.clip(a - load / 20 + (0.05 if (k >= 0 and k % 7 == 0) else 0), 0, 1)
            col[t] = a
        Xr[:, sys.acc_chan] = col
    # quantize per channel
    scale = Xr.std(0) + 1e-8
    step = sys.delta * scale
    Xr = np.round(Xr / step) * step
    return Xr


def make_dataset(seed=0, n_train=60000, n_test=40000, W=40, stride=20):
    from winfeat import window_features
    sys = sample_system(seed=seed)
    rng = np.random.default_rng(seed + 1)
    Xtr, Mtr, Atr, TYtr = generate_stream(sys, n_train, inject=False, rng=rng, quotas=150)
    Xte, Mte, Ate, TYte = generate_stream(sys, n_test, inject=True, rng=rng)
    sys.centroid = np.stack([g_k(sys, k, np.zeros((1, sys.r)))[0] for k in range(sys.K)])
    Xtr = _render(sys, Xtr, Mtr); Xte = _render(sys, Xte, Mte)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

    def win(Z, M, A, TY):
        Xw, y, at, md = [], [], [], []
        for i in range(0, len(Z) - W + 1, stride):
            Xw.append(window_features(Z[i:i + W], "stats"))
            aw = A[i:i + W]; y.append(int(aw.mean() > 0.5))
            tw = TY[i:i + W][aw == 1]; at.append(tw[0] if len(tw) else "none")
            mm = M[i:i + W][M[i:i + W] >= 0]
            md.append(int(np.bincount(mm).argmax()) if len(mm) else -1)
        return (np.asarray(Xw, np.float32), np.asarray(y, int),
                np.asarray(at, object), np.asarray(md, int))

    Xw_tr, y_tr, _, mode_tr = win(Ztr, Mtr, Atr, TYtr)
    Xw_te, y_te, at_te, mode_te = win(Zte, Mte, Ate, TYte)
    return dict(x_train=Xw_tr, y_train=y_tr, mode_train=mode_tr,
                x_test=Xw_te, y_test=y_te, atype_test=at_te, mode_test=mode_te,
                n_features=Xw_tr.shape[1], K=sys.K, system=sys)


if __name__ == "__main__":
    d = make_dataset(seed=0)
    print(f"MIIM unified: K={d['K']}  train={len(d['x_train'])} test={len(d['x_test'])} "
          f"winfeat={d['n_features']}  anomaly-frac={d['y_test'].mean():.3f}")
    print(f"distinct modes seen (train): {len(np.unique(d['mode_train'][d['mode_train']>=0]))}/{d['K']}")
    at = d['atype_test']
    for t in ["none", "pocket", "drift", "ood", "wrong_for_regime", "bad_transition"]:
        print(f"  {t:<16} {int((at == t).sum())} windows")
    import os
    os.makedirs("datasets/miim", exist_ok=True)
    np.savez_compressed("datasets/miim/miim_unified_seed0.npz",
                        x_train=d["x_train"], y_train=d["y_train"], mode_train=d["mode_train"],
                        x_test=d["x_test"], y_test=d["y_test"],
                        atype_test=d["atype_test"].astype(str), mode_test=d["mode_test"])
    print("saved -> datasets/miim/miim_unified_seed0.npz")
