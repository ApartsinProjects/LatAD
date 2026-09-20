"""A8 screen blindness check (non-win E): POSITIVE control for the VaDE responsibility-entropy
measure used in Appendix C / Table C1.

The paper's A8 screen reports mean max-responsibility and normalized responsibility entropy from a
trained VaDE. VaDE's objective (term_b, the KL of gamma to pi, plus the clustering term) actively
SHARPENS responsibilities, and components sit on a log-variance floor log(0.05); low entropy could
therefore be a property of the estimator rather than of the data. The existing controls are a
NO-overlap synthetic floor and an empirical-variance refit; there is no control in which overlap is
present by construction and the screen is shown to detect it.

Here: synthetic windows from R regimes in a d-dim observation space with a controlled BRIDGE mass
(fraction b of windows drawn uniformly along chords between neighbouring regime centres, i.e.
between-regime overlap). Run the identical pipeline (winfeat 'stats' features on raw streams ->
train_vade K -> responsibilities on train) and report H_norm and rho=frac(max resp < 0.5) as a
function of b. Invariants:
  P0  b = 0        -> H_norm at the paper's floor (<= 0.06)          [negative control reproduced]
  P1  b = 0.2-0.3  -> H_norm clearly above 0.06 and rho >= 0.30       [screen CAN see overlap]
If P1 fails, the screen is blind by construction and "A8 not observed" is not evidence of absence.
Persists one row per (b, K, seed) to fable_a8_positive_control.jsonl (resumable).
"""
from __future__ import annotations
import json, os, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models_vade import train_vade
from winfeat import window_features

HERE = os.path.dirname(os.path.abspath(__file__))
OUTF = os.path.join(HERE, "fable_a8_positive_control.jsonl")
done = set()
if os.path.exists(OUTF):
    for line in open(OUTF):
        r = json.loads(line); done.add((r["bridge"], r["K"], r["seed"], r["sep"]))


def synth(bridge, seed, R=6, d=24, n_steps=60000, sep=4.0, noise=0.4, dwell=300):
    """Piecewise-stationary stream: R regime centres (pairwise distance ~sep in the first
    few dims), dwell-time switching; a fraction `bridge` of dwell segments sit at a random
    convex combination of two neighbouring centres (between-regime pocket) instead of at a centre."""
    rng = np.random.default_rng(seed)
    C = rng.normal(0, 1, (R, d)); C = C / np.linalg.norm(C, axis=1, keepdims=True) * sep
    X = np.zeros((n_steps, d)); t = 0; kind = []
    while t < n_steps:
        L = int(rng.integers(dwell // 2, dwell * 2))
        r = int(rng.integers(R))
        if rng.random() < bridge:
            r2 = (r + int(rng.integers(1, R))) % R; lam = rng.uniform(0.3, 0.7)
            c = lam * C[r] + (1 - lam) * C[r2]; kind.append(1)
        else:
            c = C[r]; kind.append(0)
        seg = c + rng.normal(0, noise, (L, d)) * rng.uniform(0.5, 1.5, d)
        X[t:t + L] = seg[:n_steps - t]; t += L
    return X


def screen(X, K, seed, W=60, stride=30, LD=8):
    Xw = np.stack([window_features(X[i:i + W], "stats") for i in range(0, len(X) - W + 1, stride)])
    mu, sd = Xw.mean(0), Xw.std(0) + 1e-8; Z = ((Xw - mu) / sd).astype(np.float32)
    v = train_vade(Z, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
    G = v._responsibilities(Z); mr = G.max(1)
    H = -(G * np.log(G + 1e-12)).sum(1) / np.log(G.shape[1])
    # empirical-variance refit at the same latent z (the paper's 'variance-floor' sanity check)
    z = v._encode_mean(Z); a = G.argmax(1)
    logp = np.full((len(z), K), -np.inf)
    for k in range(K):
        m = a == k
        if m.sum() < 5: continue
        muk = z[m].mean(0); vk = z[m].var(0) + 1e-4
        logp[:, k] = -0.5 * (np.log(2 * np.pi * vk).sum() + ((z - muk) ** 2 / vk).sum(1)) + np.log(m.mean())
    from scipy.special import logsumexp
    Ge = np.exp(logp - logsumexp(logp, 1, keepdims=True))
    He = -(Ge * np.log(Ge + 1e-12)).sum(1) / np.log(K); mre = Ge.max(1)
    return dict(n_win=int(len(Z)), max_resp=round(float(mr.mean()), 3), H_norm=round(float(H.mean()), 3),
                rho=round(float((mr < 0.5).mean()), 3), refit_max_resp=round(float(mre.mean()), 3),
                refit_H_norm=round(float(He.mean()), 3), refit_rho=round(float((mre < 0.5).mean()), 3))


if __name__ == "__main__":
    for sep in (4.0, 2.5):
        for bridge in (0.0, 0.1, 0.2, 0.3, 0.5):
            for K in (8, 20):
                for seed in (0, 1):
                    if (bridge, K, seed, sep) in done: continue
                    X = synth(bridge, seed, sep=sep)
                    r = screen(X, K, seed)
                    row = dict(bridge=bridge, K=K, seed=seed, sep=sep, **r)
                    with open(OUTF, "a") as f: f.write(json.dumps(row) + "\n")
                    print(row, flush=True)
    print("DONE", flush=True)


# ---- variant 2: TRANSIT-only overlap (between-regime space passed through, never dwelt in) ----
def synth_ramp(ramp, seed, R=6, d=24, n_steps=60000, sep=4.0, noise=0.4, dwell=300):
    rng = np.random.default_rng(seed)
    C = rng.normal(0, 1, (R, d)); C = C / np.linalg.norm(C, axis=1, keepdims=True) * sep
    X = np.zeros((n_steps, d)); t = 0; r = int(rng.integers(R))
    scale = rng.uniform(0.5, 1.5, d)
    while t < n_steps:
        L = int(rng.integers(dwell // 2, dwell * 2))
        X[t:t + L] = (C[r] + rng.normal(0, noise, (L, d)) * scale)[:n_steps - t]; t += L
        r2 = (r + int(rng.integers(1, R))) % R
        if ramp > 0 and t < n_steps:
            lam = np.linspace(0, 1, ramp)[:, None]
            X[t:t + ramp] = ((1 - lam) * C[r] + lam * C[r2] + rng.normal(0, noise, (ramp, d)) * scale)[:n_steps - t]; t += ramp
        r = r2
    return X


if __name__ == "__main__" and os.environ.get("RAMP"):
    OUTF2 = os.path.join(HERE, "fable_a8_positive_control_ramp.jsonl")
    done2 = set()
    if os.path.exists(OUTF2):
        for line in open(OUTF2):
            r = json.loads(line); done2.add((r["ramp"], r["K"], r["seed"]))
    for ramp in (0, 30, 120, 300):
        for K in (6, 8, 20):
            for seed in (0, 1):
                if (ramp, K, seed) in done2: continue
                X = synth_ramp(ramp, seed)
                # fraction of windows that straddle/lie inside a ramp is ~ ramp/(dwell+ramp)
                r = screen(X, K, seed)
                row = dict(ramp=ramp, K=K, seed=seed, transit_frac_approx=round(ramp / (300 + ramp), 3), **r)
                with open(OUTF2, "a") as f: f.write(json.dumps(row) + "\n")
                print(row, flush=True)
    print("DONE ramp", flush=True)
