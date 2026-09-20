"""Test 1a (R1-2 / E2): ORACLE regime-imbalance sweep for the nearest-component head.

Ideal case for the mechanism the manuscript asserts: K isolated diagonal-Gaussian regimes in a
d-dim latent, components FIXED to the true regimes, pi = empirical train share. One regime is
made rare (train share f); all others share the rest equally. Scores on held-out VALID windows
of the rare regime:
  mixture  = -logsumexp_c(log pi_c + log N_c)   (pi-weighted, the density-head form)
  nearest  = -max_c log N_c                       (Eq. 5, pi-free)
  tempered = -logsumexp_c(0.5 log pi_c + log N_c) (imbalance-aware, pi^0.5)
  balanced = -logsumexp_c(log N_c) + log K        (imbalance-aware, uniform pi)
Threshold = train 99th percentile of each score (the operational rule). Reports the rare-regime
FPR vs imbalance ratio, plus the closed form for the isolated mixture:
  rare point flagged  iff  chi2_d > q - 2*log(pi_common/pi_rare), with q the train-p99 chi2 level.
Invariants (stated in advance): (I1) nearest/balanced FPR_rare == 0.01 +- MC noise at every f
(pi-free); (I2) mixture FPR_rare is non-decreasing in the imbalance ratio and equals 0.01 at
f = 1/K; (I3) Monte-Carlo mixture FPR_rare matches the closed form within MC noise.
Writes nc_oracle_sweep.json.
"""
from __future__ import annotations
import os, json, numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__))


def closed_form(d, f, K):
    """Isolated regimes, unit variance: mixture NLL_c = -log pi_c + const + chi2_d/2.
    Train-p99 threshold level q (in chi2 units, referenced to a common regime) solves
    (1-f) P(chi2_d > q) + f P(chi2_d > q - 2 log(pi_common/pi_rare)) = 0.01."""
    pc, pr = (1 - f) / (K - 1), f
    lr = np.log(pc / pr)
    g = lambda q: (1 - f) * chi2.sf(q, d) + f * chi2.sf(q - 2 * lr, d) - 0.01
    q = brentq(g, 1e-6, 1e4)
    return float(chi2.sf(q - 2 * lr, d)), float(lr)


def run(d, f, K=10, n_train=300_000, n_val=50_000, seed=0, sep=40.0):
    rng = np.random.default_rng(seed)
    pi = np.full(K, (1 - f) / (K - 1)); pi[0] = f
    mu = rng.normal(0, sep, (K, d))                       # isolated regimes
    counts = rng.multinomial(n_train, pi)
    ztr = np.concatenate([mu[c] + rng.normal(0, 1, (counts[c], d)) for c in range(K)])
    z_rare = mu[0] + rng.normal(0, 1, (n_val, d)); z_common = mu[1] + rng.normal(0, 1, (n_val, d))
    logpi = np.log(np.maximum(counts, 1) / counts.sum())

    def L(z):
        return -0.5 * (d * np.log(2 * np.pi) + ((z[:, None, :] - mu[None]) ** 2).sum(2))

    def scores(z):
        l = L(z)
        return dict(mixture=-logsumexp(logpi[None] + l, 1), nearest=-l.max(1),
                    tempered=-logsumexp(0.5 * logpi[None] + l, 1), balanced=-logsumexp(l, 1) + np.log(K))

    s_tr = {k: np.concatenate([scores(ztr[i:i + 50_000])[k] for i in range(0, len(ztr), 50_000)]) for k in ("mixture", "nearest", "tempered", "balanced")}
    s_r, s_c = scores(z_rare), scores(z_common)
    cf, lr = closed_form(d, f, K)
    out = dict(d=d, f=f, K=K, seed=seed, n_rare_train=int(counts[0]), imbalance_ratio=float(pi[1] / pi[0]),
               log_ratio_nats=lr, closed_form_fpr_rare_mixture=cf)
    for k in s_tr:
        thr = np.quantile(s_tr[k], .99)
        out[k] = dict(fpr_rare=float((s_r[k] > thr).mean()), fpr_common=float((s_c[k] > thr).mean()))
    return out


if __name__ == "__main__":
    rows = []
    for d in (4, 10, 16):
        for f in (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0002):
            r = run(d, f); rows.append(r)
            print(f"d={d:2d} f={f:<7} ratio={r['imbalance_ratio']:7.1f} logratio={r['log_ratio_nats']:.2f} | FPR_rare mix={r['mixture']['fpr_rare']:.4f} "
                  f"(closed {r['closed_form_fpr_rare_mixture']:.4f}) near={r['nearest']['fpr_rare']:.4f} temp={r['tempered']['fpr_rare']:.4f} "
                  f"bal={r['balanced']['fpr_rare']:.4f} | FPR_common mix={r['mixture']['fpr_common']:.4f} near={r['nearest']['fpr_common']:.4f}", flush=True)
    json.dump(rows, open(os.path.join(HERE, "nc_oracle_sweep.json"), "w"), indent=1)
    # invariants
    I1 = all(abs(r["nearest"]["fpr_rare"] - 0.01) < 0.003 and abs(r["balanced"]["fpr_rare"] - 0.01) < 0.003 for r in rows)
    I3 = all(abs(r["mixture"]["fpr_rare"] - r["closed_form_fpr_rare_mixture"]) < 0.01 for r in rows)
    I2 = all(np.all(np.diff([r["mixture"]["fpr_rare"] for r in rows if r["d"] == d]) >= -0.003) for d in (4, 10, 16))
    print(f"\nInvariants: I1 nearest/balanced flat at 1% -> {I1}; I2 mixture monotone in imbalance -> {I2}; I3 MC == closed form -> {I3}")
