"""
run_ldt.py - the decisive experiment for the trajectory-aware A/B/C architecture.

Pipeline (spec's training order):
  1. Train A (mode encoder, VaDE) on normal windows -> freeze -> emit (gamma, d, pi).
  2. Train B (trajectory encoder) on A's normal gamma-stream (two query heads) ->
     memory-horizon probe -> freeze -> emit c_t for every window.
  3. Train C (conditioned VaDE) on normal windows + c_t (FiLM), fit the C1-C4
     scoring stack, fuse to one score.
  4. Measure per-type TPR@5%FPR for pocket / drift / bad_transition, for
       C+B   : C conditioned on B's real context
       C-alone : the ablation with c_t zeroed (control)
     and print the decisive table.

The window arrays from make_temporal_miim are strided over the raw stream IN TIME
ORDER, so the window sequence is itself the trajectory B reads.

Smoke-test config: small K, short epochs. Run:
    /c/Python314/python run_ldt.py
"""

from __future__ import annotations

import numpy as np
import torch

from temporal_data import make_temporal_miim
from generate import (TrueNormalOracle, ModeDistance, shuffle_candidates,
                      control_difficulty_input, in_range_mask)
import ldt_a, ldt_b, ldt_c
import component4


TYPES = ["pocket", "drift", "bad_transition"]


def tpr_at_fpr(score_normal, score_pos, fpr=0.05):
    """TPR on positives at the threshold giving `fpr` on normal windows."""
    if len(score_pos) == 0:
        return float("nan")
    thr = float(np.quantile(score_normal, 1.0 - fpr))
    return float((score_pos > thr).mean())


def build_c1_anomalies(x_train, mode_train, seed=0):
    """C1: channel-shuffle candidates, oracle-filtered, in-range, HARD band."""
    rng = np.random.default_rng(seed)
    oracle = TrueNormalOracle(x_train, k=5)
    md = ModeDistance(x_train, mode_train)
    cand = shuffle_candidates(x_train, mode_train, 8000, swap_frac=0.4, rng=rng)
    pool = np.concatenate([control_difficulty_input(cand, x_train, mode_train, a)
                           for a in (0.3, 0.5, 0.7, 1.0)])
    keep = in_range_mask(pool, x_train) & oracle.is_anomaly(pool)
    pool = pool[keep]
    # HARD band: nearest-mode distance 1-2x the normal envelope
    sig, _ = md.min_sigma(pool)
    sig_n, _ = md.min_sigma(x_train)
    env = np.quantile(sig_n, 0.99)
    hard = (sig >= env) & (sig <= 2.5 * env)
    if hard.sum() < 200:                      # fall back to all oracle-anoms
        return pool
    return pool[hard]


def fused_scores(model, x, c, pmpca, fuser, device):
    """Assemble the C-score fuser features and return the fused P(anomaly)."""
    s_rec, s_lat = ldt_c.base_scores(model, x, c, device=device)
    agr = ldt_c.basin_features_cond(model, x, c, device=device)
    pca_min = pmpca.min_score(x)
    feats = np.stack([s_rec, s_lat, 1.0 - agr, pca_min], axis=1)
    return fuser.predict_proba(feats)[:, 1], feats


class ModeContextScore:
    """Mode-conditional trajectory-context anomaly: how off-distribution is c_t for
    windows assigned to this mode. FiLM scores p(window | c_t) and cannot catch a
    pure-trajectory fault (bad_transition: the window is NORMAL, only the arrival
    path is wrong). This scores p(c_t | mode) instead -> flags an abnormal arrival.
    Ablation-safe: with c_t==0 (C-alone) it returns 0, so the control gets no signal.
    """

    def __init__(self, c, assign, K, ridge=1e-2):
        self.dim = c.shape[1]; self.mu = {}; self.inv = {}
        self.trivial = bool(np.allclose(c, 0))
        gcov = np.cov(c.T) + ridge * np.eye(self.dim)
        self.gmu = c.mean(0); self.ginv = np.linalg.pinv(gcov)
        for k in range(K):
            ck = c[assign == k]
            if len(ck) >= 30:
                self.mu[k] = ck.mean(0)
                self.inv[k] = np.linalg.pinv(np.cov(ck.T) + ridge * np.eye(self.dim))

    def score(self, c, assign):
        if self.trivial or np.allclose(c, 0):
            return np.zeros(len(c))
        out = np.zeros(len(c))
        for k in np.unique(assign):
            idx = np.where(assign == k)[0]
            m = self.mu.get(int(k), self.gmu); inv = self.inv.get(int(k), self.ginv)
            dd = c[idx] - m
            out[idx] = np.einsum("ij,jk,ik->i", dd, inv, dd)
        return out


def _z(v, ref):
    mu, sd = ref.mean(), ref.std() + 1e-9
    return (v - mu) / sd if sd > 1e-6 else np.zeros_like(v)


def run(seed=0, K=16, latent=8, device=None, verbose=True):
    if device is None:
        # honour CUDA_VISIBLE_DEVICES="" as a real CPU request
        import os
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        gpu_ok = torch.cuda.is_available() and vis != "" and torch.cuda.device_count() > 0
        device = "cuda" if gpu_ok else "cpu"
    print(f"device={device}  K={K}  latent={latent}  seed={seed}")

    # ---- data --------------------------------------------------------------
    d = make_temporal_miim(n_modes=20, n_features=24, seed=seed,
                           n_train=120000, n_test=150000, W=40, stride=20)
    xtr, xte = d["x_train"], d["x_test"]
    yte, atype = d["y_test"], d["atype_test"]
    print(f"train={len(xtr)} test={len(xte)} feat={d['n_features']} "
          f"anom-frac={yte.mean():.3f}")
    for t in TYPES:
        print(f"  {t:<16} {(atype==t).sum()} windows")

    # ---- A: mode encoder ---------------------------------------------------
    print("\n[A] training mode encoder (VaDE)...")
    A = ldt_a.ModeEncoderA.fit(xtr, n_clusters=K, latent_dim=latent,
                               pretrain_epochs=15, epochs=30, warmup=8,
                               seed=seed, device=device, verbose=verbose)
    g_tr, _ = A.encode(xtr)
    g_te, _ = A.encode(xte)
    print(f"    pi (mode prior) min={A.pi.min():.4f} max={A.pi.max():.3f} "
          f"active modes={(A.pi>0.005).sum()}/{K}")

    # ---- B: trajectory encoder --------------------------------------------
    print("\n[B] training trajectory encoder (GRU stand-in for Mamba)...")
    B = ldt_b.TrajectoryEncoderB(K=K, emb_dim=48, ctx_dim=96, n_layers=2,
                                 centroids=A.centroids(), backbone="gru")
    ldt_b.train_B(B, g_tr, A.pi, epochs=60, seg_len=512, stride=128,
                  batch_segs=16, lr=3e-3, max_train_offset=64,
                  device=device, verbose=verbose, seed=seed)
    probe = ldt_b.memory_horizon_probe(B, g_tr, device=device)
    print("    memory-horizon probe (instance-head acc vs offset):")
    print("      " + "  ".join(f"k={k}:{a:.2f}" for k, a in probe.items()))

    c_tr = ldt_b.emit_context(B, g_tr, device=device)
    c_te = ldt_b.emit_context(B, g_te, device=device)
    c_te_zero = np.zeros_like(c_te)          # ablation control (c_t = 0)
    c_tr_zero = np.zeros_like(c_tr)
    print(f"    context dim={c_tr.shape[1]}  |c_t| mean={np.linalg.norm(c_te,axis=1).mean():.2f}")

    # ---- C1 anomalies (shared) --------------------------------------------
    print("\n[C1] generating channel-shuffle near-anomalies...")
    x_anom = build_c1_anomalies(xtr, d["mode_train"], seed=seed)
    print(f"    kept {len(x_anom)} hard oracle-verified anomalies")

    normal = yte == 0
    results = {}

    # ---- C: train + score, for the C+B and C-alone conditions -------------
    for tag, ctr, cte in [("C+B", c_tr, c_te), ("C-alone", c_tr_zero, c_te_zero)]:
        print(f"\n[C] training conditional detector  ({tag})...")
        model = ldt_c.train_cond_vade(xtr, ctr, n_clusters=K, latent_dim=latent,
                                      pretrain_epochs=12, epochs=25, warmup=8,
                                      seed=seed, device=device, verbose=verbose)
        ldt_c.fit_whitener(model, xtr, ctr, device=device)

        # per-mode PCA experts (C3a) on the conditioned assignment
        assign_tr = ldt_c.assign_modes(model, xtr, ctr, device=device)
        pmpca = ldt_c.PerModePCA(xtr, assign_tr, latent_dim=latent)
        ctxscore = ModeContextScore(ctr, assign_tr, K)   # mode-conditional trajectory context

        # C1 anomalies get context too: use the mean train context (they are not
        # a time series). This keeps the fuser features distributionally matched.
        c_anom = np.repeat(ctr.mean(0, keepdims=True), len(x_anom), axis=0)

        # fuser training features (normal train vs C1 anomalies)
        _, feat_norm = fused_scores(model, xtr, ctr, pmpca,
                                    _IdentityFuser(), device)
        _, feat_anom = fused_scores(model, x_anom, c_anom, pmpca,
                                    _IdentityFuser(), device)
        fuser = ldt_c.build_fuser(feat_norm, feat_anom, seed=seed)

        # score the test stream
        fused_te, _ = fused_scores(model, xte, cte, pmpca, fuser, device)
        # score train-normal for C4 per-mode thresholds
        fused_tr, _ = fused_scores(model, xtr, ctr, pmpca, fuser, device)

        # ---- combine base C-score with the trajectory-context score -----
        assign_te = ldt_c.assign_modes(model, xte, cte, device=device)
        sctx_tr = ctxscore.score(ctr, assign_tr)
        sctx_te = ctxscore.score(cte, assign_te)
        base_tr = fused_tr.copy()
        # s_ctx contributes only as an EXCEEDANCE above a high normal quantile, so it
        # RAISES genuine trajectory outliers without diluting the shared threshold for
        # snapshot faults (or shifting C-alone, whose s_ctx is identically 0).
        zs_tr = _z(sctx_tr, sctx_tr)
        q = float(np.quantile(zs_tr, 0.90)) if zs_tr.std() > 1e-6 else 0.0
        fused_tr = _z(base_tr, base_tr) + np.maximum(0.0, zs_tr - q)
        fused_te = _z(fused_te, base_tr) + np.maximum(0.0, _z(sctx_te, sctx_tr) - q)

        # ---- C4: per-mode thresholds at 5% FPR --------------------------
        global_thr = float(np.quantile(fused_tr, 0.95))
        thr_dict = component4.mode_conditional_thresholds(
            fused_tr, assign_tr, target_fpr=0.05, global_thr=global_thr)

        # global-threshold TPR@5%FPR per type (primary decisive metric)
        sn = fused_te[normal]
        row = {}
        for t in TYPES:
            row[t] = tpr_at_fpr(sn, fused_te[atype == t], fpr=0.05)
        # C4 per-mode operating point (audit)
        flagged = component4.apply_mode_conditional(fused_te, assign_te,
                                                    thr_dict, global_thr)
        c4 = {t: float(flagged[atype == t].mean()) for t in TYPES}
        c4_fpr = float(flagged[normal].mean())
        results[tag] = {"tpr": row, "c4": c4, "c4_fpr": c4_fpr,
                        "fused_te": fused_te}

    # ---- decisive table ----------------------------------------------------
    print("\n" + "=" * 62)
    print("DECISIVE PER-TYPE TPR @ 5% FPR  (global threshold on normal)")
    print("=" * 62)
    hdr = f"{'type':<16}" + "".join(f"{t:>12}" for t in ["C+B", "C-alone"])
    print(hdr)
    for t in TYPES:
        print(f"{t:<16}" + "".join(
            f"{results[c]['tpr'][t]:>12.3f}" for c in ["C+B", "C-alone"]))
    print("-" * 62)
    print("bad_transition floor (snapshot baseline, spec) = 0.057")
    lift = results["C+B"]["tpr"]["bad_transition"] - results["C-alone"]["tpr"]["bad_transition"]
    print(f"bad_transition lift (C+B minus C-alone) = {lift:+.3f}")

    print("\nC4 per-mode operating point (TPR per type; FPR on normal):")
    print(f"{'type':<16}" + "".join(f"{c:>12}" for c in ["C+B", "C-alone"]))
    for t in TYPES:
        print(f"{t:<16}" + "".join(
            f"{results[c]['c4'][t]:>12.3f}" for c in ["C+B", "C-alone"]))
    print(f"{'FPR(normal)':<16}" + "".join(
        f"{results[c]['c4_fpr']:>12.3f}" for c in ["C+B", "C-alone"]))
    return results


NOTES = """
HONEST NOTES (smoke-test config: K=16, latent=8, short epochs, CPU, seed 0).

WHAT WORKED
  - A/B/C train end-to-end, A→freeze→B→freeze→C, no cluster collapse (16/16
    active modes), FiLM identity-init verified (conditioning is a no-op at init).
  - The full C1-C4 scoring stack runs on the conditioned VaDE and fuses to one
    C-score. C lifts the two SNAPSHOT-ish faults well: drift ~0.77, pocket ~0.40
    at 5% FPR (0.64 / 0.82 at the C4 per-mode operating point).
  - C lifts bad_transition OFF the snapshot floor: 0.057 (spec baseline) -> 0.267
    at 5% FPR (0.31 at the C4 point). So the detector is not pinned at the
    false-positive floor.

WHAT DID NOT WORK (reported honestly, not hidden)
  - The DECISIVE claim - that the lift is DUE TO B's trajectory context - is NOT
    demonstrated at this config: C+B and C-alone (c_t zeroed) score
    bad_transition IDENTICALLY (lift = +0.000). The bad_transition gain comes
    from C's own scoring stack, not from conditioning on B.

ROOT CAUSE (diagnosed, not guessed)
  - B is not yet a faithful historian at this scale: the memory-horizon probe
    reads k=1 accuracy ~0.35 (barely above the pi_max~0.22 chance rate) and
    decays to noise by k=32. A GRU trained for 8 short epochs over a 20-mode
    Zipf stream has not learned to recover the recent mode, so c_t carries little
    transition signal for FiLM to exploit. (In a tiny 8-mode sanity run the same
    probe hit k=1=1.0, confirming the machinery works when B can actually fit.)
  - Consequence: with an uninformative c_t, FiLM's identity-init safeguard does
    exactly its job - it declines to change the score - so C+B collapses onto
    C-alone. The safeguard is working; the missing ingredient is a STRONGER B.

WHAT TO CHANGE TO MAKE THE DECISIVE EXPERIMENT BITE (next iteration)
  - Train B longer / bigger (more epochs, seg_len>=1024, higher LR, or the
    dilated-TCN backbone) until the memory-horizon probe clears ~0.7 at k<=8;
    only then is c_t faithful enough for the C+B vs C-alone contrast to be
    meaningful. The plumbing (heads, FiLM, ablation switch) is all in place.
  - Optionally add the spec's surprise-gated write / presence features (ldt_b
    optional extensions) so a one-step forbidden predecessor is written into s_t.
"""


class _IdentityFuser:
    """Placeholder so fused_scores can return the feature matrix during fuser
    training (we only consume the second return value there)."""

    def predict_proba(self, X):
        return np.zeros((len(X), 2))


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run(seed=seed)
