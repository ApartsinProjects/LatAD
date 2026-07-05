"""
CPS normal-behaviour data with the MIIM property
(Massive, Implicit, Imbalanced Multimodality), plus two anomaly families that
correspond to the two dangerous detection errors named in the proposal.

  - Normal data  : a mixture of K operating modes on a nonlinear manifold,
                   with power-law (Zipf) imbalanced mode weights. The rarest
                   modes carry only a handful of samples. Mode labels are kept
                   for evaluation only; no detector ever sees them (implicit).
  - OOD anomaly  : a genuine fault whose signature lies well outside every mode.
  - Pocket anomaly: a fault whose signature falls in the empty region *between*
                   two well-sampled modes, where a single smooth model
                   over-interpolates a bridge of "normal" (the false negative).

A real dataset (WADI) plugs in through load_wadi(); the experiment code only
needs the CPSDataset container this module returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CPSDataset:
    """A CPS anomaly-detection split. All arrays are float32 / int.

    x_train : normal-only training data (no anomalies, no labels used).
    x_test, y_test : test points; y_test == 1 marks an anomaly.
    mode_train, mode_test : ground-truth operating-mode id per point
        (evaluation only, -1 for anomalies). Never fed to a detector.
    atype_test : anomaly family per test point ('none' | 'ood' | 'pocket').
    """

    x_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    mode_train: np.ndarray
    mode_test: np.ndarray
    atype_test: np.ndarray
    name: str = "dataset"

    @property
    def n_features(self) -> int:
        return self.x_train.shape[1]


def _zipf_weights(k: int, exponent: float, rng: np.random.Generator) -> np.ndarray:
    """Power-law mode weights: a few common regimes, a long tail of rare ones."""
    ranks = np.arange(1, k + 1)
    w = 1.0 / np.power(ranks, exponent)
    rng.shuffle(w)  # decouple mode index from frequency rank
    return w / w.sum()


def _mode_manifold(latent: np.ndarray, mode_id: int, n_features: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Map a low-dim regime coordinate to observed signal space through a
    smooth, mode-specific nonlinear embedding. Mimics 'valid states must stay
    consistent with physics/control': each regime is a curved low-dim manifold
    of correlated channels, not an arbitrary Gaussian blob."""
    d_lat = latent.shape[1]
    # Mode-specific random linear map + offset (deterministic given mode_id).
    m_rng = np.random.default_rng(1000 + mode_id)
    A = m_rng.normal(0, 1, size=(d_lat, n_features)) / np.sqrt(d_lat)
    b = m_rng.normal(0, 1, size=(n_features,)) * 3.0
    x = latent @ A + b
    # Smooth nonlinearity couples channels (engineered control response).
    x = x + 0.35 * np.tanh(x[:, ::-1] * 1.3)
    return x


def make_miim(
    n_features: int = 24,
    n_modes: int = 40,
    latent_dim: int = 3,
    n_train: int = 24000,
    n_test_normal: int = 6000,
    n_ood: int = 800,
    n_pocket: int = 800,
    zipf_exponent: float = 2.2,
    rare_quantile: float = 0.40,
    noise: float = 0.15,
    noise_mode: str = "correlated",
    bounded: bool = True,
    quantize: bool = True,
    seed: int = 0,
) -> tuple[CPSDataset, dict]:
    """Generate the controlled MIIM benchmark.

    Returns the dataset and a meta dict (mode weights, which modes are 'rare',
    the mode centroids) used by the evaluation to score the rare-mode FPR and
    the pocket false-negative rate.
    """
    rng = np.random.default_rng(seed)
    weights = _zipf_weights(n_modes, zipf_exponent, rng)

    # Per-mode intrinsic spread in regime (latent) space.
    mode_scale = rng.uniform(0.25, 0.7, size=n_modes)

    # Observation-noise model. 'correlated' gives noise a fixed non-isotropic
    # covariance Sigma = W W^T + diag(s^2): cross-channel correlation (low-rank W)
    # plus heteroscedastic per-channel scale s. The decoder cannot reconstruct
    # noise, so residuals inherit Sigma - which is exactly what a whitened
    # (Mahalanobis) score exploits and a plain sum-of-squares cannot. 'isotropic'
    # reproduces the earlier white noise (whitening is then provably neutral).
    n_nf = 6
    Wn = rng.normal(0, 1, size=(n_features, n_nf)) * 0.55
    chan_scale = rng.uniform(0.4, 2.4, size=n_features)
    # Heterogeneous, a-priori-unknown per-channel resolution: sensors and
    # actuators differ in precision/quantization (fine ADC vs coarse vs near-binary
    # actuator states). q_frac is the quantisation step as a fraction of the
    # channel scale - most channels fine, a few very coarse.
    q_frac = rng.choice([0.02, 0.05, 0.10, 0.30, 0.80], size=n_features,
                        p=[0.40, 0.25, 0.20, 0.10, 0.05])

    def add_noise(X, r):
        if noise_mode == "isotropic":
            return X + r.normal(0, noise, X.shape)
        common = r.normal(0, 1, size=(len(X), n_nf)) @ Wn.T      # correlated part
        indep = r.normal(0, 1, size=X.shape) * chan_scale        # heteroscedastic
        return X + noise * (common + indep)

    def sample_mode(mode_id: int, n: int) -> np.ndarray:
        if n == 0:
            return np.empty((0, n_features), np.float64)
        s = mode_scale[mode_id]
        lat = rng.normal(0, s, size=(n, latent_dim))
        if bounded:
            # Physical/control limits bound a mode's extent: truncate the regime
            # coordinate to +/-2 sigma (a clipped Gaussian), giving each mode a
            # bounded manifold patch with hard edges rather than infinite tails.
            lat = np.clip(lat, -2.0 * s, 2.0 * s)
        return _mode_manifold(lat, mode_id, n_features, rng)

    # ---- normal training data: implicit, imbalanced multimodality ----
    train_counts = rng.multinomial(n_train, weights)
    x_tr, mode_tr = [], []
    for m in range(n_modes):
        xm = sample_mode(m, int(train_counts[m]))
        x_tr.append(xm)
        mode_tr.append(np.full(len(xm), m))
    x_train = np.concatenate(x_tr)
    mode_train = np.concatenate(mode_tr)

    # "rare" modes = smallest-weight tail (targets of the false-positive test).
    rare_cut = np.quantile(weights, rare_quantile)
    rare_modes = set(np.where(weights <= rare_cut)[0].tolist())

    # ---- normal test data: keep rare modes represented on purpose ----
    # Half drawn by the true (imbalanced) prior, half forced from rare modes so
    # the rare-mode FPR is measurable with enough samples.
    n_by_prior = n_test_normal // 2
    n_rare_forced = n_test_normal - n_by_prior
    test_counts = rng.multinomial(n_by_prior, weights)
    x_te, mode_te = [], []
    for m in range(n_modes):
        xm = sample_mode(m, int(test_counts[m]))
        x_te.append(xm); mode_te.append(np.full(len(xm), m))
    rare_list = sorted(rare_modes) or list(range(n_modes))
    forced = rng.choice(rare_list, size=n_rare_forced)
    for m in forced:
        xm = sample_mode(int(m), 1)
        x_te.append(xm); mode_te.append(np.array([m]))
    x_test_normal = np.concatenate(x_te)
    mode_test_normal = np.concatenate(mode_te)

    # centroids in signal space (for pocket construction + eval).
    centroids = np.stack([sample_mode(m, 400).mean(0) for m in range(n_modes)])

    # ---- OOD anomalies: far outside every mode ----
    span = x_train.std(0) * 6.0
    x_ood = rng.uniform(-1, 1, size=(n_ood, n_features)) * span + x_train.mean(0)
    # push each OOD point away from its nearest centroid to guarantee it is out
    d = ((x_ood[:, None, :] - centroids[None]) ** 2).sum(-1)
    nearest = centroids[d.argmin(1)]
    x_ood = nearest + (x_ood - nearest) * 1.8

    # ---- pocket anomalies: midpoints between two well-sampled modes ----
    common_modes = [m for m in range(n_modes) if m not in rare_modes]
    if len(common_modes) < 2:
        common_modes = list(range(n_modes))
    pockets = []
    for _ in range(n_pocket):
        a, b = rng.choice(common_modes, size=2, replace=False)
        t = rng.uniform(0.35, 0.65)
        mid = centroids[a] * (1 - t) + centroids[b] * t
        pockets.append(mid)
    x_pocket = np.stack(pockets)
    x_pocket += rng.normal(0, x_train.std(0) * 0.05, size=x_pocket.shape)

    # ---- assemble, add observation noise, quantise, standardise on train ----
    x_train = add_noise(x_train, rng)
    x_test = np.concatenate([x_test_normal, x_ood, x_pocket])
    x_test = add_noise(x_test, rng)

    # heterogeneous per-channel quantisation (fixed step, physical frame)
    q_step = (q_frac * (x_train.std(0) + 1e-8)) if quantize else None

    def _quant(X):
        return np.round(X / q_step) * q_step if quantize else X

    x_train = _quant(x_train)
    x_test = _quant(x_test)
    y_test = np.concatenate([
        np.zeros(len(x_test_normal)), np.ones(len(x_ood)), np.ones(len(x_pocket))])
    atype_test = np.array(
        ["none"] * len(x_test_normal) + ["ood"] * len(x_ood) + ["pocket"] * len(x_pocket))
    mode_test = np.concatenate([
        mode_test_normal, np.full(len(x_ood), -1), np.full(len(x_pocket), -1)])

    mu, sd = x_train.mean(0), x_train.std(0) + 1e-8
    # Large fresh true-normal reference for the Component-1 generator oracle,
    # drawn from the SAME modes and standardised in the SAME frame as the data,
    # so kNN-to-normal distances are meaningful (a different seed would give
    # different manifolds and a different frame - not a valid reference).
    oracle_counts = rng.multinomial(8000, weights)
    x_or = np.concatenate([sample_mode(m, int(oracle_counts[m])) for m in range(n_modes)])
    x_or = _quant(add_noise(x_or, rng))
    oracle_normal = ((x_or - mu) / sd).astype(np.float32)

    x_train = ((x_train - mu) / sd).astype(np.float32)
    x_test = ((x_test - mu) / sd).astype(np.float32)

    ds = CPSDataset(
        x_train=x_train, x_test=x_test, y_test=y_test.astype(int),
        mode_train=mode_train.astype(int), mode_test=mode_test.astype(int),
        atype_test=atype_test, name="MIIM-synthetic")
    meta = {
        "weights": weights, "rare_modes": rare_modes, "n_modes": n_modes,
        "train_counts": train_counts, "oracle_normal": oracle_normal,
    }
    return ds, meta


def load_wadi(root: str) -> CPSDataset:
    """Drop-in loader for the real WADI dataset once iTrust access is granted.

    Expects the standard WADI CSVs under `root`:
        WADI_14days*.csv          (normal operation)
        WADI_attackdata*.csv      (attack period, with an attack-label column)
    Kept as a stub with the exact wiring so switching from synthetic to WADI is
    a one-line change in run.py. Not called until the files are present.
    """
    raise NotImplementedError(
        "WADI files not present yet. Once iTrust grants access, place the CSVs "
        f"under {root!r} and implement the parse here (drop first ~2 rows of "
        "sensor warm-up, forward-fill actuator states, standardise on the "
        "14-day normal split, map the attack column to y_test).")
