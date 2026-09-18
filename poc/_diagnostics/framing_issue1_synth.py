"""Issue 1 (framing brainstorm): synthetic crossover, density vs reconstruction as a function of
how "reconstructable-but-improbable" the faults are, at several regime-overlap levels.

Generator. Normal data: K regimes in R^D whose centres and within-regime spread live in a d_m-dim
manifold U (the decoder's reachable span), plus small isotropic noise. `delta` scales centre separation
(the regime-overlap / multimodality knob). Faults start from a normal point of regime k and are displaced by
    v = m * ((1-p) * u_off + p * u_in),   |u_off| = |u_in| = 1,
where u_in points along U toward the midpoint between regime k and its nearest neighbour (in-envelope:
reachable by the decoder, but in the density gap between regimes), and u_off is orthogonal to U
(off-manifold: unreachable). p in [0,1] is the in-envelope-ness knob; p=1 is the purely
reconstructable-but-improbable fault, p=0 the purely unreconstructable one.

Detectors (train-normal only). recon: a small MLP autoencoder with latent d_m, score = reconstruction
error. density: a K-component GMM fitted in the AE latent, score = -log p(z). Both mirror the paper's
Table 6 heads at toy scale. Writes framing_issue1_synth.json (+ .svg figure). CPU, seconds per cell.
Invariants stated in advance: (i) p=0 -> recon AUROC ~1 and recon > density; (ii) p=1 -> density > recon
and recon AUROC near chance; (iii) the crossover point in p moves little with delta.
"""
from __future__ import annotations
import json, os, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

HERE = os.path.dirname(os.path.abspath(__file__))
D, DM, K = 20, 6, 5
N_TRAIN, N_TEST_N, N_FAULT = 4000, 1000, 300
SIG_W, SIG_N = 0.6, 0.05          # within-regime spread along U, isotropic noise
OFF_Z = 4.0                       # pure off-manifold fault = +4 sd of normal reconstruction error
P_GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
DELTA_GRID = [4.0, 6.0, 10.0]     # centre separation scale (small = more overlap)
SEEDS = [0, 1, 2]


def make_world(delta, seed):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(D, D)))
    U_in, U_off = U[:, :DM], U[:, DM:]
    C = rng.normal(size=(K, DM)); C *= delta / np.linalg.norm(C, axis=1, keepdims=True)  # centres on a sphere of radius delta
    return rng, U_in, U_off, C


def sample_normal(rng, U_in, C, n):
    k = rng.integers(0, K, n)
    z = C[k] + SIG_W * rng.normal(size=(n, DM))
    return z @ U_in.T + SIG_N * rng.normal(size=(n, D)), k


def sample_faults(rng, U_in, U_off, C, n, p):
    x, k = sample_normal(rng, U_in, C, n)
    dist = np.linalg.norm(C[:, None] - C[None], axis=2) + np.eye(K) * 1e9
    nn_ = dist.argmin(1)
    mid = 0.5 * (C[k] + C[nn_[k]])
    u_in = (mid - C[k]); m = np.linalg.norm(u_in, axis=1, keepdims=True); u_in = u_in / m   # unit, along U
    u_off = rng.normal(size=(n, D - DM)); u_off /= np.linalg.norm(u_off, axis=1, keepdims=True)
    # off-manifold magnitude is set on the NOISE scale (a pure off-manifold fault, p=0, adds about
    # OFF_Z standard deviations to the normal reconstruction-error distribution); the in-manifold
    # magnitude is set on the REGIME scale (p=1 lands at the midpoint of the gap to the nearest regime).
    # Without this, any off-manifold component is ~100x the noise floor and recon AUROC pins at 1.0.
    sd_rec = np.sqrt(2 * (D - DM)) * SIG_N ** 2
    m_off = np.sqrt(OFF_Z * sd_rec)
    v = (1 - p) * m_off * (u_off @ U_off.T) + p * m * (u_in @ U_in.T)
    return x + v


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, DM))
        self.dec = nn.Sequential(nn.Linear(DM, 32), nn.ReLU(), nn.Linear(32, D))
    def forward(self, x):
        z = self.enc(x); return self.dec(z), z


def fit_ae(Xtr, seed, epochs=150):
    torch.manual_seed(seed)
    ae = AE(); opt = torch.optim.Adam(ae.parameters(), 1e-2)
    X = torch.tensor(Xtr, dtype=torch.float32)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            xb = X[perm[i:i + 256]]; xr, _ = ae(xb); loss = ((xr - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return ae


def scores(ae, gmm, X):
    with torch.no_grad():
        xr, z = ae(torch.tensor(X, dtype=torch.float32))
    rec = ((xr - torch.tensor(X, dtype=torch.float32)) ** 2).mean(1).numpy()
    den = -gmm.score_samples(z.numpy())
    return rec, den


rows = []
t0 = time.time()
for delta in DELTA_GRID:
    for seed in SEEDS:
        rng, U_in, U_off, C = make_world(delta, seed)
        Xtr, _ = sample_normal(rng, U_in, C, N_TRAIN)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr_s = (Xtr - mu) / sd
        ae = fit_ae(Xtr_s, seed)
        with torch.no_grad():
            _, ztr = ae(torch.tensor(Xtr_s, dtype=torch.float32))
        gmm = GaussianMixture(K, covariance_type="full", random_state=seed, reg_covar=1e-4).fit(ztr.numpy())
        Xn, _ = sample_normal(rng, U_in, C, N_TEST_N)
        # multimodality of THIS world in observation space: silhouette of true regimes
        from sklearn.metrics import silhouette_score
        _, kk = sample_normal(rng, U_in, C, 600); Xs, kk = sample_normal(rng, U_in, C, 600)
        sil = float(silhouette_score(Xs, kk))
        for p in P_GRID:
            Xf = sample_faults(rng, U_in, U_off, C, N_FAULT, p)
            X = np.vstack([Xn, Xf]); y = np.r_[np.zeros(N_TEST_N), np.ones(N_FAULT)]
            rec, den = scores(ae, gmm, (X - mu) / sd)
            row = dict(delta=delta, seed=seed, p=p, silhouette=sil,
                       auroc_recon=float(roc_auc_score(y, rec)), auroc_density=float(roc_auc_score(y, den)))
            row["gap"] = row["auroc_density"] - row["auroc_recon"]
            rows.append(row)
            print(f"delta={delta} seed={seed} p={p:.1f} sil={sil:.2f}  recon={row['auroc_recon']:.3f} density={row['auroc_density']:.3f} gap={row['gap']:+.3f}  [{time.time()-t0:.0f}s]", flush=True)
        json.dump(rows, open(os.path.join(HERE, "framing_issue1_synth.json"), "w"), indent=1)

# summary: seed-mean per (delta, p)
summ = {}
for delta in DELTA_GRID:
    for p in P_GRID:
        r = [x for x in rows if x["delta"] == delta and x["p"] == p]
        summ[f"delta={delta},p={p}"] = dict(recon=float(np.mean([x["auroc_recon"] for x in r])),
                                          density=float(np.mean([x["auroc_density"] for x in r])),
                                          gap=float(np.mean([x["gap"] for x in r])),
                                          silhouette=float(np.mean([x["silhouette"] for x in r])))
json.dump(dict(rows=rows, summary=summ), open(os.path.join(HERE, "framing_issue1_synth.json"), "w"), indent=1)
print("\nSUMMARY (seed-mean)")
for k, v in summ.items():
    print(f"  {k:18s} sil={v['silhouette']:.2f} recon={v['recon']:.3f} density={v['density']:.3f} gap={v['gap']:+.3f}")

# figure: gap vs recon-AUROC (the RBI index), one curve per delta, plus the three real datasets
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    real = json.load(open(os.path.join(HERE, "framing_issue1_index.json")))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for delta in DELTA_GRID:
        xs = [summ[f"delta={delta},p={p}"]["recon"] for p in P_GRID]
        ys = [summ[f"delta={delta},p={p}"]["gap"] for p in P_GRID]
        ax[0].plot(P_GRID, ys, marker="o", label=f"separation {delta} (sil {summ[f'delta={delta},p=0.0']['silhouette']:.2f})")
        ax[1].plot(xs, ys, marker="o", label=f"separation {delta}")
    for ds, lab in [("HAI", "HAI"), ("WADI_clean", "WADI"), ("SWaT_canon", "SWaT")]:
        if ds in real:
            ax[1].scatter([real[ds]["ext_recon_index"]], [real[ds]["gap_density_minus_recon"]], marker="*", s=160, zorder=5, label=lab)
    ax[0].axhline(0, color="k", lw=0.6); ax[1].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("in-envelope-ness p of the fault (0 = off-manifold, 1 = between-regime)"); ax[0].set_ylabel("AUROC density - AUROC recon")
    ax[1].set_xlabel("reconstruction-detector AUROC on the faults (reconstructability index)"); ax[1].set_ylabel("AUROC density - AUROC recon")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "framing_issue1_synth.svg")); fig.savefig(os.path.join(HERE, "framing_issue1_synth.png"), dpi=130)
    print("figure saved")
except Exception as e:
    print("figure skipped:", e)
