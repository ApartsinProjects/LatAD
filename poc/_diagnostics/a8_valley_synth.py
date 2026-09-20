"""A8 valley-smoothing synthetic control: two close normal regimes, three anomaly directions of equal size.

World. D = 20 observed dims. A d_m = 4 dim manifold U_in (random orthonormal frame). Two regimes A, B whose
centres differ by `gap` (in within-regime sd units, sigma_w = 1) along the first manifold axis e1; within-regime
spread sigma_w along all d_m manifold axes; isotropic noise sigma_n = 0.05 in R^D. Optional SINGLE-regime world
(gap = None): only A exists. Detectors are trained on normal data only:
  recon   : MLP autoencoder with latent d_m, score = reconstruction error (the "deep reconstructor" stand-in)
  density : GMM in the AE latent, K by BIC in {1,2,3,4}, score = -log p(z)  (the latent-density head stand-in)
  linrec  : PCA-d_m reconstruction error (linear reference; a linear model cannot "curve" the manifold)
Anomalies. Start from a normal A point, displace by magnitude m = gap/2 (lands at the inter-regime midpoint):
  valley : along +e1 (toward B)                 -> the hypothesised blind spot of the reconstructor
  beyond : along -e1 (away from B)              -> same magnitude, same axis, no regime on the other side
  off    : orthogonal to U_in (off-manifold)    -> unreachable by any decoder
Also a 1-D profile along the A->B axis (t from -1.5 to 2.5, in units of the gap) of recon error and -log p.
Invariants stated in advance: (S1) single-regime world: valley == beyond for both detectors (symmetric); any
valley-vs-beyond asymmetry with B present is the smoothing effect. (S2) off anomalies are caught by the recon
head at every gap. (S3) if the mechanism is real, recon AUROC on valley falls below recon AUROC on beyond, the
deficit is largest at small gaps and shrinks at large gaps; density AUROC on valley stays high. (S4) the linear
reconstructor treats valley and beyond identically (it cannot bend), so any asymmetry is a nonlinear-decoder effect.
Writes a8_valley_synth.jsonl (one row per (gap, seed), resumable) and a8_valley_synth.png/.svg.
Usage (from poc/): python _diagnostics/a8_valley_synth.py
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "a8_valley_synth.jsonl")
D, DM = 20, 4
SIG_W, SIG_N = 1.0, 0.05
N_TRAIN, N_TEST_N, N_ANOM = 6000, 1500, 400
GAPS = [None, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0]     # None = single regime (invariant S1); m = gap/2, single regime uses m = 2.5
SEEDS = [0, 1, 2]
T_GRID = np.linspace(-1.5, 2.5, 81)


def emit(row):
    with open(OUT, "a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n"); f.flush()


def done():
    s = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            try: r = json.loads(l); s.add((r["gap"], r["seed"]))
            except Exception: pass
    return s


class World:
    def __init__(self, gap, seed):
        self.rng = np.random.default_rng(seed); self.gap = gap
        Q, _ = np.linalg.qr(self.rng.normal(size=(D, D))); self.U_in, self.U_off = Q[:, :DM], Q[:, DM:]
        self.cA = np.zeros(DM); self.cB = np.zeros(DM); self.cB[0] = gap if gap is not None else 0.0
        self.e1 = self.U_in[:, 0]                    # observed-space direction A -> B
        self.m = (gap / 2) if gap is not None else 2.5

    def normal(self, n):
        if self.gap is None: k = np.zeros(n, int)
        else: k = self.rng.integers(0, 2, n)
        c = np.where(k[:, None] == 0, self.cA[None], self.cB[None])
        z = c + SIG_W * self.rng.normal(size=(n, DM))
        return z @ self.U_in.T + SIG_N * self.rng.normal(size=(n, D)), k

    def anomalies(self, n, kind):
        x, k = self.normal(2 * n); x = x[k == 0][:n]        # start from regime A points only
        if kind == "valley": v = self.m * self.e1[None]
        elif kind == "beyond": v = -self.m * self.e1[None]
        elif kind == "off":
            u = self.rng.normal(size=(len(x), D - DM)); u /= np.linalg.norm(u, axis=1, keepdims=True); v = self.m * (u @ self.U_off.T)
        return x + v

    def profile(self):
        # points along the A->B axis at t*gap (or t*2m in the single world), with within-regime spread in the OTHER manifold dims
        L = self.gap if self.gap is not None else 2 * self.m
        z = SIG_W * self.rng.normal(size=(len(T_GRID), 200, DM)); z[:, :, 0] = (T_GRID * L)[:, None]
        return z.reshape(-1, DM) @ self.U_in.T + SIG_N * self.rng.normal(size=(len(T_GRID) * 200, D))


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, DM))
        self.dec = nn.Sequential(nn.Linear(DM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, D))
    def forward(self, x): z = self.enc(x); return self.dec(z), z


def fit_ae(X, seed, epochs=200):
    torch.manual_seed(seed); ae = AE(); opt = torch.optim.Adam(ae.parameters(), 3e-3)
    Xt = torch.tensor(X, dtype=torch.float32)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            xb = Xt[perm[i:i + 256]]; xr, _ = ae(xb); loss = ((xr - xb) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return ae


def score(ae, gmm, pca, X):
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32); xr, z = ae(Xt)
    rec = ((xr - Xt) ** 2).mean(1).numpy(); den = -gmm.score_samples(z.numpy())
    lin = ((pca.inverse_transform(pca.transform(X)) - X) ** 2).mean(1)
    return dict(recon=rec, density=den, linrec=lin)


def auroc(sn, sa): return float(roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(sa))], np.r_[sn, sa]))


t0 = time.time()
for gap in GAPS:
    for seed in SEEDS:
        if (gap, seed) in done(): continue
        Wd = World(gap, seed); Xtr, _ = Wd.normal(N_TRAIN); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        S = lambda X: (X - mu) / sd
        ae = fit_ae(S(Xtr), seed)
        with torch.no_grad(): _, ztr = ae(torch.tensor(S(Xtr), dtype=torch.float32))
        ztr = ztr.numpy()
        fits = [GaussianMixture(k, covariance_type="full", random_state=seed, reg_covar=1e-4).fit(ztr) for k in (1, 2, 3, 4)]
        gmm = fits[int(np.argmin([g.bic(ztr) for g in fits]))]
        pca = PCA(DM).fit(S(Xtr))
        Xn, _ = Wd.normal(N_TEST_N); sn = score(ae, gmm, pca, S(Xn))
        row = dict(gap=gap, seed=seed, m=Wd.m, K_bic=gmm.n_components, train_recon_q99=float(np.quantile(sn["recon"], .99)))
        for kind in ["valley", "beyond", "off"]:
            Xa = Wd.anomalies(N_ANOM, kind); sa = score(ae, gmm, pca, S(Xa))
            for det in ["recon", "density", "linrec"]:
                row[f"auroc_{det}_{kind}"] = auroc(sn[det], sa[det])
                row[f"rec99_{det}_{kind}"] = float((sa[det] > np.quantile(sn[det], .99)).mean())
                row[f"medpct_{det}_{kind}"] = float(np.median(np.searchsorted(np.sort(sn[det]), sa[det]) / len(sn[det])))
        # 1-D profile along the axis
        Xp = Wd.profile(); sp = score(ae, gmm, pca, S(Xp))
        for det in ["recon", "density", "linrec"]:
            v = sp[det].reshape(len(T_GRID), 200)
            # express as percentile of the test-normal score distribution (0..1), median over the 200 draws per t
            pct = np.searchsorted(np.sort(sn[det]), v) / len(sn[det])
            row[f"profile_{det}"] = np.median(pct, 1).round(4).tolist()
        row["profile_t"] = T_GRID.round(3).tolist(); row["secs"] = round(time.time() - t0, 1)
        emit(row)
        print(f"gap={gap} seed={seed} K={gmm.n_components} | recon valley/beyond/off = {row['auroc_recon_valley']:.3f}/{row['auroc_recon_beyond']:.3f}/{row['auroc_recon_off']:.3f} | density = {row['auroc_density_valley']:.3f}/{row['auroc_density_beyond']:.3f}/{row['auroc_density_off']:.3f} | linrec = {row['auroc_linrec_valley']:.3f}/{row['auroc_linrec_beyond']:.3f}/{row['auroc_linrec_off']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

# ---- summary + figure
rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
print("\nSUMMARY seed-mean AUROC (recon valley / beyond / off | density valley / beyond / off | linrec valley / beyond)")
summ = {}
for gap in GAPS:
    rs = [r for r in rows if r["gap"] == gap]
    if not rs: continue
    g = lambda k: float(np.mean([r[k] for r in rs]))
    summ[str(gap)] = {k: g(k) for k in rs[0] if k.startswith(("auroc_", "rec99_", "medpct_"))}
    print(f"  gap={str(gap):5s} n={len(rs)} | {g('auroc_recon_valley'):.3f} / {g('auroc_recon_beyond'):.3f} / {g('auroc_recon_off'):.3f} | {g('auroc_density_valley'):.3f} / {g('auroc_density_beyond'):.3f} / {g('auroc_density_off'):.3f} | {g('auroc_linrec_valley'):.3f} / {g('auroc_linrec_beyond'):.3f}   recall@1%: recon v/b {g('rec99_recon_valley'):.2f}/{g('rec99_recon_beyond'):.2f} density v/b {g('rec99_density_valley'):.2f}/{g('rec99_density_beyond'):.2f}")
json.dump(summ, open(os.path.join(HERE, "a8_valley_synth_summary.json"), "w"), indent=1)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    gaps = [g for g in GAPS if g is not None and str(g) in summ]
    for det, ls in [("recon", "-"), ("density", "--")]:
        for kind, c in [("valley", "C3"), ("beyond", "C0"), ("off", "C2")]:
            ax[0].plot(gaps, [summ[str(g)][f"auroc_{det}_{kind}"] for g in gaps], ls, color=c, marker="o", label=f"{det} / {kind}")
    if "None" in summ:
        ax[0].axhline(summ["None"]["auroc_recon_valley"], color="C3", lw=0.6, ls=":"); ax[0].axhline(summ["None"]["auroc_density_valley"], color="C3", lw=0.6, ls=":")
    ax[0].set_xlabel("gap between regime centres (sd units); anomaly size = gap/2"); ax[0].set_ylabel("AUROC vs test normal"); ax[0].legend(fontsize=6.5); ax[0].set_ylim(0.4, 1.02)
    for gi, gap in enumerate([g for g in (4.0, 8.0) if str(g) in summ]):
        rs = [r for r in rows if r["gap"] == gap]; t = np.array(rs[0]["profile_t"])
        for det, c in [("recon", "C1"), ("density", "C4"), ("linrec", "C7")]:
            p = np.mean([r[f"profile_{det}"] for r in rs], 0); ax[1 + gi].plot(t, p, color=c, label=det)
        ax[1 + gi].axvline(0, color="k", lw=0.5); ax[1 + gi].axvline(1, color="k", lw=0.5); ax[1 + gi].axhline(0.99, color="k", lw=0.5, ls=":")
        ax[1 + gi].set_title(f"score along the A->B axis, gap = {gap:g}"); ax[1 + gi].set_xlabel("position t (0 = centre A, 1 = centre B)"); ax[1 + gi].set_ylabel("score as test-normal percentile"); ax[1 + gi].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "a8_valley_synth.png"), dpi=130); fig.savefig(os.path.join(HERE, "a8_valley_synth.svg")); print("figure saved")
except Exception as e:
    print("figure skipped:", e)
