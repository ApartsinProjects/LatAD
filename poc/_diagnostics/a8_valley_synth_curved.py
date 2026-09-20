"""A8 valley-smoothing synthetic control, CURVED-manifold version (separates interpolation from extrapolation).

The linear-manifold control (a8_valley_synth.py) cannot separate "valley" (between regimes) from "beyond"
(past a regime, away from the other) for a reconstructor: any in-subspace displacement is reconstructable.
Here the normal manifold is curved:  x = U_in z + kappa * U_q q(z) / gap_ref + sigma_n eps, with q(z) the 10
quadratic monomials z_i z_j of the 4 manifold coordinates and U_q a 10-dim frame orthogonal to U_in. Regimes A
and B sit at z1 = 0 and z1 = gap on the surface. Anomalies (all from regime-A points, all of size gap/2 in z):
  valley : ON the surface at z1 = gap/2   (extreme of A drifting toward B along the physical manifold;
           reachable by a decoder that interpolates between the two regimes = the hypothesised blind spot)
  beyond : ON the surface at z1 = -gap/2  (extreme of A away from B; reachable only by extrapolating the curve)
  chord  : straight-line midpoint in observation space between the A point and its B image (off the surface
           by the sagitta; what a decoder that interpolates linearly in x would produce)
  off    : displacement of the same size orthogonal to both U_in and U_q
kappa = 0 reproduces the linear world. Detectors as before (MLP AE recon, GMM-in-latent density, PCA linear recon).
Invariants: (C1) kappa = 0 -> valley ~ beyond for recon (linear world result); (C2) off is caught at every setting;
(C3) if the smoothing mechanism is real: with kappa > 0, recon AUROC(valley) << recon AUROC(beyond) and the
density head catches both; (C4) the deficit should be largest for gaps where the valley is a genuine density
trough (gap >= 6) and vanish for gaps where the midpoint is inside the envelopes (gap <= 4).
Writes a8_valley_synth_curved.jsonl (resumable) and a8_valley_synth_curved.png.
"""
from __future__ import annotations
import json, os, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "a8_valley_synth_curved.jsonl")
D, DM, NQ = 24, 4, 10
SIG_W, SIG_N = 1.0, 0.05
N_TRAIN, N_TEST_N, N_ANOM = 6000, 1500, 400
GAPS = [4.0, 6.0, 8.0, 12.0]; KAPPAS = [0.0, 0.5, 1.0]; SEEDS = [0, 1, 2]
KINDS = ["valley", "beyond", "chord", "off"]


def emit(row):
    with open(OUT, "a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n"); f.flush()


def done():
    s = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            try: r = json.loads(l); s.add((r["gap"], r["kappa"], r["seed"]))
            except Exception: pass
    return s


class World:
    def __init__(self, gap, kappa, seed):
        self.rng = np.random.default_rng(seed); self.gap, self.kappa = gap, kappa
        Q, _ = np.linalg.qr(self.rng.normal(size=(D, D))); self.U_in, self.U_q, self.U_off = Q[:, :DM], Q[:, DM:DM + NQ], Q[:, DM + NQ:]
        self.iu = np.triu_indices(DM); self.gref = max(gap, 4.0); self.m = gap / 2

    def surf(self, z):
        q = (z[:, :, None] * z[:, None, :])[:, self.iu[0], self.iu[1]]          # (n, 10) quadratic monomials
        return z @ self.U_in.T + self.kappa * (q @ self.U_q.T) / self.gref

    def normal(self, n, k=None):
        if k is None: k = self.rng.integers(0, 2, n)
        z = SIG_W * self.rng.normal(size=(n, DM)); z[:, 0] += self.gap * k
        return self.surf(z) + SIG_N * self.rng.normal(size=(n, D)), z, k

    def anomalies(self, n, kind):
        _, z, _ = self.normal(n, k=np.zeros(n, int))
        if kind in ("valley", "beyond"):
            z2 = z.copy(); z2[:, 0] += (self.m if kind == "valley" else -self.m); return self.surf(z2) + SIG_N * self.rng.normal(size=(n, D))
        if kind == "chord":
            zB = z.copy(); zB[:, 0] += self.gap
            return 0.5 * (self.surf(z) + self.surf(zB)) + SIG_N * self.rng.normal(size=(n, D))
        if kind == "off":
            u = self.rng.normal(size=(n, D - DM - NQ)); u /= np.linalg.norm(u, axis=1, keepdims=True)
            return self.surf(z) + self.m * (u @ self.U_off.T) + SIG_N * self.rng.normal(size=(n, D))


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, DM))
        self.dec = nn.Sequential(nn.Linear(DM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, D))
    def forward(self, x): z = self.enc(x); return self.dec(z), z


def fit_ae(X, seed, epochs=200):
    torch.manual_seed(seed); ae = AE(); opt = torch.optim.Adam(ae.parameters(), 3e-3); Xt = torch.tensor(X, dtype=torch.float32)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            xb = Xt[perm[i:i + 256]]; xr, _ = ae(xb); loss = ((xr - xb) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return ae


def score(ae, gmm, pca, X):
    with torch.no_grad(): Xt = torch.tensor(X, dtype=torch.float32); xr, z = ae(Xt)
    return dict(recon=((xr - Xt) ** 2).mean(1).numpy(), density=-gmm.score_samples(z.numpy()), linrec=((pca.inverse_transform(pca.transform(X)) - X) ** 2).mean(1))


def auroc(sn, sa): return float(roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(sa))], np.r_[sn, sa]))


t0 = time.time()
for kappa in KAPPAS:
    for gap in GAPS:
        for seed in SEEDS:
            if (gap, kappa, seed) in done(): continue
            Wd = World(gap, kappa, seed); Xtr, _, _ = Wd.normal(N_TRAIN); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; S = lambda X: (X - mu) / sd
            ae = fit_ae(S(Xtr), seed)
            with torch.no_grad(): _, ztr = ae(torch.tensor(S(Xtr), dtype=torch.float32))
            ztr = ztr.numpy(); fits = [GaussianMixture(k, covariance_type="full", random_state=seed, reg_covar=1e-4).fit(ztr) for k in (1, 2, 3, 4)]
            gmm = fits[int(np.argmin([g.bic(ztr) for g in fits]))]; pca = PCA(DM).fit(S(Xtr))
            Xn, _, _ = Wd.normal(N_TEST_N); sn = score(ae, gmm, pca, S(Xn))
            row = dict(gap=gap, kappa=kappa, seed=seed, K_bic=gmm.n_components, train_recon_med=float(np.median(sn["recon"])), train_linrec_med=float(np.median(sn["linrec"])))
            for kind in KINDS:
                Xa = Wd.anomalies(N_ANOM, kind); sa = score(ae, gmm, pca, S(Xa))
                for det in ["recon", "density", "linrec"]:
                    row[f"auroc_{det}_{kind}"] = auroc(sn[det], sa[det]); row[f"rec99_{det}_{kind}"] = float((sa[det] > np.quantile(sn[det], .99)).mean())
                    row[f"medratio_{det}_{kind}"] = float(np.median(sa[det]) / np.median(sn[det]))
            row["secs"] = round(time.time() - t0, 1); emit(row)
            print(f"kappa={kappa} gap={gap} seed={seed} K={gmm.n_components} | recon v/b/c/o = " + "/".join(f"{row[f'auroc_recon_{k}']:.3f}" for k in KINDS) + " | density = " + "/".join(f"{row[f'auroc_density_{k}']:.3f}" for k in KINDS) + " | linrec = " + "/".join(f"{row[f'auroc_linrec_{k}']:.3f}" for k in KINDS) + f"  [{time.time()-t0:.0f}s]", flush=True)

rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
summ = {}
print("\nSUMMARY seed-mean AUROC, kinds valley/beyond/chord/off")
for kappa in KAPPAS:
    for gap in GAPS:
        rs = [r for r in rows if r["gap"] == gap and r["kappa"] == kappa]
        if not rs: continue
        g = lambda k: float(np.mean([r[k] for r in rs])); summ[f"kappa={kappa},gap={gap}"] = {k: g(k) for k in rs[0] if k.startswith(("auroc_", "rec99_", "medratio_"))}
        print(f"  kappa={kappa} gap={gap:5.1f} | recon " + "/".join(f"{g(f'auroc_recon_{k}'):.3f}" for k in KINDS) + " | density " + "/".join(f"{g(f'auroc_density_{k}'):.3f}" for k in KINDS) + " | linrec " + "/".join(f"{g(f'auroc_linrec_{k}'):.3f}" for k in KINDS) + f" | recall@1% recon v/b {g('rec99_recon_valley'):.2f}/{g('rec99_recon_beyond'):.2f} density v/b {g('rec99_density_valley'):.2f}/{g('rec99_density_beyond'):.2f}")
json.dump(summ, open(os.path.join(HERE, "a8_valley_synth_curved_summary.json"), "w"), indent=1)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(KAPPAS), figsize=(4.3 * len(KAPPAS), 3.8), sharey=True)
    for a, kappa in zip(ax, KAPPAS):
        for det, ls in [("recon", "-"), ("density", "--")]:
            for kind, c in [("valley", "C3"), ("beyond", "C0"), ("chord", "C1"), ("off", "C2")]:
                a.plot(GAPS, [summ[f"kappa={kappa},gap={g}"][f"auroc_{det}_{kind}"] for g in GAPS], ls, color=c, marker="o", label=f"{det} / {kind}")
        a.set_title(f"curvature kappa = {kappa}"); a.set_xlabel("gap (within-regime sd); anomaly size = gap/2"); a.set_ylim(0.4, 1.02); a.axhline(0.5, color="k", lw=0.5)
    ax[0].set_ylabel("AUROC vs test normal"); ax[-1].legend(fontsize=6.5); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "a8_valley_synth_curved.png"), dpi=130); fig.savefig(os.path.join(HERE, "a8_valley_synth_curved.svg")); print("figure saved")
except Exception as e:
    print("figure skipped:", e)
