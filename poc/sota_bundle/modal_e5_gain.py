"""E5 - gain ablation for the LatAD paper (MDPI IoT).

Difficult-subset AUROC (5 seeds, mean +/- sd) on 3 datasets (WADI, HAI, SWaT).
The difficult-subset protocol is taken verbatim from rev4_stats.py:
    easy      = (y == 1) & (maxz > maxz_thr)
    difficult = (y == 1) & ~easy
    keep      = (y == 0) | difficult
    per-seed AUROC on keep, then mean over seeds.

Arms
----
reference (always):
    the current reported LatAD head -- VaDE latent + high-K density GMM, scored with
    anomaly_score_hard(use_resid='auto', use_basin='auto'). The number the paper reports.

arm-b (DECISIVE, always):
    isolates whether the gain is cross-channel density. SAME pipeline otherwise; two
    density models on the stats window features:
      * crosschannel_latent : the current LatAD latent-density head (learned joint latent
                              + high-K diagonal GMM) -- models cross-channel structure.
      * marginal_product    : channel-independent product of per-dimension 1-D densities
                              (per-feature 1-D GMM; sum of per-dim log-densities) -- ignores
                              cross-channel structure.
    The crosschannel_latent - marginal_product gap on the difficult subset is the decisive
    "cross-channel density is what carries the difficult anomalies" test.

arm-a (OPTIONAL, behind --with-arm-a; the heavy part):
    representation sweep -- the SAME cross-channel latent-density stack applied to three
    representations:
      (i)  raw_conv               : raw 60xC windows via a 1-D CNN/GRU VaDE encoder,
      (ii) stats                  : the current stats window features (bundle Xn_w/Xa_w),
      (iii) stats_spectral_slopes : stats concatenated with per-channel spectral band
                                    energies + linear slopes (built from raw windows).

Results are written incrementally (one JSON row per (dataset, arm, variant, rep, seed),
appended + flushed to the results volume) and are resumable: a row already present is
skipped. Per-cell aggregation (mean +/- sd over seeds) is computed at the end.

Run (decisive arms only, all datasets):
    modal run modal_e5_gain.py
With the heavy representation sweep:
    modal run modal_e5_gain.py --with-arm-a
CPU:
    modal run modal_e5_gain.py --device cpu

  ==> PREPARE ONLY. Do not `modal run` until the user provides the Modal key + go.

TODOs a human must confirm before the first real run are marked  # TODO(confirm):
"""
from __future__ import annotations
import json
from pathlib import Path
import modal

HERE = Path(__file__).parent.resolve()

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python="3.11")
    .pip_install("numpy<2", "scikit-learn", "scipy")
    .add_local_file(str(HERE.parent / "models_vade.py"), "/app/models_vade.py")
    .add_local_file(str(HERE.parent / "rev4_stats.py"), "/app/rev4_stats.py")   # protocol reference
)
for ds in ["WADI", "HAI", "SWaT"]:
    b = HERE / "ens_bundle" / f"bundle_{ds}.npz"
    if b.exists():
        image = image.add_local_file(str(b), f"/app/bundle_{ds}.npz")
# per-timestep arrays: needed ONLY for arm-a (raw windows + spectral/slopes).
for pfx in ["wadi", "HAI", "SWaT"]:
    for suf in ["train", "test"]:
        lp = HERE / f"{pfx}_{suf}.npy"
        if lp.exists():
            image = image.add_local_file(str(lp), f"/app/{pfx}_{suf}.npy")

app = modal.App("latad-e5-gain", image=image)
results_vol = modal.Volume.from_name("latad-e3e5-results", create_if_missing=True)

# LatAD VaDE hyperparameters, mirroring export_checkpoints.py CFG = (K clusters, latent_dim).
# SWaT is absent there -> default to HAI's (40, 16).  # TODO(confirm): SWaT global-VaDE (K, latent_dim).
LATAD_CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}
PFX = {"WADI": "wadi", "HAI": "HAI", "SWaT": "SWaT"}
WIN, STRIDE = 60, 30       # bundle windowing (from checkpoint config); reproduces bundle window counts
NSEEDS = 5


# --------------------------------------------------------------------------- #
#  density models
# --------------------------------------------------------------------------- #
def _crosschannel_latent_score(Ztr, Zte, K, LD, seed, device, k_density):
    """The current LatAD latent-density head: VaDE joint latent + high-K diagonal GMM.
    Returns the pure cross-channel density z-score on Zte (use_near/resid/basin off, so
    ONLY the cross-channel latent density is measured)."""
    import numpy as np
    from models_vade import train_vade
    v = train_vade(Ztr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device=device)
    v.fit_latent_density(Ztr, k_density=k_density, seed=seed)
    s = v.anomaly_score_hard(Zte, use_near=False, use_recon=False, use_resid=False, use_basin=False)
    return np.asarray(s, dtype=float)


def _reference_score(Ztr, Zte, K, LD, seed, device, k_density):
    """The full reported LatAD head (auto resid + auto basin, nearest-mode on)."""
    import numpy as np
    from models_vade import train_vade
    v = train_vade(Ztr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device=device)
    v.fit_residual_whitener(Ztr); v.fit_latent_density(Ztr, k_density=k_density, seed=seed)
    v.fit_resid_head(Ztr, seed=seed); v.fit_basin_head(Ztr, seed=seed)
    s = v.anomaly_score_hard(Zte, use_resid="auto", use_basin="auto")
    return np.asarray(s, dtype=float)


def _marginal_product_score(Ztr, Zte, seed, k_marg=5, max_fit=20000):
    """Channel-independent product-of-1-D-marginal density. For each feature dimension
    fit a 1-D GMM (k_marg components) on train-normal; the test score is the sum over
    dimensions of the per-dim NLL (= product of marginals in probability space). This
    deliberately ignores ALL cross-channel structure, isolating the arm-b contrast."""
    import numpy as np
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(seed)
    n, d = Ztr.shape
    fit = Ztr if n <= max_fit else Ztr[rng.choice(n, max_fit, replace=False)]
    nll = np.zeros(len(Zte), dtype=float)
    for j in range(d):
        col = fit[:, j:j + 1]
        kj = min(k_marg, max(1, len(np.unique(col))))
        try:
            g = GaussianMixture(n_components=kj, covariance_type="full",
                                reg_covar=1e-3, random_state=seed).fit(col)
            nll += -g.score_samples(Zte[:, j:j + 1])
        except Exception:
            # degenerate/constant column -> standardized-Gaussian fallback
            mu, sd = col.mean(), col.std() + 1e-9
            nll += 0.5 * ((Zte[:, j] - mu) / sd) ** 2
    return nll


# --------------------------------------------------------------------------- #
#  arm-a representations (raw windows + spectral/slopes) -- OPTIONAL / HEAVY
# --------------------------------------------------------------------------- #
def _raw_windows(pfx, w=WIN, stride=STRIDE):
    """(n_win, w, C) raw windows for train and test from the per-timestep arrays.
    W=60/stride=30 reproduces the bundle window counts (verified: WADI 575, HAI 14819,
    SWaT 1087).  # TODO(confirm): window ORDER matches bundle_<DS>.npz exactly (same
    start index, same drop-last convention) so bundle y/maxz masks line up 1:1."""
    import numpy as np
    def win(X):
        idx = list(range(0, len(X) - w + 1, stride))
        return np.stack([X[i:i + w] for i in idx]).astype(np.float32)
    tr = win(np.load(f"/app/{pfx}_train.npy"))
    te = win(np.load(f"/app/{pfx}_test.npy"))
    return tr, te


def _spectral_slopes(win):
    """Per-channel spectral band energies (3 bands) + linear slope from (n,W,C) windows
    -> (n, C*4) extra features."""
    import numpy as np
    n, W, C = win.shape
    mag = np.abs(np.fft.rfft(win - win.mean(1, keepdims=True), axis=1))   # (n, F, C)
    F = mag.shape[1]
    edges = [0, F // 3, 2 * F // 3, F]
    bands = np.concatenate([mag[:, edges[b]:edges[b + 1], :].sum(1) for b in range(3)], axis=1)  # (n, 3C)
    t = np.arange(W, dtype=np.float32); tc = t - t.mean()
    slope = (win * tc[None, :, None]).sum(1) / (tc ** 2).sum()            # (n, C)
    return np.concatenate([bands, slope], axis=1).astype(np.float32)


class _ConvVaDE:
    """Minimal 1-D-CNN VaDE-style encoder for the raw_conv arm: Conv1d encoder -> latent,
    then the SAME high-K diagonal-GMM density head as models_vade.fit_latent_density.
    Kept self-contained (models_vade.VaDE has only a dense MLP encoder).
    # TODO(confirm): whether the paper wants a GRU encoder instead of Conv1d, and the
    #   exact encoder width/epochs; this is a reasonable default for the ablation."""
    def __init__(self, C, latent_dim, seed, device):
        import torch, torch.nn as nn
        torch.manual_seed(seed)
        self.dev = torch.device(device); self.d = latent_dim
        self.enc = nn.Sequential(
            nn.Conv1d(C, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(8), nn.Flatten(),
            nn.Linear(32 * 8, 64), nn.ReLU())
        self.fc_mu = nn.Linear(64, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, C))
        self.net = nn.ModuleList([self.enc, self.fc_mu, self.dec]).to(self.dev)
        self.gmm = None; self._ref = None

    def _z(self, X):
        import torch
        xt = torch.as_tensor(X, dtype=torch.float32, device=self.dev).transpose(1, 2)  # (n,C,W)
        return self.fc_mu(self.enc(xt))

    def fit(self, X, epochs=30, k_density=80, seed=0):
        import torch, numpy as np
        from sklearn.mixture import GaussianMixture
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        xt = torch.as_tensor(X, dtype=torch.float32, device=self.dev)
        tgt = xt.mean(1)                                     # reconstruct the per-channel window mean
        for _ in range(epochs):
            for i in range(0, len(xt), 256):
                z = self.fc_mu(self.enc(xt[i:i + 256].transpose(1, 2)))
                loss = ((self.dec(z) - tgt[i:i + 256]) ** 2).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ztr = self._z(X).cpu().numpy()
        self.gmm = GaussianMixture(n_components=k_density, covariance_type="diag",
                                   reg_covar=1e-3, random_state=seed).fit(ztr)
        d = -self.gmm.score_samples(ztr); self._ref = (d.mean(), d.std() + 1e-9)
        return self

    def score(self, X):
        import torch, numpy as np
        with torch.no_grad():
            z = self._z(X).cpu().numpy()
        d = -self.gmm.score_samples(z); m, s = self._ref
        return (d - m) / s


def _raw_conv_score(Wtr, Wte, LD, seed, device, k_density):
    import numpy as np
    C = Wtr.shape[2]
    m = _ConvVaDE(C, LD, seed, device).fit(Wtr, epochs=30, k_density=k_density, seed=seed)
    return np.asarray(m.score(Wte), dtype=float)


# --------------------------------------------------------------------------- #
#  per-dataset driver
# --------------------------------------------------------------------------- #
def _cells(with_arm_a):
    """(arm, variant, rep) cells to evaluate."""
    cells = [("reference", "latad_full", "stats"),
             ("arm_b", "crosschannel_latent", "stats"),
             ("arm_b", "marginal_product", "stats")]
    if with_arm_a:
        cells += [("arm_a", "crosschannel_latent", "raw_conv"),
                  ("arm_a", "crosschannel_latent", "stats"),
                  ("arm_a", "crosschannel_latent", "stats_spectral_slopes")]
    return cells


def _run_ds(name, nseeds, with_arm_a, device):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    B = np.load(f"/app/bundle_{name}.npz")
    Xn = B["Xn_w"].astype(np.float32); Xa = B["Xa_w"].astype(np.float32)
    y = B["y"].astype(int); thr = float(B["maxz_thr"]); maxz = B["maxz"]
    easy = (y == 1) & (maxz > thr); hard = (y == 1) & ~easy
    keep = np.where((y == 0) | hard)[0]
    n_hard = int(hard.sum()); n_norm = int((y == 0).sum())
    K, LD = LATAD_CFG[name]
    kd = min(80, max(20, len(Xn) // 10))

    # standardized stats features (shared by reference / arm-b / arm-a stats rung)
    mu, sig = Xn.mean(0), Xn.std(0) + 1e-8
    Zn = ((Xn - mu) / sig).astype(np.float32); Za = ((Xa - mu) / sig).astype(np.float32)

    # arm-a representations (built once per dataset; only if requested)
    reps = {"stats": (Zn, Za)}
    if with_arm_a:
        Wtr, Wte = _raw_windows(PFX[name])
        assert len(Wte) == len(y), (
            f"raw window count {len(Wte)} != bundle windows {len(y)} for {name}; "
            "windowing does not line up -- see _raw_windows TODO")
        reps["raw_conv"] = (Wtr, Wte)
        Sn = np.hstack([Xn, _spectral_slopes(Wtr)]); Sa = np.hstack([Xa, _spectral_slopes(Wte)])
        smu, ssig = Sn.mean(0), Sn.std(0) + 1e-8
        reps["stats_spectral_slopes"] = (((Sn - smu) / ssig).astype(np.float32),
                                         ((Sa - smu) / ssig).astype(np.float32))

    outdir = Path("/results/e5"); outdir.mkdir(parents=True, exist_ok=True)
    rows_path = outdir / f"rows_{device}.jsonl"
    done = set()
    if rows_path.exists():
        for ln in rows_path.read_text().splitlines():
            try:
                r = json.loads(ln); done.add((r["dataset"], r["arm"], r["variant"], r["rep"], r["seed"]))
            except Exception:
                pass

    rows = []
    for (arm, variant, rep) in _cells(with_arm_a):
        for seed in range(nseeds):
            key = (name, arm, variant, rep, seed)
            if key in done:
                print(f"[e5] skip (done): {key}", flush=True); continue
            try:
                if variant == "marginal_product":
                    s = _marginal_product_score(Zn, Za, seed)
                elif variant == "latad_full":
                    s = _reference_score(Zn, Za, K, LD, seed, device, kd)
                elif variant == "crosschannel_latent" and rep == "raw_conv":
                    Wtr, Wte = reps["raw_conv"]
                    s = _raw_conv_score(Wtr, Wte, LD, seed, device, kd)
                elif variant == "crosschannel_latent":
                    Ztr, Zte = reps[rep]
                    s = _crosschannel_latent_score(Ztr, Zte, K, LD, seed, device, kd)
                else:
                    raise ValueError(f"unknown cell {arm}/{variant}/{rep}")
                s = np.nan_to_num(np.asarray(s, dtype=float))
                auroc = float(roc_auc_score(y[keep], s[keep]))
                row = {"dataset": name, "arm": arm, "variant": variant, "rep": rep,
                       "seed": seed, "auroc_difficult": round(auroc, 4),
                       "n_difficult": n_hard, "n_normal": n_norm}
            except Exception as e:
                import traceback
                row = {"dataset": name, "arm": arm, "variant": variant, "rep": rep, "seed": seed,
                       "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1200:]}
            with open(rows_path, "a") as fh:
                fh.write(json.dumps(row) + "\n"); fh.flush()
            results_vol.commit()
            rows.append(row)
            print(f"[e5] {name} {arm}/{variant}/{rep} seed{seed} "
                  f"auroc={row.get('auroc_difficult')} err={row.get('error')}", flush=True)
    return rows


@app.function(gpu="A10G", cpu=8.0, timeout=5 * 60 * 60, memory=32768, volumes={"/results": results_vol})
def run_ds_gpu(name: str, nseeds: int, with_arm_a: bool) -> list:
    import sys; sys.path.insert(0, "/app")
    return _run_ds(name, nseeds, with_arm_a, "cuda")


@app.function(cpu=8.0, timeout=5 * 60 * 60, memory=32768, volumes={"/results": results_vol})
def run_ds_cpu(name: str, nseeds: int, with_arm_a: bool) -> list:
    import sys; sys.path.insert(0, "/app")
    return _run_ds(name, nseeds, with_arm_a, "cpu")


def _aggregate(all_rows):
    """mean +/- sd of difficult-subset AUROC over seeds, per (dataset, arm, variant, rep)."""
    import statistics
    agg = {}
    for r in all_rows:
        if "auroc_difficult" not in r:
            continue
        k = (r["dataset"], r["arm"], r["variant"], r["rep"])
        agg.setdefault(k, []).append(r["auroc_difficult"])
    out = []
    for k, vals in sorted(agg.items()):
        out.append({"dataset": k[0], "arm": k[1], "variant": k[2], "rep": k[3],
                    "auroc_mean": round(statistics.mean(vals), 4),
                    "auroc_sd": round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 4),
                    "n_seeds": len(vals)})
    return out


@app.local_entrypoint()
def main(datasets: str = "WADI,HAI,SWaT", seeds: int = NSEEDS,
         with_arm_a: bool = False, device: str = "cuda"):
    ds = [d for d in datasets.split(",") if d]
    fn = run_ds_gpu if device == "cuda" else run_ds_cpu
    print(f"[e5] launching device={device} datasets={ds} seeds={seeds} arm_a={with_arm_a}", flush=True)
    results = list(fn.starmap([(d, seeds, with_arm_a) for d in ds]))   # one container per dataset
    all_rows = [r for sub in results for r in sub]
    agg = _aggregate(all_rows)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"e5_gain_{device}.json").write_text(
        json.dumps({"rows": all_rows, "aggregate": agg}, indent=2))
    print("\n==================== E5 GAIN SUMMARY (difficult-subset AUROC, mean+/-sd) ====================")
    for a in agg:
        print(f"{a['dataset']:5} {a['arm']:10} {a['variant']:20} {a['rep']:22} "
              f"AUROC {a['auroc_mean']} +/- {a['auroc_sd']} (n={a['n_seeds']})")
    # highlight the decisive arm-b gap per dataset
    print("\n---- decisive arm-b gap (crosschannel_latent - marginal_product) ----")
    by = {(a["dataset"], a["variant"]): a["auroc_mean"] for a in agg if a["arm"] == "arm_b"}
    for d in ds:
        cc = by.get((d, "crosschannel_latent")); mp = by.get((d, "marginal_product"))
        if cc is not None and mp is not None:
            print(f"{d:5} cross-channel {cc}  -  marginal {mp}  =  gain {round(cc - mp, 4)}")
