"""Pilot part 2: (a) gap measure in VaDE LATENT space (second view), (b) SKAB gate stability
across seeds at the reported config (K16/LD6/ep40): rho, H_norm, basin_lam. Appends to fable_a3_pilot.json."""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from fable_a3_pilot import geometry, summarise, graded_from_logN, load_skab, OUT
from sklearn.neighbors import NearestNeighbors
from scipy.special import logsumexp
RES = json.load(open(OUT))
def save():
    json.dump(RES, open(OUT, "w"), indent=1)

def latent_gap(z, logN, logpi, mu_c, tag):
    lp = logpi[None] + logN; G = np.exp(lp - logsumexp(lp, 1, keepdims=True)); lab = G.argmax(1)
    K = len(mu_c); z = z.astype(np.float64); C = mu_c.astype(np.float64)
    nn = NearestNeighbors(n_neighbors=11).fit(z); rad = nn.kneighbors(z)[0][:, -1]
    dc = ((z[:, None, :] - C[None]) ** 2).sum(-1); o = np.argsort(dc, 1); a, b = o[:, 0], o[:, 1]
    v = C[b] - C[a]; L2 = (v ** 2).sum(1) + 1e-12; t = ((z - C[a]) * v).sum(1) / L2
    proj = C[a] + np.clip(t, 0, 1)[:, None] * v; perp = np.linalg.norm(z - proj, axis=1)
    r_own = np.sqrt(dc[np.arange(len(z)), lab])
    med_r = np.array([np.median(r_own[lab == c]) if (lab == c).sum() > 0 else np.nan for c in range(K)])
    thr = np.nan_to_num(med_r[a], nan=np.nanmedian(med_r))
    interior = (t > 0.2) & (t < 0.8); between = interior & (perp <= thr); core = ~interior
    q90 = np.quantile(rad[core], 0.9); gap = between & (rad > q90)
    r = dict(interior=float(interior.mean()), between=float(between.mean()), gap=float(gap.mean()),
             gap_over_between=float(gap.sum() / max(1, between.sum())),
             rad_between_over_core=float(np.median(rad[between]) / np.median(rad[core])) if between.any() else None)
    print(f"  [{tag}] latent-space: {r}", flush=True); return r

RES.setdefault("latent_gap", {})
for name in ["WADI", "HAI", "SWaT"]:
    if name in RES["latent_gap"]: continue
    d = np.load(os.path.join(HERE, f"e2_fable_{name}.npz"))
    RES["latent_gap"][name] = latent_gap(d["ztr"], d["logN_tr"], d["logpi"], d["mu_c"], name); save()

from models_vade import train_vade
Xtr_s = load_skab()[0]
RES.setdefault("skab_gate_seeds", {})
for seed in range(5):
    key = f"s{seed}"
    if key in RES["skab_gate_seeds"]: continue
    v = train_vade(Xtr_s, n_clusters=16, latent_dim=6, epochs=40, warmup=8, seed=seed, device="cpu")
    v.fit_basin_head(Xtr_s)
    with torch.no_grad():
        mu = v.encode(torch.as_tensor(Xtr_s))[0]; logN = v._log_pz_given_c(mu).cpu().numpy()
        logpi = torch.log_softmax(v.pi_logit, 0).cpu().numpy(); z = mu.cpu().numpy()
        mu_c = v.mu_c.detach().cpu().numpy() if hasattr(v, "mu_c") else None
    g = graded_from_logN(logN, logpi); g["basin_lam"] = float(v._basin_lam)
    if mu_c is not None and mu_c.shape[0] == 16:
        g["latent_gap"] = latent_gap(z, logN, logpi, mu_c, f"SKAB s{seed}")
    else:
        g["latent_gap"] = "mu_c attr not found"
    RES["skab_gate_seeds"][key] = g; print(f"  [SKAB seed {seed}] rho={g['rho_0p5']:.3f} H={g['H_norm_mean']:.3f} lam={g['basin_lam']:.2f}", flush=True); save()
print("done")
