"""
Component 2 - the basin-of-attraction test (the proposal's false-positive fix).

Idea: a rare-but-VALID operating mode and a genuine fault look alike to a
density score (both sit in a low-density region - the valid one only because its
mode is under-sampled). They differ in the DYNAMICS of the latent potential:

  U(z) = -log sum_c N(z | mu_c, Sigma_c)          (mixture-density potential)

  - A valid point (common or rare) lies inside ONE basin of U. Gradient descent
    from several perturbed restarts all rolls into the SAME mode, over a SHORT
    distance: high 'agreement', small 'convergence distance'.
  - A pocket fault sits on the ridge/separatrix BETWEEN two modes. Perturbed
    restarts split between the two basins: LOW agreement.
  - An OOD fault is far from every basin: LARGE convergence distance.

Both signals are read from geometry, not from the (mis-estimated) density of a
rare component, which is exactly why the test rescues rare-but-valid modes that
the density score wrongly flags.
"""

from __future__ import annotations

import numpy as np
import torch

from models_vade import LOG2PI


def _potential(z, mu_c, lvc):
    """U(z) = -logsumexp_c log N(z|mu_c,Sigma_c), and per-component log-density."""
    z_e = z.unsqueeze(1)                       # (B,1,d)
    logN = -0.5 * (LOG2PI * z.shape[1]
                   + torch.sum(lvc.unsqueeze(0) + (z_e - mu_c.unsqueeze(0)) ** 2
                               / torch.exp(lvc.unsqueeze(0)), dim=2))   # (B,K)
    U = -torch.logsumexp(logN, dim=1)          # (B,)
    return U, logN


@torch.no_grad()
def _encode(vade, x):
    dev = next(vade.parameters()).device
    xt = torch.as_tensor(x, dtype=torch.float32, device=dev)
    return vade.encode(xt)[0]


def basin_features(vade, x, restarts=8, steps=80, step_size=0.35, pert=0.3,
                   chunk=4096):
    """Return per-point (agreement, conv_distance).

    agreement     : fraction of restarts that converge to the modal cluster
                    (1 = one stable basin; ~0.5 = split across two = pocket).
    conv_distance : mean latent distance travelled to the basin (large = OOD).

    The step budget (steps * step_size / 2 ~= 14 latent units of path length)
    is set large enough to actually cross between neighbouring mode basins - a
    smaller budget only measures local stability, not true basin convergence.
    """
    dev = next(vade.parameters()).device
    z0_all = _encode(vade, x)
    mu_c = vade.mu_c.detach()
    lvc = vade._lvc().detach()
    N, d = z0_all.shape
    K = mu_c.shape[0]
    agreement = np.zeros(N, dtype=np.float32)
    conv_dist = np.zeros(N, dtype=np.float32)

    for s in range(0, N, chunk):
        z0 = z0_all[s:s + chunk]
        B = z0.shape[0]
        finals = torch.empty((restarts, B), dtype=torch.long, device=dev)
        dsum = torch.zeros(B, device=dev)
        for r in range(restarts):
            z = (z0 + pert * torch.randn_like(z0)).detach().requires_grad_(True)
            for t in range(steps):
                U, _ = _potential(z, mu_c, lvc)
                g = torch.autograd.grad(U.sum(), z)[0]
                g = g / (g.norm(dim=1, keepdim=True) + 1e-8)   # unit-speed descent
                lr = step_size * (1.0 - t / steps)             # decay -> settles
                z = (z - lr * g).detach().requires_grad_(True)
            with torch.no_grad():
                _, logN = _potential(z, mu_c, lvc)
                finals[r] = logN.argmax(1)
                dsum += torch.norm(z - z0, dim=1)
        finals_np = finals.cpu().numpy()                       # (R,B)
        for i in range(B):
            agreement[s + i] = np.bincount(finals_np[:, i], minlength=K).max() / restarts
        conv_dist[s:s + B] = (dsum / restarts).cpu().numpy()

    return agreement, conv_dist


def _standardizer(train_vals):
    mu, sd = float(np.mean(train_vals)), float(np.std(train_vals) + 1e-8)
    return lambda v: (v - mu) / sd


def vade_scores(vade, x_train, x_test, **basin_kw):
    """Build the base and basin-augmented anomaly scores for a trained VaDE.

    Returns a dict method_name -> (train_score, test_score). Every term is
    standardised on NORMAL training statistics before fusion (equal-weight,
    calibration-free), so no abnormal data and no hand-tuned weights are used.
    """
    # --- reconstruction + latent-density terms (base score, already fused) ---
    base_tr = vade.anomaly_score(x_train)
    base_te = vade.anomaly_score(x_test)

    # --- basin features ---
    agr_tr, dist_tr = basin_features(vade, x_train, **basin_kw)
    agr_te, dist_te = basin_features(vade, x_test, **basin_kw)

    # Component-2 score = reconstruction energy + basin INSTABILITY (1-agreement).
    # Reconstruction already flags OOD (bad reconstruction); instability flags a
    # pocket (split basin) WITHOUT penalising a rare-but-valid mode (stable basin,
    # high agreement). The density term is dropped precisely because it is the
    # source of the rare-mode false positive. conv_distance is kept only for
    # diagnostics (it re-penalises rare modes, so it is not in the score).
    rec_tr = _recon_only(vade, x_train)
    rec_te = _recon_only(vade, x_test)
    z_rec = _standardizer(rec_tr)
    z_instab = _standardizer(1 - agr_tr)

    full_tr = z_rec(rec_tr) + z_instab(1 - agr_tr)
    full_te = z_rec(rec_te) + z_instab(1 - agr_te)

    return {
        "VaDE (joint, ours)": (base_tr, base_te),
        "VaDE + basin (full, ours)": (full_tr, full_te),
    }, {"agreement_test": agr_te, "dist_test": dist_te}


@torch.no_grad()
def _recon_only(vade, x):
    from models_vade import _recon_energy, _as_tensor
    xt = _as_tensor(x, vade)
    mu, _ = vade.encode(xt)
    return _recon_energy(xt, vade.decode(mu), vade.res_whitener)


# --------------------------------------------------------------------------- #
#  Standalone check: do the basin features separate the groups as claimed?
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from data import make_miim
    from models_vade import train_vade

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, meta = make_miim(n_modes=40, seed=0)
    vade = train_vade(ds.x_train, n_clusters=40, latent_dim=10, epochs=60,
                      seed=0, device=device)
    vade.fit_residual_whitener(ds.x_train)
    agr, dist = basin_features(vade, ds.x_test)

    rare = set(meta["rare_modes"])
    groups = {
        "common-normal": (ds.y_test == 0) & np.array([m not in rare for m in ds.mode_test]),
        "rare-normal": (ds.y_test == 0) & np.array([m in rare for m in ds.mode_test]),
        "pocket": ds.atype_test == "pocket",
        "ood": ds.atype_test == "ood",
    }
    print(f"{'group':<16}{'n':>7}{'agreement':>12}{'conv_dist':>12}")
    for name, mask in groups.items():
        if mask.any():
            print(f"{name:<16}{int(mask.sum()):>7}{agr[mask].mean():>12.3f}{dist[mask].mean():>12.3f}")
