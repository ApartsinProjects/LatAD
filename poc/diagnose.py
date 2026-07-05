"""
Root-cause diagnostics for 'why does joint VaDE underperform sequential VAE+GMM'.

For one seed it trains both models and reports, side by side:
  - reconstruction MSE on normal data (did joint training hurt the decoder?)
  - clustering quality vs GROUND-TRUTH modes (NMI, ACC) - does joint discover
    the operating modes better or worse than a post-hoc GMM?
  - score DECOMPOSITION: AUROC from reconstruction alone, from latent density
    alone, and combined - which term carries each method?
  - per-fault-type AUROC: OOD-vs-normal and POCKET-vs-normal separately, so we
    see exactly where joint loses.

Run:  C:/Python314/python.exe diagnose.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score

from data import make_miim
from models_vade import LOG2PI, train_plain_vae, train_vade

SEED = 0
N_MODES = 40
LATENT = 10
EPOCHS = 60
device = "cuda" if torch.cuda.is_available() else "cpu"


def cluster_acc(true, pred):
    """Hungarian-matched clustering accuracy."""
    true, pred = np.asarray(true), np.asarray(pred)
    D = max(pred.max(), true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for t, p in zip(true, pred):
        w[p, t] += 1
    row, col = linear_sum_assignment(-w)
    return w[row, col].sum() / len(true)


def _nearest_comp_logdensity(z, means, covs):
    d = z.shape[1]
    diff2 = (z[:, None, :] - means[None]) ** 2
    log_comp = -0.5 * (d * LOG2PI + np.sum(np.log(covs)[None] + diff2 / covs[None], axis=2))
    return log_comp.max(1), log_comp.argmax(1)


def decompose_plain(pv, x):
    with torch.no_grad():
        xt = torch.as_tensor(x, dtype=torch.float32, device=device)
        mu, _ = pv.encode(xt)
        recon = F.mse_loss(pv.decoder(mu), xt, reduction="none").sum(1).cpu().numpy()
        z = mu.cpu().numpy()
    log_near, assign = _nearest_comp_logdensity(z, pv.gmm.means_, pv.gmm.covariances_)
    return recon, -log_near, assign


def decompose_vade(vade, x):
    with torch.no_grad():
        xt = torch.as_tensor(x, dtype=torch.float32, device=device)
        mu, _ = vade.encode(xt)
        recon = F.mse_loss(vade.decode(mu), xt, reduction="none").sum(1).cpu().numpy()
        comp = vade._log_pz_given_c(mu)              # (N,K) component log-density
        log_near = comp.max(1).values.cpu().numpy()
        assign = comp.argmax(1).cpu().numpy()
    return recon, -log_near, assign


def auroc(y, s):
    return roc_auc_score(y, s)


def report(name, recon, dens, ds):
    y = ds.y_test
    combined = recon + dens
    normal = y == 0
    ood = ds.atype_test == "ood"
    pkt = ds.atype_test == "pocket"
    print(f"\n[{name}]")
    print(f"  AUROC  recon-only={auroc(y, recon):.3f}  "
          f"density-only={auroc(y, dens):.3f}  combined={auroc(y, combined):.3f}")
    print(f"  OOD-vs-normal    recon={auroc(np.r_[np.zeros(normal.sum()), np.ones(ood.sum())], np.r_[recon[normal], recon[ood]]):.3f}  "
          f"dens={auroc(np.r_[np.zeros(normal.sum()), np.ones(ood.sum())], np.r_[dens[normal], dens[ood]]):.3f}")
    print(f"  POCKET-vs-normal recon={auroc(np.r_[np.zeros(normal.sum()), np.ones(pkt.sum())], np.r_[recon[normal], recon[pkt]]):.3f}  "
          f"dens={auroc(np.r_[np.zeros(normal.sum()), np.ones(pkt.sum())], np.r_[dens[normal], dens[pkt]]):.3f}")
    return combined


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    ds, meta = make_miim(n_modes=N_MODES, seed=SEED)
    print(f"data: train={len(ds.x_train)} test={len(ds.x_test)} "
          f"modes={N_MODES} rare={len(meta['rare_modes'])} "
          f"rarest_train={int(meta['train_counts'].min())}")

    print("\ntraining sequential (plain VAE + GMM) ...")
    pv = train_plain_vae(ds.x_train, latent_dim=LATENT, epochs=EPOCHS, seed=SEED, device=device)
    pv.fit_gmm(ds.x_train, n_clusters=N_MODES, seed=SEED)

    print("training joint VaDE ...")
    vade = train_vade(ds.x_train, n_clusters=N_MODES, latent_dim=LATENT,
                      epochs=EPOCHS, seed=SEED, device=device)

    # ---- reconstruction quality on normal train ----
    rec_pv, _, asg_pv = decompose_plain(pv, ds.x_train)
    rec_vd, _, asg_vd = decompose_vade(vade, ds.x_train)
    print(f"\nreconstruction MSE on train normal:  sequential={rec_pv.mean():.3f}  "
          f"joint={rec_vd.mean():.3f}")

    # ---- clustering vs ground-truth modes ----
    print(f"clustering vs true modes (train):")
    print(f"  sequential GMM : NMI={normalized_mutual_info_score(ds.mode_train, asg_pv):.3f}  "
          f"ACC={cluster_acc(ds.mode_train, asg_pv):.3f}")
    print(f"  joint VaDE     : NMI={normalized_mutual_info_score(ds.mode_train, asg_vd):.3f}  "
          f"ACC={cluster_acc(ds.mode_train, asg_vd):.3f}")
    # how many clusters actually used?
    print(f"  clusters used  : sequential={len(np.unique(asg_pv))}/{N_MODES}  "
          f"joint={len(np.unique(asg_vd))}/{N_MODES}")

    # ---- score decomposition on test ----
    r_pv, d_pv, _ = decompose_plain(pv, ds.x_test)
    r_vd, d_vd, _ = decompose_vade(vade, ds.x_test)
    report("sequential VAE+GMM", r_pv, d_pv, ds)
    report("joint VaDE", r_vd, d_vd, ds)


if __name__ == "__main__":
    main()
