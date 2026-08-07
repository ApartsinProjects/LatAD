"""Fix (a): CONSTRAINED reconstruction that cannot reconstruct off-manifold.
  flexible : ||x - decode(encode(x))||     (current recon; cheats on CPS correlation breaks)
  prototype: ||x - decode(mu_c*)||          (decode the NEAREST MODE MEAN -> can't cheat)
  snap     : ||x - decode(snap z to mu_c*)||
Hypothesis: prototype residual is high for ANY off-manifold window (CPS joint-break OR IT
outlier), so it may be the one reconstruction that helps CPS AND server-IT.
Also tests a better DENSITY: kNN distance in latent (non-parametric) vs the current GMM-NLL.
Difficult-subset AUROC, seeds 0-2, all 6 datasets.
"""
from __future__ import annotations
import os, sys, json, numpy as np, torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from models_vade import train_vade, _as_tensor
import eda_real as E

CFG = {"WADI": (20, 10), "HAI": (40, 16), "SKAB": (16, 6), "SWaT": (40, 16), "PSM": (40, 16), "SMD": (40, 16)}
SEEDS = [0, 1, 2]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics"); os.makedirs(OUT, exist_ok=True)


def au(y, s, m):
    k = (y == 0) | m
    return float(roc_auc_score(y[k], s[k])) if m.sum() >= 3 else float("nan")


def zt(s, ref): return (s - ref.mean()) / (ref.std() + 1e-9)


@torch.no_grad()
def decode_np(v, Z):
    return v.decode(torch.as_tensor(Z, dtype=torch.float32, device=v.mu_c.device)).cpu().numpy()


def run(name):
    D = E.load(name); Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"]
    C6 = Xte0.shape[1] // 6; triv = np.abs(Xte0[:, :C6]).max(1); trn = np.abs(Xtr0[:, :C6]).max(1)
    hard = (y == 1) & ~(triv > np.quantile(trn, 0.99))
    K, LD = CFG[name]; mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
    R = {k: [] for k in ["density", "knn_latent", "flexible", "prototype", "snap"]}
    for seed in SEEDS:
        v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
        kd = min(80, max(20, len(Xtr) // 10)); v.fit_latent_density(Xtr, k_density=kd); v.fit_residual_whitener(Xtr)
        Zte = v._encode_mean(Xte); Ztr = v._encode_mean(Xtr)
        muc = v.mu_c.detach().cpu().numpy()                            # (K, d)
        cst_te = ((Zte[:, None, :] - muc[None]) ** 2).sum(-1).argmin(1)
        cst_tr = ((Ztr[:, None, :] - muc[None]) ** 2).sum(-1).argmin(1)
        # signals (z-normalised on train)
        dte, _ = v._hard_components(Xte); dtr, _ = v._hard_components(Xtr)
        R["density"].append(au(y, zt(dte, dtr), hard))
        nn = NearestNeighbors(n_neighbors=5).fit(Ztr)
        kte = nn.kneighbors(Zte)[0][:, -1]; ktr = nn.kneighbors(Ztr)[0][:, -1]
        R["knn_latent"].append(au(y, zt(kte, ktr), hard))
        fte = ((Xte - decode_np(v, Zte)) ** 2).sum(1); ftr = ((Xtr - decode_np(v, Ztr)) ** 2).sum(1)
        R["flexible"].append(au(y, zt(fte, ftr), hard))
        pte = ((Xte - decode_np(v, muc[cst_te])) ** 2).sum(1); ptr = ((Xtr - decode_np(v, muc[cst_tr])) ** 2).sum(1)
        R["prototype"].append(au(y, zt(pte, ptr), hard))
        ste = ((Xte - decode_np(v, muc[cst_te])) ** 2).sum(1)  # snap==prototype for hard assign; keep for clarity
        R["snap"].append(au(y, zt(ste, ptr), hard))
    r = {"dataset": name, **{k: round(float(np.nanmean(v)), 3) for k, v in R.items()}}
    json.dump(r, open(f"{OUT}/proto_recon_{name}.json", "w"), indent=1)
    print(f"{name:5} density={r['density']:.3f} kNN={r['knn_latent']:.3f} flex_recon={r['flexible']:.3f} "
          f"PROTOTYPE={r['prototype']:.3f}", flush=True)


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["PSM", "SWaT", "SMD", "WADI", "HAI", "SKAB"]):
        run(nm)
