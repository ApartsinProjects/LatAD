"""Implement + test the two FP/FN-targeted variants:
  C2 noise-agreement RESCUE  (FP fix): perturb z with Gaussian noise R times; 'agreement' =
      fraction of copies whose argmax mode == the clean argmax. A rare-but-VALID point sits deep
      in one basin -> high agreement -> DEMOTE its anomaly score. Rescues the low-density FPs.
  C3 off-manifold ratio-gap  (FN fix): per-mode input-space PCA experts (ClusterEnsemble).
      min recon err = off-manifold energy; RATIO gap = 2nd-min/min (scale-free) vs the current
      DIFFERENCE gap. Target the in-mode FNs a single mode still explains.
Report ALL/EASY/HARD AUROC + FP count and hard-catch at the 95%-train threshold.
"""
from __future__ import annotations
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from models_vade import train_vade, _as_tensor
from winfeat import window_features
import eda_real as E
import component3 as C3

W, ST = 60, 30
def win(X, y=None):
    A, B = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], "stats"))
        if y is not None: B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)

def au(y, s, mask): k = (y == 0) | mask; return roc_auc_score(y[k], s[k])
def z(s, ref): return (s - ref.mean()) / (ref.std() + 1e-9)

@torch.no_grad()
def noise_agreement(v, Xs, zstd, R=16, frac=0.5, seed=0):
    zt = v.encode(_as_tensor(Xs, v))[0]
    base = v._log_pz_given_c(zt).argmax(1).cpu().numpy()
    g = torch.Generator(device=zt.device).manual_seed(seed)
    sig = torch.as_tensor(frac * zstd, dtype=zt.dtype, device=zt.device)
    agree = np.zeros(len(Xs))
    for _ in range(R):
        zp = zt + torch.randn(zt.shape, generator=g, device=zt.device, dtype=zt.dtype) * sig
        agree += (v._log_pz_given_c(zp).argmax(1).cpu().numpy() == base)
    return agree / R

def main(name, K, seed=0):
    D = E.load(name); Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = ((Xtr - m) / sd).astype(np.float32), ((Xte - m) / sd).astype(np.float32)
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy

    v = train_vade(Xtr_s, n_clusters=K, latent_dim=10, epochs=40, warmup=8, seed=seed)
    v.fit_residual_whitener(Xtr_s); v.fit_latent_density(Xtr_s, k_density=80); v.fit_resid_head(Xtr_s)
    base_te = v.anomaly_score_hard(Xte_s, use_resid="auto")
    base_tr = v.anomaly_score_hard(Xtr_s, use_resid="auto")
    thr = np.quantile(base_tr, 0.95)
    def report(tag, s_te, s_tr):
        t = np.quantile(s_tr, 0.95)
        fp = int(((yw == 0) & (s_te > t)).sum()); hc = int((hard & (s_te > t)).sum())
        print(f"{tag:<26}{au(yw,s_te,yw==1):>6.3f}{au(yw,s_te,easy):>7.3f}{au(yw,s_te,hard):>7.3f}"
              f"{fp:>7}{hc:>6}/{int(hard.sum())}")

    zstd = v._encode_mean(Xtr_s).std(0)
    ag_te = noise_agreement(v, Xte_s, zstd); ag_tr = noise_agreement(v, Xtr_s, zstd)

    ens = C3.ClusterEnsemble(v, Xtr_s, latent_dim=10)
    Mte, Mtr = ens.expert_matrix(Xte_s), ens.expert_matrix(Xtr_s)
    def feats(M):
        S = np.sort(M, 1); mn = S[:, 0] + 1e-9
        return mn, (S[:, 1] - S[:, 0]), (S[:, 1] / mn)          # min(off-manifold), diff-gap, ratio-gap
    mn_te, dgap_te, rgap_te = feats(Mte); mn_tr, dgap_tr, rgap_tr = feats(Mtr)

    print(f"\n##### {name} (easy={int(easy.sum())} hard={int(hard.sum())}) #####")
    print(f"{'variant':<26}{'ALL':>6}{'EASY':>7}{'HARD':>7}{'FP':>7}{'hardCatch':>9}")
    report("VaDE-hard (base)", base_te, base_tr)
    for lam in [0.5, 1.0, 2.0]:                                 # C2 rescue: subtract agreement
        report(f"+C2 rescue lam={lam}", z(base_te, base_tr) - lam * z(ag_te, ag_tr), z(base_tr, base_tr) - lam * z(ag_tr, ag_tr))
    report("C3 off-manifold min", mn_te, mn_tr)
    report("C3 ratio-gap", rgap_te, rgap_tr)
    report("+C3 ratio-gap (fused)", z(base_te, base_tr) + z(rgap_te, rgap_tr), z(base_tr, base_tr) + z(rgap_tr, rgap_tr))
    report("+C3 diff-gap (fused)", z(base_te, base_tr) + z(dgap_te, dgap_tr), z(base_tr, base_tr) + z(dgap_tr, dgap_tr))


if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI"]):
        main(nm, {"WADI": 20, "HAI": 24, "SKAB": 16}.get(nm, 20))
