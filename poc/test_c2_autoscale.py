"""Auto-scale the C2 noise-agreement rescue from TRAINING data so it can't backfire.
Gate/scale on train-normal mean max-responsibility (crisp modes -> no rescue; overlapping modes ->
rescue): lam_eff = base_lam * max(0, thr - maxresp_train). WADI 0.94 / HAI 0.89 -> ~0 (no-op),
SKAB 0.57 -> meaningful. Verify it lifts SKAB without touching WADI/HAI.
"""
from __future__ import annotations
import numpy as np, torch
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from models_vade import train_vade, _as_tensor
from winfeat import window_features
import eda_real as E

W, ST = 60, 30
def win(X, y=None):
    A, B = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], "stats"))
        if y is not None: B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)

def au(y, s, mask): k = (y == 0) | mask; return roc_auc_score(y[k], s[k])
def z(s, r): return (s - r.mean()) / (r.std() + 1e-9)

@torch.no_grad()
def maxresp_and_agree(v, Xs, zstd, R=16, frac=0.5, seed=0):
    zt = v.encode(_as_tensor(Xs, v))[0]
    lN = v._log_pz_given_c(zt).cpu().numpy()
    lp = torch.log_softmax(v.pi_logit, 0).cpu().numpy()[None] + lN
    G = np.exp(lp - logsumexp(lp, 1, keepdims=True)); maxr = G.max(1)
    base = lN.argmax(1)
    g = torch.Generator(device=zt.device).manual_seed(seed)
    sig = torch.as_tensor(frac * zstd, dtype=zt.dtype, device=zt.device)
    agree = np.zeros(len(Xs))
    for _ in range(R):
        zp = zt + torch.randn(zt.shape, generator=g, device=zt.device, dtype=zt.dtype) * sig
        agree += (v._log_pz_given_c(zp).argmax(1).cpu().numpy() == base)
    return maxr, agree / R

# Gate calibrated purely on training normals: the RATIO of train-normal windows that are
# 'between modes' (max responsibility < 0.5). Crisp-mode data (WADI/HAI) -> ~0 -> C2 off;
# overlapping-mode data (SKAB) -> sizeable -> C2 on. No magic score threshold.
AMB_LEVEL, BASE_LAM, DEADZONE = 0.5, 2.5, 0.15

def main(name, K, seed=0):
    D = E.load(name); Xtr, _ = win(D["Xn_raw"]); Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = ((Xtr - m) / sd).astype(np.float32), ((Xte - m) / sd).astype(np.float32)
    C6 = Xte.shape[1] // 6; triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy

    v = train_vade(Xtr_s, n_clusters=K, latent_dim=10, epochs=40, warmup=8, seed=seed)
    v.fit_residual_whitener(Xtr_s); v.fit_latent_density(Xtr_s, k_density=80); v.fit_resid_head(Xtr_s)
    base_te = v.anomaly_score_hard(Xte_s, use_resid="auto"); base_tr = v.anomaly_score_hard(Xtr_s, use_resid="auto")
    zstd = v._encode_mean(Xtr_s).std(0)
    mr_tr, ag_tr = maxresp_and_agree(v, Xtr_s, zstd); _, ag_te = maxresp_and_agree(v, Xte_s, zstd)

    frac_amb = float((mr_tr < AMB_LEVEL).mean())                 # ratio of train-normal between modes
    lam_eff = BASE_LAM * max(0.0, frac_amb - DEADZONE)           # train-only, no score-threshold constant
    rescued = z(base_te, base_tr) - lam_eff * z(ag_te, ag_tr)
    print(f"{name:5} train frac_ambiguous(maxresp<0.5)={frac_amb:.2f}  lam_eff={lam_eff:.2f}   "
          f"| base HARD {au(yw,base_te,hard):.3f} -> auto {au(yw,rescued,hard):.3f}   "
          f"ALL {au(yw,base_te,yw==1):.3f}->{au(yw,rescued,yw==1):.3f}  "
          f"EASY {au(yw,base_te,easy):.3f}->{au(yw,rescued,easy):.3f}")


if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI", "HAI", "SKAB"]):
        main(nm, {"WADI": 20, "HAI": 24, "SKAB": 16}.get(nm, 20))
