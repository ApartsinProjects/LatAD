"""Explore Model C: FiLM (condition) vs the user's proposal - a VaDE over the
CONCATENATION [window (+) context], so the mixture clusters represent the JOINT
(content, current-point) and a pure-trajectory fault (bad_transition) falls off
every joint cluster (high latent-NLL).

Caches A+B once; then trains window-only vs joint-concat VaDE and prints per-type
TPR@5%FPR for the full base score and for the latent-NLL term alone.
"""
from __future__ import annotations
import os, numpy as np, torch
from temporal_data import make_temporal_miim
import ldt_a, ldt_b
from models_vade import train_vade

CACHE = "datasets/miim/_abcache.npz"
TYPES = ["pocket", "drift", "bad_transition"]


def build_cache(seed=0, K=20, latent=8, device="cpu"):
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        print(f"[cache] loaded  xtr={z['xtr'].shape} c={z['c_tr'].shape}")
        return {k: z[k] for k in z.files}
    d = make_temporal_miim(n_modes=K, n_features=24, seed=seed,
                           n_train=120000, n_test=150000, W=40, stride=20)
    xtr, xte = d["x_train"], d["x_test"]
    print(f"[A] training mode encoder ({len(xtr)} windows)...")
    A = ldt_a.ModeEncoderA.fit(xtr, n_clusters=K, latent_dim=latent,
                               pretrain_epochs=15, epochs=30, warmup=8,
                               seed=seed, device=device, verbose=False)
    g_tr, _ = A.encode(xtr); g_te, _ = A.encode(xte)
    print(f"[B] training trajectory encoder (60 ep, focused offsets)...")
    B = ldt_b.TrajectoryEncoderB(K=K, emb_dim=48, ctx_dim=96, n_layers=2,
                                 centroids=A.centroids(), backbone="gru")
    ldt_b.train_B(B, g_tr, A.pi, epochs=60, seg_len=512, stride=128, batch_segs=16,
                  lr=3e-3, max_train_offset=64, device=device, seed=seed)
    probe = ldt_b.memory_horizon_probe(B, g_tr, device=device)
    print("    probe: " + "  ".join(f"k={k}:{a:.2f}" for k, a in probe.items() if k <= 16))
    c_tr = ldt_b.emit_context(B, g_tr, device=device)
    c_te = ldt_b.emit_context(B, g_te, device=device)
    out = dict(xtr=xtr, xte=xte, yte=d["y_test"], atype=d["atype_test"].astype(str),
               mode_tr=d["mode_train"], c_tr=c_tr, c_te=c_te)
    os.makedirs("datasets/miim", exist_ok=True)
    np.savez(CACHE, **out)
    print("[cache] saved")
    return out


def tpr_at(sn, sp, fpr=0.05):
    if len(sp) == 0:
        return float("nan")
    return float((sp > np.quantile(sn, 1 - fpr)).mean())


def scores(model, X):
    """Return (base = recon+latentNLL, latent_nll_only)."""
    dev = next(model.parameters()).device
    xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    with torch.no_grad():
        mu, _ = model.encode(xt)
        lnll = -model._log_pz_given_c(mu).max(1).values.cpu().numpy()
    base = model.anomaly_score(X)
    return base, lnll


def eval_variant(name, Xtr, Xte, atype, yte, K, latent, seed, device):
    v = train_vade(Xtr, n_clusters=K, latent_dim=latent, epochs=40, warmup=8,
                   seed=seed, device=device)
    v.fit_residual_whitener(Xtr)
    base, lnll = scores(v, Xte)
    normal = yte == 0
    row_b = {t: tpr_at(base[normal], base[atype == t]) for t in TYPES}
    row_l = {t: tpr_at(lnll[normal], lnll[atype == t]) for t in TYPES}
    print(f"\n[{name}]  input dim={Xtr.shape[1]}")
    print("   base (recon+latentNLL): " + "  ".join(f"{t}:{row_b[t]:.3f}" for t in TYPES))
    print("   latent-NLL only       : " + "  ".join(f"{t}:{row_l[t]:.3f}" for t in TYPES))
    return row_b, row_l


def mode_ctx_score(c_tr, assign_tr, c_te, assign_te, K, min_n=40):
    """Per-mode Ledoit-Wolf Mahalanobis of c_t: mode-specific arrival-context law
    p(c_t|mode), shrunk so it is robust. A forbidden arrival is off the mode's
    tight normal-context manifold -> high score. Pooling would wash this out."""
    from sklearn.covariance import LedoitWolf
    gm = c_tr.mean(0)
    pooled = LedoitWolf().fit(c_tr - gm)
    models = {}
    for k in range(K):
        ck = c_tr[assign_tr == k]
        if len(ck) >= min_n:
            models[k] = LedoitWolf().fit(ck)          # location + shrunk cov
    def sc(c, a):
        out = np.empty(len(c))
        for k in np.unique(a):
            idx = np.where(a == k)[0]
            mdl = models.get(int(k))
            out[idx] = mdl.mahalanobis(c[idx]) if mdl is not None else pooled.mahalanobis(c[idx] - gm)
        return out
    return sc(c_tr, assign_tr), sc(c_te, assign_te)


def soft_gamma(v, X):
    """Soft responsibilities gamma_t from the window VaDE (no argmax)."""
    import torch.nn.functional as F
    dev = next(v.parameters()).device
    with torch.no_grad():
        mu, _ = v.encode(torch.as_tensor(X, dtype=torch.float32, device=dev))
        logpcz = F.log_softmax(v.pi_logit, 0).unsqueeze(0) + v._log_pz_given_c(mu)
        return torch.softmax(logpcz, 1).cpu().numpy()


def soft_ctx_score(c_tr, g_tr, w_tr, c_te, g_te, shrink=0.2, ridge=1e-3):
    """Responsibility-weighted (soft) per-mode context density, fit with per-window
    rarity weights w_tr (soft coverage). No window is hard-assigned to a mode."""
    N, D = c_tr.shape; K = g_tr.shape[1]
    Wt = g_tr * w_tr[:, None]                          # (N,K) combined weights
    Wk = Wt.sum(0) + 1e-9
    mu = (Wt.T @ c_tr) / Wk[:, None]                   # (K,D) gamma-weighted means
    inv = np.empty((K, D, D))
    for k in range(K):
        d = c_tr - mu[k]
        cov = (Wt[:, k:k + 1] * d).T @ d / Wk[k]
        cov = (1 - shrink) * cov + shrink * np.trace(cov) / D * np.eye(D) + ridge * np.eye(D)
        inv[k] = np.linalg.inv(cov)

    def sc(c, g):
        out = np.zeros(len(c))
        for k in range(K):
            d = c - mu[k]
            out += g[:, k] * np.einsum("ij,jk,ik->i", d, inv[k], d)
        return out
    return sc(c_tr, g_tr), sc(c_te, g_te)


def soft_rarity_weights(g, lag=2, clip=8.0):
    """Per-window soft transition-rarity weight from responsibility outer-products.
    T[i,j] = sum_t g[t-lag,i] g[t,j] (fuzzy transition graph, lag clears window overlap)."""
    N, K = g.shape
    T = g[:-lag].T @ g[lag:]                            # (K,K) soft edge counts
    F = T / (T.sum() + 1e-9)
    r = np.ones(N)
    edge_mass = np.einsum("ti,tj->t", g[:-lag], g[lag:])          # per-t total soft mass ~1
    rare = np.einsum("ti,ij,tj->t", g[:-lag], 1.0 / (F + 1e-6), g[lag:])
    r[lag:] = np.clip(rare / (edge_mass + 1e-9), 0.2, clip)
    r /= r.mean()
    return r, T


def main():
    seed = 0; K = 20; latent = 8
    device = "cuda" if (torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "") else "cpu"
    C = build_cache(seed, K, latent, device)
    xtr, xte, atype, yte = C["xtr"], C["xte"], C["atype"], C["yte"]
    c_tr, c_te = C["c_tr"], C["c_te"]
    normal = yte == 0

    def std(a, ref):
        return (a - ref.mean(0)) / (ref.std(0) + 1e-8)
    xtr_s, xte_s = std(xtr, xtr).astype(np.float32), std(xte, xtr).astype(np.float32)

    # window VaDE + its cluster assignment (content mode)
    v = train_vade(xtr_s, n_clusters=K, latent_dim=latent, epochs=40, warmup=8,
                   seed=seed, device=device)
    v.fit_residual_whitener(xtr_s)
    sw_tr, sw_te = v.anomaly_score(xtr_s), v.anomaly_score(xte_s)
    dev = next(v.parameters()).device
    with torch.no_grad():
        atr = v._log_pz_given_c(v.encode(torch.as_tensor(xtr_s, device=dev))[0]).argmax(1).cpu().numpy()
        ate = v._log_pz_given_c(v.encode(torch.as_tensor(xte_s, device=dev))[0]).argmax(1).cpu().numpy()

    # mode-conditional context branch (robust, whitened). PCA-reduce c_t first to cut the
    # heavy-tail FPR cost (chi^2_96 -> chi^2_m), keeping the top transition directions.
    from sklearn.decomposition import PCA
    pca = PCA(n_components=24, random_state=0).fit(c_tr)
    cr_tr, cr_te = pca.transform(c_tr), pca.transform(c_te)
    # soft responsibilities from the window VaDE (no argmax); soft rarity weights
    g_tr, g_te = soft_gamma(v, xtr_s), soft_gamma(v, xte_s)
    r_tr, T = soft_rarity_weights(g_tr, lag=2)
    ones = np.ones(len(g_tr))
    contexts = {
        "hard-argmax LW":       mode_ctx_score(cr_tr, atr, cr_te, ate, K),
        "soft gamma-weighted":  soft_ctx_score(cr_tr, g_tr, ones, cr_te, g_te),
        "soft + rarity-weight": soft_ctx_score(cr_tr, g_tr, r_tr, cr_te, g_te),
    }
    thrw = np.quantile(sw_te[normal], 0.95)
    win_row = {t: float((sw_te[atype == t] > thrw).mean()) for t in TYPES}

    def evalctx(name, sc_te):
        print(f"\n--- context: {name} ---")
        print(f"{'combination':<26}" + "".join(f"{t:>15}" for t in TYPES) + f"{'FPR':>7}")
        print(f"{'window-only (C-alone)':<26}" + "".join(f"{win_row[t]:>15.3f}" for t in TYPES)
              + f"{float((sw_te[normal] > thrw).mean()):>7.3f}")
        for qw, qc in [(0.955, 0.995), (0.96, 0.99), (0.965, 0.985), (0.97, 0.98)]:
            tw = np.quantile(sw_te[normal], qw); tc = np.quantile(sc_te[normal], qc)
            flag = (sw_te > tw) | (sc_te > tc)
            row = {t: float(flag[atype == t].mean()) for t in TYPES}
            print(f"{f'OR win@{1-qw:.3f} ctx@{1-qc:.3f}':<26}"
                  + "".join(f"{row[t]:>15.3f}" for t in TYPES)
                  + f"{float(flag[normal].mean()):>7.3f}")

    print("\n=== window + mode-conditional context (no FiLM), exact 5% FPR ===")
    for name, (_, sc_te) in contexts.items():
        evalctx(name, sc_te)
    print("\nfloor bad_transition (snapshot baseline) = 0.057")


if __name__ == "__main__":
    main()
