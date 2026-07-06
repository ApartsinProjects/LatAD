"""Port the A->B->C soft-context exploration onto the NEW unified generator (miim_gen),
whose two-tier graph HAS a rare-valid-edge tail -> the place the coverage/rarity lever
should finally bite. Compares hard-argmax vs soft vs soft+rarity context on bad_transition.
"""
from __future__ import annotations
import os, numpy as np, torch
import ldt_a, ldt_b
from models_vade import train_vade
from miim_gen import make_dataset
import explore_c as E

CACHE = "datasets/miim/_abcache_miim.npz"
TYPES = ["pocket", "drift", "bad_transition"]


def build(seed=0, K=64, latent=8, device="cpu"):
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        print(f"[cache] loaded xtr={z['xtr'].shape}")
        return {k: z[k] for k in z.files}
    d = make_dataset(seed=seed, n_train=300000, n_test=300000, W=40, stride=20)
    xtr, xte = d["x_train"], d["x_test"]
    at = d["atype_test"]
    print(f"[data] miim: K={d['K']} train={len(xtr)} test={len(xte)} "
          + " ".join(f"{t}={int((at==t).sum())}" for t in TYPES))
    print("[A] training mode encoder...")
    A = ldt_a.ModeEncoderA.fit(xtr, n_clusters=K, latent_dim=latent, pretrain_epochs=15,
                               epochs=30, warmup=8, seed=seed, device=device, verbose=False)
    g_tr, _ = A.encode(xtr); g_te, _ = A.encode(xte)
    print(f"    active modes={(A.pi>0.005).sum()}/{K}")
    print("[B] training trajectory encoder (60 ep)...")
    B = ldt_b.TrajectoryEncoderB(K=K, emb_dim=64, ctx_dim=96, n_layers=2,
                                 centroids=A.centroids(), backbone="gru")
    ldt_b.train_B(B, g_tr, A.pi, epochs=60, seg_len=512, stride=128, batch_segs=16,
                  lr=3e-3, max_train_offset=64, device=device, seed=seed)
    probe = ldt_b.memory_horizon_probe(B, g_tr, device=device)
    print("    probe: " + "  ".join(f"k={k}:{a:.2f}" for k, a in probe.items() if k <= 16))
    out = dict(xtr=xtr, xte=xte, yte=d["y_test"], atype=at.astype(str),
               c_tr=ldt_b.emit_context(B, g_tr, device=device),
               c_te=ldt_b.emit_context(B, g_te, device=device))
    os.makedirs("datasets/miim", exist_ok=True); np.savez(CACHE, **out)
    print("[cache] saved")
    return out


def main():
    seed = 0; K = 64; latent = 8
    device = "cuda" if (torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "") else "cpu"
    C = build(seed, K, latent, device)
    xtr, xte, atype, yte = C["xtr"], C["xte"], C["atype"], C["yte"]
    c_tr, c_te = C["c_tr"], C["c_te"]; normal = yte == 0

    def std(a, r): return ((a - r.mean(0)) / (r.std(0) + 1e-8)).astype(np.float32)
    xtr_s, xte_s = std(xtr, xtr), std(xte, xtr)
    v = train_vade(xtr_s, n_clusters=K, latent_dim=latent, epochs=40, warmup=8, seed=seed, device=device)
    v.fit_residual_whitener(xtr_s)
    sw_te = v.anomaly_score(xte_s)
    dev = next(v.parameters()).device
    with torch.no_grad():
        atr = v._log_pz_given_c(v.encode(torch.as_tensor(xtr_s, device=dev))[0]).argmax(1).cpu().numpy()
        ate = v._log_pz_given_c(v.encode(torch.as_tensor(xte_s, device=dev))[0]).argmax(1).cpu().numpy()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=24, random_state=0).fit(c_tr)
    cr_tr, cr_te = pca.transform(c_tr), pca.transform(c_te)
    g_tr, g_te = E.soft_gamma(v, xtr_s), E.soft_gamma(v, xte_s)
    r_tr, T = E.soft_rarity_weights(g_tr, lag=2)
    ones = np.ones(len(g_tr))
    contexts = {
        "hard-argmax LW":       E.mode_ctx_score(cr_tr, atr, cr_te, ate, K),
        "soft gamma-weighted":  E.soft_ctx_score(cr_tr, g_tr, ones, cr_te, g_te),
        "soft + rarity-weight": E.soft_ctx_score(cr_tr, g_tr, r_tr, cr_te, g_te),
    }
    thrw = np.quantile(sw_te[normal], 0.95)
    win = {t: float((sw_te[atype == t] > thrw).mean()) for t in TYPES}
    print(f"\nrarity weights: std={r_tr.std():.3f} (temporal was 0.016)")
    print("\n=== miim_gen: window + mode-conditional context, exact 5% FPR ===")
    for name, (_, sc_te) in contexts.items():
        print(f"\n--- context: {name} ---")
        print(f"{'combination':<26}" + "".join(f"{t:>15}" for t in TYPES) + f"{'FPR':>7}")
        print(f"{'window-only (C-alone)':<26}" + "".join(f"{win[t]:>15.3f}" for t in TYPES)
              + f"{float((sw_te[normal] > thrw).mean()):>7.3f}")
        for qw, qc in [(0.955, 0.995), (0.96, 0.99), (0.965, 0.985), (0.97, 0.98)]:
            tw = np.quantile(sw_te[normal], qw); tc = np.quantile(sc_te[normal], qc)
            flag = (sw_te > tw) | (sc_te > tc)
            row = {t: float(flag[atype == t].mean()) for t in TYPES}
            print(f"{f'OR win@{1-qw:.3f} ctx@{1-qc:.3f}':<26}"
                  + "".join(f"{row[t]:>15.3f}" for t in TYPES) + f"{float(flag[normal].mean()):>7.3f}")


if __name__ == "__main__":
    main()
