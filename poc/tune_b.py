"""Fast iteration harness for Model B: train A once (cached), then train B under a
given config and print the memory-horizon probe. Goal: probe >~0.7 at k<=8.

Usage: /c/Python314/python tune_b.py <emb_dim> <ctx_dim> <n_layers> <epochs> <lr> <backbone>
"""
from __future__ import annotations
import sys, os
import numpy as np, torch
from temporal_data import make_temporal_miim
import ldt_a, ldt_b

CACHE = "datasets/miim/_Acache.npz"


def get_A(seed=0, K=20, latent=8, n_train=120000, device="cpu"):
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if int(z["K"]) == K and int(z["seed"]) == seed:
            print(f"[A] loaded cache: g_tr {z['g_tr'].shape}")
            return z["g_tr"], z["pi"], z["centroids"]
    d = make_temporal_miim(n_modes=K, n_features=24, seed=seed,
                           n_train=n_train, n_test=100, W=40, stride=20)
    print(f"[A] training mode encoder on {len(d['x_train'])} windows...")
    A = ldt_a.ModeEncoderA.fit(d["x_train"], n_clusters=K, latent_dim=latent,
                               pretrain_epochs=15, epochs=30, warmup=8,
                               seed=seed, device=device, verbose=False)
    g_tr, _ = A.encode(d["x_train"])
    os.makedirs("datasets/miim", exist_ok=True)
    np.savez(CACHE, g_tr=g_tr, pi=A.pi, centroids=A.centroids(), K=K, seed=seed)
    print(f"[A] cached. active modes={(A.pi>0.005).sum()}/{K}")
    return g_tr, A.pi, A.centroids()


def main():
    emb = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    nl = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    ep = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    lr = float(sys.argv[5]) if len(sys.argv) > 5 else 2e-3
    bk = sys.argv[6] if len(sys.argv) > 6 else "gru"
    device = "cuda" if (torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "") else "cpu"
    print(f"cfg: emb={emb} ctx={ctx} layers={nl} epochs={ep} lr={lr} backbone={bk} device={device}")

    g_tr, pi, cents = get_A(device=device)
    K = g_tr.shape[1]
    B = ldt_b.TrajectoryEncoderB(K=K, emb_dim=emb, ctx_dim=ctx, centroids=cents,
                                 backbone=bk, n_layers=nl)
    ldt_b.train_B(B, g_tr, pi, epochs=ep, seg_len=512, stride=128,
                  batch_segs=16, lr=lr, device=device, verbose=True, seed=0)
    probe = ldt_b.memory_horizon_probe(B, g_tr, device=device)
    near = np.mean([probe[k] for k in probe if k <= 8])
    print("\nprobe:  " + "  ".join(f"k={k}:{a:.2f}" for k, a in probe.items()))
    print(f"MEAN(k<=8) = {near:.3f}   pi_max(chance)={pi.max():.3f}")
    print("PASS" if near > 0.7 else "FAIL (need k<=8 mean > 0.70)")


if __name__ == "__main__":
    main()
