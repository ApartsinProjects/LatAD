"""Shared measures for the A3 space/K robustness check (report-only diagnostic).

overlap_in_space(Z, K): VaDE-free overlap read in an arbitrary representation Z.
  gmm_H    : mean normalised responsibility entropy of a diag GMM(K) fit on Z (H / log K)
  gmm_rho  : frac(max responsibility < 0.5)
  knn_amb  : LOO 10-NN share of same GMM-argmax label < 0.5   (dimension-robust twin of rho)
  knn_mix  : mean(1 - same-label share)
  (if true labels given) knn_amb_true / knn_mix_true with the TRUE labels, and ARI(gmm, true).

vade_overlap(X, K, LD): VaDE's own responsibility entropy plus the same kNN read in its latent.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def H_norm(G):
    K = G.shape[1]
    return -(G * np.log(G + 1e-12)).sum(1) / np.log(K)


def knn_share(Z, lab, k=10):
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=-1).fit(Z)
    _, idx = nn.kneighbors(Z)
    idx = idx[:, 1:]
    return (lab[idx] == lab[:, None]).mean(1)


def overlap_in_space(Z, K, seed=0, true_lab=None, kmeans_too=True):
    Z = np.asarray(Z, np.float64)
    g = GaussianMixture(K, covariance_type="diag", reg_covar=1e-4, random_state=seed, n_init=1).fit(Z)
    G = g.predict_proba(Z); lab = G.argmax(1)
    H = H_norm(G); mr = G.max(1)
    sh = knn_share(Z, lab)
    out = dict(dim=int(Z.shape[1]), K=int(K), gmm_H=float(H.mean()), gmm_rho=float((mr < 0.5).mean()),
               knn_amb=float((sh < 0.5).mean()), knn_mix=float((1 - sh).mean()),
               n_used_components=int((np.bincount(lab, minlength=K) > 0).sum()))
    if kmeans_too:
        km = KMeans(K, n_init=4, random_state=seed).fit(Z)
        sk = knn_share(Z, km.labels_)
        out["knn_amb_kmeans"] = float((sk < 0.5).mean()); out["knn_mix_kmeans"] = float((1 - sk).mean())
    if true_lab is not None:
        st = knn_share(Z, true_lab)
        out["knn_amb_true"] = float((st < 0.5).mean()); out["knn_mix_true"] = float((1 - st).mean())
        out["ari_gmm_true"] = float(adjusted_rand_score(true_lab, lab))
    return out


def vade_overlap(X, K, LD, seed=0, epochs=40, warmup=8, true_lab=None):
    import torch
    from models_vade import train_vade
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    v = train_vade(np.asarray(X, np.float32), n_clusters=K, latent_dim=LD, epochs=epochs, warmup=warmup,
                   seed=seed, device="cpu")
    G = v._responsibilities(X); lab = G.argmax(1)
    H = H_norm(G); mr = G.max(1)
    with torch.no_grad():
        z = v.encode(torch.as_tensor(np.asarray(X, np.float32)))[0].numpy()
    sh = knn_share(z, lab)
    out = dict(LD=int(LD), K=int(K), vade_H=float(H.mean()), vade_rho=float((mr < 0.5).mean()),
               knn_amb=float((sh < 0.5).mean()), knn_mix=float((1 - sh).mean()),
               n_used_components=int((np.bincount(lab, minlength=K) > 0).sum()))
    if true_lab is not None:
        st = knn_share(z, true_lab)
        out["knn_amb_true"] = float((st < 0.5).mean()); out["knn_mix_true"] = float((1 - st).mean())
        out["ari_vade_true"] = float(adjusted_rand_score(true_lab, lab))
    return out, z, G


def pca_spaces(X, ks, seed=0):
    X = np.asarray(X, np.float64); d = X.shape[1]
    full = min(ks[-1], d) if ks else d
    p = PCA(min(max(ks), d), random_state=seed).fit(X)
    Zall = p.transform(X)
    return {k: Zall[:, :k] for k in ks if k <= d}, p.explained_variance_ratio_


class JsonStore:
    def __init__(self, path):
        self.path = path
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.d, f, indent=1); f.flush(); os.fsync(f.fileno())
