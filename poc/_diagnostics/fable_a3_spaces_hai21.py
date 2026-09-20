"""H1b: HAI regime-21 (test-normal block, test windows 6436..7081, 0.4% train mass). Is it MERGED into a
big component at the paper's K=40, and if it is forced to be its own cluster, does between-overlap
(ambiguous train mass) appear? Uses the cached paper-config latent e2_fable_HAI.npz (LD16, K40).

Invariants stated before running:
  B1 the block's test windows are >0.95 in a single VaDE component (they form one regime).
  B2 if merged: that component's train mass >> 0.4% (a big train regime absorbed it). If not merged:
     train mass ~0.4%, i.e. the paper's K already gives it its own component.
  B3 forcing a dedicated component (mean/var from the block) changes train H_norm by < 0.01 and makes
     < 1% of train windows ambiguous toward it, if the block is separated from train mass; a rise of
     > 0.05 in H_norm would mean the block sits in an overlap region with a train regime (hidden A3).
  B4 coarser GMMs (K=8,16) on the same latent merge the block into bigger components (report mass).
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fable_a3_spaces_lib import H_norm, HERE

LOG2PI = np.log(2 * np.pi)
e = np.load(os.path.join(HERE, "e2_fable_HAI.npz"))
ztr, zte, mu_c, lvc, logpi, yw = e["ztr"], e["zte"], e["mu_c"], e["lvc"], e["logpi"], e["yw"]
Xtr, Xte = e["Xtr_s"], e["Xte_s"]
K, LD = len(logpi), ztr.shape[1]
blk = np.zeros(len(zte), bool); blk[6436:7082] = True
assert (yw[blk] == 0).all(), "block must be test-normal"
tn = (yw == 0) & ~blk
R = {}


def resp(z, mu, lv, lp):
    logN = -0.5 * (LOG2PI * z.shape[1] + (lv[None] + (z[:, None, :] - mu[None]) ** 2 / np.exp(lv[None])).sum(2))
    l = lp[None] + logN
    return np.exp(l - logsumexp(l, 1, keepdims=True))


# ---- 1. paper model (K=40): where does the block land? ----
Gtr, Gte = resp(ztr, mu_c, lvc, logpi), resp(zte, mu_c, lvc, logpi)
ltr, lte = Gtr.argmax(1), Gte.argmax(1)
ptr = np.bincount(ltr, minlength=K) / len(ltr)
cb = np.bincount(lte[blk], minlength=K) / blk.sum()
top = np.argsort(-cb)[:3]
R["paper_K40"] = dict(block_n=int(blk.sum()), block_top_components=[dict(c=int(c), block_share=float(cb[c]), train_mass=float(ptr[c])) for c in top],
                      H_block=float(H_norm(Gte[blk]).mean()), H_test_normal_outside=float(H_norm(Gte[tn]).mean()),
                      H_train=float(H_norm(Gtr).mean()), rho_block=float((Gte[blk].max(1) < 0.5).mean()),
                      H_train_in_block_components=float(H_norm(Gtr[np.isin(ltr, top[:1])]).mean()) if ptr[top[0]] > 0 else None,
                      n_train_in_top_component=int((ltr == top[0]).sum()))
print("paper K40:", json.dumps(R["paper_K40"]), flush=True)

# distance structure: block vs train in latent and in full feature space
for nm, A, B in (("latent", ztr, zte), ("full_feat", Xtr.astype(np.float64), Xte.astype(np.float64))):
    nn = NearestNeighbors(n_neighbors=2, algorithm="brute", n_jobs=-1).fit(A)
    d_loo = nn.kneighbors(A)[0][:, 1]; d_blk = nn.kneighbors(B[blk])[0][:, 0]; d_tn = nn.kneighbors(B[tn])[0][:, 0]
    R[f"nn_dist_{nm}"] = dict(train_loo_med=float(np.median(d_loo)), train_loo_q99=float(np.quantile(d_loo, .99)),
                              block_med=float(np.median(d_blk)), block_frac_gt_q99=float((d_blk > np.quantile(d_loo, .99)).mean()),
                              test_normal_outside_med=float(np.median(d_tn)), tn_frac_gt_q99=float((d_tn > np.quantile(d_loo, .99)).mean()))
    print(nm, R[f"nn_dist_{nm}"], flush=True)

# ---- 2. coarser GMMs on the same latent: does the block get merged? ----
R["gmm_latent"] = {}
for k in (8, 16, 40):
    g = GaussianMixture(k, covariance_type="diag", reg_covar=1e-4, random_state=0, n_init=2).fit(ztr)
    G1, G2 = g.predict_proba(ztr), g.predict_proba(zte)
    l1, l2 = G1.argmax(1), G2.argmax(1); p1 = np.bincount(l1, minlength=k) / len(l1)
    c2 = np.bincount(l2[blk], minlength=k) / blk.sum(); t = int(np.argmax(c2))
    R["gmm_latent"][f"K{k}"] = dict(block_top_component=t, block_share=float(c2[t]), train_mass_of_that_component=float(p1[t]),
                                    H_train=float(H_norm(G1).mean()), H_block=float(H_norm(G2[blk]).mean()),
                                    H_train_in_that_component=float(H_norm(G1[l1 == t]).mean()) if (l1 == t).any() else None)
    print(f"gmm K{k}:", R["gmm_latent"][f"K{k}"], flush=True)

# ---- 3. force the block to be its own component (paper K=40 + 1, and GMM K=8 + 1) ----
mb, vb = zte[blk].mean(0), zte[blk].var(0) + 1e-3
R["forced_component"] = {}
for w in (0.004, 0.063):
    mu2 = np.vstack([mu_c, mb]); lv2 = np.vstack([lvc, np.log(vb)])
    lp2 = np.log(np.r_[np.exp(logpi) * (1 - w), w])
    G2 = resp(ztr, mu2, lv2, lp2); H2 = H_norm(G2)
    amb_new = (G2[:, -1] > 0.2) & (G2[:, -1] < 0.8)
    R["forced_component"][f"K41_w{w}"] = dict(H_train_before=float(H_norm(Gtr).mean()), H_train_after=float(H2.mean()),
                                              rho_before=float((Gtr.max(1) < 0.5).mean()), rho_after=float((G2.max(1) < 0.5).mean()),
                                              train_frac_ambiguous_toward_new=float(amb_new.mean()),
                                              train_frac_assigned_new=float((G2.argmax(1) == K).mean()),
                                              block_frac_assigned_new=float((resp(zte[blk], mu2, lv2, lp2).argmax(1) == K).mean()))
    print(f"forced K41 w={w}:", R["forced_component"][f"K41_w{w}"], flush=True)
g8 = GaussianMixture(8, covariance_type="diag", reg_covar=1e-4, random_state=0, n_init=2).fit(ztr)
mu9 = np.vstack([g8.means_, mb]); lv9 = np.log(np.vstack([g8.covariances_, vb])); lp9 = np.log(np.r_[g8.weights_ * (1 - 0.063), 0.063])
G9 = resp(ztr, mu9, lv9, lp9); G8 = g8.predict_proba(ztr)
R["forced_component"]["K8plus1"] = dict(H_train_before=float(H_norm(G8).mean()), H_train_after=float(H_norm(G9).mean()),
                                        train_frac_ambiguous_toward_new=float(((G9[:, -1] > 0.2) & (G9[:, -1] < 0.8)).mean()),
                                        train_frac_assigned_new=float((G9.argmax(1) == 8).mean()))
print("forced K8+1:", R["forced_component"]["K8plus1"], flush=True)
json.dump(R, open(os.path.join(HERE, "fable_a3_spaces_hai21.json"), "w"), indent=1)
print("done", flush=True)
