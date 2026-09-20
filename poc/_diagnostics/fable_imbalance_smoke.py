"""Feasibility smoke for the induced-imbalance design on SKAB (1 seed, single thread).
Checks: (1) external regime split (k-means on PCA of train windows, K by silhouette, train only);
(2) beta sampler: share_c(beta) ∝ p_c^(1+beta), hold N constant by upsample-with-replacement of
common regimes; invariants beta=0 -> identity multiset, monotone rare share; (3) IF / AE / VaDE
FPR on test-normals of the starved regime vs beta; beta=0 must equal the plain baseline.
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
POC = "E:/Projects/Backlog/LatAD/poc"; sys.path.insert(0, POC); os.chdir(POC)
import numpy as np, torch
torch.set_num_threads(1)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.ensemble import IsolationForest
import eda_real as E
from models_vade import train_vade
from compare_baselines import ae_scores

SEED = 0
BETAS = [0.0, 0.5, 1.0, 2.0, 4.0]
D = E.load("SKAB")
Xtr0, Xte0, y = np.asarray(D["Xn_w"], np.float32), np.asarray(D["Xa_w"], np.float32), np.asarray(D["ya_w"], int)
N = len(Xtr0)
mu, sd = Xtr0.mean(0), Xtr0.std(0) + 1e-8
Ztr = ((Xtr0 - mu) / sd); Zte = ((Xte0 - mu) / sd)

# ---- (1) external regimes: PCA(8) on TRAIN, k-means, K by silhouette on TRAIN only ----
pca = PCA(8, random_state=SEED).fit(Ztr)
Ptr, Pte = pca.transform(Ztr), pca.transform(Zte)
best = None
for K in range(3, 9):
    km = KMeans(K, n_init=10, random_state=SEED).fit(Ptr)
    s = silhouette_score(Ptr, km.labels_)
    if best is None or s > best[0]: best = (s, K, km)
sil, K, km = best
r_tr = km.labels_; r_te = km.predict(Pte)
p = np.bincount(r_tr, minlength=K) / N
print(f"regimes K={K} (silhouette {sil:.3f}) train shares {np.round(p,3)}  test-normal counts {np.bincount(r_te[y==0], minlength=K)}")

# ---- (2) sampler ----
def sample(beta, rng):
    w = p ** (1.0 + beta); w /= w.sum()
    n_c = np.floor(w * N).astype(int)
    n_c[np.argmax(w)] += N - n_c.sum()                  # fix rounding so sum == N exactly
    idx = []
    for c in range(K):
        pool = np.where(r_tr == c)[0]
        if n_c[c] <= len(pool): idx.append(rng.choice(pool, n_c[c], replace=False))   # down-sample rare
        else:                    idx.append(np.r_[pool, rng.choice(pool, n_c[c] - len(pool), replace=True)])  # up-sample common w/ replacement
    return np.concatenate(idx), n_c

rng = np.random.default_rng(SEED)
idx0, n0 = sample(0.0, rng)
assert sorted(idx0.tolist()) == list(range(N)), "beta=0 must be the identity multiset"
print("INVARIANT beta=0 == identity multiset: OK")
rare = int(np.argmin(p)); print(f"rarest regime = {rare} (share {p[rare]:.3f}, {int(p[rare]*N)} windows; test-normal in it: {int(((r_te==rare)&(y==0)).sum())})")
prev = None
for b in BETAS:
    _, n_c = sample(b, rng); share = n_c[rare] / N
    assert prev is None or share <= prev + 1e-12, "rare share must be non-increasing in beta"
    prev = share
    print(f"  beta={b:<4} n_c={n_c.tolist()} rare share={share:.4f} dup-frac(common)={(n_c.sum()-np.minimum(n_c, np.bincount(r_tr,minlength=K)).sum())/N:.3f}")
print("INVARIANT rare share monotone non-increasing: OK")

# ---- (3) FPR on starved-regime test-normals vs beta, 3 methods ----
C6 = Xte0.shape[1] // 6
hard = (y == 1) & (np.abs(Xte0[:, :C6]).max(1) <= np.quantile(np.abs(Xtr0[:, :C6]).max(1), 0.99))
norm = y == 0; rare_te = norm & (r_te == rare); other_te = norm & (r_te != rare)
def au(s, mask):
    k = norm | mask; return float(roc_auc_score(y[k], s[k])) if hard.sum() >= 2 else float("nan")
def fprs(s_tr, s_te):
    thr = np.quantile(s_tr, 0.99)                      # train-99pct threshold (paper-style)
    thr_m = np.quantile(s_te[other_te], 0.99)          # matched: 1% FPR on non-starved test normals
    return dict(fpr_rare=float((s_te[rare_te] > thr).mean()), fpr_other=float((s_te[other_te] > thr).mean()),
                fpr_rare_matched=float((s_te[rare_te] > thr_m).mean()), diff_auroc=round(au(s_te, hard), 3))
rows = []
for b in BETAS:
    rng = np.random.default_rng(SEED)                  # same rng per beta -> deterministic subsample
    idx, n_c = sample(b, rng); X = Ztr[idx].astype(np.float32); Xt = Zte.astype(np.float32)
    t0 = time.time()
    out = dict(beta=b, n_rare_train=int(n_c[rare]))
    IF = IsolationForest(n_estimators=200, random_state=SEED).fit(X)
    out["IF"] = fprs(-IF.decision_function(X), -IF.decision_function(Xt))
    out["AE"] = fprs(ae_scores(X, X, seed=SEED), ae_scores(X, Xt, seed=SEED))
    v = train_vade(X, n_clusters=16, latent_dim=6, epochs=40, warmup=8, seed=SEED, device="cpu")
    v.fit_latent_density(X, k_density=min(80, max(20, len(X) // 10)))
    out["LatAD"] = fprs(v.anomaly_score_hard(X), v.anomaly_score_hard(Xt))
    out["secs"] = round(time.time() - t0, 1); rows.append(out)
    print(f"beta={b:<4} rare_train={n_c[rare]:3d} | " + " | ".join(f"{m} rareFPR={out[m]['fpr_rare']:.3f} matched={out[m]['fpr_rare_matched']:.3f} otherFPR={out[m]['fpr_other']:.3f} diffAU={out[m]['diff_auroc']}" for m in ("IF","AE","LatAD")) + f" ({out['secs']}s)", flush=True)
json.dump(dict(K=K, silhouette=sil, shares=p.tolist(), rare=rare, rows=rows),
          open("E:/tmp/claude/E--Projects-Backlog-LatAD/6ae0c3f3-0eaf-4beb-8a09-88ce1e618d2a/scratchpad/imb_smoke_SKAB.json", "w"), indent=1)
