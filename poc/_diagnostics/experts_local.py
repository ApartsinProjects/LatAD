"""Local CPU re-implementation of sota_bundle/modal_experts.experts (use_full=1), parameterised by the
community-construction variant, so partition variants can be evaluated without Modal.
Mirrors the Modal code line by line: HAC on |rho| of train-normal window means (active channels), nested
dendrogram subtrees with MINSZ <= size <= MAXSZ, per-community VaDE (K=min(20,max(6,|G|)), latent
min(8,max(3,|G|//2)), 15 epochs, warmup 4), latent GMM k=min(50,max(12,nfit//12)), residual whitener + resid
head + basin head, scores standardised on the 20% calibration slice. 5 seeds.
Saves sota_bundle/experts_variants/<tag>/expert_<ds>.npz in the exact format ensemble_final expects.
Usage: python experts_local.py <ds> <tag> <method> <maxsz> <minsz> [nseeds]"""
from __future__ import annotations
import os, sys, time, gc, numpy as np, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from models_vade import train_vade

name, tag, method, MAXSZ, MINSZ = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
nseeds = int(sys.argv[6]) if len(sys.argv) > 6 else 5
TRIM = int(sys.argv[7]) if len(sys.argv) > 7 else 0          # leading TRAIN windows to drop (startup transient)
outdir = os.path.join(ROOT, "sota_bundle", "experts_variants", tag); os.makedirs(outdir, exist_ok=True)
outp = os.path.join(outdir, f"expert_{name}.npz")
if os.path.exists(outp):
    print("exists", outp); sys.exit(0)

B = np.load(os.path.join(ROOT, "sota_bundle", "ens_bundle", f"bundle_{name}.npz"))
Xn, Xa = B["Xn_w"].astype(float)[TRIM:], B["Xa_w"].astype(float)
print(f"[{name}/{tag}] TRIM={TRIM} leading train windows dropped -> {len(Xn)} train windows", flush=True)
nch = int(B["nch"]); nfit = len(Xn) * 4 // 5
means = Xn[:, :nch]; sdc = means.std(0); active = np.where(sdc > 1e-6)[0]
Zact = (means[:, active] - means[:, active].mean(0)) / sdc[active]
Cabs = np.abs(np.nan_to_num(np.corrcoef(Zact.T))); np.fill_diagonal(Cabs, 0.0)
dist = 1 - Cabs; dist = (dist + dist.T) / 2
L = linkage(squareform(dist, checks=False), method=method); _, nodes = to_tree(L, rd=True)
leaves = lambda n: [n.id] if n.is_leaf() else leaves(n.left) + leaves(n.right)
seen, comms, coh = set(), [], []
a2p = {int(a): i for i, a in enumerate(active)}
for n in nodes:
    if not n.is_leaf():
        lv = sorted(leaves(n))
        if MINSZ <= len(lv) <= MAXSZ and tuple(lv) not in seen:
            seen.add(tuple(lv)); G = [int(active[i]) for i in lv]; comms.append(G)
            pos = [a2p[c] for c in G]; sub = Cabs[np.ix_(pos, pos)]
            coh.append(float(sub.sum() / (len(pos) * (len(pos) - 1))))
sf = lambda Xw, S: np.nan_to_num(np.ascontiguousarray(Xw[:, [b * nch + c for b in range(6) for c in S]]))
S = len(comms); ncal = len(Xn) - nfit; ntest = len(Xa)
print(f"[{name}/{tag}] S={S} communities, method={method} maxsz={MAXSZ} minsz={MINSZ}, nfit={nfit} ncal={ncal}", flush=True)
Cal = np.zeros((nseeds, S, ncal), np.float32); Tst = np.zeros((nseeds, S, ntest), np.float32)
Fit = np.zeros((nseeds, S, nfit), np.float32)          # in-sample fit-slice scores (for a train-only stationarity gate)
t0 = time.time()
for si in range(nseeds):
    nfail = 0
    for gi, G in enumerate(comms):
        try:
            Ff, Fc, Ft = sf(Xn[:nfit], G), sf(Xn[nfit:], G), sf(Xa, G)
            m2, s2 = Ff.mean(0), Ff.std(0) + 1e-9
            cl = lambda A: np.clip(np.nan_to_num(((A - m2) / s2), posinf=10.0, neginf=-10.0), -10, 10).astype(np.float32)
            Ztr, Zc2, Zt2 = cl(Ff), cl(Fc), cl(Ft)
            v = train_vade(Ztr, n_clusters=min(20, max(6, len(G))), latent_dim=min(8, max(3, len(G) // 2)),
                           epochs=15, warmup=4, seed=si, device="cpu")
            v.fit_latent_density(Ztr, k_density=min(50, max(12, nfit // 12)))
            v.fit_residual_whitener(Ztr); v.fit_resid_head(Ztr); v.fit_basin_head(Ztr)
            cc = np.asarray(v.anomaly_score_hard(Zc2, use_resid="auto", use_basin="auto"))
            tt = np.asarray(v.anomaly_score_hard(Zt2, use_resid="auto", use_basin="auto"))
            ff = np.asarray(v.anomaly_score_hard(Ztr, use_resid="auto", use_basin="auto")); del v
            cm, cs = cc.mean(), cc.std() + 1e-9
            if not (np.isfinite(cm) and cs > 0):
                raise ValueError("non-finite expert scores")
            Cal[si, gi] = np.nan_to_num((cc - cm) / cs); Tst[si, gi] = np.nan_to_num((tt - cm) / cs)
            Fit[si, gi] = np.nan_to_num((ff - cm) / cs)
        except Exception as e:
            Cal[si, gi] = 0.0; Tst[si, gi] = 0.0; nfail += 1
            print(f"  seed {si} comm {gi} (|G|={len(G)}) skipped: {e}", flush=True)
        gc.collect()
    print(f"[{name}/{tag}] seed {si}: {S} experts ({nfail} skipped) {time.time()-t0:.0f}s", flush=True)
mx = max(len(g) for g in comms); chpad = np.full((S, mx), -1, int)
for i, g in enumerate(comms):
    chpad[i, :len(g)] = g
y = B["y"].astype(int)
np.savez(outp, comm_size=np.array([len(g) for g in comms]), comm_cohesion=np.array(coh), comm_channels=chpad,
         calib_surprise=Cal, test_surprise=Tst, fit_surprise=Fit, y=y, hard=(y == 1) & ~(B["maxz"] > float(B["maxz_thr"])))
print("saved", outp, flush=True)
