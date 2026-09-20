"""Why does PCA-SPE on the detector feature space flag almost none of the max|z|-difficult anomalies
that LinRes flags? Diagnostics on one dataset:
  1. where the difficult anomalies sit in the TRAIN SPE / T2 distribution (percentile);
  2. which stat blocks carry the train SPE energy (is the residual dominated by noisy blocks?);
  3. variants of the PCA input: means-only block (the LinRes feature space, minus one-hot), and
     LinRes-style one-hot features (Fn/Fa from build_feats) -> does SPE then agree with LinRes?
"""
import os, sys, numpy as np
POC = r"E:\Projects\Backlog\LatAD\poc"; os.chdir(POC); sys.path.insert(0, POC)
import eda_real as E
from onehot_filter import build_feats, loco_residual
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


def detector_inputs(name):
    D = E.load(name)
    Xtr0, Xte0 = D["Xn_w"].astype(np.float64), D["Xa_w"].astype(np.float64)
    clipv = E.CLIP.get(name)
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr, Xte = (Xtr0 - mu) / sig, (Xte0 - mu) / sig
    if clipv:
        Xtr, Xte = np.clip(Xtr, -clipv, clipv), np.clip(Xte, -clipv, clipv)
    return D, Xtr, Xte


def pca_stats(Xtr, Xte, var_keep):
    m = Xtr.mean(0); Ztr, Zte = Xtr - m, Xte - m
    U, S, Vt = np.linalg.svd(Ztr, full_matrices=False)
    lam = S ** 2 / (len(Ztr) - 1); cum = np.cumsum(lam) / lam.sum()
    k = int(min(np.searchsorted(cum, var_keep) + 1, len(lam)))
    lam_k = lam[:k]; P = Vt[:k].T
    def stats(Z):
        sc = Z @ P
        return (sc ** 2 / lam_k).sum(1), np.maximum((Z ** 2).sum(1) - (sc ** 2).sum(1), 0.0)
    return stats(Ztr), stats(Zte), k, lam, cum

name = sys.argv[1] if len(sys.argv) > 1 else "WADI_clean"
fn, W, stride = E.RAW[name]
D, Xtr, Xte = detector_inputs(name)
d = np.load(f"{POC}/_diagnostics/scores_{name}.npz"); y = d["label"].astype(int)
maxz, mthr = d["maxz"], float(d["maxz_thr"]); canon = (y == 1) & (maxz <= mthr)
Xn_raw, Xa_raw = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
Fn, Fa, grp = build_feats(Xn_raw, Xa_raw, W, stride, onehot=True)
r_tr, r_te = loco_residual(Fn, Fa[:len(y)], grp); lin_thr = float(np.quantile(r_tr, 0.99))
C = Xtr.shape[1] // 6
print(f"{name}: d={Xtr.shape[1]} (C={C}), LinRes feats={Fn.shape[1]}, difficult={canon.sum()}, LinRes flags {(r_te[canon]>lin_thr).sum()}")

def pct(ref, v):
    o = np.sort(ref); return np.searchsorted(o, v, side="right") / len(o)

def report(tag, Ztr, Zte, var=0.95):
    (t2_tr, spe_tr), (t2_te, spe_te), k, lam, cum = pca_stats(Ztr, Zte, var)
    thr_t2, thr_spe = np.quantile(t2_tr, .99), np.quantile(spe_tr, .99)
    fl_t2 = t2_te[canon] > thr_t2; fl_spe = spe_te[canon] > thr_spe; fl_lin = r_te[canon] > lin_thr
    keepD = (y == 0) | canon
    print(f"  [{tag:22}] d={Ztr.shape[1]:4} k={k:3}  flags on difficult: T2 {fl_t2.sum():2} SPE {fl_spe.sum():2} "
          f"union {(fl_t2|fl_spe).sum():2} | LinRes {fl_lin.sum():2} (SPE&Lin {(fl_spe&fl_lin).sum()}) "
          f"| difficult-AUROC as detector: T2 {roc_auc_score(y[keepD], t2_te[keepD]):.3f} SPE {roc_auc_score(y[keepD], spe_te[keepD]):.3f} "
          f"LinRes {roc_auc_score(y[keepD], r_te[keepD]):.3f} | rho(SPE,Lin|difficult)={spearmanr(spe_te[canon], r_te[canon]).correlation:.2f}")
    return t2_tr, spe_tr, t2_te, spe_te, thr_t2, thr_spe

# 1. full detector feature space (the spec'd filter)
t2_tr, spe_tr, t2_te, spe_te, thr_t2, thr_spe = report("stats x6 (spec)", Xtr, Xte)
p_spe = pct(spe_tr, spe_te[canon]); p_t2 = pct(t2_tr, t2_te[canon]); p_lin = pct(r_tr, r_te[canon])
print("  difficult anomalies, train-percentile of  SPE :", np.round(np.sort(p_spe), 3))
print("  difficult anomalies, train-percentile of  T2  :", np.round(np.sort(p_t2), 3))
print("  difficult anomalies, train-percentile of  LinRes:", np.round(np.sort(p_lin), 3))
# 2. which stat blocks carry the train SPE energy
m = Xtr.mean(0); Ztr = Xtr - m
U, S, Vt = np.linalg.svd(Ztr, full_matrices=False); lam = S ** 2 / (len(Ztr) - 1)
k = int(np.searchsorted(np.cumsum(lam) / lam.sum(), .95) + 1); P = Vt[:k].T
Rtr = Ztr - (Ztr @ P) @ P.T; Rte = (Xte - m) - ((Xte - m) @ P) @ P.T
blk = ["mean", "std", "min", "max", "slope?", "block6"]
etr = [(Rtr[:, b*C:(b+1)*C] ** 2).sum(1).mean() for b in range(6)]
ete = [(Rte[canon][:, b*C:(b+1)*C] ** 2).sum(1).mean() for b in range(6)]
print("  train SPE energy by stat block     :", np.round(etr, 2), " total", round(sum(etr), 2))
print("  difficult-anomaly SPE energy/block :", np.round(ete, 2), " total", round(sum(ete), 2))
# per-column: how many columns carry the top 50% of train SPE energy
col = (Rtr ** 2).mean(0); o = np.argsort(col)[::-1]; cs = np.cumsum(col[o]) / col.sum()
print(f"  columns carrying 50% / 90% of train SPE energy: {int(np.searchsorted(cs,.5)+1)} / {int(np.searchsorted(cs,.9)+1)} of {Xtr.shape[1]}")
# 3. variants of PCA input
report("means block only", Xtr[:, :C], Xte[:, :C])
report("means, 99% var", Xtr[:, :C], Xte[:, :C], 0.99)
report("one-hot LinRes feats", Fn, Fa[:len(y)])
report("one-hot feats, 99%", Fn, Fa[:len(y)], 0.99)
# 4. per-feature-normalized SPE (each residual column scaled by its train residual std), full space
sd_r = Rtr.std(0) + 1e-9
spe_n_tr = ((Rtr / sd_r) ** 2).sum(1); spe_n_te = ((Rte / sd_r) ** 2).sum(1)
thr = np.quantile(spe_n_tr, .99); keepD = (y == 0) | canon
print(f"  [normalized SPE, x6   ] flags on difficult {int((spe_n_te[canon] > thr).sum())}  difficult-AUROC {roc_auc_score(y[keepD], spe_n_te[keepD]):.3f}")
