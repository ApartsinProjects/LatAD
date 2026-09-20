"""HAI: why does PCA-T2 (k=60, 95% var) flag 90/167 max|z|-difficult anomalies and reach 0.807 difficult AUROC?
Decompose T2 per stat block (z_j * (P L^-1 P^T z)_j summed over columns of each block), check the retained
eigenvalue range, and list the top contributing columns for flagged-by-T2-but-not-LinRes windows."""
import os, sys, numpy as np
POC = r"E:\Projects\Backlog\LatAD\poc"; os.chdir(POC); sys.path.insert(0, POC)
import eda_real as E
from sklearn.metrics import roc_auc_score
name = sys.argv[1] if len(sys.argv) > 1 else "HAI"
D = E.load(name); ch = D["ch"]
Xtr0, Xte0 = D["Xn_w"].astype(np.float64), D["Xa_w"].astype(np.float64)
clipv = E.CLIP.get(name); mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
Xtr, Xte = (Xtr0 - mu) / sig, (Xte0 - mu) / sig
if clipv: Xtr, Xte = np.clip(Xtr, -clipv, clipv), np.clip(Xte, -clipv, clipv)
P_ = np.load(f"{POC}/_diagnostics/pca_filter_doublehard_{name}.npz")
y, t2, spe, thr_t2, thr_spe, lin, lin_thr = P_["y"], P_["t2"], P_["spe"], float(P_["thr_t2"]), float(P_["thr_spe"]), P_["linres_te"], float(P_["lin_thr"])
canon = (y == 1) & (P_["maxz"] <= float(P_["maxz_thr"]))
m = Xtr.mean(0); Ztr, Zte = Xtr - m, Xte - m
U, S, Vt = np.linalg.svd(Ztr, full_matrices=False); lam = S ** 2 / (len(Ztr) - 1); cum = np.cumsum(lam) / lam.sum()
k = int(np.searchsorted(cum, .95) + 1); P = Vt[:k].T; C = Xtr.shape[1] // 6
print(f"{name}: k={k}, lam[0]={lam[0]:.2f} lam[k-1]={lam[k-1]:.3f} ratio {lam[0]/lam[k-1]:.0f}; n train cols with std<1e-6 (constant): {(Xtr0.std(0)<1e-6).sum()}")
M = P @ np.diag(1 / lam[:k]) @ P.T                       # T2 = z^T M z
def t2_blocks(Z):
    contrib = Z * (Z @ M)                                # per column, sums to T2
    return np.stack([contrib[:, b*C:(b+1)*C].sum(1) for b in range(6)], 1)
blk = ["mean", "std", "min", "max", "blk5", "blk6"]
Btr = t2_blocks(Ztr); Bd = t2_blocks(Zte[canon])
print("  train mean T2 by block            :", np.round(Btr.mean(0), 1), " sum", round(Btr.sum(1).mean(), 1))
print("  difficult-anomaly mean T2 by block:", np.round(Bd.mean(0), 1), " sum", round(Bd.sum(1).mean(), 1))
print("  difficult-anomaly MEDIAN T2 by block:", np.round(np.median(Bd, 0), 1))
# recompute T2 with the MEAN block only vs the other blocks (PCA refit on the sub-block), as detectors on difficult
keepD = (y == 0) | canon
for tag, cols in [("means only", slice(0, C)), ("std only", slice(C, 2*C)), ("blocks 2-6 (no means)", slice(C, 6*C))]:
    Ztr_b, Zte_b = Ztr[:, cols], Zte[:, cols]
    U_, S_, Vt_ = np.linalg.svd(Ztr_b, full_matrices=False); l_ = S_ ** 2 / (len(Ztr_b) - 1); c_ = np.cumsum(l_) / l_.sum()
    k_ = int(np.searchsorted(c_, .95) + 1); Pb = Vt_[:k_].T
    t2b = ((Zte_b @ Pb) ** 2 / l_[:k_]).sum(1); t2b_tr = ((Ztr_b @ Pb) ** 2 / l_[:k_]).sum(1)
    speb = (Zte_b ** 2).sum(1) - ((Zte_b @ Pb) ** 2).sum(1); speb_tr = (Ztr_b ** 2).sum(1) - ((Ztr_b @ Pb) ** 2).sum(1)
    print(f"  [{tag:22}] k={k_:3}  T2 difficult-AUROC {roc_auc_score(y[keepD], t2b[keepD]):.3f}  flags {(t2b[canon] > np.quantile(t2b_tr,.99)).sum():3}/{canon.sum()}"
          f" | SPE difficult-AUROC {roc_auc_score(y[keepD], speb[keepD]):.3f} flags {(speb[canon] > np.quantile(speb_tr,.99)).sum():3}")
# the T2-flagged-but-not-LinRes windows: top contributing columns
idx = np.where(canon & (t2 > thr_t2) & (lin <= lin_thr))[0]
print(f"  T2-flagged, LinRes-passed difficult windows: {len(idx)}; T2 train-99th {thr_t2:.1f}")
contrib = Zte[idx] * (Zte[idx] @ M)
for i, w in enumerate(idx[:6]):
    top = np.argsort(contrib[i])[::-1][:4]
    print(f"    win {w}: T2={t2[w]:.0f} top cols " + ", ".join(f"{blk[c//C]}:{ch[c%C]}({contrib[i][c]:.0f}, z={Zte[w][c]:.1f})" for c in top))
# train-percentile view for the full T2 statistic vs a robust reference: does T2 exceed chi2(k) tails on train?
from scipy.stats import chi2
print(f"  train T2 quantiles 50/90/99: {np.quantile(t2, [.5,.9,.99]) if False else np.round(np.quantile(((Ztr @ P) ** 2 / lam[:k]).sum(1), [.5,.9,.99]),1)}  chi2({k}) 50/90/99: {np.round(chi2.ppf([.5,.9,.99], k),1)}")
