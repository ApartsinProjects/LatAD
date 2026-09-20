import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
import eda_real as E
def sf(Xw,nch,S): return np.ascontiguousarray(Xw[:,[b*nch+c for b in range(6) for c in S]])
name="WADI"; D=E.load(name); nch=len(D["ch"])
Xn=np.asarray(D["Xn_w"],float); Xa=np.asarray(D["Xa_w"],float); y=np.asarray(D["ya_w"],int)
d=np.load("_diagnostics/scores_WADI.npz"); thr=float(d["maxz_thr"]); hard=(y==1)&~((y==1)&(d["maxz"]>thr)); keep=(y==0)|hard
nfit=len(Xn)*4//5
for sd in [0,1,2]:
    rng=np.random.default_rng(sd); cols=[]
    for _ in range(24):
        S=np.sort(rng.choice(nch,size=24,replace=False))
        Ftr=sf(Xn,nch,S); Fte=sf(Xa,nch,S); mu,sg=Ftr[:nfit].mean(0),Ftr[:nfit].std(0)+1e-9
        Ztr=((Ftr-mu)/sg).astype(np.float32); Zte=((Fte-mu)/sg).astype(np.float32)
        v=train_vade(Ztr[:nfit],n_clusters=20,latent_dim=8,epochs=20,warmup=5,seed=0,device="cpu")
        v.fit_latent_density(Ztr[:nfit],k_density=min(60,max(15,nfit//10)))
        cal=np.asarray(v.anomaly_score_hard(Ztr[nfit:],use_resid=False,use_basin=False))
        sc=np.asarray(v.anomaly_score_hard(Zte,use_resid=False,use_basin=False))
        cols.append(np.nan_to_num((sc-cal.mean())/(cal.std()+1e-9)))
    top3=np.sort(np.stack(cols),axis=0)[-3:].mean(0)
    print(f"subset-seed {sd}: WADI ens difficult-AUROC = {round(float(roc_auc_score(y[keep],top3[keep])),3)}",flush=True)
print("DONE")
