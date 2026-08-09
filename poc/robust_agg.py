import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
import eda_real as E
def sf(Xw,nch,S): return np.ascontiguousarray(Xw[:,[b*nch+c for b in range(6) for c in S]])
D=E.load("WADI"); nch=len(D["ch"])
Xn=np.asarray(D["Xn_w"],float); Xa=np.asarray(D["Xa_w"],float); y=np.asarray(D["ya_w"],int)
d=np.load("_diagnostics/scores_WADI.npz"); thr=float(d["maxz_thr"]); hard=(y==1)&~((y==1)&(d["maxz"]>thr)); keep=(y==0)|hard
nfit=len(Xn)*4//5; K=48; m=24
def agg_auroc(Zc,Zt):  # members x windows (per-member standardized); calibrate aggregate on held-out normal
    out={}
    for nm,fn in [("max",lambda A:A.max(0)),("q99",lambda A:np.percentile(A,99,0)),
                  ("q95",lambda A:np.percentile(A,95,0)),("q90",lambda A:np.percentile(A,90,0)),
                  ("top3",lambda A:np.sort(A,0)[-3:].mean(0)),("top10",lambda A:np.sort(A,0)[-10:].mean(0))]:
        ac=fn(Zc); at=fn(Zt); s=(at-ac.mean())/(ac.std()+1e-9)
        out[nm]=round(float(roc_auc_score(y[keep],s[keep])),3)
    return out
for sd in [1,2]:
    rng=np.random.default_rng(sd); Zc=[]; Zt=[]
    for j in range(K):
        S=np.sort(rng.choice(nch,size=m,replace=False))
        Ftr=sf(Xn,nch,S); Fte=sf(Xa,nch,S); mu,sg=Ftr[:nfit].mean(0),Ftr[:nfit].std(0)+1e-9
        Ztr=((Ftr-mu)/sg).astype(np.float32); Zte=((Fte-mu)/sg).astype(np.float32)
        v=train_vade(Ztr[:nfit],n_clusters=20,latent_dim=8,epochs=15,warmup=4,seed=0,device="cpu")
        v.fit_latent_density(Ztr[:nfit],k_density=min(60,max(15,nfit//10)))
        cal=np.asarray(v.anomaly_score_hard(Ztr[nfit:],use_resid=False,use_basin=False))
        sc =np.asarray(v.anomaly_score_hard(Zte,use_resid=False,use_basin=False))
        cm,cs=cal.mean(),cal.std()+1e-9
        Zc.append(np.nan_to_num((cal-cm)/cs)); Zt.append(np.nan_to_num((sc-cm)/cs))
        if (j+1)%24==0: print(f"  seed {sd} {j+1}/{K}",flush=True)
    print(f"seed {sd}: {agg_auroc(np.stack(Zc),np.stack(Zt))}",flush=True)
print("DONE")
