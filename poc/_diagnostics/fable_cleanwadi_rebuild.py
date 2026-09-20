"""Rebuild the cleaned-WADI per-window score table WITH the paper's WADI clip (10 sigma), which
E.load('WADI_clean') silently skipped (CLIP has no 'WADI_clean' key). One pass, one artifact:
label, maxz, maxz_thr, linres, l2, LatAD/IF/AE (5 seeds), plus the LatAD head components per seed
(density, nearest, forced-residual, forced-basin, gate decisions) so head ablations are offline.
Optional: --K --LD --tag to rebuild LatAD only under another config (baselines untouched).
Usage: python fable_cleanwadi_rebuild.py [--clip 10] [--K 20 --LD 10 --tag default]
"""
from __future__ import annotations
import os, sys, json, argparse, numpy as np, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.dirname(HERE); sys.path.insert(0, POC)
os.chdir(POC)
from sklearn.ensemble import IsolationForest
from models_vade import train_vade
from compare_baselines import ae_scores
import eda_real as E

ap = argparse.ArgumentParser()
ap.add_argument("--clip", type=float, default=10.0)
ap.add_argument("--K", type=int, default=20); ap.add_argument("--LD", type=int, default=10)
ap.add_argument("--tag", default="default"); ap.add_argument("--latad_only", action="store_true")
ap.add_argument("--seeds", default="0,1,2,3,4")
ap.add_argument("--drop", default="", help="EXPLORATORY: extra channels to drop (comma list); flagged, not headline")
a = ap.parse_args()
SEEDS = [int(s) for s in a.seeds.split(",")]
OUT = os.path.join(HERE, f"fable_cleanwadi_scores_clip{int(a.clip)}_{a.tag}.npz")

if a.drop:
    extra = set(a.drop.split(","))
    def _raw_drop(downsample=10):
        Xn, Xa, ya, sens = E._raw_wadi_clean(downsample=downsample)
        keep = [i for i, c in enumerate(sens) if c not in extra]
        assert len(keep) == len(sens) - len(extra), "unknown channel in --drop"
        return Xn[:, keep], Xa[:, keep], ya, [sens[i] for i in keep]
    E.RAW["WADI_clean"] = (_raw_drop, 60, 30)
D = E.load("WADI_clean", clip=a.clip); fn, W, stride = E.RAW["WADI_clean"]
Xn_raw, Xa_raw = np.asarray(D["Xn_raw"], float), np.asarray(D["Xa_raw"], float)
Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"].astype(int)
C6 = Xte0.shape[1] // 6
maxz = np.abs(Xte0[:, :C6]).max(1).astype(np.float32)
maxz_thr = float(np.quantile(np.abs(Xtr0[:, :C6]).max(1), 0.99))
mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
print(f"clip={a.clip} n_tr={len(Xtr)} n_te={len(Xte)} d={Xtr.shape[1]} anom={y.sum()} thr={maxz_thr:.3f} "
      f"hard={int(((y==1)&(maxz<=maxz_thr)).sum())} easy={int(((y==1)&(maxz>maxz_thr)).sum())}", flush=True)

cols = dict(label=y, maxz=maxz, maxz_thr=np.float32(maxz_thr), seeds=np.array(SEEDS),
            K=a.K, LD=a.LD, clip=a.clip)
if not a.latad_only:
    l2 = np.sqrt((Xte ** 2).mean(1)).astype(np.float32)
    from onehot_filter import build_feats, loco_residual as loco_oh
    Fn, Fa, grp = build_feats(Xn_raw, Xa_raw, W, stride, onehot=True)
    _, linres = loco_oh(Fn, Fa[:len(y)], grp)
    cols["linres"] = linres.astype(np.float32); cols["l2"] = l2
    print("  linres/l2 done", flush=True)

kd = min(80, max(20, len(Xtr) // 10))
comp = {k: [] for k in ("LatAD", "dens", "near", "resid", "basin", "LatAD_tr", "IF", "AE")}
gates = []
for sd in SEEDS:
    v = train_vade(Xtr, n_clusters=a.K, latent_dim=a.LD, epochs=40, warmup=8, seed=sd, device="cpu")
    v.fit_residual_whitener(Xtr); v.fit_latent_density(Xtr, k_density=kd)
    v.fit_resid_head(Xtr); v.fit_basin_head(Xtr)
    d_te, n_te = v._hard_components(Xte); dm, ds, nm, ns = v._hd_ref
    comp["dens"].append((d_te - dm) / ds); comp["near"].append((n_te - nm) / ns)
    base = v.anomaly_score_hard(Xte, use_resid=False, use_basin=False)
    comp["resid"].append(v.anomaly_score_hard(Xte, use_resid=True, use_basin=False) - base)
    comp["basin"].append(v.anomaly_score_hard(Xte, use_resid=False, use_basin=True) - base)
    comp["LatAD"].append(v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto"))
    comp["LatAD_tr"].append(v.anomaly_score_hard(Xtr, use_resid="auto", use_basin="auto"))
    gates.append(dict(seed=sd, resid_auto=bool(v._resid_auto), resid_gen_ratio=float(v._resid_gen_ratio),
                      basin_lam=float(v._basin_lam), basin_frac_amb=float(v._basin_frac_amb)))
    if not a.latad_only:
        comp["IF"].append(-IsolationForest(n_estimators=200, random_state=sd).fit(Xtr).score_samples(Xte))
        comp["AE"].append(ae_scores(Xtr, Xte, seed=sd))
    print(f"  seed {sd} done  gates={gates[-1]}", flush=True)
for k, vals in comp.items():
    if vals: cols[k] = np.stack(vals).astype(np.float32)
cols["gates"] = json.dumps(gates)
np.savez(OUT, **cols)
print("saved", OUT, flush=True)
