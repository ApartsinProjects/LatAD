"""Train-normal-only auto-gate probe for the LatAD-null fusion term (zl) in ensemble_final.

Mechanism mirrors models_vade.fit_resid_head's auto gate: fit the global VaDE on the 80% train-normal
slice A (with the FIX-2 clip), score the fit slice A (in-sample) and the held-out 20% slice B, and
measure whether the global LatAD score GENERALIZES to held-out normal. NO test scores/labels are read.

Candidate decision variables (all train-normal only), reported per dataset so the reader can see whether
they separate off-WADI / on-HAI-SWaT WITHOUT test data:
  R1  q95(sB)/q95(sA)                 (residual-head convention; gate ON if < 1.5)
  R2  frac(sB > q99(sA))              (held-out tail exceedance; ideal ~0.01)
  R3  (mean sB - mean sA)/std(sA)     (z-shift of held-out normal)
  R4  q99(sB)/q99(sA)                 (deeper tail ratio)
Persisted incrementally to clean_gate_probe.jsonl / clean_gate_probe.json.
"""
from __future__ import annotations
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
POC = r"E:\Projects\Backlog\LatAD\poc"
os.chdir(POC); sys.path.insert(0, POC)
from models_vade import train_vade
import eda_real as E

CFG = {"WADI_clean": (20, 10), "HAI": (40, 16), "SWaT_canon": (40, 16)}
SEEDS = [0, 1, 2, 3, 4]
OUT = os.path.dirname(os.path.abspath(__file__))
ROWS = os.path.join(OUT, "clean_gate_probe.jsonl")
if os.path.exists(ROWS):
    os.remove(ROWS)


def run(name):
    K, LD = CFG[name]
    D = E.load(name)
    Xtr0 = D["Xn_w"].astype(np.float32)
    clipv = E.CLIP.get(name)
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = (Xtr0 - mu) / sig
    if clipv:
        Xtr = np.clip(Xtr, -clipv, clipv)      # FIX 2
    Xtr = Xtr.astype(np.float32)
    nfit = int(0.8 * len(Xtr))
    A, B = Xtr[:nfit], Xtr[nfit:]
    kd = min(80, max(20, len(A) // 10))
    per = []
    for sd in SEEDS:
        v = train_vade(A, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
        v.fit_residual_whitener(A); v.fit_latent_density(A, k_density=kd)
        v.fit_resid_head(A); v.fit_basin_head(A)
        sA = np.asarray(v.anomaly_score_hard(A, use_resid="auto", use_basin="auto"))
        sB = np.asarray(v.anomaly_score_hard(B, use_resid="auto", use_basin="auto"))
        qA95, qB95 = np.quantile(sA, 0.95), np.quantile(sB, 0.95)
        qA99, qB99 = np.quantile(sA, 0.99), np.quantile(sB, 0.99)
        row = dict(dataset=name, seed=sd,
                   R1_q95ratio=float(qB95 / (qA95 + 1e-9)),
                   R2_tail_exceed=float((sB > qA99).mean()),
                   R3_zshift=float((sB.mean() - sA.mean()) / (sA.std() + 1e-9)),
                   R4_q99ratio=float(qB99 / (qA99 + 1e-9)),
                   resid_auto=bool(getattr(v, "_resid_auto", False)),
                   resid_gen_ratio=float(getattr(v, "_resid_gen_ratio", float("nan"))))
        per.append(row)
        with open(ROWS, "a") as fh:
            fh.write(json.dumps(row) + "\n"); fh.flush()
        print(f"  [{name}] seed {sd}: R1={row['R1_q95ratio']:.3f} R2={row['R2_tail_exceed']:.4f} "
              f"R3={row['R3_zshift']:+.3f} R4={row['R4_q99ratio']:.3f} resid_auto={row['resid_auto']}", flush=True)
    agg = {k: [np.mean([r[k] for r in per]), np.std([r[k] for r in per])]
           for k in ("R1_q95ratio", "R2_tail_exceed", "R3_zshift", "R4_q99ratio")}
    return dict(name=name, nfit=nfit, nB=len(B), per_seed=per, agg={k: [round(v[0], 4), round(v[1], 4)] for k, v in agg.items()})


if __name__ == "__main__":
    res = {}
    for nm in (sys.argv[1:] or ["WADI_clean", "HAI", "SWaT_canon"]):
        res[nm] = run(nm)
        json.dump(res, open(os.path.join(OUT, "clean_gate_probe.json"), "w"), indent=1)
    print("\n=== gate decision variables (train-normal only, mean over 5 seeds) ===")
    for nm, r in res.items():
        a = r["agg"]
        print(f"  {nm:12} R1 q95ratio={a['R1_q95ratio'][0]:.3f}  R2 tail_exceed={a['R2_tail_exceed'][0]:.4f}  "
              f"R3 zshift={a['R3_zshift'][0]:+.3f}  R4 q99ratio={a['R4_q99ratio'][0]:.3f}")
    print("saved -> clean_gate_probe.json", flush=True)
