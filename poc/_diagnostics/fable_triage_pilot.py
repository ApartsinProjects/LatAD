"""Post-hoc regime-triage pilot on the FROZEN LatAD latent (report-only; touches no paper/model file).

A flagged window (score > 1%-FPR operating threshold fixed on TRAIN-normal) gets a SECOND label:
  RARE-VALID  iff  novel latent region (it is flagged)  AND  physics holds:
                   low whitened residual (A8)  AND  basin agreement holds (A3)
                   AND it sits in a coherent under-represented cluster:
                       code has >= NMIN train windows and < RARE_FRAC train share
                       AND the test run-length of that code is >= LRUN windows.
  FAULT       otherwise.
The anomaly score is NEVER modified: the triage only adds a label.

Partitions compared for the "cluster" term (same rule, same constants):
  P0  VaDE nearest-component argmax (K components)
  P1  P0 with high-variance components split post hoc (F8), depth <= 2
  P2  k-means codebook on the train latent, M in {64,128,256}
  CTL M=1 codebook (degenerate control; must label nothing RARE-VALID)

Pre-stated constants (fixed on train-normal, never on test):
  FLAG   : score > q99(train score)                 (1% train FPR operating point)
  RESID  : resid z <= q99(train resid z)
  BASIN  : agreement >= q05(train agreement)
  NMIN=20, RARE_FRAC=0.02, LRUN=10
  SPLIT  : component trace-variance > 2x median  -> 2-GMM split, recursive, depth<=2

Pre-stated invariants:
  (a) per-seed test scores reproduce _diagnostics/scores_HAI.npz LatAD[seed]; full/diff AUROC to 4 digits
  (b) the witness regimes (occ<1%, test-normal share >3x train and >0.5%; seed 0: comps 8, 21) are
      labelled RARE-VALID for the majority of their flagged windows
  (c) CTL M=1 labels nothing RARE-VALID
Win bar: >= 50% of rare-normal FPs reclassified RARE-VALID, <= 5% of labelled anomalies RARE-VALID.
"""
from __future__ import annotations
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.dirname(HERE)
sys.path.insert(0, POC); os.chdir(POC)
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

CFG = {"HAI": (40, 16), "SKAB": (16, 6), "WADI": (20, 10), "SWaT": (40, 16)}
SEEDS = [0, 1, 2, 3, 4]
NMIN, RARE_FRAC, LRUN = 20, 0.02, 10
SPLIT_RATIO, SPLIT_DEPTH = 2.0, 2
MS = [64, 128, 256]


def au(y, s, mask):
    k = (y == 0) | mask; yy = y[k]
    if yy.sum() < 2 or (yy == 0).sum() < 2: return float("nan")
    return float(roc_auc_score(yy, s[k]))


# ------------------------------------------------------------------ stage
def stage(name, sd, nthreads):
    import torch
    torch.set_num_threads(nthreads)
    import eda_real as E
    from models_vade import train_vade
    D = E.load(name); K, LD = CFG[name]
    Xtr0 = np.asarray(D["Xn_w"], np.float32); Xte0 = np.asarray(D["Xa_w"], np.float32)
    y = np.asarray(D["ya_w"], int)
    C6 = Xte0.shape[1] // 6
    maxz = np.abs(Xte0[:, :C6]).max(1); thr = float(np.quantile(np.abs(Xtr0[:, :C6]).max(1), 0.99))
    hard = maxz > thr
    mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
    Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
    kd = min(80, max(20, len(Xtr0) // 10))
    t0 = time.time()
    # NOTE: torch CPU training is only bit-reproducible at the thread count the stored table was
    # built with (default = all cores). A 1-thread run of seed 0 gave AUROC 0.9381 vs stored 0.9409.
    v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
    v.fit_residual_whitener(Xtr)
    # The stored table predates the "heads default to the model seed" change (old code: seed 0 for
    # every head). Try the current default first, fall back to head seed 0, record which reproduced.
    head_seed_used, repro = None, None
    for hs in (sd, 0):
        v.fit_latent_density(Xtr, k_density=kd, seed=hs)
        v.fit_resid_head(Xtr, seed=hs); v.fit_basin_head(Xtr, seed=hs)
        s_te = np.asarray(v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto"), float)
        repro = check_repro(name, sd, dict(s_te=s_te, y=y, hard=hard)); head_seed_used = hs
        if repro.get("PASS", True): break
        print(f"  [{name}] seed {sd}: head seed {hs} did not reproduce ({repro}), retrying", flush=True)
    s_tr = np.asarray(v.anomaly_score_hard(Xtr, use_resid="auto", use_basin="auto"), float)
    Ztr, Zte = v._encode_mean(Xtr), v._encode_mean(Xte)
    with torch.no_grad():
        from models_vade import _as_tensor
        a_tr = v._log_pz_given_c(_as_tensor(Ztr, v)).argmax(1).cpu().numpy()
        a_te = v._log_pz_given_c(_as_tensor(Zte, v)).argmax(1).cpu().numpy()
    rm, rsd = v._rd_ref
    def rz(X):
        Q = v._rp.transform(v._residual(X)); G = v._responsibilities(X)
        return (v._resid_score(Q, G) - rm) / rsd
    rs_tr, rs_te = rz(Xtr), rz(Xte)
    ag_tr, ag_te = v._noise_agreement(Xtr), v._noise_agreement(Xte)
    out = dict(s_tr=s_tr, s_te=s_te, Ztr=Ztr, Zte=Zte, a_tr=a_tr, a_te=a_te, rs_tr=rs_tr, rs_te=rs_te,
               ag_tr=ag_tr, ag_te=ag_te, y=y, hard=hard, resid_auto=bool(v._resid_auto),
               basin_lam=float(v._basin_lam), K=K, head_seed_used=head_seed_used, nthreads=nthreads)
    np.savez(os.path.join(HERE, f"fable_triage_{name}_s{sd}.npz"), **out)
    print(f"  [{name}] seed {sd} staged in {time.time()-t0:.0f}s  resid_auto={v._resid_auto} basin_lam={v._basin_lam:.3f}", flush=True)
    return out


def check_repro(name, sd, d):
    """Invariant (a): scores reproduce the stored table."""
    p = os.path.join(HERE, f"scores_{name}.npz")
    if not os.path.exists(p): return {"stored": False}
    S = np.load(p); i = list(S["seeds"]).index(sd)
    st = S["LatAD"][i].astype(float); y = S["label"]; hard = S["maxz"] > S["maxz_thr"]
    assert (y == d["y"]).all() and (hard == d["hard"]).all()
    r = dict(stored=True, max_abs_diff=float(np.abs(st - d["s_te"]).max()),
             stored_full=round(roc_auc_score(y, st), 4), new_full=round(roc_auc_score(y, d["s_te"]), 4),
             stored_diff=round(au(y, st, hard), 4), new_diff=round(au(y, d["s_te"], hard), 4))
    r["PASS"] = bool(r["stored_full"] == r["new_full"] and r["stored_diff"] == r["new_diff"])
    return r


# ------------------------------------------------------------------ partitions
def split_partition(a_tr, a_te, Ztr, Zte, seed):
    """F8: split components whose trace-variance > SPLIT_RATIO x median, recursively."""
    codes_tr, codes_te = a_tr.copy(), a_te.copy()
    nxt = int(max(codes_tr.max(), codes_te.max())) + 1
    for _ in range(SPLIT_DEPTH):
        ids = [c for c in np.unique(codes_tr) if (codes_tr == c).sum() >= 2 * NMIN]
        tv = {c: Ztr[codes_tr == c].var(0).sum() for c in ids}
        med = np.median(list(tv.values()))
        big = [c for c in ids if tv[c] > SPLIT_RATIO * med]
        if not big: break
        for c in big:
            g = GaussianMixture(2, covariance_type="diag", random_state=seed, reg_covar=1e-4).fit(Ztr[codes_tr == c])
            lt = g.predict(Ztr[codes_tr == c]); le = g.predict(Zte[codes_te == c]) if (codes_te == c).any() else None
            if min((lt == 0).sum(), (lt == 1).sum()) < NMIN: continue    # do not create an empty child
            idx = np.where(codes_tr == c)[0]; codes_tr[idx[lt == 1]] = nxt
            if le is not None:
                jdx = np.where(codes_te == c)[0]; codes_te[jdx[le == 1]] = nxt
            nxt += 1
    return codes_tr, codes_te


def kmeans_partition(Ztr, Zte, M, seed):
    km = KMeans(M, n_init=3, random_state=seed).fit(Ztr)
    return km.labels_, km.predict(Zte)


def run_length(codes):
    """for each position, length of the maximal contiguous run of equal codes containing it."""
    n = len(codes); rl = np.zeros(n, int); i = 0
    while i < n:
        j = i
        while j + 1 < n and codes[j + 1] == codes[i]: j += 1
        rl[i:j + 1] = j - i + 1; i = j + 1
    return rl


# ------------------------------------------------------------------ triage
def triage(d, codes_tr, codes_te):
    s_tr, s_te, y = d["s_tr"], d["s_te"], d["y"]
    flag_thr = np.quantile(s_tr, 0.99)
    FLAG = s_te > flag_thr
    RESID = d["rs_te"] <= np.quantile(d["rs_tr"], 0.99)
    BASIN = d["ag_te"] >= np.quantile(d["ag_tr"], 0.05)
    cnt = np.bincount(codes_tr, minlength=int(max(codes_tr.max(), codes_te.max())) + 1)
    frac = cnt / len(codes_tr)
    RARE = (cnt[codes_te] >= NMIN) & (frac[codes_te] < RARE_FRAC)
    COH = run_length(codes_te) >= LRUN
    RV = FLAG & RESID & BASIN & RARE & COH
    return dict(FLAG=FLAG, RESID=RESID, BASIN=BASIN, RARE=RARE, COH=COH, RV=RV, flag_thr=float(flag_thr))


def metrics(d, T, rare_norm, witness_mask):
    y, hard = d["y"], d["hard"]; norm = y == 0; anom = y == 1
    FLAG, RV = T["FLAG"], T["RV"]
    rn_fp = rare_norm & FLAG
    m = {}
    m["n_flag"] = int(FLAG.sum()); m["n_flag_norm"] = int((FLAG & norm).sum()); m["n_flag_anom"] = int((FLAG & anom).sum())
    m["n_rare_norm_fp"] = int(rn_fp.sum())
    m["rare_norm_fp_reclassified"] = float(RV[rn_fp].mean()) if rn_fp.sum() else float("nan")
    m["common_norm_fp_reclassified"] = float(RV[norm & FLAG & ~rare_norm].mean()) if (norm & FLAG & ~rare_norm).sum() else float("nan")
    m["anom_RV_of_all_anom"] = float(RV[anom].mean())
    m["anom_RV_of_flagged_anom"] = float(RV[anom & FLAG].mean()) if (anom & FLAG).sum() else float("nan")
    m["hard_anom_RV_of_flagged_hard"] = float(RV[anom & hard & FLAG].mean()) if (anom & hard & FLAG).sum() else float("nan")
    # confusion both directions, among FLAGGED windows: rows true (normal / anomaly), cols label (RARE-VALID / FAULT)
    m["confusion_flagged"] = {"normal": {"RARE_VALID": int((FLAG & norm & RV).sum()), "FAULT": int((FLAG & norm & ~RV).sum())},
                              "anomaly": {"RARE_VALID": int((FLAG & anom & RV).sum()), "FAULT": int((FLAG & anom & ~RV).sum())}}
    # FPR / recall at the operating point, before and after dropping RARE-VALID alarms
    m["fpr_before"] = float(FLAG[norm].mean()); m["fpr_after"] = float((FLAG & ~RV)[norm].mean())
    m["recall_before"] = float(FLAG[anom].mean()); m["recall_after"] = float((FLAG & ~RV)[anom].mean())
    m["anoms_dropped"] = int((FLAG & RV & anom).sum())
    # AUROC with RARE-VALID windows removed from evaluation (method-output version of the manual exclusion)
    keep = ~RV
    m["auroc_full"] = round(roc_auc_score(y, d["s_te"]), 4); m["auroc_full_excl_RV"] = round(roc_auc_score(y[keep], d["s_te"][keep]), 4)
    m["auroc_diff"] = round(au(y, d["s_te"], hard), 4); m["auroc_diff_excl_RV"] = round(au(y[keep], d["s_te"][keep], hard[keep]), 4)
    m["n_RV_norm"] = int((RV & norm).sum()); m["n_RV_anom"] = int((RV & anom).sum())
    # witness
    wf = witness_mask & FLAG
    m["witness_n"] = int(witness_mask.sum()); m["witness_flagged"] = float(FLAG[witness_mask].mean()) if witness_mask.sum() else float("nan")
    m["witness_flagged_RV"] = float(RV[wf].mean()) if wf.sum() else float("nan")
    # which guard kills rare-normal FPs that were NOT reclassified
    miss = rn_fp & ~RV
    m["rare_fp_missed_by"] = {k: float((~T[k])[miss].mean()) if miss.sum() else float("nan") for k in ("RESID", "BASIN", "RARE", "COH")}
    return m


def analyze(name, sd, d):
    y = d["y"]; norm = y == 0; K = int(d["K"])
    a_tr, a_te = d["a_tr"], d["a_te"]
    occ = np.bincount(a_tr, minlength=K) / len(a_tr)
    occ_te = np.bincount(a_te[norm], minlength=K) / norm.sum()
    rare_ids = np.where(occ < RARE_FRAC)[0]
    rare_norm = norm & np.isin(a_te, rare_ids)              # E2 definition, partition-independent (P0)
    witness_ids = [int(c) for c in range(K) if occ[c] < 0.01 and occ_te[c] > 3 * occ[c] and occ_te[c] > 0.005]
    witness = norm & np.isin(a_te, witness_ids)
    # contiguity of the witness block
    wi = np.where(witness)[0]; nruns = int((np.diff(wi) > 1).sum() + 1) if len(wi) else 0
    res = dict(seed=sd, witness_ids=witness_ids, witness_n=int(witness.sum()), witness_runs=nruns,
               n_rare_norm=int(rare_norm.sum()), rare_ids=rare_ids.tolist(), partitions={})
    parts = {"P0_vade": (a_tr, a_te)}
    parts["P1_split"] = split_partition(a_tr, a_te, d["Ztr"], d["Zte"], sd)
    for M in MS:
        parts[f"P2_km{M}"] = kmeans_partition(d["Ztr"], d["Zte"], M, sd)
    parts["CTL_M1"] = (np.zeros(len(a_tr), int), np.zeros(len(a_te), int))
    for pn, (ctr, cte) in parts.items():
        T = triage(d, ctr, cte)
        m = metrics(d, T, rare_norm, witness)
        m["n_codes"] = int(len(np.unique(ctr)))
        res["partitions"][pn] = m
    return res


def agg(results):
    """mean/std over seeds per partition per metric."""
    out = {}
    pn_all = results[0]["partitions"].keys()
    for pn in pn_all:
        out[pn] = {}
        for k in results[0]["partitions"][pn]:
            vals = [r["partitions"][pn][k] for r in results]
            if isinstance(vals[0], (int, float)):
                v = np.array(vals, float)
                out[pn][k] = dict(mean=round(float(np.nanmean(v)), 4), std=round(float(np.nanstd(v)), 4), per_seed=[round(float(x), 4) for x in v])
            elif k == "confusion_flagged":
                out[pn][k] = {r_: {c_: int(sum(v_[r_][c_] for v_ in vals)) for c_ in ("RARE_VALID", "FAULT")} for r_ in ("normal", "anomaly")}
            elif k == "rare_fp_missed_by":
                out[pn][k] = {g: round(float(np.nanmean([v_[g] for v_ in vals])), 3) for g in vals[0]}
    return out


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "HAI"
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else SEEDS
    nthreads = int(sys.argv[3]) if len(sys.argv) > 3 else 8   # must match the stored table's build (see stage)
    results, repro = [], []
    for sd in seeds:
        p = os.path.join(HERE, f"fable_triage_{name}_s{sd}.npz")
        if os.path.exists(p):
            z = np.load(p); d = {k: z[k] for k in z.files}; print(f"  [{name}] seed {sd} loaded from cache", flush=True)
        else:
            d = stage(name, sd, nthreads)
        r = check_repro(name, sd, d); repro.append(dict(seed=sd, **r))
        print(f"  [{name}] seed {sd} repro: {json.dumps(r)}", flush=True)
        res = analyze(name, sd, d); results.append(res)
        for pn, m in res["partitions"].items():
            print(f"    {pn:9s} codes={m['n_codes']:4d} rareFP={m['n_rare_norm_fp']:4d} recl={m['rare_norm_fp_reclassified']:.3f} "
                  f"anomRV(all)={m['anom_RV_of_all_anom']:.4f} anomRV(flag)={m['anom_RV_of_flagged_anom']:.4f} "
                  f"fpr {m['fpr_before']:.4f}->{m['fpr_after']:.4f} recall {m['recall_before']:.3f}->{m['recall_after']:.3f} "
                  f"witnessRV={m['witness_flagged_RV']:.3f} auroc {m['auroc_full']}->{m['auroc_full_excl_RV']}", flush=True)
        print(f"    witness ids={res['witness_ids']} n={res['witness_n']} runs={res['witness_runs']} rare_norm={res['n_rare_norm']}", flush=True)
        json.dump(dict(dataset=name, repro=repro, per_seed=results, agg=agg(results),
                       constants=dict(NMIN=NMIN, RARE_FRAC=RARE_FRAC, LRUN=LRUN, SPLIT_RATIO=SPLIT_RATIO, SPLIT_DEPTH=SPLIT_DEPTH, MS=MS)),
                  open(os.path.join(HERE, f"fable_triage_{name}.json"), "w"), indent=1)
    print("AGG:", json.dumps(agg(results), indent=1), flush=True)
