"""Quantify the fusion-calibration leak in ensemble_final.py (lines 119-139).

ensemble_final.ensemble_scores() z-scores the HC/HC_coh aggregate and builds the LatAD null tail
against nm = (y == 0), i.e. the TEST-normal windows, selected with the test labels. The headline
HCcoh+LatAD = z(hc_coh) + z(surv(lat[nm], lat)) therefore takes its fusion SCALE from test-normal
statistics. This script recomputes the headline under alternative calibrations and reports the
difficult-subset AUROC delta:

  test_normal  : current code (must reproduce 0.824 / 0.849 / 0.840 exactly -> invariant I1)
  test_all     : label-free transductive (all test windows as reference; no label use)
  train_cal    : experts' held-out train-normal calibration slice for HC/HC_coh (Cal surprises are
                 stored in expert_<DS>.npz) + LatAD null tail against the global model's own
                 TRAIN-window scores (retrained here, identical config to build_scores_table.py)
  train_heldout: (small datasets only) global model retrained on the first 80% of train windows,
                 LatAD tail calibrated on the last 20% (fully held-out; the test score changes too)

Also: weight sweep a*z(hc_coh)+zl under train_cal, to show how sensitive the fused ranking is to
the relative scale at all. Persisted incrementally: retrained train scores per seed to
fable_leak_trainscores_<DS>.npz (resumable), results to fable_leak_calib_<DS>.json.

Invariants: I1 test_normal reproduces the stored headline numbers to 3 decimals.
            I2 retrained test scores give the same difficult AUROC as scores_<DS>.npz['LatAD'] per seed
               (within +-0.01; torch CPU determinism across machines is not guaranteed).
            I3 HC_coh alone (no calibration) is identical across all variants.
"""
from __future__ import annotations
import json, os, sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import roc_auc_score
import eda_real as E
from models_vade import train_vade
from ensemble_final import surv, pval, HC, episodes, boot, OUT

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {"WADI_clean": (20, 10), "SWaT_canon": (40, 16), "HAI": (40, 16)}
HEADLINE = {"WADI_clean": 0.824, "SWaT_canon": 0.840, "HAI": 0.849}
REPS = int(os.environ.get("BOOT_REPS", "2000"))


def train_scores(name, seeds, frac=1.0):
    """Retrain the global VaDE (build_scores_table config) per seed; return (test, train_insample,
    heldout) LatAD scores. frac<1 -> fit on the first frac of train windows, held-out = the rest."""
    tag = "" if frac == 1.0 else f"_f{int(frac*100)}"
    path = os.path.join(HERE, f"fable_leak_trainscores_{name}{tag}.npz")
    store = dict(np.load(path)) if os.path.exists(path) else {}
    D = E.load(name)
    Xtr0, Xte0 = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32)
    nfit = int(len(Xtr0) * frac)
    Xfit0 = Xtr0[:nfit]
    mu, sig = Xfit0.mean(0), Xfit0.std(0) + 1e-8            # identical to build_scores_table
    Xfit = ((Xfit0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
    Xall = ((Xtr0 - mu) / sig).astype(np.float32)
    K, LD = CFG[name]; kd = min(80, max(20, len(Xfit0) // 10))
    for sd in seeds:
        if f"test_{sd}" in store:
            continue
        t0 = time.time()
        v = train_vade(Xfit, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
        v.fit_residual_whitener(Xfit); v.fit_latent_density(Xfit, k_density=kd)
        v.fit_resid_head(Xfit); v.fit_basin_head(Xfit)
        store[f"test_{sd}"] = np.asarray(v.anomaly_score_hard(Xte, use_resid="auto", use_basin="auto"), np.float32)
        store[f"train_{sd}"] = np.asarray(v.anomaly_score_hard(Xall, use_resid="auto", use_basin="auto"), np.float32)
        store["nfit"] = np.int64(nfit)
        np.savez(path, **store)                              # flush per seed (resumable)
        print(f"  [{name}{tag}] seed {sd} retrained in {time.time()-t0:.0f}s  resid_auto={v._resid_auto} "
              f"ratio={v._resid_gen_ratio:.2f}", flush=True)
    return store


def fuse(name, hc_raw, hccoh_raw, cohmax_raw, lat_test, ref_hc, ref_hccoh, ref_cohmax, ref_lat, a=1.0):
    """Build the fused scores given calibration references (arrays of reference-window values)."""
    z = lambda s, r: (s - r.mean()) / (r.std() + 1e-9)
    tail = surv(ref_lat, lat_test); tail_ref = surv(ref_lat, ref_lat)
    zl = z(tail, tail_ref)
    return {"HCcoh+LatAD": a * z(hccoh_raw, ref_hccoh) + zl,
            "null+HC": np.maximum(z(hc_raw, ref_hc), zl),
            "cohmax+LatAD": a * z(cohmax_raw, ref_cohmax) + zl,
            "HC+LatAD": a * z(hc_raw, ref_hc) + zl,
            "HC_coh": hccoh_raw, "LatAD": lat_test}


def run(name):
    d = np.load(f"{OUT}/scores_{name}.npz"); y = d["label"].astype(int)
    Ex = np.load(f"sota_bundle/experts_full/expert_{name}.npz", allow_pickle=True)
    assert np.array_equal(Ex["y"].astype(int), y)
    Cal, Tst = Ex["calib_surprise"], Ex["test_surprise"]
    coh = Ex["comm_cohesion"].astype(float); size = Ex["comm_size"].astype(float); w = coh * np.sqrt(size)
    wn = w / (w.max() + 1e-9)
    lat = d["LatAD"]; nseed = min(Tst.shape[0], lat.shape[0]); S = Tst.shape[1]
    mthr = float(d["maxz_thr"]); maxz = d["maxz"]
    easy = (y == 1) & (maxz > mthr); diff = (y == 1) & (maxz <= mthr)
    # double-hard mask (same construction as ensemble_final.run)
    from onehot_filter import build_feats, loco_residual
    fn, W, stride = E.RAW[name]; Dd = E.load(name)
    Fn, Fa, grp = build_feats(np.asarray(Dd["Xn_raw"], float), np.asarray(Dd["Xa_raw"], float), W, stride, onehot=True)
    r_tr, r_te = loco_residual(Fn, Fa[:len(y)], grp); lin_thr = float(np.quantile(r_tr, 0.99))
    dhard = diff & (r_te <= lin_thr)
    subsets = {"All": (y == 1), "Easy": easy, "Difficult": diff, "DoubleHard": dhard}
    L = int(np.ceil(W / stride)) + 1
    linres = d["linres"]

    # per-seed raw aggregates on test and on the calib slice
    raw = {"test": [], "cal": []}
    for sd in range(nseed):
        for split, A in (("test", Tst), ("cal", Cal)):
            tails = np.stack([surv(Cal[sd, g], A[sd, g]) for g in range(S)])
            P = np.stack([pval(Cal[sd, g], A[sd, g]) for g in range(S)])
            raw[split].append(dict(hc=HC(P), hccoh=HC(P, wt=w), cohmax=(wn[:, None] * tails).max(0)))

    TS = train_scores(name, range(nseed))
    variants = {}
    nm = y == 0
    for sd in range(nseed):
        rt, rc = raw["test"][sd], raw["cal"][sd]
        lt = lat[sd]; ltr = TS[f"train_{sd}"]; lt_re = TS[f"test_{sd}"]
        V = {
            "test_normal": fuse(name, rt["hc"], rt["hccoh"], rt["cohmax"], lt,
                                rt["hc"][nm], rt["hccoh"][nm], rt["cohmax"][nm], lt[nm]),
            "test_all": fuse(name, rt["hc"], rt["hccoh"], rt["cohmax"], lt,
                             rt["hc"], rt["hccoh"], rt["cohmax"], lt),
            "train_cal": fuse(name, rt["hc"], rt["hccoh"], rt["cohmax"], lt,
                              rc["hc"], rc["hccoh"], rc["cohmax"], ltr),
            "train_cal_retrained_test": fuse(name, rt["hc"], rt["hccoh"], rt["cohmax"], lt_re,
                                             rc["hc"], rc["hccoh"], rc["cohmax"], ltr),
        }
        for a in (0.25, 0.5, 2.0, 4.0):
            V[f"train_cal_a{a}"] = fuse(name, rt["hc"], rt["hccoh"], rt["cohmax"], lt,
                                        rc["hc"], rc["hccoh"], rc["cohmax"], ltr, a=a)
        for k, sc in V.items():
            variants.setdefault(k, {m: [] for m in sc})
            for m, s in sc.items():
                variants[k][m].append(np.asarray(s, np.float64))

    res = {"name": name, "nseed": nseed, "n_subset": {s: int(m.sum()) for s, m in subsets.items()},
           "n_episodes": {s: len([e for e in episodes(y) if m[e].any()]) for s, m in subsets.items()},
           "invariants": {}, "variants": {}}
    # I2: retrained test vs stored per-seed difficult AUROC
    keep = np.where((y == 0) | diff)[0]
    i2 = [(round(float(roc_auc_score(y[keep], lat[sd][keep])), 4),
           round(float(roc_auc_score(y[keep], TS[f'test_{sd}'][keep])), 4),
           round(float(np.corrcoef(np.argsort(np.argsort(lat[sd])), np.argsort(np.argsort(TS[f'test_{sd}'])))[0, 1]), 4))
          for sd in range(nseed)]
    res["invariants"]["I2_stored_vs_retrained_difficult_auroc_and_rankcorr"] = i2
    for k, M in variants.items():
        res["variants"][k] = {}
        for m, lst in M.items():
            arr = np.stack(lst); row = {}
            for s, mk in subsets.items():
                kp = np.arange(len(y)) if s == "All" else np.where((y == 0) | mk)[0]
                aus = [roc_auc_score(y[kp], arr[i][kp]) for i in range(arr.shape[0])]
                row[s] = [round(float(np.mean(aus)), 3), round(float(np.std(aus)), 3)]
            res["variants"][k][m] = row
        if k in ("test_normal", "test_all", "train_cal", "train_cal_retrained_test"):
            arr = np.stack(M["HCcoh+LatAD"])
            res["variants"][k]["sig_HCcoh+LatAD_vs_linres"] = {
                "Difficult": boot(y, arr, linres, diff, L, reps=REPS),
                "DoubleHard": boot(y, arr, linres, dhard, L, reps=REPS)}
        json.dump(res, open(os.path.join(HERE, f"fable_leak_calib_{name}.json"), "w"), indent=1)
        print(f"[{name}] {k:28} HCcoh+LatAD Difficult {res['variants'][k]['HCcoh+LatAD']['Difficult']} "
              f"null+HC {res['variants'][k]['null+HC']['Difficult']} HC_coh {res['variants'][k]['HC_coh']['Difficult']}", flush=True)
    res["invariants"]["I1_headline_reproduced"] = (res["variants"]["test_normal"]["HCcoh+LatAD"]["Difficult"][0] == HEADLINE[name])
    json.dump(res, open(os.path.join(HERE, f"fable_leak_calib_{name}.json"), "w"), indent=1)

    # train_heldout (small datasets): 80/20 global retrain, tail calibrated on held-out 20%
    if len(y) < 3000 or os.environ.get("FORCE_HELDOUT"):
        TH = train_scores(name, range(nseed), frac=0.8); nfit = int(TH["nfit"])
        M = {m: [] for m in ("HCcoh+LatAD", "null+HC", "LatAD")}
        for sd in range(nseed):
            rt, rc = raw["test"][sd], raw["cal"][sd]
            ho = TH[f"train_{sd}"][nfit:]
            sc = fuse(name, rt["hc"], rt["hccoh"], rt["cohmax"], TH[f"test_{sd}"],
                      rc["hc"], rc["hccoh"], rc["cohmax"], ho)
            for m in M: M[m].append(sc[m])
        res["variants"]["train_heldout80"] = {}
        for m, lst in M.items():
            arr = np.stack(lst); row = {}
            for s, mk in subsets.items():
                kp = np.arange(len(y)) if s == "All" else np.where((y == 0) | mk)[0]
                aus = [roc_auc_score(y[kp], arr[i][kp]) for i in range(arr.shape[0])]
                row[s] = [round(float(np.mean(aus)), 3), round(float(np.std(aus)), 3)]
            res["variants"]["train_heldout80"][m] = row
        arr = np.stack(M["HCcoh+LatAD"])
        res["variants"]["train_heldout80"]["sig_HCcoh+LatAD_vs_linres"] = {
            "Difficult": boot(y, arr, linres, diff, L, reps=REPS), "DoubleHard": boot(y, arr, linres, dhard, L, reps=REPS)}
        print(f"[{name}] train_heldout80  HCcoh+LatAD Difficult {res['variants']['train_heldout80']['HCcoh+LatAD']['Difficult']}", flush=True)
        json.dump(res, open(os.path.join(HERE, f"fable_leak_calib_{name}.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["WADI_clean", "SWaT_canon", "HAI"]):
        run(nm)
    print("DONE", flush=True)
