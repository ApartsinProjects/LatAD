"""Issue 1 (framing brainstorm): does a per-dataset "reconstructable-but-improbable" (RBI) index
predict the density-minus-reconstruction head gap on the difficult subset?

Inputs (all pre-existing artifacts, nothing retrained):
  _diagnostics/heads_<ds>.npz   : per-seed head scores (recon, density, ...) on the test windows,
                                  same 5-seed run behind Table 6; label, hard
  _diagnostics/scores_<ds>.npz  : windowed external baselines (USAD, TranAD, AE, linres), maxz, maxz_thr
Difficult subset = (label==1) & (maxz <= maxz_thr)  [identical to the 'hard' mask]; scored vs ALL normals.

Indices computed per dataset:
  gap            = AUROC(density) - AUROC(recon)   (Table 6 head level, seed-mean)
  ext_recon      = mean AUROC of the external reconstruction detectors USAD, TranAD on the difficult subset
  RBI_ext        = fraction of difficult anomalies whose EXTERNAL recon score (USAD/TranAD mean rank) is
                   at or below the normal median (reconstructs as well as a typical normal window) while
                   the LatAD density head puts them above the normal 90th percentile (improbable)
  IBR_ext        = the converse (unreconstructable but probable)
  RBI_head/IBR_head: same, using the in-model recon head (near-tautological with gap, reported as decomposition)
Writes _diagnostics/framing_issue1_index.json
"""
from __future__ import annotations
import json, os
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DS = ["WADI_clean", "HAI", "SWaT_canon"]
PAPER_NAME = {"WADI_clean": "WADI", "HAI": "HAI", "SWaT_canon": "SWaT"}


def pct_vs_normal(score, normal_mask):
    """empirical percentile of each window's score within the normal-window score distribution."""
    ref = np.sort(score[normal_mask])
    return np.searchsorted(ref, score, side="right") / len(ref)


out = {}
for ds in DS:
    h = np.load(os.path.join(HERE, f"heads_{ds}.npz"), allow_pickle=True)
    s = np.load(os.path.join(HERE, f"scores_{ds}.npz"), allow_pickle=True)
    y = h["label"].astype(int)
    # canonical difficult mask = scores_<ds>.npz maxz <= train-p99 (WADI_clean: the FIX3 30-window mask;
    # heads_WADI_clean.npz still carries the pre-fix 43-window mask, so the scores mask is authoritative)
    hard = (s["label"].astype(int) == 1) & (s["maxz"] <= float(s["maxz_thr"]))
    assert np.array_equal(y, s["label"].astype(int)), ds
    if not np.array_equal(hard, h["hard"].astype(bool)):
        print(f"  note {ds}: heads npz hard mask n={int(h['hard'].sum())} differs from canonical n={int(hard.sum())}; using canonical")
    keep = (y == 0) | hard
    normal = y == 0
    res = dict(n_difficult=int(hard.sum()), n_normal=int(normal.sum()))

    # head-level AUROCs (5-seed mean, matches Table 6)
    def auc_seeds(arr):
        return [roc_auc_score(y[keep], np.nan_to_num(a)[keep]) for a in arr]
    a_rec = auc_seeds(h["recon"]); a_den = auc_seeds(h["density"])
    res["auroc_recon_head"] = float(np.mean(a_rec)); res["auroc_density_head"] = float(np.mean(a_den))
    res["gap_density_minus_recon"] = res["auroc_density_head"] - res["auroc_recon_head"]
    res["gap_per_seed"] = [float(d - r) for d, r in zip(a_den, a_rec)]

    # external reconstruction detectors on the SAME difficult subset
    ext = {}
    for b in ["USAD", "TranAD", "AE"]:
        arr = s[b]
        arr = arr if arr.ndim == 1 else arr.mean(0)
        ext[b] = float(roc_auc_score(y[keep], np.nan_to_num(arr)[keep]))
    res["auroc_external"] = ext
    res["ext_recon_index"] = float(np.mean([ext["USAD"], ext["TranAD"]]))
    res["ext_recon_index_incl_AE"] = float(np.mean([ext["USAD"], ext["TranAD"], ext["AE"]]))

    # per-window quadrant decomposition of the difficult anomalies
    den = h["density"].mean(0); rec = h["recon"].mean(0)
    p_den = pct_vs_normal(den, normal); p_rec = pct_vs_normal(rec, normal)
    # external recon rank = mean of USAD/TranAD percentiles vs normal
    p_ext = np.mean([pct_vs_normal(np.nan_to_num(s[b] if s[b].ndim == 1 else s[b].mean(0)), normal)
                     for b in ["USAD", "TranAD"]], axis=0)
    d = hard
    def quad(pr, pd, lo=0.5, hi=0.9):
        return dict(RBI=float(np.mean((pr[d] <= lo) & (pd[d] >= hi))),   # reconstructable, improbable
                    IBR=float(np.mean((pd[d] <= lo) & (pr[d] >= hi))),   # unreconstructable, probable
                    both=float(np.mean((pr[d] >= hi) & (pd[d] >= hi))),
                    neither=float(np.mean((pr[d] < hi) & (pd[d] < hi))),
                    recon_seen=float(np.mean(pr[d] >= hi)), density_seen=float(np.mean(pd[d] >= hi)))
    res["quadrants_external_recon"] = quad(p_ext, p_den)
    res["quadrants_head_recon"] = quad(p_rec, p_den)
    # median percentile of difficult anomalies under each score (how "normal-looking" they are)
    res["median_pct_difficult"] = dict(ext_recon=float(np.median(p_ext[d])), head_recon=float(np.median(p_rec[d])),
                                       density=float(np.median(p_den[d])))
    out[ds] = res
    print(f"{ds}: gap={res['gap_density_minus_recon']:+.3f} (dens {res['auroc_density_head']:.3f} / rec {res['auroc_recon_head']:.3f})"
          f"  ext_recon={res['ext_recon_index']:.3f} {ext}  RBI_ext={res['quadrants_external_recon']['RBI']:.2f}"
          f" IBR_ext={res['quadrants_external_recon']['IBR']:.2f}  RBI_head={res['quadrants_head_recon']['RBI']:.2f}"
          f" IBR_head={res['quadrants_head_recon']['IBR']:.2f}  med pct: {res['median_pct_difficult']}")

# cross-dataset association (n=3: report, do not over-read)
gaps = [out[d]["gap_density_minus_recon"] for d in DS]
for key, vals in [("ext_recon_index", [out[d]["ext_recon_index"] for d in DS]),
                  ("RBI_ext_minus_IBR_ext", [out[d]["quadrants_external_recon"]["RBI"] - out[d]["quadrants_external_recon"]["IBR"] for d in DS]),
                  ("RBI_ext", [out[d]["quadrants_external_recon"]["RBI"] for d in DS])]:
    rho = spearmanr(vals, gaps).correlation
    out.setdefault("cross_dataset", {})[key] = dict(values=dict(zip(DS, vals)), gaps=dict(zip(DS, gaps)), spearman=float(rho))
    print(f"cross-dataset {key}: {dict(zip(DS, np.round(vals,3)))} vs gaps {np.round(gaps,3)} spearman={rho:+.2f}")

json.dump(out, open(os.path.join(HERE, "framing_issue1_index.json"), "w"), indent=1)
print("saved framing_issue1_index.json")
