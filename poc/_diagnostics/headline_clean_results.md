# Headline regime-community model on the CLEAN datasets — FULL-HEAD (construct-matched with HAI 0.849)

Decisive run of the paper's headline model **LatAD (regime-community)** on the two clean datasets,
scored on the canonical difficult (subtle-anomaly) subset against all baselines, 5-seed and
construct-matched. **FULL-HEAD config**: each per-community VaDE expert applies the whitened-residual
(A8) + basin (A3) heads inside the community (`modal_experts.py --full 1`, `use_full=1`), identical to
the config behind the HAI headline (0.849). This supersedes the earlier density-only run so a single
Table 3 does not mix two model variants.

Experts: `sota_bundle/experts_full/expert_{WADI_clean,SWaT_canon}.npz`. Scores via `ensemble_final.py`
with `EXPERTS_DIR=sota_bundle/experts_full`, `BOOT_REPS=2000`. JSONs (with real bootstrap CIs):
`_diagnostics/headline_full_{WADI_clean,SWaT_canon}.json` (null+HC head),
`headline_full_HCcoh_{WADI_clean,SWaT_canon}.json` (HCcoh+LatAD head).

Headline aggregators: **null+HC** = max(z(HC), z(LatAD null-expert)) (primary paper headline);
**HCcoh+LatAD** = z(cohesion-weighted HC) + z(LatAD) (density-fusion headline, the HAI-0.849 variant);
**cohmax+LatAD** shown as the strongest sibling.

Difficult subset = SUBTLE anomalies: `difficult=(y==1)&(maxz<=train_p99)`, scored against ALL normals
(`keep=(y==0)|difficult`). Double-hard additionally requires the linear co-regression residual below its
train-99th. Verified against the code before trusting output.

---

## WADI_clean  (575 windows, 56 anomalies, 5 seeds; Difficult n=43 / 11 episodes, DoubleHard n=27 / 10 episodes)

| method | Difficult AUROC | DoubleHard AUROC | All | Easy |
|---|---|---|---|---|
| linres (strongest classical) | 0.787±0.000 | 0.660±0.000 | 0.834 | 0.993 |
| AE | 0.739±0.003 | 0.671±0.003 | 0.799 | 0.999 |
| IF | 0.681±0.006 | 0.581±0.007 | 0.725 | 0.871 |
| USAD (5-seed) | 0.685±0.000 | 0.595±0.000 | 0.757 | 0.994 |
| TranAD (5-seed) | 0.722±0.000 | 0.629±0.000 | 0.786 | 0.998 |
| **global LatAD** | **0.734±0.008** | 0.682±0.014 | 0.793 | 0.991 |
| HC (community) | 0.789±0.008 | 0.688±0.017 | 0.831 | 0.969 |
| HC_coh | 0.805±0.012 | 0.706±0.022 | 0.843 | 0.968 |
| **null+HC (headline)** | **0.806±0.005** | **0.720±0.017** | 0.850 | 0.995 |
| **HCcoh+LatAD (headline)** | **0.824±0.012** | **0.743±0.024** | 0.864 | 0.996 |
| cohmax+LatAD | 0.828±0.016 | 0.754±0.027 | 0.867 | 0.996 |

Episode-block bootstrap (2000 reps) vs strongest classical baseline (linres):
- null+HC: Difficult diff +0.019, 95% CI [-0.108, 0.163], P(diff<=0)=0.41; DoubleHard diff +0.060, CI [-0.158, 0.250], P=0.30
- HCcoh+LatAD: Difficult diff +0.037, CI [-0.088, 0.181], P=0.32; DoubleHard diff +0.083, CI [-0.134, 0.267], P=0.22

**Verdict WADI_clean: the regime-community model LEADS.** null+HC 0.806 vs global LatAD 0.734 on
Difficult (+0.072); HCcoh+LatAD 0.824. The gain is genuine community signal (full-head HC alone 0.789 >
global 0.734). It beats every learned baseline (AE/IF/USAD/TranAD) and the global model on both subtle
subsets. Against the best classical baseline (linres 0.787) it is numerically ahead but the
episode-block bootstrap is underpowered (11 anomaly episodes, wide CI) and does not reach significance.

## SWaT_canon  (1498 windows, 233 anomalies, 5 seeds; Difficult n=85 / 23 episodes, DoubleHard n=59 / 18 episodes)

| method | Difficult AUROC | DoubleHard AUROC | All | Easy |
|---|---|---|---|---|
| linres (strongest classical) | 0.782±0.000 | 0.688±0.000 | 0.920 | 1.000 |
| AE | 0.729±0.007 | 0.666±0.010 | 0.899 | 0.996 |
| IF | 0.627±0.007 | 0.578±0.009 | 0.821 | 0.933 |
| USAD (5-seed) | 0.658±0.000 | 0.609±0.000 | 0.873 | 0.996 |
| TranAD (5-seed) | 0.655±0.000 | 0.604±0.000 | 0.873 | 0.998 |
| **global LatAD** | **0.804±0.007** | 0.728±0.010 | 0.927 | 0.998 |
| HC (community) | 0.811±0.011 | 0.735±0.014 | 0.918 | 0.980 |
| HC_coh | 0.822±0.014 | 0.757±0.015 | 0.919 | 0.974 |
| **null+HC (headline)** | **0.823±0.011** | **0.752±0.016** | 0.935 | 0.999 |
| **HCcoh+LatAD (headline)** | **0.840±0.004** | **0.775±0.006** | 0.941 | 0.999 |
| cohmax+LatAD | 0.837±0.004 | 0.775±0.005 | 0.939 | 0.998 |

Episode-block bootstrap (2000 reps) vs linres:
- null+HC: Difficult diff +0.040, CI [-0.001, 0.092], P=0.0275; DoubleHard diff +0.065, CI [0.008, 0.134], P=0.0145 (significant)
- **HCcoh+LatAD: Difficult diff +0.058, CI [0.017, 0.107], P=0.0025; DoubleHard diff +0.087, CI [0.030, 0.153], P=0.0005 (significant)**

**Verdict SWaT_canon: the regime-community model LEADS.** With the full-head experts, null+HC 0.823
now beats global LatAD 0.804 on Difficult (+0.019, was a tie under density-only), because full-head HC
(0.811) overtakes the global model. HCcoh+LatAD 0.840 / cohmax+LatAD 0.837 lead global by +0.036/+0.033
on Difficult and +0.047 on DoubleHard, and the margin over the strongest classical baseline (linres)
is significant on both Difficult (HCcoh P=0.0025; null+HC P=0.028) and DoubleHard (HCcoh P=0.0005).
Beats every learned baseline on both subtle subsets.

---

## Bottom line (FULL-HEAD, Difficult subset)

| dataset | global LatAD | null+HC | HCcoh+LatAD | cohmax+LatAD | verdict |
|---|---|---|---|---|---|
| WADI_clean | 0.734 | 0.806 (+0.072) | 0.824 | 0.828 | **LEADS** global (bootstrap vs linres underpowered) |
| SWaT_canon | 0.804 | 0.823 (+0.019) | 0.840 | 0.837 | **LEADS** global; significant vs linres (P<=0.003) |
| HAI (ref, full-head) | — | — | 0.849 | — | construct-matched config |

The three HCcoh+LatAD headline numbers — WADI_clean 0.824, SWaT_canon 0.840, HAI 0.849 — now all come
from the same full-head, per-community, whitened-residual config and belong in one Table 3.

The regime-community model LEADS the global model on the subtle subset of BOTH clean datasets. The lead
is large on high-dimensional WADI_clean (+0.072 headline), where subtle faults localize to a correlated
sub-system the whole-system model dilutes, and smaller but statistically significant on SWaT_canon.

### Full-head vs density-only (the audit's expectation)
The move is small, as predicted, and never negative:

| head, Difficult | WADI density-only → full | SWaT density-only → full |
|---|---|---|
| null+HC | 0.799 → 0.806 (+0.007) | 0.803 → 0.823 (+0.020) |
| HCcoh+LatAD | 0.816 → 0.824 (+0.008) | 0.827 → 0.840 (+0.013) |
| cohmax+LatAD | 0.824 → 0.828 (+0.004) | 0.828 → 0.837 (+0.009) |

WADI matches the audit's "negligible" expectation (~+0.007). SWaT_canon moves up more (~+0.02 on
null+HC): the per-community whitened-residual head adds real signal there, enough to flip null+HC from
a tie with global to a lead. Reported plainly rather than rounded away.

## Provenance / construct-matching

- **Full-head experts rebuilt** for both clean datasets via `modal_experts.py --full 1` (use_full=1,
  whitened-residual + basin heads inside each community), written to `sota_bundle/experts_full/`
  (WADI_clean S=44, SWaT_canon S=25, 5 seeds each), same schema and training as the existing HAI/WADI/SWaT
  full-head experts. Expert `y` equals scores `label` exactly, both 5 seeds, same window grid →
  construct-matched to the baselines and to the HAI 0.849 config.
- Bundles (`sota_bundle/ens_bundle/bundle_{WADI_clean,SWaT_canon}.npz`) built from
  `E.load('WADI_clean'|'SWaT_canon')` and verified label-for-label + maxz-for-maxz against `scores_*.npz`.
- USAD/TranAD (5 seeds, 30 epochs) were added to `scores_SWaT_canon.npz` earlier (per-timestep dumps of
  length 44992 = len(Xa_raw), seed-averaged and re-windowed W=60/stride=30 exactly as build_scores_table).
- **No science modified.** Config-key additions only: dataset entries in `ensemble_final.py` (CFG/COMPET),
  the bundle-upload loops in `modal_experts.py` / `modal_sota.py`. All AUROCs re-derived from the same
  subset masks the main tables use; the global-LatAD WADI number reproduces the known 0.734 exactly,
  cross-validating the pipeline; the only significant bootstrap wins are on SWaT_canon and are reported
  as such, with WADI_clean's underpowered CIs stated plainly.
