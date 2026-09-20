# SWaT_canon in-paper results recomputed on the CLEAN official pipeline

Scope: SWaT_canon only. HAI and WADI untouched. Nothing committed.
Pipeline: official Dec-2015 normal (0% train/test overlap, `_assert_swat_no_leak` guard),
`CLIP["SWaT_canon"]=10`, coverage-stable v2 difficulty axis (Easy/Difficult/DoubleHard = 142/91/31,
trivial max|z| difficult AUROC 0.627 invariant HOLDS).

Reporting set: community fusion (HCcoh+LatAD, and HC_coh) as the detector; baselines IF / AE / USAD /
TranAD / GDN / LinRes / boosted-LOO. Global-LatAD and trivial max|z| are shown only as ablation/floor
context rows, never as competitors.

Difficult subset changed with the leak fix: OLD-leaked 85 windows (maxz_thr 5.03) -> NEW-clean 91
windows (maxz_thr 3.163). All NEW baseline numbers reproduce `headline_full_SWaT_canon_OFFICIAL_v2.json`
exactly (verified by direct recompute from `scores_SWaT_canon.npz`).

---

## CRITICAL DISCREPANCY FOUND (verify-before-report catch)

`headline_full_SWaT_canon_OFFICIAL_v2.json` records the community fusion as **HC_coh 0.742 /
HCcoh+LatAD 0.703** on the difficult subset (and All 0.898/0.883), with a significance of diff 0.27 vs
LinRes. **These fusion numbers are NOT reproducible from the committed clean pipeline.**

- The fusion is fully determined by `sota_bundle/experts_full/expert_SWaT_canon.npz` + `scores_*.npz`.
  `ensemble_final.ensemble_scores` (the canonical fusion path, which matched the leaked headline to 3 dp)
  yields **HC_coh 0.523 / HCcoh+LatAD 0.524** on the difficult subset from the committed expert.
- I rebuilt the expert independently from the current clean official bundle
  (`ens_bundle/bundle_SWaT_canon.npz`, 1616 train windows, maxz_thr 3.163, canonical params
  method=average/MAXSZ=25/MINSZ=3 -> S=24 communities). It reproduces **HC_coh 0.523 / HCcoh+LatAD 0.524**,
  matching the committed expert exactly.
- The high number in OFFICIAL_v2 traces to a differently-trained expert: `experts_variants/cur_avg25`
  (S=25, built 2026-09-19 00:32, before the official-normal bundle) gives HC_coh 0.808. OFFICIAL_v2's
  0.742 sits in that inflated family, NOT the clean official (S=24) family.

**Conclusion:** the committed clean-official expert is correct; the OFFICIAL_v2 fusion/significance rows
are stale (built on a non-official-clean train) and overstate the result by ~0.18-0.22 AUROC on the
Difficult/All subsets. The corrected clean-official community fusion is reported below.

---

## 1. Table 5 - source of gain (difficult subset, cross-channel latent density vs channel-independent marginal product)

5-seed mean; NEW on the 91-window clean difficult subset (fresh VaDE, K=40/LD=16, verbatim arms from
`e5_gain_clean.py`). Invariant: no arm below chance except as a genuine collapse (corroborated by the
independent LatAD-global 0.472 in the scores npz).

| Density model | OLD-leaked SWaT | NEW-clean SWaT | direction |
|---|---|---|---|
| cross-channel latent density | 0.795 ± 0.010 | **0.472 ± 0.020** | collapses to chance |
| channel-independent product of marginals | 0.654 ± 0.002 | 0.522 ± 0.003 | down, still ~chance |
| **cross-channel gain** | **+0.141** | **-0.050** | **gain vanishes / inverts** |

The "source of gain" story for SWaT does not survive: on clean official data the single cross-channel
latent density is at chance on the difficult subset and does NOT beat the marginal product.
(`clean_swat_tables56.json`)

## 2. Table 6 - difficult-subset AUROC by score head (SWaT column)

5-seed mean; NEW on 91-window clean difficult subset.

| Score head | OLD-leaked SWaT | NEW-clean SWaT | direction |
|---|---|---|---|
| reconstruction residual (dropped term) | 0.835 ± 0.000 | 0.567 ± 0.004 | down (still highest head) |
| latent density | 0.794 ± 0.006 | 0.472 ± 0.020 | collapses to chance |
| nearest-component NLL | 0.723 ± 0.015 | 0.479 ± 0.025 | collapses to chance |
| base (density + nearest) | 0.775 ± 0.009 | 0.472 ± 0.019 | collapses to chance |
| base + resid (auto) = LatAD (global density) | 0.807 ± 0.008 | 0.472 ± 0.019 | collapses to chance |

Every head collapses to near chance on clean SWaT difficult; the whitened reconstruction residual is now
the strongest single head (0.567) but is itself only marginally above chance. (`clean_swat_tables56.json`)

## 3. Table 7 - subsystem localization (SWaT row, attack-episode level, 24 episodes, ground-truth targets)

NEW computed on the confirmed-clean expert (`framing_issue2_localization.py`; the paper §7 rule is the
train-normal p-value "rank" scorer, "raw" = argmax surprise). Coverage-aware random baseline.

| Metric | OLD-leaked (paper) | NEW-clean (raw) | NEW-clean (rank/§7) | random |
|---|---|---|---|---|
| Top-1 | 0.46 (11/24), P=0.004 | 0.25 (6/24), p=0.48 | 0.33 (8/24), p=0.16 | 0.229 |
| Top-3 | 0.79 | 0.708 | 0.458 | 0.542 |

On clean official data the SWaT top-1 localization is NOT significantly above the coverage-aware random
baseline under either scorer (p = 0.16-0.48). The paper's SWaT localization claim collapses.
(`framing_loc/loc2_summary.json`)

## 4. Density-vs-reconstruction framing numbers for SWaT

- Paper claim: "the density head alone beats USAD/TranAD on every difficult subset." OLD-leaked SWaT:
  density 0.794 vs USAD 0.658 / TranAD 0.655. **NEW-clean: density 0.472 vs USAD 0.477 / TranAD 0.477 -
  density no longer beats them (ties/marginally below).** The framing does not hold on clean SWaT.
- Paper: "on SWaT reconstruction is the strongest single head (0.835)." NEW-clean: recon is still the
  strongest head but drops to 0.567 (near chance).
- The deep reconstruction detectors did NOT stay strong: their difficult-subset AUROC drops from
  ~0.66 (leaked) to ~0.477 (clean), consistent with the leak having let them memorize test-normals.

## 5. SWaT significance statement feeding the paper

Paper text: "on canonical SWaT the lead is also significant (0.837 vs the linear baseline 0.782,
P = 0.007)." Reported detector = community fusion.

| Quantity | OLD-leaked | NEW-clean official (corrected) | (OFFICIAL_v2 file, non-reproducible) |
|---|---|---|---|
| community fusion HCcoh+LatAD, Difficult | 0.840 | **0.524** | 0.703 |
| community fusion HC_coh, Difficult | 0.822 | **0.523** | 0.742 |
| community fusion HCcoh+LatAD, All | 0.941 | **0.788** | 0.883 |
| LinRes, Difficult | 0.782 | 0.464 | 0.464 |
| best baseline (IF), Difficult | 0.627 | 0.550 | 0.550 |
| best baseline (IF), All | 0.821 | 0.800 | 0.800 |
| significance (fusion - LinRes, Difficult) | +0.058, P=0.007 | **+0.059 (vs LinRes); vs best baseline IF the fusion is BELOW it (0.524 < 0.55)** | +0.27 (inflated) |

On the clean official pipeline the SWaT headline evaporates: community fusion All 0.788 < IF 0.80
(no best-overall win), and Difficult 0.524 < IF 0.55 and < trivial floor 0.627 (no significant
difficult win). The SWaT "significant lead" does not hold.

### Full clean-official difficult-subset table (reproducible from committed artifacts)

| Method | OLD-leaked diff | NEW-clean diff |
|---|---|---|
| community fusion HCcoh+LatAD | 0.840 | 0.524 |
| community fusion HC_coh | 0.822 | 0.523 |
| IF | 0.627 | 0.550 |
| AE | 0.729 | 0.517 |
| USAD | 0.658 | 0.477 |
| TranAD | 0.655 | 0.477 |
| GDN | (n/a) | 0.473 |
| LinRes | 0.782 | 0.464 |
| boosted channel-wise LOO | 0.881 | **0.412** |
| _(ablation) LatAD global density_ | 0.804 | 0.472 |
| _(floor) trivial max\|z\|_ | 0.646 | 0.627 |

boosted-LOO: clean difficult **0.412** (was 0.881), doublehard 0.124 (was 0.830); linear-LOO invariant
reproduces `scores_SWaT_canon.npz['linres']` to rel-err 6e-8. (`clean_swat_boosted_loo.json`)

---

## A8 CLEAN RE-CHECK (valley smoothing on clean SWaT_canon)

`a8_valley_real.py SWaT_canon`, 3 seeds, clean regenerated `raw_SWaT_canon.npz` (train Xn = official
normal, 48510 rows). Per-partition AUROC vs all test-normals, seed-mean:

| Partition (n/seed) | LatAD/density | USAD | TranAD | AE | IF | LinRes |
|---|---|---|---|---|---|---|
| valley (16-21) | 0.281 | 0.260 | 0.246 | 0.331 | 0.495 | 0.362 |
| off-manifold (188-197) | 0.861 | 0.871 | 0.871 | 0.888 | 0.874 | 0.876 |
| in-cluster (15-17) | 0.151 | 0.146 | 0.142 | 0.170 | 0.304 | 0.139 |

Interaction (LatAD-deep)[valley] - (LatAD-deep)[off], per seed: vs USAD obs +0.020/+0.048/+0.024
(p_perm 0.50/0.10/0.41); vs TranAD +0.041/+0.058/+0.034 (p 0.15/0.05/0.22); vs AE negative. All CIs
include 0.

**Finding:** on clean SWaT the valley hypothesis is NOT supported. The expected pattern (deep baselines
miss valley anomalies while density/community catches them) does not appear: in the valley EVERY
detector, including LatAD/density (0.28), is near chance and is statistically indistinguishable from
USAD/TranAD (0.25-0.26); off-manifold everyone catches (~0.86-0.89). The clean deep-baseline collapse is
uniform across partitions, not valley-specific, so the SWaT valley evidence is null - it does not revive
the A8 valley finding. This confirms (does not re-open) the SWaT component of the "drop A8 valley"
verdict. (`a8_valley_real.jsonl`; leaked rows backed up to `a8_valley_real.jsonl.leaked_bak`)

---

## Artifacts written (this run, uncommitted)
- `_diagnostics/clean_swat_tables56.{py,json,log}` - Tables 5 & 6
- `_diagnostics/clean_swat_boosted_loo.{py,json,log}` - boosted-LOO
- `_diagnostics/framing_loc/loc2_summary.json` (+ per-episode jsonl) - Table 7 (overwrote leaked)
- `_diagnostics/a8_valley_real.jsonl` (SWaT rows rerun clean; leaked backup kept)
- `_diagnostics/experts_local_swat_cleanreb.log` + `sota_bundle/experts_variants/clean_reb/` -
  independent expert rebuild that confirmed the committed clean expert and exposed the OFFICIAL_v2 gap
- clean `raw_SWaT_canon.npz` regenerated in scratchpad (leak guard passed)
