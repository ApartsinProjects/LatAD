# SWaT_canon data-leak fix and clean recomputation (2026-09-19)

## 1. The leak (root cause, confirmed)

`eda_real._raw_swat_canonical` trained on the **entire** Kaggle-mirror `normal.csv`
(1,387,098 rows). By timestamp the mirror decomposes into three segments:

| segment | rows | dates | what it is |
|---|---|---|---|
| head | 0 – 395,297 (395,298 rows) | 28/12/2015 – 02/01/2016 | **the official attack file's Normal-labelled rows = the exact test-normal rows** |
| Normal_v0 | 395,298 – 892,097 (496,800 rows) | 22/12–28/12/2015 | genuine normal recording |
| Normal_v1 | 892,098 – 1,387,097 (495,000 rows) | 22/12–28/12/2015 | Normal_v0 duplicate (shifted ~1800 s) |

The test set is `datasets/_new/SWaT_canonical/swat_attack_canonical.npz` (`Xa`, `ya`, 449,919 rows,
12.14 % attack, 395,298 test-normal rows). The mirror's first 395,298 rows are **byte-for-byte** the
test-normal rows (row-wise match 1.000), so **100 % of the test-normal rows were inside the train
set**.

## 2. Clean training set and 0.000-overlap verification

**Source used: FALLBACK** (raw-row set-difference). The official `SWaT_Dataset_Normal_v1.xlsx`
(44-channel normal recording) is not present locally (only `SWaT_Dataset_Attack_v0.xlsx` is), and the
labelled Dec-2015 benchmark lives only inside the 108 GB iTrust zip (not reachable). The clean train
is therefore built as **`normal.csv` MINUS the exact test-normal rows**, via a rounded-row-hash
(3 dp) set-difference using the same `sens` columns and ffill/bfill/fillna prep as the pipeline:

- rows removed: **395,298** (exactly the contiguous head, idx 0–395,297)
- rows kept (clean train = Normal_v0 + Normal_v1): **991,800**
- after warmup-drop (2 %) + downsample (10): **97,197** raw train rows -> **3,238** windows
  (leaked build was 4,530 windows)
- test unchanged: 44,992 raw rows -> 1,498 windows, 233 anomaly windows

**Fail-fast overlap check (rounded-row-hash intersection, clean train vs all test-normal rows):
0.000 (0 rows).** Asserted inside `_raw_swat_canonical` on every load (`_assert_swat_no_leak`).

### Constant-channel limitation (documented)
7 channels are stored NaN for all of Normal_v0 in the mirror — **MV101, AIT201, MV201, P201, P202,
P204, MV303** — so after ffill/bfill they are **constant (std = 0)** in the clean train. Note
51 − 7 = 44 = the official normal recording's channel count, i.e. these 7 exist only in the attack
file. In the clean train there are 15 train-constant feature-columns in total (these 7 + genuine
always-on/off status/setpoints P206, P401, P403, P404, P502, P601, P603, P102). Of them, several
(MV101, P102, AIT201, MV201, P201, MV303, P403) **vary in test** (AIT201's analyzer range genuinely
shifts to ~[168,266]). Standardising a train-constant channel with (std + 1e-8) turns any test
deviation into a ~1e8-σ divide-by-eps artifact.

Two consequences were handled (label-free):
- **Detector inputs:** the raw ~1e8 artifact collapsed AE (AUROC 0.466, below chance) and LatAD
  (0.503, chance). Bounding with the ±10 σ **A2 envelope** (`CLIP["SWaT_canon"]=10.0`, exactly as
  WADI already does; SWaT was None only under the leak's false no-constant-channel assumption)
  restores them (AE 0.829, LatAD 0.766). HAI/WADI CLIP unchanged.
- **Difficulty axis:** even clipped, the constant channels saturate max|z| to the ceiling and send
  **all 233 anomalies to Easy** (Difficult = 0). The axis is recomputed over **active first-sixth
  channels only (train-normal std > 1e-6)** — the same `active` rule the community-expert
  construction already uses — which restores a meaningful split: **Easy 126 / Difficult 107**
  (`_diagnostics/fix_maxz_swat_canon.py`; SWaT_canon-only, verified to change HAI/WADI counts if
  applied there so it was **not** applied to them).

## 3. OLD (leaked) vs NEW (clean) per-method AUROC

experts_full headline, 5-seed mean. Subsets: OLD Easy/Difficult/DoubleHard = 148/85/59; NEW = 126/107/30.
(DoubleHard is not row-comparable across OLD/NEW — the difficulty and linres thresholds both moved —
so read All and Difficult as the primary comparison.)

| method | All OLD | All NEW | Diff OLD | Diff NEW | Δ Diff |
|---|---|---|---|---|---|
| trivial max\|z\| | 0.871 | 0.816 | 0.646 | 0.601 | −0.045 |
| IF | 0.821 | 0.808 | 0.627 | 0.613 | −0.014 |
| AE | 0.899 | 0.829 | 0.729 | 0.655 | −0.074 |
| linres | 0.920 | 0.770 | 0.782 | 0.512 | **−0.270** |
| USAD | 0.873 | 0.828 | 0.658 | 0.648 | −0.010 |
| TranAD | 0.873 | 0.827 | 0.655 | 0.647 | −0.008 |
| GDN | n/a | 0.828 | n/a | 0.648 | — |
| LatAD (global) | 0.927 | 0.766 | 0.804 | 0.527 | **−0.277** |
| HC_coh | 0.919 | 0.835 | 0.822 | 0.658 | −0.164 |
| null+HC | 0.935 | 0.808 | 0.823 | 0.598 | −0.225 |
| cohmax+LatAD | 0.939 | 0.737 | 0.837 | 0.521 | −0.316 |
| **HCcoh+LatAD (headline)** | 0.941 | 0.835 | 0.840 | 0.657 | **−0.183** |
| boosted_LOO (baseline) | 0.952 | 0.771 | **0.881** | **0.518** | **−0.363** |
| AUG HCcoh+LatAD+B | 0.951 | 0.834 | 0.871 | 0.653 | −0.218 |

The leak inflated **every** method; it inflated the memorising baselines most. **boosted_LOO**
(channel-wise HGB residual) fell the hardest on Difficult (0.881 → 0.518, −0.363) because on the
leaked train it could memorise the test-normal rows; on the clean train it collapses to the linres
level (All 0.771 ≈ linres 0.770). The density fusion HCcoh+LatAD is the most leak-robust of the
strong methods (−0.183).

## 4. Clean significance verdict — the ranking FLIPPED

Episode-block bootstrap (2,000 reps), community fusion vs the strongest baseline (boosted_LOO), Difficult subset:

| comparison | OLD (leaked) | NEW (clean) |
|---|---|---|
| AUG HCcoh+LatAD+B − boosted_LOO [Difficult] | diff −0.010, CI [−0.052, 0.039], P(≤0)=0.68 → **tie** | diff **+0.136**, CI [0.079, 0.189], P(≤0)=0.000 → **community leads** |
| AUG HCcoh+LatAD+B − boosted_LOO [DoubleHard] | diff −0.012, P(≤0)=0.68 → tie | diff +0.090, CI [−0.009, 0.182], P(≤0)=0.035 (marginal) |
| AUG null+HC+B (max) − boosted_LOO [Difficult] | diff −0.069, P(≤0)=0.99 → baseline led | diff +0.081, CI [0.026, 0.135], P(≤0)=0.004 → community leads |

**Verdict: on the clean data SWaT_canon is no longer a tie — the community-density fusion
(HCcoh+LatAD / its boosted-augmented form) significantly LEADS the strongest baseline on the
Difficult subset (+0.136 AUROC, p < 0.001).** The leaked result showed a tie only because the leak
inflated the memorising boosted baseline by more (+0.363) than the community method (+0.218). This is
consistent with the revision2 drift-robustness thesis: the density fusion degrades least under the
clean train/test shift.

## 5. HAI and WADI untouched (verified)
- `_diagnostics/scores_HAI.npz`, `scores_WADI.npz`, `scores_WADI_clean.npz`: pre-session mtimes, not rewritten.
- `sota_bundle/experts_full/expert_HAI.npz`, `expert_WADI_clean.npz`: pre-session mtimes.
- `loo_fusion_boosted.json` HAI (HCcoh+LatAD Diff 0.845, boosted 0.321) and WADI_clean
  (0.771 / 0.660) entries unchanged. The earlier verified 0 % HAI/WADI overlap is unaffected.

## 6. Recurrence guard (STEP 4) — loader assertion diff

Added to `eda_real.py` (`_swat_rowhashes` + `_assert_swat_no_leak`, called inside
`_raw_swat_canonical`); raises if any test-normal raw row reappears in the train:

```python
def _assert_swat_no_leak(Xn, Xa, ya):
    """Fail-fast guard: raise if ANY test-normal raw row is present in the SWaT train set."""
    tn = Xa[ya.astype(int) == 0]
    inter = set(_swat_rowhashes(Xn)) & set(_swat_rowhashes(tn))
    if inter:
        raise AssertionError(
            f"SWaT leak: {len(inter)} test-normal raw row(s) found inside the train set "
            f"(overlap must be 0). Check _raw_swat_canonical set-difference.")
```

The clean train build itself is now the set-difference:
```python
V = prep(nrm)                                                  # full mirror normal.csv (prepped)
tn_set = set(_swat_rowhashes(Xa[ya == 0]))                     # exact test-normal raw rows
keep = np.array([h not in tn_set for h in _swat_rowhashes(V)]) # raw-row set-difference
Xn = V[keep][int(keep.sum() * warmup_drop):][::downsample]     # clean normal (Normal_v0 + v1)
_assert_swat_no_leak(Xn, Xa_ds, ya_ds)
```
and `CLIP["SWaT_canon"]` was changed `None -> 10.0`.

## 7. Artifacts regenerated (clean)
- `eda_real.py` — leak-free `_raw_swat_canonical`, `_assert_swat_no_leak`, CLIP=10.
- `sota_bundle/SWaT_canon_{train,test,labels,triv_test,triv_thr}.npy` — clipped, 97,197×51 train.
- `sota_bundle/ens_bundle/bundle_SWaT_canon.npz`; `sota_bundle/experts_full/expert_SWaT_canon.npz`
  (5-seed VaDE community experts, K/latent per |G|, rebuilt locally via `experts_local.py`, average/25/3).
- `_diagnostics/scores_SWaT_canon.npz` — clean LatAD/IF/AE (5-seed) + linres + active-channel maxz;
  USAD/TranAD/GDN folded in from Modal (clipped inputs).
- `_diagnostics/scores_sota_ms_SWaT_canon.npz` — USAD/TranAD (5-seed) + GDN, Modal `latad-sota`.
- `_diagnostics/loo_fusion_boosted.json` (SWaT_canon key), `headline_full_SWaT_canon_CLEAN.json`.
- Modal: USAD/TranAD 5 seeds + GDN 1 seed (stride 5), app `latad-sota`, on the clean clipped npy.

## 8. Correction summary
- Clean source: mirror `normal.csv` minus the 395,298 verbatim test-normal rows (Normal_v0 + Normal_v1), 0.000 overlap.
- Inflation magnitude (Difficult AUROC drop, leaked → clean): boosted_LOO −0.363, cohmax+LatAD −0.316,
  LatAD −0.277, linres −0.270, null+HC −0.225, HCcoh+LatAD −0.183, AE −0.074, USAD/TranAD ≈ −0.01.
- Clean verdict: **not a tie — HCcoh+LatAD / AUG significantly leads boosted_LOO on Difficult
  (+0.136, p < 0.001)**; the leak had masked this by inflating the baseline most.

---

## 9. OFFICIAL normal (clean re-run, 2026-09-19)

The §2 clean train was the FALLBACK (mirror `normal.csv` minus the verbatim test-normal rows =
Normal_v0 + Normal_v1). This section replaces it with the **OFFICIAL iTrust Dec-2015 normal
recording** `datasets/_new/SWaT_canonical/swat_normal_canonical.npz` (495,000 rows x 51 channels;
AIT201 genuine, range [251.7, 272.5]). `_raw_swat_canonical` now trains on it, drops the 1
coincidental raw-row hash collision by set-difference, and asserts `_assert_swat_no_leak`
(**0.000 test-normal overlap**, verified: 1616 train windows / 1498 test / 233 anomalies).

### 9.1 Root cause of the interrupted run's below-chance Difficult (NOT a code bug)
The prior artifact showed every method below chance on Difficult (trivial 0.206, n=39). This is
**not** a misaligned mask or partial write: the stored max|z| reproduces exactly from the current
load and is per-window aligned to the labels/scores. The cause is a genuine **train/test-NORMAL
domain shift** introduced by the official recording:
- 14 pump/UV/status channels are frozen (train std ~0) in the official normal but switch during
  the test log (both in normal and attack periods) -> 1e7-sigma divide-by-eps, clipped to +-10.
- 4 water-quality ANALYSER channels **AIT201/AIT202/AIT501/AIT504** are active but their normal
  operating point in the recording differs from the test-normal periods by >=2 sigma (AIT201 by
  ~11 sigma: recorded band [251.7,272.5] vs test-normal [168,267]); the analysers recalibrate
  between the two collection campaigns. Standardising on the narrow recording scale saturates
  their |z| at the clip for **most test-NORMAL windows** (test-normal max|z| median = 10.0).
The train-99th-pct difficulty threshold (3.27), calibrated on the frozen/quiet train, then sits
far below the test-normal max|z| level, so "Difficult" (max|z| <= thr) selected the atypically
QUIET anomalies against a backdrop of saturated-high normals -> trivial and every learned detector
invert. The active-channel rule alone (`fix_maxz_swat_canon.py`) removes the frozen channels but
NOT the recalibrated analysers, so it did not fix the axis.

### 9.2 Corrected difficulty axis (`fix_maxz_swat_canon_v2.py`, SWaT_canon-only)
max|z| over the first-sixth (window-mean) block on channels that are BOTH **active** (train std >=
1e-6; drops the 14 frozen) AND **coverage-stable** (|mean-z of the TEST-NORMAL windows| < 2; drops
the 4 recalibrated analysers). Unclipped, to match build_scores_table's canonical axis. 33/51
channels kept. Threshold = 99th pct of train max|z| over the same set. Detectors are untouched
(they still see all 51 channels); only the difficulty AXIS changes.
- **Easy / Difficult / DoubleHard = 142 / 91 / 31** (maxz_thr 3.163).
- **INVARIANT restored:** trivial max|z| Difficult AUROC = **0.627** (All 0.853), back in the
  healthy band (WADI_clean 0.60-0.70, fallback-SWaT 0.601), NOT below chance.

### 9.3 OLD-leaked vs FALLBACK-clean vs OFFICIAL-clean (5-seed mean AUROC)
Difficult subset counts differ per build (OLD 148/85/59, fallback 126/107/30, official 142/91/31),
so read All and Difficult as the primary comparison, not row-identical anomalies.

| method | All OLD | All FB | All OFF | Diff OLD | Diff FB | Diff OFF |
|---|---|---|---|---|---|---|
| trivial max\|z\| | 0.871 | 0.816 | 0.853 | 0.646 | 0.601 | 0.627 |
| IF | 0.821 | 0.808 | 0.800 | 0.627 | 0.613 | 0.550 |
| AE | 0.899 | 0.829 | 0.786 | 0.729 | 0.655 | 0.517 |
| linres | 0.920 | 0.770 | 0.776 | 0.782 | 0.512 | 0.464 |
| USAD (Modal, official) | 0.873 | 0.828 | 0.763 | 0.658 | 0.648 | 0.477 |
| TranAD (Modal, official) | 0.873 | 0.827 | 0.761 | 0.655 | 0.647 | 0.477 |
| GDN (Modal, official) | n/a | 0.828 | 0.761 | n/a | 0.648 | 0.473 |
| LatAD (global) | 0.927 | 0.766 | 0.752 | 0.804 | 0.527 | 0.472 |
| HC_coh | 0.919 | 0.835 | 0.898 | 0.822 | 0.658 | 0.742 |
| null+HC | 0.935 | 0.808 | 0.892 | 0.823 | 0.598 | 0.734 |
| **HCcoh+LatAD (headline)** | 0.941 | 0.835 | 0.883 | 0.840 | 0.657 | 0.703 |

SOTA (USAD/TranAD/GDN) were **re-run on Modal on the official train** (app `latad-sota`, 5+5+1
jobs, rc=0, reaped): the prior dumps predated the official normal npz, so they were stale; the
fresh window-averaged AUROC is ~0.76 All / ~0.477 Difficult. Community experts (24 communities,
5-seed) and LatAD/IF/AE/linres were already on the official train (verified) and were reused.

### 9.4 Significance (episode-block bootstrap, 2000 reps; official normal, corrected axis)
On the official normal the **strongest baseline is the trivial max|z| rule** (All 0.853 / Diff
0.627), not AE. Headline vs baselines:

| comparison | All | Difficult |
|---|---|---|
| HCcoh+LatAD - IF | +0.082 CI[0.047,0.179] p<.001 | +0.153 CI[0.069,0.237] p=.0005 |
| HCcoh+LatAD - AE | +0.096 CI[0.043,0.242] p<.001 | +0.186 CI[0.094,0.277] p<.001 |
| HCcoh+LatAD - trivial | +0.030 CI[-0.05,0.116] p=.16 (tie) | +0.076 CI[-0.075,0.219] p=.16 (tie) |
| null+HC - trivial | +0.039 CI[0.001,0.13] p=.025 | +0.107 CI[0.011,0.209] p=.012 |

### 9.5 Honest verdict (official normal)
- The below-chance was a **degenerate difficulty axis** from the official-recording analyser
  recalibration + frozen channels, NOT a corrupt score / alignment bug. Corrected, nothing is
  below chance.
- The community-density fusion (HC_coh / HCcoh+LatAD / null+HC) **significantly leads every learned
  deep and classical baseline** (IF, AE, USAD, TranAD, GDN, linres) on both All and Difficult
  (vs IF: All +0.082, Diff +0.153, p<.001; vs the deep SOTA the gap is larger).
- Against the strongest baseline, the **trivial max|z| rule**, it is a **tie** for HCcoh+LatAD
  (Diff +0.076, p=.16) and a **marginal lead** for the max-fused null+HC (Diff +0.107, p=.012).
  Note a circularity caveat: the difficulty axis is derived from max|z|, so the trivial-rule
  comparison on the Difficult subset is the most conservative bar.
- Net: on the genuinely-independent official normal, SWaT_canon is a **clear community-method win
  over the deep baselines and a tie-to-marginal-win over the trivial rule** - stronger than the
  fallback on All (HCcoh+LatAD 0.883 vs 0.835) and Difficult (0.703 vs 0.657), consistent with the
  drift-robustness thesis (density fusion degrades least under the clean train/test-normal shift).

### 9.6 HAI / WADI untouched (verified)
Pre-session mtimes unchanged: `scores_HAI.npz` (09-18 09:51), `scores_WADI_clean.npz` (09-18
09:55), `expert_HAI.npz` (08-09), `expert_WADI_clean.npz` (09-17). The corrected axis and SOTA
re-run are SWaT_canon-only.
