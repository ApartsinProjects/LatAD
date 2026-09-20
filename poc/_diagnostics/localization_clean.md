# Subsystem localization on the CLEAN pipeline (Table 9 re-derivation)

**Why.** The published Table 9 (IoT2.html) used a stale 25-community SWaT expert
(`experts_variants/cur_avg25`). This re-derives localization on the CLEAN experts,
`sota_bundle/experts_full/expert_{HAI,WADI_clean,SWaT_canon}.npz`. Confirmed clean SWaT
expert is **S=24 with `fit_surprise`** (keys include `comm_channels (24,19)`,
`test_surprise (5,24,1498)`, `fit_surprise (5,24,1292)`), not the 25-community variant.

**Method (unchanged from the script that produced Table 9).**
`_diagnostics/framing_issue2_localization.py`, episode level, **`rank` rule** = per-community
train-normal upper-tail empirical p-value (seed-averaged), degenerate-calibration communities
unranked. Ground truth = **published iTrust attack-target mapping** (SWaT DeepSentinel
`attack_list.csv`, 36 physically-labelled attacks; WADI `WADI_attacklabels.csv`, 12 rows;
HAI 20.07 timetable, 38 attacks). Coverage-aware random baseline `1 - C(S-kc,k)/C(S,k)`.
Reproduced fresh 2026-09-20; matches stored `framing_loc/loc2_summary.json` exactly.

## Clean per-dataset numbers (episode level, rank rule)

| Dataset | Episodes (targets known) | Named-comm size (ch/plant) | Top-1 | Random top-1 | Top-3 | Random top-3 | MC p (top-1) |
|---|---|---|---|---|---|---|---|
| HAI        | 38          | 5.5 / 59  | **0.395** | 0.246 | **0.684** | 0.557 | **0.024** |
| WADI_clean | 9 (7 cov.)  | 4.6 / 122 | 0.333 | 0.131 | 0.556 | 0.308 | 0.086 |
| SWaT_canon | 24          | 6.4 / 51  | 0.333 | 0.229 | 0.458 | 0.542 | 0.157 |

(MC p = Monte-Carlo permutation p-value for observed top-1 episode hits vs the coverage-aware
random baseline, 20k draws.)

## What changed vs the stale Table 9

| Dataset | Stale (published) T1 / T3 | Clean T1 / T3 | Verdict |
|---|---|---|---|
| SWaT | 0.46 / **0.79** | 0.33 / **0.458** | **Collapses.** Top-3 falls from above-random to **below** random (0.458 < 0.542). |
| HAI  | 0.39 / 0.68 | 0.395 / 0.684 | Unchanged (expert was already clean). |
| WADI | 0.33 / 0.56 | 0.333 / 0.556 | Unchanged (expert was already clean). |

Only SWaT moves. The stale 0.79 top-3 was an artifact of the 25-community expert.

## Verification (not a bug / coverage artifact)

- **Oracle upper bound.** On SWaT the non-deployable test-normal-calibrated score localizes at
  top-1 0.792 / top-3 0.833 (episode). The community structure *can* localize SWaT; the
  deployable **train-normal** calibration cannot. The gap is the SWaT drift (69% of test-normal
  windows), consistent with the paper's drift thesis, not a code error.
- **Loud-community monopoly.** SWaT normal-window top-1 profile under `rank` = [0.54, 0.14, ...]:
  one community holds 54% of normal top-1 slots, so it dominates the top rank on attacks too and
  the top-3 lands below chance. Milder on HAI ([0.39, ...]) and absent on WADI ([0.15, ...]).
- **Coverage.** SWaT episode coverage = 1.00, HAI 0.95, WADI 0.78 (7/9). SWaT's failure is not a
  coverage gap. GT sanity: SWaT 36 GT attacks 0 unmatched; HAI 38 one-to-one; WADI 12 rows, 0
  unmatched. Only unobserved points: SWaT MV504, WADI 1_AIT_006 (both in-record but not sensed).

## Table 9 replacement values (drop-in)

```
SWaT   | 24         | 6.4 / 51  | 0.33 | 0.23 | 0.46 | 0.54
HAI    | 38         | 5.6 / 59  | 0.39 | 0.25 | 0.68 | 0.56
WADI   | 9 (7 cov.) | 5.4 / 122 | 0.33 | 0.13 | 0.56 | 0.31
```

(HAI and WADI rows are unchanged from the current table. SWaT row is corrected: Top-1
0.46->0.33, Random-1 0.22->0.23, Top-3 0.79->0.46, Random-3 0.49->0.54, size 6.5->6.4.)

## Which datasets the localization claim survives on

- **HAI: survives.** Top-1 0.395 vs 0.246 random (1.6x, MC p=0.024, significant); top-3 0.684 vs
  0.557; median target-community rank = 2. The literal top-1 "most-surprised community" claim holds.
- **WADI: directional only.** Top-1 0.333 vs 0.131 (2.5x) and top-3 0.556 vs 0.308 (1.8x) are the
  largest *lifts*, but only 9 episodes (7 coverable), MC p=0.086: underpowered, does not reach
  significance.
- **SWaT: does NOT survive.** Top-1 0.333 vs 0.229 (not significant, p=0.157) and top-3 0.458
  **below** the 0.542 random baseline. Deployable train-normal ranking is drift-swamped; only the
  non-deployable oracle localizes it.

## Scoping recommendation

**Scope the localization claim to HAI (significant), with WADI as directional support; drop SWaT.**

The current paper text ("a three-community shortlist contains an attacked subsystem in **68 to 79
percent of HAI and SWaT attacks**", lines 656, 1238-1239) leans on the stale SWaT 0.79 and is no
longer supported: clean SWaT top-3 is 0.458 (below random). Suggested rewording: the top-3
shortlist contains the attacked subsystem in **68 percent of HAI attacks** (vs 56 percent random,
top-1 0.39 vs 0.25, p=0.024); WADI is directionally stronger (top-1 2.5x, top-3 1.8x random) but
has too few episodes (9) to test. On SWaT the deployable train-normal ranking is defeated by the
record's drift (localizable only by a non-deployable test-normal oracle), which is exactly the
non-stationarity the drift-aware typing of Section 7 addresses.

Text/table spots to update: Table 9 body (line ~1261 SWaT row), abstract-adjacent claim line 656,
deployment paragraph lines 1238-1253 ("68 to 79 percent of HAI and SWaT").
