# Post-hoc regime-triage pilot on the frozen LatAD latent (HAI, 5 seeds)

Report-only, 2026-09-17. Script: `fable_triage_pilot.py`; per-seed caches `fable_triage_HAI_s{0-4}.npz`;
aggregate `fable_triage_HAI.json`; log `fable_triage_HAI.log`. No paper, model, or checkpoint touched.
Detection scores are byte-identical to the submitted pipeline (invariant a below); the triage only adds a
second label to windows the detector already flags.

## Verdict: FAILS the pre-stated win bar. Registry only.

| Requirement (pre-stated) | Bar | Best partition (P0, VaDE argmax) | Result |
|---|---|---|---|
| Rare-normal false positives reclassified RARE-VALID | >= 50% | 0.64, 0.05, 0.61, 0.63, 0.15 (mean 0.42) | 3 of 5 seeds |
| Labelled anomalies mislabelled RARE-VALID | <= 5% | 8.0%, 3.8%, 3.5%, 10.1%, 5.1% (mean 6.1%) | 2 of 5 seeds |
| Both on the same seed | | seed 2 only (0.61 / 3.5%) | 1 of 5 seeds |

No partition clears both halves of the bar on a seed majority. The k-means codebook (M=64) keeps anomaly
mislabelling at 1.1% but reclassifies only 0.47, 0.07, 0.10, 0.10, 0.17 of the rare-normal FPs. Step 6
(witness dataset) is therefore not triggered.

## Rule as pre-stated (every threshold fixed on train-normal)

A window is FLAGGED if its LatAD score exceeds the 99th percentile of the train-normal score (the 1%
train-FPR operating point). A flagged window is RARE-VALID iff all four hold, else FAULT:

- RESID: whitened-residual z (A8) <= q99 of train-normal residual z
- BASIN: basin agreement (A3) >= q05 of train-normal agreement
- RARE: its code has >= 20 train windows and < 2% train share (an under-represented but seen regime)
- COH: the contiguous run of that code in the test sequence is >= 10 windows

Partitions for the code: P0 VaDE nearest-component argmax (K=40); P1 = P0 with components whose latent
trace-variance exceeds 2x the median split by a 2-GMM, depth <= 2 (F8); P2 k-means on the train latent,
M in {64, 128, 256}; CTL M=1. Rare-normal (for the win bar) = test-normal whose P0 component has < 2%
train share, the E2 definition, partition-independent. Witness = P0 components with < 1% train share
whose test-normal share exceeds 3x train and 0.5% (seed 0: components 8 and 21).

## Invariant checks

| Invariant | Expected | Observed | Status |
|---|---|---|---|
| (a) score path reproduces `scores_HAI.npz` | full and difficult AUROC equal to 4 digits, all seeds | 0.9409/0.9789, 0.9239/0.9692, 0.9323/0.9739, 0.9403/0.9791, 0.9285/0.9743, all equal | PASS |
| (b) witness block labelled RARE-VALID | majority of its flagged windows | 0.68, 0.03, 0.84, 0.85, 0.16 | 3 of 5 seeds |
| (c) M=1 codebook labels nothing | 0 RARE-VALID | 0 on all seeds; every metric unchanged | PASS |

Invariant (a) failed on the first attempt and the root cause was found before anything was read: the
coordinator's single-thread instruction. Torch CPU training is only bit-reproducible at the thread count
the table was built with (default, 8); a 1-thread run of seed 0 gave full AUROC 0.9381 against the stored
0.9409 (score correlation 0.69 after excluding the 35 windows that sit at ~1e30 in both runs; those 35
are a property of the residual head and identical in both). Re-staging at 8 threads reproduced all five
seeds to the digit. A second hazard, the head-seed default change committed after the table was built,
did not bite (the current default reproduced on every seed). The 35 ~1e30 windows are in the stored
table too, so they are part of the submitted score path, not an artefact of this pilot.

Invariant (b) failure on seeds 1 and 4, by guard: the un-reclassified witness windows fail RESID (0.83
and 0.54 of them) and COH (0.81 and 0.89); on seed 4 one witness component has 9 train windows, under
NMIN, so RARE fails for 25%. Both witness regimes on those seeds are visited in short alternating runs
(88 to 120 runs over 1000 to 1380 windows) and carry a residual above the train q99. The rule's physics
guards reject a real under-represented regime on 2 of 5 seeds.

## Confusion matrix among flagged windows (5 seeds summed), both directions

| P0 VaDE argmax | labelled RARE-VALID | labelled FAULT |
|---|---|---|
| true normal (6789 flagged) | 2359 | 4430 |
| true anomaly (2864 flagged) | **199** | 2665 |

| P2 k-means M=64 | labelled RARE-VALID | labelled FAULT |
|---|---|---|
| true normal (6789 flagged) | 1111 | 5678 |
| true anomaly (2864 flagged) | **40** | 2824 |

P1 (split) is P0 with 174 anomalies hidden instead of 199; P2 M=128: 528 / 36; CTL: 0 / 0.

## FPR after triage (drop RARE-VALID alarms) at the train-1% operating point, mean of 5 seeds

| Partition | test-normal FPR before -> after | recall before -> after | anomalies dropped (5-seed sum) | full AUROC excl. RARE-VALID | difficult AUROC excl. RARE-VALID |
|---|---|---|---|---|---|
| P0 VaDE argmax | 0.096 -> 0.063 | 0.879 -> 0.818 | 199 | 0.933 -> 0.940 | 0.975 -> 0.982 |
| P1 split | 0.096 -> 0.063 | 0.879 -> 0.825 | 174 | 0.933 -> 0.940 | 0.975 -> 0.982 |
| P2 km64 | 0.096 -> 0.080 | 0.879 -> 0.866 | 40 | 0.933 -> 0.936 | 0.975 -> 0.978 |
| CTL M=1 | 0.096 -> 0.096 | unchanged | 0 | 0.933 | 0.975 |

The test-normal FPR at the train-1% threshold is 9.6%, nine times the train rate, which is the
distribution shift the rare regimes cause. The method-output version of the manual exclusion reaches
0.940, not the 0.950 the manual exclusion gives, and it costs 6 points of recall. The requirement that
the drop touch ~0 real anomalies is not met: 199 of 3260 anomaly-windows (6.1%) over 5 seeds, 15 of 38
attack episodes on seed 0 alone.

## What the hidden anomalies are (root cause of the anomaly-side failure)

They are not label-edge windows: median in-window attack fraction 1.00, 83% at >= 0.9 (same as the anomaly
population). They are mostly HARD anomalies (0.61 to 0.96 of them per seed), they land in the witness
components (100% on seeds 0, 1, 4), and their residual z is 0.2 to 0.8 against 40 to 100 for the anomalies
labelled FAULT. The detector flags them because the regime is rare, exactly as it flags the rare-valid
normals, and the decoder reconstructs them as well as it reconstructs normal operation.

Separability of RARE-VALID anomalies from RARE-VALID normals on each guard statistic (AUROC, P0,
> 0.5 means anomalies score higher):

| seed | score | resid z | agreement | within-component z-dist | code occupancy | run length |
|---|---|---|---|---|---|---|
| 0 | 0.31 | 0.24 | 0.49 | 0.14 | 0.53 | 0.93 |
| 1 | 0.90 | 0.35 | 0.05 | 0.06 | 0.96 | 0.90 |
| 2 | 0.16 | 0.75 | 0.53 | 0.55 | 0.32 | 0.87 |
| 3 | 0.32 | 0.73 | 0.56 | 0.59 | 0.41 | 0.87 |
| 4 | 0.05 | 0.26 | 0.33 | 0.05 | 0.50 | 0.76 |

On score, residual, and distance to the component centre the hidden anomalies look as normal as, or more
normal than, the rare-valid normals on 3 to 4 of 5 seeds. No tightening of the physics guards separates
them; the failure is structural for this rule family, not a threshold choice. The one statistic that
separates them is run length, with anomalies forming *longer* contiguous blocks than the rare-regime
visits (attacks on HAI are sustained; rare-regime visits alternate). A rule with an upper bound on run
length would be a new rule chosen after seeing test labels, so it is recorded here as an observation
for a future pre-registered test and is not evaluated.

## Which partition worked cleanest

None cleanly. P0 (VaDE argmax) and P1 (split) are the only partitions that reach the reclassification
bar on any seed and they carry the anomaly-hiding cost; P1's split step changes little because the
rare components are not the high-variance ones. The k-means codebooks are cleaner on the anomaly side
only because they merge the rare regime into common codes (RARE guard fails for 65 to 92% of witness
windows at M=64), which is the collapse the codebook was supposed to avoid, arriving through a different
mechanism: k-means allocates codes by variance, and a 0.4% regime does not earn one at M <= 256.
The finding on partitions is therefore the same as the backbone analysis predicted: post-hoc
partitioning of this latent cannot name the rare regimes more reliably than VaDE itself does.

## Constants and reproduction

NMIN=20, RARE_FRAC=0.02, LRUN=10, SPLIT_RATIO=2.0, SPLIT_DEPTH=2, M in {64,128,256}; VaDE (40,16),
40 epochs, warm-up 8, k_density 80, heads auto-gated as in `build_scores_table.py`; HAI 18359 train /
14819 test windows, 652 anomaly windows. Rerun: `python _diagnostics/fable_triage_pilot.py HAI 0,1,2,3,4 8`
(the trailing 8 is the thread count; it must match the stored table for invariant (a)). Agent execution:
about 9 minutes staging plus 3 minutes diagnostics, CPU only.
