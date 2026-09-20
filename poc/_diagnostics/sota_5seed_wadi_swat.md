# USAD / TranAD five-seed runs on WADI_clean and SWaT_canon

Five seeds (0-4) each of USAD and TranAD, run on WADI_clean and SWaT_canon (20 jobs total,
CPU-only, 8 cores / 16 GB per job, Modal workspace `apersteiny`). Matches the protocol already
used for HAI's 5-seed rows: same `sota_bundle/modal_sota.py` harness
(`modal run modal_sota.py --datasets "WADI_clean,SWaT_canon" --models "USAD,TranAD" --seeds "0,1,2,3,4"`),
per-timestep score dumps window-averaged onto the same grid as `scores_<DS>.npz`
(`_diagnostics/sota_5seed_aggregate.py`, adapted from `rev4_sota_aggregate.py`), difficulty and
double-hard masks from the CLEAN `scores_WADI_clean.npz` / `scores_SWaT_canon.npz` (max|z| Easy/
Difficult split + LinRes leave-one-channel-out residual for DoubleHard, identical definitions to
`ensemble_final.py`).

All 20 jobs returned rc=0 (no failures, no watchdog kills, no spend-limit issues).

## Mean +- SD AUROC (5 seeds), by subset

### WADI_clean (n_windows=575, anom=56; Easy=26, Difficult=30, DoubleHard=19)

| Model  | All | Easy | Difficult | DoubleHard |
|---|---|---|---|---|
| USAD   | 0.7565 +- 0.0035 | 0.9622 +- 0.0015 | 0.5783 +- 0.0062 | 0.4647 +- 0.0108 |
| TranAD | 0.7840 +- 0.0017 | 0.9846 +- 0.0006 | 0.6102 +- 0.0029 | 0.4778 +- 0.0041 |

### SWaT_canon (n_windows=1498, anom=233; Easy=148, Difficult=85, DoubleHard=59)

| Model  | All | Easy | Difficult | DoubleHard |
|---|---|---|---|---|
| USAD   | 0.8730 +- 0.0005 | 0.9962 +- 0.0001 | 0.6584 +- 0.0015 | 0.6092 +- 0.0016 |
| TranAD | 0.8727 +- 0.0009 | 0.9980 +- 0.0006 | 0.6546 +- 0.0016 | 0.6025 +- 0.0017 |

(Difficulty/DoubleHard subset sizes above are the CLEAN counts: WADI difficult=30/double-hard=19,
SWaT difficult=85/double-hard=59, exactly matching the task spec.)

## Sanity check vs current Table 3 point estimates

All four means fall within 0.003 of the existing single-run point estimates (well under the
0.02 flag threshold), so no config mismatch:

| Dataset | Model | Subset | 5-seed mean | current point estimate | delta |
|---|---|---|---|---|---|
| WADI_clean | USAD | All | 0.756 | 0.757 | -0.001 |
| WADI_clean | USAD | Difficult | 0.578 | 0.579 | -0.001 |
| WADI_clean | TranAD | All | 0.784 | 0.786 | -0.002 |
| WADI_clean | TranAD | Difficult | 0.610 | 0.613 | -0.003 |
| SWaT_canon | USAD | All | 0.873 | 0.873 | +0.000 |
| SWaT_canon | USAD | Difficult | 0.658 | 0.658 | +0.000 |
| SWaT_canon | TranAD | All | 0.873 | 0.873 | -0.000 |
| SWaT_canon | TranAD | Difficult | 0.655 | 0.655 | -0.000 |

No flags. The single-run values already in the paper were seed-0 draws that happen to sit close
to the 5-seed mean; the paper's SDs can be added directly.

## Per-seed AUROC values

### WADI_clean

| Model | Subset | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| USAD | All | 0.7556 | 0.7580 | 0.7625 | 0.7531 | 0.7534 |
| USAD | Easy | 0.9625 | 0.9595 | 0.9643 | 0.9623 | 0.9624 |
| USAD | Difficult | 0.5763 | 0.5834 | 0.5875 | 0.5718 | 0.5723 |
| USAD | DoubleHard | 0.4620 | 0.4801 | 0.4739 | 0.4514 | 0.4559 |
| TranAD | All | 0.7835 | 0.7819 | 0.7870 | 0.7844 | 0.7833 |
| TranAD | Easy | 0.9843 | 0.9835 | 0.9849 | 0.9851 | 0.9853 |
| TranAD | Difficult | 0.6095 | 0.6072 | 0.6155 | 0.6105 | 0.6082 |
| TranAD | DoubleHard | 0.4794 | 0.4727 | 0.4849 | 0.4760 | 0.4762 |

### SWaT_canon

| Model | Subset | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|
| USAD | All | 0.8740 | 0.8725 | 0.8729 | 0.8727 | 0.8726 |
| USAD | Easy | 0.9961 | 0.9962 | 0.9963 | 0.9962 | 0.9961 |
| USAD | Difficult | 0.6613 | 0.6571 | 0.6581 | 0.6576 | 0.6576 |
| USAD | DoubleHard | 0.6120 | 0.6071 | 0.6082 | 0.6088 | 0.6097 |
| TranAD | All | 0.8731 | 0.8727 | 0.8730 | 0.8737 | 0.8711 |
| TranAD | Easy | 0.9982 | 0.9975 | 0.9984 | 0.9987 | 0.9972 |
| TranAD | Difficult | 0.6553 | 0.6553 | 0.6547 | 0.6561 | 0.6516 |
| TranAD | DoubleHard | 0.6036 | 0.6041 | 0.6024 | 0.6033 | 0.5993 |

(Exact values also saved in `_diagnostics/sota_5seed_wadi_swat.json`; per-job train_s and raw
per-run All/Hard metrics printed by the harness itself are in `_diagnostics/sota_5seed_launch.log`.)

## Modal wall-clock and cost

- App `ap-i0cJrUtMguqSlfLVfZGggm`, workspace `apersteiny`: created 2026-09-18 21:49:34, stopped
  21:55:58 -> **6 min 24 s wall-clock** for all 20 jobs run in parallel (one container per
  (dataset, model, seed), CPU-only, no GPU).
- Per-job training time (`train_s` from the harness, sum over 20 jobs) = 5,157 s; CPU-only,
  8 cores / 16 GB per container.
- Estimated cost at Modal's published on-demand CPU rate ($0.0000131/core-sec) and memory rate
  ($0.00000222/GiB-sec), using container lifetime ~= train_s + ~20 s setup/git-clone overhead
  per job (~5,560 s total across all 20 containers): **~$0.6-0.8** (CPU ~$0.58 + memory ~$0.20).
  No GPU was requested by this harness (`run_one` has no `gpu=` argument), so this run's cost is
  CPU/RAM only, not GPU-hour.
- No spend-limit hit, no failed or watchdog-killed jobs (all rc=0).

## Agent execution time

This task (read harness, smoke test, launch, poll, pull, aggregate, sanity-check, write report)
took about 15 Modal-side minutes of wall-clock (smoke test ~1 min + full run ~6.5 min + polling/
pull/aggregation), and roughly 15-20 minutes of my own tool-call time end to end. Compute cost
under $1.

## Files

- `_diagnostics/sota_5seed_aggregate.py` -- aggregation script (adapted from `rev4_sota_aggregate.py`
  + `ensemble_final.py`'s DoubleHard definition, targeting `WADI_clean`/`SWaT_canon`)
- `_diagnostics/sota_5seed_wadi_swat.json` -- full per-seed, per-subset results
- `_diagnostics/sota_5seed_launch.log` -- full Modal run log (per-job raw metrics, train_s)
- `_diagnostics/sota_pull_5seed/` -- pulled `score_<MODEL>_<DS>_s<seed>.npy`,
  `labels_<MODEL>_<DS>_s<seed>.npy`, `one_<MODEL>_<DS>_s<seed>.json` (40 npy + 20 json)

Paper (`paper.html` / `IoT2.html`) was NOT edited. These numbers are ready to drop into Table 3/4
as +-SD alongside the existing WADI_clean/SWaT_canon USAD/TranAD point estimates.
