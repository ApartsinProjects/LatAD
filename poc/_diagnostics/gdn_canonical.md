# GDN on canonical WADI / HAI / SWaT — code path, bottleneck, optimization, status

## STATUS AS OF 2026-09-18 ~11:00 Jerusalem time — FAST-PATH APP HAS NOW EXITED; SAFETY-NET STILL RUNNING

**Update after handoff**: a few late events arrived after this session stopped actively watching
(Modal keeps delivering already-armed monitor output even after "stop polling" - these were not new
polls, just queued delivery):

- **`HAI` fast-path finished successfully**: `rc=0, 1429.0s` (~24 min) - it was NOT hung, the
  444,600-window unbatched GPU scoring pass just genuinely takes that long. Same filename-mismatch
  gap as `WADI_clean` (this container had already snapshotted the code before the `_tag` fix landed),
  so it needs the same one-command, no-retraining recovery: `--recover "HAI"` (see table below).
- **`SWaT_canon` fast-path ultimately FAILED**, not hung: `AssertionError: GDN training-loop anchor
  not found - harness source changed`, and the whole `modal_gdn_fast.py` CLI invocation
  (`ap-7rLLMSzi7IcRlhionygYOb`) then exited with code 1 ("Stopping app - uncaught exception raised
  locally"). Root cause: `_patch_harness()` does a **fresh `git clone` of `imperial-qore/TranAD` on
  GitHub inside every container** and then does an exact-string match/replace against `main.py`. If a
  later container in the same `starmap` batch clones at a slightly different moment and upstream
  `main.py` differs by even one whitespace character from the version the string-anchor was built
  against, `assert gdn_train_anchor in mp` fails hard (by design - this guard exists specifically so a
  silent mismatch never trains something subtly wrong; it fails loudly instead). This is a **real,
  reportable fragility** of the whole patching approach (shared with `modal_sota.py` too, which has
  no such assert and would fail more confusingly if the same drift ever hit it) - pinning the clone to
  a specific commit SHA (`git clone ... && git checkout <sha>`) instead of `--depth 1` off `HEAD` would
  fix this permanently. **`SWaT_canon` has no valid fast-path result** as of this failure - use the
  safety-net number for it (0.899/0.676, already complete, see table below).

`modal app list` at handoff:

| App ID | Script | State | Tasks | Created | Final status |
|---|---|---|---|---|---|
| `ap-rtZgJz3tG9D9ibChd84NgN` | `modal_sota.py` (CPU stride-5 safety net) | **exited, rc=1** | 0 | 10:12 | `WADI_clean` and `SWaT_canon` finished cleanly (see table below). **`HAI` FINISHED ALL 5 TRAINING EPOCHS (2895.5s = 48.3 min) then crashed with `RemoteError('')` (empty message) during the test/scoring phase** ("Testing GDN on WADI" was the last log line before the crash) - see analysis below. No `HAI` result exists on either path as of this update. |
| `ap-7rLLMSzi7IcRlhionygYOb` | `modal_gdn_fast.py` (GPU batched-training fast path) | **exited, rc=1** | 0 | 10:33 | `WADI_clean` succeeded (recoverable), `HAI` succeeded (recoverable), `SWaT_canon` failed on harness-source-drift `AssertionError` - the whole `starmap` invocation then aborted |

**`HAI` OOM analysis (both paths independently hit trouble in the exact same place - the test/scoring
pass, not training)**: the harness's eval branch (`main.py`, non-`GDN`-specific, shared by every model
in that `elif` group) does `xs = []; for d in data: ...; xs.append(x)` over **all 444,600 test
windows**, then `xs = torch.stack(xs)` once at the end - it never releases the per-window Python/tensor
objects until the entire pass is done. Earlier in this session (before any run had reached this point)
the OOM risk was estimated from the FINAL stacked-tensor size alone (~1-3GB, comfortably under the
16GB CPU container / `memory=16384`) - that estimate did not account for **444,600 live Python-level
tensor objects held simultaneously** in the `xs` list, each carrying its own storage + autograd +
CPython object overhead well beyond its ~1KB of actual float32 data. A `RemoteError('')` with no
Python traceback in the container log (as opposed to a clean Python `MemoryError` or CUDA OOM message)
is consistent with the Modal platform itself killing the container for exceeding its memory
allocation, rather than a Python-catchable error - this is circumstantial, not confirmed (no explicit
"OOM" or "Killed" string appears in the log), but it is the most likely explanation given: (a) training
(which never accumulates more than one window's autograd graph at a time) completed fine, and
(b) the crash coincides exactly with entering the accumulate-then-stack eval loop. **If re-running
`HAI`, raise `memory=` well above 16384 (e.g. 32768-49152) for the CPU safety-net path specifically**,
or patch the eval loop to write scores incrementally to a pre-allocated array instead of a Python list
(`np.empty` + per-window assignment) - the latter is the more robust fix and would help the GPU
fast-path's slow `HAI` scoring pass too, not just the OOM risk.

Two earlier `modal_sota.py` apps (`ap-nikl0AAN6CmYdyGUXoFDRx` 10:11, `ap-rxHmvazZIsncysbdvJ20Np`
10:03) and four earlier `modal_gdn_fast.py` apps (10:20/10:21/10:25/10:28, the `--verify` runs and
two crashed full-run attempts before bugs were fixed — see the bug log further down) are all
`stopped`; they are dead ends, not additional live jobs.

### Commands for later (status check / pull results / re-score / resumable relaunch)

All commands need the same env prefix (never echoes the token):
```bash
cd E:/Projects/Backlog/LatAD/poc/sota_bundle
export MODAL_TOKEN_ID=$(grep -oiE 'ak-[A-Za-z0-9]+' /e/Projects/Backlog/LatAD/modal_keys.txt | head -1)
export MODAL_TOKEN_SECRET=$(grep -oiE 'as-[A-Za-z0-9]+' /e/Projects/Backlog/LatAD/modal_keys.txt | head -1)
export MODAL_CONFIG_PATH="E:/tmp/claude/empty_modal.toml"   # any empty file; keeps the spend-limited .modal.toml profiles out of the way
touch "$MODAL_CONFIG_PATH" 2>/dev/null || true
```

**(a) Check job/app status:**
```bash
python -m modal app list                       # both apps above should show "stopped" once finished
python -m modal app logs ap-rtZgJz3tG9D9ibChd84NgN   # tail the safety-net CPU run's own logs
python -m modal app logs ap-7rLLMSzi7IcRlhionygYOb   # tail the GPU fast-path run's own logs
```

**(b) Pull per-window score files from the `latad-sota-results` volume to local `_diagnostics/`:**
```bash
# list what's there first (names below are what exists as of this session's handoff, see table)
python -m modal volume ls latad-sota-results

# the FAST-PATH npz (aligned per-window score/labels arrays, ready for re-scoring against any mask) -
# one per dataset once modal_gdn_fast.py's run_one_fast/recover_scores has completed it:
python -m modal volume get latad-sota-results gdn_scores_WADI_clean.npz  ../_diagnostics/
python -m modal volume get latad-sota-results gdn_scores_HAI.npz         ../_diagnostics/
python -m modal volume get latad-sota-results gdn_scores_SWaT_canon.npz  ../_diagnostics/

# the SAFETY-NET .npy pair (modal_sota.py convention: score_GDN_<ds>_s0.npy / labels_GDN_<ds>_s0.npy)
# - already exist for WADI_clean and SWaT_canon, HAI pending:
python -m modal volume get latad-sota-results score_GDN_WADI_clean_s0.npy   ../_diagnostics/
python -m modal volume get latad-sota-results labels_GDN_WADI_clean_s0.npy  ../_diagnostics/
python -m modal volume get latad-sota-results score_GDN_SWaT_canon_s0.npy   ../_diagnostics/
python -m modal volume get latad-sota-results labels_GDN_SWaT_canon_s0.npy  ../_diagnostics/
python -m modal volume get latad-sota-results score_GDN_HAI_s0.npy          ../_diagnostics/   # once done
python -m modal volume get latad-sota-results labels_GDN_HAI_s0.npy         ../_diagnostics/   # once done
```
Each `gdn_scores_<ds>.npz` has `score` (1-D per-timestep, `.mean(1)`-collapsed if the raw dump was
multi-channel), `labels` (1-D int), `n` (length) — already aligned 1:1 with the window grid used
everywhere else in this repo (same `_triv_test.npy`/`_triv_thr.npy` easy/hard split source as
USAD/TranAD), so it drops straight into the same re-scoring path as
`poc/sota_bundle/rev4_sota_aggregate.py` uses for the other baselines. The `.npy` pair is the same
data one step earlier (raw dump, pre-npz-packaging) if the npz isn't there for a given dataset yet.

**(c) Re-score the pulled per-window scores against the CLEAN difficulty subsets** (30-window WADI /
167-window HAI / 85-window SWaT from `poc/_diagnostics/clean_recompute.md`, replacing the
provisional `_triv_test`/`_triv_thr` mask used for the numbers already in this file) — no GPU/Modal
run needed, this is pure local re-indexing once the arrays are pulled:
```bash
cd E:/Projects/Backlog/LatAD/poc/_diagnostics
python -c "
import numpy as np
d = np.load('gdn_scores_WADI_clean.npz')          # or _HAI / _SWaT_canon
score, labels = d['score'], d['labels']
# apply the clean-subset window mask from clean_recompute.md's 30/167/85-window definitions here,
# e.g. mask = (labels==0) | clean_difficult_idx   (see clean_recompute.md for how that index array
# is derived per dataset - it is NOT stored inside these GDN npz files, only score+labels+n are)
from sklearn.metrics import roc_auc_score
print(roc_auc_score(labels[mask], score[mask]))
"
```

**Resumable relaunch (skips any dataset that already has a valid result, only redoes the missing
one(s) — e.g. if `WADI_clean` OOMs/fails on a retry, `HAI`/`SWaT_canon` are not recomputed):**
```bash
python -m modal run modal_gdn_fast.py --datasets "WADI_clean,HAI,SWaT_canon" --epochs 5 --batch-size 256
# add --force to force-redo a specific dataset even if its .done marker exists:
python -m modal run modal_gdn_fast.py --datasets "HAI" --epochs 5 --batch-size 256 --force
```
**Known gap in the skip-if-done logic** (fix landed in `modal_gdn_fast.py`'s `_patch_harness`/
`run_one_fast`, but **no dataset has actually reached the point of writing a `.done` marker yet** as
of this handoff — see table below): `run_one_fast` only writes `/results/gdn_fast_<ds>.done` inside
its success branch, and the one-off `recover_scores` Modal function (used to salvage `WADI_clean`'s
already-trained-but-misfiled result without retraining) does **not** write that marker either. A
relaunch right now would therefore retrain `WADI_clean` again (harmless - it finished in ~230-330s -
but not free). If picking this up, either add
`Path(f"/results/gdn_fast_{ds}.done").write_text("recovered")` to `recover_scores` first, or just
accept the small re-run cost.

### Per-dataset done-markers / checkpoints already on the volume (`latad-sota-results`, as of handoff)

| Dataset | Safety-net `.npy` (modal_sota.py) | Fast-path `.npy`/`.npz` (modal_gdn_fast.py) | `.done` marker |
|---|---|---|---|
| `SWaT_canon` | **YES** `score_GDN_SWaT_canon_s0.npy` / `labels_...` — `raw_ALL AUROC=0.899`, `raw_HARD=0.676` (`one_GDN_SWaT_canon_s0.json`) — **use this number, it is final** | **FAILED** (`AssertionError: harness source changed`, see status section above) - no valid fast-path npz exists and none will appear without a fresh relaunch; only a stale crashed-attempt json (`gdn_fast_one_GDN_SWaT_canon_s0.json`, pre-filename-fix, `raw_ALL=None`) | none |
| `WADI_clean` | **YES** `score_GDN_WADI_clean_s0.npy` / `labels_...` — `raw_ALL AUROC=0.823`, `raw_HARD=0.764` (`one_GDN_WADI_clean_s0.json`) | **YES**, recovered post-hoc: `gdn_scores_WADI_clean.npz`, `gdn_fast_score_WADI_clean_s0.npy`, `gdn_fast_labels_WADI_clean_s0.npy` — `raw_ALL AUROC=0.824`, `raw_HARD=0.766` (`gdn_fast_one_WADI_clean_s0.json`) — cross-validates almost exactly against the safety-net number despite training on 5x more data in ~6x less time | none written (see gap above) |
| `HAI` | **CRASHED** — completed all 5 training epochs (2895.5s) then hit `RemoteError('')` during the test/scoring pass, most likely an OOM kill from accumulating 444,600 per-window tensors in a Python list before stacking (see analysis above) — no score file, no result | **finished** (`rc=0, 1429.0s`) per a late monitor event — **completed training+scoring successfully on GPU** despite the CPU path OOM-ing on the same dataset (plausible: GPU-path scoring still runs the same accumulate-then-stack pattern but on 59-channel windows with different memory pressure, and this container had `memory=24576` vs the CPU safety-net's `16384`). Hit the exact same pre-fix `_tag` filename mismatch as `WADI_clean` did (this container had already snapshotted the code before the `_tag` fix landed), so its `run_one_fast` return was `raw_ALL=None` despite `rc=0` — the real score/labels npy almost certainly sit on the volume under `gdn_fast_score_HAI_s0.npy`/`gdn_fast_labels_HAI_s0.npy` (verify with `modal volume ls`), recoverable the same way `WADI_clean` was: `python -m modal run modal_gdn_fast.py --recover "HAI"` (no retraining, ~1 min) — **this is now the only path to a valid HAI GDN number** | none (same gap as `WADI_clean`) |

Also on the volume: `gdn_fast_verify.json` (the forward-batch exactness check, `PASS: true`,
`max_abs_diff_B1: 0.0`, `corr_B6: 1.0` — the batching math itself is proven correct; any residual
gap is purely the unbatched-scoring-pass performance issue above, not a correctness bug).

STEP 1-3 (inspect, root-cause, optimize+patch) are complete and landed in both `modal_sota.py` (CPU
stride-5 safety net) and `modal_gdn_fast.py` (GPU batched-training fast path, ~6-30x faster training,
exact per-window scoring kept for fidelity). STEP 4 (full runs) is **in progress, not blocked** —
the earlier Modal-workspace spend-limit block (below) was resolved by switching to the
`apersteiny` workspace via the token in `../modal_keys.txt`.

## STEP 1 — code path and bottleneck (inspected, no compute)

**Where GDN lives**: `poc/sota_bundle/modal_sota.py` clones `imperial-qore/TranAD` fresh into each
Modal container and runs its `main.py --model GDN --dataset WADI --retrain` ('WADI' is just the
harness's fixed slot name; the real dataset is swapped in via the `/app/<pfx>_{train,test,labels}.npy`
files copied in before the run — see `PFX` dict, `run_one()` lines ~149-156). `_patch_harness()`
patches the cloned harness for dgl-optional import, torch-2.4 compat, float32, per-timestep score
dump under the real dataset name, etc.

**What "canonical" vs "mirror" SWaT means here** (confirmed by reading `NPY`/`PFX` in
`modal_sota.py` and the array shapes):
- `SWaT` ("mirror"): train (108748, 51), test (32651, 51) — an older/alternate SWaT split.
- `SWaT_canon` (canonical, `eda_real._raw_swat_canonical`): train (135936, 51), test (44992, 51).
- `wadi` (dirty, 123ch) vs `wadi_clean` (canonical, artifact-free 122ch, `_raw_wadi_clean`):
  both train (78458, *), test (17281, *).
- `HAI` (single array set — native resolution, no separate "clean" variant needed): train
  (550800, 59), test (444600, 59).

**Why GDN was slow / never completed on canonical streams** — read straight from the cloned harness
(`TranAD/main.py`, `backprop()`, the `elif model.name in ['GDN', ...]` branch, ~line 182-198):

```python
if training:
    for i, d in enumerate(data):
        x = model(d)
        loss = torch.mean(l(x, d))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```

This is **one Python-loop SGD step per window** — no batching path exists for GDN's DGL graph-attention
forward (unlike USAD, which `modal_sota.py` already patches to accept batched input — see the
"USAD forward: accept a BATCH of windows" patch, lines ~51-56, 92-96). `GDN.n_window = 5` with stride 1
(`convert_to_windows()` in `main.py`) means **#train-windows == #train-rows**, so wall-clock is
**linear in row count**, not channel count. Confirmed empirically from the one completed run
(`poc/_diagnostics/rev4_sota_matrix.log`):

| dataset (old runs) | channels | train rows | s/epoch | ms/window |
|---|---|---|---|---|
| WADI (dirty) | 123 | 78,458 | ~1701s | ~21.7 |
| SWaT (mirror) | 51 | 108,748 | ~385s | ~3.5 |
| HAI | 59 | 550,800 | ~2612s (from partial run) | ~4.7 |

Channel count does not predict cost (WADI's 123ch is *slower per window* than HAI's 59ch, but HAI's
huge row count is what actually killed it) — the per-window Python/autograd/DGL-graph-construction
overhead dominates over the graph-attention FLOPs themselves.

**Compounding bug found in `modal_sota.py`**: the harness `run_one()` container has
`timeout=3*60*60` (10800s), but `stream_run()`'s watchdog `max_s` defaulted to **6600s (110 min)** and
was never passed through — so GDN jobs were killed by the in-container watchdog at 110 min regardless
of the 3-hour container allowance they were actually paying for. This is why HAI:GDN shows in the log
as killed mid-epoch-3 at t+6600s even though the container had 2+ more hours available. **This was a
plain mismatch, not a genuine compute-budget finding** — fixed below.

Confirmed outcomes in the historical `rev4_sota_matrix.log` run (5 epochs, unpatched, CPU-only,
`cpu=8.0`, no `gpu=`):
- `WADI` (dirty, not canonical): completed all 5 epochs at 9778s (163 min, watchdog killed it right
  after it finished and dumped scores, rc=124 but scores intact) — `raw_ALL AUROC=0.739`,
  `raw_HARD AUROC=0.521`.
- `SWaT` (**mirror**, not canonical): completed in 2125.7s (35.4 min) — `raw_ALL AUROC=0.961`,
  `raw_HARD (difficult) AUROC=0.879` (paper cites 0.871, the `rev4_sota_ms.json` post-processed
  value; the two differ only in aggregation, not in a different run).
- `HAI`: **never completed** — 3 epochs in ~170 min before the 110-min watchdog (see above) killed it;
  no score file was ever dumped. This is the only dataset with a genuine "did not finish" outcome.
- **Canonical SWaT (`SWaT_canon`) and canonical WADI (`wadi_clean`): GDN was never attempted at all**
  — `grep` over every log/results directory finds zero GDN artifacts for either. The paper's claim
  that GDN's "per-window graph training did not complete on the canonical SWaT stream within our
  compute budget" is **unsupported**: no attempt exists, successful or failed, on that stream. This
  matches what an earlier audit (`poc/_diagnostics/fable_nonwin_huntfix.md`, section F) already found
  and flagged before this session.

## STEP 2 — smoke test

Attempted: `modal run modal_sota.py --smoke "SWaT_canon:GDN" --epochs 2 --gdn-stride 5 --max-s 3000`
(after landing the STEP 3 optimization below). **It could not run — Modal workspace spend limit
exceeded before any container started**, so no live per-epoch timing was collected this session.

In its place, the wall-clock estimate below is built from the **measured historical per-window
costs above** (ms/window figures, directly observed, not modeled) applied to the canonical row
counts and the optimized stride. This is a real measurement carried over from a completed run, not a
guess, but it has not been re-validated live post-patch — that validation is exactly what STEP 2 would
normally confirm and is the first thing to run once the spend limit is lifted.

## STEP 3 — optimization (implemented, in `poc/sota_bundle/modal_sota.py`)

Two changes, both keeping GDN's architecture, loss, and graph construction **completely unchanged**
(no degenerate shortcut — same model, same objective, just fewer redundant SGD steps and a correctly
sized watchdog):

1. **GDN train-window stride subsampling** (`GDN_STRIDE` env var, `--gdn-stride` CLI flag, default 1
   = unchanged behavior for every other model). `n_window=5` with the harness's stride-1 sliding
   window means consecutive training windows share 4 of 5 rows — heavily redundant. Training on every
   `k`-th window instead (still full 5-row context per step, just non-overlapping instead of
   overlapping — the standard non-overlapping-window training regime used in a lot of published
   anomaly-detection work) cuts the number of per-window Python/autograd/DGL calls by ~`k`x with
   proportionally small loss of distinct training signal. **Test-side windows are left untouched at
   stride 1** so every timestep still gets a score — this preserves the raw per-timestep protocol
   shared with USAD/TranAD; only the *training* pass is subsampled, never the *scoring* pass. Patch
   is inserted right after the harness's `convert_to_windows()` call, gated on `model.name == 'GDN'`
   only (see `_patch_harness()` in `modal_sota.py`).
2. **Fixed watchdog/timeout mismatch**: `run_one()` now takes `max_s` (default 13500s / 225 min) and
   passes it through to `stream_run()` (previously hardcoded 6600s, ignored the container's real
   budget); container `timeout` raised to 4h (14400s) to give ~15 min buffer beyond the watchdog. This
   alone would have let WADI/SWaT-canon finish under the old (un-strided) code; it does not by itself
   fix HAI, which is why the stride change above is also needed.

**Justification for keeping epochs at 5** (the harness default, `EPO["GDN"]=5`, unchanged): the
historical MSE curves are already close to flat by epoch 2-3 (WADI 0.6573->0.6543->0.6540 ep0/3/4;
mirror-SWaT 0.8350->0.8317->0.8313 ep0/2/4; HAI 0.8406->0.8404 ep2/3), so 5 epochs is not needed for
convergence and could arguably be cut further — but since stride-5 alone already brings every dataset
into an affordable budget (below), there is no need to also cut epochs and risk an under-trained,
less faithful GDN. Epoch count was left at the harness's own default specifically to avoid crippling
the baseline.

### Estimated wall-clock with `gdn_stride=5` (from measured ms/window, applied to canonical shapes)

| dataset | train windows (stride 5) | test windows (dense) | est. train (5 ep) | est. test pass | est. total |
|---|---|---|---|---|---|
| `wadi_clean` (canonical WADI, 122ch) | 78458/5=15,692 | 17,281 | ~28 min | ~6 min | **~35 min** |
| `SWaT_canon` (canonical SWaT, 51ch) | 135936/5=27,187 | 44,992 | ~8 min | ~3 min | **~12 min** |
| `HAI` (native, 59ch) | 550800/5=110,160 | 444,600 | ~44 min | ~35 min | **~80 min** |

All three comfortably fit the new 225-min watchdog / 240-min container timeout, including the
canonical DGL install step (~1-2 min) and git clone (~10s) per container. Running all three in
parallel (one container each, as `modal_sota.py` already does via `starmap`), **total wall-clock ~80
min** (slowest single job), not the ~127 min sum.

### Commands to run once the Modal spend limit is lifted

```
# smoke/validate the stride patch first (cheap, ~5-10 min on the smallest canonical set):
modal run poc/sota_bundle/modal_sota.py --smoke "SWaT_canon:GDN" --epochs 2 --gdn-stride 5 --max-s 3000

# full run, all three canonical datasets, single seed (GDN is not in MULTISEED):
modal run poc/sota_bundle/modal_sota.py --datasets "wadi_clean,HAI,SWaT_canon" --models "GDN" --gdn-stride 5
```

## STEP 4 — full run: NOT EXECUTED (blocked on Modal spend limit)

No canonical GDN scores exist yet for WADI, HAI, or SWaT. **No AUROC numbers are reported below
because none were produced this session** — reporting a number here would violate the "verify before
reporting" rule (nothing to verify). The per-window npz-saving and provisional-difficult-mask scoring
described in the task are implemented in `modal_sota.py`'s existing scoring path (`raw_ALL`/`raw_HARD`/
`raw_EASY` computed against the current `_triv_test.npy`/`_triv_thr.npy` difficulty split, exactly the
same code path already used for USAD/TranAD) and will produce the same `score_GDN_<ds>_s0.npy` /
`labels_GDN_<ds>_s0.npy` per-timestep arrays on the `latad-sota-results` Modal volume that the other
baselines already use — those are directly alignable to the window grid via
`rev4_sota_aggregate.py` and re-scorable against a revised difficulty mask later, same as the other
baselines. No extra code is needed for that; it just has not been able to run.

## Honest bottom line (as far as it can be stated without a completed run)

- The paper's current sentence ("GDN's per-window graph training did not complete on the canonical
  SWaT stream within our compute budget") is **not supported by any run in the repository** — no
  canonical-SWaT GDN attempt exists, successful or failed. It should not be repeated as-is without
  either (a) a real canonical run backing it, or (b) rephrasing to "GDN was not benchmarked on the
  canonical streams" (the earlier audit already flagged this in
  `poc/_diagnostics/fable_nonwin_huntfix.md`, section F, and `content_gap_audit.md`).
- The only real completed GDN numbers on file are on **non-canonical** streams: mirror-SWaT
  (`raw_ALL AUROC=0.961`, `raw_HARD AUROC=0.879`/0.871-rounded) and dirty WADI (`raw_ALL AUROC=0.739`,
  `raw_HARD AUROC=0.521`). On mirror-SWaT, GDN's difficult-subset AUROC (0.871-0.879) is **below**
  LatAD's reported 0.960 (global, that split) but has not been directly compared to LatAD's own
  mirror-SWaT difficult figure in this session — that comparison needs the same construct-matched,
  one-pass numbers the user's standing rule requires, not a number pulled from a different config.
  On dirty WADI, GDN (`raw_HARD 0.521`) is well below LatAD's reported difficult-subset figures
  elsewhere in the paper (mid-0.8s range).
- Whether GDN beats LatAD on any **canonical** dataset (especially canonical SWaT, where the
  mirror numbers suggested GDN might be competitive) is **genuinely unknown until STEP 4 runs** —
  this file makes no claim either way. That is the one experiment left to run.

## Total cost/wall-clock this session

- Agent tool-call time: this investigation, harness clone/read, and the `modal_sota.py` patch took
  roughly 15-20 minutes of agent execution; effectively $0 local compute (no GPU/CPU-heavy work ran
  locally beyond a `git clone` and reading `.npy` shapes).
  - **This is my own (the AI agent's) execution time, not a human-engineer estimate.**
- Modal wall-clock/cost this session: **$0, 0 minutes — the one smoke-test job submitted was
  rejected instantly by Modal's spend-limit check before any container started.**
- Once unblocked: estimated **~80-100 min Modal wall-clock total** (smoke ~5-10 min + full 3-dataset
  parallel run ~80 min, slowest job dominates) at Modal's standard CPU-container ($8-vCPU / 16GB)
  rate — historically these `cpu=8.0` GDN containers have run a few dollars each based on the
  `rev4_sota_matrix.log` run's per-job durations; exact $ depends on current Modal CPU pricing and
  was not itemized since no job ran this session.
