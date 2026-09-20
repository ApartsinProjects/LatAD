# E3 / E5 Modal experiments — LatAD (MDPI IoT)

Two Modal jobs, staged next to the existing `sota_bundle/modal_*.py` so they share its
image / volume / `add_local_file` conventions.

- **`modal_e3_cost.py`** — cost / compute benchmark (training wall-clock, peak GPU mem,
  peak host RAM, params, batch-1 / batch-256 latency, throughput; per dataset × method;
  LatAD reported as *global density* and *regime-community* separately, plus per-community
  ms for the CalexNet early-exit remark). CPU-only path for the edge proxy.
- **`modal_e5_gain.py`** — gain ablation (difficult-subset AUROC, 5 seeds, 3 datasets;
  decisive arm-b cross-channel-vs-marginal density; optional arm-a representation sweep).

> **STATUS: PREPARE ONLY.** Nothing here has been run. Do not `modal run` until the user
> supplies the Modal token and says go. No paper HTML is touched.

---

## 1. GPU / image (matched to the existing scripts, with one deliberate change)

The existing `modal_sota.py`, `modal_experts.py`, `modal_ensemble.py` all run **CPU-only**
(`@app.function(cpu=8.0, ...)`, no `gpu=`, `train_vade(..., device="cpu")`). The E3 GPU
numbers (peak GPU memory, GPU latency) require an actual GPU, so **E3 and E5 add a GPU
function (`gpu="A10G"`) plus a CPU function**, selected with `--device {cuda,cpu}`.

- `--device cuda` → `gpu="A10G"` (A10G = cheap, 24 GB; matches the gpu2modal "encoding /
  small-batch" recommendation, ample for these tiny models). Gives the GPU-memory and
  GPU-latency rows.
- `--device cpu` → `gpu=None`, `cpu=8.0` → the **edge-proxy latency** numbers (and it
  reproduces the existing pipeline's CPU behavior exactly).

Base image (identical registry + deps to `modal_sota` / `modal_experts`, plus `psutil`
for host RAM and `git` for the TranAD harness clone):

```
pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel  (add_python=3.11)
apt:  git
pip:  numpy<2  pandas  scikit-learn  scipy  tqdm  matplotlib  psutil
```

`zenodo_bundle/requirements.txt` (`torch>=2.0 numpy>=1.24 scikit-learn>=1.3 scipy>=1.10`)
is a strict subset of the above — satisfied by the pinned image.

---

## 2. Exact `modal run` commands

E3 — cost benchmark:
```bash
# GPU run (all datasets, all methods): TranAD, USAD, AE, LatAD-global, LatAD-community
modal run sota_bundle/modal_e3_cost.py --device cuda

# CPU edge-proxy latency (same script, same rows, device=cpu)
modal run sota_bundle/modal_e3_cost.py --device cpu

# smoke one cell
modal run sota_bundle/modal_e3_cost.py --device cuda --datasets WADI --methods LatAD-global
```

E5 — gain ablation:
```bash
# decisive arms only (reference + arm-b cross-channel vs marginal), 5 seeds, 3 datasets
modal run sota_bundle/modal_e5_gain.py

# add the heavy representation sweep (arm-a: raw_conv / stats / stats+spectral+slopes)
modal run sota_bundle/modal_e5_gain.py --with-arm-a

# CPU
modal run sota_bundle/modal_e5_gain.py --device cpu
```

(Or via the skill runner: `python C:/Users/apart/.claude/skills/gpu2modal/modal_runner.py run
--script sota_bundle/modal_e3_cost.py --args "--device cuda"`.)

---

## 3. Where results land + how to pull them

Both jobs write to one Modal volume: **`latad-e3e5-results`** (mounted at `/results`).

- **Incremental + resumable**: every cell/seed appends one JSON row to
  `/results/e3/rows_<device>.jsonl` (E3) or `/results/e5/rows_<device>.jsonl` (E5),
  flushed and `volume.commit()`-ed immediately. A re-launch skips any row already present,
  so a killed run resumes instead of repaying. E3 also drops a per-cell
  `/results/e3/e3_<ds>_<method>_<device>.json`.
- The local entrypoint additionally saves an aggregated
  `sota_bundle/results/e3_cost_<device>.json` / `sota_bundle/results/e5_gain_<device>.json`
  from the returned rows when the client run completes.

Pull the volume artifacts back (remote paths are **relative to the volume root — no leading
slash**, per the gpu2modal note; make the local dir first and verify non-empty):

```bash
mkdir -p sota_bundle/results/e3 sota_bundle/results/e5
modal volume ls  latad-e3e5-results e3
modal volume get latad-e3e5-results e3/rows_cuda.jsonl sota_bundle/results/e3/
modal volume get latad-e3e5-results e5/rows_cuda.jsonl sota_bundle/results/e5/
ls -lh sota_bundle/results/e3/rows_cuda.jsonl     # confirm it is non-empty (exit 0 is NOT proof)
```

---

## 4. Data / checkpoint staging (what the existing scripts do vs what E3/E5 add)

`DATA_MANIFEST.md` lists every file + size. Summary of what is `add_local_file`-staged into
`/app/`:

| Artifact | Used by existing scripts | E3 | E5 |
|---|---|---|---|
| `{wadi,HAI,SWaT}_{train,test,labels}.npy` (per-timestep) | `modal_sota` (deep baselines) | ✔ TranAD/USAD training | train+test only, **arm-a only** |
| `{wadi,HAI,SWaT}_triv_{test,thr}.npy` | `modal_sota` (easy/hard split) | ✔ (mounted, unused) | — |
| `ens_bundle/bundle_{WADI,HAI,SWaT}.npz` (window features) | `modal_experts` / `modal_ensemble` | ✔ LatAD family | ✔ all arms |
| `models_vade.py` | `modal_experts` / `modal_ensemble` | ✔ | ✔ |
| `modal_sota.py` | — | ✔ (imports `_patch_harness`) | — |
| `rev4_stats.py` | — | — | ✔ (protocol reference) |
| `zenodo_bundle/checkpoints/vade_{WADI,HAI,SKAB}.pt` | `export_checkpoints` produces them | ✔ (LatAD-global inference/params/mem) | — |

**What E3/E5 add beyond the existing pipeline:** the VaDE checkpoints (for inference-time
params/memory without re-training), `modal_sota.py` on the import path (E3 reuses its
proven `_patch_harness`), `rev4_stats.py` (E5's difficult-subset protocol), `psutil`, and
a GPU function.

### SWaT VaDE checkpoint is ABSENT
`zenodo_bundle/checkpoints/` has `vade_WADI.pt`, `vade_HAI.pt`, `vade_SKAB.pt` — **no
`vade_SWaT.pt`** (SKAB is not an E3/E5 dataset). Consequences:
- **E3 LatAD-global on SWaT** cannot load a checkpoint → it trains fresh and reports
  `"source": "trained (no checkpoint on volume)"`; its params/latency/memory are from that
  freshly-trained model. WADI and HAI load the shipped checkpoint.
- The SWaT global-VaDE hyperparameters `(K, latent_dim)` are not in `export_checkpoints.py`
  either; both scripts **default SWaT to HAI's `(40, 16)`** — see the TODO below.
- To remove this gap, run `python export_checkpoints.py SWaT` first (needs
  `eda_real`/raw-data access + a confirmed SWaT `(K, latent_dim)`), then re-stage the new
  `vade_SWaT.pt`.

### Upload footprint
Union of all staged files (both scripts): **≈ 392 MB**.
- E3 stages essentially all of it (per-timestep arrays dominate: HAI_train 130 MB,
  HAI_test 105 MB).
- E5 **decisive arms only** need just bundles + `models_vade.py` + `rev4_stats.py` ≈ **64 MB**;
  `--with-arm-a` additionally stages the train/test `.npy` (→ ≈ 374 MB) for raw windows.

---

## 5. TODOs a human MUST confirm before the first real run

Marked `# TODO(confirm):` in the scripts. In priority order:

1. **GPU choice is a deliberate addition** — the published pipeline is CPU-only. Confirm
   A10G is acceptable for the "GPU memory / latency" table, or switch the `gpu=` string.
2. **`AE` baseline definition (E3)** — the imperial-qore/TranAD harness has no plain `AE`
   model, and the bundle's `AE` column is a window-feature reconstruction baseline. E3
   implements `AE` as a **dense autoencoder over the window-feature vector** (same input as
   LatAD). Confirm this matches the paper's "AE"; if the paper means a per-timestep sequence
   AE, swap the architecture and train on the `{pfx}_train.npy` arrays.
3. **Deep-baseline inference interface (E3, `_bench_deep`)** — training wall-clock reuses
   `modal_sota._patch_harness` faithfully, but the direct model build for params/latency
   assumes `src.models.<Model>(n_features)` and a per-window forward shape
   (TranAD `(W,1,feats)`; USAD `(1, W*feats)`). Confirm the constructor arg and forward
   signature against the cloned harness; a mismatch is caught and reported as
   `infer_error` (train_s stays valid) rather than crashing the cell.
4. **Deep-baseline training device** — the harness trains float32-on-CPU by design (forcing
   CUDA broke `torch.FloatTensor(data)` per `modal_sota`). So E3's `train_s` for TranAD/USAD
   uses the harness's own device policy regardless of `--device`; GPU mem/latency come from
   the separate on-device model build. Confirm this is the intended accounting.
5. **SWaT `(K, latent_dim)`** — defaulted to HAI's `(40, 16)` in both scripts
   (`LATAD_CFG`). Confirm or set the real SWaT values (and ideally export a `vade_SWaT.pt`).
6. **arm-a raw-window alignment (E5)** — `W=60/stride=30` reproduces the bundle window
   *counts* exactly (WADI 575, HAI 14819, SWaT 1087); the script `assert`s the count match.
   Confirm the window **order/start-index/drop-last** convention matches how
   `bundle_<DS>.npz` was built, so the bundle `y`/`maxz` difficult masks line up 1:1.
7. **arm-a raw_conv encoder (E5)** — implemented as a compact Conv1d VaDE-style encoder +
   the same high-K density GMM (models_vade's `VaDE` has only a dense MLP encoder). Confirm
   whether the paper wants Conv1d vs GRU, and the encoder width/epochs.

Everything compiles clean: `python -m py_compile modal_e3_cost.py modal_e5_gain.py`.
