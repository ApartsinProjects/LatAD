# E3 / E5 data manifest — files staged to Modal `/app/`

All paths are relative to the repo root `E:\Projects\Backlog\LatAD\poc\`. Files are
`add_local_file`-staged into `/app/` by `modal_e3_cost.py` / `modal_e5_gain.py` (same
mechanism as the existing `modal_sota.py` / `modal_experts.py`). Sizes are `stat`-measured.

| File | Bytes | MB | E3 | E5 | Purpose |
|---|--:|--:|:-:|:-:|---|
| `models_vade.py` | 23,075 | 0.02 | ✔ | ✔ | VaDE model + `anomaly_score_hard` scoring |
| `sota_bundle/modal_sota.py` | 16,072 | 0.02 | ✔ | | reuse `_patch_harness` for TranAD/USAD training |
| `rev4_stats.py` | 8,205 | 0.01 | | ✔ | difficult-subset AUROC protocol reference |
| `sota_bundle/wadi_train.npy` | 38,601,464 | 38.6 | ✔ | arm-a | WADI per-timestep train (123 ch) |
| `sota_bundle/wadi_test.npy` | 8,502,380 | 8.5 | ✔ | arm-a | WADI per-timestep test |
| `sota_bundle/wadi_labels.npy` | 138,376 | 0.14 | ✔ | | WADI per-timestep labels |
| `sota_bundle/wadi_triv_test.npy` | 69,252 | 0.07 | ✔ | | (mounted, unused by E3) |
| `sota_bundle/wadi_triv_thr.npy` | 132 | 0.00 | ✔ | | (mounted, unused by E3) |
| `sota_bundle/HAI_train.npy` | 129,988,928 | 130.0 | ✔ | arm-a | HAI per-timestep train (59 ch) |
| `sota_bundle/HAI_test.npy` | 104,925,728 | 104.9 | ✔ | arm-a | HAI per-timestep test |
| `sota_bundle/HAI_labels.npy` | 3,556,928 | 3.56 | ✔ | | HAI per-timestep labels |
| `sota_bundle/HAI_triv_test.npy` | 1,778,528 | 1.78 | ✔ | | (mounted, unused by E3) |
| `sota_bundle/HAI_triv_thr.npy` | 132 | 0.00 | ✔ | | (mounted, unused by E3) |
| `sota_bundle/SWaT_train.npy` | 22,184,720 | 22.2 | ✔ | arm-a | SWaT per-timestep train (51 ch) |
| `sota_bundle/SWaT_test.npy` | 6,660,932 | 6.66 | ✔ | arm-a | SWaT per-timestep test |
| `sota_bundle/SWaT_labels.npy` | 261,336 | 0.26 | ✔ | | SWaT per-timestep labels |
| `sota_bundle/SWaT_triv_test.npy` | 130,732 | 0.13 | ✔ | | (mounted, unused by E3) |
| `sota_bundle/SWaT_triv_thr.npy` | 132 | 0.00 | ✔ | | (mounted, unused by E3) |
| `sota_bundle/ens_bundle/bundle_WADI.npz` | 9,464,902 | 9.46 | ✔ | ✔ | WADI window features (Xn_w 2614×738), y, maxz |
| `sota_bundle/ens_bundle/bundle_HAI.npz` | 48,227,518 | 48.2 | ✔ | ✔ | HAI window features (Xn_w 18359×354) |
| `sota_bundle/ens_bundle/bundle_SWaT.npz` | 5,859,022 | 5.86 | ✔ | ✔ | SWaT window features (Xn_w 3623×306) |
| `zenodo_bundle/checkpoints/vade_WADI.pt` | 8,203,806 | 8.20 | ✔ | | LatAD-global fitted VaDE (feat 738) |
| `zenodo_bundle/checkpoints/vade_HAI.pt` | 2,977,607 | 2.98 | ✔ | | LatAD-global fitted VaDE (feat 354) |
| `zenodo_bundle/checkpoints/vade_SKAB.pt` | 334,750 | 0.33 | ✔ | | staged for completeness; SKAB is not an E3/E5 dataset |

**`vade_SWaT.pt` does NOT exist** — E3 trains SWaT's LatAD-global fresh (see README §4).

### Footprint
- **Union of both scripts:** ≈ **392 MB** (E3 stages nearly all of it; the HAI per-timestep
  arrays are 235 MB of the total).
- **E5 decisive arms only** (no `--with-arm-a`): only the 3 bundles + `models_vade.py` +
  `rev4_stats.py` ≈ **64 MB**. `--with-arm-a` adds the train/test `.npy` → ≈ **374 MB**.

Bundle window widths = `nch × 6` stats per channel: WADI 738 (123 ch), HAI 354 (59 ch),
SWaT 306 (51 ch). Bundle window counts match `W=60, stride=30` over the per-timestep test
arrays (WADI 575, HAI 14819, SWaT 1087).

### Results volume (outputs, pulled back)
`latad-e3e5-results` (created on first run): `e3/rows_<device>.jsonl`,
`e3/e3_<ds>_<method>_<device>.json`, `e5/rows_<device>.jsonl`. See README §3.
