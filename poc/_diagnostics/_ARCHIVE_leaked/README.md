# Archived leaked / stale / superseded artifacts

These files were moved here (not deleted) to stop leaked or superseded numbers
from sitting alongside clean ones and confusing future analysis. A leaked GDN
dump previously misled a verification pass into "correcting" a clean manuscript
cell to a leaked value; quarantining removes that trap.

Reference point: the SWaT_canon train/test data-leak fix, git commit `1224d49`
(2026-09-19). Before the fix, `SWaT_canon` trained on data containing the test-normal
rows; after, it uses the official Dec-2015 iTrust Normal_v1. A separate stale-bundle
issue: some expert bundles are S=25 or lack the `fit_surprise` calibration array; the
clean canonical bundle is `sota_bundle/experts_full` (S=24, has `fit_surprise`).

Nothing here is referenced by an active code path any more (the GDN scripts and the
`ensemble_final` default were rewired/hardened before these were moved). Do not cite
any number from these files.

| archived file (original path) | why quarantined | clean replacement |
|---|---|---|
| `score_GDN_SWaT_canon_s0.npy` (`_diagnostics/`) | Leaked + numerically corrupted GDN dump (pre-fix mtime 2026-09-18, values up to 3.9e14). Produced the spurious SWaT double-hard GDN=0.571. | `_diagnostics/sota_pull_official/score_GDN_SWaT_canon_s0.npy` (clean official-normal; gives the verified 0.115). GDN scripts now point here. |
| `labels_GDN_SWaT_canon_s0.npy` (`_diagnostics/`) | Label companion of the leaked dump above. | regenerated from the clean batch. |
| `a8_valley_real.jsonl.leaked_bak` (`_diagnostics/`) | Self-marked pre-fix backup. | `_diagnostics/a8_valley_real.jsonl`. |
| `a8_drift_unification.v0.json` / `.v0.log` (`_diagnostics/`) | Self-marked earlier version. | `_diagnostics/a8_drift_unification.json`. |
| `clean_recompute_backup/` (`_diagnostics/`, 8 files) | Pre-fix snapshot (mtimes 2026-09-18), including a pre-official `scores_SWaT_canon.npz`. | `_diagnostics/scores_SWaT_canon.npz` (current, official-normal batch). |
| `swat_canon_cache.log` (`_diagnostics/`) | Stale EDA-load log, predates the fix. | n/a (log only). |
| `scores_sota_ms_SWaT.npz` (`_diagnostics/`) | Legacy non-canonical `"SWaT"` SOTA bundle (>40 days stale); the pipeline uses `SWaT_canon`. | `_diagnostics/scores_sota_ms_SWaT_canon.npz`. |
| `experts_variants__cur_avg25/` (`sota_bundle/experts_variants/cur_avg25/`) | Stale S=25 bundle, no `fit_surprise`; produced wrong SWaT numbers. Was read by `dig_swat_perwindow.py` (exploratory); that script must be repointed at `experts_full` before reuse. | `sota_bundle/experts_full` (S=24). |
| `experts_variants__var_avg15/` (`sota_bundle/experts_variants/var_avg15/`) | Stale bundle, no `fit_surprise`, no live reference. | `sota_bundle/experts_full`. |

## Code changes made alongside this archival (so the leak cannot re-enter)
- `rev4_doublehard_pca_all3.py`, `rev4_doublehard_pca.py`, `_diagnostics/gdn_doublehard.py`,
  `_diagnostics/gdn_rescore_clean.py`: GDN SWaT input repointed to the clean
  `sota_pull_official/score_GDN_SWaT_canon_s0.npy`.
- `ensemble_final.py`: default `EXPERTS_DIR` changed from `sota_bundle/experts` (old
  format) to `sota_bundle/experts_full`, plus a fail-fast assert that the loaded bundle
  contains `fit_surprise` (a stale/old-format bundle now errors loudly instead of
  silently resurrecting bad numbers).
- `models_vade.py`: `anomaly_score_hard` docstring corrected (the reported model uses the
  density+nearest base, `use_near=True`; the old comment wrongly claimed density-only).
- `_diagnostics/doublehard_pca_all3.json` and `.md` regenerated from the clean GDN input
  (SWaT double-hard GDN now 0.115, matching the manuscript).
