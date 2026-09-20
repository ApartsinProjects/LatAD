# LatAD `poc/` cleanup inventory — leaked / stale / superseded artifacts

Read-only manifest. Nothing moved, deleted, or edited. Reference point: SWaT leak fix commit
`1224d49` (2026-09-19 21:41:56 +03:00, "fix SWaT_canon train/test data leak"). Generated
2026-09-20 by an inventory pass; mtimes are local filesystem mtimes (may differ from git commit
times when a file was regenerated locally without being committed).

Legend for the "recommended action" column: **quarantine** = safe to move to an archive dir later
with no code-path impact found; **regenerate** = a live script reads this file, so quarantining it
before regenerating a replacement would break that path; **keep** = looks stale by name/date but is
either still authoritative or ambiguous — do not touch; **investigate** = contradiction found, needs
a human decision before any action.

## 1. Leaked GDN / SOTA score dumps for SWaT

| path | mtime | category | why | clean supersede | referenced by (scripts) | recommended action |
|---|---|---|---|---|---|---|
| `_diagnostics/score_GDN_SWaT_canon_s0.npy` | 2026-09-18 11:03 (**pre-fix**) | leaked | `GDN_FILE["SWaT_canon"]` in `rev4_doublehard_pca.py`/`rev4_doublehard_pca_all3.py` resolves to exactly this path (`OUT = _diagnostics`). Values are corrupted/absurd: min 0.208, **max 3.9e14**, mean 5.2e10 (6 values > 1e6) — not a plausible per-timestep GDN reconstruction score. Never regenerated after the fix (mtime unchanged since before `1224d49`). | `_diagnostics/sota_pull_official/score_GDN_SWaT_canon_s0.npy` (mtime 2026-09-19 21:16, sane range: min 0.194, max 31.2, mean 4.87) — **but this clean file is not wired into the pipeline anywhere**; see "REGENERATE FIRST" below. | `rev4_doublehard_pca.py:122`, `rev4_doublehard_pca_all3.py:122`, `_diagnostics/gdn_doublehard.py:14` (hardcoded abs path), `_diagnostics/gdn_rescore_clean.py:10` | **regenerate first** (highest risk — see §5) |
| `_diagnostics/labels_GDN_SWaT_canon_s0.npy` | 2026-09-18 11:03 (pre-fix, paired with the above) | leaked | Label companion to the corrupted score dump above; same generation run. | Regenerate alongside the score file from `sota_pull_official/` or a fresh run. | Not directly opened by name in any `.py` found (companion artifact of the run that produced `score_GDN_SWaT_canon_s0.npy`) | quarantine after regenerate |
| `_diagnostics/sota_pull_official/score_GDN_SWaT_canon_s0.npy` (+ 9 sibling `score_{TranAD,USAD}_SWaT_canon_s{0..4}.npy`) | 2026-09-19 21:16–21:17 (~25 min **before** the 21:41 fix commit landed, i.e. the regenerated-official-data batch that the commit codifies) | **clean, orphaned** | Sane value range (max 31.2 vs the leaked file's 3.9e14); this looks like the official post-fix pull, but it is not referenced by `GDN_FILE`, `gdn_rescore_clean.py`, or `gdn_doublehard.py` — all three still point at the stale sibling in `_diagnostics/` directly. | — (this *is* the clean candidate) | none found (orphaned) | keep; wire in before quarantining the stale sibling |
| `_diagnostics/scores_sota_ms_SWaT_canon.npz` | 2026-09-19 21:17:02 (same official-pull batch) | likely clean | Multi-seed SOTA bundle for `SWaT_canon`, timestamp matches the clean `sota_pull_official/` batch, not the corrupted single-file GDN dump. | — | `_diagnostics/merge_sota_swat_canon.py` (writer, not consumer, of this path) | keep |
| `_diagnostics/scores_sota_ms_SWaT.npz` | 2026-08-09 14:47 (over a month stale, and keyed to `"SWaT"` not `"SWaT_canon"`) | superseded | Legacy non-canonical `SWaT` variant, predates the canonical `SWaT_canon` pipeline entirely; the manuscript now uses `SWaT_canon` throughout. | `scores_sota_ms_SWaT_canon.npz` | none found referencing `"SWaT"` (non-canon) key in active `.py` (only `scores_sota_ms_SWaT_canon` is read) | quarantine |
| `_diagnostics/scores_SWaT_canon.npz` | 2026-09-19 21:17:02 (same official-pull batch) | likely clean | Matches the same clean batch as `sota_pull_official/`; not the pre-fix `clean_recompute_backup` copy. | — | multiple scripts read `scores_SWaT_canon.npz` via `eda_real`/build-table helpers (this is the live cache) | keep |

## 2. Stale expert bundles (`sota_bundle/experts*`)

| path | mtime | S (expert count) | has `fit_surprise`? | category | evidence | referenced by | recommended action |
|---|---|---|---|---|---|---|---|
| `sota_bundle/experts_full/expert_SWaT_canon.npz` | 2026-09-19 12:33 | 24 | yes, shape `(5,24,1292)` | **clean (canonical)** | leak-free official normal, S=24, has `fit_surprise` calibration array | default `EXPERTS_DIR` in nearly every active `_diagnostics/*.py` script and `rev4_doublehard_pca*.py` (11+ scripts) | keep |
| `sota_bundle/experts_variants/clean_reb/expert_SWaT_canon.npz` | 2026-09-19 21:57 | 24 | yes | clean (variant, same S=24) | identical shape/keys to `experts_full`, newer mtime — likely a rebuild-and-verify copy | not found referenced by path in any `.py` | keep (or fold into `experts_full` later — not a leak risk either way) |
| `sota_bundle/experts_variants/clean_swat/expert_SWaT_canon.npz` | 2026-09-19 12:32 | 24 | yes | clean (variant, same S=24) | identical to `experts_full` | `_diagnostics/comm_gate_eval.py` (takes `tag` as CLI/var arg — only stale if invoked with a stale tag) | keep |
| `sota_bundle/experts/expert_SWaT_canon.npz` | 2026-09-17 13:25 | — (no `fit_surprise` key at all) | **no** | **stale (old format)** | predates the `fit_surprise` calibration field entirely; this is the OLD default `EXPERTS_DIR` fallback (`ensemble_final.py:118` falls back to `sota_bundle/experts` — not `experts_full` — if `EXPERTS_DIR` is unset) | `ensemble_final.py:118` (soft default only; every caller found explicitly overrides `EXPERTS_DIR` to `experts_full` before importing) | **investigate** — the unset-env-var default trap should be fixed in code, not just archived (would silently resurrect stale numbers for any future script that forgets to set `EXPERTS_DIR`) |
| `sota_bundle/experts_variants/cur_avg25/expert_SWaT_canon.npz` | 2026-09-19 00:32 | **25** (per task background, confirmed no `fit_surprise` key) | no | **stale (S=25 bug)** | This is the bundle named in the background as producing the wrong SWaT double-hard GDN=0.571-adjacent numbers; S=25 not S=24, no calibration array | `_diagnostics/dig_swat_perwindow.py:118` (explicitly sets `EXPERTS_DIR` to this dir for a secondary/exploratory block in that script) | **regenerate first** if `dig_swat_perwindow.py`'s output is still cited anywhere; otherwise quarantine |
| `sota_bundle/experts_variants/cur_fit/expert_SWaT_canon.npz` | 2026-09-19 00:57 | 25, `fit_surprise` shape `(5,25,3624)` | yes | stale (S=25 variant) | S=25, different window count (3624 vs 1292) than the canonical S=24 bundle — a different trim/windowing experiment | `_diagnostics/drift_fix_eval.py:73` | keep (active experiment input) — do not quarantine without checking `drift_fix_eval.py`'s current use |
| `sota_bundle/experts_variants/trim6h/expert_SWaT_canon.npz` | 2026-09-19 07:12 | 25, `fit_surprise` shape `(5,25,3566)` | yes | stale (S=25, TRIM=72 variant) | S=25, distinct window count; named for a specific trim ablation | `_diagnostics/drift_fix_eval.py:73`, `_diagnostics/trim_fair_swat.py:68` | keep (active experiment input) |
| `sota_bundle/experts_variants/var_avg15/expert_SWaT_canon.npz` | 2026-09-19 00:37 | — (no `fit_surprise`) | no | stale (S=15-ish variant, no calibration) | Same old-format issue as `experts/` above | not found referenced by path in any `.py` | quarantine |

Note: `experts_variants/cur_fit` and `trim6h` are NOT simply "leaked GDN" residue — they are
deliberate ablation variants (different trim/windowing) still read by `drift_fix_eval.py` and
`trim_fair_swat.py`. Do not quarantine those two without first confirming those scripts' outputs
are not still cited in the manuscript or an open diagnostic.

## 3. Explicitly-marked residue (`*.leaked_bak`, `*.v0.*`)

| path | mtime | category | why | clean supersede | referenced by | recommended action |
|---|---|---|---|---|---|---|
| `_diagnostics/a8_valley_real.jsonl.leaked_bak` | 2026-09-19 21:48 | explicitly-marked leaked | Filename self-declares it as the pre-fix backup of `a8_valley_real.jsonl` | `_diagnostics/a8_valley_real.jsonl` (if present/regenerated; not separately verified in this pass) | none found (`.leaked_bak` extension means nothing globs/imports it) | quarantine (safe — self-marked, no live reference) |
| `_diagnostics/a8_drift_unification.v0.json` | 2026-09-20 08:16 | explicitly-marked superseded | `.v0` suffix self-declares an earlier version | `_diagnostics/a8_drift_unification.json` (current, unsuffixed) | none found | quarantine (safe) |
| `_diagnostics/a8_drift_unification.v0.log` | 2026-09-20 08:16 | explicitly-marked superseded | Log companion to the `.v0.json` above | `_diagnostics/a8_drift_unification.json`'s current run log | none found | quarantine (safe) |

## 4. Pre-leak-fix SWaT_canon caches / score npzs

| path | mtime | category | why | clean supersede | referenced by | recommended action |
|---|---|---|---|---|---|---|
| `_diagnostics/clean_recompute_backup/` (whole dir: `scores_SWaT_canon.npz`, `scores_HAI.npz`, `scores_WADI_clean.ORIG.npz`, `ens_hai_swat_sig.log`, `gate_probe.log`, `rebuild_WADI_clean.log`, `wadi_t56.log`) | all 2026-09-18 09:48–10:46 (pre-fix) | stale | Directory name and mtimes both mark this as a backup snapshot taken before the leak fix; `scores_SWaT_canon.npz` inside it predates the official-data regeneration by ~11 hours | `_diagnostics/scores_SWaT_canon.npz` (current, 2026-09-19 21:17) | none found referencing `clean_recompute_backup/` by path in any `.py` | quarantine (safe — self-contained backup dir, zero references) |
| `_diagnostics/swat_canon_cache.log` | 2026-09-17 11:47 | stale (log only) | Console-output log from an EDA load, predates the fix by 2+ days; low risk (not a score artifact, just a log) | n/a | none found | quarantine (safe, cosmetic) |

## 5. Diagnostic JSON/MD quoting a disputed or known-leaked number — REGENERATE, do not just archive

| path | mtime | what it quotes | status | recommended action |
|---|---|---|---|---|
| `_diagnostics/swat_leak_fix.md` | pre-existing | boosted-LOO Difficult `0.881→0.518`, HCcoh+LatAD Difficult `0.703` | These are explicitly the **pre-fix, documented-as-such** numbers inside the leak-fix writeup itself (the doc's own subject is the leak) — not mislabeled, but do not copy these numbers into the manuscript or any fresh table without the "(pre-fix)" qualifier already present in this file. | keep as historical record; do not cite standalone |
| `_diagnostics/swat_staleness_verification.md` | pre-existing (untracked, from a prior session) | Table 4 GDN SWaT double-hard: declares manuscript's `0.115` a single-cell typo and `0.571` (from `doublehard_pca_all3.json`) the "correct replacement value" | **CONTRADICTION FOUND, needs human review.** This pass traced `doublehard_pca_all3.json`'s GDN column back through `rev4_doublehard_pca_all3.py`'s `GDN_FILE` dict to `_diagnostics/score_GDN_SWaT_canon_s0.npy` — the file flagged **leaked** in §1 above (mtime pre-fix, max value 3.9e14, never regenerated). `doublehard_pca_all3.json` itself has a post-fix mtime (2026-09-19 22:13), but its GDN *input* was not refreshed at that run, so its GDN column may itself rest on the stale/leaked score dump despite the json's own timestamp looking clean. **This means 0.571 is not verified clean either** — `swat_staleness_verification.md`'s STEP 4 verdict for the GDN cell should be treated as unresolved until GDN is rescored against `_diagnostics/sota_pull_official/score_GDN_SWaT_canon_s0.npy` (or a fresh regeneration) and `doublehard_pca_all3.json` is regenerated from that clean input. | **investigate** — do not act on either `0.115` or `0.571` for the manuscript Table 4 GDN/SWaT/double-hard cell until GDN is re-run against the clean score file |
| `paper/IoT2.html` line 915 | current manuscript | prints `GDN | 0.648 | 0.354 | 0.115` (WADI, HAI, SWaT double-hard) | The `0.115` SWaT figure is the disputed cell above — confirmed by `swat_staleness_verification.md` to not match any known artifact, clean or leaked | **do not edit without regenerating GDN first** (out of scope for this read-only pass; flagging only) |
| `_diagnostics/doublehard_pca_all3.json` / `.md` | 2026-09-19 22:13 | `SWaT_canon.rows_pca.GDN.auroc = 0.571` | Downstream of the leaked `score_GDN_SWaT_canon_s0.npy` input (see above) despite its own post-fix mtime | **regenerate** after rewiring `GDN_FILE`/`gdn_rescore_clean.py`/`gdn_doublehard.py` to the clean `sota_pull_official/` score file |
| `_diagnostics/gdn_doublehard.py`, `_diagnostics/gdn_rescore_clean.py` | 2026-09-18 (both pre-fix scripts, never edited since) | n/a (scripts, not results) | Both hardcode `score_GDN_SWaT_canon_s0.npy` (the leaked path) as their GDN input | **regenerate** their outputs after fixing the input path; do not quarantine the scripts themselves (still structurally valid, just pointed at the wrong file) |
| `_diagnostics/swat_clean_ablations.md` | pre-existing | boosted-LOO Difficult `0.412 (was 0.881)`, HCcoh+LatAD Difficult `0.703` | Self-labels the `0.881` as "was" (pre-fix) and reports `0.703` as the current clean HCcoh+LatAD figure (not GDN) — this looks correctly annotated, no GDN dependency, lower risk than the GDN cell above | keep; not part of the GDN contradiction |

---

## SAFE TO QUARANTINE (no live references found)

- `_diagnostics/a8_valley_real.jsonl.leaked_bak`
- `_diagnostics/a8_drift_unification.v0.json`
- `_diagnostics/a8_drift_unification.v0.log`
- `_diagnostics/clean_recompute_backup/` (whole directory — 8 files)
- `_diagnostics/swat_canon_cache.log`
- `_diagnostics/scores_sota_ms_SWaT.npz` (legacy non-canonical `"SWaT"` key, 40+ days stale)
- `sota_bundle/experts_variants/var_avg15/expert_SWaT_canon.npz` (no `fit_surprise`, no referencer found)
- `_diagnostics/labels_GDN_SWaT_canon_s0.npy` (only after the paired score file is regenerated — see below)

## REGENERATE FIRST (referenced by active scripts — quarantining before fixing would break a live path)

- `_diagnostics/score_GDN_SWaT_canon_s0.npy` — read by `rev4_doublehard_pca.py`, `rev4_doublehard_pca_all3.py`, `_diagnostics/gdn_doublehard.py`, `_diagnostics/gdn_rescore_clean.py`. **Highest-risk item in this inventory**: it is both leaked (pre-fix, corrupted magnitude) and the sole live GDN input for the SWaT double-hard pipeline. Fix by pointing these four scripts at `_diagnostics/sota_pull_official/score_GDN_SWaT_canon_s0.npy` (or a fresh regeneration) and re-running `rev4_doublehard_pca_all3.py` before touching Table 4's GDN/SWaT cells.
- `sota_bundle/experts_variants/cur_avg25/` — read by `_diagnostics/dig_swat_perwindow.py` (line 118, secondary block). Confirm that script's output isn't cited anywhere before archiving the bundle.
- `sota_bundle/experts/` (old-format, no `fit_surprise`) — this is the silent *default* `EXPERTS_DIR` in `ensemble_final.py:118` if the env var is ever left unset. Every current caller overrides it explicitly, but the trap itself should be fixed in code (default to `experts_full`, not `experts`) before this directory is archived, or a future script will silently pick up stale numbers.

---

## Summary

- **Leaked**: 3 items (2 score/label npy pairs + `sota_pull_official` clean counterpart identified but unwired)
- **Stale**: 6 items (4 expert-bundle variants confirmed S=25 or missing `fit_surprise`, 1 legacy `scores_sota_ms_SWaT.npz`, 1 old-format `sota_bundle/experts/`)
- **Explicitly-marked residue**: 3 items (`.leaked_bak`, 2× `.v0.*`)
- **Pre-fix caches**: 1 backup directory (8 files) + 1 stale log
- **Disputed/needs-regeneration**: 1 confirmed contradiction (Table 4 GDN/SWaT double-hard cell — neither `0.115` nor `0.571` is verified clean) plus the 4 scripts/artifacts feeding it

**Highest risk**: `_diagnostics/score_GDN_SWaT_canon_s0.npy` is simultaneously (a) leaked/corrupted
(pre-fix, magnitude up to 3.9e14) and (b) the file three active scripts actually read to compute
GDN's SWaT double-hard AUROC — including the diagnostic that a prior verification pass
(`swat_staleness_verification.md`) relied on to declare `0.571` "the correct replacement value" for
the manuscript. That declaration should be treated as unverified until GDN is rescored against the
orphaned clean file at `_diagnostics/sota_pull_official/score_GDN_SWaT_canon_s0.npy`.
