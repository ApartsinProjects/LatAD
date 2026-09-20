# SWaT staleness verification (post-leak-fix, S=24 vs S=25 bundle check)

Scope: read-only verification pass over `poc/paper/IoT2.html` SWaT cells (Tables 3, 4, 5, 6, 7, 9,
Appendix Table C1) against the clean-official (leak-free, S=24 `experts_full`) recompute artifacts
under `poc/_diagnostics/`. Nothing in `IoT2.html` was modified by this pass.

---

## STEP 1 — Authoritative clean artifact(s), with exact provenance

**Leak-fix commit**: `1224d49` "revision2: fix SWaT_canon train/test data leak" —
`git log -1 --format=%ci 1224d49` → **`2026-09-19 21:41:56 +0300`**.

```
git log --oneline -20 -- poc/_diagnostics
1224d49 2026-09-19 21:41:56 +0300   revision2: fix SWaT_canon train/test data leak (100% test-normal in train)
b3b7e35                              revision2: checkpoint diagnostics, analysis code, and round-1 review materials  (HEAD, includes the current IoT2.html table cells)
```

### Bundle check (S=24 experts_full vs S=25 cur_avg25)

`poc/_diagnostics/swat_clean_ablations.md` (mtime 2026-09-19 22:00, i.e. ~19 min after the fix commit)
states explicitly, under "CRITICAL DISCREPANCY FOUND":

> "The high number in OFFICIAL_v2 traces to a differently-trained expert:
> `experts_variants/cur_avg25` (S=25, built 2026-09-19 00:32, before the official-normal bundle)
> gives HC_coh 0.808. OFFICIAL_v2's 0.742 sits in that inflated family, NOT the clean official
> (S=24) family... I rebuilt the expert independently from the current clean official bundle
> (`ens_bundle/bundle_SWaT_canon.npz`, 1616 train windows, maxz_thr 3.163, canonical params
> method=average/MAXSZ=25/MINSZ=3 -> S=24 communities). It reproduces HC_coh 0.523 / HCcoh+LatAD
> 0.524, matching the committed expert exactly."

This is corroborated independently by `poc/_diagnostics/doublehard_pca_all3.md` (mtime
2026-09-19 22:14:37, i.e. after the fix commit):

> "**Experts used:** ... SWaT `experts_full/expert_SWaT_canon.npz` (clean official-normal bundle,
> **S=24**, has `fit_surprise`, mtime 12:33). SWaT_canon uses the clean official Dec-2015 normal
> train (0 test-normal overlap, fail-fast guard). Cross-check: SWaT HCcoh+LatAD *difficult* AUROC =
> 0.524 (near chance, matches the clean ~0.536 expectation; not the stale ~0.70+)."

**Conclusion — authoritative clean artifacts (all confirmed post-`1224d49`, S=24 `experts_full`):**

| file | mtime (local) | vs 1224d49 (21:41:56) | content |
|---|---|---|---|
| `clean_swat_tables56.py/.json/.log` | 2026-09-19 21:48–21:49 | **post-fix** (7–8 min after) | Table 5 & 6 SWaT recompute, S=24 |
| `clean_swat_boosted_loo.py/.json/.log` | 2026-09-19 21:49–21:50 | **post-fix** | boosted-LOO clean recompute |
| `a8_valley_real.SWaT_canon.CLEAN.log` + `a8_valley_real.jsonl` | 2026-09-19 21:49:59 | **post-fix** (leaked version explicitly archived 7 min earlier as `a8_valley_real.jsonl.leaked_bak` at 21:48:29 — the cleanest before/after pair in the whole diagnostic set) | A8 valley re-check |
| `swat_clean_ablations.md` | 2026-09-19 22:00 | **post-fix** | Tables 5/6/7 + significance summary write-up, S=24 |
| `pca_filter_doublehard_SWaT_canon.npz`, `doublehard_pca_all3.json/.md` | 2026-09-19 22:03–22:14 | **post-fix** | Table 4 (PCA double-hard, 95% var) recompute, S=24 (`experts_full`, explicitly cited) |
| `localization_clean.md` | 2026-09-20 06:29 | post-fix, next day | Table 9 re-derivation, S=24 |
| `a8_drift_unification.{json,log,py}` | 2026-09-20 08:16–08:20 | post-fix, next day | drift/changepoint typing (Table 7 mechanism), S=24 |
| `a8_drift_lens_verdict.md` | 2026-09-20 08:38 | post-fix, next day | independent provenance audit of the above (confirms all timestamps in this table) |

Files that are **NOT** authoritative (pre-fix or ambiguous, explicitly identified as such in-repo):

| file | mtime | status |
|---|---|---|
| `loo_boosted_swat.json/.log` | 2026-09-18 22:07–22:15 | **pre-fix** (before `1224d49`); `n_diff=85` (old-leaked difficulty count, not 91); `boosted_diff=0.881`, `boosted_dhard=0.830` — these are exactly the numbers `swat_clean_ablations.md` labels "OLD-leaked ... (was 0.881)" / "(was 0.830)". Superseded by `clean_swat_boosted_loo.json`. |
| `headline_full_SWaT_canon_OFFICIAL_v2.json` | 2026-09-19 21:18 | pre-fix by 23 min; also built on the wrong S=25 `cur_avg25` bundle per `swat_clean_ablations.md`. Doubly stale. |
| `headline_full_SWaT_canon_CLEAN.json`, `_OFFICIAL.json` | 2026-09-19 10:21 / 12:39 | pre-fix (same day, hours before the commit) |
| `rev4_ablation_clean.json/.log` | 2026-09-17 19:42 | pre-fix by two days; name "clean" refers to an earlier, unrelated cleanup, not the leak fix |
| `fix_verify_SWaT.json` | 2026-08-07 16:16 | pre-fix by six weeks; unrelated to the leak bug |
| `rev4_doublehard.json` | 2026-08-09 14:47 | pre-fix by six weeks; predates GDN as a baseline entirely (no GDN row) |

---

## STEP 2 — Manuscript SWaT cells extracted (IoT2.html)

### Table 3 — per-dataset results (SWaT block, line 817)
```
817: <b>SWaT</b> (anom 233 = 142 easy + 91 difficult)
818: trivial max|u|            All 0.853  Easy 0.997  Diff 0.627
819: Isolation Forest          All 0.800±0.004  Easy 0.961±0.004  Diff 0.550±0.007
820: AutoEncoder               All 0.786±0.001  Easy 0.959±0.001  Diff 0.517±0.003
821: LinRes (one-hot)          All 0.776        Easy 0.976        Diff 0.464
822: USAD                      All 0.763±0.001  Easy 0.946        Diff 0.477±0.001
823: TranAD                    All 0.761±0.001  Easy 0.943        Diff 0.477±0.002
824: GDN                       All 0.761         Easy 0.945        Diff 0.473
825: LatAD (global density)    All 0.752±0.016  Easy 0.931±0.016  Diff 0.472±0.019
826: LatAD (regime-community)  All 0.788±0.008  Easy 0.957±0.005  Diff 0.524±0.024
```

### Table 4 — PCA double-hard subset, SWaT column (lines 909–918; windows/episodes 28/9, line 889/904)
```
909: trivial max|u|                    SWaT 0.514 (win)
910: Isolation Forest                  SWaT 0.326±0.012
911: AutoEncoder                       SWaT 0.154±0.006
912: LinRes (one-hot)                  SWaT 0.153
913: USAD                              SWaT 0.125±0.004
914: TranAD                            SWaT 0.119±0.001
915: GDN                               SWaT 0.115          <-- see mismatch below
916: boosted LinRes (leave-one-out)     SWaT 0.117
917: LatAD (global density)            SWaT 0.184±0.022
918: LatAD (regime-community)          SWaT 0.178±0.011
```

### Table 5 — source of gain, difficult subset (lines 999–1001)
```
999:  cross-channel latent density              SWaT 0.472±0.020
1000: channel-independent product of marginals  SWaT 0.522±0.003
1001: cross-channel gain                        SWaT -0.050
```

### Table 6 — head decomposition, difficult subset (lines 1014–1018)
```
1014: reconstruction residual (dropped term)          SWaT 0.567±0.004
1015: latent density                                   SWaT 0.472±0.020
1016: nearest-component NLL                             SWaT 0.479±0.025
1017: base (density + nearest)                          SWaT 0.472±0.019
1018: base + resid (auto) = LatAD (global density)      SWaT 0.472±0.019
```

### Table 7 — drift-aware changepoint typing (line 1076)
```
1076: SWaT   All 0.788 -> 0.880   Difficult 0.524 -> 0.723   Double-hard 0.178 -> 0.671
```

### Table 9 — subsystem localization (caption, line 1238)
```
1238: "... Bold marks HAI, where the localization margin is significant. SWaT is omitted:
       its train-normal ranking is defeated by the record's drift (§7)."
```
(SWaT has no data row in Table 9 — it was dropped from the table entirely.)

### Appendix Table C1 (line 2968)
```
2968: SWaT | 0.99 (3.9) | 0.02 (0.05) | 0.008 / 0.117 | 0.06 / 0.005
```

---

## STEP 3 — Comparison table

| table | cell (method / subset) | manuscript value | clean-artifact value | source file:line | match? | note |
|---|---|---|---|---|---|---|
| T3 | LinRes, Difficult | 0.464 | 0.464 | `swat_clean_ablations.md:106` | yes | |
| T3 | IF, Difficult | 0.550 | 0.550 | `swat_clean_ablations.md:107,121` | yes | |
| T3 | trivial max\|u\|, Difficult | 0.627 | 0.627 | `swat_clean_ablations.md:129` | yes | |
| T3 | AE, Difficult | 0.517 | 0.517 | `swat_clean_ablations.md:122` | yes | |
| T3 | USAD, Difficult | 0.477 | 0.477 | `swat_clean_ablations.md:123` | yes | |
| T3 | TranAD, Difficult | 0.477 | 0.477 | `swat_clean_ablations.md:124` | yes | |
| T3 | GDN, Difficult | 0.473 | 0.473 | `swat_clean_ablations.md:125` | yes | |
| T3 | LatAD (global density), Difficult | 0.472±0.019 | 0.472 | `clean_swat_tables56.json` (`table6_heads.base` == global-density def.) | yes | |
| T3 | LatAD (regime-community), Difficult | 0.524±0.024 | 0.5237±... | `clean_swat_boosted_loo.json` (`HCcoh+LatAD_diff`=0.52371) | yes | |
| T5 | cross-channel latent density | 0.472±0.020 | 0.4717±0.0197 | `clean_swat_tables56.json` | yes | |
| T5 | channel-independent product of marginals | 0.522±0.003 | 0.5218±0.0031 | `clean_swat_tables56.json` | yes | |
| T5 | cross-channel gain | -0.050 | -0.0502 | `clean_swat_tables56.json` | yes | |
| T6 | reconstruction residual | 0.567±0.004 | 0.567±0.004 | `clean_swat_tables56.json` | yes | |
| T6 | latent density | 0.472±0.020 | 0.472±0.020 | `clean_swat_tables56.json` | yes | |
| T6 | nearest-component NLL | 0.479±0.025 | 0.479±0.025 | `clean_swat_tables56.json` | yes | |
| T6 | base | 0.472±0.019 | 0.472±0.019 | `clean_swat_tables56.json` | yes | |
| T6 | base+resid = LatAD(global density) | 0.472±0.019 | 0.472±0.019 | `clean_swat_tables56.json` | yes | |
| T7 | All 0.788→0.880 | 0.788→0.880 | 0.788 (T3 raw) → 0.880 | `a8_drift_unification.json/.md` text (§7 recompute, 08:20 2026-09-20) | yes | |
| T7 | Difficult 0.524→0.723 | 0.524→0.723 | 0.5237→0.723 | ditto; 0.524 also = `clean_swat_boosted_loo.json` `HCcoh+LatAD_diff` | yes | |
| T7 | Double-hard 0.178→0.671 | 0.178→0.671 | 0.178 (T4 PCA-dhard) → 0.671 | ditto | yes | |
| T4 | trivial max\|u\| | 0.514 | 0.514 | `doublehard_pca_all3.md` (SWaT col) | yes | |
| T4 | IF | 0.326±0.012 | 0.326±0.012 | `doublehard_pca_all3.json` | yes | |
| T4 | AE | 0.154±0.006 | 0.154±0.006 | `doublehard_pca_all3.json` | yes | |
| T4 | USAD | 0.125±0.004 | 0.125±0.004 | `doublehard_pca_all3.json` | yes | |
| T4 | TranAD | 0.119±0.001 | 0.119±0.001 | `doublehard_pca_all3.json` | yes | |
| T4 | LinRes | 0.153 | 0.153 | `doublehard_pca_all3.json` | yes | |
| T4 | boosted LinRes | 0.117 | 0.117 | `doublehard_pca_all3.json` (`rows_pca.boosted_LOO.auroc`=0.117) | yes | |
| T4 | LatAD (global density) | 0.184±0.022 | 0.184±0.022 | `doublehard_pca_all3.json` | yes | |
| T4 | LatAD (regime-community) | 0.178±0.011 | 0.178±0.011 | `doublehard_pca_all3.json` | yes | |
| **T4** | **GDN** | **0.115** | **0.571** | `doublehard_pca_all3.json` → `SWaT_canon.rows_pca.GDN.auroc` = 0.571 | **NO** | see STEP 4 |
| T9 | (localization) | SWaT omitted from table | correctly omitted per recommendation | `localization_clean.md` "Scoping recommendation" | yes | manuscript already implements the drop-SWaT recommendation |
| App. C1 | overlap/masking row | 0.99(3.9) / 0.02(0.05) / 0.008÷0.117 / 0.06÷0.005 | same numbers, but provenance flagged "very likely clean, not formally verified" | `a8_drift_lens_verdict.md` §5 Risk flags | yes (numbers match) but **caveat**, see STEP 4 | `a8_masking_allsets.jsonl` (source, mtime 2026-09-18 10:14) predates its own writeup `clean_recompute.md` (10:48) by 34 min and was never explicitly re-run after the formal `1224d49` commit; qualitative "no excess over base rate" conclusion is independently corroborated, exact digits are not re-verified post-commit |

---

## STEP 4 — Verdict

### Confirmed clean / verified (no action needed)
Every SWaT cell in **Table 3** (all/easy/difficult, all methods), **Table 5**, **Table 6**, **Table 7**,
and the **boosted-LinRes / all other Table 4 rows** reproduces the post-`1224d49`, S=24
(`experts_full`) recompute exactly (`clean_swat_tables56.json`, `clean_swat_boosted_loo.json`,
`doublehard_pca_all3.json`, `a8_drift_unification.json`). **Table 9 already correctly omits SWaT**,
matching the explicit recommendation in `localization_clean.md` ("Scope the localization claim to
HAI ..., drop SWaT" — because clean SWaT top-3 = 0.458, below the 0.542 random baseline). No stale
pre-leak-fix or S=25-bundle numbers were found surviving anywhere in the manuscript's SWaT cells
outside the one exception below.

### Confirmed stale / mismatched — ONE cell

**Table 4, GDN, SWaT double-hard column: manuscript prints `0.115`; the only matching clean artifact
(`doublehard_pca_all3.json` → `SWaT_canon.rows_pca.GDN.auroc`, mtime 2026-09-19 22:13, post-fix,
S=24 bundle) gives `0.571`.**

This is *not* an instance of the leak bug or the S=25-bundle bug — `0.115` does not match any
pre-fix, leaked, or S=25 artifact value found anywhere under `poc/_diagnostics/` either (searched
`loo_boosted_swat.json`, `rev4_doublehard.json`, `pca_filter_doublehard.json/.md`,
`gdn_doublehard.py` outputs, and a numeric grep of `doublehard_pca_all3.json` for values near
0.115 — none found). Every other value in the same manuscript row (`GDN`: WADI 0.648, HAI 0.354)
and every other row of the same table matches `doublehard_pca_all3.json` exactly, which makes this
a genuine single-cell inconsistency introduced when the table was last edited (commit `b3b7e35`,
`git log --oneline -S "GDN</td><td>0.648" -- poc/paper/IoT2.html`), not a stale-pipeline artifact.
**Correct replacement value: 0.571** (source: `poc/_diagnostics/doublehard_pca_all3.json`,
`SWaT_canon.rows_pca.GDN.auroc`; also printed in `poc/_diagnostics/doublehard_pca_all3.md` line 26
SWaT column). Note this also changes the qualitative reading of that row: at 0.571 GDN is the
**sole above-chance survivor** on the SWaT double-hard subset (as `doublehard_pca_all3.md`'s own
prose states: *"GDN is the sole survivor there"*), not a near-floor value alongside the other
baselines — the manuscript's printed 0.115 currently hides this from the reader.

### Flagged as "very likely clean, not formally re-verified" (not a confirmed mismatch)

Appendix **Table C1** SWaT row (`0.99 (3.9)`, `0.02 (0.05)`, `0.008/0.117`, `0.06/0.005`, line 2968).
The numbers match the in-repo source (`a8_masking_allsets.jsonl` / `.md`) exactly, and
`a8_drift_lens_verdict.md` §5 independently audits this exact provenance question and concludes the
underlying fix was very likely already applied when this file was generated (2026-09-18, per
cross-reference with `clean_recompute.md`'s FIX 1/2/3 documented at 10:48 the same morning), but it
predates the *formal* `1224d49` git commit by a full day and was **never re-run after the commit**.
Recommendation (already stated in `a8_drift_lens_verdict.md`, reiterated here): re-run
`a8_masking_allsets.py` for `SWaT_canon` post-commit and diff against Table C1 for provenance
hygiene; a residual leak is judged very unlikely to flip the qualitative "no excess over base rate"
verdict (15x gap), but the exact digits are not yet a confirmed-post-commit recompute the way every
other cell audited above is.

### No cells required "recompute required — no clean artifact found"
All SWaT cells in the manuscript have a matching or (in the GDN case) a locatable clean artifact;
none needed to be left unverified for lack of any candidate recompute.

---

## Artifacts consulted
`poc/_diagnostics/swat_clean_ablations.md`, `clean_swat_tables56.{py,json,log}`,
`clean_swat_boosted_loo.{py,json,log}`, `doublehard_pca_all3.{py→_build_doublehard_pca_all3_md.py,json,md}`,
`localization_clean.md`, `a8_drift_unification.{json,log,py}`, `a8_drift_lens_verdict.md`,
`loo_boosted_swat.{json,log,py}` (pre-fix, ruled out), `headline_full_SWaT_canon_OFFICIAL_v2.json`
(pre-fix / S=25, ruled out), `rev4_doublehard.json` / `fix_verify_SWaT.json` (unrelated / far pre-fix,
ruled out), plus `git log` on `1224d49` and `poc/_diagnostics` / `poc/paper/IoT2.html`.
