# Review-2 recompute (CLEAN pipeline, FIX 1-3)

WADI = `WADI_clean`; difficulty subset = clean 30-window mask from `scores_WADI_clean.npz`
(`maxz <= maxz_thr`, thr 6.0076, 30 difficult windows of 56 anomalies / 575 windows). Experts
`sota_bundle/experts_full`, HEAD `HCcoh+LatAD` (the paper's "LatAD regime-community" headline).
5 seeds. Python `C:\Python314\python`. Raw numbers persisted to `_diagnostics/review2_recompute.json`;
run log `_diagnostics/review2_recompute.log`; script `_diagnostics/review2_recompute.py`.

---

## 1. Table A2 (WADI) — feature-representation check on the CLEAN 30-window subset

Single-latent **global density** (no factorization, `use_resid=False, use_basin=False`), difficult
subset, 5 seeds, difficulty mask = clean 30-window subset (NOT repr_ablation's own 43-window mask).

| dataset | representation | dim | difficult AUROC | temporal − stats |
|---|---|---|---|---|
| **WADI_clean (30)** | six statistics | 732 | **0.634 ± 0.011** | |
| **WADI_clean (30)** | ten temporal/spectral | 1220 | **0.626 ± 0.017** | **−0.009** |
| HAI (167) | statistics / temporal | | 0.743 / 0.814 | +0.072 (unchanged, I2) |
| SWaT_canon (85) | statistics / temporal | | 0.750 / 0.776 | +0.026 (unchanged, I2) |

**Reading:** on the clean WADI subset the richer temporal/spectral representation does **not** help
(temporal − stats = −0.009 ≤ +0.01), matching HAI/SWaT in direction. The representation is a
secondary lever; the latent-density mechanism carries the result. Conclusion supporting §6 is
unchanged and WADI is now measured (previously A2 WADI was "not re-run").

---

## 2. WADI seed SDs for Tables 3 & 4

5-seed mean ± SD, WADI_clean, subsets All 56 / Easy 26 / Difficult 30 / DoubleHard 19. Point AUROCs
reproduce Table 3 exactly (invariant check passes). Single-run methods (LinRes, trivial, USAD,
TranAD, GDN) carry no SD, matching the table convention.

| method | All | Easy | Difficult | DoubleHard |
|---|---|---|---|---|
| IsolationForest | 0.725 ± 0.005 | 0.829 ± 0.007 | 0.634 ± 0.006 | 0.530 ± 0.009 |
| AutoEncoder | 0.792 ± 0.002 | 0.981 ± 0.001 | 0.628 ± 0.004 | 0.535 ± 0.004 |
| LatAD-global (density) | 0.717 ± 0.006 | 0.813 ± 0.016 | 0.634 ± 0.011 | 0.571 ± 0.013 |
| **LatAD regime-community (HCcoh+LatAD)** | 0.827 ± 0.009 | 0.892 ± 0.009 | **0.771 ± 0.023** | 0.662 ± 0.040 |

The WADI block can now carry SDs like HAI/SWaT. Headline WADI difficult **0.771 ± 0.023**; WADI
DoubleHard headline **0.662 ± 0.040**.

---

## 3. WADI significance — HCcoh+LatAD vs EACH learned detector (difficult subset)

Paired episode-block bootstrap (moving-block normals + episode bootstrap of the 8 WADI attack
episodes with a difficult window, 2000 reps, 95% CI, one-sided P). HCcoh+LatAD difficult AUROC 0.771.

| competitor | competitor AUROC | diff | 95% CI | P(diff ≤ 0) | significant (α=0.05) |
|---|---|---|---|---|---|
| AutoEncoder | 0.628 | +0.142 | [0.059, 0.223] | 0.0010 | yes |
| USAD | 0.579 | +0.192 | [0.066, 0.338] | 0.0000 | yes |
| TranAD | 0.613 | +0.157 | [0.054, 0.276] | 0.0005 | yes |
| IsolationForest | 0.634 | +0.136 | [0.049, 0.237] | 0.0000 | yes |
| GDN | 0.660 | +0.111 | [0.045, 0.190] | 0.0000 | yes |

**VERDICT: YES — "significantly ahead of every learned detector on WADI" IS defensible.** All five
learned detectors are beaten with P < 0.05 on the clean 30-window / 8-episode subset.

**Why this differs from the earlier "expected NO":** the non-significant WADI comparison is vs
**LinRes** (the linear one-hot residual baseline, 0.750), which the machinery reproduces exactly
(diff +0.020, CI [−0.162, 0.197], P = 0.46 — invariant validated). LinRes is **not a learned
detector**; it is the only close WADI competitor, and LatAD ties it. Every *learned* detector
(AE/USAD/TranAD/IF/GDN) sits at 0.58–0.66, far below 0.771, so the gap is significant. The
distinction "ties the linear baseline, significantly ahead of every learned detector" holds.

**Validation performed (per verify-before-report):** (a) invariant — vs-LinRes reproduces the paper's
non-significant result to the decimal; (b) regime point AUROC on the difficult subset = 0.7706 (matches
the 0.771 headline); (c) GDN window-averaged difficult AUROC = 0.660 (matches Table 3); (d) 8 hard
episodes confirmed. The clean, all-significant learned-detector result is not a bug.

---

## 4. Figure 3 (poc/paper/IoT2.html) — regenerated

- **Dual-scale bug fixed.** Every bar and whisker regenerated with the single axis-consistent
  transform `y = 250 − (v − 0.3) × 271.4286` (0.3→y=250, 1.0→y=60). Previously some bars used a
  1.0→y=64 scale (span 265.71), so LatAD read 0.761/0.833/0.826 against the axis.
- **GDN bar added** to each of the three dataset groups (difficult WADI 0.660 / HAI 0.481 / SWaT 0.665,
  matching Table 3), colour `#9578b8`, inserted after TranAD before LatAD; legend gains a GDN entry.
  7 bars/group at width 16, step 18; groups start x=100/310/520, max x=644 (< axis end 700), no overflow.
- **WADI whiskers added** (previously WADI had none): IF ±0.006, AE ±0.004, LatAD ±0.023 (5-seed SDs
  from §2). USAD/TranAD/GDN are single-run → no whisker, matching convention. HAI/SWaT whiskers
  re-emitted on the corrected scale.

**Decode verification (re-parsed from the edited HTML, all 21 bars):**

| dataset | IF | AE | LinRes | USAD | TranAD | GDN | LatAD |
|---|---|---|---|---|---|---|---|
| WADI | 0.634 | 0.628 | 0.750 | 0.579 | 0.613 | 0.660 | **0.771** |
| HAI | 0.627 | 0.757 | 0.586 | 0.477 | 0.444 | 0.481 | **0.845** |
| SWaT | 0.627 | 0.729 | 0.782 | 0.658 | 0.655 | 0.665 | **0.837** |

Every decoded value matches Table 3; **LatAD now reads 0.771 / 0.845 / 0.837** exactly. No other
figure or table was touched.

---

## Numbers that change / refresh a paper claim

1. **Task 3 (potential upgrade):** "significantly ahead of every learned detector on WADI" is now
   supported (all P < 0.05), stronger than the current wording "ahead of every learned detector"
   (point estimate). The abstract/caption phrasing "ahead of every learned detector on WADI (0.771),
   tying the linear baseline" remains correct and can optionally be strengthened to "significantly
   ahead of every learned detector." The vs-LinRes tie (P = 0.46) is unchanged.
2. **Task 1 (refresh, no direction change):** WADI Table A2 is now measured on the clean 30-window
   subset — stats 0.634, temporal 0.626, temporal − stats −0.009 (representation does not help),
   filling the "WADI A2 not re-run" gap noted in `clean_recompute.md`.
3. **Task 2 (additive):** WADI difficult headline now carries SD **0.771 ± 0.023** and DoubleHard
   **0.662 ± 0.040**; IF/AE also gain WADI SDs. No point AUROC changed (Table 3 reproduced exactly).
4. **Task 4 (correction):** Figure 3 bars previously mis-scaled (LatAD read 0.761/0.833/0.826); now
   read 0.771/0.845/0.837, consistent with Table 3, plus GDN bars and WADI whiskers.
