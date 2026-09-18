# Consistency audit of `poc/paper/IoT2.html` (report only, no edits made)

Date: 2026-09-17. Checklist: paper-reviewer SC-1..SC-11 (plus SC-12 same-config and SC-15 wins-only scan, run because they are cheap and turned up findings).
Two layers: (A) internal consistency of the paper as it stands; (B) the paper against the corrected session results (clipped WADI without `2B_AIT_002_PV`, canonical SWaT `SWaT_Dataset_Attack_v0`, canonical difficult subset `(y==1)&(maxz<=thr)` vs all normals, headline `LatAD (regime-community)`). Section (C) resolves the HAI 0.849 vs 0.814 discrepancy.

Sources read: `paper/IoT2.html` (13.5k words extracted), `paper/BUILD_REFERENCE.md`, `_diagnostics/ensemble_{head,full,density,final}.json`, `rev4_{stats,stats_HAI,doublehard,ablation,stronger_split,sota_table3,timeaware}.json`, `a3_skab_witness.json`, `multimodality.json`, `cov_excl2.json`, `hai_exclusion_community.{py,json}`, `headline_clean_results.md`, `headline_{WADI_clean,SWaT_canon}.json`, `headline_HCcoh_{WADI_clean,SWaT_canon}.json`, `paper_framing_update_DRAFT.md`, `../review_round_1/RESEARCH_REGISTRY.md`, git history of `paper/paper.html` / `IoT2.html`, and a direct re-run of `ensemble_final.ensemble_scores("HAI")` under both expert libraries.

---

## 0. Headline verdicts (read this first)

1. **HAI 0.849 is CORRECT for the paper's headline model and is NOT stale.** It reproduces today, to four decimals (0.8491 ± 0.018, All 0.9490), from `EXPERTS_DIR=sota_bundle/experts_full` (full per-community heads). The session's "0.814" comes from `sota_bundle/experts/` (density-only experts), which is the paper's own Table A1 row "regime-community, density-only experts = 0.814". Root cause: `ensemble_final.py` defaults `EXPERTS_DIR` to the density-only library; today's `hai_exclusion_community.py` and the framing draft ran with that default. Full analysis in §C.
2. **Corollary that matters for the corrected basis:** the new clean-data experts (`expert_WADI_clean.npz`, `expert_SWaT_canon.npz`) live in `sota_bundle/experts/` and were built with `use_full=0` (the script writes to `experts_full/` only when `full=1`). So the corrected WADI 0.816 / SWaT 0.827 numbers are the **density-only-expert configuration**, i.e. the Table A1 row-2 configuration, not the paper's headline row-3 configuration. On the old data that distinction is negligible on WADI (0.799 vs 0.796) and SWaT (0.968 vs 0.969) because the residual head is gated off on WADI and adds little on SWaT, but it is +0.035 on HAI. Before the edit, the full-head experts must be built on WADI_clean and SWaT_canon (one `modal_experts.py --full` run) so that every Table 3 headline cell is the same configuration (SC-12). Until then the corrected WADI/SWaT numbers and the retained HAI 0.849 are from two different expert configurations.
3. **Internal BLOCKING defect independent of the data corrections:** the paper has two captions labelled "Figure 1." (the IIoT-context figure in §1 and the difficult-subset bar chart in §6) and §6 prose refers to the results chart as "Figure 1". The results figure must become Figure 3 and the §6 reference updated (SC-5 / SC-8b).
4. **Stale against the corrected basis (abstract, §1, §6, §7, §8, Table 3, Table 4, Figure caption):** every WADI number (rode the `2B_AIT_002_PV` rescale artifact) and every SWaT number (non-canonical attack-only mirror), plus every derived claim ("USAD and TranAD fall to 0.30–0.48", "leads the difficult subset of all three ... near-significantly on WADI", "SWaT is a ceiling case", "38 windows from a single attack episode"). HAI numbers survive unchanged.
5. **Untraceable-to-artifact numbers (finding in itself):** the three bootstrap CIs quoted in prose and captions (+0.092 [0.046, 0.160]; +0.089 [0.020, 0.194]; +0.12 [−0.005, 0.257], P ≈ 0.03), the gate ratios 5.24 / 1.17 / 0.85, and the Table 2 single-model numbers exist only in the hand-written `paper/BUILD_REFERENCE.md`; no JSON carries them (`ensemble_head.json` has `diff_ci: null`). They are reproducible via the documented command (`EXPERTS_DIR=sota_bundle/experts_full HEAD="HCcoh+LatAD" BOOT_REPS=2000 python ensemble_final.py`) but that output was never saved.

---

## 1. Number inventory (every numeric result, with location)

Location key: Abs = abstract; §n = section; T = table; Fig = figure caption; Concl = §8. "Backed by" names the artifact that reproduces the value; "notes" flags the corrected-basis status.

### 1.1 Abstract
| # | Value | Claim | Backed by (internal) | Status vs corrected basis |
|---|---|---|---|---|
| A1 | ten MIIM assumptions | A1–A10 | T1 | ok |
| A2 | three real IIoT testbeds | WADI, HAI, SWaT | §5.1 | ok |
| A3 | WADI 0.862 (best All AUROC) | T3 WADI regime-community All | `ensemble_head.json` WADI/HCcoh+LatAD/All 0.862 | **STALE**: rode the WADI artifact; clean 0.857 (HCcoh, density-only experts) |
| A4 | HAI 0.949 | T3 HAI All | `ensemble_head.json` 0.949; reproduced today 0.9490 (experts_full) | ok |
| A5 | SWaT 0.993 | T3 SWaT All | `ensemble_head.json` 0.993 | **STALE**: mirror; canonical 0.936 (HCcoh) |
| A6 | "leads the difficult subset of all three" | T3 Difficult column | 0.796 / 0.849 / 0.969 all top of column | **PARTLY STALE**: clean WADI 0.816 vs linres 0.787 is numerically ahead but P=0.35 (not significant); SWaT canon 0.827 vs 0.782 significant P=0.024; HAI unchanged |
| A7 | HAI 0.849 | T3 HAI Difficult | reproduced today 0.8491 | ok (see §C) |
| A8 | +0.09, 95% CI [0.046, 0.160] | §6 prose, Fig caption | `paper/BUILD_REFERENCE.md` only (no JSON) | traceable to notes, not to a saved artifact |
| A9 | USAD and TranAD 0.30–0.48 | T3 difficult (WADI 0.305/0.325, HAI 0.477/0.444) | `ensemble_head.json` | **STALE**: on clean WADI USAD 0.685, TranAD 0.722; SWaT canon 0.658/0.655; range becomes 0.44–0.72 and the "collapse to near chance" holds on HAI only |

Abstract word count 198 (budget 200–250; MDPI cap 200), 6 sentences, 0 citations.

### 1.2 Introduction (§1)
| Value | Location | Backed | Status |
|---|---|---|---|
| "best overall AUROC on every dataset", "leads the difficult subset of all three, most decisively on HAI ... statistically significant" | §1 contributions (3) | T3 + §6 CI | WADI/SWaT parts stale (see A3, A5, A6) |
| "single trained configuration per dataset; AUROC five-seed mean" | §1 last para | §5.4, T3 caption | ok |

### 1.3 §2.5 and §3 (dataset characterization)
| Value | Location | Backed | Status |
|---|---|---|---|
| WADI 123 ch, SWaT 51 ch, HAI 59 ch | §2.5, §5.1 | loaders | clean WADI = **122** ch (artifact channel dropped) |
| BIC K* ≥ 22; K* = 22, 24, 25 | §2.5, §3 | `multimodality.json` Kstar_BIC 22/24/25 (PCA-20, K grid to 25) | ok internally; note the registry MIIM matrix (§7a) reports K* 64/128/64 on a wider grid: different measurement, not a contradiction, but the paper should say "on a grid up to 25" |
| silhouette 0.06, 0.08, 0.29 | §3 | `multimodality.json` 0.06 / 0.077 / 0.291 | ok |

### 1.4 §4 Method
| Value | Location | Backed | Status |
|---|---|---|---|
| T2: recon 0.490 / 0.689; latent NLL 0.663 / 0.760; joint 0.494 / 0.689 (WADI / HAI, single model) | T2 | `BUILD_REFERENCE.md` names `recon_why.py` seed 0; no JSON found containing these values | untraced to a saved artifact; WADI column stale (artifact data) |
| M = 80 density components; PCA 30; R = 16; noise 0.5σ; δ = 0.15; λ0 = 2.5 | §4.3, §5.2 | code / BUILD_REFERENCE hyperparameters | ok |
| gate ratios 5.24 (WADI off), 1.17 (HAI on), 0.85 (SWaT on) | §4.3 iii | `BUILD_REFERENCE.md` only | untraced to artifact; WADI/SWaT values need re-measuring on clean data |
| ambiguous fraction ρ ≈ 0.05 on all three (λ = 0) | §4.3 iv, §4.3 summary, §5.2 | `a3_skab_witness.json`: 0.013 / 0.023 / 0.000 | **INTERNAL MISMATCH** with T6 (0.01 / 0.02 / 0.00) and §7 text "≤ 0.02" |
| 738-dim feature = 123 × 6 | §4.4 | arithmetic | clean WADI: 732 = 122 × 6 |
| communities of size 3–25; 45 / 28 / 26 communities | §4.4, §5.2 | expert npz S = 45/28/26; registry E3 | clean: **44** (WADI_clean) / 28 / **25** (SWaT_canon) |
| K, latent: WADI (20, 10), HAI (40, 16), SWaT (40, 16) | §5.2 | `ensemble_final.CFG` | ok (same for clean keys) |

### 1.5 §5 Methodology
| Value | Location | Backed | Status |
|---|---|---|---|
| clip ±10σ (WADI) | §5.1 | `eda_real.CLIP` | ok; note the session bug fix added `WADI_clean: 10.0` |
| SWaT: drop first 2% normal, hold out last 20% normal; "attack-only mirror" disclosure | §5.1 | registry §3 | **OBSOLETE** once canonical SWaT is adopted: canonical file has interleaved normals (1498 windows, 233 anomalies = 12.14% attack); the mirror paragraph must be replaced |
| W = 60, stride 30, 5% label rule, 6 stats | §5.2 | code | ok |
| GDN SWaT difficult 0.871 | §5.5, T3 | `rev4_sota_table3.json` | mirror-SWaT only; no canonical GDN run exists |

### 1.6 Table 3 (all cells traced to `ensemble_head.json`, 5-seed, experts_full for the headline row)
WADI (anom 56 = 37 easy + 19 difficult): all 8 rows match `ensemble_head.json` cell-for-cell (AUROC, sd, F1). **Whole block stale** (artifact data; the difficulty split itself was saturated by the artifact: clean split is 13 easy + 43 difficult).
HAI (anom 652 = 485 easy + 167 difficult): all 8 rows match; headline row reproduced today (0.949 / 0.983 / 0.849 ± 0.018). **Current.**
SWaT (anom 182 = 144 easy + 38 difficult): all 9 rows match (GDN from `rev4_sota_table3.json`). **Whole block stale** (mirror; canonical: 233 anomalies = 148 easy + 85 difficult, 23 difficult episodes).

### 1.7 §6 prose
| Value | Location | Backed | Status |
|---|---|---|---|
| global density 0.792 / 0.933 / 0.991 | Overall detection | T3 | WADI/SWaT stale (clean 0.793 / canon 0.927) |
| HAI 0.849, 167 windows, 26 episodes; AE 0.757; USAD 0.477; TranAD 0.444 | Difficult para | T3, `rev4_stats_HAI.json` (26 episodes) | ok |
| +0.092, 95% CI [0.046, 0.160], P ≈ 0 | Difficult para | `BUILD_REFERENCE.md` only | untraced to artifact (see §2, item S-5) |
| WADI: deep 0.30–0.33, LatAD 0.796, IF 0.677, +0.12, CI [−0.005, 0.257], P ≈ 0.03; single-latent 0.690 | Difficult para | T3; CI from BUILD_REFERENCE only | **STALE**: clean WADI LatAD (HCcoh) 0.816, strongest classical is linres 0.787 (not IF 0.681), diff +0.029 CI [−0.096, 0.174] P=0.35; deep 0.685/0.722; global 0.734 |
| SWaT: trivial 0.943, LatAD 0.969, linres 0.959, 38 windows / single episode | Difficult para | T3 | **STALE**: canonical trivial 0.646, LatAD 0.827, linres 0.782, 85 windows / 23 episodes; SWaT is no longer a ceiling case and becomes the significant win (P=0.024) |
| trivial Easy 1.000 / 0.966 / 0.687; Difficult 0.307 / 0.340 / 0.943 | Validity para | T3 | WADI, SWaT stale (clean 1.000 / 0.699; canon 1.000 / 0.646) |
| stronger split HAI 110 windows: ours 0.675, linres 0.622, AE 0.604, IF 0.528; WADI IF 0.710 vs ours 0.697; SWaT trivial 0.929 | Robustness para | `rev4_stronger_split.json` | traced, BUT the "ours" row there is `VaDE-hard+resid` = the **global** model, while T3's headline is regime-community (**SC-12 mixed-config**; see item I-4). WADI/SWaT stale |
| double-hard: LatAD 0.728 / 0.819 / 0.936; HAI +0.089 CI [0.020, 0.194], 19 episodes; WADI 0.728 vs IF 0.599, P ≈ 0.05; SWaT 18 windows / 1 episode; trivial 0.943→0.880; AE WADI 0.336 | Double-hard para, T4 | T4 cells all match `ensemble_head.json` DoubleHard; CI from BUILD_REFERENCE only (note `rev4_doublehard.json` gives the **global**-model HAI CI +0.076 [0.038, 0.119]) | WADI/SWaT stale (clean WADI DH: HCcoh 0.736, cohmax 0.756, linres 0.660, AE 0.671, n=27/10 ep; SWaT canon DH: HCcoh 0.760, linres 0.688, n=59/18 ep, P=0.007) |
| deep SOTA raw: 0.477, 0.444, 0.305, 0.325; "far below our 0.796–0.849"; Easy 0.90–0.97 | Deep SOTA para | T3 | WADI stale |
| Fig caption: 0.849, +0.09, [0.046, 0.160]; 0.796 vs 0.677; 0.30–0.48 | results figure (mislabelled "Figure 1") | as above | stale WADI parts; caption numbering blocking |

### 1.8 §7 Discussion
| Value | Location | Backed | Status |
|---|---|---|---|
| HAI 0.849, CI [0.046, 0.160]; 0.44–0.48; WADI 0.796 P ≈ 0.03; SWaT ceiling | What the results show | as §6 | WADI/SWaT stale |
| WADI probe (one seed): PCA off-subspace 0.43, AE recon 0.43, latent density 0.74; synthetic recon 0.08, density 0.72 | Why reconstruction fails | not located in any `_diagnostics` JSON in this audit | untraced; computed on artifact WADI |
| T5: recon 0.475/0.695/0.965; density 0.694/0.802/0.952; nearest 0.665/0.797/0.922; base 0.685/0.801/0.947; base+resid 0.685/0.820/0.960 | T5 | `rev4_ablation.json` exact | WADI/SWaT columns stale |
| SKAB ρ = 0.58, λ = 1.07, 0.500 → 0.605, λ = 2 → 0.645; T6 WADI 0.01/0.94/0.726/0.717, HAI 0.02/0.91/0.831/0.842, SWaT 0.00/1.00/0.959/0.943; "≤ 0.02"; "+0.10"; "within ±0.02" | Basin validation, T6 | `a3_skab_witness.json` exact (ρ 0.013/0.023/0.000) | ok internally except the ρ ≈ 0.05 mismatch (item I-2); WADI/SWaT single-seed off/on values computed on artifact/mirror data |
| CalexNet 30–70% | Edge para | ref [42] | ok |

### 1.9 §8 Conclusion
"best overall AUROC on every dataset", "leads the difficult subset of all three, significantly on HAI (paired 95% CI [0.046, 0.160]) and near-significantly on WADI": WADI clause stale (P=0.35 on clean data); SWaT becomes the second significant dataset.

### 1.10 Appendix A, Table A1 (all cells traced)
| Row | WADI / HAI / SWaT | Source |
|---|---|---|
| global density | 0.690 / 0.811 / 0.960 | `ensemble_head.json` LatAD Difficult |
| density-only experts | 0.799 / 0.814 / 0.968 | `ensemble_density.json` HCcoh+LatAD (this is the configuration the clean-data runs used) |
| full per-community heads | 0.796 / 0.849 / 0.969 | `ensemble_head.json` HCcoh+LatAD |
| sum / max / HC / HCcoh / null-max | 0.747, 0.749, 0.753, 0.796, 0.744 / 0.828, 0.802, 0.828, 0.849, 0.846 / 0.968, 0.964, 0.968, 0.969, 0.964 | `ensemble_head.json` sum+LatAD, max+LatAD, HC+LatAD, HCcoh+LatAD, null+HC |
WADI and SWaT columns stale.

---

## 2. Prioritized inconsistency list

### BLOCKING (internal, fix regardless of data corrections)
| ID | Finding | Paper value | Correct value | Fix |
|---|---|---|---|---|
| B-1 | Duplicate figure number (SC-5/SC-8b). Two "Figure 1." captions: IIoT context (§1, html line 157) and difficult-subset chart (§6, html line 977). §6 prose "(Figure 1)" (line 835) and the Discussion point at the wrong figure. No "Figure 3" exists anywhere. | Figure 1 (twice), Figure 2 | Figure 1 (context), Figure 2 (workflow), **Figure 3** (results chart) | Renumber the results figure and its two in-text references (§6 "earns its keep (Figure 1)" and the caption). |
| B-2 | Corrected-basis configuration mismatch (SC-12). Clean-data headline numbers (WADI 0.816, SWaT 0.827) come from density-only experts (`sota_bundle/experts/`), the HAI 0.849 from full-head experts (`experts_full/`). A single Table 3 would mix two model configurations. | n/a (pre-edit) | Build `expert_WADI_clean.npz` and `expert_SWaT_canon.npz` with `use_full=1` (to `experts_full/`), re-run `EXPERTS_DIR=sota_bundle/experts_full ensemble_final.py` for WADI_clean, SWaT_canon, HAI with BOOT_REPS=2000, and save the JSON with the CIs. | One Modal expert job; then regenerate T3/T4/T5/A1/figure from one artifact. |

### STALE vs corrected basis (numbers correct for the old data, wrong for the corrected data)
| ID | Location | Paper value | Corrected value (density-only experts; see B-2) | Note |
|---|---|---|---|---|
| S-1 | Abstract, §6, §7, T3 | WADI All AUROC 0.862 | 0.857 (HCcoh), 0.864 (cohmax) | rode the artifact; global 0.793 |
| S-2 | Abstract, §6, §7, T3 | SWaT All AUROC 0.993 | 0.936 (HCcoh / cohmax) | mirror → canonical |
| S-3 | Abstract, §1, §6, §7, §8, Fig caption | "leads the difficult subset of all three ... near-significantly on WADI (0.796 vs IF 0.677, +0.12 [−0.005, 0.257], P ≈ 0.03)" | WADI clean difficult: HCcoh 0.816 ± 0.010 (cohmax 0.824, null+HC 0.799) vs linres 0.787 (strongest classical is now linres, not IF), AE 0.739, TranAD 0.722, USAD 0.685, IF 0.681, global 0.734. Bootstrap HCcoh vs linres: +0.029, 95% CI [−0.096, 0.174], P(≤0)=0.35, 11 episodes (`headline_HCcoh_WADI_clean.json`). | Numerically ahead, not significant. The brief's "+0.036 CI [−0.097, +0.188] p=0.31" is not in any saved JSON; the artifact says +0.029 / P=0.35. Use the artifact value. |
| S-4 | Abstract, §6 ("0.30–0.48"), §7 ("0.44–0.48"), Fig caption | USAD/TranAD collapse on WADI to 0.305 / 0.325 | clean WADI USAD 0.685, TranAD 0.722 | The WADI collapse was artifact-induced (registry: removing the channel lifts USAD/TranAD by +0.1 to +0.24). Only the HAI collapse (0.477 / 0.444) survives; SWaT canon 0.658 / 0.655. Rewrite the range as HAI-only or "0.44–0.72". |
| S-5 | §6, §7, §8, T4 caption | SWaT "ceiling case", "38 windows from a single attack episode", trivial 0.943, LatAD 0.969 vs linres 0.959 | canonical SWaT: 85 difficult windows / 23 episodes; trivial 0.646; HCcoh 0.827 ± 0.004 vs linres 0.782, AE 0.729, USAD 0.658, TranAD 0.655, IF 0.627, global 0.804; vs linres +0.045, CI [0.000, 0.097], P=0.0245 (significant); double-hard +0.073, CI [0.013, 0.140], P=0.007 | SWaT flips from "ceiling, uninformative" to the second statistically significant win. Every "SWaT is a ceiling" sentence (§3 end, §6 ×3, §7, T4 caption) must go. |
| S-6 | T3 WADI block header, §6 | WADI anom 56 = 37 easy + 19 difficult; 5 difficult episodes | 56 = 13 easy + 43 difficult; 11 difficult episodes | The artifact saturated max\|u\|; the split changes. |
| S-7 | T3 SWaT block header, T4 caption | SWaT anom 182 = 144 + 38; DH 18 / 1 | 233 = 148 easy + 85 difficult; DH 59 / 18 episodes | |
| S-8 | T4 | WADI DH: 0.728 (ours), IF 0.599, AE 0.336, linres 0.251, USAD 0.258, TranAD 0.272, trivial 0.283, global 0.601; "+0.129 vs IF P ≈ 0.05" (text says 0.728 vs 0.599 P ≈ 0.05); n = 11 / 5 | clean DH (n=27 / 10): HCcoh 0.736, cohmax 0.756, null+HC 0.715, global 0.682, linres 0.660, AE 0.671, TranAD 0.629, USAD 0.595, IF 0.581, trivial 0.612; vs linres +0.076, CI [−0.137, 0.261], P=0.24 | |
| S-9 | T4 | SWaT DH: 0.936, linres 0.914, AE 0.890, IF 0.789, USAD 0.780, TranAD 0.777, trivial 0.880, global 0.921 | canon DH (n=59 / 18): HCcoh 0.760, cohmax 0.765, global 0.728, linres 0.688, AE 0.666, USAD 0.609, TranAD 0.604, IF 0.578, trivial 0.581; P=0.007 vs linres | |
| S-10 | §5.1 SWaT paragraph | "openly accessible mirror ... attack-only ... not directly comparable to canonical-SWaT figures" | Canonical `SWaT_Dataset_Attack_v0` (449,919 rows, 51 ch, 12.14% attack) now used; the 2% warm-up drop / 20% held-out normal split must be re-described for the canonical stream | Replace the paragraph. |
| S-11 | §5.1 WADI paragraph | "A few channels carry physically impossible standardized values ... clipped to ±10σ" | Add: channel `2B_AIT_002_PV` is rescaled between the normal and attack recordings (train mean 9.09, σ 0.16 → test-normal 4503) and is dropped; 122 channels; clipping retained | |
| S-12 | §2.5, §4.4, §5.2 | 123 channels, 738-dim, 45 communities (WADI); 26 communities (SWaT) | 122 ch, 732-dim, 44 communities; SWaT_canon 25 communities | |
| S-13 | T5, T2, §4.3 gate ratios, §7 WADI probe, T6 WADI/SWaT rows, stronger-split WADI/SWaT rows, T A1 WADI/SWaT columns | all computed on artifact WADI / mirror SWaT | need re-running on WADI_clean / SWaT_canon (single seed for T2/T6/probe; 5 seeds for T5/A1) | HAI columns of every table stay. |
| S-14 | §6 "Overall detection": "clearing every baseline ... on all three" | HCcoh All: WADI clean 0.857 vs linres 0.834, AE 0.799; SWaT canon 0.936 vs linres 0.920 | still true on All, but margins are small; fine to keep with new values |
| S-15 | Abstract/§8 significance framing | HAI only significant | HAI (26 ep) and SWaT-canonical (23 ep) both significant; WADI numerical | The abstract "notably on HAI" beat should become "on HAI and SWaT". |
| S-16 | §7 "What the results show", §8 | "on WADI the regime-community factorization turns a prior single-latent tie into a near-significant lead" | clean WADI: global 0.734 → HCcoh 0.816 (+0.082) is still the largest factorization gain, but vs linres it is numerical only (P=0.35) | Reframe as "largest factorization gain; ahead of every learned baseline; not separable from the linear baseline with 11 episodes". |

Not stale: HAI T3 row, HAI T4 row, HAI columns of T5/T6/A1, HAI stronger split, HAI exclusion analysis (registry), SKAB witness (T6), time-aware bootstrap (registry E4).

### INTERNAL MISMATCH (present in the current text, independent of data)
| ID | Finding | Values | Fix |
|---|---|---|---|
| I-1 | SC-15 wins-only scanner: 3 hits, all `&mdash;` in §4.1 / §4.3 / §4.4 headings ("— the representation stage (Figure 2, top)" etc.). | em-dashes | Replace with a colon or parentheses. |
| I-2 | Ambiguous-window fraction stated three ways: §4.3 iv and §4.3 summary and §5.2 say "ρ ≈ 0.05 on all three datasets"; §7 says "≤ 0.02"; T6 says 0.01 / 0.02 / 0.00; artifact `a3_skab_witness.json` says 0.013 / 0.023 / 0.000. | 0.05 vs ≤0.02 | Use "ρ ≤ 0.02 (0.01 / 0.02 / 0.00)" everywhere; both are below δ = 0.15 so the conclusion is unchanged. |
| I-3 | Three quoted CIs and the WADI P-value exist only in `paper/BUILD_REFERENCE.md`; `ensemble_head.json` stores `diff_ci: null`, and the E4 time-aware JSON reports the **global** model (+0.054 [0.012, 0.089]), not the headline. A reviewer cannot regenerate the CIs from any committed JSON. | +0.092 [0.046, 0.160]; +0.089 [0.020, 0.194]; +0.12 [−0.005, 0.257] P ≈ 0.03 | Re-run the documented command and commit the JSON (folds into B-2). |
| I-4 | SC-12: the "stronger difficulty definition" paragraph (§6) reports the **global** model (`VaDE-hard+resid` in `rev4_stronger_split.json`: HAI 0.675, WADI 0.697) while calling it "the detector" right after Table 3, whose headline is the regime-community model (0.849 / 0.796). The paragraph's WADI "tie" (0.697 vs IF 0.710) contradicts the Table 3 WADI lead a reader has just seen, because it is a different model. | global vs regime-community | Either run the stronger split on the headline model or label the paragraph "single-latent LatAD (global density)". |
| I-5 | T4 double-hard HAI CI: prose "+0.089 (95% CI [0.020, 0.194])" is the regime-community vs AE figure (BUILD_REFERENCE); `rev4_doublehard.json` holds a different, global-model bootstrap (+0.076 [0.038, 0.119]). Not a contradiction, but the two artifacts are easy to confuse; the JSON that backs the printed number does not exist. | | Folds into I-3 / B-2. |
| I-6 | §3 states K* = 22, 24, 25 "keep improving past two dozen components"; the BIC curve in `multimodality.json` was evaluated only up to K = 25, so K* = 25 on SWaT is the grid edge, and the registry MIIM matrix (wider grid) finds 64 / 128 / 64. | 22/24/25 vs 64/128/64 | State the grid ("on a grid up to 25") or quote the wider-grid values; the A2 claim is strengthened by the registry numbers. |
| I-7 | Table 5 caption says the last row "is by definition the single-latent LatAD (global density) configuration and reproduces its Table 3 row within seed variation": HAI 0.820 ± 0.010 vs T3 0.811 ± 0.016 is within variation; fine. WADI 0.685 vs 0.690 fine. No action; recorded because a reader will check it. | | none |
| I-8 | SC-2 seed / n statements are consistent (five seeds in §1, §5.4, T3, T4, T5, Fig captions; single seed disclosed for T2, T6, the WADI probe; GDN single-run). HAI 167 / 26, 110 stronger, 84 / 19 DH all match artifacts. WADI/SWaT counts are internally consistent but stale (S-6, S-7). | | none internally |

### MINOR
| ID | Finding | Fix |
|---|---|---|
| M-1 | Citation numbering is not in order of first appearance (MDPI requirement). First-appearance sequence of reference numbers: 1, 9, 5, 2, 3, 4, 6, 7, 8, 46, 10, ..., 25, 43, 45, 44, 26, ..., 37, 41, 38, 40, 39, 47, 42. All 47 references are cited at least once and all anchors resolve (no rot). | Renumber the list by first citation, or accept as-is if the journal template tolerates it. |
| M-2 | SC-11 abstract arc: 198 words (ok); beats present: problem (s1–2), mechanism/lens (s3), solution (s4–5), key finding + cross-dataset (s6). Missing beat 7 (impact / code pointer; the Zenodo DOI is only in the Data Availability statement). Sentence 6 lists per-dataset numbers ("WADI 0.862, HAI 0.949, SWaT 0.993"), an anti-pattern per the checklist, and stacks two qualifiers on the HAI claim ("+0.09, 95% CI [...]"). | When the abstract is rewritten for the corrected numbers, keep one headline number per claim and end with the Zenodo pointer. |
| M-3 | §4.3 iii says the residual "fires only where it provably generalizes"; BUILD_REFERENCE's pending list already asked for "demonstrably". | Word swap. |
| M-4 | §2.4 says GDN was re-run "where tractable (SWaT)"; no canonical-SWaT GDN exists, so after the SWaT correction the GDN row either needs a canonical run or must be dropped. | Decide at edit time. |
| M-5 | `paper_framing_update_DRAFT.md` §1 table lists HAI baselines as linres 0.586, AE 0.447, TranAD 0.257, USAD 0.328, IF 0.439. Only linres matches the paper/artifacts; AE 0.757, TranAD 0.444, USAD 0.477, IF 0.627 are the correct current-pipeline values (`cov_excl2.json` HAI: 0.7575 / 0.4450 / 0.4967 / 0.6300; `ensemble_head.json`). The draft's HAI row must not be copied into the paper. | Correct the draft before it is used. |

### SC checklist summary
| Check | Result |
|---|---|
| SC-1 phantom sections | pass: every §n / §n.m cited (1, 2, 3, 4, 4.1–4.3, 4.3, 4.4, 5, 5.1, 5.3, 5.5, 5–6, 6, 7, Appendix A) exists |
| SC-2 sample sizes | pass internally (I-8); WADI/SWaT counts stale (S-6, S-7) |
| SC-3 abstract parity | every abstract number is in T3 or §6; CI backed by notes only (I-3) |
| SC-4 Concl/Disc/Limits | consistent with each other; jointly stale on WADI/SWaT |
| SC-5 cross-ref rot | **fail**: duplicate Figure 1 (B-1); tables 1–6, A1 all referenced and unique; all 47 reference anchors resolve |
| SC-6 forward-reference graph | pass: §1 promises §3, §4, §4.4, §5–6, §7; all delivered |
| SC-7 table-vs-text | pass: every prose number appears in a table/caption with the same value (checked T3, T4, T5, T6, A1 cell-for-cell against artifacts) |
| SC-8 caption-figure | results figure caption numbers match T3; figure mislabelled (B-1) |
| SC-9 artifact reality | headline reproduces from `experts_full` today (0.8491); clean-data experts are a different configuration (B-2) |
| SC-10 subset-n | pass: 167/26, 110, 84/19, 11/5, 18/1, 38/1 all stated inline |
| SC-11 abstract arc | 198 words, lead in sentence 6 (late), per-dataset list, missing impact beat (M-2) |
| SC-12 same-config | **fail** on the stronger-split paragraph (I-4) and on the corrected basis (B-2) |
| SC-15 wins-only scan | 3 em-dash hits in headings (I-1) |

---

## 3. HAI 0.849 investigation (conclusion)

**Question.** Abstract, T3, §6, §7, §8, Fig caption and T A1 all say HAI difficult = 0.849 for the headline model; the session pipeline returned 0.814.

**Trace.**
- `git log -S"0.849" -- paper/paper.html` → entered at 8106a44 (2026-08-09, "finalize MDPI IoT submission ... headline HCcoh+LatAD reproduces 0.862/0.949/0.993 exactly from experts_full"). Same commit added `ensemble_final.py`, replaced `sota_bundle/experts/*.npz` and `modal_experts.py` (which gained the `use_full` switch writing to `experts_full/`).
- `_diagnostics/ensemble_head.json` (Aug 9 13:23): HAI `HCcoh+LatAD` Difficult 0.849 ± 0.018, All 0.949 → the exact Table 3 row. `BUILD_REFERENCE.md` records the regenerate command with `EXPERTS_DIR=sota_bundle/experts_full`.
- `_diagnostics/ensemble_density.json` (density-only experts): HAI `HCcoh+LatAD` Difficult 0.814 → the exact Table A1 "density-only experts" row.
- `_diagnostics/hai_exclusion_community.json` (today 14:26): `HCcoh+LatAD` auroc_difficult 0.8137 (full test set) → 0.8509 (uncovered regime excluded). The script calls `ensemble_scores("HAI")` with no `EXPERTS_DIR`, so it read `sota_bundle/experts/expert_HAI.npz` = density-only.
- Direct re-run today (this audit), same `scores_HAI.npz`, same 167-window canonical difficult subset, 5 seeds:
  - `EXPERTS_DIR=sota_bundle/experts` (density-only): null+HC 0.8134, HCcoh+LatAD **0.8137 ± 0.010**, cohmax+LatAD 0.8064, global 0.8115.
  - `EXPERTS_DIR=sota_bundle/experts_full` (full per-community heads): null+HC 0.8461, HCcoh+LatAD **0.8491 ± 0.018** (All 0.9490), cohmax+LatAD 0.8283, global 0.8115.

**Root cause.** Not a different K/latent (both use (40, 16)), not a different subset (both 167 windows, canonical mask), not a different seed set (5 seeds each), not a stale number. It is a **different model variant**: the paper's headline uses per-community experts with the full scoring stack (density + nearest + auto-gated whitened residual + basin), stored in `sota_bundle/experts_full/`; the session used the density-only experts in `sota_bundle/experts/`, which `ensemble_final.py` selects by default. On HAI the per-community residual head is what adds +0.035 (the gate is on for HAI, cf. §4.3 iii), which is why the two libraries differ there and are near-identical on WADI (0.799 vs 0.796) and SWaT (0.968 vs 0.969).

**Which number is correct on the current corrected pipeline.** HAI is unaffected by the WADI artifact and the SWaT mirror, so the headline HAI difficult AUROC remains **0.849 ± 0.018** (All 0.949), reproduced today from `experts_full`. The 0.814 figure is the density-only ablation and belongs only in Table A1 row 2. The HAI exclusion numbers in the registry / draft (0.814 → 0.851, TPR@5% 0.888) are also density-only; on the headline model the exclusion effect is 0.849 → (to be re-run with `EXPERTS_DIR=sota_bundle/experts_full`; the global-model exclusion is 0.8115 → 0.8498, `hai_exclusion_community.json`).

**Action items.** (1) Set `EXPERTS_DIR=sota_bundle/experts_full` explicitly (or make it the default) in every headline script; (2) build full-head experts for WADI_clean and SWaT_canon (B-2); (3) re-run `hai_exclusion_community.py` under `experts_full` before quoting an exclusion number for the headline model.

---

## 4. Mapping table: current paper number → corrected number (for the eventual edit)

Corrected values are 5-seed, clipped clean WADI (`WADI_clean`, 575 windows, 56 anomalies), canonical SWaT (`SWaT_canon`, 1498 windows, 233 anomalies), canonical difficult subset. **Caveat B-2:** the WADI_clean / SWaT_canon regime-community values below come from density-only experts; the HAI values from full-head experts. Rebuild the clean experts with full heads before the final edit and refresh the WADI/SWaT regime-community cells from that run (the baselines and global-LatAD cells will not change).

### 4.1 Headline and abstract
| Item | Current | Corrected | Source |
|---|---|---|---|
| WADI All AUROC (headline) | 0.862 | 0.857 (HCcoh) / 0.864 (cohmax) | `headline_HCcoh_WADI_clean.json` |
| HAI All AUROC | 0.949 | 0.949 | reproduced |
| SWaT All AUROC | 0.993 | 0.936 | `headline_HCcoh_SWaT_canon.json` |
| WADI difficult (headline) | 0.796 ± 0.041 | 0.816 ± 0.010 (HCcoh); 0.824 ± 0.007 (cohmax); 0.799 ± 0.013 (null+HC) | same |
| HAI difficult | 0.849 ± 0.018 | 0.849 ± 0.018 | reproduced |
| SWaT difficult | 0.969 ± 0.006 | 0.827 ± 0.004 (HCcoh); 0.828 (cohmax); 0.803 (null+HC) | same |
| HAI CI vs AE | +0.09 [0.046, 0.160] | unchanged (re-save the JSON) | BUILD_REFERENCE |
| WADI significance | vs IF +0.12 [−0.005, 0.257], P ≈ 0.03 (near-significant) | vs linres +0.029 [−0.096, 0.174], P = 0.35, 11 episodes (not significant) | `headline_HCcoh_WADI_clean.json` |
| SWaT significance | single episode, none claimed | vs linres +0.045 [0.000, 0.097], P = 0.0245; DH +0.073 [0.013, 0.140], P = 0.007; 23 / 18 episodes | `headline_HCcoh_SWaT_canon.json` |
| USAD / TranAD difficult range | 0.30–0.48 | HAI 0.477 / 0.444; WADI 0.685 / 0.722; SWaT 0.658 / 0.655 | |
| "leads difficult subset of all three" | all three, significant on HAI | HAI and SWaT significant; WADI numerically ahead of linres, beats all learned baselines | |

### 4.2 Table 3, WADI block (corrected; AUROC ± sd / F1)
| Method | All | Easy | Difficult |
|---|---|---|---|
| header | anom 56 = 13 easy + 43 difficult | | |
| trivial max\|u\| | 0.769 / 0.438 | 1.000 / 0.963 | 0.699 / 0.260 |
| Isolation Forest | 0.725 ± 0.005 / 0.388 | 0.871 ± 0.007 / 0.379 | 0.681 ± 0.006 / 0.282 |
| AutoEncoder | 0.799 ± 0.002 / 0.613 | 0.999 / 0.889 | 0.739 ± 0.003 / 0.458 |
| LinRes (one-hot) | 0.834 / 0.680 | 0.993 / 0.880 | 0.787 / 0.595 |
| USAD | 0.757 / 0.488 | 0.994 / 0.889 | 0.685 / 0.337 |
| TranAD | 0.786 / 0.542 | 0.998 / 0.889 | 0.722 / 0.390 |
| LatAD (global density) | 0.793 ± 0.006 / 0.573 | 0.991 ± 0.003 / 0.806 | 0.734 ± 0.008 / 0.440 |
| LatAD (regime-community, HCcoh) | 0.857 ± 0.007 / 0.667 | 0.994 ± 0.001 / 0.794 | 0.816 ± 0.010 / 0.563 |
(current WADI rows: 0.558/0.726/0.743/0.595/0.700/0.717/0.792/0.862 All; 0.307/0.677/0.425/0.392/0.305/0.325/0.690/0.796 Difficult)

### 4.3 Table 3, SWaT block (corrected)
| Method | All | Easy | Difficult |
|---|---|---|---|
| header | anom 233 = 148 easy + 85 difficult | | |
| trivial max\|u\| | 0.871 / 0.771 | 1.000 / 0.969 | 0.646 / 0.206 |
| Isolation Forest | 0.821 / 0.680 | 0.933 / 0.874 | 0.627 ± 0.007 / 0.259 |
| AutoEncoder | 0.899 / 0.800 | 0.996 / 0.980 | 0.729 ± 0.007 / 0.349 |
| LinRes (one-hot) | 0.920 / 0.821 | 1.000 / 0.976 | 0.782 / 0.391 |
| USAD | 0.873 / 0.796 | 0.996 / 0.980 | 0.658 / 0.270 |
| TranAD | 0.873 / 0.796 | 0.998 / 0.980 | 0.655 / 0.266 |
| GDN | (no canonical run) | | |
| LatAD (global density) | 0.927 / 0.797 | 0.998 / 0.952 | 0.804 ± 0.007 / 0.437 |
| LatAD (regime-community, HCcoh) | 0.936 / 0.832 | 0.998 / 0.958 | 0.827 ± 0.004 / 0.507 |
(current SWaT rows: 0.988/0.959/0.987/0.991/0.972/0.972/0.973/0.991/0.993 All; 0.943/0.853/0.939/0.959/0.867/0.867/0.871/0.960/0.969 Difficult)

### 4.4 Table 3, HAI block: unchanged (all 8 rows verified against `ensemble_head.json`; headline reproduced today).

### 4.5 Table 4 (double-hard)
| Method | WADI current → corrected (n 11/5 → 27/10) | HAI (unchanged, 84/19) | SWaT current → corrected (n 18/1 → 59/18) |
|---|---|---|---|
| trivial | 0.283 → 0.612 | 0.349 | 0.880 → 0.581 |
| Isolation Forest | 0.599 → 0.581 | 0.635 | 0.789 → 0.578 |
| AutoEncoder | 0.336 → 0.671 | 0.730 | 0.890 → 0.666 |
| LinRes | 0.251 → 0.660 | 0.465 | 0.914 → 0.688 |
| USAD | 0.258 → 0.595 | 0.442 | 0.780 → 0.609 |
| TranAD | 0.272 → 0.629 | 0.418 | 0.777 → 0.604 |
| LatAD (global) | 0.601 → 0.682 | 0.806 | 0.921 → 0.728 |
| LatAD (regime-community) | 0.728 → 0.736 (HCcoh) / 0.756 (cohmax) | 0.819 | 0.936 → 0.760 |
| significance | vs IF P ≈ 0.05 → vs linres +0.076 [−0.137, 0.261], P = 0.24 | +0.089 [0.020, 0.194] | none → vs linres +0.073 [0.013, 0.140], P = 0.007 |

### 4.6 Other cells
| Item | Current | Corrected |
|---|---|---|
| Table A1 WADI column (global / density-only / full) | 0.690 / 0.799 / 0.796 | 0.734 / 0.816 / (full-head: to run) |
| Table A1 SWaT column | 0.960 / 0.968 / 0.969 | 0.804 / 0.827 / (to run) |
| Table A1 WADI aggregation rows (sum / max / HC / HCcoh / null-max) | 0.747 / 0.749 / 0.753 / 0.796 / 0.744 | 0.788 / 0.807 / 0.807 / 0.816 / 0.799 (`headline_WADI_clean.json`: sum+LatAD, max+LatAD, HC+LatAD, HCcoh+LatAD, null+HC) |
| Table A1 SWaT aggregation rows | 0.968 / 0.964 / 0.968 / 0.969 / 0.964 | from `headline_SWaT_canon.json` (sum/max/HC+LatAD not printed in this audit; HCcoh 0.827, null+HC 0.803) |
| WADI channels / feature dim / communities | 123 / 738 / 45 | 122 / 732 / 44 |
| SWaT communities | 26 | 25 |
| SWaT description (§5.1) | attack-only mirror | canonical iTrust Dec-2015 attack log, 12.14% attack |
| WADI description (§5.1) | clip ±10σ | clip ±10σ and drop `2B_AIT_002_PV` (rescaled between recordings) |
| T2, T5, T6, gate ratios, §7 WADI probe, stronger split | artifact/mirror data | re-run on WADI_clean / SWaT_canon (HAI cells unchanged) |
| ρ statements | "≈ 0.05" (§4.3, §5.2) | "≤ 0.02 (0.01 / 0.02 / 0.00)" per T6 and `a3_skab_witness.json` |
| Results figure | "Figure 1" | "Figure 3", bars from the corrected T3 difficult column |
| A3 / SKAB (T6, §7) | 0.500 → 0.605, ρ = 0.58, λ = 1.07; no-op on WADI/HAI/SWaT | unchanged (verified against `a3_skab_witness.json`) |
| HAI exclusion (if promoted, registry §7b) | not in paper | global 0.934 → 0.950 AUROC, 0.814 → 0.841 difficult (`cov_excl2.json`); headline model needs an `experts_full` re-run before quoting |

---

## 5. Things this audit could not trace (findings in their own right)
1. The prose/caption CIs (+0.092 [0.046, 0.160]; +0.089 [0.020, 0.194]; +0.12 [−0.005, 0.257], P ≈ 0.03) and P ≈ 0 for HAI: only in `paper/BUILD_REFERENCE.md`; no JSON.
2. Table 2 single-model values (0.490 / 0.689 / 0.663 / 0.760 / 0.494 / 0.689): `BUILD_REFERENCE.md` attributes them to `recon_why.py` seed 0; no saved output found.
3. Gate ratios 5.24 / 1.17 / 0.85: `BUILD_REFERENCE.md` "from code"; no saved output found.
4. §7 WADI mechanism probe (0.43 / 0.43 / 0.74; 0.08 / 0.72): no artifact located under `_diagnostics`.
5. The brief's WADI bootstrap "+0.036 CI [−0.097, +0.188] p = 0.31": not in any saved JSON; the saved HCcoh run says +0.029 [−0.096, 0.174], P = 0.35 (`headline_HCcoh_WADI_clean.json`) and the null+HC run +0.013 [−0.114, 0.155], P = 0.45 (`headline_WADI_clean.json`). Both are non-significant; use the saved value.
6. `paper_framing_update_DRAFT.md` HAI baseline row (AE 0.447, TranAD 0.257, USAD 0.328, IF 0.439) does not match any artifact; the correct current values are AE 0.757, TranAD 0.444, USAD 0.477, IF 0.627.
