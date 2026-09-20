# Scout: imbalance-aware / "normalized AUC" metrics in TSAD & CPS anomaly detection (2018-2026)

## Answer (summary)

The "sort of normalized AUC" the literature keeps returning to is **VUS-ROC / VUS-PR** (Volume Under the Surface; Paparrizos et al., VLDB 2022) — the current de-facto robust default in time-series AD benchmarks. The classic "normalized AUC" in the statistics/diagnostics sense is **McClish-standardized partial AUC** (pAUC renormalized so 0.5 = chance over a restricted low-FPR band), which maps almost exactly onto our TPR@fixed-FPR direction. For the plain imbalance argument, **AUPRC/Average Precision with the prevalence baseline** (Davis & Goadrich 2006; Saito & Rehmsmeier 2015) is the canonical citation we are already adding.

## Metric-by-metric

**1. Partial AUC (pAUC) and McClish-standardized pAUC**
AUC over a restricted FPR range (e.g. FPR <= 0.1); McClish (1989) rescales so 0.5 = chance, 1 = optimal within that band, comparable to full AUROC. Prevalence-invariant (ROC-based), focused on the low-false-alarm regime operators care about. Adoption: standard in anomalous-sound detection (DCASE task 2 reports pAUC over low FPR), graph AD, and as a training objective (2025). It is essentially the integral version of TPR@FPR.
- McClish standardization: https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.5777 (Ma et al., Stat. in Medicine 2013)
- pAUC as AD objective: https://arxiv.org/html/2502.11570v2/ ; DCASE 2022 Task 2: https://arxiv.org/pdf/2206.05876

**2. VUS-ROC / VUS-PR (Volume Under the Surface)** — RECOMMENDED
Paparrizos, Boniol, Palpanas, Tsay, Elmore, Franklin, PVLDB 15(11):2774-2787, 2022. Integrates the ROC/PR surface over a range of tolerance/buffer lengths around true anomaly ranges, free of both threshold and hand-picked buffer width. VUS-PR inherits AUPRC's imbalance-awareness (prevalence-sensitive PR surface). Now the recommended robust default in TSB-UAD/TSB-AD.
- VLDB PDF: https://www.vldb.org/pvldb/vol15/p2774-paparrizos.pdf ; VLDBJ 2025 ext: https://link.springer.com/article/10.1007/s00778-025-00907-x ; code: https://github.com/thedatumorg/VUS
- Taxonomy endorsing it ("Navigating the Metric Maze"): https://arxiv.org/pdf/2303.01272

**3. AUPRC / Average Precision + prevalence baseline**
No-skill baseline = positive prevalence (unlike AUROC's 0.5) -> the imbalance-aware complement. Canonical "AUROC is optimistic under skew -> use PR" argument.
- Davis & Goadrich, ICML 2006: https://mark.goadrich.com/articles/davisgoadrichcamera2.pdf
- Saito & Rehmsmeier, PLOS ONE 2015: https://www.researchgate.net/publication/273155496

**4. Time-aware precision/recall families**
- Range-based P/R (Tatbul et al., NeurIPS 2018): https://arxiv.org/abs/1803.03639
- Affiliation-based P/R (Huet, Navarro, Rossi; KDD 2022), parameter-free: https://arxiv.org/pdf/2206.13167
- PATE (Ghorbani, Reinders, Tax; KDD 2024): https://arxiv.org/pdf/2405.12096
- eTaPR: enhanced time-aware P/R, niche (HAI ecosystem).

**5. Point-adjustment critiques and their recommendations**
- Kim et al., AAAI 2022: PA-F1 lets random scores hit SOTA; recommend PA%K + non-adjusted metrics. https://cdn.aaai.org/ojs/20680/20680-13-24693-1-2-20220628.pdf
- Garg et al., IEEE TNNLS 2022: independent PA critique; composite/threshold-independent metrics. https://arxiv.org/pdf/2212.03637
- Consensus replacement suite: VUS-ROC/PR, affiliation P/R, PATE, PA%K.

**6. Imbalance-robust scalars (MCC / balanced accuracy) in CPS/ICS**
MCC uses all four confusion cells, argued as the most honest single thresholded scalar under imbalance; appears in ICS/IDS work on SWaT/WADI. A thresholded companion to best-F1, not a replacement for a threshold-free curve.

## RECOMMENDATION

Add **VUS-PR** (Volume Under the PR Surface; Paparrizos et al., PVLDB 15(11):2774-2787, 2022; https://www.vldb.org/pvldb/vol15/p2774-paparrizos.pdf ; code https://github.com/thedatumorg/VUS).

Why VUS-PR specifically:
- The literature's current recommended robust "normalized AUC" for TSAD; naming it preempts both the "AUROC is optimistic under imbalance" objection and the point-adjustment objection at once (threshold-free and range/buffer-free).
- The imbalance-aware member of the VUS pair (PR-surface baseline tracks prevalence), complementing AUROC + AUPRC + TPR@FPR without duplicating them.
- One library call over existing scores; no retraining.

Secondary (ROC-side "normalized AUC" formalizing TPR@FPR): McClish-standardized partial AUROC over a low-FPR band (McClish 1989 via Ma et al. 2013). Pick VUS-PR if adding only one.

## Note on applicability to our difficulty-stratified evaluation
VUS is defined over a CONTIGUOUS time series with anomaly RANGES + tolerance buffers. Our difficulty-stratified evaluation scores a FILTERED window set (normal union difficult-anomaly windows), which is not a contiguous series, so VUS-PR is cleanly defined on the ALL subset (full series), not on the difficult/double-hard subsets. Plan: report AUROC + AUPRC + TPR@FPR on the difficulty subsets (the win-everywhere table), and VUS-PR on the full set as the field-standard robustness metric.

(Report generated by the web-researcher scout; claims from official proceedings PDFs/landing pages linked above.)
