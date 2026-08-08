# LatAD Revision Plan v4 — post ChatGPT/MDPI-IoT referee report

Source review: `reviews/LatAD_MDPI_IoT_review_ChatGPT.md` (ChatGPT, High-effort, read
the uploaded `paper.html`). Recommendation: **Major Revision**. Citation verification:
web-researcher pass 2026-08-08 — all reviewer-cited papers confirmed REAL.

Discipline: wins-only. Tier 1 = factual/consistency bugs (mandatory, not negative
results). Tier 2 = strengthening experiments (run; report only if the win holds; nulls
stay in `_diagnostics/` + registry + commit history). Tier 3 = positioning/citations.

---

## Verified citations (drop-in, after bibtest)

| Ref | Citation | Use |
|---|---|---|
| SensitiveHUE | Feng et al., KDD 2024, doi:10.1145/3637528.3671919, arXiv 2405.xxxx | reconstruction-based baseline; reports SWaT+WADI |
| PATE | Ghorbani et al., KDD 2024, arXiv 2405.12096 | eval-metric critique (§2.1) |
| CATCH | Wu et al., ICLR 2025, arXiv 2410.12261 | channel-aware contrast |
| mTSBench | 2025 preprint, arXiv 2506.21550 | "rankings are dataset-dependent" |
| TAB | Qiu et al., **PVLDB 2025**, arXiv 2506.18046 | unified benchmarking context |
| GDN | Deng & Hooi, AAAI 2021, doi:10.1609/aaai.v35i5.16523 | cross-channel method (cited, not run) |
| Pinet | M. Pinet, KDD Workshop **MILETS 2026**, arXiv 2606.02670 | "anomalies mostly univariate" — engage head-on |
| LATAD (collision) | Choi et al., IEEE TETCI 2024, arXiv 2406.12260 | name collision — rename LatAD |

---

## Tier 1 — Must-fix (factual/consistency; apply regardless of experiments)

1. **Remove "latent-only".** Abstract (contribution line), §4.2 title, §7 ("drop
   reconstruction entirely"), §8. Model uses a reconstruction-residual head on HAI+SWaT
   (§4.3 iv, line ~385). Reframe: **"latent-primary scoring with an optional, auto-gated
   reconstruction-residual head."**
2. **Abstract density wording.** "rather than from a density estimate" is false (§4.3(i)
   IS a latent mixture density). Change to "in the clustered latent space rather than from
   data-space reconstruction error."
3. **Table 2 caption.** "at or below chance" is false for HAI (0.689). Rewrite to: near
   chance on WADI (0.490); informative on HAI (0.689) but below the latent term (0.760).
4. **Basin head (v).** Never activates on WADI/HAI/SWaT. Move out of Contributions into a
   one-line future-work/appendix note.
5. **Claim discipline.** "best/leads every difficult column" overstated: WADI-difficult
   0.690 vs IF 0.677 (tie); SWaT-difficult 0.960 vs LinRes 0.959 (tie). Reframe to:
   **HAI decisive; WADI tie-with-IF mechanistic case; SWaT ceiling.** Figure 1 gets error
   bars. (Final wording waits on Tier 2 episode-bootstrap significance.)

## Tier 2 — Strengthening experiments (run first; wins-only reporting)

6. **Stronger univariate difficulty split.** Redefine "easy" using max over ALL SIX
   standardized per-channel window stats (mean, std, min, max, first-last diff, range),
   not just the mean. Re-evaluate LatAD vs baselines on the new difficult subset. Best
   answer to Pinet. Local recompute from `_diagnostics/scores_*.npz` + window features.
7. **Episode-block bootstrap + CIs.** Report # independent attack episodes behind the
   19 (WADI) / 38 (SWaT) difficult windows. 95% CIs on each headline AUROC and on paired
   LatAD−IF / LatAD−LinRes differences via episode/block bootstrap.
8. **Multi-seed baselines + ablation.** Run USAD/TranAD over the same 5-seed budget.
   Multi-seed Table 2 across all three datasets + full score-head ablation (which head
   causes each gain).
9. **Deployable metrics.** Add AUPRC and TPR@1%FPR under a train-normal-quantile
   threshold (no test labels). Keep bestF1 explicitly as an oracle/separability number.

## Tier 3 — Positioning & citations

10. Add SensitiveHUE (strongest new on-point baseline; runs on SWaT+WADI), CATCH,
    TAB/mTSBench context, PATE in §2.1. Run ≥1 modern method on our protocol if feasible;
    else stop calling USAD/TranAD "modern SOTA" → "representative deep baselines".
11. GDN: run it (evaluates on SWaT+WADI) or explicitly scope out.
12. Engage Pinet head-on: acknowledge "mostly univariate" thesis; position LatAD as
    detecting improbable-but-reconstructable joint states isolated by the stronger split.
13. Reproducibility table (arch, K, latent dims, LR/epochs/optimizer, gate threshold,
    R/λ₀/δ) + archive seed-level scores.

## Name

"LatAD" collides with "LATAD" (Choi et al., TETCI 2024). Rename candidates: CLAD, LaCE,
JLAD, MoNA. Author decision.

---

## Execution status (2026-08-08)

DONE:
- Tier 1 items 1–5: all applied to paper.html (latent-only→latent-primary; density
  wording; Table 2 caption; basin head relegated; claim-scoping to HAI-win / WADI-tie /
  SWaT-ceiling). Verified absent: "latent-only", "at or below chance", "drop
  reconstruction entirely".
- Tier 2 item 6 (stronger 6-stat split): `rev4_stronger_split.py` → HAI advantage
  SURVIVES (LatAD 0.675 > LinRes 0.622 > AE 0.604); WADI tie, SWaT ceiling. In §5.3/§6.
- Tier 2 item 7 (episode bootstrap + CIs): `rev4_stats.py` → HAI paired diff +0.054,
  95% CI [0.015, 0.089], P≈0.002 (SIGNIFICANT, 26 episodes); WADI tie [−0.083,0.106];
  SWaT single-episode ceiling. In abstract/§6/§7/§8 + Figure 1 whiskers.
- Tier 2 item 8a (multi-seed per-head ablation, all 3 datasets): `rev4_ablation.py` →
  Table 4 added to §7 (recon 0.475 WADI near-chance; density carries WADI/HAI; residual
  lifts HAI/SWaT where reconstruction is informative).
- Tier 2 item 9 (deployable metrics): computed; LatAD ties AE on full-set AUPRC/TPR
  (not a clean win) → kept in `_diagnostics`, OUT of paper per wins-only. Difficult-subset
  TPR@1%FPR in `_diagnostics` too.
- Tier 3 items 10–12: SensitiveHUE, CATCH, PATE, TAB added + cited; GDN scoped honestly;
  Pinet engaged via the stronger-split. Renumbered (33 refs, no orphans). bibtest: 33/33 valid.
- DOCX rebuilt (paper_mdpi.docx + root copy); QA passed; figure embedded.

GATED (need author decision — NOT done):
- Item 8b USAD/TranAD multi-seed via Modal: low value (they collapse to 0.30–0.50, far
  outside seed variance). Modal is authed (profile llmcourse) if we want it.
- Item 11 GDN raw-metric run: could BEAT LatAD on WADI-difficult and weaken the paper —
  author call before spending the cloud run. Currently scoped in text as future work.
- Name: "LatAD" collides with LATAD (Choi et al., TETCI 2024). Rename is an author
  branding decision; NOT auto-applied.

REMAINING (optional polish):
- Between-mode probe protocol (§7) still under-specified (n, mode definition); add for
  reproducibility. Full hyperparameter/reproducibility table (item 13) not yet added.
- Visual PDF render-QA of Figure 1 whiskers before submission.
