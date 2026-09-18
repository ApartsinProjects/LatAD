# Edit-pass report — `paper/IoT2.html` corrected/construct-matched revision

Date: 2026-09-17. Source of truth: `_diagnostics/consistency_audit.md` + `_diagnostics/content_gap_audit.md`.
All headline numbers are FULL-HEAD, construct-matched, from one artifact (`_diagnostics/edit_pass_fullhead.json`,
`EXPERTS_DIR=sota_bundle/experts_full HEAD=HCcoh+LatAD BOOT_REPS=2000`). No git commit made.

## STEP 1 — full-head numbers locked (resolves B-2 / SC-12)
Ran `EXPERTS_DIR=sota_bundle/experts_full HEAD="HCcoh+LatAD" BOOT_REPS=2000 ensemble_final.py WADI_clean SWaT_canon HAI`
→ `_diagnostics/edit_pass_fullhead.json` (+ `.log`). Confirmed full-head HCcoh+LatAD **difficult** = WADI_clean **0.824**,
SWaT_canon **0.840**, HAI **0.849** (cohmax 0.828/0.837; null+HC 0.806/0.823) — matches the brief exactly.
Bootstrap CIs (2000 reps, episode-block, vs strongest baseline per dataset):
- HAI difficult vs AE: **+0.092 [0.047, 0.157]**, P≈0, 26 ep; HAI double-hard vs AE +0.089 [0.022, 0.186], 19 ep.
- SWaT difficult vs linres: **+0.058 [0.015, 0.107]**, P=0.003, 23 ep; DH +0.087 [0.029, 0.151], P=0.0005, 18 ep.
- WADI difficult vs linres: **+0.037 [-0.088, 0.181]**, P=0.32, 11 ep (numerical); DH +0.083 [-0.134, 0.267], P=0.22, 10 ep.
HAI coverage (full-head, `_diagnostics/hai_exclusion_fullhead.log`): AUROC 0.9490→0.9652, difficult 0.8491→0.8896, TPR@5% 0.789→0.916.

### IMPORTANT decision — full-head supersedes the mapping-table (density-only) figures
STEP 1 is explicit that the mapping tables' WADI 0.816 / SWaT 0.827 are DENSITY-ONLY and must be replaced by FULL-HEAD
for **every** Table 3/4/A1 headline cell. GROUP 2 of the brief still carried the density-only carry-overs
(abstract 0.857/0.936; WADI vs linres +0.029/P=0.35; SWaT +0.045/P=0.0245). To keep the paper construct-matched
(the whole point of B-2/SC-12) I used the FULL-HEAD values consistently and did NOT mix configs:
- abstract/Table 3 All: **WADI 0.864, SWaT 0.941** (not 0.857/0.936).
- WADI difficult vs linres **+0.037/P=0.32** (not +0.029/P=0.35); SWaT **+0.058/P=0.003** (not +0.045/P=0.0245); DH SWaT P=0.0005.
Every one of these is arithmetically tied to the full-head difficult cells (0.824−0.787=0.037; 0.840−0.782=0.058), so the
table and text now agree (SC-7). Baselines and global-LatAD are config-independent and taken from the same run.

## Per-group changelog

### GROUP 1 (blocking / internal)
- **B-1** results figure renumbered **Figure 1 → Figure 3**; both in-text refs fixed (§6 "earns its keep (Figure 3)" and the caption). No duplicate "Figure 1" remains (verified: caption labels = Figure 1 context, Figure 2 workflow, Figure 3 results).
- **B-2 / SC-12** resolved via full-head lock (above); every Table 3/4/A1 headline cell now from one config.
- **I-1** three `&mdash;` in §4.1/§4.3/§4.4 headings → parenthetical "(… stage; …)".
- **I-2** ρ made consistent everywhere: "ρ ≤ 0.02 (0.01 / 0.02 / 0.00 on WADI / HAI / SWaT)" (§4.3 iv, §5.2).
- **I-4** the "stronger difficulty" paragraph now labeled **single-latent LatAD (global density)**; stale WADI-tie / SWaT-ceiling lines removed.
- **M-1** reference renumber to first appearance: **NOT applied** — see Not-completed below. New refs appended as [48]/[49]/[50].
- **M-3** "provably" → "demonstrably" (§4.2).

### GROUP 2 (numbers) — all from `edit_pass_fullhead.json` unless noted
- **Abstract**: full-AUROC WADI 0.864 / HAI 0.949 / SWaT 0.941; difficult framing rewritten (HAI + SWaT significant, WADI ahead of every reconstruction detector; deep collapse HAI-only 0.44–0.48); Zenodo impact beat added. Trimmed to **205 words** (was 235; pre-edit 198).
- **Table 3** WADI block (header 56 = 13 easy + 43 difficult) and SWaT block (header 233 = 148 easy + 85 difficult) fully replaced with full-head rows; HAI block unchanged; bold/best-in-column re-marked (trivial now tops WADI/SWaT Easy at 1.000).
- **Table 4** double-hard: WADI 27/10, SWaT 59/18, HAI 84/19; all rows full-head; significance in caption+prose.
- **Table 5** head decomposition: WADI/SWaT columns replaced from `_diagnostics/rev4_ablation_clean.json` (5-seed); HAI unchanged.
- **Table 6** basin-gate WADI/SWaT rows replaced from `_diagnostics/a3_witness_clean.json`; HAI/SKAB unchanged.
- **Table A1** factorization + aggregation blocks: WADI/SWaT columns full-head; HAI unchanged.
- Subset counts, channels 123→122, dim 738→732, communities 45→44 / 26→25 all applied.

### GROUP 3 (disclosures / claims)
- **§5.1 WADI**: states the `2B_AIT_002_PV` drop with the **27,359σ** measurement (not "a documented issue"); 122 channels; WADI.A1 release named.
- **§5.1 SWaT**: mirror paragraph replaced with canonical `SWaT_Dataset_Attack_v0` (449,919 rows, 12.14% attack, interleaved normals); 2%/20% mirror workaround removed.
- **A8 (G6)**: Table 1 A8 renamed "Mixed signals (heterogeneous scale and noise)"; design cell narrows the typed-channel claim ("no channel is given a categorical likelihood"); §3/§4.4 "typed" dropped; Figure 2 legend reconciled to "[A8] correlated channels / subsystems" (three conflicting A8 tags reconciled to one).
- Stale claims from content_gap A1–A5 rewritten: "leads all three" → HAI+SWaT significant / WADI competitive; USAD/TranAD "0.30–0.48" moved to HAI-only; every "SWaT ceiling / single episode" removed; "deep detectors collapse on WADI" removed.
- **Discussion**: added "Benchmark coverage as a diagnosable property" paragraph (full-head HAI coverage numbers, forward framing).
- **Limitations**: added WADI-cleaning and HAI-coverage boundary conditions.
- **GDN on SWaT**: no canonical run → GDN row DROPPED from Table 3, caption notes it, §2.4/§5.5 reworded ("did not complete on the canonical stream"); no number fabricated.

### GROUP 4 (literature)
- Added **HI-VAE** [48] (Nazábal 2020, 10.1016/j.patcog.2020.107501), **VAEM** [49] (Ma 2020, arXiv 2006.11941), **TABOR** [50] (Lin 2018, 10.1145/3196494.3196546) in the §7 "Alternative realizations" A8 future-work sentence and the reference list. All three verified by bibtest.
- Out-of-sample positioning statement added (combiner frozen before the clean/canonical re-evaluation).
- Density-side related-work currency gap and self-preprint: see Not-completed.

## Corrected finding beyond the numeric mapping (flagged for author review)
On **clean WADI the whitened reconstruction residual is informative** (Table 5 recon **0.813**±0.002; Table 2 single-model **0.816**),
NOT near-chance. The old "0.475/0.490 near-chance on WADI" was itself an artifact of the `2B_AIT_002_PV` channel. Consequences applied:
- §4.2 / Table 2 reframed: on HAI reconstruction (0.689) trails the latent (0.760) and drags the joint down (demotion helps); on WADI the residual separates the difficult subset but **fails the held-out-normal generalization gate** and is switched off — the shipped gain comes from factorizing the density.
- §7 "Why reconstruction fails and density wins": the **dirty-WADI single-model probe (0.43/0.43/0.74; synthetic 0.08/0.72) was removed** (artifact-driven; untraced per audit finding #4). The mechanism is now anchored on HAI's head decomposition and on the reconstruction-based deep detectors (which trail on all three and collapse only on HAI).
- Table 6: force-enabling the basin head no longer "degrades WADI and SWaT" — on canonical SWaT it changes +0.019 (within ±0.02); wording corrected to WADI −0.005, HAI +0.011, SWaT +0.019.
- Gate ratios re-measured on clean/canon: WADI 4.95 (off), HAI 1.23 (on), SWaT 0.85 (on).

## STEP 3 — build & verify
- **DOCX**: `E:\Projects\Backlog\LatAD\poc\paper\paper_mdpi.docx`, rebuilt from the final HTML via
  `html2doc/scripts/mdpi_from_html.py` (3/3 figures embedded, 84 native OMML equations, 195 body blocks transplanted).
- **Change-marking in the DOCX** (coordinator requirement): implemented by a sentinel-inject + python-docx post-pass
  (helper `scripts_docx_revmark.py`, removed after use). Verified in `word/document.xml`:
  **yellow highlight ×218** (prior `rev-highlight`), **green highlight ×699** (`rev2-highlight`, this round),
  **strikethrough ×26 + red color C0392B ×26** (`rev-del`), **0 sentinels remaining**. Mechanism: `rev2-highlight`→
  `WD_COLOR_INDEX.BRIGHT_GREEN`; `rev-del`→`<del>` (pandoc strike) + red font. The pipeline's built-in
  `rev-highlight`→yellow was preserved. Marks confirmed by opening the XML (per coordinator's "or opening the XML").
- **bibtest**: `python -m bibtest --summary check-html paper/IoT2.html` → **50/50 valid, "All references resolve. OK"**.
- **Self-consistency**: every abstract number has a Table 3/4 backing (WADI 0.864, HAI 0.949, SWaT 0.941, HAI 0.849 +0.09 [0.047,0.157], SWaT P=0.003, deep 0.44–0.48). No stale headline (0.862 / SWaT 0.993 / "0.30–0.48" / "attack-only mirror" / 738-dim / "near-significant") remains in prose. Figures 1/2/3 unique.

## git diff --stat
```
 poc/paper/IoT2.html | 687 ++++++++++++++++++++++++-------------------
 1 file changed, 461 insertions(+), 226 deletions(-)
```
Also modified: `poc/paper/paper_mdpi.docx` (rebuilt). New provenance artifacts (untracked): `_diagnostics/edit_pass_fullhead.json`/`.log`,
`_diagnostics/rev4_ablation_clean.json`/`.log`, `_diagnostics/a3_witness_clean.json`/`.log`, `_diagnostics/table2_clean.py`/`.log`.

## NOT completed (flagged, with reason)
1. **M-1 reference renumber to first-appearance order.** The bibtest `renumber_to_first_appearance.py` targets numeric
   `id="rN"` / `href="#rN"` anchors with `<span class="rnum">[N]</span>`. This paper uses **semantic anchors**
   (`id="r-chandola"`, hardcoded link-text numbers), so the script cannot run mechanically, and a hand renumber of 50
   references risks introducing citation errors that would fail bibtest. Left in current order (all 50 anchors resolve).
   New refs appended as 48/49/50. Recommend a dedicated renumber pass.
2. **Density-side related-work currency (content_gap D2/D6):** Birihanu & Lendak 2025, Islam & Carden 2026, Graph-MoE 2025
   NOT added — they are outside the validated set in `discrete_method_citations.md` and would each need a bibtest pass.
   Recommended as a follow-up (would go in §2.3/§2.4).
3. **Self arXiv preprint (D1):** the brief said "if a placeholder exists" — none exists in the HTML, so not inserted
   (belongs in the cover letter per MDPI practice).
4. **Detection-set complementarity (G4)** and **extended difficulty stratification (G5)**: not added — the audits require
   re-verification on the headline (full-head) model before quoting, which was out of scope for this pass.
5. **Abstract = 205 words** (MDPI soft cap 200). Trimmed from 235; kept the significance + Zenodo beats. Marginally over.

---

## POLISH PASS (2026-09-17) — 3 validated citations + first-appearance renumber + DOCX rebuild

Completes the two items flagged "NOT completed" in the previous pass (M-1 renumber; density-side related-work currency).

### TASK 3 — 3 validated citations added (all bibtest-verified)
All three inserted as `<li ... class="rev2-highlight">` (green) in `<ol class="refs">` and each with one non-apologetic positioning sentence (rev2-highlight span, in-text anchor) in Related Work:
- `r-islam` — **final number [17]**. §2.1 (IIoT): "Global autoencoders trained across such heterogeneity develop detection blind spots on multi-product CPS that grade-specific models recover [17], reinforcing our regime-latent factorization over a single plant-wide model."
- `r-birihanu` — **final number [31]**. §2.3 (latent/clustering): "Closest to our aim, Birihanu and Lendák [31] flag ICS anomalies from a latent correlation matrix scored by a multivariate Gaussian, whereas we factorize a learned latent density over correlation communities and combine them by Higher Criticism."
- `r-graphmoe` — **final number [38]**. §2.4 (deep multivariate CPS): "Graph mixture-of-experts routing likewise injects hierarchical graph structure into entity representations for GNN detectors [38], whereas we route per-community density experts by cohesion-weighted Higher Criticism rather than a learned gate."

### TASK 2 — reference renumber to FIRST-APPEARANCE order — DONE (verified clean, NOT reverted)
The named-anchor scheme (`<li id="r-NAME">`, in-text `<a href="#r-NAME">[N]</a>`) was renumbered by parsing the body for the first-distinct-appearance sequence of anchors, reordering the `<li>` entries into that sequence (auto-numbered by position), and rewriting every in-text `[N]` with a collision-safe NAME->token->number two-pass.

Bug caught and fixed during the pass (per verify-before-report): two anchors were line-broken in source (`<a\nhref="#r-tranad">`, `<a\nhref="#r-gdn">`); the initial single-space regex skipped them in both ordering and renumbering. Regex widened to `<a\s+href=`; 68 body anchors (+3 new = 71) then all matched. TranAD/GDN correctly landed at 36/37 (consecutive with USAD 35 in §2.4).

**old->new map** (old = position in the previous list; 53 refs total, 0 uncited):
```
chandola 1->1   pang 9->2    ruff 5->3    iotsurvey 2->4   iottaxon 3->5   iotae 4->6
kim 6->7        garg 7->8    wukeogh 8->9 incip 46->10     tsreview 10->11 icssurvey 11->12
tsadreview 12->13 edgeint 13->14 iotlora 14->15 iotedgeai 15->16 islam NEW->17 doshi 16->18
pate 17->19     univar 18->20 tab 19->21   bouman 20->22   memae 21->23    sarfraz 22->24
timesead 23->25 vade 24->26  dagmm 25->27  deepsvdd 43->28 thoc 45->29     omni 44->30
birihanu NEW->31 lw 26->32   if 27->33     aead 28->34     usad 29->35     tranad 30->36
gdn 31->37      graphmoe NEW->38 sensitivehue 32->39 catch 33->40 gearbox 34->41 vibdecomp 35->42
bolt 36->43     rotorcraft 37->44 hc 41->45 wadi 38->46    hai 40->47      swat 39->48
skab 47->49     hivae 48->50 vaem 49->51   tabor 50->52    calexnet 42->53
```

### VERIFICATION
- **Checker (a)**: custom script — every in-text `[N]` equals the list position of its href anchor: **71 anchors checked, 0 mismatches, 0 dangling `#r-` anchors**.
- **bibtest (b)**: `python -m bibtest --summary check-html paper/IoT2.html` -> **53/53 valid, "All references resolve. OK"**.
- **Finding (c)**: 5 hardcoded plain-text bracket citations remain (`[1, 9, 5]`, `[38]`, `[39]`, `[40]`, `[6, 8]`). All are inside `rev-del` (struck, deleted) text, are NOT `<a href>` anchors, and are therefore out of scope for the anchor renumber; they display their original (pre-renumber) numbers, consistent with being marked as removed old text. Left as-is (not shipped content).

### DOCX REBUILD — `paper/paper_mdpi.docx` (489 KB)
Rebuilt via `html2doc/scripts/mdpi_from_html.py` with `--template paper/_journal/IoT-template.docx` (the filled IoT template carrying the real title/authors/affiliations; confirmed correct in the output front matter). Change-marking reproduced via the same sentinel-inject + python-docx post-pass mechanism as the prior pass:
- `rev-highlight` -> **yellow** (pipeline built-in).
- `rev2-highlight` -> **BRIGHT_GREEN** (`w:highlight val="green"`); rev2 takes precedence over yellow when nested (round-2 additions win), so new sentences inside a previously-yellow paragraph and citation numbers inside rev2 sentences render green.
- `rev-del` -> **strikethrough + red font C0392B**.
Verified in `word/document.xml`: **green x558, yellow x215, strike x26 + red C0392B x26, 0 sentinels remaining**, **84 native OMML equations**, **3/3 figures embedded** (inline SVGs rasterized). Green run count differs from the prior pass's 699 due to run-fragmentation differences in the sentinel approach and the renumbered/added content, not missing regions; per-region audit confirms all rev2 regions (new sentences, inline rev2-in-yellow spans, figure captions, table cells, nested citation numbers) are green and all rev-del regions are struck.

---

## CORRECTIONS WIRING PASS (2026-09-17) — in-hand corrections wired into IoT2.html + letter, DOCX rebuilt

Approved edit. `/c/Python314/python`, wd `E:\Projects\Backlog\LatAD\poc`. No git commit. Marking scheme kept:
new/changed text `rev2-highlight` (green), deletions `rev-del` (red strike). Every number below is quoted from
the artifact cited; nothing invented.

### 1. A3 / SKAB witness — single-seed → 5-seed honesty correction (source `_diagnostics/a3_seed_robustness.json`)
- §7 SKAB paragraph and Table 6 SKAB row rewritten from the single lucky seed ("0.500 → 0.605") to the **5-seed**
  result: difficult AUROC **OFF 0.483±0.024 → basin-ON (λ=1) 0.536±0.041, lift +0.053±0.031 (positive on all 5
  seeds)**; **λ=2 → 0.566±0.045**. Verified means/SDs recomputed from the JSON (OFF mean 0.483, SD 0.024; ON1 mean
  0.536, SD 0.041; lift mean 0.053, SD 0.031; ON2 mean 0.566, SD 0.045; all 5 lift1 values > 0).
- Kept **ρ = 0.58 as the seed-0 value** and added that **ρ is seed-variable (0.37±0.20 over 5 seeds)** while the
  effect is consistent (ρ over seeds 0.58/0.552/0.453/0.177/0.075 → mean 0.367, SD 0.204).
- Table 6 caption updated: SKAB over five seeds (mean±SD), ρ/max-resp/λ shown at seed 0; benchmarks single-seed.
  Table 6 column header → "diff. AUROC, forced on (λ=1)". The in-text "material gain only on SKAB (+0.10)" → "(+0.053
  over five seeds)".
- **Reworded A3 from thin-pocket to mode overlap** (evidence `_diagnostics/fable_a3_measures.md`,
  `per_community_rho.json`): Table 1 A3 name "Hard envelopes (bounded regimes; thin between-regime pockets)" →
  **"Overlapping regimes (heavy mode overlap between regimes)"**; description and motivates cell rewritten (basin head
  "detects overlap"). §4.3(iv) opening reworded ("Between-regime pockets" → "Heavy mode overlap between regimes";
  "between-regime point" → "mode-ambiguous point in a zone of overlap"; §4.3 "between-regime faults absent" → "heavy
  mode overlap is essentially absent"). §7 witness heading → "…on real mode-overlap faults"; wording aligned.

### 2. Per-community ρ (source `_diagnostics/per_community_rho.json`)
Added one sentence to §7 (after the Table 6 discussion): A3 scarce globally AND per subsystem — every community
**ρ < 0.16** on all three benchmarks (WADI ρ_max 0.029 / 44 comm; HAI 0.155 / 28; SWaT 0.122 / 25; none ≥ 0.30),
vs SKAB global ρ ≈ 0.58. All values match the JSON (rho_max and n_comm per dataset; n_comm_A3_ge0p30 = 0 for all).

### 3. E3 cost table (sources `sota_bundle/results/e3_cost_{cuda,cpu}.json`)
Replaced the §7 "We report no parameter count, memory, latency, or energy measurements here" sentence with measured
numbers and added **Table 7** (rev2-highlight). Numbers pulled from the JSON:
- Train (WADI/HAI/SWaT s): LatAD-global 3.2/16/3.5; TranAD 231/898/163 (train_only_s 231.3/897.8/163.5); USAD
  96/592/95; LatAD-community 50/195/38; AE 0.2/0.0/0.0. → global trains ~47–72× faster than TranAD (72/56/47).
- GPU ms/win (b1): LatAD-global 0.7/13.0/13.0; community 32.4/18.5/17.2 (b256 up to ~52, WADI 51.6); AE 0.16/0.15/0.15.
- CPU-edge ms/win (b1): LatAD-global 0.6/14.6/14.5; community 25.0/16.2/14.5; AE 0.08/0.07/0.07.
- Params: global 209k/112k/100k; community 1.25M/0.81M/0.78M; TranAD 1.27M/0.30M/0.23M; USAD 32k/16k/14k; AE 97k/48k/42k.
- Host RAM ~3.5–3.9 GB (CPU-edge peak_host_ram 3529.6–3865.7 MB). CalexNet early-exit kept as further-reduction path.

### 4. A-01 abbreviation expansions at first use
Added (rev2-highlight, first textual use): F1 "(the harmonic mean of precision and recall)"; VaDE "Variational Deep
Embedding"; Higher Criticism "(HC)"; AUROC "(area under the ROC curve)"; GMM "Gaussian mixture model (GMM)"; NLL
"negative log-likelihood (NLL)"; HIL "hardware-in-the-loop (HIL)". CPS, SCADA, SOTA, BIC, IIoT, LatAD already
expanded. (SOTA already at §2.2; verified present.)

### 5. A-14 Table 1 "Realized by (§)" column
Added a fifth column mapping each A1–A10 to its LatAD realization (A1/A7 VaDE regimes §4.1; A2 K + mixture-density
head §4.1/§4.3; A3 basin head §4.3; A4 mixture-density head §4.3; A5 low-dim latent §4.1; A6 nearest-component NLL
§4.3; A8 standardization/whitened residual §4.3 + community factorization §4.4; A9/A10 future work §7). Facts from §4,
no new numbers.

### 6. Response letter (`review_round_1/response_letter.md`) synced
R1-1/A3 and E1(Ed-A) updated to the 5-seed numbers (0.483→0.536, +0.053±0.031, all-5-seed, ρ 0.37±0.20 / 0.58 s0)
and the mode-overlap rewording; per-community-ρ note marked "now in §7". Also refreshed the "pending" coordinator
notes I actually landed this pass: R1-3/E3/R2-10 cost → "folded into §7, Table 7"; R2-1 abbreviations → "now
expanded"; R2-8 Realized-by column → "now added". Confirmed no "0.500/0.605" remains in the letter.

### E5 / A-25 — NOT done by design (separate agent recomputing on clean data)
Left `<!-- TODO E5/A-25 pending clean recompute -->` placeholder before §7 where the consolidated "where the gain
comes from" table will go. E5/A-25 source-of-gain content untouched.

### Build & verify
- **bibtest**: `python -m bibtest --summary check-html paper/IoT2.html` → **53/53 valid, "All references resolve. OK"** (unchanged; no refs touched).
- **DOCX**: `E:\Projects\Backlog\LatAD\poc\paper\paper_mdpi.docx` rebuilt via `html2doc/scripts/mdpi_from_html.py`
  (`-t paper/_journal/IoT-template.docx`). Change-marking via a sentinel-inject + python-docx post-pass (build script
  `scratchpad/build_iot2_docx.py`): rev-highlight→yellow (pipeline built-in), rev2-highlight→green, rev-del→strike+red
  C0392B; SVG/MathML blackboxed to protect them from BS4 attr-lowercasing / tag re-nesting.
  Verified in `word/document.xml`: **green ×882, yellow ×212, strike ×31 + red C0392B ×31, 0 residual sentinels**,
  **3/3 figures embedded**, **native OMML equations present (converter reported 90)**. Temp files cleaned.
- **Consistency**: SKAB 5-seed numbers (0.483 / 0.536 / 0.566 / +0.053 / 0.37) appear identically in §7 prose,
  Table 6 caption + row, and the letter; no live "0.500 → 0.605" remains (only inside struck rev-del A3 text and the
  unrelated LinRes F1 0.605). Abstract/§6/§7/Table numbers cross-checked; no contradiction found.
- **Not committed** (per instruction).

---

## A3-renumber + E5 pass (2026-09-17)

Approved edit. `/c/Python314/python`, wd `E:\Projects\Backlog\LatAD\poc`. No git commit. Marking scheme kept:
new/changed text `rev2-highlight` (green), deletions `rev-del` (red strike). Every number quoted from its
cited artifact; nothing invented.

### TASK A — MIIM renumber + drop A3 (now known to be a variance-floor artifact)
**Collision-safe two-pass renumber applied** (`scratchpad/renumber.py`, old->TOKEN->new, protecting the
`A1-A8` range from endpoint drift; CRLF preserved on the HTML, LF on the letter). old->new map:
```
A1->A1  A2->A2  A4->A3  A5->A4  A6->A5  A7->A6  A8->A7  A3->A8  A9->A9  A10->A10
```
Applied EVERYWHERE: Table 1 rows + "Realized by (§)" column, §3, §4 mechanism<->assumption refs, the
Figure 2 SVG assumption tags (`[A8]`->`[A7]`, `[A1/A7]`->`[A1/A6]`, `[A4]`->`[A3]`, `[A6]`->`[A5]`; the old
`[A3]` basin block deleted), §7, and the response letter. No A3-A8 false positives existed (Table A1 / WADI.A1 /
A-## are all A1 or hyphenated), so only the `A1-A8` range needed protection.

- **A1 renumber, collision-safe:** DONE. Verified: Table 1 rows now read A1 Regime mixture, A2 Regime explosion,
  A3 Thin fringes (was A4), A4 Few levers (was A5), A5 Heavy tail (was A6), A6 Hidden regimes (was A7),
  A7 Mixed signals (was A8), A8 Between-regime overlap (was A3), A9 Many clocks, A10 Path dependence.
  Spot-check requested by the brief: old-A4 "thin fringes" now reads **A3** (confirmed).
- **A2 Table 1 reorder + two visually-grouped blocks:** DONE. Block 1 "Realized and validated (A1-A7)",
  block 2 "Specified, not observed in this corpus (A8-A10)".
- **A3 new-A8 graded/specification wording (no "validated"):** DONE (Table 1 A8 row + §1 scope note +
  §4 preamble): "specified for completeness; across eight real CPS datasets not observed (responsibility
  entropy <= 0.06); the earlier SKAB signal is a variance-floor artifact of the estimator on n=400; left as
  future work with A9-A10."
- **A4 graded A1-A10 framing in §3 (2 sentences):** DONE (checklist, not a universal law; a dataset realizes
  some strongly and others weakly/not at all; LatAD exploits each when present and reduces to a standard
  latent-density model when absent).
- **A5 basin head withdrawn (paper text only; models_vade.py untouched):** DONE.
  * §4.3(iv) retitled "The combined window score"; Eq. (7) reduced to `s(x)=s0+s_resid` (equation count kept at 9);
    basin mechanics replaced by a green withdrawal note + a `rev-del` deletion marker.
  * **Table 6 (SKAB basin witness) DROPPED.** Cost table renumbered 7->6->7 so numbering is contiguous
    (final: Table 5 source-of-gain [new], Table 6 head-ablation, Table 7 cost, Table A1 appendix).
  * Head-ablation table had no "base+basin" row; its caption reframed ("overlap head withdrawn, not shown").
  * Figure 2 SVG basin block + `[A8]` tag + its connector arrow removed; legend + figcaption reconciled to
    one optional auto-gated whitened-residual head.
  * §7 basin/SKAB validation paragraphs + Table 6 replaced with a scarcity+artifact discussion
    (`scratchpad/drop_table6.py`): 8-dataset screen (entropy <=0.06); per-community rho<0.16; SKAB entropy
    0.29 -> 0.045-0.048 with empirical variances (indistinguishable from WADI 0.053-0.057); rho gate + lift
    collapse to ~0 at floor 0.01 / empirical variances, every seed (fable_a3_spaces.md §7). SKAB ref [49]
    re-anchored so it stays cited.
  * §5.2 basin hyperparameter sentence struck; §5.4 "basin scale and reference" calibration item removed.
- **A6 response letter:** R1-1 and E1 rewritten to the honest framing (A8 not demonstrated; SKAB = variance-floor
  artifact; 8-dataset + per-community scarcity; basin head withdrawn; no "validated" claim). Overview line, R2-8
  "Realized by" mapping (basin removed, two-block grouping), and R1-fig updated for the dropped table.

### TASK B — E5 source-of-gain table (placeholder filled)
Replaced `<!-- TODO E5/A-25 -->` with a "Where the gain comes from" paragraph + new **Table 5** using the
CLEAN 5-seed recompute (`_diagnostics/e5_gain_clean.json`): cross-channel latent density vs channel-independent
product of marginals, difficult-subset AUROC gain **WADI_clean -0.004, SWaT_canon +0.141, HAI +0.200** (cross
0.743/0.789/0.795 vs marginal 0.747/0.589/0.654 on WADI/HAI/SWaT). States plainly that the gain is carried by
HAI and SWaT, is ~0 on cleaned WADI (single-channel/linear difficult anomalies), and that the earlier
WADI +0.343 was the `2B_AIT_002_PV` artifact. Consolidated narratively with the head-level ablation (Table 6)
and the factorization ablation (Table A1). Letter R1-5 / E5 updated to the clean numbers and "now in the manuscript".

### VERIFICATION
1. **Stale-tag grep:** every A1-A10 reference resolves under the NEW mapping; no assumption name attached to the
   wrong number (all inline tags dumped with context and audited: thin fringes=A3, few levers=A4, heavy-tail=A5,
   hidden regimes=A6, mixed/community/whitened-residual=A7, between-regime overlap=A8, VaDE regimes=A1/A6,
   trajectory=A9/A10). No A-tag > A10. Table 1 rows verified A1..A10 in order with correct names.
2. **No residual stale language:** no live "Table 6 (SKAB)" / "validated on SKAB" / "0.500->0.605" / basin-as-shipped-head.
   Remaining "basin"/"SKAB"/"rho=0.58"/"0.343" occurrences are all inside the withdrawal/artifact-correction text
   or `rev-del` deletion markers. Equations (1)-(9) intact; Figures 1/2/3; Tables 1-7 + A1 contiguous.
3. **bibtest:** `python -m bibtest --summary check-html paper/IoT2.html` -> **53/53 valid, "All references resolve. OK"**.
4. **Figure count/tags:** 3 figures; Figure 2 SVG tags now [A7],[A1/A6],[A3],[A5] (basin [A8] removed), consistent
   with Table 1 and the §4 text.
5. **Cross-file numbering:** Table 1 / §3 / §4 / §7 / Figure 2 / letter all use the same A1-A10 mapping and the
   same table numbers (source-of-gain 5, heads 6, cost 7) with no contradiction found.

### DOCX
`E:\Projects\Backlog\LatAD\poc\paper\paper_mdpi.docx` (rebuilt via `scratchpad/build_iot2_docx.py`:
`html2doc/scripts/mdpi_from_html.py -t paper/_journal/IoT-template.docx` + sentinel-inject/python-docx post-pass;
stale `~$per_mdpi.docx` lock removed first). `word/document.xml`: **green x806, yellow x212, strike x30 + red
C0392B x30, 0 residual sentinels, 0 @@BBSLOT**, **3/3 figures embedded**, **69 native OMML equations**,
200 body blocks transplanted. Temp intermediates and renumber backups cleaned.

### Honesty note
This pass REMOVES a core claim (the A3 basin validation). The renumber was verified clean (zero stale/misattributed
tags) BEFORE declaring done, so TASK A's renumber was kept (not reverted). Not committed (per instruction).

---

## FINAL POLISH (2026-09-18) — A-07 English/tone pass, A-20 indentation, render-QA, letter sync

Approved final pre-submission polish. /c/Python314/python, wd E:\Projects\Backlog\LatAD\poc. No git commit.
Coordinator mid-task rule applied: **every change this round is colour-coded** (rev2-highlight green for new/changed
text, rev-del red strike for removals) in BOTH the HTML and the rebuilt DOCX.

### STEP 1 — A-07 English/tone pass (whole manuscript)
Tone audit: **em-dashes fixed to zero.** 3 mdash occurrences removed: §7 A8 paragraph (mdash -> comma, inside an
already-green span); Table 7 caption and the two Table 7 "not measured" cells (mdash -> ndash, the cells greened).
No literal U+2014, no double-hyphen prose, and zero forbidden tone words
(honestly/frankly/candidly/admittedly/unfortunately/it should be noted/we concede) — grep-clean. wins-only scan: clean.
No apologetic/defensive phrasing introduced.

English edits (light-touch, all MARKED this round): 3 over-length run-on splits — §1 "...hard part of the problem.
Cyber-physical..." (was ~61 words), §3 "...crisply on SWaT (0.29). This ordering..." (was ~70), §4.4 "...plus one global
expert). We read its calibrated surprise..." (was ~90) — each marked with a rev-del strike of the removed connective plus
a rev2-highlight green of the reworded punctuation. One nested-paren cleanup in §1 ("A8 (between-regime overlap),
specified" -> "A8, between-regime overlap, specified", inside an already-green span). One §7 CalexNet split was reverted
(it sat inside a round-1 region and marking it cleanly was not worth the clutter).

### Marking-completeness audit (coordinator verification #1 and #3)
Word-level diff of rendered text, current IoT2.html vs git HEAD (74eddbb), flagging any changed text NOT inside a
rev-span. Result: **5 unmarked prose rewordings from earlier passes found and greened this round** —
(1) §2 intro "On time series, anomaly detection must respect..." ; (2) §2.3 opening "Latent and clustering models detect
anomalies..." + the VaDE/DAGMM survey sentences ; (3) §4.2 "The mechanism is that a flexible decoder can reconstruct a
fault faithfully..." ; (4) §5.1 "Remaining standardized features are clipped to +/-10 sigma..." ; (5) Figure 3 caption
first half ("Difficult-subset AUROC by method (five-seed mean for the learned detectors; whiskers...)"). Old text for
these was already removed by the earlier passes, so per the coordinator rule they were greened as reworded (no
strikethrough to reconstruct). After fixing: **0 significant unmarked prose runs remain** (>=3 real words, refs excluded).
**In-place NUMBER changes: 0 unmarked** — every changed table cell (Tables 3/4/5/6/A1) and every abstract headline number
is already inside a rev2-highlight row/span; the audit found no un-greened number change.
Intentionally left unmarked (reported, not defects): the reference-list first-appearance REORDER (a structural renumber;
references are conventionally unmarked), the Table 1 assumption-ID renumber (signposted by the two green block-header rows
"Realized and validated (A1-A7)" / "Specified, not observed (A8-A10)"), and boundary punctuation adjacent to green inserts
(e.g. the "." after "0.000)" where the substantive GDN caption change is already green).

### STEP 2 — A-20 indentation (build/profile)
No HTML edit needed: the MDPI template already encodes the convention and the mdpi_from_html transplant maps pandoc
FirstParagraph -> MDPI_3.2_text_no_indent (firstLine 0, flush) and BodyText -> MDPI_3.1_text (firstLine 425 twips ~=0.30",
indented). Verified in the built DOCX: 34 flush first-after-heading paras, 44 indented continuation paras; references,
abstract, and figure/table captions all firstLine=None (no indent). **A-20 applied and confirmed.**

### STEP 3 — DOCX rebuild + render-QA  (paper/paper_mdpi.docx, 529,722 bytes)
Rebuilt via scratchpad/build_iot2_docx.py = html2doc/scripts/mdpi_from_html.py -t paper/_journal/IoT-template.docx
+ sentinel-inject/python-docx marking post-pass. Rendered to PDF (Word COM ExportAsFixedFormat) and rasterized (PyMuPDF)
for visual QA.

**OMML-drift investigation (84->90->69->67): explained as content-driven, not a rendering fault.** The drop across passes
tracks the basin-head withdrawal (Eq. (7) reduced), the Table 6 (SKAB) deletion, and this pass's eq(2) fix. **But visual
QA caught a REAL REGRESSION:** eq(2) (the VaDE prior p(z)=Sum pi_c N(...)) was the only equation authored as a
display="block" math EMBEDDED mid-paragraph inside a <p> (eqs 3-9 are standalone between paragraphs; eq(1) is
display="inline"). A prior "number all equations" pass had added a <span class="eqno">(2)</span> before that embedded
block-math, which made pandoc split the paragraph and the transplant DROP the entire §4.1 VaDE prose ("variational
autoencoder whose latent prior...", "An encoder maps...", "negative evidence lower bound", "per window it is"), jumbling
the math onto one line mis-numbered "(3)". The committed HEAD DOCX did NOT have this (prose FOUND), confirming a
regression. **Fix:** changed eq(2)'s display="block" -> display="inline" (matching the proven eq(1) pattern) — an
invisible attribute change, no visible content/number change, so no rev-marking applies. After the fix, all §4.1 prose is
restored and every equation renders (verified visually).

Render-QA table:

| Item | Result |
|---|---|
| Display/inline equations | **9/9 present as real OMML** (67 native OMML runs total). Eqs (1),(2) inline (number before eq, prose intact); eqs (3)-(9) proper centered display equations with right-margin numbers (visually verified pp. 6, 11, 12, 13, 14). No equation rasterized-as-text; no equation missing. |
| Figures | **3/3 embedded, uncropped, legible** (Fig 1 IIoT context, Fig 2 workflow with assumption tags [A7]/[A1/A6]/[A3]/[A5], Fig 3 difficult-subset chart). |
| Tables | **9 rendered** (Table 1,2,3,4,5,6,7,A1 + 1 front-matter); content present, no overflow, three-line MDPI format (Tables 2 and 3 visually verified; row counts 13/4/29/9/4/6/6/11). |
| Change-marking | green highlight x837 runs, yellow x208, red strikethrough x33 (+ red font C0392B x33), 0 sentinels, 0 @@BBSLOT. Green + red strike both render (verified pp. 6, 11, 12, 16). |

**HTML rev-span <-> DOCX mark correspondence (coordinator verification #2):** HTML has rev2-highlight 198 elements,
rev-del 14, rev-highlight 68. DOCX: green 837 runs >= 198 (spans fragment into runs; every rev2 region green),
strike 33 >= 14 (every rev-del region struck), yellow 208 >= 68. No rev-span renders as plain text; no sentinel leaks.

### STEP 4 — Letter <-> paper consistency (SC-2/5/7) + letter DOCX
Cross-checked review_round_1/response_letter.md against the renumbered paper. Headline numbers all match
(WADI 0.824, SWaT 0.840 P=0.003, HAI 0.849; full AUROC 0.864/0.949/0.941; E5 -0.004/+0.141/+0.200; HAI CI [0.047,0.157],
SWaT CI [0.015,0.107], DH P=0.0005; 122 channels, 449,919 rows, 12.14%). A1-A10 numbering identical in both (A8 =
between-regime overlap; A9-A10 trajectory/future). No stale A3/basin/Table-6/0.605/0.500/near-significant CLAIM
(the SKAB/basin mentions are the correct withdrawal narrative). **2 stale citation numbers fixed IN THE LETTER**
(paper numbers were verified correct, not touched): R1-5 "[ref 9]" -> "[ref 20]" (the "univariate on almost all timesteps,
no cross-time-only segments" finding is Pinet = ref 20; ref 9 is Wu&Keogh, a different claim); R2-6 "(refs 43-45)" ->
"(refs 28-30)" (Deep SVDD/THOC/OmniAnomaly are 28/29/30 after the renumber, not the pre-renumber 43-45). Letter DOCX
rebuilt via pandoc -> review_round_1/response_letter.docx (22,890 bytes); fixes confirmed present, stale tokens absent.

### Gates
- **bibtest**: python -m bibtest --summary check-html paper/IoT2.html -> **53/53 valid, "All references resolve. OK"**.
- **em-dash**: 0 (mdash and U+2014). **wins-only scan**: clean. **tone words**: 0.
- Temp intermediates cleaned (_iot2_marked.html, inline-svg PNGs, ~$per_mdpi.docx lock).

### Deliverables (no commit)
- E:\Projects\Backlog\LatAD\poc\paper\paper_mdpi.docx
- E:\Projects\Backlog\LatAD\review_round_1\response_letter.docx

### Honesty / flags
- One genuine **regression fixed**: the §4.1 VaDE paragraph was being dropped from the DOCX (eq(2) embedded block-math +
  its later-added number); caught by visual render-QA, root-caused against the committed HEAD DOCX, and fixed with a
  minimal display attribute change. All 9 equations now render.
- Nothing could NOT be fixed. No claim, number, result, table cell, or figure content was changed; the eq(2) fix is a
  render-only attribute and the two letter edits are citation-number corrections in the letter.
