# framing_applied.md — content updates applied to the LatAD manuscript and letters

Date: 2026-09-18. Branch: revision2. HTML edits only; DOCX not rebuilt; citation numbers not renumbered.

## Target file resolution (important)

The active manuscript is **`poc/paper/IoT2.html`**, not the root `paper.html`.
Reasons: (a) memory rule `iot-active-version` says edit IoT2; (b) only `poc/paper/IoT2.html`
contains Figure 3 and the `rev2-highlight`/`rev-del` change-marking classes; (c) it is the
most recently edited copy (today) and its table numbering matches the brainstorm
(head-decomposition = Table 6; localization = new Table 7). `paper.html` is read-only, has no
Figure 3, no change-marking classes, and a different table numbering (head-decomp = Table 5);
it was left untouched.

Table-number mapping: the brainstorm's "Table 6" (whitened-residual head decomposition) IS
Table 6 in `poc/paper/IoT2.html`. The new localization table is inserted as **Table 7** (the
paper previously ended the main sequence at Table 6; appendix tables are A1/B1/C1). This matches
the brainstork/task instruction exactly, so no numbering conflict.

Change-marking convention used (as mandated): new text wrapped in `<span class="rev2-highlight">`
(green), removed text wrapped in `<span class="rev-del">` (red strike). Letters carry no
change-marking convention (they are plain response documents using `class="resp"`), so the letter
edit is plain text in the letter's own style.

---

## UPDATE 1 — Density vs reconstruction (STRENGTHEN)

### Edit 1.1 — Section 7 "Why reconstruction fails and density wins" paragraph

Location: `poc/paper/IoT2.html`, "Why reconstruction fails and density wins" paragraph (was
lines ~1036-1041). The prior rev2 summary span was struck (`rev-del`) and replaced with the
brainstorm's Issue-1 paragraph verbatim (`rev2-highlight`).

OLD (now struck, `rev-del`):
> The reconstruction-based deep detectors show this on the difficult subsets: they trail LatAD on
> all three and fall to near chance on HAI (USAD 0.477, TranAD 0.444, and the graph detector GDN
> 0.481, which learns inter-channel relations yet collapses on the same faults), whose difficult
> faults are reconstructable but improbable. At the head level (Table 6) the whitened
> reconstruction residual is the weakest base head on HAI (0.695) while the latent density carries
> the signal (0.802).

NEW (added, `rev2-highlight`, verbatim from brainstorm Issue-1 block, line 75):
> Scoring by density rather than reconstruction is the right default for the deployed
> reconstruction detectors: the single latent-density head, with no residual term and no
> factorization, exceeds USAD and TranAD on the difficult subset of all three plants (HAI 0.802
> against 0.477 and 0.444; SWaT 0.794 against 0.658 and 0.655; WADI 0.656 against 0.579 and 0.613)
> and matches GDN on WADI (0.660). The margin is largest where the faults are reconstructable but
> improbable: on HAI the reconstruction detectors place the difficult attacks at the median of
> normal operation while the density head places them at the 93rd percentile, and 27 percent of
> HAI's difficult attacks reconstruct at least as well as a typical normal window yet fall above
> the 90th normal percentile in density; on WADI that share is zero and 70 percent of the difficult
> attacks are weak under both scores, which is why the community factorization, not the head,
> carries the WADI gain. A reconstruction residual *whitened on train-normal* (Table 6) is a
> stronger term than the reconstruction the deep detectors score with; it overtakes the density
> head on WADI and SWaT, and this is precisely why LatAD keeps it as a gated head where it
> generalizes to held-out normal instead of discarding reconstruction.

(The subsequent original sentence "Scoring by density in the clustered latent replaces
reachability with probability..." is retained unchanged.)

No index-law claim added; no SKAB/Cranfield; no synthetic figure (all deferred per instruction).

### Edit 1.2 — Table 6 caption addition

Location: `poc/paper/IoT2.html`, Table 6 caption. Appended (added, `rev2-highlight`):
> The reconstruction row is the whitened residual; the deployed reconstruction detectors of
> Table&nbsp;3 score 0.44 to 0.66 on the same difficult subsets.

---

## UPDATE 2 — Subsystem localization (MODEST POSITIVE + caveats)

### Edit 2.1 — Section 4.4 localization sentence (soften bare assertion, forward-reference)

Location: `poc/paper/IoT2.html`, Section 4.4 ("The factorization also localizes:").

OLD (struck, `rev-del`):
> turning a plant-wide alarm into a subsystem-level lead for the operator.

NEW (added, `rev2-highlight`):
> ranked by their train-normal upper-tail p-value (the quantity the Higher-Criticism combiner
> already uses), so a plant-wide alarm becomes a subsystem triage shortlist; Section&nbsp;7
> (Table&nbsp;7) validates that a three-community shortlist contains an attacked subsystem in 68 to
> 79 percent of HAI and SWaT attacks.

### Edit 2.2 — Section 7 deployment assertion (replace unvalidated claim)

Location: `poc/paper/IoT2.html`, "Implications for IIoT deployment" paragraph.

OLD (struck, `rev-del`):
> localizes an anomaly to a physical subsystem rather than issuing only a plant-wide flag

NEW (added, `rev2-highlight`):
> turns a plant-wide alarm into a subsystem triage shortlist rather than only a plant-wide flag
> (validated below, Table&nbsp;7)

### Edit 2.3 — New localization paragraph (Section 7, added `rev2-highlight`)

Inserted immediately after the "Implications for IIoT deployment" paragraph. Verbatim from the
brainstorm Issue-2 block (line 125), with the task's expanded caveat (ii) and the headline framed
as the top-3 shortlist:

> **Subsystem localization.** A three-community shortlist ranked by train-normal p-value contains
> the attacked subsystem in 68 to 79 percent of HAI and SWaT attacks, against 49 to 56 percent for
> a random shortlist of the same size. The per-community surprises also localize. Ranking
> communities by their train-normal upper-tail p-value (the quantity the Higher-Criticism combiner
> already uses; communities whose calibration surprise is degenerate carry no p-value and are not
> ranked) and naming the top community per attack episode, the named community contains an attacked
> channel in 46 percent of SWaT attacks (11 of 24, coverage-aware random 22 percent, P = 0.004) and
> 40 percent of HAI attacks (15 of 38, random 25 percent, P = 0.02), and a three-community shortlist
> contains it in 79 and 68 percent (random 49 and 56 percent). The named community averages six
> channels of 51 (SWaT) and 59 (HAI). WADI, with 9 attacks whose targets are published and 7 of
> them inside any community, is directionally the same (3 of 9 at top-1, random 13 percent) but too
> small to test. The attack targets are read from the published per-testbed attack tables [50, 51,
> 52]. Two limits apply. (i) A community whose surprise drifts above its calibration range is named
> disproportionately often even on normal windows, so the pointer is a triage shortlist rather than
> a diagnosis. (ii) Larger communities are easier to hit, so the shortlist is reported at its size;
> against a size-weighted random baseline the SWaT margin holds while HAI's top-1 margin is small,
> which is why the top-3 shortlist is the comfortable headline.

Citations use the EXISTING dataset references: `#r-wadi` [50], `#r-hai` [51], `#r-swat` [52].
These appear first in Section 5.1, so reusing them in Section 7 does not change first-appearance
order or any citation number.

### Edit 2.4 — New Table 7 (Section 7), as inserted

```html
<div class="tblwrap">
<table>
<caption class="rev2-highlight"><strong>Table 7.</strong> Subsystem localization at the attack-episode level, rank rule (communities ordered by their train-normal upper-tail p-value; degenerate-calibration communities are unranked). <em>Top-1</em> names the single most-surprised community; <em>top-3</em> is a three-community shortlist. The random baselines are coverage-aware (the chance that a randomly named community of the same resolution contains an attacked channel), averaged over episodes. Bold marks the two datasets with enough episodes to test the margin.</caption>
<thead><tr class="rev2-highlight"><th class="l">Dataset</th><th>Attack episodes (targets known)</th><th>Named-community size (channels / plant)</th><th>Top-1</th><th>Random top-1</th><th>Top-3</th><th>Random top-3</th></tr></thead>
<tbody>
<tr class="rev2-highlight"><td class="l">SWaT</td><td>24</td><td>6.5 / 51</td><td><b>0.46</b></td><td>0.22</td><td><b>0.79</b></td><td>0.49</td></tr>
<tr class="rev2-highlight"><td class="l">HAI</td><td>38</td><td>5.6 / 59</td><td><b>0.39</b></td><td>0.25</td><td><b>0.68</b></td><td>0.56</td></tr>
<tr class="rev2-highlight"><td class="l">WADI</td><td>9 (7 coverable)</td><td>5.4 / 122</td><td>0.33</td><td>0.13</td><td>0.56</td><td>0.31</td></tr>
</tbody></table>
</div>
```

Rendered:

| dataset | attack episodes (targets known) | named-community size (ch / plant) | top-1 | random top-1 | top-3 | random top-3 |
|---|---|---|---|---|---|---|
| SWaT | 24 | 6.5 / 51 | **0.46** | 0.22 | **0.79** | 0.49 |
| HAI | 38 | 5.6 / 59 | **0.39** | 0.25 | **0.68** | 0.56 |
| WADI | 9 (7 coverable) | 5.4 / 122 | 0.33 | 0.13 | 0.56 | 0.31 |

Bold = the two datasets with enough episodes to test (SWaT, HAI); WADI shown unbolded (too small).

---

## LETTERS

Searched all three letters (`letter_editor.html`, `letter_reviewer1.html`, `letter_reviewer2.html`)
for localization / reconstruction-vs-density comments.

- No letter carries a dedicated localization-validation comment or a reconstruction-vs-density
  concession that the new results contradict. No letter echoes the old "deployment architecture
  rather than an experimentally evaluated configuration" framing (that string is absent from all
  three), so nothing there needed correction.
- The one letter passage that touches the subsystem pointer is **Reviewer 2, Comment 7** (the
  system-model diagram response, which mentions "the most-surprised subsystem is forwarded to the
  operator"). Updated it to reflect the now-validated triage result. The letters use no
  change-marking convention, so this is plain text in the letter's own style:

  ADDED to end of Reviewer 2 Comment 7 response:
  > The most-surprised community that Figure 1 forwards is now a validated triage output: Section 7
  > (new Table 7) reports that a three-community shortlist ranked by train-normal p-value contains
  > the attacked subsystem in 68 to 79 percent of HAI and SWaT attacks, against 49 to 56 percent for
  > a random shortlist of the same size.

- Reviewer 1 (Comments 2 and 5) and the editor letter already frame the gain as cross-channel
  density and do not concede reconstruction wins, so they are consistent with "density beats the
  deployed reconstruction detectors on all three" without edits. No change made to those two.

---

## CITATION FLAG (per instruction: flag, do not insert)

No new reference was inserted. One soft flag for the author's judgment:

- The localization ground truth comes from the precise **per-attack target-channel tables**: the
  iTrust SWaT "List_of_attacks_Final" (36 labelled attacks), the HAI 20.07 technical-details
  timetable, and the WADI attack table (as distributed with TranAD). The manuscript cites the three
  dataset papers ([50] WADI, [51] HAI, [52] SWaT), which describe the testbeds and attack campaigns;
  those citations reasonably cover the attack lists. IF a reviewer wants the exact attack-list
  documents cited as distinct sources (rather than the dataset papers), that would require a new
  reference each (iTrust attack-list doc; WADI attack-table artifact) — NOT inserted here, flagged
  only.

---

## SELF-CHECK

- **Change-mark spans balanced**: `poc/paper/IoT2.html` has 299 `<span>` opens / 299 `</span>`
  closes (balanced). All added text in `rev2-highlight`, all removed text in `rev-del`.
- **No em-dash**: 0 U+2014 characters in the file. No double-hyphen in prose (the 24 `--` matches
  are all CSS custom properties like `--text`/`--rule` and HTML comment markers, none in body text).
- **No "honestly"** (or honest/frankly/candidly): 0 occurrences.
- **MathML/SVG unchanged**: no `<math>`/`<svg>` block was edited; the Table 6 caption's inline
  MathML (M=80) was preserved untouched; the new Table 7 contains no math.
- **Figure 3 untouched**: single `Figure 3.` caption block intact (the only one in the file); no
  edit touched the Section 6 figure/SVG region (lines ~1001).
- **Citation NUMBERS untouched**: no reference renumbered; new in-text citations reuse existing
  anchors `#r-wadi`/`#r-hai`/`#r-swat` (displayed [50]/[51]/[52]), which first appear in Section 5.1,
  so first-appearance order is unchanged. Author's post-edit renumber pass remains applicable.
- **Table 7 numbering**: main sequence previously ended at Table 6; Table 7 is the next free number
  (appendix tables are A1/B1/C1). No collision.
- **DOCX not rebuilt** (per instruction).
