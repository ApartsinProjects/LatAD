"""Assemble the real-CPS EDA report (self-contained HTML with embedded figures)."""
import base64, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def b64(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


FIG = {n: b64(f"eda_figs/{n}_eda.png") for n in ["SKAB", "HAI", "WADI", "SWaT"]}

HTML = f"""<div class="wrap">
<header>
  <div class="eyebrow">LatAD &middot; prior-research PoC</div>
  <h1>Real-CPS anomaly EDA</h1>
  <p class="lede">How hard are the anomalies in the public CPS benchmarks, and how strongly is the
  MIIM structure (A1&ndash;A10) actually present? A window-level look at <b>SKAB</b>, <b>HAI</b>,
  <b>WADI</b> and <b>SWaT</b> &mdash; anomaly simplicity, difficulty stratification, mode structure,
  per-cluster robustness, root-cause of the worst errors, and a cross-check against published SOTA.</p>
  <div class="chips">
    <span class="chip amber">benchmarks mostly easy</span>
    <span class="chip ink">MIIM present but implicit</span>
    <span class="chip teal">WADI cleaning win 0.69&rarr;0.79</span>
    <span class="chip crimson">global threshold unfair per-mode</span>
  </div>
</header>

<section>
  <h2>Verdict</h2>
  <p>A one-line <span class="mono">max|z|</span> rule already separates <b>~70&ndash;76%</b> of HAI and
  WADI anomalies; strip those and <b>every</b> detector falls toward chance. The multimodality is real
  but <b>implicit</b> (overlapping modes, silhouette 0.08&ndash;0.19). WADI carried
  <b>sensor-glitch / distribution-shift channels</b> that inflated false alarms and faked
  &ldquo;out-of-envelope&rdquo; anomalies; clipping them (A2 hard-envelope) lifts our detector past
  IsolationForest. A single global threshold is <b>wildly unfair per mode</b> (mean per-cluster FPR up
  to 0.77) &mdash; a direct mandate for mode-conditional thresholds.</p>
</section>

<section>
  <h2>The three datasets</h2>
  <div class="cards">
    <div class="card"><h3>SKAB</h3><p>Skoltech rotor / pump rig, 8 sensors @1&nbsp;Hz. Small,
      contaminated, 37% anomalous. The <b>only genuinely subtle</b> set.</p></div>
    <div class="card"><h3>HAI</h3><p>HIL ICS testbed, 59 sensors across 4 coupled processes (boiler,
      turbine, water). 4% anomalous. Genuinely multi-process &mdash; the thesis case.</p></div>
    <div class="card"><h3>WADI</h3><p>iTrust water-<i>distribution</i> CPS, 123 sensors. 10% anomalous
      (after horizon fix). Needs cleaning (see &sect;cleaning).</p></div>
    <div class="card"><h3>SWaT</h3><p>iTrust water-<i>treatment</i> CPS, 51 sensors (Kaggle mirror,
      non-canonical split). The <b>easiest</b> set &mdash; all methods 0.96&ndash;0.99 AUROC.</p></div>
  </div>
</section>

<section>
  <h2>Q2 &middot; Are the anomalies too simple?</h2>
  <p>Trivial detector = the largest per-channel window-mean z-score. &ldquo;Separable&rdquo; = anomaly
  score above the 99th percentile of normal.</p>
  <table>
    <thead><tr><th>dataset</th><th>trivial AUROC</th><th>trivially separable</th><th>reading</th></tr></thead>
    <tbody>
      <tr><td>SKAB</td><td class="num">0.57</td><td class="num">28%</td><td>&asymp; chance &mdash; genuinely subtle</td></tr>
      <tr><td>HAI</td><td class="num warn">0.88</td><td class="num warn">76%</td><td>mostly one-channel range faults</td></tr>
      <tr><td>WADI</td><td class="num warn">0.87</td><td class="num">66%</td><td>mostly out-of-range</td></tr>
      <tr><td>SWaT</td><td class="num warn">0.96</td><td class="num warn">79%</td><td>easiest &mdash; near-trivial</td></tr>
    </tbody>
  </table>
</section>

<section>
  <h2>Q3 &middot; Strip the easy anomalies &mdash; who survives?</h2>
  <p>HARD subset = all normal windows + only the anomalies the trivial rule <i>cannot</i> separate.
  AUROC shown as <span class="mono">full&nbsp;/&nbsp;hard</span>.</p>
  <div class="scroll"><table>
    <thead><tr><th>dataset</th><th>easy / hard n</th><th>trivial</th><th>IsolationForest</th><th>LOF</th><th>VaDE (ours)</th></tr></thead>
    <tbody>
      <tr><td>SKAB</td><td class="num">197 / 505</td><td class="num">.55 / .38</td><td class="num">.62 / .49</td><td class="num">.64 / .54</td><td class="num">.58 / <b>.42</b></td></tr>
      <tr><td>HAI</td><td class="num">243 / 76</td><td class="num">.82 / .34</td><td class="num">.87 / .62</td><td class="num">.91 / .68</td><td class="num">.88 / <b>.67</b></td></tr>
      <tr><td>WADI</td><td class="num">37 / 19</td><td class="num">.56 / .31</td><td class="num">.73 / .70</td><td class="num">.72 / .51</td><td class="num">.79 / <b>.51</b></td></tr>
      <tr><td>SWaT</td><td class="num">144 / 38</td><td class="num">.99 / .94</td><td class="num">.97 / .87</td><td class="num">.98 / .90</td><td class="num">.99 / <b>.95</b></td></tr>
    </tbody>
  </table></div>
  <p class="note">The headline AUROC on every benchmark is <b>carried by the easy majority</b>. On the
  hard residual all methods sit at 0.4&ndash;0.7. This is the quantified case for a generator that
  produces <i>hard</i>, history-dependent anomalies.</p>
</section>

<section>
  <h2>Q4 &middot; How is MIIM (A1&ndash;A10) expressed?</h2>
  <p>Mode-size distributions are heavy-tailed / Zipf-like (<b>A1</b> mode explosion + <b>A2</b>
  imbalance); BIC keeps improving past K=32&ndash;64 on HAI/WADI (many modes). But silhouette is
  0.08&ndash;0.19 everywhere: modes <b>overlap</b>, not crisply separable &mdash; <b>A5</b>
  &ldquo;regimes are an artifact, rarely recoverable.&rdquo; Anomaly geometry (each anomaly tagged by
  where it sits relative to the learned modes):</p>
  <table>
    <thead><tr><th>dataset</th><th>OUT of envelope (A2)</th><th>BETWEEN modes (A3)</th><th>RARE mode (A1)</th><th>IN a common mode</th></tr></thead>
    <tbody>
      <tr><td>SKAB</td><td class="num">20%</td><td class="num">2%</td><td class="num">0%</td><td class="num crit">78%</td></tr>
      <tr><td>HAI</td><td class="num crit">74%</td><td class="num">1%</td><td class="num">1%</td><td class="num">24%</td></tr>
      <tr><td>WADI (cleaned)</td><td class="num">4%</td><td class="num">4%</td><td class="num">0%</td><td class="num crit">93%</td></tr>
      <tr><td>SWaT</td><td class="num crit">81%</td><td class="num">1%</td><td class="num">0%</td><td class="num">18%</td></tr>
    </tbody>
  </table>
  <figure><img src="{FIG['SKAB']}" alt="SKAB EDA panels"><figcaption>SKAB &mdash; trivial-rule histograms
    overlap (AUROC 0.57); a clean BIC minimum at K=12; 78% of anomalies sit inside common modes; several
    clusters have &gt;0.5 error.</figcaption></figure>
  <figure><img src="{FIG['HAI']}" alt="HAI EDA panels"><figcaption>HAI &mdash; 74% out-of-envelope
    (A2); BIC never saturates (many modes); the anomaly tail is clearly separated in mode-NLL.</figcaption></figure>
  <figure><img src="{FIG['WADI']}" alt="WADI EDA panels"><figcaption>WADI (after cleaning) &mdash;
    normal and anomaly mode-NLL now <b>overlap almost entirely</b> (93% in-common); the &ldquo;easy&rdquo;
    envelope violations were the glitch artifacts.</figcaption></figure>
  <figure><img src="{FIG['SWaT']}" alt="SWaT EDA panels"><figcaption>SWaT &mdash; the cleanest structure
    (silhouette 0.31, 14+ modes); 81% out-of-envelope and 79% trivially separable. Even the &ldquo;hard&rdquo;
    residual stays at AUROC 0.95 &mdash; the easiest of the four, exactly as the SOTA literature implies.</figcaption></figure>
</section>

<section class="callout">
  <h2>Cleaning &middot; a real WADI data bug</h2>
  <p>Root-caused from the worst false alarms: a few channels carry <b>physically-impossible</b>
  standardised values.</p>
  <ul>
    <li><span class="mono">2_MCV_007_CO</span> is <b>constant = 0</b> across the 14-day normal file
      (&sigma;=0) but active later &mdash; standardising &divide;0 gives <span class="mono">+1.7e8&sigma;</span>.</li>
    <li><span class="mono">2B_AIT_002_PV</span> reads <span class="mono">9&plusmn;0.16</span> in normal
      but <span class="mono">4428</span> (up to 8128) in the attack file, even in no-attack periods
      &mdash; a recalibrated / non-comparable sensor.</li>
  </ul>
  <p>One such feature dominates VaDE&rsquo;s reconstruction and whitener and drowns every other channel
  (IsolationForest&rsquo;s tree splits are scale-invariant, so it was immune &mdash; that is why we
  trailed it). <b>Clipping standardised features to &plusmn;10&sigma;</b> (a 10&sigma; reading is a
  sensor fault, not a process state &mdash; <b>A2</b>):</p>
  <table>
    <thead><tr><th>clip</th><th>VaDE AUROC</th><th>VaDE TPR@5%FPR</th><th>VaDE F1</th><th>IF AUROC</th></tr></thead>
    <tbody>
      <tr><td>none</td><td class="num">0.691</td><td class="num">0.393</td><td class="num">0.423</td><td class="num">0.724</td></tr>
      <tr><td>&plusmn;10&sigma;</td><td class="num teal">0.791</td><td class="num teal">0.589</td><td class="num teal">0.574</td><td class="num">0.730</td></tr>
    </tbody>
  </table>
  <p class="note">Now beats IF on every metric. Clip is <b>WADI-specific</b>: on HAI it slightly hurts
  (big excursions are real signal), on SKAB it is a no-op. Baked into <span class="mono">wadi.py</span>.</p>
</section>

<section>
  <h2>Q6 &middot; Per-cluster robustness</h2>
  <p>Assign every test window to its nearest mode, score with VaDE at a <b>single global</b> 5%-FPR
  threshold, and measure the false-alarm rate <i>within</i> each mode.</p>
  <table>
    <thead><tr><th>dataset</th><th>mean per-cluster FPR</th><th>worst-cluster FPR</th><th>below-chance clusters</th></tr></thead>
    <tbody>
      <tr><td>SKAB</td><td class="num">0.13</td><td class="num crit">1.00</td><td>AUROC 0.37&ndash;0.52 on subtle in-mode clusters</td></tr>
      <tr><td>HAI</td><td class="num crit">0.77</td><td class="num crit">1.00</td><td>one mode mixes 19% anomaly, FPR 0.99</td></tr>
      <tr><td>WADI</td><td class="num crit">0.58</td><td class="num crit">0.92</td><td>clusters at AUROC 0.25&ndash;0.37</td></tr>
      <tr><td>SWaT</td><td class="num">0.16</td><td class="num crit">1.00</td><td>cleaner, but worst mode still saturates</td></tr>
    </tbody>
  </table>
  <p class="note">A global threshold is unfair per mode &mdash; direct empirical mandate for
  <b>mode-conditional thresholds</b> (architecture C4 / TFAR).</p>
</section>

<section>
  <h2>Cross-check &middot; does published SOTA agree?</h2>
  <p>Verified against the primary literature. The field's own evidence backs the same conclusion.</p>
  <table>
    <thead><tr><th>our EDA finding</th><th>literature</th><th>verdict</th></tr></thead>
    <tbody>
      <tr><td>anomalies mostly single-channel separable</td><td>SWaT ~95% / WADI ~99% univariate (arXiv 2606.02670)</td><td class="teal">confirms</td></tr>
      <tr><td>headline scores hide difficulty</td><td>a RANDOM score at PA-F1 0.97 SWaT / 0.97 WADI (Kim 2022; Garg 2021)</td><td class="teal">confirms</td></tr>
      <tr><td>simple baselines competitive; deep collapses on hard</td><td>PCA / channel-AE beat OmniAnomaly (Garg); triviality a named flaw (Wu &amp; Keogh)</td><td class="teal">confirms</td></tr>
      <tr><td>SKAB is the genuinely hard set</td><td>best F1 ~0.78, high FAR/MAR</td><td class="teal">confirms</td></tr>
    </tbody>
  </table>
  <p class="note"><b>Best honest (raw point-wise) F1:</b> SWaT ~0.81, WADI ~0.57 (GDN). The 0.89&ndash;0.96
  numbers quoted in most papers are <b>point-adjusted</b>. Our HAI 76%-separable figure and the
  strip-easy&rarr;collapse experiment are our own results &mdash; the literature confirms the mechanism
  and the SWaT/WADI direction but does not cover HAI/SKAB triviality.</p>
</section>

<section>
  <h2>Q7 &middot; What actually goes wrong</h2>
  <div class="cols">
    <div>
      <h3>False alarms</h3>
      <p>Valid windows in <b>tiny, rare modes</b> (mass &lt;2%, NLL at the 100th percentile). A rare-but-
      normal operating point looks anomalous &mdash; the <b>A1/A2</b> rare-mode problem, the same
      trajectory-domain <b>E1</b> we hit in the synthetic study.</p>
    </div>
    <div>
      <h3>Misses</h3>
      <p>Uniformly <b>&ldquo;in a common mode&rdquo;</b>, NLL at the 1&ndash;40th percentile, channels
      only &minus;1 to &minus;3&sigma;. Genuinely in-distribution subtle faults &mdash; the hard tail
      the snapshot detector cannot see, where a trajectory/context branch would have to earn its keep.</p>
    </div>
  </div>
</section>

<section>
  <h2>Next</h2>
  <ol>
    <li><b>Report on the HARD subset</b> (easy anomalies removed) &mdash; the honest, discriminative
      benchmark, and where the thesis must win.</li>
    <li><b>Mode-conditional thresholds</b> (C4/TFAR) on real data &mdash; cut the worst-mode false alarms.</li>
    <li><b>Attack the subtle in-mode misses</b> on cleaned WADI (now 93% in-common) &mdash; the natural
      home for the trajectory/context branch on real data.</li>
    <li><b>SWaT</b> is staged but blocked on the Dec-2015 labelled files (108&nbsp;GB zip on SharePoint;
      or the iTrust Drive / Kaggle mirror).</li>
  </ol>
</section>

<footer>Generated from <span class="mono">eda_real.py</span> &middot; SKAB / HAI / WADI &middot; window-level,
normal-standardised, WADI clipped &plusmn;10&sigma;.</footer>
</div>

<style>
  .wrap {{ --paper:#f5f6f3; --ink:#19212a; --sub:#5b6672; --line:#d3dad9; --teal:#0d7d74;
    --amber:#a9730a; --crim:#a5324a; max-width:1060px; margin:0 auto; padding:56px 28px 40px;
    color:var(--ink); background:var(--paper);
    font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; line-height:1.6; }}
  .wrap * {{ box-sizing:border-box; }}
  .mono {{ font-family:ui-monospace,"Cascadia Code","SF Mono",Menlo,monospace; font-size:.92em; }}
  .num {{ font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap; }}
  .eyebrow {{ text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:600;
    color:var(--teal); }}
  h1 {{ font-size:2.5rem; line-height:1.08; margin:.15em 0 .3em; letter-spacing:-.02em;
    text-wrap:balance; font-weight:800; }}
  .lede {{ font-size:1.12rem; color:var(--sub); max-width:62ch; margin:0 0 1.1em; }}
  h2 {{ font-size:1.32rem; margin:0 0 .5em; letter-spacing:-.01em; }}
  h3 {{ font-size:1rem; margin:0 0 .3em; }}
  section {{ padding:26px 0; border-top:1px solid var(--line); }}
  p {{ margin:.4em 0 .8em; max-width:70ch; }}
  .note {{ font-size:.94rem; color:var(--sub); }}
  .chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
  .chip {{ font-size:.76rem; font-weight:600; padding:5px 11px; border-radius:999px; letter-spacing:.01em;
    border:1px solid transparent; }}
  .chip.teal {{ background:#0d7d7414; color:var(--teal); border-color:#0d7d7433; }}
  .chip.amber {{ background:#a9730a14; color:var(--amber); border-color:#a9730a33; }}
  .chip.crimson {{ background:#a5324a14; color:var(--crim); border-color:#a5324a33; }}
  .chip.ink {{ background:#19212a0d; color:var(--ink); border-color:#19212a22; }}
  .cards, .cols {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }}
  .cols {{ grid-template-columns:repeat(2,1fr); }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
  .card h3 {{ color:var(--teal); }} .card p {{ font-size:.92rem; color:var(--sub); margin:0; }}
  table {{ width:100%; border-collapse:collapse; margin:.4em 0 .6em; font-size:.93rem; }}
  .scroll {{ overflow-x:auto; }}
  th, td {{ text-align:left; padding:8px 12px; border-bottom:1px solid var(--line); }}
  thead th {{ font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; color:var(--sub);
    font-weight:600; border-bottom:1.5px solid #c2cbca; }}
  tbody tr:hover {{ background:#0d7d7408; }}
  td.warn {{ color:var(--amber); font-weight:600; }}
  td.crit {{ color:var(--crim); font-weight:700; }}
  td.teal {{ color:var(--teal); font-weight:700; }}
  figure {{ margin:18px 0; }}
  figure img {{ width:100%; height:auto; border:1px solid var(--line); border-radius:8px; background:#fff; }}
  figcaption {{ font-size:.86rem; color:var(--sub); margin-top:7px; max-width:none; }}
  .callout {{ background:#0d7d7409; border:1px solid #0d7d7433; border-radius:12px; padding:22px 24px;
    margin-top:0; }}
  .callout ul {{ margin:.3em 0 .8em; padding-left:1.1em; }} .callout li {{ margin:.3em 0; }}
  ol {{ padding-left:1.2em; }} ol li {{ margin:.35em 0; }}
  footer {{ padding-top:22px; border-top:1px solid var(--line); font-size:.82rem; color:var(--sub); }}
  @media (max-width:720px) {{ .cards, .cols {{ grid-template-columns:1fr; }} h1 {{ font-size:2rem; }} }}
</style>"""

open("eda_report.html", "w", encoding="utf-8").write(HTML)
print("wrote eda_report.html", len(HTML), "bytes")
