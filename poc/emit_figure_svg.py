"""Emit Figure 1 (difficult-subset AUROC by method) SVG from comparison_table.json.
Methods: IsolationForest, AutoEncoder, LinRes, USAD, TranAD, VaDE-hard+resid(ours). Datasets
WADI/HAI/SWaT. y-axis 0.3-1.0. Writes _diagnostics/figure1.svg (full <figure> block)."""
import json, os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
TAB = json.load(open(f"{OUT}/comparison_table.json"))
M = [("Isolation Forest", "IsolationForest", "#9fb8cf"), ("AutoEncoder", "AutoEncoder", "#c9c9c9"),
     ("LinRes (one-hot)", "LinRes", "#8a9a5b"), ("USAD", "USAD", "#e3b7a0"),
     ("TranAD", "TranAD", "#d59a7a"), ("VaDE-hard+resid (ours)", "VaDE-hard+resid (ours)", "#1f4e79")]
BASE, TOP, VMIN, VMAX = 250, 60, 0.30, 1.00
BW, GAP = 16, 3
XSTART = {"WADI": 108, "HAI": 318, "SWaT": 528}


def yv(v): return BASE - (v - VMIN) / (VMAX - VMIN) * (BASE - TOP)


s = ['<figure>', '<svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" '
     'font-family="Helvetica Neue,Arial,sans-serif" font-size="12">',
     '  <text x="360" y="20" text-anchor="middle" font-size="13" font-weight="bold">'
     'Difficult-subset AUROC by method (higher is better)</text>',
     f'  <line x1="70" y1="{BASE}" x2="700" y2="{BASE}" stroke="#888"/>',
     f'  <line x1="70" y1="{TOP}" x2="70" y2="{BASE}" stroke="#888"/>']
ticks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
s.append('  <g fill="#666" text-anchor="end">')
s.append("    " + "".join(f'<text x="64" y="{yv(t)+3:.0f}">{t:.1f}</text>' for t in ticks))
s.append('  </g>')
s.append('  <g stroke="#eee">' + "".join(f'<line x1="70" y1="{yv(t):.0f}" x2="700" y2="{yv(t):.0f}"/>' for t in ticks[1:]) + '</g>')
for ds, x0 in XSTART.items():
    rows = TAB[ds]["rows"]
    s.append('  <g>')
    s.append(f'    <text x="{x0 + 3*(BW+GAP)}" y="270" text-anchor="middle" font-weight="bold">{ds}</text>')
    bars = []
    for i, (key, _, col) in enumerate(M):
        v = rows[key]["difficult"]["auroc"]; x = x0 + i * (BW + GAP); y = yv(v)
        bars.append(f'<rect x="{x}" y="{y:.1f}" width="{BW}" height="{BASE - y:.1f}" fill="{col}"/>')
    s.append("    " + "    ".join(bars))
    s.append('  </g>')
# legend
s.append('  <g font-size="11">')
lx = 80
for label, _, col in M:
    s.append(f'    <rect x="{lx}" y="34" width="12" height="12" fill="{col}"/><text x="{lx+16}" y="44">{label}</text>')
    lx += 22 + int(len(label) * 6.0)
s.append('  </g>')
s.append('</svg>')
s.append('<figcaption><strong>Figure 1.</strong> Difficult-subset AUROC by method (five-seed mean for the learned '
         'detectors). The detector leads the difficult column on every dataset. The margin over the deep detectors '
         '(USAD, TranAD) is largest on HAI (0.811 vs 0.45&ndash;0.50) and WADI (0.690 vs 0.30&ndash;0.33), the two '
         'benchmarks whose difficult faults break sensor correlations; on SWaT the difficult subset is largely '
         'reconstructable, so all methods score highly and the margin is small.</figcaption>')
s.append('</figure>')
open(f"{OUT}/figure1.svg", "w").write("\n".join(s))
print("\n".join(s))
