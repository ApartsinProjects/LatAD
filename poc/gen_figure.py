"""Generate Figure 1 (difficult-subset AUROC bars, 6 methods x 3 datasets) from the
construct-matched artifact, with correct geometry + whiskers. Emits the <svg> block."""
import json
H = json.load(open("_diagnostics/ensemble_head.json"))
METH = [("IF", "Isolation Forest", "#9fb8cf"), ("AE", "AutoEncoder", "#c9c9c9"),
        ("linres", "LinRes (one-hot)", "#8a9a5b"), ("USAD", "USAD", "#e3b7a0"),
        ("TranAD", "TranAD", "#d59a7a"), ("HCcoh+LatAD", "LatAD (regime-community, ours)", "#1f4e79")]
DS = [("WADI", 165), ("HAI", 375), ("SWaT", 585)]   # group centers
Y0, Y1, V0, V1 = 250.0, 60.0, 0.3, 1.0              # axis: y=250 -> 0.3, y=60 -> 1.0
def y(v): return Y0 - (v - V0) * (Y0 - Y1) / (V1 - V0)
BW, PITCH = 16, 19

svg = ['<svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica Neue,Arial,sans-serif" font-size="12">']
svg.append('  <text x="360" y="20" text-anchor="middle" font-size="13" font-weight="bold">Difficult-subset AUROC by method (higher is better)</text>')
svg.append('  <line x1="70" y1="250" x2="700" y2="250" stroke="#888"/>')
svg.append('  <line x1="70" y1="60" x2="70" y2="250" stroke="#888"/>')
# y ticks 0.3..1.0
ticks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
svg.append('  <g fill="#666" text-anchor="end">')
svg.append('    ' + ''.join(f'<text x="64" y="{y(t)+4:.0f}">{t:.1f}</text>' for t in ticks))
svg.append('  </g>')
svg.append('  <g stroke="#eee">' + ''.join(f'<line x1="70" y1="{y(t):.0f}" x2="700" y2="{y(t):.0f}"/>' for t in ticks[:-1]) + '</line>'.replace('</line>', '') + '</g>')
# bars + whiskers per dataset
whisk = []
for ds, cx in DS:
    x0 = cx - 3 * PITCH + 2
    svg.append('  <g>')
    svg.append(f'    <text x="{cx}" y="270" text-anchor="middle" font-weight="bold">{ds}</text>')
    for i, (key, _, col) in enumerate(METH):
        r = H[ds]["results"][key]["Difficult"]; v = r["auroc"]; sd = r["sd"]
        bx = x0 + i * PITCH; ytop = y(v); h = Y0 - ytop
        svg.append(f'    <rect x="{bx}" y="{ytop:.1f}" width="{BW}" height="{h:.1f}" fill="{col}"/>')
        if sd > 0:
            cxb = bx + BW / 2; dtop = (Y0 - Y1) / (V1 - V0) * sd
            yhi, ylo = ytop - dtop, ytop + dtop
            whisk.append(f'    <line x1="{cxb:.0f}" y1="{yhi:.1f}" x2="{cxb:.0f}" y2="{ylo:.1f}"/>'
                         f'<line x1="{cxb-4:.0f}" y1="{yhi:.1f}" x2="{cxb+4:.0f}" y2="{yhi:.1f}"/>'
                         f'<line x1="{cxb-4:.0f}" y1="{ylo:.1f}" x2="{cxb+4:.0f}" y2="{ylo:.1f}"/>')
    svg.append('  </g>')
svg.append('  <g stroke="#333" stroke-width="1">')
svg.extend(whisk)
svg.append('  </g>')
# legend: 2 rows x 3 cols, compact so it fits 720px
labs = ["Isolation Forest", "AutoEncoder", "LinRes (one-hot)", "USAD", "TranAD", "LatAD (ours)"]
cols = [m[2] for m in METH]
colx = [80, 240, 380]
svg.append('  <g font-size="11">')
for i in range(6):
    xx = colx[i % 3]; yy = 33 + (i // 3) * 15
    svg.append(f'    <rect x="{xx}" y="{yy-9}" width="12" height="12" fill="{cols[i]}"/><text x="{xx+16}" y="{yy}">{labs[i]}</text>')
svg.append('  </g>')
svg.append('</svg>')
print('\n'.join(svg))
