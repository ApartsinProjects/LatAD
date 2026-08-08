"""Emit the MDPI-style §6 comparison-table HTML body from comparison_table.json.
Columns: All/Easy/Difficult x (AUROC, F1). Multi-seed AUROC shown mean+/-std. Winning AUROC per
(dataset,subset) gets class="win". Writes _diagnostics/table_body.html."""
import json, os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics")
TAB = json.load(open(f"{OUT}/comparison_table.json"))
GROUP = {"trivial max|z|": "Baseline", "Isolation Forest": "Baseline", "AutoEncoder": "Baseline",
         "LinRes (one-hot)": "Baseline", "USAD": "SOTA", "TranAD": "SOTA",
         "VaDE-hard+resid (ours)": "Ours"}
SUBS = ["all", "easy", "difficult"]


def au_str(c):
    s = f"{c['auroc']:.3f}"
    return s + (f"&plusmn;{c['auroc_sd']:.3f}" if "auroc_sd" in c else "")


lines = []
for ds in ["WADI", "HAI", "SWaT"]:
    n = TAB[ds]["n"]; rows = TAB[ds]["rows"]
    lines.append(f'<tr><td class="l" colspan="8" style="background:#eef3f8;text-align:left"><b>{ds}</b> '
                 f'&nbsp;<span class="small">(anom {n["anom"]} = {n["easy"]} easy + {n["difficult"]} difficult)</span></td></tr>')
    # winning AUROC per subset
    win = {}
    for sub in SUBS:
        best = max(rows.values(), key=lambda c: (c[sub]["auroc"] if c[sub] else -1))
        win[sub] = best[sub]["auroc"]
    for label, c in rows.items():
        name = f"<b>{label}</b>" if GROUP[label] == "Ours" else label
        cells = ""
        for sub in SUBS:
            cc = c[sub]
            cls = ' class="win"' if abs(cc["auroc"] - win[sub]) < 1e-9 else ""
            cells += f'<td{cls}>{au_str(cc)}</td><td>{cc["f1"]:.3f}</td>'
        lines.append(f'<tr><td class="l">{name}</td><td>{GROUP[label]}</td>{cells}</tr>')
open(f"{OUT}/table_body.html", "w").write("\n".join(lines))
print("\n".join(lines))
print(f"\n-> {OUT}/table_body.html")
