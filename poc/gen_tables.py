"""Emit Table 3 (All/Easy/Difficult AUROC+F1) and Table 4 (double-hard) tbody HTML from the
construct-matched headline artifact, with column-max bolding. Paste into paper.html."""
import json
H = json.load(open("_diagnostics/ensemble_head.json"))
DS = ["WADI", "HAI", "SWaT"]
# display order + labels + group
ROWS = [("trivial max|z|", "trivial max|z|", "Baseline"), ("IF", "Isolation Forest", "Baseline"),
        ("AE", "AutoEncoder", "Baseline"), ("linres", "LinRes (one-hot)", "Baseline"),
        ("USAD", "USAD", "SOTA"), ("TranAD", "TranAD", "SOTA"), ("GDN", "GDN", "SOTA"),
        ("LatAD", "LatAD (global density)", "Ours"),
        ("HCcoh+LatAD", "<b>LatAD (regime-community)</b>", "Ours")]
ANOM = {"WADI": "56 = 37 easy + 19 difficult", "HAI": "652 = 485 easy + 167 difficult",
        "SWaT": "182 = 144 easy + 38 difficult"}


def fmt_au(x):
    if x is None: return "&mdash;"
    return f'{x["auroc"]:.3f}' + (f'&plusmn;{x["sd"]:.3f}' if x["sd"] > 0 else '')


def col_max(ds, sub):
    v = [H[ds]["results"][k][sub]["auroc"] for k, _, _ in ROWS
         if k in H[ds]["results"] and H[ds]["results"][k].get(sub)]
    return max(v) if v else None


print("<!-- ===== TABLE 3 tbody ===== -->")
for ds in DS:
    print(f'<tr><td class="l" colspan="8" style="background:#eef3f8;text-align:left"><b>{ds}</b> '
          f'&nbsp;<span class="small">(anom {ANOM[ds]})</span></td></tr>')
    res = H[ds]["results"]
    mx = {s: col_max(ds, s) for s in ["All", "Easy", "Difficult"]}
    for key, label, grp in ROWS:
        if key not in res: continue
        cells = ""
        for sub in ["All", "Easy", "Difficult"]:
            x = res[key].get(sub)
            au = fmt_au(x); f1 = "&mdash;" if x is None else f'{x["f1"]:.3f}'
            cls = ' class="win"' if (x and abs(x["auroc"] - mx[sub]) < 1e-9) else ''
            cells += f'<td{cls}>{au}</td><td>{f1}</td>'
        print(f'<tr><td class="l">{label}</td><td>{grp}</td>{cells}</tr>')

print("\n<!-- ===== TABLE 4 tbody (double-hard AUROC) ===== -->")
mx4 = {ds: max(H[ds]["results"][k]["DoubleHard"]["auroc"] for k, _, _ in ROWS
               if k in H[ds]["results"] and H[ds]["results"][k].get("DoubleHard")) for ds in DS}
for key, label, grp in ROWS:
    if key == "GDN": continue
    cells = ""
    ok = True
    for ds in DS:
        x = H[ds]["results"].get(key, {}).get("DoubleHard")
        if x is None: cells += "<td>&mdash;</td>"; continue
        cls = ' class="win"' if abs(x["auroc"] - mx4[ds]) < 1e-9 else ''
        cells += f'<td{cls}>{x["auroc"]:.3f}' + (f'&plusmn;{x["sd"]:.3f}' if x["sd"] > 0 else '') + '</td>'
    print(f'<tr><td class="l">{label}</td>{cells}</tr>')
