"""Render table.json -> a standalone HTML results table. 3 dataset super-columns, each split
easy/hard/all, each cell = AUC / F1 / FPR with the best value in that (dataset,subset,metric)
column bolded (AUC/F1 = max, FPR = min). Method rows grouped Baseline / SOTA / Ours.
"""
import json, sys

SUBS = [("easy", "Easy"), ("hard", "Difficult"), ("all", "All")]
DATASETS = ["WADI", "HAI", "SKAB"]
# rows shown (parametric baselines only -- LOF dropped as non-parametric; SOTA; VaDE-family).
# VaDE Pipeline dropped: a triv-router is a two-channel decision system, not a single-score
# detector, so its merged AUROC/F1 misrepresents it (see EXPERIMENT_LOG 2.35).
ORDER = ["trivial max|z|", "IsolationForest", "AutoEncoder", "USAD", "TranAD", "GDN",
         "VaDE", "VaDE-hard+resid(auto)"]

data = json.load(open(sys.argv[1]))
SHOWN = [m for m in ORDER if m in data["WADI"]["methods"]]

# best value per (dataset, subset, metric) computed over SHOWN rows only: 0,1 = max (AUC,F1); 2 = min (FPR)
best = {}
# best is computed on the DISPLAYED (2-decimal) value, so ALL cells tied at the best shown
# value are bolded (not just the one with the highest full-precision value).
for ds in DATASETS:
    M = data[ds]["methods"]
    for sub, _ in SUBS:
        for mi in range(3):
            vals = [round(M[m][sub][mi], 2) for m in SHOWN if M[m][sub][mi] is not None]
            if vals:
                best[(ds, sub, mi)] = (min(vals) if mi == 2 else max(vals))

def cell(ds, m, sub):
    v = data[ds]["methods"][m][sub]
    if v[0] is None:
        return '<span class="na">n/a</span>'
    out = []
    labels = ["au", "f1", "fpr"]
    for mi in range(3):
        x = v[mi]
        s = f"{x:.2f}"
        isbest = best.get((ds, sub, mi)) is not None and abs(round(x, 2) - best[(ds, sub, mi)]) < 1e-9
        out.append(f'<b class="{labels[mi]}">{s}</b>' if isbest else f'<span class="{labels[mi]}">{s}</span>')
    return "<span class='sep'>/</span>".join(out)

rows = []
last_group = None
for m in ORDER:
    if m not in data["WADI"]["methods"]:
        continue
    grp = data["WADI"]["methods"][m]["group"]
    if grp != last_group:
        rows.append(f'<tr class="grp"><td colspan="10">{grp}</td></tr>')
        last_group = grp
    tds = [f'<td class="method g-{grp}">{m}</td>']
    for ds in DATASETS:
        for sub, _ in SUBS:
            tds.append(f'<td class="num">{cell(ds, m, sub)}</td>')
    rows.append("<tr>" + "".join(tds) + "</tr>")

# counts header
def counts_cell(ds):
    c = data[ds]["counts"]
    parts = {"easy": c["n_easy"], "hard": c["n_hard"], "all": c["n_anom"]}
    nnorm = {"easy": c.get("nn_easy"), "hard": c.get("nn_hard"), "all": c.get("nn_all")}
    frac = {k: (100.0 * parts[k] / c["n_anom"]) for k in parts}
    out = []
    for sub, _ in SUBS:
        out.append(f'<td class="cnt"><b>{parts[sub]}</b> <span class="dim">({frac[sub]:.0f}%, '
                   f'#norm={nnorm[sub]})</span></td>')
    return "".join(out)

# single composition row across all datasets; per-dataset normal/win moves into the super-header
count_rows = ('<tr class="grp"><td colspan="10">Composition &mdash; positive windows per subset '
              '(% of that dataset\'s anomalies); normals are fixed per dataset (in header) and are the '
              'F1/FPR denominator</td></tr>'
              '<tr class="countrow"><td class="method">Positives <span class="dim">(% of anom)</span></td>'
              + "".join(counts_cell(ds) for ds in DATASETS) + '</tr>')

sup = "".join(f'<th colspan="3" class="ds">{ds}<div class="dsnote">{data[ds]["counts"]["n_normal"]} normal'
              f' &middot; {data[ds]["counts"]["n_test"]} win</div></th>' for ds in DATASETS)
sub_th = ""
for ds in DATASETS:
    for _, label in SUBS:
        cls = "difficult" if label == "Difficult" else ("easy" if label == "Easy" else "allc")
        sub_th += f'<th class="sub {cls}">{label}</th>'

html = f"""<title>LatAD — Results</title>
<style>
  :root {{
    --ink:#12181d; --dim:#5c6b74; --line:#dbe3e7; --ground:#f7f9fa; --card:#ffffff;
    --teal:#0f7d84; --amber:#9a6a12; --grey:#657079;
    --easy:#eef4f3; --diff:#fdf3ec; --best:#0b5f65;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; }}
  .wrap {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    color:var(--ink); background:var(--ground); padding:28px 22px 40px; }}
  h1 {{ font-size:19px; letter-spacing:-0.01em; margin:0 0 3px; }}
  .lede {{ color:var(--dim); font-size:12.5px; max-width:70ch; line-height:1.55; margin:0 0 18px; }}
  .scroll {{ overflow-x:auto; background:var(--card); border:1px solid var(--line);
    border-radius:10px; box-shadow:0 1px 2px rgba(16,40,50,.04); }}
  table {{ border-collapse:collapse; width:100%; font-size:12px;
    font-variant-numeric:tabular-nums; }}
  th,td {{ padding:7px 9px; text-align:center; white-space:nowrap; }}
  thead .ds {{ font-size:12.5px; letter-spacing:.06em; text-transform:uppercase;
    border-bottom:2px solid var(--teal); color:var(--teal); }}
  thead .ds:not(:last-child), td:nth-child(4n+1):not(.method) {{ }}
  th.sub {{ font-size:10.5px; text-transform:uppercase; letter-spacing:.05em; color:var(--dim);
    border-bottom:1px solid var(--line); font-weight:600; }}
  th.sub.easy {{ background:var(--easy); }}
  th.sub.difficult {{ background:var(--diff); }}
  .metrickey {{ font-weight:400; font-size:10px; color:var(--dim); }}
  td.method {{ text-align:left; font-weight:600; position:sticky; left:0; background:var(--card);
    border-right:1px solid var(--line); min-width:150px; }}
  td.num {{ border-bottom:1px solid #eef2f4; }}
  td.num:nth-child(3n+1) {{ }}
  tr.grp td {{ text-align:left; font-size:10.5px; letter-spacing:.08em; text-transform:uppercase;
    font-weight:700; padding:9px 9px 4px; color:var(--dim); background:var(--ground); }}
  .g-Ours {{ border-left:3px solid var(--teal); }}
  .g-SOTA {{ border-left:3px solid var(--amber); }}
  .g-Baseline {{ border-left:3px solid var(--grey); }}
  b.au {{ color:#c0392b; font-weight:800; }}
  b.f1,b.fpr {{ color:var(--best); font-weight:750; }}
  .dsnote {{ font-size:9.5px; font-weight:400; letter-spacing:0; text-transform:none; color:var(--dim); margin-top:2px; }}
  .sep {{ color:#b8c4ca; margin:0 3px; font-size:10px; }}
  .num span {{ color:#2b3840; }}
  .na {{ color:#b0bcc2; font-style:italic; }}
  tr.countrow td {{ background:#fbfcfd; border-bottom:1px solid var(--line); font-size:11px;
    padding:6px 9px; }}
  td.cnt {{ line-height:1.35; }}
  .dim {{ color:var(--dim); font-size:10px; }}
  .legend {{ display:flex; gap:18px; flex-wrap:wrap; margin:14px 2px 0; font-size:11.5px; color:var(--dim); }}
  .legend b {{ color:var(--best); }}
  .chip {{ display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px;
    vertical-align:middle; }}
  .notes {{ color:var(--dim); font-size:11px; line-height:1.6; max-width:80ch; margin:12px 2px 0; }}
  .metrics-row th {{ font-size:9.5px; color:var(--dim); letter-spacing:.04em;
    border-bottom:1px solid var(--line); font-weight:500; text-transform:none; }}
</style>
<div class="wrap">
  <h1>LatAD — anomaly detection on 3 CPS datasets</h1>
  <p class="lede">Each cell is <b style="color:#c0392b">AUC</b> / <b style="color:var(--best)">F1</b> /
  <b style="color:var(--best)">FPR</b>. Best per column in bold (AUC in red). <b>Easy</b> = a trivial per-channel
  level threshold already flags it; <b>Difficult</b> = it doesn't (correlation/dynamics fault). All fits use
  train-normal only. Metrics are made <em>comparable across subsets</em>: <b>AUC</b> is prevalence-independent;
  <b>F1</b> is prevalence-matched (normals subsampled so each subset has the same anomaly:normal ratio as All,
  mean of 5 seeds); <b>FPR</b> is at one fixed operating point (best-F1 threshold on All) so it's identical across
  subsets.</p>
  <div class="scroll">
    <table>
      <thead>
        <tr><th rowspan="2" class="method" style="text-align:left;vertical-align:bottom;padding-bottom:9px">Method</th>{sup}</tr>
        <tr>{sub_th}</tr>
        <tr class="metrics-row"><th></th>{''.join('<th>AUC/F1/FPR</th>' for _ in range(9))}</tr>
      </thead>
      <tbody>
        {count_rows}
        {''.join(rows)}
      </tbody>
    </table>
  </div>
  <div class="legend">
    <span><span class="chip" style="background:var(--grey)"></span>Baseline</span>
    <span><span class="chip" style="background:var(--amber)"></span>SOTA (USAD/TranAD, TranAD harness, raw point-F1)</span>
    <span><span class="chip" style="background:var(--teal)"></span>Ours</span>
    <span><b>bold</b> = best in column</span>
  </div>
  <p class="notes">
    <b>Baselines</b> are parametric only (LOF dropped as instance-based/non-parametric).
    &nbsp;<b>VaDE</b> = joint VAE+GMM, reconstruction + latent NLL. &nbsp;<b>VaDE-hard+resid(auto)</b> = drop
    reconstruction; score the joint latent with density + closest-mode NLL, plus a responsibility-weighted whitened
    residual gated ON only when it generalizes to held-out normal (here: HAI). &nbsp;<b>VaDE Pipeline</b> = route
    each window by the trivial detector &mdash; easy&rarr;VaDE, difficult&rarr;VaDE-hard+resid &mdash; each branch
    percentile-calibrated on train-normal. GDN omitted (device patch fixed; A100 re-run pending).
  </p>
</div>
"""
open(sys.argv[2], "w", encoding="utf-8").write(html)
print("wrote", sys.argv[2])
