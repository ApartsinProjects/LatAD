# -*- coding: utf-8 -*-
"""Generate the three figures for synthetic_data.html and splice them in."""
import numpy as np, base64, io, os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg

DOC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_data.html")
W, H = 640, 400

# ---------- density rendering ----------
def mode_field(shape, cx, cy, sx, sy, deg, peak, curv):
    Hh, Ww = shape
    ys, xs = np.mgrid[0:Hh, 0:Ww]
    dx, dy = xs - cx, ys - cy
    th = math.radians(deg); ct, st = math.cos(th), math.sin(th)
    u = ct * dx + st * dy
    v = -st * dx + ct * dy
    v = v - curv * (u * u) / sx          # parabolic banana bend
    r2 = (u / sx) ** 2 + (v / sy) ** 2
    d = peak * np.exp(-0.5 * r2)
    d[r2 > 4.0] = 0.0                     # hard clip at 2 sigma -> hard rim
    return d

def render(modes, shape):
    f = np.zeros(shape)
    for m in modes:
        f = np.maximum(f, mode_field(shape, *m))
    mx = f.max()
    return f / mx if mx > 0 else f

def png_data_uri(field):
    buf = io.BytesIO()
    mpimg.imsave(buf, field, cmap='inferno', vmin=0, vmax=1, format='png', origin='upper')
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# ---------- layout: non-overlapping modes (cores separated), heavy tail ----------
rng = np.random.default_rng(11)
# (sx, sy, peak, curv) specs, big first
specs = ([(32, 12, 1.00, 0.0), (31, 11, 0.93, 0.16)]              # big (one curved)
         + [(22, 9, 0.70, c) for c in (0.0, 0.18, 0.0, 0.14)]      # medium (two curved)
         + [(15, 6, 0.50, c) for c in (0.0, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]  # small (one curved)
         + [(11, 5, 0.35, 0.0)] * 6)                               # tiny
def core_sep_ok(cand, placed, f):
    cx, cy, sx = cand[0], cand[1], cand[2]
    for p in placed:
        if math.hypot(cx - p[0], cy - p[1]) < f * (sx + p[2]):
            return False
    return True
placed = []
for (sx, sy, peak, curv) in specs:
    for _ in range(4000):
        cx = rng.uniform(48, W - 48); cy = rng.uniform(48, H - 48)
        deg = rng.uniform(0, 180)
        cand = (cx, cy, sx, sy, deg, peak, curv)
        if core_sep_ok(cand, placed, 1.18):
            placed.append(cand); break
# two close-but-not-overlapping clusters (append members near an anchor, f just >1)
def add_cluster(ax, ay, n, sx, sy, peak):
    made = []
    for k in range(n):
        for _ in range(4000):
            ang = rng.uniform(0, 2 * math.pi); rad = rng.uniform(sx * 1.05, sx * 1.35)
            cx, cy = ax + rad * math.cos(ang), ay + rad * math.sin(ang)
            cand = (cx, cy, sx, sy, rng.uniform(0, 180), peak, 0.0)
            if core_sep_ok(cand, placed + made, 1.05) and 40 < cx < W - 40 and 40 < cy < H - 40:
                made.append(cand); break
    return made
cl1 = add_cluster(500, 300, 3, 17, 7, 0.72)     # bottom-right trio
cl2 = add_cluster(150, 250, 2, 17, 7, 0.60)     # left pair
modes = placed + cl1 + cl2
# report separation health
mn = 9e9
for i in range(len(modes)):
    for j in range(i + 1, len(modes)):
        a, b = modes[i], modes[j]
        mn = min(mn, math.hypot(a[0]-b[0], a[1]-b[1]) / (a[2]+b[2]))
print("modes:", len(modes), "min core-sep ratio:", round(mn, 3))

field = render(modes, (H, W))
uri = png_data_uri(field)

# pick trajectory nodes from actual placed modes
big = sorted(placed, key=lambda m: -m[5])
n0, end = big[0], big[2]
mid = ((n0[0] + end[0]) / 2, (n0[1] + end[1]) / 2)
n1 = min([m for m in placed if m not in (n0, end)], key=lambda m: math.hypot(m[0]-mid[0], m[1]-mid[1]))
far = max(placed, key=lambda m: math.hypot(m[0]-end[0], m[1]-end[1]))
def pt(m): return f"{m[0]:.0f},{m[1]:.0f}"
valid_pts = f"{pt(n0)} {pt(n1)} {end[0]-8:.0f},{end[1]-4:.0f}"
forb_pts = f"{pt(far)} {end[0]-6:.0f},{end[1]+10:.0f}"
def clampx(x): return max(94, min(W - 94, x))
lab_n0 = (clampx(n0[0]), max(16, n0[1] - 2 * n0[3] - 8))
lab_end = (clampx(end[0]), max(16, end[1] - 2 * end[3] - 8))

# cluster overlay rings + pocket dots
def ring(m, col="#7fe3d8"):
    return (f'<ellipse cx="{m[0]:.0f}" cy="{m[1]:.0f}" rx="{2*m[2]:.0f}" ry="{2*m[3]:.0f}" '
            f'fill="none" stroke="{col}" stroke-width="1.1" transform="rotate({m[4]:.0f} {m[0]:.0f} {m[1]:.0f})"/>')
c1x = sum(m[0] for m in cl1)/len(cl1); c1y = sum(m[1] for m in cl1)/len(cl1)
c2x = sum(m[0] for m in cl2)/len(cl2); c2y = sum(m[1] for m in cl2)/len(cl2)
rings = "\n  ".join(ring(m) for m in cl1 + cl2)

heat_svg = f'''<figure style="margin:18px 0;text-align:center">
<svg viewBox="0 0 {W} {H}" width="100%" xmlns:xlink="http://www.w3.org/1999/xlink" style="max-width:{W}px;border-radius:8px" font-family="Helvetica Neue,Arial,sans-serif">
  <defs>
    <marker id="arrGood" markerWidth="9" markerHeight="9" refX="6.5" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#2ec4b6"/></marker>
    <marker id="arrBad" markerWidth="9" markerHeight="9" refX="6.5" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#ff5a6a"/></marker>
  </defs>
  <rect x="0" y="0" width="{W}" height="{H}" fill="#0d1226"/>
  <image href="{uri}" xlink:href="{uri}" x="0" y="0" width="{W}" height="{H}" preserveAspectRatio="none"/>
  {rings}
  <circle cx="{c1x:.0f}" cy="{c1y:.0f}" r="3.3" fill="#ff5a6a"/>
  <circle cx="{c2x:.0f}" cy="{c2y:.0f}" r="3.3" fill="#ff5a6a"/>
  <polyline fill="none" stroke="#2ec4b6" stroke-width="2.2" marker-end="url(#arrGood)" points="{valid_pts}"/>
  <polyline fill="none" stroke="#ff5a6a" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#arrBad)" points="{forb_pts}"/>
  <text x="16" y="26" fill="#e8eef4" font-size="13" font-weight="600">Normal latent density (2D)</text>
  <text x="{lab_n0[0]:.0f}" y="{lab_n0[1]:.0f}" fill="#cdd9e5" font-size="11" text-anchor="middle">common, elongated (A1, A5)</text>
  <text x="{lab_end[0]:.0f}" y="{lab_end[1]:.0f}" fill="#e8eef4" font-size="11" text-anchor="middle">same state, two histories (A10)</text>
  <text x="2" y="392" fill="#cdd9e5" font-size="11" text-anchor="start">valid path (teal) vs forbidden jump (red) &#183; curved &amp; close modes below</text>
  <line x1="{c1x:.0f}" y1="{c1y:.0f}" x2="{c1x+40:.0f}" y2="{c1y+52:.0f}" stroke="#cdd9e5" stroke-width="0.8"/>
  <text x="636" y="368" fill="#cdd9e5" font-size="11" text-anchor="end">close-mode clusters &#8594; thin anomalous pockets (A3)</text>
</svg>
<figcaption style="font-size:.82rem;color:#666;font-family:'Helvetica Neue',Arial,sans-serif"><strong>Figure 1.</strong>
A 2D slice of the normal latent (a rendered density heatmap): each blob is one operating mode, none overlapping
though some come close. Many modes (A1; the real $K$ is far larger, A2); each a bounded blob with a hard rim (A3),
density fading centre&rarr;rim (A4), elongated along one dominant direction (A5). Several modes are <em>curved /
irregular</em> ($\\kappa_k>0$, Step 2). Sizes and brightness follow a heavy tail (A6). No labels are shown (A7). A
<span style="color:#2ec4b6">valid trajectory</span> hops allowed transitions while a
<span style="color:#c0392b">forbidden jump</span> reaches the same mode illegally (A10); two clusters of close
envelopes leave thin anomalous pockets (A3, red dots).</figcaption>
</figure>'''

# ---------- Figure 2: curvature knob triptych ----------
cw, chh = 600, 200
kmodes_specs = [(0.0, "&#954;=0  (flat)"), (0.16, "&#954;=0.35  (curved)"), (0.30, "&#954;=0.7  (strongly curved)")]
cx3 = [110, 300, 490]
kfield = render([(cx3[i], 96, 46, 17, -18, 1.0, s[0]) for i, s in enumerate(kmodes_specs)], (chh, cw))
kuri = png_data_uri(kfield)
klabels = "\n  ".join(f'<text x="{cx3[i]}" y="185" fill="#333" font-size="12" text-anchor="middle" font-weight="600">{s[1]}</text>'
                      for i, s in enumerate(kmodes_specs))
curv_svg = f'''<figure style="margin:14px 0;text-align:center">
<svg viewBox="0 0 {cw} {chh}" width="100%" xmlns:xlink="http://www.w3.org/1999/xlink" style="max-width:{cw}px;border-radius:8px" font-family="Helvetica Neue,Arial,sans-serif">
  <rect x="0" y="0" width="{cw}" height="{chh}" fill="#0d1226"/>
  <image href="{kuri}" xlink:href="{kuri}" x="0" y="0" width="{cw}" height="{chh}" preserveAspectRatio="none"/>
  <rect x="0" y="164" width="{cw}" height="36" fill="#fcfcfb"/>
  {klabels}
</svg>
<figcaption style="font-size:.82rem;color:#666;font-family:'Helvetica Neue',Arial,sans-serif"><strong>Figure 2.</strong>
The curvature knob $\\kappa_k$ on one mode's 2D density. $\\kappa_k=0$ is the flat clipped low-rank Gaussian (a
straight, oriented ellipse, fit exactly by PCA); as $\\kappa_k$ grows the mode bows into a curved, banana-like
manifold whose between-modes region falls off every mode (the E2 case). Curvature is a per-mode, mode-dependent
axis.</figcaption>
</figure>'''

# ---------- Figure 3: multivariate time series ----------
N = 480; x0, x1 = 110, 585
xs = np.linspace(x0, x1, N)
tr = [230, 350, 470]                       # transition x positions
seg_id = np.searchsorted(tr, xs)           # 0..3 which segment
gen = np.random.default_rng(3)
def band(vals, top, bot, pad=4):
    v = np.asarray(vals, float); lo, hi = v.min(), v.max()
    if hi - lo < 1e-9: hi = lo + 1
    return bot - pad - (v - lo) / (hi - lo) * (bot - top - 2 * pad)
def poly(ys, col, wdt=1.6):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return f'<polyline fill="none" stroke="{col}" stroke-width="{wdt}" points="{pts}"/>'

rows = [("slow sensor", 30, 74, "#e0a458"),
        ("fast / noisy", 82, 126, "#6fb1e0"),
        ("very noisy, high amp.", 134, 178, "#e0644f"),
        ("non-stationary noise", 186, 230, "#8f7fe0"),
        ("state (discrete)", 238, 282, "#b48ce0"),
        ("accumulator", 290, 334, "#e07a9a")]
t = np.linspace(0, 1, N)
slow = np.sin(2 * math.pi * (t * 1.1 + 0.1)) + 0.25 * np.sin(2 * math.pi * t * 2.3) + 0.05 * gen.standard_normal(N)
fast = 0.5 * np.sin(2 * math.pi * t * 3) + 0.9 * gen.standard_normal(N)
vhi = 3.0 * gen.standard_normal(N) + 0.6 * np.sin(2 * math.pi * t * 1.5)
nsig = np.where(seg_id % 2 == 0, 0.25, 1.6)          # noise std alternates per segment
nonstat = nsig * gen.standard_normal(N)
# discrete state levels per segment
lvl_map = {0: 0.0, 1: 2.0, 2: 1.0, 3: 2.0}
state = np.array([lvl_map[s] for s in seg_id], float)
# accumulator: drain then recharge at two points
acc = np.zeros(N); a = 1.0
for i in range(N):
    a -= 0.006
    if i in (int(N*0.42), int(N*0.80)): a = 1.0
    acc[i] = a

polys = []
polys.append(poly(band(slow, *rows[0][1:3]), rows[0][3]))
polys.append(poly(band(fast, *rows[1][1:3]), rows[1][3], 1.3))
polys.append(poly(band(vhi, *rows[2][1:3]), rows[2][3], 1.2))
polys.append(poly(band(nonstat, *rows[3][1:3]), rows[3][3], 1.2))
polys.append(poly(band(state, *rows[4][1:3]), rows[4][3]))
polys.append(poly(band(acc, *rows[5][1:3]), rows[5][3]))
labels = "\n  ".join(f'<text x="6" y="{(a+b)//2+3}" font-size="10" fill="#555">{nm}</text>' for nm, a, b, _ in rows)
tlines = "\n  ".join(f'<line x1="{tx}" y1="26" x2="{tx}" y2="338" stroke="#c9b98a" stroke-width="1" stroke-dasharray="3 4"/>' for tx in tr)
ts_svg = f'''<figure style="margin:16px 0;text-align:center">
<svg viewBox="0 0 600 354" width="100%" style="max-width:600px" font-family="Helvetica Neue,Arial,sans-serif">
  <rect x="0" y="0" width="600" height="354" fill="#fcfcfb"/>
  {tlines}
  <text x="240" y="22" font-size="10" fill="#b09b5f">mode change</text>
  {labels}
  {chr(10).join("  "+p for p in polys)}
  <line x1="108" y1="340" x2="585" y2="340" stroke="#bbb" stroke-width="1"/>
  <text x="585" y="352" font-size="10" fill="#888" text-anchor="end">time &#8594;</text>
</svg>
<figcaption style="font-size:.82rem;color:#666;font-family:'Helvetica Neue',Arial,sans-serif"><strong>Figure 3.</strong>
One normal multivariate window: six channels of different <em>type, scale and domain</em> over time. A slow
low-noise sensor; a fast noisy sensor; a <em>very noisy, high-amplitude</em> sensor; a <em>non-stationary</em>
sensor whose noise level jumps between segments; a discrete state; a slowly draining / recharging accumulator.
All change together at the shared mode transitions (dashed) but on their own time constants (A8, A9).</figcaption>
</figure>'''

# ---------- splice into the doc ----------
def replace_figure(html, token, newblock):
    idx = html.find(token)
    assert idx != -1, "token not found: " + token
    start = html.rfind("<figure", 0, idx)
    end = html.find("</figure>", idx) + len("</figure>")
    return html[:start] + newblock + html[end:]

if __name__ == "__main__":
    html = open(DOC, encoding="utf-8").read()
    html = replace_figure(html, "Normal latent density (2D)", heat_svg)
    html = replace_figure(html, "The curvature knob", curv_svg)
    html = replace_figure(html, "slow sensor", ts_svg)
    open(DOC, "w", encoding="utf-8", newline="\n").write(html)
    print("spliced OK; doc bytes:", len(html))
