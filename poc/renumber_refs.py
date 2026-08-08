"""Renumber the LatAD reference list by first appearance and sync every in-text citation number.
Works on the named-id scheme: <li id="r-xxx"> in <ol class="refs">, cited as <a href="#r-xxx">N</a>.
Reorders the <li> to first-cited order and rewrites every in-text N to the ref's new 1-based position.
Reports any reference cited-but-missing or present-but-uncited (orphan)."""
import re, sys

P = "paper/paper.html"
html = open(P, encoding="utf-8").read()

# split body vs the <ol class="refs"> ... </ol>
m = re.search(r'(<ol class="refs"[^>]*>)(.*?)(</ol>)', html, re.S)
pre, ol_open, ol_body, ol_close, post = html[:m.start()], m.group(1), m.group(2), m.group(3), html[m.end():]

# parse <li id="r-xxx"> entries
items = re.findall(r'(<li id="(r-[\w-]+)">.*?</li>)', ol_body, re.S)
by_id = {rid: li for li, rid in items}

# first-appearance order of ids in the body (citations before the ref list)
cited = []
for rid in re.findall(r'href="#(r-[\w-]+)"', pre):
    if rid not in cited:
        cited.append(rid)

present = set(by_id)
missing = [c for c in cited if c not in present]           # cited but no <li>
orphan = [r for r in present if r not in cited]            # <li> but never cited
order = [c for c in cited if c in present] + orphan        # cited-order, uncited appended

new_ol = "\n".join(by_id[r] for r in order)
num = {r: i + 1 for i, r in enumerate(order)}              # id -> 1-based position

def fix(txt):
    return re.sub(r'(<a href="#(r-[\w-]+)">)\d+(</a>)',
                  lambda mm: mm.group(1) + str(num.get(mm.group(2), 0)) + mm.group(3), txt)

out = fix(pre) + ol_open + "\n" + new_ol + "\n" + ol_close + fix(post)
open(P, "w", encoding="utf-8").write(out)
print(f"refs: {len(order)} | reordered by first appearance; in-text numbers synced")
if missing: print("  CITED BUT MISSING <li>:", missing)
if orphan: print("  UNCITED (appended at end):", orphan)
print("  final order:", [f"{num[r]}:{r}" for r in order])
