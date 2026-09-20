"""V3b: per-regime threshold with a CAP, thr_c = min(w p99_c + (1-w) g, g + cap), so sink regimes
(wide catch-all components with p99 >> global) cannot hide anomalies. cap=0 -> exactly base (invariant).
Appends rows to e2_rare_results.json (variant 'regthr_cap')."""
from __future__ import annotations
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2_rare_variants import Ctx, HERE, table


def v_regthr_cap(c, kappa, cap):
    def thr_fn(key, s_tr, s_te):
        g = np.quantile(s_tr, .99); thr_c = np.full(c.K, g)
        for k in range(c.K):
            m = c.a_tr == k; n = m.sum()
            if n < 2: continue
            w = 1.0 if kappa == 0 else n / (n + kappa)
            thr_c[k] = min(w * np.quantile(s_tr[m], .99) + (1 - w) * g, g + cap)
        ex = s_te - thr_c[c.a_te]
        return ex > 0, ex
    return c.evaluate(c.scores(c.Ltr, c.Lte), "regthr_cap", dict(kappa=kappa, cap=cap), thr_fn)


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["HAI"]
    seeds = [0, 1, 2]; rows = []
    for nm in names:
        for sd in seeds:
            c = Ctx(nm, sd); base = c.v_base(); r0 = v_regthr_cap(c, float("inf"), 0.0)
            ok = all(abs(r0[k][m] - base[k][m]) < 1e-9 for k in ("mixture", "nearest") for m in r0[k] if not np.isnan(r0[k][m]))
            print(f"[{nm} s{sd}] invariant (kappa=inf,cap=0) == base: {ok}", flush=True); assert ok
            rows.append(base)
            for cap in (5.0, 10.0, 20.0, 40.0):
                for kp in (0.0, 20.0): rows.append(v_regthr_cap(c, kp, cap))
        table(rows, nm)
    out = os.path.join(HERE, "e2_rare_results.json")
    prev = json.load(open(out)) if os.path.exists(out) else []
    prev = [r for r in prev if r["variant"] != "regthr_cap"] + [r for r in rows if r["variant"] == "regthr_cap"]
    json.dump(prev, open(out, "w"), indent=1); print(f"saved {len(prev)} rows")
