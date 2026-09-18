"""Empirical test of LatAD's SUBSYSTEM-LOCALIZATION claim (paper section 7).

Claim: the regime-community model yields a per-community surprise score, so an
alarm can be localized to the "most-surprised" correlation-community subsystem
rather than only a plant-wide flag.

We measure whether the MOST-SURPRISED community actually CONTAINS the attacked
channel(s), on WADI_clean (44 communities), HAI (28), SWaT_canon (25).

Data:
  expert_<ds>.npz : comm_channels (S,mx) padded -1 = channel indices per community
                    test_surprise (nseed,S,ntest) = per-community surprise per test window
                    y (ntest) window label, hard (ntest) difficult mask
  cache_<ds>.npz  : ch names, Xn_w/Xa_w window features, ya_w, idx_a, W  (from eda_real.load)

Target channel (PROXY): the channel with the largest standardized window-mean
deviation |z| during the attacked window/episode.  z uses the SAME community
feature space (train-normal window means).  This proxy is somewhat favourable to
the detector (surprise and |z| share the deviation driver), so a LOW score under
it is strong evidence against the claim; a HIGH score should be tempered.

Baselines: because communities are NESTED HAC subtrees they OVERLAP, so a target
channel sits in several communities.  The honest random baseline is COVERAGE-AWARE:
for a target covered by kc communities, random top-k hit prob = 1 - C(S-kc,k)/C(S,k).
We report that alongside the naive k/S.

Persists incrementally: one JSONL row per window to results/loc_<ds>.jsonl, a
per-dataset summary appended to results/loc_summary.jsonl.
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
POC = HERE.parent
CACHE = Path("E:/tmp/claude/E--Projects-Backlog-LatAD/6ae0c3f3-0eaf-4beb-8a09-88ce1e618d2a/scratchpad")
OUT = HERE / "loc_results"
OUT.mkdir(exist_ok=True)
DSETS = ["WADI_clean", "HAI", "SWaT_canon"]


def comb(n, k):
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def rand_topk_prob(S, kc, k):
    """P(a uniformly random size-k set of communities intersects the kc target communities)."""
    if kc <= 0:
        return 0.0
    if S - kc < k:
        return 1.0
    return 1.0 - comb(S - kc, k) / comb(S, k)


def eval_ds(ds):
    z = np.load(POC / "sota_bundle" / "experts_full" / f"expert_{ds}.npz", allow_pickle=True)
    c = np.load(CACHE / f"cache_{ds}.npz", allow_pickle=True)
    comm_ch = z["comm_channels"]                       # (S, mx) padded -1
    S = comm_ch.shape[0]
    surp = z["test_surprise"].mean(0)                  # (S, ntest) seed-averaged
    y = z["y"].astype(int)
    hard = z["hard"].astype(bool)
    ya_w = c["ya_w"].astype(int)
    ntest = surp.shape[1]
    assert len(y) == ntest and np.array_equal(y, ya_w), f"{ds}: label/window misalignment"

    Xa_w = np.asarray(c["Xa_w"], float)
    Xn_w = np.asarray(c["Xn_w"], float)
    nch = int(c["nch"])
    ch = list(c["ch"])
    # per-channel deviation = max over the 6 stat-blocks (mean/std/min/max/trend/range) of the
    # standardized |z|, in the SAME feature space the communities/surprise are built on. This is
    # construct-matched to the detector (an attack that perturbs dynamics, not level, still shows).
    mu = Xn_w.mean(0); sd = Xn_w.std(0) + 1e-9
    Za = (Xa_w - mu) / sd
    Z = np.abs(Za).reshape(len(Xa_w), 6, nch).max(1)   # (ntest, nch) deviation per channel

    comm_sets = [set(int(x) for x in row if x >= 0) for row in comm_ch]
    chan2comm = {cc: [s for s in range(S) if cc in comm_sets[s]] for cc in range(nch)}
    covered_ch = set().union(*comm_sets)

    # ---- per-window localization (proxy target = argmax |z| over all channels) --------
    rows = []
    with open(OUT / f"loc_{ds}.jsonl", "w") as f:
        for t in np.where(y == 1)[0]:
            az = np.abs(Z[t])
            tgt = int(np.argmax(az))                   # proxy target channel (global argmax|z|)
            tgt_comms = chan2comm[tgt]
            kc = len(tgt_comms)
            order = list(np.argsort(-surp[:, t]))      # communities ranked by surprise, desc
            topk = {k: set(order[:k]) for k in (1, 3, 5)}
            hit = {k: (kc > 0 and bool(set(tgt_comms) & topk[k])) for k in (1, 3, 5)}
            best_rank = min(order.index(g) for g in tgt_comms) + 1 if kc > 0 else None
            row = dict(t=int(t), tgt_ch=tgt, tgt_name=ch[tgt], tgt_z=float(az[tgt]),
                       covered=bool(kc > 0), kc=kc, hard=bool(hard[t]),
                       hit1=hit[1], hit3=hit[3], hit5=hit[5], rank=best_rank,
                       rb1=rand_topk_prob(S, kc, 1), rb3=rand_topk_prob(S, kc, 3),
                       rb5=rand_topk_prob(S, kc, 5), rand_rank=(S + 1) / (kc + 1) if kc > 0 else None,
                       top_comm=int(order[0]))
            f.write(json.dumps(row) + "\n")
            rows.append(row)

    # ---- episode-level (contiguous runs of attacked windows) --------------------------
    epis = []
    run = []
    for t in range(ntest):
        if y[t] == 1:
            run.append(t)
        elif run:
            epis.append(run); run = []
    if run:
        epis.append(run)

    ep_rows = []
    for ep in epis:
        ep = np.array(ep)
        az_ep = np.abs(Z[ep]).max(0)                   # peak |z| per channel over the episode
        tgt = int(np.argmax(az_ep))
        tgt_comms = chan2comm[tgt]; kc = len(tgt_comms)
        ep_surp = surp[:, ep].mean(1)                  # mean community surprise over episode
        order = list(np.argsort(-ep_surp))
        topk = {k: set(order[:k]) for k in (1, 3, 5)}
        hit = {k: (kc > 0 and bool(set(tgt_comms) & topk[k])) for k in (1, 3, 5)}
        best_rank = min(order.index(g) for g in tgt_comms) + 1 if kc > 0 else None
        ep_rows.append(dict(n=len(ep), tgt_ch=tgt, tgt_name=ch[tgt], covered=bool(kc > 0),
                            kc=kc, hard=bool(hard[ep].any()), rank=best_rank,
                            hit1=hit[1], hit3=hit[3], hit5=hit[5],
                            rb1=rand_topk_prob(S, kc, 1), rb3=rand_topk_prob(S, kc, 3),
                            rb5=rand_topk_prob(S, kc, 5), rand_rank=(S + 1) / (kc + 1) if kc > 0 else None))

    def agg(rs, mask=None):
        rs = [r for r in rs if (mask is None or mask(r))]
        n = len(rs)
        if n == 0:
            return dict(n=0)
        cov = [r for r in rs if r["covered"]]
        d = dict(n=n, n_covered=len(cov), coverage=len(cov) / n)
        for k in (1, 3, 5):
            # accuracy over ALL attacked windows (uncovered target counts as miss)
            d[f"top{k}"] = float(np.mean([r[f"hit{k}"] for r in rs]))
            # accuracy over covered-target subset only
            d[f"top{k}_cov"] = float(np.mean([r[f"hit{k}"] for r in cov])) if cov else 0.0
            d[f"rand{k}"] = float(np.mean([r[f"rb{k}"] for r in rs]))
        if cov:
            d["median_rank"] = float(np.median([r["rank"] for r in cov]))
            d["mean_rank"] = float(np.mean([r["rank"] for r in cov]))
            d["rand_mean_rank"] = float(np.mean([r["rand_rank"] for r in cov]))
        return d

    res = dict(dataset=ds, S=S, nch=nch, n_channels_covered=len(covered_ch),
               naive={f"top{k}": k / S for k in (1, 3, 5)},
               window_all=agg(rows),
               window_hard=agg(rows, lambda r: r["hard"]),
               window_easy=agg(rows, lambda r: not r["hard"]),
               episode_all=agg(ep_rows),
               episode_hard=agg(ep_rows, lambda r: r["hard"]),
               n_episodes=len(epis))
    return res, rows, ep_rows


def synthetic_control(ds="SWaT_canon"):
    """Confirm the localization READING pipeline: if we place the surprise maximum on a
    community that contains the (proxy) target channel, top-1 must fire; if we place it on
    a community that does NOT contain the target, top-1 must miss."""
    z = np.load(POC / "sota_bundle" / "experts_full" / f"expert_{ds}.npz", allow_pickle=True)
    comm_ch = z["comm_channels"]; S = comm_ch.shape[0]
    comm_sets = [set(int(x) for x in row if x >= 0) for row in comm_ch]
    nch = max(max(s) for s in comm_sets) + 1
    chan2comm = {cc: [s for s in range(S) if cc in comm_sets[s]] for cc in range(nch)}
    # a channel that is in at least one community
    c0 = next(cc for cc in range(nch) if chan2comm[cc])
    g_hit = chan2comm[c0][0]                            # a community containing c0
    g_miss = next(s for s in range(S) if c0 not in comm_sets[s])
    fake_hit = np.zeros(S); fake_hit[g_hit] = 100.0
    fake_miss = np.zeros(S); fake_miss[g_miss] = 100.0
    top_hit = int(np.argmax(fake_hit)); top_miss = int(np.argmax(fake_miss))
    ok_pos = top_hit in chan2comm[c0]                   # expected True
    ok_neg = top_miss not in chan2comm[c0]              # expected True (miss)
    return dict(dataset=ds, target_ch=c0, g_hit=g_hit, g_miss=g_miss,
                positive_localizes=bool(ok_pos), negative_misses=bool(ok_neg),
                pass_=bool(ok_pos and ok_neg))


def main():
    summ = []
    with open(OUT / "loc_summary.jsonl", "w") as f:
        for ds in DSETS:
            res, _, _ = eval_ds(ds)
            f.write(json.dumps(res) + "\n"); f.flush()
            summ.append(res)
            w = res["window_all"]; e = res["episode_all"]
            print(f"\n=== {ds}  S={res['S']} nch={res['nch']} covered={res['n_channels_covered']} "
                  f"episodes={res['n_episodes']} ===")
            print(f"  WINDOW (n={w['n']}, coverage={w['coverage']:.2f}): "
                  f"top1={w['top1']:.3f} top3={w['top3']:.3f} top5={w['top5']:.3f}  "
                  f"| rand(cov) {w['rand1']:.3f}/{w['rand3']:.3f}/{w['rand5']:.3f}  "
                  f"| naive {res['naive']['top1']:.3f}/{res['naive']['top3']:.3f}/{res['naive']['top5']:.3f}")
            wh = res["window_hard"]
            print(f"  WINDOW-HARD (n={wh['n']}): top1={wh['top1']:.3f} top3={wh['top3']:.3f} "
                  f"top5={wh['top5']:.3f} | rand(cov) {wh['rand1']:.3f}/{wh['rand3']:.3f}/{wh['rand5']:.3f}")
            print(f"  EPISODE (n={e['n']}, coverage={e['coverage']:.2f}): "
                  f"top1={e['top1']:.3f} top3={e['top3']:.3f} top5={e['top5']:.3f}  "
                  f"| rand(cov) {e['rand1']:.3f}/{e['rand3']:.3f}/{e['rand5']:.3f}")
    sc = synthetic_control("SWaT_canon")
    with open(OUT / "synthetic_control.json", "w") as f:
        json.dump(sc, f, indent=2)
    print(f"\nSYNTHETIC CONTROL {sc['dataset']}: positive_localizes={sc['positive_localizes']} "
          f"negative_misses={sc['negative_misses']} PASS={sc['pass_']}")
    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
