"""Issue 2 (framing brainstorm): subsystem localization re-evaluated with GROUND-TRUTH attack targets
and CALIBRATED per-community surprise.

Ground truth (attack -> attacked channels):
  SWaT  : iTrust List_of_attacks_Final (36 physically-labelled attacks of 41), text copy fetched from
          github yoriyuki-aist/DeepSentinel examples/swat/attack_list.csv (saved next to this script).
          Canonical attack file starts 2015-12-28 10:00:00 at 1 Hz; the paper's SWaT_canon test is that
          file downsampled 10x, so downsampled index = seconds // 10.  Verified: attack 1 (10:29:14) = raw run 1754 s.
  WADI  : iTrust WADI attack table (13 rows as shipped with TranAD, WADI_attacklabels.csv); dates in that
          copy are mangled, so each row is matched to a raw label run by TIME OF DAY on the correct day
          (attack file starts 2017-10-09 18:00:00, 48 h, 10x downsampled).  Two raw runs (11 Oct 00:07:40 and
          12:16:10) have no row in that copy (attacks 13/14 of the 15) and are scored as TARGET-UNKNOWN.
  HAI   : HAI 20.07 attack timetable (hai_dataset_technical_details.pdf v4.0, pp. 39-41), 38 attacks in
          order; the 38 raw label runs in test1+test2 match the timetable start times one-to-one.
          Per-process flags attack_P1/P2/P3 from the CSVs give a second, process-level ground truth.

Scores (all from sota_bundle/experts_full/expert_<ds>.npz, seed x community x window):
  raw    : seed-mean test_surprise, argmax (the original evaluation)
  excl   : raw, but communities whose CALIBRATION surprise is degenerate (sd < 1e-3 in any seed, i.e. a
           train-normal p-value is undefined) are removed from the ranking  [train-normal-only rule]
  rank   : per seed, per community, empirical percentile of the test surprise within that community's own
           calibration surprise (upper-tail p-value), seed-averaged; degenerate communities excluded
  onset  : rank score minus its median over the preceding L windows (online, label-free): which
           community's surprise ROSE, not which is loud
  oracle : z against the community's TEST-NORMAL distribution (not deployable; diagnostic upper bound
           for 'is the loud-community bias a calibration-drift artefact')
Hit@k: any community containing any attacked channel is among the k most-surprised.  Random baseline is
coverage-aware: 1 - C(S-kc,k)/C(S,k) for a target covered by kc communities.  Also STAGE-level hit: the
top-1 community's majority stage is one of the attacked stages (baseline = share of communities in an
attacked stage).  Persists per-window rows to framing_loc/loc2_<ds>.jsonl and a summary json.
"""
from __future__ import annotations
import json, math, os, re
from datetime import datetime, timedelta
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POC = os.path.dirname(HERE)
CACHE = "E:/tmp/claude/E--Projects-Backlog-LatAD/6ae0c3f3-0eaf-4beb-8a09-88ce1e618d2a/scratchpad"
OUT = os.path.join(HERE, "framing_loc"); os.makedirs(OUT, exist_ok=True)
L_ONSET = 10

# ----------------------------------------------------------------------------- ground truth tables
HAI_T0 = [datetime(2019, 10, 29, 11, 0, 0), datetime(2019, 11, 4, 15, 0, 0)]; HAI_SPLIT = 291600
HAI_ATTACKS = {  # No -> target points (HAI 20.07 timetable). names normalised to the CSV columns below
    1: ["P1_B3004", "P1_LIT01"], 2: ["P1_LCV01D"], 3: ["P1_LCV01D", "P1_LIT01"], 4: ["P1_B3005"], 5: ["P1_B3004"],
    6: ["P1_B2016"], 7: ["P1_B2016", "P1_PIT01"], 8: ["P1_PCV01D"], 9: ["P1_PCV01D", "P1_PIT01"],
    10: ["P2_SD01", "P2_SIT01"], 11: ["P3_LH", "P3_LCP01D"], 12: ["P3_LL", "P3_LCV01D"], 13: ["P2_SD01"],
    14: ["P2_SD01", "P2_SIT01", "P1_B2016", "P1_PIT01"], 15: ["P2_SD01", "P1_PCV01D"], 16: ["P2_SD01", "P2_SIT01"],
    17: ["P2_SD01", "P2_SIT01", "P1_LCV01D", "P1_LIT01"], 18: ["P2_SD01", "P1_B3005"], 19: ["P3_LH", "P3_LCP01D", "P1_B2016"],
    20: ["P3_LH", "P3_LCP01D", "P1_B3004", "P1_LIT01"], 21: ["P3_LH", "P3_LCP01D", "P1_LCV01D"], 22: ["P3_LH", "P3_LCP01D"],
    23: ["P1_B3004", "P1_B3005", "P1_FT03"], 24: ["P1_PCV01D", "P1_B3005", "P1_FT03"], 25: ["P1_B2016"],
    26: ["P1_B2016", "P1_PIT01", "P1_B3005"], 27: ["P1_B3005", "P1_FT03"], 28: ["P3_LL", "P3_LCV01D", "P1_B2016", "P1_PIT01"],
    29: ["P1_B2016", "P1_B3005"], 30: ["P3_LCV01D", "P1_B2016"], 31: ["P3_LCV01D", "P1_LCV01D"], 32: ["P3_LCV01D", "P1_B3005"],
    33: ["P2_SD01"], 34: ["P2_SD01", "P3_LL", "P3_LCV01D"], 35: ["P2_SD01", "P1_B3004"], 36: ["P1_B3004", "P1_B2016"],
    37: ["P1_LCV01D"], 38: ["P1_LCV01D", "P1_PCV01D"]}
SWAT_T0 = datetime(2015, 12, 28, 10, 0, 0)
WADI_T0 = datetime(2017, 10, 9, 18, 0, 0)
WADI_ROWS = [  # (day offset from 9 Oct, start, end, points) from WADI_attacklabels.csv, day fixed by the raw runs
    (0, "19:25:00", "19:50:16", "1_MV_001"), (1, "10:24:10", "10:34:00", "1_FIT_001"),
    (1, "10:55:00", "11:24:00", "2_LT_002, 1_AIT_001"), (1, "11:30:40", "11:44:50", "2_MCV_101, 2_MCV_201, 2_MCV_301, 2_MCV_401, 2_MCV_501, 2_MCV_601"),
    (1, "13:39:30", "13:50:40", "2_MCV_101, 2_MCV_201"), (1, "14:48:17", "14:59:55", "1_AIT_002, 1_AIT_003, 1_AIT_004, 1_AIT_005, 1_AIT_006, 2_MV_003"),
    (1, "17:40:00", "17:49:40", "2_MCV_007"), (2, "10:55:00", "10:56:27", "1_P_006"), (2, "11:17:54", "11:31:20", "1_MV_001"),
    (2, "11:36:31", "11:47:00", "2_MCV_007"), (2, "11:59:00", "12:05:00", "2_MCV_007"), (2, "15:26:30", "15:37:00", "2_LT_002, 1_AIT_001")]


def norm_swat(pt):
    return re.sub(r"[^A-Z0-9]", "", pt.upper()).replace("DIT301", "DPIT301")


def load_swat_gt():
    rows = []
    with open(os.path.join(HERE, "swat_attack_list_deepsentinel.csv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(",")
            aid = int(parts[0]); st = datetime.strptime(parts[1], "%Y/%m/%d %H:%M:%S"); en = datetime.strptime(parts[2], "%Y/%m/%d %H:%M:%S")
            # attack-point field may be quoted with commas
            m = re.match(r'^\d+,[^,]+,[^,]+,("([^"]*)"|([^,]*)),', line)
            pts = (m.group(2) or m.group(3))
            pts = [norm_swat(p) for p in re.split(r"[,;]", pts) if p.strip()]
            s = int((st - SWAT_T0).total_seconds()) // 10; e = int((en - SWAT_T0).total_seconds()) // 10
            rows.append(dict(id=aid, s=s, e=e, points=pts))
    return rows


def load_wadi_gt():
    rows = []
    for i, (d, st, en, pts) in enumerate(WADI_ROWS):
        base = WADI_T0 + timedelta(days=d)
        s = datetime.combine(base.date(), datetime.strptime(st, "%H:%M:%S").time()); e = datetime.combine(base.date(), datetime.strptime(en, "%H:%M:%S").time())
        rows.append(dict(id=i + 1, s=int((s - WADI_T0).total_seconds()) // 10, e=int((e - WADI_T0).total_seconds()) // 10,
                         points=[p.strip() for p in pts.split(",")]))
    return rows


def runs(y):
    y = np.asarray(y).astype(int); d = np.diff(np.r_[0, y, 0]); return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def stage_of(ds, name):
    if ds == "HAI": return name.split("_")[0]
    if ds == "SWaT_canon": return "S" + re.search(r"(\d)\d\d", name).group(1)
    return name.split("_")[0]        # WADI: 1 / 2 / 2A / 2B / 3 / other


def comb(n, k): return math.comb(n, k) if 0 <= k <= n else 0
def rand_topk(S, kc, k): return 0.0 if kc <= 0 else (1.0 if S - kc < k else 1 - comb(S - kc, k) / comb(S, k))


def eval_ds(ds):
    z = np.load(os.path.join(POC, "sota_bundle", "experts_full", f"expert_{ds}.npz"), allow_pickle=True)
    c = np.load(os.path.join(CACHE, f"cache_{ds}.npz"), allow_pickle=True)
    ch = [str(x) for x in c["ch"]]; nch = len(ch); idx_a = c["idx_a"]; W = int(c["W"]); ya_raw = c["ya_raw"].astype(int)
    y = z["y"].astype(int); hard = z["hard"].astype(bool); ntest = len(y)
    comm = z["comm_channels"]; S = comm.shape[0]
    comm_sets = [set(int(x) for x in row if x >= 0) for row in comm]
    chan2comm = {cc: [s for s in range(S) if cc in comm_sets[s]] for cc in range(nch)}
    comm_stage = []
    for s in range(S):
        st = [stage_of(ds, ch[i]) for i in comm_sets[s]]; comm_stage.append(max(set(st), key=st.count))

    # ---------------- ground truth per raw-index interval ----------------
    if ds == "SWaT_canon":
        gt = load_swat_gt()
    elif ds == "WADI_clean":
        gt = load_wadi_gt()
    else:
        rr = runs(ya_raw); assert len(rr) == 38, len(rr)
        gt = [dict(id=i + 1, s=int(a), e=int(b) - 1, points=HAI_ATTACKS[i + 1]) for i, (a, b) in enumerate(rr)]
    def match(p):   # exact name, or WADI tag with a recorded suffix (1_MV_001 -> 1_MV_001_STATUS, 2_MCV_007 -> 2_MCV_007_CO)
        return [i for i, n in enumerate(ch) if n == p or n.startswith(p + "_")]
    for g in gt:
        g["chan"] = sorted(set(i for p in g["points"] for i in match(p)))
        g["unobserved"] = [p for p in g["points"] if not match(p)]
    # sanity: every GT interval should overlap at least one labelled raw run
    rr = runs(ya_raw)
    for g in gt:
        g["matched_run"] = any(not (g["e"] < a or g["s"] >= b) for a, b in rr)
    n_unmatched = sum(not g["matched_run"] for g in gt)
    print(f"{ds}: {len(gt)} GT attacks, {n_unmatched} not overlapping any labelled run; raw runs={len(rr)}; "
          f"unobserved points: {[ (g['id'], g['unobserved']) for g in gt if g['unobserved'] ]}")

    # ---------------- scores ----------------
    cs = z["calib_surprise"].astype(np.float64); ts = z["test_surprise"].astype(np.float64)   # (5,S,n)
    ts = np.nan_to_num(ts, nan=0.0, posinf=1e30, neginf=-1e30)
    degenerate = (cs.std(2) < 1e-3).any(0)                                   # (S,)
    raw = ts.mean(0)
    # rank calibration: upper-tail empirical p within own calibration surprise (per seed), seed-mean
    rank = np.zeros((S, ntest))
    for k in range(ts.shape[0]):
        for s in range(S):
            ref = np.sort(cs[k, s]); rank[s] += np.searchsorted(ref, ts[k, s], side="left") / len(ref)
    rank /= ts.shape[0]
    excl = raw.copy(); excl[degenerate] = -np.inf
    rankx = rank.copy(); rankx[degenerate] = -np.inf
    onset = np.full_like(rank, -np.inf)
    for t in range(ntest):
        lo = max(0, t - L_ONSET)
        base = np.median(rank[:, lo:t], axis=1) if t > lo else rank[:, t]
        onset[:, t] = rank[:, t] - base
    onset[degenerate] = -np.inf
    tn = ts[:, :, y == 0]
    oracle = ((ts - tn.mean(2, keepdims=True)) / (tn.std(2, keepdims=True) + 1e-9)).mean(0); oracle[degenerate] = -np.inf
    scores = dict(raw=raw, excl=excl, rank=rankx, onset=onset, oracle=oracle)
    print(f"  S={S} degenerate communities (calib sd<1e-3 in some seed): {np.where(degenerate)[0].tolist()} "
          f"-> {[ [ch[i] for i in comm_sets[s]] for s in np.where(degenerate)[0] ]}")

    # ---------------- per-window evaluation ----------------
    rows = []
    for t in np.where(y == 1)[0]:
        a, b = int(idx_a[t]), int(idx_a[t]) + W
        hits = [g for g in gt if not (g["e"] < a or g["s"] >= b)]
        tgt = sorted(set(cc for g in hits for cc in g["chan"]))
        stages = sorted(set(stage_of(ds, ch[cc]) for g in hits for cc in g["chan"]) | set(stage_of(ds, p) for g in hits for p in g["unobserved"]))
        tgt_comms = sorted(set(s for cc in tgt for s in chan2comm[cc]))
        kc = len(tgt_comms)
        row = dict(t=int(t), attacks=[g["id"] for g in hits], gt_known=bool(hits) and bool(tgt or any(g["unobserved"] for g in hits)),
                   n_tgt_ch=len(tgt), covered=kc > 0, kc=kc, hard=bool(hard[t]), stages=stages,
                   rb1=rand_topk(S, kc, 1), rb3=rand_topk(S, kc, 3), rb5=rand_topk(S, kc, 5),
                   rb_stage=float(np.mean([comm_stage[s] in stages for s in range(S)])) if stages else 0.0)
        for name, sc in scores.items():
            order = list(np.argsort(-sc[:, t], kind="stable"))
            rk = min(order.index(g) for g in tgt_comms) + 1 if kc else None
            row[name] = dict(top=int(order[0]), rank=rk, hit1=bool(rk and rk <= 1), hit3=bool(rk and rk <= 3), hit5=bool(rk and rk <= 5),
                             stage_hit=bool(stages and comm_stage[order[0]] in stages))
        rows.append(row)
    with open(os.path.join(OUT, f"loc2_{ds}.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

    # ---------------- episode level (contiguous attacked windows; score = mean over episode) ----------------
    ep_rows = []
    for a_, b_ in runs(y):
        ep = list(range(a_, b_)); wins = [r for r in rows if r["t"] in ep]
        tgt_comms = sorted(set(s for r in wins for g in gt if g["id"] in r["attacks"] for cc in g["chan"] for s in chan2comm[cc]))
        stages = sorted(set(st for r in wins for st in r["stages"]))
        kc = len(tgt_comms)
        row = dict(n=len(ep), covered=kc > 0, kc=kc, hard=bool(hard[ep].any()), gt_known=any(r["gt_known"] for r in wins),
                   rb1=rand_topk(S, kc, 1), rb3=rand_topk(S, kc, 3), rb5=rand_topk(S, kc, 5),
                   rb_stage=float(np.mean([comm_stage[s] in stages for s in range(S)])) if stages else 0.0)
        for name, sc in scores.items():
            m = sc[:, ep]; m = np.where(np.isfinite(m), m, -np.inf).mean(1) if name != "onset" else np.where(np.isfinite(m), m, -np.inf).max(1)
            order = list(np.argsort(-m, kind="stable")); rk = min(order.index(g) for g in tgt_comms) + 1 if kc else None
            row[name] = dict(rank=rk, hit1=bool(rk and rk <= 1), hit3=bool(rk and rk <= 3), hit5=bool(rk and rk <= 5),
                             stage_hit=bool(stages and comm_stage[order[0]] in stages))
        ep_rows.append(row)

    def agg(rs, mask=None):
        rs = [r for r in rs if r["gt_known"] and (mask is None or mask(r))]
        n = len(rs)
        if n == 0: return dict(n=0)
        d = dict(n=n, coverage=float(np.mean([r["covered"] for r in rs])),
                 rand1=float(np.mean([r["rb1"] for r in rs])), rand3=float(np.mean([r["rb3"] for r in rs])),
                 rand5=float(np.mean([r["rb5"] for r in rs])), rand_stage=float(np.mean([r["rb_stage"] for r in rs])))
        for name in scores:
            d[name] = {k: float(np.mean([r[name][k] for r in rs])) for k in ("hit1", "hit3", "hit5", "stage_hit")}
            cov = [r for r in rs if r["covered"]]
            d[name]["median_rank_cov"] = float(np.median([r[name]["rank"] for r in cov])) if cov else None
        return d

    # ---------------- controls: false-localization profile on NORMAL windows, per-episode breakdown, MC p-value ----
    normal_top1 = {}
    for name, sc in scores.items():
        tops = np.argmax(sc[:, y == 0], axis=0)
        normal_top1[name] = (np.bincount(tops, minlength=S) / len(tops)).tolist()
    rng = np.random.default_rng(0)
    mc = {}
    eps_known = [r for r in ep_rows if r["gt_known"]]
    for name in scores:
        obs = sum(r[name]["hit1"] for r in eps_known)
        sims = [sum(rng.random() < r["rb1"] for r in eps_known) for _ in range(20000)]
        mc[name] = dict(episodes=len(eps_known), observed_top1_hits=int(obs), expected=float(np.mean(sims)),
                        p_value=float(np.mean(np.asarray(sims) >= obs)))
    breakdown = []
    for (a_, b_), r in zip(runs(y), ep_rows):
        wins = [w for w in rows if a_ <= w["t"] < b_]
        ids = sorted(set(i for w in wins for i in w["attacks"]))
        pts = sorted(set(p for g in gt if g["id"] in ids for p in g["points"]))
        rec = dict(attacks=ids, points=pts, n_win=len(wins), covered=r["covered"], kc=r["kc"], hard=r["hard"])
        for name in ("excl", "rank", "onset"):
            m = scores[name][:, a_:b_]; m = np.where(np.isfinite(m), m, -np.inf); m = m.max(1) if name == "onset" else m.mean(1)
            top = int(np.argmax(m))
            rec[name] = dict(top=top, top_channels=[ch[i] for i in sorted(comm_sets[top])], rank=r[name]["rank"],
                             top_normal_share=round(normal_top1[name][top], 3))
        breakdown.append(rec)
    with open(os.path.join(OUT, f"loc2_episodes_{ds}.jsonl"), "w") as f:
        for b in breakdown: f.write(json.dumps(b) + "\n")

    res = dict(dataset=ds, S=S, n_degenerate=int(degenerate.sum()), n_gt_attacks=len(gt),
               normal_top1_profile_top3={name: sorted(v, reverse=True)[:3] for name, v in normal_top1.items()},
               mc_episode_top1=mc,
               n_attacked_windows=int((y == 1).sum()), n_windows_gt_known=int(sum(r["gt_known"] for r in rows)),
               window_all=agg(rows), window_hard=agg(rows, lambda r: r["hard"]), episode_all=agg(ep_rows),
               top1_monopoly={name: [float(v) for v in sorted(np.bincount([r[name]["top"] for r in rows], minlength=S) / len(rows))[-2:][::-1]] for name in scores})
    return res


def main():
    summ = {}
    for ds in ["HAI", "SWaT_canon", "WADI_clean"]:
        r = eval_ds(ds); summ[ds] = r
        for lvl in ("window_all", "window_hard", "episode_all"):
            a = r[lvl]
            if a["n"] == 0: continue
            print(f"  {lvl:12s} n={a['n']:4d} cov={a['coverage']:.2f} rand1/3/5={a['rand1']:.3f}/{a['rand3']:.3f}/{a['rand5']:.3f} rand_stage={a['rand_stage']:.2f}")
            for name in ("raw", "excl", "rank", "onset", "oracle"):
                s = a[name]; print(f"     {name:6s} top1={s['hit1']:.3f} top3={s['hit3']:.3f} top5={s['hit5']:.3f} stage@1={s['stage_hit']:.3f} medrank(cov)={s['median_rank_cov']}")
        print(f"  top-1 monopoly (share of attacked windows taken by the two most frequent top-1 communities): "
              f"{ {k: [round(v, 2) for v in vals] for k, vals in r['top1_monopoly'].items()} }")
        print(f"  NORMAL-window top-1 profile (3 largest shares): { {k: [round(v, 2) for v in vals] for k, vals in r['normal_top1_profile_top3'].items()} }")
        print(f"  MC episode top-1 vs coverage-aware random: { {k: (v['observed_top1_hits'], round(v['expected'], 1), v['p_value']) for k, v in r['mc_episode_top1'].items()} }")
    json.dump(summ, open(os.path.join(OUT, "loc2_summary.json"), "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
