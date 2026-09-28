"""Round-2 reviewer comment 1: do LatAD's correlation-communities correspond to real
physical process stages / control loops on SWaT and WADI?

We reproduce LatAD's EXACT community construction (guided_ensemble.hac_communities:
train-normal channel-mean correlation -> dist=1-|rho| -> average linkage -> nested
subtrees of size [3,25]), then score those communities against the plant's DOCUMENTED
stages, which are encoded in the channel NAMES:
  SWaT  FIT101/LIT101/MV101 -> stage 1 (P1), AIT201 -> stage 2, ... P601 -> stage 6.
  WADI  '1_AIT_001_PV' -> area 1, '2_..'/'2A_..'/'2B_..' -> area 2/2A/2B, '3_..' -> area 3.

Reports, per dataset:
  * per-community dominant-stage PURITY (what LatAD actually scores; overlapping subtrees),
    mean size-weighted purity vs a permutation-null chance baseline;
  * a FLAT partition cut from the same linkage at k = number of stages -> ARI, NMI,
    homogeneity, completeness, v-measure vs the stage labels;
  * a per-community composition table.
Writes JSON + a short markdown summary. No paper edits.
"""
import sys, os, re, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # poc/
import eda_real as E
from scipy.cluster.hierarchy import linkage, to_tree, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             homogeneity_completeness_v_measure)

OUT = os.path.dirname(os.path.abspath(__file__))
MAXSZ = 25
rng = np.random.default_rng(0)


def swat_stage(name):
    m = re.search(r'\d', name)
    return f"P{m.group()}" if m else "P?"


def wadi_area(name):
    # WADI documented process = 3 areas: P1 primary grid, P2 secondary/booster (2,2A,2B),
    # P3 consumer/return. LEAK_/TOTAL_ etc. are derived/global sensors, not a process area.
    p = name.split('_')[0]
    if p == '1':
        return 'P1'
    if p in ('2', '2A', '2B'):
        return 'P2'
    if p == '3':
        return 'P3'
    return 'other'


def build_linkage(means):
    sdc = means.std(0)
    active = np.where(sdc > 1e-6)[0]
    Z = (means[:, active] - means[:, active].mean(0)) / sdc[active]
    C = np.nan_to_num(np.corrcoef(Z.T)); dist = 1 - np.abs(C)
    np.fill_diagonal(dist, 0.0); dist = (dist + dist.T) / 2
    L = linkage(squareform(dist, checks=False), method="average")
    return L, active


def nested_communities(L, active):
    _, nodes = to_tree(L, rd=True)
    def leaves(n): return [n.id] if n.is_leaf() else leaves(n.left) + leaves(n.right)
    seen, comms = set(), []
    for n in nodes:
        if not n.is_leaf():
            lv = sorted(leaves(n))
            if 3 <= len(lv) <= MAXSZ and tuple(lv) not in seen:
                seen.add(tuple(lv)); comms.append([int(active[i]) for i in lv])
    return comms


def purity_stats(comms, stage_of):
    """size-weighted mean dominant-stage purity + fraction pure; comms are channel-idx lists."""
    tot_ch, hit, pures, sizes, rows = 0, 0, 0, [], []
    for cm in comms:
        labs = [stage_of[c] for c in cm]
        vals, cnts = np.unique(labs, return_counts=True)
        dom = vals[cnts.argmax()]; dp = cnts.max() / len(labs)
        tot_ch += len(labs); hit += cnts.max()
        pures += 1 if dp == 1.0 else 0; sizes.append(len(labs))
        rows.append(dict(size=len(labs), dominant=str(dom), purity=round(float(dp), 3),
                         composition={str(v): int(c) for v, c in zip(vals, cnts)}))
    return dict(n_comm=len(comms), mean_size=round(float(np.mean(sizes)), 1),
                weighted_purity=round(hit / tot_ch, 4),
                frac_pure_comm=round(pures / len(comms), 4)), rows


def chance_purity(comms, stage_of, active, n_perm=2000):
    """permutation null: shuffle stage labels across ACTIVE channels, recompute weighted
    purity. comms hold ORIGINAL channel indices; map them to positions within `active`."""
    active = list(active)
    base = np.array([stage_of[int(c)] for c in active])       # labels in active order
    pos_of = {int(c): i for i, c in enumerate(active)}
    comm_pos = [[pos_of[int(c)] for c in cm] for cm in comms]
    out = []
    for _ in range(n_perm):
        p = rng.permutation(base)
        tot, hit = 0, 0
        for pos in comm_pos:
            _, c = np.unique(p[pos], return_counts=True); tot += len(pos); hit += c.max()
        out.append(hit / tot)
    return float(np.mean(out)), float(np.percentile(out, 95))


def run(name, stage_fn):
    D = E.load(name); ch = list(D["ch"]); nch = len(ch)
    Xn = np.asarray(D["Xn_w"], float)
    means = Xn[:, :nch]
    stages_all = [stage_fn(ch[i]) for i in range(nch)]
    sdc = means.std(0)
    # keep active channels that carry a documented process stage (drop constants and
    # derived/global sensors labelled 'other'); communities are built on this set.
    keep = np.array([i for i in range(nch) if sdc[i] > 1e-6 and stages_all[i] != 'other'])
    Z = (means[:, keep] - means[:, keep].mean(0)) / sdc[keep]
    C = np.nan_to_num(np.corrcoef(Z.T)); dist = 1 - np.abs(C)
    np.fill_diagonal(dist, 0.0); dist = (dist + dist.T) / 2
    L = linkage(squareform(dist, checks=False), method="average")
    active = keep
    stage_of = {int(i): stages_all[i] for i in active}
    stage_labels_active = [stage_of[int(i)] for i in active]
    uniq_stages = sorted(set(stage_labels_active))

    comms = nested_communities(L, active)
    pstats, rows = purity_stats(comms, stage_of)
    ch_mean, ch_p95 = chance_purity(comms, stage_of, active)

    # flat partition from the SAME linkage at k = number of documented stages
    k = len(uniq_stages)
    flat = fcluster(L, t=k, criterion="maxclust")           # over ACTIVE channels
    from sklearn.preprocessing import LabelEncoder
    y_true = LabelEncoder().fit_transform(stage_labels_active)
    ari = adjusted_rand_score(y_true, flat)
    nmi = normalized_mutual_info_score(y_true, flat)
    hom, comp, vme = homogeneity_completeness_v_measure(y_true, flat)

    res = dict(dataset=name, n_channels=nch, n_active=int(len(active)),
               n_stages=k, stages=uniq_stages,
               nested_communities=pstats,
               community_purity_vs_chance=dict(
                   weighted_purity=pstats["weighted_purity"],
                   chance_mean=round(ch_mean, 4), chance_p95=round(ch_p95, 4),
                   lift_over_chance=round(pstats["weighted_purity"] - ch_mean, 4)),
               flat_partition_vs_stages=dict(k=k, ARI=round(float(ari), 4),
                   NMI=round(float(nmi), 4), homogeneity=round(float(hom), 4),
                   completeness=round(float(comp), 4), v_measure=round(float(vme), 4)),
               example_communities=rows[:12])
    return res


def main():
    results = {}
    for name, fn in [("SWaT", swat_stage), ("WADI", wadi_area)]:
        try:
            results[name] = run(name, fn)
            r = results[name]
            print(f"\n=== {name}: {r['n_active']}/{r['n_channels']} active channels, "
                  f"{r['n_stages']} documented stages {r['stages']} ===")
            p = r["nested_communities"]; c = r["community_purity_vs_chance"]
            print(f"  LatAD nested communities: {p['n_comm']} (mean size {p['mean_size']})")
            print(f"  size-weighted stage purity: {p['weighted_purity']:.3f} "
                  f"(chance {c['chance_mean']:.3f}, +{c['lift_over_chance']:.3f}); "
                  f"stage-pure communities: {100*p['frac_pure_comm']:.0f}%")
            f = r["flat_partition_vs_stages"]
            print(f"  flat partition (k={f['k']}) vs stages: ARI={f['ARI']:.3f} "
                  f"NMI={f['NMI']:.3f} homogeneity={f['homogeneity']:.3f} "
                  f"completeness={f['completeness']:.3f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {"error": repr(e)}
    json.dump(results, open(f"{OUT}/r2_community_stage_align.json", "w"), indent=1)
    print(f"\nwrote {OUT}/r2_community_stage_align.json")


if __name__ == "__main__":
    main()
