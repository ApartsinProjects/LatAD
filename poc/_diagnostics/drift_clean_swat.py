"""Drift-cleaning proof case on SWaT_canon: remove LITERATURE-documented normal-drift windows (DAICS,
Abdelaty et al. 2021: pump P102 gains a second normal state in test; analyzer AIT201's normal range shifts
from ~[251,272] in train to ~[168,267] in test) from the TEST partition, and recompute the Difficult-subset
AUROC for EVERY method uniformly. Second, clearly-labelled variant: clean with ALL channels the EDA labeler
flags as train-vs-test-normal shifted.

Guardrails: only label==0 windows are ever removed; attacks intact (asserted); drift regions come from the
literature / an a-priori EDA rule, never from where any method errs; cleaned AND uncleaned reported.

Invariants stated in advance:
  I1  uncleaned Difficult AUROCs reproduce loo_fusion_boosted.json (HCcoh+LatAD .837, boosted .881,
      linres .782, LatAD .804) to 3 dp.
  I2  zero anomaly windows removed; n_diff stays 85; the anomaly windows are bit-identical before/after.
  I3  data provenance: the Kaggle-mirror normal.csv is segmented by timestamp; the claim that its first
      395,298 rows ARE the canonical attack file's normal rows must hold row-for-row (else the leak finding
      is withdrawn).
  I4  EDA labeler validation: with the PROPER train (Normal_v0 recording only) P102 (state gain) and AIT201
      (range shift) rank in the top-10 shifted channels; with the leaked train (test-normal rows inside
      train) AIT201 must NOT show a range shift. Both directions are predictions, not fits.
  I5  the removed normals' mean score-percentile (vs all test normals) is reported per method: cleaning can
      only help a method if the removed normals were high-scoring for it.
Output: drift_clean_swat.json (+ .md written by hand from it)."""
from __future__ import annotations
import os, sys, json, time, numpy as np, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("EXPERTS_DIR", os.path.join(ROOT, "sota_bundle", "experts_full"))
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
import eda_real as E
import swat as SW
import ensemble_final as EF

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "SWaT_canon"; REPS = int(os.environ.get("BOOT_REPS", "2000"))
LIT_AIT201 = (251.0, 272.0)          # DAICS-stated train-normal conductivity range
KS_FLAG = 0.5                        # a-priori: a channel is "shifted" if train-vs-test-normal KS D > 0.5
OUT = {}


def prep(df, sens):
    return df[sens].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0).values.astype(np.float32)


def au_ms(y, arr, m):
    k = (y == 0) | m
    if arr.ndim == 1:
        return float(roc_auc_score(y[k], arr[k]))
    return float(np.mean([roc_auc_score(y[k], arr[i][k]) for i in range(arr.shape[0])]))


def pct_of_removed(y, arr, rem):
    """mean percentile (vs ALL test normals, uncleaned) of the removed normals' scores; seed-averaged."""
    A = arr if arr.ndim == 2 else arr[None, :]
    out = []
    for s in A:
        ref = np.sort(s[y == 0]); out.append(float(np.mean(np.searchsorted(ref, s[rem], side="right") / len(ref))))
    return round(float(np.mean(out)), 3)


t0 = time.time()
# ------------------------------------------------------------------ data provenance (I3)
df = SW._read("normal.csv"); df.columns = [c.strip() for c in df.columns]
ts = pd.to_datetime(df["Timestamp"].str.strip(), format="%d/%m/%Y %I:%M:%S %p")
canon = np.load(os.path.join(ROOT, "datasets", "_new", "SWaT_canonical", "swat_attack_canonical.npz"), allow_pickle=True)
Xa_full, ya_full, sens = canon["Xa"].astype(np.float32), canon["ya"].astype(int), list(canon["sens"])
raw = df[sens].apply(pd.to_numeric, errors="coerce")
V = raw.ffill().bfill().fillna(0.0).values.astype(np.float32)          # identical to the eda_real/swat prep
n_cn = int((ya_full == 0).sum())
big = np.where(np.abs(ts.diff().dt.total_seconds().values) > 3600)[0]     # recording boundaries (>1h jumps)
segs = [(int(s), int(e)) for s, e in zip(np.r_[0, big], np.r_[big, len(df)])]
seg_desc = [dict(rows=[s, e], n=e - s, start=str(ts.iloc[s]), end=str(ts.iloc[e - 1])) for s, e in segs]
seg0 = V[:n_cn]; Cn = Xa_full[ya_full == 0]
rowmatch = float(np.isclose(seg0, Cn, rtol=1e-4, atol=1e-4).all(1).mean())
warm = int(len(df) * 0.02)
OUT["provenance"] = dict(normal_csv_rows=int(len(df)), segments=seg_desc, canonical_normal_rows=n_cn,
                         seg0_equals_canonical_normal_rowwise=rowmatch,
                         warmup_rows_dropped=warm, canonical_normal_rows_inside_train=int(n_cn - warm))
assert rowmatch == 1.0, "I3 failed: normal.csv head != canonical normal rows"
print(f"[I3] normal.csv = {[d['n'] for d in seg_desc]} rows; first {n_cn} rows == canonical test-normal rows "
      f"(row-wise match {rowmatch:.3f}); {n_cn - warm} of them sit inside the SWaT_canon TRAIN set", flush=True)

# ------------------------------------------------------------------ windows / scores
d = np.load(os.path.join(HERE, f"scores_{NAME}.npz")); y = d["label"].astype(int)
D = E.load(NAME); idx = D["idx_a"]; W = D["W"]; fn, _, stride = E.RAW[NAME]
assert len(idx) == len(y) and np.array_equal(D["ya_w"], y)
Xa10, ya10 = Xa_full[::10], ya_full[::10]                                # raw (unstandardised) test rows, ds=10
assert len(Xa10) == len(D["Xa_raw"]) and np.array_equal(ya10, D["ya_raw"])
maxz, mthr = d["maxz"], float(d["maxz_thr"]); diff = (y == 1) & (maxz <= mthr)
ens, y2, _, nseed = EF.ensemble_scores(NAME); assert np.array_equal(y2, y)
bl = np.load(os.path.join(HERE, f"boosted_loo_{NAME}.npz"))
meth = {"trivial max|z| (floor)": maxz, "linres": d["linres"], "IF": d["IF"], "AE": d["AE"], "USAD": d["USAD"],
        "TranAD": d["TranAD"], "LatAD (global)": d["LatAD"], "null+HC": ens["null+HC"],
        "HCcoh+LatAD (community headline)": ens["HCcoh+LatAD"], "boosted_LOO": bl["b_te"]}
unc = {k: round(au_ms(y, v, diff), 3) for k, v in meth.items()}
ref = json.load(open(os.path.join(HERE, "loo_fusion_boosted.json")))[NAME]["auroc"]
for a, b in (("HCcoh+LatAD (community headline)", "HCcoh+LatAD"), ("boosted_LOO", "boosted_LOO"),
             ("linres", "linres"), ("LatAD (global)", "LatAD_global")):
    assert abs(unc[a] - ref[b]["Difficult"]) < 5e-4, f"I1 failed {a}: {unc[a]} vs {ref[b]['Difficult']}"
print(f"[I1] uncleaned Difficult AUROCs reproduce loo_fusion_boosted.json: {unc}", flush=True)

# ------------------------------------------------------------------ STEP 1: literature drift regions
jP, jA = sens.index("P102"), sens.index("AIT201")
tr_leaked = V[warm:][::10]                                                 # the train actually used
s_v0, e_v0 = max(segs, key=lambda se: se[1] - se[0])                       # Normal_v0 recording (496,800 rows, 22/12 16:00)
assert e_v0 - s_v0 == 496800 and str(ts.iloc[s_v0]).startswith("2015-12-22")
nan_cols = [c for c in sens if raw[c].iloc[s_v0:e_v0].isna().mean() > 0.5]  # channels the mirror does NOT record in Normal_v0
v0 = V[s_v0:e_v0]; tr_proper = v0[int(len(v0) * 0.02):][::10]              # DAICS-style train: Normal_v0 only
OUT["provenance"]["normal_v0_rows"] = [s_v0, e_v0]; OUT["provenance"]["channels_unrecorded_in_normal_v0"] = nan_cols
v1 = V[e_v0:]; m = min(len(v0) - 1800, len(v1))
OUT["provenance"]["normal_v1_rowwise_equals_v0_shifted_1800s"] = float(np.isclose(v0[1800:1800 + m], v1[:m], rtol=1e-4, atol=1e-4).all(1).mean())
print(f"[I3] Normal_v0 rows {s_v0}-{e_v0}; channels NaN for all of Normal_v0 in the mirror (ffilled from the attack file's last "
      f"row in the pipeline): {nan_cols}", flush=True)
P102_states = dict(leaked_train=sorted(np.unique(tr_leaked[:, jP]).tolist()),
                   proper_train=sorted(np.unique(tr_proper[:, jP]).tolist()),
                   test_normal=sorted(np.unique(Xa10[ya10 == 0, jP]).tolist()),
                   test_attack=sorted(np.unique(Xa10[ya10 == 1, jP]).tolist()))
P102_new = [s for s in P102_states["test_normal"] + P102_states["test_attack"] if s not in P102_states["proper_train"]]
AIT_rng = dict(leaked_train=[float(tr_leaked[:, jA].min()), float(tr_leaked[:, jA].max())],
               proper_train=[float(tr_proper[:, jA].min()), float(tr_proper[:, jA].max())],
               proper_train_p1_p99=[float(np.percentile(tr_proper[:, jA], 1)), float(np.percentile(tr_proper[:, jA], 99))],
               test_normal=[float(Xa10[ya10 == 0, jA].min()), float(Xa10[ya10 == 0, jA].max())],
               literature=list(LIT_AIT201))
wmean = np.stack([Xa10[i:i + W].mean(0) for i in idx])                     # (n_win, C) raw window means
wmin = np.stack([Xa10[i:i + W].min(0) for i in idx]); wmax = np.stack([Xa10[i:i + W].max(0) for i in idx])
f_P102 = np.array([np.isin(Xa10[i:i + W, jP], P102_new).any() for i in idx])
f_AIT_lit = (wmean[:, jA] < LIT_AIT201[0]) | (wmean[:, jA] > LIT_AIT201[1])
f_AIT_proper = (wmean[:, jA] < AIT_rng["proper_train"][0]) | (wmean[:, jA] > AIT_rng["proper_train"][1])
lit_region = f_P102 | f_AIT_lit
rem_lit = lit_region & (y == 0)
OUT["step1_literature"] = dict(P102_states=P102_states, P102_state_absent_from_proper_train=P102_new,
                               AIT201_ranges=AIT_rng,
                               windows_P102_newstate=dict(total=int(f_P102.sum()), normal=int((f_P102 & (y == 0)).sum()),
                                                          attack=int((f_P102 & (y == 1)).sum())),
                               windows_AIT201_outside_lit=dict(total=int(f_AIT_lit.sum()), normal=int((f_AIT_lit & (y == 0)).sum()),
                                                               attack=int((f_AIT_lit & (y == 1)).sum())),
                               windows_AIT201_outside_proper_train_range=dict(total=int(f_AIT_proper.sum()),
                                                                              normal=int((f_AIT_proper & (y == 0)).sum())),
                               drift_normal_windows_removed=int(rem_lit.sum()), n_test_normal=int((y == 0).sum()))
print(f"[S1] P102 states: {P102_states}; new-in-test state {P102_new}; AIT201 ranges {AIT_rng}", flush=True)
print(f"[S1] windows: P102-new-state {f_P102.sum()} (normal {(f_P102&(y==0)).sum()}, attack {(f_P102&(y==1)).sum()}); "
      f"AIT201 outside lit range {f_AIT_lit.sum()} (normal {(f_AIT_lit&(y==0)).sum()}); "
      f"AIT201 outside PROPER-train range {f_AIT_proper.sum()} (normal {(f_AIT_proper&(y==0)).sum()}); "
      f"LIT drift-normals removed {rem_lit.sum()} of {(y==0).sum()}", flush=True)

# ------------------------------------------------------------------ STEP 2: EDA labeler (row-level KS, per train)
tn = Xa10[ya10 == 0]
def shift_table(tr, skip=()):
    rows = []
    for j, c in enumerate(sens):
        if c in skip:
            continue
        a, b = tr[:, j], tn[:, j]
        ks = float(ks_2samp(a, b).statistic)
        oos = float(((b < a.min()) | (b > a.max())).mean())               # test-normal mass outside train support
        rows.append(dict(ch=c, KS=round(ks, 3), out_of_train_support=round(oos, 3),
                         dmu_sd=round(float(abs(b.mean() - a.mean()) / (a.std() + 1e-9)), 2),
                         train=[round(float(a.min()), 2), round(float(a.max()), 2)],
                         test_normal=[round(float(b.min()), 2), round(float(b.max()), 2)]))
    rows.sort(key=lambda r: -r["KS"]); return rows
eda_proper, eda_leaked = shift_table(tr_proper, skip=nan_cols), shift_table(tr_leaked)
rank = lambda tab, c: (1 + [r["ch"] for r in tab].index(c)) if c in [r["ch"] for r in tab] else "n/a (channel unrecorded in mirror Normal_v0)"
OUT["step2_eda"] = dict(rule=f"channel flagged if KS(train, test-normal) > {KS_FLAG}; window flagged if its window-mean "
                             "of a flagged channel lies outside that channel's train [min,max]",
                        proper_train_top10=eda_proper[:10], leaked_train_top10=eda_leaked[:10],
                        rank_proper=dict(P102=rank(eda_proper, "P102"), AIT201=rank(eda_proper, "AIT201")),
                        rank_leaked=dict(P102=rank(eda_leaked, "P102"), AIT201=rank(eda_leaked, "AIT201")),
                        P102_KS_note="P102 is constant in the proper train (single state), so KS is degenerate; its shift is "
                                     "the state gain, caught by out_of_train_support = " +
                                     str([r["out_of_train_support"] for r in eda_proper if r["ch"] == "P102"][0]))
print("[S2] top-10 shifted channels, PROPER train (Normal_v0 only):", flush=True)
for r in eda_proper[:10]:
    print(f"      {r['ch']:8s} KS {r['KS']:.3f} outOfSupport {r['out_of_train_support']:.3f} train {r['train']} testN {r['test_normal']}", flush=True)
print(f"      rank P102 {rank(eda_proper,'P102')}, AIT201 {rank(eda_proper,'AIT201')}", flush=True)
print("[S2] top-10 shifted channels, LEAKED train (as used):", flush=True)
for r in eda_leaked[:10]:
    print(f"      {r['ch']:8s} KS {r['KS']:.3f} outOfSupport {r['out_of_train_support']:.3f} train {r['train']} testN {r['test_normal']}", flush=True)
print(f"      rank P102 {rank(eda_leaked,'P102')}, AIT201 {rank(eda_leaked,'AIT201')}", flush=True)
flag_ch = [r["ch"] for r in eda_proper if r["KS"] > KS_FLAG]
f_eda = np.zeros(len(y), bool); per_ch = {}
for c in flag_ch + ["P102"]:
    j = sens.index(c); lo, hi = tr_proper[:, j].min(), tr_proper[:, j].max()
    f = (wmean[:, j] < lo) | (wmean[:, j] > hi) if c != "P102" else f_P102
    per_ch[c] = dict(windows=int(f.sum()), normal=int((f & (y == 0)).sum())); f_eda |= f
rem_eda = (f_eda | lit_region) & (y == 0)
OUT["step2_eda"]["flagged_channels"] = flag_ch; OUT["step2_eda"]["per_channel_windows"] = per_ch
OUT["step2_eda"]["drift_normal_windows_removed"] = int(rem_eda.sum())
print(f"[S2] EDA-flagged channels (KS>{KS_FLAG}): {flag_ch}; per-channel {per_ch}; EDA(+lit) normals removed {rem_eda.sum()}", flush=True)

# ------------------------------------------------------------------ STEP 3/4/5: clean + rescore
L = int(np.ceil(W / stride)) + 1
def evaluate(rem, tag):
    assert not (rem & (y == 1)).any(), "an attack window would be removed"
    yc = y.copy(); yc[rem] = 2                                              # removed normals: neither 0 nor 1
    assert (yc == 1).sum() == (y == 1).sum() and ((yc == 1) & diff).sum() == diff.sum()
    ok = (yc == 0).sum() >= 1
    tab = {k: dict(uncleaned=unc[k], cleaned=round(au_ms(yc, v, diff), 3) if ok else float("nan"),
                   removed_pct=pct_of_removed(y, v, rem)) for k, v in meth.items()}
    for k in tab: tab[k]["delta"] = round(tab[k]["cleaned"] - tab[k]["uncleaned"], 3)
    if (yc == 0).sum() < 2 * L:
        print(f"  [{tag}] only {(yc==0).sum()} normals left: bootstrap skipped", flush=True)
        bt = bt_null = dict(diff=None, diff_ci=None, p_le_0=None, n_episodes=None, note="too few normals left")
    else:
        EF.RNG = np.random.default_rng(0)
        bt = EF.boot(yc, meth["HCcoh+LatAD (community headline)"], meth["boosted_LOO"], diff, L, reps=REPS)
        EF.RNG = np.random.default_rng(0)
        bt_null = EF.boot(yc, meth["null+HC"], meth["boosted_LOO"], diff, L, reps=REPS)
    res = dict(n_normal_removed=int(rem.sum()), n_normal_kept=int((yc == 0).sum()), n_attack_removed=0,
               n_diff=int(diff.sum()), auroc_difficult=tab,
               boot_HCcohLatAD_minus_boosted=bt, boot_nullHC_minus_boosted=bt_null,
               removed_time_footprint=dict(first_window_idx=int(np.where(rem)[0].min()) if rem.any() else None,
                                           last_window_idx=int(np.where(rem)[0].max()) if rem.any() else None,
                                           frac_of_normals_in_first_third=round(float((rem & (np.arange(len(y)) < len(y) // 3)).sum() / max(1, rem.sum())), 3)))
    print(f"\n=== {tag}: removed {rem.sum()} normal windows, kept {(yc==0).sum()}; attacks intact ({(yc==1).sum()}), diff {diff.sum()} ===", flush=True)
    for k, v in tab.items():
        print(f"  {k:34s} unclean {v['uncleaned']:.3f}  clean {v['cleaned']:.3f}  delta {v['delta']:+.3f}  removed-normals pct {v['removed_pct']:.3f}", flush=True)
    print(f"  boot HCcoh+LatAD - boosted [Difficult, cleaned]: diff {bt['diff']} CI {bt['diff_ci']} P(<=0)={bt['p_le_0']} eps {bt['n_episodes']}", flush=True)
    print(f"  boot null+HC     - boosted [Difficult, cleaned]: diff {bt_null['diff']} CI {bt_null['diff_ci']} P(<=0)={bt_null['p_le_0']}", flush=True)
    return res

EF.RNG = np.random.default_rng(0)
OUT["uncleaned_boot_HCcohLatAD_minus_boosted"] = EF.boot(y, meth["HCcoh+LatAD (community headline)"], meth["boosted_LOO"], diff, L, reps=REPS)
print(f"[unc] boot HCcoh+LatAD - boosted [Difficult]: {OUT['uncleaned_boot_HCcohLatAD_minus_boosted']}", flush=True)
OUT["variant_literature_conservative"] = evaluate(rem_lit, "LITERATURE (P102 new state | AIT201 outside [251,272])")
OUT["variant_eda_broad"] = evaluate(rem_eda, f"EDA broad (lit + channels KS>{KS_FLAG} out of proper-train support)")
# sub-variants for attribution (which literature rule does the work)
OUT["variant_lit_P102_only"] = evaluate(f_P102 & (y == 0), "P102-only")
OUT["variant_lit_AIT201_only"] = evaluate(f_AIT_lit & (y == 0), "AIT201-only")
OUT["elapsed_s"] = round(time.time() - t0, 1)
json.dump(OUT, open(os.path.join(HERE, "drift_clean_swat.json"), "w"), indent=1)
print("saved drift_clean_swat.json", flush=True)
