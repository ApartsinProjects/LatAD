"""EDA of the real CPS datasets (SKAB, HAI, WADI) — how HARD are the anomalies, and
how strongly is the MIIM structure (A1-A10) present? Answers, per dataset:

  Q2 simplicity   : does a TRIVIAL univariate rule (max |z| over channels) already
                    catch the anomalies? AUROC / TPR@5%FPR of the trivial detector,
                    and the fraction of anomaly windows that are trivially separable
                    (score above the 99th percentile of normal).
  Q3 difficulty   : strip the trivially-separable anomalies, re-score LOF/IF/VaDE on
                    the HARD residual — where do the models actually differ?
  Q4 A1-A10       : #effective modes (BIC over K), cluster-size distribution (Zipf =
                    A1 mode-explosion + A2 imbalance), multimodality (silhouette,
                    BIC(1) vs BIC(K*)), hard-envelope occupancy (A2), and a NORMAL-vs-
                    ABNORMAL geometry split: each anomaly tagged OUT / BETWEEN / RARE.
  Q6 per-cluster  : assign every test window (normal+anomaly) to its nearest mode and
                    report per-cluster FPR/TPR; scatter cluster size vs error.

Raw per-timestep channels (standardised on normal) feed the trivial/geometry parts;
windowed stats features (what the models see) feed the detectors. Figures -> eda_figs/.
"""
from __future__ import annotations
import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glob import glob
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, silhouette_score
from winfeat import window_features

FIGDIR = "eda_figs"; os.makedirs(FIGDIR, exist_ok=True)


# ----------------------------------------------------------------------------- raw loaders
def _raw_wadi(downsample=10):
    import pandas as pd
    DIR = "../datasets/itrust/WADI/WADI.A2_19 Nov 2019"
    nrm = pd.read_csv(f"{DIR}/WADI_14days_new.csv"); atk = pd.read_csv(f"{DIR}/WADI_attackdataLABLE.csv", skiprows=1)
    nrm.columns = [c.strip() for c in nrm.columns]; atk.columns = [c.strip() for c in atk.columns]
    lc = atk.columns[-1]; meta = set(list(atk.columns[:3]) + list(nrm.columns[:3]) + [lc])
    sens = [c for c in atk.columns if c not in meta and c in nrm.columns and nrm[c].isna().mean() < 0.5]
    prep = lambda df: df[sens].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0).values.astype(np.float32)
    Xn = prep(nrm)[::downsample]; Xa = prep(atk)[::downsample]
    ya = (pd.to_numeric(atk[lc], errors="coerce").values == -1).astype(int)[::downsample]
    return Xn, Xa, ya, sens


def _raw_skab():
    import pandas as pd
    CH = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
          "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"]
    root = "datasets/SKAB"
    groups = [sorted(glob(f"{root}/data/{g}/*.csv")) for g in ("valve1", "valve2", "other")]
    tr, te = [], []
    for g in groups:
        tr += g[0::2]; te += g[1::2]
    def cat(files):
        Xs, ys = [], []
        for f in files:
            d = pd.read_csv(f, sep=";"); Xs.append(d[CH].to_numpy(np.float64))
            ys.append(d["anomaly"].to_numpy(np.float64) if "anomaly" in d.columns else np.zeros(len(d)))
        return np.concatenate(Xs), np.concatenate(ys)
    Xn_all, yn = cat(tr); Xa, ya = cat(te)
    Xn = Xn_all[yn == 0]                                    # pure-normal train rows
    return Xn.astype(np.float32), Xa.astype(np.float32), ya.astype(int), CH


def _raw_hai():
    import pandas as pd
    root = "datasets/hai/hai-20.07"
    LAB = {"time", "attack", "attack_P1", "attack_P2", "attack_P3"}
    rd = lambda p: pd.read_csv(p, sep=";", compression="gzip").rename(columns=str.strip)
    trf = sorted(glob(f"{root}/train*.csv.gz")); tef = sorted(glob(f"{root}/test*.csv.gz"))
    sens = [c for c in rd(trf[0]).columns if c not in LAB]
    Xn = np.concatenate([rd(f)[sens].to_numpy(np.float64) for f in trf])
    te = [rd(f) for f in tef]
    Xa = np.concatenate([d[sens].to_numpy(np.float64) for d in te])
    ya = np.concatenate([d["attack"].to_numpy(np.float64) for d in te]).astype(int)
    return Xn.astype(np.float32), Xa.astype(np.float32), ya, sens


def _raw_swat(downsample=10, warmup_drop=0.02, test_normal_frac=0.2):
    import pandas as pd, swat as _s
    nrm, atk = _s._read("normal.csv"), _s._read("attack.csv")
    drop = {"Timestamp", _s.LABEL}
    sens = [c for c in nrm.columns if c not in drop and c in atk.columns
            and pd.api.types.is_numeric_dtype(nrm[c])]
    prep = lambda df: df[sens].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0).values.astype(np.float32)
    Xn_all = prep(nrm); Xa = prep(atk)[::downsample]
    Xn_all = Xn_all[int(len(Xn_all) * warmup_drop):][::downsample]
    cut = int(len(Xn_all) * (1 - test_normal_frac))
    Xn_tr, Xn_te = Xn_all[:cut], Xn_all[cut:]
    Xa_raw = np.concatenate([Xn_te, Xa])
    ya = np.concatenate([np.zeros(len(Xn_te), int), np.ones(len(Xa), int)])
    return Xn_tr, Xa_raw, ya, sens


# (window, stride) per dataset. Unified to (60, 30): W=60 is empirically the best SKAB window
# (0.671 vs 0.66@W30 / 0.61@W120) and HAI stride 60->30 adds overlap; consistent across all so the
# table matches the sweep scripts (SKAB at the old W=20 spuriously lost to USAD).
RAW = {"SKAB": (_raw_skab, 60, 30), "HAI": (_raw_hai, 60, 30), "WADI": (_raw_wadi, 60, 30),
       "SWaT": (_raw_swat, 60, 30)}
# clip is WADI-specific: WADI has glitch/shifted channels (up to 1e8 sigma); HAI's big
# excursions are REAL attack signal (clipping hurts), SKAB is already in range (no-op).
CLIP = {"WADI": 10.0, "HAI": None, "SKAB": None, "SWaT": None}


def load(name, clip="auto"):
    fn, W, stride = RAW[name]
    if clip == "auto":
        clip = CLIP.get(name)
    Xn, Xa, ya, ch = fn()
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-8
    Xn, Xa = (Xn - mu) / sd, (Xa - mu) / sd                 # standardise on normal
    if clip:                                                # A2 envelope: cap sensor-fault/glitch spikes
        Xn, Xa = np.clip(Xn, -clip, clip), np.clip(Xa, -clip, clip)
    def win(X, y=None):
        Xw, yl, idx = [], [], []
        for i in range(0, len(X) - W + 1, stride):
            Xw.append(window_features(X[i:i + W], "stats"))
            idx.append(i)
            if y is not None:
                yl.append(int(y[i:i + W].mean() > 0.05))
        return (np.asarray(Xw, np.float32), np.asarray(idx, int),
                np.asarray(yl, int) if y is not None else None)
    Xn_w, _, _ = win(Xn); Xa_w, idx_a, ya_w = win(Xa, ya)
    return dict(name=name, ch=ch, W=W, Xn_raw=Xn, Xa_raw=Xa, ya_raw=ya,
                Xn_w=Xn_w, Xa_w=Xa_w, ya_w=ya_w, idx_a=idx_a)


# ----------------------------------------------------------------------------- metrics
def au_tpr(y, s):
    au = roc_auc_score(y, s); thr = np.quantile(s[y == 0], 0.95)
    return au, float((s[y == 1] > thr).mean())


def trivial_score(Xw, ch, W):
    """max |z| over the LEVEL features (per-channel window means) — the simplest
    'is any channel out of its normal range' rule, on the exact windows models see."""
    C = len(ch)
    return np.abs(Xw[:, :C]).max(1)                         # feat block 0 = per-channel mean


def main(names):
    print(f"{'dataset':<8}{'Ntr':>7}{'Nte':>7}{'anom':>7}{'triv_AU':>9}{'triv_TPR':>9}"
          f"{'triv_sep%':>10}{'#modes':>8}{'silhou':>8}{'BIC_gain':>9}")
    print("-" * 92)
    summary = {}
    for nm in names:
        D = load(nm)
        Xn_w, Xa_w, ya = D["Xn_w"], D["Xa_w"], D["ya_w"]
        y = np.r_[np.zeros(len(Xn_w)), ya]; X = np.r_[Xn_w, Xa_w]

        # --- Q2 trivial detector on model-visible windows ---
        st = trivial_score(X, D["ch"], D["W"])
        tri_au, tri_tpr = au_tpr(y, st)
        sep = st[len(Xn_w):][ya == 1] > np.quantile(st[:len(Xn_w)], 0.99)
        triv_sep = float(sep.mean()) if ya.sum() else float("nan")

        # --- Q4 modes: BIC over K on NORMAL windows (PCA-reduced for numerical stability),
        #     size distribution, silhouette ---
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(30, Xn_w.shape[1]), random_state=0).fit(Xn_w)
        Zn = pca.transform(Xn_w).astype(np.float64)
        Ks = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
        bic = {}
        for K in Ks:
            g = GaussianMixture(K, covariance_type="diag", random_state=0, max_iter=100, reg_covar=1e-2).fit(Zn)
            bic[K] = g.bic(Zn)
        Kbest = min(bic, key=bic.get)
        bic_gain = (bic[1] - bic[Kbest]) / abs(bic[1])       # relative BIC improvement over unimodal
        Kop = int(min(max(Kbest, 2), 24, len(Zn) // 60))     # keep clusters populated for per-cluster stats
        gK = GaussianMixture(Kop, covariance_type="diag", random_state=0, reg_covar=1e-2).fit(Zn)
        lab = gK.predict(Zn); sizes = np.bincount(lab, minlength=gK.n_components)
        eff = int((sizes / sizes.sum() > 0.01).sum())        # modes with >1% mass
        sil = silhouette_score(Zn[:6000], lab[:6000]) if len(np.unique(lab)) > 1 else float("nan")

        print(f"{nm:<8}{len(Xn_w):>7}{len(Xa_w):>7}{ya.mean():>7.3f}{tri_au:>9.3f}{tri_tpr:>9.3f}"
              f"{triv_sep:>9.1%}{eff:>8}{sil:>8.3f}{bic_gain:>9.3f}")
        summary[nm] = dict(D=D, y=y, X=X, st=st, gK=gK, lab=lab, sizes=sizes, Kbest=Kbest,
                           pca=pca, tri_au=tri_au, tri_tpr=tri_tpr, triv_sep=triv_sep,
                           sil=sil, bic_gain=bic_gain, eff=eff, bic=bic, Ks=Ks)
    return summary


def fit_detectors(D, seed=0):
    """Scores on train-normal and test for the trivial rule, IF, LOF, and our VaDE."""
    from models_vade import train_vade
    Xn, Xa, ch, W = D["Xn_w"], D["Xa_w"], D["ch"], D["W"]
    tr, te = {}, {}
    tr["trivial"], te["trivial"] = trivial_score(Xn, ch, W), trivial_score(Xa, ch, W)
    IFm = IsolationForest(n_estimators=200, random_state=seed).fit(Xn)
    tr["IF"], te["IF"] = -IFm.decision_function(Xn), -IFm.decision_function(Xa)
    LF = LocalOutlierFactor(30, novelty=True).fit(Xn)
    tr["LOF"], te["LOF"] = -LF.decision_function(Xn), -LF.decision_function(Xa)
    v = train_vade(Xn, n_clusters=min(24, max(8, D["Kbest"])), latent_dim=10, epochs=40,
                   warmup=8, seed=seed, device="cpu")
    v.fit_residual_whitener(Xn)
    tr["VaDE"], te["VaDE"] = v.anomaly_score(Xn), v.anomaly_score(Xa)
    return tr, te


def full(names):
    S = main(names)
    rows_diff, rows_pc, rows_err = [], [], []
    for nm in names:
        s = S[nm]; D = s["D"]; ya = D["ya_w"]; gK, pca = s["gK"], s["pca"]
        D["Kbest"] = s["Kbest"]
        tr, te = fit_detectors(D)
        y = s["y"]; ntr = len(D["Xn_w"])

        # ---------- Q4 geometry: NLL + responsibility-entropy on the mode model ----------
        Ztr = pca.transform(D["Xn_w"]).astype(np.float64)
        Zte = pca.transform(D["Xa_w"]).astype(np.float64)
        nll_tr = -gK.score_samples(Ztr); nll_te = -gK.score_samples(Zte)
        R = gK.predict_proba(Zte); ent = -(R * np.log(R + 1e-12)).sum(1)     # betweenness
        assign_te = gK.predict(Zte); mass = s["sizes"] / s["sizes"].sum()
        # tag each anomaly
        a = ya == 1
        out = nll_te > np.quantile(nll_tr, 0.99)                              # A2 envelope violation
        between = (~out) & (ent > np.quantile(-(gK.predict_proba(Ztr) *
                   np.log(gK.predict_proba(Ztr) + 1e-12)).sum(1), 0.99))      # A3 pocket
        rare = (~out) & (~between) & (mass[assign_te] < 0.01)                 # A1 rare mode
        rest = a & ~(out | between | rare)
        tags = dict(OUT=float((a & out).sum() / a.sum()),
                    BETWEEN=float((a & between).sum() / a.sum()),
                    RARE=float((a & rare).sum() / a.sum()),
                    IN_COMMON=float((rest).sum() / a.sum()))

        # ---------- Q3 difficulty: strip trivially-separable anomalies, re-score ----------
        easy = np.zeros(len(ya), bool)
        if a.sum():
            easy = te["trivial"] > np.quantile(tr["trivial"], 0.99)
        # HARD subset = all normal + only the anomalies NOT trivially separable
        msk = (ya == 0) | (~easy)
        row = {"dataset": nm, "n_easy": int((a & easy).sum()), "n_hard": int((a & ~easy).sum())}
        for m in ["trivial", "IF", "LOF", "VaDE"]:
            full_au = roc_auc_score(ya, te[m])
            hard_au = roc_auc_score(ya[msk], te[m][msk]) if (ya[msk] == 1).any() else float("nan")
            row[f"{m}_full"] = round(full_au, 3); row[f"{m}_hard"] = round(hard_au, 3)
        rows_diff.append((row, tags))

        # ---------- Q6 per-cluster metric (assign ALL test windows to nearest mode) ----------
        det = "VaDE"
        sc_te = te[det]; thr = np.quantile(tr[det], 0.95)                     # global 5% FPR
        clu = gK.predict(Zte); Kc = gK.n_components
        pc = []
        for k in range(Kc):
            m = clu == k
            if m.sum() < 10:
                continue
            yk, sk = ya[m], sc_te[m]
            fpr = float((sk[yk == 0] > thr).mean()) if (yk == 0).any() else float("nan")
            tpr = float((sk[yk == 1] > thr).mean()) if (yk == 1).any() else float("nan")
            auc = roc_auc_score(yk, sk) if (yk == 0).any() and (yk == 1).any() else float("nan")
            pc.append(dict(k=k, size=int(m.sum()), mass=float(mass[k]), anom=float((yk == 1).mean()),
                           fpr=fpr, tpr=tpr, auc=auc))
        rows_pc.append((nm, pc))

        # ---------- Q7 error root-cause: worst FP / FN, characterised by A1-A3 ----------
        err = _error_examples(D, te["VaDE"], tr["VaDE"], gK, assign_te, mass, nll_tr, nll_te, ent)
        rows_err.append((nm, err))

        _figures(nm, s, tr, te, ya, nll_tr, nll_te, ent, tags, pc)

    _print_diff(rows_diff); _print_pc(rows_pc); _print_err(rows_err)


def _atype(nll_pct, ent_z, m):
    if nll_pct > 0.99:
        return "OUT (A2 envelope)"
    if m < 0.01:
        return "RARE-mode normal (A1)"
    if ent_z > 0.9:
        return "BETWEEN modes (A3)"
    return "in a common mode"


def _error_examples(D, sc_te, sc_tr, gK, assign_te, mass, nll_tr, nll_te, ent, topn=4):
    """Top false-positives (normal, high score) and false-negatives (anomaly, missed)
    with the dominant channels and the A1-A3 type of the window."""
    ya, W, ch = D["ya_w"], D["W"], D["ch"]; C = len(ch)
    Xa_raw, idx = D["Xa_raw"], D["idx_a"]
    thr = np.quantile(sc_tr, 0.95)
    nll_rank = (nll_te[:, None] > nll_tr[None, :]).mean(1) if len(nll_tr) < 4000 else \
        np.searchsorted(np.sort(nll_tr), nll_te) / len(nll_tr)
    ent_tr_q = np.quantile(ent, 0.9)                       # crude betweenness scale
    def describe(i):
        w = Xa_raw[idx[i]:idx[i] + W]; zmean = w.mean(0)   # per-channel window-mean z
        top = np.argsort(-np.abs(zmean))[:3]
        chans = ", ".join(f"{ch[j]}={zmean[j]:+.1f}σ" for j in top)
        at = _atype(nll_rank[i], (ent[i] > ent_tr_q) * 1.0, mass[assign_te[i]])
        return dict(k=int(assign_te[i]), mass=float(mass[assign_te[i]]), score=float(sc_te[i]),
                    nll_pct=float(nll_rank[i]), atype=at, chans=chans)
    fp = [i for i in range(len(ya)) if ya[i] == 0 and sc_te[i] > thr]
    fp = sorted(fp, key=lambda i: -sc_te[i])[:topn]
    fn = [i for i in range(len(ya)) if ya[i] == 1 and sc_te[i] <= thr]
    fn = sorted(fn, key=lambda i: sc_te[i])[:topn]
    return dict(FP=[describe(i) for i in fp], FN=[describe(i) for i in fn],
                n_fp=int(((ya == 0) & (sc_te > thr)).sum()), n_fn=int(((ya == 1) & (sc_te <= thr)).sum()))


def _print_err(rows):
    print("\n### Q7 error root-cause (VaDE @ global 5% FPR) — worst false alarms & misses")
    for nm, e in rows:
        print(f"\n{nm}: {e['n_fp']} false alarms, {e['n_fn']} missed anomalies")
        print("  false ALARMS (normal flagged) — why did a valid window look anomalous?")
        for d in e["FP"]:
            print(f"    clu {d['k']:>2} (mass {d['mass']:.1%}) NLL@{d['nll_pct']:.0%} "
                  f"[{d['atype']}]  {d['chans']}")
        print("  MISSES (anomaly not caught) — why did an attack look normal?")
        for d in e["FN"]:
            print(f"    clu {d['k']:>2} (mass {d['mass']:.1%}) NLL@{d['nll_pct']:.0%} "
                  f"[{d['atype']}]  {d['chans']}")


def _figures(nm, s, tr, te, ya, nll_tr, nll_te, ent, tags, pc):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8)); fig.suptitle(f"{nm} — anomaly & mode EDA", fontsize=14)
    a = ya == 1
    # (0,0) trivial-score separability
    ax[0, 0].hist(s["st"][:len(s["D"]["Xn_w"])], 60, alpha=.6, label="normal", density=True, color="steelblue")
    ax[0, 0].hist(s["st"][len(s["D"]["Xn_w"]):][a], 60, alpha=.6, label="anomaly", density=True, color="crimson")
    ax[0, 0].set_title(f"trivial max|z| (AUROC {s['tri_au']:.2f})"); ax[0, 0].legend(); ax[0, 0].set_yscale("log")
    # (0,1) cluster-size distribution (Zipf)
    sz = np.sort(s["sizes"])[::-1]
    ax[0, 1].loglog(np.arange(1, len(sz) + 1), sz + 1, "o-", color="darkorange")
    ax[0, 1].set_title(f"mode-size rank (A1/A2), {s['eff']} modes"); ax[0, 1].set_xlabel("rank"); ax[0, 1].set_ylabel("size")
    # (0,2) BIC vs K
    ax[0, 2].plot(s["Ks"], [s["bic"][k] for k in s["Ks"]], "s-", color="teal")
    ax[0, 2].set_title(f"BIC vs K (Kbest={s['Kbest']})"); ax[0, 2].set_xlabel("K")
    # (1,0) mode NLL normal vs anomaly (A2 envelope)
    ax[1, 0].hist(nll_tr, 60, alpha=.6, label="normal", density=True, color="steelblue")
    ax[1, 0].hist(nll_te[a], 60, alpha=.6, label="anomaly", density=True, color="crimson")
    ax[1, 0].set_title("mode NLL (A2 hard envelope)"); ax[1, 0].legend(); ax[1, 0].set_yscale("log")
    # (1,1) anomaly A-type composition
    ks = list(tags); ax[1, 1].bar(ks, [tags[k] for k in ks], color=["crimson", "purple", "darkorange", "gray"])
    ax[1, 1].set_title("anomaly type (A1-A3 geometry)"); ax[1, 1].set_ylim(0, 1)
    for i, k in enumerate(ks):
        ax[1, 1].text(i, tags[k] + .02, f"{tags[k]:.0%}", ha="center")
    # (1,2) per-cluster size vs error scatter
    if pc:
        sizes = [p["size"] for p in pc]; err = [1 - (p["auc"] if p["auc"] == p["auc"] else .5) for p in pc]
        anom = [p["anom"] for p in pc]
        sctr = ax[1, 2].scatter(sizes, err, c=anom, s=60, cmap="viridis", edgecolor="k")
        ax[1, 2].set_xscale("log"); ax[1, 2].set_xlabel("cluster size"); ax[1, 2].set_ylabel("1 - AUROC")
        ax[1, 2].set_title("per-cluster error vs size"); plt.colorbar(sctr, ax=ax[1, 2], label="anom frac")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); p = f"{FIGDIR}/{nm}_eda.png"; fig.savefig(p, dpi=90); plt.close(fig)
    print(f"  saved {p}")


def _print_diff(rows):
    print("\n### Q3 difficulty stratification (AUROC full vs HARD = easy anomalies removed)")
    print(f"{'dataset':<8}{'easy':>6}{'hard':>6}" + "".join(f"{m:>16}" for m in ["trivial", "IF", "LOF", "VaDE"]))
    for row, tags in rows:
        cells = "".join(f"{row[m+'_full']:>7}/{row[m+'_hard']:<8}" for m in ["trivial", "IF", "LOF", "VaDE"])
        print(f"{row['dataset']:<8}{row['n_easy']:>6}{row['n_hard']:>6}{cells}")
    print("\n### Q4 anomaly geometry (share of anomalies by A1-A3 type)")
    print(f"{'dataset':<8}{'OUT(A2)':>9}{'BETWEEN(A3)':>13}{'RARE(A1)':>10}{'IN_COMMON':>11}")
    for row, tags in rows:
        print(f"{row['dataset']:<8}{tags['OUT']:>9.0%}{tags['BETWEEN']:>13.0%}{tags['RARE']:>10.0%}{tags['IN_COMMON']:>11.0%}")


def _print_pc(rows):
    print("\n### Q6 per-cluster robustness (VaDE @ global 5% FPR)")
    for nm, pc in rows:
        if not pc:
            continue
        anom_cl = [p for p in pc if p["anom"] > 0.02]
        worst = sorted(anom_cl, key=lambda p: (p["auc"] if p["auc"] == p["auc"] else 1))[:3]
        print(f"{nm}: {len(pc)} populated clusters; "
              f"mean per-cluster FPR {np.nanmean([p['fpr'] for p in pc]):.3f}, "
              f"worst-cluster FPR {np.nanmax([p['fpr'] for p in pc]):.3f}; "
              f"anomaly-bearing clusters {len(anom_cl)}")
        for p in worst:
            print(f"    cluster {p['k']:>2} size={p['size']:<5} anom={p['anom']:.0%} "
                  f"AUROC={p['auc'] if p['auc']==p['auc'] else float('nan'):.3f} TPR={p['tpr']:.2f} FPR={p['fpr']:.2f}")


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["SKAB", "HAI", "WADI"]
    mode = sys.argv[2] if len(sys.argv) > 2 else "full"
    (full if mode == "full" else main)(names)
