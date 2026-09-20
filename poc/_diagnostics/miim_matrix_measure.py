"""MIIM assumption-validation matrix (A1-A10): per-dataset OFFLINE measurements.

Model-light, local CPU. Reuses bundle arrays + cached VaDE latents. Writes
_diagnostics/miim_matrix.json incrementally (one dataset at a time, resumable).
Does NOT touch the paper or models_vade.py.

Feature convention (winfeat 'stats', W=60/stride=30): 6 blocks per channel
[mean, std, min, max, trend, range]; the first C/6 block = per-channel window means.
"""
from __future__ import annotations
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats as sps
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, silhouette_score
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity

HERE = os.path.dirname(os.path.abspath(__file__))
POC = os.path.dirname(HERE)
sys.path.insert(0, POC)
OUT = os.path.join(HERE, "miim_matrix.json")
LOG = lambda *a: print(*a, flush=True)
SEED = 0
KS = [k for k in [1, 2, 4, 8, 16, 32, 64, 128] if k <= int(os.environ.get("MIIM_KMAX", "128"))]
VADE_CFG = {"SKAB": (16, 6), "WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16)}
REG = 1e-2      # diag-GMM variance floor (features are z-scored; 1e-4 lets components collapse onto
                # near-constant features and the held-out log-lik then blows up to -1e4)


def hac_communities(Z, maxsz=25, minsz=3):
    """E3 rule (guided_ensemble.hac_communities): average-linkage HAC on 1-|r| over the active
    per-channel window means; every internal node with minsz<=size<=maxsz is one community."""
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform
    C = np.nan_to_num(np.corrcoef(Z.T)); dist = 1 - np.abs(C)
    np.fill_diagonal(dist, 0.0); dist = (dist + dist.T) / 2
    L = linkage(squareform(dist, checks=False), method="average")
    _, nodes = to_tree(L, rd=True)

    def leaves(nd):
        return [nd.id] if nd.is_leaf() else leaves(nd.left) + leaves(nd.right)
    seen, comms = set(), []
    for nd in nodes:
        if not nd.is_leaf():
            lv = tuple(sorted(leaves(nd)))
            if minsz <= len(lv) <= maxsz and lv not in seen:
                seen.add(lv); comms.append(lv)
    return comms


# ----------------------------------------------------------------------------- data
def load_ds(name):
    """Return dict with Xn (train-normal windows, raw z-level features), Xa, y, ch names."""
    if name == "SKAB":
        import eda_real as E
        D = E.load("SKAB")
        return dict(Xn=D["Xn_w"].astype(np.float64), Xa=D["Xa_w"].astype(np.float64),
                    y=D["ya_w"].astype(int), nch=len(D["ch"]),
                    Xn_raw=D["Xn_raw"].astype(np.float64))
    z = np.load(os.path.join(POC, "sota_bundle", "ens_bundle", f"bundle_{name}.npz"))
    return dict(Xn=z["Xn_w"].astype(np.float64), Xa=z["Xa_w"].astype(np.float64),
                y=z["y"].astype(int), nch=int(z["nch"]), Xn_raw=None,
                maxz=z["maxz"].astype(float), maxz_thr=float(z["maxz_thr"]))


def standardize(Xn, Xa, clip=20.0):
    """Train mean/std; drop constant channels; floor near-constant sd at 1% of the
    median sd (else SWaT's near-constant features blow up to 1e3-sigma); clip |z|<=clip."""
    m, sd = Xn.mean(0), Xn.std(0)
    keep = sd > 1e-8
    sd_f = np.maximum(sd, 0.01 * np.median(sd[keep]))
    Zn = np.clip((Xn[:, keep] - m[keep]) / sd_f[keep], -clip, clip)
    Za = np.clip((Xa[:, keep] - m[keep]) / sd_f[keep], -clip, clip)
    return Zn, Za, keep


def gini(w):
    w = np.sort(np.asarray(w, float)); n = len(w)
    if n == 0 or w.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(w) / (n * w.sum()))


def part_ratio(lam):
    lam = np.clip(lam, 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


# ----------------------------------------------------------------------------- A1/A2
def bic_sweep(Ztr, Zho, ks):
    """Diagonal GMM BIC on the fit split + held-out mean log-lik per sample."""
    rows = []
    for K in ks:
        if K > len(Ztr) // 5:            # need >=5 samples / component
            break
        t = time.time()
        g = GaussianMixture(K, covariance_type="diag", reg_covar=REG, max_iter=200,
                            n_init=1, random_state=SEED).fit(Ztr)
        lho = g.score_samples(Zho)
        rows.append(dict(K=K, bic=float(g.bic(Ztr)), ll_fit=float(g.score(Ztr)),
                         ll_ho_mean=float(lho.mean()), ll_ho=float(np.median(lho)),
                         sec=round(time.time() - t, 1)))
        LOG(f"   GMM K={K:3d} BIC={rows[-1]['bic']:.4g} ll_fit={rows[-1]['ll_fit']:.3f} "
            f"ll_ho_med={rows[-1]['ll_ho']:.3f} ll_ho_mean={rows[-1]['ll_ho_mean']:.3f} ({rows[-1]['sec']}s)")
    return rows


def curve_summary(rows):
    bic = np.array([r["bic"] for r in rows]); ks = np.array([r["K"] for r in rows])
    llho = np.array([r["ll_ho"] for r in rows])
    k_bic = int(ks[np.argmin(bic)]); k_ho = int(ks[np.argmax(llho)])
    # relative BIC improvement per doubling step: (BIC_prev - BIC_K)/|BIC_prev|
    rel = [(float(bic[i - 1] - bic[i]) / abs(float(bic[i - 1]))) for i in range(1, len(bic))]
    k_sat = None
    for i, r in enumerate(rel):
        if r < 0.01:
            k_sat = int(ks[i + 1]); break
    still_improving = bool(bic[-1] < bic[-2]) if len(bic) > 1 else None
    last_rel = rel[-1] if rel else None
    return dict(K_bic=k_bic, K_heldout_ll=k_ho, K_max=int(ks[-1]),
                bic_still_improving_at_Kmax=still_improving,
                rel_gain_last_step=last_rel, rel_gain_per_step=rel, K_sat_1pct=k_sat,
                ll_ho_gain_K1_to_Kmax=float(llho[-1] - llho[0]))


# ----------------------------------------------------------------------------- main per dataset
def measure(name, witness):
    LOG(f"\n===== {name} =====")
    D = load_ds(name)
    Xn, Xa, y = D["Xn"], D["Xa"], D["y"]
    Zn, Za, keep = standardize(Xn, Xa)
    n, Dfull, Deff = Zn.shape[0], Xn.shape[1], Zn.shape[1]
    C = Dfull // 6
    LOG(f" n_train={n} n_test={len(y)} anomalies={int(y.sum())} D_raw={Dfull} D_eff={Deff} channels={D['nch']}")
    res = dict(dataset=name, n_train=n, n_test=int(len(y)), n_anom=int(y.sum()),
               D_raw=Dfull, D_eff=Deff, n_channels=D["nch"],
               n_const_features=int(Dfull - Deff))

    # interleaved-block 80/20 split of train-normal for held-out checks: every 5th block of
    # 50 windows is held out, so the hold-out covers every regime (a contiguous tail split
    # lands on a single regime and the held-out log-lik collapses for every K), while the
    # 50-window blocks keep leakage from the 50%-overlapping neighbour windows to the block edges.
    blk = np.arange(n) // 50
    ho_mask = (blk % 5) == 4
    Ztr, Zho = Zn[~ho_mask], Zn[ho_mask]

    # ---------------- A1 / A2: BIC sweep + covariance participation ratio
    LOG(" [A1/A2] BIC sweep")
    rows = bic_sweep(Ztr, Zho, KS)
    cs = curve_summary(rows)
    ev = np.linalg.eigvalsh(np.cov(Zn.T))
    res["A1"] = dict(K_bic=cs["K_bic"], K_heldout_ll_median=cs["K_heldout_ll"], K_max=cs["K_max"],
                     bic_still_improving_at_Kmax=cs["bic_still_improving_at_Kmax"],
                     cov_participation_ratio=part_ratio(ev),
                     cov_PR_frac_of_Deff=part_ratio(ev) / Deff, bic_curve=rows)
    res["A2"] = dict(K_sat_1pct=cs["K_sat_1pct"], rel_gain_per_step=cs["rel_gain_per_step"],
                     rel_gain_last_step=cs["rel_gain_last_step"],
                     bic_still_improving_at_Kmax=cs["bic_still_improving_at_Kmax"],
                     ll_ho_gain_K1_to_Kmax=cs["ll_ho_gain_K1_to_Kmax"],
                     ll_ho_curve_median=[(r["K"], r["ll_ho"]) for r in rows],
                     ll_ho_curve_mean=[(r["K"], r["ll_ho_mean"]) for r in rows])
    K_bic = cs["K_bic"]
    K_mode = int(min(max(K_bic, 2), 64))   # mode count used downstream (>=2 for silhouette)

    # ---------------- A3: pull rho from witness json
    w = [r for r in witness if r["dataset"] == name and r["arm"] == "OFF"]
    res["A3"] = dict(rho=w[0]["rho"], mean_maxresp=w[0]["mean_maxresp"], basin_lam=w[0]["basin_lam"],
                     source="a3_skab_witness.json OFF row") if w else dict(rho=None, note="missing")

    # ---------------- mode assignment (KMeans at K_bic) on standardized train
    LOG(f" [modes] KMeans K={K_mode}")
    km = KMeans(K_mode, n_init=4, random_state=SEED).fit(Zn)
    lab = km.labels_
    occ = np.bincount(lab, minlength=K_mode) / n

    # ---------------- A4: within-mode kurtosis + high-K vs per-mode-Gaussian gain
    LOG(" [A4] within-mode non-Gaussianity")
    kurts = []
    for k in range(K_mode):
        M = Zn[lab == k]
        if len(M) < 30:
            continue
        sdk = M.std(0); ok = sdk > 1e-6
        kv = sps.kurtosis(M[:, ok], axis=0, fisher=True, bias=False)
        kurts.append(np.median(kv))
    med_kurt = float(np.median(kurts)) if kurts else float("nan")
    # per-mode single Gaussian = diag GMM with K_mode comps, vs K=80 diag GMM; both on fit split
    Kh = int(min(80, len(Ztr) // 5))
    gA = GaussianMixture(K_mode, covariance_type="diag", reg_covar=REG, random_state=SEED).fit(Ztr)
    gB = GaussianMixture(Kh, covariance_type="diag", reg_covar=REG, random_state=SEED).fit(Ztr)
    dl = gB.score_samples(Zho) - gA.score_samples(Zho)
    ll_gain_ho = float(np.median(dl)); ll_gain_ho_mean = float(dl.mean())
    frac_ho_improved = float((dl > 0).mean())
    auA = float(roc_auc_score(y, -gA.score_samples(Za))); auB = float(roc_auc_score(y, -gB.score_samples(Za)))
    # same contrast with the PAPER's regime count (VaDE K) as the per-mode Gaussian baseline
    Kp = VADE_CFG[name][0]
    gP = GaussianMixture(Kp, covariance_type="diag", reg_covar=REG, random_state=SEED).fit(Ztr)
    dp = gB.score_samples(Zho) - gP.score_samples(Zho)
    auP = float(roc_auc_score(y, -gP.score_samples(Za)))
    LOG(f"   paperK={Kp}: median gain {np.median(dp):+.2f} frac>0 {(dp > 0).mean():.2f} AUROC {auP:.3f}->{auB:.3f}")
    res["A4"] = dict(K_mode=K_mode, K_paper=Kp,
                     heldout_ll_gain_highK_minus_paperK_median=float(np.median(dp)),
                     frac_heldout_windows_improved_vs_paperK=float((dp > 0).mean()),
                     auroc_paperK=auP, n_modes_ge30=len(kurts), median_within_mode_excess_kurtosis=med_kurt,
                     K_high=Kh, heldout_ll_gain_highK_minus_perModeGauss_median=ll_gain_ho,
                     heldout_ll_gain_mean=ll_gain_ho_mean, frac_heldout_windows_improved=frac_ho_improved,
                     auroc_perModeGauss=auA, auroc_highK=auB, auroc_gain=auB - auA)
    LOG(f"   med_kurt={med_kurt:.2f} ll_gain_ho={ll_gain_ho:+.3f} AUROC {auA:.3f}->{auB:.3f}")

    # ---------------- A5: PCA
    LOG(" [A5] PCA")
    p = PCA().fit(Zn); cum = np.cumsum(p.explained_variance_ratio_)
    n90 = int(np.searchsorted(cum, 0.90) + 1); n95 = int(np.searchsorted(cum, 0.95) + 1)
    pr = part_ratio(p.explained_variance_)
    res["A5"] = dict(n90=n90, n95=n95, PR=pr, D_eff=Deff, n_channels=D["nch"],
                     n90_frac_Deff=n90 / Deff, n95_frac_Deff=n95 / Deff, PR_frac_Deff=pr / Deff,
                     n90_frac_channels=n90 / D["nch"], PR_frac_channels=pr / D["nch"])
    LOG(f"   n90={n90} n95={n95} PR={pr:.1f} of D_eff={Deff} (channels={D['nch']})")

    # ---------------- A6: occupancy imbalance
    LOG(" [A6] occupancy")
    o = np.sort(occ)[::-1]; o = o[o > 0]
    rk = np.arange(1, len(o) + 1)
    slope = float(np.polyfit(np.log(rk), np.log(o), 1)[0]) if len(o) > 2 else float("nan")
    res["A6"] = dict(K=K_mode, occupancy_sorted=[float(v) for v in o], zipf_slope=slope,
                     max_min_ratio=float(o[0] / o[-1]), gini=gini(o),
                     top1=float(o[0]), n_modes_below_1pct=int((o < 0.01).sum()))
    LOG(f"   slope={slope:.2f} max/min={o[0]/o[-1]:.1f} gini={gini(o):.2f} top1={o[0]:.2f}")

    # ---------------- A7: silhouettes (feature space, VaDE latent)
    LOG(" [A7] silhouette")
    rng = np.random.default_rng(SEED)
    sub = rng.choice(n, min(n, 5000), replace=False)
    sil_feat = float(silhouette_score(Zn[sub], lab[sub])) if len(np.unique(lab[sub])) > 1 else float("nan")
    a7 = dict(K_mode=K_mode, sil_feature_kmeans=sil_feat)
    zpath = os.path.join(HERE, f"e2_fable_{name}.npz")
    if os.path.exists(zpath):
        e = np.load(zpath); ztr = e["ztr"]; vl = np.argmax(e["logN_tr"] + e["logpi"], 1)
        a7["vade_K"] = int(e["logpi"].shape[0]); a7["vade_latent"] = int(ztr.shape[1])
        a7["vade_modes_used"] = int(len(np.unique(vl)))
        a7["sil_latent_vade"] = float(silhouette_score(ztr[sub], vl[sub])) if len(np.unique(vl[sub])) > 1 else float("nan")
        a7["sil_latent_kmeansLabels"] = float(silhouette_score(ztr[sub], lab[sub]))
        a7["sil_feature_vadeLabels"] = float(silhouette_score(Zn[sub], vl[sub])) if len(np.unique(vl[sub])) > 1 else float("nan")
        resp = np.exp(e["logN_tr"] + e["logpi"] - (e["logN_tr"] + e["logpi"]).max(1, keepdims=True))
        resp /= resp.sum(1, keepdims=True)
        a7["vade_frac_maxresp_lt_0.5_train"] = float((resp.max(1) < 0.5).mean())
        a7["latent_source"] = os.path.basename(zpath)
    else:
        # SKAB: no cached latent -> train the reported (K=16, latent=6) VaDE on 400 windows (seconds on CPU)
        try:
            from models_vade import train_vade
            K, LD = VADE_CFG[name]
            Xs = ((Xn - Xn.mean(0)) / (Xn.std(0) + 1e-8)).astype(np.float32)
            v = train_vade(Xs, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=SEED, device="cpu")
            R = np.asarray(v._responsibilities(Xs)); vl = R.argmax(1)
            ztr = np.asarray(v._encode_mean(Xs))
            a7["vade_K"] = K; a7["vade_latent"] = LD; a7["vade_modes_used"] = int(len(np.unique(vl)))
            a7["vade_frac_maxresp_lt_0.5_train"] = float((R.max(1) < 0.5).mean())
            a7["sil_feature_vadeLabels"] = float(silhouette_score(Zn[sub], vl[sub])) if len(np.unique(vl[sub])) > 1 else float("nan")
            if ztr is not None:
                a7["sil_latent_vade"] = float(silhouette_score(ztr[sub], vl[sub])) if len(np.unique(vl[sub])) > 1 else float("nan")
                a7["sil_latent_kmeansLabels"] = float(silhouette_score(ztr[sub], lab[sub]))
            # A10 helper: fraction of TEST anomaly windows that fall between modes (max-resp<0.5)
            Xas = ((Xa - Xn.mean(0)) / (Xn.std(0) + 1e-8)).astype(np.float32)
            Ra = np.asarray(v._responsibilities(Xas))
            a7["test_anom_frac_between_modes"] = float((Ra.max(1)[y == 1] < 0.5).mean())
            a7["test_normal_frac_between_modes"] = float((Ra.max(1)[y == 0] < 0.5).mean())
            a7["latent_source"] = "fresh VaDE (K=16,latent=6,seed0,epochs40)"
        except Exception as ex:
            a7["latent_error"] = repr(ex)
    res["A7"] = a7
    LOG(f"   {a7}")

    # ---------------- A8: correlation graph on train-normal window means (level block)
    LOG(" [A8] correlation communities")
    Mn = Xn[:, :C]; sdm = Mn.std(0); live = sdm > 1e-8
    M = (Mn[:, live] - Mn[:, live].mean(0)) / sdm[live]
    A_half = M[: len(M) // 2]                       # E3 used the first half of train-normal
    Cc = np.corrcoef(A_half.T); np.fill_diagonal(Cc, 0.0); Cc = np.nan_to_num(Cc)
    Cfull = np.corrcoef(M.T); np.fill_diagonal(Cfull, 0.0); Cfull = np.nan_to_num(Cfull)

    def comps(Cm, thr, minsz):
        G = nx.from_numpy_array((np.abs(Cm) > thr).astype(int))
        cc = [c for c in nx.connected_components(G) if len(c) >= minsz]
        return G, cc

    G7, cc7 = comps(Cc, 0.7, 3)
    G5, cc5 = comps(Cfull, 0.5, 2)
    G5_1, cc5_1 = comps(Cfull, 0.5, 1)
    gm = list(greedy_modularity_communities(G5)) if G5.number_of_edges() > 0 else []
    mod = float(modularity(G5, gm)) if gm else float("nan")
    iso5 = int(sum(1 for c in cc5_1 if len(c) == 1))
    hac = hac_communities(M)                                # E3 nested-HAC rule (45/28/26 in the paper)
    hac_flat = [c for c in hac if not any(set(c) < set(o) for o in hac)]   # maximal (non-nested) ones
    # channel-type heterogeneity: near-discrete vs continuous channels
    if D["Xn_raw"] is not None:
        Xr = D["Xn_raw"]; nun = np.array([len(np.unique(np.round(Xr[:, j], 6))) for j in range(Xr.shape[1])])
        src = "raw samples"
    else:
        nun = np.array([len(np.unique(np.round(Mn[:, j], 4))) for j in range(C)])
        src = "window means (raw samples not in bundle)"
    n_const = int((sdm <= 1e-8).sum())
    n_disc = int(((nun <= 10) & live).sum())
    n_cont = int(C - n_const - n_disc)
    res["A8"] = {"n_channels": C, "n_live": int(live.sum()), "n_const": n_const,
                 "E3_hac_nested_communities_3to25": len(hac),
                 "E3_hac_maximal_communities": len(hac_flat),
                 "E3_hac_maximal_sizes": sorted([len(c) for c in hac_flat], reverse=True),
                 "mine_head_rule_thr0.7_minsz3_firsthalf": len(cc7),
                 "mine_head_rule_sizes": sorted([len(c) for c in cc7], reverse=True),
                 "thr0.5_components_ge2": len(cc5), "thr0.5_isolated": iso5,
                 "thr0.5_greedy_modularity_communities": len(gm), "thr0.5_modularity": mod,
                 "thr0.5_edge_density": float(nx.density(G5)),
                 "mean_abs_corr_offdiag": float(np.abs(Cfull[np.triu_indices_from(Cfull, 1)]).mean()),
                 "n_discrete_le10vals": n_disc, "n_continuous": n_cont,
                 "frac_discrete_of_live": n_disc / max(1, int(live.sum())), "unique_value_source": src}
    LOG(f"   E3-HAC nested={len(hac)} maximal={len(hac_flat)} | thr0.7cc={len(cc7)} thr0.5 comps={len(cc5)} greedy={len(gm)} Q={mod:.2f} "
        f"discrete={n_disc}/{int(live.sum())} const={n_const}")

    # ---------------- A9: snapshot detectability (single-window trivial max|z-level| rule)
    LOG(" [A9] trivial rule")
    triv_te = np.abs(Xa[:, :C]).max(1); triv_tr = np.abs(Xn[:, :C]).max(1)
    thr = float(np.quantile(triv_tr, 0.99))
    easy = (y == 1) & (triv_te > thr); hard = (y == 1) & ~easy

    def au(mask):
        k = (y == 0) | mask
        return float(roc_auc_score(y[k], triv_te[k])) if mask.sum() > 0 else float("nan")

    a9 = dict(thr_train_q99=thr, frac_anom_easy=float(easy.sum() / max(1, y.sum())),
              n_easy=int(easy.sum()), n_hard=int(hard.sum()),
              frac_test_normal_over_thr=float((triv_te[y == 0] > thr).mean()),
              frac_train_over_thr=float((triv_tr > thr).mean()),
              triv_auroc_all=au(y == 1), triv_auroc_easy=au(easy), triv_auroc_difficult=au(hard))
    if "maxz" in D:
        a9["bundle_maxz_thr"] = D["maxz_thr"]; a9["bundle_maxz_max_abs_diff"] = float(np.abs(D["maxz"] - triv_te).max())
    if w:
        a9["latad_witness_diff_auroc"] = w[0]["diff_auroc"]; a9["latad_witness_easy_auroc"] = w[0]["easy_auroc"]
    res["A9"] = a9
    LOG(f"   easy_frac={a9['frac_anom_easy']:.2f} AUROC all/easy/diff = {a9['triv_auroc_all']:.3f}/"
        f"{a9['triv_auroc_easy']:.3f}/{a9['triv_auroc_difficult']:.3f}")

    # ---------------- A10: history dependence (proxy: anomalies not snapshot-separable)
    # A trajectory-required fault would be invisible to any single-window rule. The
    # single-window trivial rule + the 'hard' AUROC bound what history could add.
    res["A10"] = dict(frac_anom_snapshot_separable=a9["frac_anom_easy"],
                      triv_auroc_difficult=a9["triv_auroc_difficult"],
                      between_mode_train_frac_rho=res["A3"].get("rho"),
                      test_anom_between_mode_frac=res["A7"].get("test_anom_frac_between_modes"),
                      note="no public benchmark labels a history-conditioned fault; single-window rule "
                           "already separates the 'easy' share and the LatAD OFF difficult AUROC bounds the rest")
    return res


def main(names):
    witness = json.load(open(os.path.join(HERE, "a3_skab_witness.json")))
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for nm in names:
        if nm in out and "--force" not in sys.argv:
            LOG(f"[{nm}] present, skip"); continue
        t = time.time()
        out[nm] = measure(nm, witness)
        out[nm]["sec"] = round(time.time() - t, 1)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1); f.flush(); os.fsync(f.fileno())
        LOG(f"[{nm}] saved ({out[nm]['sec']}s)")


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or ["SKAB", "WADI", "SWaT", "HAI"]
    main(names)
