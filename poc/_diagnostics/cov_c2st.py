"""Threshold-free train-normal vs test-normal distribution-mismatch measures (model-free, reuse bundle arrays).
HEADLINE: energy distance with block-permutation null (MMD as cross-check); occupancy TV/KL robust across K and partition.
SECONDARY: block structure of test-only windows; C2ST vs block-permuted label null; relative OOS ratio (footnote).
Size curve: energy distance and C2ST AUC vs train fraction. Merges into cov_coverage.json under 'threshold_free'.
"""
import json, os, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "_diagnostics", "cov_coverage.json")
DSS = sys.argv[1:] or ["WADI", "HAI", "SWaT"]
FRACS = [0.1, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2]
N_SUB = 2000            # per-side cap for two-sample tests
N_PERM = 1000
BLOCK = 25              # contiguous points per permutation block (in the even-in-time subsample)
CHUNKS = 20             # time chunks per source for C2ST blocked CV / block-permutation null
KS = (8, 16, 32, 64)


def runs(idx, bridge=2):
    idx = np.sort(np.asarray(idx))
    if len(idx) == 0: return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > bridge + 1: out.append((int(s), int(p))); s = i
        p = i
    out.append((int(s), int(p))); return out


def block_stats(pos):
    rs = runs(pos); n = len(pos)
    lens = np.array([b - a + 1 for a, b in rs]) if rs else np.array([0])
    big = [(a, b) for a, b in rs if b - a + 1 >= 30]
    n_big = sum(int(np.sum((pos >= a) & (pos <= b))) for a, b in big)
    return dict(n=int(n), n_runs=len(rs), longest_run=int(lens.max()), n_runs_ge30=len(big),
                frac_in_runs_ge30=float(n_big / max(n, 1)), frac_singletons=float((lens == 1).sum() / max(n, 1)),
                runs_ge30=[(a, b, b - a + 1) for a, b in big][:25])


def even(Z, n):
    if len(Z) <= n: return Z, np.arange(len(Z))
    idx = np.linspace(0, len(Z) - 1, n).round().astype(int); return Z[idx], idx


def rand_sub(Z, n, r):
    if len(Z) <= n: return Z, np.arange(len(Z))
    idx = np.sort(r.choice(len(Z), n, replace=False)); return Z[idx], idx


# ---------------- energy distance / MMD with permutation null ----------------
def two_sample(Z0, Z1, seed=0, n_perm=N_PERM, sub="even", n_sub=N_SUB):
    r = np.random.default_rng(seed)
    n = min(len(Z0), len(Z1), n_sub)
    A, _ = (even(Z0, n) if sub == "even" else rand_sub(Z0, n, r)); B, _ = (even(Z1, n) if sub == "even" else rand_sub(Z1, n, r))
    P = np.vstack([A, B]).astype(np.float32); N = len(P); na, nb = len(A), len(B)
    D = cdist(P, P).astype(np.float32)
    sig = float(np.median(D[np.triu_indices(N, 1)]))
    Kg = np.exp(-(D / sig) ** 2 / 2).astype(np.float32)
    totD, totK = D.sum(dtype=np.float64), Kg.sum(dtype=np.float64)

    def stats(l1):                       # l1: 0/1 label vector; matvec trick, no submatrix indexing
        l1 = l1.astype(np.float32); l0 = 1 - l1; m, k = l0.sum(), l1.sum()
        Dl1 = D @ l1; sxy = float(l0 @ Dl1); syy = float(l1 @ Dl1); sxx = totD - 2 * sxy - syy
        en = 2 * sxy / (m * k) - sxx / (m * (m - 1)) - syy / (k * (k - 1))
        exy = sxy / (m * k)
        Kl1 = Kg @ l1; kxy = float(l0 @ Kl1); kyy = float(l1 @ Kl1); kxx = totK - 2 * kxy - kyy
        mmd = (kxx - m) / (m * (m - 1)) + (kyy - k) / (k * (k - 1)) - 2 * kxy / (m * k)
        return en, mmd, exy
    lab = np.r_[np.zeros(na), np.ones(nb)]
    e_obs, m_obs, exy = stats(lab)
    H = e_obs / (2 * exy)                 # energy coefficient in [0,1] (Szekely & Rizzo), effect size
    # block permutation (contiguous blocks of BLOCK points keep autocorrelation intact under the null)
    nbk = N // BLOCK; bid = np.minimum(np.arange(N) // BLOCK, nbk - 1)
    e_null, m_null = np.empty(n_perm), np.empty(n_perm)
    for i in range(n_perm):
        lb = np.zeros(nbk); lb[r.permutation(nbk)[nbk // 2:]] = 1
        e_null[i], m_null[i], _ = stats(lb[bid])
    e_null2 = np.empty(n_perm)            # point-wise permutation (classical, ignores autocorrelation) for reference
    for i in range(n_perm):
        e_null2[i], _, _ = stats(r.permutation(lab))
    return dict(n_per_side=int(n), subsample=sub, sigma_median=sig,
                energy=float(e_obs), energy_coef_H=float(H),
                energy_null_block=dict(mean=float(e_null.mean()), p95=float(np.quantile(e_null, .95)), max=float(e_null.max())),
                energy_z_vs_block_null=float((e_obs - e_null.mean()) / max(e_null.std(), 1e-12)),
                p_energy_blockperm=float((np.sum(e_null >= e_obs) + 1) / (n_perm + 1)),
                p_energy_pointperm=float((np.sum(e_null2 >= e_obs) + 1) / (n_perm + 1)),
                energy_null_point_p95=float(np.quantile(e_null2, .95)),
                mmd2=float(m_obs), mmd2_null_block=dict(mean=float(m_null.mean()), p95=float(np.quantile(m_null, .95)), max=float(m_null.max())),
                p_mmd_blockperm=float((np.sum(m_null >= m_obs) + 1) / (n_perm + 1)), n_perm=int(n_perm))


# ---------------- C2ST with block-permuted label null ----------------
def c2st(Z0, Z1, seed=0, n_null=0, return_oof=False):
    n = min(len(Z0), len(Z1))
    A, ia = even(Z0, n); B, ib = even(Z1, n)
    X = np.vstack([A, B]); y = np.r_[np.zeros(n), np.ones(n)]
    chunk = np.r_[(np.arange(n) * CHUNKS) // n, CHUNKS + (np.arange(n) * CHUNKS) // n]   # 2*CHUNKS equal time chunks
    def fit_auc(lbl):
        oof = np.full(len(lbl), np.nan)
        for tr, te in GroupKFold(10).split(X, lbl, chunk):
            clf = LogisticRegression(C=0.1, max_iter=2000).fit(X[tr], lbl[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
        return roc_auc_score(lbl, oof), oof
    auc, oof = fit_auc(y)
    out = dict(auc=float(auc), n_per_class=int(n))
    if n_null:
        r = np.random.default_rng(seed); null = []
        for _ in range(n_null):
            cl = np.zeros(2 * CHUNKS); cl[r.permutation(2 * CHUNKS)[CHUNKS:]] = 1   # whole chunks relabelled -> block null
            null.append(fit_auc(cl[chunk])[0])
        null = np.array(null)
        out.update(null_auc_mean=float(null.mean()), null_auc_p95=float(np.quantile(null, .95)), null_auc_max=float(null.max()),
                   p_vs_block_null=float((np.sum(null >= auc) + 1) / (n_null + 1)), n_null=int(n_null))
    if return_oof: out["oof_test"] = oof[n:]; out["idx_test"] = ib
    return out


def occupancy(a_tr, a_te, K):
    p = np.bincount(a_tr, minlength=K) / len(a_tr); q = np.bincount(a_te, minlength=K) / len(a_te)
    eps = 0.5 / max(len(a_tr), len(a_te)); pe, qe = (p + eps) / (p + eps).sum(), (q + eps) / (q + eps).sum()
    tv = float(0.5 * np.abs(p - q).sum()); contrib = np.abs(p - q) / max(np.abs(p - q).sum(), 1e-12)
    top = np.argsort(q - p)[::-1][:3]
    return dict(K=int(K), tv=tv, kl_test_from_train=float(np.sum(qe * np.log(qe / pe))),
                mass_test_in_modes_with_train_lt_1pct=float(q[p < 0.01].sum()),
                top_mode_share_of_tv=float(contrib.max()),
                top_test_excess=[dict(mode=int(k), train=float(p[k]), test=float(q[k]), share_of_tv=float(contrib[k])) for k in top]), p, q


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for ds in DSS:
        t0 = time.time()
        b = np.load(os.path.join(ROOT, "sota_bundle", "ens_bundle", f"bundle_{ds}.npz"))
        e = np.load(os.path.join(ROOT, "_diagnostics", f"e2_fable_{ds}.npz"))
        Xn, Xa, y = b["Xn_w"].astype(np.float64), b["Xa_w"].astype(np.float64), b["y"]
        mu, sd = Xn.mean(0), Xn.std(0); sd = np.maximum(sd, 0.01 * np.median(sd))
        Zn, Za = (Xn - mu) / sd, np.clip((Xa - mu) / sd, -50, 50)
        nrm = np.where(y == 0)[0]; Zt = Za[nrm]; n = len(Zn); h = n // 2
        T = {}
        print(f"== {ds}: train {n} test-normal {len(Zt)}")

        # ---- (2) energy distance / MMD, headline ----
        ts = {"even_seed0": two_sample(Zn, Zt, seed=0, sub="even")}
        for s_ in SEEDS: ts[f"random_seed{s_}"] = two_sample(Zn, Zt, seed=s_, sub="random")
        ts["null_ref_train_first_vs_second_half"] = two_sample(Zn[:h], Zn[h:], seed=0, sub="even", n_perm=300)
        # within-train temporal variability reference: each contiguous fifth of train vs the rest (same test, same n cap)
        fifths = []
        for j in range(5):
            lo, hi = (j * n) // 5, ((j + 1) * n) // 5; msk = np.zeros(n, bool); msk[lo:hi] = True
            r5 = two_sample(Zn[msk], Zn[~msk], seed=0, sub="even", n_perm=100)
            fifths.append(dict(fifth=j, energy=r5["energy"], H=r5["energy_coef_H"], p_block=r5["p_energy_blockperm"], c2st_auc=c2st(Zn[msk], Zn[~msk])["auc"]))
        ts["within_train_fifth_vs_rest"] = fifths
        ref_e = [f_["energy"] for f_ in fifths]; ref_a = [f_["c2st_auc"] for f_ in fifths]
        ts["test_vs_train_over_within_train"] = dict(energy_ratio_to_max_fifth=float(ts["even_seed0"]["energy"] / max(ref_e)),
                                                     energy_ratio_to_mean_fifth=float(ts["even_seed0"]["energy"] / np.mean(ref_e)),
                                                     within_train_energy=dict(mean=float(np.mean(ref_e)), max=float(max(ref_e)), last_fifth=float(ref_e[-1])),
                                                     within_train_c2st_auc=dict(mean=float(np.mean(ref_a)), max=float(max(ref_a)), last_fifth=float(ref_a[-1])))
        T["two_sample"] = ts
        print("  within-train fifths:", fifths, " ratio:", ts["test_vs_train_over_within_train"])
        for k, v in ts.items():
            if not (isinstance(v, dict) and "n_per_side" in v): continue
            print(f"  {k}: n={v['n_per_side']} energy={v['energy']:.4f} H={v['energy_coef_H']:.4f} null_block(mean {v['energy_null_block']['mean']:.4f}, max {v['energy_null_block']['max']:.4f}) z={v['energy_z_vs_block_null']:.1f} p_block={v['p_energy_blockperm']:.4f} p_point={v['p_energy_pointperm']:.4f} | mmd2={v['mmd2']:.4f} p={v['p_mmd_blockperm']:.4f}")

        # ---- (3) occupancy divergence, robust across K and partition ----
        occ = {}
        # VaDE regime argmax (itr/ite in the npz are window START INDICES, not regimes)
        itr = np.argmax(e["logN_tr"] + e["logpi"], 1); ite = np.argmax(e["logN_te"] + e["logpi"], 1)[nrm]; Kv = int(len(e["logpi"]))
        occ["vade_regimes"], p, q = occupancy(itr, ite, Kv)
        if ds == "HAI": occ["vade_regimes"]["regime_21"] = dict(train=float(p[21]), test=float(q[21]), share_of_tv=float(abs(p[21] - q[21]) / np.abs(p - q).sum()))
        # null reference for the VaDE partition: train first half vs second half
        occ["vade_regimes"]["null_train_halves"] = {k: v for k, v in occupancy(itr[:h], itr[h:], Kv)[0].items() if k in ("tv", "kl_test_from_train")}
        pooled = np.vstack([Zn, Zt]); pca = PCA(20, random_state=0).fit(pooled); Pp = pca.transform(pooled)
        for k in KS:
            km = KMeans(k, n_init=4, random_state=0).fit(Pp)
            occ[f"kmeans{k}"], _, _ = occupancy(km.labels_[:n], km.labels_[n:], k)
            kmn = KMeans(k, n_init=2, random_state=0).fit(Pp[:n])   # null: train-only fit, first half vs second half
            occ[f"kmeans{k}"]["null_train_halves"] = {kk: vv for kk, vv in occupancy(kmn.labels_[:h], kmn.labels_[h:], k)[0].items() if kk in ("tv", "kl_test_from_train")}
        T["occupancy"] = occ
        for k, v in occ.items():
            print(f"  occ {k}: TV={v['tv']:.3f} KL={v['kl_test_from_train']:.3f} mass_in_train<1%={v['mass_test_in_modes_with_train_lt_1pct']:.3f} topmode share of TV={v['top_mode_share_of_tv']:.2f} null halves TV={v['null_train_halves']['tv']:.3f}" + (f" regime21={v['regime_21']}" if 'regime_21' in v else ""))

        # ---- C2ST with block-permuted null (secondary) + block structure of test-only windows ----
        c = c2st(Zn, Zt, n_null=20, return_oof=True)
        T["c2st"] = {k: v for k, v in c.items() if not k.startswith(("oof", "idx"))}
        T["c2st"]["null_ref_train_halves_auc"] = c2st(Zn[:h], Zn[h:])["auc"]
        print("  C2ST:", T["c2st"])
        oof, pos_all = c["oof_test"], nrm[c["idx_test"]]
        struct = {}
        for pth in (0.9, 0.95):
            struct[f"p_gt_{pth}"] = block_stats(pos_all[oof > pth]); struct[f"p_gt_{pth}"]["frac_of_test_normal"] = float((oof > pth).mean())
        struct["oof_quantiles_test_normal"] = {str(qq): float(np.quantile(oof, qq)) for qq in (0.1, 0.5, 0.9, 0.99)}
        if ds == "HAI":
            inb = (pos_all >= 6436) & (pos_all <= 7081)
            struct["hai_ref_block_6436_7081"] = dict(n_sampled_in_block=int(inb.sum()), mean_oof_in_block=float(oof[inb].mean()),
                                                     frac_gt_0p9_in_block=float((oof[inb] > 0.9).mean()), mean_oof_outside=float(oof[~inb].mean()),
                                                     frac_gt_0p9_outside=float((oof[~inb] > 0.9).mean()))
        T["c2st_structure"] = struct
        np.savez(os.path.join(ROOT, "_diagnostics", f"cov_c2st_oof_{ds}.npz"), oof=oof, pos=pos_all)
        print("  structure:", struct)

        # ---- relative OOS ratio (footnote) ----
        m = int(0.1 * n); rel = {}
        for name, hidx in [("random10", np.random.default_rng(0).choice(n, m, replace=False)), ("last10_temporal", np.arange(n - m, n))]:
            mask = np.ones(n, bool); mask[hidx] = False; ref = Zn[mask]
            nn = NearestNeighbors(n_neighbors=2, algorithm="brute", n_jobs=-1).fit(ref)
            dl = nn.kneighbors(ref)[0][:, 1]; d_ho = nn.kneighbors(Zn[hidx])[0][:, 0]; d_te = nn.kneighbors(Zt)[0][:, 0]
            r_ = {f"q{qn}": dict(heldout_train_rate=float((d_ho > np.quantile(dl, qn)).mean()), test_normal_rate=float((d_te > np.quantile(dl, qn)).mean()),
                                 ratio=float((d_te > np.quantile(dl, qn)).mean() / max((d_ho > np.quantile(dl, qn)).mean(), 1e-9))) for qn in (0.9, 0.95, 0.99)}
            r_["median_d1_ratio_test_over_heldout"] = float(np.median(d_te) / np.median(d_ho)); rel[name] = r_
        T["relative_oos"] = rel
        print("  relative OOS:", rel)

        # ---- size curve: energy distance (block-perm) and C2ST AUC vs train fraction ----
        curve = []
        for f in FRACS:
            nf = max(int(round(f * n)), 100)
            subs = [("prefix", np.arange(nf))] + ([("random", np.sort(np.random.default_rng(s_).choice(n, nf, replace=False))) for s_ in SEEDS] if f < 1 else [("random", np.arange(n))])
            for kind, idx in subs:
                tsr = two_sample(Zn[idx], Zt, seed=0, sub="even", n_perm=200)
                curve.append(dict(frac=f, n_sub=int(nf), kind=kind, energy=tsr["energy"], H=tsr["energy_coef_H"], p_energy_blockperm=tsr["p_energy_blockperm"],
                                  energy_null_block_max=tsr["energy_null_block"]["max"], c2st_auc=c2st(Zn[idx], Zt)["auc"]))
        def ag(key):
            return {str(f): dict(random_mean=float(np.mean([c_[key] for c_ in curve if c_["frac"] == f and c_["kind"] == "random"])),
                                 random_sd=float(np.std([c_[key] for c_ in curve if c_["frac"] == f and c_["kind"] == "random"])),
                                 prefix=float([c_[key] for c_ in curve if c_["frac"] == f and c_["kind"] == "prefix"][0])) for f in FRACS}
        T["size_curve"] = curve; T["size_curve_agg"] = dict(energy=ag("energy"), H=ag("H"), c2st_auc=ag("c2st_auc"), p_energy=ag("p_energy_blockperm"))
        for key in ("energy", "H", "c2st_auc"):
            print(f"  size curve {key}:", {k: (round(v["random_mean"], 4), round(v["prefix"], 4)) for k, v in T["size_curve_agg"][key].items()})
        print(f"  [{time.time()-t0:.0f}s]")
        res.setdefault(ds, {})["threshold_free"] = T
        json.dump(res, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
