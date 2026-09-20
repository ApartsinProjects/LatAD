"""E5 source-of-gain ablation RECOMPUTED on the CORRECTED datasets (local CPU).

WADI_clean (artifact channel 2B_AIT_002_PV dropped), SWaT_canon (canonical Dec-2015
attack log), HAI (unchanged, re-run for a construct-matched set). Loaded via
eda_real.load so features match scores_*.npz + the WADI clip fix.

Difficult subset is CANONICAL (build_scores_table / rev4_stats):
    C6   = n_feat // 6
    maxz = |Xte[:, :C6]|.max(1)               (test window features)
    thr  = quantile(|Xtr[:, :C6]|.max(1), 0.99)   (train-normal)
    difficult = (y==1) & (maxz <= thr)
    keep      = (y==0) | difficult            (all normals vs difficult anomalies)

Arms (verbatim from sota_bundle/modal_e5_gain.py):
    crosschannel_latent : VaDE joint latent + high-K diagonal GMM (density only).
    marginal_product    : channel-independent product of per-dim 1-D GMM densities.
gain = crosschannel_latent - marginal_product on the difficult-subset AUROC.
"""
from __future__ import annotations
import os, sys, json, statistics
POC = r"E:\Projects\Backlog\LatAD\poc"
os.chdir(POC); sys.path.insert(0, POC)
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from models_vade import train_vade
import eda_real as E

# CFG identical to build_scores_table.py / export_checkpoints.py
CFG = {"WADI_clean": (20, 10), "SWaT_canon": (40, 16), "HAI": (40, 16)}
DATASETS = ["WADI_clean", "SWaT_canon", "HAI"]
NSEEDS = 5
DIAG = os.path.join(os.path.dirname(os.path.abspath(__file__)))  # write next to script; copied to _diagnostics after


def crosschannel_latent_score(Ztr, Zte, K, LD, seed, kd):
    v = train_vade(Ztr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
    v.fit_latent_density(Ztr, k_density=kd, seed=seed)
    s = v.anomaly_score_hard(Zte, use_near=False, use_recon=False, use_resid=False, use_basin=False)
    return np.asarray(s, dtype=float)


def marginal_product_score(Ztr, Zte, seed, k_marg=5, max_fit=20000):
    rng = np.random.default_rng(seed)
    n, d = Ztr.shape
    fit = Ztr if n <= max_fit else Ztr[rng.choice(n, max_fit, replace=False)]
    nll = np.zeros(len(Zte), dtype=float)
    for j in range(d):
        col = fit[:, j:j + 1]
        kj = min(k_marg, max(1, len(np.unique(col))))
        try:
            g = GaussianMixture(n_components=kj, covariance_type="full",
                                reg_covar=1e-3, random_state=seed).fit(col)
            nll += -g.score_samples(Zte[:, j:j + 1])
        except Exception:
            mu, sd = col.mean(), col.std() + 1e-9
            nll += 0.5 * ((Zte[:, j] - mu) / sd) ** 2
    return nll


def run_ds(name, rows_path):
    D = E.load(name)
    Xn = D["Xn_w"].astype(np.float32); Xa = D["Xa_w"].astype(np.float32)
    y = D["ya_w"].astype(int)
    K, LD = CFG[name]
    kd = min(80, max(20, len(Xn) // 10))

    # canonical difficulty axis (build_scores_table lines 45-48)
    C6 = Xa.shape[1] // 6
    maxz = np.abs(Xa[:, :C6]).max(1).astype(np.float32)
    thr = float(np.quantile(np.abs(Xn[:, :C6]).max(1), 0.99))
    easy = (y == 1) & (maxz > thr)
    hard = (y == 1) & ~easy
    keep = np.where((y == 0) | hard)[0]
    n_hard = int(hard.sum()); n_norm = int((y == 0).sum())

    # cross-check against corrected scores_*.npz maxz/maxz_thr if present
    npz = os.path.join("_diagnostics", f"scores_{name}.npz")
    xcheck = None
    if os.path.exists(npz):
        d = np.load(npz, allow_pickle=True)
        if len(d["maxz"]) == len(maxz):
            same_maxz = bool(np.allclose(d["maxz"], maxz, atol=1e-4))
            same_thr = abs(float(d["maxz_thr"]) - thr) < 1e-3
            same_lab = bool(np.array_equal(d["label"].astype(int), y))
            xcheck = dict(maxz_match=same_maxz, thr_match=same_thr, label_match=same_lab,
                          npz_thr=round(float(d["maxz_thr"]), 4), my_thr=round(thr, 4))

    # standardize on train-normal (identical to modal_e5_gain / build_scores_table)
    mu, sig = Xn.mean(0), Xn.std(0) + 1e-8
    Zn = ((Xn - mu) / sig).astype(np.float32); Za = ((Xa - mu) / sig).astype(np.float32)

    rows = []
    for variant in ("crosschannel_latent", "marginal_product"):
        for seed in range(NSEEDS):
            if variant == "crosschannel_latent":
                s = crosschannel_latent_score(Zn, Za, K, LD, seed, kd)
            else:
                s = marginal_product_score(Zn, Za, seed)
            s = np.nan_to_num(np.asarray(s, dtype=float))
            auroc = float(roc_auc_score(y[keep], s[keep]))
            row = {"dataset": name, "arm": "arm_b", "variant": variant, "rep": "stats",
                   "seed": seed, "auroc_difficult": round(auroc, 4),
                   "n_difficult": n_hard, "n_normal": n_norm}
            rows.append(row)
            with open(rows_path, "a") as fh:
                fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"[e5c] {name} {variant} seed{seed} auroc={row['auroc_difficult']}", flush=True)
    return rows, dict(name=name, n_difficult=n_hard, n_normal=n_norm, n_easy=int(easy.sum()),
                      maxz_thr=round(thr, 4), C6=int(C6), n_feat=int(Xa.shape[1]),
                      K=K, latent_dim=LD, k_density=kd, xcheck=xcheck)


def aggregate(all_rows):
    agg = {}
    for r in all_rows:
        k = (r["dataset"], r["variant"])
        agg.setdefault(k, []).append(r["auroc_difficult"])
    out = {}
    for k, vals in agg.items():
        out[k] = (round(statistics.mean(vals), 4),
                  round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 4), len(vals))
    return out


def main():
    rows_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e5_gain_clean_rows.jsonl")
    if os.path.exists(rows_path):
        os.remove(rows_path)
    all_rows, meta = [], {}
    for name in DATASETS:
        rows, m = run_ds(name, rows_path)
        all_rows += rows; meta[name] = m
    agg = aggregate(all_rows)

    per_ds = []
    print("\n==================== E5 GAIN (CORRECTED datasets, difficult-subset AUROC) ====================")
    for name in DATASETS:
        cc = agg[(name, "crosschannel_latent")]; mp = agg[(name, "marginal_product")]
        gain = round(cc[0] - mp[0], 4)
        per_ds.append({
            "dataset": name,
            "crosschannel_latent_mean": cc[0], "crosschannel_latent_sd": cc[1],
            "marginal_product_mean": mp[0], "marginal_product_sd": mp[1],
            "gain": gain, "n_seeds": cc[2],
            "n_difficult": meta[name]["n_difficult"], "n_normal": meta[name]["n_normal"],
        })
        print(f"{name:12} cross-channel {cc[0]:.4f}+/-{cc[1]:.4f}  marginal {mp[0]:.4f}+/-{mp[1]:.4f}  "
              f"=> gain {gain:+.4f}  (n_diff={meta[name]['n_difficult']}, n_norm={meta[name]['n_normal']})")

    wadi_gain = next(p["gain"] for p in per_ds if p["dataset"] == "WADI_clean")
    note = (f"On WADI_clean (artifact channel 2B_AIT_002_PV removed) the cross-channel-vs-marginal "
            f"difficult-AUROC gain is {wadi_gain:+.4f}, "
            f"{'DOWN from' if wadi_gain < 0.343 else 'NOT down from'} the stale dirty-WADI +0.343: "
            f"the artifact channel {'was inflating' if wadi_gain < 0.343 else 'did not inflate'} the reported gain.")
    print("\n" + note)

    out = {"note": "E5 recomputed on corrected datasets (WADI_clean, SWaT_canon, HAI). "
                   "Canonical difficult subset (y==1)&(maxz<=thr) vs all normals; 5-seed; local CPU. "
                   "Arms verbatim from sota_bundle/modal_e5_gain.py.",
           "wadi_artifact_note": note,
           "per_dataset": per_ds, "meta": meta, "rows": all_rows}
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e5_gain_clean.json")
    with open(outpath, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
