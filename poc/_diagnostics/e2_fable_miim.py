"""E2 adversarial audit, stage 4: the SAME E2 measurement on MIIM synthetic data, where the
condition (isolated modes, Zipf beta=2.1 occupancy => genuinely rare-but-valid regimes) is
present by construction. Ground-truth mode labels define 'rare' (train-window share < 2%).
Appends to e2_fable_miim.jsonl.
"""
from __future__ import annotations
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.dirname(HERE)
sys.path.insert(0, POC); os.chdir(POC)
import numpy as np, torch
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from models_vade import train_vade, _as_tensor
from miim_gen import make_dataset

OUT = os.path.join(HERE, "e2_fable_miim.jsonl")


def logN_of(v, X):
    with torch.no_grad():
        mu = v.encode(_as_tensor(X, v))[0]
        return v._log_pz_given_c(mu).cpu().numpy().astype(np.float64), torch.log_softmax(v.pi_logit, 0).cpu().numpy().astype(np.float64)


def scores(L, lp):
    return dict(mixture=-logsumexp(lp[None] + L, 1), nearest=-L.max(1))


def run(seed, K=None, LD=10, rare_pi=0.02):
    t0 = time.time()
    D = make_dataset(seed=seed)
    Xtr, Xte, yte, mtr, mte, at = D["x_train"], D["x_test"], D["y_test"], D["mode_train"], D["mode_test"], D["atype_test"]
    Ktrue = D["K"]; K = K or Ktrue
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s = ((Xtr - m) / sd).astype(np.float32); Xte_s = ((Xte - m) / sd).astype(np.float32)
    v = train_vade(Xtr_s, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
    Ltr, lp = logN_of(v, Xtr_s); Lte, _ = logN_of(v, Xte_s)
    s_tr, s_te = scores(Ltr, lp), scores(Lte, lp)
    # ground-truth rarity from the generator's mode labels
    occ_gt = np.bincount(mtr[mtr >= 0], minlength=Ktrue) / (mtr >= 0).sum()
    rare_gt = np.where(occ_gt < rare_pi)[0]
    norm = (yte == 0)
    rare_norm = norm & np.isin(mte, rare_gt)
    plain_norm = norm & (at == "none")
    # also model-side rarity (E2's definition) for comparison
    a_tr, a_te = Ltr.argmax(1), Lte.argmax(1)
    occ_m = np.bincount(a_tr, minlength=K) / len(a_tr); rare_m = np.where(occ_m < rare_pi)[0]
    rare_norm_m = norm & np.isin(a_te, rare_m)
    # isolation margin of the fitted components
    S = np.sort(Ltr, 1); margin = S[:, -1] - S[:, -2]
    res = dict(seed=seed, K_true=int(Ktrue), K_model=int(K), latent=LD, n_train=int(len(Xtr)), n_test=int(len(Xte)),
               n_test_normal=int(norm.sum()), n_rare_gt_modes=int(len(rare_gt)), n_test_normal_rare_gt=int(rare_norm.sum()),
               n_test_normal_rare_model=int(rare_norm_m.sum()), gt_occ_sorted=[round(float(x), 4) for x in np.sort(occ_gt)[::-1][:12]],
               gt_min_occ=float(occ_gt[occ_gt > 0].min()), median_isolation_margin=float(np.median(margin)),
               frac_margin_gt7=float(np.mean(margin > 7)))
    gap = s_te["mixture"] - s_te["nearest"]
    res["gap_rare_gt_median"] = float(np.median(gap[rare_norm])) if rare_norm.sum() else None
    res["gap_common_median"] = float(np.median(gap[norm & ~np.isin(mte, rare_gt)]))
    for key in s_tr:
        thr = np.quantile(s_tr[key], .99); p = s_te[key] > thr
        r = dict(thr=float(thr), fpr_all_normal=float(p[norm].mean()), fpr_plain_normal=float(p[plain_norm].mean()),
                 fpr_rare_gt_normal=float(p[rare_norm].mean()) if rare_norm.sum() else None,
                 fpr_rare_model_normal=float(p[rare_norm_m].mean()) if rare_norm_m.sum() else None,
                 tpr_all=float(p[yte == 1].mean()), auroc_all=float(roc_auc_score(yte, s_te[key])))
        # matched overall test-normal FPR 1%: FPR on rare-gt normals
        tm = np.quantile(s_te[key][norm], .99); r["fpr_rare_gt_at_matched1pct"] = float((s_te[key][rare_norm] > tm).mean()) if rare_norm.sum() else None
        # per anomaly type TPR at threshold + AUROC vs normals
        r["tpr_by_type"] = {t: float(p[(yte == 1) & (at == t)].mean()) for t in np.unique(at[yte == 1])}
        res[key] = r
    # rank statistic: AUROC rare-gt-normal vs common-normal
    yy = rare_norm[norm].astype(int)
    if yy.sum() and (yy == 0).sum():
        res["auroc_rare_vs_common_normal"] = {k: float(roc_auc_score(yy, s_te[k][norm])) for k in s_te}
    res["secs"] = round(time.time() - t0)
    with open(OUT, "a") as f: f.write(json.dumps(res) + "\n")
    print(f"[MIIM seed {seed}] K_true={Ktrue} rare_gt_modes={len(rare_gt)} rare-gt-normal test={rare_norm.sum()}/{norm.sum()} (model-rare {rare_norm_m.sum()}) min_occ={res['gt_min_occ']:.4f} margin_med={res['median_isolation_margin']:.1f}")
    for k in ["mixture", "nearest"]:
        r = res[k]
        print(f"   {k:8s} thr={r['thr']:.2f} FPR all={r['fpr_all_normal']:.4f} plain={r['fpr_plain_normal']:.4f} RARE-gt={r['fpr_rare_gt_normal']} rare-model={r['fpr_rare_model_normal']} matched1%={r['fpr_rare_gt_at_matched1pct']} | TPR={r['tpr_all']:.3f} AUROC={r['auroc_all']:.3f}")
    print(f"   gap rare-gt median={res['gap_rare_gt_median']} common median={res['gap_common_median']:.2f}; AUROC(rare vs common normal)={res.get('auroc_rare_vs_common_normal')}  ({res['secs']}s)", flush=True)


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2]
    for s in seeds: run(s)
