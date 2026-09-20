"""Test 1b (R1-2 / E2): LEARNED regime-imbalance sweep on MIIM synthetic data (isolated modes by
construction, ground-truth mode labels). A populous ground-truth mode T is downsampled in the
training pool to ratio r; VaDE (K = K_true, LD = 10, the e2_fable_miim config) and the shipped
M = 80 density head are refit per (T, r, seed). Rare-but-VALID windows = (a) val-target: held-out
TRAIN windows of T (valid by construction), (b) test-normal windows of T.
Heads: mixture (VaDE pi-weighted), nearest (Eq. 5), tempered (pi^0.5), balanced (uniform pi),
density80 (shipped sklearn GMM, pi-weighted), base (shipped: z(density80) + z(nearest)).
FPR on rare-valid at (a) the train-p99 threshold of each head, (b) matched 1% FPR on val-other.
Guard: all-anomaly AUROC and per-type TPR at threshold (a flatter score must not hide anomalies).
Allocation diagnostics: purity / pi / occupancy of the components holding val-target.
pi-only counterfactual: the r = 1 model with the pi of T's own components scaled by r (ideal case
with the LEARNED components). Frozen across r: feature standardisation (r = 1 pool), k_density.
Appends one JSON line per (target, r, seed) to nc_miim_sweep.jsonl (resumable).
Usage: python nc_miim_sweep.py <seed>
"""
from __future__ import annotations
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.dirname(HERE)
sys.path.insert(0, POC); os.chdir(POC)
import numpy as np, torch
torch.set_num_threads(2)
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from models_vade import train_vade, _as_tensor
from miim_gen import make_dataset

RATIOS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]
LD = 10
OUT = os.path.join(HERE, "nc_miim_sweep.jsonl")


def heads(v, X):
    with torch.no_grad():
        mu = v.encode(_as_tensor(X, v))[0]
        L = v._log_pz_given_c(mu).cpu().numpy().astype(np.float64)
        lp = torch.log_softmax(v.pi_logit, 0).cpu().numpy().astype(np.float64)
        z = mu.cpu().numpy().astype(float)
    K = len(lp)
    s = dict(mixture=-logsumexp(lp[None] + L, 1), nearest=-L.max(1), tempered=-logsumexp(0.5 * lp[None] + L, 1),
             balanced=-logsumexp(L, 1) + np.log(K), density80=-v.latent_gmm.score_samples(z),
             base=np.asarray(v.anomaly_score_hard(X, use_near=True), float))
    return s, L, lp


def main(seed):
    D = make_dataset(seed=seed)
    Xtr, Xte, yte, mtr, mte, at = D["x_train"], D["x_test"], D["y_test"], D["mode_train"], D["mode_test"], D["atype_test"]
    Kt = D["K"]
    occ_gt = np.bincount(mtr[mtr >= 0], minlength=Kt) / (mtr >= 0).sum()
    targets = np.argsort(-occ_gt)[:2].tolist()
    done = set()
    if os.path.exists(OUT):
        for ln in open(OUT):
            r = json.loads(ln); done.add((r["seed"], r["target"], r["ratio"]))
    for T in targets:
        rng = np.random.default_rng(1000 + seed)
        is_t = mtr == T
        idx_t, idx_o = np.flatnonzero(is_t), np.flatnonzero(~is_t)
        idx_t, idx_o = rng.permutation(idx_t), rng.permutation(idx_o)
        nvt, nvo = max(50, int(0.3 * len(idx_t))), int(0.15 * len(idx_o))
        val_t, pool_t = idx_t[:nvt], idx_t[nvt:]
        val_o, pool_o = idx_o[:nvo], idx_o[nvo:]
        pool_full = np.concatenate([pool_o, pool_t])
        m, sd = Xtr[pool_full].mean(0), Xtr[pool_full].std(0) + 1e-8          # frozen at r = 1
        kd = min(80, max(20, len(pool_full) // 10))                            # frozen at r = 1
        S = lambda X: ((X - m) / sd).astype(np.float32)
        Xv, Xt = S(Xtr[np.concatenate([val_t, val_o])]), S(Xte)
        vt = np.r_[np.ones(len(val_t), bool), np.zeros(len(val_o), bool)]
        test_norm = yte == 0; test_norm_T = test_norm & (mte == T); test_norm_O = test_norm & (mte != T)
        print(f"[seed {seed} T={T}] occ_gt={occ_gt[T]:.3f} val_t={len(val_t)} val_o={len(val_o)} pool_t={len(pool_t)} pool_o={len(pool_o)} test_norm_T={test_norm_T.sum()} K={Kt}", flush=True)
        ref = None
        for r in RATIOS:
            if (seed, T, r) in done:
                continue
            t0 = time.time()
            keep_t = pool_t[:max(1, int(round(r * len(pool_t))))]           # nested prefix
            tr = np.concatenate([pool_o, keep_t]); Xp = S(Xtr[tr])
            v = train_vade(Xp, n_clusters=Kt, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
            v.fit_latent_density(Xp, k_density=kd, seed=seed)
            s_p, Lp, lp = heads(v, Xp); s_v, Lv, _ = heads(v, Xv); s_t, Lt, _ = heads(v, Xt)
            a_p, a_v = Lp.argmax(1), Lv.argmax(1)
            occ = np.bincount(a_p, minlength=Kt) / len(a_p)
            comps_t = np.bincount(a_v[vt], minlength=Kt)
            purity = float((comps_t / np.maximum(1, np.bincount(a_v, minlength=Kt)))[a_v[vt]].mean())
            res = dict(seed=seed, target=int(T), ratio=r, occ_gt_target=float(occ_gt[T]), n_train=int(len(tr)), n_train_target=int(len(keep_t)),
                       train_target_frac=float(len(keep_t) / len(tr)), n_val_target=int(vt.sum()), n_val_other=int((~vt).sum()),
                       n_test_normal_target=int(test_norm_T.sum()), K=int(Kt), kd=int(kd),
                       median_pi_of_valtarget_comp=float(np.median(np.exp(lp)[a_v[vt]])), median_occ_of_valtarget_comp=float(np.median(occ[a_v[vt]])),
                       valtarget_comp_purity=purity, n_comps_holding_valtarget=int((comps_t > 0).sum()),
                       med_nearNLL_val_target=float(np.median(s_v["nearest"][vt])), med_nearNLL_val_other=float(np.median(s_v["nearest"][~vt])),
                       med_gap_val_target=float(np.median(s_v["mixture"][vt] - s_v["nearest"][vt])),
                       med_gap_val_other=float(np.median(s_v["mixture"][~vt] - s_v["nearest"][~vt])))
            for k in s_p:
                thr = np.quantile(s_p[k], .99); thr_m = np.quantile(s_v[k][~vt], .99)
                pv, pm, pt = s_v[k] > thr, s_v[k] > thr_m, s_t[k] > thr
                res[k] = dict(fpr_val_target=float(pv[vt].mean()), fpr_val_other=float(pv[~vt].mean()),
                              fpr_val_target_matched1pct=float(pm[vt].mean()),
                              fpr_test_normal_target=float(pt[test_norm_T].mean()) if test_norm_T.sum() else None,
                              fpr_test_normal_other=float(pt[test_norm_O].mean()),
                              tpr_all=float(pt[yte == 1].mean()), auroc_all=float(roc_auc_score(yte, s_t[k])),
                              tpr_by_type={t: float(pt[(yte == 1) & (at == t)].mean()) for t in np.unique(at[yte == 1])})
            # pi-only counterfactual on the r = 1 model of this (seed, T)
            if r == 1.0:
                ref = (Lp.copy(), lp.copy(), Lv.copy(), a_p.copy())
            if ref is not None:
                Lp1, lp1, Lv1, a_p1 = ref
                # components whose r=1 pool windows are majority-target (rows >= len(pool_o) are pool_t)
                own = np.bincount(a_p1[len(pool_o):], minlength=Kt); tot = np.bincount(a_p1, minlength=Kt)
                tcomp = np.flatnonzero((tot > 0) & (own / np.maximum(tot, 1) > 0.5))
                lp_cf = lp1.copy(); lp_cf[tcomp] += np.log(r); lp_cf -= logsumexp(lp_cf)
                keep_rows = np.r_[np.arange(len(pool_o)), len(pool_o) + np.arange(len(keep_t))]
                cf = {}
                for k, fn in (("mixture", lambda L_, lp_: -logsumexp(lp_[None] + L_, 1)), ("nearest", lambda L_, lp_: -L_.max(1))):
                    thr = np.quantile(fn(Lp1[keep_rows], lp_cf), .99); pv = fn(Lv1, lp_cf) > thr
                    cf[k] = dict(fpr_val_target=float(pv[vt].mean()), fpr_val_other=float(pv[~vt].mean()))
                res["counterfactual_pi_only"] = cf; res["n_target_comps_r1"] = int(len(tcomp))
            res["secs"] = round(time.time() - t0)
            with open(OUT, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(f"  r={r:<6} n_t={len(keep_t):4d} frac={res['train_target_frac']:.4f} | pi(valT)={res['median_pi_of_valtarget_comp']:.4f} purity={purity:.2f} ncomp={res['n_comps_holding_valtarget']} "
                  f"nearNLL valT/valO={res['med_nearNLL_val_target']:.1f}/{res['med_nearNLL_val_other']:.1f} gap={res['med_gap_val_target']:.2f}/{res['med_gap_val_other']:.2f}\n"
                  f"         FPR valT  mix={res['mixture']['fpr_val_target']:.3f} near={res['nearest']['fpr_val_target']:.3f} temp={res['tempered']['fpr_val_target']:.3f} bal={res['balanced']['fpr_val_target']:.3f} d80={res['density80']['fpr_val_target']:.3f} base={res['base']['fpr_val_target']:.3f} "
                  f"| matched: mix={res['mixture']['fpr_val_target_matched1pct']:.3f} near={res['nearest']['fpr_val_target_matched1pct']:.3f} d80={res['density80']['fpr_val_target_matched1pct']:.3f}\n"
                  f"         FPR testT mix={res['mixture']['fpr_test_normal_target']} near={res['nearest']['fpr_test_normal_target']} d80={res['density80']['fpr_test_normal_target']} | AUROC mix={res['mixture']['auroc_all']:.3f} near={res['nearest']['auroc_all']:.3f} d80={res['density80']['auroc_all']:.3f} base={res['base']['auroc_all']:.3f} "
                  f"| cf pi-only mix={res.get('counterfactual_pi_only', {}).get('mixture', {}).get('fpr_val_target')} near={res.get('counterfactual_pi_only', {}).get('nearest', {}).get('fpr_val_target')} ({res['secs']}s)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
