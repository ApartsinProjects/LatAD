"""E2 adversarial audit, stage 3: CONTROLLED IMBALANCE SWEEP on HAI (prototype of the
reviewer's part (ii)).

Design: regimes = argmax-nearest assignment of the E2 reference model (e2_fable_HAI.npz).
Pick 3 populous, tight target regimes. Split train 85/15 pool/val (seeded). In the pool,
downsample the target regimes to ratio r (1, .1, .03, .01, .003, .001); retrain VaDE with the
E2 config; threshold = train(pool) 99th pct per score. Val-target windows are VALID normal
windows from a now-rare regime (ground truth by construction); val-other = valid common.
Metrics: FPR on val-target (rare valid) and val-other, mixture vs nearest, at (a) the
train-99pct threshold, (b) matched val-other FPR=1%; test difficult AUROC. Also the analytic
pi-only counterfactual on the reference model (components fixed, pi_c *= r): the mechanism's
ideal-case upper bound. Appends one JSON line per (r, seed) to e2_fable_sweep.jsonl (resumable).
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

NAME = "HAI"; K, LD = 40, 16
TARGETS = [1, 30, 25]                       # populous (2.3-4%), tight (logvar ~ -1.1..-1.3) regimes
RATIOS = [1.0, 0.1, 0.03, 0.01, 0.003, 0.001]
OUT = os.path.join(HERE, "e2_fable_sweep.jsonl")


def logN_of(v, X):
    with torch.no_grad():
        mu = v.encode(_as_tensor(X, v))[0]
        return v._log_pz_given_c(mu).cpu().numpy().astype(np.float64), torch.log_softmax(v.pi_logit, 0).cpu().numpy().astype(np.float64)


def scores(L, lp):
    return dict(mixture=-logsumexp(lp[None] + L, 1), nearest=-L.max(1))


def au(y, s, mask):
    k = (y == 0) | mask; yy = y[k]
    return float(roc_auc_score(yy, s[k])) if yy.sum() >= 2 else float("nan")


def main(seeds=(0, 1)):
    d = np.load(os.path.join(HERE, f"e2_fable_{NAME}.npz"))
    Xtr_raw = d["Xtr_s"] * d["sd"] + d["m"]; Xte_raw = d["Xte_s"] * d["sd"] + d["m"]   # undo E2 standardisation
    yw, hard = d["yw"], d["hard"]
    a_ref = d["logN_tr"].argmax(1)
    done = set()
    if os.path.exists(OUT):
        for ln in open(OUT):
            r = json.loads(ln); done.add((r["ratio"], r["seed"]))
    for seed in seeds:
        rng = np.random.default_rng(100 + seed)
        perm = rng.permutation(len(Xtr_raw)); nval = int(0.15 * len(perm))
        val_idx, pool_idx = perm[:nval], perm[nval:]
        is_t = np.isin(a_ref, TARGETS)
        val_t = val_idx[is_t[val_idx]]; val_o = val_idx[~is_t[val_idx]]
        pool_t = pool_idx[is_t[pool_idx]]; pool_o = pool_idx[~is_t[pool_idx]]
        print(f"[seed {seed}] val target {len(val_t)} other {len(val_o)} | pool target {len(pool_t)} other {len(pool_o)}", flush=True)
        # ---- analytic pi-only counterfactual on the REFERENCE model (components fixed) ----
        Lref_tr, lp_ref = d["logN_tr"], d["logpi"]
        for r in RATIOS:
            if (r, seed) in done: continue
            t0 = time.time()
            keep_t = rng.permutation(pool_t)[:max(1, int(round(r * len(pool_t))))]
            tr_idx = np.concatenate([pool_o, keep_t])
            # -- counterfactual: pi_c *= r for targets, renormalise; threshold on the SAME downsampled train set
            lp_cf = lp_ref.copy(); lp_cf[TARGETS] += np.log(r); lp_cf -= logsumexp(lp_cf)
            s_cf_tr = scores(Lref_tr[tr_idx], lp_cf); s_cf_v = scores(Lref_tr[val_idx], lp_cf)
            cf = {}
            for key in s_cf_tr:
                thr = np.quantile(s_cf_tr[key], .99); pv = s_cf_v[key] > thr
                cf[key] = dict(fpr_val_target=float(np.mean(pv[is_t[val_idx]])), fpr_val_other=float(np.mean(pv[~is_t[val_idx]])))
            # -- real: retrain on the downsampled pool
            Xp = Xtr_raw[tr_idx]; m, sd = Xp.mean(0), Xp.std(0) + 1e-8
            Xp_s = ((Xp - m) / sd).astype(np.float32); Xv_s = ((Xtr_raw[val_idx] - m) / sd).astype(np.float32)
            Xte_s = ((Xte_raw - m) / sd).astype(np.float32)
            v = train_vade(Xp_s, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
            Lp, lp = logN_of(v, Xp_s); Lv, _ = logN_of(v, Xv_s); Lt, _ = logN_of(v, Xte_s)
            s_p, s_v, s_t = scores(Lp, lp), scores(Lv, lp), scores(Lt, lp)
            a_p = Lp.argmax(1); a_v = Lv.argmax(1)
            occ = np.bincount(a_p, minlength=K) / len(a_p)
            vt = is_t[val_idx]
            # which retrained components hold the val-target windows, and how rare are they?
            comps_t = np.bincount(a_v[vt], minlength=K)
            pi_of_t = float(np.median(np.exp(lp)[a_v[vt]])); occ_of_t = float(np.median(occ[a_v[vt]]))
            share_t_in_comp = float((comps_t / np.maximum(1, np.bincount(a_v, minlength=K)))[a_v[vt]].mean())  # purity: how target-dominated are their comps
            res = dict(ratio=r, seed=seed, n_train=int(len(tr_idx)), n_train_target=int(len(keep_t)),
                       train_target_frac=float(len(keep_t) / len(tr_idx)), n_val_target=int(vt.sum()), n_val_other=int((~vt).sum()),
                       median_pi_of_valtarget_comp=pi_of_t, median_occ_of_valtarget_comp=occ_of_t, valtarget_comp_purity=share_t_in_comp,
                       n_comps_holding_valtarget=int((comps_t > 0).sum()),
                       med_nearNLL_val_target=float(np.median(s_v["nearest"][vt])), med_nearNLL_val_other=float(np.median(s_v["nearest"][~vt])),
                       med_gap_val_target=float(np.median(s_v["mixture"][vt] - s_v["nearest"][vt])),
                       med_gap_val_other=float(np.median(s_v["mixture"][~vt] - s_v["nearest"][~vt])),
                       counterfactual_pi_only=cf, secs=round(time.time() - t0))
            for key in s_p:
                thr = np.quantile(s_p[key], .99); pv = s_v[key] > thr
                thr_m = np.quantile(s_v[key][~vt], .99); pm = s_v[key] > thr_m       # matched val-other FPR = 1%
                res[key] = dict(fpr_val_target=float(np.mean(pv[vt])), fpr_val_other=float(np.mean(pv[~vt])),
                                fpr_val_target_at_matched_other1pct=float(np.mean(pm[vt])),
                                fpr_test_normal=float(np.mean((s_t[key] > thr)[yw == 0])),
                                diff_auroc=round(au(yw, s_t[key], hard), 3), all_auroc=round(float(roc_auc_score(yw, s_t[key])), 3))
            with open(OUT, "a") as f: f.write(json.dumps(res) + "\n")
            print(f"  r={r:<6} n_t={len(keep_t):4d} frac={res['train_target_frac']:.4f} | pi(valT comp)={pi_of_t:.4f} purity={share_t_in_comp:.2f} nearNLL valT/valO={res['med_nearNLL_val_target']:.1f}/{res['med_nearNLL_val_other']:.1f} gap valT/valO={res['med_gap_val_target']:.2f}/{res['med_gap_val_other']:.2f}\n"
                  f"         FPR valTarget mix={res['mixture']['fpr_val_target']:.3f} near={res['nearest']['fpr_val_target']:.3f} | matched1%: mix={res['mixture']['fpr_val_target_at_matched_other1pct']:.3f} near={res['nearest']['fpr_val_target_at_matched_other1pct']:.3f} | valOther mix={res['mixture']['fpr_val_other']:.3f} near={res['nearest']['fpr_val_other']:.3f} | diffAUROC mix={res['mixture']['diff_auroc']} near={res['nearest']['diff_auroc']}\n"
                  f"         pi-only counterfactual (ref model): FPR valTarget mix={cf['mixture']['fpr_val_target']:.3f} near={cf['nearest']['fpr_val_target']:.3f}  ({res['secs']}s)", flush=True)


if __name__ == "__main__":
    main()
