"""Tests 2 + 3 (R1-2 / E2), ONE pass per (dataset, seed) on the CURRENT clean loaders
(HAI, WADI_clean, SWaT_canon; SWaT after the 2026-09-19 leak fix). Config identical to
clean_{wadi,swat}_tables56.py (K/LD per dataset, epochs 40, warm-up 8, k_density = min(80, N//10)).
Per (dataset, seed) it co-computes:
  * Table 6 head AUROCs on the canonical difficult subset (mask read from scores_<name>.npz):
    recon, density (M=80), nearest (Eq. 5), base, base+resid(auto), LatAD(auto/auto)  -> Test 3
  * rare-regime false-alarm rate on TEST-NORMAL windows for: VaDE mixture, VaDE nearest,
    tempered (pi^0.5), balanced (uniform pi), density80 (shipped), base (shipped), at
    (a) the train-99th-percentile threshold of each head and (b) a matched 1% FPR on all test normals.
    Rare regimes: R1 = VaDE components with train occupancy < 2%; R2 = bottom-quartile pi;
    R3 = density80 components with train occupancy < 0.5/M. Also the in-support subset of R1
    (nearest NLL <= train q99), where the pi penalty is the only thing that can differ.  -> Test 2
  * number of test windows on which the mixture and nearest train-p99 flags disagree.
Appends one JSON line per (dataset, seed) to nc_real_rare_fpr.jsonl (resumable).
Usage: python nc_real_rare_fpr.py <dataset> [nseeds]
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
from models_vade import train_vade, _as_tensor, _recon_energy
import eda_real as E

CFG = {"HAI": (40, 16), "WADI_clean": (20, 10), "SWaT_canon": (40, 16)}
OUT = os.path.join(HERE, "nc_real_rare_fpr.jsonl")


def z(v, ref):
    return (v - ref.mean()) / (ref.std() + 1e-9)


def main(name, nseeds=5):
    K, LD = CFG[name]
    D = E.load(name)
    Xn = D["Xn_w"].astype(np.float32); Xa = D["Xa_w"].astype(np.float32); y = D["ya_w"].astype(int)
    clipv = E.CLIP.get(name); mu, sig = Xn.mean(0), Xn.std(0) + 1e-8
    Zn = np.clip((Xn - mu) / sig, -clipv, clipv).astype(np.float32) if clipv else ((Xn - mu) / sig).astype(np.float32)
    Za = np.clip((Xa - mu) / sig, -clipv, clipv).astype(np.float32) if clipv else ((Xa - mu) / sig).astype(np.float32)
    kd = min(80, max(20, len(Xn) // 10))
    d = np.load(os.path.join(HERE, f"scores_{name}.npz")); thr_mz = float(d["maxz_thr"]); maxz = d["maxz"]
    assert len(maxz) == len(y), "difficulty mask / loader window-count mismatch"
    hard = (y == 1) & (maxz <= thr_mz); keep = np.where((y == 0) | hard)[0]; norm = y == 0
    print(f"{name}: train {len(Zn)} windows, test {len(Za)} (normal {norm.sum()}, anomalies {(y==1).sum()}, difficult {hard.sum()}), clip={clipv}, K={K}, LD={LD}, kd={kd}", flush=True)
    done = set()
    if os.path.exists(OUT):
        for ln in open(OUT):
            r = json.loads(ln); done.add((r["dataset"], r["seed"]))
    for seed in range(nseeds):
        if (name, seed) in done:
            continue
        t0 = time.time()
        v = train_vade(Zn, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
        v.fit_latent_density(Zn, k_density=kd, seed=seed)
        v.fit_residual_whitener(Zn); v.fit_resid_head(Zn); v.fit_basin_head(Zn)

        def comp(X):
            with torch.no_grad():
                m_ = v.encode(_as_tensor(X, v))[0]
                L = v._log_pz_given_c(m_).cpu().numpy().astype(np.float64)
                lp = torch.log_softmax(v.pi_logit, 0).cpu().numpy().astype(np.float64)
                zz = m_.cpu().numpy().astype(float)
            dens = -v.latent_gmm.score_samples(zz)
            s = dict(mixture=-logsumexp(lp[None] + L, 1), nearest=-L.max(1), tempered=-logsumexp(0.5 * lp[None] + L, 1),
                     balanced=-logsumexp(L, 1) + np.log(K), density80=dens, base=np.asarray(v.anomaly_score_hard(X, use_near=True), float))
            return s, L, lp, v.latent_gmm.predict(zz)

        s_tr, Ltr, lp, g_tr = comp(Zn); s_te, Lte, _, g_te = comp(Za)
        a_tr, a_te = Ltr.argmax(1), Lte.argmax(1)
        occ = np.bincount(a_tr, minlength=K) / len(a_tr); pi = np.exp(lp)
        occ80 = np.bincount(g_tr, minlength=kd) / len(g_tr)
        R1 = np.flatnonzero(occ < 0.02); R2 = np.flatnonzero(pi < np.quantile(pi, .25)); R3 = np.flatnonzero(occ80 < 0.5 / kd)
        insup = s_te["nearest"] <= np.quantile(s_tr["nearest"], .99)
        groups = {"all_normal": norm, "R1_rare_vade": norm & np.isin(a_te, R1), "R1_rare_vade_insupport": norm & np.isin(a_te, R1) & insup,
                  "R2_lowpi_q25": norm & np.isin(a_te, R2), "R3_rare_density80": norm & np.isin(g_te, R3)}
        # Table 6 heads (verbatim clean_*_tables56 definitions)
        dens_te, dn_te = v._hard_components(Za); dens_tr, dn_tr = v._hard_components(Zn)
        xt, xtr = _as_tensor(Za, v), _as_tensor(Zn, v)
        r_te = np.asarray(_recon_energy(xt, v.decode(v.encode(xt)[0]), v.res_whitener)); r_tr = np.asarray(_recon_energy(xtr, v.decode(v.encode(xtr)[0]), v.res_whitener))
        t6 = dict(recon=z(r_te, r_tr), density=z(np.asarray(dens_te), np.asarray(dens_tr)), nearest=z(np.asarray(dn_te), np.asarray(dn_tr)),
                  base=np.asarray(v.anomaly_score_hard(Za, use_near=True)), **{"base+resid": np.asarray(v.anomaly_score_hard(Za, use_near=True, use_resid="auto"))},
                  LatAD=np.asarray(v.anomaly_score_hard(Za, use_resid="auto", use_basin="auto")))
        res = dict(dataset=name, seed=seed, K=K, LD=LD, kd=kd, clip=clipv, n_train=int(len(Zn)), n_test_normal=int(norm.sum()), n_difficult=int(hard.sum()),
                   pi_min=float(pi.min()), pi_median=float(np.median(pi)), pi_max=float(pi.max()), occ_sorted=np.sort(occ)[::-1].round(4).tolist(),
                   n_R1=int(len(R1)), n_R2=int(len(R2)), n_R3=int(len(R3)), train_frac_in_R1=float(np.isin(a_tr, R1).mean()),
                   group_sizes={g: int(mk.sum()) for g, mk in groups.items()}, resid_auto=bool(getattr(v, "_resid_auto", False)),
                   table6={h: round(float(roc_auc_score(y[keep], np.nan_to_num(s)[keep])), 4) for h, s in t6.items()},
                   diff_auroc={k: round(float(roc_auc_score(y[keep], np.nan_to_num(s_te[k])[keep])), 4) for k in s_te})
        flags = {}
        for k in s_tr:
            thr = np.quantile(s_tr[k], .99); thr_m = np.quantile(s_te[k][norm], .99)
            fa, fm = s_te[k] > thr, s_te[k] > thr_m; flags[k] = fa
            res[k] = dict(fpr_train_p99={g: (float(fa[mk].mean()) if mk.sum() else None) for g, mk in groups.items()},
                          fpr_matched1pct={g: (float(fm[mk].mean()) if mk.sum() else None) for g, mk in groups.items()})
        res["n_flag_disagree_mix_vs_near"] = int((flags["mixture"] != flags["nearest"]).sum())
        res["n_flag_disagree_d80_vs_near"] = int((flags["density80"] != flags["nearest"]).sum())
        res["secs"] = round(time.time() - t0)
        with open(OUT, "a") as f:
            f.write(json.dumps(res) + "\n")
        gs = res["group_sizes"]
        print(f"  seed {seed} ({res['secs']}s) T6: " + " ".join(f"{h}={a:.3f}" for h, a in res["table6"].items()) + f" | R1 comps {len(R1)} (train frac {res['train_frac_in_R1']:.3f}) sizes {gs}", flush=True)
        for k in s_tr:
            print(f"     {k:9s} FPR@p99 all={res[k]['fpr_train_p99']['all_normal']:.4f} R1={res[k]['fpr_train_p99']['R1_rare_vade']} R1ins={res[k]['fpr_train_p99']['R1_rare_vade_insupport']} R2={res[k]['fpr_train_p99']['R2_lowpi_q25']} R3={res[k]['fpr_train_p99']['R3_rare_density80']} "
                  f"| matched R1={res[k]['fpr_matched1pct']['R1_rare_vade']} R3={res[k]['fpr_matched1pct']['R3_rare_density80']} | diffAUROC={res['diff_auroc'][k]:.3f}", flush=True)
        print(f"     flags disagree mix-vs-near {res['n_flag_disagree_mix_vs_near']}, d80-vs-near {res['n_flag_disagree_d80_vs_near']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
