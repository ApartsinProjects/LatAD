"""E2 support-partition measurement on cleaned WADI (clip=10), K=20, LD=10, 5 seeds.
For each test window: nearest-component NLL  n(x) = min_c -log N(z | mu_c, sig_c)  and the prior pi of that
nearest component; mixture NLL  m(x) = -logsumexp_c (log pi_c + log N_c).
Partition of TEST NORMALS (thresholds from TRAIN only):
  out-of-support     : n(x) > q99 of train n
  in-support, low-pi : n(x) <= q99 train n AND pi_nearest < pi_low (pi_low = 0.5/K, i.e. half the uniform share)
  in-support, high-pi: the rest
Rare-regime FPR = fraction of the in-support-low-pi normals flagged at the train-99th-pct threshold of each score
(nearest-only vs mixture). Also reports occupancy: how many TRAIN windows live in low-pi components (does VaDE
keep rare regimes at all?). Writes fable_cleanwadi_e2_support.json.
"""
from __future__ import annotations
import os, sys, json, numpy as np, warnings, torch
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.dirname(HERE); sys.path.insert(0, POC); os.chdir(POC)
from scipy.special import logsumexp
from models_vade import train_vade, _as_tensor
import eda_real as E

D = E.load("WADI_clean", clip=10.0)
Xtr0, Xte0, y = D["Xn_w"].astype(np.float32), D["Xa_w"].astype(np.float32), D["ya_w"].astype(int)
mu, sig = Xtr0.mean(0), Xtr0.std(0) + 1e-8
Xtr = ((Xtr0 - mu) / sig).astype(np.float32); Xte = ((Xte0 - mu) / sig).astype(np.float32)
K, LD = 20, 10; norm = y == 0
rows = []
for sd in [0, 1, 2, 3, 4]:
    v = train_vade(Xtr, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=sd, device="cpu")
    with torch.no_grad():
        logpi = torch.log_softmax(v.pi_logit, 0).cpu().numpy()
        def comp(X):
            z = v.encode(_as_tensor(X, v))[0]
            L = v._log_pz_given_c(z).detach().cpu().numpy()   # (n, K) log N_c
            near = -L.max(1); c = L.argmax(1); mix = -logsumexp(logpi[None] + L, 1)
            return near, c, mix
        n_tr, c_tr, m_tr = comp(Xtr); n_te, c_te, m_te = comp(Xte)
    pi = np.exp(logpi); occ = np.bincount(c_tr, minlength=K) / len(c_tr)
    pi_low = 0.5 / K
    low_comps = np.flatnonzero(pi < pi_low)
    oos_thr = np.quantile(n_tr, 0.99); thr_near = np.quantile(n_tr, 0.99); thr_mix = np.quantile(m_tr, 0.99)
    oos = n_te > oos_thr
    lowpi = (~oos) & (pi[c_te] < pi_low)
    g_oos, g_low, g_high = norm & oos, norm & lowpi, norm & ~oos & ~lowpi
    r = dict(seed=sd, K=K, pi_sorted=np.sort(pi)[::-1].round(4).tolist(), occ_sorted_by_pi=occ[np.argsort(-pi)].round(4).tolist(),
             n_low_pi_components=int(len(low_comps)), train_frac_in_low_pi_components=float(np.isin(c_tr, low_comps).mean()),
             n_train_in_low_pi_components=int(np.isin(c_tr, low_comps).sum()),
             test_normals=dict(total=int(norm.sum()), out_of_support=int(g_oos.sum()), in_support_low_pi=int(g_low.sum()), in_support_high_pi=int(g_high.sum())),
             test_anoms=dict(total=int((y==1).sum()), out_of_support=int(((y==1)&oos).sum()), in_support_low_pi=int(((y==1)&lowpi).sum())),
             fpr_low_pi_group=dict(nearest=float((n_te[g_low] > thr_near).mean()) if g_low.sum() else None,
                                   mixture=float((m_te[g_low] > thr_mix).mean()) if g_low.sum() else None),
             fpr_high_pi_group=dict(nearest=float((n_te[g_high] > thr_near).mean()), mixture=float((m_te[g_high] > thr_mix).mean())),
             fpr_all_normals=dict(nearest=float((n_te[norm] > thr_near).mean()), mixture=float((m_te[norm] > thr_mix).mean())),
             # alternative low-pi definition: bottom quartile of pi
             lowpi_q25=dict(pi_cut=float(np.quantile(pi, .25)), n_test_normals=int((norm & ~oos & (pi[c_te] < np.quantile(pi, .25))).sum()),
                            fpr_nearest=float((n_te[norm & ~oos & (pi[c_te] < np.quantile(pi, .25))] > thr_near).mean()) if (norm & ~oos & (pi[c_te] < np.quantile(pi, .25))).sum() else None,
                            fpr_mixture=float((m_te[norm & ~oos & (pi[c_te] < np.quantile(pi, .25))] > thr_mix).mean()) if (norm & ~oos & (pi[c_te] < np.quantile(pi, .25))).sum() else None))
    rows.append(r)
    print(f"seed {sd}: pi min/med/max {pi.min():.4f}/{np.median(pi):.4f}/{pi.max():.4f}; low-pi comps (<{pi_low:.3f}) {len(low_comps)} holding "
          f"{r['n_train_in_low_pi_components']} train windows; test normals: OOS {g_oos.sum()}, in-support-low-pi {g_low.sum()}, high-pi {g_high.sum()}; "
          f"FPR low-pi group nearest/mixture {r['fpr_low_pi_group']}; q25 def: n={r['lowpi_q25']['n_test_normals']} {r['lowpi_q25']['fpr_nearest']}/{r['lowpi_q25']['fpr_mixture']}", flush=True)
json.dump(rows, open(os.path.join(HERE, "fable_cleanwadi_e2_support.json"), "w"), indent=1)
print("saved")
