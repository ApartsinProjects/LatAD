"""E2 (reviewer B-05 / Ed-near): direct test of the rare-regime-safe nearest-component
likelihood vs the pi-weighted regime mixture.

Claim under test (paper 4.3 ii): a valid point in a rare, low-pi regime must not be
flagged just for being rare. The pi-weighted mixture NLL penalizes rare regimes
(-log sum_c pi_c N) ; the nearest-component NLL (-max_c log N) ignores pi. So on
NORMAL test windows that fall in rare regimes, the mixture should raise MORE false
alarms than nearest. WIN = nearest-component FPR on rare-regime normals is lower,
without losing difficult-subset AUROC.

Also reports two imbalance-aware variants: pi-tempered mixture (pi**T) and the
nearest head. Retrains a small VaDE per dataset (CPU). Writes _diagnostics/e2_nearest.json.
Does NOT touch the paper.
"""
from __future__ import annotations
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from models_vade import train_vade
from winfeat import window_features
import eda_real as E

W, ST = 60, 30
CFG = {"WADI": (20, 10), "HAI": (40, 16), "SWaT": (40, 16), "SKAB": (16, 6)}
RARE_PI = 0.02          # a regime is "rare" if it holds <2% of train-normal windows
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_diagnostics", "e2_nearest.json")


def win(X, y=None):
    A, B = [], []
    for i in range(0, len(X) - W + 1, ST):
        A.append(window_features(X[i:i + W], "stats"))
        if y is not None:
            B.append(int(y[i:i + W].mean() > 0.05))
    return np.asarray(A, np.float32), (np.asarray(B, int) if y is not None else None)


def au(y, s, mask):
    k = (y == 0) | mask
    yy = y[k]
    if yy.sum() < 2 or (yy == 0).sum() < 2:
        return float("nan")
    return float(roc_auc_score(yy, s[k]))


def logN_of(v, X):
    """Per-component log N(z|mu_c, sig_c^2): (n, K), plus log-pi (K,)."""
    with torch.no_grad():
        mu = v.encode(v._as_tensor(X, v) if hasattr(v, "_as_tensor") else torch.as_tensor(X))[0]
    # models_vade._responsibilities does exactly this; replicate to get raw logN
    from models_vade import _as_tensor
    with torch.no_grad():
        mu = v.encode(_as_tensor(X, v))[0]
        logN = v._log_pz_given_c(mu).cpu().numpy()
        logpi = torch.log_softmax(v.pi_logit, 0).cpu().numpy()
    return logN, logpi


def run(name, seed=0):
    K, LD = CFG[name]
    D = E.load(name)
    Xtr, _ = win(D["Xn_raw"])
    Xte, yw = win(D["Xa_raw"], D["ya_raw"])
    m, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s = ((Xtr - m) / sd).astype(np.float32)
    Xte_s = ((Xte - m) / sd).astype(np.float32)
    C6 = Xte.shape[1] // 6
    triv = np.abs(Xte[:, :C6]).max(1); trn = np.abs(Xtr[:, :C6]).max(1)
    easy = (yw == 1) & (triv > np.quantile(trn, 0.99)); hard = (yw == 1) & ~easy

    v = train_vade(Xtr_s, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")

    logN_tr, logpi = logN_of(v, Xtr_s)
    logN_te, _ = logN_of(v, Xte_s)

    def scores(logN):
        mix = -logsumexp(logpi[None] + logN, axis=1)          # pi-weighted mixture NLL
        near = -logN.max(axis=1)                               # nearest-component NLL
        temp = -logsumexp(0.5 * logpi[None] + logN, axis=1)    # pi-tempered (T=0.5)
        return dict(mixture=mix, nearest=near, tempered=temp)

    s_tr = scores(logN_tr); s_te = scores(logN_te)

    # regime assignment + train occupancy pi_c
    a_tr = logN_tr.argmax(1); a_te = logN_te.argmax(1)
    occ = np.bincount(a_tr, minlength=K) / len(a_tr)
    rare = set(np.where(occ < RARE_PI)[0].tolist())
    norm_te = (yw == 0)
    rare_norm = norm_te & np.isin(a_te, list(rare) if rare else [-1])

    res = dict(dataset=name, K=K, latent=LD, seed=seed,
               n_regimes=int(K), n_rare_regimes=int(len(rare)),
               rare_regime_ids=sorted(rare), train_occupancy=[round(float(o), 4) for o in occ],
               n_test_normal=int(norm_te.sum()), n_test_normal_rare=int(rare_norm.sum()),
               n_difficult=int(hard.sum()))

    for key in ["mixture", "nearest", "tempered"]:
        thr = float(np.quantile(s_tr[key], 0.99))             # 99th pct of TRAIN score -> ~1% FPR by design
        pred = s_te[key] > thr
        fpr_all = float((pred & norm_te).sum() / max(1, norm_te.sum()))
        fpr_rare = float((pred & rare_norm).sum() / max(1, rare_norm.sum())) if rare_norm.sum() else float("nan")
        res[key] = dict(
            diff_auroc=round(au(yw, s_te[key], hard), 3),
            fpr_all_normal=round(fpr_all, 4),
            fpr_rare_regime_normal=round(fpr_rare, 4) if rare_norm.sum() else None,
        )
    # win flags
    if rare_norm.sum() >= 5:
        res["WIN_rare_fpr"] = bool(res["nearest"]["fpr_rare_regime_normal"] < res["mixture"]["fpr_rare_regime_normal"])
    else:
        res["WIN_rare_fpr"] = None
        res["note"] = f"only {int(rare_norm.sum())} rare-regime normal test windows; FPR unreliable"
    return res


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["WADI", "HAI", "SWaT"]
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    have = {r["dataset"] for r in rows}
    for nm in names:
        if nm in have:
            print(f"[{nm}] present, skip", flush=True); continue
        r = run(nm)
        rows.append(r)
        json.dump(rows, open(OUT, "w"), indent=1)
        print(f"\n=== {nm} ===  rare_regimes={r['n_rare_regimes']}  "
              f"rare-normal-test={r['n_test_normal_rare']}/{r['n_test_normal']}")
        for k in ["mixture", "nearest", "tempered"]:
            print(f"  {k:9s} diffAUROC={r[k]['diff_auroc']}  FPR(all)={r[k]['fpr_all_normal']}  "
                  f"FPR(rare-regime)={r[k]['fpr_rare_regime_normal']}")
        print(f"  WIN (nearest FPR_rare < mixture FPR_rare): {r['WIN_rare_fpr']}"
              + (f"  [{r.get('note','')}]" if r.get('note') else ""))
    print(f"\nsaved -> {OUT}")
