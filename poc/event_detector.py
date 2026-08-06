"""Semi-Markov EVENT detector (the compression idea): run-length-encode the trajectory
into (mode, duration) events (debounced), learn a transition matrix P(j|i) + per-mode
dwell law from normal, and score each settled window by transition surprise
-log P(current|predecessor). Compresses the haystack so rare transitions are prominent.
Compares to the window VaDE and the c_t context density on the labelled miim_gen data.
"""
from __future__ import annotations
import numpy as np, torch
from sklearn.decomposition import PCA
from models_vade import train_vade
import explore_c as E

CACHE = "datasets/miim/_cmp_miim.npz"
SNAP = ["pocket", "near_boundary", "drift", "ood"]
COLS = SNAP + ["bad_transition"]


def segment(gamma, min_dur=2):
    """Run-length encode argmax(gamma) into [mode,start,dur], debouncing runs < min_dur
    (flicker) by absorbing them into the previous run (keeps its mode)."""
    m = gamma.argmax(1); ev = []
    i = 0
    while i < len(m):
        j = i
        while j < len(m) and m[j] == m[i]:
            j += 1
        ev.append([int(m[i]), i, j - i]); i = j
    out = []
    for e in ev:
        if e[2] < min_dur and out:
            out[-1][2] += e[2]
        else:
            out.append(e)
    return out


def tpr(sn, sp, f=0.05):
    return float("nan") if len(sp) == 0 else float((sp > np.quantile(sn, 1 - f)).mean())


def main():
    device = "cpu"
    C = np.load(CACHE, allow_pickle=True)
    xtr, xte, atype, yte = C["xtr"], C["xte"], C["atype"], C["yte"]
    c_tr, c_te = C["c_tr"], C["c_te"]; normal = yte == 0; fringe = atype == "fringe"
    std = lambda a, r: ((a - r.mean(0)) / (r.std(0) + 1e-8)).astype(np.float32)
    xtr_s, xte_s = std(xtr, xtr), std(xte, xtr)

    v = train_vade(xtr_s, n_clusters=64, latent_dim=8, epochs=40, warmup=8, seed=0, device=device)
    v.fit_residual_whitener(xtr_s)
    sw_tr, sw_te = v.anomaly_score(xtr_s), v.anomaly_score(xte_s)
    g_tr, g_te = E.soft_gamma(v, xtr_s), E.soft_gamma(v, xte_s)
    K = g_tr.shape[1]

    # --- learn semi-Markov event model on normal train ---
    ev_tr = segment(g_tr)
    Tc = np.ones((K, K)) * 0.5                         # Laplace smoothing
    for a, b in zip(ev_tr[:-1], ev_tr[1:]):
        Tc[a[0], b[0]] += 1
    P = Tc / Tc.sum(1, keepdims=True)
    ld = {k: [] for k in range(K)}
    for e in ev_tr:
        ld[e[0]].append(np.log(e[2] + 1))
    dmu = np.array([np.mean(ld[k]) if ld[k] else 0.0 for k in range(K)])
    dsd = np.array([np.std(ld[k]) + 0.3 if ld[k] else 1.0 for k in range(K)])

    def event_scores(g):
        ev = segment(g); n = len(g)
        s_trans = np.zeros(n); s_dwell = np.zeros(n)
        for i in range(1, len(ev)):
            a, b = ev[i - 1][0], ev[i][0]
            st, du = ev[i][1], ev[i][2]
            s_trans[st:st + du] = -np.log(P[a, b] + 1e-12)      # transition surprise into b
            z = (np.log(du + 1) - dmu[b]) / dsd[b]
            s_dwell[st:st + du] = z * z                          # dwell surprise (both tails)
        return s_trans, s_dwell

    st_te, sd_te = event_scores(g_te)
    st_tr, _ = event_scores(g_tr)

    # c_t context density (for reference)
    pca = PCA(24, random_state=0).fit(c_tr)
    sc_tr, sc_te = E.soft_ctx_score(pca.transform(c_tr), g_tr, np.ones(len(g_tr)),
                                    pca.transform(c_te), g_te)
    z = lambda a, r: (a - r.mean()) / (r.std() + 1e-9)

    branches = {
        "window VaDE": sw_te,
        "c_t context density": z(sw_te, sw_tr) + np.maximum(0.0, z(sc_te, sc_tr)),
        "EVENT transition surprise": st_te,
        "OR(window, event)": None,   # filled below via per-branch tails
    }
    print(f"\n{'branch':<28}" + "".join(f"{c:>14}" for c in COLS) + f"{'fringeFPR':>10}")
    print("-" * 100)
    for name, s in branches.items():
        if name == "OR(window, event)":
            tw = np.quantile(sw_te[normal], 0.965); tc = np.quantile(st_te[normal], 0.985)
            flag = (sw_te > tw) | (st_te > tc)
            row = [float(flag[atype == t].mean()) for t in COLS]
            ff = float(flag[fringe].mean())
        else:
            sn = s[normal]
            row = [tpr(sn, s[atype == t]) for t in COLS]; ff = tpr(sn, s[fringe])
        print(f"{name:<28}" + "".join(f"{r:>14.3f}" for r in row) + f"{ff:>10.3f}")
    print(f"\nevents: train={len(ev_tr)}  forbidden P(j|i)~{P.min():.4f} (Laplace floor); "
          "transition surprise targets bad_transition; dwell surprise is a new signal.")


if __name__ == "__main__":
    main()
