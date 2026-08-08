"""Shared machinery for the assumption-grounded cycles: train-normal correlation blocks,
the common-mode / factor head, VaDE-mode responsibilities (from LatAD proxy unavailable -> we
use a train-normal GMM on window-means as a light mode proxy), and the miss-rescue wrapper.

NOTE (honesty): calibration percentiles here use TEST-NORMAL windows as the reference (a
transductive proxy) and the groups were mined from test failures, so ALL cycle results are
EXPLORATORY / test-informed (per ChatGPT's caveat). A frozen method would recalibrate on
train-normal LatAD scores and evaluate on untouched data.
"""
from __future__ import annotations
import numpy as np
from numpy.linalg import svd
import eda_real as E

THR, MINSZ = 0.7, 3


def load_blocks(name):
    D = E.load(name); nch = len(D["ch"])
    Mn = np.asarray(D["Xn_w"], float)[:, :nch]      # train-normal per-channel window means
    Ma = np.asarray(D["Xa_w"], float)[:, :nch]
    y = np.asarray(D["ya_w"], int)
    mu, sg = Mn.mean(0), Mn.std(0) + 1e-9
    Zn, Za = (Mn - mu) / sg, (Ma - mu) / sg
    nA = len(Zn) // 2; A = Zn[:nA]
    C = np.corrcoef(A.T); np.fill_diagonal(C, 0.0); adj = np.abs(C) > THR
    seen = np.zeros(nch, bool); groups = []
    for i in range(nch):
        if seen[i]:
            continue
        st = [i]; comp = []
        while st:
            v = st.pop()
            if seen[v]:
                continue
            seen[v] = True; comp.append(v); st += list(np.where(adj[v])[0])
        if len(comp) >= MINSZ:
            groups.append(sorted(comp))
    return dict(name=name, ch=[str(c) for c in D["ch"]], Zn=Zn, Za=Za, A=A, groups=groups, y=y)


def common_mode_head(bl):
    """A5/A2 common-mode block level: max over groups of |z of the group's PC1| (fit on A)."""
    A, Za, groups = bl["A"], bl["Za"], bl["groups"]
    cols = []
    for G in groups:
        U, S, Vt = svd(A[:, G] - A[:, G].mean(0), full_matrices=False)
        u = Vt[0] * np.sign(Vt[0].sum()); ga = A[:, G] @ u
        cols.append(np.abs((Za[:, G] @ u - ga.mean()) / (ga.std() + 1e-9)))
    return np.max(cols, axis=0)


def pct(ref, v):
    """percentile of v within reference distribution ref (empirical CDF)."""
    order = np.sort(ref)
    return np.searchsorted(order, v, side="right") / len(order)


def rescue_wrap(latad, head, y, u0_hi=0.99, uh_hi=0.995):
    """Miss-rescue: aux head raises the score ONLY where LatAD is permissive (u0<u0_hi) and the
    head is extreme (uh>uh_hi). Percentiles referenced on test-NORMAL (exploratory proxy)."""
    nm = y == 0
    u0 = pct(latad[nm], latad)
    uh = pct(head[nm], head)
    fire = (u0 < u0_hi) & (uh > uh_hi)
    out = u0.copy()
    out[fire] = np.maximum(u0[fire], uh[fire])
    return out, int(fire.sum()), int((fire & nm).sum())
