"""
Window feature extractors. `stats` is the static per-channel summary (a bag of
statistics that discards within-window dynamics). `temporal` additionally encodes
the DYNAMICS at multiple time scales - rates, within-window slope, and spectral
band-power - which is what assumption A8 (channels evolve on different time
scales: slow/inertial vs fast) makes exploitable. Both return one flat vector per
window so the VaDE framework is unchanged; only the representation differs.
"""

from __future__ import annotations

import numpy as np


def feat_stats(win):
    """6 features/channel: level, variability, extremes, net trend, range."""
    return np.concatenate([win.mean(0), win.std(0), win.min(0), win.max(0),
                           win[-1] - win[0], win.max(0) - win.min(0)])


def feat_temporal(win):
    """10 features/channel: the stats level/variability/trend/range, plus
    dynamics - within-window slope, velocity (|dx|) mean/std, max spike, and
    low- vs high-frequency band power (A8: multiscale temporal content)."""
    W = win.shape[0]
    h = W // 2
    d1 = np.diff(win, axis=0)                                  # velocity, (W-1,C)
    F = np.abs(np.fft.rfft(win - win.mean(0), axis=0))         # spectrum, (W/2+1,C)
    nb = F.shape[0]
    lowp = np.log1p((F[:max(1, nb // 3)] ** 2).sum(0))         # slow-band energy
    highp = np.log1p((F[2 * nb // 3:] ** 2).sum(0))            # fast-band energy
    return np.concatenate([
        win.mean(0), win.std(0),                              # level, variability
        win[-1] - win[0], win.max(0) - win.min(0),            # net trend, range
        win[h:].mean(0) - win[:h].mean(0),                    # within-window slope
        np.abs(d1).mean(0), d1.std(0), np.abs(d1).max(0),     # velocity, roughness, spike
        lowp, highp,                                          # spectral band power (A8)
    ])


def window_features(win, rep):
    if rep == "temporal":
        return feat_temporal(win)
    if rep == "stats":
        return feat_stats(win)
    return win.ravel()                                        # 'flatten'
