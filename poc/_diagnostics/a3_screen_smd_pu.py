"""A3 screen (entropy + rho) for extra real CPS datasets: SMD machines + Paderborn PU bearing.
Method identical to the existing screen so numbers are comparable:
  - load train-NORMAL, 6-stat window features (W=60, stride=30) via eda_real pipeline,
  - standardize window features on train-normal,
  - VaDE(n_clusters=16, latent_dim=8, epochs=30, warmup=6, seed=0, device=cpu),
  - H_norm = mean over windows of -sum(g*log g)/log K from v._responsibilities(Z),
  - rho = frac(max-resp < 0.5).
Reference: SKAB H_norm~0.27 (A3 holds); WADI/HAI/SWaT/MetroPT/WindSCADA H_norm<=0.09 (no A3).
A3 if H_norm >= ~0.15. Report-only.
"""
from __future__ import annotations
import os, sys, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eda_real as E
from models_vade import train_vade
HERE = os.path.dirname(os.path.abspath(__file__))
K, LD = 16, 8


def screen_windows(Xw):
    """Xw: (n_windows, feat) train-normal window features. Standardize on train, fit VaDE, return metrics."""
    Xw = np.asarray(Xw, np.float32)
    mu, sig = Xw.mean(0), Xw.std(0) + 1e-8
    Xs = ((Xw - mu) / sig).astype(np.float32)
    v = train_vade(Xs, n_clusters=K, latent_dim=LD, epochs=30, warmup=6, seed=0, device="cpu")
    G = v._responsibilities(Xs)                                # (n, K)
    mr = G.max(1)
    ent = float(np.mean(-(G * np.log(G + 1e-12)).sum(1) / np.log(G.shape[1])))
    return dict(n_train=int(len(Xs)), feat_dim=int(Xs.shape[1]),
                entropy=round(ent, 3), rho=round(float((mr < 0.5).mean()), 3),
                mean_maxresp=round(float(mr.mean()), 3))


def screen_named(name):
    """Dataset registered in eda_real.RAW: use its train-normal window features."""
    D = E.load(name)
    return screen_windows(D["Xn_w"])


def _windows_from_raw(Xn_raw, W=60, stride=30):
    """Standardize channels on train-normal, then 6-stat window features (matches eda_real.load)."""
    from winfeat import window_features
    mu, sd = Xn_raw.mean(0), Xn_raw.std(0) + 1e-8
    Xn = ((Xn_raw - mu) / sd).astype(np.float32)
    Xw = [window_features(Xn[i:i + W], "stats") for i in range(0, len(Xn) - W + 1, stride)]
    return np.asarray(Xw, np.float32)


def screen_smd(machine):
    Xn_raw, _, _, _ = E._raw_smd(machine)
    return screen_windows(_windows_from_raw(Xn_raw))


def _frame_bandpower(sig, fs=64000, L=2048, nbands=12, fmin=500.0, hop=None):
    """Raw vibration -> per-frame multichannel feature sequence (the 'regime stream').
    Each non-overlapping frame -> log band-powers in `nbands` log-spaced bands + log broadband RMS.
    Returns (n_frames, nbands+1). This is the defensible SCADA-analog: band-power channels over time,
    whose level/shape shift with operating condition (speed/load/force)."""
    sig = np.asarray(sig, np.float32)
    hop = hop or L
    nf = 1 + (len(sig) - L) // hop
    if nf < 1:
        return np.zeros((0, nbands + 1), np.float32)
    frames = np.stack([sig[i * hop:i * hop + L] for i in range(nf)])
    frames = frames - frames.mean(1, keepdims=True)
    P = (np.abs(np.fft.rfft(frames, axis=1)) ** 2)                 # (nf, L/2+1)
    freqs = np.fft.rfftfreq(L, 1.0 / fs)
    edges = np.geomspace(fmin, fs / 2.0, nbands + 1)
    feats = np.empty((nf, nbands + 1), np.float32)
    for b in range(nbands):
        m = (freqs >= edges[b]) & (freqs < edges[b + 1])
        feats[:, b] = np.log1p(P[:, m].sum(1)) if m.any() else 0.0
    feats[:, -1] = np.log1p(np.sqrt((frames ** 2).mean(1)))        # broadband RMS
    return feats


def screen_paderborn(bearings=("K001", "K002", "K003", "K004", "K005"), W=60, stride=30,
                     L=2048, nbands=12, hop=None, conds_keep=None):
    """Healthy-bearing A3 screen. Regimes = 4 operating conditions across bearings.
    Windows never cross recording boundaries. Channels standardized on the pooled normal set
    (mirrors load()); screen_windows then standardizes the window features (mirrors the screen)."""
    from winfeat import window_features
    recs = E._raw_paderborn(bearings)
    if conds_keep:
        recs = [r for r in recs if r["cond"] in conds_keep]
    # per-recording frame sequences + condition label
    seqs = [(r["cond"], _frame_bandpower(r["sig"], L=L, nbands=nbands, hop=hop)) for r in recs]
    seqs = [(c, f) for c, f in seqs if len(f) >= W]
    # channel standardization on pooled normal frames
    allf = np.concatenate([f for _, f in seqs], 0)
    mu, sd = allf.mean(0), allf.std(0) + 1e-8
    Xw, conds = [], []
    for c, f in seqs:
        fs = (f - mu) / sd
        for i in range(0, len(fs) - W + 1, stride):
            Xw.append(window_features(fs[i:i + W], "stats")); conds.append(c)
    Xw = np.asarray(Xw, np.float32)
    r = screen_windows(Xw)
    # sanity: are the operating-condition regimes actually distinct? (guards against a
    # degenerate feature map giving spuriously-high entropy on indistinguishable windows)
    conds = np.asarray(conds)
    Xs = (Xw - Xw.mean(0)) / (Xw.std(0) + 1e-8)
    mus = {c: Xs[conds == c].mean(0) for c in np.unique(conds)}
    pred = np.array([min(mus, key=lambda c: np.sum((x - mus[c]) ** 2)) for x in Xs])
    r["cond_nn_purity"] = round(float((pred == conds).mean()), 3)   # 1.0 = conditions perfectly separable
    r["n_conds"] = int(len(np.unique(conds)))
    r["n_bearings"] = len(bearings)
    return r


def verdict(ent):
    return "A3-WITNESS" if ent >= 0.15 else ("weak" if ent >= 0.10 else "no-A3")


if __name__ == "__main__":
    t0 = time.time()
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["SKAB"]
    out = {}
    for nm in names:
        try:
            if nm.startswith("SMD:"):
                r = screen_smd(nm[4:])
            elif nm == "Paderborn":
                r = screen_paderborn()
            elif nm.startswith("Paderborn:"):
                r = screen_paderborn(tuple(nm.split(":")[1].split("+")))
            else:
                r = screen_named(nm)
            out[nm] = r
            extra = f" purity={r['cond_nn_purity']}" if "cond_nn_purity" in r else ""
            print(f"  {nm:16s} H_norm={r['entropy']:.3f}  rho={r['rho']:.3f}  "
                  f"maxresp={r['mean_maxresp']:.3f}  n_tr={r['n_train']} d={r['feat_dim']}{extra}  -> {verdict(r['entropy'])}",
                  flush=True)
        except Exception as e:
            print(f"  {nm}: ERR {type(e).__name__}: {e}", flush=True); out[nm] = {"error": str(e)}
    json.dump(out, open(os.path.join(HERE, "a3_screen_smd_pu.json"), "w"), indent=1)
    print(f"done ({time.time()-t0:.0f}s)  [ref: SKAB~0.27 A3; benchmarks<=0.09 no-A3]")
