"""A8 (between-regime overlap) screen on real CAN-bus / OBD-II vehicle driving
telemetry (Kaggle cephasax/obdii-ds3, exp2: 19 drivers, 1 car, 1 shared route,
Buenos Aires, logged via an OBD-II Bluetooth dongle, CC0-1.0).

WHY THIS DATASET: WADI/HAI/SWaT/Cranfield are industrial process-control loops
that sit in one setpoint per "regime" with sharp switches between them (crisp,
non-overlapping regimes; A8 absent everywhere so far, and the one apparent
SKAB "witness" collapsed under the variance-floor refit). Driving is the
opposite kind of system: accelerator/brake/gear-shift/cruise blend into each
other continuously (you are always partway between "accelerating" and
"cruising"), so if between-regime overlap exists anywhere in this project it
should show up here.

Channels (8, all continuous real physical vehicle-dynamic signals sampled by
the OBD-II reader every 4s): SPEED, ENGINE_RPM, THROTTLE_POS, ENGINE_LOAD,
ENGINE_COOLANT_TEMP, INTAKE_MANIFOLD_PRESSURE, MAF, TIMING_ADVANCE.
VEHICLE_ID (19 distinct driver/trip IDs, s1..s19) segments the file into 19
continuous normal-driving trips over the same route; windows are built
per-trip (never across a trip boundary) and pooled. No TROUBLE_CODES rows in
the whole file, i.e. no fault/intrusion data -- purely normal driving.

Method: identical recipe to the WADI/HAI/SWaT/Cranfield/SKAB A8 screens --
models_vade.train_vade (VaDE) + winfeat.window_features(...,"stats"), K-sweep
mean max-responsibility / normalized responsibility entropy, then the
mandatory variance-floor sanity check (empirical per-component variance
refit at the same latent z) on the highest-entropy K across 3 seeds. A
result only counts as a genuine A8 witness if entropy stays high under the
empirical refit (does not collapse) AND is not just from components pinned to
the model's logvar floor.

Reference bands (established across this project):
  WADI/HAI/SWaT/Cranfield  H_norm ~0.01-0.15  -- NO overlap
  SKAB (before floor check) H_norm ~0.06-0.53 -- looked positive, COLLAPSED
                                                   under empirical-variance
                                                   refit (floor artifact)

Persists incrementally to canbus_a8_screen.json (one row per K, flushed).
"""
from __future__ import annotations
import os, sys, json, re, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
POC = os.path.dirname(HERE)
sys.path.insert(0, POC)
from models_vade import train_vade
from winfeat import window_features

CSV = os.path.join(POC, "datasets", "_new", "CANdrive", "exp2_19drivers_1car_1route.csv")
OUT = os.path.join(HERE, "canbus_a8_screen.json")

CHANNELS = ["SPEED", "ENGINE_RPM", "THROTTLE_POS", "ENGINE_LOAD",
            "ENGINE_COOLANT_TEMP", "INTAKE_MANIFOLD_PRESSURE", "MAF", "TIMING_ADVANCE"]

W, ST = 6, 3     # 4s sample interval -> 6 samples = 24s window / 3 samples = 12s stride
LD = 8
KS = [8, 16, 32, 64, 128]
LOG2PI = np.log(2 * np.pi)


def _clean_numeric(s):
    """Strip unit suffixes (km/h, RPM, %, C, kPa, g/s) and convert the
    comma-decimal European format to a float; garbage OBD glitches (e.g. a raw
    hex frame like '1:6032000007E803' instead of a barometric reading) coerce
    to NaN and are dropped."""
    s = s.astype(str).str.replace(r"[a-zA-Z/%]+", "", regex=True).str.strip()
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load_canbus_normal():
    df = pd.read_csv(CSV)
    for c in CHANNELS:
        df[c] = _clean_numeric(df[c])
    df = df.dropna(subset=CHANNELS + ["VEHICLE_ID", "ENGINE_RUNTIME"]).copy()
    # ENGINE_RUNTIME "HH:MM:SS" (occasionally malformed) -> seconds, sort key only
    def rt_to_sec(x):
        parts = str(x).split(":")
        if len(parts) != 3:
            return np.nan
        try:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
        except ValueError:
            return np.nan
    df["_rt"] = df["ENGINE_RUNTIME"].map(rt_to_sec)
    df = df.dropna(subset=["_rt"]).copy()
    trips = {}
    for vid, g in df.groupby("VEHICLE_ID"):
        g = g.sort_values("_rt")
        trips[vid] = g[CHANNELS].to_numpy(np.float64)
    return trips


def window(trips):
    A = []
    per_trip = {}
    for vid, X in trips.items():
        n = 0
        for i in range(0, len(X) - W + 1, ST):
            A.append(window_features(X[i:i + W], "stats"))
            n += 1
        per_trip[vid] = n
    return np.asarray(A, np.float32), per_trip


def entropy_norm(G):
    K = G.shape[1]
    return float(np.mean(-(G * np.log(G + 1e-12)).sum(1) / np.log(K)))


def resp_from_var(z, mu, var, logpi):
    lv = np.log(var)
    logN = -0.5 * (LOG2PI * z.shape[1] + (lv[None] + (z[:, None, :] - mu[None]) ** 2 / var[None]).sum(2))
    l = logpi[None] + logN
    return np.exp(l - logsumexp(l, 1, keepdims=True))


def empirical_refit(v, X):
    with torch.no_grad():
        z = v.encode(torch.as_tensor(X))[0].numpy().astype(np.float64)
    G0 = v._responsibilities(X)
    lab = G0.argmax(1)
    mu = v.mu_c.detach().numpy().astype(np.float64)
    lvc = v._lvc().detach().numpy().astype(np.float64)
    logpi = torch.log_softmax(v.pi_logit, 0).detach().numpy().astype(np.float64)
    K = mu.shape[0]
    var_hat = np.stack([
        z[lab == k].var(0) + 1e-4 if (lab == k).sum() > 1 else np.exp(lvc[k])
        for k in range(K)
    ])
    Ge = resp_from_var(z, mu, var_hat, logpi)
    frac_at_floor = float((np.abs(lvc - v.logvar_floor) < 1e-6).mean())
    return Ge, frac_at_floor, float(np.exp(0.5 * lvc).mean()), float(np.sqrt(var_hat).mean())


def main():
    print("Loading CAN-bus/OBD-II normal driving trips ...", flush=True)
    trips = load_canbus_normal()
    for vid, X in trips.items():
        print(f"  {vid}: {X.shape}", flush=True)
    print(f"{len(trips)} trips, {sum(len(x) for x in trips.values())} total rows", flush=True)

    Xw, per_trip = window(trips)
    print(f"Pooled windows: {Xw.shape}, per-trip counts {per_trip}", flush=True)

    mu, sig = Xw.mean(0), Xw.std(0) + 1e-8
    Xs = ((Xw - mu) / sig).astype(np.float32)

    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    R["_meta"] = dict(csv=CSV, channels=CHANNELS, n_trips=len(trips), n_rows_total=sum(len(x) for x in trips.values()),
                       W=W, ST=ST, n_windows=int(len(Xs)))
    json.dump(R, open(OUT, "w"), indent=1)
    t0 = time.time()
    for K in KS:
        key = f"K{K}"
        if key in R and "H_norm" in R[key]:
            print(f"cached {key}: {R[key]}", flush=True)
            continue
        v = train_vade(Xs, n_clusters=K, latent_dim=LD, epochs=40, warmup=8, seed=0, device="cpu")
        G = v._responsibilities(Xs)
        mr = G.max(1)
        row = dict(K=K, n_windows=int(len(Xs)),
                   mean_maxresp=round(float(mr.mean()), 4),
                   H_norm=round(entropy_norm(G), 4),
                   rho_lt0p5=round(float((mr < 0.5).mean()), 4),
                   rho_lt0p6=round(float((mr < 0.6).mean()), 4))
        R[key] = row
        json.dump(R, open(OUT, "w"), indent=1)
        print(f"[K={K:3d}] mean_maxresp={row['mean_maxresp']:.3f} H_norm={row['H_norm']:.3f} "
              f"rho(<0.5)={row['rho_lt0p5']:.3f} rho(<0.6)={row['rho_lt0p6']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    best_key = max((k for k in R if k.startswith("K") and "H_norm" in R[k]), key=lambda k: R[k]["H_norm"])
    best_K = R[best_key]["K"]
    print(f"\nHighest-entropy K = {best_K} (H_norm={R[best_key]['H_norm']}); "
          f"running variance-floor sanity check across seeds 0-2 ...", flush=True)

    if "variance_floor_check" not in R:
        R["variance_floor_check"] = {}
    for seed in range(3):
        skey = f"K{best_K}_seed{seed}"
        if skey in R["variance_floor_check"]:
            print(f"cached {skey}: {R['variance_floor_check'][skey]}", flush=True)
            continue
        v = train_vade(Xs, n_clusters=best_K, latent_dim=LD, epochs=40, warmup=8, seed=seed, device="cpu")
        G = v._responsibilities(Xs)
        H_floor = entropy_norm(G)
        rho_floor = float((G.max(1) < 0.5).mean())
        Ge, frac_at_floor, comp_sd, emp_sd = empirical_refit(v, Xs)
        H_emp = entropy_norm(Ge)
        rho_emp = float((Ge.max(1) < 0.5).mean())
        row = dict(K=best_K, seed=seed, H_floor=round(H_floor, 4), rho_floor=round(rho_floor, 4),
                   H_emp=round(H_emp, 4), rho_emp=round(rho_emp, 4),
                   frac_components_at_floor=round(frac_at_floor, 4),
                   floored_comp_sd=round(comp_sd, 4), empirical_comp_sd=round(emp_sd, 4),
                   collapse=bool(H_emp < 0.5 * H_floor))
        R["variance_floor_check"][skey] = row
        json.dump(R, open(OUT, "w"), indent=1)
        print(f"[floor-check K={best_K} seed={seed}] H_floor={row['H_floor']:.3f} -> H_emp={row['H_emp']:.3f} "
              f"(rho {row['rho_floor']:.3f} -> {row['rho_emp']:.3f})  frac_at_floor={row['frac_components_at_floor']:.2f} "
              f"comp_sd(floored)={row['floored_comp_sd']:.3f} vs emp_sd={row['empirical_comp_sd']:.3f}  "
              f"collapse={row['collapse']}", flush=True)

    print(f"\ndone ({time.time()-t0:.0f}s total). See {OUT}", flush=True)


if __name__ == "__main__":
    main()
