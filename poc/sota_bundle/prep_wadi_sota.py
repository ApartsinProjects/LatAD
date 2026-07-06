"""Prepare WADI arrays for the SOTA harness (USAD/TranAD/GDN) with the SAME preprocessing
our detector uses: standardize on normal + clip to +-10 sigma (NOT MinMax-to-[0,1]).

The earlier prep used MinMax-fit-on-normal, which clamps every supra-normal excursion to
1.0 and destroys the near-constant 'attack-carrier' channels (audit finding 1A) --
a confound that handicaps SOTA. This version matches eda_real/wadi exactly so the
comparison is preprocessing-fair. x10 downsample, 123 channels, per-timestep labels.
"""
from __future__ import annotations
import os, numpy as np, pandas as pd

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "datasets", "itrust", "WADI", "WADI.A2_19 Nov 2019")
OUT = os.path.dirname(os.path.abspath(__file__))
DS, CLIP = 10, 10.0


def main():
    nrm = pd.read_csv(f"{DIR}/WADI_14days_new.csv"); atk = pd.read_csv(f"{DIR}/WADI_attackdataLABLE.csv", skiprows=1)
    nrm.columns = [c.strip() for c in nrm.columns]; atk.columns = [c.strip() for c in atk.columns]
    lc = atk.columns[-1]; meta = set(list(atk.columns[:3]) + list(nrm.columns[:3]) + [lc])
    sens = [c for c in atk.columns if c not in meta and c in nrm.columns and nrm[c].isna().mean() < 0.5]
    prep = lambda df: df[sens].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0).values.astype(np.float32)
    Xn = prep(nrm)[::DS]; Xa = prep(atk)[::DS]
    ya = (pd.to_numeric(atk[lc], errors="coerce").values == -1).astype(np.int64)[::DS]
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-8                       # standardize on normal (matches ours)
    Xn = np.clip((Xn - mu) / sd, -CLIP, CLIP).astype(np.float32)
    Xa = np.clip((Xa - mu) / sd, -CLIP, CLIP).astype(np.float32)
    np.save(f"{OUT}/wadi_train.npy", Xn); np.save(f"{OUT}/wadi_test.npy", Xa); np.save(f"{OUT}/wadi_labels.npy", ya)
    # trivial score on the SAME clipped features (window-level easy/hard uses this convention)
    triv = np.abs(Xa).max(1).astype(np.float32); thr = float(np.quantile(np.abs(Xn).max(1), 0.99))
    np.save(f"{OUT}/wadi_triv_test.npy", triv); np.save(f"{OUT}/wadi_triv_thr.npy", np.float32(thr))
    print(f"train {Xn.shape} test {Xa.shape} anom {ya.mean():.3f} chan {len(sens)} "
          f"clip +-{CLIP} triv_thr {thr:.2f} (standardize+clip, matches ours)")


if __name__ == "__main__":
    main()
