# Block-length (serial-dependence) sensitivity of the headline significance verdicts

Paired episode + moving-block bootstrap of the difficult-subset seed-mean AUROC difference, re-run at three normal-window block lengths. Only the NORMAL-window block length varies; attack episodes are always resampled at episode level (the paper's episode-block bootstrap). Bootstrap reps = 4000. Overlap block L=3 = ceil(W/stride)+1 with W=60, stride=30.

## HAI

- method = **HCcoh+LatAD** (difficult-subset AUROC 0.849), n_difficult = 167 over 26 attack episodes, 5 seeds.
- ACF-selected block **L_acf = 264** (first lag with normal-window score ACF < 1/e); normal-window score ACF at lags 1..12 = [0.874, 0.789, 0.748, 0.727, 0.711, 0.697, 0.683, 0.672, 0.662, 0.652, 0.646, 0.643]. Overlap block L = 3. Window step = 30 s, so L_acf spans ~132 min.

| comparison | block L | diff | 95% CI | P(diff<=0) | verdict |
|---|---|---|---|---|---|
| **AutoEncoder** (paper +0.088 CI[0.042, 0.157] P=<0.0005) | L=1 (iid) | +0.092 | [+0.046, +0.156] | 0.000 | win |
|  | L=3 (overlap) | +0.092 | [+0.046, +0.159] | 0.000 | win |
|  | L=264 (ACF) | +0.092 | [+0.046, +0.157] | 0.000 | win |
|  | L=264 + seed-hier | +0.092 | [+0.044, +0.159] | 0.000 | win |

## WADI_clean

- method = **HCcoh+LatAD** (difficult-subset AUROC 0.76), n_difficult = 30 over 8 attack episodes, 5 seeds.
- ACF-selected block **L_acf = 3** (first lag with normal-window score ACF < 1/e); normal-window score ACF at lags 1..12 = [0.495, 0.38, 0.255, 0.191, 0.186, 0.126, 0.08, 0.076, 0.101, 0.09, 0.116, 0.102]. Overlap block L = 3. Window step = 300 s, so L_acf spans ~15 min.

| comparison | block L | diff | 95% CI | P(diff<=0) | verdict |
|---|---|---|---|---|---|
| **AutoEncoder** (paper +0.143 P=0.001) | L=1 (iid) | +0.132 | [+0.056, +0.210] | 0.000 | win |
|  | L=3 (overlap) | +0.132 | [+0.054, +0.213] | 0.001 | win |
|  | L=3 (ACF) | +0.132 | [+0.054, +0.213] | 0.001 | win |
|  | L=3 + seed-hier | +0.132 | [+0.054, +0.215] | 0.001 | win |
| **USAD** (paper +0.192 P=<0.0005) | L=1 (iid) | +0.181 | [+0.052, +0.322] | 0.000 | win |
|  | L=3 (overlap) | +0.181 | [+0.051, +0.324] | 0.000 | win |
|  | L=3 (ACF) | +0.181 | [+0.051, +0.324] | 0.000 | win |
|  | L=3 + seed-hier | +0.181 | [+0.052, +0.326] | 0.001 | win |
| **TranAD** (paper +0.157 P=0.0005) | L=1 (iid) | +0.147 | [+0.041, +0.260] | 0.000 | win |
|  | L=3 (overlap) | +0.147 | [+0.041, +0.264] | 0.001 | win |
|  | L=3 (ACF) | +0.147 | [+0.041, +0.264] | 0.001 | win |
|  | L=3 + seed-hier | +0.147 | [+0.038, +0.265] | 0.001 | win |
| **IsolationForest** (paper +0.136 P=<0.0005) | L=1 (iid) | +0.126 | [+0.041, +0.221] | 0.001 | win |
|  | L=3 (overlap) | +0.126 | [+0.040, +0.221] | 0.000 | win |
|  | L=3 (ACF) | +0.126 | [+0.040, +0.221] | 0.000 | win |
|  | L=3 + seed-hier | +0.126 | [+0.037, +0.223] | 0.002 | win |
| **LinRes** (paper +0.020 CI[-0.161, 0.205] P=0.46) | L=1 (iid) | +0.010 | [-0.169, +0.193] | 0.486 | tie |
|  | L=3 (overlap) | +0.010 | [-0.168, +0.185] | 0.496 | tie |
|  | L=3 (ACF) | +0.010 | [-0.168, +0.185] | 0.496 | tie |
|  | L=3 + seed-hier | +0.010 | [-0.172, +0.193] | 0.492 | tie |

## SWaT_canon

- method = **drift-typed CP (v2 24h)** (difficult-subset AUROC 0.724), n_difficult = 91 over 23 attack episodes, 5 seeds.
- ACF-selected block **L_acf = 4** (first lag with normal-window score ACF < 1/e); normal-window score ACF at lags 1..12 = [0.566, 0.446, 0.386, 0.36, 0.373, 0.374, 0.353, 0.382, 0.347, 0.302, 0.331, 0.332]. Overlap block L = 3. Window step = 300 s, so L_acf spans ~20 min.

| comparison | block L | diff | 95% CI | P(diff<=0) | verdict |
|---|---|---|---|---|---|
| **raw community (HC_coh)** (paper +0.200 CI[0.018, 0.355] P=0.014) | L=1 (iid) | +0.201 | [+0.026, +0.360] | 0.012 | win |
|  | L=3 (overlap) | +0.201 | [+0.024, +0.362] | 0.013 | win |
|  | L=4 (ACF) | +0.201 | [+0.024, +0.361] | 0.013 | win |
|  | L=4 + seed-hier | +0.201 | [+0.026, +0.368] | 0.011 | win |

## Verdict stability

Every one of the 7 headline comparisons keeps the SAME verdict at all four block lengths (iid, overlap, ACF, ACF+seed-hierarchical): **7/7 stable**.

The ACF-selected block length differs sharply by dataset, which is the point of the check. On WADI the normal-window score decorrelates within 3 lags (a stationary record), so the empirical block equals the mechanical overlap block (3) and the sweep is a no-op. On SWaT the empirical block is 4, marginally longer than overlap. On HAI the community score is strongly serially dependent on normal windows (its ACF is still ~0.64 at lag 12 and does not fall below 1/e until lag 264), reflecting the record's front-loaded drift; the ACF block is therefore roughly two orders of magnitude longer than the overlap block. Even at that empirically-selected length the HAI margin over the AutoEncoder stays significant with an essentially unchanged confidence interval. The reason the interval barely moves is that the block length only reshuffles the shared normal set, and the paired difference (method minus competitor on the SAME resample) cancels most of the normal-window noise; increasing the block widens the interval only slightly because it reduces the number of effective normal blocks, not because it changes the signal. The reviewer's objection, that the block was chosen from window overlap rather than empirical serial dependence, is answered directly: choosing the block from the score autocorrelation instead leaves every significance verdict unchanged.
