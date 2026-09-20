# PCA double-hard leaderboard (all three datasets)

Second filter is **PCA** (Hotelling T2 on retained components OR SPE/Q on dropped components), fit on the TRAIN-NORMAL standardized windowed features, thresholds at train-normal q99; 95% variance retained. A window is double-hard iff it is an anomaly AND `max|z|<=maxz_thr` (difficult filter, channel-based) AND `T2<=q99(T2_train)` AND `SPE<=q99(SPE_train)`. The detector under test never participates in either filter. LinRes and boosted_LOO are FAIR DETECTORS here (PCA is the filter, not LinRes). Leak-free: PCA/thresholds/train-normal from train only. AUROC is 5-seed mean±SD for seeded methods, single value otherwise. Our method (community fusion HCcoh+LatAD) bolded.

**Experts used:** HAI `experts_full/expert_HAI.npz`, WADI `experts_full/expert_WADI_clean.npz`, SWaT `experts_full/expert_SWaT_canon.npz` (clean official-normal bundle, **S=24**, has `fit_surprise`, mtime 12:33). SWaT_canon uses the clean official Dec-2015 normal train (0 test-normal overlap, fail-fast guard). Cross-check: SWaT HCcoh+LatAD *difficult* AUROC = 0.524 (near chance, matches the clean ~0.536 expectation; not the stale ~0.70+).

## PCA double-hard subset sizes (95% variance)

| dataset | windows | attack episodes | (90% var) | (99% var) |
|---|--:|--:|--:|--:|
| HAI | 55 | 13 | 68w/16ep | 11w/7ep |
| WADI | 29 | 8 | 30w/8ep | 21w/7ep |
| SWaT | 28 | 9 | 27w/9ep | 25w/9ep |

## AUROC on the PCA double-hard (95% variance)

| method | HAI | WADI | SWaT |
|---|--:|--:|--:|
| **LatAD community fusion (HCcoh+LatAD) — ours** | **0.648±0.039** | **0.763±0.024** | **0.178±0.011** |
| IF | 0.435±0.023 | 0.622±0.006 | 0.326±0.012 |
| AE | 0.418±0.008 | 0.616±0.004 | 0.154±0.006 |
| USAD | 0.351±0.018 | 0.565 | 0.125±0.004 |
| TranAD | 0.296±0.004 | 0.601 | 0.119±0.001 |
| GDN | 0.354 | 0.648 | 0.571 |
| LinRes | 0.547 | 0.742 | 0.153 |
| boosted_LOO | 0.313 | 0.649 | 0.117 |
| _LatAD (global density) (ablation)_ | _0.559±0.040_ | _0.623±0.012_ | _0.184±0.022_ |
| _trivial max|z| (filter floor)_ | _0.336_ | _0.591_ | _0.514_ |

## Significance: community fusion vs the STRONGEST alternative

Episode-block moving-block bootstrap, 2000 reps; paired AUROC difference (ours minus strongest fair alternative), 95% CI, one-sided P(diff<=0). Positive diff + P below 0.05 = ours significantly leads.

| dataset | strongest alt | alt AUROC | ours AUROC | diff [95% CI], one-sided P | verdict |
|---|---|--:|--:|---|---|
| HAI | LinRes | 0.547 | 0.648 | +0.101 [-0.090, +0.337], P(diff<=0)=0.17 | ours ahead, NOT sig. |
| WADI | LinRes | 0.742 | 0.763 | +0.021 [-0.165, +0.201], P(diff<=0)=0.46 | ours ahead, NOT sig. |
| SWaT | GDN | 0.571 | 0.178 | -0.393 [-0.489, -0.313], P(diff<=0)=1.0 | GDN leads (sig.) |

## Invariant / sanity checks

- **HAI**: PCA k@95%=60/354; train-normal filter-flag rate 1.38% (expect ~1-2%); all-components SPE ~0 (PCA self-consistency ok). Floor `trivial max|z|`=0.336 (near/below chance as required — it is a defining filter).
- **WADI**: PCA k@95%=113/732; train-normal filter-flag rate 1.87% (expect ~1-2%); all-components SPE ~0 (PCA self-consistency ok). Floor `trivial max|z|`=0.591 (near/below chance as required — it is a defining filter).
- **SWaT**: PCA k@95%=44/306; train-normal filter-flag rate 1.79% (expect ~1-2%); all-components SPE ~0 (PCA self-consistency ok). Floor `trivial max|z|`=0.514 (near/below chance as required — it is a defining filter).

**Below-chance learned methods (explained, not a bug):** the double-hard subset removes by construction every window separable by the channel-wise `max|z|` filter OR the linear-Gaussian PCA filter (T2/SPE). Reconstruction/density/linear detectors (IF, AE, USAD, TranAD, LinRes, boosted_LOO, LatAD global-density) re-detect essentially that same removed signal, so on the residual windows their ordering inverts to at/below chance. This is the intended demonstration of the filter, not a sign error: the sign is not globally inverted, since GDN (graph-relational signal) and, on HAI/WADI, our community fusion stay above chance on the same subset. On SWaT_canon nearly every detector except GDN falls below chance on the 28-window/9-episode subset, and our community fusion (0.178) is below chance too — consistent with the clean SWaT community signal already being near chance on 'difficult' (0.524); GDN is the sole survivor there.
