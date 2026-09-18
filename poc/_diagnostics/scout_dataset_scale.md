# Scout: Dataset Scale vs Operational Scale (LatAD)

Purpose: support the claim that public CPS/ICS anomaly-detection benchmarks are small in
channel count, duration, and regime variability relative to a real operating plant/vehicle/
aircraft, so structural assumptions A8 (between-regime proximity/masking), A9 (multiscale/
many-clocks), A10 (path dependence) can only be verified at operational scale.

NOTE: every DOI tagged "[DOI from memory]" MUST pass bibtest before use. Only URL-verified
figures (SWaT via iTrust, HAI via USENIX PDF, MetroPT via Nature, BDG2 via GitHub/PMC, arXiv
ids) are portal-confirmed.

## (a) Benchmark-size table

| Dataset | Channels | Duration / samples | Sampling | Regimes | Anomalies | Source (verify DOI with bibtest) |
|---|---|---|---|---|---|---|
| SWaT (Dec 2015) | 51 sensors+actuators (6-stage plant) | 11 days (7 normal + 4 attack) | 1 s [FLAG: rate not on iTrust page; community-cited] | 1 nominal regime | 41 attacks | iTrust SWaT portal (verified); Goh et al., CRITIS 2016, DOI 10.1007/978-3-319-71368-7_8 [from memory] |
| WADI | 123 sensors+actuators (~88 sensors, 35 actuators; 3 subsystems) | 16 days (14 normal + 2 attack) | 1 s [FLAG: secondary sources] | 1 nominal | 15 attacks | iTrust WADI portal; Ahmed, Palleti, Mathur, CySWATER 2017, DOI 10.1145/3055366.3055375 [from memory] |
| BATADAL (C-Town) | 43 SCADA variables (7 tanks, 11 pumps, 9 PLCs) | ~1 yr train + 6 mo + 4 mo test; hourly | 1 hour | 1 network | 14 attacks (7 in test) [FLAG] | Taormina et al., J. Water Res. Plan. Manag. 2018, DOI 10.1061/(ASCE)WR.1943-5452.0000969 [from memory] |
| HAI (20.07 / 21.03) | 59-86 columns (SCADA points) by version | ~25 days (~20 normal + ~5.5 attack) [FLAG] | 1 s | 1 nominal | 38 attacks (14 primitives) | Shin et al., USENIX CSET 2020 (verified usenix.org) |
| SKAB v0.9 | 8 sensor signals | 34 series, 37,401 pts (~1,100 each) | 1 s | testbed states | 34 collective-anomaly files | Katser & Kozitsin 2020, DOI 10.34740/kaggle/dsv/1693952 [from memory]; GitHub waico/SKAB (verified) |
| SMD | 38 dims x 28 machines | 5 weeks; 50/50 train/test | 1 min [FLAG] | per-machine | labeled segments | Su et al. (OmniAnomaly), KDD 2019, DOI 10.1145/3292500.3330672 [from memory]; GitHub NetManAIOps/OmniAnomaly (verified) |
| PSM | 25 dims (eBay servers) | [FLAG: duration not confirmed] | - | - | labeled | Abdulaal et al., KDD 2021, DOI 10.1145/3447548.3467174 [from memory] |
| MSL | 55 dims, 27 entities | [FLAG: length not confirmed] | - | telemetry | labeled | Hundman et al., KDD 2018, DOI 10.1145/3219819.3219845 [from memory] |
| SMAP | 25 dims, 55 entities [FLAG] | - | - | telemetry | labeled | Hundman et al., KDD 2018 |
| MetroPT-3 | 20 variables (analog+digital APU + GPS) | Jan-Jun 2022, 10,979,547 pts | 1 Hz | metro service | compressor failures | Veloso et al., Scientific Data 2022, DOI 10.1038/s41597-022-01877-3 (verified) |
| TEP (Tennessee Eastman) | 52 variables (41 measured + 11 manipulated) | sim runs (480 train / 960 test per run) | 3 min | plant modes | 21 (later 28) fault types | Downs & Vogel, Comput. Chem. Eng. 1993, DOI 10.1016/0098-1354(93)80018-I [from memory] |
| N-CMAPSS | 14 sensors (+ ~4 op-conditions) | 40 run-to-failure units, 3 flight classes | 1 Hz | 3 flight classes | run-to-failure degradation | Arias Chao et al., Data 2021, 6(1):5, DOI 10.3390/data6010005 [from memory] |

Consistent picture: nearly all CPS/ICS AD benchmarks sit at ~10-120 channels, days-to-weeks
duration, single nominal regime. Only BATADAL (1 yr, 43 vars, hourly) and N-CMAPSS (40 units,
14 sensors, prognostics not intrusion) reach longer horizons; neither reaches operational
channel count.

## (b) Operational-scale numbers (with citations)

- Process plant / refinery: thousands to tens of thousands of instrumented historian tags;
  mid-size 500-5,000 tags, large facilities more (OSIsoft/AVEVA PI dominant). [FLAG:
  vendor/industry pages, not peer-reviewed; pair with a process-industry academic citation.]
- Commercial aircraft: Quick Access Recorders (QAR) record thousands of flight parameters,
  superset of the FDR frame. Review: "A 25-year journey in Quick Access Recorder (QAR) data"
  (ScienceDirect S0376042126000308) [FLAG: verify exact parameter-count phrasing].
- Automotive fleet: 80+ ECUs, 100+ sensors across multiple CAN buses. "A Survey of Anomaly
  Detection in In-Vehicle Networks" (arXiv 2409.07505). [FLAG: precise "thousands of CAN
  signals" not pinned; supported claim = 80+ ECUs, 100+ sensors.]
- Building portfolio: BDG2 is itself operational scale: 3,053 meters / 1,636 buildings.

Net: real systems instrument 10^3-10^4+ channels over months-to-years; benchmarks are 1-2
orders of magnitude smaller on channels, shorter in duration, poorer in regime coverage.

## (c) Closest-to-operational public datasets

- BDG2: 3,053 meters, 1,636 buildings, 19 sites, 2 years (2016-2017) hourly, ~53.6M
  measurements. Closest on channels AND duration. Miller et al., Scientific Data 2020, 7:368,
  DOI 10.1038/s41597-020-00712-x (sizes verified GitHub buds-lab / PMC7591488).
- N-CMAPSS: closest on duration/regime realism for aerospace (40 units, full run-to-failure,
  real flight profiles) but only 14 channels. Strong on A10 (path dependence).
- BATADAL: longest duration among ICS security sets (~1 yr) but 43 vars, hourly.

By axis: BDG2 wins channels+duration; N-CMAPSS wins regime variability + path dependence; no
public CPS security benchmark reaches operational scale on all three axes at once. This is the
gap A8/A9/A10 need closed.

## (d) Citation-ready references

1. Wu & Keogh, "Current Time Series Anomaly Detection Benchmarks are Flawed...", IEEE TKDE 2023 (arXiv:2009.13807), DOI 10.1109/TKDE.2021.3112126 [DOI from memory; arXiv verified].
2. Sarfraz et al., "Position: Quo Vadis, Unsupervised Time Series Anomaly Detection?", ICML 2024 (arXiv:2405.02678) [verified].
3. Paparrizos et al., "TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection", PVLDB 15(8) 2022, DOI 10.14778/3529337.3529354 [verified].
4. Goh et al., SWaT dataset, CRITIS 2016, DOI 10.1007/978-3-319-71368-7_8 [from memory].
5. Ahmed, Palleti & Mathur, WADI, CySWATER 2017, DOI 10.1145/3055366.3055375 [from memory].
6. Taormina et al., BATADAL, J. Water Res. Plan. Manag. 144(8) 2018, DOI 10.1061/(ASCE)WR.1943-5452.0000969 [from memory].
7. Arias Chao et al., N-CMAPSS, Data 6(1):5 2021, DOI 10.3390/data6010005 [from memory].
8. Miller et al., Building Data Genome Project 2, Scientific Data 7:368 2020, DOI 10.1038/s41597-020-00712-x [sizes verified].

Also: Veloso et al. MetroPT (Scientific Data 2022, DOI 10.1038/s41597-022-01877-3, verified);
Su et al. OmniAnomaly/SMD (KDD 2019).

## (e) Suggested prose (confident, no em-dashes)

Introduction (scope note):
> The public benchmarks that anchor CPS anomaly detection are small. Standard sets such as
> SWaT (51 channels, 11 days), WADI (123 channels, 16 days), HAI (up to 86 points, about 25
> days), SKAB (8 signals), and SMD (38 dimensions, five weeks) each span tens of channels,
> days to weeks of operation, and a single nominal regime. A real operating plant, aircraft,
> or vehicle instruments orders of magnitude more streams over months to years and moves
> through many operating regimes, so results obtained only on these benchmarks describe a
> narrow slice of the operational envelope.

Related Work (benchmark datasets):
> The water-treatment and ICS security datasets (SWaT, WADI, BATADAL, HAI) and the
> multivariate telemetry sets (SMD, MSL, SMAP, PSM) share a scale profile: roughly 10 to 120
> channels, days-to-weeks duration, and one nominal operating mode. BATADAL extends to about a
> year but keeps only 43 hourly variables; N-CMAPSS reaches a 40-unit fleet with full
> run-to-failure trajectories but exposes only 14 sensors. The Building Data Genome Project 2,
> with 3,053 meters across 1,636 buildings over two years, and process historians that carry
> thousands to tens of thousands of tags illustrate the operational scale these security
> benchmarks do not reach. A parallel line of work (Wu and Keogh; Sarfraz et al.; TSB-UAD)
> argues that benchmark realism, not model novelty, now limits measured progress.

Discussion / future work:
> Assumptions A8 (between-regime proximity and masking), A9 (multiscale, many-clocks
> structure), and A10 (path dependence) concern properties that only appear when a system has
> many channels, long duration, and rich regime variability. The current benchmarks cannot
> exercise them: with a single nominal regime and days of data, there is little between-regime
> masking to observe, few independent clocks, and short trajectories with limited path
> history. Verifying A8 through A10 therefore requires operational-scale data, for example the
> multi-year, thousand-stream BDG2 portfolio, the fleet run-to-failure trajectories of
> N-CMAPSS, or historian exports from an instrumented plant. We flag this as the primary axis
> for follow-up evaluation.

## Gaps / flags
- SWaT/WADI/HAI/SMD sampling rates (1 Hz / 1 min) are community-standard but not all confirmed
  on a canonical portal this pass; verify before quoting a rate.
- All "[DOI from memory]" must pass bibtest before use.
- PSM/MSL/SMAP sequence lengths/sample counts not re-verified (only dimensionality).
- "Process plant = tens of thousands of tags" rests on vendor/industry pages; pair with a
  process-industry academic citation before formal use.
- Precise "thousands of CAN signals" not pinned; supported claim = "80+ ECUs, 100+ sensors,
  multiple CAN buses."
