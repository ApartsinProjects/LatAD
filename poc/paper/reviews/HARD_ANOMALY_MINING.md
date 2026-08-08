# Hardest-anomaly mining (difficult-subset windows LatAD scores most normal)

## WADI — 19 difficult, 8 worst-caught characterized

- Median score-percentile among all windows (low = missed): **LatAD 0.48**, IF 0.439, AE 0.221, LinRes 0.259
- Recurring top channels: 2B_AIT_004_PV×8, 2A_AIT_004_PV×8, 2B_AIT_003_PV×6, 1_AIT_004_PV×5, 3_FIT_001_PV×4, 1_AIT_003_PV×4, 2_FIC_501_SP×4, 3_LT_001_PV×3
- Temporal shapes: [['flat-offset', 48]]

| window | ep | LatAD_pct | IF_pct | AE_pct | LinRes_pct | top channels (z, shape) |
|---|---|---|---|---|---|---|
| 18 | 1 | 0.176 | 0.59 | 0.16 | 0.123 | 2B_AIT_004_PV (z=-2.65, flat-offset); 1_AIT_004_PV (z=-2.42, flat-offset); 2A_AIT_004_PV (z=-2.11, flat-offset) |
| 16 | 1 | 0.198 | 0.343 | 0.186 | 0.101 | 2B_AIT_004_PV (z=-2.63, flat-offset); 2A_AIT_004_PV (z=-2.1, flat-offset); 3_FIT_001_PV (z=-1.93, flat-offset) |
| 21 | 1 | 0.409 | 0.588 | 0.235 | 0.012 | 1_AIT_004_PV (z=-3.07, flat-offset); 2B_AIT_004_PV (z=-2.62, flat-offset); 2A_AIT_004_PV (z=-2.15, flat-offset) |
| 238 | 4 | 0.452 | 0.36 | 0.233 | 0.395 | 2A_AIT_004_PV (z=-2.91, flat-offset); 2B_AIT_004_PV (z=-2.85, flat-offset); 2B_AIT_002_PV (z=-2.24, flat-offset) |
| 20 | 1 | 0.508 | 0.607 | 0.212 | 0.089 | 1_AIT_004_PV (z=-2.95, flat-offset); 2B_AIT_004_PV (z=-2.67, flat-offset); 2A_AIT_004_PV (z=-2.14, flat-offset) |
| 195 | 2 | 0.529 | 0.174 | 0.073 | 0.409 | 2_FIC_401_SP (z=2.29, flat-offset); 2B_AIT_004_PV (z=-1.94, flat-offset); 2A_AIT_004_PV (z=-1.83, flat-offset) |
| 237 | 4 | 0.541 | 0.362 | 0.23 | 0.419 | 2A_AIT_004_PV (z=-3.02, flat-offset); 2B_AIT_004_PV (z=-2.9, flat-offset); 2B_AIT_002_PV (z=-2.46, flat-offset) |
| 236 | 4 | 0.57 | 0.515 | 0.343 | 0.431 | 2A_AIT_004_PV (z=-2.99, flat-offset); 2B_AIT_004_PV (z=-2.61, flat-offset); 3_LT_001_PV (z=2.37, flat-offset) |

## HAI — 167 difficult, 8 worst-caught characterized

- Median score-percentile among all windows (low = missed): **LatAD 0.17**, IF 0.063, AE 0.111, LinRes 0.635
- Recurring top channels: P1_FT01×5, P1_FCV02Z×5, P1_PCV01D×4, P1_PCV02Z×4, P3_LT01×4, P1_PCV01Z×3, P3_LCP01D×3, P1_FCV03D×3
- Temporal shapes: [['flat-offset', 48]]

| window | ep | LatAD_pct | IF_pct | AE_pct | LinRes_pct | top channels (z, shape) |
|---|---|---|---|---|---|---|
| 3432 | 11 | 0.026 | 0.037 | 0.102 | 0.609 | P1_PCV01D (z=1.76, flat-offset); P1_PCV01Z (z=1.74, flat-offset); P1_B2016 (z=1.52, flat-offset) |
| 14755 | 38 | 0.058 | 0.089 | 0.278 | 0.714 | P1_PCV02Z (z=-3.26, flat-offset); P1_FCV03D (z=1.89, flat-offset); P1_B2004 (z=-1.76, flat-offset) |
| 11939 | 31 | 0.139 | 0.213 | 0.009 | 0.654 | P1_FCV03Z (z=2.12, flat-offset); P1_FCV03D (z=1.88, flat-offset); P1_B2004 (z=-1.77, flat-offset) |
| 11940 | 31 | 0.168 | 0.276 | 0.047 | 0.987 | P1_FCV03Z (z=2.0, flat-offset); P1_FCV03D (z=1.99, flat-offset); P1_B2004 (z=-1.77, flat-offset) |
| 2960 | 8 | 0.171 | 0.019 | 0.062 | 0.635 | P1_B3004 (z=1.55, flat-offset); P1_LIT01 (z=1.44, flat-offset); P3_LT01 (z=-1.36, flat-offset) |
| 3076 | 9 | 0.175 | 0.461 | 0.251 | 0.575 | P1_B2016 (z=2.29, flat-offset); P4_ST_LD (z=1.9, flat-offset); P4_ST_PO (z=1.89, flat-offset) |
| 2958 | 8 | 0.177 | 0.014 | 0.123 | 0.634 | P1_B3004 (z=1.55, flat-offset); P1_LIT01 (z=1.51, flat-offset); P3_LT01 (z=-1.37, flat-offset) |
| 2959 | 8 | 0.184 | 0.01 | 0.12 | 0.633 | P1_B3004 (z=1.55, flat-offset); P1_LIT01 (z=1.47, flat-offset); P3_LT01 (z=-1.41, flat-offset) |

## SWaT — 38 difficult, 8 worst-caught characterized

- Median score-percentile among all windows (low = missed): **LatAD 0.768**, IF 0.564, AE 0.788, LinRes 0.798
- Recurring top channels: P201×8, PIT502×6, AIT402×6, AIT501×5, AIT502×5, AIT202×4, LIT401×3, FIT301×2
- Temporal shapes: [['flat-offset', 45], ['transient', 3]]

| window | ep | LatAD_pct | IF_pct | AE_pct | LinRes_pct | top channels (z, shape) |
|---|---|---|---|---|---|---|
| 949 | 1 | 0.423 | 0.285 | 0.341 | 0.795 | P201 (z=-1.57, flat-offset); AIT501 (z=-1.34, flat-offset); PIT502 (z=-1.32, flat-offset) |
| 948 | 1 | 0.463 | 0.24 | 0.327 | 0.801 | P201 (z=-1.57, flat-offset); AIT501 (z=-1.39, flat-offset); PIT502 (z=-1.3, flat-offset) |
| 950 | 1 | 0.739 | 0.779 | 0.638 | 0.763 | P201 (z=-1.57, flat-offset); PIT502 (z=-1.28, flat-offset); AIT501 (z=-1.27, flat-offset) |
| 951 | 1 | 0.747 | 0.818 | 0.777 | 0.778 | FIT301 (z=-2.08, flat-offset); P302 (z=-2.05, flat-offset); DPIT301 (z=-1.99, flat-offset) |
| 947 | 1 | 0.788 | 0.321 | 0.798 | 0.794 | P201 (z=-1.57, flat-offset); AIT501 (z=-1.44, flat-offset); PIT502 (z=-1.26, flat-offset) |
| 1072 | 1 | 0.816 | 0.881 | 0.84 | 0.838 | MV101 (z=-1.68, transient); P201 (z=-1.57, flat-offset); AIT202 (z=1.52, flat-offset) |
| 946 | 1 | 0.82 | 0.349 | 0.81 | 0.828 | LIT401 (z=-1.64, flat-offset); P201 (z=-1.57, flat-offset); AIT501 (z=-1.46, flat-offset) |
| 911 | 1 | 0.823 | 0.855 | 0.808 | 0.8 | AIT201 (z=2.99, flat-offset); PIT502 (z=2.15, flat-offset); MV201 (z=-1.75, flat-offset) |
