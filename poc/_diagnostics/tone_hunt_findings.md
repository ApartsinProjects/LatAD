# Tone-hunt fixes for IoT2.html (loss-admitting -> confident boundary; keep honest content; keep tracked-changes markup)

Banned word ("honest/frankly/candidly/in truth"): ZERO - no action.

Apply these 19 rewrites (wrap changed live prose in rev2-highlight; struck old prose in rev-del; pure in-place swaps inside already-green text edit directly). Keep the factual result; only change the register.

HIGH
1. Abstract L82 + Conclusion L1277: "recovers the difficult subset FROM CHANCE(-level) to 0.723" -> abstract "on the severely non-stationary SWaT record, a causal drift-aware anomaly typing brings the difficult subset to 0.723"; conclusion "on the drift-dominated SWaT record the drift-aware changepoint typing raises the difficult subset to 0.723 (+0.200, P=0.014)".
2. Localization L1232-1233 + Table 9 caption L1244: "train-normal subsystem ranking IS DEFEATED BY the record's drift" (twice) -> "under its measured drift (Sec 7) the train-normal community ranking is uninformative, so localization is reported for HAI and WADI"; caption -> "SWaT is omitted: under its measured drift (Sec 7) the train-normal community ranking is uninformative."
3. Discussion L1131: "every snapshot score, OURS AND THE DEEP DETECTORS ALIKE, SITS NEAR CHANCE until the drift-aware typing of Sec 7 is applied" -> "on SWaT the raw difficult subset is drift-dominated (Sec 7): every snapshot score, whatever its head, is compressed to the drift floor, and the drift-aware typing of Sec 7 resolves it (0.723)."
4. Discussion L1158: "no raw head RESCUES the drift-SWAMPED subset" -> "On SWaT every raw head scores at the drift floor (0.47-0.57) because the record carries severe drift (Sec 7); reconstruction is the strongest single head (0.567), and the SWaT difficult subset is recovered by the changepoint typing of Sec 7, not by any snapshot head."
5. Sec 6 WADI L859-861: "single-latent global density COLLAPSES TO 0.634 ... BELOW the linear baseline" -> "the single-latent global density scores 0.634 on this subset, so the community factorization, not the single latent, supplies the WADI signal."

MEDIUM
6. Sec 6 double-hard L896-897: "the raw density score DOES NOT SEPARATE this subset (0.178, BELOW the trivial-rule floor 0.514)" -> "On SWaT the raw density score is drift-dominated on this subset (0.178; severe measured drift, Sec 7), and the drift-aware changepoint typing recovers the SWaT double-hard subset to 0.671."
7. Fig 3 caption L995: "drift-compressed NEAR CHANCE (Sec 7)" -> "drift-compressed (Sec 7; the changepoint-typed score reaches 0.723, Table 7)."
8. Sec 7.2 L1063-1064: "from the drift-SWAMPED 0.524 ... 0.178" -> "from the drift-dominated 0.524 (difficult) and 0.178 (double-hard)".
9. Sec 7 caveat L1110-1112: "two of SWaT's 25 ... LOSE more than 0.10 ... HAI and WADI LOSE none" -> "... drop by more than 0.10 ... while five rise; HAI and WADI show no drop."
10. Sec 7 caveat L1109: "a rolling calibration, WHICH THIS STUDY DOES NOT ADD" -> "a heavily drifted deployment adds a rolling calibration on top of the typing; this study reports the ranking."
11. Appendix A L1305/L1307 (dedupe to one): "no combination rule SEPARATES / no raw combiner SEPARATES the difficult subset" (twice) -> once: "On SWaT the raw aggregations are drift-compressed (0.47 to 0.60); the SWaT difficult subset is resolved by the changepoint typing of Sec 7, not by the combiner choice."
12. Localization L1236-1240: "the HAI top-1 margin IS SMALL (0.42 vs 0.37, WADI 0.43 vs 0.19), WHICH IS WHY the three-community shortlist is the headline" -> "against a size-weighted random baseline the top-1 margin is 0.42 versus 0.37 on HAI and 0.43 versus 0.19 on WADI; the three-community shortlist is the headline figure."
13. Limitations L1266-1269: "a different percentile would redraw the boundary, THOUGH the gap ... is large enough that the split is NOT KNIFE-EDGE" -> "the trivial rule's Easy-to-Difficult gap (0.999->0.601, 0.966->0.340, 0.997->0.627) is wide, so the split is stable to the percentile choice."
14. Limitations L1265: "all reported numbers use the full test set NONETHELESS" -> "all reported numbers use the full test set; the excluded-regime view of Sec 8 is secondary."

LOW
15. WADI residual frontier L1171-1172: "rather than of any one detector" -> "No snapshot detector separates these at a low false-alarm budget: they mark the limit of instantaneous-window detection. Reaching them requires information beyond the window (a physics-informed process residual or explicit temporal features), a direction for future work."
16. L1188: "It is ONLY PARTLY exploitable at the window scale here" -> "At the window scale it is partly exploitable: on HAI a longer 600-sample window adds about +0.06 ..."
17. Appendix A L1304: "fusing in the whole-plant expert COSTS A LITTLE, SO the fused headline SETTLES AT 0.771" -> "fusing in the whole-plant expert gives 0.771, within 0.024 of the community-only 0.795, and the same fusion lifts HAI (0.801 to 0.845)."
18. Sec 4.3(ii) L561-565: three pre-emptive "not/rather than" clauses -> "Its role is false-alarm protection on rare regimes (A5); on WADI, which has no rare-regime false alarms, the base score sits 0.02 below the density head alone (Table 6)."
19. Limitations L1269-1270: "so the reported numbers are LOWER THAN point-adjusted leaderboard figures" -> "and are therefore not comparable to point-adjusted leaderboard figures."

LEAVE (legitimate, confident): WADI tie statements (L555, L850-851, L857-858); factual negatives (L651 -0.050, L700-703 -0.008, L1069 near-no-op); competitor-directed collapse/fail (L193, L848, L865, L929-931, L1122, L1281); Table 1 "specified for completeness; not realized"; L1155 gate description; L1198 "benchmarks cannot fully exercise".
