Construct-valid and leakage-free, with one important caveat. The definition is clean for the question “which detector handles anomalies that defeat both simple marginal and linear cross-channel rules?” because both thresholds are fixed from train-normal and LatAD does not participate in subset construction. There is no model-selection leakage.

The main reviewer objection is conditioning on baseline failure: max|z| and LinRes are guaranteed to look worse on a subset explicitly restricted to examples they fail to flag. Therefore do not present double-hard as an unbiased general-purpose comparison against those two baselines. Present it as a residual-difficulty stress test. A good sentence is:

“The double-hard subset is not intended as an independent benchmark against the two detectors used to define it; rather, it isolates anomaly windows that remain after removing cases separable by either a univariate range rule or a simple linear cross-channel model. Both filtering thresholds are fixed exclusively from train-normal data, and LatAD plays no role in subset construction.”

Claim numerical leadership, not statistically established superiority on all three. I would accept:

“On the double-hard subset, LatAD attains the highest mean AUROC on all three datasets: 0.675 on WADI, 0.812 on HAI, and 0.902 on SWaT. Statistical superiority is established only on HAI, where 26 attack episodes support episode-level resampling. The WADI result is based on five attack episodes and should therefore be interpreted as numerical leadership rather than a significant difference, while the 14 SWaT windows arise from a single attack episode and do not support an episode-level generalization claim.”

For the abstract/conclusion, shorten it to:

“LatAD has the highest mean AUROC on the strict double-hard subset of all three datasets, with statistically supported superiority on HAI; the smaller WADI and single-episode SWaT subsets are reported as numerical, not statistical, leads.”

Avoid “significantly outperforms on all three,” “consistent superiority,” or “wins all three datasets.”

Add it; do not replace the existing Difficult column. The cleanest structure is a small standalone robustness table immediately after Table 3, containing only double-hard AUROC plus # windows and # attack episodes. Keep Easy/Difficult/All unchanged so readers can see that the original evaluation protocol was not retroactively replaced.

This also makes the SWaT result more persuasive: the progression from 0.960 on Difficult to 0.902 on Double-hard shows that the stricter subset genuinely removes much of the ceiling rather than merely producing an easier favorable subset.

Cherry-picking remains the largest perception risk. Neutralize it in four ways:

Call double-hard a secondary robustness/stress-test analysis, not the primary endpoint.

Give the definition mechanistically: it is the logical intersection of failures of the two simplest detector families—marginal and linear cross-channel—not a threshold selected to maximize LatAD.

Report every comparator, counts, and episode counts, including the inconvenient SWaT n
episodes
	​

=1.

Show the complete hierarchy: original Difficult → Double-hard, rather than showing only the favorable final subset.

Do not use “pre-registered” unless it genuinely was preregistered before examining test results. “Predefined using train-normal-calibrated thresholds” is accurate; “pre-registered” would create an avoidable credibility problem.

The result is defensible as: LatAD is numerically best on all three under the strictest baseline-independent residual-difficulty test, with strong statistical evidence on HAI and appropriately limited claims on WADI/SWaT.