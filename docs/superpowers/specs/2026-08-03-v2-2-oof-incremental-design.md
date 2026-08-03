# NFF v2.2 OOF Incremental Research Design

## Objective

Replace the same-cross-section in-sample ridge diagnostic with a fixed-symbol-fold out-of-fold (OOF) screen that can support a limited claim of cross-sectional bundle incrementality. Keep the existing corrected factor IC and same-sleeve portfolio proxy, but make VWAP-to-VWAP the primary execution result and label other portfolio combinations by research applicability.

The screen remains a same-day cross-sectional validation. It is stronger than an in-sample fit but is not a temporal walk-forward backtest and must not be described as one.

## Selected Approach

For every sampled decision minute, label, and horizon:

1. Build one common complete-case sample using the label and every feature required by the largest bundle step.
2. Assign every symbol to one of five stable folds using SHA256 of the normalized symbol. Normalize with `str(symbol).strip().upper()`, encode as UTF-8, interpret the first eight digest bytes as an unsigned big-endian integer, and take modulo five. The assignment is independent of process, date, and Python hash randomization.
3. Rank features with `method="average", pct=True` and demean them across the complete decision-time common sample. Feature ranking is an available-time transform and may include all prediction symbols.
4. For each held-out fold, rank the training labels with `method="average", pct=True` using only the four training folds and demean those training-label ranks within that training sample. Learn feature mean, feature standard deviation, ridge coefficients, and intercept from those training folds only, then apply those learned transforms to the held-out symbols.
5. Concatenate the five held-out predictions into a complete OOF prediction vector.
6. Evaluate the OOF prediction against the full-cross-section percentile-ranked and demeaned label. The full target rank is evaluation-only and never enters model fitting.

The same common sample and fixed fold assignment are reused for Traditional, Traditional + Minute-NVG, Traditional + Minute-NVG + Trade-NVG, and the full Hawkes step. This isolates feature-set changes from sample and universe changes.

## Incremental Metrics

Each step emits:

- OOF minute RankIC and the equal-minute-weighted daily mean minute RankIC;
- corrected pooled OOF prediction IC: concatenate fold predictions, percentile-rank and demean prediction and target separately within each minute, then correlate all admitted rows across minutes;
- per-minute OOF R-squared and mean squared error against the full-cross-section percentile-ranked, demeaned target, followed by equal-minute-weighted daily means;
- sample count, sampled minute count, fold count, and common feature count;
- delta mean minute RankIC, delta pooled IC, delta R-squared, and MSE improvement versus the immediately preceding step. IC and R-squared deltas are `new - previous`; MSE improvement is `previous - new`, so positive always means improvement;
- correlation of the incremental OOF prediction (`new_prediction - previous_prediction`) with the ranked target as a supplementary diagnostic.

Per-minute step deltas are calculated before daily averaging. Pooled deltas compare step metrics on the identical admitted row set. The aggregate applies HAC inference directly to the resulting daily OOF metrics and daily deltas; it does not subtract independently aggregated estimates. The report describes these as fixed-symbol-fold cross-sectional OOF evidence, not time-OOS evidence.

## Fold and Failure Contracts

- Fold assignment follows the exact normalized UTF-8 / first-eight-byte SHA256 contract above and is covered by known-vector fixtures.
- A minute is evaluated only when its common sample reaches `min_cross_section_n` and every held-out fold is non-empty.
- A fold fit is rejected when its training sample is below `oof_min_train_n: 40` or ridge cannot be solved.
- A minute contributes metrics only when all five folds produced finite predictions for every bundle step. If one fold or step fails, that minute is discarded from every step.
- Bundle steps never fall back to step-specific complete cases.
- Every step exposes the identical admitted minute index and sample index in test diagnostics; equal counts alone are insufficient acceptance evidence.
- Effective configuration freezes `oof_folds: 5`, `oof_ridge_alpha: 0.001`, and `oof_min_train_n: 40`; all three enter the research contract and run hash.

## Portfolio Scope

VWAP-to-VWAP is the primary execution label. Open-to-open remains diagnostic. Aggregate outputs are split into primary and diagnostic execution tables.

Portfolio variants receive an applicability flag using this contract:

| Horizon | Primary rebalance intervals | Diagnostic intervals |
| --- | --- | --- |
| 15m | 15m | 30m, 60m |
| 30m | 15m, 30m | 60m |
| 60m | 30m, 60m | 15m |
| 120m | 60m | 15m, 30m |

The first Hawkes implementation remains a stock-level eligibility gate. The gated and paired ungated variants first take the same rows with finite signal, label, Hawkes total intensity, and PIT ADV20. Sort by Hawkes intensity descending and normalized symbol ascending, then remove exactly `ceil(0.20 * n)` rows from the gated variant. This makes ties deterministic. Both variants share a `gate_pair_id` and the same pre-gate sample.

Outputs add pre-gate count, eligible count, selected count, kept ratio, pre-gate ADV20 sum, eligible ADV20 sum, selected ADV20 sum, and eligible ADV20 kept ratio. ADV20 sums are explicitly capacity proxies, not executable capacity estimates. Aggregate comparisons report delta gross, turnover, cost, net return, selected count, and ADV20 capacity proxy between paired gated and ungated variants. The mechanism is not called a market-regime gate.

## Versioning and Artifacts

The current Python entrypoint name is retained to avoid a large mechanical file move. Version identity is made explicit through:

- `research_version: "2.2"` and `base_runner_version: "2.1"` in effective configuration and contracts;
- `research_contract_v2_2.json`;
- `final_report_v2_2.md`;
- OOF-specific output columns and contract descriptions.

The run contract includes fold count, fold algorithm, ridge alpha, common-sample policy, portfolio applicability policy, and primary execution label. Existing v2.1 outputs are not reused because the run-contract hash changes.

The formal run must use a new v2.2 run name and output root; attempting to point v2.2 at an existing v2.1 directory remains a hard contract-mismatch error. Primary portfolio artifacts are named `staggered_portfolio_proxy_primary_vwap_overall.*`; open-to-open diagnostic artifacts are named `staggered_portfolio_proxy_diagnostic_open_overall.*`. Both retain grouping by universe, feature, bundle, label family, horizon, portfolio variant, applicability, signal direction, quantile, gate pair/mode, and rebalance interval.

## Testing

A dedicated synthetic test module covers:

- stable symbol fold assignment;
- known-vector stable symbol fold assignment;
- perturbing held-out labels cannot influence their own fitted predictions or any learned preprocessing transform;
- all bundle steps expose exactly the same admitted sample indices, minute indices, and fold assignments;
- one-step failure discards the minute from every bundle step;
- a deterministic high-dimensional noise fixture has in-sample RankIC above `0.90` while five-fold OOF absolute RankIC remains below `0.20`;
- exact hand-calculated fixtures for minute IC, pooled demeaned percentile-rank IC, R-squared, MSE, and all delta signs;
- pooled prediction IC is invariant to changing cross-section sizes when there is no within-minute association;
- contrarian weights reverse the signal;
- q05 selects 5% tails;
- turnover compares only the same sleeve;
- the Hawkes gate removes the top 20% and reports capacity;
- VWAP portfolio results are emitted and marked primary;
- applicability flags and aggregate group separation are preserved.

CI runs this research test module on Windows and Linux in addition to adapter tests and entrypoint compilation.

## Resource Impact

Five-fold OOF increases ridge solves by roughly five times versus one in-sample fit. This change bounds only the incremental OOF overhead: the screen processes one decision-minute block at a time, reuses one common mask and feature-rank matrix, and retains only the four step prediction vectors for the current label/horizon. The worker still retains the existing full-day feature, label, and control frames, so this design does not claim full-worker streaming memory.

Before the formal run, execute the same representative trading date at worker ladders 1, 4, 8, and 12 in isolated output roots. Record elapsed time, throughput, peak worker and system RSS, and output checksums. Select the lowest concurrency within 5% of peak throughput while preserving the configured 78% memory high-water mark and 32 GB available-memory floor. A synthetic subprocess smoke test guards against accidental OOF-array retention, while the real single-date concurrency sweep is the acceptance test for configured concurrency.

## Explicit Limits

- Fixed symbol-fold OOF tests cross-sectional generalization to held-out symbols at the same decision time; it does not test generalization to future dates.
- Overlapping horizons remain statistically dependent; daily aggregation and HAC inference mitigate but do not eliminate this dependence.
- The portfolio remains a same-sleeve proxy, not an aggregate-account execution simulator.
- A later walk-forward model stage is required before claiming temporal OOS model performance or a deployable alpha.
