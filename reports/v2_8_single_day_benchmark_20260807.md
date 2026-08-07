# v2.8 Single-Day Staged Benchmark

Date: `2026-01-02`

Run root: `C:\NFF_research\runs\v2_8_single_day_benchmark_20260807`

This is a pipeline and resource benchmark. The one-day selection and
portfolio results are not research evidence because the candidate freeze uses
one day and `allow_in_sample_portfolio=true` only for this benchmark.

## Completion

All five staged checkpoints completed:

| Stage | Scope | Result | Elapsed |
| --- | ---: | --- | ---: |
| `materialize` | 464 physical factors, 582,728 rows | `_SUCCESS` | 442.8 s |
| `basic_screen` | 464 factors, 57,536 rows | `_SUCCESS` | 1,129.3 s |
| `select` | 460 eligible, 102 detailed, 40 portfolio | `_SUCCESS` | 11.2 s |
| `detailed` | 102 candidates | `_SUCCESS` | 1,553.1 s |
| `portfolio` | 40 candidates, 126,580 rows | `_SUCCESS` | 214.7 s |

Successful stage compute totals 3,351.2 seconds, or about 55.9 minutes.
The wall clock was longer because the first portfolio attempt intentionally
exposed and then reproduced a pre-existing index-order bug before retrying.

## Outputs

- `materialize/support.parquet`, `labels.parquet`, `controls.parquet`,
  `universes.parquet`, `feature_registry.parquet`, and factor inventory
- `basic_screen/basic_factor_screen.parquet`
- `detailed/factor_rank_ic_summary.parquet` with 87,720 rows
- `detailed/decile_curves.parquet` with 61,200 rows
- `portfolio/staggered_portfolio_proxy.parquet` with 126,580 rows
- `pipeline/selection/all_selection_scores.parquet` and `candidates.parquet`

The materialized universe contract contains all PIT masks, including
`all_pit_eligible`, `common_structural`, ADV Top500/1000/2000/3000, and
`final_trading_universe`.

## Observed resource profile

These are direct process observations during the benchmark, not synthetic
estimates:

| Stage | Approx. peak worker RSS | Observation |
| --- | ---: | --- |
| `materialize` | 14.1 GB | factor-block checkpoint reached 464/464 |
| `basic_screen` | 1.3 GB | block streaming; 68/68 blocks |
| `detailed` | 6.8 GB | candidate-only residualization and deciles |
| `portfolio` | 1.6 GB | one weight path expanded across cost scenarios |

The benchmark ran with the old v2.7 campaign stopped so these measurements are
not contaminated by another research run.

## Fixed during benchmark

The first portfolio attempt failed with `KeyError` because
`_turnover_controlled_weights()` emitted a `(datetime, instrument)`
MultiIndex while the feature block used `(instrument, datetime)`. The fix
reorders weights to the input block's index-level contract. A regression test
was added and the v2.8 test file now passes `11/11`.

## Interpretation for multi-day scheduling

- `materialize` is the heavier memory stage but is checkpointed at factor
  block level and is suitable for a small date-worker pool.
- `basic_screen` is the lightest stage and is the best candidate for broad
  date parallelism; its 464-factor work is resumable by 68 blocks per date.
- `detailed` remains the main candidate-stage CPU wall-time bottleneck.
- `portfolio` is relatively light after selection and should be scheduled
  only after the frozen candidate contract exists.
