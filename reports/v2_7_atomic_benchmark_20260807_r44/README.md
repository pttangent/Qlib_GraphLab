# v2.7 Atomic Single-Day Benchmark

This directory contains the complete single-day IC, decile, and portfolio-proxy
artifacts for `2026-01-02`. The source run completed with `_SUCCESS` and the
Parquet outputs are copied here for review.

## Scope

- 646 loaded feature fields; 464 physical analysis features evaluated.
- 7 label families: open/open, VWAP/VWAP, close/close, liquidity deterioration,
  realized volatility, jump-tail event, and execution-cost proxy.
- 5 horizons: 1, 5, 15, 30, and 60 bars.
- 8 IC universes and the configured decile/portfolio universes.
- America/New_York regular session, `09:30 <= local time < 16:00`.
- Run contract hash: `5fdc553f83d2e7b58d579fe165a865a1029c43431c1c4f53b238373a0c439630`.

## Output Sizes

| Artifact | Rows | Purpose |
|---|---:|---|
| `factor_rank_ic_summary.parquet` | 399,040 | Daily raw/residualized minute-mean and corrected pooled IC observations |
| `decile_curves.parquet` | 278,400 | Decile curves and diagnostic portfolio spreads |
| `staggered_portfolio_proxy.parquet` | 350,404 | Same-sleeve portfolio proxy with turnover, cost, and net return fields |
| `feature_registry.parquet` | 646 | Evaluated/excluded feature registry |
| `label_dependency_diagnostics.parquet` | 10 | Per-minute label-dependency diagnostics |

The full output columns are preserved in Parquet. CSV exports remain in the
local source run but are intentionally omitted from Git to keep the review
branch compact.

## Stage Timing

The single-day wall-clock bottleneck was `portfolio_proxy` at approximately
`3,593.0 s`, followed by `ic_and_neutralization` at `2,353.8 s`. Factor
derivation took `155.9 s`; venue, supplement, condition, and sketch merges
together took about `154.1 s`.

## Interpretation Boundary

This is a complete single-day validity and portfolio-proxy report, not evidence
of out-of-sample performance. `incremental_model_rows` is `0` for one date, so
no temporal walk-forward or OOS conclusion can be drawn from this benchmark.
The portfolio artifact is same-sleeve accounting with turnover controls; it is
not yet a full cash/order/fill/borrow/impact account backtest.

The active 138-date run is separate and remains under the durable scheduler.
