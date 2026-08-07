# v2.8 Three-Day Staged Benchmark

Dates: `2026-01-02`, `2026-01-05`, `2026-01-06`

Run root: `C:\NFF_research\runs\v2_8_multi_day_benchmark_20260807_3d_r2`

This is a staged pipeline and scheduling benchmark. The three-day selection
is not a valid walk-forward research result; it exists to verify multi-date
parallelism, checkpoint reuse, and stage contracts.

## Completion

| Stage | Date workers | Result | Per-date elapsed range |
| --- | ---: | --- | ---: |
| `materialize` | 3 | `3/3` `_SUCCESS` | 307.1–399.0 s |
| `basic_screen` | 3 | `3/3` `_SUCCESS`, 68/68 blocks/date | 596.3–639.2 s |
| `select` | 1 | chronological freeze complete | 3 dates |
| `detailed` | 3 | `3/3` `_SUCCESS` | 1,135.3–1,252.6 s |
| `portfolio` | 3 | `3/3` `_SUCCESS` | 150.1–153.1 s |

The r2 selection contract records the dates in chronological order and freezes
`selection_end_date=2026-01-06`.

## Output scale

- All three dates materialized 464 physical factors.
- Basic screen: 57,536 rows per date.
- Detailed IC: 95,460 rows per date, 286,380 total.
- Detailed deciles: 66,600 rows per date, 199,800 total.
- Portfolio proxy: 126,580 rows per date, 379,740 total.
- Selection: 111 detailed candidates and 40 portfolio candidates.

The risk/regime and cost/liquidity tracks remain diagnostic and do not emit
portfolio candidates. Only the Alpha track contributes the 40 portfolio
features.

## Resource observations

| Stage | Observed scheduling/resource behavior |
| --- | --- |
| `materialize` | 3 workers admitted; RSS about 10.9–16.0 GB per worker, available memory about 63 GB at the high-water point; memory cap temporarily reduced admission to 2 without killing active workers. |
| `basic_screen` | 3 workers, about 1.0–1.2 GB RSS each, more than 100 GB available; configured cap was 16 and the stage was CPU-parallel safe. |
| `detailed` | 3 workers, about 3.9–6.7 GB RSS each, no admission pause; configured cap was 4 and available memory stayed above 87 GB. |
| `portfolio` | 3 workers, about 1.6 GB RSS each; all dates completed without retries. |

## Issues caught and fixed

The first three-day attempt exposed two correctness problems before its
detailed outputs were accepted:

1. Catalog date enumeration was not a guaranteed chronological contract. The
   first selection wrote `2026-01-05, 2026-01-06, 2026-01-02`; the run was
   stopped and retained as an audit sample. `_dates()` now sorts explicitly,
   and the r2 selection contract verifies the correct order.
2. Turnover-controlled portfolio weights used `(datetime, instrument)` while
   feature blocks used `(instrument, datetime)`. The weights are now reordered
   to the input block's MultiIndex contract. The regression test reproduces
   the former KeyError and passes after the fix.

The v2.8 pipeline test suite passes `12/12` after both fixes.

## Scheduling conclusion

The staged architecture is supported by the benchmark:

- materialize should start conservatively at 3–4 date workers;
- basic screen can use broad date parallelism and factor-block checkpoints;
- detailed should use 3–4 workers on this 128 GB machine;
- portfolio is light after selection, but must remain downstream of the
  frozen candidate contract.
