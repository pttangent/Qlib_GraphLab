# v2.7 Resource Snapshot: v2_7_atomic_full_defined_20260807_r45

Snapshot UTC: `2026-08-07T02:05:07.680830+00:00`
Status: `running`; stage: `neutralized_factor_diagnostics`
Progress: `0/138` dates; running `1`; pending `137`; failures `0`.
Contract: `1df2fb50a57b8a396c043d8707137a4623f240cb11b5741a5cb0bc38430dfe31`; git: `9caaa5a60b88fa03bc722be39d39f1f03e2bfc80`

## Resource Summary

| Metric | Min | Median | P90 | Max | Latest |
|---|---:|---:|---:|---:|---:|
| CPU % | 9.300 | 28.400 | 49.700 | 73.700 | 30.700 |
| Memory % | 17.200 | 47.600 | 71.080 | 93.100 | 49.300 |
| Available memory GB | 8.800 | 66.720 | 87.678 | 105.410 | 64.510 |
| Worker RSS total GB | 0.010 | 36.618 | 67.990 | 96.282 | 39.384 |
| Worker RSS max GB | 0.010 | 23.834 | 36.527 | 43.822 | 39.374 |
| Disk free GB | 365.770 | 367.320 | 374.850 | 375.230 | 365.770 |

## Scheduling

Resource samples: `563`; scheduler tuning events: `51`.
Latest tuning: `{"sample_utc": "2026-08-07T02:05:01.458616+00:00", "previous_parallel": 1, "target_parallel": 1, "reason": "awaiting_safe_streak", "safe_streak": 1, "running_workers": 1, "stage_parallel_cap": 2, "active_worker_stages": {"dates": {"2026-02-04": "date"}, "counts": {"date": 1}, "cap_by_stage": {"date": 3}, "effective_cap": 2}, "resources": {"cpu_percent": 18.3, "memory_percent": 44.1, "memory_available_gb": 71.17, "disk_free_gb": 365.78, "worker_process_count": 2, "worker_rss_total_gb": 32.194, "worker_rss_max_gb": 32.185}}`

## Factor Progress

Dates with checkpoints: `6`; dates at `464/464`: `3`; max factor progress: `100.000%`.

| Date | Factors | Progress | Last stage | State |
|---|---:|---:|---|---|
| 2026-02-03 | 0/464 | 0.000% | date | running |
| 2026-02-04 | 464/464 | 100.000% | date | running |
| 2026-03-03 | 0/464 | 0.000% | date | running |
| 2026-03-09 | 0/464 | 0.000% | date | running |
| 2026-06-01 | 464/464 | 100.000% | ic_and_neutralization | running |
| 2026-06-09 | 464/464 | 100.000% | ic_and_neutralization | running |

## Benchmark Stage Timing

| Stage | Count | Total seconds | Median seconds | P90 seconds | Max seconds |
|---|---:|---:|---:|---:|---:|
| derive_factors | 1 | 155.906 | 155.906 | 155.906 | 155.906 |
| ic_and_neutralization | 1 | 2353.814 | 2353.814 | 2353.814 | 2353.814 |
| merge_condition | 1 | 22.640 | 22.640 | 22.640 | 22.640 |
| merge_sketch | 1 | 17.208 | 17.208 | 17.208 | 17.208 |
| merge_supplements | 1 | 63.140 | 63.140 | 63.140 | 63.140 |
| merge_venue | 1 | 51.100 | 51.100 | 51.100 | 51.100 |
| portfolio_proxy | 1 | 3593.008 | 3593.008 | 3593.008 | 3593.008 |

## Audit Notes

Non-empty worker stderr logs: `2`; traceback/exception logs: `0`; warning-only logs: `2`.
This is a live snapshot. It does not assert that the 138-day campaign is complete.
The benchmark stage timing is from the completed one-day run and is included to expose the portfolio_proxy bottleneck.
