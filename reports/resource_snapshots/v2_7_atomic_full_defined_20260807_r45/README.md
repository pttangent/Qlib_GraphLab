# v2.7 Resource Snapshot: v2_7_atomic_full_defined_20260807_r45

Snapshot UTC: `2026-08-07T02:50:07.196147+00:00`
Status: `running`; stage: `neutralized_factor_diagnostics`
Progress: `0/138` dates; running `2`; pending `136`; failures `0`.
Contract: `1df2fb50a57b8a396c043d8707137a4623f240cb11b5741a5cb0bc38430dfe31`; git: `9caaa5a60b88fa03bc722be39d39f1f03e2bfc80`

## Resource Summary

| Metric | Min | Median | P90 | Max | Latest |
|---|---:|---:|---:|---:|---:|
| CPU % | 9.300 | 28.200 | 47.530 | 73.700 | 27.600 |
| Memory % | 17.200 | 50.000 | 65.600 | 93.100 | 65.500 |
| Available memory GB | 8.800 | 63.660 | 83.914 | 105.410 | 43.920 |
| Worker RSS total GB | 0.010 | 40.382 | 60.722 | 96.282 | 60.737 |
| Worker RSS max GB | 0.010 | 27.361 | 36.982 | 43.822 | 33.128 |
| Disk free GB | 357.220 | 366.100 | 373.080 | 375.230 | 357.220 |

## Scheduling

Resource samples: `1078`; scheduler tuning events: `95`.
Latest tuning: `{"sample_utc": "2026-08-07T02:50:04.378974+00:00", "previous_parallel": 2, "target_parallel": 2, "reason": "within_target_or_limit", "safe_streak": 0, "running_workers": 2, "stage_parallel_cap": 2, "active_worker_stages": {"dates": {"2026-03-03": "ic_and_neutralization", "2026-03-09": "ic_and_neutralization"}, "counts": {"ic_and_neutralization": 2}, "cap_by_stage": {"ic_and_neutralization": 3}, "effective_cap": 2}, "resources": {"cpu_percent": 27.6, "memory_percent": 65.5, "memory_available_gb": 43.92, "disk_free_gb": 357.22, "worker_process_count": 3, "worker_rss_total_gb": 60.737, "worker_rss_max_gb": 33.128}}`

## Factor Progress

Dates with checkpoints: `6`; dates at `464/464`: `5`; max factor progress: `100.000%`.

| Date | Factors | Progress | Last stage | State |
|---|---:|---:|---|---|
| 2026-02-03 | 0/464 | 0.000% | date | running |
| 2026-02-04 | 464/464 | 100.000% | ic_and_neutralization | running |
| 2026-03-03 | 464/464 | 100.000% | ic_and_neutralization | running |
| 2026-03-09 | 464/464 | 100.000% | ic_and_neutralization | running |
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

Non-empty worker stderr logs: `5`; traceback/exception logs: `0`; warning-only logs: `5`.
This is a live snapshot. It does not assert that the 138-day campaign is complete.
The benchmark stage timing is from the completed one-day run and is included to expose the portfolio_proxy bottleneck.
