# v2.7 Resource Snapshot: v2_7_atomic_full_defined_20260807_r45

Snapshot UTC: `2026-08-07T02:19:37.361661+00:00`
Status: `running`; stage: `neutralized_factor_diagnostics`
Progress: `0/138` dates; running `2`; pending `136`; failures `0`.
Contract: `1df2fb50a57b8a396c043d8707137a4623f240cb11b5741a5cb0bc38430dfe31`; git: `9caaa5a60b88fa03bc722be39d39f1f03e2bfc80`

## Resource Summary

| Metric | Min | Median | P90 | Max | Latest |
|---|---:|---:|---:|---:|---:|
| CPU % | 9.300 | 27.600 | 48.620 | 73.700 | 30.500 |
| Memory % | 17.200 | 46.900 | 69.060 | 93.100 | 58.300 |
| Available memory GB | 8.800 | 67.620 | 87.882 | 105.410 | 53.060 |
| Worker RSS total GB | 0.010 | 35.677 | 65.360 | 96.282 | 51.617 |
| Worker RSS max GB | 0.010 | 23.632 | 36.524 | 43.822 | 37.190 |
| Disk free GB | 365.010 | 367.300 | 373.820 | 375.230 | 365.010 |

## Scheduling

Resource samples: `729`; scheduler tuning events: `64`.
Latest tuning: `{"sample_utc": "2026-08-07T02:18:38.863227+00:00", "previous_parallel": 2, "target_parallel": 2, "reason": "within_target_or_limit", "safe_streak": 14, "running_workers": 2, "stage_parallel_cap": 2, "active_worker_stages": {"dates": {"2026-02-04": "ic_and_neutralization", "2026-03-03": "ic_and_neutralization"}, "counts": {"ic_and_neutralization": 2}, "cap_by_stage": {"ic_and_neutralization": 3}, "effective_cap": 2}, "resources": {"cpu_percent": 34.8, "memory_percent": 57.7, "memory_available_gb": 53.92, "disk_free_gb": 365.01, "worker_process_count": 3, "worker_rss_total_gb": 48.216, "worker_rss_max_gb": 36.533}}`

## Factor Progress

Dates with checkpoints: `6`; dates at `464/464`: `4`; max factor progress: `100.000%`.

| Date | Factors | Progress | Last stage | State |
|---|---:|---:|---|---|
| 2026-02-03 | 0/464 | 0.000% | date | running |
| 2026-02-04 | 464/464 | 100.000% | ic_and_neutralization | running |
| 2026-03-03 | 464/464 | 100.000% | ic_and_neutralization | running |
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

Non-empty worker stderr logs: `4`; traceback/exception logs: `0`; warning-only logs: `4`.
This is a live snapshot. It does not assert that the 138-day campaign is complete.
The benchmark stage timing is from the completed one-day run and is included to expose the portfolio_proxy bottleneck.
