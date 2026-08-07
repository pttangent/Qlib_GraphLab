# v2.7 Resource Snapshot: v2_7_atomic_full_defined_20260807_r45

Snapshot UTC: `2026-08-07T01:33:35.123949+00:00`
Status: `running`; stage: `neutralized_factor_diagnostics`
Progress: `0/138` dates; running `1`; pending `137`; failures `0`.
Contract: `1df2fb50a57b8a396c043d8707137a4623f240cb11b5741a5cb0bc38430dfe31`; git: `9caaa5a60b88fa03bc722be39d39f1f03e2bfc80`

## Resource Summary

| Metric | Min | Median | P90 | Max | Latest |
|---|---:|---:|---:|---:|---:|
| CPU % | 12.100 | 41.000 | 54.380 | 71.700 | 22.800 |
| Memory % | 21.000 | 54.800 | 78.600 | 93.100 | 36.500 |
| Available memory GB | 8.800 | 57.580 | 75.638 | 100.560 | 80.820 |
| Worker RSS total GB | 0.010 | 44.276 | 76.808 | 96.282 | 25.256 |
| Worker RSS max GB | 0.010 | 19.578 | 34.094 | 39.394 | 25.247 |
| Disk free GB | 367.670 | 372.700 | 375.220 | 375.230 | 367.670 |

## Scheduling

Resource samples: `229`; scheduler tuning events: `20`.
Latest tuning: `{"sample_utc": "2026-08-07T01:33:33.647172+00:00", "previous_parallel": 1, "target_parallel": 1, "reason": "awaiting_safe_streak", "safe_streak": 1, "running_workers": 1, "resources": {"cpu_percent": 22.8, "memory_percent": 36.5, "memory_available_gb": 80.82, "disk_free_gb": 367.67, "worker_process_count": 2, "worker_rss_total_gb": 25.256, "worker_rss_max_gb": 25.247}}`

## Factor Progress

Dates with checkpoints: `3`; dates at `464/464`: `3`; max factor progress: `100.000%`.

| Date | Factors | Progress | Last stage | State |
|---|---:|---:|---|---|
| 2026-02-04 | 464/464 | 100.000% | ic_and_neutralization | running |
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

Non-empty worker stderr logs: `3`; traceback/exception logs: `0`; warning-only logs: `3`.
This is a live snapshot. It does not assert that the 138-day campaign is complete.
The benchmark stage timing is from the completed one-day run and is included to expose the portfolio_proxy bottleneck.
