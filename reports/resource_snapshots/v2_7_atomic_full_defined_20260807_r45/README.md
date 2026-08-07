# v2.7 Resource Snapshot: v2_7_atomic_full_defined_20260807_r45

Snapshot UTC: `2026-08-07T02:00:27.593167+00:00`
Status: `running`; stage: `neutralized_factor_diagnostics`
Progress: `0/138` dates; running `1`; pending `137`; failures `0`.
Contract: `1df2fb50a57b8a396c043d8707137a4623f240cb11b5741a5cb0bc38430dfe31`; git: `9caaa5a60b88fa03bc722be39d39f1f03e2bfc80`

## Resource Summary

| Metric | Min | Median | P90 | Max | Latest |
|---|---:|---:|---:|---:|---:|
| CPU % | 9.300 | 29.500 | 50.220 | 73.700 | 19.600 |
| Memory % | 17.200 | 47.600 | 71.380 | 93.100 | 43.500 |
| Available memory GB | 8.800 | 66.720 | 87.696 | 105.410 | 71.900 |
| Worker RSS total GB | 0.010 | 36.000 | 68.314 | 96.282 | 32.948 |
| Worker RSS max GB | 0.010 | 22.081 | 35.303 | 39.394 | 32.939 |
| Disk free GB | 367.010 | 367.330 | 375.010 | 375.230 | 367.300 |

## Scheduling

Resource samples: `509`; scheduler tuning events: `45`.
Latest tuning: `{"sample_utc": "2026-08-07T01:59:47.011534+00:00", "previous_parallel": 2, "target_parallel": 1, "reason": "active_worker_drain", "drain_actions": [{"trade_date": "2026-06-01", "pid": 16004, "rss_gb": 35.633, "reason": "memory_guard", "action": "terminate_and_requeue"}], "active_worker_stages": {"dates": {"2026-06-09": "ic_and_neutralization", "2026-06-01": "ic_and_neutralization"}, "counts": {"ic_and_neutralization": 2}, "cap_by_stage": {"ic_and_neutralization": 3}, "effective_cap": 2}, "resources": {"cpu_percent": 30.3, "memory_percent": 71.8, "memory_available_gb": 35.94, "disk_free_gb": 367.3, "worker_process_count": 3, "worker_rss_total_gb": 68.794, "worker_rss_max_gb": 35.643}}`

## Factor Progress

Dates with checkpoints: `6`; dates at `464/464`: `3`; max factor progress: `100.000%`.

| Date | Factors | Progress | Last stage | State |
|---|---:|---:|---|---|
| 2026-02-03 | 0/464 | 0.000% | date | running |
| 2026-02-04 | 464/464 | 100.000% | ic_and_neutralization | running |
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

Non-empty worker stderr logs: `3`; traceback/exception logs: `0`; warning-only logs: `3`.
This is a live snapshot. It does not assert that the 138-day campaign is complete.
The benchmark stage timing is from the completed one-day run and is included to expose the portfolio_proxy bottleneck.
