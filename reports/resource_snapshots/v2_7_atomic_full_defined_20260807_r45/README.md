# v2.7 Resource Snapshot: v2_7_atomic_full_defined_20260807_r45

Snapshot UTC: `2026-08-07T02:51:52.479581+00:00`
Status: `running`; stage: `neutralized_factor_diagnostics`
Progress: `0/138` dates; running `1`; pending `137`; failures `0`.
Contract: `1df2fb50a57b8a396c043d8707137a4623f240cb11b5741a5cb0bc38430dfe31`; git: `9caaa5a60b88fa03bc722be39d39f1f03e2bfc80`

## Resource Summary

| Metric | Min | Median | P90 | Max | Latest |
|---|---:|---:|---:|---:|---:|
| CPU % | 9.300 | 28.050 | 47.290 | 73.700 | 15.200 |
| Memory % | 17.200 | 50.050 | 66.130 | 93.100 | 40.800 |
| Available memory GB | 8.800 | 63.580 | 83.366 | 105.410 | 75.450 |
| Worker RSS total GB | 0.010 | 40.521 | 61.194 | 96.282 | 28.734 |
| Worker RSS max GB | 0.010 | 27.548 | 36.979 | 43.822 | 28.725 |
| Disk free GB | 356.890 | 365.795 | 373.030 | 375.230 | 356.900 |

## Scheduling

Resource samples: `1098`; scheduler tuning events: `97`.
Latest tuning: `{"sample_utc": "2026-08-07T02:51:07.155474+00:00", "previous_parallel": 2, "target_parallel": 1, "reason": "active_worker_drain", "drain_actions": [{"trade_date": "2026-03-09", "pid": 73452, "rss_gb": 36.53, "reason": "memory_guard", "action": "terminate_and_requeue"}], "active_worker_stages": {"dates": {"2026-03-03": "ic_and_neutralization", "2026-03-09": "ic_and_neutralization"}, "counts": {"ic_and_neutralization": 2}, "cap_by_stage": {"ic_and_neutralization": 3}, "effective_cap": 2}, "resources": {"cpu_percent": 29.8, "memory_percent": 69.5, "memory_available_gb": 38.8, "disk_free_gb": 356.89, "worker_process_count": 3, "worker_rss_total_gb": 65.269, "worker_rss_max_gb": 36.531}}`

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
