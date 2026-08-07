# NFF v2.8 staged-selection runbook

Branch: `agent/nff-research-pipelined-selection-v2-8`

Config: `configs/v2_8_pipelined_full_campaign.yaml`

Run ID: `v2_8_pipelined_selection_20260807_r2`

The v2.8 run writes to `C:/NFF_research/runs/<run_id>` and does not modify the
existing v2.7 r45 output directory.

## Recommended first run

First produce the exact chronological 60-day training window and freeze the
candidate manifest:

```powershell
python nff_research/v2_8_launch.py `
  --config configs/v2_8_pipelined_full_campaign.yaml `
  --stage bootstrap
```

This executes:

```text
first 60 dates materialize
-> first 60 dates basic_screen
-> multi-track candidate selection
```

Review:

```text
C:/NFF_research/runs/v2_8_pipelined_selection_20260807_r2/
  pipeline/selection/meta.json
  pipeline/selection/all_selection_scores.parquet
  pipeline/selection/candidates.parquet
```

The candidate manifest contains:

- up to 80 return-alpha factors for detailed diagnostics;
- up to 30 risk/regime factors;
- up to 10 cost/liquidity factors;
- up to 40 alpha factors for portfolio proxy.

Risk and cost factor signs describe association with their own target and are
never interpreted as long/short direction. Portfolio factor signs are frozen
from the training-period return IC.

## Continue the complete campaign

```powershell
python nff_research/v2_8_launch.py `
  --config configs/v2_8_pipelined_full_campaign.yaml `
  --stage all
```

Already valid bootstrap outputs receive a cheap contract check and are skipped.
The scheduler then processes the remaining dates, candidate-only detailed
analysis, and post-freeze portfolio proxy.

Individual stages can also be resumed:

```powershell
python nff_research/v2_8_launch.py --config configs/v2_8_pipelined_full_campaign.yaml --stage materialize
python nff_research/v2_8_launch.py --config configs/v2_8_pipelined_full_campaign.yaml --stage basic_screen
python nff_research/v2_8_launch.py --config configs/v2_8_pipelined_full_campaign.yaml --stage select
python nff_research/v2_8_launch.py --config configs/v2_8_pipelined_full_campaign.yaml --stage detailed
python nff_research/v2_8_launch.py --config configs/v2_8_pipelined_full_campaign.yaml --stage portfolio
```

## Stage outputs

```text
pipeline/dates/date=YYYY-MM-DD/materialize/
  support.parquet
  labels.parquet
  label_masks.parquet
  controls.parquet
  universes.parquet
  feature_registry.parquet
  factor_block_inventory.parquet

pipeline/dates/date=YYYY-MM-DD/basic_screen/
  blocks/*.parquet
  blocks/*.json
  basic_factor_screen.parquet

pipeline/dates/date=YYYY-MM-DD/detailed/
  factor_rank_ic_summary.parquet
  decile_curves.parquet

pipeline/dates/date=YYYY-MM-DD/portfolio/
  staggered_portfolio_proxy.parquet
```

Each stage/date has `_SUCCESS` and `meta.json`. The scheduler still launches a
lightweight contract check for existing outputs, so rebuilt supplements, changed
candidate manifests or stage-specific semantic config invalidate only the
necessary downstream stage.

## Initial concurrency on the 128 GB / 16-core machine

| Stage | Max date processes | Intra-date threads | Admission estimate |
|---|---:|---:|---:|
| materialize | 4 | 4 | 12 GB/process |
| basic_screen | 16 | mostly process-level | 5 GB/process |
| detailed | 4 | 4 | 18 GB/process |
| portfolio | 4 | 4 | 10 GB/process |

A 28 GB memory reserve is maintained. New process admission pauses when the
reserve or per-worker estimate cannot be satisfied; currently running workers
are allowed to checkpoint or finish.

These values are starting limits, not measured guarantees. Recalibrate them
after one completed date per stage using actual RSS and stage timing.

## Performance changes relative to v2.7

- all 464 factors still receive a raw multi-label screen;
- raw screen is checkpointed by factor block;
- factor ranks are reused for labels with the same valid cross-section mask;
- neutralization and deciles load no more than 120 selected factors;
- portfolio loads no more than 40 alpha factors;
- all cost scenarios reuse one ranking/gate/weight/turnover path;
- the wide materialization frame is released before downstream analysis.

## Validation still required locally

Before marking the branch production-ready, run:

1. one complete date for every stage;
2. forced interruption during a raw-screen block and resume;
3. forced interruption during detailed/portfolio and resume;
4. RSS comparison against v2.7 r45;
5. mathematical comparison of overlapping factor/IC/portfolio rows.
