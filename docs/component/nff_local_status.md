# NFF local warehouse status for Qlib

This note records the local `AnotherNetworkFactory` layout observed on
2026-08-02. It is intentionally machine-specific: use it to run the custom
Qlib adapter against the local NFF warehouse, not as a generic upstream Qlib
data recipe.

## Repository checkouts

- Qlib fork: `D:\DEV\AnotherNetworkFactory\Qlib_GraphLab`
- Qlib branch: `agent/nff-direct-qlib-adapter`
- Qlib HEAD before this note: `619bc26413aad5a3a055222a7f91c7fe2473979f`
- NFF worktree inspected: `C:\Users\A001\.config\superpowers\worktrees\NodeFactorFactory\agent-nff-incremental-research-v2`
- NFF HEAD inspected: `efab45ad92f98c1283e1e56a1202b535d1ee510e`

If the Qlib fork is not present, place it under the project root with:

```powershell
cd D:\DEV\AnotherNetworkFactory
git clone https://github.com/pttangent/Qlib_GraphLab.git
cd D:\DEV\AnotherNetworkFactory\Qlib_GraphLab
git checkout agent/nff-direct-qlib-adapter
```

Use Python 3.11 or 3.12 for this fork. The machine default `python` is 3.14,
which is outside the project metadata's supported range.

## Raw material layout

NFF, not Qlib, consumes the raw ZIPs. Qlib should read the governed NFF
warehouse after NFF has converted raw Bars/Trades into canonical and feature
Parquet.

NFF raw discovery supports both of these month layouts:

```text
D:\DEV\AnotherNetworkFactory\RAW_DATA\1m\YYYYMM\YYYYMMDD.zip
D:\DEV\AnotherNetworkFactory\RAW_DATA\1m\YYYY\YYYYMM\YYYYMMDD.zip
D:\DEV\AnotherNetworkFactory\RAW_DATA\Trades\YYYYMM\YYYYMMDD.zip
D:\DEV\AnotherNetworkFactory\RAW_DATA\Trades\YYYY\YYYYMM\YYYYMMDD.zip
```

Reference inputs currently present:

```text
D:\DEV\AnotherNetworkFactory\RAW_DATA\splits.csv
D:\DEV\AnotherNetworkFactory\RAW_DATA\README\splits.csv
D:\DEV\AnotherNetworkFactory\RAW_DATA\metadata\symbol_metadata.parquet
```

Observed raw ZIP availability on 2026-08-02:

| Raw side | Month | Current files | Note |
|---|---:|---:|---|
| `1m` | `202606` | 21 | Bars are present for June trading days. |
| `1m` | `202607` | 15 | Bars are present through `20260722.zip`. |
| `Trades` | `202606` | 2 | Only `20260612.zip` and `20260616.zip` are present. |
| `Trades` | `202607` | 0 | No July Trade ZIPs are currently staged under `D:\DEV\AnotherNetworkFactory\RAW_DATA`. |

This means raw-driven NFF extension beyond the existing warehouse requires
restaging missing Trade ZIPs before running incremental updates. The current
Qlib adapter can still use already-published NFF warehouse products.

## NFF warehouse products

Main warehouse:

```text
D:\DEV\AnotherNetworkFactory\warehouses\NFF_warehouse
```

Work/cache root:

```text
D:\DEV\AnotherNetworkFactory\warehouses\NFF_work
```

Canonical products:

| Group | Dataset | Schema | Dates | Range | Parquet files |
|---|---|---|---:|---|---:|
| canonical | `bars_1m` | `v1` | 138 | 2026-01-02 to 2026-07-22 | 789 |
| canonical | `trades_1m_core` | `v1` | 138 | 2026-01-02 to 2026-07-22 | 138 |
| canonical | `trades_1m_sketch` | `v1` | 138 | 2026-01-02 to 2026-07-22 | 138 |
| canonical | `trades_condition_1m` | `v1` | 138 | 2026-01-02 to 2026-07-22 | 138 |
| canonical | `trades_venue_1m` | `v1` | 138 | 2026-01-02 to 2026-07-22 | 138 |

Feature family products:

| Family | Active schema | Dates | Range | Parquet files | QA |
|---|---|---:|---|---:|---|
| `minute_nvg` | `v3` | 138 | 2026-01-02 to 2026-07-22 | 17,317 | 138/138 passed |
| `trade_nvg` | `v3` | 138 | 2026-01-02 to 2026-07-22 | 53,766 | 138/138 passed |
| `hawkes_lite` | `v3` | 138 | 2026-01-02 to 2026-07-22 | 53,766 | 138/138 passed |

The empty `schema=v2` directories are historical placeholders. The usable
family contract for Qlib is `schema=v3`. Representative manifests for
`2026-01-30` and `2026-07-22` have `status=complete`, `contract.schema_version=v3`,
`output_record.schema_version=v3`, and `qa_record.passed=true`.

There is one non-blocking compatibility detail: some older `completed_buckets`
entries still contain internal `schema=v2` output path strings. The adapter does
not load those internal paths; it validates the manifest contract and reads the
selected `schema=v3/date=...` partition directly.

## How Qlib uses NFF

The custom adapter lives in:

```text
D:\DEV\AnotherNetworkFactory\Qlib_GraphLab\qlib\contrib\data\nff.py
```

The flow is:

```text
NFF warehouse Parquet
  -> NFFWarehouseCatalog
  -> NFFDataLoader
  -> NFFDataHandlerLP
  -> DatasetH
  -> LightGBM or another Qlib model
  -> Recorder / predictions / RankIC
```

There is no `dump_bin.py` conversion step for NFF. The NFF warehouse remains the
source of truth. Qlib receives a standard `MultiIndex(datetime, instrument)`
frame with `feature` and optional `label` column groups.

Use an end date no later than the warehouse's current last published date unless
the missing Trade ZIPs have been staged and NFF has extended the warehouse:

```python
import qlib
from qlib.constant import REG_US
from qlib.contrib.data.nff import NFFDataHandlerLP
from qlib.data.dataset import DatasetH

WAREHOUSE_ROOT = r"D:\DEV\AnotherNetworkFactory\warehouses\NFF_warehouse"

qlib.init(provider_uri=WAREHOUSE_ROOT, region=REG_US)

handler = NFFDataHandlerLP(
    warehouse_root=WAREHOUSE_ROOT,
    canonical_sets={
        "bars_1m": ["close", "volume", "dollar_volume"],
    },
    feature_sets={
        "minute_nvg": {
            "schema_version": "v3",
            "columns": [
                "price_nvg_30m_top_bottom_asymmetry",
                "price_path_30m_efficiency",
                "price_volume_terminal_overlap_30m",
            ],
        },
        "trade_nvg": {
            "schema_version": "v3",
            "columns": [
                "trade_flow_nvg_300s_top_bottom_asymmetry",
                "trade_flow_path_300s_efficiency",
                "trade_price_flow_terminal_overlap_300s",
            ],
        },
        "hawkes_lite": {
            "schema_version": "v3",
            "columns": [
                "hawkes_ready",
                "hawkes_buy_intensity_300s_mean",
                "hawkes_sell_intensity_300s_mean",
            ],
        },
    },
    execution={"frequency": "1min", "delay_bars": 1},
    label={
        "name": "LABEL0",
        "dataset": "bars_1m",
        "entry_column": "open",
        "exit_column": "open",
        "horizon_bars": 30,
        "future_sessions": 5,
        "same_session": True,
    },
    instruments="all",
    start_time="2026-05-01T13:30:00Z",
    end_time="2026-07-22T20:00:00Z",
    infer_processors=[
        {"class": "ProcessInf"},
        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature", "method": "robust"}},
    ],
    learn_processors=[{"class": "DropnaLabel"}],
    drop_raw=True,
)

dataset = DatasetH(
    handler=handler,
    segments={
        "train": ("2026-05-01", "2026-06-15"),
        "valid": ("2026-06-16", "2026-07-10"),
        "test": ("2026-07-11", "2026-07-22"),
    },
)
```

The executable example remains:

```powershell
cd D:\DEV\AnotherNetworkFactory\Qlib_GraphLab
py -3.11 examples\nff_qlib_pipeline.py inspect --warehouse-root D:\DEV\AnotherNetworkFactory\warehouses\NFF_warehouse
py -3.11 examples\nff_qlib_pipeline.py run --config examples\configs\nff_qlib_example.json
```

## Current assessment

The adapter architecture is in the right shape for research: it prunes
partitions and columns through PyArrow, keeps NFF manifests and `_SUCCESS` as
the completion contract, aligns sources on `symbol_id + timestamp`, and maps
to Qlib `datetime` only after taking the maximum upstream `available_time`.

The local warehouse is strong enough for a first real Qlib smoke/training run
over 2026-05-01 to 2026-07-22, using narrow feature bundles. It is not yet a
complete 2026-07 month acceptance because current local raw Trades for July are
not staged, and the PR still lacks a real-data end-to-end CI job.

Recommended near-term sequence:

1. Use Python 3.11 or 3.12 and install the Qlib fork in editable mode.
2. Run `examples\nff_qlib_pipeline.py inspect` against `NFF_warehouse`.
3. Run a narrow LightGBM training job ending on `2026-07-22`.
4. Stage missing Trade ZIPs from cold storage before extending the warehouse
   beyond the existing published date range.
5. Keep the PR draft until a real warehouse-backed training run records
   predictions, labels, daily RankIC, and model artifacts.
