# NodeFactorFactory → Qlib direct adapter

`qlib.contrib.data.nff` reads a governed NodeFactorFactory (NFF) warehouse directly. It does **not** create a second Qlib binary copy of the warehouse.

## Data path

```text
NFF canonical/features Parquet
  -> schema + date partition discovery
  -> _SUCCESS / manifest validation
  -> Parquet column and date pruning in PyArrow
  -> join on (symbol_id, event timestamp)
  -> max(upstream available_time)
  -> conservative Qlib decision_time
  -> DataHandlerLP / DatasetH / Qlib model
```

The adapter deliberately separates three clocks:

- `timestamp`: the event/bar interval described by the factor;
- `available_time`: when that NFF row is fully known;
- Qlib `datetime`: the first configured decision point after all requested inputs are available.

With the default minute policy, `datetime = ceil(max_available_time, 1min) + 1min`. A factor for the 10:00 bar that is available at 10:01 is therefore assigned to Qlib at 10:02.

## Warehouse layout

The adapter discovers the existing NFF layout:

```text
<warehouse>/canonical/<dataset>/schema=<version>/date=YYYY-MM-DD/*.parquet
<warehouse>/features/<family>/schema=<version>/date=YYYY-MM-DD/*.parquet
<warehouse>/manifests/date=YYYY-MM-DD.json
<warehouse>/manifests/features/<family>/date=YYYY-MM-DD.json
```

It selects exactly one schema per source for an experiment. By default, mixed family versions or implementation hashes across selected dates are rejected rather than silently combined.

## Inspect current products

```python
from qlib.contrib.data.nff import NFFWarehouseCatalog

catalog = NFFWarehouseCatalog(r"D:\DEV\AnotherNetworkFactory\warehouses\NFF_warehouse")
print(catalog.describe())
print(catalog.columns("feature", "minute_nvg", "v3"))
```

The runnable example also provides an inspection command:

```powershell
python examples/nff_qlib_pipeline.py inspect `
  --warehouse-root "D:\DEV\AnotherNetworkFactory\warehouses\NFF_warehouse"
```

## Direct DataLoader use

```python
from qlib.contrib.data.nff import NFFDataLoader

loader = NFFDataLoader(
    warehouse_root=r"D:\DEV\AnotherNetworkFactory\warehouses\NFF_warehouse",
    canonical_sets={
        "bars_1m": ["close", "volume", "dollar_volume"],
        "trades_1m_core": ["trade_count", "signed_notional"],
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
    },
    execution={"frequency": "1min", "delay_bars": 1},
    label={
        "name": "LABEL0",
        "dataset": "bars_1m",
        "entry_column": "open",
        "exit_column": "open",
        "horizon_bars": 30,
    },
)

df = loader.load(
    instruments=["AAPL", "AMD", "NVDA"],
    start_time="2026-05-01T13:30:00Z",
    end_time="2026-07-31T20:00:00Z",
)
print(df)
print(loader.last_load_report)
```

The returned frame has Qlib's standard structure:

```text
MultiIndex rows:    datetime, instrument
MultiIndex columns: feature, label
```

## Standard Qlib pipeline

Qlib's model classes log training metrics through the Recorder. Initialize the framework and run model fitting inside `R.start()`; the provider URI may point to the existing NFF root because this adapter does not ask the native Qlib feature provider to read it.

```python
import qlib
from qlib.constant import REG_US
from qlib.contrib.data.nff import NFFDataHandlerLP
from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel
from qlib.workflow import R

qlib.init(provider_uri=WAREHOUSE_ROOT, region=REG_US)

handler = NFFDataHandlerLP(
    warehouse_root=WAREHOUSE_ROOT,
    canonical_sets=CANONICAL_SETS,
    feature_sets=FEATURE_SETS,
    execution={"frequency": "1min", "delay_bars": 1},
    label={"name": "LABEL0", "horizon_bars": 30},
    instruments="all",
    start_time="2026-05-01T13:30:00Z",
    end_time="2026-07-31T20:00:00Z",
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
        "test": ("2026-07-11", "2026-07-31"),
    },
)
model = LGBModel(
    loss="mse",
    num_leaves=31,
    learning_rate=0.05,
    num_boost_round=300,
)
with R.start(experiment_name="nff_three_month_screening"):
    model.fit(dataset)
    prediction = model.predict(dataset, segment="test")
```

The complete executable version is `examples/nff_qlib_pipeline.py`; it saves predictions, labels, daily RankIC, summary metrics and the trained model.

## Performance properties

- Only selected schemas, dates, symbols and columns are scanned.
- Trade-NVG's full 261-column table is not loaded when an experiment requests three fields.
- NFF float32 schema-v3 values remain float32 in Qlib by default.
- No raw trades are read and no NVG/Hawkes family is recomputed.
- Each source is aligned on the original event key before the decision clock is assigned.
- `drop_raw=True` is the default in `NFFDataHandlerLP` to avoid retaining raw, infer and learn copies simultaneously.
- Cache/materialization is intentionally absent in the first version. Repeated experiments can later add a bounded, content-addressed narrow-view cache without changing the NFF source of truth.

For three-month screening, run feature bundles of roughly 16–32 columns rather than loading every experimental field into one handler. This controls DataHandler/Pandas memory expansion while preserving Qlib's standard experiment semantics.

## Safety rules

- Do not set `delay_bars=0` unless the execution venue and timestamp convention prove the row can be traded at that boundary.
- Do not set `allow_mixed_contracts=True` for formal research unless the differences have been audited.
- Missing values are left as missing; the adapter never turns `not_ready` or `not_observed` into zero.
- Labels begin at Qlib decision time, not at the original NFF event timestamp.
