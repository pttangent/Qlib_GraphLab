from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qlib.contrib.data.nff import NFFDataHandlerLP, NFFDataLoader, NFFWarehouseCatalog
from qlib.data.dataset.handler import DataHandlerLP


def _publish_partition(
    warehouse: Path,
    *,
    kind: str,
    dataset: str,
    schema: str,
    trade_date: str,
    frame: pd.DataFrame,
    family_version: str = "v1-test",
    implementation_hash: str = "impl-a",
) -> Path:
    group = "canonical" if kind == "canonical" else "features"
    root = warehouse / group / dataset / f"schema={schema}" / f"date={trade_date}"
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), root / "part-00000.parquet")
    (root / "_SUCCESS").write_text("", encoding="utf-8")

    if kind == "feature":
        manifest = warehouse / "manifests" / "features" / dataset / f"date={trade_date}.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "contract": {
                        "schema_version": schema,
                        "family_version": family_version,
                        "implementation_hash": implementation_hash,
                    },
                }
            ),
            encoding="utf-8",
        )
    else:
        manifest = warehouse / "manifests" / f"date={trade_date}.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"status": "complete", "outputs": []}), encoding="utf-8")
    return root


def _bars(symbol: str = "AAPL") -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-05T14:30:00Z", periods=5, freq="1min")
    return pd.DataFrame(
        {
            "trade_date": ["2026-01-05"] * 5,
            "symbol_id": np.array([1] * 5, dtype="int64"),
            "symbol": [symbol] * 5,
            "timestamp": timestamps,
            "available_time": timestamps + pd.Timedelta(minutes=1),
            "open": np.array([10, 11, 12, 13, 14], dtype="float64"),
            "high": np.array([11, 12, 13, 14, 15], dtype="float64"),
            "low": np.array([9, 10, 11, 12, 13], dtype="float64"),
            "close": np.array([10.5, 11.5, 12.5, 13.5, 14.5], dtype="float64"),
            "volume": np.array([100, 110, 120, 130, 140], dtype="float64"),
        }
    )


def test_nff_loader_projects_sources_aligns_pit_and_builds_label(tmp_path: Path):
    warehouse = tmp_path / "warehouse"
    bars = _bars()
    _publish_partition(
        warehouse,
        kind="canonical",
        dataset="bars_1m",
        schema="v1",
        trade_date="2026-01-05",
        frame=bars,
    )
    feature = pd.DataFrame(
        {
            "trade_date": ["2026-01-05"],
            "symbol_id": np.array([1], dtype="int64"),
            "symbol": ["AAPL"],
            "timestamp": pd.to_datetime(["2026-01-05T14:30:00Z"]),
            "available_time": pd.to_datetime(["2026-01-05T14:31:00Z"]),
            "shape": np.array([1.5], dtype="float32"),
            "unrequested_large_column": np.array([999.0], dtype="float32"),
        }
    )
    _publish_partition(
        warehouse,
        kind="feature",
        dataset="minute_nvg",
        schema="v3",
        trade_date="2026-01-05",
        frame=feature,
        family_version="v3-exact-test",
    )

    loader = NFFDataLoader(
        warehouse_root=warehouse,
        feature_sets={"minute_nvg": ["shape"]},
        canonical_sets={"bars_1m": ["close"]},
        execution={"frequency": "1min", "delay_bars": 1},
        label={
            "name": "LABEL0",
            "dataset": "bars_1m",
            "schema_version": "v1",
            "entry_column": "open",
            "exit_column": "open",
            "horizon_bars": 2,
            "future_sessions": 1,
        },
        arrow_use_threads=False,
    )
    result = loader.load(
        instruments=["AAPL"],
        start_time="2026-01-05T14:32:00Z",
        end_time="2026-01-05T14:32:00Z",
    )

    assert list(result.index.names) == ["datetime", "instrument"]
    assert list(result.columns.get_level_values(0).unique()) == ["feature", "label"]
    assert ("feature", "minute_nvg__shape") in result.columns
    assert ("feature", "bars_1m__close") in result.columns
    assert ("feature", "unrequested_large_column") not in result.columns
    assert result.index[0] == (pd.Timestamp("2026-01-05T14:32:00"), "AAPL")
    assert result.iloc[0][("feature", "minute_nvg__shape")] == pytest.approx(1.5)
    assert result.iloc[0][("feature", "bars_1m__close")] == pytest.approx(10.5)
    assert result.iloc[0][("label", "LABEL0")] == pytest.approx(14.0 / 12.0 - 1.0)
    assert result.dtypes.eq(np.dtype("float32")).all()
    assert loader.last_load_report["execution"]["delay_bars"] == 1
    assert loader.last_load_report["label"]["non_null_labels"] == 1
    assert loader.last_load_report["label"]["same_session"] is True
    assert loader.last_load_report["execution"]["qlib_datetime"] == "UTC-naive"


def test_nff_loader_rejects_mixed_family_contracts(tmp_path: Path):
    warehouse = tmp_path / "warehouse"
    for trade_date, version, value in [
        ("2026-01-05", "v3-a", 1.0),
        ("2026-01-06", "v3-b", 2.0),
    ]:
        timestamp = pd.Timestamp(f"{trade_date}T14:30:00Z")
        frame = pd.DataFrame(
            {
                "trade_date": [trade_date],
                "symbol_id": np.array([1], dtype="int64"),
                "symbol": ["AAPL"],
                "timestamp": [timestamp],
                "available_time": [timestamp + pd.Timedelta(minutes=1)],
                "shape": np.array([value], dtype="float32"),
            }
        )
        _publish_partition(
            warehouse,
            kind="feature",
            dataset="minute_nvg",
            schema="v3",
            trade_date=trade_date,
            frame=frame,
            family_version=version,
            implementation_hash=f"impl-{version}",
        )

    loader = NFFDataLoader(
        warehouse_root=warehouse,
        feature_sets={"minute_nvg": ["shape"]},
        execution={"frequency": "1min", "delay_bars": 1},
        arrow_use_threads=False,
    )
    with pytest.raises(ValueError, match="Mixed NFF contracts"):
        loader.load(
            start_time="2026-01-05T14:30:00Z",
            end_time="2026-01-06T15:00:00Z",
        )


def test_catalog_discovers_latest_schema_and_columns(tmp_path: Path):
    warehouse = tmp_path / "warehouse"
    _publish_partition(
        warehouse,
        kind="canonical",
        dataset="bars_1m",
        schema="v1",
        trade_date="2026-01-05",
        frame=_bars(),
    )
    _publish_partition(
        warehouse,
        kind="canonical",
        dataset="bars_1m",
        schema="v3",
        trade_date="2026-01-05",
        frame=_bars(),
    )
    catalog = NFFWarehouseCatalog(warehouse)
    assert catalog.available_schemas("canonical", "bars_1m") == ["v1", "v3"]
    schema, _ = catalog.resolve_schema("canonical", "bars_1m")
    assert schema == "v3"
    assert "available_time" in catalog.columns("canonical", "bars_1m", "v3")


def test_nff_datahandler_lp_enters_standard_qlib_fetch(tmp_path: Path):
    warehouse = tmp_path / "warehouse"
    _publish_partition(
        warehouse,
        kind="canonical",
        dataset="bars_1m",
        schema="v1",
        trade_date="2026-01-05",
        frame=_bars(),
    )
    feature = pd.DataFrame(
        {
            "trade_date": ["2026-01-05"],
            "symbol_id": np.array([1], dtype="int64"),
            "symbol": ["AAPL"],
            "timestamp": pd.to_datetime(["2026-01-05T14:30:00Z"]),
            "available_time": pd.to_datetime(["2026-01-05T14:31:00Z"]),
            "shape": np.array([1.5], dtype="float32"),
        }
    )
    _publish_partition(
        warehouse,
        kind="feature",
        dataset="minute_nvg",
        schema="v3",
        trade_date="2026-01-05",
        frame=feature,
    )
    handler = NFFDataHandlerLP(
        warehouse_root=warehouse,
        feature_sets={"minute_nvg": ["shape"]},
        label={"name": "LABEL0", "horizon_bars": 1, "future_sessions": 1},
        instruments=["AAPL"],
        start_time="2026-01-05T14:32:00Z",
        end_time="2026-01-05T14:32:00Z",
        learn_processors=[{"class": "DropnaLabel"}],
        loader_kwargs={"arrow_use_threads": False},
    )
    fetched = handler.fetch(
        selector=slice("2026-01-05T14:32:00", "2026-01-05T14:32:00"),
        level="datetime",
        col_set=["feature", "label"],
        data_key=DataHandlerLP.DK_L,
    )
    assert len(fetched) == 1
    assert ("feature", "minute_nvg__shape") in fetched.columns
    assert ("label", "LABEL0") in fetched.columns
