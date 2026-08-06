from __future__ import annotations

from pathlib import Path

import pandas as pd

from nff_research.nff_schema_contract import (
    build_schema_audit,
    directional_supplement_fields,
    discover_warehouse_schema,
    minute_windows,
    second_windows,
    validate_directional_supplement,
)


def _write_partition(root: Path, relative: str, date: str, columns: dict[str, list[object]]) -> None:
    path = root / relative / f"date={date}"
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_parquet(path / "part-000.parquet", index=False)


def test_discovers_rebuilt_directional_supplement_fields(tmp_path: Path) -> None:
    date = "2026-01-02"
    fields_15 = directional_supplement_fields(15)
    fields_30 = directional_supplement_fields(30)
    base = {
        "trade_date": [date],
        "symbol_id": [1],
        "symbol": ["AAA"],
        "timestamp": [pd.Timestamp("2026-01-02 14:30:00", tz="UTC")],
        "available_time": [pd.Timestamp("2026-01-02 14:31:00", tz="UTC")],
    }
    supplement = dict(base)
    for field in [*fields_15.values(), *fields_30.values()]:
        supplement[field] = [0.1]
    _write_partition(
        tmp_path,
        "nvg_supplement/minute_nvg_edge_raw/schema=v1",
        date,
        supplement,
    )
    _write_partition(
        tmp_path,
        "features/minute_nvg/schema=v1",
        date,
        {**base, "momentum_15m": [0.01], "price_nvg_15m_top_bottom_asymmetry": [0.2]},
    )
    _write_partition(
        tmp_path,
        "features/trade_nvg/schema=v1",
        date,
        {**base, "trade_flow_nvg_60s_top_bottom_asymmetry": [0.2], "trade_flow_nvg_300s_top_bottom_asymmetry": [0.3]},
    )

    schemas = discover_warehouse_schema(tmp_path, date)
    direction_schema = schemas["nvg_supplement/minute_nvg_edge_raw"]
    assert direction_schema.status == "PRESENT"
    assert set(fields_15.values()).issubset(direction_schema.columns)
    assert set(fields_30.values()).issubset(direction_schema.columns)
    assert minute_windows(direction_schema.columns) == (15, 30)
    assert second_windows(schemas["features/trade_nvg"].columns) == (60, 300)

    direction = validate_directional_supplement(schemas, windows=(15, 30))
    assert not direction.empty
    assert direction["present"].all()
    assert set(direction["semantic_name"]) == set(fields_15)


def test_schema_audit_exposes_missing_requested_fields(tmp_path: Path) -> None:
    date = "2026-01-02"
    base = {
        "trade_date": [date],
        "symbol_id": [1],
        "symbol": ["AAA"],
        "timestamp": [pd.Timestamp("2026-01-02 14:30:00", tz="UTC")],
        "available_time": [pd.Timestamp("2026-01-02 14:31:00", tz="UTC")],
        "price_nvg_15m_terminal_signed_edge_balance": [0.1],
    }
    _write_partition(
        tmp_path,
        "nvg_supplement/minute_nvg_edge_raw/schema=v1",
        date,
        base,
    )
    audit = build_schema_audit(
        tmp_path,
        date,
        requested_fields={
            "nvg_supplement/minute_nvg_edge_raw": {
                "price_nvg_15m_terminal_signed_edge_balance",
                "price_detrended_nvg_15m_terminal_signed_edge_balance",
            }
        },
    )
    requested = audit.loc[audit["requested"].fillna(False)]
    assert "PRESENT_REQUESTED" in set(requested["field_status"])
    assert "REQUESTED_MISSING" in set(requested["field_status"])
    required_direction = audit.loc[
        audit["field"].eq("price_detrended_nvg_15m_terminal_signed_edge_balance")
    ]
    assert not required_direction.empty
    assert not required_direction["present"].all()


def test_global_schema_probe_is_bounded_per_date(tmp_path: Path) -> None:
    base = {
        "trade_date": ["2026-01-02"],
        "symbol_id": [1],
        "symbol": ["AAA"],
        "timestamp": [pd.Timestamp("2026-01-02 14:30:00", tz="UTC")],
        "available_time": [pd.Timestamp("2026-01-02 14:31:00", tz="UTC")],
        "price_nvg_15m_terminal_signed_edge_balance": [0.1],
    }
    root = tmp_path / "nvg_supplement/minute_nvg_edge_raw/schema=v1"
    for date, count in (("2026-01-02", 3), ("2026-01-05", 2)):
        partition = root / f"date={date}"
        partition.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            pd.DataFrame(base).to_parquet(partition / f"part-{index:03d}.parquet", index=False)

    schema = discover_warehouse_schema(tmp_path, None)["nvg_supplement/minute_nvg_edge_raw"]
    assert len(schema.files) == 5
    assert schema.schema_probe_count == 4
    assert schema.schema_scan_policy == "first_last_per_date"
