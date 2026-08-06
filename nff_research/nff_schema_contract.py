from __future__ import annotations

"""Runtime schema contract for NodeFactorFactory and its NVG supplement.

The local NFF warehouse is authoritative. This module never invents a field
from a similarly named column: it reads published Parquet schemas, records what
is present for the exact date, and exposes the rebuilt directional supplement
fields explicitly.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import pandas as pd
import pyarrow.parquet as pq


NFF_SOURCE_REPOSITORY = "pttangent/NodeFactorFactory"
NFF_SOURCE_BRANCH = "agent/nvg-trade-hawkes-families"

BASE_DATASETS = {
    "features/minute_nvg": "features/minute_nvg/schema=v1",
    "features/trade_nvg": "features/trade_nvg/schema=v1",
    "features/hawkes_lite": "features/hawkes_lite/schema=v1",
    "canonical/bars_1m": "canonical/bars_1m/schema=v1",
    "canonical/trades_1m_core": "canonical/trades_1m_core/schema=v1",
    "canonical/trades_1m_sketch": "canonical/trades_1m_sketch/schema=v1",
    "canonical/trades_venue_1m": "canonical/trades_venue_1m/schema=v1",
}

SUPPLEMENT_DATASETS = {
    "nvg_supplement/minute_hvg_risk_raw": "nvg_supplement/minute_hvg_risk_raw/schema=v1",
    "nvg_supplement/minute_nvg_edge_raw": "nvg_supplement/minute_nvg_edge_raw/schema=v1",
    "nvg_supplement/minute_visibility_topology_raw": "nvg_supplement/minute_visibility_topology_raw/schema=v1",
    "nvg_supplement/trade_visibility_edge_raw": "nvg_supplement/trade_visibility_edge_raw/schema=v1",
    "nvg_supplement/trade_visibility_topology_raw": "nvg_supplement/trade_visibility_topology_raw/schema=v1",
}

KEY_COLUMNS = {
    "trade_date",
    "symbol_id",
    "symbol",
    "timestamp",
    "available_time",
    "date",
    "schema",
}

_MINUTE_WINDOW_RE = re.compile(r"_(\d+)m(?:_|$)")
_SECOND_WINDOW_RE = re.compile(r"_(\d+)s(?:_|$)")


@dataclass(frozen=True)
class DatasetSchema:
    namespace: str
    relative_root: str
    trade_date: str | None
    files: tuple[str, ...]
    columns: tuple[str, ...]
    schema_sha256: str | None
    status: str


def _schema_hash(columns: Sequence[str]) -> str:
    payload = "\n".join(columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _partition_files(root: Path, trade_date: str | None) -> list[Path]:
    if trade_date:
        return sorted((root / f"date={trade_date}").glob("*.parquet"))
    return sorted(root.glob("date=*/*.parquet"))


def discover_dataset_schema(
    warehouse_root: Path,
    namespace: str,
    relative_root: str,
    trade_date: str | None = None,
) -> DatasetSchema:
    root = warehouse_root / relative_root
    files = _partition_files(root, trade_date)
    if not files:
        return DatasetSchema(
            namespace=namespace,
            relative_root=relative_root,
            trade_date=trade_date,
            files=(),
            columns=(),
            schema_sha256=None,
            status="MISSING_PARTITION",
        )
    schemas = [tuple(pq.read_schema(path).names) for path in files]
    first = schemas[0]
    if any(schema != first for schema in schemas[1:]):
        union = tuple(sorted({column for schema in schemas for column in schema}))
        return DatasetSchema(
            namespace=namespace,
            relative_root=relative_root,
            trade_date=trade_date,
            files=tuple(str(path) for path in files),
            columns=union,
            schema_sha256=_schema_hash(union),
            status="MIXED_PARQUET_SCHEMAS",
        )
    return DatasetSchema(
        namespace=namespace,
        relative_root=relative_root,
        trade_date=trade_date,
        files=tuple(str(path) for path in files),
        columns=first,
        schema_sha256=_schema_hash(first),
        status="PRESENT",
    )


def discover_warehouse_schema(
    warehouse_root: Path | str,
    trade_date: str | None = None,
) -> dict[str, DatasetSchema]:
    root = Path(warehouse_root)
    return {
        namespace: discover_dataset_schema(root, namespace, relative, trade_date)
        for namespace, relative in {**BASE_DATASETS, **SUPPLEMENT_DATASETS}.items()
    }


def minute_windows(columns: Iterable[str]) -> tuple[int, ...]:
    values = {
        int(match.group(1))
        for column in columns
        for match in [_MINUTE_WINDOW_RE.search(column)]
        if match
    }
    return tuple(sorted(values))


def second_windows(columns: Iterable[str]) -> tuple[int, ...]:
    values = {
        int(match.group(1))
        for column in columns
        for match in [_SECOND_WINDOW_RE.search(column)]
        if match
    }
    return tuple(sorted(values))


def directional_supplement_fields(window: int) -> dict[str, str]:
    """Exact direction fields emitted by the rebuilt NVG edge supplement."""
    suffix = f"{int(window)}m"
    return {
        "price_edge_balance": f"price_nvg_{suffix}_terminal_signed_edge_balance",
        "price_long_edge_slope": f"price_nvg_{suffix}_terminal_long_edge_signed_slope",
        "detrended_edge_balance": f"price_detrended_nvg_{suffix}_terminal_signed_edge_balance",
        "detrended_long_edge_slope": f"price_detrended_nvg_{suffix}_terminal_long_edge_signed_slope",
        "volume_edge_balance": f"volume_nvg_{suffix}_terminal_signed_edge_balance",
        "volume_long_edge_slope": f"volume_nvg_{suffix}_terminal_long_edge_signed_slope",
        "price_volume_jaccard": f"price_volume_nvg_{suffix}_edge_weighted_jaccard",
        "price_volume_slope_corr": f"price_volume_nvg_{suffix}_common_edge_slope_corr",
    }


def validate_directional_supplement(
    schemas: Mapping[str, DatasetSchema],
    windows: Sequence[int] = (15, 30),
) -> pd.DataFrame:
    schema = schemas.get("nvg_supplement/minute_nvg_edge_raw")
    available = set(schema.columns if schema else ())
    rows: list[dict[str, object]] = []
    for window in windows:
        for semantic_name, field in directional_supplement_fields(window).items():
            rows.append(
                {
                    "dataset": "nvg_supplement/minute_nvg_edge_raw",
                    "window": f"{window}m",
                    "semantic_name": semantic_name,
                    "field": field,
                    "present": field in available,
                    "status": "PRESENT" if field in available else "MISSING",
                    "schema_sha256": None if schema is None else schema.schema_sha256,
                }
            )
    return pd.DataFrame(rows)


def build_schema_audit(
    warehouse_root: Path | str,
    trade_date: str | None = None,
    requested_fields: Mapping[str, Iterable[str]] | None = None,
) -> pd.DataFrame:
    schemas = discover_warehouse_schema(warehouse_root, trade_date)
    rows: list[dict[str, object]] = []
    requested = {key: set(values) for key, values in (requested_fields or {}).items()}
    for namespace, schema in schemas.items():
        wanted = requested.get(namespace, set())
        actual = set(schema.columns)
        for field in sorted(actual | wanted):
            rows.append(
                {
                    "namespace": namespace,
                    "relative_root": schema.relative_root,
                    "trade_date": trade_date,
                    "dataset_status": schema.status,
                    "schema_sha256": schema.schema_sha256,
                    "file_count": len(schema.files),
                    "field": field,
                    "is_key": field in KEY_COLUMNS,
                    "present": field in actual,
                    "requested": field in wanted,
                    "field_status": (
                        "PRESENT_REQUESTED"
                        if field in actual and field in wanted
                        else "PRESENT_SOURCE"
                        if field in actual
                        else "REQUESTED_MISSING"
                    ),
                    "nff_repository": NFF_SOURCE_REPOSITORY,
                    "nff_branch": NFF_SOURCE_BRANCH,
                }
            )
    directional = validate_directional_supplement(schemas)
    if directional.empty:
        return pd.DataFrame(rows)
    directional = directional.assign(
        namespace=directional["dataset"],
        relative_root=SUPPLEMENT_DATASETS["nvg_supplement/minute_nvg_edge_raw"],
        trade_date=trade_date,
        dataset_status=schemas["nvg_supplement/minute_nvg_edge_raw"].status,
        file_count=len(schemas["nvg_supplement/minute_nvg_edge_raw"].files),
        is_key=False,
        requested=True,
        field_status=directional["status"].map(
            {"PRESENT": "PRESENT_REQUIRED_DIRECTION", "MISSING": "REQUIRED_DIRECTION_MISSING"}
        ),
        nff_repository=NFF_SOURCE_REPOSITORY,
        nff_branch=NFF_SOURCE_BRANCH,
    )
    directional = directional[
        [
            "namespace",
            "relative_root",
            "trade_date",
            "dataset_status",
            "schema_sha256",
            "file_count",
            "field",
            "is_key",
            "present",
            "requested",
            "field_status",
            "nff_repository",
            "nff_branch",
            "window",
            "semantic_name",
        ]
    ]
    base = pd.DataFrame(rows)
    for column in ("window", "semantic_name"):
        if column not in base:
            base[column] = None
    return pd.concat([base, directional], ignore_index=True, sort=False)


def write_schema_audit(
    warehouse_root: Path | str,
    output_root: Path | str,
    trade_date: str | None = None,
    requested_fields: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, object]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    audit = build_schema_audit(warehouse_root, trade_date, requested_fields)
    parquet_path = output / "nff_schema_audit.parquet"
    csv_path = output / "nff_schema_audit.csv"
    json_path = output / "nff_schema_summary.json"
    audit.to_parquet(parquet_path, index=False)
    audit.to_csv(csv_path, index=False)
    status = audit.get("field_status", pd.Series(dtype=str))
    summary = {
        "trade_date": trade_date,
        "rows": int(len(audit)),
        "required_direction_missing": int((status == "REQUIRED_DIRECTION_MISSING").sum()),
        "requested_missing": int((status == "REQUESTED_MISSING").sum()),
        "nff_repository": NFF_SOURCE_REPOSITORY,
        "nff_branch": NFF_SOURCE_BRANCH,
        "parquet": str(parquet_path),
        "csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
