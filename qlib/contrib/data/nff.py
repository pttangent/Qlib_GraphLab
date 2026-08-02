# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Direct, point-in-time-safe access from a NodeFactorFactory warehouse to Qlib.

The adapter keeps NFF Parquet datasets as the source of truth. It selects one
schema per source, prunes date partitions and columns before materialization,
checks NFF completion manifests, aligns all requested fields on the original
``(symbol_id, timestamp)`` key, and only then maps the row to a Qlib decision
clock using the maximum upstream ``available_time``.

No Qlib binary copy of the NFF warehouse is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import DataLoader


KEY_COLUMNS = ("symbol_id", "symbol", "timestamp", "available_time")
_SCHEMA_RE = re.compile(r"^schema=(.+)$")
_DATE_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")


@dataclass(frozen=True)
class NFFSourceSpec:
    """One NFF canonical or feature dataset requested by an experiment."""

    kind: str
    dataset: str
    columns: Tuple[str, ...]
    schema_version: Optional[str] = None
    prefix: Optional[str] = None
    aliases: Optional[Mapping[str, str]] = None

    @property
    def name(self) -> str:
        return f"{self.kind}:{self.dataset}"

    @property
    def feature_prefix(self) -> str:
        return self.prefix or self.dataset


@dataclass(frozen=True)
class NFFExecutionPolicy:
    """Map NFF availability to the Qlib decision clock.

    ``decision_time = ceil(max_available_time, frequency) + delay_bars * frequency``

    The default one-bar delay is deliberately conservative for minute data: a
    bar timestamped 10:00 and available at 10:01 is assigned to 10:02.
    """

    frequency: str = "1min"
    delay_bars: int = 1
    collision_policy: str = "latest"

    def timedelta(self) -> pd.Timedelta:
        delta = pd.Timedelta(self.frequency)
        if delta <= pd.Timedelta(0):
            raise ValueError("execution frequency must be positive")
        if self.delay_bars < 0:
            raise ValueError("delay_bars must be non-negative")
        if self.collision_policy not in {"latest", "error"}:
            raise ValueError("collision_policy must be 'latest' or 'error'")
        return delta


@dataclass(frozen=True)
class NFFForwardReturnLabel:
    """Forward return calculated from canonical ``bars_1m`` at decision time.

    Entry is the selected bar field at ``decision_time``. Exit is the selected
    field ``horizon_bars`` later in the symbol's observed bar sequence. This
    makes the label start after all requested NFF inputs are available.
    """

    name: str = "LABEL0"
    dataset: str = "bars_1m"
    schema_version: Optional[str] = None
    entry_column: str = "open"
    exit_column: str = "open"
    horizon_bars: int = 30
    future_sessions: int = 5

    def validate(self) -> None:
        if self.horizon_bars <= 0:
            raise ValueError("label horizon_bars must be positive")
        if self.future_sessions < 0:
            raise ValueError("label future_sessions must be non-negative")


def _utc_timestamp(value: Optional[Union[str, pd.Timestamp]]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _schema_sort_key(value: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", value)
    return (int(match.group(1)) if match else -1, value)


def _normalise_columns(value: Union[Sequence[str], Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
        columns = result.get("columns")
        if not columns:
            raise ValueError("source config requires a non-empty 'columns' list")
        result["columns"] = tuple(str(column) for column in columns)
        aliases = result.get("aliases") or {}
        result["aliases"] = {str(key): str(alias) for key, alias in aliases.items()}
        return result
    return {"columns": tuple(str(column) for column in value), "aliases": {}}


def _build_source_specs(
    kind: str,
    configs: Optional[Mapping[str, Union[Sequence[str], Mapping[str, Any]]]],
) -> List[NFFSourceSpec]:
    specs: List[NFFSourceSpec] = []
    for dataset, raw in (configs or {}).items():
        config = _normalise_columns(raw)
        specs.append(
            NFFSourceSpec(
                kind=kind,
                dataset=str(dataset),
                columns=tuple(config["columns"]),
                schema_version=config.get("schema_version"),
                prefix=config.get("prefix"),
                aliases=config.get("aliases"),
            )
        )
    return specs


def _parse_execution(value: Optional[Mapping[str, Any]]) -> NFFExecutionPolicy:
    return NFFExecutionPolicy(**dict(value or {}))


def _parse_label(value: Optional[Mapping[str, Any]]) -> Optional[NFFForwardReturnLabel]:
    if value is None:
        return None
    label = NFFForwardReturnLabel(**dict(value))
    label.validate()
    return label


class NFFWarehouseCatalog:
    """Filesystem/catalog inspection for an NFF warehouse."""

    def __init__(self, warehouse_root: Union[str, Path]):
        self.warehouse_root = Path(warehouse_root).expanduser().resolve()

    def _base(self, kind: str, dataset: str) -> Path:
        if kind not in {"canonical", "feature"}:
            raise ValueError("source kind must be 'canonical' or 'feature'")
        group = "canonical" if kind == "canonical" else "features"
        return self.warehouse_root / group / dataset

    def available_sources(self, kind: Optional[str] = None) -> Dict[str, List[str]]:
        kinds = [kind] if kind else ["canonical", "feature"]
        result: Dict[str, List[str]] = {}
        for item in kinds:
            group = "canonical" if item == "canonical" else "features"
            root = self.warehouse_root / group
            result[item] = sorted(path.name for path in root.iterdir() if path.is_dir()) if root.exists() else []
        return result

    def available_schemas(self, kind: str, dataset: str) -> List[str]:
        root = self._base(kind, dataset)
        schemas = []
        if root.exists():
            for path in root.iterdir():
                match = _SCHEMA_RE.match(path.name)
                if path.is_dir() and match:
                    schemas.append(match.group(1))
        return sorted(schemas, key=_schema_sort_key)

    def resolve_schema(self, kind: str, dataset: str, requested: Optional[str] = None) -> Tuple[str, Path]:
        schemas = self.available_schemas(kind, dataset)
        if requested:
            if requested not in schemas:
                raise FileNotFoundError(
                    f"NFF source {kind}:{dataset} has no schema={requested}; available={schemas}"
                )
            schema = requested
        else:
            if not schemas:
                raise FileNotFoundError(f"No schemas found for NFF source {kind}:{dataset}")
            schema = schemas[-1]
        return schema, self._base(kind, dataset) / f"schema={schema}"

    def available_dates(self, kind: str, dataset: str, schema_version: Optional[str] = None) -> List[str]:
        _, root = self.resolve_schema(kind, dataset, schema_version)
        dates = []
        for path in root.iterdir() if root.exists() else ():
            match = _DATE_RE.match(path.name)
            if path.is_dir() and match:
                dates.append(match.group(1))
        return sorted(dates)

    def columns(self, kind: str, dataset: str, schema_version: Optional[str] = None) -> List[str]:
        schema, root = self.resolve_schema(kind, dataset, schema_version)
        files = sorted(root.glob("date=*/**/*.parquet"))
        if not files:
            files = sorted(root.glob("date=*/*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files for {kind}:{dataset}:schema={schema}")
        return list(pq.read_schema(files[0]).names)

    def describe(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"warehouse_root": str(self.warehouse_root), "sources": {}}
        for kind, datasets in self.available_sources().items():
            for dataset in datasets:
                schemas = self.available_schemas(kind, dataset)
                selected = schemas[-1] if schemas else None
                key = f"{kind}:{dataset}"
                result["sources"][key] = {
                    "schemas": schemas,
                    "selected_schema": selected,
                    "dates": self.available_dates(kind, dataset, selected) if selected else [],
                    "columns": self.columns(kind, dataset, selected) if selected else [],
                }
        return result


class NFFDataLoader(DataLoader):
    """Load NFF canonical and feature Parquet directly into Qlib.

    Parameters are plain dictionaries so the class can be instantiated from a
    Qlib YAML/JSON config using ``module_path: qlib.contrib.data.nff``.
    """

    def __init__(
        self,
        warehouse_root: Union[str, Path],
        feature_sets: Optional[Mapping[str, Union[Sequence[str], Mapping[str, Any]]]] = None,
        canonical_sets: Optional[Mapping[str, Union[Sequence[str], Mapping[str, Any]]]] = None,
        execution: Optional[Mapping[str, Any]] = None,
        label: Optional[Mapping[str, Any]] = None,
        join: str = "outer",
        strict_manifests: bool = True,
        allow_mixed_contracts: bool = False,
        output_float32: bool = True,
        arrow_use_threads: bool = True,
    ):
        self.warehouse_root = Path(warehouse_root).expanduser().resolve()
        self.catalog = NFFWarehouseCatalog(self.warehouse_root)
        self.sources = _build_source_specs("feature", feature_sets) + _build_source_specs(
            "canonical", canonical_sets
        )
        if not self.sources:
            raise ValueError("At least one NFF feature_sets or canonical_sets source is required")
        output_names: List[str] = []
        for spec in self.sources:
            aliases = dict(spec.aliases or {})
            output_names.extend(aliases.get(column, f"{spec.feature_prefix}__{column}") for column in spec.columns)
        duplicates = sorted({name for name in output_names if output_names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate NFF output feature names: {duplicates}")
        if join not in {"inner", "outer"}:
            raise ValueError("join must be 'inner' or 'outer'")
        self.join = join
        self.execution = _parse_execution(execution)
        self.label = _parse_label(label)
        self.strict_manifests = bool(strict_manifests)
        self.allow_mixed_contracts = bool(allow_mixed_contracts)
        self.output_float32 = bool(output_float32)
        self.arrow_use_threads = bool(arrow_use_threads)
        self.last_load_report: Dict[str, Any] = {}

    def _date_manifest(self, trade_date: str) -> Path:
        return self.warehouse_root / "manifests" / f"date={trade_date}.json"

    def _family_manifest(self, dataset: str, trade_date: str) -> Path:
        return self.warehouse_root / "manifests" / "features" / dataset / f"date={trade_date}.json"

    @staticmethod
    def _read_json(path: Path) -> Optional[Dict[str, Any]]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    def _validate_partition(
        self,
        spec: NFFSourceSpec,
        schema_version: str,
        trade_date: str,
        partition: Path,
    ) -> Optional[Tuple[Any, ...]]:
        if not (partition / "_SUCCESS").exists():
            raise FileNotFoundError(f"NFF partition is not published: {partition / '_SUCCESS'}")

        family_manifest = self._family_manifest(spec.dataset, trade_date)
        manifest = self._read_json(family_manifest) if family_manifest.exists() else None
        if manifest is None:
            manifest = self._read_json(self._date_manifest(trade_date))

        if manifest is None:
            if self.strict_manifests:
                raise FileNotFoundError(f"No NFF completion manifest for {spec.name} date={trade_date}")
            return None
        if manifest.get("status") != "complete":
            raise ValueError(f"NFF manifest is not complete for {spec.name} date={trade_date}")

        contract = manifest.get("contract") if isinstance(manifest.get("contract"), dict) else {}
        actual_schema = contract.get("schema_version")
        if actual_schema is not None and str(actual_schema) != str(schema_version):
            raise ValueError(
                f"NFF manifest/schema mismatch for {spec.name} date={trade_date}: "
                f"manifest={actual_schema}, path={schema_version}"
            )
        if not contract:
            return None
        return (
            contract.get("schema_version"),
            contract.get("family_version"),
            contract.get("implementation_hash"),
        )

    def _select_partitions(
        self,
        spec: NFFSourceSpec,
        start_time: Optional[pd.Timestamp],
        end_time: Optional[pd.Timestamp],
        future_sessions: int = 0,
    ) -> Tuple[str, List[Tuple[str, Path]], List[Tuple[Any, ...]]]:
        schema, root = self.catalog.resolve_schema(spec.kind, spec.dataset, spec.schema_version)
        available: List[Tuple[str, Path]] = []
        for path in root.iterdir() if root.exists() else ():
            match = _DATE_RE.match(path.name)
            if path.is_dir() and match:
                available.append((match.group(1), path))
        available.sort(key=lambda item: item[0])

        start_date = (start_time - pd.Timedelta(days=1)).date() if start_time is not None else None
        end_date = end_time.date() if end_time is not None else None
        selected = [
            item
            for item in available
            if (start_date is None or pd.Timestamp(item[0]).date() >= start_date)
            and (end_date is None or pd.Timestamp(item[0]).date() <= end_date)
        ]
        if future_sessions and end_date is not None:
            later = [item for item in available if pd.Timestamp(item[0]).date() > end_date]
            selected.extend(later[:future_sessions])
            selected = sorted(dict(selected).items())

        if not selected:
            raise FileNotFoundError(
                f"No NFF partitions selected for {spec.name}:schema={schema}, "
                f"start={start_time}, end={end_time}"
            )

        signatures: List[Tuple[Any, ...]] = []
        for trade_date, partition in selected:
            signature = self._validate_partition(spec, schema, trade_date, partition)
            if signature is not None:
                signatures.append(signature)
        unique_signatures = sorted({signature for signature in signatures}, key=repr)
        if len(unique_signatures) > 1 and not self.allow_mixed_contracts:
            raise ValueError(
                f"Mixed NFF contracts for {spec.name}:schema={schema}: {unique_signatures}. "
                "Pin a homogeneous rebuild or set allow_mixed_contracts=True explicitly."
            )
        return schema, selected, unique_signatures

    def _read_source(
        self,
        spec: NFFSourceSpec,
        start_time: Optional[pd.Timestamp],
        end_time: Optional[pd.Timestamp],
        instruments: Optional[List[str]],
        future_sessions: int = 0,
        extra_columns: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        schema, partitions, signatures = self._select_partitions(
            spec, start_time, end_time, future_sessions=future_sessions
        )
        files: List[str] = []
        for _, partition in partitions:
            files.extend(str(path) for path in sorted(partition.rglob("*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet files selected for {spec.name}:schema={schema}")

        dataset = pads.dataset(files, format="parquet")
        available_columns = set(dataset.schema.names)
        requested_features = list(spec.columns)
        requested = list(KEY_COLUMNS) + requested_features + list(extra_columns or ())
        requested = list(dict.fromkeys(requested))
        missing = sorted(set(requested) - available_columns)
        if missing:
            raise KeyError(
                f"NFF source {spec.name}:schema={schema} missing columns {missing}; "
                f"available={sorted(available_columns)}"
            )

        filter_expr = None
        if start_time is not None:
            source_start = (start_time - pd.Timedelta(days=1)).to_pydatetime()
            filter_expr = pads.field("timestamp") >= pa.scalar(source_start)
        if end_time is not None and future_sessions == 0:
            end_expr = pads.field("timestamp") <= pa.scalar(end_time.to_pydatetime())
            filter_expr = end_expr if filter_expr is None else filter_expr & end_expr
        if instruments:
            symbol_expr = pads.field("symbol").isin(instruments)
            filter_expr = symbol_expr if filter_expr is None else filter_expr & symbol_expr
        table = dataset.to_table(
            columns=requested,
            filter=filter_expr,
            use_threads=self.arrow_use_threads,
        )
        frame = table.to_pandas(split_blocks=True)

        if frame.empty:
            report = {
                "source": spec.name,
                "schema_version": schema,
                "dates": [date for date, _ in partitions],
                "files": len(files),
                "rows": 0,
                "contracts": signatures,
            }
            return frame, report

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame["available_time"] = pd.to_datetime(frame["available_time"], utc=True, errors="coerce")
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        frame = frame.dropna(subset=["symbol_id", "symbol", "timestamp", "available_time"])
        duplicate_count = int(frame.duplicated(["symbol_id", "timestamp"]).sum())
        if duplicate_count:
            raise ValueError(f"NFF source {spec.name} has {duplicate_count} duplicate symbol/timestamp rows")

        aliases = dict(spec.aliases or {})
        rename = {}
        for column in requested_features:
            output_name = aliases.get(column, f"{spec.feature_prefix}__{column}")
            rename[column] = output_name
        frame = frame.rename(
            columns={
                "symbol": f"__symbol__{spec.feature_prefix}",
                "available_time": f"__available__{spec.feature_prefix}",
                **rename,
            }
        )
        report = {
            "source": spec.name,
            "schema_version": schema,
            "dates": [date for date, _ in partitions],
            "files": len(files),
            "rows": int(len(frame)),
            "contracts": signatures,
            "feature_columns": list(rename.values()),
        }
        return frame, report

    @staticmethod
    def _normalise_instruments(instruments: Any) -> Optional[List[str]]:
        if instruments is None:
            return None
        if isinstance(instruments, str):
            if instruments.lower() in {"all", "*"}:
                return None
            return [instruments.upper().strip()]
        if isinstance(instruments, Mapping):
            values = list(instruments.keys())
        else:
            try:
                values = list(instruments)
            except TypeError as exc:
                raise KeyError(f"Unsupported instruments filter: {type(instruments)!r}") from exc
        return sorted({str(value).upper().strip() for value in values if str(value).strip()})

    def _merge_sources(self, frames: List[pd.DataFrame]) -> pd.DataFrame:
        key_columns = ["symbol_id", "timestamp"]
        indexed = [frame.set_index(key_columns) for frame in frames if not frame.empty]
        if not indexed:
            return pd.DataFrame()
        merged = pd.concat(indexed, axis=1, join=self.join, copy=False).reset_index()
        if merged.empty:
            return merged

        symbol_columns = [column for column in merged.columns if column.startswith("__symbol__")]
        available_columns = [column for column in merged.columns if column.startswith("__available__")]
        if not symbol_columns or not available_columns:
            raise ValueError("NFF sources did not provide symbol and available_time metadata")

        symbol_matrix = merged[symbol_columns]
        conflict = symbol_matrix.nunique(axis=1, dropna=True) > 1
        if bool(conflict.any()):
            raise ValueError(f"Conflicting NFF symbols for {int(conflict.sum())} merged rows")
        merged["instrument"] = symbol_matrix.bfill(axis=1).iloc[:, 0]
        merged["available_time"] = merged[available_columns].max(axis=1)
        merged["event_time"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
        return merged.drop(columns=symbol_columns + available_columns)

    def _apply_execution_clock(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        if frame.empty:
            return frame, 0
        delta = self.execution.timedelta()
        frame = frame.dropna(subset=["instrument", "event_time", "available_time"]).copy()
        frame["datetime"] = frame["available_time"].dt.ceil(self.execution.frequency) + (
            self.execution.delay_bars * delta
        )
        if bool((frame["datetime"] < frame["available_time"]).any()):
            raise AssertionError("Qlib decision_time precedes NFF available_time")

        duplicates = frame.duplicated(["datetime", "symbol_id"], keep=False)
        collision_rows = int(duplicates.sum())
        if collision_rows:
            if self.execution.collision_policy == "error":
                raise ValueError(f"{collision_rows} NFF rows collide on the Qlib decision clock")
            frame = frame.sort_values(
                ["datetime", "symbol_id", "available_time", "event_time"], kind="mergesort"
            ).drop_duplicates(["datetime", "symbol_id"], keep="last")
        return frame, collision_rows

    def _attach_label(
        self,
        frame: pd.DataFrame,
        start_time: Optional[pd.Timestamp],
        end_time: Optional[pd.Timestamp],
        instruments: Optional[List[str]],
    ) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
        if self.label is None or frame.empty:
            return frame, None
        label = self.label
        spec = NFFSourceSpec(
            kind="canonical",
            dataset=label.dataset,
            columns=(),
            schema_version=label.schema_version,
            prefix="__label_bars",
        )
        bars, report = self._read_source(
            spec,
            start_time,
            end_time,
            instruments,
            future_sessions=label.future_sessions,
            extra_columns=(label.entry_column, label.exit_column),
        )
        if bars.empty:
            frame[label.name] = np.nan
            return frame, report

        symbol_column = "__symbol____label_bars"
        available_column = "__available____label_bars"
        bars = bars.rename(columns={symbol_column: "__label_symbol"})
        bars = bars.drop(columns=[available_column], errors="ignore")
        bars = bars.sort_values(["symbol_id", "timestamp"], kind="mergesort")
        exit_values = bars.groupby("symbol_id", sort=False)[label.exit_column].shift(-label.horizon_bars)
        entry_values = pd.to_numeric(bars[label.entry_column], errors="coerce")
        bars[label.name] = pd.to_numeric(exit_values, errors="coerce") / entry_values - 1.0
        labels = bars[["symbol_id", "timestamp", label.name]].rename(columns={"timestamp": "datetime"})
        frame = frame.merge(labels, on=["symbol_id", "datetime"], how="left", validate="many_to_one")
        report = dict(report)
        report.update(
            {
                "label_name": label.name,
                "entry_column": label.entry_column,
                "exit_column": label.exit_column,
                "horizon_bars": label.horizon_bars,
                "non_null_labels": int(frame[label.name].notna().sum()),
            }
        )
        return frame, report

    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        start = _utc_timestamp(start_time)
        end = _utc_timestamp(end_time)
        if start is not None and end is not None and start > end:
            raise ValueError("start_time must not be after end_time")
        selected_instruments = self._normalise_instruments(instruments)

        source_frames: List[pd.DataFrame] = []
        source_reports: List[Dict[str, Any]] = []
        for spec in self.sources:
            frame, report = self._read_source(
                spec, start, end, selected_instruments, future_sessions=0
            )
            source_frames.append(frame)
            source_reports.append(report)
        merged = self._merge_sources(source_frames)
        merged, collision_rows = self._apply_execution_clock(merged)
        merged, label_report = self._attach_label(
            merged, start, end, selected_instruments
        )

        if merged.empty:
            empty_index = pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])
            result = pd.DataFrame(index=empty_index)
            self.last_load_report = {
                "rows": 0,
                "sources": source_reports,
                "label": label_report,
            }
            return result

        if start is not None:
            merged = merged[merged["datetime"] >= start]
        if end is not None:
            merged = merged[merged["datetime"] <= end]
        if selected_instruments:
            merged = merged[merged["instrument"].isin(selected_instruments)]

        metadata = {
            "symbol_id",
            "timestamp",
            "event_time",
            "available_time",
            "datetime",
            "instrument",
        }
        label_columns = [self.label.name] if self.label is not None else []
        feature_columns = [
            column for column in merged.columns if column not in metadata and column not in label_columns
        ]
        if not feature_columns:
            raise ValueError("NFF adapter produced no feature columns")

        for column in feature_columns + label_columns:
            if column in merged:
                converted = pd.to_numeric(merged[column], errors="coerce")
                if self.output_float32:
                    converted = converted.astype("float32")
                merged[column] = converted

        merged = merged.sort_values(["datetime", "instrument"], kind="mergesort")
        index = pd.MultiIndex.from_frame(merged[["datetime", "instrument"]])
        index.names = ["datetime", "instrument"]
        feature_block = merged[feature_columns].copy()
        feature_block.index = index
        blocks = {"feature": feature_block}
        if label_columns:
            label_block = merged[label_columns].copy()
            label_block.index = index
            blocks["label"] = label_block
        result = pd.concat(blocks, axis=1).sort_index()

        self.last_load_report = {
            "warehouse_root": str(self.warehouse_root),
            "rows": int(len(result)),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "start_time": None if start is None else start.isoformat(),
            "end_time": None if end is None else end.isoformat(),
            "execution": {
                "frequency": self.execution.frequency,
                "delay_bars": self.execution.delay_bars,
                "collision_policy": self.execution.collision_policy,
                "collision_rows": collision_rows,
            },
            "sources": source_reports,
            "label": label_report,
        }
        return result


class NFFDataHandlerLP(DataHandlerLP):
    """Qlib ``DataHandlerLP`` pre-wired with :class:`NFFDataLoader`."""

    def __init__(
        self,
        warehouse_root: Union[str, Path],
        feature_sets: Optional[Mapping[str, Union[Sequence[str], Mapping[str, Any]]]] = None,
        canonical_sets: Optional[Mapping[str, Union[Sequence[str], Mapping[str, Any]]]] = None,
        execution: Optional[Mapping[str, Any]] = None,
        label: Optional[Mapping[str, Any]] = None,
        instruments=None,
        start_time=None,
        end_time=None,
        infer_processors: List[Any] = [],
        learn_processors: List[Any] = [],
        shared_processors: List[Any] = [],
        process_type=DataHandlerLP.PTYPE_A,
        drop_raw: bool = True,
        loader_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        loader_config = dict(loader_kwargs or {})
        loader = NFFDataLoader(
            warehouse_root=warehouse_root,
            feature_sets=feature_sets,
            canonical_sets=canonical_sets,
            execution=execution,
            label=label,
            **loader_config,
        )
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            shared_processors=shared_processors,
            process_type=process_type,
            drop_raw=drop_raw,
            **kwargs,
        )


def make_nff_handler_config(**kwargs: Any) -> Dict[str, Any]:
    """Return a config accepted by Qlib's ``init_instance_by_config``."""

    return {
        "class": "NFFDataHandlerLP",
        "module_path": "qlib.contrib.data.nff",
        "kwargs": kwargs,
    }
