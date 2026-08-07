# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Point-in-time-safe stock-day episode construction from an NFF warehouse.

The existing :mod:`qlib.contrib.data.nff` adapter remains the source reader and
clock authority.  This module adds a second, task-oriented view:

``stock x trading day -> support window + causal query windows + future labels``.

No GFF/GAL input is required and no second wide warehouse copy is created.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from qlib.contrib.data.nff import NFFDataLoader, NFFSourceSpec


_METADATA_COLUMNS = {
    "symbol_id",
    "timestamp",
    "event_time",
    "available_time",
    "datetime",
    "instrument",
    "trade_date",
}


def _utc_timestamp(value: Union[str, pd.Timestamp]) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _clock_on_day(trade_date: Union[str, date], clock: str, timezone: str) -> pd.Timestamp:
    local = pd.Timestamp(f"{trade_date} {clock}", tz=timezone)
    return local.tz_convert("UTC").tz_localize(None)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def contract_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SupportQueryPolicy:
    """Intraday support/query contract expressed in exchange-local time."""

    timezone: str = "America/New_York"
    support_start: str = "09:30"
    support_end: str = "10:00"
    query_start: str = "10:00"
    query_end: str = "15:00"
    session_end: str = "16:00"
    query_stride_minutes: int = 15
    query_lookback_minutes: int = 60
    min_support_coverage: float = 0.60
    min_query_coverage: float = 0.60
    max_symbols_per_day: Optional[int] = None
    universe_rank_column: Optional[str] = None

    def validate(self) -> None:
        if self.query_stride_minutes <= 0:
            raise ValueError("query_stride_minutes must be positive")
        if self.query_lookback_minutes <= 0:
            raise ValueError("query_lookback_minutes must be positive")
        for name, value in {
            "min_support_coverage": self.min_support_coverage,
            "min_query_coverage": self.min_query_coverage,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_symbols_per_day is not None and self.max_symbols_per_day <= 0:
            raise ValueError("max_symbols_per_day must be positive when configured")
        anchor = "2026-01-05"
        if _clock_on_day(anchor, self.support_start, self.timezone) >= _clock_on_day(
            anchor, self.support_end, self.timezone
        ):
            raise ValueError("support_start must be before support_end")
        if _clock_on_day(anchor, self.query_start, self.timezone) > _clock_on_day(
            anchor, self.query_end, self.timezone
        ):
            raise ValueError("query_start must not be after query_end")
        if _clock_on_day(anchor, self.support_end, self.timezone) > _clock_on_day(
            anchor, self.query_start, self.timezone
        ):
            raise ValueError("support_end must not be after query_start")
        if _clock_on_day(anchor, self.query_end, self.timezone) >= _clock_on_day(
            anchor, self.session_end, self.timezone
        ):
            raise ValueError("query_end must be before session_end")


@dataclass(frozen=True)
class EpisodeTargetSpec:
    """One causal target evaluated at every admitted query timestamp."""

    name: str
    kind: str
    horizon_minutes: int
    price_column: str = "bars_1m__open"
    entry_delay_minutes: int = 1
    minimum_observations: int = 5

    def validate(self) -> None:
        if self.kind not in {"forward_return", "realized_volatility"}:
            raise ValueError(f"Unsupported target kind: {self.kind}")
        if self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        if self.entry_delay_minutes < 0:
            raise ValueError("entry_delay_minutes must be non-negative")
        if self.minimum_observations <= 0:
            raise ValueError("minimum_observations must be positive")


@dataclass
class StockDayEpisode:
    """One stock-day task with a fixed support block and multiple query points."""

    symbol: str
    trade_date: str
    feature_names: Tuple[str, ...]
    target_names: Tuple[str, ...]
    support_times_ns: np.ndarray
    support_x: np.ndarray
    support_observed: np.ndarray
    query_times_ns: np.ndarray
    query_x: np.ndarray
    query_observed: np.ndarray
    targets: np.ndarray
    target_observed: np.ndarray
    audit: Dict[str, Any]

    @property
    def query_count(self) -> int:
        return int(len(self.query_times_ns))

    def manifest_row(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "trade_date": self.trade_date,
            "feature_count": len(self.feature_names),
            "target_count": len(self.target_names),
            "support_rows": int(self.support_x.shape[0]),
            "query_count": self.query_count,
            "support_coverage": float(self.audit["support_coverage"]),
            "query_coverage_mean": float(self.audit["query_coverage_mean"]),
            "target_coverage": float(self.audit["target_coverage"]),
            "first_query_time": pd.Timestamp(self.query_times_ns[0]).isoformat() if self.query_count else None,
            "last_query_time": pd.Timestamp(self.query_times_ns[-1]).isoformat() if self.query_count else None,
        }


@dataclass
class EpisodeBuildResult:
    episodes: List[StockDayEpisode]
    rejections: List[Dict[str, Any]]
    feature_names: Tuple[str, ...]
    target_names: Tuple[str, ...]

    def manifest_frame(self) -> pd.DataFrame:
        return pd.DataFrame([episode.manifest_row() for episode in self.episodes])

    def rejection_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rejections)


def _feature_columns(frame: pd.DataFrame, explicit: Optional[Sequence[str]] = None) -> List[str]:
    if explicit is not None:
        missing = sorted(set(explicit) - set(frame.columns))
        if missing:
            raise KeyError(f"Aligned NFF frame is missing requested episode features: {missing}")
        return list(explicit)
    return [
        column
        for column in frame.columns
        if column not in _METADATA_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _coverage(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.isfinite(values).mean())


def _select_universe(
    frame: pd.DataFrame,
    trade_date: str,
    policy: SupportQueryPolicy,
    feature_columns: Sequence[str],
) -> List[str]:
    symbols = sorted(frame["instrument"].dropna().astype(str).unique())
    if policy.max_symbols_per_day is None or len(symbols) <= policy.max_symbols_per_day:
        return symbols
    rank_column = policy.universe_rank_column
    if rank_column is None:
        return symbols[: policy.max_symbols_per_day]
    if rank_column not in feature_columns:
        raise KeyError(f"universe_rank_column={rank_column!r} is not an episode feature")
    support_start = _clock_on_day(trade_date, policy.support_start, policy.timezone)
    support_end = _clock_on_day(trade_date, policy.support_end, policy.timezone)
    support = frame[(frame["datetime"] >= support_start) & (frame["datetime"] < support_end)]
    ranking = support.groupby("instrument", sort=False)[rank_column].mean().dropna().sort_values(ascending=False)
    return list(ranking.head(policy.max_symbols_per_day).index.astype(str))


def _target_value(
    bars: pd.DataFrame,
    query_time: pd.Timestamp,
    spec: EpisodeTargetSpec,
) -> float:
    entry_time = query_time + pd.Timedelta(minutes=spec.entry_delay_minutes)
    exit_time = entry_time + pd.Timedelta(minutes=spec.horizon_minutes)
    if spec.price_column not in bars.columns:
        return np.nan
    series = pd.to_numeric(bars[spec.price_column], errors="coerce")
    if spec.kind == "forward_return":
        try:
            entry = float(series.loc[entry_time])
            exit_value = float(series.loc[exit_time])
        except (KeyError, TypeError, ValueError):
            return np.nan
        if not np.isfinite(entry) or not np.isfinite(exit_value) or entry == 0.0:
            return np.nan
        return exit_value / entry - 1.0

    window = series.loc[(series.index >= entry_time) & (series.index <= exit_time)].dropna()
    if len(window) < max(2, spec.minimum_observations):
        return np.nan
    values = window.to_numpy(dtype="float64")
    if np.any(values <= 0) or not np.isfinite(values).all():
        return np.nan
    returns = np.diff(np.log(values))
    if len(returns) < 1:
        return np.nan
    return float(np.sqrt(np.square(returns).sum()))


def build_stock_day_episodes(
    aligned: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    policy: SupportQueryPolicy,
    targets: Sequence[EpisodeTargetSpec],
    feature_columns: Optional[Sequence[str]] = None,
) -> EpisodeBuildResult:
    """Build stock-day episodes from aligned NFF rows and raw bar clocks."""

    policy.validate()
    for target in targets:
        target.validate()
    if aligned.empty:
        return EpisodeBuildResult([], [], tuple(feature_columns or ()), tuple(target.name for target in targets))
    required_aligned = {"datetime", "instrument"}
    required_bars = {"datetime", "instrument"}
    if missing := sorted(required_aligned - set(aligned.columns)):
        raise KeyError(f"Aligned frame missing columns: {missing}")
    if missing := sorted(required_bars - set(bars.columns)):
        raise KeyError(f"Bars frame missing columns: {missing}")

    aligned = aligned.copy()
    aligned["datetime"] = pd.to_datetime(aligned["datetime"], utc=True, errors="coerce").dt.tz_localize(None)
    aligned = aligned.dropna(subset=["datetime", "instrument"])
    aligned["trade_date"] = (
        pd.DatetimeIndex(aligned["datetime"]).tz_localize("UTC").tz_convert(policy.timezone).date.astype(str)
    )

    bars = bars.copy()
    bars["datetime"] = pd.to_datetime(bars["datetime"], utc=True, errors="coerce").dt.tz_localize(None)
    bars = bars.dropna(subset=["datetime", "instrument"])
    bars["trade_date"] = (
        pd.DatetimeIndex(bars["datetime"]).tz_localize("UTC").tz_convert(policy.timezone).date.astype(str)
    )

    features = _feature_columns(aligned, feature_columns)
    target_names = tuple(target.name for target in targets)
    episodes: List[StockDayEpisode] = []
    rejections: List[Dict[str, Any]] = []

    for trade_date, day_frame in aligned.groupby("trade_date", sort=True):
        admitted_symbols = set(_select_universe(day_frame, trade_date, policy, features))
        support_start = _clock_on_day(trade_date, policy.support_start, policy.timezone)
        support_end = _clock_on_day(trade_date, policy.support_end, policy.timezone)
        query_start = _clock_on_day(trade_date, policy.query_start, policy.timezone)
        query_end = _clock_on_day(trade_date, policy.query_end, policy.timezone)
        session_end = _clock_on_day(trade_date, policy.session_end, policy.timezone)
        support_grid = pd.date_range(support_start, support_end - pd.Timedelta(minutes=1), freq="1min")
        query_grid = pd.date_range(query_start, query_end, freq=f"{policy.query_stride_minutes}min")

        for symbol, symbol_frame in day_frame.groupby("instrument", sort=True):
            symbol = str(symbol)
            if symbol not in admitted_symbols:
                rejections.append({"symbol": symbol, "trade_date": trade_date, "reason": "outside_universe"})
                continue
            symbol_frame = symbol_frame.sort_values("datetime", kind="mergesort").drop_duplicates("datetime", keep="last")
            indexed = symbol_frame.set_index("datetime")[features]
            support = indexed.reindex(support_grid)
            support_values = support.to_numpy(dtype="float32")
            support_observed = np.isfinite(support_values)
            support_coverage = _coverage(support_values)
            if support_coverage < policy.min_support_coverage:
                rejections.append(
                    {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "reason": "insufficient_support_coverage",
                        "support_coverage": support_coverage,
                    }
                )
                continue

            symbol_bars = bars[(bars["trade_date"] == trade_date) & (bars["instrument"].astype(str) == symbol)]
            symbol_bars = symbol_bars.sort_values("datetime", kind="mergesort").drop_duplicates("datetime", keep="last")
            symbol_bars = symbol_bars.set_index("datetime")
            query_sequences: List[np.ndarray] = []
            query_observed: List[np.ndarray] = []
            query_times: List[np.datetime64] = []
            query_targets: List[np.ndarray] = []
            query_target_observed: List[np.ndarray] = []
            query_coverages: List[float] = []

            for query_time in query_grid:
                lookback_start = query_time - pd.Timedelta(minutes=policy.query_lookback_minutes - 1)
                lookback_grid = pd.date_range(lookback_start, query_time, freq="1min")
                sequence = indexed.reindex(lookback_grid).to_numpy(dtype="float32")
                sequence_observed = np.isfinite(sequence)
                coverage = _coverage(sequence)
                if coverage < policy.min_query_coverage:
                    continue
                values = np.asarray(
                    [
                        _target_value(symbol_bars, query_time, spec)
                        if query_time + pd.Timedelta(minutes=spec.entry_delay_minutes + spec.horizon_minutes) < session_end
                        else np.nan
                        for spec in targets
                    ],
                    dtype="float32",
                )
                observed = np.isfinite(values)
                if not observed.any():
                    continue
                query_sequences.append(sequence)
                query_observed.append(sequence_observed)
                query_times.append(np.datetime64(query_time.to_datetime64(), "ns"))
                query_targets.append(values)
                query_target_observed.append(observed)
                query_coverages.append(coverage)

            if not query_sequences:
                rejections.append({"symbol": symbol, "trade_date": trade_date, "reason": "no_admissible_queries"})
                continue

            target_matrix = np.stack(query_targets).astype("float32", copy=False)
            target_mask = np.stack(query_target_observed).astype(bool, copy=False)
            episode = StockDayEpisode(
                symbol=symbol,
                trade_date=trade_date,
                feature_names=tuple(features),
                target_names=target_names,
                support_times_ns=support_grid.to_numpy(dtype="datetime64[ns]").astype("int64"),
                support_x=support_values,
                support_observed=support_observed,
                query_times_ns=np.asarray(query_times, dtype="datetime64[ns]").astype("int64"),
                query_x=np.stack(query_sequences).astype("float32", copy=False),
                query_observed=np.stack(query_observed).astype(bool, copy=False),
                targets=target_matrix,
                target_observed=target_mask,
                audit={
                    "support_coverage": support_coverage,
                    "query_coverage_mean": float(np.mean(query_coverages)),
                    "target_coverage": float(target_mask.mean()),
                    "latest_support_time": support_grid[-1].isoformat() if len(support_grid) else None,
                    "first_query_time": pd.Timestamp(query_times[0]).isoformat(),
                    "causal": bool(support_grid[-1] < pd.Timestamp(query_times[0])) if len(support_grid) else True,
                },
            )
            if not episode.audit["causal"]:
                raise AssertionError("support window is not strictly before the first query")
            episodes.append(episode)

    return EpisodeBuildResult(episodes, rejections, tuple(features), target_names)


class NFFStockDayEpisodeFactory:
    """Read NFF partitions and construct stock-day episodes on demand."""

    def __init__(
        self,
        *,
        warehouse_root: Union[str, Path],
        feature_sets: Optional[Mapping[str, Any]] = None,
        canonical_sets: Optional[Mapping[str, Any]] = None,
        execution: Optional[Mapping[str, Any]] = None,
        policy: Optional[Union[SupportQueryPolicy, Mapping[str, Any]]] = None,
        targets: Optional[Sequence[Union[EpisodeTargetSpec, Mapping[str, Any]]]] = None,
        episode_features: Optional[Sequence[str]] = None,
        loader_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        self.warehouse_root = Path(warehouse_root).expanduser().resolve()
        self.policy = policy if isinstance(policy, SupportQueryPolicy) else SupportQueryPolicy(**dict(policy or {}))
        self.policy.validate()
        self.targets = tuple(
            item if isinstance(item, EpisodeTargetSpec) else EpisodeTargetSpec(**dict(item))
            for item in (targets or ())
        )
        if not self.targets:
            raise ValueError("At least one episode target is required")
        for target in self.targets:
            target.validate()
        self.episode_features = tuple(episode_features) if episode_features is not None else None
        kwargs = dict(loader_kwargs or {})
        self.loader = NFFDataLoader(
            warehouse_root=self.warehouse_root,
            feature_sets=feature_sets,
            canonical_sets=canonical_sets,
            execution=execution,
            label=None,
            **kwargs,
        )
        self.last_load_report: Dict[str, Any] = {}

    @property
    def contract(self) -> Dict[str, Any]:
        return {
            "warehouse_root": str(self.warehouse_root),
            "policy": asdict(self.policy),
            "targets": [asdict(target) for target in self.targets],
            "episode_features": list(self.episode_features) if self.episode_features is not None else None,
            "sources": [asdict(source) for source in self.loader.sources],
            "execution": asdict(self.loader.execution),
        }

    @property
    def contract_hash(self) -> str:
        return contract_hash(self.contract)

    def available_dates(self) -> List[str]:
        date_sets: List[set[str]] = []
        for source in self.loader.sources:
            date_sets.append(set(self.loader.catalog.available_dates(source.kind, source.dataset, source.schema_version)))
        return sorted(set.intersection(*date_sets)) if date_sets else []

    def _load_aligned(
        self,
        *,
        start_time: Union[str, pd.Timestamp],
        end_time: Union[str, pd.Timestamp],
        instruments: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]], int]:
        start = _utc_timestamp(start_time)
        end = _utc_timestamp(end_time)
        selected = self.loader._normalise_instruments(instruments)
        source_frames: List[pd.DataFrame] = []
        source_reports: List[Dict[str, Any]] = []
        for source in self.loader.sources:
            frame, report = self.loader._read_source(source, start, end, selected, future_sessions=0)
            source_frames.append(frame)
            source_reports.append(report)
        merged = self.loader._merge_sources(source_frames)
        merged, collision_rows = self.loader._apply_execution_clock(merged)
        if merged.empty:
            return merged, source_reports, collision_rows
        merged = merged[(merged["datetime"] >= start) & (merged["datetime"] <= end)].copy()
        if selected:
            merged = merged[merged["instrument"].isin(selected)].copy()
        merged["datetime"] = pd.to_datetime(merged["datetime"], utc=True, errors="coerce").dt.tz_localize(None)
        return merged.sort_values(["datetime", "instrument"], kind="mergesort"), source_reports, collision_rows

    def _load_bars(
        self,
        *,
        start_time: Union[str, pd.Timestamp],
        end_time: Union[str, pd.Timestamp],
        instruments: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        needed_columns = sorted({target.price_column.split("__", 1)[-1] for target in self.targets})
        candidates = [source for source in self.loader.sources if source.kind == "canonical" and source.dataset == "bars_1m"]
        if not candidates:
            raise ValueError("Episode targets require canonical bars_1m in canonical_sets")
        configured = candidates[0]
        configured_columns = set(configured.columns)
        if missing := sorted(set(needed_columns) - configured_columns):
            raise KeyError(f"bars_1m canonical_sets must include target price columns: {missing}")
        spec = NFFSourceSpec(
            kind="canonical",
            dataset="bars_1m",
            columns=tuple(needed_columns),
            schema_version=configured.schema_version,
            prefix="bars_1m",
        )
        start = _utc_timestamp(start_time)
        end = _utc_timestamp(end_time)
        selected = self.loader._normalise_instruments(instruments)
        frame, report = self.loader._read_source(
            spec,
            start,
            end + pd.Timedelta(minutes=max(target.horizon_minutes + target.entry_delay_minutes for target in self.targets)),
            selected,
            future_sessions=0,
            extra_columns=("trade_date",),
        )
        if frame.empty:
            return pd.DataFrame(columns=["datetime", "instrument"]), report
        frame = frame.rename(columns={"__symbol__bars_1m": "instrument", "timestamp": "datetime"})
        frame = frame.drop(columns=["__available__bars_1m", "symbol_id"], errors="ignore")
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce").dt.tz_localize(None)
        return frame.sort_values(["datetime", "instrument"], kind="mergesort"), report

    def load_date(
        self,
        trade_date: str,
        *,
        instruments: Optional[Sequence[str]] = None,
    ) -> EpisodeBuildResult:
        support_start = pd.Timestamp(f"{trade_date} {self.policy.support_start}", tz=self.policy.timezone)
        query_start = pd.Timestamp(f"{trade_date} {self.policy.query_start}", tz=self.policy.timezone)
        query_end = pd.Timestamp(f"{trade_date} {self.policy.query_end}", tz=self.policy.timezone)
        lookback_start = query_start - pd.Timedelta(minutes=self.policy.query_lookback_minutes - 1)
        start_local = min(support_start, lookback_start)
        end_local = query_end
        start = start_local.tz_convert("UTC")
        end = end_local.tz_convert("UTC")
        aligned, source_reports, collision_rows = self._load_aligned(
            start_time=start,
            end_time=end,
            instruments=instruments,
        )
        bars, bar_report = self._load_bars(start_time=start, end_time=end, instruments=instruments)
        result = build_stock_day_episodes(
            aligned,
            bars,
            policy=self.policy,
            targets=self.targets,
            feature_columns=self.episode_features,
        )
        self.last_load_report = {
            "trade_date": trade_date,
            "rows": int(len(aligned)),
            "episodes": len(result.episodes),
            "rejections": len(result.rejections),
            "collision_rows": int(collision_rows),
            "sources": source_reports,
            "bars": bar_report,
            "contract_hash": self.contract_hash,
        }
        return result


def make_episode_factory_config(**kwargs: Any) -> Dict[str, Any]:
    return {
        "class": "NFFStockDayEpisodeFactory",
        "module_path": "qlib.contrib.data.nff_episode",
        "kwargs": kwargs,
    }
