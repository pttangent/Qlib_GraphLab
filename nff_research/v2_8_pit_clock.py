from __future__ import annotations

"""Strict event/availability/decision clock for the v2.8 NFF research path.

The generic NFF adapter intentionally maps source rows onto a conservative
Qlib decision clock.  The v2.6/v2.7 research layer then joined several
published NFF tables (supplements, sketch, condition and venue) by treating
that decision-clock index as if it were the original source event timestamp.
That can attach a later event-minute venue/flow observation to an earlier
research decision and contaminates short-horizon labels.

This module installs a research-only correction without changing the NFF
warehouse:

* the loader emits hidden event/available/decision timing columns;
* every manually joined NFF table is keyed by the original event timestamp;
* sources carrying ``available_time`` must also satisfy
  ``source_available_time <= decision_time``;
* labels are rebuilt from independent canonical market bars using the actual
  decision time as the clock anchor (the configured +1 minute entry offset is
  preserved);
* selection is constrained by the factor registry's research role, so REGIME
  families such as J cannot enter the Alpha portfolio merely because a return
  screen is large;
* the stage contract is salted with this PIT policy so stale pre-fix
  materialize/basic/detailed/portfolio checkpoints cannot be reused.
"""

from collections.abc import Mapping
import gc
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.dataset as pads


PIT_CLOCK_VERSION = "v2.8.1-event-available-decision"
SESSION_TZ = ZoneInfo("America/New_York")
TIMING_EVENT = "__nff_event_time_ns"
TIMING_AVAILABLE = "__nff_available_time_ns"
TIMING_DECISION = "__nff_decision_time_ns"

ROLE_POLICY: dict[str, tuple[str, ...]] = {
    "alpha": ("DIRECTION_ALPHA", "CONFIRMATION"),
    "risk_regime": ("RISK", "REGIME", "LIQUIDITY", "CONFIRMATION"),
    "cost_liquidity": ("LIQUIDITY", "REGIME", "RISK"),
}


def _sqlish_dates(values: pd.Series) -> list[str]:
    times = pd.to_datetime(values, utc=True, errors="coerce")
    return sorted(
        times.dt.tz_convert(SESSION_TZ).dt.strftime("%Y-%m-%d").dropna().unique()
    )


def _timing_work(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index()
    missing = [name for name in (TIMING_EVENT, TIMING_AVAILABLE) if name not in frame]
    if missing:
        raise RuntimeError(
            "PIT research frame is missing loader timing metadata: " + ", ".join(missing)
        )
    work = frame.reset_index()
    work["__symbol"] = work["instrument"].astype("string").str.upper().str.strip()
    work["__event_time"] = pd.to_datetime(
        pd.to_numeric(work[TIMING_EVENT], errors="coerce"), unit="ns", utc=True, errors="coerce"
    )
    work["__base_available_time"] = pd.to_datetime(
        pd.to_numeric(work[TIMING_AVAILABLE], errors="coerce"), unit="ns", utc=True, errors="coerce"
    )
    work["__decision_time"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
    invalid = (
        work["__event_time"].isna()
        | work["__base_available_time"].isna()
        | work["__decision_time"].isna()
        | (work["__base_available_time"] > work["__decision_time"])
    )
    if bool(invalid.any()):
        raise RuntimeError(
            f"PIT timing metadata invalid for {int(invalid.sum())} research rows"
        )
    return work


def _restore_index(work: pd.DataFrame) -> pd.DataFrame:
    return (
        work.drop(
            columns=["__symbol", "__event_time", "__base_available_time", "__decision_time"],
            errors="ignore",
        )
        .set_index(["instrument", "datetime"])
        .sort_index()
    )


def _decision_keys(work: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    dates = pd.to_datetime(work["__event_time"], utc=True, errors="coerce").dt.tz_convert(
        SESSION_TZ
    ).dt.strftime("%Y-%m-%d")
    keys = work.loc[
        dates.eq(trade_date),
        ["__symbol", "__event_time", "__decision_time"],
    ].drop_duplicates(["__symbol", "__event_time"], keep="last")
    return keys


def _filter_visible_source(
    source: pd.DataFrame,
    keys: pd.DataFrame,
    *,
    symbol_column: str = "symbol",
    timestamp_column: str = "timestamp",
    available_column: str = "available_time",
) -> pd.DataFrame:
    """Attach decision time to event rows and reject observations not yet visible."""
    if source.empty or keys.empty:
        return source.iloc[0:0].copy()
    if available_column not in source:
        raise KeyError(f"PIT source requires {available_column!r}")
    work = source.copy(deep=False)
    work["__symbol"] = work[symbol_column].astype("string").str.upper().str.strip()
    work["__event_time"] = pd.to_datetime(
        work[timestamp_column], utc=True, errors="coerce"
    )
    work["__source_available"] = pd.to_datetime(
        work[available_column], utc=True, errors="coerce"
    )
    work = work.merge(
        keys,
        on=["__symbol", "__event_time"],
        how="inner",
        validate="many_to_one",
        sort=False,
    )
    visible = (
        work["__source_available"].notna()
        & work["__decision_time"].notna()
        & (work["__source_available"] <= work["__decision_time"])
    )
    return work.loc[visible].copy(deep=False)


def _pit_merge_supplements(P: Any, frame: pd.DataFrame, warehouse_root: Path) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = _timing_work(frame)
    for dataset in P.FF.SUPPLEMENT_DATASETS:
        root = warehouse_root / "nvg_supplement" / dataset / "schema=v1"
        for trade_date in _sqlish_dates(work["__event_time"]):
            files = sorted((root / f"date={trade_date}").glob("*.parquet"))
            if not files:
                continue
            supplement = pads.dataset(
                [str(path) for path in files], format="parquet"
            ).to_table().to_pandas()
            if supplement.empty or "symbol" not in supplement or "timestamp" not in supplement:
                continue
            visible = _filter_visible_source(
                supplement,
                _decision_keys(work, trade_date),
            )
            if visible.empty:
                continue
            excluded = {
                "symbol", "timestamp", "available_time", "trade_date", "symbol_id",
                "date", "__symbol", "__event_time", "__source_available", "__decision_time",
            }
            values = [column for column in supplement.columns if column not in excluded]
            if not values:
                continue
            part = visible[["__symbol", "__event_time", *values]].drop_duplicates(
                ["__symbol", "__event_time"], keep="last"
            )
            overlap = [column for column in values if column in work.columns]
            if overlap:
                work = work.drop(columns=overlap)
            work = work.merge(
                part,
                on=["__symbol", "__event_time"],
                how="left",
                validate="one_to_one",
                sort=False,
                copy=False,
            )
    result = _restore_index(work)
    add_aliases = getattr(P.C, "_add_exact_aliases", None)
    return add_aliases(result) if callable(add_aliases) else result


def _pit_merge_sketch(P: Any, frame: pd.DataFrame, warehouse_root: Path) -> pd.DataFrame:
    """Join sketch rows by event time.

    ``trades_1m_sketch`` has no per-row available_time in the published schema.
    The research base loader always includes ``trades_1m_core``; its
    symbol-minute available_time is the max report availability across the
    same eligible trade set and therefore provides the conservative gate for
    sketch features derived from that set.
    """
    if frame.empty:
        return frame
    work = _timing_work(frame)
    root = warehouse_root / "canonical" / "trades_1m_sketch" / "schema=v1"
    for trade_date in _sqlish_dates(work["__event_time"]):
        files = sorted((root / f"date={trade_date}").glob("*.parquet"))
        if not files:
            continue
        sketch = pads.dataset([str(path) for path in files], format="parquet").to_table().to_pandas()
        if sketch.empty or "symbol" not in sketch or "timestamp" not in sketch:
            continue
        sketch["__symbol"] = sketch["symbol"].astype("string").str.upper().str.strip()
        sketch["__event_time"] = pd.to_datetime(sketch["timestamp"], utc=True, errors="coerce")
        keys = _decision_keys(work, trade_date)
        sketch = sketch.merge(
            keys[["__symbol", "__event_time"]],
            on=["__symbol", "__event_time"],
            how="inner",
            validate="many_to_one",
            sort=False,
        )
        excluded = {
            "trade_date", "symbol_id", "symbol", "timestamp", "date",
            "__symbol", "__event_time",
        }
        values = [column for column in sketch.columns if column not in excluded]
        if not values:
            continue
        part = sketch[["__symbol", "__event_time", *values]].drop_duplicates(
            ["__symbol", "__event_time"], keep="last"
        )
        rename = {column: f"trades_1m_sketch__{column}" for column in values}
        part = part.rename(columns=rename)
        overlap = [column for column in rename.values() if column in work.columns]
        if overlap:
            work = work.drop(columns=overlap)
        work = work.merge(
            part,
            on=["__symbol", "__event_time"],
            how="left",
            validate="one_to_one",
            sort=False,
            copy=False,
        )
    return _restore_index(work)


def _pit_merge_condition(P: Any, frame: pd.DataFrame, warehouse_root: Path) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = _timing_work(frame)
    root = warehouse_root / "canonical" / "trades_condition_1m" / "schema=v1"
    for trade_date in _sqlish_dates(work["__event_time"]):
        files = sorted((root / f"date={trade_date}").glob("*.parquet"))
        if not files:
            continue
        condition = pads.dataset([str(path) for path in files], format="parquet").to_table().to_pandas()
        if condition.empty or "symbol" not in condition or "timestamp" not in condition:
            continue
        visible = _filter_visible_source(condition, _decision_keys(work, trade_date))
        if visible.empty:
            continue
        numeric = [
            column
            for column in condition.columns
            if column.endswith("_count") or column.endswith("_volume") or column == "dollar_volume"
        ]
        if not numeric:
            continue
        for column in numeric:
            visible[column] = pd.to_numeric(visible[column], errors="coerce").fillna(0.0)
        agg = (
            visible.groupby(["__symbol", "__event_time"], sort=False)[numeric]
            .sum()
            .reset_index()
            .rename(columns={column: f"trades_condition_1m__{column}" for column in numeric})
        )
        overlap = [column for column in agg.columns if column in work.columns and column not in {"__symbol", "__event_time"}]
        if overlap:
            work = work.drop(columns=overlap)
        work = work.merge(
            agg,
            on=["__symbol", "__event_time"],
            how="left",
            validate="one_to_one",
            sort=False,
            copy=False,
        )
    return _restore_index(work)


def _pit_merge_venue(P: Any, frame: pd.DataFrame, warehouse_root: Path) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = _timing_work(frame)
    root = warehouse_root / "canonical" / "trades_venue_1m" / "schema=v1"
    for trade_date in _sqlish_dates(work["__event_time"]):
        files = sorted((root / f"date={trade_date}").glob("*.parquet"))
        if not files:
            continue
        venue = pads.dataset([str(path) for path in files], format="parquet").to_table().to_pandas()
        if venue.empty or "symbol" not in venue or "timestamp" not in venue:
            continue
        visible = _filter_visible_source(venue, _decision_keys(work, trade_date))
        if visible.empty:
            continue
        visible["volume"] = pd.to_numeric(visible["volume"], errors="coerce").fillna(0.0)
        visible["signed_dollar_flow_proxy"] = pd.to_numeric(
            visible["signed_dollar_flow_proxy"], errors="coerce"
        ).fillna(0.0)
        visible["is_off_exchange"] = visible["is_off_exchange"].fillna(False).astype(bool)
        visible["__venue"] = (
            visible["exchange"].astype("string").fillna("NA")
            + ":"
            + visible.get("trf_id", pd.Series("NA", index=visible.index)).astype("string").fillna("NA")
            + ":"
            + visible["is_off_exchange"].astype(int).astype(str)
        )
        level = (
            visible.groupby(
                ["__symbol", "__event_time", "__venue", "is_off_exchange"],
                sort=False,
            )
            .agg(volume=("volume", "sum"), flow=("signed_dollar_flow_proxy", "sum"))
            .reset_index()
        )
        total = level.groupby(["__symbol", "__event_time"], sort=False)["volume"].transform("sum")
        share = level["volume"] / total.replace(0, np.nan)
        level["hhi"] = share.pow(2)
        level["entropy"] = -(share.where(share > 0) * np.log(share.where(share > 0)))
        level["off_volume"] = level["volume"].where(level["is_off_exchange"], 0.0)
        level["lit_volume"] = level["volume"].where(~level["is_off_exchange"], 0.0)
        level["dark_flow"] = level["flow"].where(level["is_off_exchange"], 0.0)
        level["lit_flow"] = level["flow"].where(~level["is_off_exchange"], 0.0)
        agg = (
            level.groupby(["__symbol", "__event_time"], sort=False)
            .agg(
                off_exchange_volume=("off_volume", "sum"),
                lit_volume=("lit_volume", "sum"),
                dark_signed_flow=("dark_flow", "sum"),
                lit_signed_flow=("lit_flow", "sum"),
                venue_hhi=("hhi", "sum"),
                venue_entropy=("entropy", "sum"),
                dominant_venue_share=("volume", lambda values: np.nan),
                venue_count=("__venue", "nunique"),
            )
            .reset_index()
        )
        # Compute dominant share without a Python group callback in the hot path.
        dominant = (
            level.assign(__share=share)
            .groupby(["__symbol", "__event_time"], sort=False)["__share"]
            .max()
            .rename("dominant_venue_share")
            .reset_index()
        )
        agg = agg.drop(columns=["dominant_venue_share"]).merge(
            dominant, on=["__symbol", "__event_time"], how="left", validate="one_to_one"
        )
        total_volume = agg["off_exchange_volume"] + agg["lit_volume"]
        agg["off_exchange_share"] = agg["off_exchange_volume"] / total_volume.replace(0, np.nan)
        agg["dark_lit_divergence"] = agg["dark_signed_flow"] - agg["lit_signed_flow"]
        overlap = [
            column for column in agg.columns
            if column in work.columns and column not in {"__symbol", "__event_time"}
        ]
        if overlap:
            work = work.drop(columns=overlap)
        work = work.merge(
            agg,
            on=["__symbol", "__event_time"],
            how="left",
            validate="one_to_one",
            sort=False,
            copy=False,
        )
    return _restore_index(work)


def _load_execution_frame(P: Any, decision_index: pd.MultiIndex) -> pd.DataFrame:
    decision_times = pd.Series(
        pd.to_datetime(decision_index.get_level_values("datetime"), utc=True, errors="coerce")
    )
    dates = _sqlish_dates(decision_times)
    bars_root = P.R.WAREHOUSE_ROOT / "canonical" / "bars_1m" / "schema=v1"
    trades_root = P.R.WAREHOUSE_ROOT / "canonical" / "trades_1m_core" / "schema=v1"
    bar_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    for trade_date in dates:
        bar_files = sorted((bars_root / f"date={trade_date}").glob("*.parquet"))
        if bar_files:
            dataset = pads.dataset([str(path) for path in bar_files], format="parquet")
            available = set(dataset.schema.names)
            wanted = [
                column
                for column in (
                    "symbol", "timestamp", "open", "high", "low", "close", "volume",
                    "dollar_volume", "vwap",
                )
                if column in available
            ]
            bar_parts.append(dataset.to_table(columns=wanted).to_pandas())
        trade_files = sorted((trades_root / f"date={trade_date}").glob("*.parquet"))
        if trade_files:
            dataset = pads.dataset([str(path) for path in trade_files], format="parquet")
            available = set(dataset.schema.names)
            wanted = [column for column in ("symbol", "timestamp", "trade_count") if column in available]
            if {"symbol", "timestamp"}.issubset(wanted):
                trade_parts.append(dataset.to_table(columns=wanted).to_pandas())
    if not bar_parts:
        raise RuntimeError("decision-clock labels could not load canonical bars")
    bars = pd.concat(bar_parts, ignore_index=True, copy=False)
    bars["instrument"] = bars["symbol"].astype("string").str.upper().str.strip()
    bars["datetime"] = (
        pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    bars = bars.dropna(subset=["instrument", "datetime"]).drop_duplicates(
        ["instrument", "datetime"], keep="last"
    )
    rename = {
        column: f"bars_1m__{column}"
        for column in ("open", "high", "low", "close", "volume", "dollar_volume", "vwap")
        if column in bars
    }
    execution = bars[["instrument", "datetime", *rename]].rename(columns=rename).set_index(
        ["instrument", "datetime"]
    ).sort_index()
    if trade_parts:
        trades = pd.concat(trade_parts, ignore_index=True, copy=False)
        trades["instrument"] = trades["symbol"].astype("string").str.upper().str.strip()
        trades["datetime"] = (
            pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )
        trades = trades.dropna(subset=["instrument", "datetime"]).drop_duplicates(
            ["instrument", "datetime"], keep="last"
        )
        if "trade_count" in trades:
            trade_count = trades[["instrument", "datetime", "trade_count"]].rename(
                columns={"trade_count": "trades_1m_core__trade_count"}
            ).set_index(["instrument", "datetime"])
            execution = execution.join(trade_count, how="left")
    return execution


def _decision_clock_labels(P: Any, original_builder: Any, features: pd.DataFrame, horizons: list[int]):
    execution = _load_execution_frame(P, features.index)
    labels, masks = original_builder(execution, horizons)
    labels = labels.reindex(features.index)
    mask_frame = pd.DataFrame(masks, index=execution.index).reindex(features.index)
    masks_out = {
        column: mask_frame[column].fillna(False).astype("boolean")
        for column in mask_frame.columns
    }
    audit = dict(getattr(P.V26, "LAST_LABEL_AUDIT", {}) or {})
    audit.update(
        {
            "pit_clock_version": PIT_CLOCK_VERSION,
            "label_anchor": "decision_time",
            "entry_offset_minutes": 1,
            "execution_source": "canonical bars_1m + trades_1m_core",
            "feature_decision_rows": int(len(features)),
            "execution_market_rows": int(len(execution)),
            "label_non_null_after_decision_reindex": {
                str(column): int(labels[column].notna().sum()) for column in labels
            },
        }
    )
    P.V26.LAST_LABEL_AUDIT = audit
    del execution, mask_frame
    gc.collect()
    return labels, masks_out


def _install_role_gate(P: Any) -> None:
    globals_dict = getattr(P.select_candidates, "__globals__", {})
    original = globals_dict.get("_track_summary")
    if not callable(original) or getattr(original, "_pit_role_gate", False):
        return

    def role_aware(data: pd.DataFrame, track: Mapping[str, Any], universe: str) -> pd.DataFrame:
        summary = original(data, track, universe)
        if summary.empty:
            return summary
        registry = getattr(P.V26, "SPEC_REGISTRY", pd.DataFrame())
        if registry.empty or "factor_id" not in registry:
            raise RuntimeError("factor role registry unavailable during v2.8 selection")
        role_map = registry.set_index("factor_id")["role"].astype(str).to_dict()
        neutral_map = (
            registry.set_index("factor_id")["neutralization_allowed"].astype(bool).to_dict()
            if "neutralization_allowed" in registry
            else {}
        )
        name = str(track.get("name"))
        allowed = set(ROLE_POLICY.get(name, ()))
        summary = summary.copy()
        summary["factor_role"] = summary["feature"].map(role_map).fillna("UNREGISTERED")
        summary["neutralization_allowed"] = summary["feature"].map(neutral_map).fillna(False)
        keep = summary["factor_role"].isin(allowed)
        if name == "alpha":
            keep &= summary["neutralization_allowed"]
        return summary.loc[keep].reset_index(drop=True)

    role_aware._pit_role_gate = True  # type: ignore[attr-defined]
    globals_dict["_track_summary"] = role_aware


def _install_runtime(P: Any, config: Mapping[str, Any]) -> None:
    if getattr(P, "_PIT_CLOCK_RUNTIME_INSTALLED", False):
        return
    base_loader = P.R.NFFDataLoader

    class TimingNFFDataLoader(base_loader):
        """NFF loader variant that preserves event/availability timing metadata."""

        def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
            start = P.R._utc_timestamp(start_time) if hasattr(P.R, "_utc_timestamp") else None
            end = P.R._utc_timestamp(end_time) if hasattr(P.R, "_utc_timestamp") else None
            # The adapter helpers live in qlib.contrib.data.nff, not the research module.
            if start is None and start_time is not None:
                start = pd.Timestamp(start_time)
                start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
            if end is None and end_time is not None:
                end = pd.Timestamp(end_time)
                end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
            if start is not None and end is not None and start > end:
                raise ValueError("start_time must not be after end_time")
            selected_instruments = self._normalise_instruments(instruments)
            source_frames: list[pd.DataFrame] = []
            source_reports: list[dict[str, Any]] = []
            for spec in self.sources:
                source_frame, report = self._read_source(
                    spec, start, end, selected_instruments, future_sessions=0
                )
                source_frames.append(source_frame)
                source_reports.append(report)
            merged = self._merge_sources(source_frames)
            merged, collision_rows = self._apply_execution_clock(merged)
            merged, label_report = self._attach_label(merged, start, end, selected_instruments)
            if merged.empty:
                empty_index = pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])
                self.last_load_report = {"rows": 0, "sources": source_reports, "label": label_report}
                return pd.DataFrame(index=empty_index)
            if start is not None:
                merged = merged[merged["datetime"] >= start]
            if end is not None:
                merged = merged[merged["datetime"] <= end]
            if selected_instruments:
                merged = merged[merged["instrument"].isin(selected_instruments)]
            metadata = {
                "symbol_id", "timestamp", "event_time", "available_time", "datetime", "instrument"
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
            event_ns = pd.to_datetime(merged["event_time"], utc=True, errors="coerce").astype("int64")
            available_ns = pd.to_datetime(merged["available_time"], utc=True, errors="coerce").astype("int64")
            decision_utc = pd.to_datetime(merged["datetime"], utc=True, errors="coerce")
            decision_ns = decision_utc.astype("int64")
            merged["datetime"] = decision_utc.dt.tz_convert("UTC").dt.tz_localize(None)
            merged = merged.sort_values(["datetime", "instrument"], kind="mergesort")
            # Recompute the timing arrays after the same stable sort.
            sort_indexer = merged.index
            event_ns = event_ns.loc[sort_indexer]
            available_ns = available_ns.loc[sort_indexer]
            decision_ns = decision_ns.loc[sort_indexer]
            index = pd.MultiIndex.from_frame(merged[["datetime", "instrument"]])
            index.names = ["datetime", "instrument"]
            feature_block = merged[feature_columns].copy()
            feature_block[TIMING_EVENT] = event_ns.to_numpy(dtype="int64")
            feature_block[TIMING_AVAILABLE] = available_ns.to_numpy(dtype="int64")
            feature_block[TIMING_DECISION] = decision_ns.to_numpy(dtype="int64")
            feature_block.index = index
            blocks: dict[str, pd.DataFrame] = {"feature": feature_block}
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
                "pit_timing_columns": [TIMING_EVENT, TIMING_AVAILABLE, TIMING_DECISION],
                "pit_clock_version": PIT_CLOCK_VERSION,
                "start_time": None if start is None else start.isoformat(),
                "end_time": None if end is None else end.isoformat(),
                "execution": {
                    "frequency": self.execution.frequency,
                    "delay_bars": self.execution.delay_bars,
                    "collision_policy": self.execution.collision_policy,
                    "collision_rows": collision_rows,
                    "qlib_datetime": "UTC-naive decision_time",
                },
                "sources": source_reports,
                "label": label_report,
            }
            return result

    P.R.NFFDataLoader = TimingNFFDataLoader
    # Preserve the post-v2.7 exact alias behavior while replacing only the clock join.
    P.FF.merge_supplements = lambda frame, root: _pit_merge_supplements(P, frame, root)
    P.FF.merge_canonical_sketch = lambda frame, root: _pit_merge_sketch(P, frame, root)
    P.FF.merge_condition_aggregates = lambda frame, root: _pit_merge_condition(P, frame, root)
    P.FF.merge_venue_aggregates = lambda frame, root: _pit_merge_venue(P, frame, root)

    current_derive = P.FF.derive_prototype

    def derive_with_venue_semantics(frame: pd.DataFrame, prototype_id: str, window: str):
        value = current_derive(frame, prototype_id, window)
        if prototype_id == "J__ALL__" and isinstance(value, dict):
            direct = P.FF._column(frame, "dominant_venue_share")
            if direct is not None:
                value = dict(value)
                value["J08"] = direct
        elif prototype_id == "J08":
            direct = P.FF._column(frame, "dominant_venue_share")
            if direct is not None:
                return pd.to_numeric(direct, errors="coerce").astype("float32")
        return value

    P.FF.derive_prototype = derive_with_venue_semantics

    original_builder = P.R.build_labels_and_masks
    P.R.build_labels_and_masks = lambda features, horizons: _decision_clock_labels(
        P, original_builder, features, list(horizons)
    )
    P._PIT_CLOCK_RUNTIME_INSTALLED = True


def install(P: Any) -> None:
    """Install contract, selection-role and runtime PIT corrections."""
    original_contract_hash = P._contract_hash

    def pit_contract_hash(config: Mapping[str, Any]) -> str:
        return P._json_hash(
            {
                "base_contract": original_contract_hash(config),
                "pit_clock_version": PIT_CLOCK_VERSION,
                "manual_join_key": "symbol+event_time",
                "availability_gate": "source_available_time<=decision_time",
                "label_anchor": "decision_time+1m exact canonical market bars",
                "selection_role_policy": ROLE_POLICY,
                "off_exchange_semantics": "TRF/off-exchange proxy; not asserted to be ATS dark-pool flow",
            }
        )

    P._contract_hash = pit_contract_hash
    original_bootstrap = P.bootstrap

    def bootstrap_with_pit(config: dict[str, Any]) -> None:
        original_bootstrap(config)
        _install_runtime(P, config)

    P.bootstrap = bootstrap_with_pit
    _install_role_gate(P)
