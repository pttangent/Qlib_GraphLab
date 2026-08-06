from __future__ import annotations

"""Runtime hardening shared by parent and detached v2.7 date workers."""

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-8
FIELD_RESOLUTION: dict[str, dict[str, Any]] = defaultdict(
    lambda: {"calls": 0, "resolved": None, "resolver": None, "non_null_rate": None}
)


def _shift(series: pd.Series, periods: int, instruments: pd.Index) -> pd.Series:
    return series.groupby(instruments, sort=False, group_keys=False).shift(periods)


def _rolling(series: pd.Series, window: int, operation: str, minimum: int) -> pd.Series:
    result = getattr(
        series.groupby(level="instrument", sort=False).rolling(window, min_periods=minimum),
        operation,
    )()
    return result.reset_index(level=0, drop=True).reindex(series.index)


def _path_metrics(values: pd.Series, window: int) -> dict[str, pd.Series]:
    instruments = values.index.get_level_values("instrument")
    change = values - _shift(values, window - 1, instruments)
    step = values.groupby(instruments, sort=False, group_keys=False).diff().abs()
    path_length = _rolling(step, window - 1, "sum", window - 1)
    high = _rolling(values, window, "max", window)
    low = _rolling(values, window, "min", window)
    value_range = high - low
    return {
        "signed_change": change,
        "efficiency": change.abs() / (path_length + EPS),
        "roughness": path_length / (change.abs() + EPS),
        "range": value_range,
        "terminal_position": (values - low) / (value_range + EPS),
    }


def _enrich_causal_paths(frame: pd.DataFrame, campaign: Any) -> pd.DataFrame:
    generated: dict[str, pd.Series] = {}
    windows = sorted(
        {
            int(str(window).removesuffix("m"))
            for window in campaign.RUNTIME_WINDOWS.get("A", ())
            if str(window).endswith("m")
        }
    )
    close = frame.get("bars_1m__close")
    dollar = frame.get("bars_1m__dollar_volume")
    if close is not None:
        log_close = np.log(pd.to_numeric(close, errors="coerce").replace(0, np.nan))
        for window in windows:
            for metric, value in _path_metrics(log_close, window).items():
                generated[f"minute_nvg__price_path_{window}m_{metric}"] = value.astype("float32")
            generated[f"traditional__momentum_{window}m"] = _path_metrics(log_close, window)[
                "signed_change"
            ].astype("float32")
    if dollar is not None:
        log_dollar = np.log1p(pd.to_numeric(dollar, errors="coerce").clip(lower=0))
        for window in windows:
            for metric, value in _path_metrics(log_dollar, window).items():
                generated[f"minute_nvg__volume_path_{window}m_{metric}"] = value.astype("float32")

    active = frame.get("trade_nvg__active_second_ratio_60s")
    if active is None:
        active = frame.get("trade_nvg__trade_active_second_ratio_60s")
    if active is not None:
        active = pd.to_numeric(active, errors="coerce")
        for seconds, minutes in ((60, 1), (180, 3), (300, 5)):
            ratio = active if minutes == 1 else _rolling(active, minutes, "mean", 1)
            generated[f"trade_nvg__trade_active_second_ratio_{seconds}s"] = ratio.astype("float32")
            generated[f"trade_nvg__trade_price_stale_ratio_{seconds}s"] = (1.0 - ratio).astype("float32")
            observed = active.notna().astype(float)
            coverage = observed if minutes == 1 else _rolling(observed, minutes, "mean", 1)
            generated[f"trade_nvg__trade_observation_coverage_{seconds}s"] = coverage.astype("float32")

    if not generated:
        return frame
    block = pd.concat(generated, axis=1, copy=False)
    block.columns = list(generated)
    missing = [column for column in block if column not in frame]
    return pd.concat([frame, block[missing]], axis=1, copy=False) if missing else frame


def _candidate_names(campaign: Any, name: str) -> list[str]:
    candidates = [name, campaign._strip_namespace(name)]
    bare = campaign._strip_namespace(name)
    aliases = {
        "_top_terminal_slope_mean": "_top_terminal_signed_mean_slope",
        "_bottom_terminal_slope_mean": "_bottom_terminal_signed_mean_slope",
        "_top_terminal_value_gap_mean": "_top_terminal_value_delta_mean",
        "_bottom_terminal_value_gap_mean": "_bottom_terminal_value_delta_mean",
    }
    for old, new in aliases.items():
        if bare.endswith(old):
            candidates.append(bare[: -len(old)] + new)
    if "_detrended_top_bottom_asymmetry" in bare:
        candidates.append(
            bare.replace("price_nvg_", "price_detrended_nvg_").replace(
                "_detrended_top_bottom_asymmetry", "_terminal_signed_edge_balance"
            )
        )
    return list(dict.fromkeys(candidates))


def _write_resolution(campaign: Any, result: pd.DataFrame) -> None:
    context = campaign.CTX
    if context is None:
        return
    root = context.root / "schema"
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {"requested_field": name, **record}
        for name, record in sorted(FIELD_RESOLUTION.items())
    ]
    pd.DataFrame(rows).to_parquet(root / "field_resolution.parquet", index=False)
    pd.DataFrame(rows).to_csv(root / "field_resolution.csv", index=False)

    registry = campaign.V26.SPEC_REGISTRY.copy()
    runtime = campaign.V26.RUNTIME_FACTOR_STATUS
    registry["runtime_status"] = registry["factor_id"].map(
        lambda factor: runtime.get(factor, {}).get("status", "NOT_MATERIALIZED")
    )
    registry["non_null_rate"] = registry["factor_id"].map(
        lambda factor: runtime.get(factor, {}).get("non_null_rate", 0.0)
    )
    registry["column_present"] = registry["factor_id"].isin(result.columns)
    registry.to_parquet(root / "factor_resolution.parquet", index=False)
    registry.to_csv(root / "factor_resolution.csv", index=False)
    summary = {
        "factor_specs": int(len(registry)),
        "successful": int(registry["runtime_status"].eq("SUCCESS").sum()),
        "low_coverage": int(registry["runtime_status"].eq("LOW_COVERAGE").sum()),
        "unavailable": int(registry["runtime_status"].str.contains("UNAVAILABLE", na=False).sum()),
        "not_materialized": int(registry["runtime_status"].eq("NOT_MATERIALIZED").sum()),
        "field_requests": len(rows),
        "unresolved_field_requests": int(sum(row.get("resolved") is None for row in rows)),
    }
    (root / "factor_resolution_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def install(campaign: Any) -> None:
    base_column = campaign._column_exact
    base_add_all = campaign._add_all_features

    def tracked_column(frame: pd.DataFrame, name: str):
        record = FIELD_RESOLUTION[name]
        record["calls"] += 1
        selected = next((candidate for candidate in _candidate_names(campaign, name) if candidate in frame), None)
        value = base_column(frame, name)
        if selected is not None:
            record["resolved"] = selected
            record["resolver"] = "exact_or_documented_alias"
        elif value is not None:
            record["resolved"] = "legacy_suffix_resolver"
            record["resolver"] = "legacy_suffix_resolver"
        if value is not None:
            record["non_null_rate"] = float(pd.to_numeric(value, errors="coerce").notna().mean())
        return value

    def add_all(frame: pd.DataFrame) -> pd.DataFrame:
        FIELD_RESOLUTION.clear()
        enriched = _enrich_causal_paths(frame, campaign)
        result = base_add_all(enriched)
        _write_resolution(campaign, result)
        return result

    campaign._column_exact = tracked_column
    campaign._add_all_features = add_all
