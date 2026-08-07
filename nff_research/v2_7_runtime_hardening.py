from __future__ import annotations

"""Runtime field and factor resolution audit for v2.7."""

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


FIELD_RESOLUTION: dict[str, dict[str, Any]] = defaultdict(
    lambda: {"calls": 0, "resolved": None, "resolver": None, "non_null_rate": 0.0}
)


def _candidate_names(campaign: Any, name: str) -> list[str]:
    candidates = [name]
    for prefix in ("minute_nvg__", "trade_nvg__", "hawkes_lite__", "hawkes_derived__"):
        if name.startswith(prefix):
            candidates.append(name[len(prefix) :])
    bare = candidates[-1]
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


def _groups(series: pd.Series) -> pd.Index:
    return series.index.get_level_values("instrument")


def _path_metrics(series: pd.Series, window: int) -> dict[str, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    groups = _groups(numeric)
    grouped = numeric.groupby(groups, sort=False, group_keys=False)
    start = grouped.shift(window - 1)
    signed_change = numeric - start
    difference = grouped.diff().abs()
    path_length = difference.groupby(groups, sort=False, group_keys=False).transform(
        lambda value: value.rolling(window, min_periods=window - 1).sum()
    )
    rolling_min = grouped.transform(
        lambda value: value.rolling(window, min_periods=window).min()
    )
    rolling_max = grouped.transform(
        lambda value: value.rolling(window, min_periods=window).max()
    )
    range_value = rolling_max - rolling_min
    efficiency = signed_change.abs() / (path_length + 1e-8)
    roughness = path_length / (signed_change.abs() + 1e-8)
    terminal_position = (numeric - rolling_min) / (range_value + 1e-8)
    return {
        "signed_change": signed_change.astype("float32"),
        "efficiency": efficiency.astype("float32"),
        "roughness": roughness.astype("float32"),
        "range": range_value.astype("float32"),
        "terminal_position": terminal_position.astype("float32"),
    }


def _enrich_causal_paths(frame: pd.DataFrame, campaign: Any) -> pd.DataFrame:
    generated: dict[str, pd.Series] = {}
    close = frame.get("bars_1m__close")
    dollar = frame.get("bars_1m__dollar_volume")
    for window_text in sorted(
        {
            value
            for family in ("A", "B", "C", "D")
            for value in campaign.RUNTIME_WINDOWS.get(family, ())
            if str(value).endswith("m")
        }
    ):
        window = int(str(window_text)[:-1])
        if close is not None:
            price = np.log(pd.to_numeric(close, errors="coerce").replace(0, np.nan))
            for metric, value in _path_metrics(price, window).items():
                target = f"minute_nvg__price_path_{window_text}_{metric}"
                if target not in frame:
                    generated[target] = value
        if dollar is not None:
            volume = np.log1p(pd.to_numeric(dollar, errors="coerce").clip(lower=0))
            for metric, value in _path_metrics(volume, window).items():
                target = f"minute_nvg__volume_path_{window_text}_{metric}"
                if target not in frame:
                    generated[target] = value
    if not generated:
        return frame
    block = pd.concat(generated, axis=1, copy=False)
    block.columns = list(generated)
    return pd.concat([frame, block], axis=1, copy=False)


def _write_resolution(campaign: Any, result: pd.DataFrame) -> None:
    if campaign.CTX is None:
        return
    root = campaign.CTX.root / "schema"
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {"requested_field": name, **record}
        for name, record in sorted(FIELD_RESOLUTION.items())
    ]
    frame = pd.DataFrame(rows)
    frame.to_parquet(root / "field_resolution.parquet", index=False)
    frame.to_csv(root / "field_resolution.csv", index=False)

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
        candidates = _candidate_names(campaign, name)
        selected = next((candidate for candidate in candidates if candidate in frame), None)
        try:
            value = base_column(frame, name)
        except KeyError as exc:
            # `_column_exact` normally delegates to ORIGINAL["column"], which
            # is populated by the worker install path. Formula/unit consumers
            # can legitimately call the final resolver before `main()`; in
            # that case use the underlying exact/suffix resolver rather than
            # raising a lifecycle-dependent KeyError.
            if exc.args != ("column",):
                raise
            value = campaign.FF._column(frame, name)
            if value is not None:
                record["resolver"] = "preinstall_factor_engine_fallback"
        if selected is not None:
            record["resolved"] = selected
            record["resolver"] = "exact_or_documented_alias"
        elif value is not None:
            record["resolved"] = (
                "factor_engine_fallback"
                if record.get("resolver") == "preinstall_factor_engine_fallback"
                else "legacy_suffix_resolver"
            )
            record["resolver"] = record.get("resolver") or "legacy_suffix_resolver"
        if value is not None:
            value = pd.to_numeric(value, errors="coerce").astype("float32")
            record["non_null_rate"] = float(value.notna().mean())
        return value

    def add_all(frame: pd.DataFrame) -> pd.DataFrame:
        FIELD_RESOLUTION.clear()
        enriched = _enrich_causal_paths(frame, campaign)
        result = base_add_all(enriched)
        _write_resolution(campaign, result)
        return result

    campaign._column_exact = tracked_column
    campaign._add_all_features = add_all
