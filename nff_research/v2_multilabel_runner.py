from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from qlib.contrib.data.nff import NFFDataLoader, NFFWarehouseCatalog


WAREHOUSE_ROOT = Path(r"D:\DEV\AnotherNetworkFactory\warehouses\NFF_warehouse")
RESEARCH_ROOT = Path(r"D:\DEV\AnotherNetworkFactory\warehouses\NFF_research")

KEY_COLUMNS = {"trade_date", "symbol_id", "symbol", "timestamp", "available_time", "schema", "date"}
BAR_COLUMNS = ["open", "close", "dollar_volume", "vwap"]
TRADES_CORE_COLUMNS = [
    "trade_count",
    "dollar_volume",
    "large_trade_volume",
    "lit_trade_count",
    "lit_volume",
    "off_exchange_volume",
    "buy_volume_proxy",
    "sell_volume_proxy",
    "signed_dollar_flow_proxy",
    "report_lag_p90_ns",
]

MINUTE_FEATURE_RE = re.compile(
    r"(momentum_\d+m|reversal_\d+m|price_path_\d+m_(efficiency|roughness|terminal_position|signed_change|range)|"
    r"price_nvg_\d+m_top_bottom_asymmetry|price_volume_(terminal_overlap|nvg_confirmation)_\d+m|"
    r"price_nvg_\d+m_(overextension|detrended)_change_1m|price_path_efficiency_\d+m_change_1m)"
)
TRADE_FEATURE_RE = re.compile(
    r"(trade_active_second_(ratio|count)_\d+s|trade_observation_coverage_\d+s|trade_count_\d+s|"
    r"trade_price_stale_ratio_\d+s|trade_flow_path_\d+s_(efficiency|roughness|terminal_position|signed_change|range)|"
    r"trade_price_path_\d+s_(efficiency|roughness|terminal_position|signed_change|range)|"
    r"trade_price_flow_terminal_overlap_\d+s|trade_(price|flow)_nvg_\d+s_top_bottom_asymmetry_change_1m|"
    r"trade_active_second_ratio_\d+s_change_1m)"
)
HAWKES_COLUMNS = [
    "hawkes_ready",
    "hawkes_warmup_fraction",
    "hawkes_total_intensity",
    "hawkes_total_baseline",
    "hawkes_intensity_imbalance",
    "hawkes_endogenous_share",
    "hawkes_self_excitation_share",
    "hawkes_cross_excitation_share",
    "hawkes_branching_ratio_max",
    "hawkes_short_excitation_share",
    "hawkes_medium_excitation_share",
    "hawkes_long_excitation_share",
    "hawkes_flow_surprise",
    "hawkes_total_surprise",
    "hawkes_surprise_energy",
    "hawkes_total_intensity_mean_60s",
    "hawkes_total_intensity_mean_300s",
    "hawkes_endogenous_share_mean_60s",
    "hawkes_endogenous_share_mean_180s",
    "hawkes_endogenous_share_mean_300s",
    "hawkes_shock_score_60s",
    "hawkes_shock_score_180s",
    "hawkes_shock_score_300s",
]
TRAD_WINDOWS = (15, 30, 60, 120)
LABEL_NAMES = (
    "return_open_to_open",
    "liquidity_deterioration",
    "realized_volatility",
    "jump_risk",
    "execution_cost_proxy",
)
ANALYSIS_FEATURE_RE = re.compile(
    r"^(signal__|traditional__(momentum|reversal|realized_vol|dollar_volume_change|vwap_dislocation|log_dollar_volume)|"
    r"bars_1m__(close|dollar_volume|vwap)|trades_1m_core__(trade_count|dollar_volume|signed_dollar_flow_proxy|"
    r"buy_volume_proxy|sell_volume_proxy|off_exchange_volume|lit_volume|large_trade_volume|report_lag_p90_ns)|"
    r"minute_nvg__(momentum|reversal)_(10|15|30)m|"
    r"minute_nvg__price_nvg_(10|15|30)m_top_bottom_asymmetry|"
    r"minute_nvg__price_path_(10|15|30)m_(efficiency|terminal_position|signed_change|range)|"
    r"minute_nvg__price_volume_(terminal_overlap|nvg_confirmation)_(10|15|30)m|"
    r"trade_nvg__trade_(active_second_ratio|active_second_count|observation_coverage|count|price_stale_ratio)_(60|180|300)s|"
    r"trade_nvg__trade_flow_path_(60|180|300)s_(efficiency|terminal_position|signed_change|range)|"
    r"trade_nvg__trade_price_path_(60|180|300)s_(efficiency|terminal_position|signed_change|range)|"
    r"trade_nvg__trade_price_flow_terminal_overlap_(60|180|300)s|"
    r"hawkes_lite__(hawkes_ready|hawkes_warmup_fraction|hawkes_total_intensity|hawkes_total_baseline|"
    r"hawkes_intensity_imbalance|hawkes_endogenous_share|hawkes_cross_excitation_share|hawkes_branching_ratio_max|"
    r"hawkes_long_excitation_share|hawkes_flow_surprise|hawkes_surprise_energy|hawkes_shock_score_(60|180|300)s)|"
    r"hawkes_derived__)"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def read_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_status(path: Path, **updates: Any) -> None:
    status = read_status(path)
    status.update(updates)
    status["heartbeat_utc"] = utc_now()
    atomic_write_json(path, status)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, default=str, ensure_ascii=False) + "\n")


def load_dates(start_date: str, end_date: str) -> list[str]:
    catalog = NFFWarehouseCatalog(WAREHOUSE_ROOT)
    dates = catalog.available_dates("feature", "minute_nvg", "v3")
    return [date for date in dates if start_date <= date <= end_date]


def source_columns(kind: str, dataset: str, schema: str | None = None) -> list[str]:
    catalog = NFFWarehouseCatalog(WAREHOUSE_ROOT)
    return [column for column in catalog.columns(kind, dataset, schema) if column not in KEY_COLUMNS]


def selected_nff_columns(dataset: str) -> list[str]:
    columns = source_columns("feature", dataset, "v3")
    if dataset == "hawkes_lite":
        available = set(columns)
        return [column for column in HAWKES_COLUMNS if column in available]
    if dataset == "minute_nvg":
        return [column for column in columns if MINUTE_FEATURE_RE.search(column)]
    if dataset == "trade_nvg":
        return [column for column in columns if TRADE_FEATURE_RE.search(column)]
    return columns


def feature_sets() -> dict[str, dict[str, Any]]:
    return {
        "minute_nvg": {"schema_version": "v3", "columns": selected_nff_columns("minute_nvg")},
        "trade_nvg": {"schema_version": "v3", "columns": selected_nff_columns("trade_nvg")},
        "hawkes_lite": {"schema_version": "v3", "columns": selected_nff_columns("hawkes_lite")},
    }


def canonical_sets() -> dict[str, dict[str, Any]]:
    available_core = set(source_columns("canonical", "trades_1m_core", "v1"))
    return {
        "bars_1m": {"schema_version": "v1", "columns": BAR_COLUMNS},
        "trades_1m_core": {
            "schema_version": "v1",
            "columns": [column for column in TRADES_CORE_COLUMNS if column in available_core],
        },
    }


def add_hawkes_derived(features: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-12
    p = "hawkes_lite__"
    required = [
        "hawkes_total_intensity",
        "hawkes_total_baseline",
        "hawkes_intensity_imbalance",
        "hawkes_endogenous_share",
        "hawkes_cross_excitation_share",
        "hawkes_branching_ratio_max",
        "hawkes_ready",
        "hawkes_short_excitation_share",
        "hawkes_medium_excitation_share",
        "hawkes_long_excitation_share",
        "hawkes_total_intensity_mean_60s",
        "hawkes_total_intensity_mean_300s",
        "hawkes_endogenous_share_mean_60s",
        "hawkes_endogenous_share_mean_300s",
        "hawkes_shock_score_60s",
        "hawkes_shock_score_180s",
        "hawkes_shock_score_300s",
        "hawkes_endogenous_share_mean_180s",
    ]
    if any(p + column not in features.columns for column in required):
        return features
    result = features.copy()
    intensity = result[p + "hawkes_total_intensity"].astype(float)
    baseline = result[p + "hawkes_total_baseline"].astype(float)
    imbalance = result[p + "hawkes_intensity_imbalance"].astype(float).clip(-1, 1)
    endogenous = result[p + "hawkes_endogenous_share"].astype(float).clip(0, 1)
    cross = result[p + "hawkes_cross_excitation_share"].astype(float).clip(0, 1)
    branching = result[p + "hawkes_branching_ratio_max"].astype(float).clip(0, 1)
    ready = result[p + "hawkes_ready"].astype(float).clip(0, 1)
    excess = np.log((intensity + eps) / (baseline + eps)).clip(-5, 5)
    positive = excess.clip(lower=0)

    result["hawkes_derived__hawkes_excess_intensity"] = excess
    result["hawkes_derived__hawkes_signed_pressure"] = imbalance * positive
    result["hawkes_derived__hawkes_pressure_strength"] = imbalance.abs() * positive
    result["hawkes_derived__hawkes_persistence"] = ready * endogenous * branching
    result["hawkes_derived__hawkes_cross_reaction"] = cross * positive
    duration = (
        2.0 * result[p + "hawkes_short_excitation_share"].astype(float)
        + 10.0 * result[p + "hawkes_medium_excitation_share"].astype(float)
        + 60.0 * result[p + "hawkes_long_excitation_share"].astype(float)
    )
    result["hawkes_derived__hawkes_effective_duration"] = duration.clip(0, 60)
    result["hawkes_derived__hawkes_effective_duration_norm"] = (duration / 60.0).clip(0, 1)

    for window in (60, 180, 300):
        shock = result[p + f"hawkes_shock_score_{window}s"].astype(float).clip(lower=0)
        endo = result[p + f"hawkes_endogenous_share_mean_{window}s"].astype(float).clip(0, 1)
        result[f"hawkes_derived__hawkes_exogenous_shock_{window}s"] = shock * (1 - endo)
        result[f"hawkes_derived__hawkes_endogenous_shock_{window}s"] = shock * endo

    result["hawkes_derived__hawkes_intensity_regime_change"] = (
        np.log(result[p + "hawkes_total_intensity_mean_60s"].astype(float) + eps)
        - np.log(result[p + "hawkes_total_intensity_mean_300s"].astype(float) + eps)
    )
    result["hawkes_derived__hawkes_endogeneity_regime_change"] = (
        result[p + "hawkes_endogenous_share_mean_60s"].astype(float)
        - result[p + "hawkes_endogenous_share_mean_300s"].astype(float)
    )
    result["hawkes_derived__hawkes_shock_regime_change"] = (
        np.log1p(result[p + "hawkes_shock_score_60s"].astype(float).clip(lower=0))
        - np.log1p(result[p + "hawkes_shock_score_300s"].astype(float).clip(lower=0))
    )
    return result


def _future_roll(series: pd.Series, window: int, op: str) -> pd.Series:
    shifted = series.shift(-1)
    reversed_series = shifted.iloc[::-1]
    min_periods = max(3, min(window, window // 3))
    roller = reversed_series.rolling(window, min_periods=min_periods)
    if op == "sum":
        out = roller.sum()
    elif op == "mean":
        out = roller.mean()
    elif op == "max":
        out = roller.max()
    else:
        raise ValueError(op)
    return out.iloc[::-1]


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    min_periods = max(5, min(window, window // 3))
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def add_traditional_factors(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    group = result.groupby(level="instrument", sort=False, group_keys=False)
    close = result["bars_1m__close"].astype(float).replace(0, np.nan)
    vwap = result["bars_1m__vwap"].astype(float).replace(0, np.nan)
    dollar_volume = result["bars_1m__dollar_volume"].astype(float).clip(lower=0)
    ret1 = group["bars_1m__close"].pct_change(fill_method=None)
    result["traditional__vwap_dislocation"] = vwap / close - 1.0
    result["traditional__log_dollar_volume"] = np.log1p(dollar_volume)
    for window in TRAD_WINDOWS:
        momentum = group["bars_1m__close"].pct_change(periods=window, fill_method=None)
        result[f"traditional__momentum_{window}m"] = momentum
        result[f"traditional__reversal_{window}m"] = -momentum
        result[f"traditional__dollar_volume_change_{window}m"] = group["bars_1m__dollar_volume"].pct_change(
            periods=window, fill_method=None
        )
        result[f"traditional__realized_vol_{window}m"] = (
            ret1.groupby(level="instrument", sort=False, group_keys=False)
            .transform(lambda s, w=window: np.sqrt((s.astype(float) ** 2).rolling(w, min_periods=max(3, min(w, w // 3))).sum()))
        )
    return result


def _cs_zscore(series: pd.Series) -> pd.Series:
    grouped = series.groupby(level="datetime", sort=False)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return (series - mean) / std


def add_research_signals(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    sign_sources = {
        "signal__neg_minute_nvg_price_nvg_30m_top_bottom_asymmetry": "minute_nvg__price_nvg_30m_top_bottom_asymmetry",
        "signal__neg_trade_flow_path_300s_terminal_position": "trade_nvg__trade_flow_path_300s_terminal_position",
        "signal__neg_trade_flow_path_300s_signed_change": "trade_nvg__trade_flow_path_300s_signed_change",
        "signal__neg_hawkes_intensity_imbalance": "hawkes_lite__hawkes_intensity_imbalance",
        "signal__neg_hawkes_signed_pressure": "hawkes_derived__hawkes_signed_pressure",
    }
    for target, source in sign_sources.items():
        if source in result.columns:
            result[target] = -result[source].astype(float)

    consensus_columns = [
        column
        for column in [
            "signal__neg_minute_nvg_price_nvg_30m_top_bottom_asymmetry",
            "signal__neg_trade_flow_path_300s_terminal_position",
            "signal__neg_hawkes_signed_pressure",
        ]
        if column in result.columns
    ]
    if consensus_columns:
        zscores = [_cs_zscore(result[column].astype(float)) for column in consensus_columns]
        result["signal__structural_overextension_consensus"] = pd.concat(zscores, axis=1).mean(axis=1)
    return result


def build_labels(frame: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    labels = pd.DataFrame(index=frame.index)
    group = frame.groupby(level="instrument", sort=False, group_keys=False)
    open_px = frame["bars_1m__open"].astype(float).replace(0, np.nan)
    close = frame["bars_1m__close"].astype(float).replace(0, np.nan)
    dollar_volume = frame["bars_1m__dollar_volume"].astype(float).clip(lower=0)
    trade_count = frame.get("trades_1m_core__trade_count", pd.Series(np.nan, index=frame.index)).astype(float).clip(lower=0)
    ret1 = group["bars_1m__close"].pct_change().astype(float)
    liquidity_log = np.log1p(dollar_volume) + 0.25 * np.log1p(trade_count)
    amihud = ret1.abs() / (dollar_volume + 1.0) * 1e8

    for horizon in horizons:
        exit_open = open_px.groupby(level="instrument", sort=False, group_keys=False).shift(-horizon)
        labels[f"return_open_to_open__h{horizon}"] = exit_open / open_px - 1.0
        future_liquidity = liquidity_log.groupby(level="instrument", sort=False, group_keys=False).transform(
            lambda s, w=horizon: _future_roll(s.astype(float), w, "mean")
        )
        labels[f"liquidity_deterioration__h{horizon}"] = liquidity_log - future_liquidity
        future_var = (ret1.astype(float) ** 2).groupby(level="instrument", sort=False, group_keys=False).transform(
            lambda s, w=horizon: _future_roll(s, w, "sum")
        )
        labels[f"realized_volatility__h{horizon}"] = np.sqrt(future_var)
        labels[f"jump_risk__h{horizon}"] = ret1.abs().groupby(level="instrument", sort=False, group_keys=False).transform(
            lambda s, w=horizon: _future_roll(s.astype(float), w, "max")
        )
        labels[f"execution_cost_proxy__h{horizon}"] = amihud.groupby(level="instrument", sort=False, group_keys=False).transform(
            lambda s, w=horizon: _future_roll(s.astype(float), w, "mean")
        )
    return labels.astype("float32")


def infer_bundle(name: str) -> str:
    if name.startswith("traditional__"):
        return "TRAD"
    if name.startswith("bars_1m__"):
        return "BARS"
    if name.startswith("trades_1m_core__"):
        return "TRADES_CORE"
    if name.startswith("minute_nvg__"):
        return "MINUTE_NVG"
    if name.startswith("trade_nvg__"):
        return "TRADE_NVG"
    if name.startswith("hawkes_lite__"):
        return "HAWKES_LITE"
    if name.startswith("hawkes_derived__"):
        return "HAWKES_DERIVED"
    if name.startswith("signal__"):
        return "SIGNAL"
    return "OTHER"


def universe_masks(features: pd.DataFrame) -> dict[str, pd.Series]:
    all_mask = pd.Series(True, index=features.index)
    required = [
        "minute_nvg__price_nvg_30m_top_bottom_asymmetry",
        "trade_nvg__trade_flow_path_300s_terminal_position",
        "hawkes_derived__hawkes_signed_pressure",
    ]
    present_required = [column for column in required if column in features.columns]
    common_mask = features[present_required].notna().all(axis=1) if present_required else all_mask.copy()
    liquid_mask = all_mask.copy()
    if "bars_1m__close" in features.columns:
        liquid_mask &= features["bars_1m__close"].astype(float) >= 5.0
    if "bars_1m__dollar_volume" in features.columns:
        liquid_mask &= features["bars_1m__dollar_volume"].astype(float) > 0.0
    if "trades_1m_core__trade_count" in features.columns:
        liquid_mask &= features["trades_1m_core__trade_count"].astype(float) >= 1.0
    if "trade_nvg__trade_active_second_ratio_300s" in features.columns:
        liquid_mask &= features["trade_nvg__trade_active_second_ratio_300s"].astype(float) >= 0.01
    if "trade_nvg__trade_price_stale_ratio_300s" in features.columns:
        liquid_mask &= features["trade_nvg__trade_price_stale_ratio_300s"].astype(float) <= 0.95
    return {
        "all": all_mask.fillna(False),
        "common_structural": common_mask.fillna(False),
        "liquid_basic": (common_mask & liquid_mask).fillna(False),
    }


def _corrwith_ranked_block(x_rank: pd.DataFrame, y_rank: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    y = y_rank.to_numpy(dtype="float64")
    x = x_rank.to_numpy(dtype="float64")
    valid = np.isfinite(x) & np.isfinite(y[:, None])
    n = valid.sum(axis=0).astype("float64")
    x0 = np.where(valid, x, 0.0)
    y0 = np.where(np.isfinite(y), y, 0.0)
    sum_x = x0.sum(axis=0)
    sum_y = (valid * y0[:, None]).sum(axis=0)
    sum_xy = (x0 * y0[:, None]).sum(axis=0)
    sum_x2 = (x0 * x0).sum(axis=0)
    sum_y2 = (valid * (y0[:, None] * y0[:, None])).sum(axis=0)
    numerator = sum_xy - (sum_x * sum_y / np.maximum(n, 1.0))
    var_x = sum_x2 - (sum_x * sum_x / np.maximum(n, 1.0))
    var_y = sum_y2 - (sum_y * sum_y / np.maximum(n, 1.0))
    denominator = np.sqrt(var_x * var_y)
    corr = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype="float64"), where=denominator > 0)
    corr[n < 8] = np.nan
    return corr, n


def rank_ic_summary(features: pd.DataFrame, labels: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    feature_columns = [
        column for column in features.columns if ANALYSIS_FEATURE_RE.search(column) and features[column].notna().any()
    ]
    rows: list[dict[str, Any]] = []
    masks = universe_masks(features)
    for universe, universe_mask in masks.items():
        universe_features = features.loc[universe_mask, feature_columns]
        universe_labels = labels.loc[universe_mask]
        ranked_labels = universe_labels.groupby(level="datetime", sort=False).rank(method="average")
        feature_rank_chunks: list[tuple[list[str], pd.DataFrame]] = []
        chunk_size = 32
        for start in range(0, len(feature_columns), chunk_size):
            chunk = feature_columns[start : start + chunk_size]
            feature_rank_chunks.append(
                (chunk, universe_features[chunk].groupby(level="datetime", sort=False).rank(method="average"))
            )
        for label_column in labels.columns:
            label = universe_labels[label_column]
            label_name, horizon_text = label_column.rsplit("__h", 1)
            horizon = int(horizon_text)
            label_non_null = int(label.notna().sum())
            coverage = (
                universe_features.notna()
                .where(label.notna(), False)
                .sum(axis=0)
                .reindex(feature_columns)
                .to_numpy(dtype="float64")
                / max(1, label_non_null)
            )
            y_rank = ranked_labels[label_column]
            offset = 0
            for chunk, x_rank in feature_rank_chunks:
                corrs, counts = _corrwith_ranked_block(x_rank, y_rank)
                for local_idx, feature in enumerate(chunk):
                    idx = offset + local_idx
                    value = corrs[local_idx]
                    rows.append(
                        {
                            "trade_date": trade_date,
                            "universe": universe,
                            "horizon_bars": horizon,
                            "label_family": label_name,
                            "feature": feature,
                            "bundle": infer_bundle(feature),
                            "rank_ic_method": "pooled_cs_rank_ic",
                            "ic_count": int(counts[local_idx]),
                            "rank_ic_mean": float(value) if np.isfinite(value) else math.nan,
                            "rank_ic_std": math.nan,
                            "rank_ic_positive_ratio": float(value > 0) if np.isfinite(value) else math.nan,
                            "coverage": float(coverage[idx]) if np.isfinite(coverage[idx]) else 0.0,
                            "label_non_null": label_non_null,
                        }
                    )
                offset += len(chunk)
    return pd.DataFrame(rows)


def unit_paths(out_root: Path, trade_date: str) -> tuple[Path, Path, Path]:
    out_dir = out_root / "02_multilabel_factor_diagnostics" / f"date={trade_date}"
    return out_dir, out_dir / "factor_rank_ic_summary.parquet", out_dir / "_SUCCESS"


def run_date(trade_date: str, horizons: list[int], out_root: Path) -> dict[str, Any]:
    out_dir, summary_path, success_path = unit_paths(out_root, trade_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    if success_path.exists() and summary_path.exists():
        return {"trade_date": trade_date, "status": "skipped", "out_dir": str(out_dir)}

    loader = NFFDataLoader(
        warehouse_root=WAREHOUSE_ROOT,
        canonical_sets=canonical_sets(),
        feature_sets=feature_sets(),
        execution={"frequency": "1min", "delay_bars": 1, "collision_policy": "latest"},
        label=None,
        join="inner",
        strict_manifests=True,
        allow_mixed_contracts=True,
        output_float32=True,
        arrow_use_threads=True,
    )
    frame = loader.load(
        instruments="all",
        start_time=f"{trade_date}T14:30:00Z",
        end_time=f"{trade_date}T20:00:00Z",
    )
    features = frame["feature"].sort_index()
    features = add_hawkes_derived(features)
    features = add_traditional_factors(features)
    features = add_research_signals(features)
    features = features.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).astype("float32")
    labels = build_labels(features, horizons).replace([np.inf, -np.inf], np.nan)
    summary = rank_ic_summary(features, labels, trade_date)
    summary.to_parquet(summary_path, index=False)
    summary.to_csv(out_dir / "factor_rank_ic_summary.csv", index=False)
    meta = {
        "trade_date": trade_date,
        "horizons": horizons,
        "rows": int(len(features)),
        "feature_count": int(features.shape[1]),
        "label_non_null": {column: int(labels[column].notna().sum()) for column in labels.columns},
            "feature_sets": {key: len(value["columns"]) for key, value in feature_sets().items()},
            "canonical_sets": {key: len(value["columns"]) for key, value in canonical_sets().items()},
            "universe_counts": {name: int(mask.sum()) for name, mask in universe_masks(features).items()},
            "loader_report": loader.last_load_report,
        }
    atomic_write_json(meta_path, meta)
    success_path.write_text(utc_now(), encoding="utf-8")
    return {"trade_date": trade_date, "status": "completed", **meta}


def aggregate(out_root: Path) -> None:
    agg = out_root / "02_multilabel_factor_diagnostics" / "_aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    frames = [
        pd.read_parquet(path)
        for path in sorted((out_root / "02_multilabel_factor_diagnostics").glob("date=*/factor_rank_ic_summary.parquet"))
    ]
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(agg / "factor_rank_ic_by_date.parquet", index=False)
    data.to_csv(agg / "factor_rank_ic_by_date.csv", index=False)
    grouped = (
        data.groupby(["universe", "label_family", "horizon_bars", "bundle", "feature"], dropna=False)
        .agg(
            dates=("trade_date", "nunique"),
            total_ic_count=("ic_count", "sum"),
            mean_rank_ic=("rank_ic_mean", "mean"),
            median_rank_ic=("rank_ic_mean", "median"),
            std_daily_rank_ic=("rank_ic_mean", "std"),
            mean_positive_ratio=("rank_ic_positive_ratio", "mean"),
            mean_coverage=("coverage", "mean"),
            mean_label_non_null=("label_non_null", "mean"),
        )
        .reset_index()
        .sort_values(["universe", "label_family", "horizon_bars", "mean_rank_ic"], ascending=[True, True, True, False])
    )
    grouped.to_parquet(agg / "factor_rank_ic_overall_summary.parquet", index=False)
    grouped.to_csv(agg / "factor_rank_ic_overall_summary.csv", index=False)
    top = grouped.groupby(["universe", "label_family", "horizon_bars"], group_keys=False).head(25)
    top.to_csv(agg / "top25_by_label_horizon.csv", index=False)


def resource_snapshot(out_root: Path) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.2)
    d_usage = shutil.disk_usage(out_root.anchor or str(out_root.drive) + "\\")
    return {
        "cpu_percent": round(float(cpu), 2),
        "memory_percent": round(float(memory.percent), 2),
        "memory_available_gb": round(memory.available / 1024**3, 2),
        "disk_free_gb": round(d_usage.free / 1024**3, 2),
    }


def worker_command(args: argparse.Namespace, out_root: Path, trade_date: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--run-id",
        args.run_id,
        "--out-root",
        str(out_root),
        "--worker-date",
        trade_date,
        "--horizons",
        *[str(value) for value in args.horizons],
    ]


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "trade_date": result.get("trade_date"),
        "status": result.get("status", "completed"),
        "rows": result.get("rows"),
        "feature_count": result.get("feature_count"),
        "label_non_null_min": min(result.get("label_non_null", {"x": 0}).values())
        if isinstance(result.get("label_non_null"), dict)
        else result.get("label_non_null"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--out-root")
    parser.add_argument("--horizons", nargs="+", type=int, default=[15, 30, 60, 120])
    parser.add_argument("--start-date", default="2026-01-02")
    parser.add_argument("--end-date", default="2026-07-22")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-date")
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--max-parallel", type=int, default=24)
    parser.add_argument("--min-parallel", type=int, default=4)
    parser.add_argument("--target-cpu", type=float, default=90.0)
    parser.add_argument("--memory-high-water", type=float, default=88.0)
    parser.add_argument("--disk-free-floor-gb", type=float, default=50.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--launch-batch-size", type=int, default=4)
    args = parser.parse_args()

    out_root = Path(args.out_root) if args.out_root else RESEARCH_ROOT / "runs" / f"v2_multilabel_{args.run_id}"
    out_root.mkdir(parents=True, exist_ok=True)
    status_path = out_root / "status.json"
    resource_log_path = out_root / "resource_samples.ndjson"

    if args.worker:
        if not args.worker_date:
            raise ValueError("--worker requires --worker-date")
        print(json.dumps(run_date(args.worker_date, args.horizons, out_root), indent=2, default=str))
        return 0

    dates = load_dates(args.start_date, args.end_date)
    total = len(dates)
    completed = 0
    failures: list[dict[str, Any]] = []
    pending = list(dates)
    attempts: dict[str, int] = {}
    running: dict[str, dict[str, Any]] = {}
    current_parallel = max(args.min_parallel, min(args.parallel, args.max_parallel))
    last_tune = 0.0
    update_status(
        status_path,
        pid=os.getpid(),
        run_id=args.run_id,
        out_root=str(out_root),
        started_utc=utc_now(),
        stage="multilabel_factor_diagnostics",
        status="running",
        total_units=total,
        completed_units=completed,
        parallel=current_parallel,
        running_workers=0,
        resources=resource_snapshot(out_root),
        horizons=args.horizons,
        labels=list(LABEL_NAMES),
    )
    append_jsonl(
        resource_log_path,
        {
            "sample_utc": utc_now(),
            "stage": "started",
            "completed_units": completed,
            "total_units": total,
            "parallel": current_parallel,
            "running_workers": 0,
            "pending_units": len(pending),
            "resources": resource_snapshot(out_root),
        },
    )

    while pending or running:
        resources = resource_snapshot(out_root)
        now = time.time()
        if now - last_tune >= 60:
            last_tune = now
            if resources["disk_free_gb"] < args.disk_free_floor_gb:
                current_parallel = 0
            elif resources["memory_percent"] >= args.memory_high_water:
                current_parallel = max(args.min_parallel, max(1, current_parallel // 2))
            elif (
                resources["cpu_percent"] < args.target_cpu - 12
                and resources["memory_percent"] < args.memory_high_water - 15
                and current_parallel < args.max_parallel
                and len(running) >= current_parallel
            ):
                current_parallel = min(args.max_parallel, current_parallel + 2)
            elif resources["cpu_percent"] > 98 and current_parallel > args.min_parallel:
                current_parallel = max(args.min_parallel, current_parallel - 1)

        launched_this_cycle = 0
        while pending and len(running) < current_parallel and launched_this_cycle < max(1, args.launch_batch_size):
            trade_date = pending.pop(0)
            out_dir, _, success = unit_paths(out_root, trade_date)
            if success.exists():
                completed += 1
                meta_path = out_dir / "meta.json"
                result = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
                    "trade_date": trade_date,
                    "status": "skipped",
                }
                update_status(
                    status_path,
                    last_unit=compact_result({**result, "status": "skipped"}),
                    completed_units=completed,
                    total_units=total,
                    parallel=current_parallel,
                    running_workers=len(running),
                    failure_count=len(failures),
                    resources=resources,
                )
                continue
            attempts[trade_date] = attempts.get(trade_date, 0) + 1
            attempt = attempts[trade_date]
            log_dir = out_root / "worker_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = log_dir / f"date={trade_date}_attempt={attempt}.out.log"
            stderr_path = log_dir / f"date={trade_date}_attempt={attempt}.err.log"
            out = stdout_path.open("w", encoding="utf-8")
            err = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(worker_command(args, out_root, trade_date), stdout=out, stderr=err)
            running[trade_date] = {
                "process": process,
                "stdout": out,
                "stderr": err,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "started": time.perf_counter(),
                "attempt": attempt,
            }
            launched_this_cycle += 1

        finished: list[str] = []
        for trade_date, info in list(running.items()):
            process: subprocess.Popen[Any] = info["process"]
            rc = process.poll()
            if rc is None:
                continue
            info["stdout"].close()
            info["stderr"].close()
            elapsed = round(time.perf_counter() - info["started"], 3)
            out_dir, _, success = unit_paths(out_root, trade_date)
            if rc == 0 and success.exists():
                completed += 1
                meta_path = out_dir / "meta.json"
                result = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
                    "trade_date": trade_date,
                    "status": "completed_no_meta",
                }
                update_status(
                    status_path,
                    current_unit={"trade_date": trade_date},
                    last_unit=compact_result(result),
                    completed_units=completed,
                    total_units=total,
                    last_unit_seconds=elapsed,
                    parallel=current_parallel,
                    running_workers=max(0, len(running) - 1),
                    running_units=[{"trade_date": d} for d in running.keys() if d != trade_date][:20],
                    failure_count=len(failures),
                    resources=resources,
                )
            else:
                stderr_tail = Path(info["stderr_path"]).read_text(encoding="utf-8", errors="replace")[-4000:]
                failure = {
                    "trade_date": trade_date,
                    "attempt": info["attempt"],
                    "returncode": rc,
                    "error": stderr_tail or "worker failed without stderr",
                    "stdout": str(info["stdout_path"]),
                    "stderr": str(info["stderr_path"]),
                }
                if info["attempt"] <= args.retries:
                    pending.append(trade_date)
                    failure["healing_action"] = "requeued"
                else:
                    failures.append(failure)
                    fail_dir = out_root / "failures"
                    fail_dir.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(fail_dir / f"date={trade_date}.json", failure)
                update_status(
                    status_path,
                    current_unit={"trade_date": trade_date},
                    completed_units=completed,
                    total_units=total,
                    parallel=current_parallel,
                    running_workers=max(0, len(running) - 1),
                    failure_count=len(failures),
                    last_failure=failure,
                    resources=resources,
                )
            finished.append(trade_date)
        for trade_date in finished:
            running.pop(trade_date, None)

        update_status(
            status_path,
            completed_units=completed,
            total_units=total,
            parallel=current_parallel,
            running_workers=len(running),
            running_units=[{"trade_date": d} for d in running.keys()][:20],
            pending_units=len(pending),
            failure_count=len(failures),
            resources=resources,
        )
        append_jsonl(
            resource_log_path,
            {
                "sample_utc": utc_now(),
                "stage": "multilabel_factor_diagnostics",
                "completed_units": completed,
                "total_units": total,
                "parallel": current_parallel,
                "running_workers": len(running),
                "pending_units": len(pending),
                "failure_count": len(failures),
                "running_units": [{"trade_date": d} for d in running.keys()][:20],
                "resources": resources,
            },
        )
        time.sleep(5)

    update_status(status_path, stage="aggregate", completed_units=completed, total_units=total, failure_count=len(failures))
    aggregate(out_root)
    terminal_status = "partial_success" if failures else "complete"
    update_status(
        status_path,
        stage="finished",
        status=terminal_status,
        completed_units=completed,
        total_units=total,
        failure_count=len(failures),
        finished_utc=utc_now(),
        aggregate_dir=str(out_root / "02_multilabel_factor_diagnostics" / "_aggregate"),
    )
    append_jsonl(
        resource_log_path,
        {
            "sample_utc": utc_now(),
            "stage": "finished",
            "status": terminal_status,
            "completed_units": completed,
            "total_units": total,
            "failure_count": len(failures),
            "resources": resource_snapshot(out_root),
        },
    )
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
