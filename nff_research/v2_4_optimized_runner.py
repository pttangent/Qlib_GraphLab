from __future__ import annotations

"""CPU-optimized execution layer for the v2.3 NFF research contract.

The reference runner remains authoritative for formulas, labels, masks, output
schemas, contracts, checkpointing, and aggregation.  This module replaces only
hot execution paths whose repeated Pandas scans dominated wall time:

* exact warehouse column projection, including direct hawkes_derived reuse;
* one cross-sectional rank pass for minute-mean and pooled IC statistics;
* residualized feature reuse when return labels share an identical sample;
* one ranked feature matrix and one symbol-fold vector per OOF minute;
* 15-minute portfolio/decile prefiltering before expensive groupby loops;
* LPT date ordering from cached warehouse byte-cost estimates;
* per-stage timing written into each daily meta.json.

Run this file with the same CLI as v2_1_neutralized_runner.py.
"""

import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nff_research import v2_1_neutralized_runner as R


FASTPATH_VERSION = "2.4"
REFERENCE_RUNNER_VERSION = "2.3"
_STAGE_SECONDS: dict[str, float] = defaultdict(float)
_ORIGINALS: dict[str, Any] = {}


def _timed(stage: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    started = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        _STAGE_SECONDS[stage] += time.perf_counter() - started


def _suffixes(prefix: str, columns: list[str]) -> set[str]:
    token = prefix + "__"
    return {column[len(token) :] for column in columns if column.startswith(token)}


def _first_schema(dataset: str, candidates: tuple[str, ...]) -> tuple[str | None, set[str]]:
    for schema in candidates:
        try:
            columns = set(R.source_columns("feature", dataset, schema))
        except Exception:
            continue
        if columns:
            return schema, columns
    return None, set()


def research_required_columns_manifest() -> dict[str, Any]:
    """Return the exact warehouse inputs needed by the v2.3/v2.4 study.

    The manifest is derived from the representative registry and gate/control
    contracts, rather than broad regex selectors.  It is safe to persist in the
    run contract and makes source projection auditable.
    """

    representative = list(R.REPRESENTATIVE_ALPHA_FEATURES)
    minute_required = _suffixes("minute_nvg", representative)
    trade_required = _suffixes("trade_nvg", representative)
    hawkes_required = _suffixes("hawkes_lite", representative)
    derived_required = _suffixes("hawkes_derived", representative)

    # Eligibility and neutralization inputs that are intentionally not alpha.
    trade_required.update({"trade_active_second_ratio_300s", "trade_price_stale_ratio_300s"})
    for spec in R.HAWKES_GATE_SPECS:
        column = str(spec["column"])
        if column.startswith("hawkes_lite__"):
            hawkes_required.add(column.split("__", 1)[1])
        elif column.startswith("hawkes_derived__"):
            derived_required.add(column.split("__", 1)[1])

    # Fallback dependencies required only when a materialized derived family is
    # unavailable.  They are still projected narrowly.
    derived_dependencies = {
        "hawkes_ready",
        "hawkes_total_intensity",
        "hawkes_total_baseline",
        "hawkes_intensity_imbalance",
        "hawkes_endogenous_share",
        "hawkes_cross_excitation_share",
        "hawkes_branching_ratio_max",
        "hawkes_short_excitation_share",
        "hawkes_medium_excitation_share",
        "hawkes_long_excitation_share",
        "hawkes_total_intensity_mean_60s",
        "hawkes_total_intensity_mean_300s",
        "hawkes_endogenous_share_mean_60s",
        "hawkes_endogenous_share_mean_180s",
        "hawkes_endogenous_share_mean_300s",
        "hawkes_shock_score_60s",
        "hawkes_shock_score_180s",
        "hawkes_shock_score_300s",
    }

    schemas: dict[str, str | None] = {}
    available: dict[str, set[str]] = {}
    for dataset, candidates in {
        "minute_nvg": ("v4", "v3", "v2", "v1"),
        "trade_nvg": ("v4", "v3", "v2", "v1"),
        "hawkes_lite": ("v4", "v3", "v2", "v1"),
        "hawkes_derived": ("v4", "v3", "v2", "v1"),
    }.items():
        schema, columns = _first_schema(dataset, candidates)
        schemas[dataset] = schema
        available[dataset] = columns

    derived_materialized = bool(schemas["hawkes_derived"] and derived_required <= available["hawkes_derived"])
    if not derived_materialized:
        hawkes_required.update(derived_dependencies)

    requested = {
        "minute_nvg": sorted(minute_required & available["minute_nvg"]),
        "trade_nvg": sorted(trade_required & available["trade_nvg"]),
        "hawkes_lite": sorted(hawkes_required & available["hawkes_lite"]),
        "hawkes_derived": sorted(derived_required & available["hawkes_derived"]) if derived_materialized else [],
    }
    return {
        "version": "research_required_columns_v1",
        "schemas": schemas,
        "columns": requested,
        "derived_materialized": derived_materialized,
        "requested_count": sum(len(values) for values in requested.values()),
    }


def narrow_feature_sets() -> dict[str, dict[str, Any]]:
    manifest = research_required_columns_manifest()
    out: dict[str, dict[str, Any]] = {}
    for dataset, columns in manifest["columns"].items():
        schema = manifest["schemas"].get(dataset)
        if schema and columns:
            out[dataset] = {"schema_version": schema, "columns": columns}
    return out


def add_all_features_fast(frame: pd.DataFrame) -> pd.DataFrame:
    manifest = research_required_columns_manifest()
    derived_columns = [f"hawkes_derived__{value}" for value in manifest["columns"].get("hawkes_derived", [])]
    if not derived_columns or any(column not in frame.columns for column in derived_columns):
        frame = R.V2.add_hawkes_derived(frame)
    # Traditional factors are required; deprecated pre-specified signals are
    # deliberately not materialized because they are never alpha inputs.
    frame = R.V2.add_traditional_factors(frame)
    keep = set(R.REPRESENTATIVE_ALPHA_FEATURES)
    keep.update({
        "bars_1m__open", "bars_1m__close", "bars_1m__vwap", "bars_1m__volume", "bars_1m__dollar_volume",
        "trades_1m_core__trade_count",
        "trade_nvg__trade_active_second_ratio_300s",
        "trade_nvg__trade_price_stale_ratio_300s",
    })
    keep.update(str(spec["column"]) for spec in R.HAWKES_GATE_SPECS)
    selected = [column for column in frame.columns if column in keep]
    return frame[selected].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).astype("float32")


def _online_corr_state() -> dict[str, float]:
    return {"n": 0.0, "sx": 0.0, "sy": 0.0, "sxx": 0.0, "syy": 0.0, "sxy": 0.0}


def _update_corr_state(
    state: dict[str, float], x: np.ndarray, y: np.ndarray, valid: np.ndarray | None = None
) -> None:
    # Use the pre-transform validity mask when supplied.  Percentile ranking
    # must not alter the sample contract through a divide-by-zero or an
    # implementation-specific non-finite intermediate.
    if valid is None:
        valid = np.isfinite(x) & np.isfinite(y)
    else:
        valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return
    xv = x[valid].astype("float64", copy=False)
    yv = y[valid].astype("float64", copy=False)
    state["n"] += float(len(xv))
    state["sx"] += float(xv.sum())
    state["sy"] += float(yv.sum())
    state["sxx"] += float(xv @ xv)
    state["syy"] += float(yv @ yv)
    state["sxy"] += float(xv @ yv)


def _finish_corr_state(state: dict[str, float], min_n: int) -> tuple[float, int]:
    n = int(state["n"])
    if n < min_n:
        return math.nan, n
    cov = state["sxy"] - state["sx"] * state["sy"] / n
    vx = state["sxx"] - state["sx"] ** 2 / n
    vy = state["syy"] - state["sy"] ** 2 / n
    denom = math.sqrt(max(vx, 0.0) * max(vy, 0.0))
    return (float(cov / denom), n) if denom > 0 else (math.nan, n)


def ranked_ic_stats_once(
    feature_frame: pd.DataFrame,
    label: pd.Series,
    feature_columns: list[str],
    min_n: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Compute minute-mean and corrected pooled IC from one rank pass."""

    minute_values: dict[str, list[float]] = {column: [] for column in feature_columns}
    minute_counts: dict[str, int] = {column: 0 for column in feature_columns}
    pooled = {column: _online_corr_state() for column in feature_columns}
    admitted_minutes = 0
    work = pd.concat([feature_frame[feature_columns], label.rename("__label")], axis=1)
    for _, block in work.groupby(level="datetime", sort=False):
        y = block["__label"]
        if int(y.notna().sum()) < min_n:
            continue
        admitted_minutes += 1
        # Keep the two IC contracts distinct.  Reference minute-mean IC uses
        # average ranks directly; corrected pooled IC uses per-minute
        # percentile ranks demeaned within each cross-section.  Percentile
        # ranks are an affine transform only when the valid count is identical,
        # which is not true with feature-specific missingness.
        ranked_y_raw = y.rank(method="average")
        label_count = int(ranked_y_raw.notna().sum())
        ranked_y_pooled = ranked_y_raw / float(label_count)
        ranked_y_pooled = ranked_y_pooled - ranked_y_pooled.mean()
        ranked_x_raw = block[feature_columns].rank(method="average")
        feature_counts = ranked_x_raw.notna().sum(axis=0).astype("float64")
        ranked_x_pooled = ranked_x_raw.divide(feature_counts, axis="columns")
        ranked_x_pooled = ranked_x_pooled - ranked_x_pooled.mean(axis=0)
        yv = ranked_y_raw.to_numpy(dtype="float64")
        yv_pooled = ranked_y_pooled.to_numpy(dtype="float64")
        xv = ranked_x_raw.to_numpy(dtype="float64")
        xv_pooled = ranked_x_pooled.to_numpy(dtype="float64")
        for idx, feature in enumerate(feature_columns):
            corr, count = R._corr_with_min_n(xv[:, idx], yv, min_n=min_n)
            minute_counts[feature] += int(count)
            if np.isfinite(corr):
                minute_values[feature].append(float(corr))
            pooled_valid = np.isfinite(xv[:, idx]) & np.isfinite(yv)
            _update_corr_state(pooled[feature], xv_pooled[:, idx], yv_pooled, valid=pooled_valid)

    minute_out: dict[str, dict[str, float]] = {}
    pooled_out: dict[str, dict[str, float]] = {}
    for feature in feature_columns:
        values = np.asarray(minute_values[feature], dtype="float64")
        minute_out[feature] = {
            "ic_minutes": int(values.size),
            "ic_count": int(minute_counts[feature]),
            "rank_ic_mean": float(values.mean()) if values.size else math.nan,
            "rank_ic_std": float(values.std(ddof=1)) if values.size > 1 else math.nan,
            "rank_ic_positive_ratio": float((values > 0).mean()) if values.size else math.nan,
        }
        value, count = _finish_corr_state(pooled[feature], min_n=min_n)
        pooled_out[feature] = {
            "ic_minutes": admitted_minutes,
            "ic_count": int(count),
            "rank_ic_mean": value,
            "rank_ic_std": math.nan,
            "rank_ic_positive_ratio": float(value > 0) if np.isfinite(value) else math.nan,
        }
    return minute_out, pooled_out


def _index_cache_key(index: pd.Index) -> tuple[int, int]:
    values = pd.util.hash_pandas_object(index, index=False).to_numpy(dtype="uint64")
    return len(index), int(values.sum(dtype="uint64"))


def minute_rank_ic_summary_fast(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_masks: dict[str, pd.Series],
    controls: pd.DataFrame,
    trade_date: str,
    min_n: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str], tuple[pd.DataFrame, pd.Series]]]:
    feature_columns = R.analysis_features(features)
    decile_cache_features = [column for column in R.CORE_DECILE_FEATURES if column in feature_columns]
    masks = R.universe_masks(features, controls)
    rows: list[dict[str, Any]] = []
    residual_cache: dict[tuple[str, str], tuple[pd.DataFrame, pd.Series]] = {}
    residual_feature_cache: dict[tuple[str, tuple[int, int]], pd.DataFrame] = {}

    for universe, universe_mask in masks.items():
        for label_column in labels.columns:
            family, horizon_text = label_column.rsplit("__h", 1)
            horizon = int(horizon_text)
            base_mask = (universe_mask & labels[label_column].notna() & label_masks[label_column].fillna(False)).fillna(False)
            if int(base_mask.sum()) < min_n:
                continue
            feature_frame = features.loc[base_mask, feature_columns]
            label = labels.loc[base_mask, label_column]
            minute_stats, pooled_stats = ranked_ic_stats_once(feature_frame, label, feature_columns, min_n)
            label_non_null = int(label.notna().sum())
            coverage = feature_frame.notna().sum(axis=0) / max(1, label_non_null)
            R.append_ic_rows(rows, minute_stats, trade_date, universe, "raw", "minute_mean_cs_rank_ic", family, horizon, coverage, label_non_null)
            R.append_ic_rows(rows, pooled_stats, trade_date, universe, "raw", "pooled_cs_demeaned_pct_rank_ic", family, horizon, coverage, label_non_null)

            if not family.startswith("return_") or universe == "own_feature_universe":
                continue
            controls_sub = controls.loc[base_mask]
            cache_key = (universe, _index_cache_key(feature_frame.index))
            feature_resid = residual_feature_cache.get(cache_key)
            if feature_resid is None:
                feature_resid = R._residualize_matrix(feature_frame, controls_sub, min_n=max(min_n, 40))
                residual_feature_cache[cache_key] = feature_resid
            label_resid = R._residualize_matrix(label.to_frame(label_column), controls_sub, min_n=max(min_n, 40))[label_column]
            neut_minute, neut_pooled = ranked_ic_stats_once(feature_resid, label_resid, feature_columns, min_n)
            neutral_label_non_null = int(label_resid.notna().sum())
            neutral_coverage = feature_resid.notna().sum(axis=0) / max(1, neutral_label_non_null)
            if family in {"return_open_to_open", "return_vwap_to_vwap"} and decile_cache_features:
                residual_cache[(universe, label_column)] = (
                    feature_resid[decile_cache_features].copy(deep=False),
                    label_resid.copy(deep=False),
                )
            R.append_ic_rows(rows, neut_minute, trade_date, universe, "neutralized", "minute_mean_cs_rank_ic_residualized", family, horizon, neutral_coverage, neutral_label_non_null)
            R.append_ic_rows(rows, neut_pooled, trade_date, universe, "neutralized", "pooled_cs_demeaned_pct_rank_ic_residualized", family, horizon, neutral_coverage, neutral_label_non_null)
    return pd.DataFrame(rows), residual_cache


@lru_cache(maxsize=32768)
def _symbol_fold(symbol: str, folds: int) -> int:
    return R.stable_symbol_fold(symbol, folds)


def _fold_ids(symbols: np.ndarray, folds: int) -> np.ndarray:
    return np.fromiter((_symbol_fold(str(symbol).strip().upper(), folds) for symbol in symbols), dtype="int16", count=len(symbols))


def _batched_oof_predictions(
    ranked_all: pd.DataFrame,
    raw_y: pd.Series,
    fold_ids: np.ndarray,
    step_columns: list[tuple[str, list[str]]],
    folds: int,
    alpha: float,
    min_train_n: int,
) -> dict[str, np.ndarray] | None:
    outputs = {step: np.full(len(raw_y), np.nan, dtype="float64") for step, _ in step_columns}
    y_values = pd.to_numeric(raw_y, errors="coerce").to_numpy(dtype="float64")
    for held_out in range(folds):
        test = fold_ids == held_out
        train = ~test
        if not test.any() or int(train.sum()) < min_train_n:
            return None
        train_y = pd.Series(y_values[train]).rank(method="average", pct=True).to_numpy(dtype="float64")
        train_y -= train_y.mean()
        for step, columns in step_columns:
            matrix = ranked_all[columns].to_numpy(dtype="float64", copy=False)
            fitted = R._ridge_fit(matrix[train], train_y, alpha)
            if fitted is None:
                return None
            outputs[step][test] = R._ridge_predict(matrix[test], fitted)
    return outputs if all(np.isfinite(values).all() for values in outputs.values()) else None


def bundle_incremental_model_screen_fast(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_masks: dict[str, pd.Series],
    controls: pd.DataFrame,
    trade_date: str,
    min_n: int,
    *,
    oof_folds: int = R.OOF_FOLDS,
    oof_ridge_alpha: float = R.OOF_RIDGE_ALPHA,
    oof_min_train_n: int = R.OOF_MIN_TRAIN_N,
) -> pd.DataFrame:
    feature_columns = R.analysis_features(features)
    bundle_columns = {
        step: [column for column in feature_columns if R.infer_bundle(column) in bundles]
        for step, bundles in R.BUNDLE_MODEL_STEPS
    }
    active_steps = [(step, bundle_columns[step]) for step, _ in R.BUNDLE_MODEL_STEPS if bundle_columns[step]]
    needed = sorted({column for _, columns in active_steps for column in columns})
    if not active_steps or not needed:
        return pd.DataFrame()

    universe_mask = R.universe_masks(features, controls)["liquid_common_adv20_top1000"]
    sampled_mask = _rebalance_mask(features.index, 15)
    rows: list[dict[str, Any]] = []
    for label_column in labels.columns:
        family, horizon_text = label_column.rsplit("__h", 1)
        if family not in R.BUNDLE_MODEL_LABELS:
            continue
        horizon = int(horizon_text)
        model_features = features
        steps = active_steps
        label_needed = needed
        model_baseline = "representative_feature_bundles"
        if family == "execution_cost_proxy":
            baseline = [column for column in R.MECHANICAL_PROXY_BASELINE_COLUMNS if column in controls.columns]
            if len(baseline) != len(R.MECHANICAL_PROXY_BASELINE_COLUMNS):
                continue
            model_features = pd.concat([features, controls[baseline]], axis=1)
            steps = [("mechanical_baseline_controls", baseline)] + [
                (f"mechanical_baseline_plus_{step}", baseline + columns) for step, columns in active_steps
            ]
            label_needed = sorted({column for _, columns in steps for column in columns})
            model_baseline = "current_log_intraday_dollar_volume_and_realized_vol_60m"

        base_mask = (
            universe_mask
            & labels[label_column].notna()
            & label_masks[label_column].fillna(False)
            & sampled_mask
        ).fillna(False)
        if int(base_mask.sum()) < min_n:
            continue
        work = pd.concat([model_features.loc[base_mask, label_needed], labels.loc[base_mask, label_column].rename("label")], axis=1)
        prediction_parts: dict[str, list[pd.Series]] = {step: [] for step, _ in steps}
        target_parts: list[pd.Series] = []
        for _, raw_block in work.groupby(level="datetime", sort=False):
            block = raw_block.dropna(subset=label_needed + ["label"])
            if len(block) < min_n:
                continue
            ranked_all = block[label_needed].rank(method="average", pct=True)
            ranked_all -= ranked_all.mean(axis=0)
            symbols = block.index.get_level_values("instrument").astype(str).to_numpy()
            predictions = _batched_oof_predictions(
                ranked_all,
                block["label"],
                _fold_ids(symbols, oof_folds),
                steps,
                oof_folds,
                oof_ridge_alpha,
                oof_min_train_n,
            )
            if predictions is None:
                continue
            target_parts.append(block["label"].astype("float64"))
            for step, _ in steps:
                prediction_parts[step].append(pd.Series(predictions[step], index=block.index, dtype="float64"))

        if not target_parts:
            continue
        target = pd.concat(target_parts)
        sample_hash = R._index_identity_hash(target.index)
        fold_hash = R._fold_identity_hash(target.index, oof_folds)
        previous_step: str | None = None
        previous_prediction: pd.Series | None = None
        previous_metrics: dict[str, Any] | None = None
        for step, columns in steps:
            prediction = pd.concat(prediction_parts[step])
            metrics = R._oof_prediction_metrics(prediction, target, min_n=min_n)
            deltas = {
                "delta_mean_minute_pred_rank_ic": math.nan,
                "delta_pooled_pred_rank_ic": math.nan,
                "delta_mean_minute_r2": math.nan,
                "mse_improvement_vs_previous": math.nan,
                "incremental_oof_prediction_rank_ic": math.nan,
            }
            if previous_prediction is not None and previous_metrics is not None:
                deltas = R._oof_metric_deltas(metrics, previous_metrics, prediction - previous_prediction, target, min_n=min_n)
            rows.append({
                "trade_date": trade_date,
                "universe": "liquid_common_adv20_top1000",
                "label_family": family,
                "label_evidence_role": R.BUNDLE_MODEL_LABEL_ROLES[family],
                "horizon_bars": horizon,
                "step": step,
                "previous_step": previous_step,
                "feature_count": len(columns),
                "common_feature_count": len(label_needed),
                "fold_count": oof_folds,
                "sampled_minutes": metrics["sampled_minutes"],
                "sample_count": metrics["sample_count"],
                "sample_identity_hash": sample_hash,
                "fold_identity_hash": fold_hash,
                "mean_minute_pred_rank_ic": metrics["mean_minute_pred_rank_ic"],
                "pooled_demeaned_pct_rank_pred_ic": metrics["pooled_demeaned_pct_rank_pred_ic"],
                "mean_minute_r2": metrics["mean_minute_r2"],
                "mean_minute_mse": metrics["mean_minute_mse"],
                **deltas,
                "model_baseline": model_baseline,
                "contract": "same-day 15m fixed-symbol-fold cross-sectional OOF ridge; all-step common sample; ranked feature matrix and folds cached once per minute; not temporal OOS",
            })
            previous_step = step
            previous_prediction = prediction
            previous_metrics = metrics
    return pd.DataFrame(rows)


def _rebalance_mask(index: pd.Index, minutes: int = 15) -> pd.Series:
    dt = pd.DatetimeIndex(index.get_level_values("datetime"))
    if dt.tz is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    local = dt.tz_convert(R.SESSION_TZ)
    minute_of_day = local.hour * 60 + local.minute
    open_minute = R.SESSION_OPEN[0] * 60 + R.SESSION_OPEN[1]
    close_minute = R.SESSION_CLOSE[0] * 60 + R.SESSION_CLOSE[1]
    values = (
        (minute_of_day >= open_minute)
        & (minute_of_day < close_minute)
        & (((minute_of_day - open_minute) % minutes) == 0)
    )
    return pd.Series(np.asarray(values, dtype=bool), index=index)


def decile_curves_prefiltered(*args: Any, **kwargs: Any) -> pd.DataFrame:
    features, labels, label_masks, controls = args[:4]
    mask = _rebalance_mask(features.index, 15)
    reduced_labels = labels.loc[mask]
    reduced_masks = {name: value.loc[mask] for name, value in label_masks.items()}
    residual_cache = args[4]
    reduced_residual: dict[tuple[str, str], tuple[pd.DataFrame, pd.Series]] = {}
    for key, (frame, series) in residual_cache.items():
        local_mask = _rebalance_mask(frame.index, 15)
        reduced_residual[key] = (frame.loc[local_mask], series.loc[local_mask])
    return _ORIGINALS["decile_curves"](
        features.loc[mask], reduced_labels, reduced_masks, controls.loc[mask], reduced_residual, *args[5:], **kwargs
    )


def portfolio_prefiltered(*args: Any, **kwargs: Any) -> pd.DataFrame:
    features, labels, label_masks, controls = args[:4]
    mask = _rebalance_mask(features.index, 15)
    return _ORIGINALS["staggered_portfolio_proxy"](
        features.loc[mask],
        labels.loc[mask],
        {name: value.loc[mask] for name, value in label_masks.items()},
        controls.loc[mask],
        *args[4:],
        **kwargs,
    )


def _date_cost_cache_path() -> Path:
    return R.RESEARCH_ROOT / "derived_inputs" / "research_date_cost_v2_4.json"


def _estimate_date_cost(trade_date: str) -> int:
    total = 0
    for family in ("minute_nvg", "trade_nvg", "hawkes_lite", "hawkes_derived"):
        root = R.WAREHOUSE_ROOT / "features" / family
        if not root.exists():
            continue
        for part in root.glob(f"schema=*/date={trade_date}/*.parquet"):
            try:
                total += int(part.stat().st_size)
            except OSError:
                continue
    return total


def load_dates_lpt(start_date: str, end_date: str) -> list[str]:
    dates = _ORIGINALS["load_dates"](start_date, end_date)
    path = _date_cost_cache_path()
    cached: dict[str, int] = {}
    if path.exists():
        try:
            cached = {str(k): int(v) for k, v in json.loads(path.read_text(encoding="utf-8")).get("cost_bytes", {}).items()}
        except Exception:
            cached = {}
    changed = False
    for trade_date in dates:
        if trade_date not in cached:
            cached[trade_date] = _estimate_date_cost(trade_date)
            changed = True
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({"version": "file_bytes_lpt_v1", "cost_bytes": cached}, indent=2), encoding="utf-8")
        tmp.replace(path)
    # Longest-processing-time first minimizes the final campaign tail.
    return sorted(dates, key=lambda value: (-cached.get(value, 0), value))


def _profile_wrapper(stage: str, fn: Any) -> Any:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return _timed(stage, fn, *args, **kwargs)
    return wrapped


def run_date_profiled(*args: Any, **kwargs: Any) -> dict[str, Any]:
    _STAGE_SECONDS.clear()
    started = time.perf_counter()
    result = _ORIGINALS["run_date"](*args, **kwargs)
    total = time.perf_counter() - started
    trade_date = str(result.get("trade_date") or (args[0] if args else ""))
    out_root = Path(args[2] if len(args) > 2 else kwargs["out_root"])
    meta_path = R.unit_paths(out_root, trade_date)[0] / "meta.json"
    profile = {
        "fastpath_version": FASTPATH_VERSION,
        "reference_runner_version": REFERENCE_RUNNER_VERSION,
        "total_seconds": round(total, 6),
        "stage_seconds": {key: round(value, 6) for key, value in sorted(_STAGE_SECONDS.items())},
        "required_columns": research_required_columns_manifest(),
        "symbol_fold_cache": str(_symbol_fold.cache_info()),
        "scheduler": "LPT file-byte estimate",
    }
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["optimization_profile"] = profile
        R.atomic_write_json(meta_path, meta)
    result["optimization_profile"] = profile
    return result


def install_fastpath() -> None:
    if _ORIGINALS:
        return
    for name in (
        "add_all_features",
        "build_labels_and_masks",
        "join_daily_controls",
        "minute_rank_ic_summary",
        "decile_curves",
        "bundle_incremental_model_screen",
        "staggered_portfolio_proxy",
        "load_dates",
        "run_date",
    ):
        _ORIGINALS[name] = getattr(R, name)

    R.RESEARCH_VERSION = FASTPATH_VERSION
    R.BASE_RUNNER_VERSION = REFERENCE_RUNNER_VERSION
    R.__file__ = __file__  # worker subprocesses and run contracts point to this fast path.
    R.feature_sets = narrow_feature_sets
    R.add_all_features = _profile_wrapper("feature_build", add_all_features_fast)
    R.build_labels_and_masks = _profile_wrapper("labels", _ORIGINALS["build_labels_and_masks"])
    R.join_daily_controls = _profile_wrapper("controls", _ORIGINALS["join_daily_controls"])
    R.minute_rank_ic_summary = _profile_wrapper("ic_and_neutralization", minute_rank_ic_summary_fast)
    R.decile_curves = _profile_wrapper("deciles", decile_curves_prefiltered)
    R.bundle_incremental_model_screen = _profile_wrapper("oof", bundle_incremental_model_screen_fast)
    R.staggered_portfolio_proxy = _profile_wrapper("portfolio", portfolio_prefiltered)
    R.load_dates = load_dates_lpt
    R.run_date = run_date_profiled

    # Keep nested numerical libraries from oversubscribing inside date workers.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "1")


def main() -> int:
    install_fastpath()
    return R.main()


if __name__ == "__main__":
    raise SystemExit(main())
