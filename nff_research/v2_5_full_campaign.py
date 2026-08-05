"""v2.5 full campaign wrapper.

The v2.4 engine remains the authoritative daily factor/label/decile/gate
implementation.  This module adds the missing campaign layer: a derived NVG
supplement factor, PIT universe layers, temporal walk-forward OOS models,
Qlib Recorder artifacts, and an explicit account ledger.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import yaml

# Detached date workers execute this module by file path. Make the repository
# package importable without relying on an inherited PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nff_research import v2_4_optimized_runner as fast
from nff_research import v2_1_neutralized_runner as R


DERIVED_FEATURES = [
    "derived_supplement__nvg_directional_factor_15m",
    "derived_supplement__nvg_directional_factor_30m",
]
SUPPLEMENT_COMPONENTS = {
    15: [
        "price_nvg_15m_terminal_signed_edge_balance",
        "price_nvg_15m_terminal_long_edge_signed_slope",
        "price_detrended_nvg_15m_terminal_signed_edge_balance",
        "price_detrended_nvg_15m_terminal_long_edge_signed_slope",
        "volume_nvg_15m_terminal_signed_edge_balance",
        "volume_nvg_15m_terminal_long_edge_signed_slope",
    ],
    30: [
        "price_nvg_30m_terminal_signed_edge_balance",
        "price_nvg_30m_terminal_long_edge_signed_slope",
        "price_detrended_nvg_30m_terminal_signed_edge_balance",
        "price_detrended_nvg_30m_terminal_long_edge_signed_slope",
        "volume_nvg_30m_terminal_signed_edge_balance",
        "volume_nvg_30m_terminal_long_edge_signed_slope",
    ],
}
CONFIDENCE_COMPONENTS = {
    15: ["price_volume_nvg_15m_edge_weighted_jaccard", "price_volume_nvg_15m_common_edge_slope_corr"],
    30: ["price_volume_nvg_30m_edge_weighted_jaccard", "price_volume_nvg_30m_common_edge_slope_corr"],
}
_BASE_RUN_DATE: Any = None
_BASE_ANALYSIS_FEATURES: Any = None


def _utc(value: Any) -> pd.Series:
    return pd.to_datetime(value, utc=True, errors="coerce")


def _read_supplement(trade_date: str) -> pd.DataFrame:
    root = R.WAREHOUSE_ROOT / "nvg_supplement" / "minute_nvg_edge_raw" / "schema=v1" / f"date={trade_date}"
    paths = sorted(root.glob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    columns = ["symbol", "timestamp", *sum(SUPPLEMENT_COMPONENTS.values(), []), *sum(CONFIDENCE_COMPONENTS.values(), [])]
    available = set(pq.read_schema(paths[0]).names)
    selected = [c for c in columns if c in available]
    if not selected:
        return pd.DataFrame()
    data = pads.dataset([str(p) for p in paths], format="parquet").to_table(columns=selected).to_pandas()
    data["symbol"] = data["symbol"].astype("string").str.upper().str.strip()
    data["timestamp"] = _utc(data["timestamp"])
    return data.dropna(subset=["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"], keep="last")


def _cs_zscore(values: pd.Series, groups: pd.Series) -> pd.Series:
    mean = values.groupby(groups, sort=False).transform("mean")
    std = values.groupby(groups, sort=False).transform("std").replace(0, np.nan)
    return (values - mean) / std


def add_supplement_directional_factor(features: pd.DataFrame) -> pd.DataFrame:
    """Append only the directional consensus derived from base NVG + supplement."""
    if features.empty:
        return features
    result = features.copy()
    work = result.reset_index()
    work["__symbol"] = work["instrument"].astype("string").str.upper().str.strip()
    work["__timestamp"] = _utc(work["datetime"])
    for window in (15, 30):
        date_values = work["__timestamp"].dt.tz_convert(R.SESSION_TZ).dt.strftime("%Y-%m-%d")
        out = pd.Series(np.nan, index=work.index, dtype="float32")
        for trade_date in sorted(date_values.dropna().unique()):
            mask = date_values == trade_date
            part = work.loc[mask].copy()
            supplement = _read_supplement(str(trade_date))
            if supplement.empty:
                continue
            part = part.merge(supplement, left_on=["__symbol", "__timestamp"], right_on=["symbol", "timestamp"], how="left", sort=False)
            base = [
                f"minute_nvg__price_nvg_{window}m_top_bottom_asymmetry",
                f"minute_nvg__price_path_{window}m_signed_change",
                f"minute_nvg__price_nvg_{window}m_detrended_top_bottom_asymmetry",
            ]
            supplement_cols = [c for c in SUPPLEMENT_COMPONENTS[window] if c in part.columns]
            direction_cols = [c for c in base + supplement_cols if c in part.columns]
            if not direction_cols:
                continue
            z = pd.DataFrame(index=part.index)
            for col in direction_cols:
                z[col] = _cs_zscore(pd.to_numeric(part[col], errors="coerce"), part["__timestamp"])
            values = z.to_numpy(dtype="float64")
            valid = np.isfinite(values)
            count = valid.sum(axis=1)
            signed = np.where(valid, np.tanh(np.nan_to_num(values, nan=0.0)), np.nan)
            consensus = np.divide(np.nansum(signed, axis=1), count, out=np.zeros(len(part)), where=count > 0)
            completeness = count / max(len(direction_cols), 1)
            confidence_cols = [c for c in CONFIDENCE_COMPONENTS[window] if c in part.columns]
            confidence = part[confidence_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0.0).clip(-1, 1) if confidence_cols else 0.0
            factor = consensus * np.sqrt(completeness) * (0.5 + 0.5 * np.asarray(confidence))
            out.loc[part.index] = factor.astype("float32")
        result[f"derived_supplement__nvg_directional_factor_{window}m"] = out.to_numpy()
    return result


def _analysis_features(features: pd.DataFrame) -> list[str]:
    columns = list(_BASE_ANALYSIS_FEATURES(features)) if _BASE_ANALYSIS_FEATURES else []
    return columns + [c for c in DERIVED_FEATURES if c in features and features[c].notna().any()]


def _install() -> None:
    global _BASE_RUN_DATE, _BASE_ANALYSIS_FEATURES
    fast.install_fastpath()
    R.canonical_sets = lambda: {
        "bars_1m": {"schema_version": "v1", "columns": [c for c in ("open", "close", "volume", "dollar_volume", "vwap") if c in R.source_columns("canonical", "bars_1m", "v1")]},
        "trades_1m_core": {"schema_version": "v1", "columns": ["trade_count"]},
    }
    _BASE_RUN_DATE = R.run_date
    _BASE_ANALYSIS_FEATURES = R.analysis_features
    R.analysis_features = _analysis_features
    R.add_all_features = lambda frame: add_supplement_directional_factor(fast.add_all_features_fast(frame))
    base_universe_masks = R.universe_masks
    R.universe_masks = lambda features, controls: {
        key: value for key, value in base_universe_masks(features, controls).items() if key != "own_feature_universe"
    }
    # The v2.4 same-day fixed-symbol OOF screen is superseded in this
    # campaign by the temporal OOS stage below. Avoid paying for both
    # regressions per date; daily IC, deciles, costs and gates remain active.
    R.bundle_incremental_model_screen = lambda *args, **kwargs: pd.DataFrame()
    R.infer_bundle = lambda name: "MINUTE_NVG" if str(name).startswith("derived_supplement__") else R.V2.infer_bundle(name)
    stage_dir = None
    if "--out-root" in sys.argv:
        try:
            stage_dir = Path(sys.argv[sys.argv.index("--out-root") + 1]) / "worker_stages"
        except (IndexError, ValueError):
            stage_dir = None
    for stage_name in ("minute_rank_ic_summary", "decile_curves", "staggered_portfolio_proxy"):
        original = getattr(R, stage_name)
        def timed_stage(*args: Any, _original=original, _name=stage_name, **kwargs: Any) -> Any:
            started = time.perf_counter()
            if stage_dir is not None:
                stage_dir.mkdir(parents=True, exist_ok=True)
                (stage_dir / f"pid={os.getpid()}.json").write_text(json.dumps({"stage": _name, "state": "running", "started": started}), encoding="utf-8")
            try:
                return _original(*args, **kwargs)
            finally:
                if stage_dir is not None:
                    (stage_dir / f"pid={os.getpid()}.json").write_text(json.dumps({"stage": _name, "state": "complete", "elapsed_seconds": time.perf_counter() - started}), encoding="utf-8")
        setattr(R, stage_name, timed_stage)
    R.__file__ = str(Path(__file__).resolve())


def _model_cache_path(out_root: Path, trade_date: str) -> Path:
    return out_root / "model_cache" / f"date={trade_date}" / "dataset.parquet"


def _materialize_model_cache(trade_date: str, horizons: Sequence[int], out_root: Path, controls_path: Path) -> None:
    out_path = _model_cache_path(out_root, trade_date)
    success = out_path.with_name("_SUCCESS")
    if out_path.exists() and success.exists():
        return
    loader = R.NFFDataLoader(
        warehouse_root=R.WAREHOUSE_ROOT,
        canonical_sets=R.canonical_sets(),
        feature_sets=R.feature_sets(),
        execution={"frequency": "1min", "delay_bars": 1, "collision_policy": "latest"},
        label=None,
        join="inner",
        strict_manifests=True,
        allow_mixed_contracts=False,
        output_float32=True,
        arrow_use_threads=True,
    )
    start_time, end_time = R.loader_session_range(trade_date)
    frame = loader.load(instruments="all", start_time=start_time, end_time=end_time)
    features = R.add_all_features(frame["feature"].sort_index())
    labels, label_masks = R.build_labels_and_masks(features, [30])
    controls = R.join_daily_controls(features, trade_date, controls_path)
    masks = R.universe_masks(features, controls)
    selected = masks["final_trading_universe"] & R._rebalance_mask(features.index, 15)
    selected &= labels["return_vwap_to_vwap__h30"].notna() & label_masks["return_vwap_to_vwap__h30"].fillna(False)
    alpha = _analysis_features(features)
    columns = [c for c in alpha if c in features]
    out = features.loc[selected, columns].copy()
    out["label"] = labels.loc[selected, "return_vwap_to_vwap__h30"]
    out["vwap"] = features.loc[selected, "bars_1m__vwap"]
    out["dollar_volume"] = features.loc[selected, "bars_1m__dollar_volume"]
    out["trade_count"] = features.loc[selected, "trades_1m_core__trade_count"]
    out["hawkes_total_intensity"] = features.loc[selected, "hawkes_lite__hawkes_total_intensity"]
    out = out.reset_index().rename(columns={"instrument": "symbol", "datetime": "decision_time"})
    out["trade_date"] = trade_date
    out["symbol"] = out["symbol"].astype("string").str.upper().str.strip()
    metadata_path = R.INDUSTRY_METADATA_PATH
    if metadata_path is not None and metadata_path.exists():
        metadata = pd.read_parquet(metadata_path, columns=["symbol", "sector_code", "industry_code", "market_cap"])
        metadata["symbol"] = metadata["symbol"].astype("string").str.upper().str.strip()
        out = out.merge(metadata.drop_duplicates("symbol"), on="symbol", how="left", sort=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")
    success.write_text(json.dumps({"trade_date": trade_date, "rows": len(out), "features": columns}), encoding="utf-8")


def _run_date_with_cache(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = _BASE_RUN_DATE(*args, **kwargs)
    trade_date = str(args[0] if args else kwargs["trade_date"])
    horizons = args[1] if len(args) > 1 else kwargs["horizons"]
    out_root = Path(args[2] if len(args) > 2 else kwargs["out_root"])
    controls_path = Path(args[3] if len(args) > 3 else kwargs["controls_path"])
    if result.get("status") in {"completed", "skipped"}:
        _materialize_model_cache(trade_date, horizons, out_root, controls_path)
    return result


def _folds(dates: list[str], train: int = 60, validation: int = 10, test: int = 10, step: int = 10) -> list[dict[str, Any]]:
    result = []
    start = train + validation
    fold_id = 0
    while start + test <= len(dates):
        result.append({"fold_id": fold_id, "train": dates[start - validation - train : start - validation], "validation": dates[start - validation : start], "test": dates[start : start + test]})
        fold_id += 1
        start += step
    return result


def _read_cache(paths: dict[str, Path], dates: Sequence[str], columns: Sequence[str]) -> pd.DataFrame:
    frames = []
    for date in dates:
        path = paths.get(date)
        if path is None or not path.exists():
            continue
        frame = pd.read_parquet(path, columns=[c for c in columns if c in pq.read_schema(path).names])
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(columns))


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train_x, axis=0)
    std = np.nanstd(train_x, axis=0)
    std[~np.isfinite(std) | (std < 1e-8)] = 1.0
    x = np.nan_to_num((train_x - mean) / std)
    y = np.asarray(train_y, dtype="float64")
    xtx = x.T @ x + float(alpha) * np.eye(x.shape[1])
    coef = np.linalg.solve(xtx, x.T @ y)
    return mean, std, coef


def _predict_ridge(x: np.ndarray, fitted: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std, coef = fitted
    return np.nan_to_num((x - mean) / std) @ coef


def _rank_ic(frame: pd.DataFrame, prediction: str = "prediction") -> float:
    values = []
    for _, block in frame.groupby("decision_time", sort=False):
        block = block[[prediction, "label"]].dropna()
        if len(block) < 30:
            continue
        values.append(block[prediction].rank().corr(block["label"].rank()))
    return float(np.nanmean(values)) if values else math.nan


def _account_replay(predictions: pd.DataFrame, run_root: Path, initial_equity: float = 1_000_000.0, participation: float = 0.05, one_way_bps: float = 2.5) -> dict[str, Any]:
    cash = float(initial_equity)
    positions: dict[str, float] = {}
    orders: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    previous_prices: dict[str, float] = {}
    for timestamp, block in predictions.sort_values("decision_time").groupby("decision_time", sort=True):
        block = block.dropna(subset=["prediction", "vwap", "dollar_volume"])
        if len(block) < 40:
            continue
        block = block.loc[block["hawkes_total_intensity"].rank(pct=True).fillna(0) <= 0.80]
        n = max(1, int(len(block) * 0.05))
        long = block.nsmallest(n, "prediction")
        short = block.nlargest(n, "prediction")
        target: dict[str, float] = {str(s): 0.5 / n for s in long["symbol"]}
        target.update({str(s): -0.5 / n for s in short["symbol"]})
        prices = {str(row.symbol): float(row.vwap) for row in block.itertuples() if np.isfinite(row.vwap) and row.vwap > 0}
        equity_before = cash + sum(shares * prices.get(symbol, previous_prices.get(symbol, 0.0)) for symbol, shares in positions.items())
        current_values = {symbol: shares * prices.get(symbol, previous_prices.get(symbol, 0.0)) for symbol, shares in positions.items()}
        for symbol in set(positions) | set(target):
            px = prices.get(symbol, previous_prices.get(symbol, np.nan))
            if not np.isfinite(px) or px <= 0:
                continue
            current_weight = current_values.get(symbol, 0.0) / max(equity_before, 1.0)
            desired_weight = target.get(symbol, 0.0)
            order_value = (desired_weight - current_weight) * equity_before
            row = block.loc[block["symbol"] == symbol]
            available = float(row["dollar_volume"].iloc[0]) if not row.empty else 0.0
            max_value = max(0.0, available * participation)
            fill_value = float(np.sign(order_value) * min(abs(order_value), max_value))
            fill_shares = fill_value / px if px else 0.0
            if abs(fill_shares) < 1e-10:
                continue
            impact_bps = 5.0 * math.sqrt(min(1.0, abs(fill_value) / max(available, 1.0)))
            execution_px = px * (1.0 + np.sign(fill_shares) * (one_way_bps + impact_bps) / 10000.0)
            commission = abs(fill_value) * 0.2 / 10000.0
            spread = abs(fill_value) * one_way_bps / 10000.0
            impact = abs(fill_value) * impact_bps / 10000.0
            orders.append({"decision_time": timestamp, "symbol": symbol, "target_value": order_value, "submitted_value": order_value, "filled_value": fill_value, "unfilled_value": order_value - fill_value, "participation": abs(fill_value) / max(available, 1.0)})
            fills.append({"decision_time": timestamp, "symbol": symbol, "shares": fill_shares, "fill_price": execution_px, "notional": abs(fill_value), "commission": commission, "spread": spread, "impact": impact, "borrow": 0.0})
            cash -= fill_shares * execution_px + commission
            positions[symbol] = positions.get(symbol, 0.0) + fill_shares
            if abs(positions[symbol]) < 1e-10:
                positions.pop(symbol, None)
        short_value = sum(abs(shares * prices.get(symbol, previous_prices.get(symbol, 0.0))) for symbol, shares in positions.items() if shares < 0)
        borrow = short_value * 50.0 / 10000.0 / 252.0 * 15.0 / 390.0
        cash -= borrow
        equity = cash + sum(shares * prices.get(symbol, previous_prices.get(symbol, 0.0)) for symbol, shares in positions.items())
        snapshots.append({"decision_time": timestamp, "cash": cash, "equity": equity, "equity_before": equity_before, "gross_exposure": sum(abs(shares * prices.get(symbol, previous_prices.get(symbol, 0.0))) for symbol, shares in positions.items()) / max(equity, 1.0), "net_exposure": sum(shares * prices.get(symbol, previous_prices.get(symbol, 0.0)) for symbol, shares in positions.items()) / max(equity, 1.0), "borrow": borrow, "position_count": len(positions), "position_drift": equity - equity_before - sum(-f["commission"] - f["spread"] - f["impact"] for f in fills if f["decision_time"] == timestamp)})
        previous_prices = prices
    orders_df = pd.DataFrame(orders)
    fills_df = pd.DataFrame(fills)
    snaps = pd.DataFrame(snapshots)
    run_root.joinpath("account").mkdir(parents=True, exist_ok=True)
    orders_df.to_parquet(run_root / "account" / "orders.parquet", index=False)
    fills_df.to_parquet(run_root / "account" / "fills.parquet", index=False)
    snaps.to_parquet(run_root / "account" / "snapshots.parquet", index=False)
    if snaps.empty:
        return {"status": "NO_OOS_ROWS"}
    daily = snaps.assign(decision_time=pd.to_datetime(snaps["decision_time"], utc=True)).set_index("decision_time")["equity"].resample("1D").last().dropna()
    ret = daily.pct_change().dropna()
    drawdown = daily / daily.cummax() - 1.0
    costs = {"commission": float(fills_df["commission"].sum()) if not fills_df.empty else 0.0, "spread": float(fills_df["spread"].sum()) if not fills_df.empty else 0.0, "impact": float(fills_df["impact"].sum()) if not fills_df.empty else 0.0, "borrow": float(snaps["borrow"].sum())}
    return {"start_equity": float(daily.iloc[0]), "end_equity": float(daily.iloc[-1]), "net_return": float(daily.iloc[-1] / daily.iloc[0] - 1.0), "sharpe": float(ret.mean() / ret.std() * math.sqrt(252)) if len(ret) > 1 and ret.std() > 0 else math.nan, "max_drawdown": float(drawdown.min()), "days": int(len(daily)), "orders": int(len(orders_df)), "fills": int(len(fills_df)), "fill_ratio": float(fills_df["notional"].sum() / orders_df["submitted_value"].abs().sum()) if not orders_df.empty and orders_df["submitted_value"].abs().sum() > 0 else math.nan, "turnover_notional": float(fills_df["notional"].sum()) if not fills_df.empty else 0.0, "costs": costs, "cost_after_return": float((daily.iloc[-1] - sum(costs.values())) / daily.iloc[0] - 1.0)}


def run_temporal_oos(run_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    cache_root = run_root / "model_cache"
    paths = {p.parent.name.removeprefix("date="): p for p in cache_root.glob("date=*/dataset.parquet") if p.with_name("_SUCCESS").exists()}
    dates = sorted(paths)
    folds = _folds(dates, int(config.get("walk_forward", {}).get("train_days", 60)), int(config.get("walk_forward", {}).get("validation_days", 10)), int(config.get("walk_forward", {}).get("test_days", 10)), int(config.get("walk_forward", {}).get("step_days", 10)))
    if not folds:
        return {"status": "MODEL_FAILED", "reason": "no temporal folds"}
    sample = pd.read_parquet(next(iter(paths.values())))
    features = [c for c in DERIVED_FEATURES + list(R.REPRESENTATIVE_ALPHA_FEATURES) if c in sample.columns]
    primary = "return_vwap_to_vwap__h30"
    all_predictions = []
    fold_rows = []
    for fold in folds:
        train = _read_cache(paths, fold["train"], features + ["label", "symbol", "decision_time", "sector_code", "industry_code", "market_cap"])
        validation = _read_cache(paths, fold["validation"], features + ["label", "symbol", "decision_time", "sector_code", "industry_code", "market_cap"])
        test = _read_cache(paths, fold["test"], features + ["label", "symbol", "decision_time", "sector_code", "industry_code", "market_cap", "vwap", "dollar_volume", "trade_count", "hawkes_total_intensity"])
        train = train.dropna(subset=features + ["label"])
        validation = validation.dropna(subset=features + ["label"])
        test = test.dropna(subset=features + ["label"])
        if len(train) < 100 or test.empty:
            continue
        fitted = _fit_ridge(train[features].to_numpy(dtype="float64"), train["label"].to_numpy(dtype="float64"), 0.1)
        validation = validation.copy(); validation["prediction"] = _predict_ridge(validation[features].to_numpy(dtype="float64"), fitted)
        test = test.copy(); test["prediction"] = _predict_ridge(test[features].to_numpy(dtype="float64"), fitted)
        test["fold_id"] = fold["fold_id"]
        test["model"] = "ridge"
        all_predictions.append(test)
        fold_rows.append({"fold_id": fold["fold_id"], "train_start": fold["train"][0], "train_end": fold["train"][-1], "validation_start": fold["validation"][0], "validation_end": fold["validation"][-1], "test_start": fold["test"][0], "test_end": fold["test"][-1], "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test), "validation_rank_ic": _rank_ic(validation), "test_rank_ic": _rank_ic(test), "feature_count": len(features)})
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    predictions.to_parquet(run_root / "predictions_oos.parquet", index=False)
    pd.DataFrame(fold_rows).to_csv(run_root / "walk_forward_fold_metrics.csv", index=False)
    account = _account_replay(predictions, run_root)
    (run_root / "account_metrics.json").write_text(json.dumps(account, indent=2, default=str), encoding="utf-8")
    recorder = {"status": "not_attempted", "folds": len(fold_rows)}
    try:
        import qlib
        from qlib.workflow import R as QlibR
        qlib.init(provider_uri=str(run_root / "qlib_provider"), region="us")
        for row in fold_rows:
            with QlibR.start(experiment_name="nff_v2_5_full_campaign", recorder_name=f"ridge_fold_{row['fold_id']:03d}"):
                QlibR.log_params(**{"research_version": "2.5", "model": "ridge", "fold_id": row["fold_id"], "feature_count": row["feature_count"], "primary_label": primary, "git_commit": _git_commit()})
                QlibR.log_metrics(test_rank_ic=float(row["test_rank_ic"]), validation_rank_ic=float(row["validation_rank_ic"]))
        recorder = {"status": "complete", "experiment_name": "nff_v2_5_full_campaign", "folds": len(fold_rows)}
    except Exception as exc:
        recorder = {"status": "FAILED", "error": repr(exc), "folds": len(fold_rows)}
    (run_root / "recorder_audit.json").write_text(json.dumps(recorder, indent=2, default=str), encoding="utf-8")
    return {"status": "complete", "dates": len(dates), "folds": fold_rows, "prediction_rows": len(predictions), "account": account, "recorder": recorder, "feature_count": len(features)}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _preflight(run_root: Path, config: dict[str, Any]) -> None:
    inventory = []
    for dataset in ("minute_nvg", "trade_nvg", "hawkes_lite", "hawkes_derived"):
        for schema in ("v1", "v2", "v3", "v4"):
            try:
                cols = R.source_columns("feature", dataset, schema)
            except Exception:
                cols = []
            for col in cols:
                inventory.append({"dataset": dataset, "schema": schema, "field": col, "role": "QUALITY_GATE" if R.QUALITY_OR_CONTROL_RE.search(col) else "CANDIDATE_FIELD", "status": "INVENTORIED"})
    report_root = run_root / "reports" / "factor_registry"
    report_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory).drop_duplicates(["dataset", "schema", "field"]).to_parquet(report_root / "source_field_inventory.parquet", index=False)
    pd.DataFrame(inventory).drop_duplicates(["dataset", "schema", "field"]).to_csv(report_root / "source_field_inventory.csv", index=False)
    assumptions = {"research_version": "2.5", "git_commit": _git_commit(), "config": config, "known_data_limits": ["quotes/order-book tape is unavailable; spread and impact are estimated proxies", "the implemented primary multifactor model is temporal OOS Ridge; additional model names remain diagnostic until trained"], "universe_policy": "all PIT and ADV layers are parallel diagnostics; final account replay uses Top1000-equivalent final_trading_universe"}
    (run_root / "reports" / "assumptions.json").parent.mkdir(parents=True, exist_ok=True)
    (run_root / "reports" / "assumptions.json").write_text(json.dumps(assumptions, indent=2, default=str), encoding="utf-8")


def _write_final_report(run_root: Path, model_result: dict[str, Any], config: dict[str, Any]) -> None:
    lines = [
        "# NFF v2.5 Full Campaign Report",
        "",
        "This campaign combines the v2.4 55-feature multi-label factor screen with a derived NVG supplement factor, PIT universe layers, temporal OOS Ridge and an explicit account ledger.",
        "",
        "## Scope",
        f"- Research dates: `{config['run']['start_date']}` to `{config['run']['end_date']}`",
        "- Alpha screen: 55 representative factors plus two derived NVG supplement directional factors; readiness/activity axes remain controls or gates.",
        "- Labels: open/open, VWAP/VWAP, close/close, liquidity deterioration, realized volatility, jump-tail diagnostic, execution-cost mechanical diagnostic.",
        "- Universes: all PIT eligible, common structural, ADV Top500/1000/2000/3000, final trading universe.",
        "- Neutralization: numeric controls plus sector/industry metadata where present; raw and residualized IC are both retained.",
        "",
        "## Walk-forward OOS",
        f"- Status: `{model_result.get('status')}`; folds: `{len(model_result.get('folds', []))}`; prediction rows: `{model_result.get('prediction_rows', 0)}`.",
        "- Primary model: Ridge trained only on earlier train/validation dates; test predictions are chronological OOS.",
        pd.DataFrame(model_result.get("folds", [])).to_markdown(index=False) if model_result.get("folds") else "No completed temporal folds.",
        "",
        "## Account",
        "The account artifacts include orders, fills, cash/equity snapshots, participation-limited fills, estimated spread/impact/commission and borrow costs. These are estimated execution costs because the warehouse does not contain a complete quote/order-book tape.",
        "```json",
        json.dumps(model_result.get("account", {}), indent=2, default=str),
        "```",
        "",
        "## Recorder",
        json.dumps(model_result.get("recorder", {}), indent=2, default=str),
        "",
        "## Interpretation policy",
        "A factor is not promoted by IC alone. Final assessment must combine PIT validity, raw-vs-neutralized stability, decile monotonicity, turnover, cost-after return, OOS prediction, gate impact, drawdown and capacity. Missing quote-derived costs are explicitly ESTIMATED_COST, not broker-realized costs.",
    ]
    (run_root / "reports").mkdir(parents=True, exist_ok=True)
    (run_root / "reports" / "final_report_v2_5.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_root = Path(config["local_paths"]["research_root"]) / "runs" / config["run"]["name"]
    run_root.mkdir(parents=True, exist_ok=True)
    _preflight(run_root, config)
    # The reference scheduler accepts the v2.5 config but uses a backwards-
    # compatible default run prefix. Pass the explicit campaign root so the
    # model cache, daily outputs and final report share one durable identity.
    original_argv = list(sys.argv)
    if "--out-root" not in sys.argv:
        sys.argv.extend(["--out-root", str(run_root)])
    try:
        rc = R.main()
    finally:
        sys.argv[:] = original_argv
    if rc != 0:
        return rc
    model = run_temporal_oos(run_root, config)
    _write_final_report(run_root, model, config)
    status = {"phase": "complete" if model.get("status") == "complete" else "partial", "daily_phase_returncode": rc, "temporal_oos": model, "finished_at": datetime.now(timezone.utc).isoformat()}
    (run_root / "status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    return 0 if model.get("status") == "complete" else 4


def main() -> int:
    if "--worker" in sys.argv:
        _install()
        return R.main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v2_5_full_campaign.yaml")
    args = parser.parse_args()
    _install()
    return run(Path(args.config).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
