from __future__ import annotations

"""NFF v2.8 staged research pipeline.

The v2.7 runner executes load/derive, IC, neutralization, deciles and portfolio
inside one date process.  This module keeps the warehouse-exact v2.7 factor
math, but turns those operations into independent restartable stages:

    materialize -> basic_screen -> select -> detailed -> portfolio

All 464 physical factors enter the cheap raw screen.  Only candidates selected
from a frozen training window enter neutralization, deciles and portfolio
simulation.  Selection directions are frozen before any portfolio date, so the
pipeline never uses same-day results to choose the same-day trade direction.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import psutil
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nff_research import v2_7_launch as V27

C = V27.C
R = C.R
V26 = C.V26
VERSION = "2.8-pipelined-selection"


@dataclass(frozen=True)
class StageSpec:
    name: str
    max_workers: int
    estimated_worker_gb: float


_BOOTSTRAPPED = False
_CONFIG: dict[str, Any] = {}


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.part")
    try:
        temp.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path, *, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.part")
    try:
        frame.to_parquet(temp, index=index, compression="zstd")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return data


def _run_root(config: Mapping[str, Any]) -> Path:
    return (
        Path(config["local_paths"]["research_root"])
        / "runs"
        / str(config["run"]["name"])
    )


def _controls_path(config: Mapping[str, Any]) -> Path:
    explicit = config.get("pipeline", {}).get("controls_path")
    if explicit:
        return Path(str(explicit))
    return (
        Path(config["local_paths"]["research_root"])
        / "derived_inputs"
        / "daily_bar_controls_v2_1"
        / f"daily_controls_{config['run']['start_date']}_{config['run']['end_date']}.parquet"
    )


def _contract_hash(config: Mapping[str, Any]) -> str:
    return _json_hash(
        {
            "version": VERSION,
            "v27_version": getattr(C, "VERSION", "unknown"),
            "factor_registry": list(getattr(V26, "FULL_FACTOR_NAMES", ())),
            "pipeline": config.get("pipeline", {}),
            "selection": config.get("selection", {}),
            "labels": config.get("labels", {}),
            "warehouse_root": config.get("local_paths", {}).get("warehouse_root"),
        }
    )


def bootstrap(config: dict[str, Any]) -> None:
    global _BOOTSTRAPPED, _CONFIG
    if _BOOTSTRAPPED:
        return
    _CONFIG = config
    R.apply_config_globals(config)
    V26._patch()
    V26._wrap_run_date()
    C.configure_registry(config)
    C.install(config)
    _BOOTSTRAPPED = True


def _pipeline_root(config: Mapping[str, Any]) -> Path:
    return _run_root(config) / "pipeline"


def _date_root(config: Mapping[str, Any], trade_date: str) -> Path:
    return _pipeline_root(config) / "dates" / f"date={trade_date}"


def _stage_root(config: Mapping[str, Any], stage: str, trade_date: str) -> Path:
    return _date_root(config, trade_date) / stage


def _stage_success(config: Mapping[str, Any], stage: str, trade_date: str) -> Path:
    return _stage_root(config, stage, trade_date) / "_SUCCESS"


def _dates(config: Mapping[str, Any]) -> list[str]:
    # Warehouse catalog enumeration is not a training-window ordering
    # contract; freeze candidates on an explicit chronological sequence.
    return sorted(
        str(value)
        for value in R.load_dates(
            str(config["run"]["start_date"]), str(config["run"]["end_date"])
        )
    )


def _factor_manifests(config: Mapping[str, Any], trade_date: str) -> list[Path]:
    root = _run_root(config) / "atomic_checkpoints" / f"date={trade_date}" / "factors"
    return sorted(root.glob("family=*/window=*/manifest.json"))


def _factor_block_inventory(config: Mapping[str, Any], trade_date: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest_path in _factor_manifests(config, trade_date):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        for block in manifest.get("blocks", []):
            path = manifest_path.parent / str(block["path"])
            for column in block.get("columns", []):
                rows.append(
                    {
                        "feature": str(column),
                        "path": str(path),
                        "family": manifest.get("family"),
                        "window": manifest.get("window"),
                    }
                )
    return pd.DataFrame(rows)


def _load_selected_factors(
    config: Mapping[str, Any],
    trade_date: str,
    selected: Iterable[str],
    expected_index: pd.Index,
) -> pd.DataFrame:
    wanted = set(str(value) for value in selected)
    if not wanted:
        return pd.DataFrame(index=expected_index)
    inventory = _factor_block_inventory(config, trade_date)
    if inventory.empty:
        raise RuntimeError(f"no factor blocks for {trade_date}")
    inventory = inventory[inventory["feature"].isin(wanted)]
    missing = wanted - set(inventory["feature"])
    if missing:
        raise RuntimeError(f"missing selected factors for {trade_date}: {sorted(missing)[:20]}")
    parts: list[pd.DataFrame] = []
    for path, group in inventory.groupby("path", sort=False):
        columns = list(group["feature"])
        part = pd.read_parquet(path, columns=columns)
        part.index = expected_index
        parts.append(part)
    result = pd.concat(parts, axis=1, copy=False)
    return result.loc[:, [name for name in wanted if name in result]]


def _iter_factor_blocks(
    config: Mapping[str, Any], trade_date: str, expected_index: pd.Index
) -> Iterable[pd.DataFrame]:
    seen: set[str] = set()
    for manifest_path in _factor_manifests(config, trade_date):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        for block in manifest.get("blocks", []):
            path = manifest_path.parent / str(block["path"])
            part = pd.read_parquet(path)
            part.index = expected_index
            columns = [column for column in part.columns if column not in seen]
            if not columns:
                continue
            seen.update(columns)
            yield part[columns]


def _support_columns(features: pd.DataFrame) -> list[str]:
    exact = {
        "bars_1m__open",
        "bars_1m__close",
        "bars_1m__vwap",
        "bars_1m__volume",
        "bars_1m__dollar_volume",
        "trades_1m_core__trade_count",
        "trades_1m_core__dollar_volume",
        "trades_1m_core__signed_dollar_flow_proxy",
        "trade_nvg__trade_active_second_ratio_300s",
        "trade_nvg__trade_price_stale_ratio_300s",
        "trade_nvg__trade_observation_coverage_300s",
    }
    exact.update(
        str(spec["column"])
        for spec in getattr(R, "HAWKES_GATE_SPECS", ())
        if isinstance(spec, Mapping) and spec.get("column")
    )
    prefixes = ("hawkes_lite__hawkes_ready", "hawkes_lite__hawkes_warmup")
    return [
        column
        for column in features.columns
        if column in exact or any(column.startswith(prefix) for prefix in prefixes)
    ]


def materialize_date(config: dict[str, Any], trade_date: str) -> dict[str, Any]:
    bootstrap(config)
    stage_root = _stage_root(config, "materialize", trade_date)
    success = _stage_success(config, "materialize", trade_date)
    contract = _contract_hash(config)
    meta_path = stage_root / "meta.json"
    if success.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("contract_hash") == contract:
            return {"trade_date": trade_date, "stage": "materialize", "status": "skipped"}

    started = time.perf_counter()
    stage_root.mkdir(parents=True, exist_ok=True)
    atomic = config.get("atomic", {})
    C.CTX = C.Context(
        trade_date=trade_date,
        out_root=_run_root(config),
        contract_hash=contract,
        factor_block_size=int(atomic.get("factor_block_size", 8)),
        intra_workers=int(atomic.get("intra_date_workers", 4)),
    )
    C.CTX.root.mkdir(parents=True, exist_ok=True)
    C.write_schema_audit(R.WAREHOUSE_ROOT, C.CTX.root / "schema", trade_date)
    C._event("pipeline_materialize", "running")
    try:
        loader = R.NFFDataLoader(
            warehouse_root=R.WAREHOUSE_ROOT,
            canonical_sets=R.canonical_sets(),
            feature_sets=R.feature_sets(),
            execution={"frequency": "1min", "delay_bars": 1, "collision_policy": "latest"},
            label=None,
            join="inner",
            strict_manifests=True,
            allow_mixed_contracts=bool(config["run"].get("allow_mixed_contracts", True)),
            output_float32=True,
            arrow_use_threads=True,
        )
        start_time, end_time = R.loader_session_range(trade_date)
        loaded = loader.load(instruments="all", start_time=start_time, end_time=end_time)
        features = R.add_all_features(loaded["feature"].sort_index())
        del loaded
        gc.collect()
        horizons = [int(value) for value in config["run"]["horizons"]]
        labels, label_masks = R.build_labels_and_masks(features, horizons)
        controls = R.join_daily_controls(features, trade_date, _controls_path(config))
        universes = pd.DataFrame(R.universe_masks(features, controls), index=features.index)
        analysis = list(R.analysis_features(features))
        expected = set(V26.FULL_FACTOR_NAMES)
        missing = expected - set(analysis)
        if missing:
            raise RuntimeError(f"physical factor completion failed for {trade_date}: {len(missing)} missing")
        support_columns = _support_columns(features)
        support = features[support_columns].copy(deep=False)
        registry = R.feature_registry(features, analysis, trade_date)

        _atomic_parquet(support, stage_root / "support.parquet")
        _atomic_parquet(labels, stage_root / "labels.parquet")
        _atomic_parquet(pd.DataFrame(label_masks, index=features.index), stage_root / "label_masks.parquet")
        _atomic_parquet(controls, stage_root / "controls.parquet")
        _atomic_parquet(universes.astype("boolean"), stage_root / "universes.parquet")
        _atomic_parquet(registry, stage_root / "feature_registry.parquet", index=False)
        inventory = _factor_block_inventory(config, trade_date)
        if len(set(inventory.get("feature", ()))) != len(expected):
            raise RuntimeError(
                f"factor block inventory mismatch for {trade_date}: "
                f"{len(set(inventory.get('feature', ())))}/{len(expected)}"
            )
        _atomic_parquet(inventory, stage_root / "factor_block_inventory.parquet", index=False)
        meta = {
            "version": VERSION,
            "trade_date": trade_date,
            "stage": "materialize",
            "status": "complete",
            "contract_hash": contract,
            "rows": int(len(features)),
            "physical_factor_count": int(len(expected)),
            "support_columns": support_columns,
            "label_columns": list(labels.columns),
            "universe_columns": list(universes.columns),
            "elapsed_seconds": time.perf_counter() - started,
            "loader_report": loader.last_load_report,
        }
        _atomic_json(meta_path, meta)
        success.write_text(pd.Timestamp.utcnow().isoformat(), encoding="utf-8")
        C._event("pipeline_materialize", "complete", elapsed_seconds=meta["elapsed_seconds"])
        return meta
    except BaseException as exc:
        C._event("pipeline_materialize", "failed", error=repr(exc))
        raise
    finally:
        C.FUTURE_CACHE.clear()
        C.FUTURE_WIDE_CACHE.clear()
        C.CTX = None
        gc.collect()


def _load_materialized(
    config: Mapping[str, Any], trade_date: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = _stage_root(config, "materialize", trade_date)
    if not (root / "_SUCCESS").exists():
        raise RuntimeError(f"materialize stage incomplete for {trade_date}")
    support = pd.read_parquet(root / "support.parquet")
    labels = pd.read_parquet(root / "labels.parquet")
    masks = pd.read_parquet(root / "label_masks.parquet").astype("boolean")
    controls = pd.read_parquet(root / "controls.parquet")
    universes = pd.read_parquet(root / "universes.parquet").astype("boolean")
    for frame in (labels, masks, controls, universes):
        frame.index = support.index
    return support, labels, masks, controls, universes


def _screen_label_columns(config: Mapping[str, Any], labels: pd.DataFrame) -> list[str]:
    screen = config.get("selection", {}).get("screen_labels")
    if not screen:
        return list(labels.columns)
    allowed: list[str] = []
    for item in screen:
        family = str(item.get("family"))
        horizons = {int(value) for value in item.get("horizons", [])}
        for column in labels.columns:
            if "__h" not in column:
                continue
            current_family, horizon = column.rsplit("__h", 1)
            if current_family == family and (not horizons or int(horizon) in horizons):
                allowed.append(column)
    return list(dict.fromkeys(allowed))


def basic_screen_date(config: dict[str, Any], trade_date: str) -> dict[str, Any]:
    bootstrap(config)
    root = _stage_root(config, "basic_screen", trade_date)
    success = _stage_success(config, "basic_screen", trade_date)
    contract = _contract_hash(config)
    meta_path = root / "meta.json"
    if success.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("contract_hash") == contract:
            return {"trade_date": trade_date, "stage": "basic_screen", "status": "skipped"}
    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    support, labels, label_masks, _, universes = _load_materialized(config, trade_date)
    label_columns = _screen_label_columns(config, labels)
    universe_names = list(
        config.get("selection", {}).get(
            "screen_universes", ["all_pit_eligible", "final_trading_universe"]
        )
    )
    min_n = int(config["run"].get("min_cross_section_n", 30))
    rows: list[dict[str, Any]] = []

    for block in _iter_factor_blocks(config, trade_date, support.index):
        feature_columns = list(block.columns)
        for universe in universe_names:
            if universe not in universes:
                continue
            universe_mask = universes[universe].fillna(False)
            for label_column in label_columns:
                base_mask = (
                    universe_mask
                    & labels[label_column].notna()
                    & label_masks[label_column].fillna(False)
                ).fillna(False)
                if int(base_mask.sum()) < min_n:
                    continue
                factor_frame = block.loc[base_mask]
                label = labels.loc[base_mask, label_column]
                ranked = C._rank_frame_by_datetime_average(factor_frame)
                minute_stats, pooled_stats = C._ranked_ic_stats_from_ranked_features(
                    ranked, label, feature_columns, min_n
                )
                family, horizon_text = label_column.rsplit("__h", 1)
                coverage = factor_frame.notna().mean()
                for method, stats in (
                    ("minute_mean_cs_rank_ic", minute_stats),
                    ("pooled_cs_demeaned_pct_rank_ic", pooled_stats),
                ):
                    for feature in feature_columns:
                        item = stats[feature]
                        rows.append(
                            {
                                "trade_date": trade_date,
                                "feature": feature,
                                "factor_family": _factor_family(feature),
                                "universe": universe,
                                "label_family": family,
                                "horizon_bars": int(horizon_text),
                                "rank_ic_method": method,
                                "rank_ic_mean": item["rank_ic_mean"],
                                "rank_ic_std": item["rank_ic_std"],
                                "rank_ic_positive_ratio": item["rank_ic_positive_ratio"],
                                "ic_minutes": item["ic_minutes"],
                                "ic_count": item["ic_count"],
                                "coverage": float(coverage.get(feature, math.nan)),
                            }
                        )
        del block
        gc.collect()

    result = pd.DataFrame(rows)
    _atomic_parquet(result, root / "basic_factor_screen.parquet", index=False)
    meta = {
        "version": VERSION,
        "trade_date": trade_date,
        "stage": "basic_screen",
        "status": "complete",
        "contract_hash": contract,
        "rows": int(len(result)),
        "feature_count": int(result["feature"].nunique()) if not result.empty else 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(meta_path, meta)
    success.write_text(pd.Timestamp.utcnow().isoformat(), encoding="utf-8")
    return meta


def _factor_family(feature: str) -> str:
    pieces = str(feature).split("__")
    if len(pieces) >= 2 and pieces[0] == "full_factor" and pieces[1]:
        return pieces[1][0].upper()
    return "OTHER"


def _candidate_rows(
    aggregated: pd.DataFrame,
    *,
    max_features: int,
    min_per_family: int,
    max_per_family: int,
) -> pd.DataFrame:
    if aggregated.empty:
        return aggregated
    ordered = aggregated.sort_values(
        ["selection_score", "valid_days", "coverage_mean", "feature"],
        ascending=[False, False, False, True],
    )
    selected: list[int] = []
    family_counts: dict[str, int] = {}
    for family, group in ordered.groupby("factor_family", sort=True):
        for index in group.head(min_per_family).index:
            if len(selected) >= max_features:
                break
            selected.append(index)
            family_counts[family] = family_counts.get(family, 0) + 1
    for index, row in ordered.iterrows():
        if len(selected) >= max_features:
            break
        if index in selected:
            continue
        family = str(row["factor_family"])
        if family_counts.get(family, 0) >= max_per_family:
            continue
        selected.append(index)
        family_counts[family] = family_counts.get(family, 0) + 1
    return ordered.loc[selected].sort_values("selection_score", ascending=False).reset_index(drop=True)


def select_candidates(config: dict[str, Any]) -> dict[str, Any]:
    bootstrap(config)
    root = _pipeline_root(config) / "selection"
    root.mkdir(parents=True, exist_ok=True)
    contract = _contract_hash(config)
    success = root / "_SUCCESS"
    meta_path = root / "meta.json"
    if success.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("contract_hash") == contract:
            return {"stage": "select", "status": "skipped", **meta}

    available_dates = [
        trade_date
        for trade_date in _dates(config)
        if _stage_success(config, "basic_screen", trade_date).exists()
    ]
    selection = config.get("selection", {})
    train_days = int(selection.get("train_days", 60))
    if len(available_dates) < train_days:
        raise RuntimeError(
            f"candidate selection requires {train_days} completed screen dates; "
            f"found {len(available_dates)}"
        )
    train_dates = available_dates[:train_days]
    frames = [
        pd.read_parquet(_stage_root(config, "basic_screen", date) / "basic_factor_screen.parquet")
        for date in train_dates
    ]
    data = pd.concat(frames, ignore_index=True)
    primary_family = str(selection.get("primary_label", "return_vwap_to_vwap"))
    primary_horizons = {int(value) for value in selection.get("primary_horizons", [5, 15, 30, 60])}
    primary_universe = str(selection.get("primary_universe", "final_trading_universe"))
    data = data[
        (data["label_family"] == primary_family)
        & (data["horizon_bars"].isin(primary_horizons))
        & (data["universe"] == primary_universe)
        & (data["rank_ic_method"] == "minute_mean_cs_rank_ic")
    ].copy()
    if data.empty:
        raise RuntimeError("no primary-label basic screen rows available for selection")

    grouped = data.groupby(["feature", "factor_family", "horizon_bars"], sort=False)
    summary = grouped.agg(
        mean_ic=("rank_ic_mean", "mean"),
        std_ic=("rank_ic_mean", "std"),
        valid_days=("rank_ic_mean", "count"),
        positive_day_ratio=("rank_ic_mean", lambda value: float((value > 0).mean())),
        coverage_mean=("coverage", "mean"),
        minute_positive_ratio=("rank_ic_positive_ratio", "mean"),
    ).reset_index()
    summary["direction"] = np.where(summary["mean_ic"] >= 0, 1.0, -1.0)
    summary["directional_day_ratio"] = np.where(
        summary["direction"] > 0,
        summary["positive_day_ratio"],
        1.0 - summary["positive_day_ratio"],
    )
    summary["directional_minute_ratio"] = np.where(
        summary["direction"] > 0,
        summary["minute_positive_ratio"],
        1.0 - summary["minute_positive_ratio"],
    )
    summary["icir"] = summary["mean_ic"].abs() / summary["std_ic"].replace(0, np.nan)
    summary["selection_score"] = (
        summary["mean_ic"].abs()
        * np.sqrt(summary["valid_days"].clip(lower=1))
        * summary["directional_day_ratio"].clip(lower=0.5)
        * summary["directional_minute_ratio"].clip(lower=0.5)
        * np.sqrt(summary["coverage_mean"].clip(lower=0))
    )
    summary = summary.sort_values("selection_score", ascending=False)
    summary = summary.groupby("feature", sort=False).head(1).reset_index(drop=True)

    minimum_days = int(selection.get("minimum_valid_days", max(10, train_days // 3)))
    minimum_coverage = float(selection.get("minimum_coverage", 0.70))
    minimum_stability = float(selection.get("minimum_directional_day_ratio", 0.55))
    eligible = summary[
        (summary["valid_days"] >= minimum_days)
        & (summary["coverage_mean"] >= minimum_coverage)
        & (summary["directional_day_ratio"] >= minimum_stability)
    ].copy()
    if eligible.empty:
        eligible = summary.copy()
        eligible["selection_warning"] = "FALLBACK_WEAK_NO_FACTOR_PASSED_THRESHOLDS"
    else:
        eligible["selection_warning"] = ""

    diagnostic = _candidate_rows(
        eligible,
        max_features=int(selection.get("diagnostic_max_features", 120)),
        min_per_family=int(selection.get("minimum_per_family", 2)),
        max_per_family=int(selection.get("diagnostic_max_per_family", 20)),
    )
    portfolio = _candidate_rows(
        eligible,
        max_features=int(selection.get("portfolio_max_features", 40)),
        min_per_family=int(selection.get("minimum_per_family", 2)),
        max_per_family=int(selection.get("portfolio_max_per_family", 8)),
    )
    diagnostic["candidate_scope"] = "detailed"
    portfolio["candidate_scope"] = "portfolio"
    candidates = pd.concat([diagnostic, portfolio], ignore_index=True)
    _atomic_parquet(summary, root / "all_selection_scores.parquet", index=False)
    _atomic_parquet(candidates, root / "candidates.parquet", index=False)
    freeze_date = train_dates[-1]
    meta = {
        "version": VERSION,
        "stage": "select",
        "status": "complete",
        "contract_hash": contract,
        "selection_dates": train_dates,
        "selection_start_date": train_dates[0],
        "selection_end_date": freeze_date,
        "portfolio_eligible_after": freeze_date,
        "direction_contract": "sign(mean training-date rank IC); frozen before portfolio dates",
        "diagnostic_candidate_count": int(diagnostic["feature"].nunique()),
        "portfolio_candidate_count": int(portfolio["feature"].nunique()),
        "thresholds": {
            "minimum_valid_days": minimum_days,
            "minimum_coverage": minimum_coverage,
            "minimum_directional_day_ratio": minimum_stability,
        },
    }
    _atomic_json(meta_path, meta)
    success.write_text(pd.Timestamp.utcnow().isoformat(), encoding="utf-8")
    return meta


def _load_candidates(config: Mapping[str, Any], scope: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = _pipeline_root(config) / "selection"
    if not (root / "_SUCCESS").exists():
        raise RuntimeError("candidate selection has not completed")
    candidates = pd.read_parquet(root / "candidates.parquet")
    candidates = candidates[candidates["candidate_scope"] == scope].copy()
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    return candidates, meta


def _with_saved_universes(universes: pd.DataFrame):
    original = R.universe_masks

    def saved(_features: pd.DataFrame, _controls: pd.DataFrame) -> dict[str, pd.Series]:
        return {column: universes[column].astype("boolean") for column in universes.columns}

    R.universe_masks = saved
    return original


def detailed_date(config: dict[str, Any], trade_date: str) -> dict[str, Any]:
    bootstrap(config)
    root = _stage_root(config, "detailed", trade_date)
    success = _stage_success(config, "detailed", trade_date)
    contract = _contract_hash(config)
    meta_path = root / "meta.json"
    if success.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("contract_hash") == contract:
            return {"trade_date": trade_date, "stage": "detailed", "status": "skipped"}
    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    candidates, selection_meta = _load_candidates(config, "detailed")
    names = list(candidates["feature"].drop_duplicates())
    support, labels, label_masks, controls, universes = _load_materialized(config, trade_date)
    factors = _load_selected_factors(config, trade_date, names, support.index)
    features = pd.concat([support, factors], axis=1, copy=False)
    original_universes = _with_saved_universes(universes)
    original_deciles = list(R.CORE_DECILE_FEATURES)
    R.CORE_DECILE_FEATURES = names
    try:
        min_n = int(config["run"].get("min_cross_section_n", 30))
        summary, residual_cache = R.minute_rank_ic_summary(
            features, labels, label_masks, controls, trade_date, min_n
        )
        deciles = R.decile_curves(
            features, labels, label_masks, controls, residual_cache, trade_date, min_n
        )
        del residual_cache
        gc.collect()
    finally:
        R.universe_masks = original_universes
        R.CORE_DECILE_FEATURES = original_deciles
    _atomic_parquet(summary, root / "factor_rank_ic_summary.parquet", index=False)
    if not deciles.empty:
        _atomic_parquet(deciles, root / "decile_curves.parquet", index=False)
    meta = {
        "version": VERSION,
        "trade_date": trade_date,
        "stage": "detailed",
        "status": "complete",
        "contract_hash": contract,
        "selection_end_date": selection_meta["selection_end_date"],
        "candidate_count": len(names),
        "summary_rows": int(len(summary)),
        "decile_rows": int(len(deciles)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(meta_path, meta)
    success.write_text(pd.Timestamp.utcnow().isoformat(), encoding="utf-8")
    return meta


def _selected_portfolio_variants(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    variants = [
        {
            "name": "selected_long_high_q10_15m",
            "direction": 1.0,
            "quantile": 0.10,
            "rebalance_minutes": 15,
            "gate_pair_id": None,
            "gate_mode": "none",
        },
        {
            "name": "selected_long_high_q05_30m",
            "direction": 1.0,
            "quantile": 0.05,
            "rebalance_minutes": 30,
            "gate_pair_id": None,
            "gate_mode": "none",
        },
        {
            "name": "selected_long_high_q05_30m_turnover_controlled",
            "direction": 1.0,
            "quantile": 0.05,
            "rebalance_minutes": 30,
            "gate_pair_id": None,
            "gate_mode": "none",
            "turnover_controlled": True,
        },
    ]
    for spec in getattr(R, "HAWKES_GATE_SPECS", ()):
        pair = f"selected_hawkes_{spec['name']}_top20_q05_30m"
        for mode in ("ungated_shared_sample", "exclude_top20"):
            variants.append(
                {
                    "name": f"{pair}_{mode}",
                    "direction": 1.0,
                    "quantile": 0.05,
                    "rebalance_minutes": 30,
                    "gate_pair_id": pair,
                    "gate_mode": mode,
                    "gate_column": spec["column"],
                    "exclude_fraction": spec.get("exclude_fraction", 0.20),
                }
            )
    return variants


def _selected_portfolio_proxy(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_masks: pd.DataFrame,
    controls: pd.DataFrame,
    universes: pd.DataFrame,
    trade_date: str,
    candidate_names: list[str],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    universe_name = str(config.get("portfolio_proxy", {}).get("universe", "final_trading_universe"))
    if universe_name not in universes:
        raise RuntimeError(f"portfolio universe not materialized: {universe_name}")
    universe_mask = universes[universe_name].fillna(False)
    primary_family = str(
        config.get("portfolio_proxy", {}).get("primary_execution_label", "return_vwap_to_vwap")
    )
    horizon_filter = {
        int(value) for value in config.get("portfolio_proxy", {}).get("horizons", [5, 15, 30, 60])
    }
    min_n = int(config["run"].get("min_cross_section_n", 30))
    costs = [float(value) for value in config.get("portfolio_proxy", {}).get("costs_bps_one_way", [1.0])]
    variants = _selected_portfolio_variants(config)
    gate_columns = [
        str(spec["column"])
        for spec in getattr(R, "HAWKES_GATE_SPECS", ())
        if str(spec["column"]) in features.columns
    ]
    work_columns = list(dict.fromkeys(candidate_names + gate_columns))
    rows: list[dict[str, Any]] = []
    for label_column in labels.columns:
        if "__h" not in label_column:
            continue
        family, horizon_text = label_column.rsplit("__h", 1)
        horizon = int(horizon_text)
        if family != primary_family or horizon not in horizon_filter:
            continue
        base_mask = (
            universe_mask
            & labels[label_column].notna()
            & label_masks[label_column].fillna(False)
        ).fillna(False)
        if int(base_mask.sum()) < min_n:
            continue
        adv = np.expm1(
            pd.to_numeric(controls.loc[base_mask, "control__log_adv20"], errors="coerce")
        ).rename("__adv20")
        work = pd.concat(
            [features.loc[base_mask, work_columns], labels.loc[base_mask, label_column].rename("label"), adv],
            axis=1,
        )
        for cost in costs:
            workers = min(
                int(config.get("pipeline", {}).get("portfolio_threads", 4)),
                max(1, len(variants)),
            )
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="selected-portfolio") as executor:
                futures = [
                    executor.submit(
                        R._portfolio_variant_rows_batched,
                        work,
                        candidate_names,
                        variant,
                        trade_date,
                        universe_name,
                        family,
                        horizon,
                        min_n,
                        cost,
                    )
                    for variant in variants
                ]
                for future in futures:
                    result = future.result()
                    for row in result:
                        row["candidate_contract"] = "training-window selected and direction-frozen"
                        rows.append(row)
    return pd.DataFrame(rows)


def portfolio_date(config: dict[str, Any], trade_date: str) -> dict[str, Any]:
    bootstrap(config)
    root = _stage_root(config, "portfolio", trade_date)
    success = _stage_success(config, "portfolio", trade_date)
    contract = _contract_hash(config)
    meta_path = root / "meta.json"
    if success.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("contract_hash") == contract:
            return {"trade_date": trade_date, "stage": "portfolio", "status": "skipped"}
    candidates, selection_meta = _load_candidates(config, "portfolio")
    allow_in_sample = bool(config.get("selection", {}).get("allow_in_sample_portfolio", False))
    if not allow_in_sample and trade_date <= str(selection_meta["portfolio_eligible_after"]):
        root.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": VERSION,
            "trade_date": trade_date,
            "stage": "portfolio",
            "status": "skipped_training_period",
            "contract_hash": contract,
            "portfolio_eligible_after": selection_meta["portfolio_eligible_after"],
        }
        _atomic_json(meta_path, meta)
        success.write_text(pd.Timestamp.utcnow().isoformat(), encoding="utf-8")
        return meta

    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    names = list(candidates["feature"].drop_duplicates())
    directions = candidates.drop_duplicates("feature").set_index("feature")["direction"].to_dict()
    support, labels, label_masks, controls, universes = _load_materialized(config, trade_date)
    factors = _load_selected_factors(config, trade_date, names, support.index)
    for name in names:
        factors[name] = factors[name].astype("float32") * float(directions[name])
    features = pd.concat([support, factors], axis=1, copy=False)
    portfolio = _selected_portfolio_proxy(
        features,
        labels,
        label_masks,
        controls,
        universes,
        trade_date,
        names,
        config,
    )
    if not portfolio.empty:
        _atomic_parquet(portfolio, root / "staggered_portfolio_proxy.parquet", index=False)
    meta = {
        "version": VERSION,
        "trade_date": trade_date,
        "stage": "portfolio",
        "status": "complete",
        "contract_hash": contract,
        "selection_end_date": selection_meta["selection_end_date"],
        "candidate_count": len(names),
        "portfolio_rows": int(len(portfolio)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(meta_path, meta)
    success.write_text(pd.Timestamp.utcnow().isoformat(), encoding="utf-8")
    return meta


def _stage_specs(config: Mapping[str, Any]) -> dict[str, StageSpec]:
    supplied = config.get("pipeline", {}).get("stages", {})
    defaults = {
        "materialize": (6, 12.0),
        "basic_screen": (8, 6.0),
        "detailed": (4, 16.0),
        "portfolio": (6, 8.0),
    }
    result: dict[str, StageSpec] = {}
    for stage, (workers, memory) in defaults.items():
        item = supplied.get(stage, {}) if isinstance(supplied, Mapping) else {}
        result[stage] = StageSpec(
            stage,
            int(item.get("max_workers", workers)),
            float(item.get("estimated_worker_gb", memory)),
        )
    return result


def _worker_command(config_path: Path, stage: str, trade_date: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve().with_name("v2_8_launch.py")),
        "--config",
        str(config_path),
        "--worker-stage",
        stage,
        "--worker-date",
        trade_date,
    ]


def _run_date_stage(config_path: Path, config: dict[str, Any], stage: str) -> dict[str, Any]:
    spec = _stage_specs(config)[stage]
    dates = _dates(config)
    if stage == "portfolio" and not bool(config.get("selection", {}).get("allow_in_sample_portfolio", False)):
        _, meta = _load_candidates(config, "portfolio")
        dates = [date for date in dates if date > str(meta["portfolio_eligible_after"])]
    pending = [date for date in dates if not _stage_success(config, stage, date).exists()]
    running: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    retries = int(config["run"].get("retries", 3))
    reserve_gb = float(config.get("pipeline", {}).get("memory_reserve_gb", 28.0))
    status_path = _pipeline_root(config) / f"status_{stage}.json"
    log_root = _pipeline_root(config) / "worker_logs" / stage
    log_root.mkdir(parents=True, exist_ok=True)
    attempts: dict[str, int] = {}

    while pending or running:
        available_gb = psutil.virtual_memory().available / 1024**3
        memory_cap = max(1, int(max(0.0, available_gb - reserve_gb) // spec.estimated_worker_gb))
        cap = max(1, min(spec.max_workers, memory_cap))
        while pending and len(running) < cap:
            trade_date = pending.pop(0)
            attempts[trade_date] = attempts.get(trade_date, 0) + 1
            attempt = attempts[trade_date]
            stdout_path = log_root / f"date={trade_date}_attempt={attempt}.out.log"
            stderr_path = log_root / f"date={trade_date}_attempt={attempt}.err.log"
            stdout = stdout_path.open("w", encoding="utf-8")
            stderr = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                _worker_command(config_path, stage, trade_date),
                stdout=stdout,
                stderr=stderr,
            )
            running[trade_date] = {
                "process": process,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "started": time.perf_counter(),
                "attempt": attempt,
            }
        finished: list[str] = []
        for trade_date, info in list(running.items()):
            rc = info["process"].poll()
            if rc is None:
                continue
            info["stdout"].close()
            info["stderr"].close()
            if rc != 0 or not _stage_success(config, stage, trade_date).exists():
                error = info["stderr_path"].read_text(encoding="utf-8", errors="replace")[-4000:]
                failure = {
                    "trade_date": trade_date,
                    "stage": stage,
                    "attempt": info["attempt"],
                    "returncode": rc,
                    "error": error,
                }
                if info["attempt"] <= retries:
                    pending.append(trade_date)
                else:
                    failures.append(failure)
                    _atomic_json(
                        _pipeline_root(config) / "failures" / stage / f"date={trade_date}.json",
                        failure,
                    )
            finished.append(trade_date)
        for trade_date in finished:
            running.pop(trade_date, None)
        _atomic_json(
            status_path,
            {
                "version": VERSION,
                "stage": stage,
                "pending": len(pending),
                "running": list(running),
                "completed": sum(_stage_success(config, stage, date).exists() for date in dates),
                "total": len(dates),
                "failures": len(failures),
                "max_workers": spec.max_workers,
                "memory_limited_cap": cap,
                "memory_available_gb": available_gb,
                "estimated_worker_gb": spec.estimated_worker_gb,
                "updated_utc": pd.Timestamp.utcnow().isoformat(),
            },
        )
        time.sleep(2)
    return {"stage": stage, "status": "partial_success" if failures else "complete", "failures": failures}


def run_worker(config: dict[str, Any], stage: str, trade_date: str) -> dict[str, Any]:
    functions = {
        "materialize": materialize_date,
        "basic_screen": basic_screen_date,
        "detailed": detailed_date,
        "portfolio": portfolio_date,
    }
    if stage not in functions:
        raise ValueError(f"unsupported worker stage: {stage}")
    return functions[stage](config, trade_date)


def write_architecture(config: Mapping[str, Any]) -> None:
    root = _pipeline_root(config) / "architecture"
    root.mkdir(parents=True, exist_ok=True)
    specs = _stage_specs(config)
    _atomic_json(
        root / "v2_8_pipeline.json",
        {
            "version": VERSION,
            "dag": ["materialize", "basic_screen", "select", "detailed", "portfolio"],
            "stage_parallelism": {
                name: {
                    "max_workers": spec.max_workers,
                    "estimated_worker_gb": spec.estimated_worker_gb,
                }
                for name, spec in specs.items()
            },
            "factor_scope": {
                "materialize": 464,
                "basic_screen": 464,
                "detailed": "training-selected diagnostic candidates",
                "portfolio": "training-selected and direction-frozen portfolio candidates",
            },
            "anti_leakage": [
                "candidate list fitted only on first selection.train_days completed dates",
                "factor direction is sign of training-window mean rank IC",
                "portfolio dates must be later than selection_end_date unless explicitly overridden",
            ],
            "memory_model": "each stage is a separate process; wide materialization frames are released before IC/portfolio stages",
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=["materialize", "basic_screen", "select", "detailed", "portfolio", "all"],
        default="all",
    )
    parser.add_argument("--worker-stage", choices=["materialize", "basic_screen", "detailed", "portfolio"])
    parser.add_argument("--worker-date")
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    bootstrap(config)
    write_architecture(config)

    if args.worker_stage:
        if not args.worker_date:
            raise ValueError("--worker-stage requires --worker-date")
        print(json.dumps(run_worker(config, args.worker_stage, args.worker_date), indent=2, default=str))
        return 0

    stages = (
        ["materialize", "basic_screen", "select", "detailed", "portfolio"]
        if args.stage == "all"
        else [args.stage]
    )
    results: list[dict[str, Any]] = []
    for stage in stages:
        if stage == "select":
            result = select_candidates(config)
        else:
            result = _run_date_stage(config_path, config, stage)
        results.append(result)
        if result.get("status") not in {"complete", "skipped"}:
            print(json.dumps(results, indent=2, default=str))
            return 2
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
