"""Full A-K defined-factor campaign.

This is deliberately a separate runner from the v2.5 representative screen.
It loads every available NFF source column needed by the governing A-K
contract, derives all executable prototypes, retains unavailable prototypes in
the registry, and then reuses the tested PIT/IC/decile/account/OOS machinery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nff_research import full_factor_engine as FF
from nff_research import v2_5_full_campaign as BASE
from nff_research import v2_1_neutralized_runner as R
from nff_research import v2_4_optimized_runner as FAST


PROTOTYPES = FF.parse_prototypes()
SPEC_REGISTRY = FF.expand_specs(PROTOTYPES)
_ORIGINAL_BUILD_LABELS = R.build_labels_and_masks
SUPPLEMENT_FACTOR_NAMES = [
    "full_factor__s01__w15m",
    "full_factor__s01__w30m",
]
FULL_FACTOR_NAMES = [
    FF.factor_name(row["prototype_id"], row["window"])
    for _, row in SPEC_REGISTRY.iterrows()
] + SUPPLEMENT_FACTOR_NAMES
SUPPLEMENT_SPECS = pd.DataFrame(
    [
        {
            "factor_id": name,
            "prototype_id": "S01",
            "family": "S",
            "title": "NVG supplement directional consensus",
            "role": "DIRECTION_ALPHA",
            "direction_prior": "TWO_SIDED",
            "source_fields": "minute_nvg + nvg_supplement minute_nvg_edge_raw",
            "formula": "tanh(CSZ(base price path, raw NVG edge, detrended edge, supplement edge)) * sqrt(completeness) * confidence",
            "financial_hypothesis": "directional consensus using complete NVG edge information",
            "window": window,
            "transform": "CSZ,confidence-weighted consensus",
            "label_group": "return,state,risk,liquidity,cost",
            "required_gate": "PIT; quality/readiness gate",
            "available_time_rule": "available_time <= planned_order_time; entry t+1",
            "expected_range": "bounded by tanh and completeness",
            "neutralization_allowed": True,
            "cost_relevance": "direct",
            "status": "PENDING_SCHEMA_RESOLUTION",
            "instruction_line": None,
        }
        for name, window in zip(SUPPLEMENT_FACTOR_NAMES, ("15m", "30m"))
    ]
)
SPEC_REGISTRY = pd.concat([SPEC_REGISTRY, SUPPLEMENT_SPECS], ignore_index=True)
RUNTIME_FACTOR_STATUS: dict[str, dict[str, Any]] = {}


def _source_columns_optional(kind: str, dataset: str, schema: str) -> list[str]:
    try:
        return R.source_columns(kind, dataset, schema)
    except (FileNotFoundError, KeyError):
        return []


def _all_feature_sets() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for dataset in ("minute_nvg", "trade_nvg", "hawkes_lite"):
        schema = next((s for s in ("v4", "v3", "v2", "v1") if _source_columns_optional("feature", dataset, s)), None)
        if schema:
            result[dataset] = {
                "schema_version": schema,
                "columns": [c for c in _source_columns_optional("feature", dataset, schema) if c not in R.KEY_COLUMNS],
            }
    return result


def _all_canonical_sets() -> dict[str, dict[str, Any]]:
    bars = [c for c in _source_columns_optional("canonical", "bars_1m", "v1") if c not in R.KEY_COLUMNS]
    core = [c for c in _source_columns_optional("canonical", "trades_1m_core", "v1") if c not in R.KEY_COLUMNS]
    return {
        "bars_1m": {"schema_version": "v1", "columns": bars},
        "trades_1m_core": {"schema_version": "v1", "columns": core},
    }


def _full_add_all_features(frame: pd.DataFrame) -> pd.DataFrame:
    # Keep every numeric source column.  The raw columns are controls and
    # dependencies; only full_factor__ columns are admitted as Alpha inputs.
    frame = FF.merge_supplements(frame, R.WAREHOUSE_ROOT)
    frame = FF.merge_canonical_sketch(frame, R.WAREHOUSE_ROOT)
    frame = FF.merge_condition_aggregates(frame, R.WAREHOUSE_ROOT)
    frame = FF.merge_venue_aggregates(frame, R.WAREHOUSE_ROOT)
    # The legacy helpers copy their whole input frame.  Run each helper on
    # its narrow dependency block, then join only generated columns.  This
    # keeps the full A-K source inventory while avoiding three wide-frame
    # copies at the peak of a worker.
    hawkes_source = [c for c in frame.columns if c.startswith("hawkes_lite__")]
    hawkes_block = R.V2.add_hawkes_derived(frame[hawkes_source]) if hawkes_source else frame.iloc[:, 0:0].copy()
    hawkes_columns = [c for c in hawkes_block.columns if c.startswith("hawkes_derived__")]
    features = frame.join(hawkes_block[hawkes_columns], how="left") if hawkes_columns else frame
    del hawkes_block
    traditional_columns = [c for c in ("bars_1m__close", "bars_1m__vwap", "bars_1m__dollar_volume") if c in features.columns]
    traditional_block = R.V2.add_traditional_factors(features[traditional_columns]) if traditional_columns else features.iloc[:, 0:0].copy()
    traditional_generated = [c for c in traditional_block.columns if c.startswith("traditional__")]
    features = features.join(traditional_block[traditional_generated], how="left") if traditional_generated else features
    del traditional_block
    features, runtime = FF.derive_all(features, PROTOTYPES)
    # The supplement is deliberately combined with base NVG components into a
    # directional factor; raw supplement fields never enter Alpha directly.
    features = BASE.add_supplement_directional_factor(features)
    supplement_names = [c for c in BASE.DERIVED_FEATURES if c in features]
    for name in supplement_names:
        window = "15m" if name.endswith("15m") else "30m"
        factor = f"full_factor__s01__w{window}"
        features[factor] = features[name]
        runtime[factor] = {
            "status": "SUCCESS" if features[name].notna().any() else "LOW_COVERAGE",
            "non_null_rate": float(features[name].notna().mean()),
        }
    RUNTIME_FACTOR_STATUS.clear()
    RUNTIME_FACTOR_STATUS.update(runtime)
    keep = list(features.columns)
    return features[keep].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).astype("float32")


def _full_analysis_features(features: pd.DataFrame) -> list[str]:
    return [c for c in FULL_FACTOR_NAMES if c in features.columns and features[c].notna().any()]


def _future_exact(frame: pd.DataFrame, column: str, offset_minutes: int) -> pd.Series:
    """Return a column at the exact future timestamp for each symbol-minute row."""
    index = frame.index
    times = pd.to_datetime(index.get_level_values("datetime"), utc=True)
    symbols = index.get_level_values("instrument")
    lookup_index = pd.MultiIndex.from_arrays(
        [times + pd.Timedelta(minutes=offset_minutes), symbols],
        names=index.names,
    )
    source = pd.to_numeric(frame[column], errors="coerce").copy()
    source.index = pd.MultiIndex.from_arrays([times, symbols], names=index.names)
    result = source.reindex(lookup_index)
    result.index = index
    return result


def _ensure_research_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize loader index aliases before label code uses named levels."""
    index = frame.index
    if not isinstance(index, pd.MultiIndex):
        raise TypeError(f"research frame must have a MultiIndex, got {type(index).__name__}")
    names = list(index.names)

    def datetime_score(position: int) -> float:
        values = index.get_level_values(position)
        if pd.api.types.is_datetime64_any_dtype(values):
            return 1.0
        # Inspect the full level by evenly spaced samples.  A leading slice
        # can look datetime-like while a later symbol block contains values
        # such as ``A``; that silently mislabels the levels and breaks labels.
        sample_size = min(len(values), 4096)
        if len(values) > sample_size:
            positions = np.linspace(0, len(values) - 1, sample_size, dtype="int64")
            sampled = values.take(positions)
        else:
            sampled = values
        sample = pd.Series(sampled, dtype="object")
        try:
            parsed = pd.to_datetime(sample, utc=True, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(sample, utc=True, errors="coerce")
        if not len(parsed):
            return 0.0
        plausible = parsed.notna() & parsed.ge(pd.Timestamp("2015-01-01", tz="UTC")) & parsed.lt(
            pd.Timestamp("2035-01-01", tz="UTC")
        )
        return float(plausible.mean())

    def datetime_position() -> int:
        named_candidates = {
            position
            for position, name in enumerate(names)
            if name in {"datetime", "timestamp", "event_time", "decision_time"}
        }
        positions = sorted(named_candidates) + [position for position in range(index.nlevels) if position not in named_candidates]
        best_position = max(positions, key=datetime_score)
        best_rate = datetime_score(best_position)
        if best_rate < 0.9:
            if names == [None, None] and index.nlevels == 2:
                # The NFF loader's unnamed fallback contract is still
                # instrument, datetime.  Prefer an actual datetime dtype;
                # otherwise retain that fixed two-level order rather than
                # failing after a concat has stripped only the names.
                for position in range(index.nlevels):
                    if pd.api.types.is_datetime64_any_dtype(index.get_level_values(position)):
                        return position
                return 1
            raise KeyError(f"cannot identify datetime level from index names={names!r}")
        return best_position

    dt_position = datetime_position()
    instrument_position = next(
        (
            position
            for position, name in enumerate(names)
            if position != dt_position and name in {"instrument", "symbol", "security", "ticker"}
        ),
        next(position for position in range(index.nlevels) if position != dt_position),
    )
    names[dt_position] = "datetime"
    names[instrument_position] = "instrument"
    if names == list(index.names):
        return frame
    normalized = frame.copy(deep=False)
    normalized.index = index.set_names(names)
    return normalized


def _session_valid(index: pd.MultiIndex, horizon: int) -> pd.Series:
    times = pd.to_datetime(index.get_level_values("datetime"), utc=True)
    entry = times + pd.Timedelta(minutes=1)
    exit_time = times + pd.Timedelta(minutes=horizon + 1)
    local_entry = entry.tz_convert("America/New_York")
    local_exit = exit_time.tz_convert("America/New_York")
    entry_minute = local_entry.hour * 60 + local_entry.minute
    exit_minute = local_exit.hour * 60 + local_exit.minute
    valid_entry = (entry_minute >= 570) & (entry_minute < 960)
    valid_exit = (exit_minute >= 570) & (exit_minute < 960)
    return pd.Series(
        valid_entry & valid_exit & (local_entry.date == local_exit.date),
        index=index,
        dtype="boolean",
    )


def _full_build_labels_and_masks(frame: pd.DataFrame, horizons: list[int]):
    """Build labels using exact elapsed-minute joins, never observed-row shifts.

    The entry is always t+1 and the return exit is t+h+1. Future RV/BPV,
    absolute return, MFE/MAE and cost windows are accumulated only from exact
    future minutes. Missing minutes therefore invalidate the corresponding
    label instead of silently stretching the horizon.
    """
    frame = _ensure_research_index(frame)
    labels = pd.DataFrame(index=frame.index)
    masks: dict[str, pd.Series] = {}
    close_col = "bars_1m__close"
    open_col = "bars_1m__open"
    vwap_col = "bars_1m__vwap"
    high_col = "bars_1m__high" if "bars_1m__high" in frame else close_col
    low_col = "bars_1m__low" if "bars_1m__low" in frame else close_col
    volume_col = "bars_1m__volume" if "bars_1m__volume" in frame else None
    dollar_col = "bars_1m__dollar_volume" if "bars_1m__dollar_volume" in frame else None
    trade_col = "trades_1m_core__trade_count" if "trades_1m_core__trade_count" in frame else None

    current_close = pd.to_numeric(frame[close_col], errors="coerce").replace(0, np.nan)
    for horizon in sorted(set(int(value) for value in horizons)):
        session_valid = _session_valid(frame.index, horizon)
        entry_open = _future_exact(frame, open_col, 1).replace(0, np.nan)
        entry_vwap = _future_exact(frame, vwap_col, 1).replace(0, np.nan)
        entry_close = _future_exact(frame, close_col, 1).replace(0, np.nan)
        exit_open = _future_exact(frame, open_col, horizon + 1).replace(0, np.nan)
        exit_vwap = _future_exact(frame, vwap_col, horizon + 1).replace(0, np.nan)
        exit_close = _future_exact(frame, close_col, horizon + 1).replace(0, np.nan)

        return_specs = {
            "return_open_to_open": (exit_open / entry_open - 1.0),
            "return_vwap_to_vwap": (exit_vwap / entry_vwap - 1.0),
            "return_close_to_close": (exit_close / entry_close - 1.0),
        }
        for family, value in return_specs.items():
            name = f"{family}__h{horizon}"
            labels[name] = value.where(session_valid)
            masks[name] = (session_valid & value.notna()).astype("boolean")

        # Build future-window statistics from exact t+1 ... t+h+1 prices.
        previous_close = _future_exact(frame, close_col, 1).replace(0, np.nan)
        sum_sq = pd.Series(0.0, index=frame.index, dtype="float64")
        sum_abs = pd.Series(0.0, index=frame.index, dtype="float64")
        sum_bpv = pd.Series(0.0, index=frame.index, dtype="float64")
        max_high = pd.Series(-np.inf, index=frame.index, dtype="float64")
        min_low = pd.Series(np.inf, index=frame.index, dtype="float64")
        sum_cost = pd.Series(0.0, index=frame.index, dtype="float64")
        valid_returns = pd.Series(True, index=frame.index, dtype="boolean")
        valid_cost = pd.Series(True, index=frame.index, dtype="boolean")
        previous_return: pd.Series | None = None
        for offset in range(2, horizon + 2):
            future_close = _future_exact(frame, close_col, offset).replace(0, np.nan)
            future_high = _future_exact(frame, high_col, offset)
            future_low = _future_exact(frame, low_col, offset)
            future_dollar = _future_exact(frame, dollar_col, offset) if dollar_col else pd.Series(np.nan, index=frame.index)
            one_minute_return = np.log(future_close / previous_close)
            valid_returns &= one_minute_return.notna()
            sum_sq += one_minute_return.pow(2).fillna(0.0)
            sum_abs += one_minute_return.abs().fillna(0.0)
            if previous_return is not None:
                sum_bpv += previous_return.abs().mul(one_minute_return.abs()).fillna(0.0)
            previous_return = one_minute_return
            # Array-wise extrema avoid allocating a two-column frame on every
            # future minute. NaN remains invalid for the whole exact window.
            max_high = pd.Series(
                np.maximum(max_high.to_numpy(dtype="float64"), future_high.to_numpy(dtype="float64")),
                index=frame.index,
            )
            min_low = pd.Series(
                np.minimum(min_low.to_numpy(dtype="float64"), future_low.to_numpy(dtype="float64")),
                index=frame.index,
            )
            valid_cost &= future_dollar.gt(0) & one_minute_return.notna()
            sum_cost += (one_minute_return.abs() / future_dollar.replace(0, np.nan)).fillna(0.0)
            previous_close = future_close

        window_valid = session_valid & valid_returns
        rv_name = f"realized_volatility__h{horizon}"
        labels[rv_name] = np.sqrt(sum_sq).where(window_valid)
        masks[rv_name] = window_valid.astype("boolean")

        abs_name = f"fwd_abs_return__h{horizon}"
        labels[abs_name] = (sum_abs / max(horizon, 1)).where(window_valid)
        masks[abs_name] = window_valid.astype("boolean")

        mfe_name = f"fwd_mfe__h{horizon}"
        mae_name = f"fwd_mae__h{horizon}"
        labels[mfe_name] = (max_high / entry_open - 1.0).where(window_valid)
        labels[mae_name] = (min_low / entry_open - 1.0).where(window_valid)
        masks[mfe_name] = window_valid.astype("boolean")
        masks[mae_name] = window_valid.astype("boolean")

        bpv = (np.pi / 2.0) * sum_bpv
        jump_variation = (sum_sq - bpv).clip(lower=0.0)
        threshold = jump_variation.groupby(level="datetime", sort=False).transform("quantile", q=0.99)
        jump_name = f"jump_tail_event__h{horizon}"
        jump_valid = window_valid & threshold.notna()
        labels[jump_name] = (jump_variation > threshold).astype("float32").where(jump_valid)
        masks[jump_name] = jump_valid.astype("boolean")

        entry_dollar = _future_exact(frame, dollar_col, 1) if dollar_col else pd.Series(np.nan, index=frame.index)
        entry_trade = _future_exact(frame, trade_col, 1) if trade_col else pd.Series(np.nan, index=frame.index)
        entry_liquidity = np.log1p(entry_dollar.clip(lower=0)) + 0.25 * np.log1p(entry_trade.clip(lower=0))
        future_liquidity_sum = pd.Series(0.0, index=frame.index, dtype="float64")
        valid_liquidity = pd.Series(True, index=frame.index, dtype="boolean")
        for offset in range(1, horizon + 1):
            future_dollar = _future_exact(frame, dollar_col, offset) if dollar_col else pd.Series(np.nan, index=frame.index)
            future_trade = _future_exact(frame, trade_col, offset) if trade_col else pd.Series(np.nan, index=frame.index)
            future_liquidity = np.log1p(future_dollar.clip(lower=0)) + 0.25 * np.log1p(future_trade.clip(lower=0))
            valid_liquidity &= future_liquidity.notna()
            future_liquidity_sum += future_liquidity.fillna(0.0)
        liq_name = f"liquidity_deterioration__h{horizon}"
        liq_valid = session_valid & valid_liquidity & entry_liquidity.notna()
        labels[liq_name] = (entry_liquidity - future_liquidity_sum / max(horizon, 1)).where(liq_valid)
        masks[liq_name] = liq_valid.astype("boolean")

        cost_name = f"execution_cost_proxy__h{horizon}"
        cost_valid = session_valid & valid_cost
        labels[cost_name] = (sum_cost / max(horizon, 1)).where(cost_valid)
        masks[cost_name] = cost_valid.astype("boolean")

    return labels.replace([np.inf, -np.inf], np.nan).astype("float32"), masks


def _full_feature_registry(features: pd.DataFrame, evaluated: list[str], trade_date: str) -> pd.DataFrame:
    # Preserve the existing field-level registry and add the authoritative
    # prototype/spec registry beside it.
    base = R.feature_registry(features, evaluated, trade_date)
    specs = SPEC_REGISTRY.copy()
    specs["trade_date"] = trade_date
    specs["runtime_feature"] = specs.apply(lambda row: FF.factor_name(row.prototype_id, row.window), axis=1)
    specs["runtime_status"] = specs.runtime_feature.map(lambda x: RUNTIME_FACTOR_STATUS.get(x, {}).get("status", "DATA_UNAVAILABLE"))
    specs["runtime_non_null_rate"] = specs.runtime_feature.map(lambda x: RUNTIME_FACTOR_STATUS.get(x, {}).get("non_null_rate", 0.0))
    specs["status"] = specs["runtime_status"]
    return base, specs


def _patch() -> None:
    global PROTOTYPES
    FAST.install_fastpath()
    R.feature_sets = _all_feature_sets
    R.canonical_sets = _all_canonical_sets
    R.add_all_features = _full_add_all_features
    R.analysis_features = _full_analysis_features
    R.build_labels_and_masks = _full_build_labels_and_masks
    # Match the authoritative label contract: RET_1M, RET_5M, RET_15M,
    # RET_30M and RET_60M.  The loader's bar clock is one minute, so these
    # are elapsed-minute horizons, not arbitrary observed-row counts.
    R.HORIZONS = (1, 5, 15, 30, 60)
    R.CORE_DECILE_FEATURES = FULL_FACTOR_NAMES
    R.infer_bundle = lambda name: {
        "a": "A_TRADITIONAL",
        "b": "B_MINUTE_NVG",
        "c": "C_TOPOLOGY",
        "d": "D_VOLUME_NVG",
        "e": "E_TRADE_NVG",
        "f": "F_ACTIVITY",
        "g": "G_HVG",
        "h": "H_HAWKES",
        "i": "I_MICROSTRUCTURE",
        "j": "J_VENUE",
        "k": "K_COMPOSITE",
        "s": "S_SUPPLEMENT",
    }.get(str(name).lower().split("__")[1][:1], "FULL_DEFINED") if str(name).startswith("full_factor__") else R.V2.infer_bundle(name)
    R.bundle_incremental_model_screen = lambda *args, **kwargs: pd.DataFrame()
    original_worker_command = R.worker_command

    def full_worker_command(*args: Any, **kwargs: Any) -> list[str]:
        command = original_worker_command(*args, **kwargs)
        legacy_script = str(Path(R.__file__).resolve())
        current_script = str(Path(__file__).resolve())
        return [current_script if item == legacy_script else item for item in command]

    R.worker_command = full_worker_command
    BASE.DERIVED_FEATURES = [c for c in BASE.DERIVED_FEATURES if c in FULL_FACTOR_NAMES]
    BASE._analysis_features = _full_analysis_features
    BASE._materialize_model_cache.__globals__["_analysis_features"] = _full_analysis_features
    BASE.run_temporal_oos.__globals__["DERIVED_FEATURES"] = FULL_FACTOR_NAMES


def _preflight(run_root: Path, config: dict[str, Any]) -> None:
    report = run_root / "reports" / "factor_registry"
    report.mkdir(parents=True, exist_ok=True)
    inventory_rows: list[dict[str, Any]] = []
    for dataset, spec in {**_all_feature_sets(), **_all_canonical_sets()}.items():
        schema = spec["schema_version"]
        kind = "canonical" if dataset in {"bars_1m", "trades_1m_core"} else "feature"
        for field in spec["columns"]:
            inventory_rows.append({"kind": kind, "dataset": dataset, "schema": schema, "field": field, "status": "INVENTORIED"})
    pd.DataFrame(inventory_rows).to_parquet(report / "source_field_inventory.parquet", index=False)
    pd.DataFrame(inventory_rows).to_csv(report / "source_field_inventory.csv", index=False)
    specs = SPEC_REGISTRY.copy()
    specs["status"] = "PENDING_SCHEMA_RESOLUTION"
    specs["instruction_sha256"] = FF.instruction_hash()
    specs.to_parquet(report / "factor_spec_registry.parquet", index=False)
    specs.to_csv(report / "factor_spec_registry.csv", index=False)
    assumptions = {
        "research_version": "2.6-full-defined",
        "governing_instruction": str(FF.INSTRUCTION_PATH),
        "instruction_sha256": FF.instruction_hash(),
        "prototype_count": int(len(PROTOTYPES)),
        "spec_count": int(len(specs)),
        "available_source_field_count": int(len(inventory_rows)),
        "known_unavailable_families": [],
        "source_resolution_policy": "C/G use published nvg_supplement; J uses canonical trades_venue_1m plus sketch venue fields; any individual missing prototype dependency remains DATA_UNAVAILABLE in the registry.",
        "known_cost_limit": "complete quote/order-book tape is unavailable; spread/impact are explicitly estimated proxies",
        "model_policy": "temporal OOS Ridge is the executable primary model; unimplemented model names are not claimed as run",
        "normalization_policy": "raw, per-minute CSZ standardization, and residualized return IC are retained",
        "mixed_contract_policy": "April implementation-hash boundary is allowed only with per-date hash audit",
        "config": config,
    }
    (run_root / "reports" / "assumptions.json").write_text(json.dumps(assumptions, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (run_root / "reports" / "factor_registry" / "instruction_manifest.json").write_text(json.dumps(assumptions, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _wrap_run_date() -> None:
    original = R.run_date

    def run_date(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        trade_date = str(args[0] if args else kwargs["trade_date"])
        horizons = args[1] if len(args) > 1 else kwargs["horizons"]
        out_root = Path(args[2] if len(args) > 2 else kwargs["out_root"])
        controls_path = Path(args[3] if len(args) > 3 else kwargs["controls_path"])
        out_dir = out_root / "02_neutralized_factor_diagnostics" / f"date={trade_date}"
        _, specs = _full_feature_registry(pd.DataFrame(), [], trade_date)
        # Replace the preflight placeholder with runtime status, while keeping
        # the full 184-prototype/expanded-spec rows even when unavailable.
        specs.to_parquet(out_dir / "factor_spec_registry.parquet", index=False)
        specs.to_csv(out_dir / "factor_spec_registry.csv", index=False)
        summary_path = out_dir / "factor_rank_ic_summary.parquet"
        if summary_path.exists():
            summary = pd.read_parquet(summary_path)
            if "normalization_variant" not in summary.columns:
                summary["normalization_variant"] = np.where(
                    summary["neutralization"].eq("none"), "raw", "neutralized"
                )
                standardized = summary.loc[summary["neutralization"].eq("none")].copy()
                standardized["normalization_variant"] = "standardized_cs_zscore"
                standardized["neutralization"] = "standardized_cs_zscore"
                summary = pd.concat([summary, standardized], ignore_index=True)
                summary.to_parquet(summary_path, index=False)
                summary.to_csv(out_dir / "factor_rank_ic_summary.csv", index=False)
        if result.get("status") in {"completed", "skipped"}:
            # The temporal OOS stage consumes the same full-defined factor
            # block. Materialize it at the date checkpoint while the loader
            # cache is warm; otherwise a successful single-factor phase would
            # leave no auditable training input for the walk-forward stage.
            BASE._materialize_model_cache(trade_date, horizons, out_root, controls_path)
        return result

    R.run_date = run_date


def run(config_path: Path) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_root = Path(config["local_paths"]["research_root"]) / "runs" / config["run"]["name"]
    run_root.mkdir(parents=True, exist_ok=True)
    _preflight(run_root, config)
    original_argv = list(sys.argv)
    if "--out-root" not in sys.argv:
        sys.argv.extend(["--out-root", str(run_root)])
    try:
        rc = R.main()
    finally:
        sys.argv[:] = original_argv
    if rc != 0:
        return rc
    model = BASE.run_temporal_oos(run_root, config)
    BASE._write_final_report(run_root, model, config)
    (run_root / "reports" / "final_report_v2_6_full_defined.md").write_text(
        (run_root / "reports" / "final_report_v2_5.md").read_text(encoding="utf-8")
        .replace("v2.5", "v2.6 full-defined A-K")
        .replace("55 representative factors plus two derived NVG supplement directional factors", f"{len(PROTOTYPES)} A-K prototypes expanded to {len(SPEC_REGISTRY)} formal specs; executable factors are recorded per-date with unavailable specs preserved")
        + "\n\n## Governing instruction audit\n\n"
        + json.dumps({"instruction": str(FF.INSTRUCTION_PATH), "sha256": FF.instruction_hash(), "prototype_count": len(PROTOTYPES), "spec_count": len(SPEC_REGISTRY)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    status = {"phase": "complete" if model.get("status") == "complete" else "partial", "daily_phase_returncode": rc, "temporal_oos": model, "finished_at": BASE.datetime.now(BASE.timezone.utc).isoformat(), "prototype_count": len(PROTOTYPES), "spec_count": len(SPEC_REGISTRY), "instruction_sha256": FF.instruction_hash()}
    (run_root / "status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    return 0 if model.get("status") == "complete" else 4


def main() -> int:
    _patch()
    _wrap_run_date()
    if "--worker" in sys.argv:
        return R.main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v2_6_full_defined_campaign.yaml")
    args = parser.parse_args()
    return run(Path(args.config).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
