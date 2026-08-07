"""NFF v2.7 real-schema atomic fast path.

This runner preserves the v2.6 research contract while making the expensive
single-factor pass restartable and materially faster. It treats the rebuilt
``nvg_supplement`` as a first-class source, discovers its actual Parquet schema,
uses only exact or financially equivalent aliases, caches exact future-minute
lookups, vectorizes decile aggregation, and checkpoints factor blocks and later
stages below the date level.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import yaml
try:
    from scipy.stats import rankdata
except ImportError:  # pragma: no cover - scipy is part of the research runtime
    rankdata = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nff_research import full_factor_engine as FF
from nff_research import v2_1_neutralized_runner as R
from nff_research import v2_5_full_campaign as BASE
from nff_research import v2_6_full_defined_campaign as V26
from nff_research.nff_schema_contract import (
    directional_supplement_fields,
    discover_warehouse_schema,
    minute_windows,
    second_windows,
    write_schema_audit,
)


EPS = 1e-8
VERSION = "2.7-real-schema-atomic"
SESSION_TZ = "America/New_York"
SUPPLEMENT_WINDOWS = ("15m", "30m")


@dataclass
class Context:
    trade_date: str
    out_root: Path
    contract_hash: str
    factor_block_size: int
    intra_workers: int

    @property
    def root(self) -> Path:
        return self.out_root / "atomic_checkpoints" / f"date={self.trade_date}"


CTX: Context | None = None
ORIGINAL: dict[str, Any] = {}
RUNTIME_WINDOWS: dict[str, tuple[str, ...]] = {}
FUTURE_CACHE: dict[tuple[int, str, int], pd.Series] = {}
FUTURE_WIDE_CACHE: dict[tuple[int, str], tuple[pd.DatetimeIndex, pd.Index, np.ndarray]] = {}


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _index_hash(index: pd.Index) -> str:
    values = pd.util.hash_pandas_object(index, index=True).to_numpy(dtype="uint64")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _rank_frame_average(frame: pd.DataFrame) -> pd.DataFrame:
    """Match pandas average rank/NaN semantics without DataFrame rank overhead."""
    if rankdata is None:
        return frame.rank(method="average")
    values = rankdata(
        frame.to_numpy(dtype="float64", copy=False),
        axis=0,
        method="average",
        nan_policy="omit",
    )
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def _rank_frame_by_datetime_average(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank feature values within each decision-minute cross-section."""
    if frame.empty:
        return frame.copy()
    values = frame.to_numpy(dtype="float64", copy=False)
    ranked = np.full(values.shape, np.nan, dtype="float64")
    for positions in frame.groupby(level="datetime", sort=False).indices.values():
        positions = np.asarray(positions, dtype="int64")
        block = values[positions]
        if rankdata is not None:
            ranked[positions] = rankdata(
                block, axis=0, method="average", nan_policy="omit"
            )
        else:
            ranked[positions] = pd.DataFrame(block).rank(method="average").to_numpy(
                dtype="float64"
            )
    return pd.DataFrame(ranked, index=frame.index, columns=frame.columns)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Windows can briefly retain a fixed sibling while a monitor reads it.
    # Use a unique temp path so concurrent progress/status writers cannot
    # replace or delete one another's staging file.
    temp = path.with_name(
        f"{path.name}.{os.getpid()}.{time.time_ns()}.part"
    )
    try:
        temp.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        for attempt in range(6):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.03 * (attempt + 1))
    finally:
        temp.unlink(missing_ok=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.unlink(missing_ok=True)
    frame.to_parquet(temp, index=index, compression="zstd")
    os.replace(temp, path)


def _factor_progress_snapshot(stage: str | None = None, state: str | None = None, **extra: Any) -> dict[str, Any]:
    """Summarize family/window factor checkpoints for the outer status file."""
    if CTX is None:
        return {}
    expected_names = set(getattr(V26, "FULL_FACTOR_NAMES", ()))
    completed_names: set[str] = set()
    completed_blocks = 0
    for manifest_path in (CTX.root / "factors").glob("family=*/window=*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("status") != "complete":
            continue
        completed_blocks += len(manifest.get("blocks", []))
        for block in manifest.get("blocks", []):
            completed_names.update(str(name) for name in block.get("columns", []))
    if expected_names:
        completed_names &= expected_names
    completed = len(completed_names)
    expected = len(expected_names)
    payload: dict[str, Any] = {
        "factor_completed": completed,
        "factor_expected": expected,
        "factor_progress_pct": round(100.0 * completed / expected, 3) if expected else 0.0,
        "factor_completed_blocks": completed_blocks,
        "factor_checkpoint_level": "family/window/factor-block",
        "factor_progress_path": str(CTX.root / "factors"),
        "factor_last_event_utc": pd.Timestamp.utcnow().isoformat(),
    }
    if stage is not None:
        payload["factor_current_stage"] = stage
    if state is not None:
        payload["factor_current_state"] = state
    for key in ("family", "window", "factors"):
        if key in extra:
            payload[f"factor_current_{key}"] = extra[key]
    _atomic_json(CTX.out_root / "factor_progress.json", payload)
    return payload


def _event(stage: str, state: str, **extra: Any) -> None:
    if CTX is None:
        return
    path = CTX.root / "stage_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "time_utc": pd.Timestamp.utcnow().isoformat(),
                    "trade_date": CTX.trade_date,
                    "stage": stage,
                    "state": state,
                    **extra,
                },
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )
    _factor_progress_snapshot(stage, state, **extra)


def _timed(stage: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    started = time.perf_counter()
    _event(stage, "running")
    try:
        value = fn(*args, **kwargs)
    except BaseException as exc:
        _event(stage, "failed", elapsed_seconds=time.perf_counter() - started, error=repr(exc))
        raise
    _event(stage, "complete", elapsed_seconds=time.perf_counter() - started)
    return value


def _window_strings(values: Iterable[int], suffix: str) -> tuple[str, ...]:
    return tuple(f"{int(value)}{suffix}" for value in sorted(set(values)))


def configure_registry(config: Mapping[str, Any]) -> None:
    """Build the factor registry from the actual NFF/supplement schemas."""
    global RUNTIME_WINDOWS
    warehouse = Path(config["local_paths"]["warehouse_root"])
    first_date = str(config["run"]["start_date"])
    schemas = discover_warehouse_schema(warehouse, first_date)

    def columns(namespace: str) -> tuple[str, ...]:
        item = schemas.get(namespace)
        return tuple(item.columns) if item is not None else ()

    minute_direction = tuple(
        sorted(
            set(minute_windows(columns("features/minute_nvg")))
            | set(minute_windows(columns("nvg_supplement/minute_nvg_edge_raw")))
        )
    ) or (15, 30, 60, 120)
    minute_topology = tuple(
        sorted(
            set(minute_direction)
            | set(minute_windows(columns("nvg_supplement/minute_visibility_topology_raw")))
        )
    )
    hvg = tuple(
        sorted(
            set(minute_direction)
            | set(minute_windows(columns("nvg_supplement/minute_hvg_risk_raw")))
        )
    )
    trade = tuple(
        sorted(
            set(second_windows(columns("features/trade_nvg")))
            | set(second_windows(columns("nvg_supplement/trade_visibility_edge_raw")))
        )
    ) or (60, 180, 300)

    FF.FAMILY_WINDOWS = {
        "A": _window_strings(minute_direction, "m"),
        "B": _window_strings(minute_direction, "m"),
        "C": _window_strings(minute_topology, "m"),
        "D": _window_strings(minute_direction, "m"),
        "E": _window_strings(trade, "s"),
        "F": _window_strings(trade, "s"),
        "G": _window_strings(hvg, "m"),
        "H": _window_strings(trade, "s"),
        "I": ("1m",),
        "J": ("1m",),
        "K": ("1m",),
    }
    RUNTIME_WINDOWS = dict(FF.FAMILY_WINDOWS)
    specs = FF.expand_specs(V26.PROTOTYPES)
    supplement_specs = []
    for window in (15, 30):
        supplement_specs.append(
            {
                "factor_id": f"full_factor__s01__w{window}m",
                "prototype_id": "S01",
                "family": "S",
                "title": "Exact NVG supplement directional consensus",
                "role": "DIRECTION_ALPHA",
                "direction_prior": "TWO_SIDED",
                "source_fields": "|".join(directional_supplement_fields(window).values()),
                "formula": "raw+detrended price direction; volume and cross-graph structure as confidence",
                "financial_hypothesis": "price direction confirmed by detrended geometry and volume structure is more persistent",
                "window": f"{window}m",
                "transform": "robust cross-sectional z-score and bounded confidence",
                "label_group": "return,state,risk,liquidity,cost",
                "required_gate": "PIT and liquidity/coverage",
                "available_time_rule": "supplement available_time <= decision time; entry next exact minute",
                "expected_range": "approximately [-1,1]",
                "neutralization_allowed": True,
                "cost_relevance": "direct",
                "status": "PENDING_SCHEMA_RESOLUTION",
                "instruction_line": None,
            }
        )
    V26.SPEC_REGISTRY = pd.concat([specs, pd.DataFrame(supplement_specs)], ignore_index=True)
    V26.SUPPLEMENT_FACTOR_NAMES = [row["factor_id"] for row in supplement_specs]
    V26.FULL_FACTOR_NAMES = list(V26.SPEC_REGISTRY["factor_id"])
    R.CORE_DECILE_FEATURES = list(V26.FULL_FACTOR_NAMES)
    BASE.run_temporal_oos.__globals__["DERIVED_FEATURES"] = list(V26.FULL_FACTOR_NAMES)


def _strip_namespace(name: str) -> str:
    for prefix in ("minute_nvg__", "trade_nvg__", "hawkes_lite__", "hawkes_derived__"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _column_exact(frame: pd.DataFrame, name: str) -> pd.Series | None:
    candidates = [name, _strip_namespace(name)]
    bare = _strip_namespace(name)
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
    for candidate in dict.fromkeys(candidates):
        if candidate in frame.columns:
            return pd.to_numeric(frame[candidate], errors="coerce").astype("float32")
    return ORIGINAL["column"](frame, name)


def _add_exact_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    generated: dict[str, pd.Series] = {}
    available = set(frame.columns)
    for window_text in RUNTIME_WINDOWS.get("B", ()):
        if not window_text.endswith("m"):
            continue
        window = int(window_text[:-1])
        fields = directional_supplement_fields(window)
        aliases = {
            f"minute_nvg__price_nvg_{window_text}_terminal_signed_edge_balance": fields["price_edge_balance"],
            f"minute_nvg__price_nvg_{window_text}_terminal_long_edge_signed_slope": fields["price_long_edge_slope"],
            f"minute_nvg__price_nvg_{window_text}_detrended_top_bottom_asymmetry": fields["detrended_edge_balance"],
            f"minute_nvg__price_detrended_nvg_{window_text}_terminal_signed_edge_balance": fields["detrended_edge_balance"],
            f"minute_nvg__price_detrended_nvg_{window_text}_terminal_long_edge_signed_slope": fields["detrended_long_edge_slope"],
            f"minute_nvg__volume_nvg_{window_text}_terminal_signed_edge_balance": fields["volume_edge_balance"],
            f"minute_nvg__volume_nvg_{window_text}_terminal_long_edge_signed_slope": fields["volume_long_edge_slope"],
            f"minute_nvg__price_volume_nvg_{window_text}_edge_weighted_jaccard": fields["price_volume_jaccard"],
            f"minute_nvg__price_volume_nvg_{window_text}_common_edge_slope_corr": fields["price_volume_slope_corr"],
        }
        for target, source in aliases.items():
            if target not in available and source in available:
                generated[target] = pd.to_numeric(frame[source], errors="coerce").astype("float32")
    for source in sorted(available):
        if source.startswith(("trade_price_nvg_", "trade_flow_nvg_", "trade_price_flow_nvg_")):
            target = f"trade_nvg__{source}"
            if target not in available:
                generated[target] = pd.to_numeric(frame[source], errors="coerce").astype("float32")
    if not generated:
        return frame
    block = pd.concat(generated, axis=1, copy=False)
    block.columns = list(generated)
    return pd.concat([frame, block], axis=1, copy=False)


def _merge_supplements_exact(frame: pd.DataFrame, warehouse_root: Path) -> pd.DataFrame:
    return _add_exact_aliases(ORIGINAL["merge_supplements"](frame, warehouse_root))


def _series(frame: pd.DataFrame, *names: str) -> pd.Series | None:
    for name in names:
        value = _column_exact(frame, name)
        if value is not None:
            return value
    return None


def _shift(series: pd.Series, periods: int, frame: pd.DataFrame) -> pd.Series:
    groups = frame.index.get_level_values("instrument")
    return series.groupby(groups, sort=False, group_keys=False).shift(periods)


def _correct_family(
    frame: pd.DataFrame,
    values: dict[str, pd.Series | None],
    family: str,
    window: str,
) -> dict[str, pd.Series | None]:
    result = dict(values)
    if family == "C":
        asym = _series(frame, f"price_nvg_{window}_top_bottom_asymmetry")
        replacement = _series(frame, f"price_nvg_{window}_hub_replacement_strength")
        if asym is not None and replacement is not None:
            result["C08"] = np.sign(asym - _shift(asym, 1, frame)) * replacement
    elif family == "F":
        activity = _series(frame, f"trade_nvg__trade_active_second_ratio_{window}")
        stale = _series(frame, f"trade_nvg__trade_price_stale_ratio_{window}")
        if activity is not None and stale is not None:
            result["F03"] = np.maximum(-stale.diff(), 0) * np.maximum(activity.diff(), 0)
    elif family == "G":
        momentum = _series(frame, "minute_nvg__momentum_15m", "traditional__momentum_15m")
        irreversibility = _series(frame, f"return_hvg_{window}_degree_irreversibility_js")
        if momentum is not None and irreversibility is not None:
            result["G09"] = -np.sign(momentum) * np.maximum(_shift(irreversibility, 5, frame) - irreversibility, 0)
    elif family == "H":
        duration = _series(frame, "hawkes_derived__hawkes_effective_duration_norm")
        if duration is not None:
            result["H23"] = duration
    elif family == "I":
        buy = _series(frame, "trades_1m_core__large_trade_buy_volume_proxy")
        sell = _series(frame, "trades_1m_core__large_trade_sell_volume_proxy")
        share = _series(frame, "trades_1m_core__large_trade_dollar_share")
        if buy is not None and sell is not None and share is not None:
            result["I06"] = (buy - sell) / (buy + sell + EPS) * share
    return result


def _derive_corrected(
    frame: pd.DataFrame, prototype_id: str, window: str
) -> pd.Series | dict[str, pd.Series | None] | None:
    raw = ORIGINAL["derive_prototype"](frame, prototype_id, window)
    family = prototype_id[:1]
    if prototype_id.endswith("__ALL__"):
        return _correct_family(frame, raw if isinstance(raw, dict) else {}, family, window)
    corrected = _correct_family(
        frame,
        {prototype_id: raw if isinstance(raw, pd.Series) else None},
        family,
        window,
    )
    return corrected.get(prototype_id)


def _factor_contract(group: pd.DataFrame, frame: pd.DataFrame) -> str:
    return _hash(
        {
            "version": VERSION,
            "run_contract": None if CTX is None else CTX.contract_hash,
            "index": _index_hash(frame.index),
            "instruction": FF.instruction_hash(),
            "factors": group[["factor_id", "prototype_id", "window", "formula"]].to_dict("records"),
        }
    )


def _derive_all_checkpointed(
    frame: pd.DataFrame, prototypes: list[dict[str, Any]]
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    if CTX is None:
        return ORIGINAL["derive_all"](frame, prototypes)
    specs = V26.SPEC_REGISTRY[V26.SPEC_REGISTRY["family"].isin(list("ABCDEFGHIJK"))]
    blocks: list[pd.DataFrame] = []
    runtime: dict[str, dict[str, Any]] = {}
    for (family, window), group in specs.groupby(["family", "window"], sort=False):
        root = CTX.root / "factors" / f"family={family}" / f"window={window}"
        contract = _factor_contract(group, frame)
        manifest_path = root / "manifest.json"
        reused = False
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "complete" and manifest.get("contract_hash") == contract:
                loaded = []
                valid = True
                for item in manifest.get("blocks", []):
                    path = root / item["path"]
                    if not path.exists():
                        valid = False
                        break
                    part = pd.read_parquet(path)
                    if not part.index.equals(frame.index):
                        valid = False
                        break
                    # Parquet preserves index values but some writers/readers
                    # drop MultiIndex level names.  Restore the live loader
                    # index contract before label code consumes the block.
                    part.index = frame.index
                    loaded.append(part)
                    runtime.update(item.get("factors", {}))
                if valid:
                    blocks.extend(loaded)
                    reused = True
                    _event("factor_block", "reused", family=family, window=window)
        if reused:
            continue
        started = time.perf_counter()
        raw = _derive_corrected(frame, f"{family}__ALL__", str(window))
        values = raw if isinstance(raw, dict) else {}
        columns: dict[str, pd.Series] = {}
        for row in group.itertuples(index=False):
            name = FF.factor_name(row.prototype_id, row.window)
            value = values.get(row.prototype_id)
            if value is None:
                runtime[name] = {"status": "DATA_UNAVAILABLE", "non_null_rate": 0.0}
                continue
            numeric = pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
            columns[name] = numeric
            rate = float(numeric.notna().mean())
            runtime[name] = {"status": "SUCCESS" if rate > 0 else "LOW_COVERAGE", "non_null_rate": rate}
        entries = []
        names = list(columns)
        for block_id, start in enumerate(range(0, len(names), CTX.factor_block_size)):
            selected = names[start : start + CTX.factor_block_size]
            part = pd.concat({name: columns[name] for name in selected}, axis=1, copy=False)
            part.columns = selected
            part.index = frame.index
            path = root / f"block={block_id:03d}.parquet"
            _atomic_parquet(part, path)
            status = {name: runtime[name] for name in selected}
            entries.append({"path": path.name, "columns": selected, "factors": status})
            blocks.append(part)
        _atomic_json(
            manifest_path,
            {
                "status": "complete",
                "contract_hash": contract,
                "family": family,
                "window": window,
                "blocks": entries,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        _event("factor_block", "complete", family=family, window=window, factors=len(names))
    FF._DERIVE_CACHE.clear()
    combined = pd.concat([frame, *blocks], axis=1, copy=False) if blocks else frame
    combined.index = frame.index
    return combined, runtime


def _csz(series: pd.Series, groups: pd.Index) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    grouped = numeric.groupby(groups, sort=False, group_keys=False)
    lower = grouped.transform("quantile", q=0.01)
    upper = grouped.transform("quantile", q=0.99)
    clipped = numeric.clip(lower=lower, upper=upper)
    median = clipped.groupby(groups, sort=False, group_keys=False).transform("median")
    mad = (clipped - median).abs().groupby(groups, sort=False, group_keys=False).transform("median")
    return (clipped - median) / (1.4826 * mad + EPS)


def _checkpoint_supplement_factors(
    frame: pd.DataFrame, generated: dict[str, pd.Series]
) -> dict[str, pd.Series]:
    """Persist and reuse the two S factors so progress reaches all 464 fields."""
    if CTX is None:
        return generated
    result = dict(generated)
    for name, value in list(generated.items()):
        window = name.rsplit("__w", 1)[-1]
        root = CTX.root / "factors" / "family=S" / f"window={window}"
        contract = _hash(
            {
                "version": VERSION,
                "run_contract": CTX.contract_hash,
                "index": _index_hash(frame.index),
                "instruction": FF.instruction_hash(),
                "factor": name,
            }
        )
        manifest_path = root / "manifest.json"
        block_path = root / "block=000.parquet"
        if manifest_path.exists() and block_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                cached = pd.read_parquet(block_path)
                if (
                    manifest.get("status") == "complete"
                    and manifest.get("contract_hash") == contract
                    and cached.index.equals(frame.index)
                    and name in cached.columns
                ):
                    cached.index = frame.index
                    result[name] = cached[name].astype("float32")
                    _event("factor_block", "reused", family="S", window=window, factors=1)
                    continue
            except Exception:
                pass
        part = value.to_frame(name=name)
        part.index = frame.index
        _atomic_parquet(part, block_path)
        _atomic_json(
            manifest_path,
            {
                "status": "complete",
                "contract_hash": contract,
                "family": "S",
                "window": window,
                "blocks": [{"path": block_path.name, "columns": [name]}],
            },
        )
        _event("factor_block", "complete", family="S", window=window, factors=1)
    return result


def _supplement_direction(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    groups = frame.index.get_level_values("datetime")
    generated: dict[str, pd.Series] = {}
    runtime: dict[str, dict[str, Any]] = {}
    for window_text in SUPPLEMENT_WINDOWS:
        window = int(window_text[:-1])
        fields = directional_supplement_fields(window)
        price = [
            _series(frame, fields["price_edge_balance"]),
            _series(frame, fields["price_long_edge_slope"]),
            _series(frame, fields["detrended_edge_balance"]),
            _series(frame, fields["detrended_long_edge_slope"]),
        ]
        price = [value for value in price if value is not None]
        name = f"full_factor__s01__w{window}m"
        if not price:
            runtime[name] = {"status": "DATA_UNAVAILABLE", "non_null_rate": 0.0}
            continue
        z = pd.concat([_csz(value, groups) for value in price], axis=1)
        direction = np.tanh(z).mean(axis=1, skipna=True)
        completeness = z.notna().sum(axis=1) / len(price)
        volume = [
            _series(frame, fields["volume_edge_balance"]),
            _series(frame, fields["volume_long_edge_slope"]),
        ]
        volume = [value for value in volume if value is not None]
        volume_confidence = pd.Series(0.75, index=frame.index)
        if volume:
            strength = pd.concat([_csz(value, groups).abs() for value in volume], axis=1).mean(axis=1)
            volume_confidence = 0.75 + 0.25 * np.tanh(strength.clip(lower=0))
        jaccard = _series(frame, fields["price_volume_jaccard"])
        slope = _series(frame, fields["price_volume_slope_corr"])
        graph_confidence = pd.Series(0.5, index=frame.index)
        if jaccard is not None:
            graph_confidence += 0.25 * jaccard.clip(0, 1).fillna(0)
        if slope is not None:
            graph_confidence += 0.25 * ((slope.clip(-1, 1) + 1) / 2).fillna(0)
        factor = direction * np.sqrt(completeness) * volume_confidence * graph_confidence.clip(0, 1)
        generated[name] = factor.astype("float32")
        rate = float(factor.notna().mean())
        runtime[name] = {"status": "SUCCESS" if rate > 0 else "LOW_COVERAGE", "non_null_rate": rate}
    if not generated:
        return frame, runtime
    generated = _checkpoint_supplement_factors(frame, generated)
    block = pd.concat(generated, axis=1, copy=False)
    block.columns = list(generated)
    result = pd.concat([frame, block], axis=1, copy=False)
    # Concatenating an unnamed checkpoint block can clear MultiIndex level
    # names even when values and row order are unchanged. Restore the live
    # loader index contract before label construction.
    result.index = frame.index
    return result, runtime


def _aggregate_venue_level(level: pd.DataFrame) -> pd.DataFrame:
    """Aggregate venue rows without Python callbacks inside groupby.agg."""
    keys = ["__symbol", "__timestamp"]
    total = level.groupby(keys, sort=False)["volume"].transform("sum")
    share = level["volume"].div(total.replace(0, np.nan))
    work = level.copy(deep=False)
    work["venue_share"] = share
    return work.groupby(keys, sort=False).agg(
        off_exchange_volume=("off_volume", "sum"),
        lit_volume=("lit_volume", "sum"),
        dark_signed_flow=("dark_flow", "sum"),
        lit_signed_flow=("lit_flow", "sum"),
        venue_hhi=("hhi", "sum"),
        venue_entropy=("entropy", "sum"),
        dominant_venue_share=("venue_share", "max"),
        venue_count=("__venue", "nunique"),
    ).reset_index()


def _merge_venue_exact(frame: pd.DataFrame, warehouse_root: Path) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.reset_index()
    work["__symbol"] = work["instrument"].astype("string").str.upper().str.strip()
    work["__timestamp"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
    dates = sorted(work["__timestamp"].dt.tz_convert(SESSION_TZ).dt.strftime("%Y-%m-%d").dropna().unique())
    root = warehouse_root / "canonical" / "trades_venue_1m" / "schema=v1"
    for trade_date in dates:
        paths = sorted((root / f"date={trade_date}").glob("*.parquet"))
        if not paths:
            continue
        venue = pads.dataset([str(path) for path in paths], format="parquet").to_table().to_pandas()
        if venue.empty:
            continue
        venue["__symbol"] = venue["symbol"].astype("string").str.upper().str.strip()
        venue["__timestamp"] = pd.to_datetime(venue["timestamp"], utc=True, errors="coerce")
        if "available_time" in venue:
            available = pd.to_datetime(venue["available_time"], utc=True, errors="coerce")
            venue = venue[available.isna() | (available <= venue["__timestamp"] + pd.Timedelta(minutes=1))]
        venue["volume"] = pd.to_numeric(venue["volume"], errors="coerce").fillna(0)
        venue["signed_dollar_flow_proxy"] = pd.to_numeric(venue["signed_dollar_flow_proxy"], errors="coerce").fillna(0)
        venue["is_off_exchange"] = venue["is_off_exchange"].fillna(False).astype(bool)
        venue["__venue"] = (
            venue["exchange"].astype("string").fillna("NA")
            + ":"
            + venue.get("trf_id", pd.Series("NA", index=venue.index)).astype("string").fillna("NA")
            + ":"
            + venue["is_off_exchange"].astype(int).astype(str)
        )
        level = venue.groupby(["__symbol", "__timestamp", "__venue", "is_off_exchange"], sort=False).agg(
            volume=("volume", "sum"), flow=("signed_dollar_flow_proxy", "sum")
        ).reset_index()
        total = level.groupby(["__symbol", "__timestamp"], sort=False)["volume"].transform("sum")
        share = level["volume"] / total.replace(0, np.nan)
        level["hhi"] = share.pow(2)
        level["entropy"] = -(share.where(share > 0) * np.log(share.where(share > 0)))
        level["off_volume"] = level["volume"].where(level["is_off_exchange"], 0)
        level["lit_volume"] = level["volume"].where(~level["is_off_exchange"], 0)
        level["dark_flow"] = level["flow"].where(level["is_off_exchange"], 0)
        level["lit_flow"] = level["flow"].where(~level["is_off_exchange"], 0)
        agg = _aggregate_venue_level(level)
        total_volume = agg["off_exchange_volume"] + agg["lit_volume"]
        agg["off_exchange_share"] = agg["off_exchange_volume"] / total_volume.replace(0, np.nan)
        agg["dark_lit_divergence"] = agg["dark_signed_flow"] - agg["lit_signed_flow"]
        work = work.merge(agg, on=["__symbol", "__timestamp"], how="left", sort=False, copy=False)
    return work.drop(columns=["__symbol", "__timestamp"], errors="ignore").set_index(["instrument", "datetime"]).sort_index()


def _add_all_features(frame: pd.DataFrame) -> pd.DataFrame:
    _event(
        "feature_input",
        "running",
        rows=int(len(frame)),
        index_names=list(frame.index.names),
        close_non_null=int(
            pd.to_numeric(frame.get("bars_1m__close", pd.Series(dtype="float64")), errors="coerce")
            .notna()
            .sum()
        ),
    )
    frame = V26._ensure_research_index(frame)
    _event(
        "feature_input",
        "canonicalized",
        rows=int(len(frame)),
        index_names=list(frame.index.names),
        close_non_null=int(
            pd.to_numeric(frame.get("bars_1m__close", pd.Series(dtype="float64")), errors="coerce")
            .notna()
            .sum()
        ),
    )
    frame = _timed("merge_supplements", FF.merge_supplements, frame, R.WAREHOUSE_ROOT)
    frame = _timed("merge_sketch", FF.merge_canonical_sketch, frame, R.WAREHOUSE_ROOT)
    frame = _timed("merge_condition", FF.merge_condition_aggregates, frame, R.WAREHOUSE_ROOT)
    frame = _timed("merge_venue", FF.merge_venue_aggregates, frame, R.WAREHOUSE_ROOT)
    hawkes_columns = [column for column in frame if column.startswith("hawkes_lite__")]
    if hawkes_columns:
        block = R.V2.add_hawkes_derived(frame[hawkes_columns])
        derived = [column for column in block if column.startswith("hawkes_derived__")]
        if derived:
            frame = pd.concat([frame, block[derived]], axis=1, copy=False)
    traditional_columns = [
        column
        for column in ("bars_1m__close", "bars_1m__vwap", "bars_1m__dollar_volume")
        if column in frame
    ]
    if traditional_columns:
        block = R.V2.add_traditional_factors(frame[traditional_columns])
        derived = [column for column in block if column.startswith("traditional__")]
        if derived:
            frame = pd.concat([frame, block[derived]], axis=1, copy=False)
    frame, runtime = _timed("derive_factors", FF.derive_all, frame, V26.PROTOTYPES)
    frame, supplement_runtime = _supplement_direction(frame)
    runtime.update(supplement_runtime)
    V26.RUNTIME_FACTOR_STATUS.clear()
    V26.RUNTIME_FACTOR_STATUS.update(runtime)
    keep_prefixes = (
        "full_factor__",
        "bars_1m__",
        "trades_1m_core__",
        "traditional__",
        "hawkes_lite__",
        "hawkes_derived__",
    )
    keep_exact = {
        "off_exchange_share",
        "dark_signed_flow",
        "lit_signed_flow",
        "dark_lit_divergence",
        "venue_hhi",
        "venue_entropy",
        "dominant_venue_share",
    }
    keep = [column for column in frame if column.startswith(keep_prefixes) or column in keep_exact]
    # The v2.7 installer replaces the v2.6 builder, so canonicalize here as
    # well.  Otherwise labels see a repaired local frame while controls and
    # portfolio stages still receive the original unnamed/reversed index.
    result = frame[keep].select_dtypes(include=[np.number]).replace(
        [np.inf, -np.inf], np.nan
    ).astype("float32")
    result = V26._ensure_research_index(result)
    _event(
        "feature_output",
        "complete",
        rows=int(len(result)),
        index_names=list(result.index.names),
        close_non_null=int(
            pd.to_numeric(result.get("bars_1m__close", pd.Series(dtype="float64")), errors="coerce")
            .notna()
            .sum()
        ),
        columns=int(result.shape[1]),
    )
    return result


def _future_exact_fast(frame: pd.DataFrame, column: str, offset_minutes: int) -> pd.Series:
    """Lookup exact future timestamps through one cached symbol-wide matrix."""
    key = (id(frame), column, int(offset_minutes))
    cached = FUTURE_CACHE.get(key)
    if cached is not None:
        return cached
    wide_key = (id(frame), column)
    try:
        cached_wide = FUTURE_WIDE_CACHE.get(wide_key)
        if cached_wide is None:
            source = pd.to_numeric(frame[column], errors="coerce")
            wide = source.unstack(level="instrument")
            cached_wide = (
                pd.DatetimeIndex(pd.to_datetime(wide.index, utc=True)),
                wide.columns,
                np.asarray(wide, dtype="float64"),
            )
            FUTURE_WIDE_CACHE[wide_key] = cached_wide
        wide_times, wide_symbols, wide_values = cached_wide
        index = frame.index
        times = pd.DatetimeIndex(pd.to_datetime(index.get_level_values("datetime"), utc=True))
        symbols = pd.Index(index.get_level_values("instrument"))
        time_positions = wide_times.get_indexer(times + pd.Timedelta(minutes=int(offset_minutes)))
        symbol_positions = wide_symbols.get_indexer(symbols)
        valid = (time_positions >= 0) & (symbol_positions >= 0)
        values = np.full(len(index), np.nan, dtype="float64")
        values[valid] = wide_values[time_positions[valid], symbol_positions[valid]]
        result = pd.Series(values, index=index, name=column)
    except (KeyError, ValueError, TypeError):
        result = ORIGINAL["future_exact"](frame, column, int(offset_minutes))
    FUTURE_CACHE[key] = result
    return result


def _future_exact_cached(frame: pd.DataFrame, column: str, offset_minutes: int) -> pd.Series:
    key = (id(frame), column, int(offset_minutes))
    cached = FUTURE_CACHE.get(key)
    if cached is not None:
        return cached
    return _future_exact_fast(frame, column, offset_minutes)


def _ranked_ic_stats_vectorized(
    feature_frame: pd.DataFrame,
    label: pd.Series,
    feature_columns: list[str],
    min_n: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Compute the reference IC statistics with matrix-level accumulators.

    The v2.4 fast path called a Python/NumPy correlation helper once per
    feature per minute. That preserved the formula but made the full A-K
    screen spend most of its time in millions of tiny ``isfinite`` calls.
    This implementation keeps the same average-rank and demeaned-percentile
    rank contracts while accumulating all feature columns in one vectorized
    pass per minute.
    """
    feature_count = len(feature_columns)
    minute_values: list[list[float]] = [[] for _ in feature_columns]
    minute_counts = np.zeros(feature_count, dtype="int64")
    pooled_n = np.zeros(feature_count, dtype="float64")
    pooled_sx = np.zeros(feature_count, dtype="float64")
    pooled_sy = np.zeros(feature_count, dtype="float64")
    pooled_sxx = np.zeros(feature_count, dtype="float64")
    pooled_syy = np.zeros(feature_count, dtype="float64")
    pooled_sxy = np.zeros(feature_count, dtype="float64")
    admitted_minutes = 0
    work = pd.concat([feature_frame[feature_columns], label.rename("__label")], axis=1)

    for _, block in work.groupby(level="datetime", sort=False):
        ranked_y_raw = block["__label"].rank(method="average")
        y_raw = ranked_y_raw.to_numpy(dtype="float64")
        y_valid = np.isfinite(y_raw)
        if int(y_valid.sum()) < min_n:
            continue
        admitted_minutes += 1

        ranked_x_raw = _rank_frame_average(block[feature_columns])
        x_raw = ranked_x_raw.to_numpy(dtype="float64", copy=False)
        valid = np.isfinite(x_raw) & y_valid[:, None]
        n = valid.sum(axis=0).astype("float64")
        x0 = np.where(valid, x_raw, 0.0)
        y0 = np.where(valid, y_raw[:, None], 0.0)
        sx = x0.sum(axis=0, dtype="float64")
        sy = y0.sum(axis=0, dtype="float64")
        sxx = (x0 * x0).sum(axis=0, dtype="float64")
        syy = (y0 * y0).sum(axis=0, dtype="float64")
        sxy = (x0 * y0).sum(axis=0, dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = sxy - sx * sy / n
            vx = sxx - sx * sx / n
            vy = syy - sy * sy / n
            corr = cov / np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0))
        eligible = (n >= min_n) & np.isfinite(corr)
        minute_counts += n.astype("int64")
        for idx in np.flatnonzero(eligible):
            minute_values[idx].append(float(corr[idx]))

        label_count = int(y_valid.sum())
        y_pooled = y_raw / float(label_count)
        y_pooled[~y_valid] = np.nan
        y_pooled -= np.nanmean(y_pooled)
        feature_counts = np.isfinite(x_raw).sum(axis=0).astype("float64")
        x_pooled = x_raw / feature_counts[None, :]
        feature_means = np.divide(
            np.nansum(x_pooled, axis=0),
            feature_counts,
            out=np.full(feature_count, np.nan, dtype="float64"),
            where=feature_counts > 0,
        )
        x_pooled -= feature_means[None, :]
        xp0 = np.where(valid, x_pooled, 0.0)
        yp0 = np.where(valid, y_pooled[:, None], 0.0)
        pooled_n += n
        pooled_sx += xp0.sum(axis=0, dtype="float64")
        pooled_sy += yp0.sum(axis=0, dtype="float64")
        pooled_sxx += (xp0 * xp0).sum(axis=0, dtype="float64")
        pooled_syy += (yp0 * yp0).sum(axis=0, dtype="float64")
        pooled_sxy += (xp0 * yp0).sum(axis=0, dtype="float64")

    minute_out: dict[str, dict[str, float]] = {}
    pooled_out: dict[str, dict[str, float]] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled_cov = pooled_sxy - pooled_sx * pooled_sy / pooled_n
        pooled_vx = pooled_sxx - pooled_sx * pooled_sx / pooled_n
        pooled_vy = pooled_syy - pooled_sy * pooled_sy / pooled_n
        pooled_corr = pooled_cov / np.sqrt(
            np.maximum(pooled_vx, 0.0) * np.maximum(pooled_vy, 0.0)
        )
    for idx, feature in enumerate(feature_columns):
        values = np.asarray(minute_values[idx], dtype="float64")
        value = float(pooled_corr[idx]) if pooled_n[idx] >= min_n and np.isfinite(pooled_corr[idx]) else math.nan
        minute_out[feature] = {
            "ic_minutes": int(values.size),
            "ic_count": int(minute_counts[idx]),
            "rank_ic_mean": float(values.mean()) if values.size else math.nan,
            "rank_ic_std": float(values.std(ddof=1)) if values.size > 1 else math.nan,
            "rank_ic_positive_ratio": float((values > 0).mean()) if values.size else math.nan,
        }
        pooled_out[feature] = {
            "ic_minutes": admitted_minutes,
            "ic_count": int(pooled_n[idx]),
            "rank_ic_mean": value,
            "rank_ic_std": math.nan,
            "rank_ic_positive_ratio": float(value > 0) if np.isfinite(value) else math.nan,
        }
    return minute_out, pooled_out


def _ranked_ic_stats_from_ranked_features_fast(
    ranked_features: pd.DataFrame,
    label: pd.Series,
    feature_columns: list[str],
    min_n: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Compute IC from a cached rank matrix without per-minute DataFrame concat.

    The feature ranks are already cross-sectional.  The remaining work is
    performed on NumPy blocks grouped by integer datetime codes, so the
    464-column matrix is scanned once per label instead of repeatedly copying
    and aligning a wide pandas frame.
    """
    feature_count = len(feature_columns)
    feature_values = ranked_features[feature_columns].to_numpy(dtype="float64", copy=False)
    aligned_label = label.reindex(ranked_features.index).to_numpy(dtype="float64", copy=False)
    datetimes = ranked_features.index.get_level_values("datetime")
    datetime_codes = pd.factorize(datetimes, sort=False)[0]
    order = np.argsort(datetime_codes, kind="stable")
    sorted_codes = datetime_codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(order)]

    minute_corrs: list[np.ndarray] = []
    minute_counts = np.zeros(feature_count, dtype="int64")
    pooled_n = np.zeros(feature_count, dtype="float64")
    pooled_sx = np.zeros(feature_count, dtype="float64")
    pooled_sy = np.zeros(feature_count, dtype="float64")
    pooled_sxx = np.zeros(feature_count, dtype="float64")
    pooled_syy = np.zeros(feature_count, dtype="float64")
    pooled_sxy = np.zeros(feature_count, dtype="float64")
    admitted_minutes = 0

    for start, end in zip(starts, ends):
        positions = order[start:end]
        x_raw = feature_values[positions]
        y_values = aligned_label[positions]
        ranked_y = pd.Series(y_values).rank(method="average").to_numpy(dtype="float64")
        y_valid = np.isfinite(ranked_y)
        if int(y_valid.sum()) < min_n:
            continue
        admitted_minutes += 1

        feature_valid = np.isfinite(x_raw)
        valid = feature_valid & y_valid[:, None]
        n = valid.sum(axis=0).astype("float64")
        x0 = np.where(valid, x_raw, 0.0)
        y0 = np.where(y_valid, ranked_y, 0.0)
        sx = x0.sum(axis=0, dtype="float64")
        sy = y0 @ valid
        sxx = (x0 * x0).sum(axis=0, dtype="float64")
        syy = (y0 * y0) @ valid
        sxy = y0 @ x0
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = sxy - sx * sy / n
            vx = sxx - sx * sx / n
            vy = syy - sy * sy / n
            corr = cov / np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0))
        corr[(n < min_n) | ~np.isfinite(corr)] = np.nan
        minute_corrs.append(corr)
        minute_counts += n.astype("int64")

        label_count = int(y_valid.sum())
        y_pooled = ranked_y / float(label_count)
        y_pooled[~y_valid] = np.nan
        y_pooled -= np.nanmean(y_pooled)
        feature_counts = feature_valid.sum(axis=0).astype("float64")
        x_pooled = np.divide(
            x_raw,
            feature_counts[None, :],
            out=np.full_like(x_raw, np.nan, dtype="float64"),
            where=feature_counts[None, :] > 0,
        )
        feature_means = np.divide(
            np.nansum(x_pooled, axis=0),
            feature_counts,
            out=np.full(feature_count, np.nan, dtype="float64"),
            where=feature_counts > 0,
        )
        x_pooled -= feature_means[None, :]
        xp0 = np.where(valid, x_pooled, 0.0)
        yp0 = np.where(valid, y_pooled[:, None], 0.0)
        pooled_n += n
        pooled_sx += xp0.sum(axis=0, dtype="float64")
        pooled_sy += yp0.sum(axis=0, dtype="float64")
        pooled_sxx += (xp0 * xp0).sum(axis=0, dtype="float64")
        pooled_syy += ((y_pooled * y_pooled)[:, None] * valid).sum(axis=0, dtype="float64")
        pooled_sxy += y_pooled @ xp0

    with np.errstate(divide="ignore", invalid="ignore"):
        pooled_cov = pooled_sxy - pooled_sx * pooled_sy / pooled_n
        pooled_vx = pooled_sxx - pooled_sx * pooled_sx / pooled_n
        pooled_vy = pooled_syy - pooled_sy * pooled_sy / pooled_n
        pooled_corr = pooled_cov / np.sqrt(
            np.maximum(pooled_vx, 0.0) * np.maximum(pooled_vy, 0.0)
        )
    minute_matrix = (
        np.vstack(minute_corrs) if minute_corrs else np.empty((0, feature_count), dtype="float64")
    )
    minute_out: dict[str, dict[str, float]] = {}
    pooled_out: dict[str, dict[str, float]] = {}
    for idx, feature in enumerate(feature_columns):
        values = minute_matrix[:, idx]
        values = values[np.isfinite(values)]
        value = float(pooled_corr[idx]) if pooled_n[idx] >= min_n and np.isfinite(pooled_corr[idx]) else math.nan
        minute_out[feature] = {
            "ic_minutes": int(values.size),
            "ic_count": int(minute_counts[idx]),
            "rank_ic_mean": float(values.mean()) if values.size else math.nan,
            "rank_ic_std": float(values.std(ddof=1)) if values.size > 1 else math.nan,
            "rank_ic_positive_ratio": float((values > 0).mean()) if values.size else math.nan,
        }
        pooled_out[feature] = {
            "ic_minutes": admitted_minutes,
            "ic_count": int(pooled_n[idx]),
            "rank_ic_mean": value,
            "rank_ic_std": math.nan,
            "rank_ic_positive_ratio": float(value > 0) if np.isfinite(value) else math.nan,
        }
    return minute_out, pooled_out


def _ranked_ic_stats_from_ranked_features(
    ranked_features: pd.DataFrame,
    label: pd.Series,
    feature_columns: list[str],
    min_n: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Compatibility name for the batched ranked-feature IC implementation."""
    return _ranked_ic_stats_from_ranked_features_fast(
        ranked_features, label, feature_columns, min_n
    )


def _minute_rank_ic_summary_rank_cache(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_masks: dict[str, pd.Series],
    controls: pd.DataFrame,
    trade_date: str,
    min_n: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str], tuple[pd.DataFrame, pd.Series]]]:
    """Run the fast summary while ranking each universe mask only once.

    Universes are independent at this stage. Running them concurrently keeps
    the exact per-universe mask and residualization contracts while avoiding a
    long single-threaded IC pass over all 464 executable factors.
    """
    feature_columns = R.analysis_features(features)
    decile_cache_features = [column for column in R.CORE_DECILE_FEATURES if column in feature_columns]
    masks = R.universe_masks(features, controls)
    # ``own_feature_universe`` is a compatibility alias for the full PIT mask.
    # Do not repeat the expensive per-label/per-factor cross-sectional ranking.
    canonical_masks: dict[str, pd.Series] = {}
    universe_aliases: dict[str, str] = {}
    for universe, universe_mask in masks.items():
        if (
            universe == "own_feature_universe"
            and "all_pit_eligible" in canonical_masks
            and universe_mask.equals(canonical_masks["all_pit_eligible"])
        ):
            universe_aliases[universe] = "all_pit_eligible"
            continue
        canonical_masks[universe] = universe_mask
    def _process_universe(
        universe: str, universe_mask: pd.Series
    ) -> tuple[list[dict[str, Any]], dict[tuple[str, str], tuple[pd.DataFrame, pd.Series]]]:
        universe_rows: list[dict[str, Any]] = []
        universe_residual_cache: dict[tuple[str, str], tuple[pd.DataFrame, pd.Series]] = {}
        raw_rank_cache: dict[str, pd.DataFrame] = {}
        neutral_rank_cache: dict[str, pd.DataFrame] = {}
        residual_feature_cache: dict[tuple[str, str], pd.DataFrame] = {}
        for label_column in labels.columns:
            family, horizon_text = label_column.rsplit("__h", 1)
            horizon = int(horizon_text)
            base_mask = (
                universe_mask
                & labels[label_column].notna()
                & label_masks[label_column].fillna(False)
            ).fillna(False)
            if int(base_mask.sum()) < min_n:
                continue
            feature_frame = features.loc[base_mask, feature_columns]
            label = labels.loc[base_mask, label_column]
            index_key = _index_hash(feature_frame.index)
            ranked_raw = raw_rank_cache.get(index_key)
            if ranked_raw is None:
                ranked_raw = _rank_frame_by_datetime_average(feature_frame[feature_columns])
                raw_rank_cache[index_key] = ranked_raw
            minute_stats, pooled_stats = _ranked_ic_stats_from_ranked_features(
                ranked_raw, label, feature_columns, min_n
            )
            label_non_null = int(label.notna().sum())
            coverage = feature_frame.notna().sum(axis=0) / max(1, label_non_null)
            R.append_ic_rows(universe_rows, minute_stats, trade_date, universe, "raw", "minute_mean_cs_rank_ic", family, horizon, coverage, label_non_null)
            R.append_ic_rows(universe_rows, pooled_stats, trade_date, universe, "raw", "pooled_cs_demeaned_pct_rank_ic", family, horizon, coverage, label_non_null)

            if not family.startswith("return_") or universe not in {"common_structural", "liquid_common_adv20_top1000"}:
                continue
            controls_sub = controls.loc[base_mask]
            residual_key = _index_hash(feature_frame.index)
            feature_resid = residual_feature_cache.get((universe, residual_key))
            if feature_resid is None:
                feature_resid = R._residualize_matrix(feature_frame, controls_sub, min_n=max(min_n, 40))
                residual_feature_cache[(universe, residual_key)] = feature_resid
            label_resid = R._residualize_matrix(label.to_frame(label_column), controls_sub, min_n=max(min_n, 40))[label_column]
            ranked_resid = neutral_rank_cache.get(residual_key)
            if ranked_resid is None:
                ranked_resid = _rank_frame_by_datetime_average(feature_resid[feature_columns])
                neutral_rank_cache[residual_key] = ranked_resid
            neut_minute, neut_pooled = _ranked_ic_stats_from_ranked_features(ranked_resid, label_resid, feature_columns, min_n)
            neutral_label_non_null = int(label_resid.notna().sum())
            neutral_coverage = feature_resid.notna().sum(axis=0) / max(1, neutral_label_non_null)
            if family in {"return_open_to_open", "return_vwap_to_vwap"} and decile_cache_features:
                universe_residual_cache[(universe, label_column)] = (
                    feature_resid[decile_cache_features].copy(deep=False), label_resid.copy(deep=False)
                )
            R.append_ic_rows(universe_rows, neut_minute, trade_date, universe, "neutralized", "minute_mean_cs_rank_ic_residualized", family, horizon, neutral_coverage, neutral_label_non_null)
            R.append_ic_rows(universe_rows, neut_pooled, trade_date, universe, "neutralized", "pooled_cs_demeaned_pct_rank_ic_residualized", family, horizon, neutral_coverage, neutral_label_non_null)
        raw_rank_cache.clear()
        neutral_rank_cache.clear()
        residual_feature_cache.clear()
        return universe_rows, universe_residual_cache

    rows: list[dict[str, Any]] = []
    residual_cache: dict[tuple[str, str], tuple[pd.DataFrame, pd.Series]] = {}
    max_workers = min(3, max(1, len(masks)))
    canonical_rows: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ic-universe") as executor:
        futures = [
            executor.submit(_process_universe, universe, universe_mask)
            for universe, universe_mask in canonical_masks.items()
        ]
        for future in futures:
            universe_rows, universe_cache = future.result()
            rows.extend(universe_rows)
            residual_cache.update(universe_cache)
            if universe_rows:
                canonical_rows[universe_rows[0]["universe"]] = universe_rows
    for alias, canonical in universe_aliases.items():
        rows.extend(
            [{**row, "universe": alias} for row in canonical_rows.get(canonical, [])]
        )
    return pd.DataFrame(rows), residual_cache


def _stage_cache(name: str, function: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    def wrapped(*args: Any, **kwargs: Any) -> pd.DataFrame:
        path = None if CTX is None else CTX.root / "stages" / f"{name}.parquet"
        if path is not None and path.exists():
            _event(name, "reused")
            return pd.read_parquet(path)
        result = _timed(name, function, *args, **kwargs)
        if path is not None:
            _atomic_parquet(result, path, index=False)
        return result
    return wrapped


def _datetime_first_portfolio_frame(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Keep portfolio weights index-compatible with the reference implementation."""
    index = frame.index
    if not isinstance(index, pd.MultiIndex):
        return frame
    names = list(index.names)
    if "datetime" not in names or "instrument" not in names:
        return frame
    desired = ["datetime", "instrument"] + [name for name in names if name not in {"datetime", "instrument"}]
    if names == desired:
        return frame
    return frame.reorder_levels(desired).sort_index()


def _portfolio_index_safe(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Normalize the four aligned portfolio inputs before legacy weight lookup."""
    if len(args) < 4:
        return R.staggered_portfolio_proxy(*args, **kwargs)
    normalized = list(args)
    normalized[0] = _datetime_first_portfolio_frame(args[0])
    normalized[1] = _datetime_first_portfolio_frame(args[1])
    normalized[2] = {
        name: _datetime_first_portfolio_frame(value)
        for name, value in args[2].items()
    }
    normalized[3] = _datetime_first_portfolio_frame(args[3])
    return ORIGINAL["portfolio"](*normalized, **kwargs)


def _residualize_cached(values: pd.DataFrame, controls: pd.DataFrame, min_n: int = 40) -> pd.DataFrame:
    if CTX is None:
        return ORIGINAL["residualize"](values, controls, min_n=min_n)
    token = _hash(
        {
            "version": VERSION,
            "index": _index_hash(values.index),
            "columns": list(values.columns),
            "controls": list(controls.columns),
            "min_n": min_n,
            "contract": CTX.contract_hash,
        }
    )[:20]
    path = CTX.root / "neutralization" / f"residual={token}.parquet"
    if path.exists():
        cached = pd.read_parquet(path)
        if cached.index.equals(values.index) and list(cached.columns) == list(values.columns):
            return cached
    result = ORIGINAL["residualize"](values, controls, min_n=min_n)
    _atomic_parquet(result, path)
    return result


def _decile_feature_rows(
    feature: str,
    signal: pd.Series,
    label: pd.Series,
    adv: pd.Series,
    price: pd.Series,
    trades: pd.Series,
    metadata: dict[str, Any],
    min_n: int,
) -> list[dict[str, Any]]:
    work = pd.concat(
        [
            signal.rename("signal"),
            label.rename("label"),
            adv.rename("adv"),
            price.rename("price"),
            trades.rename("trades"),
        ],
        axis=1,
    ).dropna(subset=["signal", "label"])
    if work.empty:
        return []
    minute = work.index.get_level_values("datetime").minute
    work = work.loc[(minute % 15) == 0]
    if work.empty:
        return []
    categories = pd.Categorical(work.index.get_level_values("datetime"))
    minute_codes = categories.codes.astype("int32")
    minute_counts = np.bincount(minute_codes, minlength=len(categories.categories))
    eligible = minute_counts >= min_n
    ranks = work["signal"].groupby(level="datetime", sort=False).rank(method="first", pct=True)
    deciles = np.ceil(ranks.to_numpy(dtype="float64") * 10).clip(1, 10).astype("int16")
    combined = minute_codes * 10 + deciles - 1
    size = len(categories.categories) * 10

    def aggregate(column: str) -> tuple[np.ndarray, np.ndarray]:
        values = pd.to_numeric(work[column], errors="coerce").to_numpy(dtype="float64")
        valid = eligible[minute_codes] & np.isfinite(values)
        sums = np.bincount(combined[valid], weights=values[valid], minlength=size).reshape(-1, 10)
        counts = np.bincount(combined[valid], minlength=size).reshape(-1, 10)
        means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        return means, counts

    label_means, label_counts = aggregate("label")
    adv_means, _ = aggregate("adv")
    price_means, _ = aggregate("price")
    trade_means, _ = aggregate("trades")
    rows = []
    for decile in range(10):
        def mean(values: np.ndarray) -> float:
            column = values[:, decile]
            return float(np.nanmean(column)) if np.isfinite(column).any() else math.nan

        rows.append(
            {
                **metadata,
                "feature": feature,
                "bundle": R.infer_bundle(feature),
                "rebalance": "sample_15m",
                "decile": decile + 1,
                "mean_label": mean(label_means),
                "count": int(label_counts[:, decile].sum()),
                "mean_log_adv20": mean(adv_means),
                "mean_price": mean(price_means),
                "mean_trade_count": mean(trade_means),
            }
        )
    return rows


def _deciles_fast(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_masks: dict[str, pd.Series],
    controls: pd.DataFrame,
    residual_cache: dict[tuple[str, str], tuple[pd.DataFrame, pd.Series]],
    trade_date: str,
    min_n: int,
) -> pd.DataFrame:
    path = None if CTX is None else CTX.root / "stages" / "decile_curves.parquet"
    if path is not None and path.exists():
        return pd.read_parquet(path)
    masks = R.universe_masks(features, controls)
    factor_columns = [column for column in R.CORE_DECILE_FEATURES if column in features]
    rows: list[dict[str, Any]] = []
    workers = 1 if CTX is None else max(1, CTX.intra_workers)
    for universe in ("liquid_common_adv20_top1000", "common_structural"):
        for label_column in (
            column
            for column in labels
            if column.startswith("return_open_to_open__h") or column.startswith("return_vwap_to_vwap__h")
        ):
            family, horizon = label_column.rsplit("__h", 1)
            base_mask = (masks[universe] & labels[label_column].notna() & label_masks[label_column].fillna(False)).fillna(False)
            if int(base_mask.sum()) < min_n:
                continue
            base_label = labels.loc[base_mask, label_column]
            adv = controls.loc[base_mask, "control__log_adv20"]
            price = features.loc[base_mask, "bars_1m__close"]
            trades = features.loc[base_mask].get(
                "trades_1m_core__trade_count", pd.Series(np.nan, index=base_label.index)
            )
            metadata = {
                "trade_date": trade_date,
                "universe": universe,
                "label_family": family,
                "horizon_bars": int(horizon),
            }
            for variant in ("raw", "winsorized", "neutralized"):
                if variant == "neutralized":
                    cached = residual_cache.get((universe, label_column))
                    if cached is None:
                        continue
                    signals = cached[0].reindex(base_label.index)
                    target = cached[1].reindex(base_label.index)
                    available = [column for column in factor_columns if column in signals]
                else:
                    signals = features.loc[base_mask, factor_columns]
                    target = base_label
                    available = list(signals.columns)
                    if variant == "winsorized":
                        groups = signals.index.get_level_values("datetime")
                        grouped = signals.groupby(groups, sort=False, group_keys=False)
                        signals = signals.clip(
                            lower=grouped.transform("quantile", q=0.01),
                            upper=grouped.transform("quantile", q=0.99),
                        )
                variant_meta = {**metadata, "variant": variant}
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(
                            _decile_feature_rows,
                            feature,
                            signals[feature],
                            target,
                            adv,
                            price,
                            trades,
                            variant_meta,
                            min_n,
                        )
                        for feature in available
                    ]
                    for future in futures:
                        rows.extend(future.result())
    result = pd.DataFrame(rows)
    if path is not None and not result.empty:
        _atomic_parquet(result, path, index=False)
    return result


def _install_context(config: Mapping[str, Any]) -> None:
    # Call the original v2.1 date implementation directly.  The v2.4
    # profiling wrapper captures the pre-v2.6 label/selector functions and
    # can otherwise silently route workers back to the legacy path.
    original = V26.FAST._ORIGINALS.get("run_date", R.run_date)
    original_globals = getattr(original, "__globals__", {})
    # The optimized runner stores the pre-patch run_date function and its
    # module globals separately. Updating only R.<name> is insufficient when
    # that captured function belongs to a legacy module dictionary.
    original_globals.update(
        {
            "add_all_features": _add_all_features,
            "build_labels_and_masks": V26._full_build_labels_and_masks,
            "analysis_features": V26._full_analysis_features,
        }
    )
    atomic = config.get("atomic", {}) if isinstance(config.get("atomic"), Mapping) else {}
    factor_block_size = int(atomic.get("factor_block_size", 8))
    intra_workers = int(atomic.get("intra_date_workers", 8))

    def run_date(*args: Any, **kwargs: Any) -> dict[str, Any]:
        global CTX
        trade_date = str(args[0] if args else kwargs["trade_date"])
        out_root = Path(args[2] if len(args) > 2 else kwargs["out_root"])
        contract = str(kwargs.get("contract_hash") or (args[5] if len(args) > 5 else "unknown"))
        CTX = Context(trade_date, out_root, contract, factor_block_size, intra_workers)
        CTX.root.mkdir(parents=True, exist_ok=True)
        write_schema_audit(R.WAREHOUSE_ROOT, CTX.root / "schema", trade_date)
        _event("date", "running", factor_block_size=factor_block_size, intra_workers=intra_workers)
        try:
            # Rebind the live module entry points immediately before the date
            # call.  The legacy optimized runner captures wrappers at import
            # time; without this guard a worker can silently use the old
            # observed-row labels and representative selector.
            R.add_all_features = _add_all_features
            R.build_labels_and_masks = V26._full_build_labels_and_masks
            R.analysis_features = V26._full_analysis_features
            result = original(*args, **kwargs)
            _event("date", "complete", status=result.get("status"))
            return result
        except BaseException as exc:
            _event("date", "failed", error=repr(exc))
            raise
        finally:
            audit_dir = out_root / "02_neutralized_factor_diagnostics" / f"date={trade_date}"
            audit_dir.mkdir(parents=True, exist_ok=True)
            (audit_dir / "label_index_audit.json").write_text(
                json.dumps(V26.LAST_LABEL_AUDIT, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            FUTURE_CACHE.clear()
            FUTURE_WIDE_CACHE.clear()
            CTX = None

    R.run_date = run_date


def _install_worker_command() -> None:
    original = R.worker_command

    def worker_command(*args: Any, **kwargs: Any) -> list[str]:
        command = original(*args, **kwargs)
        current = str(Path(__file__).resolve())
        legacy = {str(Path(R.__file__).resolve()), str(Path(V26.__file__).resolve())}
        return [current if value in legacy else value for value in command]

    R.worker_command = worker_command


def _install_progress_status_bridge() -> None:
    original_update_status = R.update_status

    def update_status(path: Path, **updates: Any) -> None:
        progress_path = path.parent / "factor_progress.json"
        if progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except Exception:
                progress = {}
            if isinstance(progress, dict):
                updates = {**progress, **updates}
        original_update_status(path, **updates)

    R.update_status = update_status


def install(config: Mapping[str, Any]) -> None:
    ORIGINAL.update(
        {
            "column": FF._column,
            "merge_supplements": FF.merge_supplements,
            "derive_prototype": FF.derive_prototype,
            "derive_all": FF.derive_all,
            "future_exact": V26._future_exact,
            "residualize": R._residualize_matrix,
            "portfolio": R.staggered_portfolio_proxy,
        }
    )
    FF._column = _column_exact
    FF.merge_supplements = _merge_supplements_exact
    FF.merge_venue_aggregates = _merge_venue_exact
    FF.derive_prototype = _derive_corrected
    FF.derive_all = _derive_all_checkpointed
    V26._future_exact = _future_exact_cached
    # The v2.4 summary function resolves ranked_ic_stats_once in its own
    # module globals, so replace that hot helper explicitly after the fast
    # path has been installed.
    V26.FAST.ranked_ic_stats_once = _ranked_ic_stats_vectorized
    R.minute_rank_ic_summary = lambda *args, **kwargs: _timed(
        "ic_and_neutralization", _minute_rank_ic_summary_rank_cache, *args, **kwargs
    )
    R.add_all_features = _add_all_features
    original_join_controls = R.join_daily_controls

    def join_daily_controls_canonical(
        features: pd.DataFrame, trade_date: str, controls_path: Path
    ) -> pd.DataFrame:
        # A legacy/profile wrapper can return a fresh frame after the feature
        # builder.  Canonicalize at the last shared boundary and update the
        # caller object in place so labels, controls and later portfolio code
        # cannot observe different index names/order or duplicate keys.
        normalized = V26._ensure_research_index(features)
        if normalized is not features or not normalized.index.equals(features.index):
            features.__init__(normalized)
        return original_join_controls(features, trade_date, controls_path)

    R.join_daily_controls = join_daily_controls_canonical
    R._residualize_matrix = _residualize_cached
    R.decile_curves = _deciles_fast
    R.staggered_portfolio_proxy = _stage_cache("portfolio_proxy", _portfolio_index_safe)
    _install_context(config)
    _install_worker_command()
    _install_progress_status_bridge()


def _write_architecture(run_root: Path, config: Mapping[str, Any]) -> None:
    atomic = config.get("atomic", {}) if isinstance(config.get("atomic"), Mapping) else {}
    outer = int(config["run"].get("initial_parallel", 2))
    inner = int(atomic.get("intra_date_workers", 8))
    path = run_root / "reports" / "architecture"
    path.mkdir(parents=True, exist_ok=True)
    (path / "v2_7_atomic_architecture.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "outer_date_processes": outer,
                "intra_date_workers": inner,
                "nominal_compute_slots": outer * inner,
                "reason": "date workers retain large shared frames; intra-date threads fill CPU without duplicating the frame",
                "checkpoint_levels": [
                    "date/schema",
                    "family/window/factor block",
                    "neutralization matrix",
                    "decile stage",
                    "portfolio stage",
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _config_path() -> Path:
    if "--config" in sys.argv:
        position = sys.argv.index("--config")
        if position + 1 < len(sys.argv):
            return Path(sys.argv[position + 1]).resolve()
    return Path("configs/v2_7_atomic_full_campaign.yaml").resolve()


def main() -> int:
    config_path = _config_path()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if "--worker" in sys.argv:
        V26._patch()
        V26._wrap_run_date()
        configure_registry(config)
        install(config)
        return R.main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--run-name")
    parser.add_argument("--research-root")
    parser.add_argument("--controls-path")
    # Runtime-only scheduler flags are intentionally forwarded to the base
    # runner. They do not alter the research contract when resuming an
    # existing run with --resume-existing-contract.
    parser.add_argument("--run-id")
    parser.add_argument("--out-root")
    parser.add_argument("--contract-hash")
    parser.add_argument("--resume-existing-contract", action="store_true")
    parser.add_argument("--parallel", type=int)
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--min-parallel", type=int)
    parser.add_argument("--memory-min-available-gb", type=float)
    cli = parser.parse_args()
    if cli.start_date:
        config["run"]["start_date"] = cli.start_date
    if cli.end_date:
        config["run"]["end_date"] = cli.end_date
    if cli.run_name:
        config["run"]["name"] = cli.run_name
    if cli.research_root:
        config.setdefault("local_paths", {})["research_root"] = cli.research_root
    V26._patch()
    V26._wrap_run_date()
    configure_registry(config)
    install(config)
    run_root = Path(config["local_paths"]["research_root"]) / "runs" / config["run"]["name"]
    _write_architecture(run_root, config)
    # V2.6 reloads its YAML inside run(); persist the CLI-resolved contract so
    # bounded benchmarks and resumed runs cannot silently use the base dates.
    effective_config_path = run_root / "effective_config.yaml"
    effective_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    # The legacy scheduler does not know --run-name; feed it the resolved
    # config and its accepted --run-id equivalent for bounded runs.
    original_argv = list(sys.argv)
    forwarded_argv: list[str] = []
    skip_next = False
    for value in sys.argv:
        if skip_next:
            skip_next = False
            continue
        if value in {"--run-name", "--research-root"}:
            skip_next = True
            continue
        if value == "--config":
            forwarded_argv.extend([value, str(effective_config_path)])
            skip_next = True
            continue
        forwarded_argv.append(value)
    if "--run-id" not in forwarded_argv:
        forwarded_argv.extend(["--run-id", str(config["run"]["name"])])
    sys.argv[:] = forwarded_argv
    try:
        return V26.run(effective_config_path)
    finally:
        sys.argv[:] = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
