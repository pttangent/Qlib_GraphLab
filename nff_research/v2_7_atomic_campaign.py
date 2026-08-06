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


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _index_hash(index: pd.Index) -> str:
    values = pd.util.hash_pandas_object(index, index=True).to_numpy(dtype="uint64")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temp, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.unlink(missing_ok=True)
    frame.to_parquet(temp, index=index, compression="zstd")
    os.replace(temp, path)


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
    for window in minute_direction:
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
    return (pd.concat([frame, *blocks], axis=1, copy=False) if blocks else frame), runtime


def _csz(series: pd.Series, groups: pd.Index) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    grouped = numeric.groupby(groups, sort=False, group_keys=False)
    lower = grouped.transform("quantile", q=0.01)
    upper = grouped.transform("quantile", q=0.99)
    clipped = numeric.clip(lower=lower, upper=upper)
    median = clipped.groupby(groups, sort=False, group_keys=False).transform("median")
    mad = (clipped - median).abs().groupby(groups, sort=False, group_keys=False).transform("median")
    return (clipped - median) / (1.4826 * mad + EPS)


def _supplement_direction(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    groups = frame.index.get_level_values("datetime")
    generated: dict[str, pd.Series] = {}
    runtime: dict[str, dict[str, Any]] = {}
    for window_text in RUNTIME_WINDOWS.get("B", ()):
        if not window_text.endswith("m"):
            continue
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
    block = pd.concat(generated, axis=1, copy=False)
    block.columns = list(generated)
    return pd.concat([frame, block], axis=1, copy=False), runtime


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
        agg = level.groupby(["__symbol", "__timestamp"], sort=False).agg(
            off_exchange_volume=("off_volume", "sum"),
            lit_volume=("lit_volume", "sum"),
            dark_signed_flow=("dark_flow", "sum"),
            lit_signed_flow=("lit_flow", "sum"),
            venue_hhi=("hhi", "sum"),
            venue_entropy=("entropy", "sum"),
            dominant_venue_share=("volume", lambda s: float(s.max() / s.sum()) if float(s.sum()) > 0 else np.nan),
            venue_count=("__venue", "nunique"),
        ).reset_index()
        total_volume = agg["off_exchange_volume"] + agg["lit_volume"]
        agg["off_exchange_share"] = agg["off_exchange_volume"] / total_volume.replace(0, np.nan)
        agg["dark_lit_divergence"] = agg["dark_signed_flow"] - agg["lit_signed_flow"]
        work = work.merge(agg, on=["__symbol", "__timestamp"], how="left", sort=False, copy=False)
    return work.drop(columns=["__symbol", "__timestamp"], errors="ignore").set_index(["instrument", "datetime"]).sort_index()


def _add_all_features(frame: pd.DataFrame) -> pd.DataFrame:
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
    return frame[keep].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).astype("float32")


def _future_exact_cached(frame: pd.DataFrame, column: str, offset_minutes: int) -> pd.Series:
    key = (id(frame), column, int(offset_minutes))
    cached = FUTURE_CACHE.get(key)
    if cached is not None:
        return cached
    value = ORIGINAL["future_exact"](frame, column, offset_minutes)
    FUTURE_CACHE[key] = value
    return value


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
    original = R.run_date
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
            result = original(*args, **kwargs)
            _event("date", "complete", status=result.get("status"))
            return result
        except BaseException as exc:
            _event("date", "failed", error=repr(exc))
            raise
        finally:
            FUTURE_CACHE.clear()
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
    R.add_all_features = _add_all_features
    R._residualize_matrix = _residualize_cached
    R.decile_curves = _deciles_fast
    R.staggered_portfolio_proxy = _stage_cache("portfolio_proxy", R.staggered_portfolio_proxy)
    _install_context(config)
    _install_worker_command()


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
    V26._patch()
    V26._wrap_run_date()
    configure_registry(config)
    install(config)
    if "--worker" in sys.argv:
        return R.main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(config_path))
    args = parser.parse_args()
    run_root = Path(config["local_paths"]["research_root"]) / "runs" / config["run"]["name"]
    _write_architecture(run_root, config)
    return V26.run(Path(args.config).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
