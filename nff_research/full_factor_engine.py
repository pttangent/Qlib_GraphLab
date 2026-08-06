"""Executable registry and derivations for the A-K factor contract.

The governing specification lives outside the repository.  This module reads
its prototype headings/formula blocks at run time, stores the source hash in
the run contract, and only marks a prototype executable when all of its data
dependencies are present.  Missing families are retained as explicit
DATA_UNAVAILABLE registry rows instead of being silently dropped.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as pads


EPS = 1e-8
_DERIVE_CACHE: dict[tuple[int, int], dict[str, Any]] = {}
INSTRUCTION_PATH = Path(
    r"C:\Users\A001\Downloads\NFF → Qlib 分钟级全因子研究、Walk-forward 与账户回测无人值守总指令.md"
)
FAMILY_WINDOWS = {
    "A": ("10m", "15m", "30m"),
    "B": ("10m", "15m", "30m"),
    "C": ("10m", "15m", "30m", "60m"),
    "D": ("10m", "15m", "30m"),
    "E": ("60s", "180s", "300s"),
    "F": ("60s", "180s", "300s"),
    "G": ("15m", "30m", "60m"),
    "H": ("60s", "180s", "300s"),
    "I": ("1m",),
    "J": ("1m",),
    "K": ("1m",),
}


def instruction_hash(path: Path = INSTRUCTION_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _section_family(lines: list[str], position: int) -> str:
    for line in reversed(lines[:position]):
        match = re.match(r"^# ([A-K])\. ", line.strip())
        if match:
            return match.group(1)
    return "?"


def parse_prototypes(path: Path = INSTRUCTION_PATH) -> list[dict[str, Any]]:
    """Parse the authoritative A01..K12 headings and first formula block."""
    lines = path.read_text(encoding="utf-8").splitlines()
    heading = re.compile(r"^## ([A-K]\d{2}) (.+?)\s*$")
    found: list[tuple[int, str, str]] = []
    for index, line in enumerate(lines):
        match = heading.match(line.strip())
        if match:
            found.append((index, match.group(1), match.group(2)))
    rows: list[dict[str, Any]] = []
    for item, (start, prototype_id, title) in enumerate(found):
        end = found[item + 1][0] if item + 1 < len(found) else len(lines)
        section = _section_family(lines, start)
        body = lines[start:end]
        formula = ""
        in_block = False
        block: list[str] = []
        for line in body[1:]:
            if line.strip() == "```text":
                if not in_block:
                    in_block = True
                    block = []
                continue
            if in_block and line.strip() == "```":
                if block:
                    formula = "\n".join(block).strip()
                    break
                in_block = False
                continue
            if in_block:
                block.append(line)
        source_tokens = sorted(
            set(
                re.findall(
                    r"[A-Za-z][A-Za-z0-9_]*(?:_[A-Za-z0-9]+)*",
                    formula,
                )
            )
        )
        rows.append(
            {
                "prototype_id": prototype_id,
                "family": section,
                "title": title,
                "formula": formula,
                "source_tokens": source_tokens,
                "instruction_line": start + 1,
            }
        )
    return rows


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()


SUPPLEMENT_DATASETS = (
    "minute_hvg_risk_raw",
    "minute_nvg_edge_raw",
    "minute_visibility_topology_raw",
    "trade_visibility_edge_raw",
    "trade_visibility_topology_raw",
)


def merge_canonical_sketch(frame: pd.DataFrame, warehouse_root: Path) -> pd.DataFrame:
    """Join the sketch table, whose published schema has no available_time key."""
    if frame.empty:
        return frame
    work = frame.reset_index()
    work["__symbol"] = work["instrument"].astype("string").str.upper().str.strip()
    work["__timestamp"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
    dates = sorted(work["__timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d").dropna().unique())
    root = warehouse_root / "canonical" / "trades_1m_sketch" / "schema=v1"
    for trade_date in dates:
        files = sorted((root / f"date={trade_date}").glob("*.parquet"))
        if not files:
            continue
        try:
            sketch = pads.dataset([str(path) for path in files], format="parquet").to_table().to_pandas()
        except Exception:
            continue
        if sketch.empty or "symbol" not in sketch.columns or "timestamp" not in sketch.columns:
            continue
        sketch["__symbol"] = sketch["symbol"].astype("string").str.upper().str.strip()
        sketch["__timestamp"] = pd.to_datetime(sketch["timestamp"], utc=True, errors="coerce")
        value_columns = [c for c in sketch.columns if c not in {"trade_date", "symbol_id", "symbol", "timestamp", "date", "__symbol", "__timestamp"}]
        if not value_columns:
            continue
        sketch = sketch[["__symbol", "__timestamp", *value_columns]].drop_duplicates(["__symbol", "__timestamp"], keep="last")
        sketch = sketch.rename(columns={c: f"trades_1m_sketch__{c}" for c in value_columns})
        work = work.merge(sketch, on=["__symbol", "__timestamp"], how="left", sort=False, copy=False)
    return work.drop(columns=["__symbol", "__timestamp"], errors="ignore").set_index(["instrument", "datetime"]).sort_index()


def merge_condition_aggregates(frame: pd.DataFrame, warehouse_root: Path) -> pd.DataFrame:
    """Aggregate condition-code rows before joining them to symbol-minute."""
    if frame.empty:
        return frame
    work = frame.reset_index()
    work["__symbol"] = work["instrument"].astype("string").str.upper().str.strip()
    work["__timestamp"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
    dates = sorted(work["__timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d").dropna().unique())
    root = warehouse_root / "canonical" / "trades_condition_1m" / "schema=v1"
    for trade_date in dates:
        files = sorted((root / f"date={trade_date}").glob("*.parquet"))
        if not files:
            continue
        try:
            condition = pads.dataset([str(path) for path in files], format="parquet").to_table().to_pandas()
        except Exception:
            continue
        if condition.empty or "symbol" not in condition.columns or "timestamp" not in condition.columns:
            continue
        condition["__symbol"] = condition["symbol"].astype("string").str.upper().str.strip()
        condition["__timestamp"] = pd.to_datetime(condition["timestamp"], utc=True, errors="coerce")
        numeric = [c for c in condition.columns if c.endswith("_count") or c.endswith("_volume") or c == "dollar_volume"]
        if not numeric:
            continue
        for col in numeric:
            condition[col] = pd.to_numeric(condition[col], errors="coerce").fillna(0.0)
        agg = condition.groupby(["__symbol", "__timestamp"], sort=False)[numeric].sum().reset_index()
        agg = agg.rename(columns={c: f"trades_condition_1m__{c}" for c in numeric})
        work = work.merge(agg, on=["__symbol", "__timestamp"], how="left", sort=False, copy=False)
    return work.drop(columns=["__symbol", "__timestamp"], errors="ignore").set_index(["instrument", "datetime"]).sort_index()


def merge_venue_aggregates(frame: pd.DataFrame, warehouse_root: Path) -> pd.DataFrame:
    """Aggregate the published venue tape to the symbol-minute research key."""
    if frame.empty:
        return frame
    work = frame.reset_index()
    work["__symbol"] = work["instrument"].astype("string").str.upper().str.strip()
    work["__timestamp"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
    dates = sorted(work["__timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d").dropna().unique())
    root = warehouse_root / "canonical" / "trades_venue_1m" / "schema=v1"
    for trade_date in dates:
        partition = root / f"date={trade_date}"
        files = sorted(partition.glob("*.parquet"))
        if not files:
            continue
        try:
            venue = pads.dataset([str(path) for path in files], format="parquet").to_table().to_pandas()
        except Exception:
            continue
        if venue.empty:
            continue
        venue["__symbol"] = venue["symbol"].astype("string").str.upper().str.strip()
        venue["__timestamp"] = pd.to_datetime(venue["timestamp"], utc=True, errors="coerce")
        if "available_time" in venue.columns:
            available = pd.to_datetime(venue["available_time"], utc=True, errors="coerce")
            venue = venue[available.isna() | (available <= venue["__timestamp"] + pd.Timedelta(minutes=1))]
        for col in ("volume", "dollar_volume", "buy_volume_proxy", "sell_volume_proxy", "signed_dollar_flow_proxy"):
            if col in venue:
                venue[col] = pd.to_numeric(venue[col], errors="coerce").fillna(0.0)
        venue["is_off_exchange"] = venue["is_off_exchange"].fillna(False).astype(bool)
        venue["__off_volume"] = venue["volume"].where(venue["is_off_exchange"], 0.0)
        venue["__lit_volume"] = venue["volume"].where(~venue["is_off_exchange"], 0.0)
        venue["__off_signed_flow"] = venue["signed_dollar_flow_proxy"].where(venue["is_off_exchange"], 0.0)
        venue["__lit_signed_flow"] = venue["signed_dollar_flow_proxy"].where(~venue["is_off_exchange"], 0.0)
        grouped = venue.groupby(["__symbol", "__timestamp"], sort=False)
        agg = grouped.agg(
            off_exchange_volume=("__off_volume", "sum"),
            lit_volume=("__lit_volume", "sum"),
            dark_signed_flow=("__off_signed_flow", "sum"),
            lit_signed_flow=("__lit_signed_flow", "sum"),
            venue_count=("exchange", "nunique"),
        ).reset_index()
        total = agg["off_exchange_volume"] + agg["lit_volume"]
        agg["off_exchange_share"] = agg["off_exchange_volume"] / total.replace(0, np.nan)
        agg["dark_lit_divergence"] = agg["dark_signed_flow"] - agg["lit_signed_flow"]
        agg["venue_hhi"] = 1.0 / agg["venue_count"].clip(lower=1)
        agg["venue_entropy"] = np.log(agg["venue_count"].clip(lower=1))
        work = work.merge(agg, on=["__symbol", "__timestamp"], how="left", sort=False, copy=False)
    return work.drop(columns=["__symbol", "__timestamp"], errors="ignore").set_index(["instrument", "datetime"]).sort_index()


def merge_supplements(frame: pd.DataFrame, warehouse_root: Path) -> pd.DataFrame:
    """Join all published NVG/HVG/topology supplement tables by PIT key.

    The supplement is intentionally loaded outside NFFDataLoader because it is
    a separate, published warehouse namespace.  Only columns from the exact
    local trade date are joined and all source keys are normalized before the
    merge.  Raw supplement columns are audit/dependency inputs; the caller
    decides which derived factors become Alpha.
    """
    if frame.empty:
        return frame
    work = frame.reset_index()
    work["__symbol"] = work["instrument"].astype("string").str.upper().str.strip()
    work["__timestamp"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
    dates = sorted(work["__timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d").dropna().unique())
    for dataset in SUPPLEMENT_DATASETS:
        root = warehouse_root / "nvg_supplement" / dataset / "schema=v1"
        for trade_date in dates:
            partition = root / f"date={trade_date}"
            files = sorted(partition.glob("*.parquet"))
            if not files:
                continue
            try:
                supplement = pads.dataset([str(path) for path in files], format="parquet").to_table().to_pandas()
            except Exception:
                continue
            if supplement.empty or "symbol" not in supplement.columns or "timestamp" not in supplement.columns:
                continue
            supplement["__symbol"] = supplement["symbol"].astype("string").str.upper().str.strip()
            supplement["__timestamp"] = pd.to_datetime(supplement["timestamp"], utc=True, errors="coerce")
            if "available_time" in supplement.columns:
                available = pd.to_datetime(supplement["available_time"], utc=True, errors="coerce")
                supplement = supplement[available.isna() | (available <= supplement["__timestamp"] + pd.Timedelta(minutes=1))]
            keep = [c for c in supplement.columns if c not in {"symbol", "timestamp", "available_time", "trade_date", "symbol_id", "date", "__symbol", "__timestamp"}]
            if not keep:
                continue
            supplement = supplement[["__symbol", "__timestamp", *keep]].drop_duplicates(["__symbol", "__timestamp"], keep="last")
            work = work.merge(supplement, on=["__symbol", "__timestamp"], how="left", sort=False, copy=False)
    result = work.drop(columns=["__symbol", "__timestamp"], errors="ignore").set_index(["instrument", "datetime"])
    return result.sort_index()


def factor_name(prototype_id: str, window: str) -> str:
    return f"full_factor__{prototype_id.lower()}__w{_safe_name(window)}"


def expand_specs(prototypes: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    roles = {
        "A": "DIRECTION_ALPHA",
        "B": "DIRECTION_ALPHA",
        "C": "REGIME",
        "D": "CONFIRMATION",
        "E": "DIRECTION_ALPHA",
        "F": "LIQUIDITY",
        "G": "RISK",
        "H": "RISK",
        "I": "LIQUIDITY",
        "J": "REGIME",
        "K": "CONFIRMATION",
    }
    directions = {
        "A": "TWO_SIDED",
        "B": "TWO_SIDED",
        "C": "NONE",
        "D": "TWO_SIDED",
        "E": "TWO_SIDED",
        "F": "NONE",
        "G": "NONE",
        "H": "TWO_SIDED",
        "I": "TWO_SIDED",
        "J": "TWO_SIDED",
        "K": "TWO_SIDED",
    }
    for prototype in prototypes:
        family = prototype["family"]
        for window in FAMILY_WINDOWS.get(family, ("1m",)):
            rows.append(
                {
                    "factor_id": factor_name(prototype["prototype_id"], window),
                    "prototype_id": prototype["prototype_id"],
                    "family": family,
                    "title": prototype["title"],
                    "role": roles.get(family, "CONTROL"),
                    "direction_prior": directions.get(family, "TWO_SIDED"),
                    "source_fields": "|".join(prototype["source_tokens"]),
                    "formula": prototype["formula"],
                    "financial_hypothesis": prototype["title"],
                    "window": window,
                    "transform": "raw,CSZ,CSR,neutralized",
                    "label_group": "return,state,risk,liquidity,cost",
                    "required_gate": "PIT; quality gate for activity/Hawkes families",
                    "available_time_rule": "available_time <= planned_order_time; entry t+1",
                    "expected_range": "unbounded or documented bounded transform",
                    "neutralization_allowed": family not in {"F", "G", "J"},
                    "cost_relevance": "direct" if family in {"E", "F", "H", "I", "J"} else "diagnostic",
                    "status": "PENDING_SCHEMA_RESOLUTION",
                    "instruction_line": prototype["instruction_line"],
                }
            )
    return pd.DataFrame(rows)


def _groups(frame: pd.DataFrame) -> pd.Series:
    key = (id(frame), len(frame.columns))
    bucket = _DERIVE_CACHE.setdefault(key, {})
    return bucket.setdefault("groups", frame.index.get_level_values("datetime"))


def _instrument_groups(frame: pd.DataFrame) -> pd.Series:
    key = (id(frame), len(frame.columns))
    bucket = _DERIVE_CACHE.setdefault(key, {})
    return bucket.setdefault("instrument_groups", frame.index.get_level_values("instrument"))


def _group_shift(series: pd.Series, periods: int, groups: pd.Series) -> pd.Series:
    return series.groupby(groups, sort=False, group_keys=False).shift(periods)


def _csz(series: pd.Series, groups: pd.Series) -> pd.Series:
    cacheable = series.name is not None
    cache = _DERIVE_CACHE.setdefault((id(series), len(series)), {}) if cacheable else {}
    cache_key = ("csz", id(groups))
    if cacheable and cache_key in cache:
        return cache[cache_key]
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    grouped = numeric.groupby(groups, sort=False, group_keys=False)
    # Keep the same 1%/99% winsorization contract, but use grouped quantile
    # transforms instead of a Python lambda per minute. The lambda path is
    # disproportionately expensive for a full day of symbol-minute rows.
    lower = grouped.transform("quantile", q=0.01)
    upper = grouped.transform("quantile", q=0.99)
    clipped = numeric.clip(lower=lower, upper=upper)
    median = clipped.groupby(groups, sort=False, group_keys=False).transform("median")
    mad = (clipped - median).abs().groupby(groups, sort=False, group_keys=False).transform("median")
    result = (clipped - median) / (1.4826 * mad + EPS)
    if cacheable:
        cache[cache_key] = result
    return result


def _csr(series: pd.Series, groups: pd.Series) -> pd.Series:
    cacheable = series.name is not None
    cache = _DERIVE_CACHE.setdefault((id(series), len(series)), {}) if cacheable else {}
    cache_key = ("csr", id(groups))
    if cacheable and cache_key not in cache:
        cache[cache_key] = series.groupby(groups, sort=False, group_keys=False).rank(pct=True).mul(2.0).sub(1.0)
    return cache[cache_key] if cacheable else series.groupby(groups, sort=False, group_keys=False).rank(pct=True).mul(2.0).sub(1.0)


def _same(left: pd.Series, right: pd.Series) -> pd.Series:
    return ((np.sign(left) == np.sign(right)) & (np.sign(left) != 0)).astype("float32")


def _conf(left: pd.Series, right: pd.Series, groups: pd.Series) -> pd.Series:
    return np.sign(left) * np.minimum(_csz(left, groups).abs(), _csz(right, groups).abs()) * _same(left, right)


def _asym(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left - right) / (left.abs() + right.abs() + EPS)


def _window_number(window: str) -> int:
    match = re.search(r"\d+", window)
    return int(match.group(0)) if match else 1


def _column(frame: pd.DataFrame, name: str) -> pd.Series | None:
    """Resolve an instruction field to a namespaced NFF column."""
    bucket = _DERIVE_CACHE.setdefault((id(frame), len(frame.columns)), {})
    column_cache = bucket.setdefault("columns", {})
    if name in column_cache:
        return column_cache[name]
    candidates = [name]
    if name.startswith("bars_1m__") or name.startswith("trades_1m_core__"):
        candidates.append(name)
    else:
        candidates.extend(
            [
                f"minute_nvg__{name}",
                f"trade_nvg__{name}",
                f"hawkes_lite__{name}",
                f"hawkes_derived__{name}",
            ]
        )
    for candidate in candidates:
        if candidate in frame.columns:
            value = pd.to_numeric(frame[candidate], errors="coerce").astype("float32")
            column_cache[name] = value
            return value
    column_cache[name] = None
    return None
    return None


def _raw(frame: pd.DataFrame, stem: str, window: str) -> pd.Series | None:
    candidates = [
        f"{stem}_{window}",
        f"{stem}_{window.replace('m', 'm').replace('s', 's')}",
    ]
    for candidate in candidates:
        value = _column(frame, candidate)
        if value is not None:
            return value
    return None


def _first(frame: pd.DataFrame, *values: pd.Series | None) -> pd.Series | None:
    for value in values:
        if value is not None:
            return value
    return None


def _session_features(frame: pd.DataFrame) -> dict[str, pd.Series]:
    bucket = _DERIVE_CACHE.setdefault((id(frame), len(frame.columns)), {})
    if "session" in bucket:
        return bucket["session"]
    close = _column(frame, "bars_1m__close")
    open_px = _column(frame, "bars_1m__open")
    if close is None or open_px is None:
        bucket["session"] = {}
        return bucket["session"]
    groups = _instrument_groups(frame)
    first_open = open_px.groupby(groups, sort=False, group_keys=False).transform("first")
    running_high = close.groupby(groups, sort=False, group_keys=False).cummax()
    running_low = close.groupby(groups, sort=False, group_keys=False).cummin()
    result = {
        "open_mom": close / first_open.replace(0, np.nan) - 1.0,
        "session_range_dir": 2.0 * (close - running_low) / (running_high - running_low + EPS) - 1.0,
        "drawdown": close / running_high.replace(0, np.nan) - 1.0,
    }
    bucket["session"] = result
    return result


def _hawkes(frame: pd.DataFrame, name: str, window: str | None = None) -> pd.Series | None:
    stem = name if window is None else f"{name}_{window}"
    return _column(frame, stem)


def derive_prototype(frame: pd.DataFrame, prototype_id: str, window: str) -> pd.Series | dict[str, pd.Series | None] | None:
    """Derive one contract prototype.  Missing source dependencies return None."""
    groups = _groups(frame)
    w = _window_number(window)
    minutes = f"{w}m" if window.endswith("m") else window
    sec = window
    session = _session_features(frame)
    mom = _first(_raw(frame, "momentum", minutes), _raw(frame, "traditional__momentum", minutes))
    path_change = _raw(frame, "price_path", minutes)
    if path_change is None:
        path_change = _column(frame, f"minute_nvg__price_path_{minutes}_signed_change")
    path_eff = _column(frame, f"minute_nvg__price_path_{minutes}_efficiency")
    path_rough = _column(frame, f"minute_nvg__price_path_{minutes}_roughness")
    path_pos = _column(frame, f"minute_nvg__price_path_{minutes}_terminal_position")
    path_range = _column(frame, f"minute_nvg__price_path_{minutes}_range")
    edge = _first(_column(frame, f"minute_nvg__price_nvg_{minutes}_terminal_signed_edge_balance"), _column(frame, f"minute_nvg__price_nvg_{minutes}_top_bottom_asymmetry"))
    top_edge = _column(frame, f"minute_nvg__price_nvg_{minutes}_top_terminal_slope_mean")
    bottom_edge = _column(frame, f"minute_nvg__price_nvg_{minutes}_bottom_terminal_slope_mean")
    top_long = _column(frame, f"minute_nvg__price_nvg_{minutes}_top_terminal_long_edge_ratio")
    bottom_long = _column(frame, f"minute_nvg__price_nvg_{minutes}_bottom_terminal_long_edge_ratio")
    top_visible = _column(frame, f"minute_nvg__price_nvg_{minutes}_top_terminal_visible_fraction")
    bottom_visible = _column(frame, f"minute_nvg__price_nvg_{minutes}_bottom_terminal_visible_fraction")
    top_degree = _column(frame, f"minute_nvg__price_nvg_{minutes}_top_terminal_degree")
    bottom_degree = _column(frame, f"minute_nvg__price_nvg_{minutes}_bottom_terminal_degree")
    top_span = _column(frame, f"minute_nvg__price_nvg_{minutes}_top_terminal_span_mean")
    bottom_span = _column(frame, f"minute_nvg__price_nvg_{minutes}_bottom_terminal_span_mean")
    top_gap = _column(frame, f"minute_nvg__price_nvg_{minutes}_top_terminal_value_gap_mean")
    bottom_gap = _column(frame, f"minute_nvg__price_nvg_{minutes}_bottom_terminal_value_gap_mean")
    top_mass = _column(frame, f"minute_nvg__price_nvg_{minutes}_top_terminal_visibility_mass")
    bottom_mass = _column(frame, f"minute_nvg__price_nvg_{minutes}_bottom_terminal_visibility_mass")
    volume_edge = _column(frame, f"minute_nvg__volume_nvg_{minutes}_terminal_signed_edge_balance")
    volume_long = _column(frame, f"minute_nvg__volume_nvg_{minutes}_terminal_long_edge_signed_slope")
    volume_change = _column(frame, f"minute_nvg__volume_path_{minutes}_signed_change")
    volume_eff = _column(frame, f"minute_nvg__volume_path_{minutes}_efficiency")
    overlap = _column(frame, f"minute_nvg__price_volume_terminal_overlap_{minutes}")
    confirmation = _column(frame, f"minute_nvg__price_volume_nvg_confirmation_{minutes}")
    tprice = _column(frame, f"trade_nvg__trade_price_path_{sec}_signed_change")
    tflow = _column(frame, f"trade_nvg__trade_flow_path_{sec}_signed_change")
    tprice_eff = _column(frame, f"trade_nvg__trade_price_path_{sec}_efficiency")
    tflow_eff = _column(frame, f"trade_nvg__trade_flow_path_{sec}_efficiency")
    tprice_rough = _column(frame, f"trade_nvg__trade_price_path_{sec}_roughness")
    tflow_rough = _column(frame, f"trade_nvg__trade_flow_path_{sec}_roughness")
    tprice_edge = _column(frame, f"trade_nvg__trade_price_nvg_{sec}_top_bottom_asymmetry")
    tflow_edge = _column(frame, f"trade_nvg__trade_flow_nvg_{sec}_top_bottom_asymmetry")
    toverlap = _column(frame, f"trade_nvg__trade_price_flow_terminal_overlap_{sec}")
    activity = _column(frame, f"trade_nvg__trade_active_second_ratio_{sec}")
    coverage = _column(frame, f"trade_nvg__trade_observation_coverage_{sec}")
    stale = _column(frame, f"trade_nvg__trade_price_stale_ratio_{sec}")
    hawkes_pressure = _hawkes(frame, "hawkes_derived__hawkes_signed_pressure")
    hawkes_intensity = _hawkes(frame, "hawkes_lite__hawkes_total_intensity")
    hawkes_imbalance = _hawkes(frame, "hawkes_lite__hawkes_intensity_imbalance")
    hawkes_endogenous = _hawkes(frame, "hawkes_lite__hawkes_endogenous_share")
    hawkes_branching = _hawkes(frame, "hawkes_lite__hawkes_branching_ratio_max")
    hawkes_persistence = _hawkes(frame, "hawkes_derived__hawkes_persistence")
    hawkes_exog = _hawkes(frame, "hawkes_derived__hawkes_exogenous_shock", sec)
    hawkes_endog = _hawkes(frame, "hawkes_derived__hawkes_endogenous_shock", sec)
    hawkes_shock = _hawkes(frame, "hawkes_lite__hawkes_shock_score", sec)
    signed_flow = _column(frame, "trades_1m_core__signed_dollar_flow_proxy")
    ofi = _column(frame, "trades_1m_core__buy_volume_proxy")
    sell = _column(frame, "trades_1m_core__sell_volume_proxy")
    volume = _column(frame, "trades_1m_core__volume")
    if volume is None:
        volume = _column(frame, "bars_1m__volume")
    dollar = _column(frame, "trades_1m_core__dollar_volume")
    if dollar is None:
        dollar = _column(frame, "bars_1m__dollar_volume")
    close = _column(frame, "bars_1m__close")
    frame_cache = _DERIVE_CACHE.setdefault((id(frame), len(frame.columns)), {})
    if close is not None and "ret1" not in frame_cache:
        frame_cache["ret1"] = close.groupby(_instrument_groups(frame), sort=False, group_keys=False).pct_change(fill_method=None)
    ret1 = frame_cache.get("ret1") if close is not None else None

    def c(value: pd.Series | None) -> pd.Series | None:
        return value

    p = prototype_id
    m10 = _raw(frame, "momentum", "10m")
    m15 = _raw(frame, "momentum", "15m")
    m30 = _raw(frame, "momentum", "30m")

    def finish(value: pd.Series | None) -> pd.Series | None:
        if value is None:
            return None
        return pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")

    def finish_map(values: dict[str, pd.Series | None]) -> dict[str, pd.Series | None]:
        return {key: finish(value) for key, value in values.items()}
    # A: traditional/path baselines.
    a = {
        "A01": mom,
        "A02": -mom if mom is not None else None,
        "A03": path_change * path_eff if path_change is not None and path_eff is not None else None,
        "A04": path_change / (path_range.abs() + EPS) if path_change is not None and path_range is not None else None,
        "A05": path_change * path_eff / (path_rough.abs() + EPS) if path_change is not None and path_eff is not None and path_rough is not None else None,
        "A06": (2 * path_pos - 1) * path_eff if path_pos is not None and path_eff is not None else None,
        "A07": -np.sign(path_change) * (1 - path_eff) * path_rough if path_change is not None and path_eff is not None and path_rough is not None else None,
        "A08": 1.0 / (1.0 + _csz(path_range, groups).abs() + _csz(path_rough, groups).abs()) if path_range is not None and path_rough is not None else None,
        "A09": session.get("open_mom"),
        "A10": _column(frame, "traditional__vwap_dislocation"),
        "A11": -_column(frame, "traditional__vwap_dislocation") if _column(frame, "traditional__vwap_dislocation") is not None else None,
        "A12": session.get("session_range_dir"),
        "A13": -_csz(session["drawdown"], groups) if "drawdown" in session else None,
        "A14": pd.concat([m10, m15, m30], axis=1).apply(np.sign).mean(axis=1) if all(value is not None for value in (m10, m15, m30)) else None,
        "A15": _csz(m10, groups) - _csz(m30, groups) if m10 is not None and m30 is not None else None,
        "A16": _csz(m10, groups) - 2 * _csz(m15, groups) + _csz(m30, groups) if all(value is not None for value in (m10, m15, m30)) else None,
    }
    if p == "A__ALL__":
        return finish_map(a)
    if p.startswith("A"):
        return finish(a.get(p))
    b = {
        "B01": edge,
        "B02": _first(_column(frame, f"minute_nvg__price_nvg_{minutes}_terminal_long_edge_signed_slope"), top_long),
        "B03": edge / (_column(frame, f"minute_nvg__price_nvg_{minutes}_terminal_slope_std").abs() + EPS) if edge is not None and _column(frame, f"minute_nvg__price_nvg_{minutes}_terminal_slope_std") is not None else None,
        "B04": _csz(_first(top_long, edge), groups) - _csz(_first(top_edge, edge), groups) if _first(top_long, edge) is not None and _first(top_edge, edge) is not None else None,
        "B05": _column(frame, f"minute_nvg__price_nvg_{minutes}_top_bottom_asymmetry"),
        "B06": _column(frame, f"minute_nvg__price_nvg_{minutes}_overextension_change_1m"),
        "B07": np.sign(mom) * _column(frame, f"minute_nvg__price_nvg_{minutes}_top_bottom_asymmetry").abs() if mom is not None and _column(frame, f"minute_nvg__price_nvg_{minutes}_top_bottom_asymmetry") is not None else None,
        "B08": -np.sign(mom) * _column(frame, f"minute_nvg__price_nvg_{minutes}_top_bottom_asymmetry").abs() if mom is not None and _column(frame, f"minute_nvg__price_nvg_{minutes}_top_bottom_asymmetry") is not None else None,
        "B09": edge * (top_visible + bottom_visible) / 2 if edge is not None and top_visible is not None and bottom_visible is not None else None,
        "B10": edge * np.log1p(top_mass + bottom_mass) if edge is not None and top_mass is not None and bottom_mass is not None else None,
        "B11": _asym(top_degree, bottom_degree) if top_degree is not None and bottom_degree is not None else None,
        "B12": _asym(top_visible, bottom_visible) if top_visible is not None and bottom_visible is not None else None,
        "B13": _asym(top_long, bottom_long) if top_long is not None and bottom_long is not None else None,
        "B14": _asym(top_span, bottom_span) if top_span is not None and bottom_span is not None else None,
        "B15": _asym(top_gap, bottom_gap) if top_gap is not None and bottom_gap is not None else None,
        "B16": _asym(top_edge, bottom_edge.abs()) if top_edge is not None and bottom_edge is not None else None,
        "B17": np.sign(edge) / (top_span + bottom_span + EPS) if edge is not None and top_span is not None and bottom_span is not None else None,
        "B18": edge * (top_long + bottom_long) / 2 if edge is not None and top_long is not None and bottom_long is not None else None,
        "B19": edge * (1 - _column(frame, f"minute_nvg__price_nvg_{minutes}_top_terminal_span_entropy").fillna(0)) if edge is not None and _column(frame, f"minute_nvg__price_nvg_{minutes}_top_terminal_span_entropy") is not None else None,
        "B20": np.sign(edge) * (top_gap + bottom_gap) / 2 if edge is not None and top_gap is not None and bottom_gap is not None else None,
        "B21": _csz(edge, groups) - _csz(_column(frame, f"minute_nvg__price_nvg_{minutes}_detrended_top_bottom_asymmetry"), groups) if edge is not None and _column(frame, f"minute_nvg__price_nvg_{minutes}_detrended_top_bottom_asymmetry") is not None else None,
        "B22": _conf(edge, _column(frame, f"minute_nvg__price_nvg_{minutes}_detrended_top_bottom_asymmetry"), groups) if edge is not None and _column(frame, f"minute_nvg__price_nvg_{minutes}_detrended_top_bottom_asymmetry") is not None else None,
        "B23": _column(frame, f"minute_nvg__price_nvg_{minutes}_detrended_top_bottom_asymmetry") * (np.sign(edge) != np.sign(_column(frame, f"minute_nvg__price_nvg_{minutes}_detrended_top_bottom_asymmetry"))) if edge is not None and _column(frame, f"minute_nvg__price_nvg_{minutes}_detrended_top_bottom_asymmetry") is not None else None,
        "B24": _column(frame, f"minute_nvg__price_nvg_{minutes}_detrended_change_1m"),
    }
    if p == "B__ALL__":
        return finish_map(b)
    if p.startswith("B"):
        return finish(b.get(p))
    # C: minute-NVG topology.  These fields are published by the topology
    # supplement; they are not approximated from the raw NVG edge table.
    c_degree = _column(frame, f"price_nvg_{minutes}_full_degree_gini")
    c_hub = _column(frame, f"price_nvg_{minutes}_full_hub_share")
    c_age = _column(frame, f"price_nvg_{minutes}_full_hub_age_norm")
    c_motif = _column(frame, f"price_nvg_{minutes}_full_motif_entropy")
    c_span_entropy = _column(frame, f"price_nvg_{minutes}_full_edge_span_entropy")
    c_replace = _column(frame, f"price_nvg_{minutes}_hub_replacement_strength")
    c_edge = _first(edge, _column(frame, f"price_nvg_{minutes}_terminal_signed_edge_balance"))
    c_slope_std = _column(frame, f"price_nvg_{minutes}_terminal_slope_std")
    c_range = _first(path_range, _column(frame, f"price_path_{minutes}_range"))
    c_asym = _first(
        _column(frame, f"price_nvg_{minutes}_top_bottom_asymmetry"),
        _column(frame, f"price_nvg_{minutes}_terminal_signed_edge_balance"),
    )
    c = {
        "C01": c_degree,
        "C02": c_hub,
        "C03": c_age,
        "C04": c_motif,
        "C05": c_span_entropy,
        "C06": c_replace,
        "C07": -np.sign(c_edge) * c_replace if c_edge is not None and c_replace is not None else None,
        "C08": np.sign(_group_shift(c_asym, 1, _instrument_groups(frame))) * c_replace if c_asym is not None and c_replace is not None else None,
        "C09": np.sign(c_edge) * c_edge.abs() * c_degree * (1 - c_motif) if all(x is not None for x in (c_edge, c_degree, c_motif)) else None,
        "C10": np.sign(c_edge) * c_edge.abs() * c_motif * c_slope_std if all(x is not None for x in (c_edge, c_motif, c_slope_std)) else None,
        "C11": (1 - c_edge.abs()) * (1 - c_motif) * (1 - c_range.abs()) if all(x is not None for x in (c_edge, c_motif, c_range)) else None,
        "C12": pd.concat(
            [_csr(c_replace, groups), _csr(_group_shift(c_degree, 1, _instrument_groups(frame)).sub(c_degree).abs(), groups),
             _csr(_group_shift(c_motif, 1, _instrument_groups(frame)).sub(c_motif).abs(), groups),
             _csr(_group_shift(c_age, 1, _instrument_groups(frame)).sub(c_age).abs(), groups)], axis=1
        ).mean(axis=1) if all(x is not None for x in (c_replace, c_degree, c_motif, c_age)) else None,
    }
    if p == "C__ALL__":
        return finish_map(c)
    if p.startswith("C"):
        return finish(c.get(p))
    d = {
        "D01": volume_edge,
        "D02": volume_long,
        "D03": volume_change * volume_eff if volume_change is not None and volume_eff is not None else None,
        "D04": _csz(volume_change, groups) if volume_change is not None else None,
        "D05": _conf(edge, volume_edge, groups) if edge is not None and volume_edge is not None else None,
        "D06": edge * overlap if edge is not None and overlap is not None else None,
        "D07": edge * confirmation if edge is not None and confirmation is not None else None,
        "D08": edge * _column(frame, f"minute_nvg__price_volume_nvg_{minutes}_edge_weighted_jaccard") if edge is not None and _column(frame, f"minute_nvg__price_volume_nvg_{minutes}_edge_weighted_jaccard") is not None else None,
        "D09": edge * _column(frame, f"minute_nvg__price_volume_nvg_{minutes}_common_edge_slope_corr") if edge is not None and _column(frame, f"minute_nvg__price_volume_nvg_{minutes}_common_edge_slope_corr") is not None else None,
        "D10": edge * _column(frame, f"minute_nvg__price_volume_nvg_{minutes}_full_motif_cosine") if edge is not None and _column(frame, f"minute_nvg__price_volume_nvg_{minutes}_full_motif_cosine") is not None else None,
        "D11": edge * _column(frame, f"minute_nvg__price_volume_nvg_{minutes}_full_hub_time_overlap") if edge is not None and _column(frame, f"minute_nvg__price_volume_nvg_{minutes}_full_hub_time_overlap") is not None else None,
        "D12": edge * (1 - overlap) if edge is not None and overlap is not None else None,
        "D13": volume_edge * (np.abs(volume_edge) > np.abs(edge)) if volume_edge is not None and edge is not None else None,
        "D14": _column(frame, f"minute_nvg__price_volume_terminal_overlap_change_1m"),
    }
    if p == "D__ALL__":
        return finish_map(d)
    if p.startswith("D"):
        return finish(d.get(p))
    e = {
        "E01": _first(_column(frame, f"trade_nvg__trade_price_nvg_{sec}_terminal_signed_edge_balance"), tprice_edge),
        "E02": _first(_column(frame, f"trade_nvg__trade_flow_nvg_{sec}_terminal_signed_edge_balance"), tflow_edge),
        "E03": _column(frame, f"trade_nvg__trade_price_nvg_{sec}_terminal_long_edge_signed_slope"),
        "E04": _column(frame, f"trade_nvg__trade_flow_nvg_{sec}_terminal_long_edge_signed_slope"),
        "E05": tprice * tprice_eff if tprice is not None and tprice_eff is not None else None,
        "E06": tflow * tflow_eff if tflow is not None and tflow_eff is not None else None,
        "E07": tprice * tprice_eff / (tprice_rough.abs() + EPS) if tprice is not None and tprice_eff is not None and tprice_rough is not None else None,
        "E08": tflow * tflow_eff / (tflow_rough.abs() + EPS) if tflow is not None and tflow_eff is not None and tflow_rough is not None else None,
        "E09": _conf(tprice_edge, tflow_edge, groups) if tprice_edge is not None and tflow_edge is not None else None,
        "E10": tprice_edge * toverlap if tprice_edge is not None and toverlap is not None else None,
        "E11": tprice_edge * _column(frame, f"trade_nvg__trade_price_flow_nvg_{sec}_edge_weighted_jaccard") if tprice_edge is not None and _column(frame, f"trade_nvg__trade_price_flow_nvg_{sec}_edge_weighted_jaccard") is not None else None,
        "E12": tprice_edge * _column(frame, f"trade_nvg__trade_price_flow_nvg_{sec}_common_edge_slope_corr") if tprice_edge is not None and _column(frame, f"trade_nvg__trade_price_flow_nvg_{sec}_common_edge_slope_corr") is not None else None,
        "E13": tprice_edge * _column(frame, f"trade_nvg__trade_price_flow_nvg_{sec}_full_motif_cosine") if tprice_edge is not None and _column(frame, f"trade_nvg__trade_price_flow_nvg_{sec}_full_motif_cosine") is not None else None,
        "E14": tprice_edge * _column(frame, f"trade_nvg__trade_price_flow_nvg_{sec}_full_hub_time_overlap") if tprice_edge is not None and _column(frame, f"trade_nvg__trade_price_flow_nvg_{sec}_full_hub_time_overlap") is not None else None,
        "E15": -np.sign(tflow_edge) * tflow_edge.abs() * (np.sign(tprice_edge) != np.sign(tflow_edge)) if tprice_edge is not None and tflow_edge is not None else None,
        "E16": tprice_edge * (1 - toverlap) if tprice_edge is not None and toverlap is not None else None,
        "E17": tprice_edge if tprice_edge is not None else tflow_edge,
        "E18": _column(frame, f"trade_nvg__trade_price_nvg_{sec}_top_bottom_asymmetry_change_1m"),
        "E19": -np.sign(tprice_edge) * _column(frame, f"trade_nvg__trade_price_nvg_{sec}_hub_replacement_strength") if tprice_edge is not None and _column(frame, f"trade_nvg__trade_price_nvg_{sec}_hub_replacement_strength") is not None else None,
        "E20": _csz(_column(frame, "trade_nvg__trade_price_nvg_60s_top_bottom_asymmetry"), groups) - _csz(_column(frame, "trade_nvg__trade_price_nvg_300s_top_bottom_asymmetry"), groups) if _column(frame, "trade_nvg__trade_price_nvg_60s_top_bottom_asymmetry") is not None and _column(frame, "trade_nvg__trade_price_nvg_300s_top_bottom_asymmetry") is not None else None,
        "E21": pd.concat([tprice_edge, _column(frame, f"trade_nvg__trade_price_nvg_180s_top_bottom_asymmetry"), _column(frame, f"trade_nvg__trade_price_nvg_300s_top_bottom_asymmetry")], axis=1).apply(np.sign).mean(axis=1) if tprice_edge is not None and _column(frame, f"trade_nvg__trade_price_nvg_180s_top_bottom_asymmetry") is not None and _column(frame, f"trade_nvg__trade_price_nvg_300s_top_bottom_asymmetry") is not None else None,
        "E22": tprice_edge * (np.sign(tprice_edge) != np.sign(edge)) if tprice_edge is not None and edge is not None else None,
    }
    if p == "E__ALL__":
        return finish_map(e)
    if p.startswith("E"):
        return finish(e.get(p))
    f = {
        "F01": activity,
        "F02": _csz(activity, groups) if activity is not None else None,
        "F03": -_group_shift(stale, 0, _instrument_groups(frame)) * np.maximum(_group_shift(activity, 0, _instrument_groups(frame)), 0) if stale is not None and activity is not None else None,
        "F04": coverage,
        "F05": _column(frame, f"trade_nvg__trade_activity_nvg_{sec}_terminal_slope_mean"),
        "F06": _column(frame, f"trade_nvg__trade_activity_nvg_{sec}_terminal_long_edge_ratio"),
        "F07": _column(frame, f"trade_nvg__trade_activity_nvg_{sec}_terminal_span_entropy"),
        "F08": _column(frame, f"trade_nvg__trade_activity_nvg_{sec}_hub_replacement_strength"),
    }
    if p == "F__ALL__":
        return finish_map(f)
    if p.startswith("F"):
        return finish(f.get(p))
    # G: minute-HVG risk and irreversibility supplement.
    g_return_degree = _column(frame, f"return_hvg_{minutes}_terminal_degree")
    g_return_long = _column(frame, f"return_hvg_{minutes}_terminal_long_edge_ratio")
    g_return_irrev = _column(frame, f"return_hvg_{minutes}_degree_irreversibility_js")
    g_return_motif_irrev = _column(frame, f"return_hvg_{minutes}_motif_irreversibility_js")
    g_abs_degree = _column(frame, f"abs_return_hvg_{minutes}_terminal_degree")
    g_abs_irrev = _column(frame, f"abs_return_hvg_{minutes}_degree_irreversibility_js")
    g_volume_irrev = _column(frame, f"volume_hvg_{minutes}_degree_irreversibility_js")
    g_trade_price_irrev = _column(frame, f"trade_price_hvg_{sec}_degree_irreversibility_js")
    g_trade_flow_irrev = _column(frame, f"trade_flow_hvg_{sec}_degree_irreversibility_js")
    g_trade_activity_irrev = _column(frame, f"trade_activity_hvg_{sec}_degree_irreversibility_js")
    g_trade_motif = _column(frame, f"trade_price_flow_hvg_{sec}_full_motif_cosine")
    g_trade_hub = _column(frame, f"trade_price_flow_hvg_{sec}_full_hub_time_overlap")
    g = {
        "G01": g_return_degree,
        "G02": g_return_long,
        "G03": g_return_irrev,
        "G04": g_return_motif_irrev,
        "G05": g_abs_degree,
        "G06": g_abs_irrev,
        "G07": g_volume_irrev,
        "G08": g_return_irrev - _group_shift(g_return_irrev, 5, _instrument_groups(frame)) if g_return_irrev is not None else None,
        "G09": -np.sign(mom) * np.maximum(-(_group_shift(g_return_irrev, 5, _instrument_groups(frame)) - g_return_irrev), 0) if mom is not None and g_return_irrev is not None else None,
        "G10": g_trade_price_irrev,
        "G11": g_trade_flow_irrev,
        "G12": g_trade_activity_irrev,
        "G13": g_trade_motif,
        "G14": g_trade_hub,
    }
    if p == "G__ALL__":
        return finish_map(g)
    if p.startswith("G"):
        return finish(g.get(p))
    h = {
        "H01": hawkes_imbalance,
        "H02": hawkes_pressure,
        "H03": _hawkes(frame, "hawkes_derived__hawkes_pressure_strength"),
        "H04": _hawkes(frame, "hawkes_derived__hawkes_excess_intensity"),
        "H05": _hawkes(frame, "hawkes_lite__hawkes_flow_surprise"),
        "H06": _hawkes(frame, "hawkes_lite__hawkes_buy_surprise") - _hawkes(frame, "hawkes_lite__hawkes_sell_surprise") if _hawkes(frame, "hawkes_lite__hawkes_buy_surprise") is not None and _hawkes(frame, "hawkes_lite__hawkes_sell_surprise") is not None else None,
        "H07": _hawkes(frame, "hawkes_lite__hawkes_surprise_energy"),
        "H08": hawkes_pressure * hawkes_persistence if hawkes_pressure is not None and hawkes_persistence is not None else None,
        "H09": hawkes_pressure * hawkes_endogenous if hawkes_pressure is not None and hawkes_endogenous is not None else None,
        "H10": hawkes_pressure * hawkes_branching if hawkes_pressure is not None and hawkes_branching is not None else None,
        "H11": np.sign(hawkes_pressure) * _hawkes(frame, "hawkes_derived__hawkes_cross_reaction") if hawkes_pressure is not None and _hawkes(frame, "hawkes_derived__hawkes_cross_reaction") is not None else None,
        "H12": np.sign(hawkes_pressure) * hawkes_exog if hawkes_pressure is not None and hawkes_exog is not None else None,
        "H13": np.sign(hawkes_pressure) * hawkes_endog if hawkes_pressure is not None and hawkes_endog is not None else None,
        "H14": _csz(hawkes_exog, groups) - _csz(hawkes_endog, groups) if hawkes_exog is not None and hawkes_endog is not None else None,
        "H15": _csz(_hawkes(frame, "hawkes_lite__hawkes_shock_score", "60s"), groups) - _csz(_hawkes(frame, "hawkes_lite__hawkes_shock_score", "300s"), groups) if _hawkes(frame, "hawkes_lite__hawkes_shock_score", "60s") is not None and _hawkes(frame, "hawkes_lite__hawkes_shock_score", "300s") is not None else None,
        "H16": _hawkes(frame, "hawkes_lite__hawkes_total_intensity_change", sec),
        "H17": _hawkes(frame, "hawkes_lite__hawkes_endogenous_share_change", sec),
        "H18": _hawkes(frame, "hawkes_lite__hawkes_imbalance_std", sec),
        "H19": _hawkes(frame, "hawkes_lite__hawkes_flow_surprise_sum", sec) / (_hawkes(frame, "hawkes_lite__hawkes_flow_surprise_maxabs", sec).abs() + EPS) if _hawkes(frame, "hawkes_lite__hawkes_flow_surprise_sum", sec) is not None and _hawkes(frame, "hawkes_lite__hawkes_flow_surprise_maxabs", sec) is not None else None,
        "H20": _hawkes(frame, "hawkes_lite__hawkes_total_surprise_maxabs", sec) / (_hawkes(frame, "hawkes_lite__hawkes_total_surprise_sum", sec).abs() + _hawkes(frame, "hawkes_lite__hawkes_total_surprise_std", sec).abs() + EPS) if _hawkes(frame, "hawkes_lite__hawkes_total_surprise_maxabs", sec) is not None and _hawkes(frame, "hawkes_lite__hawkes_total_surprise_sum", sec) is not None and _hawkes(frame, "hawkes_lite__hawkes_total_surprise_std", sec) is not None else None,
        "H21": _hawkes(frame, "hawkes_lite__hawkes_short_excitation_share") - _hawkes(frame, "hawkes_lite__hawkes_long_excitation_share") if _hawkes(frame, "hawkes_lite__hawkes_short_excitation_share") is not None and _hawkes(frame, "hawkes_lite__hawkes_long_excitation_share") is not None else None,
        "H22": hawkes_pressure * (_hawkes(frame, "hawkes_lite__hawkes_short_excitation_share") - _hawkes(frame, "hawkes_lite__hawkes_long_excitation_share")) if hawkes_pressure is not None and _hawkes(frame, "hawkes_lite__hawkes_short_excitation_share") is not None and _hawkes(frame, "hawkes_lite__hawkes_long_excitation_share") is not None else None,
        "H23": hawkes_persistence,
        "H24": pd.concat([_hawkes(frame, "hawkes_derived__hawkes_intensity_regime_change"), _hawkes(frame, "hawkes_derived__hawkes_endogeneity_regime_change"), _hawkes(frame, "hawkes_derived__hawkes_shock_regime_change")], axis=1).max(axis=1) if all(_hawkes(frame, x) is not None for x in ("hawkes_derived__hawkes_intensity_regime_change", "hawkes_derived__hawkes_endogeneity_regime_change", "hawkes_derived__hawkes_shock_regime_change")) else None,
    }
    if p == "H__ALL__":
        return finish_map(h)
    if p.startswith("H"):
        return finish(h.get(p))
    i = {
        "I01": (ofi - sell) / (ofi + sell + EPS) if ofi is not None and sell is not None else None,
        "I02": _csz(signed_flow, groups) if signed_flow is not None else None,
        "I03": _column(frame, "trades_1m_core__flow_persistence_15m"),
        "I04": _column(frame, "trades_1m_core__flow_persistence_15m") * _csz(signed_flow, groups) if _column(frame, "trades_1m_core__flow_persistence_15m") is not None and signed_flow is not None else None,
        "I05": (_column(frame, "trades_1m_core__large_trade_buy_volume_proxy") - _column(frame, "trades_1m_core__large_trade_sell_volume_proxy")) / (_column(frame, "trades_1m_core__large_trade_buy_volume_proxy") + _column(frame, "trades_1m_core__large_trade_sell_volume_proxy") + EPS) if _column(frame, "trades_1m_core__large_trade_buy_volume_proxy") is not None and _column(frame, "trades_1m_core__large_trade_sell_volume_proxy") is not None else None,
        "I06": None,
        "I07": _column(frame, "trades_1m_core__block_trade_volume") / (volume + EPS) if _column(frame, "trades_1m_core__block_trade_volume") is not None and volume is not None else None,
        "I08": _column(frame, "trades_1m_core__odd_lot_volume") / (volume + EPS) if _column(frame, "trades_1m_core__odd_lot_volume") is not None and volume is not None else None,
        "I09": _column(frame, "trades_1m_core__conditioned_volume") / (volume + EPS) if _column(frame, "trades_1m_core__conditioned_volume") is not None and volume is not None else None,
        "I10": _column(frame, "trades_1m_core__trade_size_p95") / (_column(frame, "trades_1m_core__median_trade_size") + EPS) if _column(frame, "trades_1m_core__trade_size_p95") is not None and _column(frame, "trades_1m_core__median_trade_size") is not None else None,
        "I11": _column(frame, "trades_1m_core__trade_size_hhi"),
        "I12": _column(frame, "trades_1m_core__top_1pct_volume_share"),
        "I13": _column(frame, "trades_1m_core__burstiness") * np.sign(signed_flow) if _column(frame, "trades_1m_core__burstiness") is not None and signed_flow is not None else None,
        "I14": _column(frame, "trades_1m_core__burstiness") * _column(frame, "trades_1m_core__large_trade_dollar_share") * (i["I05"] if i["I05"] is not None else 0) if _column(frame, "trades_1m_core__burstiness") is not None and _column(frame, "trades_1m_core__large_trade_dollar_share") is not None else None,
        "I15": _column(frame, "trades_1m_core__sign_run_mean") * np.sign((ofi - sell) if ofi is not None and sell is not None else 0) if _column(frame, "trades_1m_core__sign_run_mean") is not None and ofi is not None and sell is not None else None,
        "I16": _column(frame, "trades_1m_core__sign_run_max") * np.sign((ofi - sell) if ofi is not None and sell is not None else 0) if _column(frame, "trades_1m_core__sign_run_max") is not None and ofi is not None and sell is not None else None,
        "I17": np.sign(ofi - sell) * (1 - _csz(_column(frame, "trades_1m_core__flow_sign_changes"), groups)) if ofi is not None and sell is not None and _column(frame, "trades_1m_core__flow_sign_changes") is not None else None,
        "I18": _column(frame, "trades_1m_core__trade_price_std"),
        "I19": _column(frame, "trades_1m_core__max_within_minute_silence_ns"),
        "I20": _column(frame, "trades_1m_core__within_minute_gap_p90_ns"),
        "I21": _column(frame, "trades_1m_core__subsecond_trade_share"),
        "I22": ret1.abs() / (dollar + EPS) if ret1 is not None and dollar is not None else None,
        "I23": np.sign(ret1) * _csz(ret1.abs() / (dollar + EPS), groups) if ret1 is not None and dollar is not None else None,
        "I24": ret1 / (_csz(signed_flow, groups).abs() + EPS) if ret1 is not None and signed_flow is not None else None,
        "I25": -np.sign(ofi - sell) * (ofi - sell).abs() * (np.sign(ret1) != np.sign(ofi - sell)) if ret1 is not None and ofi is not None and sell is not None else None,
        "I26": _column(frame, "trades_1m_core__volume_at_price_hhi"),
        "I27": _column(frame, "trades_1m_core__report_lag_p90_ns"),
        "I28": _column(frame, "trades_1m_core__corrected_replacement_count") / (_column(frame, "trades_1m_core__trade_count") + EPS) if _column(frame, "trades_1m_core__corrected_replacement_count") is not None and _column(frame, "trades_1m_core__trade_count") is not None else None,
    }
    if p == "I__ALL__":
        return finish_map(i)
    if p.startswith("I"):
        return finish(i.get(p))
    j = {
        "J01": _column(frame, "off_exchange_share") if _column(frame, "off_exchange_share") is not None else (_column(frame, "trades_1m_core__off_exchange_volume") / (volume + EPS) if _column(frame, "trades_1m_core__off_exchange_volume") is not None and volume is not None else None),
        "J02": _column(frame, "dark_signed_flow"),
        "J03": _column(frame, "lit_signed_flow") if _column(frame, "lit_signed_flow") is not None else signed_flow,
        "J04": _csz(_column(frame, "dark_signed_flow"), groups) - _csz(_column(frame, "lit_signed_flow"), groups) if _column(frame, "dark_signed_flow") is not None and _column(frame, "lit_signed_flow") is not None else None,
        "J05": _conf(_column(frame, "dark_signed_flow"), _column(frame, "lit_signed_flow"), groups) if _column(frame, "dark_signed_flow") is not None and _column(frame, "lit_signed_flow") is not None else None,
        "J06": _first(_column(frame, "venue_hhi"), _column(frame, "trades_1m_sketch__venue_volume_hhi")),
        "J07": _first(_column(frame, "venue_entropy"), _column(frame, "trades_1m_sketch__venue_entropy")),
        "J08": _column(frame, "trades_1m_sketch__dominant_venue_share"),
        "J09": _csz(_first(_column(frame, "venue_hhi"), _column(frame, "trades_1m_sketch__venue_volume_hhi")), groups) if _first(_column(frame, "venue_hhi"), _column(frame, "trades_1m_sketch__venue_volume_hhi")) is not None else None,
        "J10": _conf(i["I05"], _column(frame, "dark_signed_flow"), groups) if i["I05"] is not None and _column(frame, "dark_signed_flow") is not None else None,
    }
    if p == "J__ALL__":
        return finish_map(j)
    if p.startswith("J"):
        return finish(j.get(p))
    # K uses the fixed contract anchors (10m minute price, 60s flow, 300s
    # trade-price) irrespective of the registry's one-minute display window.
    k_price = _first(
        _column(frame, "minute_nvg__price_nvg_10m_terminal_signed_edge_balance"),
        _column(frame, "minute_nvg__price_nvg_10m_top_bottom_asymmetry"),
    )
    k_flow = _first(
        _column(frame, "trade_nvg__trade_flow_nvg_60s_terminal_signed_edge_balance"),
        _column(frame, "trade_nvg__trade_flow_nvg_60s_top_bottom_asymmetry"),
    )
    k_trade_price = _first(
        _column(frame, "trade_nvg__trade_price_nvg_300s_terminal_signed_edge_balance"),
        _column(frame, "trade_nvg__trade_price_nvg_300s_top_bottom_asymmetry"),
    )
    k_hawkes_pressure = hawkes_pressure
    k_hawkes_strength = _hawkes(frame, "hawkes_derived__hawkes_pressure_strength")
    k_surprise_energy = _hawkes(frame, "hawkes_lite__hawkes_surprise_energy")
    k_burstiness = _column(frame, "trades_1m_sketch__burstiness")
    k_activity = _column(frame, "trade_nvg__trade_active_second_ratio_60s")
    k_activity_breakout = _csz(k_activity, groups) if k_activity is not None else None
    k_large_imbalance = i.get("I05")
    k_amihud = i.get("I22")
    k_stale = _column(frame, "trade_nvg__trade_price_stale_ratio_60s")
    k_hawkes_regime = _first(
        _hawkes(frame, "hawkes_derived__hawkes_shock_regime_change"),
        _hawkes(frame, "hawkes_derived__hawkes_intensity_regime_change"),
    )
    k_c_degree = _column(frame, "price_nvg_10m_full_degree_gini")
    k_c_motif = _column(frame, "price_nvg_10m_full_motif_entropy")
    k_c_replace = _column(frame, "price_nvg_10m_hub_replacement_strength")
    k_c_slope_std = _column(frame, "price_nvg_10m_terminal_slope_std")
    k_c_range = _first(path_range, _column(frame, "price_path_10m_range"))
    k = {
        "K01": _conf(k_price, k_trade_price, groups) if k_price is not None and k_trade_price is not None else None,
        "K02": _conf(k_price, k_flow, groups) if k_price is not None and k_flow is not None else None,
        "K03": _conf(k_price, k_hawkes_pressure, groups) if k_price is not None and k_hawkes_pressure is not None else None,
        "K04": _conf(k_flow, k_hawkes_pressure, groups) if k_flow is not None and k_hawkes_pressure is not None else None,
        "K05": pd.concat([_csr(k_price, groups), _csr(k_flow, groups), _csr(k_hawkes_pressure, groups)], axis=1).mean(axis=1) * ((np.sign(k_price) == np.sign(k_flow)) & (np.sign(k_price) == np.sign(k_hawkes_pressure))).astype(float) if k_price is not None and k_flow is not None and k_hawkes_pressure is not None else None,
        "K06": -np.sign(k_price) * _csz(k_price, groups).abs() * ((np.sign(k_flow) != np.sign(k_price)) & (np.sign(k_hawkes_pressure) != np.sign(k_price))).astype(float) if k_price is not None and k_flow is not None and k_hawkes_pressure is not None else None,
        "K07": pd.concat([_csr(k_flow, groups), _csr(k_hawkes_pressure, groups)], axis=1).mean(axis=1) * (k_price.abs() < k_price.abs().groupby(groups, sort=False, group_keys=False).transform("median")).astype(float) if k_price is not None and k_flow is not None and k_hawkes_pressure is not None else None,
        "K08": k_price * confirmation * (0.5 + 0.5 * _same(k_price, k_flow)) if k_price is not None and confirmation is not None and k_flow is not None else None,
        "K09": np.sign(k_price) * k_hawkes_strength * _same(k_price, k_hawkes_pressure) if k_price is not None and k_hawkes_pressure is not None and k_hawkes_strength is not None else None,
        "K10": -np.sign(k_hawkes_pressure) * k_hawkes_strength * ((np.sign(k_price) != np.sign(k_hawkes_pressure)) | (k_price.abs() < k_price.abs().groupby(groups, sort=False, group_keys=False).transform("median"))).astype(float) if k_price is not None and k_hawkes_pressure is not None and k_hawkes_strength is not None else None,
        "K11": _csr(k_price, groups) * pd.concat([path_eff, confirmation], axis=1).mean(axis=1) if k_price is not None and path_eff is not None and confirmation is not None else None,
        "K12": pd.concat([b.get("B08"), e.get("E15"), h.get("H09")], axis=1).mean(axis=1) if b.get("B08") is not None and e.get("E15") is not None and h.get("H09") is not None else None,
        "K13": pd.concat([_csr(k_surprise_energy, groups), _csr(k_burstiness, groups), _csr(k_activity_breakout, groups), _csr(k_large_imbalance.abs(), groups)], axis=1).mean(axis=1) if all(x is not None for x in (k_surprise_energy, k_burstiness, k_activity_breakout, k_large_imbalance)) else None,
        "K14": pd.concat([_csr(k_amihud, groups), _csr(_column(frame, "trades_1m_sketch__max_within_minute_silence_ns"), groups), _csr(k_stale, groups), _csr(1 - k_activity, groups), _csr(_column(frame, "trades_1m_sketch__trade_size_hhi"), groups)], axis=1).mean(axis=1) if all(x is not None for x in (k_amihud, k_activity, k_stale, _column(frame, "trades_1m_sketch__max_within_minute_silence_ns"), _column(frame, "trades_1m_sketch__trade_size_hhi"))) else None,
        "K15": pd.concat([_csr(k_c_replace, groups), _csr(_column(frame, "trade_nvg__trade_price_nvg_60s_hub_replacement_strength"), groups), _csr(_column(frame, "trade_nvg__trade_flow_nvg_60s_hub_replacement_strength"), groups), _csr(k_hawkes_regime, groups)], axis=1).mean(axis=1) if all(x is not None for x in (k_c_replace, k_hawkes_regime, _column(frame, "trade_nvg__trade_price_nvg_60s_hub_replacement_strength"), _column(frame, "trade_nvg__trade_flow_nvg_60s_hub_replacement_strength"))) else None,
        "K16": _csr(k_price, groups).abs() * _column(frame, "minute_nvg__price_path_10m_efficiency") * (1 - _csr(k_c_motif, groups).abs()) if k_price is not None and _column(frame, "minute_nvg__price_path_10m_efficiency") is not None and k_c_motif is not None else None,
        "K17": pd.concat([_csr(k_c_slope_std, groups), _csr(k_c_motif, groups), _csr(_hawkes(frame, "hawkes_lite__hawkes_imbalance_std", "60s"), groups), _csr(_column(frame, "traditional__realized_vol_5m"), groups)], axis=1).mean(axis=1) if all(x is not None for x in (k_c_slope_std, k_c_motif, _hawkes(frame, "hawkes_lite__hawkes_imbalance_std", "60s"), _column(frame, "traditional__realized_vol_5m"))) else None,
        "K18": pd.concat([_csr(-_column(frame, "traditional__realized_vol_5m"), groups), _csr(-k_c_range, groups), _csr(-k_price.abs(), groups), _csr(-_hawkes(frame, "hawkes_derived__hawkes_excess_intensity"), groups)], axis=1).mean(axis=1) if all(x is not None for x in (k_c_range, k_price, _column(frame, "traditional__realized_vol_5m"), _hawkes(frame, "hawkes_derived__hawkes_excess_intensity"))) else None,
        "K19": 1 / (1 + np.exp(-(_csz(hawkes_persistence, groups) + _csz(hawkes_endogenous, groups) - _csz(_hawkes(frame, "hawkes_lite__hawkes_imbalance_std", "60s"), groups)))) if all(x is not None for x in (hawkes_persistence, hawkes_endogenous, _hawkes(frame, "hawkes_lite__hawkes_imbalance_std", "60s"))) else None,
        "K20": 1 / (1 + np.exp(-(_csz(_hawkes(frame, "hawkes_derived__hawkes_exogenous_shock", "60s"), groups) + _csz(k_hawkes_regime, groups) - _csz(hawkes_persistence, groups)))) if all(x is not None for x in (k_hawkes_regime, hawkes_persistence, _hawkes(frame, "hawkes_derived__hawkes_exogenous_shock", "60s"))) else None,
        "K21": _column(frame, "traditional__momentum_15m") / (_column(frame, "traditional__realized_vol_30m").abs() + EPS) if _column(frame, "traditional__momentum_15m") is not None and _column(frame, "traditional__realized_vol_30m") is not None else None,
        "K22": pd.concat([_csr(hawkes_persistence, groups), _csr(_column(frame, "return_hvg_15m_terminal_long_edge_ratio"), groups), _csr(_column(frame, "trade_nvg__trade_price_nvg_60s_full_edge_span_entropy"), groups), _csr(_column(frame, "trades_1m_core__flow_persistence_15m"), groups)], axis=1).mean(axis=1) if all(x is not None for x in (hawkes_persistence, _column(frame, "return_hvg_15m_terminal_long_edge_ratio"), _column(frame, "trade_nvg__trade_price_nvg_60s_full_edge_span_entropy"), _column(frame, "trades_1m_core__flow_persistence_15m"))) else None,
    }
    if p == "K__ALL__":
        return finish_map(k)
    values = {**a, **b, **d, **e, **f, **h, **i, **j, **k}
    return finish(values.get(p))


def derive_all(frame: pd.DataFrame, prototypes: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    result = frame.copy()
    # Prototype formulas depend only on the stable source/dependency block.
    # Passing the growing result frame would widen every subsequent lookup and
    # turn a source-column scan into an avoidable O(specs * generated_columns)
    # cost.  Keep writes in result, but read every formula from source_frame.
    source_frame = frame
    registry: dict[str, dict[str, Any]] = {}
    generated_columns: dict[str, pd.Series] = {}
    by_family: dict[str, list[dict[str, Any]]] = {}
    for prototype in prototypes:
        by_family.setdefault(prototype["family"], []).append(prototype)
    for family, family_prototypes in by_family.items():
        for window in FAMILY_WINDOWS.get(family, ("1m",)):
            values = derive_prototype(source_frame, f"{family}__ALL__", window)
            value_map = values if isinstance(values, dict) else {}
            for prototype in family_prototypes:
                name = factor_name(prototype["prototype_id"], window)
                value = value_map.get(prototype["prototype_id"])
                if value is None:
                    registry[name] = {"status": "DATA_UNAVAILABLE", "non_null_rate": 0.0}
                    continue
                # Accumulate generated columns and attach them in one block. Repeated
                # frame insertion creates hundreds of tiny pandas blocks and turns
                # later rank/groupby work into a costly consolidation pass.
                generated_columns[name] = value
                rate = float(value.notna().mean())
                registry[name] = {"status": "SUCCESS" if rate > 0 else "LOW_COVERAGE", "non_null_rate": rate}
    if generated_columns:
        generated = pd.concat(generated_columns, axis=1, copy=False)
        generated.columns = list(generated_columns)
        result = pd.concat([result, generated], axis=1, copy=False)
    # The cache is deliberately date-local; keeping source Series across dates
    # would trade CPU for an unbounded worker RSS increase.
    _DERIVE_CACHE.clear()
    return result, registry


def resolve_registry(frame_columns: Iterable[str], path: Path = INSTRUCTION_PATH) -> pd.DataFrame:
    prototypes = parse_prototypes(path)
    specs = expand_specs(prototypes)
    available = set(frame_columns)
    # The actual source resolver is conservative: a generated factor column is
    # the authoritative evidence that its dependencies were present.
    specs["status"] = specs.apply(
        lambda row: "SUCCESS" if factor_name(row.prototype_id, row.window) in available else "DATA_UNAVAILABLE",
        axis=1,
    )
    specs["instruction_sha256"] = instruction_hash(path)
    return specs
