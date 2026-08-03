from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psutil

from qlib.contrib.data.nff import NFFDataLoader, NFFWarehouseCatalog


WAREHOUSE_ROOT = Path(r"D:\DEV\AnotherNetworkFactory\warehouses\NFF_warehouse")
RESEARCH_ROOT = Path(r"D:\DEV\AnotherNetworkFactory\warehouses\NFF_research")
RAW_1M_ROOT = Path(r"D:\DEV\AnotherNetworkFactory\RAW_DATA\1m")
NFF_SRC_ROOT = Path(r"D:\DEV\AnotherNetworkFactory\NodeFactorFactory\src")
CONTROLS_ROOT = RESEARCH_ROOT / "derived_inputs" / "daily_bar_controls_v2_1"
SALT = "vvtr123!@#qwe"
SESSION_TZ = ZoneInfo("America/New_York")
SESSION_OPEN = (9, 30)
SESSION_CLOSE = (16, 0)

KEY_COLUMNS = {"trade_date", "symbol_id", "symbol", "timestamp", "available_time", "schema", "date"}
BAR_COLUMNS = ["open", "close", "volume", "dollar_volume", "vwap"]
TRADES_CORE_COLUMNS = [
    "trade_count",
    "volume",
    "dollar_volume",
    "vwap",
    "avg_trade_size",
    "median_trade_size",
    "max_trade_size",
    "large_trade_count",
    "large_trade_volume",
    "large_trade_dollar_volume",
    "lit_trade_count",
    "lit_volume",
    "off_exchange_volume",
    "buy_volume_proxy",
    "sell_volume_proxy",
    "signed_dollar_flow_proxy",
    "report_lag_p90_ns",
]

HORIZONS = (15, 30, 60, 120)
LABEL_FAMILIES = (
    "return_open_to_open",
    "return_vwap_to_vwap",
    "return_close_to_close",
    "liquidity_deterioration",
    "realized_volatility",
    "jump_tail_event",
    "execution_cost_proxy",
)
CORE_DECILE_FEATURES = [
    "minute_nvg__price_nvg_30m_top_bottom_asymmetry",
    "minute_nvg__momentum_30m",
    "minute_nvg__price_path_30m_terminal_position",
    "minute_nvg__price_path_30m_signed_change",
    "traditional__reversal_15m",
    "traditional__reversal_30m",
    "traditional__momentum_15m",
    "traditional__realized_vol_30m",
    "traditional__log_dollar_volume",
    "trade_nvg__trade_price_path_60s_signed_change",
    "trade_nvg__trade_flow_path_300s_terminal_position",
    "trade_nvg__trade_flow_path_300s_signed_change",
    "hawkes_derived__hawkes_signed_pressure",
    "hawkes_derived__hawkes_endogenous_shock_60s",
    "minute_nvg__price_path_30m_efficiency",
    "minute_nvg__price_volume_terminal_overlap_30m",
    "hawkes_derived__hawkes_exogenous_shock_300s",
    "hawkes_derived__hawkes_effective_duration_norm",
]

REPRESENTATIVE_ALPHA_FEATURES = [
    "traditional__momentum_15m",
    "traditional__momentum_30m",
    "traditional__momentum_60m",
    "traditional__momentum_120m",
    "traditional__reversal_15m",
    "traditional__reversal_30m",
    "traditional__reversal_60m",
    "traditional__reversal_120m",
    "traditional__realized_vol_15m",
    "traditional__realized_vol_30m",
    "traditional__realized_vol_60m",
    "traditional__realized_vol_120m",
    "traditional__vwap_dislocation",
    "traditional__log_dollar_volume",
    "traditional__dollar_volume_change_15m",
    "traditional__dollar_volume_change_30m",
    "minute_nvg__price_nvg_15m_top_bottom_asymmetry",
    "minute_nvg__price_nvg_30m_top_bottom_asymmetry",
    "minute_nvg__price_path_15m_efficiency",
    "minute_nvg__price_path_30m_efficiency",
    "minute_nvg__price_path_15m_terminal_position",
    "minute_nvg__price_path_30m_terminal_position",
    "minute_nvg__price_path_15m_signed_change",
    "minute_nvg__price_path_30m_signed_change",
    "minute_nvg__price_path_15m_range",
    "minute_nvg__price_path_30m_range",
    "minute_nvg__price_volume_terminal_overlap_15m",
    "minute_nvg__price_volume_terminal_overlap_30m",
    "trade_nvg__trade_price_path_60s_efficiency",
    "trade_nvg__trade_price_path_60s_terminal_position",
    "trade_nvg__trade_price_path_60s_signed_change",
    "trade_nvg__trade_price_path_300s_range",
    "trade_nvg__trade_flow_path_60s_efficiency",
    "trade_nvg__trade_flow_path_300s_terminal_position",
    "trade_nvg__trade_flow_path_300s_signed_change",
    "trade_nvg__trade_flow_path_300s_range",
    "trade_nvg__trade_price_flow_terminal_overlap_300s",
    "hawkes_lite__hawkes_total_intensity",
    "hawkes_lite__hawkes_intensity_imbalance",
    "hawkes_lite__hawkes_endogenous_share",
    "hawkes_lite__hawkes_cross_excitation_share",
    "hawkes_lite__hawkes_branching_ratio_max",
    "hawkes_lite__hawkes_flow_surprise",
    "hawkes_lite__hawkes_shock_score_60s",
    "hawkes_lite__hawkes_shock_score_300s",
    "hawkes_derived__hawkes_signed_pressure",
    "hawkes_derived__hawkes_persistence",
    "hawkes_derived__hawkes_cross_reaction",
    "hawkes_derived__hawkes_effective_duration_norm",
    "hawkes_derived__hawkes_exogenous_shock_60s",
    "hawkes_derived__hawkes_exogenous_shock_300s",
    "hawkes_derived__hawkes_endogenous_shock_60s",
    "hawkes_derived__hawkes_endogenous_shock_300s",
    "hawkes_derived__hawkes_intensity_regime_change",
    "hawkes_derived__hawkes_endogeneity_regime_change",
]

QUALITY_OR_CONTROL_RE = re.compile(
    r"(hawkes_ready|warmup_fraction|observation_coverage|active_second_(ratio|count)|price_stale_ratio|"
    r"(^|__)trade_count(_|$)|bars_1m__(open|close|vwap|volume|dollar_volume)|"
    r"trades_1m_core__(trade_count|volume|dollar_volume|vwap|lit_trade_count|lit_volume|off_exchange_volume))"
)

DEPRECATED_SIGNAL_RE = re.compile(r"^signal__")

BUNDLE_ORDER = ["TRAD", "MINUTE_NVG", "TRADE_NVG", "HAWKES_LITE", "HAWKES_DERIVED"]


def _load_v2_module() -> Any:
    local_path = Path(__file__).resolve().with_name("v2_multilabel_runner.py")
    path = local_path if local_path.exists() else RESEARCH_ROOT / "scripts" / "v2_multilabel_runner.py"
    spec = importlib.util.spec_from_file_location("v2_multilabel_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2 = _load_v2_module()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
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


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config requires PyYAML; install pyyaml or pass explicit CLI arguments") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping: {config_path}")
    return data


def apply_config_globals(config: dict[str, Any]) -> None:
    global WAREHOUSE_ROOT, RESEARCH_ROOT, RAW_1M_ROOT, NFF_SRC_ROOT, CONTROLS_ROOT
    paths = config.get("local_paths", {}) if isinstance(config.get("local_paths", {}), dict) else {}
    if paths.get("warehouse_root"):
        WAREHOUSE_ROOT = Path(paths["warehouse_root"])
    if paths.get("research_root"):
        RESEARCH_ROOT = Path(paths["research_root"])
    if paths.get("raw_1m_root"):
        RAW_1M_ROOT = Path(paths["raw_1m_root"])
    if paths.get("nodefactorfactory_src"):
        NFF_SRC_ROOT = Path(paths["nodefactorfactory_src"])
    CONTROLS_ROOT = RESEARCH_ROOT / "derived_inputs" / "daily_bar_controls_v2_1"
    for attr, value in {
        "WAREHOUSE_ROOT": WAREHOUSE_ROOT,
        "RESEARCH_ROOT": RESEARCH_ROOT,
        "RAW_1M_ROOT": RAW_1M_ROOT,
        "NFF_SRC_ROOT": NFF_SRC_ROOT,
    }.items():
        if hasattr(V2, attr):
            setattr(V2, attr, value)


def _cli_flags(argv: list[str]) -> set[str]:
    return {arg.split("=", 1)[0] for arg in argv if arg.startswith("--")}


def apply_config_args(args: argparse.Namespace, config: dict[str, Any], explicit_flags: set[str]) -> dict[str, Any]:
    run = config.get("run", {}) if isinstance(config.get("run", {}), dict) else {}
    paths = config.get("local_paths", {}) if isinstance(config.get("local_paths", {}), dict) else {}
    portfolio = config.get("portfolio_proxy", {}) if isinstance(config.get("portfolio_proxy", {}), dict) else {}
    mappings = {
        "name": ("run_id", "--run-id"),
        "start_date": ("start_date", "--start-date"),
        "end_date": ("end_date", "--end-date"),
        "horizons": ("horizons", "--horizons"),
        "initial_parallel": ("parallel", "--parallel"),
        "max_parallel": ("max_parallel", "--max-parallel"),
        "min_parallel": ("min_parallel", "--min-parallel"),
        "target_cpu_percent": ("target_cpu", "--target-cpu"),
        "memory_high_water_percent": ("memory_high_water", "--memory-high-water"),
        "memory_min_available_gb": ("memory_min_available_gb", "--memory-min-available-gb"),
        "disk_free_floor_gb": ("disk_free_floor_gb", "--disk-free-floor-gb"),
        "launch_batch_size": ("launch_batch_size", "--launch-batch-size"),
        "retries": ("retries", "--retries"),
        "min_cross_section_n": ("min_cs_n", "--min-cs-n"),
        "allow_mixed_contracts": ("allow_mixed_contracts", "--allow-mixed-contracts"),
    }
    for key, (attr, flag) in mappings.items():
        if key in run and flag not in explicit_flags:
            setattr(args, attr, run[key])
    if "controls_path" in paths and "--controls-path" not in explicit_flags:
        args.controls_path = paths["controls_path"]
    if "rebalance_minutes" in portfolio and "--rebalance-minutes" not in explicit_flags:
        args.rebalance_minutes = int(portfolio["rebalance_minutes"])
    if "cost_bps_per_turnover" in portfolio and "--cost-bps-per-turnover" not in explicit_flags:
        args.cost_bps_per_turnover = float(portfolio["cost_bps_per_turnover"])
    if not args.run_id:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "run_id": args.run_id,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "horizons": list(args.horizons),
        "parallel": args.parallel,
        "max_parallel": args.max_parallel,
        "min_parallel": args.min_parallel,
        "target_cpu": args.target_cpu,
        "memory_high_water": args.memory_high_water,
        "memory_min_available_gb": args.memory_min_available_gb,
        "disk_free_floor_gb": args.disk_free_floor_gb,
        "launch_batch_size": args.launch_batch_size,
        "retries": args.retries,
        "min_cs_n": args.min_cs_n,
        "rebalance_minutes": args.rebalance_minutes,
        "cost_bps_per_turnover": args.cost_bps_per_turnover,
        "allow_mixed_contracts": bool(args.allow_mixed_contracts),
        "paths": {
            "warehouse_root": str(WAREHOUSE_ROOT),
            "research_root": str(RESEARCH_ROOT),
            "raw_1m_root": str(RAW_1M_ROOT),
            "nodefactorfactory_src": str(NFF_SRC_ROOT),
            "controls_path": str(args.controls_path) if args.controls_path else None,
        },
        "session": {
            "timezone": str(SESSION_TZ),
            "regular_open": f"{SESSION_OPEN[0]:02d}:{SESSION_OPEN[1]:02d}",
            "regular_close_exclusive": f"{SESSION_CLOSE[0]:02d}:{SESSION_CLOSE[1]:02d}",
        },
    }


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def build_run_contract(out_root: Path, controls_path: Path, effective_config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    runner_path = Path(__file__).resolve()
    dependency_path = Path(V2.__file__).resolve() if getattr(V2, "__file__", None) else None
    contract = {
        "contract_version": "v2_1_20260803_dst_contract_v1",
        "created_utc": utc_now(),
        "git_commit": current_git_commit(),
        "runner_path": str(runner_path),
        "runner_sha256": file_sha256(runner_path),
        "dependency_runner_path": str(dependency_path) if dependency_path else None,
        "dependency_runner_sha256": file_sha256(dependency_path) if dependency_path else None,
        "config_sha256": json_sha256(effective_config),
        "feature_registry_sha256": json_sha256(REPRESENTATIVE_ALPHA_FEATURES),
        "label_contract_sha256": json_sha256(list(LABEL_FAMILIES)),
        "controls_path": str(controls_path),
        "controls_sha256": file_sha256(controls_path),
        "warehouse_root": str(WAREHOUSE_ROOT),
        "session": effective_config["session"],
        "portfolio_accounting": "same_sleeve_turnover",
        "effective_config": effective_config,
    }
    contract_hash = json_sha256({k: v for k, v in contract.items() if k != "created_utc"})
    contract["run_contract_hash"] = contract_hash
    path = out_root / "run_contract.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("run_contract_hash") != contract_hash:
            raise RuntimeError(
                f"run contract mismatch for {out_root}; existing={existing.get('run_contract_hash')} current={contract_hash}. "
                "Use a new run id or remove the old exploratory run explicitly."
            )
    else:
        atomic_write_json(path, contract)
    atomic_write_json(out_root / "effective_config.json", effective_config)
    return contract, contract_hash


def read_run_contract_hash(out_root: Path) -> str | None:
    path = out_root / "run_contract.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("run_contract_hash")
    except Exception:
        return None


def load_dates(start_date: str, end_date: str) -> list[str]:
    dates = NFFWarehouseCatalog(WAREHOUSE_ROOT).available_dates("feature", "minute_nvg", "v3")
    return [date for date in dates if start_date <= date <= end_date]


def source_columns(kind: str, dataset: str, schema: str | None = None) -> list[str]:
    catalog = NFFWarehouseCatalog(WAREHOUSE_ROOT)
    return [column for column in catalog.columns(kind, dataset, schema) if column not in KEY_COLUMNS]


def feature_sets() -> dict[str, dict[str, Any]]:
    return V2.feature_sets()


def canonical_sets() -> dict[str, dict[str, Any]]:
    available_bars = set(source_columns("canonical", "bars_1m", "v1"))
    available_core = set(source_columns("canonical", "trades_1m_core", "v1"))
    return {
        "bars_1m": {"schema_version": "v1", "columns": [c for c in BAR_COLUMNS if c in available_bars]},
        "trades_1m_core": {
            "schema_version": "v1",
            "columns": [c for c in TRADES_CORE_COLUMNS if c in available_core],
        },
    }


def warehouse_bar_day_path(trade_date: str) -> Path:
    return WAREHOUSE_ROOT / "canonical" / "bars_1m" / "schema=v1" / f"date={trade_date}"


def regular_session_bounds_utc(trade_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    local_day = datetime.fromisoformat(trade_date).date()
    open_local = datetime(local_day.year, local_day.month, local_day.day, SESSION_OPEN[0], SESSION_OPEN[1], tzinfo=SESSION_TZ)
    close_local = datetime(local_day.year, local_day.month, local_day.day, SESSION_CLOSE[0], SESSION_CLOSE[1], tzinfo=SESSION_TZ)
    return pd.Timestamp(open_local.astimezone(timezone.utc)), pd.Timestamp(close_local.astimezone(timezone.utc))


def loader_session_range(trade_date: str) -> tuple[str, str]:
    open_utc, close_utc = regular_session_bounds_utc(trade_date)
    # NFFDataLoader time filters are inclusive; use the last regular-session minute.
    end_utc = close_utc - pd.Timedelta(minutes=1)
    return open_utc.isoformat().replace("+00:00", "Z"), end_utc.isoformat().replace("+00:00", "Z")


def _regular_session_filter(df: pd.DataFrame, column: str) -> pd.Series:
    ts = pd.to_datetime(df[column], utc=True, errors="coerce")
    local = ts.dt.tz_convert(SESSION_TZ)
    minute = local.dt.hour * 60 + local.dt.minute
    open_minute = SESSION_OPEN[0] * 60 + SESSION_OPEN[1]
    close_minute = SESSION_CLOSE[0] * 60 + SESSION_CLOSE[1]
    return ts.notna() & (minute >= open_minute) & (minute < close_minute)


def warehouse_daily_stats(trade_date: str) -> pd.DataFrame:
    day_dir = warehouse_bar_day_path(trade_date)
    parts = sorted(day_dir.glob("*.parquet"))
    frames: list[pd.DataFrame] = []
    for part in parts:
        df = pd.read_parquet(part, columns=["symbol", "timestamp", "close", "volume", "dollar_volume"])
        if df.empty:
            continue
        df = df.loc[_regular_session_filter(df, "timestamp")]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce").clip(lower=0)
    data["dollar_volume"] = pd.to_numeric(data["dollar_volume"], errors="coerce").clip(lower=0)
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.sort_values(["symbol", "timestamp"])
    grouped = data.groupby("symbol", sort=False)
    out = grouped.agg(
        daily_dollar_volume=("dollar_volume", "sum"),
        daily_volume=("volume", "sum"),
        last_close=("close", "last"),
        bar_count=("close", "count"),
    ).reset_index()
    out["daily_vwap"] = out["daily_dollar_volume"] / out["daily_volume"].replace(0, np.nan)
    out["trade_date"] = trade_date
    out["source"] = "warehouse_bars_1m"
    return out


def raw_daily_stats(zip_path: Path, target_symbols: set[str] | None = None) -> pd.DataFrame:
    if str(NFF_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(NFF_SRC_ROOT))
    import pyzipper
    from nodefactor_factory.zipio import zip_password

    trade_date = zip_path.stem[:4] + "-" + zip_path.stem[4:6] + "-" + zip_path.stem[6:8]
    rows: list[dict[str, Any]] = []
    with pyzipper.AESZipFile(zip_path) as archive:
        archive.setpassword(zip_password(zip_path, SALT))
        members = [name for name in archive.namelist() if name.endswith(".csv")]
        for member in members:
            symbol = Path(member).stem
            if target_symbols is not None and symbol not in target_symbols:
                continue
            try:
                with archive.open(member) as handle:
                    df = pd.read_csv(handle, usecols=["close", "volume", "amount", "bob"], low_memory=False)
            except Exception:
                continue
            if df.empty:
                continue
            df = df.loc[_regular_session_filter(df, "bob")]
            volume = pd.to_numeric(df["volume"], errors="coerce").clip(lower=0)
            amount = pd.to_numeric(df["amount"], errors="coerce").clip(lower=0)
            close = pd.to_numeric(df["close"], errors="coerce").dropna()
            volume_sum = float(volume.sum())
            dollar_sum = float(amount.sum())
            rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "daily_dollar_volume": dollar_sum,
                    "daily_volume": volume_sum,
                    "daily_vwap": dollar_sum / volume_sum if volume_sum > 0 else math.nan,
                    "last_close": float(close.iloc[-1]) if len(close) else math.nan,
                    "bar_count": int(close.shape[0]),
                    "source": "raw_1m_zip",
                }
            )
    return pd.DataFrame(rows)


def available_raw_dates() -> list[str]:
    paths = sorted(RAW_1M_ROOT.glob("20????/20??????.zip"))
    out = []
    for path in paths:
        stem = path.stem
        if len(stem) == 8 and stem.isdigit():
            out.append(f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}")
    return out


def collect_research_symbols(start_date: str, end_date: str, controls_root: Path) -> set[str]:
    cache_path = controls_root / f"research_symbols_{start_date}_{end_date}.json"
    if cache_path.exists():
        return set(json.loads(cache_path.read_text(encoding="utf-8"))["symbols"])
    symbols: set[str] = set()
    for trade_date in load_dates(start_date, end_date):
        day_dir = WAREHOUSE_ROOT / "features" / "minute_nvg" / "schema=v3" / f"date={trade_date}"
        for part in sorted(day_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(part, columns=["symbol"])
            except Exception:
                continue
            symbols.update(str(value) for value in df["symbol"].dropna().unique())
    atomic_write_json(cache_path, {"created_utc": utc_now(), "symbols": sorted(symbols), "count": len(symbols)})
    return symbols


def _cached_daily_stats(cache_dir: Path, key: str, builder: Any) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.parquet"
    success = path.with_suffix(path.suffix + "._SUCCESS")
    if path.exists() and success.exists():
        return pd.read_parquet(path)
    df = builder()
    if not df.empty:
        df.to_parquet(path, index=False)
        success.write_text(utc_now(), encoding="utf-8")
    return df


def build_daily_controls(start_date: str, end_date: str, controls_root: Path, raw_workers: int = 8, raw_lookback_days: int = 25) -> Path:
    controls_root.mkdir(parents=True, exist_ok=True)
    output_path = controls_root / f"daily_controls_{start_date}_{end_date}.parquet"
    success_path = output_path.with_suffix(output_path.suffix + "._SUCCESS")
    if output_path.exists() and success_path.exists():
        return output_path

    warehouse_dates = load_dates(start_date, end_date)
    raw_dates = [date for date in available_raw_dates() if date < start_date][-raw_lookback_days:] if raw_lookback_days > 0 else []
    frames: list[pd.DataFrame] = []
    target_symbols = collect_research_symbols(start_date, end_date, controls_root)

    raw_zip_paths = [RAW_1M_ROOT / date[:7].replace("-", "") / date.replace("-", "") for date in raw_dates]
    raw_zip_paths = [path.with_suffix(".zip") for path in raw_zip_paths if path.with_suffix(".zip").exists()]
    with ThreadPoolExecutor(max_workers=max(1, raw_workers)) as pool:
        futures = {
            pool.submit(
                _cached_daily_stats,
                controls_root / "daily_stats_raw",
                path.stem,
                lambda p=path, s=target_symbols: raw_daily_stats(p, s),
            ): path
            for path in raw_zip_paths
        }
        for future in as_completed(futures):
            df = future.result()
            if not df.empty:
                frames.append(df)

    for trade_date in warehouse_dates:
        df = _cached_daily_stats(
            controls_root / "daily_stats_warehouse",
            trade_date,
            lambda d=trade_date: warehouse_daily_stats(d),
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("no daily bar stats available")
    daily = pd.concat(frames, ignore_index=True)
    daily = daily.sort_values(["symbol", "trade_date"])
    daily["daily_dollar_volume"] = pd.to_numeric(daily["daily_dollar_volume"], errors="coerce").clip(lower=0)
    daily["daily_volume"] = pd.to_numeric(daily["daily_volume"], errors="coerce").clip(lower=0)
    daily["last_close"] = pd.to_numeric(daily["last_close"], errors="coerce")
    daily["bar_count"] = pd.to_numeric(daily["bar_count"], errors="coerce").fillna(0).astype("int32")

    controls: list[pd.DataFrame] = []
    for _, group in daily.groupby("symbol", sort=False):
        group = group.sort_values("trade_date").copy()
        prev_dollar = group["daily_dollar_volume"].shift(1)
        prev_volume = group["daily_volume"].shift(1)
        group["adv20"] = prev_dollar.rolling(20, min_periods=10).mean()
        group["adv20_days"] = prev_dollar.rolling(20, min_periods=1).count()
        group["vwap20"] = prev_dollar.rolling(20, min_periods=10).sum() / prev_volume.rolling(20, min_periods=10).sum().replace(0, np.nan)
        controls.append(group)
    result = pd.concat(controls, ignore_index=True)
    result = result[result["trade_date"].between(start_date, end_date)].copy()
    result["adv20_rank_desc"] = result.groupby("trade_date")["adv20"].rank(method="first", ascending=False)
    result["adv20_top1000"] = result["adv20_rank_desc"] <= 1000
    result.to_parquet(output_path, index=False)
    result.to_csv(output_path.with_suffix(".csv"), index=False)
    atomic_write_json(
        controls_root / f"daily_controls_{start_date}_{end_date}_meta.json",
        {
            "created_utc": utc_now(),
            "start_date": start_date,
            "end_date": end_date,
            "raw_lookback_dates": raw_dates,
            "warehouse_dates": warehouse_dates,
            "rows": int(len(result)),
            "symbols": int(result["symbol"].nunique()),
            "output_path": str(output_path),
        },
    )
    success_path.write_text(utc_now(), encoding="utf-8")
    return output_path


def _future_roll(series: pd.Series, window: int, op: str) -> pd.Series:
    return V2._future_roll(series, window, op)


def add_all_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = V2.add_hawkes_derived(frame)
    features = V2.add_traditional_factors(features)
    features = V2.add_research_signals(features)
    features = features.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).astype("float32")
    return features


def build_labels_and_masks(frame: pd.DataFrame, horizons: list[int]) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    labels = pd.DataFrame(index=frame.index)
    masks: dict[str, pd.Series] = {}
    group = frame.groupby(level="instrument", sort=False, group_keys=False)
    open_px = frame["bars_1m__open"].astype(float).replace(0, np.nan)
    close_px = frame["bars_1m__close"].astype(float).replace(0, np.nan)
    vwap_px = frame["bars_1m__vwap"].astype(float).replace(0, np.nan)
    volume = frame.get("bars_1m__volume", pd.Series(np.nan, index=frame.index)).astype(float).clip(lower=0)
    dollar_volume = frame["bars_1m__dollar_volume"].astype(float).clip(lower=0)
    trade_count = frame.get("trades_1m_core__trade_count", pd.Series(np.nan, index=frame.index)).astype(float).clip(lower=0)
    liquidity_log = np.log1p(dollar_volume) + 0.25 * np.log1p(trade_count)
    ret1 = group["bars_1m__close"].pct_change(fill_method=None).astype(float)
    amihud = ret1.abs() / (dollar_volume + 1.0) * 1e8

    for horizon in horizons:
        entry_volume = volume.groupby(level="instrument", sort=False, group_keys=False).shift(-1)
        exit_volume = volume.groupby(level="instrument", sort=False, group_keys=False).shift(-(horizon + 1))
        real_volume_mask = (entry_volume > 0) & (exit_volume > 0)
        for family, px in [
            ("return_open_to_open", open_px),
            ("return_vwap_to_vwap", vwap_px),
            ("return_close_to_close", close_px),
        ]:
            entry = px.groupby(level="instrument", sort=False, group_keys=False).shift(-1)
            exit_px = px.groupby(level="instrument", sort=False, group_keys=False).shift(-(horizon + 1))
            name = f"{family}__h{horizon}"
            labels[name] = exit_px / entry - 1.0
            masks[name] = real_volume_mask

        entry_liquidity = liquidity_log.groupby(level="instrument", sort=False, group_keys=False).shift(-1)
        future_liquidity = entry_liquidity.groupby(level="instrument", sort=False, group_keys=False).transform(
            lambda s, w=horizon: _future_roll(s.astype(float), w, "mean")
        )
        name = f"liquidity_deterioration__h{horizon}"
        labels[name] = entry_liquidity - future_liquidity
        masks[name] = real_volume_mask

        name = f"realized_volatility__h{horizon}"
        future_var = (ret1.astype(float) ** 2).groupby(level="instrument", sort=False, group_keys=False).transform(
            lambda s, w=horizon: _future_roll(s, w, "sum")
        )
        labels[name] = np.sqrt(future_var)
        masks[name] = real_volume_mask

        future_abs_max = ret1.abs().groupby(level="instrument", sort=False, group_keys=False).transform(
            lambda s, w=horizon: _future_roll(s.astype(float), w, "max")
        )
        threshold = future_abs_max.groupby(level="datetime", sort=False).transform(lambda s: s.quantile(0.99))
        name = f"jump_tail_event__h{horizon}"
        labels[name] = (future_abs_max > threshold).astype("float32")
        labels.loc[future_abs_max.isna() | threshold.isna(), name] = np.nan
        masks[name] = real_volume_mask

        name = f"execution_cost_proxy__h{horizon}"
        labels[name] = amihud.groupby(level="instrument", sort=False, group_keys=False).transform(
            lambda s, w=horizon: _future_roll(s.astype(float), w, "mean")
        )
        masks[name] = real_volume_mask

    return labels.replace([np.inf, -np.inf], np.nan).astype("float32"), masks


def join_daily_controls(features: pd.DataFrame, trade_date: str, controls_path: Path) -> pd.DataFrame:
    daily = pd.read_parquet(controls_path, filters=[("trade_date", "==", trade_date)])
    daily = daily.set_index("symbol")
    symbols = pd.Index(features.index.get_level_values("instrument"), name="symbol")
    matched = daily.reindex(symbols)
    controls = pd.DataFrame(index=features.index)
    close = features["bars_1m__close"].astype(float).replace(0, np.nan)
    dollar_volume = features["bars_1m__dollar_volume"].astype(float).clip(lower=0)
    trade_count = features.get("trades_1m_core__trade_count", pd.Series(np.nan, index=features.index)).astype(float).clip(lower=0)
    controls["control__log_price"] = np.log(close)
    controls["control__log_adv20"] = np.log1p(pd.Series(matched["adv20"].to_numpy(), index=features.index).astype(float))
    controls["control__log_intraday_dollar_volume"] = np.log1p(dollar_volume)
    controls["control__active_second_ratio_300s"] = features.get(
        "trade_nvg__trade_active_second_ratio_300s", pd.Series(np.nan, index=features.index)
    ).astype(float)
    controls["control__log_trade_count"] = np.log1p(trade_count)
    ret1 = features.groupby(level="instrument", sort=False)["bars_1m__close"].pct_change(fill_method=None).astype(float)
    controls["control__realized_vol_60m"] = ret1.groupby(level="instrument", sort=False, group_keys=False).transform(
        lambda s: np.sqrt((s.astype(float) ** 2).rolling(60, min_periods=20).sum())
    )
    market_ret = ret1.groupby(level="datetime", sort=False).transform("mean")
    rm = ret1 * market_ret
    tmp_beta = pd.DataFrame({"ret": ret1, "market_ret": market_ret, "ret_market": rm}, index=features.index)
    by_instrument = tmp_beta.groupby(level="instrument", sort=False, group_keys=False)
    mean_r = by_instrument["ret"].transform(lambda s: s.rolling(60, min_periods=20).mean())
    mean_m = by_instrument["market_ret"].transform(lambda s: s.rolling(60, min_periods=20).mean())
    mean_rm = by_instrument["ret_market"].transform(lambda s: s.rolling(60, min_periods=20).mean())
    var_m = by_instrument["market_ret"].transform(lambda s: s.rolling(60, min_periods=20).var())
    beta = (mean_rm - mean_r * mean_m) / var_m.replace(0, np.nan)
    controls["control__beta_60m_intraday_proxy"] = beta.reindex(features.index)
    controls["control__adv20_days"] = pd.Series(matched["adv20_days"].to_numpy(), index=features.index).astype(float)
    controls["control__adv20_rank_desc"] = pd.Series(matched["adv20_rank_desc"].to_numpy(), index=features.index).astype(float)
    controls["control__adv20_top1000"] = pd.Series(matched["adv20_top1000"].to_numpy(), index=features.index).astype(float)
    controls["control__last_close_prevday"] = pd.Series(matched["last_close"].to_numpy(), index=features.index).astype(float)
    return controls.replace([np.inf, -np.inf], np.nan).astype("float32")


def infer_bundle(name: str) -> str:
    return V2.infer_bundle(name)


def analysis_features(features: pd.DataFrame) -> list[str]:
    available = set(features.columns)
    selected = [
        column
        for column in REPRESENTATIVE_ALPHA_FEATURES
        if column in available and features[column].notna().any() and not QUALITY_OR_CONTROL_RE.search(column)
    ]
    return selected


def feature_exclusion_reason(column: str, evaluated: bool) -> str:
    if evaluated:
        return "evaluated_representative_alpha_feature"
    if DEPRECATED_SIGNAL_RE.search(column):
        return "deprecated_prespecified_negative_signal_formula_documented_not_used_as_alpha"
    if QUALITY_OR_CONTROL_RE.search(column):
        return "quality_liquidity_or_control_axis_used_for_masks_or_neutralization"
    if not V2.ANALYSIS_FEATURE_RE.search(column):
        return "outside_v2_1_representative_selector_scope"
    if column not in REPRESENTATIVE_ALPHA_FEATURES:
        return "redundant_or_nonrepresentative_feature_excluded_by_v2_1_selector"
    return "all_nan_or_unavailable"


def feature_registry(features: pd.DataFrame, evaluated_features: list[str], trade_date: str) -> pd.DataFrame:
    evaluated = set(evaluated_features)
    rows: list[dict[str, Any]] = []
    for column in features.columns:
        series = pd.to_numeric(features[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        non_null = int(series.notna().sum())
        rows.append(
            {
                "trade_date": trade_date,
                "feature": column,
                "bundle": infer_bundle(column),
                "evaluated": column in evaluated,
                "exclusion_reason": feature_exclusion_reason(column, column in evaluated),
                "non_null_count": non_null,
                "non_null_rate": float(non_null / max(1, len(series))),
                "unique_count": int(series.nunique(dropna=True)),
                "variance": float(series.var()) if non_null > 1 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def universe_masks(features: pd.DataFrame, controls: pd.DataFrame) -> dict[str, pd.Series]:
    all_mask = pd.Series(True, index=features.index)
    required = [
        "minute_nvg__price_nvg_30m_top_bottom_asymmetry",
        "trade_nvg__trade_flow_path_300s_terminal_position",
        "hawkes_derived__hawkes_signed_pressure",
    ]
    present_required = [column for column in required if column in features.columns]
    common = features[present_required].notna().all(axis=1) if present_required else all_mask.copy()
    liquid = common.copy()
    liquid &= features["bars_1m__close"].astype(float) >= 5.0
    liquid &= features.get("bars_1m__volume", pd.Series(np.nan, index=features.index)).astype(float) > 0.0
    liquid &= controls["control__adv20_top1000"].fillna(0).astype(bool)
    liquid &= controls["control__adv20_days"].fillna(0) >= 20
    liquid &= features.get("trades_1m_core__trade_count", pd.Series(np.nan, index=features.index)).astype(float) >= 1.0
    if "trade_nvg__trade_active_second_ratio_300s" in features.columns:
        liquid &= features["trade_nvg__trade_active_second_ratio_300s"].astype(float) >= 0.01
    if "trade_nvg__trade_price_stale_ratio_300s" in features.columns:
        liquid &= features["trade_nvg__trade_price_stale_ratio_300s"].astype(float) <= 0.95
    return {
        "own_feature_universe": all_mask.fillna(False),
        "common_structural": common.fillna(False),
        "liquid_common_adv20_top1000": liquid.fillna(False),
    }


def _corr(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())
    if n < 30:
        return math.nan, n
    xv = x[valid].astype("float64")
    yv = y[valid].astype("float64")
    x0 = xv - xv.mean()
    y0 = yv - yv.mean()
    denom = math.sqrt(float((x0 * x0).sum() * (y0 * y0).sum()))
    if denom <= 0:
        return math.nan, n
    return float((x0 * y0).sum() / denom), n


def _rank_ic_for_block(x: pd.DataFrame, y: pd.Series, min_n: int = 30) -> tuple[np.ndarray, np.ndarray]:
    xr = x.rank(method="average").to_numpy(dtype="float64")
    yr = y.rank(method="average").to_numpy(dtype="float64")
    corrs = np.full(x.shape[1], np.nan, dtype="float64")
    counts = np.zeros(x.shape[1], dtype="int64")
    for idx in range(x.shape[1]):
        corrs[idx], counts[idx] = _corr(xr[:, idx], yr)
        if counts[idx] < min_n:
            corrs[idx] = math.nan
    return corrs, counts


def _minute_mean_ic(
    feature_frame: pd.DataFrame,
    label: pd.Series,
    feature_columns: list[str],
    min_n: int,
) -> dict[str, dict[str, float]]:
    store: dict[str, list[float]] = {column: [] for column in feature_columns}
    counts: dict[str, int] = {column: 0 for column in feature_columns}
    label_name = "__label"
    for _, group in pd.concat([feature_frame[feature_columns], label.rename(label_name)], axis=1).groupby(
        level="datetime", sort=False
    ):
        y = group[label_name]
        if int(y.notna().sum()) < min_n:
            continue
        for start in range(0, len(feature_columns), 24):
            chunk = feature_columns[start : start + 24]
            corrs, ns = _rank_ic_for_block(group[chunk], y, min_n=min_n)
            for i, feature in enumerate(chunk):
                value = corrs[i]
                counts[feature] += int(ns[i])
                if np.isfinite(value):
                    store[feature].append(float(value))
    out: dict[str, dict[str, float]] = {}
    for feature, values in store.items():
        arr = np.asarray(values, dtype="float64")
        out[feature] = {
            "ic_minutes": int(arr.size),
            "ic_count": int(counts[feature]),
            "rank_ic_mean": float(arr.mean()) if arr.size else math.nan,
            "rank_ic_std": float(arr.std(ddof=1)) if arr.size > 1 else math.nan,
            "rank_ic_positive_ratio": float((arr > 0).mean()) if arr.size else math.nan,
        }
    return out


def _pooled_ic(
    feature_frame: pd.DataFrame,
    label: pd.Series,
    feature_columns: list[str],
    min_n: int,
) -> dict[str, dict[str, float]]:
    ranked_label = label.groupby(level="datetime", sort=False).rank(method="average")
    out: dict[str, dict[str, float]] = {}
    y = ranked_label.to_numpy(dtype="float64")
    ic_minutes = int(ranked_label.groupby(level="datetime", sort=False).count().ge(min_n).sum())
    for start in range(0, len(feature_columns), 24):
        chunk = feature_columns[start : start + 24]
        ranked_features = feature_frame[chunk].groupby(level="datetime", sort=False).rank(method="average")
        for feature in chunk:
            value, n = _corr(ranked_features[feature].to_numpy(dtype="float64"), y)
            if n < min_n:
                value = math.nan
            out[feature] = {
                "ic_minutes": ic_minutes,
                "ic_count": int(n),
                "rank_ic_mean": float(value) if np.isfinite(value) else math.nan,
                "rank_ic_std": math.nan,
                "rank_ic_positive_ratio": float(value > 0) if np.isfinite(value) else math.nan,
            }
        del ranked_features
    return out


def append_ic_rows(
    rows: list[dict[str, Any]],
    stats: dict[str, dict[str, float]],
    trade_date: str,
    universe: str,
    neutralization: str,
    method: str,
    family: str,
    horizon: int,
    coverage: pd.Series,
    label_non_null: int,
) -> None:
    for feature, stat in stats.items():
        rows.append(
            {
                "trade_date": trade_date,
                "universe": universe,
                "neutralization": neutralization,
                "rank_ic_method": method,
                "label_family": family,
                "horizon_bars": horizon,
                "feature": feature,
                "bundle": infer_bundle(feature),
                "ic_minutes": stat["ic_minutes"],
                "ic_count": stat["ic_count"],
                "rank_ic_mean": stat["rank_ic_mean"],
                "rank_ic_std": stat["rank_ic_std"],
                "rank_ic_positive_ratio": stat["rank_ic_positive_ratio"],
                "coverage": float(coverage.get(feature, 0.0)),
                "label_non_null": label_non_null,
            }
        )


def _residualize_matrix(values: pd.DataFrame, controls: pd.DataFrame, min_n: int = 40) -> pd.DataFrame:
    result = pd.DataFrame(index=values.index, columns=values.columns, dtype="float32")
    control_cols = [
        "control__log_price",
        "control__log_adv20",
        "control__log_intraday_dollar_volume",
        "control__active_second_ratio_300s",
        "control__log_trade_count",
        "control__realized_vol_60m",
        "control__beta_60m_intraday_proxy",
    ]
    control_cols = [column for column in control_cols if column in controls.columns]
    for dt, idx in values.groupby(level="datetime", sort=False).groups.items():
        c = controls.loc[idx, control_cols]
        v = values.loc[idx]
        c_clean = c.replace([np.inf, -np.inf], np.nan)
        control_valid = c_clean.notna().all(axis=1)
        if int(control_valid.sum()) < min_n:
            continue
        v_clean = v.replace([np.inf, -np.inf], np.nan)
        grouped_columns: dict[bytes, list[str]] = {}
        grouped_masks: dict[bytes, np.ndarray] = {}
        control_valid_arr = control_valid.to_numpy(dtype=bool)
        for column in values.columns:
            base_arr = control_valid_arr & v_clean[column].notna().to_numpy(dtype=bool)
            if int(base_arr.sum()) < min_n:
                continue
            key = np.packbits(base_arr).tobytes()
            grouped_columns.setdefault(key, []).append(column)
            grouped_masks[key] = base_arr
        for key, columns in grouped_columns.items():
            base_arr = grouped_masks[key]
            row_index = v_clean.index[base_arr]
            x = c_clean.loc[row_index].to_numpy(dtype="float64")
            x = np.column_stack([np.ones(x.shape[0]), x])
            y = v_clean.loc[row_index, columns].to_numpy(dtype="float64")
            try:
                coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            result.loc[row_index, columns] = (y - x @ coef).astype("float32")
    return result


def minute_rank_ic_summary(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_masks: dict[str, pd.Series],
    controls: pd.DataFrame,
    trade_date: str,
    min_n: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    feature_columns = analysis_features(features)
    decile_cache_features = [column for column in CORE_DECILE_FEATURES if column in feature_columns]
    masks = universe_masks(features, controls)
    rows: list[dict[str, Any]] = []
    residual_cache: dict[tuple[str, str], tuple[pd.DataFrame, pd.Series]] = {}

    for universe, universe_mask in masks.items():
        for label_column in labels.columns:
            family, horizon_text = label_column.rsplit("__h", 1)
            horizon = int(horizon_text)
            label_valid = labels[label_column].notna() & label_masks[label_column].fillna(False)
            base_mask = (universe_mask & label_valid).fillna(False)
            if int(base_mask.sum()) < min_n:
                continue
            feature_frame = features.loc[base_mask, feature_columns]
            label = labels.loc[base_mask, label_column]
            raw_minute_stats = _minute_mean_ic(feature_frame, label, feature_columns, min_n=min_n)
            raw_pooled_stats = _pooled_ic(feature_frame, label, feature_columns, min_n=min_n)
            label_non_null = int(label.notna().sum())
            coverage = feature_frame.notna().sum(axis=0) / max(1, label_non_null)
            append_ic_rows(
                rows,
                raw_minute_stats,
                trade_date,
                universe,
                "raw",
                "minute_mean_cs_rank_ic",
                family,
                horizon,
                coverage,
                label_non_null,
            )
            append_ic_rows(
                rows,
                raw_pooled_stats,
                trade_date,
                universe,
                "raw",
                "pooled_cs_rank_ic",
                family,
                horizon,
                coverage,
                label_non_null,
            )

            if not family.startswith("return_") or universe == "own_feature_universe":
                continue

            neutral_features = [c for c in feature_columns if c in feature_frame.columns]
            controls_sub = controls.loc[base_mask]
            feature_resid = _residualize_matrix(feature_frame[neutral_features], controls_sub, min_n=max(min_n, 40))
            label_resid = _residualize_matrix(label.to_frame(label_column), controls_sub, min_n=max(min_n, 40))[label_column]
            neut_minute_stats = _minute_mean_ic(feature_resid, label_resid, neutral_features, min_n=min_n)
            neut_pooled_stats = _pooled_ic(feature_resid, label_resid, neutral_features, min_n=min_n)
            neutral_label_non_null = int(label_resid.notna().sum())
            neutral_coverage = feature_resid.notna().sum(axis=0) / max(1, neutral_label_non_null)
            if family == "return_open_to_open" and decile_cache_features:
                cache_columns = [column for column in decile_cache_features if column in feature_resid.columns]
                if cache_columns:
                    residual_cache[(universe, label_column)] = (
                        feature_resid[cache_columns].copy(deep=True),
                        label_resid.copy(deep=True),
                    )
            append_ic_rows(
                rows,
                neut_minute_stats,
                trade_date,
                universe,
                "neutralized",
                "minute_mean_cs_rank_ic_residualized",
                family,
                horizon,
                neutral_coverage,
                neutral_label_non_null,
            )
            append_ic_rows(
                rows,
                neut_pooled_stats,
                trade_date,
                universe,
                "neutralized",
                "pooled_cs_rank_ic_residualized",
                family,
                horizon,
                neutral_coverage,
                neutral_label_non_null,
            )
            del feature_resid, label_resid, controls_sub
            gc.collect()
    return pd.DataFrame(rows), residual_cache


def _winsorize_by_minute(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    def clip_one(s: pd.Series) -> pd.Series:
        lo = s.quantile(lower)
        hi = s.quantile(upper)
        return s.clip(lo, hi)

    return series.groupby(level="datetime", sort=False, group_keys=False).transform(clip_one)


def decile_curves(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_masks: dict[str, pd.Series],
    controls: pd.DataFrame,
    residual_cache: dict[tuple[str, str], tuple[pd.DataFrame, pd.Series]],
    trade_date: str,
    min_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    masks = universe_masks(features, controls)
    decile_features = [column for column in CORE_DECILE_FEATURES if column in features.columns]
    if not decile_features:
        return pd.DataFrame()
    for universe in ["liquid_common_adv20_top1000", "common_structural"]:
        universe_mask = masks[universe]
        for label_column in [c for c in labels.columns if c.startswith("return_open_to_open__h")]:
            family, horizon_text = label_column.rsplit("__h", 1)
            horizon = int(horizon_text)
            base_mask = (universe_mask & labels[label_column].notna() & label_masks[label_column].fillna(False)).fillna(False)
            if int(base_mask.sum()) < min_n:
                continue
            base = pd.DataFrame(
                {
                    "label": labels.loc[base_mask, label_column],
                    "adv20": controls.loc[base_mask, "control__log_adv20"],
                    "price": features.loc[base_mask, "bars_1m__close"],
                    "trade_count": features.loc[base_mask].get(
                        "trades_1m_core__trade_count", pd.Series(np.nan, index=features.loc[base_mask].index)
                    ),
                }
            )
            for variant in ("raw", "winsorized", "neutralized"):
                for feature in decile_features:
                    if variant == "neutralized":
                        cached = residual_cache.get((universe, label_column))
                        if cached is None or feature not in cached[0].columns:
                            continue
                        signal = cached[0][feature]
                        label = cached[1].rename("label")
                        work = pd.concat([signal.rename("signal"), label, base[["adv20", "price", "trade_count"]]], axis=1)
                    else:
                        signal = features.loc[base_mask, feature]
                        if variant == "winsorized":
                            signal = _winsorize_by_minute(signal)
                        work = pd.concat([signal.rename("signal"), base], axis=1)
                    for rebalance in ("sample_15m",):
                        minutes = work.index.get_level_values("datetime").minute
                        sampled = work.loc[(minutes % 15) == 0]
                        sampled = sampled.dropna(subset=["signal", "label"])
                        if sampled.empty:
                            continue
                        for _, minute_block in sampled.groupby(level="datetime", sort=False):
                            if len(minute_block) < min_n:
                                continue
                            try:
                                decile = pd.qcut(minute_block["signal"].rank(method="first"), 10, labels=False) + 1
                            except ValueError:
                                continue
                            minute_block = minute_block.assign(decile=decile.astype("int16"))
                            grouped = minute_block.groupby("decile", sort=True)
                            for decile_id, decile_frame in grouped:
                                rows.append(
                                    {
                                        "trade_date": trade_date,
                                        "universe": universe,
                                        "feature": feature,
                                        "bundle": infer_bundle(feature),
                                        "label_family": family,
                                        "horizon_bars": horizon,
                                        "variant": variant,
                                        "rebalance": rebalance,
                                        "decile": int(decile_id),
                                        "mean_label": float(decile_frame["label"].mean()),
                                        "count": int(decile_frame["label"].notna().sum()),
                                        "mean_log_adv20": float(decile_frame["adv20"].mean()),
                                        "mean_price": float(decile_frame["price"].mean()),
                                        "mean_trade_count": float(decile_frame["trade_count"].mean()),
                                    }
                                )
    if not rows:
        return pd.DataFrame()
    data = pd.DataFrame(rows)
    return (
        data.groupby(
            [
                "trade_date",
                "universe",
                "feature",
                "bundle",
                "label_family",
                "horizon_bars",
                "variant",
                "rebalance",
                "decile",
            ],
            dropna=False,
        )
        .agg(
            mean_label=("mean_label", "mean"),
            count=("count", "sum"),
            mean_log_adv20=("mean_log_adv20", "mean"),
            mean_price=("mean_price", "mean"),
            mean_trade_count=("mean_trade_count", "mean"),
        )
        .reset_index()
    )


def _portfolio_weights(block: pd.DataFrame, feature: str) -> pd.Series:
    ranked = block[feature].rank(method="first")
    try:
        bucket = pd.qcut(ranked, 10, labels=False) + 1
    except ValueError:
        return pd.Series(dtype="float64")
    long_index = block.index[bucket == 10]
    short_index = block.index[bucket == 1]
    if len(long_index) == 0 or len(short_index) == 0:
        return pd.Series(dtype="float64")
    weights = pd.Series(0.0, index=block.index, dtype="float64")
    weights.loc[long_index] = 0.5 / len(long_index)
    weights.loc[short_index] = -0.5 / len(short_index)
    return weights[weights != 0.0]


def _is_rebalance_minute(dt: Any, rebalance_minutes: int) -> bool:
    stamp = pd.Timestamp(dt)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    local = stamp.tz_convert(SESSION_TZ)
    minute = local.hour * 60 + local.minute
    open_minute = SESSION_OPEN[0] * 60 + SESSION_OPEN[1]
    close_minute = SESSION_CLOSE[0] * 60 + SESSION_CLOSE[1]
    return open_minute <= minute < close_minute and (minute - open_minute) % rebalance_minutes == 0


def staggered_portfolio_proxy(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_masks: dict[str, pd.Series],
    controls: pd.DataFrame,
    trade_date: str,
    min_n: int,
    cost_bps_per_turnover: float = 1.0,
    rebalance_minutes: int = 15,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    masks = universe_masks(features, controls)
    universe = "liquid_common_adv20_top1000"
    universe_mask = masks[universe]
    portfolio_features = [column for column in CORE_DECILE_FEATURES if column in features.columns and column in analysis_features(features)]
    if not portfolio_features:
        return pd.DataFrame()
    for label_column in [c for c in labels.columns if c.startswith("return_open_to_open__h")]:
        family, horizon_text = label_column.rsplit("__h", 1)
        horizon = int(horizon_text)
        sleeve_count = max(1, int(math.ceil(horizon / rebalance_minutes)))
        base_mask = (universe_mask & labels[label_column].notna() & label_masks[label_column].fillna(False)).fillna(False)
        if int(base_mask.sum()) < min_n:
            continue
        work = pd.concat([features.loc[base_mask, portfolio_features], labels.loc[base_mask, label_column].rename("label")], axis=1)
        for feature in portfolio_features:
            prev_by_sleeve: dict[int, pd.Series] = {}
            rebalance_ordinal = 0
            for dt, block in work[[feature, "label"]].dropna().groupby(level="datetime", sort=True):
                if not _is_rebalance_minute(dt, rebalance_minutes) or len(block) < min_n:
                    continue
                sleeve_id = rebalance_ordinal % sleeve_count
                rebalance_ordinal += 1
                weights = _portfolio_weights(block, feature)
                if weights.empty:
                    continue
                long_mask = weights > 0
                short_mask = weights < 0
                gross_return = float((weights * block.loc[weights.index, "label"]).sum())
                prev_weights = prev_by_sleeve.get(sleeve_id)
                if prev_weights is None:
                    turnover = float(weights.abs().sum())
                else:
                    combined = weights.reindex(weights.index.union(prev_weights.index), fill_value=0.0)
                    previous = prev_weights.reindex(combined.index, fill_value=0.0)
                    turnover = float((combined - previous).abs().sum())
                cost = turnover * cost_bps_per_turnover / 10000.0
                rows.append(
                    {
                        "trade_date": trade_date,
                        "datetime": dt,
                        "universe": universe,
                        "feature": feature,
                        "bundle": infer_bundle(feature),
                        "label_family": family,
                        "horizon_bars": horizon,
                        "rebalance_minutes": rebalance_minutes,
                        "sleeve_count": sleeve_count,
                        "sleeve_id": sleeve_id,
                        "portfolio_accounting": "same_sleeve_turnover",
                        "long_count": int(long_mask.sum()),
                        "short_count": int(short_mask.sum()),
                        "gross_return": gross_return,
                        "turnover": turnover,
                        "cost_bps_per_turnover": cost_bps_per_turnover,
                        "net_return": gross_return - cost,
                    }
                )
                prev_by_sleeve[sleeve_id] = weights
    return pd.DataFrame(rows)


def unit_paths(out_root: Path, trade_date: str) -> tuple[Path, Path, Path]:
    out_dir = out_root / "02_neutralized_factor_diagnostics" / f"date={trade_date}"
    return out_dir, out_dir / "factor_rank_ic_summary.parquet", out_dir / "_SUCCESS"


def run_date(
    trade_date: str,
    horizons: list[int],
    out_root: Path,
    controls_path: Path,
    min_n: int,
    contract_hash: str | None,
    allow_mixed_contracts: bool,
    rebalance_minutes: int,
    cost_bps_per_turnover: float,
) -> dict[str, Any]:
    out_dir, summary_path, success_path = unit_paths(out_root, trade_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    if success_path.exists() and summary_path.exists():
        if meta_path.exists():
            try:
                existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                existing_meta = {}
            if existing_meta.get("run_contract_hash") == contract_hash:
                return {"trade_date": trade_date, "status": "skipped", "out_dir": str(out_dir)}

    loader = NFFDataLoader(
        warehouse_root=WAREHOUSE_ROOT,
        canonical_sets=canonical_sets(),
        feature_sets=feature_sets(),
        execution={"frequency": "1min", "delay_bars": 1, "collision_policy": "latest"},
        label=None,
        join="inner",
        strict_manifests=True,
        allow_mixed_contracts=allow_mixed_contracts,
        output_float32=True,
        arrow_use_threads=True,
    )
    start_time, end_time = loader_session_range(trade_date)
    frame = loader.load(
        instruments="all",
        start_time=start_time,
        end_time=end_time,
    )
    features = add_all_features(frame["feature"].sort_index())
    del frame
    gc.collect()
    labels, label_masks = build_labels_and_masks(features, horizons)
    controls = join_daily_controls(features, trade_date, controls_path)
    evaluated_features = analysis_features(features)
    registry = feature_registry(features, evaluated_features, trade_date)
    summary, residual_cache = minute_rank_ic_summary(features, labels, label_masks, controls, trade_date, min_n)
    deciles = decile_curves(features, labels, label_masks, controls, residual_cache, trade_date, min_n)
    del residual_cache
    gc.collect()
    portfolio = staggered_portfolio_proxy(
        features,
        labels,
        label_masks,
        controls,
        trade_date,
        min_n,
        cost_bps_per_turnover=cost_bps_per_turnover,
        rebalance_minutes=rebalance_minutes,
    )

    summary.to_parquet(summary_path, index=False)
    summary.to_csv(out_dir / "factor_rank_ic_summary.csv", index=False)
    registry.to_parquet(out_dir / "feature_registry.parquet", index=False)
    registry.to_csv(out_dir / "feature_registry.csv", index=False)
    if not deciles.empty:
        deciles.to_parquet(out_dir / "decile_curves.parquet", index=False)
        deciles.to_csv(out_dir / "decile_curves.csv", index=False)
    if not portfolio.empty:
        portfolio.to_parquet(out_dir / "staggered_portfolio_proxy.parquet", index=False)
        portfolio.to_csv(out_dir / "staggered_portfolio_proxy.csv", index=False)

    masks = universe_masks(features, controls)
    meta = {
        "trade_date": trade_date,
        "run_contract_hash": contract_hash,
        "regular_session": {
            "timezone": str(SESSION_TZ),
            "open_local": f"{SESSION_OPEN[0]:02d}:{SESSION_OPEN[1]:02d}",
            "close_local_exclusive": f"{SESSION_CLOSE[0]:02d}:{SESSION_CLOSE[1]:02d}",
            "load_start_utc": start_time,
            "load_end_utc_inclusive": end_time,
        },
        "horizons": horizons,
        "label_families": list(LABEL_FAMILIES),
        "rows": int(len(features)),
        "feature_count": int(features.shape[1]),
        "analysis_feature_count": int(len(evaluated_features)),
        "analysis_features": evaluated_features,
        "summary_rows": int(len(summary)),
        "decile_rows": int(len(deciles)),
        "portfolio_rows": int(len(portfolio)),
        "portfolio_accounting": "same_sleeve_turnover",
        "label_non_null": {column: int(labels[column].notna().sum()) for column in labels.columns},
        "label_real_volume_valid": {column: int(label_masks[column].fillna(False).sum()) for column in labels.columns},
        "universe_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "controls_path": str(controls_path),
        "controls_available": {column: int(controls[column].notna().sum()) for column in controls.columns},
        "neutralization_controls": [
            "log_price",
            "previous_20d_adv",
            "same_minute_dollar_volume",
            "active_second_ratio_300s",
            "trade_count",
            "realized_vol_60m",
            "intraday_beta_60m_proxy",
        ],
        "sector_neutralization": "not_available_no_local_sector_reference_found",
        "label_contract": {
            "return_open_to_open__hN": "open[t+N+1] / open[t+1] - 1; decision at t; requires entry and exit real bar volume > 0; same-session only.",
            "return_vwap_to_vwap__hN": "vwap[t+N+1] / vwap[t+1] - 1; decision at t; requires entry and exit real bar volume > 0; same-session only.",
            "return_close_to_close__hN": "close[t+N+1] / close[t+1] - 1; decision at t; requires entry and exit real bar volume > 0; same-session only.",
            "liquidity_deterioration__hN": "log liquidity at entry minute minus mean future log liquidity over horizon; log liquidity = log1p(dollar_volume) + 0.25*log1p(trade_count); requires entry and exit real bar volume > 0.",
            "realized_volatility__hN": "sqrt(sum of future one-minute close-return squared over horizon); requires entry and exit real bar volume > 0.",
            "jump_tail_event__hN": "1 if future max absolute one-minute close return over horizon is above the same decision-minute cross-sectional 99th percentile, else 0; requires entry and exit real bar volume > 0.",
            "execution_cost_proxy__hN": "mean future Amihud-style abs(1m close return)/(dollar_volume+1)*1e8 over horizon; requires entry and exit real bar volume > 0.",
        },
        "universe_contract": {
            "own_feature_universe": "inner-join base universe loaded by the NFF/Qlib decision-time join; retained name is backwards-compatible and is not a true per-feature own universe.",
            "common_structural": "rows with non-null 30m price NVG asymmetry, 300s trade-flow terminal position, and Hawkes signed pressure.",
            "liquid_common_adv20_top1000": "common_structural plus price >= 5, current bar real volume > 0, PIT previous-20-trading-day ADV top 1000, 20 ADV lookback days, trade_count >= 1, active_second_ratio_300s >= 0.01, and stale_ratio_300s <= 0.95 when available.",
        },
        "signal_contract": {
            "signal__neg_minute_nvg_price_nvg_30m_top_bottom_asymmetry": "-minute_nvg__price_nvg_30m_top_bottom_asymmetry; documented but excluded from v2.1 alpha screen.",
            "signal__neg_trade_flow_path_300s_terminal_position": "-trade_nvg__trade_flow_path_300s_terminal_position; documented but excluded from v2.1 alpha screen.",
            "signal__neg_trade_flow_path_300s_signed_change": "-trade_nvg__trade_flow_path_300s_signed_change; documented but excluded from v2.1 alpha screen.",
            "signal__neg_hawkes_signed_pressure": "-hawkes_derived__hawkes_signed_pressure; documented but excluded from v2.1 alpha screen.",
            "signal__structural_overextension_consensus": "cross-sectional z-score mean of selected negative overextension signals; documented but excluded from v2.1 alpha screen.",
        },
        "loader_report": loader.last_load_report,
    }
    atomic_write_json(meta_path, meta)
    success_path.write_text(utc_now(), encoding="utf-8")
    return {"trade_date": trade_date, "status": "completed", **meta}


def _normal_two_sided_pvalue(t_value: float) -> float:
    if not np.isfinite(t_value):
        return math.nan
    return float(math.erfc(abs(t_value) / math.sqrt(2.0)))


def _hac_tstat(values: pd.Series, max_lag: int = 5) -> tuple[float, float, int]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype="float64")
    n = int(x.size)
    if n < 3:
        return math.nan, math.nan, n
    centered = x - x.mean()
    lag = min(max_lag, n - 1)
    gamma0 = float(np.dot(centered, centered) / n)
    var = gamma0
    for k in range(1, lag + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        var += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    se = math.sqrt(max(var, 0.0) / n)
    if se <= 0:
        return math.nan, math.nan, n
    t_value = float(x.mean() / se)
    return t_value, _normal_two_sided_pvalue(t_value), n


def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype="float64")
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return out
    ranks = np.arange(1, m + 1, dtype="float64")
    q = (valid.to_numpy(dtype="float64") * m / ranks)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out.loc[valid.index] = np.clip(q, 0, 1)
    return out


def _block_bootstrap_mean(values: pd.Series, reps: int = 300, block_size: int = 5) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype="float64")
    n = x.size
    if n < block_size:
        return {"bootstrap_mean": math.nan, "bootstrap_ci_low": math.nan, "bootstrap_ci_high": math.nan, "bootstrap_zero_crossing_rate": math.nan}
    rng = np.random.default_rng(20260803)
    starts = np.arange(0, n)
    draws = np.empty(reps, dtype="float64")
    for i in range(reps):
        sampled: list[np.ndarray] = []
        while sum(len(chunk) for chunk in sampled) < n:
            start = int(rng.choice(starts))
            end = min(n, start + block_size)
            sampled.append(x[start:end])
        draw = np.concatenate(sampled)[:n]
        draws[i] = draw.mean()
    p_two = 2.0 * min(float((draws <= 0).mean()), float((draws >= 0).mean()))
    return {
        "bootstrap_mean": float(draws.mean()),
        "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
        "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_zero_crossing_rate": min(1.0, p_two),
    }


def write_research_contract(out_root: Path) -> None:
    contract = {
        "name": "NFF v2.1 neutralized representative validity and same-sleeve portfolio proxy",
        "rank_ic_methods": [
            "minute_mean_cs_rank_ic",
            "pooled_cs_rank_ic",
            "minute_mean_cs_rank_ic_residualized",
            "pooled_cs_rank_ic_residualized",
        ],
        "alpha_feature_policy": {
            "included": REPRESENTATIVE_ALPHA_FEATURES,
            "excluded_quality_control_regex": QUALITY_OR_CONTROL_RE.pattern,
            "deprecated_signal_regex": DEPRECATED_SIGNAL_RE.pattern,
            "note": "Readiness, warmup, coverage, active-second, stale, and trade-count axes are masks/controls, not alpha inputs.",
        },
        "universe_contract": {
            "own_feature_universe": "inner-join base universe loaded by the NFF/Qlib decision-time join; retained name is backwards-compatible and is not a true per-feature own universe.",
            "common_structural": "non-null structural price/trade/Hawkes representative features.",
            "liquid_common_adv20_top1000": "common_structural + price >= 5 + real current volume + PIT previous-20d ADV top 1000 + 20 lookback days + trade_count >= 1 + active/stale filters.",
        },
        "label_contract": {
            "return_open_to_open__hN": "open[t+N+1] / open[t+1] - 1.",
            "return_vwap_to_vwap__hN": "vwap[t+N+1] / vwap[t+1] - 1.",
            "return_close_to_close__hN": "close[t+N+1] / close[t+1] - 1.",
            "liquidity_deterioration__hN": "log liquidity at entry minute minus mean future log liquidity.",
            "realized_volatility__hN": "sqrt(sum future 1m close-return squared).",
            "jump_tail_event__hN": "future max abs 1m close-return above same-minute cross-sectional 99th percentile.",
            "execution_cost_proxy__hN": "future mean Amihud-style abs(1m close return)/(dollar_volume+1)*1e8.",
        },
        "portfolio_proxy_contract": {
            "rebalance": "Every 15 NY regular-session minutes.",
            "portfolio": "Equal-weight long top decile and short bottom decile in liquid_common_adv20_top1000.",
            "capital": "Same-sleeve accounting; for horizon H and 15m rebalance, ceil(H/15) sleeves are tracked and turnover compares a sleeve only with its own prior weights.",
            "cost": "net_return = gross_return - turnover * cost_bps_per_turnover / 10000.",
        },
        "session_contract": {
            "timezone": str(SESSION_TZ),
            "regular_session": "09:30 <= local_time < 16:00",
        },
        "qlib_recorder_assessment": {
            "status": "not_used_for_v2_1_validity_runner",
            "reason": "The current artifact is an intraday factor validity and sleeve-proxy research pass. Qlib Recorder can be added after predictions/positions are materialized, but custom overlapping sleeve accounting is clearer in a dedicated simulator.",
            "recommended_next_step": "Export prediction, target weight, executed sleeve weight, and realized return tables; then attach them to a Qlib Recorder or a custom recorder-compatible artifact store.",
        },
        "known_limits": [
            "Sector neutralization is not applied because no local sector reference file was found.",
            "Intraday beta is a 60-minute proxy, not a 20-day market beta.",
            "Portfolio output is a transparent same-sleeve proxy, not a full Qlib Recorder/model backtest.",
            "July 2026 should be read as a stress slice, not true out-of-sample.",
        ],
    }
    atomic_write_json(out_root / "research_contract_v2_1.json", contract)


def aggregate(out_root: Path) -> None:
    base = out_root / "02_neutralized_factor_diagnostics"
    agg = base / "_aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    write_research_contract(out_root)
    frames = [pd.read_parquet(path) for path in sorted(base.glob("date=*/factor_rank_ic_summary.parquet"))]
    if frames:
        data = pd.concat(frames, ignore_index=True)
        data["month"] = data["trade_date"].str.slice(0, 7)
        data.to_parquet(agg / "factor_rank_ic_by_date.parquet", index=False)
        data.to_csv(agg / "factor_rank_ic_by_date.csv", index=False)
        monthly = (
            data.groupby(
                ["month", "universe", "neutralization", "rank_ic_method", "label_family", "horizon_bars", "bundle", "feature"],
                dropna=False,
            )
            .agg(
                dates=("trade_date", "nunique"),
                mean_rank_ic=("rank_ic_mean", "mean"),
                median_rank_ic=("rank_ic_mean", "median"),
                std_daily_rank_ic=("rank_ic_mean", "std"),
                mean_positive_ratio=("rank_ic_positive_ratio", "mean"),
            )
            .reset_index()
        )
        monthly.to_parquet(agg / "factor_rank_ic_monthly_summary.parquet", index=False)
        monthly.to_csv(agg / "factor_rank_ic_monthly_summary.csv", index=False)
        grouped = (
            data.groupby(
                ["universe", "neutralization", "rank_ic_method", "label_family", "horizon_bars", "bundle", "feature"],
                dropna=False,
            )
            .agg(
                dates=("trade_date", "nunique"),
                total_ic_minutes=("ic_minutes", "sum"),
                total_ic_count=("ic_count", "sum"),
                mean_rank_ic=("rank_ic_mean", "mean"),
                median_rank_ic=("rank_ic_mean", "median"),
                std_daily_rank_ic=("rank_ic_mean", "std"),
                mean_positive_ratio=("rank_ic_positive_ratio", "mean"),
                mean_coverage=("coverage", "mean"),
                mean_label_non_null=("label_non_null", "mean"),
            )
            .reset_index()
            .sort_values(
                ["universe", "neutralization", "rank_ic_method", "label_family", "horizon_bars", "mean_rank_ic"],
                ascending=[True, True, True, True, True, False],
            )
        )
        hac_rows = []
        for keys, group in data.groupby(
            ["universe", "neutralization", "rank_ic_method", "label_family", "horizon_bars", "bundle", "feature"],
            sort=False,
            dropna=False,
        ):
            t_value, p_value, n = _hac_tstat(group.sort_values("trade_date")["rank_ic_mean"])
            hac_rows.append((*keys, t_value, p_value, n))
        hac = pd.DataFrame(
            hac_rows,
            columns=[
                "universe",
                "neutralization",
                "rank_ic_method",
                "label_family",
                "horizon_bars",
                "bundle",
                "feature",
                "hac_tstat_lag5",
                "hac_pvalue",
                "hac_n",
            ],
        )
        grouped = grouped.merge(
            hac,
            on=["universe", "neutralization", "rank_ic_method", "label_family", "horizon_bars", "bundle", "feature"],
            how="left",
        )
        grouped["fdr_qvalue"] = grouped.groupby(
            ["universe", "neutralization", "rank_ic_method", "label_family", "horizon_bars"], dropna=False
        )["hac_pvalue"].transform(_benjamini_hochberg)
        grouped.to_parquet(agg / "factor_rank_ic_overall_summary.parquet", index=False)
        grouped.to_csv(agg / "factor_rank_ic_overall_summary.csv", index=False)
        top = grouped.groupby(["universe", "neutralization", "rank_ic_method", "label_family", "horizon_bars"], group_keys=False).head(30)
        top.to_csv(agg / "top30_by_label_horizon_neutralization.csv", index=False)
        bootstrap_targets = grouped.reindex(grouped["mean_rank_ic"].abs().sort_values(ascending=False).index).head(1000)
        boot_rows = []
        key_cols = ["universe", "neutralization", "rank_ic_method", "label_family", "horizon_bars", "bundle", "feature"]
        indexed = data.set_index(key_cols, drop=False)
        for _, target in bootstrap_targets.iterrows():
            key = tuple(target[col] for col in key_cols)
            try:
                series = indexed.loc[key, "rank_ic_mean"]
            except KeyError:
                continue
            if not isinstance(series, pd.Series):
                series = pd.Series([series])
            boot_rows.append({**{col: target[col] for col in key_cols}, **_block_bootstrap_mean(series)})
        pd.DataFrame(boot_rows).to_csv(agg / "block_bootstrap_top1000.csv", index=False)

        bundle_steps = [
            ("traditional", {"TRAD"}),
            ("traditional_plus_minute_nvg", {"TRAD", "MINUTE_NVG"}),
            ("traditional_plus_minute_trade_nvg", {"TRAD", "MINUTE_NVG", "TRADE_NVG"}),
            ("traditional_plus_minute_trade_hawkes", {"TRAD", "MINUTE_NVG", "TRADE_NVG", "HAWKES_LITE", "HAWKES_DERIVED"}),
        ]
        inc_rows = []
        for keys, group in grouped.groupby(["universe", "neutralization", "rank_ic_method", "label_family", "horizon_bars"], dropna=False):
            for step_name, bundles in bundle_steps:
                subset = group[group["bundle"].isin(bundles)]
                if subset.empty:
                    continue
                best = subset.iloc[subset["mean_rank_ic"].abs().to_numpy().argmax()]
                inc_rows.append(
                    {
                        "universe": keys[0],
                        "neutralization": keys[1],
                        "rank_ic_method": keys[2],
                        "label_family": keys[3],
                        "horizon_bars": keys[4],
                        "step": step_name,
                        "bundle_count": len(bundles),
                        "feature_count": int(subset["feature"].nunique()),
                        "mean_abs_rank_ic": float(subset["mean_rank_ic"].abs().mean()),
                        "best_feature": best["feature"],
                        "best_bundle": best["bundle"],
                        "best_mean_rank_ic": float(best["mean_rank_ic"]),
                        "best_hac_tstat_lag5": float(best["hac_tstat_lag5"]) if np.isfinite(best["hac_tstat_lag5"]) else math.nan,
                        "best_fdr_qvalue": float(best["fdr_qvalue"]) if np.isfinite(best["fdr_qvalue"]) else math.nan,
                    }
                )
        pd.DataFrame(inc_rows).to_csv(agg / "incremental_bundle_screen.csv", index=False)

    registry_frames = [pd.read_parquet(path) for path in sorted(base.glob("date=*/feature_registry.parquet"))]
    if registry_frames:
        registry = pd.concat(registry_frames, ignore_index=True)
        registry.to_parquet(agg / "feature_registry_by_date.parquet", index=False)
        registry_summary = (
            registry.groupby(["feature", "bundle", "evaluated", "exclusion_reason"], dropna=False)
            .agg(
                dates=("trade_date", "nunique"),
                mean_non_null_rate=("non_null_rate", "mean"),
                median_unique_count=("unique_count", "median"),
                mean_variance=("variance", "mean"),
            )
            .reset_index()
            .sort_values(["evaluated", "bundle", "feature"], ascending=[False, True, True])
        )
        registry_summary.to_csv(agg / "feature_registry_summary.csv", index=False)

    decile_frames = [pd.read_parquet(path) for path in sorted(base.glob("date=*/decile_curves.parquet"))]
    if decile_frames:
        deciles = pd.concat(decile_frames, ignore_index=True)
        deciles.to_parquet(agg / "decile_curves_by_date.parquet", index=False)
        daily_deciles = (
            deciles.groupby(
                [
                    "trade_date",
                    "universe",
                    "feature",
                    "bundle",
                    "label_family",
                    "horizon_bars",
                    "variant",
                    "rebalance",
                    "decile",
                ],
                dropna=False,
            )
            .agg(
                mean_label=("mean_label", "mean"),
                count=("count", "sum"),
                mean_log_adv20=("mean_log_adv20", "mean"),
                mean_price=("mean_price", "mean"),
                mean_trade_count=("mean_trade_count", "mean"),
            )
            .reset_index()
        )
        daily_deciles.to_csv(agg / "decile_curves_by_date.csv", index=False)
        overall_deciles = (
            daily_deciles.groupby(
                ["universe", "feature", "bundle", "label_family", "horizon_bars", "variant", "rebalance", "decile"],
                dropna=False,
            )
            .agg(
                dates=("trade_date", "nunique"),
                mean_label=("mean_label", "mean"),
                total_count=("count", "sum"),
                mean_log_adv20=("mean_log_adv20", "mean"),
                mean_price=("mean_price", "mean"),
                mean_trade_count=("mean_trade_count", "mean"),
            )
            .reset_index()
        )
        overall_deciles.to_parquet(agg / "decile_curves_overall.parquet", index=False)
        overall_deciles.to_csv(agg / "decile_curves_overall.csv", index=False)

    portfolio_frames = [pd.read_parquet(path) for path in sorted(base.glob("date=*/staggered_portfolio_proxy.parquet"))]
    if portfolio_frames:
        portfolio = pd.concat(portfolio_frames, ignore_index=True)
        portfolio.to_parquet(agg / "staggered_portfolio_proxy_by_cohort.parquet", index=False)
        portfolio_daily = (
            portfolio.groupby(["trade_date", "universe", "feature", "bundle", "label_family", "horizon_bars"], dropna=False)
            .agg(
                cohorts=("net_return", "count"),
                gross_return_mean=("gross_return", "mean"),
                net_return_mean=("net_return", "mean"),
                turnover_mean=("turnover", "mean"),
                long_count_mean=("long_count", "mean"),
                short_count_mean=("short_count", "mean"),
            )
            .reset_index()
        )
        portfolio_daily.to_csv(agg / "staggered_portfolio_proxy_by_date.csv", index=False)
        portfolio_overall = (
            portfolio_daily.groupby(["universe", "feature", "bundle", "label_family", "horizon_bars"], dropna=False)
            .agg(
                dates=("trade_date", "nunique"),
                cohorts=("cohorts", "sum"),
                gross_return_mean=("gross_return_mean", "mean"),
                net_return_mean=("net_return_mean", "mean"),
                turnover_mean=("turnover_mean", "mean"),
                positive_net_day_ratio=("net_return_mean", lambda s: float((s > 0).mean())),
            )
            .reset_index()
            .sort_values(["label_family", "horizon_bars", "net_return_mean"], ascending=[True, True, False])
        )
        portfolio_overall.to_parquet(agg / "staggered_portfolio_proxy_overall.parquet", index=False)
        portfolio_overall.to_csv(agg / "staggered_portfolio_proxy_overall.csv", index=False)

    report = [
        "# NFF v2.1 neutralized research report",
        "",
        f"- Finished UTC: {utc_now()}",
        "- RankIC method: v1-compatible minute-level cross-sectional Spearman IC, then daily mean over minutes.",
        "- Also emitted pooled cross-sectional RankIC in the same run for v2 comparability.",
        "- Neutralization: per-minute OLS residualization against log price, previous 20d ADV, same-minute dollar volume, active-second ratio, trade count, realized vol, and an intraday beta proxy.",
        "- Alpha feature policy: representative semantic features only; readiness/warmup/coverage/activity/stale/trade-count axes are masks or controls, not alpha columns.",
        "- Tradability: liquid-common universe requires price >= 5, previous 20d ADV top 1000, 20 lookback days, real entry/exit volume, trade_count >= 1, and active-second/stale filters when available.",
        "- Labels: next-minute open/vwap/close entry-to-future-exit return labels plus liquidity deterioration, realized volatility, jump-tail event, and execution-cost proxy.",
        "- Portfolio proxy: 15-minute NY-session same-sleeve top/bottom decile long-short cohorts with turnover and simple cost columns.",
        "- Sector neutralization: not applied because no local sector reference file was found in the warehouse scan; this is recorded in each date meta.",
        "- July should be read as a stress slice, not true OOS.",
    ]
    (agg / "final_report_v2_1.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def resource_snapshot(out_root: Path) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.2)
    d_usage = shutil.disk_usage(str(out_root.anchor or out_root.drive + "\\"))
    return {
        "cpu_percent": round(float(cpu), 2),
        "memory_percent": round(float(memory.percent), 2),
        "memory_available_gb": round(memory.available / 1024**3, 2),
        "disk_free_gb": round(d_usage.free / 1024**3, 2),
    }


def worker_command(args: argparse.Namespace, out_root: Path, controls_path: Path, trade_date: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--run-id",
        args.run_id,
        "--out-root",
        str(out_root),
        "--controls-path",
        str(controls_path),
        "--worker-date",
        trade_date,
        "--min-cs-n",
        str(args.min_cs_n),
        "--horizons",
        *[str(value) for value in args.horizons],
        "--contract-hash",
        args.contract_hash,
        "--rebalance-minutes",
        str(args.rebalance_minutes),
        "--cost-bps-per-turnover",
        str(args.cost_bps_per_turnover),
    ]
    if args.config:
        command.extend(["--config", args.config])
    if args.allow_mixed_contracts:
        command.append("--allow-mixed-contracts")
    return command


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    label_counts = result.get("label_non_null", {})
    return {
        "trade_date": result.get("trade_date"),
        "status": result.get("status", "completed"),
        "rows": result.get("rows"),
        "analysis_feature_count": result.get("analysis_feature_count"),
        "summary_rows": result.get("summary_rows"),
        "decile_rows": result.get("decile_rows"),
        "label_non_null_min": min(label_counts.values()) if isinstance(label_counts, dict) and label_counts else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--run-id")
    parser.add_argument("--out-root")
    parser.add_argument("--controls-path")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--start-date", default="2026-01-02")
    parser.add_argument("--end-date", default="2026-07-22")
    parser.add_argument("--build-controls-only", action="store_true")
    parser.add_argument("--raw-workers", type=int, default=8)
    parser.add_argument("--raw-lookback-days", type=int, default=25)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-date")
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--max-parallel", type=int, default=28)
    parser.add_argument("--min-parallel", type=int, default=4)
    parser.add_argument("--target-cpu", type=float, default=90.0)
    parser.add_argument("--memory-high-water", type=float, default=88.0)
    parser.add_argument("--memory-min-available-gb", type=float, default=16.0)
    parser.add_argument("--disk-free-floor-gb", type=float, default=50.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--launch-batch-size", type=int, default=4)
    parser.add_argument("--min-cs-n", type=int, default=30)
    parser.add_argument("--contract-hash")
    parser.add_argument("--allow-mixed-contracts", action="store_true")
    parser.add_argument("--rebalance-minutes", type=int, default=15)
    parser.add_argument("--cost-bps-per-turnover", type=float, default=1.0)
    args = parser.parse_args()

    explicit_flags = _cli_flags(sys.argv[1:])
    config = load_yaml_config(args.config)
    apply_config_globals(config)
    effective_config = apply_config_args(args, config, explicit_flags)
    out_root = Path(args.out_root) if args.out_root else RESEARCH_ROOT / "runs" / f"v2_1_neutralized_{args.run_id}"
    out_root.mkdir(parents=True, exist_ok=True)
    controls_path = Path(args.controls_path) if args.controls_path else build_daily_controls(
        args.start_date,
        args.end_date,
        CONTROLS_ROOT,
        raw_workers=args.raw_workers,
        raw_lookback_days=args.raw_lookback_days,
    )

    if args.build_controls_only:
        print(json.dumps({"controls_path": str(controls_path)}, indent=2))
        return 0

    effective_config["paths"]["controls_path"] = str(controls_path)
    if args.worker and args.contract_hash:
        existing_hash = read_run_contract_hash(out_root)
        if existing_hash and existing_hash != args.contract_hash:
            raise RuntimeError(f"worker contract hash {args.contract_hash} does not match existing run contract {existing_hash}")
        contract_hash = args.contract_hash
    else:
        _, contract_hash = build_run_contract(out_root, controls_path, effective_config)
        args.contract_hash = args.contract_hash or contract_hash
        if args.contract_hash != contract_hash:
            raise RuntimeError(f"--contract-hash {args.contract_hash} does not match current run contract {contract_hash}")

    if args.worker:
        if not args.worker_date:
            raise ValueError("--worker requires --worker-date")
        print(
            json.dumps(
                run_date(
                    args.worker_date,
                    args.horizons,
                    out_root,
                    controls_path,
                    args.min_cs_n,
                    args.contract_hash,
                    args.allow_mixed_contracts,
                    args.rebalance_minutes,
                    args.cost_bps_per_turnover,
                ),
                indent=2,
                default=str,
            )
        )
        return 0

    status_path = out_root / "status.json"
    resource_log_path = out_root / "resource_samples.ndjson"
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
        controls_path=str(controls_path),
        started_utc=utc_now(),
        stage="neutralized_factor_diagnostics",
        status="running",
        total_units=total,
        completed_units=completed,
        parallel=current_parallel,
        running_workers=0,
        resources=resource_snapshot(out_root),
        horizons=args.horizons,
        label_families=list(LABEL_FAMILIES),
        run_contract_hash=contract_hash,
        effective_config=str(out_root / "effective_config.json"),
    )

    while pending or running:
        resources = resource_snapshot(out_root)
        now = time.time()
        if now - last_tune >= 60:
            last_tune = now
            if resources["disk_free_gb"] < args.disk_free_floor_gb:
                failure = {
                    "reason": "disk_free_floor_breached",
                    "disk_free_gb": resources["disk_free_gb"],
                    "disk_free_floor_gb": args.disk_free_floor_gb,
                    "pending_units": len(pending),
                    "running_workers": len(running),
                }
                fail_dir = out_root / "failures"
                fail_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_json(fail_dir / "scheduler_disk_floor.json", failure)
                update_status(status_path, status="blocked_disk_free_floor", stage="paused", last_failure=failure, resources=resources)
                return 3
            elif resources["memory_percent"] >= args.memory_high_water or resources["memory_available_gb"] <= args.memory_min_available_gb:
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
            launch_resources = resource_snapshot(out_root)
            if (
                launch_resources["memory_percent"] >= args.memory_high_water
                or launch_resources["memory_available_gb"] <= args.memory_min_available_gb
            ):
                update_status(
                    status_path,
                    parallel=current_parallel,
                    running_workers=len(running),
                    pending_units=len(pending),
                    resources=launch_resources,
                    admission_paused_reason="memory_guard",
                )
                break
            trade_date = pending.pop(0)
            out_dir, summary_path, success = unit_paths(out_root, trade_date)
            meta_path = out_dir / "meta.json"
            if success.exists() and summary_path.exists() and meta_path.exists():
                result = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"trade_date": trade_date}
                if result.get("run_contract_hash") != contract_hash:
                    attempts[trade_date] = attempts.get(trade_date, 0)
                else:
                    completed += 1
                    update_status(status_path, last_unit=compact_result({**result, "status": "skipped"}), completed_units=completed)
                    continue
            if resources["disk_free_gb"] < args.disk_free_floor_gb:
                failure = {
                    "reason": "disk_free_floor_breached",
                    "disk_free_gb": resources["disk_free_gb"],
                    "disk_free_floor_gb": args.disk_free_floor_gb,
                    "pending_units": len(pending) + 1,
                    "running_workers": len(running),
                }
                pending.insert(0, trade_date)
                fail_dir = out_root / "failures"
                fail_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_json(fail_dir / "scheduler_disk_floor.json", failure)
                update_status(status_path, status="blocked_disk_free_floor", stage="paused", last_failure=failure, resources=resources)
                return 3
            attempts[trade_date] = attempts.get(trade_date, 0) + 1
            attempt = attempts[trade_date]
            log_dir = out_root / "worker_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = log_dir / f"date={trade_date}_attempt={attempt}.out.log"
            stderr_path = log_dir / f"date={trade_date}_attempt={attempt}.err.log"
            out = stdout_path.open("w", encoding="utf-8")
            err = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(worker_command(args, out_root, controls_path, trade_date), stdout=out, stderr=err)
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
                result = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"trade_date": trade_date}
                update_status(
                    status_path,
                    last_unit=compact_result(result),
                    completed_units=completed,
                    total_units=total,
                    last_unit_seconds=elapsed,
                    parallel=current_parallel,
                    running_workers=max(0, len(running) - 1),
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
                update_status(status_path, last_failure=failure, completed_units=completed, failure_count=len(failures), resources=resources)
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
                "stage": "neutralized_factor_diagnostics",
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
        aggregate_dir=str(out_root / "02_neutralized_factor_diagnostics" / "_aggregate"),
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
