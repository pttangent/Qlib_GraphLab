from __future__ import annotations

"""Build resumable NFF-only stock identity and market context caches."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlib.contrib.data.nff_context import (
    ContextCacheConfig,
    NFFContextCacheStore,
    build_context_date,
    context_cache_contract,
)
from qlib.contrib.data.nff_episode import NFFStockDayEpisodeFactory
from qlib.contrib.model.nff_generic import NormalizationState


def _read_config(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Config must contain a YAML mapping")
    return value


def _factory(config: Mapping[str, Any]) -> NFFStockDayEpisodeFactory:
    value = dict(config.get("episode_factory") or {})
    if not value:
        raise ValueError("Config requires episode_factory")
    return NFFStockDayEpisodeFactory(**value)


def _normalization(config: Mapping[str, Any]) -> NormalizationState:
    root = Path(config.get("output", {}).get("generic_model_root", "model_runs/nff_generic_pilot"))
    path = root.expanduser().resolve() / "normalization.json"
    if not path.exists():
        raise FileNotFoundError(f"P2 normalization not found: {path}")
    return NormalizationState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _manifest_dates(config: Mapping[str, Any], explicit: Optional[Path]) -> list[str]:
    path = explicit or Path(config.get("output", {}).get("episode_run_root", "episode_runs/pilot")) / "aggregate" / "episode_manifest.parquet"
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"P1 episode manifest not found: {path}")
    frame = pd.read_parquet(path, columns=["trade_date"])
    return sorted(frame["trade_date"].astype(str).unique())


def _cache_config(config: Mapping[str, Any]) -> ContextCacheConfig:
    return ContextCacheConfig(**dict(config.get("context_cache") or {}))


def _cache_root(config: Mapping[str, Any], explicit: Optional[Path]) -> Path:
    return (explicit or Path(config.get("output", {}).get("context_cache_root", "context_cache/pilot"))).expanduser().resolve()


def _build_one(
    config: Mapping[str, Any],
    trade_date: str,
    normalization_dict: Mapping[str, Any],
    cache_root: str,
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    factory = _factory(config)
    normalization = NormalizationState.from_dict(normalization_dict)
    cache_config = _cache_config(config)
    store = NFFContextCacheStore(cache_root, contract)
    result = factory.load_date(trade_date)
    context = build_context_date(
        result.episodes,
        normalization,
        episode_contract_hash=factory.contract_hash,
        config=cache_config,
    )
    context.audit.update(
        {
            "trade_date": trade_date,
            "episode_rejections": int(len(result.rejections)),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    )
    store.write_date(trade_date, context)
    return context.audit


def _aggregate(store: NFFContextCacheStore, dates: list[str]) -> Dict[str, Any]:
    rows = []
    for trade_date in dates:
        meta_path = store.date_root(trade_date) / "meta.json"
        if meta_path.exists():
            rows.append(json.loads(meta_path.read_text(encoding="utf-8")))
    frame = pd.DataFrame(rows)
    aggregate_root = store.root / "aggregate"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    if not frame.empty:
        tmp = aggregate_root / "context_daily_audit.parquet.tmp"
        frame.to_parquet(tmp, index=False)
        tmp.replace(aggregate_root / "context_daily_audit.parquet")
    summary = {
        "status": "complete" if len(rows) == len(dates) else "partial",
        "contract_hash": store.contract_hash,
        "requested_dates": len(dates),
        "completed_dates": len(rows),
        "identity_rows": int(frame.get("identity_rows", pd.Series(dtype=int)).sum()) if not frame.empty else 0,
        "market_rows": int(frame.get("market_rows", pd.Series(dtype=int)).sum()) if not frame.empty else 0,
        "market_valid_rows": int(frame.get("market_valid_rows", pd.Series(dtype=int)).sum()) if not frame.empty else 0,
    }
    path = aggregate_root / "summary.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    report = [
        "# NFF context cache",
        "",
        f"- Status: {summary['status']}",
        f"- Completed dates: {summary['completed_dates']} / {summary['requested_dates']}",
        f"- Identity rows: {summary['identity_rows']}",
        f"- Market query rows: {summary['market_rows']}",
        f"- Valid market rows: {summary['market_valid_rows']}",
        "- Inputs: NFF only; no GFF/GAL access.",
        "- Identity rows are consumed only by later trading dates.",
        "- Market context is exact leave-one-out at each query timestamp.",
    ]
    (aggregate_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def command_inspect(config: Mapping[str, Any], manifest: Optional[Path], root: Optional[Path]) -> int:
    factory = _factory(config)
    normalization = _normalization(config)
    cache_config = _cache_config(config)
    dates = _manifest_dates(config, manifest)
    contract = context_cache_contract(
        episode_contract_hash=factory.contract_hash,
        normalization=normalization,
        config=cache_config,
    )
    payload = {
        "context_cache_root": str(_cache_root(config, root)),
        "dates": {"count": len(dates), "first": dates[0], "last": dates[-1]},
        "episode_contract_hash": factory.contract_hash,
        "context_contract": contract,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def command_build(config: Mapping[str, Any], manifest: Optional[Path], root: Optional[Path], force: bool) -> int:
    factory = _factory(config)
    normalization = _normalization(config)
    cache_config = _cache_config(config)
    dates = _manifest_dates(config, manifest)
    contract = context_cache_contract(
        episode_contract_hash=factory.contract_hash,
        normalization=normalization,
        config=cache_config,
    )
    store = NFFContextCacheStore(_cache_root(config, root), contract)
    store.initialize()
    pending = dates if force else [date for date in dates if not store.is_complete(date)]
    workers = max(1, int(config.get("run", {}).get("context_workers", 4)))
    if workers == 1:
        for trade_date in pending:
            audit = _build_one(config, trade_date, normalization.to_dict(), str(store.root), contract)
            print(json.dumps(audit, default=str))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _build_one,
                    config,
                    trade_date,
                    normalization.to_dict(),
                    str(store.root),
                    contract,
                ): trade_date
                for trade_date in pending
            }
            for future in as_completed(futures):
                trade_date = futures[future]
                try:
                    print(json.dumps(future.result(), default=str))
                except Exception as exc:
                    raise RuntimeError(f"Context cache failed for {trade_date}: {exc}") from exc
    summary = _aggregate(store, dates)
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["inspect", "build", "aggregate"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--cache-root")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = _read_config(Path(args.config).expanduser().resolve())
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else None
    root = Path(args.cache_root).expanduser().resolve() if args.cache_root else None
    if args.command == "inspect":
        return command_inspect(config, manifest, root)
    if args.command == "build":
        return command_build(config, manifest, root, args.force)
    factory = _factory(config)
    normalization = _normalization(config)
    cache_config = _cache_config(config)
    contract = context_cache_contract(
        episode_contract_hash=factory.contract_hash,
        normalization=normalization,
        config=cache_config,
    )
    store = NFFContextCacheStore(_cache_root(config, root), contract)
    store.initialize()
    summary = _aggregate(store, _manifest_dates(config, manifest))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
