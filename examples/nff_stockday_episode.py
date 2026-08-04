from __future__ import annotations

"""Build resumable, point-in-time-safe stock-day episode manifests from NFF."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlib.contrib.data.nff_episode import NFFStockDayEpisodeFactory, contract_hash


def _read_config(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Episode config must contain a YAML mapping")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _factory(config: Mapping[str, Any]) -> NFFStockDayEpisodeFactory:
    factory_config = dict(config.get("episode_factory") or {})
    if not factory_config:
        raise ValueError("Config requires episode_factory")
    return NFFStockDayEpisodeFactory(**factory_config)


def _resolved_dates(factory: NFFStockDayEpisodeFactory, config: Mapping[str, Any]) -> list[str]:
    available = factory.available_dates()
    date_config = dict(config.get("dates") or {})
    start = str(date_config.get("start") or available[0]) if available else None
    end = str(date_config.get("end") or available[-1]) if available else None
    if start is None or end is None:
        return []
    return [value for value in available if start <= value <= end]


def _date_root(run_root: Path, trade_date: str) -> Path:
    return run_root / "dates" / f"date={trade_date}"


def _existing_complete(run_root: Path, trade_date: str, expected_hash: str) -> bool:
    meta_path = _date_root(run_root, trade_date) / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return meta.get("status") == "complete" and meta.get("contract_hash") == expected_hash


def _date_worker(
    config: Dict[str, Any],
    trade_date: str,
    run_root_text: str,
    expected_hash: str,
    force: bool,
) -> Dict[str, Any]:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    run_root = Path(run_root_text)
    date_root = _date_root(run_root, trade_date)
    meta_path = date_root / "meta.json"
    if _existing_complete(run_root, trade_date, expected_hash) and not force:
        return {"trade_date": trade_date, "status": "reused"}
    if meta_path.exists() and not force:
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        existing_hash = existing.get("contract_hash")
        if existing_hash and existing_hash != expected_hash:
            raise RuntimeError(
                f"date={trade_date} checkpoint contract mismatch: {existing_hash} != {expected_hash}"
            )

    started = time.perf_counter()
    factory = _factory(config)
    result = factory.load_date(trade_date)
    manifest = result.manifest_frame()
    rejections = result.rejection_frame()
    date_root.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(manifest, date_root / "manifest.parquet")
    _atomic_parquet(rejections, date_root / "rejections.parquet")
    meta = {
        "status": "complete",
        "trade_date": trade_date,
        "contract_hash": expected_hash,
        "episode_count": int(len(manifest)),
        "rejection_count": int(len(rejections)),
        "feature_names": list(result.feature_names),
        "target_names": list(result.target_names),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "load_report": factory.last_load_report,
    }
    _atomic_json(meta_path, meta)
    return {
        "trade_date": trade_date,
        "status": "completed",
        "episode_count": int(len(manifest)),
        "rejection_count": int(len(rejections)),
        "elapsed_seconds": meta["elapsed_seconds"],
    }


def _aggregate(run_root: Path, dates: Iterable[str], expected_hash: str) -> Dict[str, Any]:
    manifests = []
    rejections = []
    daily_rows = []
    for trade_date in dates:
        root = _date_root(run_root, trade_date)
        meta_path = root / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "complete" or meta.get("contract_hash") != expected_hash:
            continue
        manifest_path = root / "manifest.parquet"
        rejection_path = root / "rejections.parquet"
        if manifest_path.exists():
            frame = pd.read_parquet(manifest_path)
            if not frame.empty:
                manifests.append(frame)
        if rejection_path.exists():
            frame = pd.read_parquet(rejection_path)
            if not frame.empty:
                rejections.append(frame)
        daily_rows.append(
            {
                "trade_date": trade_date,
                "episode_count": int(meta.get("episode_count", 0)),
                "rejection_count": int(meta.get("rejection_count", 0)),
                "elapsed_seconds": float(meta.get("elapsed_seconds", 0.0)),
            }
        )

    aggregate = run_root / "aggregate"
    manifest = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame()
    rejection = pd.concat(rejections, ignore_index=True) if rejections else pd.DataFrame()
    daily = pd.DataFrame(daily_rows).sort_values("trade_date") if daily_rows else pd.DataFrame()
    _atomic_parquet(manifest, aggregate / "episode_manifest.parquet")
    _atomic_parquet(rejection, aggregate / "episode_rejections.parquet")
    _atomic_parquet(daily, aggregate / "episode_daily_summary.parquet")

    reason_counts = (
        rejection.groupby("reason", dropna=False).size().sort_values(ascending=False).to_dict()
        if not rejection.empty and "reason" in rejection
        else {}
    )
    summary = {
        "contract_hash": expected_hash,
        "completed_dates": int(len(daily)),
        "episode_count": int(len(manifest)),
        "unique_symbols": int(manifest["symbol"].nunique()) if not manifest.empty else 0,
        "rejection_count": int(len(rejection)),
        "rejection_reasons": {str(key): int(value) for key, value in reason_counts.items()},
        "support_coverage_mean": float(manifest["support_coverage"].mean()) if not manifest.empty else None,
        "query_coverage_mean": float(manifest["query_coverage_mean"].mean()) if not manifest.empty else None,
        "target_coverage_mean": float(manifest["target_coverage"].mean()) if not manifest.empty else None,
    }
    _atomic_json(aggregate / "summary.json", summary)
    report = [
        "# NFF stock-day episode manifest",
        "",
        f"- Contract: `{expected_hash}`",
        f"- Completed dates: {summary['completed_dates']}",
        f"- Accepted episodes: {summary['episode_count']}",
        f"- Unique symbols: {summary['unique_symbols']}",
        f"- Rejections: {summary['rejection_count']}",
        f"- Mean support coverage: {summary['support_coverage_mean']}",
        f"- Mean query coverage: {summary['query_coverage_mean']}",
        f"- Mean target coverage: {summary['target_coverage_mean']}",
        "",
        "This manifest is NFF-only. GFF and GAL are not read or required.",
    ]
    aggregate.mkdir(parents=True, exist_ok=True)
    (aggregate / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def command_inspect(config: Dict[str, Any]) -> int:
    factory = _factory(config)
    dates = _resolved_dates(factory, config)
    print(
        json.dumps(
            {
                "contract_hash": factory.contract_hash,
                "available_dates": len(factory.available_dates()),
                "selected_dates": dates,
                "policy": asdict(factory.policy),
                "targets": [asdict(target) for target in factory.targets],
                "sources": [asdict(source) for source in factory.loader.sources],
            },
            indent=2,
            default=str,
        )
    )
    return 0


def command_build(config: Dict[str, Any], run_root: Path, workers: int, force: bool) -> int:
    factory = _factory(config)
    dates = _resolved_dates(factory, config)
    if not dates:
        raise RuntimeError("No common NFF dates are available for the selected sources")
    run_contract = {
        "version": "nff_stockday_episode_v1",
        "factory": factory.contract,
        "dates": dates,
    }
    expected_hash = contract_hash(run_contract)
    run_root.mkdir(parents=True, exist_ok=True)
    contract_path = run_root / "run_contract.json"
    if contract_path.exists() and not force:
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing.get("contract_hash") != expected_hash:
            raise RuntimeError("Existing run_root has a different episode contract; use a new run ID or --force")
    _atomic_json(contract_path, {"contract_hash": expected_hash, **run_contract})

    pending = [value for value in dates if force or not _existing_complete(run_root, value, expected_hash)]
    status = {
        "status": "running",
        "contract_hash": expected_hash,
        "total_dates": len(dates),
        "pending_dates": len(pending),
        "completed_dates": len(dates) - len(pending),
        "workers": workers,
    }
    _atomic_json(run_root / "status.json", status)
    failures = []
    completed = status["completed_dates"]
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(_date_worker, config, trade_date, str(run_root), expected_hash, force): trade_date
                for trade_date in pending
            }
            for future in as_completed(futures):
                trade_date = futures[future]
                try:
                    result = future.result()
                    completed += 1
                    print(json.dumps(result, default=str))
                except Exception as exc:
                    failure = {"trade_date": trade_date, "error": repr(exc)}
                    failures.append(failure)
                    print(json.dumps({"status": "failed", **failure}), file=sys.stderr)
                _atomic_json(
                    run_root / "status.json",
                    {
                        **status,
                        "completed_dates": completed,
                        "pending_dates": len(dates) - completed,
                        "failure_count": len(failures),
                        "last_failure": failures[-1] if failures else None,
                    },
                )

    summary = _aggregate(run_root, dates, expected_hash)
    final_status = {
        **status,
        **summary,
        "status": "complete" if not failures else "complete_with_failures",
        "failures": failures,
    }
    _atomic_json(run_root / "status.json", final_status)
    print(json.dumps(final_status, indent=2, default=str))
    return 0 if not failures else 2


def command_inspect_episode(
    config: Dict[str, Any], symbol: str, trade_date: str, output: Optional[Path]
) -> int:
    factory = _factory(config)
    result = factory.load_date(trade_date, instruments=[symbol])
    matches = [episode for episode in result.episodes if episode.symbol.upper() == symbol.upper()]
    if not matches:
        print(json.dumps({"symbol": symbol, "trade_date": trade_date, "rejections": result.rejections}, indent=2))
        return 1
    episode = matches[0]
    payload = {
        "manifest": episode.manifest_row(),
        "feature_names": list(episode.feature_names),
        "target_names": list(episode.target_names),
        "audit": episode.audit,
        "query_times": [pd.Timestamp(value).isoformat() for value in episode.query_times_ns],
        "targets": episode.targets.tolist(),
        "target_observed": episode.target_observed.tolist(),
        "load_report": factory.last_load_report,
    }
    print(json.dumps(payload, indent=2, default=str))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            support_times_ns=episode.support_times_ns,
            support_x=episode.support_x,
            support_observed=episode.support_observed,
            query_times_ns=episode.query_times_ns,
            query_x=episode.query_x,
            query_observed=episode.query_observed,
            targets=episode.targets,
            target_observed=episode.target_observed,
            feature_names=np.asarray(episode.feature_names),
            target_names=np.asarray(episode.target_names),
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["inspect", "build-manifest", "inspect-episode"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--symbol")
    parser.add_argument("--date")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = _read_config(Path(args.config).expanduser().resolve())
    if args.command == "inspect":
        return command_inspect(config)
    if args.command == "build-manifest":
        run_root = Path(args.run_root or config.get("output", {}).get("episode_run_root", "episode_runs/pilot"))
        workers = args.workers or int(config.get("run", {}).get("workers", 4))
        return command_build(config, run_root.expanduser().resolve(), workers, args.force)
    if not args.symbol or not args.date:
        parser.error("inspect-episode requires --symbol and --date")
    return command_inspect_episode(
        config,
        args.symbol,
        args.date,
        Path(args.output).expanduser().resolve() if args.output else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
