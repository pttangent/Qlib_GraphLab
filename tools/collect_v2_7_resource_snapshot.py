"""Collect a reproducible resource/stage snapshot for a running v2.7 campaign."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = []
    for row in rows:
        value = row.get("resources", {}).get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": percentile(values, 0.90),
        "max": max(values) if values else None,
        "latest": values[-1] if values else None,
    }


def factor_progress(run_root: Path, expected: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for date_root in sorted((run_root / "atomic_checkpoints").glob("date=*")):
        completed: set[str] = set()
        manifest_count = 0
        for manifest_path in date_root.glob("factors/family=*/window=*/manifest.json"):
            try:
                manifest = read_json(manifest_path)
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("status") != "complete":
                continue
            manifest_count += 1
            for block in manifest.get("blocks", []):
                completed.update(str(column) for column in block.get("columns", []))
        events = read_jsonl(date_root / "stage_events.jsonl")
        result.append(
            {
                "trade_date": date_root.name.removeprefix("date="),
                "factor_completed": min(len(completed), expected),
                "factor_expected": expected,
                "factor_progress_pct": round(100.0 * min(len(completed), expected) / expected, 3) if expected else 0.0,
                "factor_manifest_count": manifest_count,
                "last_stage": events[-1].get("stage") if events else None,
                "last_state": events[-1].get("state") if events else None,
            }
        )
    return result


def stage_timing(benchmark_root: Path) -> dict[str, Any]:
    events = read_jsonl(benchmark_root / "atomic_checkpoints/date=2026-01-02/stage_events.jsonl")
    completed: list[dict[str, Any]] = []
    for event in events:
        if event.get("state") == "complete" and isinstance(event.get("elapsed_seconds"), (int, float)):
            completed.append(
                {
                    "stage": event.get("stage"),
                    "elapsed_seconds": float(event["elapsed_seconds"]),
                    "trade_date": event.get("trade_date"),
                }
            )
    by_stage: dict[str, list[float]] = {}
    for row in completed:
        by_stage.setdefault(str(row["stage"]), []).append(row["elapsed_seconds"])
    summary = {}
    for stage, values in sorted(by_stage.items()):
        summary[stage] = {
            "count": len(values),
            "total_seconds": sum(values),
            "median_seconds": statistics.median(values),
            "p90_seconds": percentile(values, 0.90),
            "max_seconds": max(values),
        }
    return {"source": str(benchmark_root), "completed_stage_events": completed, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--expected-factors", type=int, default=464)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    output_root = (args.output_root or Path("reports/resource_snapshots") / run_root.name).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    status = read_json(run_root / "status.json")
    resource_rows = read_jsonl(run_root / "resource_samples.ndjson")
    tuning_rows = read_jsonl(run_root / "scheduler_tuning.ndjson")
    contract = read_json(run_root / "run_contract.json")
    effective_config = read_json(run_root / "effective_config.json")
    factor_rows = factor_progress(run_root, args.expected_factors)
    stage_report = stage_timing(args.benchmark_root.resolve())

    error_logs = []
    for path in sorted((run_root / "worker_logs").rglob("*.err.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        error_logs.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "has_traceback": "Traceback (most recent call last)" in text,
                "has_error": "Error:" in text or "Exception:" in text,
                "tail": text[-2000:],
            }
        )
    nonempty_errors = [row for row in error_logs if row["bytes"] > 0]
    traceback_errors = [row for row in nonempty_errors if row["has_traceback"] or row["has_error"]]

    latest_tuning = tuning_rows[-1] if tuning_rows else {}
    snapshot = {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": status.get("run_id", run_root.name),
        "run_root": str(run_root),
        "status": status,
        "resource_sample_count": len(resource_rows),
        "resource_summary": {
            key: numeric_summary(resource_rows, key)
            for key in (
                "cpu_percent",
                "memory_percent",
                "memory_available_gb",
                "worker_rss_total_gb",
                "worker_rss_max_gb",
                "disk_free_gb",
            )
        },
        "scheduler_tuning_count": len(tuning_rows),
        "latest_scheduler_tuning": latest_tuning,
        "factor_progress": factor_rows,
        "factor_progress_summary": {
            "dates_seen": len(factor_rows),
            "dates_factor_complete": sum(row["factor_completed"] >= args.expected_factors for row in factor_rows),
            "max_factor_progress_pct": max((row["factor_progress_pct"] for row in factor_rows), default=0.0),
        },
        "stage_timing_benchmark": stage_report,
        "worker_log_audit": {
            "files": error_logs,
            "nonempty_log_count": len(nonempty_errors),
            "traceback_or_exception_count": len(traceback_errors),
            "traceback_or_exception_logs": traceback_errors,
            "warning_only_count": len(nonempty_errors) - len(traceback_errors),
        },
        "contract_hash": contract.get("run_contract_hash"),
        "git_commit": contract.get("git_commit"),
        "effective_config": effective_config,
    }
    (output_root / "resource_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    (output_root / "factor_progress.json").write_text(
        json.dumps(factor_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "stage_timing_benchmark.json").write_text(
        json.dumps(stage_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for filename in (
        "status.json",
        "resource_samples.ndjson",
        "scheduler_tuning.ndjson",
        "run_contract.json",
        "effective_config.json",
        "factor_progress.json",
    ):
        source = run_root / filename
        if source.exists():
            shutil.copy2(source, output_root / f"run_{filename}")
    benchmark_events = args.benchmark_root.resolve() / "atomic_checkpoints/date=2026-01-02/stage_events.jsonl"
    if benchmark_events.exists():
        shutil.copy2(benchmark_events, output_root / "benchmark_stage_events.jsonl")

    lines = [
        f"# v2.7 Resource Snapshot: {run_root.name}",
        "",
        f"Snapshot UTC: `{snapshot['snapshot_utc']}`",
        f"Status: `{status.get('status')}`; stage: `{status.get('stage')}`",
        f"Progress: `{status.get('completed_units', 0)}/{status.get('total_units', 0)}` dates; running `{status.get('running_workers', 0)}`; pending `{status.get('pending_units', 0)}`; failures `{status.get('failure_count', 0)}`.",
        f"Contract: `{contract.get('run_contract_hash')}`; git: `{contract.get('git_commit')}`",
        "",
        "## Resource Summary",
        "",
        "| Metric | Min | Median | P90 | Max | Latest |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "cpu_percent": "CPU %",
        "memory_percent": "Memory %",
        "memory_available_gb": "Available memory GB",
        "worker_rss_total_gb": "Worker RSS total GB",
        "worker_rss_max_gb": "Worker RSS max GB",
        "disk_free_gb": "Disk free GB",
    }
    for key, label in labels.items():
        row = snapshot["resource_summary"][key]
        values = [row[name] for name in ("min", "median", "p90", "max", "latest")]
        rendered = ["-" if value is None else f"{value:.3f}" for value in values]
        lines.append(f"| {label} | " + " | ".join(rendered) + " |")
    lines.extend(
        [
            "",
            "## Scheduling",
            "",
            f"Resource samples: `{len(resource_rows)}`; scheduler tuning events: `{len(tuning_rows)}`.",
            f"Latest tuning: `{json.dumps(latest_tuning, ensure_ascii=False)}`",
            "",
            "## Factor Progress",
            "",
            f"Dates with checkpoints: `{len(factor_rows)}`; dates at `464/464`: `{snapshot['factor_progress_summary']['dates_factor_complete']}`; max factor progress: `{snapshot['factor_progress_summary']['max_factor_progress_pct']:.3f}%`.",
            "",
            "| Date | Factors | Progress | Last stage | State |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in factor_rows:
        lines.append(
            f"| {row['trade_date']} | {row['factor_completed']}/{row['factor_expected']} | {row['factor_progress_pct']:.3f}% | {row['last_stage'] or '-'} | {row['last_state'] or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Benchmark Stage Timing",
            "",
            "| Stage | Count | Total seconds | Median seconds | P90 seconds | Max seconds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for stage, row in stage_report["summary"].items():
        lines.append(
            f"| {stage} | {row['count']} | {row['total_seconds']:.3f} | {row['median_seconds']:.3f} | {row['p90_seconds']:.3f} | {row['max_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Audit Notes",
            "",
            f"Non-empty worker stderr logs: `{len(nonempty_errors)}`; traceback/exception logs: `{len(traceback_errors)}`; warning-only logs: `{len(nonempty_errors) - len(traceback_errors)}`.",
            "This is a live snapshot. It does not assert that the 138-day campaign is complete.",
            "The benchmark stage timing is from the completed one-day run and is included to expose the portfolio_proxy bottleneck.",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
