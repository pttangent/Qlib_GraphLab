from __future__ import annotations

"""Production hardening for the v2.8 staged pipeline."""

import gc
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import pandas as pd
import psutil


def _admission_cap(
    spec: Any,
    stage: str,
    *,
    usable_gb: float,
    minimum_launch_headroom_gb: float,
    config: Mapping[str, Any],
) -> int:
    """Return process queue width while leaving peak memory to its own gate."""
    if usable_gb < minimum_launch_headroom_gb:
        return 0
    materialize_internal = config.get("pipeline", {}).get(
        "materialize_internal", {}
    )
    if (
        stage == "materialize"
        and isinstance(materialize_internal, Mapping)
        and int(materialize_internal.get("peak_slots", 0)) > 0
        and bool(materialize_internal.get("stream_factor_blocks", False))
    ):
        # Date workers spend most of their lifetime waiting at the file-based
        # peak lease. Their queue width is therefore independent of the
        # guarded wide-merge RSS estimate; peak_slots is the memory control.
        return int(spec.max_workers)
    memory_cap = int(max(0.0, usable_gb) // spec.estimated_worker_gb)
    return min(spec.max_workers, memory_cap)


def install(P: Any) -> None:
    original_load_selected = P._load_selected_factors
    original_detailed = P.detailed_date
    original_portfolio = P.portfolio_date

    def load_selected_ordered(
        config: Mapping[str, Any],
        trade_date: str,
        selected: Any,
        expected_index: pd.Index,
    ) -> pd.DataFrame:
        ordered = list(dict.fromkeys(str(value) for value in selected))
        result = original_load_selected(config, trade_date, ordered, expected_index)
        missing = [feature for feature in ordered if feature not in result]
        if missing:
            raise RuntimeError(
                f"selected factor load lost {len(missing)} columns for {trade_date}: {missing[:20]}"
            )
        return result.loc[:, ordered]

    def with_atomic_context(fn: Any, stage_name: str):
        def wrapped(config: dict[str, Any], trade_date: str) -> dict[str, Any]:
            if P.C.CTX is not None:
                return fn(config, trade_date)
            atomic = config.get("atomic", {})
            P.C.CTX = P.C.Context(
                trade_date=trade_date,
                out_root=P._run_root(config),
                contract_hash=P._contract_hash(config),
                factor_block_size=int(atomic.get("factor_block_size", 8)),
                intra_workers=int(atomic.get("intra_date_workers", 4)),
            )
            P.C.CTX.root.mkdir(parents=True, exist_ok=True)
            P.C._event(stage_name, "running")
            started = time.perf_counter()
            try:
                result = fn(config, trade_date)
                P.C._event(
                    stage_name,
                    "complete",
                    elapsed_seconds=time.perf_counter() - started,
                )
                return result
            except BaseException as exc:
                P.C._event(
                    stage_name,
                    "failed",
                    elapsed_seconds=time.perf_counter() - started,
                    error=repr(exc),
                )
                raise
            finally:
                P.C.FUTURE_CACHE.clear()
                P.C.FUTURE_WIDE_CACHE.clear()
                P.C.CTX = None
                gc.collect()

        return wrapped

    def run_date_stage_guarded(
        config_path: Path,
        config: dict[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        spec = P._stage_specs(config)[stage]
        runtime_dates = config.get("pipeline", {}).get("_runtime_dates")
        dates = (
            [str(value) for value in runtime_dates]
            if isinstance(runtime_dates, list)
            else P._dates(config)
        )
        if stage == "portfolio" and not bool(
            config.get("selection", {}).get("allow_in_sample_portfolio", False)
        ):
            _, meta = P._load_candidates(config, "portfolio")
            dates = [
                date
                for date in dates
                if date > str(meta["portfolio_eligible_after"])
            ]
        # Every date gets a cheap worker-side contract verification. Existing
        # valid outputs return immediately as skipped. This is necessary because
        # a bare _SUCCESS marker cannot reveal source/config changes to the parent.
        pending = list(dates)
        running: dict[str, dict[str, Any]] = {}
        failures: list[dict[str, Any]] = []
        retries = int(config["run"].get("retries", 3))
        reserve_gb = float(
            config.get("pipeline", {}).get("memory_reserve_gb", 28.0)
        )
        minimum_launch_headroom_gb = float(
            config.get("pipeline", {}).get(
                "minimum_launch_headroom_gb",
                min(spec.estimated_worker_gb, 8.0),
            )
        )
        status_suffix = config.get("pipeline", {}).get("_runtime_status_suffix")
        status_name = f"status_{stage}"
        if status_suffix:
            status_name += f"_{status_suffix}"
        status_path = P._pipeline_root(config) / f"{status_name}.json"
        log_root = P._pipeline_root(config) / "worker_logs" / stage
        log_root.mkdir(parents=True, exist_ok=True)
        attempts: dict[str, int] = {}
        verified = 0

        while pending or running:
            available_gb = psutil.virtual_memory().available / 1024**3
            usable_gb = max(0.0, available_gb - reserve_gb)
            launch_allowed = usable_gb >= minimum_launch_headroom_gb
            cap = _admission_cap(
                spec,
                stage,
                usable_gb=usable_gb,
                minimum_launch_headroom_gb=minimum_launch_headroom_gb,
                config=config,
            )
            while pending and len(running) < cap:
                trade_date = pending.pop(0)
                attempts[trade_date] = attempts.get(trade_date, 0) + 1
                attempt = attempts[trade_date]
                stdout_path = (
                    log_root / f"date={trade_date}_attempt={attempt}.out.log"
                )
                stderr_path = (
                    log_root / f"date={trade_date}_attempt={attempt}.err.log"
                )
                stdout = stdout_path.open("w", encoding="utf-8")
                stderr = stderr_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    P._worker_command(config_path, stage, trade_date),
                    stdout=stdout,
                    stderr=stderr,
                )
                running[trade_date] = {
                    "process": process,
                    "stdout": stdout,
                    "stderr": stderr,
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path,
                    "attempt": attempt,
                }

            finished: list[str] = []
            for trade_date, info in list(running.items()):
                rc = info["process"].poll()
                if rc is None:
                    continue
                info["stdout"].close()
                info["stderr"].close()
                if rc != 0 or not P._stage_success(
                    config, stage, trade_date
                ).exists():
                    error = info["stderr_path"].read_text(
                        encoding="utf-8", errors="replace"
                    )[-4000:]
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
                        P._atomic_json(
                            P._pipeline_root(config)
                            / "failures"
                            / stage
                            / f"date={trade_date}.json",
                            failure,
                        )
                else:
                    verified += 1
                finished.append(trade_date)
            for trade_date in finished:
                running.pop(trade_date, None)

            P._atomic_json(
                status_path,
                {
                    "version": P.VERSION,
                    "stage": stage,
                    "runtime_status_suffix": status_suffix,
                    "pending_contract_checks": len(pending),
                    "running": list(running),
                    "verified_or_completed": verified,
                    "completed_markers_in_scope": sum(
                        P._stage_success(config, stage, date).exists()
                        for date in dates
                    ),
                    "scope_total": len(dates),
                    "failures": len(failures),
                    "max_workers": spec.max_workers,
                    "memory_limited_cap": cap,
                    "admission_paused": bool(pending and cap == 0),
                    "admission_pause_reason": (
                        "memory_headroom"
                        if pending and cap == 0
                        else None
                    ),
                    "memory_available_gb": round(available_gb, 3),
                    "memory_reserve_gb": reserve_gb,
                    "usable_memory_gb": round(usable_gb, 3),
                    "estimated_worker_gb": spec.estimated_worker_gb,
                    "updated_utc": pd.Timestamp.utcnow().isoformat(),
                },
            )
            time.sleep(2 if running or cap > 0 else 10)
        return {
            "stage": stage,
            "status": "partial_success" if failures else "complete",
            "verified_or_completed": verified,
            "scope_total": len(dates),
            "failures": failures,
        }

    P._load_selected_factors = load_selected_ordered
    P.detailed_date = with_atomic_context(
        original_detailed, "pipeline_detailed"
    )
    P.portfolio_date = with_atomic_context(
        original_portfolio, "pipeline_portfolio"
    )
    P._run_date_stage = run_date_stage_guarded
