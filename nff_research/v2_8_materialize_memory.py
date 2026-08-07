from __future__ import annotations

"""Memory-bounded materialize execution for the v2.8 staged pipeline.

The v2.7 factor math is unchanged.  This layer changes only execution shape:

1. date workers may coexist, but the wide warehouse-load/merge phase is guarded
   by a cross-process slot lease so their short 25-30+ GB peaks do not align;
2. A-K factor blocks are written to their existing atomic checkpoints and then
   released instead of being accumulated and concatenated back into one 464-
   factor DataFrame;
3. resumed family/window checkpoints are validated from Parquet metadata and
   manifests without reading every completed factor block into memory;
4. after the peak merge phase and after each family/window factor group, Python,
   Arrow, and the platform allocator are asked to return unused memory.

The persisted factor values, exact-minute labels, PIT masks, controls, universe
rules, factor manifests, and downstream basic-screen contract are unchanged.
"""

from dataclasses import dataclass
import ctypes
import gc
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class _Lease:
    path: Path
    slot: int
    acquired_at: float
    trade_date: str


_ACTIVE_LEASE: _Lease | None = None
_POST_BOOTSTRAP = False
_ORIGINALS: dict[str, Any] = {}


def _internal_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("pipeline", {}).get("materialize_internal", {})


def _rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def _memory_trim() -> dict[str, float]:
    before = _rss_gb()
    gc.collect()
    try:
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    try:
        if os.name == "nt":
            ctypes.CDLL("msvcrt")._heapmin()
        else:
            libc = ctypes.CDLL(None)
            malloc_trim = getattr(libc, "malloc_trim", None)
            if malloc_trim is not None:
                malloc_trim(0)
    except Exception:
        pass
    gc.collect()
    return {"rss_before_gb": before, "rss_after_gb": _rss_gb()}


def _emit(P: Any, stage: str, state: str, **extra: Any) -> None:
    try:
        P.C._event(stage, state, pid=os.getpid(), rss_gb=round(_rss_gb(), 3), **extra)
    except Exception:
        pass


def _gate_root(P: Any, config: Mapping[str, Any]) -> Path:
    root = P._pipeline_root(config) / "resource_gates" / "materialize_peak"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _lease_payload(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _cleanup_stale_leases(P: Any, config: Mapping[str, Any]) -> None:
    root = _gate_root(P, config)
    stale_seconds = float(_internal_config(config).get("stale_lease_seconds", 3600.0))
    now = time.time()
    for path in root.glob("slot=*.lease"):
        payload = _lease_payload(path)
        pid = int(payload.get("pid", -1))
        age = max(0.0, now - float(payload.get("created_epoch", 0.0)))
        if pid > 0 and psutil.pid_exists(pid) and age <= stale_seconds:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _acquire_peak_lease(P: Any, config: Mapping[str, Any], trade_date: str) -> _Lease:
    global _ACTIVE_LEASE
    if _ACTIVE_LEASE is not None:
        return _ACTIVE_LEASE
    settings = _internal_config(config)
    slots = max(1, int(settings.get("peak_slots", 1)))
    min_available_gb = float(settings.get("peak_entry_min_available_gb", 48.0))
    poll_seconds = max(0.1, float(settings.get("gate_poll_seconds", 0.5)))
    root = _gate_root(P, config)
    started = time.perf_counter()
    _emit(P, "materialize_peak_gate", "waiting", trade_date=trade_date, peak_slots=slots)
    while True:
        _cleanup_stale_leases(P, config)
        available_gb = psutil.virtual_memory().available / 1024**3
        if available_gb >= min_available_gb:
            for slot in range(slots):
                path = root / f"slot={slot:02d}.lease"
                try:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
                payload = {
                    "pid": os.getpid(),
                    "trade_date": trade_date,
                    "slot": slot,
                    "created_epoch": time.time(),
                    "created_utc": pd.Timestamp.utcnow().isoformat(),
                }
                try:
                    os.write(fd, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
                finally:
                    os.close(fd)
                _ACTIVE_LEASE = _Lease(
                    path=path,
                    slot=slot,
                    acquired_at=time.perf_counter(),
                    trade_date=trade_date,
                )
                _emit(
                    P,
                    "materialize_peak_gate",
                    "acquired",
                    trade_date=trade_date,
                    slot=slot,
                    wait_seconds=time.perf_counter() - started,
                    available_gb=round(available_gb, 3),
                )
                return _ACTIVE_LEASE
        time.sleep(poll_seconds)


def _release_peak_lease(P: Any, reason: str) -> None:
    global _ACTIVE_LEASE
    lease = _ACTIVE_LEASE
    if lease is None:
        return
    held_seconds = time.perf_counter() - lease.acquired_at
    # Keep the slot owned until temporary Arrow/pandas/allocator buffers have
    # actually been returned.  Releasing first allows the next process to enter
    # its 30 GB peak while this process is still at its own high-water RSS.
    trim = _memory_trim()
    try:
        lease.path.unlink(missing_ok=True)
    finally:
        _ACTIVE_LEASE = None
    _emit(
        P,
        "materialize_peak_gate",
        "released",
        trade_date=lease.trade_date,
        slot=lease.slot,
        reason=reason,
        held_seconds=held_seconds,
        **{key: round(value, 3) for key, value in trim.items()},
    )


def _manifest_is_reusable(
    C: Any,
    manifest: Mapping[str, Any],
    root: Path,
    contract: str,
    rows: int,
) -> bool:
    if manifest.get("status") != "complete" or manifest.get("contract_hash") != contract:
        return False
    for item in manifest.get("blocks", []):
        path = root / str(item.get("path", ""))
        if not path.exists():
            return False
        try:
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != rows:
                return False
            expected_columns = set(str(value) for value in item.get("columns", []))
            if expected_columns and not expected_columns.issubset(set(parquet.schema_arrow.names)):
                return False
        except Exception:
            return False
    return True


def _streaming_derive_all(
    P: Any,
    frame: pd.DataFrame,
    prototypes: list[dict[str, Any]],
):
    C = P.C
    if C.CTX is None:
        return _ORIGINALS["derive_all"](frame, prototypes)
    specs = P.V26.SPEC_REGISTRY[
        P.V26.SPEC_REGISTRY["family"].isin(list("ABCDEFGHIJK"))
    ]
    runtime: dict[str, dict[str, Any]] = {}
    for (family, window), group in specs.groupby(["family", "window"], sort=False):
        root = C.CTX.root / "factors" / f"family={family}" / f"window={window}"
        contract = C._factor_contract(group, frame)
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
            if _manifest_is_reusable(C, manifest, root, contract, len(frame)):
                for item in manifest.get("blocks", []):
                    runtime.update(item.get("factors", {}))
                C._event(
                    "factor_block",
                    "reused",
                    family=family,
                    window=window,
                    streaming=True,
                )
                continue

        started = time.perf_counter()
        raw = C._derive_corrected(frame, f"{family}__ALL__", str(window))
        values = raw if isinstance(raw, dict) else {}
        group_rows = list(group.itertuples(index=False))
        entries: list[dict[str, Any]] = []
        group_factor_count = 0
        for block_id, start in enumerate(
            range(0, len(group_rows), C.CTX.factor_block_size)
        ):
            selected_rows = group_rows[start : start + C.CTX.factor_block_size]
            generated: dict[str, pd.Series] = {}
            status: dict[str, dict[str, Any]] = {}
            for row in selected_rows:
                name = C.FF.factor_name(row.prototype_id, row.window)
                value = values.get(row.prototype_id)
                if value is None:
                    runtime[name] = {
                        "status": "DATA_UNAVAILABLE",
                        "non_null_rate": 0.0,
                    }
                    status[name] = runtime[name]
                    continue
                numeric = (
                    pd.to_numeric(value, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .astype("float32")
                )
                rate = float(numeric.notna().mean())
                runtime[name] = {
                    "status": "SUCCESS" if rate > 0 else "LOW_COVERAGE",
                    "non_null_rate": rate,
                }
                generated[name] = numeric
                status[name] = runtime[name]
            if generated:
                part = pd.concat(generated, axis=1, copy=False)
                part.columns = list(generated)
                part.index = frame.index
                path = root / f"block={block_id:03d}.parquet"
                C._atomic_parquet(part, path)
                entries.append(
                    {
                        "path": path.name,
                        "columns": list(generated),
                        "factors": status,
                    }
                )
                group_factor_count += len(generated)
                del part
            del generated
        C._atomic_json(
            manifest_path,
            {
                "status": "complete",
                "contract_hash": contract,
                "family": family,
                "window": window,
                "blocks": entries,
                "elapsed_seconds": time.perf_counter() - started,
                "materialization_mode": "stream_to_checkpoint_no_wide_concat",
            },
        )
        C._event(
            "factor_block",
            "complete",
            family=family,
            window=window,
            factors=group_factor_count,
            streaming=True,
        )
        del raw, values, group_rows, entries
        C.FF._DERIVE_CACHE.clear()
        trim = _memory_trim()
        C._event(
            "factor_block_memory",
            "trimmed",
            family=family,
            window=window,
            rss_before_gb=round(trim["rss_before_gb"], 3),
            rss_after_gb=round(trim["rss_after_gb"], 3),
        )
    C.FF._DERIVE_CACHE.clear()
    # The factor blocks are now the source of truth.  Do not concatenate all
    # 462 A-K columns back into the live frame; downstream v2.8 stages read
    # selected blocks from their manifests.
    return frame, runtime


def _manifest_factor_status(
    P: Any,
    trade_date: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for manifest_path in P._factor_manifests(P._CONFIG, trade_date):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("status") != "complete":
            continue
        for block in manifest.get("blocks", []):
            for name in block.get("columns", []):
                status = dict(block.get("factors", {}).get(name, {}))
                if not status:
                    status = {
                        "status": "SUCCESS",
                        "non_null_rate": math.nan,
                    }
                result[str(name)] = status
    return result


def _streaming_feature_registry(
    P: Any,
    features: pd.DataFrame,
    trade_date: str,
    expected: set[str],
) -> pd.DataFrame:
    base = P.R.feature_registry(features, [], trade_date)
    # S15/S30 are still present in the reduced live frame.  Their authoritative
    # audit rows come from the same factor manifests as A-K, so remove the live
    # duplicates before appending manifest-backed physical-factor rows.
    base = base.loc[~base["feature"].isin(expected)].copy()
    statuses = _manifest_factor_status(P, trade_date)
    rows: list[dict[str, Any]] = []
    spec = P.V26.SPEC_REGISTRY.set_index("factor_id", drop=False)
    for feature in sorted(expected):
        item = statuses.get(feature, {})
        rate = float(item.get("non_null_rate", math.nan))
        non_null_count = (
            int(round(rate * len(features))) if math.isfinite(rate) else 0
        )
        row = {
            "trade_date": trade_date,
            "feature": feature,
            "bundle": P.R.infer_bundle(feature),
            "evaluated": True,
            "exclusion_reason": None,
            "non_null_count": non_null_count,
            "non_null_rate": rate,
            "unique_count": math.nan,
            "variance": math.nan,
            "runtime_status": item.get("status", "UNKNOWN"),
            "registry_stat_source": "factor_manifest",
        }
        if feature in spec.index:
            row["family"] = spec.loc[feature].get("family")
            row["window"] = spec.loc[feature].get("window")
            row["prototype_id"] = spec.loc[feature].get("prototype_id")
        rows.append(row)
    factors = pd.DataFrame(rows)
    base["registry_stat_source"] = "live_support_frame"
    return pd.concat([base, factors], ignore_index=True, sort=False)


def _optimized_materialize_date(
    P: Any,
    config: dict[str, Any],
    trade_date: str,
) -> dict[str, Any]:
    P.bootstrap(config)
    C = P.C
    stage_root = P._stage_root(config, "materialize", trade_date)
    success = P._stage_success(config, "materialize", trade_date)
    contract = P._contract_hash(config)
    meta_path = stage_root / "meta.json"
    if success.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("contract_hash") == contract:
            return {
                "trade_date": trade_date,
                "stage": "materialize",
                "status": "skipped",
            }

    started = time.perf_counter()
    stage_root.mkdir(parents=True, exist_ok=True)
    atomic = config.get("atomic", {})
    C.CTX = C.Context(
        trade_date=trade_date,
        out_root=P._run_root(config),
        contract_hash=contract,
        factor_block_size=int(atomic.get("factor_block_size", 8)),
        intra_workers=int(atomic.get("intra_date_workers", 4)),
    )
    C.CTX.root.mkdir(parents=True, exist_ok=True)
    C.write_schema_audit(
        P.R.WAREHOUSE_ROOT,
        C.CTX.root / "schema",
        trade_date,
    )
    C._event(
        "pipeline_materialize",
        "running",
        execution_mode="memory_streaming_peak_gated",
    )
    try:
        loader = P.R.NFFDataLoader(
            warehouse_root=P.R.WAREHOUSE_ROOT,
            canonical_sets=P.R.canonical_sets(),
            feature_sets=P.R.feature_sets(),
            execution={
                "frequency": "1min",
                "delay_bars": 1,
                "collision_policy": "latest",
            },
            label=None,
            join="inner",
            strict_manifests=True,
            allow_mixed_contracts=bool(
                config["run"].get("allow_mixed_contracts", True)
            ),
            output_float32=True,
            arrow_use_threads=True,
        )
        start_time, end_time = P.R.loader_session_range(trade_date)
        loaded = loader.load(
            instruments="all",
            start_time=start_time,
            end_time=end_time,
        )
        features = P.R.add_all_features(loaded["feature"].sort_index())
        del loaded
        _memory_trim()

        horizons = [int(value) for value in config["run"]["horizons"]]
        labels, label_masks = P.R.build_labels_and_masks(features, horizons)
        controls = P.R.join_daily_controls(
            features,
            trade_date,
            P._controls_path(config),
        )
        universes = pd.DataFrame(
            P.R.universe_masks(features, controls),
            index=features.index,
        )

        expected = set(P.V26.FULL_FACTOR_NAMES)
        inventory = P._factor_block_inventory(config, trade_date)
        present = (
            set(inventory.get("feature", ())) if not inventory.empty else set()
        )
        missing = expected - present
        extra = present - expected
        if missing or extra:
            raise RuntimeError(
                f"factor block inventory mismatch for {trade_date}: "
                f"present={len(present)} expected={len(expected)} "
                f"missing={sorted(missing)[:20]} extra={sorted(extra)[:20]}"
            )

        support_columns = P._support_columns(features)
        support = features[support_columns].copy(deep=False)
        registry = _streaming_feature_registry(
            P,
            features,
            trade_date,
            expected,
        )

        P._atomic_parquet(support, stage_root / "support.parquet")
        P._atomic_parquet(labels, stage_root / "labels.parquet")
        P._atomic_parquet(
            pd.DataFrame(label_masks, index=features.index),
            stage_root / "label_masks.parquet",
        )
        P._atomic_parquet(controls, stage_root / "controls.parquet")
        P._atomic_parquet(
            universes.astype("boolean"),
            stage_root / "universes.parquet",
        )
        P._atomic_parquet(
            registry,
            stage_root / "feature_registry.parquet",
            index=False,
        )
        P._atomic_parquet(
            inventory,
            stage_root / "factor_block_inventory.parquet",
            index=False,
        )

        settings = dict(_internal_config(config))
        meta = {
            "version": P.VERSION,
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
            "materialization_mode": "stream_factor_blocks",
            "peak_phase_gate": {
                "peak_slots": int(settings.get("peak_slots", 1)),
                "peak_entry_min_available_gb": float(
                    settings.get("peak_entry_min_available_gb", 48.0)
                ),
            },
            "rss_gb_before_success": round(_rss_gb(), 3),
        }
        P._atomic_json(meta_path, meta)
        success.write_text(
            pd.Timestamp.utcnow().isoformat(),
            encoding="utf-8",
        )
        C._event(
            "pipeline_materialize",
            "complete",
            elapsed_seconds=meta["elapsed_seconds"],
        )
        return meta
    except BaseException as exc:
        C._event(
            "pipeline_materialize",
            "failed",
            error=repr(exc),
        )
        raise
    finally:
        _release_peak_lease(P, "materialize_finally")
        C.FUTURE_CACHE.clear()
        C.FUTURE_WIDE_CACHE.clear()
        C.CTX = None
        _memory_trim()


def _install_after_bootstrap(
    P: Any,
    config: dict[str, Any],
) -> None:
    global _POST_BOOTSTRAP
    if _POST_BOOTSTRAP:
        return
    _POST_BOOTSTRAP = True

    loader_class = P.R.NFFDataLoader
    _ORIGINALS["loader_load"] = loader_class.load
    _ORIGINALS["derive_all"] = P.C.FF.derive_all
    _ORIGINALS["merge_venue"] = P.C.FF.merge_venue_aggregates

    def gated_loader_load(self: Any, *args: Any, **kwargs: Any):
        trade_date = (
            P.C.CTX.trade_date if P.C.CTX is not None else "unknown"
        )
        _acquire_peak_lease(P, config, trade_date)
        _emit(
            P,
            "materialize_peak_phase",
            "warehouse_load_running",
            trade_date=trade_date,
        )
        try:
            return _ORIGINALS["loader_load"](self, *args, **kwargs)
        finally:
            _emit(
                P,
                "materialize_peak_phase",
                "warehouse_load_complete",
                trade_date=trade_date,
            )

    def release_after_venue(*args: Any, **kwargs: Any):
        trade_date = (
            P.C.CTX.trade_date if P.C.CTX is not None else "unknown"
        )
        try:
            return _ORIGINALS["merge_venue"](*args, **kwargs)
        finally:
            _release_peak_lease(P, "after_merge_venue")
            _emit(
                P,
                "materialize_peak_phase",
                "wide_merge_complete",
                trade_date=trade_date,
            )

    loader_class.load = gated_loader_load
    P.C.FF.merge_venue_aggregates = release_after_venue
    P.C.FF.derive_all = lambda frame, prototypes: _streaming_derive_all(
        P,
        frame,
        prototypes,
    )


def install(P: Any) -> None:
    original_bootstrap = P.bootstrap

    def bootstrap(config: dict[str, Any]) -> None:
        original_bootstrap(config)
        _install_after_bootstrap(P, config)

    P.bootstrap = bootstrap
    P.materialize_date = lambda config, trade_date: _optimized_materialize_date(
        P,
        config,
        trade_date,
    )
