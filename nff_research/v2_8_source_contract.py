from __future__ import annotations

"""Source-aware contracts for v2.8 stage success markers."""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from nff_research import v2_7_checkpoint_hardening as CHECKPOINTS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_fingerprint(paths: list[Path]) -> str:
    records = []
    for path in sorted(paths):
        if not path.exists():
            continue
        stat = path.stat()
        records.append(
            {
                "path": path.as_posix(),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _write_meta(P: Any, path: Path, updates: Mapping[str, Any]) -> None:
    meta = _read_meta(path)
    meta.update(updates)
    P._atomic_json(path, meta)


def _invalidate(success: Path) -> None:
    success.unlink(missing_ok=True)


def _invalidate_internal_stage_cache(
    P: Any,
    config: Mapping[str, Any],
    trade_date: str,
    stage: str,
) -> None:
    stages = (
        P._run_root(config)
        / "atomic_checkpoints"
        / f"date={trade_date}"
        / "stages"
    )
    names = {
        "detailed": ("decile_curves.parquet", "decile_curves.json"),
        "portfolio": ("portfolio_proxy.parquet", "portfolio_proxy.json"),
    }
    for name in names.get(stage, ()):
        (stages / name).unlink(missing_ok=True)


def install(P: Any) -> None:
    original_materialize = P.materialize_date
    original_basic = P.basic_screen_date
    original_select = P.select_candidates
    original_detailed = P.detailed_date
    original_portfolio = P.portfolio_date

    def materialize_source_aware(
        config: dict[str, Any], trade_date: str
    ) -> dict[str, Any]:
        source = CHECKPOINTS._upstream_fingerprint(P.C, trade_date)
        root = P._stage_root(config, "materialize", trade_date)
        success = P._stage_success(config, "materialize", trade_date)
        meta_path = root / "meta.json"
        if success.exists() and _read_meta(meta_path).get("upstream_fingerprint") != source:
            _invalidate(success)
        result = original_materialize(config, trade_date)
        _write_meta(
            P,
            meta_path,
            {
                "upstream_fingerprint": source,
                "source_contract": "v2.7 warehouse parquet size+mtime fingerprint",
            },
        )
        return result

    def basic_source_aware(config: dict[str, Any], trade_date: str) -> dict[str, Any]:
        materialize_meta = _read_meta(
            P._stage_root(config, "materialize", trade_date) / "meta.json"
        )
        source = materialize_meta.get("upstream_fingerprint")
        root = P._stage_root(config, "basic_screen", trade_date)
        success = P._stage_success(config, "basic_screen", trade_date)
        meta_path = root / "meta.json"
        if success.exists() and _read_meta(meta_path).get("upstream_fingerprint") != source:
            _invalidate(success)
        result = original_basic(config, trade_date)
        _write_meta(P, meta_path, {"upstream_fingerprint": source})
        return result

    def screen_fingerprint(config: Mapping[str, Any]) -> str:
        paths = [
            P._stage_root(config, "basic_screen", date)
            / "basic_factor_screen.parquet"
            for date in P._dates(config)
            if P._stage_success(config, "basic_screen", date).exists()
        ]
        return _path_fingerprint(paths)

    def select_source_aware(config: dict[str, Any]) -> dict[str, Any]:
        source = screen_fingerprint(config)
        root = P._pipeline_root(config) / "selection"
        success = root / "_SUCCESS"
        meta_path = root / "meta.json"
        if success.exists() and _read_meta(meta_path).get("screen_fingerprint") != source:
            _invalidate(success)
        result = original_select(config)
        _write_meta(P, meta_path, {"screen_fingerprint": source})
        return result

    def downstream_wrapper(
        original: Callable[[dict[str, Any], str], dict[str, Any]],
        stage: str,
    ) -> Callable[[dict[str, Any], str], dict[str, Any]]:
        def wrapped(config: dict[str, Any], trade_date: str) -> dict[str, Any]:
            materialize_meta = _read_meta(
                P._stage_root(config, "materialize", trade_date) / "meta.json"
            )
            upstream = materialize_meta.get("upstream_fingerprint")
            candidates_path = P._pipeline_root(config) / "selection" / "candidates.parquet"
            candidate_sha = _sha256(candidates_path) if candidates_path.exists() else None
            root = P._stage_root(config, stage, trade_date)
            success = P._stage_success(config, stage, trade_date)
            meta_path = root / "meta.json"
            existing = _read_meta(meta_path)
            stale = (
                existing.get("upstream_fingerprint") != upstream
                or existing.get("candidate_manifest_sha256") != candidate_sha
            )
            if success.exists() and stale:
                _invalidate(success)
                _invalidate_internal_stage_cache(P, config, trade_date, stage)
            result = original(config, trade_date)
            _write_meta(
                P,
                meta_path,
                {
                    "upstream_fingerprint": upstream,
                    "candidate_manifest_sha256": candidate_sha,
                },
            )
            return result

        return wrapped

    P.materialize_date = materialize_source_aware
    P.basic_screen_date = basic_source_aware
    P.select_candidates = select_source_aware
    P.detailed_date = downstream_wrapper(original_detailed, "detailed")
    P.portfolio_date = downstream_wrapper(original_portfolio, "portfolio")
