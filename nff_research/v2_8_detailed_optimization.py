from __future__ import annotations

"""Track-aware, memory-bounded execution for the v2.8 detailed stage.

This module does not change factor values, PIT joins, label definitions, the
training-only candidate freeze, or the existing IC/neutralization/decile math.
It changes execution shape only where the research role makes the old all-by-all
matrix redundant:

- Alpha candidates are evaluated on return labels and retain the expensive
  return neutralization + decile path.
- Risk/regime candidates are evaluated only on risk/state targets and skip
  return deciles/neutralization that cannot answer their selected hypothesis.
- Cost/liquidity candidates are evaluated only on cost/liquidity targets and
  likewise skip return-only detailed work.
- Each track loads its own selected factor columns, writes a restartable track
  checkpoint, then releases the factor matrix before the next track.

The existing v2.7 residualizer already performs fixed-effect within demeaning,
validity-mask grouping, and multi-target least squares.  This layer therefore
focuses on eliminating redundant cross-track work rather than replacing an
already block-oriented linear algebra kernel.
"""

from contextlib import contextmanager
import gc
import json
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


_DEFAULT_TRACK_LABELS: dict[str, tuple[str, ...]] = {
    "alpha": (
        "return_open_to_open",
        "return_vwap_to_vwap",
        "return_close_to_close",
    ),
    "risk_regime": (
        "realized_volatility",
        "liquidity_deterioration",
        "jump_tail_event",
    ),
    "cost_liquidity": (
        "execution_cost_proxy",
        "liquidity_deterioration",
    ),
}


def _settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("pipeline", {}).get("detailed_internal", {})


def _track_label_families(config: Mapping[str, Any], track: str) -> tuple[str, ...]:
    configured = _settings(config).get("track_label_families", {})
    if isinstance(configured, Mapping) and track in configured:
        return tuple(str(value) for value in configured[track])
    return _DEFAULT_TRACK_LABELS.get(track, tuple())


def _decile_tracks(config: Mapping[str, Any]) -> set[str]:
    values = _settings(config).get("decile_tracks", ["alpha"])
    return {str(value) for value in values}


def _subset_labels(
    labels: pd.DataFrame,
    masks: pd.DataFrame,
    families: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed = set(families)
    columns = [
        column
        for column in labels.columns
        if "__h" in column and column.rsplit("__h", 1)[0] in allowed
    ]
    return labels.loc[:, columns], masks.loc[:, columns]


def _candidate_track_groups(candidates: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if candidates.empty:
        return []
    work = candidates.copy()
    if "selection_track" not in work.columns:
        work["selection_track"] = "alpha"
    order = ["alpha", "risk_regime", "cost_liquidity"]
    seen = set(work["selection_track"].astype(str))
    order.extend(sorted(seen - set(order)))
    groups: list[tuple[str, pd.DataFrame]] = []
    for track in order:
        part = work[work["selection_track"].astype(str) == track].copy()
        if not part.empty:
            groups.append((track, part))
    return groups


def _upstream_fingerprint(P: Any, config: Mapping[str, Any], trade_date: str) -> str | None:
    path = P._stage_root(config, "materialize", trade_date) / "meta.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value.get("upstream_fingerprint")


def _track_contract(
    P: Any,
    config: Mapping[str, Any],
    trade_date: str,
    track: str,
    candidate_rows: pd.DataFrame,
    label_columns: list[str],
) -> str:
    candidate_payload = candidate_rows.sort_values("feature").to_dict("records")
    return P._json_hash(
        {
            "version": P.VERSION,
            "stage_contract": P._contract_hash(config),
            "trade_date": trade_date,
            "track": track,
            "candidate_rows": candidate_payload,
            "labels": label_columns,
            "upstream_fingerprint": _upstream_fingerprint(P, config, trade_date),
            "execution": "track_streaming_detailed_v2.8",
        }
    )


def _inner_decile_cache(P: Any) -> Path | None:
    if P.C.CTX is None:
        return None
    return P.C.CTX.root / "stages" / "decile_curves.parquet"


@contextmanager
def _bounded_internal_executor(P: Any):
    """Bound legacy internal ThreadPoolExecutor calls to the stage budget.

    v2.7's rank-cache path has an internal universe pool whose historical cap
    was three workers.  At eight date processes that would silently turn the
    intended 8×2 shape into as many as 24 Python threads before BLAS threads are
    counted.  Patch only the function-global executor during this date worker.
    """
    limit = max(1, int(getattr(P.C.CTX, "intra_workers", 1))) if P.C.CTX is not None else 1
    globals_dict = getattr(P.R.minute_rank_ic_summary, "__globals__", {})
    original = globals_dict.get("ThreadPoolExecutor")
    if original is None:
        yield
        return

    def bounded(*args: Any, **kwargs: Any):
        if args:
            requested = max(1, int(args[0]))
            args = (min(requested, limit), *args[1:])
        else:
            requested = max(1, int(kwargs.get("max_workers", limit)))
            kwargs["max_workers"] = min(requested, limit)
        return original(*args, **kwargs)

    globals_dict["ThreadPoolExecutor"] = bounded
    try:
        yield
    finally:
        globals_dict["ThreadPoolExecutor"] = original


def install(P: Any) -> None:
    def detailed_date_track_streaming(
        config: dict[str, Any], trade_date: str
    ) -> dict[str, Any]:
        P.bootstrap(config)
        root = P._stage_root(config, "detailed", trade_date)
        success = P._stage_success(config, "detailed", trade_date)
        contract = P._contract_hash(config)
        meta_path = root / "meta.json"
        if success.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("contract_hash") == contract:
                return {
                    "trade_date": trade_date,
                    "stage": "detailed",
                    "status": "skipped",
                }

        started = time.perf_counter()
        root.mkdir(parents=True, exist_ok=True)
        candidates, selection_meta = P._load_candidates(config, "detailed")
        support, labels, label_masks, controls, universes = P._load_materialized(
            config, trade_date
        )
        min_n = int(config["run"].get("min_cross_section_n", 30))
        decile_tracks = _decile_tracks(config)
        summary_parts: list[pd.DataFrame] = []
        decile_parts: list[pd.DataFrame] = []
        track_meta: list[dict[str, Any]] = []

        for track, track_candidates in _candidate_track_groups(candidates):
            names = list(track_candidates["feature"].drop_duplicates())
            families = _track_label_families(config, track)
            track_labels, track_masks = _subset_labels(labels, label_masks, families)
            track_root = root / "tracks" / f"track={track}"
            track_root.mkdir(parents=True, exist_ok=True)
            track_contract = _track_contract(
                P,
                config,
                trade_date,
                track,
                track_candidates,
                list(track_labels.columns),
            )
            track_meta_path = track_root / "meta.json"
            track_success = track_root / "_SUCCESS"
            summary_path = track_root / "factor_rank_ic_summary.parquet"
            decile_path = track_root / "decile_curves.parquet"

            existing: dict[str, Any] = {}
            if track_meta_path.exists():
                try:
                    existing = json.loads(track_meta_path.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
            reusable = (
                track_success.exists()
                and existing.get("track_contract") == track_contract
                and summary_path.exists()
            )
            if reusable:
                summary_parts.append(pd.read_parquet(summary_path))
                if decile_path.exists():
                    decile_parts.append(pd.read_parquet(decile_path))
                track_meta.append({**existing, "status": "reused"})
                continue

            # Track contracts are independently restartable.  Remove stale
            # outputs before writing a new contract so a previously enabled
            # decile path cannot leak into a later no-decile/no-work run.
            track_success.unlink(missing_ok=True)
            summary_path.unlink(missing_ok=True)
            decile_path.unlink(missing_ok=True)

            if not names or track_labels.empty:
                empty_meta = {
                    "version": P.VERSION,
                    "trade_date": trade_date,
                    "track": track,
                    "status": "no_work",
                    "track_contract": track_contract,
                    "candidate_count": len(names),
                    "label_columns": list(track_labels.columns),
                    "label_families": list(families),
                    "deciles_enabled": track in decile_tracks,
                }
                P._atomic_json(track_meta_path, empty_meta)
                track_success.write_text(pd.Timestamp.utcnow().isoformat(), encoding="utf-8")
                track_meta.append(empty_meta)
                continue

            track_started = time.perf_counter()
            factors = P._load_selected_factors(config, trade_date, names, support.index)
            features = pd.concat([support, factors], axis=1, copy=False)
            original_universes = P._with_saved_universes(universes)
            original_deciles = list(P.R.CORE_DECILE_FEATURES)
            P.R.CORE_DECILE_FEATURES = names
            try:
                with _bounded_internal_executor(P):
                    summary, residual_cache = P.R.minute_rank_ic_summary(
                        features,
                        track_labels,
                        track_masks,
                        controls,
                        trade_date,
                        min_n,
                    )
                    summary = summary.copy()
                    summary["selection_track"] = track
                    summary["detailed_label_scope"] = ",".join(families)

                    if track in decile_tracks:
                        # The v2.7 internal decile cache is date-global.  A prior
                        # all-candidate detailed run must not be mistaken for this
                        # narrower alpha-only track contract.
                        inner_cache = _inner_decile_cache(P)
                        if inner_cache is not None:
                            inner_cache.unlink(missing_ok=True)
                            inner_cache.with_suffix(".json").unlink(missing_ok=True)
                        deciles = P.R.decile_curves(
                            features,
                            track_labels,
                            track_masks,
                            controls,
                            residual_cache,
                            trade_date,
                            min_n,
                        )
                        if not deciles.empty:
                            deciles = deciles.copy()
                            deciles["selection_track"] = track
                    else:
                        deciles = pd.DataFrame()
                    del residual_cache
            finally:
                P.R.universe_masks = original_universes
                P.R.CORE_DECILE_FEATURES = original_deciles

            P._atomic_parquet(summary, summary_path, index=False)
            if not deciles.empty:
                P._atomic_parquet(deciles, decile_path, index=False)
            track_record = {
                "version": P.VERSION,
                "trade_date": trade_date,
                "track": track,
                "status": "complete",
                "track_contract": track_contract,
                "candidate_count": len(names),
                "label_columns": list(track_labels.columns),
                "label_families": list(families),
                "summary_rows": int(len(summary)),
                "decile_rows": int(len(deciles)),
                "deciles_enabled": track in decile_tracks,
                "intra_workers": int(getattr(P.C.CTX, "intra_workers", 1)) if P.C.CTX is not None else 1,
                "elapsed_seconds": time.perf_counter() - track_started,
            }
            P._atomic_json(track_meta_path, track_record)
            track_success.write_text(pd.Timestamp.utcnow().isoformat(), encoding="utf-8")
            summary_parts.append(summary)
            if not deciles.empty:
                decile_parts.append(deciles)
            track_meta.append(track_record)

            del factors, features, summary, deciles
            gc.collect()

        summary = (
            pd.concat(summary_parts, ignore_index=True, copy=False)
            if summary_parts
            else pd.DataFrame()
        )
        deciles = (
            pd.concat(decile_parts, ignore_index=True, copy=False)
            if decile_parts
            else pd.DataFrame()
        )
        P._atomic_parquet(
            summary,
            root / "factor_rank_ic_summary.parquet",
            index=False,
        )
        if not deciles.empty:
            P._atomic_parquet(deciles, root / "decile_curves.parquet", index=False)
        else:
            (root / "decile_curves.parquet").unlink(missing_ok=True)
        meta = {
            "version": P.VERSION,
            "trade_date": trade_date,
            "stage": "detailed",
            "status": "complete",
            "contract_hash": contract,
            "selection_end_date": selection_meta["selection_end_date"],
            "candidate_count": int(candidates["feature"].nunique()),
            "summary_rows": int(len(summary)),
            "decile_rows": int(len(deciles)),
            "execution_mode": "track_streaming",
            "tracks": track_meta,
            "elapsed_seconds": time.perf_counter() - started,
        }
        P._atomic_json(meta_path, meta)
        success.write_text(pd.Timestamp.utcnow().isoformat(), encoding="utf-8")
        return meta

    P.detailed_date = detailed_date_track_streaming
