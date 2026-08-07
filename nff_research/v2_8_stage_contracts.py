from __future__ import annotations

"""Stage-specific semantic contracts for the v2.8 DAG.

Changing a portfolio threshold must not invalidate physical factor blocks.
Changing a screen label must not invalidate materialization.  Each stage hashes
only its actual semantic dependencies; source/candidate fingerprints are added
by ``v2_8_source_contract``.
"""

from contextlib import contextmanager
from typing import Any, Callable, Mapping


_ACTIVE_STAGE: str | None = None


def _common(P: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": P.VERSION,
        "v27_version": getattr(P.C, "VERSION", "unknown"),
        "physical_factor_registry": list(getattr(P.V26, "FULL_FACTOR_NAMES", ())),
        "warehouse_root": config.get("local_paths", {}).get("warehouse_root"),
        "study": config.get("study", {}),
        "factor_resolution_policy": config.get("factor_resolution_policy", {}),
        "date_range": {
            "start": config.get("run", {}).get("start_date"),
            "end": config.get("run", {}).get("end_date"),
        },
    }


def contract_payload(P: Any, config: Mapping[str, Any], stage: str) -> dict[str, Any]:
    common = _common(P, config)
    run = config.get("run", {})
    if stage == "materialize":
        return {
            **common,
            "stage": stage,
            "horizons": run.get("horizons"),
            "min_cross_section_n": run.get("min_cross_section_n"),
            "atomic": config.get("atomic", {}),
            "labels": config.get("labels", {}),
            "controls_path": config.get("pipeline", {}).get("controls_path"),
        }
    if stage == "basic_screen":
        selection = config.get("selection", {})
        return {
            **common,
            "stage": stage,
            "min_cross_section_n": run.get("min_cross_section_n"),
            "screen_labels": selection.get("screen_labels"),
            "screen_universes": selection.get("screen_universes"),
        }
    if stage == "select":
        return {
            **common,
            "stage": stage,
            "selection": config.get("selection", {}),
        }
    if stage == "detailed":
        return {
            **common,
            "stage": stage,
            "min_cross_section_n": run.get("min_cross_section_n"),
            "neutralization": config.get("neutralization", {}),
            "labels": config.get("labels", {}),
            "factor_block_size": config.get("atomic", {}).get("factor_block_size"),
            # Track-to-label scope changes which diagnostics are computed and
            # is therefore semantic. Worker counts/BLAS threads remain purely
            # operational and are intentionally excluded.
            "detailed_internal": config.get("pipeline", {}).get(
                "detailed_internal", {}
            ),
        }
    if stage == "portfolio":
        return {
            **common,
            "stage": stage,
            "min_cross_section_n": run.get("min_cross_section_n"),
            "portfolio_proxy": config.get("portfolio_proxy", {}),
        }
    return {
        **common,
        "stage": stage,
        "pipeline": config.get("pipeline", {}),
    }


@contextmanager
def _stage_context(stage: str):
    global _ACTIVE_STAGE
    previous = _ACTIVE_STAGE
    _ACTIVE_STAGE = stage
    try:
        yield
    finally:
        _ACTIVE_STAGE = previous


def install(P: Any) -> None:
    def contract_hash(config: Mapping[str, Any]) -> str:
        stage = _ACTIVE_STAGE or "pipeline"
        return P._json_hash(contract_payload(P, config, stage))

    P._contract_hash = contract_hash

    def wrap(function: Callable[..., Any], stage: str) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with _stage_context(stage):
                return function(*args, **kwargs)

        return wrapped

    # Install outermost so every source/checkpoint wrapper below sees the same
    # active semantic stage while calculating its contract.
    P.materialize_date = wrap(P.materialize_date, "materialize")
    P.basic_screen_date = wrap(P.basic_screen_date, "basic_screen")
    P.select_candidates = wrap(P.select_candidates, "select")
    P.detailed_date = wrap(P.detailed_date, "detailed")
    P.portfolio_date = wrap(P.portfolio_date, "portfolio")
