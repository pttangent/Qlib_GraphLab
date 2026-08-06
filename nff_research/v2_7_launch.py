from __future__ import annotations

"""Canonical launch script for the complete v2.7 campaign."""

import json
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nff_research import v2_7_deciles as DECILES
from nff_research import v2_7_run as RUN
from nff_research import v2_7_runtime_hardening as HARDEN


C = RUN.C
HARDEN.install(C)

_BASE_VALIDATION_SCORE = RUN.MODELS._validation_score
_BASE_CONFIGURE_REGISTRY = C.configure_registry
_BASE_ADD_ALL_FEATURES = C._add_all_features
STRICT_FACTOR_POLICY = {
    "expected_factor_specs": 476,
    "fail_on_unavailable": True,
    "fail_on_zero_coverage": True,
}


def _finite_validation_score(source, predictions) -> float:
    score = float(_BASE_VALIDATION_SCORE(source, predictions))
    return score if math.isfinite(score) else -math.inf


def _exact_decile_feature_rows(
    feature,
    signal,
    label,
    adv,
    price,
    trades,
    metadata,
    min_n,
):
    return DECILES.decile_feature_rows(
        C.R,
        feature,
        signal,
        label,
        adv,
        price,
        trades,
        metadata,
        min_n,
    )


def _configure_476_registry(config) -> None:
    """Preserve the 474+2 baseline while remapping obsolete 10m slots.

    The old contract had the correct number of hypotheses but referred to a
    10m minute family that is not the governed NFF/supplement baseline. We keep
    the same number of parameter slots and move them to actual, financially
    ordered horizons rather than multiplying every prototype over every field
    discovered in the warehouse.
    """
    global STRICT_FACTOR_POLICY
    _BASE_CONFIGURE_REGISTRY(config)
    fixed_windows = {
        "A": ("15m", "30m", "60m"),
        "B": ("15m", "30m", "60m"),
        "C": ("15m", "30m", "60m", "120m"),
        "D": ("15m", "30m", "60m"),
        "E": ("60s", "180s", "300s"),
        "F": ("60s", "180s", "300s"),
        "G": ("15m", "30m", "60m"),
        "H": ("60s", "180s", "300s"),
        "I": ("1m",),
        "J": ("1m",),
        "K": ("1m",),
    }
    C.FF.FAMILY_WINDOWS = fixed_windows
    C.RUNTIME_WINDOWS = dict(fixed_windows)
    base_specs = C.FF.expand_specs(C.V26.PROTOTYPES)
    if len(base_specs) != 474:
        raise RuntimeError(f"A-K contract drift: expected 474 specs, found {len(base_specs)}")

    supplement_rows = []
    for window in (15, 30):
        fields = C.directional_supplement_fields(window)
        supplement_rows.append(
            {
                "factor_id": f"full_factor__s01__w{window}m",
                "prototype_id": "S01",
                "family": "S",
                "title": "Exact NVG supplement directional consensus",
                "role": "DIRECTION_ALPHA",
                "direction_prior": "TWO_SIDED",
                "source_fields": "|".join(fields.values()),
                "formula": "raw+detrended price direction; volume and cross-graph structure as confidence",
                "financial_hypothesis": "price direction confirmed by detrended geometry and volume structure is more persistent",
                "window": f"{window}m",
                "transform": "robust cross-sectional z-score and bounded confidence",
                "label_group": "return,state,risk,liquidity,cost",
                "required_gate": "PIT and liquidity/coverage",
                "available_time_rule": "supplement available_time <= decision time; entry next exact minute",
                "expected_range": "approximately [-1,1]",
                "neutralization_allowed": True,
                "cost_relevance": "direct",
                "status": "PENDING_SCHEMA_RESOLUTION",
                "instruction_line": None,
            }
        )
    C.V26.SPEC_REGISTRY = pd.concat(
        [base_specs, pd.DataFrame(supplement_rows)], ignore_index=True
    )
    if len(C.V26.SPEC_REGISTRY) != 476:
        raise RuntimeError(
            f"formal factor contract drift: expected 476 specs, found {len(C.V26.SPEC_REGISTRY)}"
        )
    C.V26.SUPPLEMENT_FACTOR_NAMES = [row["factor_id"] for row in supplement_rows]
    C.V26.FULL_FACTOR_NAMES = list(C.V26.SPEC_REGISTRY["factor_id"])
    C.R.CORE_DECILE_FEATURES = list(C.V26.FULL_FACTOR_NAMES)
    C.BASE.run_temporal_oos.__globals__["DERIVED_FEATURES"] = list(C.V26.FULL_FACTOR_NAMES)
    policy = config.get("factor_resolution_policy", {})
    STRICT_FACTOR_POLICY = {
        "expected_factor_specs": int(policy.get("expected_factor_specs", 476)),
        "fail_on_unavailable": bool(policy.get("fail_on_unavailable", True)),
        "fail_on_zero_coverage": bool(policy.get("fail_on_zero_coverage", True)),
    }


def _strict_add_all_features(frame):
    result = _BASE_ADD_ALL_FEATURES(frame)
    registry = C.V26.SPEC_REGISTRY
    runtime = C.V26.RUNTIME_FACTOR_STATUS
    expected = int(STRICT_FACTOR_POLICY["expected_factor_specs"])
    if len(registry) != expected:
        raise RuntimeError(f"factor registry count mismatch: expected={expected}, actual={len(registry)}")
    unavailable = []
    zero_coverage = []
    absent_columns = []
    for row in registry.itertuples(index=False):
        state = runtime.get(row.factor_id, {})
        status = str(state.get("status", "NOT_MATERIALIZED"))
        rate = float(state.get("non_null_rate", 0.0) or 0.0)
        if "UNAVAILABLE" in status or status == "NOT_MATERIALIZED":
            unavailable.append(row.factor_id)
        if rate <= 0.0:
            zero_coverage.append(row.factor_id)
        if row.factor_id not in result.columns:
            absent_columns.append(row.factor_id)
    failure = {
        "expected_factor_specs": expected,
        "registry_specs": len(registry),
        "materialized_columns": int(sum(name in result.columns for name in registry["factor_id"])),
        "unavailable_count": len(unavailable),
        "zero_coverage_count": len(zero_coverage),
        "absent_column_count": len(absent_columns),
        "unavailable": unavailable,
        "zero_coverage": zero_coverage,
        "absent_columns": absent_columns,
        "policy": STRICT_FACTOR_POLICY,
    }
    if C.CTX is not None:
        path = C.CTX.root / "schema" / "factor_completion_gate.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
    should_fail = bool(absent_columns)
    should_fail |= bool(unavailable) and STRICT_FACTOR_POLICY["fail_on_unavailable"]
    should_fail |= bool(zero_coverage) and STRICT_FACTOR_POLICY["fail_on_zero_coverage"]
    if should_fail:
        raise RuntimeError(
            "476-factor completion gate failed; inspect factor_completion_gate.json and factor_resolution.csv"
        )
    return result


RUN.MODELS._validation_score = _finite_validation_score
C._decile_feature_rows = _exact_decile_feature_rows
C.configure_registry = _configure_476_registry
C._add_all_features = _strict_add_all_features


def _install_worker_command() -> None:
    original = C.R.worker_command
    current = str(Path(__file__).resolve())
    known_names = {
        "v2_1_neutralized_runner.py",
        "v2_6_full_defined_campaign.py",
        "v2_7_atomic_campaign.py",
        "v2_7_atomic_entry.py",
        "v2_7_run.py",
        "v2_7_launch.py",
    }

    def worker_command(*args: Any, **kwargs: Any) -> list[str]:
        command = original(*args, **kwargs)
        replaced = False
        result: list[str] = []
        for value in command:
            if Path(value).name in known_names:
                result.append(current)
                replaced = True
            else:
                result.append(value)
        if not replaced:
            raise RuntimeError(f"worker command has no replaceable research entrypoint: {command}")
        return result

    C.R.worker_command = worker_command


C._install_worker_command = _install_worker_command


def main() -> int:
    return C.main()


if __name__ == "__main__":
    raise SystemExit(main())
