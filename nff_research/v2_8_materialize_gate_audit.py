from __future__ import annotations

"""Restore the v2.7 physical completion semantics on streamed v2.8 factors.

The streaming materializer intentionally keeps A-K factor columns out of the
live DataFrame, so the legacy wide-frame completion gate cannot be called
literally.  Presence therefore comes from the factor-block inventory, while
availability and coverage come from the live runtime status with manifest
status as a resume fallback.
"""

import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from nff_research import v2_7_physical_contract as PHYSICAL


def _coerce_rate(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_streaming_audit(
    *,
    expected: set[str],
    inventory: pd.DataFrame,
    statuses: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
    actual_physical_specs: int,
    wide_frame_columns: int,
    expected_legacy_specs: int,
    legacy_gap_ids: list[str],
) -> tuple[dict[str, Any], bool]:
    present = set(inventory.get("feature", pd.Series(dtype="string")).astype(str))
    absent = sorted(expected - present)
    extra = sorted(present - expected)
    unavailable: list[str] = []
    zero_coverage: list[str] = []

    for feature in sorted(expected):
        item = statuses.get(feature, {})
        status = str(item.get("status", "NOT_MATERIALIZED"))
        rate = _coerce_rate(item.get("non_null_rate", 0.0))
        if "UNAVAILABLE" in status or status == "NOT_MATERIALIZED":
            unavailable.append(feature)
        if not math.isfinite(rate) or rate <= 0.0:
            zero_coverage.append(feature)

    expected_count = int(policy.get("expected_physical_specs", len(expected)))
    audit = {
        "contract_version": PHYSICAL.CONTRACT_VERSION,
        "expected_physical_specs": expected_count,
        "actual_physical_specs": int(actual_physical_specs),
        "expected_legacy_specs": int(expected_legacy_specs),
        "legacy_gap_count": len(legacy_gap_ids),
        "legacy_gap_ids": list(legacy_gap_ids),
        "materialized_columns": len(present & expected),
        "unavailable_count": len(unavailable),
        "zero_coverage_count": len(zero_coverage),
        "absent_column_count": len(absent),
        "extra_column_count": len(extra),
        "unavailable": unavailable,
        "zero_coverage": zero_coverage,
        "absent_columns": absent,
        "extra_columns": extra,
        "policy": dict(policy),
        "materialization_mode": "stream_factor_blocks",
        "wide_frame_columns": int(wide_frame_columns),
        "gate_basis": "factor_block_inventory+runtime_or_manifest_status",
    }
    should_fail = int(actual_physical_specs) != expected_count or bool(absent) or bool(extra)
    should_fail |= bool(unavailable) and bool(policy.get("fail_on_unavailable", True))
    should_fail |= bool(zero_coverage) and bool(policy.get("fail_on_zero_coverage", True))
    return audit, should_fail


def _write_legacy_gap(P: Any, root: Path) -> None:
    gap = getattr(P.V26, "LEGACY_CONTRACT_GAP", pd.DataFrame()).copy()
    if gap.empty:
        return
    gap.to_parquet(root / "legacy_contract_gap.parquet", index=False)
    gap.to_csv(root / "legacy_contract_gap.csv", index=False)
    policy = getattr(P.C, "PHYSICAL_CONTRACT_POLICY", {})
    P._atomic_json(
        root / "legacy_contract_gap.json",
        {
            "contract_version": PHYSICAL.CONTRACT_VERSION,
            "legacy_formal_specs": int(policy.get("expected_legacy_specs", PHYSICAL.LEGACY_FORMAL_COUNT)),
            "physical_executable_specs": int(policy.get("expected_physical_specs", PHYSICAL.PHYSICAL_EXECUTABLE_COUNT)),
            "gap_count": int(len(gap)),
            "gap_reason": "no published 60m price-NVG topology; 60m HVG remains in family G",
            "factor_ids": list(gap["factor_id"]),
        },
    )


def install(M: Any, P: Any) -> None:
    def strict_streaming_completion_audit(
        expected: set[str],
        inventory: pd.DataFrame,
        *,
        wide_frame_columns: int,
    ) -> dict[str, Any]:
        ctx = P.C.CTX
        if ctx is None:
            raise RuntimeError("streaming physical audit requires an active atomic context")
        trade_date = str(ctx.trade_date)
        statuses = dict(M._manifest_factor_status(P, trade_date))
        # Live runtime status is more complete than older supplement manifests
        # and still exists on resumed A-K blocks because the streaming derive
        # loader reconstructs their status from each manifest.
        statuses.update(getattr(P.V26, "RUNTIME_FACTOR_STATUS", {}))
        policy = dict(getattr(P.C, "PHYSICAL_CONTRACT_POLICY", {}))
        legacy_gap_ids = list(policy.get("legacy_gap_ids", []))
        audit, should_fail = build_streaming_audit(
            expected=expected,
            inventory=inventory,
            statuses=statuses,
            policy=policy,
            actual_physical_specs=int(len(P.V26.SPEC_REGISTRY)),
            wide_frame_columns=wide_frame_columns,
            expected_legacy_specs=int(policy.get("expected_legacy_specs", PHYSICAL.LEGACY_FORMAL_COUNT)),
            legacy_gap_ids=legacy_gap_ids,
        )
        schema_root = ctx.root / "schema"
        schema_root.mkdir(parents=True, exist_ok=True)
        _write_legacy_gap(P, schema_root)
        if should_fail:
            P._atomic_json(schema_root / "physical_factor_completion_gate.json", audit)
            raise RuntimeError(
                "warehouse-exact streamed factor completion gate failed; "
                "inspect physical_factor_completion_gate.json"
            )
        return audit

    M._streaming_completion_audit = strict_streaming_completion_audit
