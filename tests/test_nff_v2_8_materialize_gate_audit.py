from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from nff_research import v2_8_materialize_gate_audit as GATE


def test_streaming_audit_passes_complete_physical_inventory() -> None:
    expected = {"f1", "f2"}
    inventory = pd.DataFrame({"feature": ["f1", "f2"]})
    statuses = {
        "f1": {"status": "SUCCESS", "non_null_rate": 0.95},
        "f2": {"status": "SUCCESS", "non_null_rate": 0.80},
    }
    audit, should_fail = GATE.build_streaming_audit(
        expected=expected,
        inventory=inventory,
        statuses=statuses,
        policy={
            "expected_physical_specs": 2,
            "fail_on_unavailable": True,
            "fail_on_zero_coverage": True,
        },
        actual_physical_specs=2,
        wide_frame_columns=17,
        expected_legacy_specs=3,
        legacy_gap_ids=["legacy-only"],
    )

    assert should_fail is False
    assert audit["expected_physical_specs"] == 2
    assert audit["actual_physical_specs"] == 2
    assert audit["materialized_columns"] == 2
    assert audit["unavailable_count"] == 0
    assert audit["zero_coverage_count"] == 0
    assert audit["absent_column_count"] == 0
    assert audit["gate_basis"] == "factor_block_inventory+runtime_or_manifest_status"


def test_streaming_audit_rejects_present_but_zero_coverage_factor() -> None:
    expected = {"f1", "f2"}
    inventory = pd.DataFrame({"feature": ["f1", "f2"]})
    statuses = {
        "f1": {"status": "SUCCESS", "non_null_rate": 0.9},
        "f2": {"status": "LOW_COVERAGE", "non_null_rate": 0.0},
    }
    audit, should_fail = GATE.build_streaming_audit(
        expected=expected,
        inventory=inventory,
        statuses=statuses,
        policy={
            "expected_physical_specs": 2,
            "fail_on_unavailable": True,
            "fail_on_zero_coverage": True,
        },
        actual_physical_specs=2,
        wide_frame_columns=9,
        expected_legacy_specs=2,
        legacy_gap_ids=[],
    )

    assert should_fail is True
    assert audit["materialized_columns"] == 2
    assert audit["zero_coverage"] == ["f2"]
    assert audit["zero_coverage_count"] == 1


def test_streaming_audit_rejects_missing_runtime_status() -> None:
    expected = {"f1"}
    inventory = pd.DataFrame({"feature": ["f1"]})
    audit, should_fail = GATE.build_streaming_audit(
        expected=expected,
        inventory=inventory,
        statuses={},
        policy={
            "expected_physical_specs": 1,
            "fail_on_unavailable": True,
            "fail_on_zero_coverage": True,
        },
        actual_physical_specs=1,
        wide_frame_columns=5,
        expected_legacy_specs=1,
        legacy_gap_ids=[],
    )

    assert should_fail is True
    assert audit["unavailable"] == ["f1"]
    assert audit["zero_coverage"] == ["f1"]
