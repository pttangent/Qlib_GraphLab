from __future__ import annotations

import pandas as pd

from nff_research import v2_8_pipeline as P
from nff_research import v2_8_launch as LAUNCH


def test_factor_family_and_candidate_quotas() -> None:
    rows = []
    for family in "ABCEH":
        for index in range(6):
            rows.append(
                {
                    "feature": f"full_factor__{family.lower()}{index:02d}__w15m",
                    "factor_family": family,
                    "selection_score": 100.0 - len(rows),
                    "valid_days": 60,
                    "coverage_mean": 0.95,
                }
            )
    frame = pd.DataFrame(rows)
    selected = P._candidate_rows(
        frame,
        max_features=12,
        min_per_family=2,
        max_per_family=3,
    )
    assert len(selected) == 12
    assert selected.groupby("factor_family").size().min() >= 2
    assert selected.groupby("factor_family").size().max() <= 3
    assert P._factor_family("full_factor__s01__w30m") == "S"


def test_selected_portfolio_variants_never_reverse_frozen_direction() -> None:
    variants = P._selected_portfolio_variants({})
    assert variants
    assert all(float(item["direction"]) == 1.0 for item in variants)
    paired = [item for item in variants if item.get("gate_pair_id")]
    assert paired
    modes = {item["gate_mode"] for item in paired}
    assert modes == {"ungated_shared_sample", "exclude_top20"}


def test_stage_specs_are_decoupled() -> None:
    config = {
        "pipeline": {
            "stages": {
                "materialize": {"max_workers": 5, "estimated_worker_gb": 11},
                "basic_screen": {"max_workers": 9, "estimated_worker_gb": 5},
                "detailed": {"max_workers": 3, "estimated_worker_gb": 17},
                "portfolio": {"max_workers": 7, "estimated_worker_gb": 7},
            }
        }
    }
    specs = P._stage_specs(config)
    assert specs["materialize"].max_workers == 5
    assert specs["basic_screen"].max_workers == 9
    assert specs["detailed"].estimated_worker_gb == 17
    assert specs["portfolio"].max_workers == 7


def test_runtime_hardening_is_installed_by_canonical_launcher() -> None:
    assert LAUNCH.main is P.main
    assert P._run_date_stage.__module__.endswith("v2_8_runtime_hardening")
    assert P.materialize_date.__module__.endswith("v2_8_source_contract")
    assert P.basic_screen_date.__module__.endswith("v2_8_source_contract")
    assert P.select_candidates.__module__.endswith("v2_8_source_contract")
    assert P.detailed_date.__module__.endswith("v2_8_source_contract")
    assert P.portfolio_date.__module__.endswith("v2_8_source_contract")


def test_pipeline_source_contains_training_freeze_guards() -> None:
    pathlib = __import__("pathlib").Path
    source = pathlib(P.__file__).read_text(encoding="utf-8")
    hardening = pathlib(
        __import__("nff_research.v2_8_runtime_hardening", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    source_contract = pathlib(
        __import__("nff_research.v2_8_source_contract", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    assert "portfolio_eligible_after" in source
    assert "skipped_training_period" in source
    assert "direction_contract" in source
    assert "basic_screen" in source and "detailed" in source and "portfolio" in source
    assert "candidate_contract" in source
    assert "admission_paused" in hardening
    assert "memory_headroom" in hardening
    assert "with_atomic_context" in hardening
    assert "upstream_fingerprint" in source_contract
    assert "candidate_manifest_sha256" in source_contract
    assert "screen_fingerprint" in source_contract
