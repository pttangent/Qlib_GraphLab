from __future__ import annotations

import pandas as pd

from nff_research import v2_8_pipeline as P
from nff_research import v2_8_launch as LAUNCH
from nff_research import v2_8_portfolio_optimization as PORTFOLIO_OPT
from nff_research import v2_8_selection_tracks as TRACKS


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


def test_cost_scenarios_reuse_one_weight_path() -> None:
    base = pd.DataFrame(
        {
            "gross_return": [0.002, -0.001],
            "turnover": [0.5, 1.0],
            "cost_bps_per_turnover": [0.0, 0.0],
            "cost": [0.0, 0.0],
            "net_return": [0.002, -0.001],
        }
    )
    expanded = PORTFOLIO_OPT.expand_cost_scenarios(base, [0.0, 5.0, 10.0])
    assert len(expanded) == 6
    assert set(expanded["cost_bps_per_turnover"]) == {0.0, 5.0, 10.0}
    ten = expanded[expanded["cost_bps_per_turnover"] == 10.0].reset_index(drop=True)
    assert ten.loc[0, "cost"] == 0.0005
    assert ten.loc[0, "net_return"] == 0.0015
    assert set(expanded["cost_scenario_source"]) == {"single_weight_path_expansion"}


def test_default_selection_tracks_separate_alpha_and_risk() -> None:
    tracks = {item["name"]: item for item in TRACKS._track_defaults()}
    assert tracks["alpha"]["portfolio_max_features"] == 40
    assert tracks["risk_regime"]["portfolio_max_features"] == 0
    assert tracks["cost_liquidity"]["portfolio_max_features"] == 0
    assert "realized_volatility" in tracks["risk_regime"]["label_families"]
    assert "execution_cost_proxy" in tracks["cost_liquidity"]["label_families"]


def test_risk_track_direction_is_not_trade_direction() -> None:
    data = pd.DataFrame(
        {
            "feature": ["full_factor__g01__w15m"] * 3,
            "factor_family": ["G"] * 3,
            "label_family": ["realized_volatility"] * 3,
            "horizon_bars": [15] * 3,
            "universe": ["final_trading_universe"] * 3,
            "rank_ic_method": ["minute_mean_cs_rank_ic"] * 3,
            "rank_ic_mean": [0.10, 0.12, 0.08],
            "coverage": [0.9, 0.9, 0.9],
            "rank_ic_positive_ratio": [0.7, 0.8, 0.7],
        }
    )
    track = next(item for item in TRACKS._track_defaults() if item["name"] == "risk_regime")
    result = TRACKS._track_summary(data, track, "final_trading_universe")
    assert len(result) == 1
    assert result.iloc[0]["direction_semantics"] == "target_association_not_trade_direction"


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
    assert P._selected_portfolio_proxy.__module__.endswith("v2_8_portfolio_optimization")


def test_pipeline_source_contains_training_freeze_guards() -> None:
    pathlib = __import__("pathlib").Path
    source = pathlib(P.__file__).read_text(encoding="utf-8")
    hardening = pathlib(
        __import__("nff_research.v2_8_runtime_hardening", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    source_contract = pathlib(
        __import__("nff_research.v2_8_source_contract", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    strict = pathlib(
        __import__("nff_research.v2_8_strict_training_window", fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    tracks = pathlib(TRACKS.__file__).read_text(encoding="utf-8")
    portfolio_opt = pathlib(PORTFOLIO_OPT.__file__).read_text(encoding="utf-8")
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
    assert "screen_fingerprint_dates" in source_contract
    assert "exact first" in strict and "chronological" in strict
    assert "only the alpha selection track may emit portfolio candidates" in tracks
    assert "target_association_not_trade_direction" in tracks
    assert "single_weight_path_expansion" in portfolio_opt
