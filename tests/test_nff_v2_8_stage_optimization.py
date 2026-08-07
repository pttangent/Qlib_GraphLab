from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from nff_research import v2_8_detailed_optimization as D
from nff_research import v2_8_portfolio_optimization as PORT
from nff_research import v2_8_runtime_hardening as H
from nff_research import v2_8_stage_contracts as CONTRACTS


def test_detailed_track_label_scopes_are_role_specific() -> None:
    index = pd.RangeIndex(3)
    labels = pd.DataFrame(
        {
            "return_open_to_open__h5": [1.0, 2.0, 3.0],
            "return_vwap_to_vwap__h5": [1.0, 2.0, 3.0],
            "realized_volatility__h5": [1.0, 2.0, 3.0],
            "liquidity_deterioration__h5": [1.0, 2.0, 3.0],
            "jump_tail_event__h5": [0.0, 1.0, 0.0],
            "execution_cost_proxy__h5": [1.0, 2.0, 3.0],
        },
        index=index,
    )
    masks = labels.notna()
    config = {"pipeline": {"detailed_internal": {}}}

    alpha, _ = D._subset_labels(labels, masks, D._track_label_families(config, "alpha"))
    risk, _ = D._subset_labels(labels, masks, D._track_label_families(config, "risk_regime"))
    cost, _ = D._subset_labels(labels, masks, D._track_label_families(config, "cost_liquidity"))

    assert set(alpha) == {"return_open_to_open__h5", "return_vwap_to_vwap__h5"}
    assert set(risk) == {
        "realized_volatility__h5",
        "liquidity_deterioration__h5",
        "jump_tail_event__h5",
    }
    assert set(cost) == {
        "execution_cost_proxy__h5",
        "liquidity_deterioration__h5",
    }


def test_candidate_track_groups_keep_alpha_risk_cost_order() -> None:
    candidates = pd.DataFrame(
        {
            "feature": ["risk", "alpha", "cost"],
            "selection_track": ["risk_regime", "alpha", "cost_liquidity"],
        }
    )
    groups = D._candidate_track_groups(candidates)
    assert [name for name, _ in groups] == ["alpha", "risk_regime", "cost_liquidity"]
    assert [part.iloc[0]["feature"] for _, part in groups] == ["alpha", "risk", "cost"]


def test_portfolio_cost_expansion_preserves_cost_major_order_and_math() -> None:
    base = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "gross_return": [0.01, -0.02],
            "turnover": [1.0, 0.5],
            "cost_bps_per_turnover": [0.0, 0.0],
            "cost": [0.0, 0.0],
            "net_return": [0.01, -0.02],
        }
    )
    out = PORT.expand_cost_scenarios(base, [0.0, 10.0])
    assert out["feature"].tolist() == ["a", "b", "a", "b"]
    assert out["cost_bps_per_turnover"].tolist() == [0.0, 0.0, 10.0, 10.0]
    np.testing.assert_allclose(out["cost"].to_numpy(), [0.0, 0.0, 0.001, 0.0005])
    np.testing.assert_allclose(out["net_return"].to_numpy(), [0.01, -0.02, 0.009, -0.0205])
    # Keep the historical output contract while the implementation itself is
    # vectorized; downstream audits should not change merely for a speedup.
    assert set(out["cost_scenario_source"]) == {"single_weight_path_expansion"}


def test_stage_worker_env_caps_hidden_blas_threads() -> None:
    config = {
        "atomic": {"intra_date_workers": 4},
        "pipeline": {
            "stage_intra_workers": {"detailed": 2, "portfolio": 2},
            "stage_blas_threads": {"detailed": 2, "portfolio": 1},
        },
    }
    assert H._stage_intra_workers(config, "detailed") == 2
    assert H._stage_intra_workers(config, "portfolio") == 2
    detailed = H._worker_env(config, "detailed")
    portfolio = H._worker_env(config, "portfolio")
    assert detailed["OMP_NUM_THREADS"] == "2"
    assert detailed["MKL_NUM_THREADS"] == "2"
    assert portfolio["OPENBLAS_NUM_THREADS"] == "1"
    assert portfolio["NUMEXPR_NUM_THREADS"] == "1"


def test_detailed_semantic_scope_is_hashed_but_worker_shape_is_not() -> None:
    p = SimpleNamespace(
        VERSION="2.8-pipelined-selection",
        C=SimpleNamespace(VERSION="2.7-real-schema-atomic"),
        V26=SimpleNamespace(FULL_FACTOR_NAMES=["f1"]),
    )
    base = {
        "run": {"start_date": "2026-01-02", "end_date": "2026-07-22", "min_cross_section_n": 30},
        "local_paths": {"warehouse_root": "D:/warehouse"},
        "study": {},
        "factor_resolution_policy": {},
        "atomic": {"factor_block_size": 8, "intra_date_workers": 4},
        "labels": {},
        "neutralization": {},
        "pipeline": {
            "detailed_internal": {
                "decile_tracks": ["alpha"],
                "track_label_families": {"alpha": ["return_vwap_to_vwap"]},
            },
            "stage_intra_workers": {"detailed": 2},
            "stage_blas_threads": {"detailed": 2},
            "stages": {"detailed": {"max_workers": 8, "estimated_worker_gb": 9}},
        },
    }
    operational = json.loads(json.dumps(base))
    operational["pipeline"]["stage_intra_workers"]["detailed"] = 1
    operational["pipeline"]["stage_blas_threads"]["detailed"] = 1
    operational["pipeline"]["stages"]["detailed"]["max_workers"] = 12
    semantic = json.loads(json.dumps(base))
    semantic["pipeline"]["detailed_internal"]["decile_tracks"] = []

    assert CONTRACTS.contract_payload(p, base, "detailed") == CONTRACTS.contract_payload(
        p, operational, "detailed"
    )
    assert CONTRACTS.contract_payload(p, base, "detailed") != CONTRACTS.contract_payload(
        p, semantic, "detailed"
    )


def test_campaign_uses_8x2_shape_for_detailed_and_portfolio() -> None:
    path = Path("configs/v2_8_pipelined_full_campaign.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["pipeline"]["stages"]["detailed"] == {
        "max_workers": 8,
        "estimated_worker_gb": 9,
    }
    assert config["pipeline"]["stages"]["portfolio"] == {
        "max_workers": 8,
        "estimated_worker_gb": 3,
    }
    assert config["pipeline"]["stage_intra_workers"]["detailed"] == 2
    assert config["pipeline"]["stage_intra_workers"]["portfolio"] == 2
    assert config["pipeline"]["portfolio_threads"] == 2
