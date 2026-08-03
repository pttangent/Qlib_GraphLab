from __future__ import annotations

import gc
import json
import math
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nff_research import v2_1_neutralized_runner as runner


def _symbols(n: int) -> np.ndarray:
    return np.array([f"SYM{i:04d}" for i in range(n)], dtype=object)


def test_symbol_fold_known_vectors():
    expected = {"AAPL": 4, "MSFT": 2, "NVDA": 4, "BRK.B": 0, "TSLA": 1}

    assert {symbol: runner.stable_symbol_fold(symbol, 5) for symbol in expected} == expected
    assert runner.stable_symbol_fold("  aapl ", 5) == expected["AAPL"]


def test_oof_ridge_heldout_label_perturbation_does_not_change_heldout_prediction():
    rng = np.random.default_rng(20260803)
    symbols = _symbols(150)
    x = pd.DataFrame(rng.normal(size=(len(symbols), 8)), columns=[f"x{i}" for i in range(8)])
    y = pd.Series(rng.normal(size=len(symbols)))
    folds = np.array([runner.stable_symbol_fold(symbol, 5) for symbol in symbols])

    original = runner._fixed_symbol_fold_oof_predict(x, y, symbols, folds=5, alpha=0.001, min_train_n=40)
    perturbed_y = y.copy()
    perturbed_y.loc[folds == 0] = perturbed_y.loc[folds == 0] * 1000.0 + 10000.0
    perturbed = runner._fixed_symbol_fold_oof_predict(x, perturbed_y, symbols, folds=5, alpha=0.001, min_train_n=40)

    assert original is not None
    assert perturbed is not None
    np.testing.assert_allclose(original[folds == 0], perturbed[folds == 0], atol=1e-12, rtol=0)
    assert not np.allclose(original[folds != 0], perturbed[folds != 0])


def test_ridge_fit_statistics_are_learned_from_training_rows_only():
    train_x = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
    train_y = np.array([-0.25, 0.0, 0.25])
    held_out_x = np.array([[1000.0, -500.0]])

    fitted = runner._ridge_fit(train_x, train_y, alpha=0.001)

    assert fitted is not None
    mean, std, _ = fitted
    np.testing.assert_allclose(mean, [3.0, 14.0])
    np.testing.assert_allclose(std, np.std(train_x, axis=0))
    prediction = runner._ridge_predict(held_out_x, fitted)
    assert np.isfinite(prediction).all()


def test_oof_high_dimensional_noise_does_not_reproduce_insample_overfit():
    rng = np.random.default_rng(71)
    symbols = _symbols(80)
    x = pd.DataFrame(rng.normal(size=(80, 70)))
    y = pd.Series(rng.normal(size=80))
    ranked_x = x.rank(method="average", pct=True)
    ranked_x -= ranked_x.mean(axis=0)
    ranked_y = y.rank(method="average", pct=True)
    ranked_y -= ranked_y.mean()

    in_sample, _, _ = runner._ridge_fit_predict(ranked_x.to_numpy(), ranked_y.to_numpy(), alpha=1e-9)
    oof = runner._fixed_symbol_fold_oof_predict(x, y, symbols, folds=5, alpha=0.001, min_train_n=40)

    assert oof is not None
    assert np.corrcoef(in_sample, ranked_y.to_numpy())[0, 1] > 0.90
    assert abs(np.corrcoef(oof, ranked_y.to_numpy())[0, 1]) < 0.20


def _rank_fixture(sizes: list[int]) -> tuple[pd.Series, pd.Series]:
    zero_corr_permutations = {
        4: [2, 4, 1, 3],
        8: [1, 4, 6, 7, 8, 5, 3, 2],
    }
    pred_parts = []
    target_parts = []
    index_parts = []
    for minute, size in enumerate(sizes):
        stamp = pd.Timestamp("2026-07-06T13:30:00Z") + pd.Timedelta(minutes=15 * minute)
        index = pd.MultiIndex.from_arrays(
            [[stamp] * size, [f"M{minute}_{i}" for i in range(size)]],
            names=["datetime", "instrument"],
        )
        index_parts.extend(index.tolist())
        pred_parts.extend(range(1, size + 1))
        target_parts.extend(zero_corr_permutations[size])
    index = pd.MultiIndex.from_tuples(index_parts, names=["datetime", "instrument"])
    return pd.Series(pred_parts, index=index, dtype=float), pd.Series(target_parts, index=index, dtype=float)


def test_pooled_prediction_ic_has_no_cross_section_size_bias():
    pred_a, target_a = _rank_fixture([4, 4])
    pred_b, target_b = _rank_fixture([4, 8])

    corr_a, n_a = runner._pooled_demeaned_pct_rank_corr(pred_a, target_a, min_n=1)
    corr_b, n_b = runner._pooled_demeaned_pct_rank_corr(pred_b, target_b, min_n=1)

    assert n_a == 8
    assert n_b == 12
    assert corr_a == pytest.approx(0.0, abs=1e-12)
    assert corr_b == pytest.approx(0.0, abs=1e-12)
    assert corr_a == pytest.approx(corr_b, abs=1e-12)


def test_oof_metric_formulas_and_delta_signs():
    index = pd.MultiIndex.from_arrays(
        [
            [pd.Timestamp("2026-07-06T13:30:00Z")] * 4,
            ["A", "B", "C", "D"],
        ],
        names=["datetime", "instrument"],
    )
    target = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    previous = pd.Series([-0.375, 0.125, -0.125, 0.375], index=index)
    current = pd.Series([-0.375, -0.125, 0.125, 0.375], index=index)

    previous_metrics = runner._oof_prediction_metrics(previous, target, min_n=1)
    current_metrics = runner._oof_prediction_metrics(current, target, min_n=1)
    deltas = runner._oof_metric_deltas(current_metrics, previous_metrics, current - previous, target, min_n=1)

    assert current_metrics["mean_minute_pred_rank_ic"] == pytest.approx(1.0)
    assert current_metrics["pooled_demeaned_pct_rank_pred_ic"] == pytest.approx(1.0)
    assert current_metrics["mean_minute_r2"] == pytest.approx(1.0)
    assert current_metrics["mean_minute_mse"] == pytest.approx(0.0)
    assert previous_metrics["mean_minute_pred_rank_ic"] == pytest.approx(0.8)
    assert previous_metrics["mean_minute_r2"] == pytest.approx(0.6)
    assert previous_metrics["mean_minute_mse"] == pytest.approx(0.03125)
    assert deltas["delta_mean_minute_pred_rank_ic"] == pytest.approx(0.2)
    assert deltas["delta_pooled_pred_rank_ic"] == pytest.approx(0.2)
    assert deltas["delta_mean_minute_r2"] == pytest.approx(0.4)
    assert deltas["mse_improvement_vs_previous"] == pytest.approx(0.03125)
    assert deltas["incremental_oof_prediction_rank_ic"] == pytest.approx(0.316227766016838)


def _bundle_screen_inputs(n_symbols: int = 75, minutes: int = 2):
    rng = np.random.default_rng(123)
    symbols = _symbols(n_symbols)
    timestamps = [pd.Timestamp("2026-07-06T13:30:00Z") + pd.Timedelta(minutes=15 * i) for i in range(minutes)]
    index = pd.MultiIndex.from_product([timestamps, symbols], names=["datetime", "instrument"])
    n = len(index)
    features = pd.DataFrame(
        {
            "traditional__momentum_15m": rng.normal(size=n),
            "minute_nvg__price_path_15m_range": rng.normal(size=n),
            "trade_nvg__trade_price_path_300s_range": rng.normal(size=n),
            "hawkes_lite__hawkes_total_intensity": rng.normal(size=n),
            "minute_nvg__price_nvg_30m_top_bottom_asymmetry": rng.normal(size=n),
            "trade_nvg__trade_flow_path_300s_terminal_position": rng.normal(size=n),
            "hawkes_derived__hawkes_signed_pressure": rng.normal(size=n),
            "bars_1m__close": 20.0,
            "bars_1m__volume": 1000.0,
            "trades_1m_core__trade_count": 10.0,
        },
        index=index,
    )
    # Force a largest-step common sample rather than step-specific complete cases.
    features.iloc[0, features.columns.get_loc("minute_nvg__price_path_15m_range")] = np.nan
    features.iloc[1, features.columns.get_loc("trade_nvg__trade_price_path_300s_range")] = np.nan
    controls = pd.DataFrame(
        {
            "control__adv20_top1000": True,
            "control__adv20_days": 20.0,
            "control__log_adv20": math.log1p(1_000_000.0),
        },
        index=index,
    )
    labels = pd.DataFrame({"return_vwap_to_vwap__h15": rng.normal(size=n)}, index=index)
    masks = {"return_vwap_to_vwap__h15": pd.Series(True, index=index)}
    return features, labels, masks, controls


def test_bundle_oof_screen_uses_identical_common_sample_indices_minutes_and_folds():
    features, labels, masks, controls = _bundle_screen_inputs()

    result = runner.bundle_incremental_model_screen(features, labels, masks, controls, "2026-07-06", min_n=30)

    assert len(result) == 4
    assert set(result["fold_count"]) == {5}
    assert result["common_feature_count"].nunique() == 1
    assert result["sample_count"].nunique() == 1
    assert result["sampled_minutes"].nunique() == 1
    assert result["sample_identity_hash"].nunique() == 1
    assert result["fold_identity_hash"].nunique() == 1


def test_bundle_oof_screen_discards_minute_when_any_step_fails(monkeypatch: pytest.MonkeyPatch):
    features, labels, masks, controls = _bundle_screen_inputs()
    original = runner._fixed_symbol_fold_oof_predict
    call_count = 0

    def fail_largest_step_once(x, y, symbols, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == len(runner.BUNDLE_MODEL_STEPS):
            return None
        return original(x, y, symbols, **kwargs)

    monkeypatch.setattr(runner, "_fixed_symbol_fold_oof_predict", fail_largest_step_once)
    result = runner.bundle_incremental_model_screen(features, labels, masks, controls, "2026-07-06", min_n=30)

    assert len(result) == 4
    assert set(result["sampled_minutes"]) == {1}
    assert result["sample_count"].nunique() == 1


def test_contrarian_q05_weights_reverse_signal_and_select_exact_tails():
    index = pd.MultiIndex.from_arrays(
        [[pd.Timestamp("2026-07-06T13:30:00Z")] * 100, _symbols(100)],
        names=["datetime", "instrument"],
    )
    block = pd.DataFrame({"signal": -np.arange(100, dtype=float)}, index=index)

    weights = runner._portfolio_weights(block, "signal", quantile=0.05)

    assert len(weights) == 10
    assert set(weights[weights > 0].index.get_level_values("instrument")) == set(_symbols(5))
    assert set(weights[weights < 0].index.get_level_values("instrument")) == set(_symbols(100)[-5:])


def test_same_sleeve_turnover_ignores_other_active_sleeves():
    current = pd.Series({"A": 0.5, "B": -0.5})
    same_sleeve_previous = pd.Series({"A": 0.25, "C": -0.25})
    other_active_sleeve = pd.Series({"D": 0.5, "E": -0.5})

    turnover = runner._same_sleeve_turnover(current, same_sleeve_previous)

    assert turnover == pytest.approx(1.0)
    assert turnover != runner._same_sleeve_turnover(current, other_active_sleeve)


def test_calendar_sleeve_identity_survives_skipped_rebalance():
    features, labels, masks, controls = _bundle_screen_inputs(n_symbols=100, minutes=3)
    old_times = features.index.get_level_values("datetime").unique()
    new_times = [old_times[0], old_times[0] + pd.Timedelta(minutes=30), old_times[0] + pd.Timedelta(minutes=60)]
    remap = dict(zip(old_times, new_times))
    for frame in (features, labels, controls):
        frame.index = pd.MultiIndex.from_arrays(
            [frame.index.get_level_values("datetime").map(remap), frame.index.get_level_values("instrument")],
            names=["datetime", "instrument"],
        )
    masks = {column: pd.Series(True, index=labels.index) for column in labels.columns}
    labels.columns = ["return_vwap_to_vwap__h60"]
    masks = {"return_vwap_to_vwap__h60": pd.Series(True, index=labels.index)}
    middle = labels.index.get_level_values("datetime") == new_times[1]
    labels.loc[middle, "return_vwap_to_vwap__h60"] = np.nan

    result = runner.staggered_portfolio_proxy(features, labels, masks, controls, "2026-07-06", min_n=30)
    cohorts = result[
        (result["portfolio_variant"] == "contrarian_q05_30m")
        & (result["feature"] == "traditional__momentum_15m")
        & (result["horizon_bars"] == 60)
    ].sort_values("datetime")

    assert list(cohorts["sleeve_id"]) == [0, 0]


@pytest.mark.parametrize(
    ("horizon", "rebalance", "expected"),
    [
        (15, 15, "primary"),
        (15, 30, "diagnostic"),
        (30, 15, "primary"),
        (30, 30, "primary"),
        (30, 60, "diagnostic"),
        (60, 15, "diagnostic"),
        (60, 30, "primary"),
        (60, 60, "primary"),
        (120, 15, "diagnostic"),
        (120, 30, "diagnostic"),
        (120, 60, "primary"),
    ],
)
def test_portfolio_applicability_contract(horizon: int, rebalance: int, expected: str):
    assert runner._portfolio_variant_applicability(horizon, rebalance) == expected


def test_hawkes_gate_pair_uses_shared_sample_and_deterministic_top_20_percent():
    symbols = np.array(["Z", "A", "B", "C", "D", "E", "F", "G", "H", "MISS"])
    index = pd.MultiIndex.from_arrays(
        [[pd.Timestamp("2026-07-06T13:30:00Z")] * len(symbols), symbols],
        names=["datetime", "instrument"],
    )
    block = pd.DataFrame(
        {
            "signal": np.arange(len(symbols), dtype=float),
            "label": np.arange(len(symbols), dtype=float),
            "hawkes": [10, 10, 9, 8, 7, 6, 5, 4, 3, np.nan],
            "adv20": [1_000_000.0] * len(symbols),
        },
        index=index,
    )

    pre_gate, eligible, stats = runner._hawkes_gate_pair(block, "hawkes", "adv20", exclude_fraction=0.20)

    assert list(pre_gate.index.get_level_values("instrument")) == list(symbols[:-1])
    assert set(pre_gate.index) - set(eligible.index) == {
        (pd.Timestamp("2026-07-06T13:30:00Z"), "A"),
        (pd.Timestamp("2026-07-06T13:30:00Z"), "Z"),
    }
    assert stats["pre_gate_count"] == 9
    assert stats["eligible_count"] == 7
    assert stats["gate_kept_ratio"] == pytest.approx(7 / 9)
    assert stats["eligible_adv20_kept_ratio"] == pytest.approx(7 / 9)


def test_hawkes_gate_catalog_covers_intensity_shock_endogeneity_exogeneity_and_persistence():
    gate_columns = {spec["column"] for spec in runner.HAWKES_GATE_SPECS}

    assert gate_columns == {
        "hawkes_lite__hawkes_total_intensity",
        "hawkes_lite__hawkes_shock_score_300s",
        "hawkes_derived__hawkes_endogenous_shock_300s",
        "hawkes_derived__hawkes_exogenous_shock_300s",
        "hawkes_derived__hawkes_persistence",
    }
    assert all(spec["exclude_fraction"] == pytest.approx(0.20) for spec in runner.HAWKES_GATE_SPECS)


def test_turnover_controlled_weights_keep_minimum_hold_and_cap_replacements():
    symbols = _symbols(20)
    index = pd.MultiIndex.from_arrays(
        [[pd.Timestamp("2026-07-06T13:30:00Z")] * len(symbols), symbols],
        names=["datetime", "instrument"],
    )
    block = pd.DataFrame({"signal": np.arange(len(symbols), dtype=float)}, index=index)
    previous = pd.Series({symbols[0]: 0.25, symbols[-1]: -0.25})
    previous.index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2026-07-06T13:00:00Z")], previous.index], names=["datetime", "instrument"]
    )
    previous_signal = pd.Series({symbols[0]: 0.9, symbols[-1]: 0.1})
    previous_hold = {symbols[0]: 1, symbols[-1]: 1}

    weights, state = runner._turnover_controlled_weights(
        block,
        "signal",
        previous,
        previous_signal,
        previous_hold,
        quantile=0.05,
        buffer_quantile=0.10,
        min_hold_periods=2,
        signal_change_threshold=0.10,
        max_replacement_fraction=0.50,
    )

    held_symbols = set(weights.index.get_level_values("instrument"))
    assert {symbols[0], symbols[-1]}.issubset(held_symbols)
    assert state["replacement_count"] <= state["replacement_budget"]
    assert state["minimum_hold_retained_count"] == 2


def test_turnover_control_forces_exit_when_previous_symbol_is_not_currently_eligible():
    symbols = _symbols(20)
    index = pd.MultiIndex.from_arrays(
        [[pd.Timestamp("2026-07-06T13:30:00Z")] * len(symbols), symbols],
        names=["datetime", "instrument"],
    )
    block = pd.DataFrame({"signal": np.arange(len(symbols), dtype=float)}, index=index)
    previous_index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-07-06T13:00:00Z"), "MISSING")], names=["datetime", "instrument"]
    )
    previous = pd.Series([0.5], index=previous_index)
    previous_signal = pd.Series({"MISSING": 0.9})

    weights, _ = runner._turnover_controlled_weights(
        block, "signal", previous, previous_signal, {"MISSING": 1},
        quantile=0.05, buffer_quantile=0.10, min_hold_periods=2,
        signal_change_threshold=0.10, max_replacement_fraction=0.50,
    )

    assert "MISSING" not in set(weights.index.get_level_values("instrument"))


def test_future_bipower_uses_only_adjacent_returns_inside_future_window():
    index = pd.MultiIndex.from_arrays(
        [[pd.Timestamp("2026-07-06T13:30:00Z")] * 6, ["A"] * 6],
        names=["datetime", "instrument"],
    )
    returns = pd.Series([1.0, 2.0, 3.0, 5.0, 7.0, 11.0], index=index)

    bpv = runner._future_bipower_variation(returns, horizon=3)

    # At t=0 the future returns are 2,3,5; BPV is 2*3 + 3*5, never 1*2.
    assert bpv.iloc[0] == pytest.approx(21.0)


def test_label_dependency_diagnostic_is_minute_mean_and_corrected_pooled():
    index = pd.MultiIndex.from_arrays(
        [
            [pd.Timestamp("2026-07-06T13:30:00Z")] * 4
            + [pd.Timestamp("2026-07-06T13:31:00Z")] * 4,
            ["A", "B", "C", "D"] * 2,
        ],
        names=["datetime", "instrument"],
    )
    labels = pd.DataFrame(
        {
            "realized_volatility__h15": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            "jump_tail_event__h15": [1.0, 2.0, 3.0, 4.0, 40.0, 30.0, 20.0, 10.0],
            "execution_cost_proxy__h15": [1.0, 2.0, 3.0, 4.0, 40.0, 30.0, 20.0, 10.0],
        },
        index=index,
    )

    result = runner.label_dependency_diagnostics(labels, "2026-07-06")

    assert {"minute_mean_rank_correlation", "pooled_demeaned_pct_rank_correlation", "positive_ratio"}.issubset(result.columns)
    jump = result[result["diagnostic"] == "jump_vs_realized_volatility"].iloc[0]
    assert jump["minute_mean_rank_correlation"] == pytest.approx(0.0)
    assert jump["pooled_demeaned_pct_rank_correlation"] == pytest.approx(0.0)


def test_label_evidence_roles_keep_jump_and_cost_diagnostic_but_screen_them_oof():
    assert set(runner.BUNDLE_MODEL_LABEL_ROLES) == set(runner.LABEL_FAMILIES)
    assert runner.BUNDLE_MODEL_LABEL_ROLES["jump_tail_event"] == "diagnostic_jump_orthogonality"
    assert runner.BUNDLE_MODEL_LABEL_ROLES["execution_cost_proxy"] == "diagnostic_mechanical_proxy"


def test_jump_tail_contract_uses_bipower_jump_variation_not_raw_future_maximum():
    contract = runner.LABEL_CONTRACTS["jump_tail_event__hN"]

    assert "bipower" in contract.lower()
    assert "max absolute" not in contract.lower()


def test_vwap_portfolio_is_primary_and_emits_gate_capacity_fields():
    features, labels, masks, controls = _bundle_screen_inputs(n_symbols=100, minutes=2)

    result = runner.staggered_portfolio_proxy(features, labels, masks, controls, "2026-07-06", min_n=30)
    paired = result[result["gate_pair_id"].notna()]

    assert not result.empty
    assert set(result["execution_role"]) == {"primary"}
    assert {"ungated_shared_sample", "exclude_top20"}.issubset(set(paired["gate_mode"]))
    assert {
        "variant_applicability",
        "pre_gate_count",
        "eligible_count",
        "selected_count",
        "gate_kept_ratio",
        "pre_gate_adv20_sum",
        "eligible_adv20_sum",
        "selected_adv20_sum",
        "eligible_adv20_kept_ratio",
        "cost",
    }.issubset(result.columns)


def test_portfolio_deduplicates_feature_that_is_also_hawkes_gate():
    features, labels, masks, controls = _bundle_screen_inputs(n_symbols=100, minutes=2)
    features["hawkes_derived__hawkes_exogenous_shock_300s"] = np.linspace(0.1, 1.0, len(features))

    result = runner.staggered_portfolio_proxy(features, labels, masks, controls, "2026-07-06", min_n=30)

    assert not result.empty


def test_portfolio_variant_catalog_is_configurable(monkeypatch: pytest.MonkeyPatch):
    features, labels, masks, controls = _bundle_screen_inputs(n_symbols=100, minutes=2)
    monkeypatch.setattr(runner, "ENABLED_PORTFOLIO_VARIANTS", ["contrarian_q05_30m_turnover_controlled"])

    result = runner.staggered_portfolio_proxy(features, labels, masks, controls, "2026-07-06", min_n=30)

    assert set(result["portfolio_variant"]) == {"contrarian_q05_30m_turnover_controlled"}


def test_v2_3_config_freezes_oof_contract():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "v2_1_neutralized_full.yaml"
    config = runner.load_yaml_config(str(config_path))

    assert config["research_version"] == "2.3"
    assert config["base_runner_version"] == "2.1"
    assert {key: config["incremental_validation"][key] for key in [
        "method", "folds", "fold_algorithm", "ridge_alpha", "min_train_n", "sample_policy"
    ]} == {
        "method": "fixed_symbol_fold_cross_sectional_oof_ridge",
        "folds": 5,
        "fold_algorithm": "sha256_normalized_symbol_first8_big_endian_modulo",
        "ridge_alpha": 0.001,
        "min_train_n": 40,
        "sample_policy": "largest_step_common_complete_case_all_steps_or_drop_minute",
    }


def test_effective_config_and_worker_command_carry_oof_parameters(tmp_path: Path):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "v2_1_neutralized_full.yaml"
    config = runner.load_yaml_config(str(config_path))
    args = Namespace(
        run_id=None,
        start_date="2026-01-02",
        end_date="2026-07-22",
        horizons=[15, 30, 60, 120],
        parallel=12,
        max_parallel=16,
        min_parallel=4,
        target_cpu=90.0,
        memory_high_water=78.0,
        memory_min_available_gb=32.0,
        disk_free_floor_gb=50.0,
        launch_batch_size=2,
        retries=1,
        min_cs_n=30,
        allow_mixed_contracts=False,
        controls_path=str(tmp_path / "controls.parquet"),
        rebalance_minutes=15,
        cost_bps_per_turnover=1.0,
        oof_folds=3,
        oof_ridge_alpha=9.0,
        oof_min_train_n=10,
        contract_hash="contract-hash",
        config=str(config_path),
    )

    effective = runner.apply_config_args(args, config, explicit_flags=set())
    command = runner.worker_command(args, tmp_path, Path(args.controls_path), "2026-06-01")

    assert effective["incremental_validation"]["folds"] == 5
    assert effective["incremental_validation"]["ridge_alpha"] == 0.001
    assert effective["incremental_validation"]["min_train_n"] == 40
    assert command[command.index("--oof-folds") + 1] == "5"
    assert command[command.index("--oof-ridge-alpha") + 1] == "0.001"
    assert command[command.index("--oof-min-train-n") + 1] == "40"


def test_run_contract_hash_changes_with_oof_policy_and_rejects_old_root(tmp_path: Path):
    controls = tmp_path / "controls.parquet"
    controls.write_bytes(b"controls")
    base = {
        "research_version": "2.3",
        "base_runner_version": "2.1",
        "session": {"timezone": "America/New_York"},
        "incremental_validation": {
            "folds": 5,
            "ridge_alpha": 0.001,
            "min_train_n": 40,
            "fold_algorithm": runner.OOF_FOLD_ALGORITHM,
            "sample_policy": runner.OOF_SAMPLE_POLICY,
        },
        "portfolio_policy": {"primary_execution_label": "return_vwap_to_vwap"},
    }
    _, hash_a = runner.build_run_contract(tmp_path / "a", controls, base)
    changed = json.loads(json.dumps(base))
    changed["incremental_validation"]["folds"] = 4
    _, hash_b = runner.build_run_contract(tmp_path / "b", controls, changed)
    assert hash_a != hash_b

    old_root = tmp_path / "old"
    old_root.mkdir()
    (old_root / "run_contract.json").write_text(json.dumps({"run_contract_hash": "v2.1-old"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="run contract mismatch"):
        runner.build_run_contract(old_root, controls, base)


def test_v2_3_research_contract_is_explicit(tmp_path: Path):
    runner.write_research_contract(tmp_path)

    contract = json.loads((tmp_path / "research_contract_v2_3.json").read_text(encoding="utf-8"))
    assert contract["research_version"] == "2.3"
    assert contract["base_runner_version"] == "2.1"
    assert contract["incremental_model_contract"]["folds"] == 5
    assert contract["incremental_model_contract"]["ridge_alpha"] == 0.001
    assert contract["incremental_model_contract"]["min_train_n"] == 40
    assert contract["portfolio_proxy_contract"]["primary_execution_label"] == "return_vwap_to_vwap"


def test_research_contract_uses_effective_oof_overrides(tmp_path: Path):
    (tmp_path / "effective_config.json").write_text(
        json.dumps(
            {
                "research_version": "2.3",
                "base_runner_version": "2.1",
                "incremental_validation": {
                    "folds": 4,
                    "ridge_alpha": 0.25,
                    "min_train_n": 55,
                    "fold_algorithm": runner.OOF_FOLD_ALGORITHM,
                    "sample_policy": runner.OOF_SAMPLE_POLICY,
                },
                "portfolio_policy": {"primary_execution_label": "return_vwap_to_vwap"},
            }
        ),
        encoding="utf-8",
    )

    runner.write_research_contract(tmp_path)

    contract = json.loads((tmp_path / "research_contract_v2_3.json").read_text(encoding="utf-8"))
    assert contract["incremental_model_contract"]["folds"] == 4
    assert contract["incremental_model_contract"]["ridge_alpha"] == 0.25
    assert contract["incremental_model_contract"]["min_train_n"] == 55


def _write_v2_3_aggregate_inputs(root: Path) -> tuple[list[float], list[float]]:
    metric_values = [0.01, 0.02, 0.00, 0.03, 0.015, 0.025]
    delta_values = [0.002, 0.004, -0.001, 0.003, 0.001, 0.005]
    for day, (metric, delta) in enumerate(zip(metric_values, delta_values), start=1):
        trade_date = f"2026-06-{day:02d}"
        out = root / "02_neutralized_factor_diagnostics" / f"date={trade_date}"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "trade_date": trade_date,
                    "universe": "liquid_common_adv20_top1000",
                    "label_family": "return_vwap_to_vwap",
                    "horizon_bars": 30,
                    "step": "traditional_plus_minute_nvg",
                    "previous_step": "traditional",
                    "feature_count": 2,
                    "common_feature_count": 4,
                    "fold_count": 5,
                    "sampled_minutes": 20,
                    "sample_count": 1500,
                    "sample_identity_hash": "same-sample",
                    "fold_identity_hash": "same-folds",
                    "mean_minute_pred_rank_ic": metric,
                    "pooled_demeaned_pct_rank_pred_ic": metric + 0.001,
                    "mean_minute_r2": metric / 2,
                    "mean_minute_mse": 0.1 - metric,
                    "delta_mean_minute_pred_rank_ic": delta,
                    "delta_pooled_pred_rank_ic": delta + 0.0005,
                    "delta_mean_minute_r2": delta / 2,
                    "mse_improvement_vs_previous": delta / 3,
                    "incremental_oof_prediction_rank_ic": delta * 2,
                    "contract": "test",
                }
            ]
        ).to_parquet(out / "bundle_oof_incremental_screen.parquet", index=False)

        portfolio_rows = []
        for label_family, execution_role in [
            ("return_vwap_to_vwap", "primary"),
            ("return_open_to_open", "diagnostic"),
        ]:
            for gate_mode, gross, turnover, cost, selected, selected_adv in [
                ("ungated_shared_sample", 0.0004, 1.2, 0.00012, 10, 8_000_000.0),
                ("exclude_top20", 0.0005, 1.0, 0.00010, 8, 7_000_000.0),
            ]:
                portfolio_rows.append(
                    {
                        "trade_date": trade_date,
                        "datetime": pd.Timestamp(f"{trade_date}T13:30:00Z"),
                        "universe": "liquid_common_adv20_top1000",
                        "feature": "traditional__momentum_15m",
                        "bundle": "TRAD",
                        "label_family": label_family,
                        "horizon_bars": 30,
                        "portfolio_variant": f"pair_{gate_mode}",
                        "signal_direction": "contrarian",
                        "quantile": 0.05,
                        "hawkes_liquidity_gate": gate_mode == "exclude_top20",
                        "gate_pair_id": "pair-1",
                        "gate_mode": gate_mode,
                        "execution_role": execution_role,
                        "variant_applicability": "primary",
                        "rebalance_minutes": 30,
                        "sleeve_count": 1,
                        "sleeve_id": 0,
                        "portfolio_accounting": "same_sleeve_turnover",
                        "long_count": selected // 2,
                        "short_count": selected // 2,
                        "selected_count": selected,
                        "pre_gate_count": 100,
                        "eligible_count": 80 if gate_mode == "exclude_top20" else 100,
                        "gate_kept_ratio": 0.8 if gate_mode == "exclude_top20" else 1.0,
                        "pre_gate_adv20_sum": 100_000_000.0,
                        "eligible_adv20_sum": 80_000_000.0 if gate_mode == "exclude_top20" else 100_000_000.0,
                        "eligible_adv20_kept_ratio": 0.8 if gate_mode == "exclude_top20" else 1.0,
                        "selected_adv20_sum": selected_adv,
                        "gross_return": gross,
                        "turnover": turnover,
                        "cost_bps_per_turnover": 1.0,
                        "cost": cost,
                        "net_return": gross - cost,
                    }
                )
        pd.DataFrame(portfolio_rows).to_parquet(out / "staggered_portfolio_proxy.parquet", index=False)
    return metric_values, delta_values


def test_aggregate_emits_v2_3_primary_diagnostic_and_gate_pair_outputs(tmp_path: Path):
    _write_v2_3_aggregate_inputs(tmp_path)

    runner.aggregate(tmp_path)

    aggregate = tmp_path / "02_neutralized_factor_diagnostics" / "_aggregate"
    expected = [
        tmp_path / "research_contract_v2_3.json",
        aggregate / "final_report_v2_3.md",
        aggregate / "staggered_portfolio_proxy_primary_vwap_overall.csv",
        aggregate / "staggered_portfolio_proxy_primary_vwap_overall.parquet",
        aggregate / "staggered_portfolio_proxy_diagnostic_open_overall.csv",
        aggregate / "staggered_portfolio_proxy_diagnostic_open_overall.parquet",
        aggregate / "hawkes_gate_pair_comparison_overall.csv",
        aggregate / "hawkes_gate_pair_comparison_overall.parquet",
    ]
    assert all(path.exists() for path in expected)
    comparison = pd.read_parquet(aggregate / "hawkes_gate_pair_comparison_overall.parquet")
    assert {
        "delta_gross_return",
        "delta_turnover",
        "delta_cost",
        "delta_net_return",
        "delta_selected_count",
        "delta_selected_adv20_sum",
    }.issubset(comparison.columns)


def test_aggregate_hac_uses_daily_oof_metric_and_delta_series_directly(tmp_path: Path):
    metric_values, delta_values = _write_v2_3_aggregate_inputs(tmp_path)

    runner.aggregate(tmp_path)

    aggregate = tmp_path / "02_neutralized_factor_diagnostics" / "_aggregate"
    overall = pd.read_parquet(aggregate / "bundle_oof_incremental_screen_overall.parquet").iloc[0]
    metric_hac = runner._hac_tstat(pd.Series(metric_values))[0]
    delta_hac = runner._hac_tstat(pd.Series(delta_values))[0]
    assert overall["hac_tstat_mean_minute_pred_rank_ic"] == pytest.approx(metric_hac)
    assert overall["hac_tstat_delta_mean_minute_pred_rank_ic"] == pytest.approx(delta_hac)


def test_gate_comparison_uses_only_exactly_paired_cohorts(tmp_path: Path):
    _write_v2_3_aggregate_inputs(tmp_path)
    first = tmp_path / "02_neutralized_factor_diagnostics" / "date=2026-06-01" / "staggered_portfolio_proxy.parquet"
    portfolio = pd.read_parquet(first)
    extra = portfolio.iloc[[0]].copy()
    extra["datetime"] = extra["datetime"] + pd.Timedelta(minutes=15)
    extra["gross_return"] = 1.0
    extra["net_return"] = 1.0
    pd.concat([portfolio, extra], ignore_index=True).to_parquet(first, index=False)

    runner.aggregate(tmp_path)

    comparison = pd.read_parquet(
        tmp_path / "02_neutralized_factor_diagnostics" / "_aggregate" / "hawkes_gate_pair_comparison_overall.parquet"
    )
    assert set(np.round(comparison["delta_gross_return"], 10)) == {0.0001}


def test_resource_snapshot_reports_worker_rss(tmp_path: Path):
    snapshot = runner.resource_snapshot(tmp_path)

    assert {"worker_process_count", "worker_rss_total_gb", "worker_rss_max_gb"}.issubset(snapshot)
    assert snapshot["worker_process_count"] >= 0
    assert snapshot["worker_rss_total_gb"] >= 0
    assert snapshot["worker_rss_max_gb"] >= 0


def test_adaptive_parallel_target_requires_safe_streak_and_respects_worker_rss_projection():
    safe_resources = {
        "cpu_percent": 60.0,
        "memory_percent": 50.0,
        "memory_available_gb": 48.0,
        "worker_process_count": 12,
        "worker_rss_total_gb": 48.0,
    }
    unchanged, reason = runner.adaptive_parallel_target(
        safe_resources, current_parallel=12, min_parallel=4, max_parallel=16, target_cpu=90.0,
        memory_high_water=78.0, memory_min_available_gb=32.0, safe_streak=2,
    )
    assert unchanged == 12
    assert reason == "awaiting_safe_streak"

    grown, reason = runner.adaptive_parallel_target(
        safe_resources, current_parallel=12, min_parallel=4, max_parallel=16, target_cpu=90.0,
        memory_high_water=78.0, memory_min_available_gb=32.0, safe_streak=3,
    )
    assert grown == 13
    assert reason == "cpu_headroom_projected_worker_rss"

    pressured, reason = runner.adaptive_parallel_target(
        {**safe_resources, "memory_percent": 80.0}, current_parallel=12, min_parallel=4, max_parallel=16,
        target_cpu=90.0, memory_high_water=78.0, memory_min_available_gb=32.0, safe_streak=3,
    )
    assert pressured == 6
    assert reason == "memory_guard"


def test_oof_subprocess_does_not_retain_minute_arrays():
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import gc
import json
import os
import numpy as np
import pandas as pd
import psutil
from nff_research import v2_1_neutralized_runner as runner

rng = np.random.default_rng(20260803)
symbols = np.array([f'SYM{i:04d}' for i in range(1000)], dtype=object)
process = psutil.Process(os.getpid())
start = process.memory_info().rss
for _ in range(20):
    x = pd.DataFrame(rng.normal(size=(1000, 55)))
    y = pd.Series(rng.normal(size=1000))
    prediction = runner._fixed_symbol_fold_oof_predict(x, y, symbols)
    assert prediction is not None
    del x, y, prediction
    gc.collect()
end = process.memory_info().rss
print(json.dumps({'growth_mb': (end - start) / 1024**2}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    growth_mb = json.loads(completed.stdout.strip().splitlines()[-1])["growth_mb"]
    assert growth_mb < 128.0
