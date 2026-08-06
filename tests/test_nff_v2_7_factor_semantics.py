from __future__ import annotations

import numpy as np
import pandas as pd

from nff_research import v2_7_launch as LAUNCH
from nff_research import full_factor_engine as FF
from nff_research import v2_6_full_defined_campaign as FULL
from nff_research import v2_7_atomic_campaign as ATOMIC
from nff_research import v2_4_optimized_runner as V24


def _cross_section(values: dict[str, list[float]]) -> pd.DataFrame:
    instruments = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    timestamp = pd.Timestamp("2026-01-02 15:00:00", tz="UTC")
    index = pd.MultiIndex.from_arrays(
        [instruments, [timestamp] * len(instruments)],
        names=["instrument", "datetime"],
    )
    return pd.DataFrame(values, index=index)


def test_vectorized_ranked_ic_matches_reference_statistics() -> None:
    times = pd.date_range("2026-01-02 15:00:00", periods=4, freq="1min", tz="UTC")
    index = pd.MultiIndex.from_product(
        [["AAA", "BBB", "CCC", "DDD"], times], names=["instrument", "datetime"]
    )
    frame = pd.DataFrame(
        {
            "x": np.arange(len(index), dtype="float64"),
            "z": np.linspace(1.0, 4.0, len(index)),
        },
        index=index,
    )
    frame.iloc[2, 0] = np.nan
    label = pd.Series(np.sin(np.arange(len(index))), index=index, name="label")
    reference = V24.ranked_ic_stats_once(frame, label, ["x", "z"], min_n=2)
    optimized = ATOMIC._ranked_ic_stats_vectorized(frame, label, ["x", "z"], min_n=2)
    for feature in ("x", "z"):
        for key in ("ic_minutes", "ic_count", "rank_ic_mean", "rank_ic_std", "rank_ic_positive_ratio"):
            assert np.isclose(
                reference[0][feature][key], optimized[0][feature][key], equal_nan=True
            )
        assert np.isclose(
            reference[1][feature]["rank_ic_mean"],
            optimized[1][feature]["rank_ic_mean"],
            equal_nan=True,
        )


def test_full_label_builder_normalizes_timestamp_index_alias() -> None:
    index = pd.MultiIndex.from_arrays(
        [["AAA", "BBB"], [pd.Timestamp("2026-01-02 15:00:00", tz="UTC")] * 2],
        names=["symbol", "timestamp"],
    )
    frame = pd.DataFrame(
        {
            "bars_1m__open": [100.0, 101.0],
            "bars_1m__close": [100.0, 101.0],
            "bars_1m__vwap": [100.0, 101.0],
        },
        index=index,
    )
    normalized = FULL._ensure_research_index(frame)
    assert normalized.index.names == ["instrument", "datetime"]


def test_full_label_builder_restores_unnamed_loader_key() -> None:
    index = pd.MultiIndex.from_arrays(
        [["AAA", "BBB"], [pd.Timestamp("2026-01-02 15:00:00", tz="UTC")] * 2],
        names=[None, None],
    )
    frame = pd.DataFrame({"bars_1m__close": [100.0, 101.0]}, index=index)
    normalized = FULL._ensure_research_index(frame)
    assert normalized.index.names == ["instrument", "datetime"]


def test_factor_progress_counts_family_window_manifests(tmp_path) -> None:
    previous_context = ATOMIC.CTX
    previous_names = getattr(ATOMIC.V26, "FULL_FACTOR_NAMES", None)
    try:
        ATOMIC.V26.FULL_FACTOR_NAMES = ["factor_a", "factor_b", "factor_c"]
        ATOMIC.CTX = ATOMIC.Context("2026-01-02", tmp_path, "contract", 8, 1)
        manifest_dir = ATOMIC.CTX.root / "factors" / "family=A" / "window=10m"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "manifest.json").write_text(
            '{"status":"complete","blocks":[{"columns":["factor_a","factor_b"]}]}',
            encoding="utf-8",
        )
        progress = ATOMIC._factor_progress_snapshot("factor_block", "complete", family="A", window="10m")
        assert progress["factor_completed"] == 2
        assert progress["factor_expected"] == 3
        assert progress["factor_progress_pct"] == 66.667
        assert progress["factor_current_family"] == "A"
        assert (tmp_path / "factor_progress.json").exists()
    finally:
        ATOMIC.CTX = previous_context
        if previous_names is None:
            delattr(ATOMIC.V26, "FULL_FACTOR_NAMES")
        else:
            ATOMIC.V26.FULL_FACTOR_NAMES = previous_names


def test_full_label_builder_corrects_swapped_level_names() -> None:
    index = pd.MultiIndex.from_arrays(
        [[pd.Timestamp("2026-01-02 15:00:00", tz="UTC")] * 2, ["AAA", "BBB"]],
        names=["instrument", "datetime"],
    )
    frame = pd.DataFrame({"bars_1m__close": [100.0, 101.0]}, index=index)
    normalized = FULL._ensure_research_index(frame)
    assert normalized.index.names == ["instrument", "datetime"]
    assert normalized.index.get_level_values("instrument")[0] == "AAA"
    assert normalized.index.get_level_values("datetime")[0] == pd.Timestamp("2026-01-02 15:00:00", tz="UTC")


def test_full_label_builder_repairs_rowwise_mixed_index_orientation() -> None:
    timestamp = pd.Timestamp("2026-01-02 15:00:00", tz="UTC")
    index = pd.MultiIndex.from_arrays(
        [[timestamp, "BBB", timestamp, "DDD"], ["AAA", timestamp, "CCC", timestamp]],
        names=["instrument", "datetime"],
    )
    normalized = FULL._ensure_research_index(pd.DataFrame({"x": 1.0}, index=index))
    assert normalized.index.names == ["instrument", "datetime"]
    assert normalized.index.get_level_values("instrument").tolist() == ["AAA", "BBB", "CCC", "DDD"]
    assert normalized.index.get_level_values("datetime").tolist() == [timestamp] * 4


def test_full_label_builder_applies_latest_collision_policy() -> None:
    timestamp = pd.Timestamp("2026-01-02 15:00:00", tz="UTC")
    index = pd.MultiIndex.from_arrays(
        [["AAA", "AAA"], [timestamp, timestamp]],
        names=["instrument", "datetime"],
    )
    normalized = FULL._ensure_research_index(pd.DataFrame({"x": [1.0, 2.0]}, index=index))
    assert len(normalized) == 1
    assert normalized.iloc[0, 0] == 2.0
    assert normalized.attrs["research_index_collision_count"] == 1


def test_future_exact_uses_canonical_instrument_datetime_order() -> None:
    times = pd.date_range("2026-01-02 15:00:00", periods=2, freq="min", tz="UTC")
    index = pd.MultiIndex.from_product(
        [["AAA", "BBB"], times], names=["instrument", "datetime"]
    )
    frame = pd.DataFrame(
        {"value": [10.0, 11.0, 20.0, 21.0]}, index=index
    )
    shifted = FULL._future_exact(frame, "value", 1)
    assert shifted.loc[("AAA", times[0])] == 11.0
    assert shifted.loc[("BBB", times[0])] == 21.0
    assert pd.isna(shifted.loc[("AAA", times[1])])


def test_full_label_builder_uses_plausible_dates_for_numeric_swapped_levels() -> None:
    times = pd.date_range("2026-01-02 14:30:00", periods=5000, freq="min", tz="UTC")
    # Numeric instrument ids can parse as Unix nanoseconds; plausibility must
    # still select the actual 2026 timestamp level.
    index = pd.MultiIndex.from_arrays(
        [times, np.arange(len(times), dtype="int64")],
        names=["instrument", "datetime"],
    )
    normalized = FULL._ensure_research_index(pd.DataFrame({"x": 1.0}, index=index))
    assert normalized.index.names == ["instrument", "datetime"]
    assert normalized.index.get_level_values("datetime")[0] == times[0]


def test_full_label_builder_detects_unnamed_string_datetime_level() -> None:
    timestamps = ["2026-01-02 14:30:00", "2026-01-02 14:31:00"]
    index = pd.MultiIndex.from_arrays(
        [timestamps, ["A", "B"]],
        names=[None, None],
    )
    normalized = FULL._ensure_research_index(pd.DataFrame({"x": 1.0}, index=index))
    assert normalized.index.names == ["instrument", "datetime"]
    assert normalized.index.get_level_values("datetime")[0] == timestamps[0]


def test_supplement_contract_has_only_15m_and_30m() -> None:
    assert ATOMIC.SUPPLEMENT_WINDOWS == ("15m", "30m")


def test_supplement_direction_preserves_named_research_index() -> None:
    index = _cross_section(
        {
            "price_nvg_15m_terminal_signed_edge_balance": [-1, -0.5, 0, 0.5, 1],
            "price_nvg_30m_terminal_signed_edge_balance": [-1, -0.5, 0, 0.5, 1],
        }
    ).index
    frame = pd.DataFrame(index=index)
    for window in (15, 30):
        frame[f"price_nvg_{window}m_terminal_long_edge_signed_slope"] = np.linspace(-1, 1, len(index))
        frame[f"price_detrended_nvg_{window}m_terminal_signed_edge_balance"] = np.linspace(1, -1, len(index))
        frame[f"price_detrended_nvg_{window}m_terminal_long_edge_signed_slope"] = np.linspace(-0.5, 0.5, len(index))
    previous_context = ATOMIC.CTX
    try:
        ATOMIC.CTX = None
        result, _ = ATOMIC._supplement_direction(frame)
        assert result.index.names == ["instrument", "datetime"]
    finally:
        ATOMIC.CTX = previous_context


def test_b04_uses_long_direction_not_long_edge_ratio() -> None:
    frame = _cross_section(
        {
            "minute_nvg__price_nvg_15m_terminal_signed_edge_balance": [-1, -0.5, 0, 0.5, 1],
            "minute_nvg__price_nvg_15m_terminal_long_edge_signed_slope": [1, 0.5, 0, -0.5, -1],
            # Deliberately opposite nuisance field. The factor must ignore it.
            "minute_nvg__price_nvg_15m_top_terminal_long_edge_ratio": [0, 0.25, 0.5, 0.75, 1],
        }
    )
    result = LAUNCH.C._correct_family(frame, {}, "B", "15m")
    groups = frame.index.get_level_values("datetime")
    expected = LAUNCH.C._csz(
        frame["minute_nvg__price_nvg_15m_terminal_long_edge_signed_slope"], groups
    ) - LAUNCH.C._csz(
        frame["minute_nvg__price_nvg_15m_terminal_signed_edge_balance"], groups
    )
    pd.testing.assert_series_equal(result["B04"], expected, check_names=False)


def test_trade_fast_slow_and_resonance_use_terminal_direction() -> None:
    frame = _cross_section(
        {
            "trade_nvg__trade_price_nvg_60s_terminal_signed_edge_balance": [-1, -0.5, 0, 0.5, 1],
            "trade_nvg__trade_price_nvg_180s_terminal_signed_edge_balance": [-1, -1, 0, 1, 1],
            "trade_nvg__trade_price_nvg_300s_terminal_signed_edge_balance": [1, 0.5, 0, -0.5, -1],
            # Opposite asymmetry values prove the old substitute is not used.
            "trade_nvg__trade_price_nvg_60s_top_bottom_asymmetry": [1, 1, 1, -1, -1],
            "trade_nvg__trade_price_nvg_180s_top_bottom_asymmetry": [1, 1, 1, -1, -1],
            "trade_nvg__trade_price_nvg_300s_top_bottom_asymmetry": [-1, -1, -1, 1, 1],
        }
    )
    result = LAUNCH.C._correct_family(frame, {}, "E", "60s")
    groups = frame.index.get_level_values("datetime")
    expected_fast_slow = LAUNCH.C._csz(
        frame["trade_nvg__trade_price_nvg_60s_terminal_signed_edge_balance"], groups
    ) - LAUNCH.C._csz(
        frame["trade_nvg__trade_price_nvg_300s_terminal_signed_edge_balance"], groups
    )
    expected_resonance = (
        pd.concat(
            [
                frame["trade_nvg__trade_price_nvg_60s_terminal_signed_edge_balance"],
                frame["trade_nvg__trade_price_nvg_180s_terminal_signed_edge_balance"],
                frame["trade_nvg__trade_price_nvg_300s_terminal_signed_edge_balance"],
            ],
            axis=1,
        )
        .apply(np.sign)
        .mean(axis=1)
        .astype("float32")
    )
    pd.testing.assert_series_equal(result["E20"], expected_fast_slow, check_names=False)
    pd.testing.assert_series_equal(result["E21"], expected_resonance, check_names=False)


def test_d04_is_causal_within_symbol_zscore() -> None:
    times = pd.date_range("2026-01-02 14:30:00", periods=40, freq="1min", tz="UTC")
    index = pd.MultiIndex.from_arrays(
        [["AAA"] * len(times), times], names=["instrument", "datetime"]
    )
    values = pd.Series(np.linspace(0.0, 3.9, len(times)), index=index)
    frame = pd.DataFrame(
        {"minute_nvg__volume_path_15m_signed_change": values}, index=index
    )
    original = LAUNCH.C._correct_family(frame, {}, "D", "15m")["D04"]
    changed = frame.copy()
    changed.iloc[-1, 0] = 10_000.0
    revised = LAUNCH.C._correct_family(changed, {}, "D", "15m")["D04"]
    # A future mutation cannot affect earlier causal z-scores.
    pd.testing.assert_series_equal(original.iloc[:-1], revised.iloc[:-1])
    assert original.notna().sum() > 0


def test_first_candidate_is_not_dropped_when_fallback_is_missing() -> None:
    times = pd.date_range("2026-01-02 14:30:00", periods=8, freq="1min", tz="UTC")
    index = pd.MultiIndex.from_arrays(
        [["AAA"] * len(times), times], names=["instrument", "datetime"]
    )
    momentum = pd.Series(np.linspace(-1.0, 1.0, len(times)), index=index)
    frame = pd.DataFrame({"minute_nvg__momentum_10m": momentum}, index=index)

    result = FF.derive_prototype(frame, "A__ALL__", "10m")

    expected = momentum.astype("float32")
    expected.name = "minute_nvg__momentum_10m"
    pd.testing.assert_series_equal(result["A01"], expected)
    pd.testing.assert_series_equal(result["A02"], (-expected).rename(expected.name))


def test_trade_sketch_fields_resolve_from_canonical_namespace() -> None:
    times = pd.date_range("2026-01-02 14:30:00", periods=8, freq="1min", tz="UTC")
    index = pd.MultiIndex.from_arrays(
        [["AAA"] * len(times), times], names=["instrument", "datetime"]
    )
    frame = pd.DataFrame(
        {
            "trades_1m_sketch__trade_size_p95": np.full(len(times), 200.0),
            "trades_1m_core__median_trade_size": np.full(len(times), 100.0),
            "trades_1m_sketch__trade_size_hhi": np.full(len(times), 0.2),
            "trades_1m_sketch__top_1pct_volume_share": np.full(len(times), 0.3),
            "trades_1m_sketch__burstiness": np.full(len(times), 0.4),
            "trades_1m_sketch__large_trade_dollar_share": np.full(len(times), 0.5),
            "trades_1m_sketch__large_trade_buy_volume_proxy": np.full(len(times), 75.0),
            "trades_1m_sketch__large_trade_sell_volume_proxy": np.full(len(times), 25.0),
        },
        index=index,
    )

    result = FF.derive_prototype(frame, "I__ALL__", "1m")

    assert result["I10"].notna().all()
    assert result["I11"].notna().all()
    assert result["I12"].notna().all()
    assert result["I14"].notna().all()
