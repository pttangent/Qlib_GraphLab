from __future__ import annotations

import numpy as np
import pandas as pd

from nff_research import v2_7_launch as LAUNCH


def _cross_section(values: dict[str, list[float]]) -> pd.DataFrame:
    instruments = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    timestamp = pd.Timestamp("2026-01-02 15:00:00", tz="UTC")
    index = pd.MultiIndex.from_arrays(
        [instruments, [timestamp] * len(instruments)],
        names=["instrument", "datetime"],
    )
    return pd.DataFrame(values, index=index)


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
    expected_resonance = pd.concat(
        [
            frame["trade_nvg__trade_price_nvg_60s_terminal_signed_edge_balance"],
            frame["trade_nvg__trade_price_nvg_180s_terminal_signed_edge_balance"],
            frame["trade_nvg__trade_price_nvg_300s_terminal_signed_edge_balance"],
        ],
        axis=1,
    ).apply(np.sign).mean(axis=1)
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
