from __future__ import annotations

import pandas as pd
import numpy as np

from nff_research import v2_7_atomic_campaign as campaign
from nff_research import v2_1_neutralized_runner as runner


def test_portfolio_index_normalizer_reorders_instrument_first_frames():
    timestamps = pd.date_range("2026-07-06T13:30:00Z", periods=2, freq="15min")
    index = pd.MultiIndex.from_product([["A", "B"], timestamps], names=["instrument", "datetime"])
    frame = pd.DataFrame({"value": [1, 2, 3, 4]}, index=index)

    normalized = campaign._datetime_first_portfolio_frame(frame)

    assert list(normalized.index.names) == ["datetime", "instrument"]
    restored = normalized.reorder_levels(["instrument", "datetime"])
    assert set(restored.index) == set(frame.index)


def test_parallel_portfolio_stream_preserves_same_sleeve_rows():
    timestamps = pd.date_range("2026-07-06T13:30:00Z", periods=3, freq="15min")
    symbols = [f"S{i:02d}" for i in range(40)]
    index = pd.MultiIndex.from_product([symbols, timestamps], names=["instrument", "datetime"])
    values = np.arange(len(index), dtype="float64")
    work = pd.DataFrame(
        {
            "signal": values,
            "label": values / 10000.0,
            "__adv20": np.full(len(index), 1_000_000.0),
        },
        index=index,
    )
    variant = {
        "name": "contrarian_q05_30m",
        "direction": -1.0,
        "quantile": 0.05,
        "rebalance_minutes": 30,
        "gate_pair_id": None,
        "gate_mode": "none",
    }
    rows = runner._portfolio_feature_rows(
        work,
        "signal",
        variant,
        "2026-07-06",
        "liquid_common_adv20_top1000",
        "return_vwap_to_vwap",
        30,
        30,
        1.0,
    )
    assert len(rows) == 2
    assert {row["sleeve_id"] for row in rows} == {0}
    assert all(row["portfolio_accounting"] == "same_sleeve_turnover" for row in rows)
    assert all(np.isfinite(row["gross_return"]) for row in rows)


def test_batched_portfolio_variant_matches_independent_feature_streams():
    timestamps = pd.date_range("2026-07-06T13:30:00Z", periods=4, freq="15min")
    symbols = [f"S{i:02d}" for i in range(40)]
    index = pd.MultiIndex.from_product([symbols, timestamps], names=["instrument", "datetime"])
    work = pd.DataFrame(
        {
            "signal_a": np.sin(np.arange(len(index), dtype="float64")),
            "signal_b": np.cos(np.arange(len(index), dtype="float64")),
            "label": np.linspace(-0.01, 0.01, len(index)),
            "__adv20": np.full(len(index), 1_000_000.0),
        },
        index=index,
    )
    work.loc[("S00", timestamps[1]), "signal_a"] = np.nan
    variant = {
        "name": "contrarian_q05_30m",
        "direction": -1.0,
        "quantile": 0.05,
        "rebalance_minutes": 30,
        "gate_pair_id": None,
        "gate_mode": "none",
    }
    expected = []
    for feature in ("signal_a", "signal_b"):
        expected.extend(
            runner._portfolio_feature_rows(
                work,
                feature,
                variant,
                "2026-07-06",
                "liquid_common_adv20_top1000",
                "return_vwap_to_vwap",
                30,
                30,
                1.0,
            )
        )
    actual = runner._portfolio_variant_rows_batched(
        work,
        ["signal_a", "signal_b"],
        variant,
        "2026-07-06",
        "liquid_common_adv20_top1000",
        "return_vwap_to_vwap",
        30,
        30,
        1.0,
    )
    expected_frame = pd.DataFrame(expected).sort_values(["feature", "datetime"]).reset_index(drop=True)
    actual_frame = pd.DataFrame(actual).sort_values(["feature", "datetime"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_frame, expected_frame, check_dtype=False, check_like=True)
