import numpy as np
import pandas as pd

from nff_research.v2_6_full_defined_campaign import _full_build_labels_and_masks


def _frame(periods: int = 8) -> pd.DataFrame:
    times = pd.date_range("2026-01-02 14:30", periods=periods, freq="min", tz="UTC")
    index = pd.MultiIndex.from_product([times, ["A"]], names=["datetime", "instrument"])
    values = np.arange(100.0, 100.0 + len(index))
    return pd.DataFrame(
        {
            "bars_1m__open": values,
            "bars_1m__close": values + 0.5,
            "bars_1m__vwap": values + 0.25,
            "bars_1m__high": values + 1.0,
            "bars_1m__low": values - 1.0,
            "bars_1m__volume": 100.0,
            "bars_1m__dollar_volume": 10_000.0,
            "trades_1m_core__trade_count": 10.0,
        },
        index=index,
    )


def test_labels_use_exact_elapsed_minutes_and_same_session_mask():
    labels, masks = _full_build_labels_and_masks(_frame(), [1, 5])
    expected_h1 = 102.0 / 101.0 - 1.0
    expected_h5 = 106.0 / 101.0 - 1.0
    assert np.isclose(labels.iloc[0]["return_open_to_open__h1"], expected_h1)
    assert np.isclose(labels.iloc[0]["return_open_to_open__h5"], expected_h5)
    assert int(masks["return_open_to_open__h1"].sum()) == 6
    assert int(masks["return_open_to_open__h5"].sum()) == 2


def test_jump_label_does_not_use_decision_time_return():
    frame = _frame(8)
    labels_a, _ = _full_build_labels_and_masks(frame, [5])
    changed = frame.copy()
    changed.iloc[0, changed.columns.get_loc("bars_1m__close")] = 10_000.0
    labels_b, _ = _full_build_labels_and_masks(changed, [5])
    # The first future-window BPV term begins with future returns only; changing
    # close[t] must not alter the label at t.
    assert np.isclose(
        labels_a.iloc[0]["jump_tail_event__h5"],
        labels_b.iloc[0]["jump_tail_event__h5"],
        equal_nan=True,
    )
