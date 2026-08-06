from __future__ import annotations

import numpy as np
import pandas as pd

from nff_research import v2_7_atomic_entry as ENTRY
from nff_research import v2_7_models as MODELS
from nff_research import v2_7_runtime_hardening as HARDEN


def _minute_index(instruments: list[str], minute: str = "2026-01-02 14:45:00") -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [instruments, [pd.Timestamp(minute, tz="UTC")] * len(instruments)],
        names=["instrument", "datetime"],
    )


def test_high_expected_return_is_long_and_low_is_short() -> None:
    block = pd.DataFrame(
        {
            "prediction": np.arange(100, dtype="float64"),
        },
        index=np.arange(100),
    )
    weights = MODELS._weights(block, quantile=0.10)
    assert weights.loc[block.nlargest(10, "prediction").index].gt(0).all()
    assert weights.loc[block.nsmallest(10, "prediction").index].lt(0).all()
    assert abs(float(weights.sum())) < 1e-12
    assert abs(float(weights.abs().sum()) - 1.0) < 1e-12


def test_vectorized_deciles_match_qcut_unique_rank_counts() -> None:
    instruments = [f"S{number:02d}" for number in range(23)]
    index = _minute_index(instruments)
    signal = pd.Series(np.arange(23, dtype="float64"), index=index)
    label = pd.Series(np.arange(23, dtype="float64") / 100.0, index=index)
    control = pd.Series(1.0, index=index)
    rows = ENTRY._decile_feature_rows_qcut_exact(
        "factor",
        signal,
        label,
        control,
        control,
        control,
        {"trade_date": "2026-01-02", "universe": "test", "variant": "raw", "label_family": "return", "horizon_bars": 5},
        min_n=10,
    )
    actual = pd.Series({int(row["decile"]): int(row["count"]) for row in rows})
    expected = pd.qcut(
        signal.rank(method="first"),
        10,
        labels=False,
        duplicates="drop",
    ).value_counts().sort_index()
    expected.index = expected.index + 1
    assert actual.to_dict() == expected.to_dict()


def test_path_metrics_are_causal_and_window_local() -> None:
    instruments = ["AAA"] * 8
    times = pd.date_range("2026-01-02 14:30:00", periods=8, freq="1min", tz="UTC")
    index = pd.MultiIndex.from_arrays([instruments, times], names=["instrument", "datetime"])
    original = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], index=index)
    changed_future = original.copy()
    changed_future.iloc[-1] = 8000.0
    first = HARDEN._path_metrics(original, 4)
    second = HARDEN._path_metrics(changed_future, 4)
    # Changing the final observation cannot alter any earlier row.
    for metric in first:
        pd.testing.assert_series_equal(first[metric].iloc[:-1], second[metric].iloc[:-1])
    assert np.isclose(float(first["signed_change"].iloc[3]), 3.0)
    assert np.isclose(float(first["efficiency"].iloc[3]), 1.0)
