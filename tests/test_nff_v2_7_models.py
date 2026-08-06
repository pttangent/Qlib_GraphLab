from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from nff_research import v2_7_deciles as DECILES
from nff_research import v2_7_launch as LAUNCH
from nff_research import v2_7_models as MODELS
from nff_research import v2_7_physical_contract as PHYSICAL
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
    research = SimpleNamespace(infer_bundle=lambda feature: "test")
    rows = DECILES.decile_feature_rows(
        research,
        "factor",
        signal,
        label,
        control,
        control,
        control,
        {
            "trade_date": "2026-01-02",
            "universe": "test",
            "variant": "raw",
            "label_family": "return",
            "horizon_bars": 5,
        },
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


def test_cached_quantile_edges_match_qcut_for_many_cross_section_sizes() -> None:
    for count in range(10, 101):
        ranks = np.arange(1, count + 1, dtype="float64")
        actual = DECILES.qcut_deciles_from_unique_ranks(ranks, np.full(count, count))
        expected = pd.qcut(
            pd.Series(ranks),
            10,
            labels=False,
            duplicates="drop",
        ).to_numpy() + 1
        np.testing.assert_array_equal(actual, expected)


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


def test_registry_separates_physical_464_from_legacy_476() -> None:
    physical_base = PHYSICAL._expanded(LAUNCH.C, PHYSICAL.PHYSICAL_WINDOWS)
    legacy_base = PHYSICAL._expanded(LAUNCH.C, PHYSICAL.LEGACY_WINDOWS)
    supplement = PHYSICAL._supplement_rows(LAUNCH.C)
    physical = pd.concat([physical_base, supplement], ignore_index=True)
    legacy = pd.concat([legacy_base, supplement], ignore_index=True)

    assert len(physical) == PHYSICAL.PHYSICAL_EXECUTABLE_COUNT == 464
    assert len(legacy) == PHYSICAL.LEGACY_FORMAL_COUNT == 476
    assert set(physical.loc[physical["family"].eq("A"), "window"]) == {"10m", "15m", "30m"}
    assert set(physical.loc[physical["family"].eq("B"), "window"]) == {"10m", "15m", "30m"}
    assert set(physical.loc[physical["family"].eq("C"), "window"]) == {"10m", "15m", "30m"}
    assert set(physical.loc[physical["family"].eq("D"), "window"]) == {"10m", "15m", "30m"}
    assert set(physical.loc[physical["family"].eq("G"), "window"]) == {"15m", "30m", "60m"}
    assert set(physical.loc[physical["family"].eq("S"), "window"]) == {"15m", "30m"}

    gap = legacy.loc[~legacy["factor_id"].isin(set(physical["factor_id"]))]
    assert len(gap) == 12
    assert gap["family"].eq("C").all()
    assert gap["window"].eq("60m").all()


def test_multiscale_a14_to_a16_use_physical_10_15_30() -> None:
    instruments = ["AAA", "BBB", "CCC"]
    index = _minute_index(instruments)
    frame = pd.DataFrame(
        {
            "traditional__momentum_10m": [0.1, -0.2, 0.3],
            "traditional__momentum_15m": [0.2, -0.1, 0.1],
            "traditional__momentum_30m": [0.3, -0.3, -0.1],
            # A conflicting 60m field proves it is not part of the formula.
            "traditional__momentum_60m": [-9.0, 9.0, -9.0],
        },
        index=index,
    )
    result = LAUNCH.C._correct_family(frame, {}, "A", "10m")
    assert result["A14"].notna().all()
    assert result["A15"].notna().all()
    assert result["A16"].notna().all()
    expected_resonance = pd.Series([1.0, -1.0, 1.0 / 3.0], index=index)
    pd.testing.assert_series_equal(result["A14"], expected_resonance, check_names=False)


def test_k_minute_anchor_maps_to_physical_10m_fields() -> None:
    index = _minute_index(["AAA", "BBB", "CCC"])
    frame = pd.DataFrame(
        {
            "minute_nvg__price_nvg_10m_terminal_signed_edge_balance": [1.0, 2.0, 3.0],
            "minute_nvg__price_nvg_15m_terminal_signed_edge_balance": [-1.0, -2.0, -3.0],
            "minute_nvg__price_path_10m_efficiency": [0.1, 0.2, 0.3],
            "minute_nvg__price_path_15m_efficiency": [0.9, 0.8, 0.7],
        },
        index=index,
    )
    mapped = PHYSICAL._k_anchor_frame(frame)
    pd.testing.assert_series_equal(
        mapped["minute_nvg__price_nvg_15m_terminal_signed_edge_balance"],
        frame["minute_nvg__price_nvg_10m_terminal_signed_edge_balance"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        mapped["minute_nvg__price_path_15m_efficiency"],
        frame["minute_nvg__price_path_10m_efficiency"],
        check_names=False,
    )
