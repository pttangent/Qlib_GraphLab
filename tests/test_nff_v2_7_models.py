from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from nff_research import v2_7_deciles as DECILES
from nff_research import v2_7_launch as LAUNCH
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
    research = SimpleNamespace(infer_bundle=lambda feature: "test")
    rows = DECILES.decile_feature_rows(
        research,
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


def test_registry_preserves_474_plus_two_contract(monkeypatch) -> None:
    monkeypatch.setattr(LAUNCH, "_BASE_CONFIGURE_REGISTRY", lambda config: None)
    LAUNCH._configure_476_registry({"factor_resolution_policy": {}})
    registry = LAUNCH.C.V26.SPEC_REGISTRY
    assert len(registry) == 476
    assert not registry["window"].eq("10m").any()
    assert set(registry.loc[registry["family"].eq("A"), "window"]) == {"15m", "30m", "60m"}
    assert set(registry.loc[registry["family"].eq("B"), "window"]) == {"15m", "30m", "60m"}
    assert set(registry.loc[registry["family"].eq("C"), "window"]) == {"15m", "30m", "60m", "120m"}
    assert set(registry.loc[registry["family"].eq("S"), "window"]) == {"15m", "30m"}


def test_multiscale_a14_to_a16_use_15_30_60() -> None:
    instruments = ["AAA", "BBB", "CCC"]
    index = _minute_index(instruments)
    frame = pd.DataFrame(
        {
            "traditional__momentum_15m": [0.1, -0.2, 0.3],
            "traditional__momentum_30m": [0.2, -0.1, 0.1],
            "traditional__momentum_60m": [0.3, -0.3, -0.1],
        },
        index=index,
    )
    result = LAUNCH.C._correct_family(frame, {}, "A", "15m")
    assert result["A14"].notna().all()
    assert result["A15"].notna().all()
    assert result["A16"].notna().all()
    expected_resonance = pd.Series([1.0, -1.0, 1.0 / 3.0], index=index)
    pd.testing.assert_series_equal(result["A14"], expected_resonance, check_names=False)
