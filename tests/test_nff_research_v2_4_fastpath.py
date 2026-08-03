from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nff_research import v2_1_neutralized_runner as reference
from nff_research import v2_4_optimized_runner as fast


def _index(minutes: int = 4, symbols: int = 80) -> pd.MultiIndex:
    timestamps = pd.date_range("2026-07-06T13:30:00Z", periods=minutes, freq="15min")
    names = [f"SYM{i:04d}" for i in range(symbols)]
    return pd.MultiIndex.from_product([timestamps, names], names=["datetime", "instrument"])


def test_ranked_ic_fastpath_matches_reference_minute_and_pooled():
    rng = np.random.default_rng(20260803)
    index = _index(minutes=5, symbols=90)
    features = pd.DataFrame(
        {
            "f1": rng.normal(size=len(index)),
            "f2": rng.normal(size=len(index)),
            "f3": rng.normal(size=len(index)),
        },
        index=index,
    )
    features.iloc[::17, 1] = np.nan
    label = pd.Series(rng.normal(size=len(index)), index=index)
    columns = list(features.columns)

    ref_minute = reference._minute_mean_ic(features, label, columns, min_n=30)
    ref_pooled = reference._pooled_ic(features, label, columns, min_n=30)
    got_minute, got_pooled = fast.ranked_ic_stats_once(features, label, columns, min_n=30)

    for column in columns:
        assert got_minute[column]["rank_ic_mean"] == pytest.approx(ref_minute[column]["rank_ic_mean"], abs=1e-12)
        assert got_minute[column]["rank_ic_std"] == pytest.approx(ref_minute[column]["rank_ic_std"], abs=1e-12)
        assert got_minute[column]["rank_ic_positive_ratio"] == pytest.approx(
            ref_minute[column]["rank_ic_positive_ratio"], abs=1e-12
        )
        assert got_pooled[column]["rank_ic_mean"] == pytest.approx(ref_pooled[column]["rank_ic_mean"], abs=1e-12)
        assert got_pooled[column]["ic_count"] == ref_pooled[column]["ic_count"]


def test_batched_oof_matches_reference_predictions():
    rng = np.random.default_rng(71)
    symbols = np.array([f"SYM{i:04d}" for i in range(125)], dtype=object)
    raw = pd.DataFrame(rng.normal(size=(len(symbols), 12)), columns=[f"x{i}" for i in range(12)])
    ranked = raw.rank(method="average", pct=True)
    ranked -= ranked.mean(axis=0)
    y = pd.Series(rng.normal(size=len(symbols)))
    steps = [("small", list(ranked.columns[:5])), ("large", list(ranked.columns))]
    folds = fast._fold_ids(symbols, 5)

    got = fast._batched_oof_predictions(ranked, y, folds, steps, folds=5, alpha=0.001, min_train_n=40)
    assert got is not None
    for step, columns in steps:
        expected = reference._fixed_symbol_fold_oof_predict(
            raw[columns], y, symbols, folds=5, alpha=0.001, min_train_n=40
        )
        assert expected is not None
        np.testing.assert_allclose(got[step], expected, atol=1e-12, rtol=0)


def test_rebalance_mask_reduces_regular_session_to_26_fifteen_minute_points():
    timestamps = pd.date_range("2026-07-06T13:30:00Z", periods=390, freq="1min")
    index = pd.MultiIndex.from_product([timestamps, ["A", "B"]], names=["datetime", "instrument"])
    mask = fast._rebalance_mask(index, 15)

    admitted = pd.Index(index.get_level_values("datetime")[mask]).nunique()
    assert admitted == 26
    assert int(mask.sum()) == 52


def test_portfolio_wrapper_prefilters_before_reference_groupby(monkeypatch: pytest.MonkeyPatch):
    timestamps = pd.date_range("2026-07-06T13:30:00Z", periods=390, freq="1min")
    index = pd.MultiIndex.from_product([timestamps, ["A", "B"]], names=["datetime", "instrument"])
    features = pd.DataFrame({"x": 1.0}, index=index)
    labels = pd.DataFrame({"return_vwap_to_vwap__h15": 0.0}, index=index)
    masks = {"return_vwap_to_vwap__h15": pd.Series(True, index=index)}
    controls = pd.DataFrame({"c": 1.0}, index=index)
    observed = {}

    def fake_reference(f, l, m, c, *args, **kwargs):
        observed["rows"] = len(f)
        observed["minutes"] = f.index.get_level_values("datetime").nunique()
        assert len(l) == len(f) == len(c)
        assert all(len(value) == len(f) for value in m.values())
        return pd.DataFrame()

    monkeypatch.setitem(fast._ORIGINALS, "staggered_portfolio_proxy", fake_reference)
    fast.portfolio_prefiltered(features, labels, masks, controls, "2026-07-06", 1)

    assert observed == {"rows": 52, "minutes": 26}


def test_lpt_orders_heaviest_dates_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(reference, "RESEARCH_ROOT", tmp_path)
    monkeypatch.setitem(fast._ORIGINALS, "load_dates", lambda start, end: ["2026-01-02", "2026-01-05", "2026-01-06"])
    costs = {"2026-01-02": 10, "2026-01-05": 50, "2026-01-06": 30}
    monkeypatch.setattr(fast, "_estimate_date_cost", lambda value: costs[value])

    ordered = fast.load_dates_lpt("2026-01-02", "2026-01-06")

    assert ordered == ["2026-01-05", "2026-01-06", "2026-01-02"]
    cache = tmp_path / "derived_inputs" / "research_date_cost_v2_4.json"
    assert cache.exists()


def test_required_manifest_is_narrow_and_reuses_materialized_derived(monkeypatch: pytest.MonkeyPatch):
    available = {
        "minute_nvg": {column.split("__", 1)[1] for column in reference.REPRESENTATIVE_ALPHA_FEATURES if column.startswith("minute_nvg__")},
        "trade_nvg": {
            *{column.split("__", 1)[1] for column in reference.REPRESENTATIVE_ALPHA_FEATURES if column.startswith("trade_nvg__")},
            "trade_active_second_ratio_300s",
            "trade_price_stale_ratio_300s",
        },
        "hawkes_lite": {column.split("__", 1)[1] for column in reference.REPRESENTATIVE_ALPHA_FEATURES if column.startswith("hawkes_lite__")},
        "hawkes_derived": {column.split("__", 1)[1] for column in reference.REPRESENTATIVE_ALPHA_FEATURES if column.startswith("hawkes_derived__")},
    }
    for spec in reference.HAWKES_GATE_SPECS:
        prefix, suffix = str(spec["column"]).split("__", 1)
        available.setdefault(prefix, set()).add(suffix)

    monkeypatch.setattr(
        fast,
        "_first_schema",
        lambda dataset, candidates: ("v4", available.get(dataset, set())),
    )
    manifest = fast.research_required_columns_manifest()

    assert manifest["derived_materialized"] is True
    assert manifest["columns"]["hawkes_derived"]
    assert "hawkes_ready" not in manifest["columns"]["hawkes_lite"]
    assert manifest["requested_count"] < 100


def test_online_pooled_corr_matches_direct_corr_with_missing_values():
    rng = np.random.default_rng(9)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    x[::13] = np.nan
    y[::17] = np.nan
    state = fast._online_corr_state()
    for start in range(0, len(x), 37):
        fast._update_corr_state(state, x[start : start + 37], y[start : start + 37])
    got, n = fast._finish_corr_state(state, min_n=30)
    expected, expected_n = reference._corr_with_min_n(x, y, min_n=30)
    assert n == expected_n
    assert got == pytest.approx(expected, abs=1e-12)


def test_fastpath_sets_blas_thread_guards_without_overwriting_user_choice(monkeypatch: pytest.MonkeyPatch):
    fast._ORIGINALS.clear()
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    fast.install_fastpath()
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == "1"
