from __future__ import annotations

import numpy as np
import pandas as pd

from qlib.contrib.data.nff_episode import StockDayEpisode
from qlib.contrib.model.nff_daily_adapter import (
    episodes_to_adapter_arrays,
    previous_day_support,
    same_day_shuffled_support,
    stable_sample_id,
    support_prefix,
    update_previous_support_bank,
)
from qlib.contrib.model.nff_generic import NormalizationState


def _episode(symbol: str, trade_date: str, value: float) -> StockDayEpisode:
    support_start = pd.Timestamp(f"{trade_date} 14:30:00")
    support_times = pd.date_range(support_start, periods=30, freq="1min").astype("int64").to_numpy()
    query_times = pd.date_range(support_start + pd.Timedelta(minutes=30), periods=2, freq="15min").astype("int64").to_numpy()
    return StockDayEpisode(
        symbol=symbol,
        trade_date=trade_date,
        feature_names=("f",),
        target_names=("y",),
        support_times_ns=support_times,
        support_x=np.full((30, 1), value, dtype="float32"),
        support_observed=np.ones((30, 1), dtype=bool),
        query_times_ns=query_times,
        query_x=np.full((2, 30, 1), value, dtype="float32"),
        query_observed=np.ones((2, 30, 1), dtype=bool),
        targets=np.full((2, 1), value, dtype="float32"),
        target_observed=np.ones((2, 1), dtype=bool),
        audit={"support_coverage": 1.0, "query_coverage_mean": 1.0},
    )


def _norm() -> NormalizationState:
    return NormalizationState(
        feature_mean=(0.0,),
        feature_std=(1.0,),
        target_mean=(0.0,),
        target_std=(1.0,),
        feature_names=("f",),
        target_names=("y",),
    )


def test_sample_ids_are_deterministic_and_query_specific():
    episode = _episode("AAPL", "2026-01-05", 1.0)
    first = stable_sample_id(episode, int(episode.query_times_ns[0]), "contract")
    repeated = stable_sample_id(episode, int(episode.query_times_ns[0]), "contract")
    second = stable_sample_id(episode, int(episode.query_times_ns[1]), "contract")
    assert first == repeated
    assert first != second


def test_support_counterfactuals_preserve_date_boundary_and_shape():
    episodes = [_episode("AAPL", "2026-01-05", 1.0), _episode("MSFT", "2026-01-05", 2.0)]
    arrays = episodes_to_adapter_arrays(episodes, _norm(), episode_contract_hash="contract")
    shuffled, sources = same_day_shuffled_support(arrays.support_x, arrays.symbols)
    assert shuffled.shape == arrays.support_x.shape
    assert sources == ("MSFT", "AAPL")
    assert np.allclose(shuffled[0], arrays.support_x[1])
    assert support_prefix(arrays.support_x, 5).shape[1] == 5

    bank = {}
    previous, available = previous_day_support(arrays.support_x, arrays.symbols, bank)
    assert not available.any()
    assert np.allclose(previous, 0.0)
    update_previous_support_bank(bank, arrays.support_x, arrays.symbols)
    previous, available = previous_day_support(arrays.support_x, arrays.symbols, bank)
    assert available.all()
    assert np.allclose(previous, arrays.support_x)
