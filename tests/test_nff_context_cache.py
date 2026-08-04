from __future__ import annotations

import numpy as np
import pandas as pd

from qlib.contrib.data.nff_context import (
    ContextCacheConfig,
    build_context_date,
    build_identity_daily_summary,
    build_market_query_context,
)
from qlib.contrib.data.nff_episode import StockDayEpisode
from qlib.contrib.model.nff_context_decomposition import episodes_to_context_arrays
from qlib.contrib.model.nff_generic import NormalizationState


def _normalization() -> NormalizationState:
    return NormalizationState(
        feature_mean=(0.0, 0.0),
        feature_std=(1.0, 1.0),
        target_mean=(0.0,),
        target_std=(1.0,),
        feature_names=("f1", "f2"),
        target_names=("y",),
    )


def _episode(symbol: str, trade_date: str, offset: float) -> StockDayEpisode:
    base = pd.Timestamp(f"{trade_date} 15:00:00", tz="UTC").value
    support = np.asarray([[offset, offset + 10], [offset + 1, offset + 11]], dtype="float32")
    query = np.asarray(
        [
            [[offset, offset + 10], [offset + 1, offset + 11], [offset + 2, offset + 12]],
            [[offset + 2, offset + 12], [offset + 3, offset + 13], [offset + 4, offset + 14]],
        ],
        dtype="float32",
    )
    return StockDayEpisode(
        symbol=symbol,
        trade_date=trade_date,
        feature_names=("f1", "f2"),
        target_names=("y",),
        support_times_ns=np.asarray([base - 120_000_000_000, base - 60_000_000_000], dtype="int64"),
        support_x=support,
        support_observed=np.ones_like(support, dtype=bool),
        query_times_ns=np.asarray([base, base + 900_000_000_000], dtype="int64"),
        query_x=query,
        query_observed=np.ones_like(query, dtype=bool),
        targets=np.asarray([[0.1], [0.2]], dtype="float32"),
        target_observed=np.ones((2, 1), dtype=bool),
        audit={"support_coverage": 1.0, "query_coverage_mean": 1.0, "target_coverage": 1.0},
    )


def test_identity_and_leave_one_out_market_context():
    episodes = [
        _episode("A", "2026-01-05", 0.0),
        _episode("B", "2026-01-05", 2.0),
        _episode("C", "2026-01-05", 4.0),
    ]
    normalization = _normalization()
    identity = build_identity_daily_summary(episodes, normalization)
    assert len(identity) == 3
    assert identity["identity__f1__coverage"].eq(1.0).all()

    market = build_market_query_context(
        episodes,
        normalization,
        episode_contract_hash="episode-contract",
        minimum_cross_section=3,
    )
    assert len(market) == 6
    first = market[(market["symbol"] == "A") & (market["query_index"] == 0)].iloc[0]
    # Query-last f1 values are A=2, B=4, C=6, therefore A's LOO mean is 5.
    assert np.isclose(first["market__f1__loo_mean"], 5.0)
    assert first["cross_section_size"] == 3
    assert bool(first["market_valid"])
    assert not market["sample_id"].duplicated().any()


def test_identity_history_is_strictly_prior():
    current = [_episode("A", "2026-01-06", 1.0)]
    normalization = _normalization()
    past = build_identity_daily_summary([_episode("A", "2026-01-05", 0.0)], normalization)
    future = build_identity_daily_summary([_episode("A", "2026-01-07", 9.0)], normalization)
    history = pd.concat([past, future], ignore_index=True)
    market = build_market_query_context(
        current,
        normalization,
        episode_contract_hash="episode-contract",
        minimum_cross_section=2,
    )
    arrays = episodes_to_context_arrays(
        current,
        normalization,
        episode_contract_hash="episode-contract",
        identity_history=history,
        market_context=market,
        history_days=20,
    )
    assert arrays.identity_valid.sum() == 1
    expected = past.filter(like="identity__").iloc[0].to_numpy(dtype="float32")
    assert np.allclose(arrays.identity_x[0, 0], np.nan_to_num(expected))


def test_context_date_audit_is_nff_only_contract():
    result = build_context_date(
        [_episode("A", "2026-01-05", 0.0), _episode("B", "2026-01-05", 2.0)],
        _normalization(),
        episode_contract_hash="episode-contract",
        config=ContextCacheConfig(minimum_cross_section=2),
    )
    assert result.audit["identity_rows"] == 2
    assert result.audit["market_rows"] == 4
