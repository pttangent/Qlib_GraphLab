from __future__ import annotations

import numpy as np
import pandas as pd

from qlib.contrib.data.nff_episode import (
    EpisodeTargetSpec,
    SupportQueryPolicy,
    build_stock_day_episodes,
)


def _synthetic_frames():
    decision_times = pd.date_range("2026-01-05T14:30:00Z", "2026-01-05T20:00:00Z", freq="1min")
    aligned = pd.DataFrame(
        {
            "datetime": decision_times,
            "instrument": ["AAA"] * len(decision_times),
            "minute_nvg__shape": np.linspace(0.0, 1.0, len(decision_times), dtype="float32"),
            "hawkes_derived__persistence": np.linspace(1.0, 0.0, len(decision_times), dtype="float32"),
        }
    )
    bar_times = pd.date_range("2026-01-05T14:30:00Z", "2026-01-05T21:00:00Z", freq="1min")
    bars = pd.DataFrame(
        {
            "datetime": bar_times,
            "instrument": ["AAA"] * len(bar_times),
            "bars_1m__open": np.arange(len(bar_times), dtype="float64") + 100.0,
            "bars_1m__close": np.arange(len(bar_times), dtype="float64") + 100.5,
        }
    )
    return aligned, bars


def test_build_stock_day_episode_is_strictly_causal():
    aligned, bars = _synthetic_frames()
    policy = SupportQueryPolicy(
        support_start="09:30",
        support_end="10:00",
        query_start="10:00",
        query_end="10:30",
        query_stride_minutes=15,
        query_lookback_minutes=10,
        min_support_coverage=1.0,
        min_query_coverage=1.0,
    )
    targets = [
        EpisodeTargetSpec(
            name="return_open_h15",
            kind="forward_return",
            horizon_minutes=15,
            price_column="bars_1m__open",
            entry_delay_minutes=1,
        ),
        EpisodeTargetSpec(
            name="rv_close_h15",
            kind="realized_volatility",
            horizon_minutes=15,
            price_column="bars_1m__close",
            entry_delay_minutes=1,
            minimum_observations=5,
        ),
    ]
    result = build_stock_day_episodes(aligned, bars, policy=policy, targets=targets)
    assert len(result.episodes) == 1
    episode = result.episodes[0]
    assert episode.symbol == "AAA"
    assert episode.trade_date == "2026-01-05"
    assert episode.support_x.shape == (30, 2)
    assert episode.query_x.shape == (3, 10, 2)
    assert episode.targets.shape == (3, 2)
    assert episode.target_observed.all()
    assert episode.audit["causal"] is True
    assert episode.support_times_ns.max() < episode.query_times_ns.min()
    first_query = pd.Timestamp(episode.query_times_ns[0])
    assert first_query == pd.Timestamp("2026-01-05T15:00:00")
    expected = (146.0 / 131.0) - 1.0
    assert episode.targets[0, 0] == np.float32(expected)


def test_episode_rejects_insufficient_support_coverage():
    aligned, bars = _synthetic_frames()
    aligned.loc[aligned.index[:25], "minute_nvg__shape"] = np.nan
    policy = SupportQueryPolicy(
        support_start="09:30",
        support_end="10:00",
        query_start="10:00",
        query_end="10:00",
        query_lookback_minutes=5,
        min_support_coverage=0.90,
    )
    result = build_stock_day_episodes(
        aligned,
        bars,
        policy=policy,
        targets=[
            EpisodeTargetSpec(
                name="return_open_h15",
                kind="forward_return",
                horizon_minutes=15,
                price_column="bars_1m__open",
            )
        ],
    )
    assert not result.episodes
    assert result.rejections[0]["reason"] == "insufficient_support_coverage"
