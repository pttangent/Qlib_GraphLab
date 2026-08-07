from __future__ import annotations

import numpy as np
import torch

from qlib.contrib.data.nff_episode import StockDayEpisode
from qlib.contrib.model.nff_generic import (
    GenericGRUConfig,
    NFFGenericGRU,
    NormalizationState,
    RunningMoments,
    episodes_to_query_arrays,
    masked_mse,
)


def _episode() -> StockDayEpisode:
    query_x = np.arange(2 * 4 * 3, dtype="float32").reshape(2, 4, 3)
    observed = np.ones_like(query_x, dtype=bool)
    observed[0, 0, 1] = False
    query_x[0, 0, 1] = np.nan
    return StockDayEpisode(
        symbol="AAA",
        trade_date="2026-01-05",
        feature_names=("f0", "f1", "f2"),
        target_names=("r15", "rv15"),
        support_times_ns=np.arange(3, dtype="int64"),
        support_x=np.zeros((3, 3), dtype="float32"),
        support_observed=np.ones((3, 3), dtype=bool),
        query_times_ns=np.array([1, 2], dtype="int64"),
        query_x=query_x,
        query_observed=observed,
        targets=np.array([[0.1, 0.2], [0.3, np.nan]], dtype="float32"),
        target_observed=np.array([[True, True], [True, False]]),
        audit={"support_coverage": 1.0, "query_coverage_mean": 1.0, "target_coverage": 0.75},
    )


def test_running_moments_and_query_array_contract():
    episode = _episode()
    feature_moments = RunningMoments(3)
    feature_moments.update(episode.query_x, episode.query_observed)
    assert np.all(feature_moments.count > 0)
    normalization = NormalizationState(
        feature_mean=tuple(feature_moments.mean),
        feature_std=tuple(feature_moments.std()),
        target_mean=(0.2, 0.2),
        target_std=(0.1, 1.0),
        feature_names=episode.feature_names,
        target_names=episode.target_names,
    )
    x, y, mask, metadata = episodes_to_query_arrays([episode], normalization)
    assert x.shape == (2, 4, 6)
    assert y.shape == (2, 2)
    assert mask.shape == (2, 2)
    assert len(metadata) == 2
    assert x[0, 0, 1] == 0.0
    assert x[0, 0, 4] == 0.0


def test_generic_gru_forward_and_masked_loss():
    model = NFFGenericGRU(3, 2, GenericGRUConfig(hidden_size=8, num_layers=1, head_hidden_size=4))
    x = torch.randn(5, 10, 6)
    prediction = model(x)
    assert prediction.shape == (5, 2)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, True], [True, False], [True, True], [False, True], [True, True]])
    loss = masked_mse(prediction, target, mask)
    assert loss.ndim == 0
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
