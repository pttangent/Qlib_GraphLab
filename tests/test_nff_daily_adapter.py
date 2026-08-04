from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from qlib.contrib.data.nff_episode import StockDayEpisode
from qlib.contrib.model.nff_daily_adapter import (
    DailyAdapterConfig,
    NFFDailyContextAdapter,
    episodes_to_adapter_arrays,
    masked_episode_mse,
    paired_prediction_frame,
)
from qlib.contrib.model.nff_generic import GenericGRUConfig, NFFGenericGRU, NormalizationState


def _episode(symbol: str = "AAPL", shift: float = 0.0) -> StockDayEpisode:
    support_times = pd.date_range("2026-01-05T14:30:00", periods=30, freq="1min").astype("int64").to_numpy()
    query_times = pd.date_range("2026-01-05T15:00:00", periods=3, freq="15min").astype("int64").to_numpy()
    support = np.full((30, 2), 1.0 + shift, dtype="float32")
    query = np.full((3, 30, 2), 2.0 + shift, dtype="float32")
    return StockDayEpisode(
        symbol=symbol,
        trade_date="2026-01-05",
        feature_names=("f0", "f1"),
        target_names=("ret", "vol"),
        support_times_ns=support_times,
        support_x=support,
        support_observed=np.ones_like(support, dtype=bool),
        query_times_ns=query_times,
        query_x=query,
        query_observed=np.ones_like(query, dtype=bool),
        targets=np.asarray([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], dtype="float32"),
        target_observed=np.ones((3, 2), dtype=bool),
        audit={"support_coverage": 1.0, "query_coverage_mean": 1.0},
    )


def _normalization() -> NormalizationState:
    return NormalizationState(
        feature_mean=(0.0, 0.0),
        feature_std=(1.0, 1.0),
        target_mean=(0.0, 0.0),
        target_std=(1.0, 1.0),
        feature_names=("f0", "f1"),
        target_names=("ret", "vol"),
    )


def test_adapter_is_exact_p2_at_zero_initialized_delta():
    arrays = episodes_to_adapter_arrays([_episode()], _normalization(), episode_contract_hash="contract-a")
    base = NFFGenericGRU(2, 2, GenericGRUConfig(hidden_size=8, num_layers=1, dropout=0.0, head_hidden_size=4))
    model = NFFDailyContextAdapter(
        base,
        DailyAdapterConfig(support_hidden_size=4, adapter_hidden_size=6, dropout=0.0),
        freeze_base=True,
    )
    support = torch.from_numpy(arrays.support_x)
    query = torch.from_numpy(arrays.query_x)
    valid = torch.from_numpy(arrays.query_valid)
    adapted, gate = model(support, query, query_valid=valid)
    baseline = model.baseline_forward(query, query_valid=valid)
    assert adapted.shape == (1, 3, 2)
    assert gate.shape == (1, 3, 1)
    assert torch.allclose(adapted, baseline)
    assert all(not parameter.requires_grad for parameter in model.base_model.parameters())


def test_adapter_pairing_is_unique_and_loss_masks_padding():
    arrays = episodes_to_adapter_arrays(
        [_episode("AAPL"), _episode("MSFT", 1.0)],
        _normalization(),
        episode_contract_hash="contract-a",
    )
    base = NFFGenericGRU(2, 2, GenericGRUConfig(hidden_size=8, num_layers=1, dropout=0.0, head_hidden_size=4))
    model = NFFDailyContextAdapter(base, DailyAdapterConfig(support_hidden_size=4, adapter_hidden_size=6, dropout=0.0))
    adapted, gate = model(
        torch.from_numpy(arrays.support_x),
        torch.from_numpy(arrays.query_x),
        query_valid=torch.from_numpy(arrays.query_valid),
    )
    baseline = model.baseline_forward(torch.from_numpy(arrays.query_x), query_valid=torch.from_numpy(arrays.query_valid))
    loss = masked_episode_mse(
        adapted,
        torch.from_numpy(arrays.targets),
        torch.from_numpy(arrays.target_observed),
        torch.from_numpy(arrays.query_valid),
    )
    assert torch.isfinite(loss)
    frame = paired_prediction_frame(
        adapted.detach().numpy(),
        baseline.detach().numpy(),
        arrays.targets,
        arrays.target_observed,
        arrays.query_valid,
        arrays.metadata,
        _normalization(),
        support_mode="same_stock_same_day",
        support_length=30,
        gate=gate.detach().numpy(),
    )
    assert len(frame) == 12
    assert not frame.duplicated(["sample_id", "target_name"]).any()
    assert (frame["support_end"] < frame["query_time"]).all()
