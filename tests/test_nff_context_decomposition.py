from __future__ import annotations

import numpy as np
import torch

from qlib.contrib.model.nff_context_decomposition import (
    ContextDecompositionConfig,
    NFFContextDecompositionModel,
    shuffled_identity,
    shuffled_market_time,
)
from qlib.contrib.model.nff_generic import GenericGRUConfig, NFFGenericGRU


def _model(variant: str) -> NFFContextDecompositionModel:
    torch.manual_seed(7)
    base = NFFGenericGRU(
        feature_count=2,
        target_count=1,
        config=GenericGRUConfig(hidden_size=8, num_layers=1, dropout=0.0, head_hidden_size=4),
    )
    model = NFFContextDecompositionModel(
        base,
        identity_width=8,
        market_width=8,
        config=ContextDecompositionConfig(
            identity_hidden_size=4,
            market_hidden_size=4,
            daily_hidden_size=4,
            branch_hidden_size=8,
            dropout=0.0,
            context_dropout=0.0,
        ),
        variant=variant,
        freeze_base=True,
    )
    return model.eval()


def test_zero_initialized_context_branches_equal_p2():
    support = torch.randn(3, 5, 4)
    query = torch.randn(3, 2, 5, 4)
    identity = torch.randn(3, 4, 8)
    identity_valid = torch.ones(3, 4, dtype=torch.bool)
    market = torch.randn(3, 2, 8)
    market_valid = torch.ones(3, 2, dtype=torch.bool)
    query_valid = torch.ones(3, 2, dtype=torch.bool)
    for variant in ("identity", "market", "combined", "full"):
        model = _model(variant)
        with torch.no_grad():
            prediction, gates = model(
                support,
                query,
                identity,
                identity_valid,
                market,
                market_valid,
                query_valid=query_valid,
            )
            baseline = model.baseline_forward(query, query_valid=query_valid)
        assert torch.equal(prediction, baseline)
        assert set(gates).issubset({"identity", "market", "daily"})
        assert not any(parameter.requires_grad for parameter in model.base_model.parameters())


def test_counterfactual_shuffles_preserve_shapes():
    identity = np.arange(48, dtype="float32").reshape(3, 2, 8)
    identity_valid = np.ones((3, 2), dtype=bool)
    shuffled_x, shuffled_valid = shuffled_identity(identity, identity_valid)
    assert shuffled_x.shape == identity.shape
    assert shuffled_valid.shape == identity_valid.shape
    assert not np.array_equal(shuffled_x, identity)

    market = np.arange(48, dtype="float32").reshape(3, 2, 8)
    market_valid = np.ones((3, 2), dtype=bool)
    shifted_x, shifted_valid = shuffled_market_time(market, market_valid)
    assert shifted_x.shape == market.shape
    assert shifted_valid.shape == market_valid.shape
    assert np.array_equal(shifted_x[:, 0], market[:, 1])
