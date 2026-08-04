# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NFF-only decomposition of persistent, market and stock-day context."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from qlib.contrib.data.nff_context import identity_context_columns, market_context_columns
from qlib.contrib.data.nff_episode import StockDayEpisode
from qlib.contrib.model.nff_daily_adapter import AdapterArrays, episodes_to_adapter_arrays
from qlib.contrib.model.nff_generic import NFFGenericGRU, NormalizationState


_ALLOWED_VARIANTS = {"identity", "market", "combined", "full"}


@dataclass(frozen=True)
class ContextDecompositionConfig:
    identity_hidden_size: int = 64
    identity_num_layers: int = 1
    market_hidden_size: int = 64
    daily_hidden_size: int = 64
    branch_hidden_size: int = 128
    dropout: float = 0.10
    gate_bias: float = -2.0
    context_dropout: float = 0.10

    def validate(self) -> None:
        for name, value in {
            "identity_hidden_size": self.identity_hidden_size,
            "identity_num_layers": self.identity_num_layers,
            "market_hidden_size": self.market_hidden_size,
            "daily_hidden_size": self.daily_hidden_size,
            "branch_hidden_size": self.branch_hidden_size,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.context_dropout < 1.0:
            raise ValueError("context_dropout must be in [0, 1)")


@dataclass
class ContextArrays:
    adapter: AdapterArrays
    identity_x: np.ndarray
    identity_valid: np.ndarray
    market_x: np.ndarray
    market_valid: np.ndarray
    identity_columns: Tuple[str, ...]
    market_columns: Tuple[str, ...]

    @property
    def episode_count(self) -> int:
        return self.adapter.episode_count


def _numeric_context(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(frame), 0), dtype="float32")
    return frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")


def episodes_to_context_arrays(
    episodes: Sequence[StockDayEpisode],
    normalization: NormalizationState,
    *,
    episode_contract_hash: str,
    identity_history: pd.DataFrame,
    market_context: pd.DataFrame,
    history_days: int = 20,
) -> ContextArrays:
    if history_days <= 0:
        raise ValueError("history_days must be positive")
    adapter = episodes_to_adapter_arrays(
        episodes,
        normalization,
        episode_contract_hash=episode_contract_hash,
    )
    identity_columns = identity_context_columns(normalization.feature_names)
    market_columns = market_context_columns(normalization.feature_names)
    identity_x = np.zeros((len(episodes), history_days, len(identity_columns)), dtype="float32")
    identity_valid = np.zeros((len(episodes), history_days), dtype=bool)

    identity_frame = identity_history.copy()
    if not identity_frame.empty:
        identity_frame["trade_date"] = identity_frame["trade_date"].astype(str)
        identity_frame["symbol"] = identity_frame["symbol"].astype(str)
    for episode_index, episode in enumerate(episodes):
        current_date = str(episode.trade_date)
        if identity_frame.empty:
            continue
        history = identity_frame[
            (identity_frame["symbol"] == str(episode.symbol))
            & (identity_frame["trade_date"] < current_date)
        ].sort_values("trade_date", kind="mergesort").tail(history_days)
        if not history.empty and history["trade_date"].max() >= current_date:
            raise AssertionError("Identity context contains current or future dates")
        values = _numeric_context(history, identity_columns)
        valid_rows = np.isfinite(values).any(axis=1)
        values = np.where(np.isfinite(values), values, 0.0).astype("float32", copy=False)
        count = len(values)
        if count:
            identity_x[episode_index, :count] = values
            identity_valid[episode_index, :count] = valid_rows

    market_x = np.zeros(
        (adapter.episode_count, adapter.query_slots, len(market_columns)), dtype="float32"
    )
    market_valid = np.zeros((adapter.episode_count, adapter.query_slots), dtype=bool)
    market_frame = market_context.copy()
    if not market_frame.empty:
        if market_frame["sample_id"].duplicated().any():
            raise ValueError("Market context contains duplicate sample_id rows")
        market_frame = market_frame.set_index("sample_id", drop=False)
    for row in adapter.metadata.itertuples(index=False):
        if market_frame.empty or row.sample_id not in market_frame.index:
            continue
        context_row = market_frame.loc[row.sample_id]
        values = pd.to_numeric(context_row.reindex(market_columns), errors="coerce").to_numpy(dtype="float32")
        valid = bool(context_row.get("market_valid", True)) and np.isfinite(values).any()
        market_x[int(row.episode_index), int(row.query_index)] = np.where(
            np.isfinite(values), values, 0.0
        )
        market_valid[int(row.episode_index), int(row.query_index)] = valid

    return ContextArrays(
        adapter=adapter,
        identity_x=identity_x,
        identity_valid=identity_valid,
        market_x=market_x,
        market_valid=market_valid,
        identity_columns=tuple(identity_columns),
        market_columns=tuple(market_columns),
    )


class _ResidualContextBranch(nn.Module):
    def __init__(self, base_hidden: int, context_hidden: int, branch_hidden: int, dropout: float, gate_bias: float):
        super().__init__()
        joint = base_hidden + context_hidden
        self.delta = nn.Sequential(
            nn.Linear(joint, branch_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_hidden, base_hidden),
        )
        self.gate = nn.Sequential(
            nn.Linear(joint, branch_hidden),
            nn.GELU(),
            nn.Linear(branch_hidden, 1),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_bias)

    def forward(self, query_hidden: Tensor, context: Tensor, valid: Tensor) -> Tuple[Tensor, Tensor]:
        joint = torch.cat([query_hidden, context], dim=-1)
        gate = torch.sigmoid(self.gate(joint)) * valid.to(query_hidden.dtype).unsqueeze(-1)
        return gate * self.delta(joint), gate


class NFFContextDecompositionModel(nn.Module):
    """Add independently gated identity, market and daily NFF context to P2."""

    def __init__(
        self,
        base_model: NFFGenericGRU,
        identity_width: int,
        market_width: int,
        config: Optional[ContextDecompositionConfig] = None,
        *,
        variant: str = "full",
        freeze_base: bool = True,
    ):
        super().__init__()
        if variant not in _ALLOWED_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(_ALLOWED_VARIANTS)}")
        self.base_model = base_model
        self.config = config or ContextDecompositionConfig()
        self.config.validate()
        self.variant = variant
        base_hidden = int(base_model.config.hidden_size)
        input_width = int(base_model.feature_count) * 2

        self.identity_encoder = nn.GRU(
            input_size=max(1, int(identity_width)),
            hidden_size=self.config.identity_hidden_size,
            num_layers=self.config.identity_num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.identity_num_layers > 1 else 0.0,
        )
        self.identity_norm = nn.LayerNorm(self.config.identity_hidden_size)
        self.market_encoder = nn.Sequential(
            nn.Linear(max(1, int(market_width)), self.config.market_hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.config.market_hidden_size),
        )
        self.daily_encoder = nn.GRU(
            input_size=input_width,
            hidden_size=self.config.daily_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.daily_norm = nn.LayerNorm(self.config.daily_hidden_size)

        self.identity_branch = _ResidualContextBranch(
            base_hidden,
            self.config.identity_hidden_size,
            self.config.branch_hidden_size,
            self.config.dropout,
            self.config.gate_bias,
        )
        self.market_branch = _ResidualContextBranch(
            base_hidden,
            self.config.market_hidden_size,
            self.config.branch_hidden_size,
            self.config.dropout,
            self.config.gate_bias,
        )
        self.daily_branch = _ResidualContextBranch(
            base_hidden,
            self.config.daily_hidden_size,
            self.config.branch_hidden_size,
            self.config.dropout,
            self.config.gate_bias,
        )
        self.identity_width = int(identity_width)
        self.market_width = int(market_width)
        self.set_base_trainable(not freeze_base)

    @property
    def uses_identity(self) -> bool:
        return self.variant in {"identity", "combined", "full"}

    @property
    def uses_market(self) -> bool:
        return self.variant in {"market", "combined", "full"}

    @property
    def uses_daily(self) -> bool:
        return self.variant == "full"

    def set_base_trainable(self, trainable: bool) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad = bool(trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.base_model.parameters()):
            self.base_model.eval()
        return self

    def encode_query(self, query_x: Tensor) -> Tensor:
        batch, queries, steps, width = query_x.shape
        flat = query_x.reshape(batch * queries, steps, width)
        _, hidden = self.base_model.encoder(flat)
        return self.base_model.norm(hidden[-1]).reshape(batch, queries, -1)

    def encode_identity(self, identity_x: Tensor, identity_valid: Tensor) -> Tuple[Tensor, Tensor]:
        if self.identity_width == 0:
            identity_x = torch.zeros((*identity_x.shape[:2], 1), device=identity_x.device, dtype=identity_x.dtype)
        masked = identity_x * identity_valid.to(identity_x.dtype).unsqueeze(-1)
        sequence, _ = self.identity_encoder(masked)
        lengths = identity_valid.sum(dim=1)
        safe_index = (lengths - 1).clamp_min(0)
        batch_index = torch.arange(len(identity_x), device=identity_x.device)
        encoded = self.identity_norm(sequence[batch_index, safe_index])
        has_history = lengths > 0
        encoded = encoded * has_history.to(encoded.dtype).unsqueeze(-1)
        return encoded, has_history

    def encode_market(self, market_x: Tensor, market_valid: Tensor) -> Tensor:
        if self.market_width == 0:
            market_x = torch.zeros((*market_x.shape[:2], 1), device=market_x.device, dtype=market_x.dtype)
        encoded = self.market_encoder(market_x)
        return encoded * market_valid.to(encoded.dtype).unsqueeze(-1)

    def encode_daily(self, support_x: Tensor) -> Tensor:
        _, hidden = self.daily_encoder(support_x)
        return self.daily_norm(hidden[-1])

    def _drop_valid(self, valid: Tensor) -> Tensor:
        if not self.training or self.config.context_dropout <= 0.0:
            return valid
        keep = torch.rand(valid.shape, device=valid.device) >= self.config.context_dropout
        return valid & keep

    def forward(
        self,
        support_x: Tensor,
        query_x: Tensor,
        identity_x: Tensor,
        identity_valid: Tensor,
        market_x: Tensor,
        market_valid: Tensor,
        *,
        query_valid: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        query_hidden = self.encode_query(query_x)
        adapted = query_hidden
        gates: Dict[str, Tensor] = {}
        batch, queries, _ = query_hidden.shape

        if self.uses_identity:
            identity_context, history_valid = self.encode_identity(identity_x, identity_valid)
            identity_context = identity_context[:, None, :].expand(-1, queries, -1)
            valid = self._drop_valid(history_valid[:, None].expand(-1, queries))
            residual, gate = self.identity_branch(query_hidden, identity_context, valid)
            adapted = adapted + residual
            gates["identity"] = gate
        if self.uses_market:
            market_context = self.encode_market(market_x, market_valid)
            valid = self._drop_valid(market_valid)
            residual, gate = self.market_branch(query_hidden, market_context, valid)
            adapted = adapted + residual
            gates["market"] = gate
        if self.uses_daily:
            daily_context = self.encode_daily(support_x)[:, None, :].expand(-1, queries, -1)
            valid = self._drop_valid(torch.ones((batch, queries), device=query_x.device, dtype=torch.bool))
            residual, gate = self.daily_branch(query_hidden, daily_context, valid)
            adapted = adapted + residual
            gates["daily"] = gate

        prediction = self.base_model.head(adapted)
        if query_valid is not None:
            mask = query_valid.to(prediction.dtype).unsqueeze(-1)
            prediction = prediction * mask
            gates = {name: value * mask for name, value in gates.items()}
        return prediction, gates

    def baseline_forward(self, query_x: Tensor, *, query_valid: Optional[Tensor] = None) -> Tensor:
        prediction = self.base_model.head(self.encode_query(query_x))
        if query_valid is not None:
            prediction = prediction * query_valid.to(prediction.dtype).unsqueeze(-1)
        return prediction

    def model_contract(self) -> Dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "variant": self.variant,
            "identity_width": self.identity_width,
            "market_width": self.market_width,
            "base_model": self.base_model.model_contract(),
            "config": asdict(self.config),
        }


def masked_context_mse(
    prediction: Tensor,
    target: Tensor,
    target_observed: Tensor,
    query_valid: Tensor,
) -> Tensor:
    mask = target_observed.to(prediction.dtype) * query_valid.to(prediction.dtype).unsqueeze(-1)
    denominator = mask.sum().clamp_min(1.0)
    return (((prediction - target) ** 2) * mask).sum() / denominator


def shuffled_identity(identity_x: np.ndarray, identity_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(identity_x) <= 1:
        return identity_x.copy(), identity_valid.copy()
    order = np.roll(np.arange(len(identity_x)), 1)
    return identity_x[order].copy(), identity_valid[order].copy()


def shuffled_market_time(market_x: np.ndarray, market_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if market_x.shape[1] <= 1:
        return market_x.copy(), market_valid.copy()
    return np.roll(market_x, 1, axis=1).copy(), np.roll(market_valid, 1, axis=1).copy()


def zero_identity(identity_x: np.ndarray, identity_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.zeros_like(identity_x), np.zeros_like(identity_valid)


def zero_market(market_x: np.ndarray, market_valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.zeros_like(market_x), np.zeros_like(market_valid)
