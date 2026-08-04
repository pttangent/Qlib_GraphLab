# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NFF-only daily context adaptation for point-in-time stock-day episodes.

The module deliberately keeps the P2 generic GRU as the query backbone. A
support encoder summarizes the fixed early-session NFF window once per
stock-day. A zero-initialized gated residual adapter then conditions every
later query representation without using GFF, GAL, ticker embeddings, or
intraday gradient updates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from qlib.contrib.data.nff_episode import StockDayEpisode
from qlib.contrib.model.nff_generic import NFFGenericGRU, NormalizationState, inverse_targets


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_episode_id(episode: StockDayEpisode, episode_contract_hash: str) -> str:
    payload = {
        "symbol": str(episode.symbol).upper().strip(),
        "trade_date": str(episode.trade_date),
        "episode_contract_hash": str(episode_contract_hash),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def stable_sample_id(
    episode: StockDayEpisode,
    query_time_ns: int,
    episode_contract_hash: str,
) -> str:
    payload = {
        "episode_id": stable_episode_id(episode, episode_contract_hash),
        "query_time_ns": int(query_time_ns),
        "feature_names": list(episode.feature_names),
        "target_names": list(episode.target_names),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DailyAdapterConfig:
    support_hidden_size: int = 64
    support_num_layers: int = 1
    adapter_hidden_size: int = 128
    dropout: float = 0.10
    gate_bias: float = -2.0

    def validate(self) -> None:
        if self.support_hidden_size <= 0 or self.adapter_hidden_size <= 0:
            raise ValueError("adapter hidden sizes must be positive")
        if self.support_num_layers <= 0:
            raise ValueError("support_num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class AdapterArrays:
    support_x: np.ndarray
    query_x: np.ndarray
    targets: np.ndarray
    target_observed: np.ndarray
    query_valid: np.ndarray
    metadata: pd.DataFrame
    symbols: Tuple[str, ...]

    @property
    def episode_count(self) -> int:
        return int(self.support_x.shape[0])

    @property
    def query_slots(self) -> int:
        return int(self.query_x.shape[1]) if self.query_x.ndim >= 2 else 0


def _normalize_feature_block(
    values: np.ndarray,
    observed: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    standardized = (values.astype("float32", copy=False) - mean) / std
    standardized = np.where(observed, standardized, 0.0).astype("float32", copy=False)
    return np.concatenate([standardized, observed.astype("float32")], axis=-1)


def episodes_to_adapter_arrays(
    episodes: Sequence[StockDayEpisode],
    normalization: NormalizationState,
    *,
    episode_contract_hash: str,
) -> AdapterArrays:
    """Pack complete stock-day tasks while encoding support only once.

    Query slots are padded per date. ``query_valid`` distinguishes real query
    points from padding; target missingness remains a separate mask.
    """

    feature_names = tuple(normalization.feature_names)
    target_names = tuple(normalization.target_names)
    if not episodes:
        return AdapterArrays(
            support_x=np.empty((0, 0, len(feature_names) * 2), dtype="float32"),
            query_x=np.empty((0, 0, 0, len(feature_names) * 2), dtype="float32"),
            targets=np.empty((0, 0, len(target_names)), dtype="float32"),
            target_observed=np.empty((0, 0, len(target_names)), dtype=bool),
            query_valid=np.empty((0, 0), dtype=bool),
            metadata=pd.DataFrame(),
            symbols=(),
        )

    for episode in episodes:
        if tuple(episode.feature_names) != feature_names:
            raise ValueError("Episode feature order does not match normalization contract")
        if tuple(episode.target_names) != target_names:
            raise ValueError("Episode target order does not match normalization contract")
        if episode.query_count <= 0:
            raise ValueError("Adapter arrays require episodes with at least one query")

    feature_mean = np.asarray(normalization.feature_mean, dtype="float32")
    feature_std = np.asarray(normalization.feature_std, dtype="float32")
    target_mean = np.asarray(normalization.target_mean, dtype="float32")
    target_std = np.asarray(normalization.target_std, dtype="float32")
    max_queries = max(episode.query_count for episode in episodes)
    support_rows = max(int(episode.support_x.shape[0]) for episode in episodes)
    lookback_rows = max(int(episode.query_x.shape[1]) for episode in episodes)
    width = len(feature_names) * 2
    target_count = len(target_names)
    episode_count = len(episodes)

    support_x = np.zeros((episode_count, support_rows, width), dtype="float32")
    query_x = np.zeros((episode_count, max_queries, lookback_rows, width), dtype="float32")
    targets = np.zeros((episode_count, max_queries, target_count), dtype="float32")
    target_observed = np.zeros((episode_count, max_queries, target_count), dtype=bool)
    query_valid = np.zeros((episode_count, max_queries), dtype=bool)
    metadata_rows: List[Dict[str, Any]] = []
    symbols: List[str] = []

    for episode_index, episode in enumerate(episodes):
        support = _normalize_feature_block(
            episode.support_x,
            episode.support_observed.astype(bool, copy=False),
            feature_mean,
            feature_std,
        )
        support_x[episode_index, : len(support)] = support
        query = _normalize_feature_block(
            episode.query_x,
            episode.query_observed.astype(bool, copy=False),
            feature_mean,
            feature_std,
        )
        query_count = episode.query_count
        query_x[episode_index, :query_count, : query.shape[1]] = query
        standardized_targets = (episode.targets.astype("float32", copy=False) - target_mean) / target_std
        standardized_targets = np.where(episode.target_observed, standardized_targets, 0.0).astype(
            "float32", copy=False
        )
        targets[episode_index, :query_count] = standardized_targets
        target_observed[episode_index, :query_count] = episode.target_observed
        query_valid[episode_index, :query_count] = True
        symbols.append(str(episode.symbol))

        support_times = np.asarray(episode.support_times_ns, dtype="int64")
        support_start_ns = int(support_times.min()) if support_times.size else 0
        support_end_ns = int(support_times.max()) if support_times.size else 0
        episode_id = stable_episode_id(episode, episode_contract_hash)
        for query_index, query_time_ns in enumerate(np.asarray(episode.query_times_ns, dtype="int64")):
            query_start_ns = int(query_time_ns) - (int(episode.query_x.shape[1]) - 1) * 60 * 1_000_000_000
            overlap = int(((support_times >= query_start_ns) & (support_times <= int(query_time_ns))).sum())
            if support_end_ns >= int(query_time_ns):
                raise AssertionError("Support must end strictly before every query")
            metadata_rows.append(
                {
                    "episode_index": episode_index,
                    "query_index": query_index,
                    "episode_id": episode_id,
                    "sample_id": stable_sample_id(episode, int(query_time_ns), episode_contract_hash),
                    "symbol": str(episode.symbol),
                    "trade_date": str(episode.trade_date),
                    "query_time": pd.Timestamp(int(query_time_ns)),
                    "support_start": pd.Timestamp(support_start_ns),
                    "support_end": pd.Timestamp(support_end_ns),
                    "support_query_overlap_minutes": overlap,
                    "support_coverage": float(episode.audit.get("support_coverage", np.nan)),
                    "query_coverage_mean": float(episode.audit.get("query_coverage_mean", np.nan)),
                }
            )

    metadata = pd.DataFrame(metadata_rows).sort_values(
        ["trade_date", "query_time", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    return AdapterArrays(
        support_x=support_x,
        query_x=query_x,
        targets=targets,
        target_observed=target_observed,
        query_valid=query_valid,
        metadata=metadata,
        symbols=tuple(symbols),
    )


class NFFDailyContextAdapter(nn.Module):
    """Condition a frozen P2 query representation on fixed early-session NFF."""

    def __init__(
        self,
        base_model: NFFGenericGRU,
        config: Optional[DailyAdapterConfig] = None,
        *,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config or DailyAdapterConfig()
        self.config.validate()
        input_width = int(base_model.feature_count) * 2
        recurrent_dropout = self.config.dropout if self.config.support_num_layers > 1 else 0.0
        self.support_encoder = nn.GRU(
            input_size=input_width,
            hidden_size=self.config.support_hidden_size,
            num_layers=self.config.support_num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.support_norm = nn.LayerNorm(self.config.support_hidden_size)
        joint_width = base_model.config.hidden_size + self.config.support_hidden_size
        self.delta = nn.Sequential(
            nn.Linear(joint_width, self.config.adapter_hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.adapter_hidden_size, base_model.config.hidden_size),
        )
        self.gate = nn.Sequential(
            nn.Linear(joint_width, self.config.adapter_hidden_size),
            nn.GELU(),
            nn.Linear(self.config.adapter_hidden_size, 1),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, self.config.gate_bias)
        self.set_base_trainable(not freeze_base)

    def set_base_trainable(self, trainable: bool) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad = bool(trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        # Adapter-only training must preserve the exact deterministic P2 path.
        # Frozen GRU/head dropout is therefore kept in evaluation mode.
        if not any(parameter.requires_grad for parameter in self.base_model.parameters()):
            self.base_model.eval()
        return self

    def set_joint_trainable(self) -> None:
        """Unfreeze the P2 head, norm and final recurrent layer only."""
        self.set_base_trainable(False)
        for parameter in self.base_model.head.parameters():
            parameter.requires_grad = True
        for parameter in self.base_model.norm.parameters():
            parameter.requires_grad = True
        last_layer = self.base_model.config.num_layers - 1
        suffix = f"_l{last_layer}"
        for name, parameter in self.base_model.encoder.named_parameters():
            if suffix in name:
                parameter.requires_grad = True

    def encode_support(self, support_x: Tensor) -> Tensor:
        if support_x.ndim != 3:
            raise ValueError(f"Expected support [batch,time,width], got {tuple(support_x.shape)}")
        _, hidden = self.support_encoder(support_x)
        return self.support_norm(hidden[-1])

    def encode_query(self, query_x: Tensor) -> Tensor:
        if query_x.ndim != 4:
            raise ValueError(f"Expected query [batch,queries,time,width], got {tuple(query_x.shape)}")
        batch, queries, steps, width = query_x.shape
        flat = query_x.reshape(batch * queries, steps, width)
        _, hidden = self.base_model.encoder(flat)
        encoded = self.base_model.norm(hidden[-1])
        return encoded.reshape(batch, queries, -1)

    def forward(
        self,
        support_x: Tensor,
        query_x: Tensor,
        *,
        query_valid: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        context = self.encode_support(support_x)
        query_hidden = self.encode_query(query_x)
        expanded_context = context[:, None, :].expand(-1, query_hidden.shape[1], -1)
        joint = torch.cat([query_hidden, expanded_context], dim=-1)
        gate = torch.sigmoid(self.gate(joint))
        adapted = query_hidden + gate * self.delta(joint)
        prediction = self.base_model.head(adapted)
        if query_valid is not None:
            mask = query_valid.to(dtype=prediction.dtype).unsqueeze(-1)
            prediction = prediction * mask
            gate = gate * mask
        return prediction, gate

    def baseline_forward(self, query_x: Tensor, *, query_valid: Optional[Tensor] = None) -> Tensor:
        prediction = self.base_model.head(self.encode_query(query_x))
        if query_valid is not None:
            prediction = prediction * query_valid.to(dtype=prediction.dtype).unsqueeze(-1)
        return prediction

    def model_contract(self) -> Dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "base_model": self.base_model.model_contract(),
            "adapter": asdict(self.config),
        }


def masked_episode_mse(
    prediction: Tensor,
    target: Tensor,
    target_observed: Tensor,
    query_valid: Tensor,
) -> Tensor:
    mask = target_observed.to(dtype=prediction.dtype) * query_valid.to(dtype=prediction.dtype).unsqueeze(-1)
    denominator = mask.sum().clamp_min(1.0)
    return (((prediction - target) ** 2) * mask).sum() / denominator


def support_prefix(support_x: np.ndarray, minutes: int) -> np.ndarray:
    if minutes <= 0:
        raise ValueError("support prefix minutes must be positive")
    if support_x.ndim != 3:
        raise ValueError("support_x must be [episodes,time,width]")
    return support_x[:, : min(minutes, support_x.shape[1]), :]


def same_day_shuffled_support(
    support_x: np.ndarray,
    symbols: Sequence[str],
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    if len(support_x) != len(symbols):
        raise ValueError("support and symbols length mismatch")
    if len(symbols) <= 1:
        return support_x.copy(), tuple(symbols)
    order = np.arange(len(symbols))
    shifted = np.roll(order, 1)
    return support_x[shifted].copy(), tuple(str(symbols[index]) for index in shifted)


def previous_day_support(
    support_x: np.ndarray,
    symbols: Sequence[str],
    bank: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    output = np.zeros_like(support_x)
    available = np.zeros(len(symbols), dtype=bool)
    for index, symbol in enumerate(symbols):
        value = bank.get(str(symbol))
        if value is None or value.shape != support_x[index].shape:
            continue
        output[index] = value
        available[index] = True
    return output, available


def update_previous_support_bank(
    bank: MutableMapping[str, np.ndarray],
    support_x: np.ndarray,
    symbols: Sequence[str],
) -> None:
    for index, symbol in enumerate(symbols):
        bank[str(symbol)] = support_x[index].copy()


def paired_prediction_frame(
    adapter_standardized: np.ndarray,
    baseline_standardized: np.ndarray,
    targets_standardized: np.ndarray,
    target_observed: np.ndarray,
    query_valid: np.ndarray,
    metadata: pd.DataFrame,
    normalization: NormalizationState,
    *,
    support_mode: str,
    support_length: int,
    gate: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Return one paired row per observed ``sample_id x target``."""

    if metadata.empty:
        return pd.DataFrame()
    adapter_values = inverse_targets(adapter_standardized, normalization)
    baseline_values = inverse_targets(baseline_standardized, normalization)
    target_values = inverse_targets(targets_standardized, normalization)
    episode_index = metadata["episode_index"].to_numpy(dtype="int64")
    query_index = metadata["query_index"].to_numpy(dtype="int64")
    valid_query_rows = query_valid[episode_index, query_index].astype(bool)
    rows: List[pd.DataFrame] = []
    for target_index, target_name in enumerate(normalization.target_names):
        observed = valid_query_rows & target_observed[episode_index, query_index, target_index].astype(bool)
        if not observed.any():
            continue
        part = metadata.loc[observed].copy()
        ei = episode_index[observed]
        qi = query_index[observed]
        part["target_name"] = target_name
        part["prediction_adapter"] = adapter_values[ei, qi, target_index]
        part["prediction_b0"] = baseline_values[ei, qi, target_index]
        part["target"] = target_values[ei, qi, target_index]
        part["support_mode"] = support_mode
        part["support_length"] = int(support_length)
        if gate is not None:
            part["gate"] = gate[ei, qi, 0]
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
