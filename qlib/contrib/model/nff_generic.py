# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NFF-only generic sequence baseline for stock-day episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from qlib.contrib.data.nff_episode import StockDayEpisode


@dataclass(frozen=True)
class GenericGRUConfig:
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.10
    head_hidden_size: int = 64

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.head_hidden_size <= 0:
            raise ValueError("hidden sizes must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class NFFGenericGRU(nn.Module):
    """Shared cross-stock model with no stock ID and no daily adapter."""

    def __init__(self, feature_count: int, target_count: int, config: Optional[GenericGRUConfig] = None):
        super().__init__()
        self.config = config or GenericGRUConfig()
        self.config.validate()
        if feature_count <= 0 or target_count <= 0:
            raise ValueError("feature_count and target_count must be positive")
        self.feature_count = int(feature_count)
        self.target_count = int(target_count)
        recurrent_dropout = self.config.dropout if self.config.num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=self.feature_count * 2,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.head_hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.head_hidden_size, self.target_count),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], got shape={tuple(x.shape)}")
        _, hidden = self.encoder(x)
        return self.head(self.norm(hidden[-1]))

    def model_contract(self) -> Dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "feature_count": self.feature_count,
            "target_count": self.target_count,
            "config": asdict(self.config),
        }


class RunningMoments:
    """Numerically stable per-column moments with missing-value support."""

    def __init__(self, width: int):
        if width <= 0:
            raise ValueError("width must be positive")
        self.count = np.zeros(width, dtype="float64")
        self.mean = np.zeros(width, dtype="float64")
        self.m2 = np.zeros(width, dtype="float64")

    @property
    def width(self) -> int:
        return int(len(self.count))

    def update(self, values: np.ndarray, observed: Optional[np.ndarray] = None) -> None:
        array = np.asarray(values, dtype="float64").reshape(-1, self.width)
        valid = np.isfinite(array) if observed is None else np.asarray(observed, dtype=bool).reshape(-1, self.width)
        valid &= np.isfinite(array)
        for column in range(self.width):
            data = array[valid[:, column], column]
            if data.size == 0:
                continue
            batch_count = float(data.size)
            batch_mean = float(data.mean())
            batch_m2 = float(np.square(data - batch_mean).sum())
            old_count = self.count[column]
            total = old_count + batch_count
            delta = batch_mean - self.mean[column]
            self.mean[column] += delta * batch_count / total
            self.m2[column] += batch_m2 + delta * delta * old_count * batch_count / total
            self.count[column] = total

    def std(self, floor: float = 1e-6) -> np.ndarray:
        denominator = np.maximum(self.count - 1.0, 1.0)
        value = np.sqrt(np.maximum(self.m2 / denominator, 0.0))
        value[~np.isfinite(value) | (value < floor)] = 1.0
        return value

    def state_dict(self) -> Dict[str, Any]:
        return {"count": self.count.tolist(), "mean": self.mean.tolist(), "m2": self.m2.tolist()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RunningMoments":
        obj = cls(len(state["count"]))
        obj.count = np.asarray(state["count"], dtype="float64")
        obj.mean = np.asarray(state["mean"], dtype="float64")
        obj.m2 = np.asarray(state["m2"], dtype="float64")
        return obj


@dataclass(frozen=True)
class NormalizationState:
    feature_mean: Tuple[float, ...]
    feature_std: Tuple[float, ...]
    target_mean: Tuple[float, ...]
    target_std: Tuple[float, ...]
    feature_names: Tuple[str, ...]
    target_names: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NormalizationState":
        return cls(
            feature_mean=tuple(float(item) for item in value["feature_mean"]),
            feature_std=tuple(float(item) for item in value["feature_std"]),
            target_mean=tuple(float(item) for item in value["target_mean"]),
            target_std=tuple(float(item) for item in value["target_std"]),
            feature_names=tuple(str(item) for item in value["feature_names"]),
            target_names=tuple(str(item) for item in value["target_names"]),
        )


def episodes_to_query_arrays(
    episodes: Sequence[StockDayEpisode],
    normalization: NormalizationState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Flatten episodes into fixed-length query samples for P2 training."""

    feature_names = tuple(normalization.feature_names)
    target_names = tuple(normalization.target_names)
    if not episodes:
        return (
            np.empty((0, 0, len(feature_names) * 2), dtype="float32"),
            np.empty((0, len(target_names)), dtype="float32"),
            np.empty((0, len(target_names)), dtype=bool),
            pd.DataFrame(columns=["symbol", "trade_date", "query_time"]),
        )
    feature_mean = np.asarray(normalization.feature_mean, dtype="float32")
    feature_std = np.asarray(normalization.feature_std, dtype="float32")
    target_mean = np.asarray(normalization.target_mean, dtype="float32")
    target_std = np.asarray(normalization.target_std, dtype="float32")

    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    y_mask_parts: List[np.ndarray] = []
    metadata_rows: List[Dict[str, Any]] = []
    for episode in episodes:
        if tuple(episode.feature_names) != feature_names:
            raise ValueError("Episode feature order does not match normalization contract")
        if tuple(episode.target_names) != target_names:
            raise ValueError("Episode target order does not match normalization contract")
        observed = episode.query_observed.astype(bool, copy=False)
        standardized = (episode.query_x.astype("float32", copy=False) - feature_mean) / feature_std
        standardized = np.where(observed, standardized, 0.0).astype("float32", copy=False)
        model_x = np.concatenate([standardized, observed.astype("float32")], axis=-1)
        target_mask = episode.target_observed.astype(bool, copy=False)
        model_y = (episode.targets.astype("float32", copy=False) - target_mean) / target_std
        model_y = np.where(target_mask, model_y, 0.0).astype("float32", copy=False)
        x_parts.append(model_x)
        y_parts.append(model_y)
        y_mask_parts.append(target_mask)
        metadata_rows.extend(
            {
                "symbol": episode.symbol,
                "trade_date": episode.trade_date,
                "query_time": pd.Timestamp(value),
            }
            for value in episode.query_times_ns
        )
    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(y_mask_parts, axis=0),
        pd.DataFrame(metadata_rows),
    )


def masked_mse(prediction: Tensor, target: Tensor, observed: Tensor) -> Tensor:
    mask = observed.to(dtype=prediction.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return (((prediction - target) ** 2) * mask).sum() / denominator


def inverse_targets(values: np.ndarray, normalization: NormalizationState) -> np.ndarray:
    mean = np.asarray(normalization.target_mean, dtype="float32")
    std = np.asarray(normalization.target_std, dtype="float32")
    return values.astype("float32", copy=False) * std + mean


def prediction_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "mse": None, "mae": None, "rank_ic_mean": None, "rank_ic_count": 0}
    valid = frame.dropna(subset=["prediction", "target"])
    if valid.empty:
        return {"rows": 0, "mse": None, "mae": None, "rank_ic_mean": None, "rank_ic_count": 0}
    error = valid["prediction"] - valid["target"]
    rank_values: List[float] = []
    for _, block in valid.groupby(["trade_date", "query_time", "target_name"], sort=False):
        if len(block) < 3:
            continue
        corr = block["prediction"].rank(method="average").corr(block["target"].rank(method="average"))
        if pd.notna(corr):
            rank_values.append(float(corr))
    return {
        "rows": int(len(valid)),
        "mse": float(np.square(error).mean()),
        "mae": float(np.abs(error).mean()),
        "rank_ic_mean": float(np.mean(rank_values)) if rank_values else None,
        "rank_ic_std": float(np.std(rank_values, ddof=1)) if len(rank_values) > 1 else None,
        "rank_ic_count": len(rank_values),
        "rank_ic_positive_ratio": float(np.mean(np.asarray(rank_values) > 0)) if rank_values else None,
    }
