# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Point-in-time-safe NFF-only identity and market context caches."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from qlib.contrib.data.nff_episode import StockDayEpisode, contract_hash
from qlib.contrib.model.nff_daily_adapter import stable_sample_id
from qlib.contrib.model.nff_generic import NormalizationState


@dataclass(frozen=True)
class ContextCacheConfig:
    history_days: int = 20
    minimum_cross_section: int = 3
    formula_version: str = "nff_context_v1"

    def validate(self) -> None:
        if self.history_days <= 0:
            raise ValueError("history_days must be positive")
        if self.minimum_cross_section < 2:
            raise ValueError("minimum_cross_section must be at least 2")
        if not self.formula_version:
            raise ValueError("formula_version must be non-empty")


def normalization_contract_hash(normalization: NormalizationState) -> str:
    return contract_hash(normalization.to_dict())


def context_cache_contract(
    *,
    episode_contract_hash: str,
    normalization: NormalizationState,
    config: ContextCacheConfig,
) -> Dict[str, Any]:
    config.validate()
    return {
        "version": config.formula_version,
        "episode_contract_hash": str(episode_contract_hash),
        "normalization_hash": normalization_contract_hash(normalization),
        "feature_names": list(normalization.feature_names),
        "history_days": config.history_days,
        "minimum_cross_section": config.minimum_cross_section,
        "identity_statistics": ["mean", "std", "last", "coverage"],
        "market_statistics": ["loo_mean", "loo_std", "loo_positive_ratio", "loo_coverage"],
    }


def _standardized(values: np.ndarray, observed: np.ndarray, normalization: NormalizationState) -> np.ndarray:
    mean = np.asarray(normalization.feature_mean, dtype="float32")
    std = np.asarray(normalization.feature_std, dtype="float32")
    array = (values.astype("float32", copy=False) - mean) / std
    return np.where(observed.astype(bool, copy=False), array, np.nan).astype("float32", copy=False)


def identity_context_columns(feature_names: Sequence[str]) -> Tuple[str, ...]:
    return tuple(
        f"identity__{feature}__{stat}"
        for feature in feature_names
        for stat in ("mean", "std", "last", "coverage")
    )


def market_context_columns(feature_names: Sequence[str]) -> Tuple[str, ...]:
    return tuple(
        f"market__{feature}__{stat}"
        for feature in feature_names
        for stat in ("loo_mean", "loo_std", "loo_positive_ratio", "loo_coverage")
    )


def build_identity_daily_summary(
    episodes: Sequence[StockDayEpisode],
    normalization: NormalizationState,
) -> pd.DataFrame:
    """Create one compact, label-free daily state row per stock.

    The summary uses the fixed support block plus the last observable row at
    each admitted query. It never reads future dates and is only consumed by
    later trading days.
    """

    feature_names = tuple(normalization.feature_names)
    columns = identity_context_columns(feature_names)
    rows = []
    for episode in episodes:
        if tuple(episode.feature_names) != feature_names:
            raise ValueError("Episode feature order differs from normalization contract")
        support = _standardized(episode.support_x, episode.support_observed, normalization)
        query_last = _standardized(
            episode.query_x[:, -1, :],
            episode.query_observed[:, -1, :],
            normalization,
        )
        values = np.concatenate([support, query_last], axis=0)
        row: Dict[str, Any] = {"symbol": str(episode.symbol), "trade_date": str(episode.trade_date)}
        for index, feature in enumerate(feature_names):
            series = values[:, index]
            valid = np.isfinite(series)
            if valid.any():
                valid_values = series[valid].astype("float64", copy=False)
                mean = float(valid_values.mean())
                std = float(valid_values.std(ddof=1)) if len(valid_values) > 1 else 0.0
                last = float(valid_values[-1])
                coverage = float(valid.mean())
            else:
                mean = std = last = np.nan
                coverage = 0.0
            row[f"identity__{feature}__mean"] = mean
            row[f"identity__{feature}__std"] = std
            row[f"identity__{feature}__last"] = last
            row[f"identity__{feature}__coverage"] = coverage
        rows.append(row)
    return pd.DataFrame(rows, columns=["symbol", "trade_date", *columns])


def _leave_one_out_statistics(values: np.ndarray, observed: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Exact leave-one-out moments for [stocks, features] standardized snapshots."""

    valid = observed & np.isfinite(values)
    safe = np.where(valid, values, 0.0).astype("float64", copy=False)
    count = valid.sum(axis=0).astype("float64")
    total = safe.sum(axis=0)
    total_sq = np.square(safe).sum(axis=0)
    positive = ((safe > 0.0) & valid).sum(axis=0).astype("float64")
    n = values.shape[0]

    own = safe
    own_valid = valid.astype("float64")
    loo_count = count[None, :] - own_valid
    loo_sum = total[None, :] - own
    loo_sq = total_sq[None, :] - np.square(own)
    denominator = np.maximum(loo_count, 1.0)
    mean = loo_sum / denominator
    variance = np.maximum(loo_sq / denominator - np.square(mean), 0.0)
    std = np.sqrt(variance)
    positive_ratio = (positive[None, :] - ((own > 0.0) & valid)) / denominator
    coverage = loo_count / max(n - 1, 1)

    absent = loo_count <= 0
    mean[absent] = np.nan
    std[absent] = np.nan
    positive_ratio[absent] = np.nan
    coverage[absent] = 0.0
    return mean, std, positive_ratio, coverage


def build_market_query_context(
    episodes: Sequence[StockDayEpisode],
    normalization: NormalizationState,
    *,
    episode_contract_hash: str,
    minimum_cross_section: int = 3,
) -> pd.DataFrame:
    """Create exact leave-one-out market context for every admitted query."""

    if minimum_cross_section < 2:
        raise ValueError("minimum_cross_section must be at least 2")
    feature_names = tuple(normalization.feature_names)
    context_columns = market_context_columns(feature_names)
    snapshots: Dict[int, list[Tuple[StockDayEpisode, int, np.ndarray, np.ndarray]]] = {}
    for episode in episodes:
        if tuple(episode.feature_names) != feature_names:
            raise ValueError("Episode feature order differs from normalization contract")
        for query_index, query_time_ns in enumerate(np.asarray(episode.query_times_ns, dtype="int64")):
            raw = episode.query_x[query_index, -1, :]
            observed = episode.query_observed[query_index, -1, :].astype(bool, copy=False)
            standardized = _standardized(raw[None, :], observed[None, :], normalization)[0]
            snapshots.setdefault(int(query_time_ns), []).append(
                (episode, query_index, standardized, np.isfinite(standardized))
            )

    rows = []
    for query_time_ns, items in sorted(snapshots.items()):
        values = np.stack([item[2] for item in items]).astype("float32", copy=False)
        observed = np.stack([item[3] for item in items]).astype(bool, copy=False)
        mean, std, positive_ratio, coverage = _leave_one_out_statistics(values, observed)
        cross_section_size = len(items)
        for stock_index, (episode, query_index, _, _) in enumerate(items):
            row: Dict[str, Any] = {
                "sample_id": stable_sample_id(episode, query_time_ns, episode_contract_hash),
                "symbol": str(episode.symbol),
                "trade_date": str(episode.trade_date),
                "query_time": pd.Timestamp(query_time_ns),
                "query_index": int(query_index),
                "cross_section_size": int(cross_section_size),
                "market_valid": bool(cross_section_size >= minimum_cross_section),
            }
            for feature_index, feature in enumerate(feature_names):
                row[f"market__{feature}__loo_mean"] = float(mean[stock_index, feature_index])
                row[f"market__{feature}__loo_std"] = float(std[stock_index, feature_index])
                row[f"market__{feature}__loo_positive_ratio"] = float(
                    positive_ratio[stock_index, feature_index]
                )
                row[f"market__{feature}__loo_coverage"] = float(coverage[stock_index, feature_index])
            rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "symbol",
            "trade_date",
            "query_time",
            "query_index",
            "cross_section_size",
            "market_valid",
            *context_columns,
        ],
    )


@dataclass
class ContextDateResult:
    identity: pd.DataFrame
    market: pd.DataFrame
    audit: Dict[str, Any]


def build_context_date(
    episodes: Sequence[StockDayEpisode],
    normalization: NormalizationState,
    *,
    episode_contract_hash: str,
    config: Optional[ContextCacheConfig] = None,
) -> ContextDateResult:
    config = config or ContextCacheConfig()
    config.validate()
    identity = build_identity_daily_summary(episodes, normalization)
    market = build_market_query_context(
        episodes,
        normalization,
        episode_contract_hash=episode_contract_hash,
        minimum_cross_section=config.minimum_cross_section,
    )
    trade_dates = sorted(set(identity.get("trade_date", pd.Series(dtype=str)).astype(str)))
    audit = {
        "trade_date": trade_dates[0] if len(trade_dates) == 1 else None,
        "episodes": int(len(episodes)),
        "identity_rows": int(len(identity)),
        "market_rows": int(len(market)),
        "market_valid_rows": int(market["market_valid"].sum()) if not market.empty else 0,
        "formula_version": config.formula_version,
    }
    return ContextDateResult(identity=identity, market=market, audit=audit)


class NFFContextCacheStore:
    """Small date-partitioned cache with strict contract validation."""

    def __init__(self, root: Union[str, Path], contract: Mapping[str, Any]):
        self.root = Path(root).expanduser().resolve()
        self.contract = dict(contract)
        self.contract_hash = contract_hash(self.contract)

    @property
    def contract_path(self) -> Path:
        return self.root / "context_contract.json"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"contract_hash": self.contract_hash, **self.contract}
        if self.contract_path.exists():
            existing = json.loads(self.contract_path.read_text(encoding="utf-8"))
            if existing.get("contract_hash") != self.contract_hash:
                raise RuntimeError("Context cache root belongs to a different contract")
            return
        self._atomic_json(self.contract_path, payload)

    def date_root(self, trade_date: str) -> Path:
        return self.root / "dates" / f"date={trade_date}"

    def is_complete(self, trade_date: str) -> bool:
        meta_path = self.date_root(trade_date) / "meta.json"
        if not meta_path.exists():
            return False
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("status") == "complete" and meta.get("contract_hash") == self.contract_hash

    def write_date(self, trade_date: str, result: ContextDateResult) -> None:
        date_root = self.date_root(trade_date)
        date_root.mkdir(parents=True, exist_ok=True)
        self._atomic_parquet(result.identity, date_root / "identity.parquet")
        self._atomic_parquet(result.market, date_root / "market.parquet")
        self._atomic_json(
            date_root / "meta.json",
            {"status": "complete", "contract_hash": self.contract_hash, **result.audit},
        )

    def read_identity(self, dates: Optional[Sequence[str]] = None) -> pd.DataFrame:
        selected = set(str(value) for value in dates) if dates is not None else None
        frames = []
        date_root = self.root / "dates"
        if not date_root.exists():
            return pd.DataFrame()
        for path in sorted(date_root.glob("date=*/identity.parquet")):
            trade_date = path.parent.name.split("=", 1)[-1]
            if selected is None or trade_date in selected:
                frames.append(pd.read_parquet(path))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def read_market(self, trade_date: str) -> pd.DataFrame:
        path = self.date_root(str(trade_date)) / "market.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    @staticmethod
    def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True, default=str), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
