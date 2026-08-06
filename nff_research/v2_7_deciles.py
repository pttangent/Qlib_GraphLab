from __future__ import annotations

"""Exact, vectorized within-minute decile aggregation.

Pandas qcut constructs linearly interpolated quantile edges and uses
right-closed bins. Floating interpolation is observable for some cross-section
sizes (for example N=91), so an algebraically simplified ceil formula is not
byte-for-byte equivalent. We cache the exact edge vector for each N and use
NumPy searchsorted; the expensive aggregation remains vectorized.
"""

import math
from typing import Any

import numpy as np
import pandas as pd


_QCUT_EDGE_CACHE: dict[int, np.ndarray] = {}


def _qcut_edges(count: int) -> np.ndarray:
    count = int(count)
    cached = _QCUT_EDGE_CACHE.get(count)
    if cached is not None:
        return cached
    if count <= 1:
        edges = np.array([1.0, 1.0], dtype="float64")
    else:
        values = np.arange(1, count + 1, dtype="float64")
        edges = np.quantile(values, np.linspace(0.0, 1.0, 11), method="linear")
    _QCUT_EDGE_CACHE[count] = edges
    return edges


def qcut_deciles_from_unique_ranks(ranks: np.ndarray, counts: np.ndarray) -> np.ndarray:
    ranks = np.asarray(ranks, dtype="float64")
    counts = np.asarray(counts, dtype="int64")
    result = np.ones(len(ranks), dtype="int16")
    for count in np.unique(counts):
        mask = counts == count
        if int(count) <= 1:
            result[mask] = 1
            continue
        # searchsorted over the complete edge vector returns 0 for the minimum;
        # clipping it to one reproduces qcut(include_lowest=True) and its
        # right-closed interval assignment for all other values.
        result[mask] = np.searchsorted(
            _qcut_edges(int(count)), ranks[mask], side="left"
        ).clip(1, 10).astype("int16")
    return result


def decile_feature_rows(
    research: Any,
    feature: str,
    signal: pd.Series,
    label: pd.Series,
    adv: pd.Series,
    price: pd.Series,
    trades: pd.Series,
    metadata: dict[str, object],
    min_n: int,
) -> list[dict[str, object]]:
    work = pd.concat(
        [
            signal.rename("signal"),
            label.rename("label"),
            adv.rename("adv"),
            price.rename("price"),
            trades.rename("trades"),
        ],
        axis=1,
    ).dropna(subset=["signal", "label"])
    if work.empty:
        return []
    minute = work.index.get_level_values("datetime").minute
    work = work.loc[(minute % 15) == 0]
    if work.empty:
        return []

    categories = pd.Categorical(work.index.get_level_values("datetime"))
    minute_codes = categories.codes.astype("int32")
    minute_counts = np.bincount(minute_codes, minlength=len(categories.categories))
    eligible = minute_counts >= min_n
    unique_ranks = work["signal"].groupby(level="datetime", sort=False).rank(method="first")
    deciles = qcut_deciles_from_unique_ranks(
        unique_ranks.to_numpy(dtype="float64"), minute_counts[minute_codes]
    )
    combined = minute_codes * 10 + deciles - 1
    size = len(categories.categories) * 10

    def aggregate(column: str) -> tuple[np.ndarray, np.ndarray]:
        values = pd.to_numeric(work[column], errors="coerce").to_numpy(dtype="float64")
        valid = eligible[minute_codes] & np.isfinite(values)
        sums = np.bincount(combined[valid], weights=values[valid], minlength=size).reshape(-1, 10)
        counts = np.bincount(combined[valid], minlength=size).reshape(-1, 10)
        means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        return means, counts

    label_means, label_counts = aggregate("label")
    adv_means, _ = aggregate("adv")
    price_means, _ = aggregate("price")
    trade_means, _ = aggregate("trades")
    rows: list[dict[str, object]] = []
    for decile in range(10):
        def mean(values: np.ndarray) -> float:
            column = values[:, decile]
            return float(np.nanmean(column)) if np.isfinite(column).any() else math.nan

        rows.append(
            {
                **metadata,
                "feature": feature,
                "bundle": research.infer_bundle(feature),
                "rebalance": "sample_15m",
                "decile": decile + 1,
                "mean_label": mean(label_means),
                "count": int(label_counts[:, decile].sum()),
                "mean_log_adv20": mean(adv_means),
                "mean_price": mean(price_means),
                "mean_trade_count": mean(trade_means),
            }
        )
    return rows
