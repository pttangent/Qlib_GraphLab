from __future__ import annotations

"""Exact, vectorized within-minute decile aggregation.

For unique ranks r=1..N, pandas qcut uses linearly interpolated quantile edges
`1 + (N-1) * k/10`. The right-closed bin index is therefore
`max(1, ceil(10*(r-1)/(N-1)))`, not `ceil(10*r/N)` and not
`floor(10*(r-1)/N)+1`.
"""

import math
from typing import Any

import numpy as np
import pandas as pd


def qcut_deciles_from_unique_ranks(ranks: np.ndarray, counts: np.ndarray) -> np.ndarray:
    ranks = np.asarray(ranks, dtype="float64")
    counts = np.asarray(counts, dtype="float64")
    denominator = np.maximum(counts - 1.0, 1.0)
    bins = np.maximum(1.0, np.ceil(10.0 * (ranks - 1.0) / denominator))
    bins = np.where(counts <= 1.0, 1.0, bins)
    return bins.clip(1, 10).astype("int16")


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
