from __future__ import annotations

"""Avoid recomputing rankings, gates and sleeves for every cost scenario."""

from typing import Any, Mapping

import numpy as np
import pandas as pd


def expand_cost_scenarios(
    base: pd.DataFrame,
    costs_bps_one_way: list[float],
) -> pd.DataFrame:
    """Expand one gross/turnover path into cost scenarios without N deep copies.

    Row order intentionally matches the prior implementation: all base rows for
    cost[0], then all base rows for cost[1], and so on.  Only accounting fields
    differ across scenarios; rankings, Hawkes gates, weights and turnover are
    computed once upstream.  The persisted audit label remains the historical
    ``single_weight_path_expansion`` contract even though the expansion itself
    is now vectorized with NumPy.
    """
    if base.empty or not costs_bps_one_way:
        return base
    costs = np.asarray([float(value) for value in costs_bps_one_way], dtype="float64")
    repeats = len(costs)
    result = pd.concat([base] * repeats, ignore_index=True, copy=False)
    scenario_cost = np.repeat(costs, len(base))
    turnover = np.tile(
        pd.to_numeric(base["turnover"], errors="coerce").to_numpy(dtype="float64"),
        repeats,
    )
    gross = np.tile(
        pd.to_numeric(base["gross_return"], errors="coerce").to_numpy(dtype="float64"),
        repeats,
    )
    cost = turnover * scenario_cost / 10000.0
    result["cost_bps_per_turnover"] = scenario_cost
    result["cost"] = cost
    result["net_return"] = gross - cost
    result["cost_scenario_source"] = "single_weight_path_expansion"
    return result


def install(P: Any) -> None:
    original = P._selected_portfolio_proxy

    def selected_portfolio_proxy_single_path(
        features: pd.DataFrame,
        labels: pd.DataFrame,
        label_masks: pd.DataFrame,
        controls: pd.DataFrame,
        universes: pd.DataFrame,
        trade_date: str,
        candidate_names: list[str],
        config: Mapping[str, Any],
    ) -> pd.DataFrame:
        costs = [
            float(value)
            for value in config.get("portfolio_proxy", {}).get(
                "costs_bps_one_way", [1.0]
            )
        ]
        # Copy only the two mappings whose accounting setting changes.  A deep
        # copy of the full campaign config is unnecessary for every date.
        one_path_config = dict(config)
        one_path_config["portfolio_proxy"] = {
            **dict(config.get("portfolio_proxy", {})),
            "costs_bps_one_way": [0.0],
        }
        base = original(
            features,
            labels,
            label_masks,
            controls,
            universes,
            trade_date,
            candidate_names,
            one_path_config,
        )
        return expand_cost_scenarios(base, costs)

    P._selected_portfolio_proxy = selected_portfolio_proxy_single_path
