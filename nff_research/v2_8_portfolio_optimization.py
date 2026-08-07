from __future__ import annotations

"""Avoid recomputing rankings, gates and sleeves for every cost scenario."""

from copy import deepcopy
from typing import Any, Mapping

import pandas as pd


def expand_cost_scenarios(
    base: pd.DataFrame,
    costs_bps_one_way: list[float],
) -> pd.DataFrame:
    if base.empty:
        return base
    parts: list[pd.DataFrame] = []
    for cost_bps in costs_bps_one_way:
        part = base.copy(deep=False).copy()
        cost = pd.to_numeric(part["turnover"], errors="coerce") * float(cost_bps) / 10000.0
        part["cost_bps_per_turnover"] = float(cost_bps)
        part["cost"] = cost
        part["net_return"] = pd.to_numeric(part["gross_return"], errors="coerce") - cost
        part["cost_scenario_source"] = "single_weight_path_expansion"
        parts.append(part)
    return pd.concat(parts, ignore_index=True, copy=False)


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
        one_path_config = deepcopy(dict(config))
        one_path_config.setdefault("portfolio_proxy", {})[
            "costs_bps_one_way"
        ] = [0.0]
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
