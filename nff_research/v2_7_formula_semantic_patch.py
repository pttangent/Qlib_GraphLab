from __future__ import annotations

"""Final semantic corrections for direction-sensitive B/D/E prototypes.

These overrides intentionally prefer the rebuilt terminal-direction supplement.
Top/bottom asymmetry remains a separate geometry factor and is never used as a
silent substitute for signed terminal direction.
"""

from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-8


def _causal_tsz(series: pd.Series, frame: pd.DataFrame, window: int = 30, minimum: int = 10) -> pd.Series:
    instruments = frame.index.get_level_values("instrument")
    prior = series.groupby(instruments, sort=False, group_keys=False).shift(1)
    mean = (
        prior.groupby(instruments, sort=False)
        .rolling(window, min_periods=minimum)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(series.index)
    )
    std = (
        prior.groupby(instruments, sort=False)
        .rolling(window, min_periods=minimum)
        .std(ddof=0)
        .reset_index(level=0, drop=True)
        .reindex(series.index)
    )
    return (series - mean) / (std.replace(0.0, np.nan) + EPS)


def install(campaign: Any) -> None:
    base_correct = campaign._correct_family

    def correct(
        frame: pd.DataFrame,
        values: dict[str, pd.Series | None],
        family: str,
        window: str,
    ) -> dict[str, pd.Series | None]:
        result = base_correct(frame, values, family, window)
        groups = frame.index.get_level_values("datetime")
        instruments = frame.index.get_level_values("instrument")

        def get(*names: str) -> pd.Series | None:
            return campaign._series(frame, *names)

        if family == "B":
            edge = get(
                f"minute_nvg__price_nvg_{window}_terminal_signed_edge_balance",
                f"price_nvg_{window}_terminal_signed_edge_balance",
            )
            long_slope = get(
                f"minute_nvg__price_nvg_{window}_terminal_long_edge_signed_slope",
                f"price_nvg_{window}_terminal_long_edge_signed_slope",
            )
            top_entropy = get(
                f"minute_nvg__price_nvg_{window}_top_terminal_span_entropy",
                f"price_nvg_{window}_top_terminal_span_entropy",
            )
            bottom_entropy = get(
                f"minute_nvg__price_nvg_{window}_bottom_terminal_span_entropy",
                f"price_nvg_{window}_bottom_terminal_span_entropy",
            )
            if edge is not None and long_slope is not None:
                # Long-horizon directional slope minus all-visible-edge
                # direction. This preserves the original fast/slow hypothesis;
                # long_edge_ratio is only a memory-strength variable.
                result["B04"] = campaign._csz(long_slope, groups) - campaign._csz(edge, groups)
            if edge is not None and top_entropy is not None and bottom_entropy is not None:
                entropy = 0.5 * (top_entropy + bottom_entropy)
                result["B19"] = edge * (1.0 - entropy.clip(0.0, 1.0))
            return result

        if family == "D":
            volume_change = get(f"minute_nvg__volume_path_{window}_signed_change")
            overlap = get(
                f"minute_nvg__price_volume_terminal_overlap_{window}",
                f"price_volume_terminal_overlap_{window}",
                f"price_volume_nvg_overlap_{window}",
            )
            if volume_change is not None:
                # D04 is a within-symbol abnormal volume expansion, not a
                # cross-sectional size/liquidity rank.
                result["D04"] = _causal_tsz(volume_change, frame, window=30, minimum=10)
            if overlap is not None:
                result["D14"] = overlap - overlap.groupby(
                    instruments, sort=False, group_keys=False
                ).shift(1)
            return result

        if family != "E":
            return result

        price_edge = get(
            f"trade_nvg__trade_price_nvg_{window}_terminal_signed_edge_balance",
            f"trade_price_nvg_{window}_terminal_signed_edge_balance",
        )
        flow_edge = get(
            f"trade_nvg__trade_flow_nvg_{window}_terminal_signed_edge_balance",
            f"trade_flow_nvg_{window}_terminal_signed_edge_balance",
        )
        price_long = get(
            f"trade_nvg__trade_price_nvg_{window}_terminal_long_edge_signed_slope",
            f"trade_price_nvg_{window}_terminal_long_edge_signed_slope",
        )
        flow_long = get(
            f"trade_nvg__trade_flow_nvg_{window}_terminal_long_edge_signed_slope",
            f"trade_flow_nvg_{window}_terminal_long_edge_signed_slope",
        )
        overlap = get(
            f"trade_nvg__trade_price_flow_terminal_overlap_{window}",
            f"trade_price_flow_terminal_overlap_{window}",
            f"trade_price_flow_nvg_overlap_{window}",
        )
        jaccard = get(
            f"trade_nvg__trade_price_flow_nvg_{window}_edge_weighted_jaccard",
            f"trade_price_flow_nvg_{window}_edge_weighted_jaccard",
        )
        slope_corr = get(
            f"trade_nvg__trade_price_flow_nvg_{window}_common_edge_slope_corr",
            f"trade_price_flow_nvg_{window}_common_edge_slope_corr",
        )
        motif = get(
            f"trade_nvg__trade_price_flow_nvg_{window}_full_motif_cosine",
            f"trade_price_flow_nvg_{window}_full_motif_cosine",
        )
        hub_overlap = get(
            f"trade_nvg__trade_price_flow_nvg_{window}_full_hub_time_overlap",
            f"trade_price_flow_nvg_{window}_full_hub_time_overlap",
        )
        price_asymmetry = get(
            f"trade_nvg__trade_price_nvg_{window}_top_bottom_asymmetry",
            f"trade_price_nvg_{window}_top_bottom_asymmetry",
        )
        price_asymmetry_change = get(
            f"trade_nvg__trade_price_nvg_{window}_top_bottom_asymmetry_change_1m",
            f"trade_price_nvg_{window}_top_bottom_asymmetry_change_1m",
        )
        price_hub_replace = get(
            f"trade_nvg__trade_price_nvg_{window}_hub_replacement_strength",
            f"trade_price_nvg_{window}_hub_replacement_strength",
        )
        minute_price = get(
            "minute_nvg__price_nvg_15m_terminal_signed_edge_balance",
            "price_nvg_15m_terminal_signed_edge_balance",
        )

        if price_edge is not None:
            result["E01"] = price_edge
        if flow_edge is not None:
            result["E02"] = flow_edge
        if price_long is not None:
            result["E03"] = price_long
        if flow_long is not None:
            result["E04"] = flow_long
        if price_edge is not None and flow_edge is not None:
            result["E09"] = campaign._conf(price_edge, flow_edge, groups)
            result["E15"] = (
                -np.sign(flow_edge)
                * campaign._csz(flow_edge, groups).abs()
                * (
                    (np.sign(price_edge) != np.sign(flow_edge))
                    | (
                        price_edge.abs()
                        < price_edge.abs().groupby(groups, sort=False).transform("median")
                    )
                ).astype(float)
            )
        if price_edge is not None and overlap is not None:
            result["E10"] = price_edge * overlap
            result["E16"] = price_edge * (1.0 - overlap)
        if price_edge is not None and jaccard is not None:
            result["E11"] = price_edge * jaccard
        if price_edge is not None and slope_corr is not None:
            result["E12"] = price_edge * slope_corr
        if price_edge is not None and motif is not None:
            result["E13"] = price_edge * motif
        if price_edge is not None and hub_overlap is not None:
            result["E14"] = price_edge * hub_overlap
        if price_asymmetry is not None:
            result["E17"] = price_asymmetry
            if price_asymmetry_change is None:
                price_asymmetry_change = price_asymmetry - price_asymmetry.groupby(
                    instruments, sort=False, group_keys=False
                ).shift(1)
        if price_asymmetry_change is not None:
            result["E18"] = price_asymmetry_change
        if price_edge is not None and price_hub_replace is not None:
            result["E19"] = -np.sign(price_edge) * price_hub_replace

        price_60 = get(
            "trade_nvg__trade_price_nvg_60s_terminal_signed_edge_balance",
            "trade_price_nvg_60s_terminal_signed_edge_balance",
        )
        price_180 = get(
            "trade_nvg__trade_price_nvg_180s_terminal_signed_edge_balance",
            "trade_price_nvg_180s_terminal_signed_edge_balance",
        )
        price_300 = get(
            "trade_nvg__trade_price_nvg_300s_terminal_signed_edge_balance",
            "trade_price_nvg_300s_terminal_signed_edge_balance",
        )
        if price_60 is not None and price_300 is not None:
            result["E20"] = campaign._csz(price_60, groups) - campaign._csz(price_300, groups)
        if all(value is not None for value in (price_60, price_180, price_300)):
            result["E21"] = pd.concat([price_60, price_180, price_300], axis=1).apply(
                np.sign
            ).mean(axis=1)
        if price_edge is not None and minute_price is not None:
            result["E22"] = price_edge * (
                np.sign(price_edge) != np.sign(minute_price)
            ).astype(float)
        return result

    campaign._correct_family = correct
