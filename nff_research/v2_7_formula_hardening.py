from __future__ import annotations

"""Financially equivalent formula remapping for the real 476-factor contract."""

from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-8


def _mean(values: list[pd.Series | None]) -> pd.Series | None:
    available = [value for value in values if value is not None]
    return pd.concat(available, axis=1).mean(axis=1) if available else None


def install(campaign: Any, hardening: Any) -> None:
    base_correct = campaign._correct_family
    base_enrich = hardening._enrich_causal_paths

    def enrich(frame: pd.DataFrame, context: Any) -> pd.DataFrame:
        result = base_enrich(frame, context)
        close = result.get("bars_1m__close")
        if close is not None and "traditional__realized_vol_5m" not in result:
            close = pd.to_numeric(close, errors="coerce")
            groups = result.index.get_level_values("instrument")
            ret1 = close.groupby(groups, sort=False, group_keys=False).pct_change(fill_method=None)
            rv5 = (
                ret1.pow(2)
                .groupby(level="instrument", sort=False)
                .rolling(5, min_periods=3)
                .sum()
                .reset_index(level=0, drop=True)
                .reindex(result.index)
                .pow(0.5)
                .astype("float32")
            )
            result = pd.concat(
                [result, rv5.rename("traditional__realized_vol_5m")], axis=1, copy=False
            )
        return result

    def correct(
        frame: pd.DataFrame,
        values: dict[str, pd.Series | None],
        family: str,
        window: str,
    ) -> dict[str, pd.Series | None]:
        result = base_correct(frame, values, family, window)
        groups = frame.index.get_level_values("datetime")

        def get(*names: str) -> pd.Series | None:
            return campaign._series(frame, *names)

        if family == "A":
            m15 = get("traditional__momentum_15m", "minute_nvg__momentum_15m")
            m30 = get("traditional__momentum_30m", "minute_nvg__momentum_30m")
            m60 = get("traditional__momentum_60m", "minute_nvg__momentum_60m")
            if all(value is not None for value in (m15, m30, m60)):
                result["A14"] = pd.concat([m15, m30, m60], axis=1).apply(np.sign).mean(axis=1)
                result["A15"] = campaign._csz(m15, groups) - campaign._csz(m60, groups)
                result["A16"] = (
                    campaign._csz(m15, groups)
                    - 2.0 * campaign._csz(m30, groups)
                    + campaign._csz(m60, groups)
                )
            return result

        if family != "K":
            return result

        price = get(
            "minute_nvg__price_nvg_15m_terminal_signed_edge_balance",
            "price_nvg_15m_terminal_signed_edge_balance",
        )
        flow = get(
            "trade_nvg__trade_flow_nvg_60s_terminal_signed_edge_balance",
            "trade_flow_nvg_60s_terminal_signed_edge_balance",
        )
        trade_price = get(
            "trade_nvg__trade_price_nvg_300s_terminal_signed_edge_balance",
            "trade_price_nvg_300s_terminal_signed_edge_balance",
        )
        hawkes_pressure = get("hawkes_derived__hawkes_signed_pressure")
        hawkes_strength = get("hawkes_derived__hawkes_pressure_strength")
        hawkes_persistence = get("hawkes_derived__hawkes_persistence")
        hawkes_endogenous = get("hawkes_lite__hawkes_endogenous_share")
        imbalance_std = get("hawkes_lite__hawkes_imbalance_std_60s")
        exogenous_shock = get("hawkes_derived__hawkes_exogenous_shock_60s")
        shock_regime = get(
            "hawkes_derived__hawkes_shock_regime_change",
            "hawkes_derived__hawkes_intensity_regime_change",
        )
        surprise_energy = get("hawkes_lite__hawkes_surprise_energy")
        excess_intensity = get("hawkes_derived__hawkes_excess_intensity")
        duration = get("hawkes_derived__hawkes_effective_duration_norm")

        path_efficiency = get("minute_nvg__price_path_15m_efficiency")
        path_range = get("minute_nvg__price_path_15m_range")
        momentum = get("traditional__momentum_15m", "minute_nvg__momentum_15m")
        rv5 = get("traditional__realized_vol_5m")
        rv30 = get("traditional__realized_vol_30m")
        price_asymmetry = get(
            "price_nvg_15m_top_bottom_asymmetry",
            "minute_nvg__price_nvg_15m_top_bottom_asymmetry",
        )
        price_volume_confirmation = get("minute_nvg__price_volume_nvg_confirmation_15m")
        if price_volume_confirmation is None:
            jaccard = get("price_volume_nvg_15m_edge_weighted_jaccard")
            slope_corr = get("price_volume_nvg_15m_common_edge_slope_corr")
            terms = []
            if jaccard is not None:
                terms.append(jaccard.clip(0, 1))
            if slope_corr is not None:
                terms.append((slope_corr.clip(-1, 1) + 1.0) / 2.0)
            price_volume_confirmation = _mean(terms)

        degree_gini = get("price_nvg_15m_full_degree_gini")
        motif_entropy = get("price_nvg_15m_full_motif_entropy")
        hub_replacement = get("price_nvg_15m_hub_replacement_strength")
        slope_std = get(
            "price_nvg_15m_terminal_slope_std",
            "minute_nvg__price_nvg_15m_terminal_slope_std",
        )
        trade_price_hub = get("trade_price_nvg_60s_hub_replacement_strength")
        trade_flow_hub = get("trade_flow_nvg_60s_hub_replacement_strength")
        trade_span_entropy = get("trade_price_nvg_60s_full_edge_span_entropy")
        return_long_ratio = get("return_hvg_15m_terminal_long_edge_ratio")

        burstiness = get("trades_1m_core__burstiness", "trades_1m_sketch__burstiness")
        silence = get(
            "trades_1m_core__max_within_minute_silence_ns",
            "trades_1m_sketch__max_within_minute_silence_ns",
        )
        trade_size_hhi = get("trades_1m_core__trade_size_hhi", "trades_1m_sketch__trade_size_hhi")
        activity = get("trade_nvg__trade_active_second_ratio_60s")
        stale = get("trade_nvg__trade_price_stale_ratio_60s")
        signed_flow = get("trades_1m_core__signed_dollar_flow_proxy")
        flow_persistence = get("trades_1m_core__flow_persistence_15m")
        large_buy = get("trades_1m_core__large_trade_buy_volume_proxy")
        large_sell = get("trades_1m_core__large_trade_sell_volume_proxy")
        large_imbalance = (
            (large_buy - large_sell) / (large_buy + large_sell + EPS)
            if large_buy is not None and large_sell is not None
            else None
        )
        close = get("bars_1m__close")
        dollar = get("bars_1m__dollar_volume", "trades_1m_core__dollar_volume")
        if close is not None:
            instruments = frame.index.get_level_values("instrument")
            ret1 = close.groupby(instruments, sort=False, group_keys=False).pct_change(fill_method=None)
        else:
            ret1 = None
        amihud = ret1.abs() / (dollar + EPS) if ret1 is not None and dollar is not None else None

        flow_absorption = None
        if flow is not None and trade_price is not None:
            flow_absorption = -np.sign(flow) * campaign._csz(flow, groups).abs() * (
                (np.sign(flow) != np.sign(trade_price))
                | (trade_price.abs() < trade_price.abs().groupby(groups, sort=False).transform("median"))
            ).astype(float)
        overextension_reversal = (
            -np.sign(momentum) * price_asymmetry.abs()
            if momentum is not None and price_asymmetry is not None
            else None
        )
        endogenous_pressure = (
            hawkes_pressure * hawkes_endogenous
            if hawkes_pressure is not None and hawkes_endogenous is not None
            else None
        )

        k: dict[str, pd.Series | None] = {}
        if price is not None and trade_price is not None:
            k["K01"] = campaign._conf(price, trade_price, groups)
        if price is not None and flow is not None:
            k["K02"] = campaign._conf(price, flow, groups)
        if price is not None and hawkes_pressure is not None:
            k["K03"] = campaign._conf(price, hawkes_pressure, groups)
        if flow is not None and hawkes_pressure is not None:
            k["K04"] = campaign._conf(flow, hawkes_pressure, groups)
        if all(value is not None for value in (price, flow, hawkes_pressure)):
            same = (
                (np.sign(price) == np.sign(flow))
                & (np.sign(price) == np.sign(hawkes_pressure))
            ).astype(float)
            k["K05"] = _mean(
                [campaign._csr(price, groups), campaign._csr(flow, groups), campaign._csr(hawkes_pressure, groups)]
            ) * same
            k["K06"] = -np.sign(price) * campaign._csz(price, groups).abs() * (
                (np.sign(flow) != np.sign(price))
                & (np.sign(hawkes_pressure) != np.sign(price))
            ).astype(float)
            weak_price = price.abs() < price.abs().groupby(groups, sort=False).transform("median")
            k["K07"] = _mean([campaign._csr(flow, groups), campaign._csr(hawkes_pressure, groups)]) * weak_price.astype(float)
        if all(value is not None for value in (price, flow, price_volume_confirmation)):
            k["K08"] = price * price_volume_confirmation * (0.5 + 0.5 * campaign._same(price, flow))
        if all(value is not None for value in (price, hawkes_pressure, hawkes_strength)):
            k["K09"] = np.sign(price) * hawkes_strength * campaign._same(price, hawkes_pressure)
            absorbed = (
                (np.sign(price) != np.sign(hawkes_pressure))
                | (price.abs() < price.abs().groupby(groups, sort=False).transform("median"))
            ).astype(float)
            k["K10"] = -np.sign(hawkes_pressure) * hawkes_strength * absorbed
        if all(value is not None for value in (price, path_efficiency, price_volume_confirmation)):
            k["K11"] = campaign._csr(price, groups) * _mean([path_efficiency, price_volume_confirmation])
        if all(value is not None for value in (overextension_reversal, flow_absorption, endogenous_pressure)):
            k["K12"] = _mean([overextension_reversal, flow_absorption, endogenous_pressure])
        if all(value is not None for value in (surprise_energy, burstiness, activity, large_imbalance)):
            activity_breakout = campaign._csz(activity, groups)
            k["K13"] = _mean(
                [
                    campaign._csr(surprise_energy, groups),
                    campaign._csr(burstiness, groups),
                    campaign._csr(activity_breakout, groups),
                    campaign._csr(large_imbalance.abs(), groups),
                ]
            )
        if all(value is not None for value in (amihud, silence, stale, activity, trade_size_hhi)):
            k["K14"] = _mean(
                [
                    campaign._csr(amihud, groups),
                    campaign._csr(silence, groups),
                    campaign._csr(stale, groups),
                    campaign._csr(1.0 - activity, groups),
                    campaign._csr(trade_size_hhi, groups),
                ]
            )
        if all(value is not None for value in (hub_replacement, trade_price_hub, trade_flow_hub, shock_regime)):
            k["K15"] = _mean(
                [
                    campaign._csr(hub_replacement, groups),
                    campaign._csr(trade_price_hub, groups),
                    campaign._csr(trade_flow_hub, groups),
                    campaign._csr(shock_regime, groups),
                ]
            )
        if all(value is not None for value in (price, path_efficiency, motif_entropy)):
            k["K16"] = (
                campaign._csr(price, groups).abs()
                * path_efficiency
                * (1.0 - campaign._csr(motif_entropy, groups).abs())
            )
        if all(value is not None for value in (slope_std, motif_entropy, imbalance_std, rv5)):
            k["K17"] = _mean(
                [
                    campaign._csr(slope_std, groups),
                    campaign._csr(motif_entropy, groups),
                    campaign._csr(imbalance_std, groups),
                    campaign._csr(rv5, groups),
                ]
            )
        if all(value is not None for value in (rv5, path_range, price, excess_intensity)):
            k["K18"] = _mean(
                [
                    campaign._csr(-rv5, groups),
                    campaign._csr(-path_range, groups),
                    campaign._csr(-price.abs(), groups),
                    campaign._csr(-excess_intensity, groups),
                ]
            )
        if all(value is not None for value in (hawkes_persistence, hawkes_endogenous, imbalance_std)):
            score = (
                campaign._csz(hawkes_persistence, groups)
                + campaign._csz(hawkes_endogenous, groups)
                - campaign._csz(imbalance_std, groups)
            ).clip(-20, 20)
            k["K19"] = 1.0 / (1.0 + np.exp(-score))
        if all(value is not None for value in (exogenous_shock, shock_regime, hawkes_persistence)):
            score = (
                campaign._csz(exogenous_shock, groups)
                + campaign._csz(shock_regime, groups)
                - campaign._csz(hawkes_persistence, groups)
            ).clip(-20, 20)
            k["K20"] = 1.0 / (1.0 + np.exp(-score))
        if momentum is not None and rv30 is not None:
            k["K21"] = momentum / (rv30.abs() + EPS)
        if all(value is not None for value in (duration, return_long_ratio, trade_span_entropy, flow_persistence)):
            k["K22"] = _mean(
                [
                    campaign._csr(duration, groups),
                    campaign._csr(return_long_ratio, groups),
                    campaign._csr(trade_span_entropy, groups),
                    campaign._csr(flow_persistence, groups),
                ]
            )
        result.update(k)
        return result

    hardening._enrich_causal_paths = enrich
    campaign._correct_family = correct
