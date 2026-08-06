import numpy as np
import pandas as pd

from nff_research.full_factor_engine import (
    FAMILY_WINDOWS,
    derive_prototype,
    expand_specs,
    parse_prototypes,
)


def _frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [["AAA", "BBB"], pd.date_range("2026-01-02 14:30", periods=3, freq="min", tz="UTC")],
        names=["instrument", "datetime"],
    )
    values = np.arange(len(index), dtype="float32") + 1
    return pd.DataFrame(
        {
            "bars_1m__open": values + 100,
            "bars_1m__close": values + 100.5,
            "bars_1m__high": values + 101,
            "bars_1m__low": values + 99.5,
            "bars_1m__volume": values * 10,
            "bars_1m__dollar_volume": values * 1000,
            "bars_1m__vwap": values + 100.25,
            "price_nvg_10m_full_degree_gini": values / 10,
            "price_nvg_10m_full_hub_share": values / 20,
            "price_nvg_10m_full_hub_age_norm": values / 30,
            "price_nvg_10m_full_motif_entropy": values / 40,
            "price_nvg_10m_full_edge_span_entropy": values / 50,
            "price_nvg_10m_hub_replacement_strength": values / 60,
            "price_nvg_10m_terminal_signed_edge_balance": values / 70,
            "price_nvg_10m_top_bottom_asymmetry": values / 80,
            "price_nvg_10m_terminal_slope_std": values / 90,
            "return_hvg_15m_terminal_degree": values / 10,
            "return_hvg_15m_terminal_long_edge_ratio": values / 20,
            "return_hvg_15m_degree_irreversibility_js": values / 30,
            "return_hvg_15m_motif_irreversibility_js": values / 40,
            "hawkes_lite__hawkes_total_intensity": values,
            "hawkes_lite__hawkes_intensity_imbalance": values / 10,
            "hawkes_derived__hawkes_signed_pressure": values / 20,
            "hawkes_derived__hawkes_pressure_strength": values / 30,
            "off_exchange_share": values / 100,
            "dark_signed_flow": values * 2,
            "lit_signed_flow": values,
            "venue_hhi": values / 50,
            "venue_entropy": values / 60,
            "dominant_venue_share": values / 70,
        },
        index=index,
    )


def test_authoritative_instruction_expands_all_a_to_k_specs():
    prototypes = parse_prototypes()
    specs = expand_specs(prototypes)
    assert len(prototypes) == 194
    assert len(specs) == 474
    assert set(FAMILY_WINDOWS) == set("ABCDEFGHIJK")
    assert {"C", "G", "J", "K"}.issubset(set(prototypes[i]["family"] for i in range(len(prototypes))))


def test_topology_hvg_and_venue_prototypes_use_published_source_fields():
    frame = _frame()
    assert derive_prototype(frame, "C01", "10m").notna().any()
    assert derive_prototype(frame, "G01", "15m").notna().any()
    assert derive_prototype(frame, "J01", "1m").notna().any()


def test_k13_to_k22_are_executable_when_dependencies_exist():
    frame = _frame()
    frame["hawkes_lite__hawkes_surprise_energy"] = 1.0
    frame["hawkes_lite__hawkes_imbalance_std_60s"] = 0.1
    frame["trade_nvg__trade_price_nvg_60s_hub_replacement_strength"] = 0.2
    frame["trade_nvg__trade_flow_nvg_60s_hub_replacement_strength"] = 0.2
    frame["hawkes_derived__hawkes_persistence"] = 0.2
    frame["hawkes_lite__hawkes_endogenous_share"] = 0.3
    frame["hawkes_derived__hawkes_intensity_regime_change"] = 0.2
    frame["hawkes_derived__hawkes_shock_regime_change"] = 0.2
    frame["trade_nvg__trade_active_second_ratio_60s"] = 0.8
    frame["trade_nvg__trade_active_second_ratio_1m"] = 0.8
    frame["trade_nvg__trade_price_stale_ratio_60s"] = 0.1
    frame["trades_1m_sketch__trade_size_hhi"] = 0.2
    frame["trades_1m_sketch__burstiness"] = 0.2
    frame["trades_1m_sketch__max_within_minute_silence_ns"] = 1.0
    frame["trades_1m_core__large_trade_buy_volume_proxy"] = 2.0
    frame["trades_1m_core__large_trade_sell_volume_proxy"] = 1.0
    frame["trades_1m_core__flow_persistence_15m"] = 0.5
    frame["minute_nvg__price_nvg_10m_terminal_slope_std"] = 0.2
    frame["price_nvg_10m_full_motif_entropy"] = 0.2
    frame["price_nvg_10m_hub_replacement_strength"] = 0.2
    frame["price_nvg_10m_full_degree_gini"] = 0.2
    frame["price_path_10m_range"] = 0.1
    frame["minute_nvg__price_path_10m_efficiency"] = 0.5
    frame["minute_nvg__price_path_10m_range"] = 0.1
    frame["traditional__realized_vol_5m"] = 0.1
    frame["traditional__realized_vol_30m"] = 0.1
    frame["traditional__momentum_15m"] = 0.1
    frame["hawkes_derived__hawkes_excess_intensity"] = 0.1
    frame["hawkes_derived__hawkes_exogenous_shock_60s"] = 0.1
    frame["hawkes_lite__hawkes_imbalance_std_60s"] = 0.1
    frame["minute_nvg__price_nvg_10m_terminal_signed_edge_balance"] = 0.1
    frame["minute_nvg__price_nvg_10m_top_bottom_asymmetry"] = 0.1
    frame["trade_nvg__trade_price_nvg_300s_terminal_signed_edge_balance"] = 0.1
    frame["trade_nvg__trade_flow_nvg_60s_terminal_signed_edge_balance"] = 0.1
    frame["trade_nvg__trade_price_nvg_60s_full_edge_span_entropy"] = 0.1
    for prototype in ("K13", "K14", "K15", "K16", "K17", "K18", "K19", "K20", "K21", "K22"):
        value = derive_prototype(frame, prototype, "1m")
        assert value is not None, prototype
