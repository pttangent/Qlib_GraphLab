from __future__ import annotations

"""Warehouse-exact factor contract for the v2.7 campaign.

The historical A-K instruction expands to 474 formal specifications and two
S-direction specifications.  The current physical warehouse supports 462 A-K
specifications plus the two S factors.  The only legacy-only gap is C@60m:
price-NVG topology is physically published at 10m/15m/30m, while 60m belongs
to the HVG risk family G.  The gap is reported explicitly and is never filled
with a renamed 30m field or an HVG substitute.
"""

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


CONTRACT_VERSION = "v2.7-warehouse-exact-20260806"
LEGACY_FORMAL_COUNT = 476
PHYSICAL_EXECUTABLE_COUNT = 464
S_WINDOWS = (15, 30)

LEGACY_WINDOWS: dict[str, tuple[str, ...]] = {
    "A": ("10m", "15m", "30m"),
    "B": ("10m", "15m", "30m"),
    "C": ("10m", "15m", "30m", "60m"),
    "D": ("10m", "15m", "30m"),
    "E": ("60s", "180s", "300s"),
    "F": ("60s", "180s", "300s"),
    "G": ("15m", "30m", "60m"),
    "H": ("60s", "180s", "300s"),
    "I": ("1m",),
    "J": ("1m",),
    "K": ("1m",),
}

PHYSICAL_WINDOWS: dict[str, tuple[str, ...]] = {
    **LEGACY_WINDOWS,
    "C": ("10m", "15m", "30m"),
}

# v2.6 K formulas were defined with a fixed 10m minute-price/topology anchor.
# v2.7_formula_hardening originally requested the same fields at 15m.  During
# K derivation only, map those requests to the audited 10m physical columns.
_K_15_TO_10: dict[str, str] = {
    "minute_nvg__price_nvg_15m_terminal_signed_edge_balance": "minute_nvg__price_nvg_10m_terminal_signed_edge_balance",
    "price_nvg_15m_terminal_signed_edge_balance": "price_nvg_10m_terminal_signed_edge_balance",
    "minute_nvg__price_path_15m_efficiency": "minute_nvg__price_path_10m_efficiency",
    "minute_nvg__price_path_15m_range": "minute_nvg__price_path_10m_range",
    "price_nvg_15m_top_bottom_asymmetry": "price_nvg_10m_top_bottom_asymmetry",
    "minute_nvg__price_nvg_15m_top_bottom_asymmetry": "minute_nvg__price_nvg_10m_top_bottom_asymmetry",
    "minute_nvg__price_volume_nvg_confirmation_15m": "minute_nvg__price_volume_nvg_confirmation_10m",
    "price_volume_nvg_15m_edge_weighted_jaccard": "price_volume_nvg_10m_edge_weighted_jaccard",
    "price_volume_nvg_15m_common_edge_slope_corr": "price_volume_nvg_10m_common_edge_slope_corr",
    "price_nvg_15m_full_degree_gini": "price_nvg_10m_full_degree_gini",
    "price_nvg_15m_full_motif_entropy": "price_nvg_10m_full_motif_entropy",
    "price_nvg_15m_hub_replacement_strength": "price_nvg_10m_hub_replacement_strength",
    "price_nvg_15m_terminal_slope_std": "price_nvg_10m_terminal_slope_std",
    "minute_nvg__price_nvg_15m_terminal_slope_std": "minute_nvg__price_nvg_10m_terminal_slope_std",
}


def _supplement_rows(campaign: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window in S_WINDOWS:
        fields = campaign.directional_supplement_fields(window)
        rows.append(
            {
                "factor_id": f"full_factor__s01__w{window}m",
                "prototype_id": "S01",
                "family": "S",
                "title": "Exact NVG supplement directional consensus",
                "role": "DIRECTION_ALPHA",
                "direction_prior": "TWO_SIDED",
                "source_fields": "|".join(fields.values()),
                "formula": "raw+detrended price direction; volume and cross-graph structure as confidence",
                "financial_hypothesis": "price direction confirmed by detrended geometry and volume structure is more persistent",
                "window": f"{window}m",
                "transform": "robust cross-sectional z-score and bounded confidence",
                "label_group": "return,state,risk,liquidity,cost",
                "required_gate": "PIT and liquidity/coverage",
                "available_time_rule": "supplement available_time <= decision time; entry next exact minute",
                "expected_range": "approximately [-1,1]",
                "neutralization_allowed": True,
                "cost_relevance": "direct",
                "status": "PENDING_SCHEMA_RESOLUTION",
                "instruction_line": None,
                "physical_support": True,
                "support_reason": "v3 directional contract publishes 15m/30m derived direction factors",
                "formula_override": "exact minute_nvg_edge_raw direction components",
            }
        )
    return pd.DataFrame(rows)


def _expanded(campaign: Any, windows: Mapping[str, tuple[str, ...]]) -> pd.DataFrame:
    previous = dict(campaign.FF.FAMILY_WINDOWS)
    try:
        campaign.FF.FAMILY_WINDOWS = dict(windows)
        specs = campaign.FF.expand_specs(campaign.V26.PROTOTYPES).copy()
    finally:
        campaign.FF.FAMILY_WINDOWS = previous
    # G01-G09 are minute-HVG (Wh); G10-G14 are Trade-HVG (We).  They share a
    # family in the written A-K contract but must retain their independent
    # physical clocks.  The ordinal mapping is the published 15m/30m/60m
    # minute horizon to 60s/180s/300s trade horizon.
    trade_windows = {"15m": "60s", "30m": "180s", "60m": "300s"}
    trade_mask = specs["family"].eq("G") & specs["prototype_id"].isin(
        [f"G{value:02d}" for value in range(10, 15)]
    )
    specs.loc[trade_mask, "window"] = specs.loc[trade_mask, "window"].map(trade_windows)
    specs.loc[trade_mask, "factor_id"] = specs.loc[trade_mask].apply(
        lambda row: campaign.FF.factor_name(row["prototype_id"], row["window"]), axis=1
    )
    return specs


def _sync_feature_consumers(campaign: Any, names: list[str]) -> None:
    campaign.V26.FULL_FACTOR_NAMES = list(names)
    campaign.R.CORE_DECILE_FEATURES = list(names)
    campaign.BASE.DERIVED_FEATURES = list(names)
    campaign.BASE.run_temporal_oos.__globals__["DERIVED_FEATURES"] = list(names)
    campaign.BASE._analysis_features = campaign.V26._full_analysis_features
    campaign.BASE._materialize_model_cache.__globals__["_analysis_features"] = (
        campaign.V26._full_analysis_features
    )


def _k_anchor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    replacements: dict[str, pd.Series] = {}
    for target, source in _K_15_TO_10.items():
        if source in frame.columns:
            replacements[target] = frame[source]
    if not replacements:
        return frame
    block = pd.concat(replacements, axis=1, copy=False)
    block.columns = list(replacements)
    keep = frame.drop(columns=[column for column in replacements if column in frame], errors="ignore")
    return pd.concat([keep, block], axis=1, copy=False)


def install(campaign: Any) -> None:
    base_configure = campaign.configure_registry
    base_add_all = campaign._add_all_features
    # Keep the pre-gate builder available to the v2.8 streaming materializer.
    # The strict wrapper below is correct for the legacy wide-frame path, but
    # streamed factor blocks are intentionally absent from that live frame.
    campaign._base_add_all_features = base_add_all
    base_correct = campaign._correct_family
    base_factor_contract = campaign._factor_contract
    state: dict[str, Any] = {
        "expected_physical_specs": PHYSICAL_EXECUTABLE_COUNT,
        "expected_legacy_specs": LEGACY_FORMAL_COUNT,
        "fail_on_unavailable": True,
        "fail_on_zero_coverage": True,
        "legacy_gap_ids": [],
    }

    def configure(config: Mapping[str, Any]) -> None:
        base_configure(config)

        physical_base = _expanded(campaign, PHYSICAL_WINDOWS)
        legacy_base = _expanded(campaign, LEGACY_WINDOWS)
        supplement = _supplement_rows(campaign)

        physical_base["physical_support"] = True
        physical_base["support_reason"] = "published physical warehouse window"
        physical_base["formula_override"] = ""
        physical = pd.concat([physical_base, supplement], ignore_index=True)

        legacy_base["physical_support"] = ~(
            legacy_base["family"].eq("C") & legacy_base["window"].eq("60m")
        )
        legacy_base["support_reason"] = np.where(
            legacy_base["physical_support"],
            "published physical warehouse window",
            "legacy-only C@60m: no published 60m price-NVG topology",
        )
        legacy_base["formula_override"] = ""
        legacy = pd.concat([legacy_base, supplement], ignore_index=True)

        if len(physical) != PHYSICAL_EXECUTABLE_COUNT:
            raise RuntimeError(
                f"physical factor contract drift: expected {PHYSICAL_EXECUTABLE_COUNT}, found {len(physical)}"
            )
        if len(legacy) != LEGACY_FORMAL_COUNT:
            raise RuntimeError(
                f"legacy factor contract drift: expected {LEGACY_FORMAL_COUNT}, found {len(legacy)}"
            )

        physical_ids = set(physical["factor_id"])
        gap = legacy.loc[~legacy["factor_id"].isin(physical_ids)].copy()
        if len(gap) != LEGACY_FORMAL_COUNT - PHYSICAL_EXECUTABLE_COUNT:
            raise RuntimeError(
                f"legacy gap drift: expected {LEGACY_FORMAL_COUNT - PHYSICAL_EXECUTABLE_COUNT}, found {len(gap)}"
            )
        if not (
            gap["family"].eq("C").all() and gap["window"].eq("60m").all()
        ):
            raise RuntimeError("legacy gap must contain only C@60m specifications")

        a_mask = physical["prototype_id"].isin(["A14", "A15", "A16"])
        physical.loc[a_mask, "formula_override"] = "multi-scale momentum uses physical 10m/15m/30m"
        k_mask = physical["family"].eq("K")
        physical.loc[k_mask, "formula_override"] = (
            "fixed minute-price/topology anchor uses physical 10m; momentum/risk anchors retain documented horizons"
        )

        campaign.FF.FAMILY_WINDOWS = dict(PHYSICAL_WINDOWS)
        campaign.RUNTIME_WINDOWS = dict(PHYSICAL_WINDOWS)
        campaign.V26.SPEC_REGISTRY = physical
        campaign.V26.LEGACY_SPEC_REGISTRY = legacy
        campaign.V26.LEGACY_CONTRACT_GAP = gap
        campaign.V26.SUPPLEMENT_FACTOR_NAMES = list(supplement["factor_id"])
        names = list(physical["factor_id"])
        _sync_feature_consumers(campaign, names)

        policy = config.get("factor_resolution_policy", {})
        state.update(
            {
                "expected_physical_specs": int(
                    policy.get("expected_physical_specs", PHYSICAL_EXECUTABLE_COUNT)
                ),
                "expected_legacy_specs": int(
                    policy.get("expected_legacy_specs", LEGACY_FORMAL_COUNT)
                ),
                "fail_on_unavailable": bool(policy.get("fail_on_unavailable", True)),
                "fail_on_zero_coverage": bool(policy.get("fail_on_zero_coverage", True)),
                "legacy_gap_ids": list(gap["factor_id"]),
            }
        )
        campaign.PHYSICAL_CONTRACT_POLICY = dict(state)

    def correct(
        frame: pd.DataFrame,
        values: dict[str, pd.Series | None],
        family: str,
        window: str,
    ) -> dict[str, pd.Series | None]:
        source = _k_anchor_frame(frame) if family == "K" else frame
        result = base_correct(source, values, family, window)
        if family != "A":
            return result
        groups = frame.index.get_level_values("datetime")
        m10 = campaign._series(frame, "traditional__momentum_10m", "minute_nvg__momentum_10m")
        m15 = campaign._series(frame, "traditional__momentum_15m", "minute_nvg__momentum_15m")
        m30 = campaign._series(frame, "traditional__momentum_30m", "minute_nvg__momentum_30m")
        if all(value is not None for value in (m10, m15, m30)):
            result["A14"] = pd.concat([m10, m15, m30], axis=1).apply(np.sign).mean(axis=1)
            result["A15"] = campaign._csz(m10, groups) - campaign._csz(m30, groups)
            result["A16"] = (
                campaign._csz(m10, groups)
                - 2.0 * campaign._csz(m15, groups)
                + campaign._csz(m30, groups)
            )
        return result

    def factor_contract(group: pd.DataFrame, frame: pd.DataFrame) -> str:
        return campaign._hash(
            {
                "base_contract": base_factor_contract(group, frame),
                "physical_contract_version": CONTRACT_VERSION,
                "physical_windows": PHYSICAL_WINDOWS,
                "legacy_gap_ids": state.get("legacy_gap_ids", []),
                "k_minute_anchor": "10m",
                "a_multiscale_windows": ["10m", "15m", "30m"],
            }
        )

    def strict_add_all(frame: pd.DataFrame) -> pd.DataFrame:
        result = base_add_all(frame)
        registry = campaign.V26.SPEC_REGISTRY
        runtime = campaign.V26.RUNTIME_FACTOR_STATUS
        expected = int(state["expected_physical_specs"])
        unavailable: list[str] = []
        zero_coverage: list[str] = []
        absent_columns: list[str] = []
        for row in registry.itertuples(index=False):
            item = runtime.get(row.factor_id, {})
            status = str(item.get("status", "NOT_MATERIALIZED"))
            rate = float(item.get("non_null_rate", 0.0) or 0.0)
            if "UNAVAILABLE" in status or status == "NOT_MATERIALIZED":
                unavailable.append(row.factor_id)
            if rate <= 0.0:
                zero_coverage.append(row.factor_id)
            if row.factor_id not in result.columns:
                absent_columns.append(row.factor_id)

        audit = {
            "contract_version": CONTRACT_VERSION,
            "expected_physical_specs": expected,
            "actual_physical_specs": len(registry),
            "expected_legacy_specs": int(state["expected_legacy_specs"]),
            "legacy_gap_count": len(state.get("legacy_gap_ids", [])),
            "legacy_gap_ids": state.get("legacy_gap_ids", []),
            "materialized_columns": int(
                sum(name in result.columns for name in registry["factor_id"])
            ),
            "unavailable_count": len(unavailable),
            "zero_coverage_count": len(zero_coverage),
            "absent_column_count": len(absent_columns),
            "unavailable": unavailable,
            "zero_coverage": zero_coverage,
            "absent_columns": absent_columns,
            "policy": dict(state),
        }
        if campaign.CTX is not None:
            root = campaign.CTX.root / "schema"
            campaign._atomic_json(root / "physical_factor_completion_gate.json", audit)
            gap = campaign.V26.LEGACY_CONTRACT_GAP.copy()
            gap.to_parquet(root / "legacy_contract_gap.parquet", index=False)
            gap.to_csv(root / "legacy_contract_gap.csv", index=False)
            campaign._atomic_json(
                root / "legacy_contract_gap.json",
                {
                    "contract_version": CONTRACT_VERSION,
                    "legacy_formal_specs": int(state["expected_legacy_specs"]),
                    "physical_executable_specs": expected,
                    "gap_count": len(gap),
                    "gap_reason": "no published 60m price-NVG topology; 60m HVG remains in family G",
                    "factor_ids": list(gap["factor_id"]),
                },
            )

        should_fail = len(registry) != expected or bool(absent_columns)
        should_fail |= bool(unavailable) and bool(state["fail_on_unavailable"])
        should_fail |= bool(zero_coverage) and bool(state["fail_on_zero_coverage"])
        if should_fail:
            raise RuntimeError(
                "warehouse-exact factor completion gate failed; inspect physical_factor_completion_gate.json"
            )
        return result

    campaign.configure_registry = configure
    campaign._correct_family = correct
    campaign._factor_contract = factor_contract
    campaign._add_all_features = strict_add_all
