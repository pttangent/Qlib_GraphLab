"""Final v2.7 entrypoint hardening.

The main v2.7 module intentionally drops most raw dependencies after factor
materialization. This layer preserves the small set required by the PIT
universe/neutralization contracts, fixes transition formulas to use
per-symbol differences, and resolves the structural universe from whichever
real base/supplement fields actually survived the date's schema audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nff_research import v2_7_atomic_campaign as C


_BASE_CORRECT_FAMILY = C._correct_family
_BASE_ADD_ALL_FEATURES = C._add_all_features
_BASE_UNIVERSE_MASKS = C.R.universe_masks


PRESERVED_SOURCE_COLUMNS = (
    "minute_nvg__price_nvg_30m_top_bottom_asymmetry",
    "minute_nvg__price_nvg_30m_terminal_signed_edge_balance",
    "trade_nvg__trade_flow_path_300s_terminal_position",
    "trade_nvg__trade_flow_nvg_300s_terminal_signed_edge_balance",
    "trade_nvg__trade_active_second_ratio_300s",
    "trade_nvg__trade_price_stale_ratio_300s",
    "trade_nvg__trade_observation_coverage_300s",
)


def _correct_family_grouped(
    frame: pd.DataFrame,
    values: dict[str, pd.Series | None],
    family: str,
    window: str,
) -> dict[str, pd.Series | None]:
    result = _BASE_CORRECT_FAMILY(frame, values, family, window)
    if family != "F":
        return result
    activity = C._series(frame, f"trade_nvg__trade_active_second_ratio_{window}")
    stale = C._series(frame, f"trade_nvg__trade_price_stale_ratio_{window}")
    if activity is None or stale is None:
        return result
    activity_change = activity - C._shift(activity, 1, frame)
    stale_change = stale - C._shift(stale, 1, frame)
    result["F03"] = (-stale_change).clip(lower=0) * activity_change.clip(lower=0)
    return result


def _add_all_features_with_universe_controls(frame: pd.DataFrame) -> pd.DataFrame:
    retained = frame[[column for column in PRESERVED_SOURCE_COLUMNS if column in frame]].copy(deep=False)
    result = _BASE_ADD_ALL_FEATURES(frame)
    missing = [column for column in retained if column not in result]
    return pd.concat([result, retained[missing]], axis=1, copy=False) if missing else result


def _first_valid_column(features: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in features and features[column].notna().any():
            return column
    return None


def _dynamic_universe_masks(
    features: pd.DataFrame,
    controls: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Build PIT universes without requiring one historical schema spelling.

    Direction is sourced first from the rebuilt NVG supplement factor/field,
    then from the older top-bottom geometry only as a structural coverage
    fallback. Trade flow and Hawkes follow the same real-field policy.
    """
    index = features.index
    all_mask = pd.Series(True, index=index, dtype="boolean")
    price_column = _first_valid_column(
        features,
        (
            "full_factor__s01__w30m",
            "full_factor__b01__w30m",
            "minute_nvg__price_nvg_30m_terminal_signed_edge_balance",
            "minute_nvg__price_nvg_30m_top_bottom_asymmetry",
        ),
    )
    trade_column = _first_valid_column(
        features,
        (
            "full_factor__e02__w300s",
            "trade_nvg__trade_flow_nvg_300s_terminal_signed_edge_balance",
            "trade_nvg__trade_flow_path_300s_terminal_position",
        ),
    )
    hawkes_column = _first_valid_column(
        features,
        (
            "full_factor__h02__w60s",
            "full_factor__h01__w60s",
            "hawkes_derived__hawkes_signed_pressure",
            "hawkes_lite__hawkes_intensity_imbalance",
        ),
    )
    required = [column for column in (price_column, trade_column, hawkes_column) if column]
    common = features[required].notna().all(axis=1) if required else all_mask.copy()

    close = pd.to_numeric(features.get("bars_1m__close"), errors="coerce")
    volume = pd.to_numeric(features.get("bars_1m__volume"), errors="coerce")
    trade_count = pd.to_numeric(
        features.get("trades_1m_core__trade_count", pd.Series(np.nan, index=index)),
        errors="coerce",
    )
    liquid = common.fillna(False)
    liquid &= close.ge(5).fillna(False)
    liquid &= volume.gt(0).fillna(False)
    liquid &= controls.get("control__adv20_days", pd.Series(0, index=index)).fillna(0).ge(20)
    liquid &= trade_count.ge(1).fillna(False)

    active_column = _first_valid_column(
        features,
        (
            "trade_nvg__trade_active_second_ratio_300s",
            "trade_nvg__active_second_ratio_60s",
        ),
    )
    stale_column = _first_valid_column(
        features,
        ("trade_nvg__trade_price_stale_ratio_300s",),
    )
    if active_column:
        liquid &= pd.to_numeric(features[active_column], errors="coerce").ge(0.01).fillna(False)
    if stale_column:
        liquid &= pd.to_numeric(features[stale_column], errors="coerce").le(0.95).fillna(False)

    result: dict[str, pd.Series] = {
        "all_pit_eligible": all_mask,
        "common_structural": common.fillna(False).astype("boolean"),
    }
    for limit in (500, 1000, 2000, 3000):
        key = f"control__adv20_top{limit}"
        layer = controls.get(key)
        if layer is None:
            layer = controls.get("control__adv20_top1000", pd.Series(False, index=index))
        result[f"liquid_common_adv20_top{limit}"] = (
            liquid & layer.fillna(False).astype(bool)
        ).astype("boolean")
    result["final_trading_universe"] = result["liquid_common_adv20_top1000"]
    result["own_feature_universe"] = result["all_pit_eligible"]
    return result


C._correct_family = _correct_family_grouped
C._add_all_features = _add_all_features_with_universe_controls
C.R.universe_masks = _dynamic_universe_masks


def main() -> int:
    return C.main()


if __name__ == "__main__":
    raise SystemExit(main())
