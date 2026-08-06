"""Final v2.7 entrypoint hardening.

This layer preserves the small set required by PIT universes, derives missing
but mathematically explicit trade controls before the A-K engine runs, fixes
per-symbol transitions, resolves universes from the real schema, and makes the
vectorized decile kernel exactly equivalent to qcut on unique within-minute
ranks.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nff_research import v2_7_atomic_campaign as C


_BASE_CORRECT_FAMILY = C._correct_family
_BASE_ADD_ALL_FEATURES = C._add_all_features
_BASE_DECILE_FEATURE_ROWS = C._decile_feature_rows
_BASE_DISCOVER_WAREHOUSE_SCHEMA = C.discover_warehouse_schema


PRESERVED_SOURCE_COLUMNS = (
    "minute_nvg__price_nvg_30m_top_bottom_asymmetry",
    "minute_nvg__price_nvg_30m_terminal_signed_edge_balance",
    "trade_nvg__trade_flow_path_300s_terminal_position",
    "trade_nvg__trade_flow_nvg_300s_terminal_signed_edge_balance",
    "trade_nvg__trade_active_second_ratio_300s",
    "trade_nvg__trade_price_stale_ratio_300s",
    "trade_nvg__trade_observation_coverage_300s",
)


def _discover_union_schema(warehouse_root, trade_date=None):
    """Build registry windows from the union of published date schemas.

    Per-date audits remain date-specific. Registry construction must not lose a
    valid supplement window merely because the first campaign date lacks that
    family partition.
    """
    return _BASE_DISCOVER_WAREHOUSE_SCHEMA(warehouse_root, None)


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


def _prederive_trade_fields(frame: pd.DataFrame) -> pd.DataFrame:
    generated: dict[str, pd.Series] = {}
    dollar = C._series(frame, "trades_1m_core__dollar_volume", "bars_1m__dollar_volume")
    large_dollar = C._series(frame, "trades_1m_core__large_trade_dollar_volume")
    if dollar is not None and large_dollar is not None:
        generated["trades_1m_core__large_trade_dollar_share"] = (
            large_dollar / (dollar + C.EPS)
        ).astype("float32")

    signed_flow = C._series(frame, "trades_1m_core__signed_dollar_flow_proxy")
    if signed_flow is not None:
        groups = frame.index.get_level_values("instrument")
        signed_sum = (
            signed_flow.groupby(groups, sort=False)
            .rolling(15, min_periods=5)
            .sum()
            .reset_index(level=0, drop=True)
            .reindex(frame.index)
        )
        absolute_sum = (
            signed_flow.abs()
            .groupby(groups, sort=False)
            .rolling(15, min_periods=5)
            .sum()
            .reset_index(level=0, drop=True)
            .reindex(frame.index)
        )
        generated["trades_1m_core__flow_persistence_15m"] = (
            signed_sum / (absolute_sum + C.EPS)
        ).astype("float32")

    if not generated:
        return frame
    block = pd.concat(generated, axis=1, copy=False)
    block.columns = list(generated)
    missing = [column for column in block if column not in frame]
    return pd.concat([frame, block[missing]], axis=1, copy=False) if missing else frame


def _add_all_features_with_universe_controls(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _prederive_trade_fields(frame)
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


def _decile_feature_rows_qcut_exact(
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
    row_counts = minute_counts[minute_codes]
    # pd.qcut on unique ranks 1..N is equivalent to these right-closed
    # equal-frequency bins. Unlike ceil(pct_rank*10), this preserves the first
    # bucket when N is not divisible by ten.
    deciles = (
        np.floor((unique_ranks.to_numpy(dtype="float64") - 1.0) * 10.0 / row_counts)
        .clip(0, 9)
        .astype("int16")
        + 1
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
                "bundle": C.R.infer_bundle(feature),
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


C.discover_warehouse_schema = _discover_union_schema
C._correct_family = _correct_family_grouped
C._add_all_features = _add_all_features_with_universe_controls
C._decile_feature_rows = _decile_feature_rows_qcut_exact
C.R.universe_masks = _dynamic_universe_masks


def main() -> int:
    return C.main()


if __name__ == "__main__":
    raise SystemExit(main())
