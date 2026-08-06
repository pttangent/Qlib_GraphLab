"""Final v2.7 entrypoint hardening.

The main v2.7 module intentionally drops most raw dependencies after factor
materialization. This layer preserves the small set required by the PIT
universe/neutralization contracts and fixes transition formulas to use
per-symbol, not global-row, differences.
"""

from __future__ import annotations

import pandas as pd

from nff_research import v2_7_atomic_campaign as C


_BASE_CORRECT_FAMILY = C._correct_family
_BASE_ADD_ALL_FEATURES = C._add_all_features


PRESERVED_SOURCE_COLUMNS = (
    "minute_nvg__price_nvg_30m_top_bottom_asymmetry",
    "trade_nvg__trade_flow_path_300s_terminal_position",
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


C._correct_family = _correct_family_grouped
C._add_all_features = _add_all_features_with_universe_controls


def main() -> int:
    return C.main()


if __name__ == "__main__":
    raise SystemExit(main())
