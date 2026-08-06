from __future__ import annotations

"""Keep generated S-direction factors identical to the formal v3 registry."""

from typing import Any

import pandas as pd


SCOPE_VERSION = "v2.7-supplement-s15-s30-only"


def install(campaign: Any) -> None:
    base_direction = campaign._supplement_direction
    base_factor_contract = campaign._factor_contract

    def scoped_direction(
        frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
        result, runtime = base_direction(frame)
        allowed = set(campaign.V26.SUPPLEMENT_FACTOR_NAMES)
        generated = {
            column for column in result.columns if column.startswith("full_factor__s01__")
        }
        extra = sorted(generated - allowed)
        if extra:
            result = result.drop(columns=extra)
        runtime = {
            name: value
            for name, value in runtime.items()
            if not name.startswith("full_factor__s01__") or name in allowed
        }
        return result, runtime

    def factor_contract(group: pd.DataFrame, frame: pd.DataFrame) -> str:
        return campaign._hash(
            {
                "base_contract": base_factor_contract(group, frame),
                "supplement_scope_version": SCOPE_VERSION,
                "allowed_s_factors": list(campaign.V26.SUPPLEMENT_FACTOR_NAMES),
            }
        )

    campaign._supplement_direction = scoped_direction
    campaign._factor_contract = factor_contract
