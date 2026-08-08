from __future__ import annotations

"""Apply the PIT-clock semantic salt after v2.8 stage contracts are installed."""

from typing import Any, Mapping

from nff_research import v2_8_pit_clock as PIT


def install(P: Any) -> None:
    original = P._contract_hash

    def contract_hash(config: Mapping[str, Any]) -> str:
        return P._json_hash(
            {
                "stage_contract": original(config),
                "pit_clock_version": PIT.PIT_CLOCK_VERSION,
                "manual_join_key": "symbol+event_time",
                "availability_gate": "source_available_time<=decision_time",
                "label_anchor": "decision_time+1m exact canonical market bars",
                "selection_role_policy": PIT.ROLE_POLICY,
            }
        )

    P._contract_hash = contract_hash
