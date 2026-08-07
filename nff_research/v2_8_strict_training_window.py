from __future__ import annotations

"""Require the exact chronological training window before candidate freeze."""

from typing import Any


def install(P: Any) -> None:
    original = P.select_candidates

    def strict_select(config: dict[str, Any]) -> dict[str, Any]:
        train_days = int(config.get("selection", {}).get("train_days", 60))
        planned = P._dates(config)[:train_days]
        missing = [
            date
            for date in planned
            if not P._stage_success(config, "basic_screen", date).exists()
        ]
        if missing:
            raise RuntimeError(
                "candidate selection requires the exact first "
                f"{train_days} chronological screen dates; "
                f"missing {len(missing)} dates: {missing[:20]}"
            )
        return original(config)

    P.select_candidates = strict_select
