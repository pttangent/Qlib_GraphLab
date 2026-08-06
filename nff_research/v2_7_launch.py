from __future__ import annotations

"""Canonical launch script for the complete v2.7 campaign."""

import math
from pathlib import Path
from typing import Any

from nff_research import v2_7_run as RUN
from nff_research import v2_7_runtime_hardening as HARDEN


C = RUN.C
HARDEN.install(C)

_BASE_VALIDATION_SCORE = RUN.MODELS._validation_score


def _finite_validation_score(source, predictions) -> float:
    score = float(_BASE_VALIDATION_SCORE(source, predictions))
    return score if math.isfinite(score) else -math.inf


RUN.MODELS._validation_score = _finite_validation_score


def _install_worker_command() -> None:
    original = C.R.worker_command
    current = str(Path(__file__).resolve())
    known_names = {
        "v2_1_neutralized_runner.py",
        "v2_6_full_defined_campaign.py",
        "v2_7_atomic_campaign.py",
        "v2_7_atomic_entry.py",
        "v2_7_run.py",
        "v2_7_launch.py",
    }

    def worker_command(*args: Any, **kwargs: Any) -> list[str]:
        command = original(*args, **kwargs)
        replaced = False
        result: list[str] = []
        for value in command:
            if Path(value).name in known_names:
                result.append(current)
                replaced = True
            else:
                result.append(value)
        if not replaced:
            raise RuntimeError(f"worker command has no replaceable research entrypoint: {command}")
        return result

    C.R.worker_command = worker_command


C._install_worker_command = _install_worker_command


def main() -> int:
    return C.main()


if __name__ == "__main__":
    raise SystemExit(main())
