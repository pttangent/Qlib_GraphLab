from __future__ import annotations

"""Canonical launch script for the complete v2.7 campaign."""

import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nff_research import v2_7_checkpoint_hardening as CHECKPOINTS
from nff_research import v2_7_deciles as DECILES
from nff_research import v2_7_formula_hardening as FORMULAS
from nff_research import v2_7_physical_contract as PHYSICAL
from nff_research import v2_7_run as RUN
from nff_research import v2_7_runtime_hardening as HARDEN
from nff_research import v2_7_supplement_scope as SUPPLEMENT_SCOPE
from nff_research import v2_7_symbol_id_supplement as SYMBOL_JOIN


C = RUN.C
HARDEN.install(C)
# Formula hardening calls the mathematical helpers from the underlying factor
# engine. Bind them explicitly before installing parent/worker patches.
C._csr = C.FF._csr
C._conf = C.FF._conf
C._same = C.FF._same
FORMULAS.install(C, HARDEN)
CHECKPOINTS.install(C)
# Restore the physical 10m/15m/30m minute-NVG contract, record the legacy-only
# C@60m gap, and include the correction in checkpoint hashes.
PHYSICAL.install(C)
# The base directional helper iterates all B windows. Keep only the two S
# factors explicitly admitted by the v3 contract (15m and 30m).
SUPPLEMENT_SCOPE.install(C)
# Preserve exact symbol_id/event_time metadata through the adapter and join the
# out-of-adapter supplement namespace on the physical key. Supplement rows are
# admitted only when their own availability maps no later than decision_time.
SYMBOL_JOIN.install(C)

_BASE_VALIDATION_SCORE = RUN.MODELS._validation_score


def _finite_validation_score(source, predictions) -> float:
    score = float(_BASE_VALIDATION_SCORE(source, predictions))
    return score if math.isfinite(score) else -math.inf


def _exact_decile_feature_rows(
    feature,
    signal,
    label,
    adv,
    price,
    trades,
    metadata,
    min_n,
):
    return DECILES.decile_feature_rows(
        C.R,
        feature,
        signal,
        label,
        adv,
        price,
        trades,
        metadata,
        min_n,
    )


RUN.MODELS._validation_score = _finite_validation_score
C._decile_feature_rows = _exact_decile_feature_rows


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
