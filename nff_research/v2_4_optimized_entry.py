from __future__ import annotations

"""Executable v2.4 entrypoint with exact feature and canonical projection."""

import sys
from pathlib import Path
from typing import Any

# File-path execution is used by detached workers; make the repository package
# importable without relying on an editable install or inherited PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nff_research import v2_4_optimized_runner as fast

R = fast.R


def narrow_canonical_sets() -> dict[str, dict[str, Any]]:
    available_bars = set(R.source_columns("canonical", "bars_1m", "v1"))
    available_core = set(R.source_columns("canonical", "trades_1m_core", "v1"))
    # Bars support all seven labels, traditional factors, price filters, and
    # current-dollar-volume neutralization.  Trade count is the only canonical
    # trade field consumed by v2.3/v2.4 after NFF families are materialized.
    bars = [column for column in ("open", "close", "volume", "dollar_volume", "vwap") if column in available_bars]
    core = [column for column in ("trade_count",) if column in available_core]
    return {
        "bars_1m": {"schema_version": "v1", "columns": bars},
        "trades_1m_core": {"schema_version": "v1", "columns": core},
    }


def install() -> None:
    fast.install_fastpath()
    R.canonical_sets = narrow_canonical_sets
    # The scheduler must launch this entrypoint so worker subprocesses install
    # the same canonical projection rather than falling back to the base file.
    R.__file__ = __file__


def main() -> int:
    install()
    return R.main()


if __name__ == "__main__":
    raise SystemExit(main())
