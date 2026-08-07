from __future__ import annotations

"""Canonical launcher for the v2.8 staged-selection campaign."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nff_research import v2_8_pipeline as PIPELINE
from nff_research import v2_8_detailed_optimization as DETAILED_OPT
from nff_research import v2_8_runtime_hardening as HARDENING
from nff_research import v2_8_screen_optimization as SCREEN_OPT
from nff_research import v2_8_selection_tracks as SELECTION_TRACKS
from nff_research import v2_8_strict_training_window as STRICT_TRAINING
from nff_research import v2_8_portfolio_optimization as PORTFOLIO_OPT
from nff_research import v2_8_materialize_memory as MATERIALIZE_MEMORY
from nff_research import v2_8_source_contract as SOURCE_CONTRACT
from nff_research import v2_8_stage_contracts as STAGE_CONTRACTS
from nff_research import v2_8_bootstrap_scheduler as BOOTSTRAP_SCHEDULER

# Detailed optimization must be installed before HARDENING so the hardening
# layer wraps the optimized date function with the same atomic context and
# worker cleanup guarantees as the reference path.
DETAILED_OPT.install(PIPELINE)
HARDENING.install(PIPELINE)
SCREEN_OPT.install(PIPELINE)
SELECTION_TRACKS.install(PIPELINE)
STRICT_TRAINING.install(PIPELINE)
PORTFOLIO_OPT.install(PIPELINE)
# Install before source/stage wrappers so source fingerprints and semantic
# stage contexts remain outermost around the optimized materialize function.
MATERIALIZE_MEMORY.install(PIPELINE)
SOURCE_CONTRACT.install(PIPELINE)
STAGE_CONTRACTS.install(PIPELINE)
BOOTSTRAP_SCHEDULER.install(PIPELINE)
main = PIPELINE.main


if __name__ == "__main__":
    raise SystemExit(main())
