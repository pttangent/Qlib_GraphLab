from __future__ import annotations

"""Canonical launcher for the v2.8 staged-selection campaign."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nff_research import v2_8_pipeline as PIPELINE
from nff_research import v2_8_runtime_hardening as HARDENING
from nff_research import v2_8_selection_tracks as SELECTION_TRACKS
from nff_research import v2_8_strict_training_window as STRICT_TRAINING
from nff_research import v2_8_portfolio_optimization as PORTFOLIO_OPT
from nff_research import v2_8_source_contract as SOURCE_CONTRACT
from nff_research import v2_8_stage_contracts as STAGE_CONTRACTS

HARDENING.install(PIPELINE)
SELECTION_TRACKS.install(PIPELINE)
STRICT_TRAINING.install(PIPELINE)
PORTFOLIO_OPT.install(PIPELINE)
SOURCE_CONTRACT.install(PIPELINE)
STAGE_CONTRACTS.install(PIPELINE)
main = PIPELINE.main


if __name__ == "__main__":
    raise SystemExit(main())
