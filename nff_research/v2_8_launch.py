from __future__ import annotations

"""Canonical launcher for the v2.8 staged-selection campaign."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nff_research import v2_8_pipeline as PIPELINE
from nff_research import v2_8_runtime_hardening as HARDENING

HARDENING.install(PIPELINE)
main = PIPELINE.main


if __name__ == "__main__":
    raise SystemExit(main())
