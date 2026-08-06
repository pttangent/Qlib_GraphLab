"""Detached entrypoint for the v2.5 full campaign."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nff_research.v2_5_full_campaign import main


if __name__ == "__main__":
    raise SystemExit(main())
