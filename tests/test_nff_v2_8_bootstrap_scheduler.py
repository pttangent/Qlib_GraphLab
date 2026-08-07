from __future__ import annotations

from nff_research import v2_8_bootstrap_scheduler as BOOTSTRAP
from nff_research import v2_8_launch as LAUNCH
from nff_research import v2_8_pipeline as P


def test_scoped_config_does_not_mutate_base() -> None:
    base = {"pipeline": {"memory_reserve_gb": 28}}
    scoped = BOOTSTRAP._scoped(base, ["2026-01-02", "2026-01-05"], "training")
    assert "_runtime_dates" not in base["pipeline"]
    assert scoped["pipeline"]["_runtime_dates"] == ["2026-01-02", "2026-01-05"]
    assert scoped["pipeline"]["_runtime_status_suffix"] == "training"


def test_canonical_main_is_training_window_first_scheduler() -> None:
    assert LAUNCH.main is P.main
    assert P.main.__module__.endswith("v2_8_bootstrap_scheduler")
    source = __import__("pathlib").Path(BOOTSTRAP.__file__).read_text(encoding="utf-8")
    first_materialize = source.index('run_stage("materialize", training)')
    first_screen = source.index('run_stage("basic_screen", training)')
    selection = source.index('run_stage("select")')
    remaining = source.index('run_stage("materialize", remaining)')
    assert first_materialize < first_screen < selection < remaining
    assert 'choices=[' in source and '"bootstrap"' in source
