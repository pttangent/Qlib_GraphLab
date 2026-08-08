from __future__ import annotations

import pandas as pd

from nff_research import v2_8_launch as LAUNCH  # noqa: F401 - installs policy
from nff_research import v2_8_pipeline as P
from nff_research import v2_8_pit_clock as PIT
from nff_research import v2_8_selection_tracks as TRACKS


def test_source_visibility_is_bounded_by_decision_time() -> None:
    source = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "timestamp": [
                "2026-01-02T15:00:00Z",
                "2026-01-02T15:00:00Z",
                "2026-01-02T15:02:00Z",
            ],
            "available_time": [
                "2026-01-02T15:01:20Z",
                "2026-01-02T15:03:00Z",
                "2026-01-02T15:03:00Z",
            ],
            "value": [1.0, 2.0, 3.0],
        }
    )
    keys = pd.DataFrame(
        {
            "__symbol": ["AAA"],
            "__event_time": [pd.Timestamp("2026-01-02T15:00:00Z")],
            "__decision_time": [pd.Timestamp("2026-01-02T15:02:00Z")],
        }
    )
    visible = PIT._filter_visible_source(source, keys)
    assert visible["value"].tolist() == [1.0]
    assert bool((visible["__source_available"] <= visible["__decision_time"]).all())


def test_timing_work_rejects_available_after_decision() -> None:
    index = pd.MultiIndex.from_tuples(
        [("AAA", pd.Timestamp("2026-01-02 15:02:00"))],
        names=["instrument", "datetime"],
    )
    frame = pd.DataFrame(
        {
            PIT.TIMING_EVENT: [pd.Timestamp("2026-01-02T15:00:00Z").value],
            PIT.TIMING_AVAILABLE: [pd.Timestamp("2026-01-02T15:03:00Z").value],
        },
        index=index,
    )
    try:
        PIT._timing_work(frame)
    except RuntimeError as exc:
        assert "PIT timing metadata invalid" in str(exc)
    else:  # pragma: no cover - explicit failure message is more useful than pytest.raises here
        raise AssertionError("available_time after decision_time must be rejected")


def test_alpha_selection_excludes_regime_family(monkeypatch) -> None:
    registry = pd.DataFrame(
        [
            {
                "factor_id": "full_factor__a01__w10m",
                "role": "DIRECTION_ALPHA",
                "neutralization_allowed": True,
            },
            {
                "factor_id": "full_factor__j03__w1m",
                "role": "REGIME",
                "neutralization_allowed": False,
            },
        ]
    )
    monkeypatch.setattr(P.V26, "SPEC_REGISTRY", registry)
    rows = []
    for feature, family in (
        ("full_factor__a01__w10m", "A"),
        ("full_factor__j03__w1m", "J"),
    ):
        for value in (0.10, 0.12, 0.11):
            rows.append(
                {
                    "feature": feature,
                    "factor_family": family,
                    "label_family": "return_vwap_to_vwap",
                    "horizon_bars": 5,
                    "universe": "final_trading_universe",
                    "rank_ic_method": "minute_mean_cs_rank_ic",
                    "rank_ic_mean": value,
                    "coverage": 0.99,
                    "rank_ic_positive_ratio": 0.8,
                }
            )
    data = pd.DataFrame(rows)
    track = next(item for item in TRACKS._track_defaults() if item["name"] == "alpha")
    selected = TRACKS._track_summary(data, track, "final_trading_universe")
    assert selected["feature"].tolist() == ["full_factor__a01__w10m"]
    assert selected.iloc[0]["factor_role"] == "DIRECTION_ALPHA"


def test_pit_policy_is_part_of_stage_contract() -> None:
    config = {
        "run": {"start_date": "2026-01-02", "end_date": "2026-01-05"},
        "pipeline": {},
        "selection": {},
        "labels": {},
        "local_paths": {"warehouse_root": "D:/warehouse"},
    }
    first = P._contract_hash(config)
    assert isinstance(first, str) and len(first) == 64
    assert PIT.PIT_CLOCK_VERSION.startswith("v2.8.1")
    assert PIT.ROLE_POLICY["alpha"] == ("DIRECTION_ALPHA", "CONFIRMATION")
