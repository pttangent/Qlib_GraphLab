from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nff_research import v2_7_launch as LAUNCH
from nff_research import v2_7_symbol_id_supplement as JOIN


def test_research_loader_preserves_exact_physical_keys() -> None:
    loader = LAUNCH.C.R.NFFDataLoader.__new__(LAUNCH.C.R.NFFDataLoader)
    loader.join = "inner"
    timestamp = pd.Timestamp("2026-01-02 15:00:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "symbol_id": pd.Series([9_007_199_254_740_001], dtype="int64"),
            "timestamp": [timestamp],
            "__symbol__bars_1m": ["AAA"],
            "__available__bars_1m": [timestamp + pd.Timedelta(seconds=30)],
            "bars_1m__close": [100.0],
        }
    )
    merged = loader._merge_sources([frame])
    assert int(merged.loc[0, JOIN.SYMBOL_ID_COLUMN]) == 9_007_199_254_740_001
    assert int(merged.loc[0, JOIN.EVENT_TIME_NS_COLUMN]) == timestamp.value


def _write_supplement(root: Path) -> None:
    partition = (
        root
        / "nvg_supplement"
        / "minute_nvg_edge_raw"
        / "schema=v1"
        / "date=2026-01-02"
    )
    partition.mkdir(parents=True)
    event = pd.Timestamp("2026-01-02 15:00:00", tz="UTC")
    pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-02"],
            "symbol_id": [101, 202],
            # Deliberately wrong/equal symbols prove the join is id-based.
            "symbol": ["COLLISION", "COLLISION"],
            "timestamp": [event, event],
            "available_time": [
                event + pd.Timedelta(seconds=30),
                event + pd.Timedelta(minutes=1, seconds=30),
            ],
            "price_nvg_15m_terminal_signed_edge_balance": [0.25, 0.75],
        }
    ).to_parquet(partition / "data.parquet", index=False)


def test_supplement_join_uses_symbol_id_event_time_and_pit_clock(tmp_path: Path) -> None:
    _write_supplement(tmp_path)
    event = pd.Timestamp("2026-01-02 15:00:00", tz="UTC")
    decision = event + pd.Timedelta(minutes=2)
    index = pd.MultiIndex.from_arrays(
        [[decision, decision], ["AAA", "BBB"]], names=["datetime", "instrument"]
    )
    frame = pd.DataFrame(
        {
            JOIN.SYMBOL_ID_COLUMN: pd.Series([101, 202], index=index, dtype="int64"),
            JOIN.EVENT_TIME_NS_COLUMN: pd.Series(
                [event.value, event.value], index=index, dtype="int64"
            ),
            "bars_1m__close": [100.0, 200.0],
        },
        index=index,
    )
    result = LAUNCH.C.FF.merge_supplements(frame, tmp_path)
    column = "price_nvg_15m_terminal_signed_edge_balance"
    assert np.isclose(float(result.loc[(decision, "AAA"), column]), 0.25)
    # available 15:01:30 -> ceil 15:02 + one bar = 15:03, later than 15:02 decision.
    assert pd.isna(result.loc[(decision, "BBB"), column])
    audit = pd.DataFrame(LAUNCH.C.SUPPLEMENT_JOIN_AUDIT)
    row = audit.loc[audit["dataset"].eq("minute_nvg_edge_raw")].iloc[0]
    assert row["join_mode"] == "symbol_id_event_time"
    assert int(row["matched_rows"]) == 2
    assert int(row["admissible_rows"]) == 1
    assert int(row["late_rows"]) == 1
