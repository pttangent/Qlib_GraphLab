from __future__ import annotations

import numpy as np
import pandas as pd

from nff_research import v2_7_launch as LAUNCH


def test_only_registered_s15_s30_are_materialized(monkeypatch) -> None:
    instruments = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    timestamp = pd.Timestamp("2026-01-02 15:00:00", tz="UTC")
    index = pd.MultiIndex.from_arrays(
        [instruments, [timestamp] * len(instruments)],
        names=["instrument", "datetime"],
    )
    columns: dict[str, np.ndarray] = {}
    for window in (10, 15, 30):
        fields = LAUNCH.C.directional_supplement_fields(window)
        columns[fields["price_edge_balance"]] = np.linspace(-1.0, 1.0, len(index))
        columns[fields["price_long_edge_slope"]] = np.linspace(-0.5, 0.5, len(index))
        columns[fields["detrended_edge_balance"]] = np.linspace(-0.8, 0.8, len(index))
        columns[fields["detrended_long_edge_slope"]] = np.linspace(-0.4, 0.4, len(index))
    frame = pd.DataFrame(columns, index=index)

    monkeypatch.setitem(LAUNCH.C.RUNTIME_WINDOWS, "B", ("10m", "15m", "30m"))
    monkeypatch.setattr(
        LAUNCH.C.V26,
        "SUPPLEMENT_FACTOR_NAMES",
        ["full_factor__s01__w15m", "full_factor__s01__w30m"],
    )
    result, runtime = LAUNCH.C._supplement_direction(frame)

    assert "full_factor__s01__w10m" not in result.columns
    assert "full_factor__s01__w10m" not in runtime
    assert "full_factor__s01__w15m" in result.columns
    assert "full_factor__s01__w30m" in result.columns
    assert set(runtime) == {"full_factor__s01__w15m", "full_factor__s01__w30m"}
