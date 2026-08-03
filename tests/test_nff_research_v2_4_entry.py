from __future__ import annotations

from nff_research import v2_4_optimized_entry as entry


def test_canonical_projection_reads_only_required_bar_and_trade_columns(monkeypatch):
    available = {
        ("canonical", "bars_1m", "v1"): ["open", "high", "low", "close", "volume", "dollar_volume", "vwap"],
        ("canonical", "trades_1m_core", "v1"): [
            "trade_count",
            "dollar_volume",
            "signed_dollar_flow_proxy",
            "large_trade_volume",
            "lit_volume",
            "off_exchange_volume",
        ],
    }
    monkeypatch.setattr(entry.R, "source_columns", lambda kind, dataset, schema: available[(kind, dataset, schema)])

    projected = entry.narrow_canonical_sets()

    assert projected["bars_1m"]["columns"] == ["open", "close", "volume", "dollar_volume", "vwap"]
    assert projected["trades_1m_core"]["columns"] == ["trade_count"]
