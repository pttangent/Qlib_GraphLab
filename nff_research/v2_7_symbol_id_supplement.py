from __future__ import annotations

"""PIT-safe symbol-id join for the out-of-adapter supplement namespace.

The base NFF adapter aligns canonical/features on ``(symbol_id, timestamp)``
and then converts rows to a Qlib decision clock.  Supplement tables are loaded
by the research layer, so this module carries the original symbol id/event time
through the adapter as temporary metadata, joins on the physical key, and
admits a supplement value only when its own availability maps to a decision
clock no later than the current research row.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pads


JOIN_VERSION = "v2.7-symbol-id-event-time-pit-v1"
SYMBOL_ID_COLUMN = "nff_meta__symbol_id"
EVENT_TIME_NS_COLUMN = "nff_meta__event_time_ns"


def _atomic_audit(campaign: Any, rows: list[dict[str, Any]]) -> None:
    campaign.SUPPLEMENT_JOIN_AUDIT = list(rows)
    if campaign.CTX is None:
        return
    root = campaign.CTX.root / "schema"
    frame = pd.DataFrame(rows)
    campaign._atomic_parquet(frame, root / "supplement_join_audit.parquet", index=False)
    frame.to_csv(root / "supplement_join_audit.csv", index=False)
    campaign._atomic_json(
        root / "supplement_join_audit.json",
        {
            "join_version": JOIN_VERSION,
            "trade_date": campaign.CTX.trade_date,
            "rows": rows,
            "fallback_used": any(row.get("join_mode") != "symbol_id_event_time" for row in rows),
        },
    )


def _normalise_symbol_id(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.astype("Int64")


def install(campaign: Any) -> None:
    base_loader = campaign.R.NFFDataLoader
    fallback_merge = campaign.FF.merge_supplements
    base_factor_contract = campaign._factor_contract

    class ResearchMetadataLoader(base_loader):
        """Opt-in research loader that retains exact physical join metadata."""

        def __init__(self, *args: Any, **kwargs: Any):
            # Preserve int64 symbol ids/event nanoseconds. The daily factor
            # builder casts actual numeric features to float32 after supplement
            # joining and drops these temporary metadata columns.
            kwargs["output_float32"] = False
            super().__init__(*args, **kwargs)

        def _merge_sources(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
            merged = super()._merge_sources(frames)
            if merged.empty:
                return merged
            if "symbol_id" not in merged or "event_time" not in merged:
                raise RuntimeError("NFF adapter did not retain symbol_id/event_time before Qlib clock mapping")
            merged[SYMBOL_ID_COLUMN] = _normalise_symbol_id(merged["symbol_id"])
            event = pd.to_datetime(merged["event_time"], utc=True, errors="coerce")
            merged[EVENT_TIME_NS_COLUMN] = event.astype("int64")
            return merged

    def merge_symbol_id(frame: pd.DataFrame, warehouse_root: Path) -> pd.DataFrame:
        if frame.empty:
            return frame
        if SYMBOL_ID_COLUMN not in frame or EVENT_TIME_NS_COLUMN not in frame:
            row = {
                "join_version": JOIN_VERSION,
                "join_mode": "symbol_fallback",
                "dataset": "ALL",
                "trade_date": None,
                "input_rows": len(frame),
                "source_rows": 0,
                "matched_rows": 0,
                "admissible_rows": 0,
                "late_rows": 0,
                "duplicate_rows": 0,
                "reason": "temporary symbol_id/event_time metadata unavailable",
            }
            _atomic_audit(campaign, [row])
            if campaign.CTX is not None:
                raise RuntimeError(
                    "production v2.7 supplement join requires nff_meta__symbol_id and nff_meta__event_time_ns"
                )
            return fallback_merge(frame, warehouse_root)

        work = frame.reset_index()
        work["__symbol_id"] = _normalise_symbol_id(work[SYMBOL_ID_COLUMN])
        event_ns = pd.to_numeric(work[EVENT_TIME_NS_COLUMN], errors="coerce").astype("Int64")
        work["__event_time"] = pd.to_datetime(event_ns, unit="ns", utc=True, errors="coerce")
        work["__decision_time"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
        valid_key = work["__symbol_id"].notna() & work["__event_time"].notna()
        if not bool(valid_key.all()):
            raise RuntimeError(
                f"research frame has {int((~valid_key).sum())} rows without exact symbol_id/event_time"
            )
        dates = sorted(
            work["__event_time"]
            .dt.tz_convert("America/New_York")
            .dt.strftime("%Y-%m-%d")
            .dropna()
            .unique()
        )
        audits: list[dict[str, Any]] = []
        occupied = set(work.columns)
        for dataset in campaign.FF.SUPPLEMENT_DATASETS:
            root = Path(warehouse_root) / "nvg_supplement" / dataset / "schema=v1"
            for trade_date in dates:
                files = sorted((root / f"date={trade_date}").glob("*.parquet"))
                if not files:
                    audits.append(
                        {
                            "join_version": JOIN_VERSION,
                            "join_mode": "symbol_id_event_time",
                            "dataset": dataset,
                            "trade_date": trade_date,
                            "input_rows": len(work),
                            "source_rows": 0,
                            "matched_rows": 0,
                            "admissible_rows": 0,
                            "late_rows": 0,
                            "duplicate_rows": 0,
                            "reason": "partition_missing",
                        }
                    )
                    continue
                supplement = pads.dataset(
                    [str(path) for path in files], format="parquet"
                ).to_table().to_pandas()
                required = {"symbol_id", "timestamp", "available_time"}
                missing = sorted(required - set(supplement.columns))
                if missing:
                    raise KeyError(
                        f"supplement {dataset} date={trade_date} missing physical keys {missing}"
                    )
                supplement["__symbol_id"] = _normalise_symbol_id(supplement["symbol_id"])
                supplement["__event_time"] = pd.to_datetime(
                    supplement["timestamp"], utc=True, errors="coerce"
                )
                supplement["__supplement_available"] = pd.to_datetime(
                    supplement["available_time"], utc=True, errors="coerce"
                )
                valid = (
                    supplement["__symbol_id"].notna()
                    & supplement["__event_time"].notna()
                    & supplement["__supplement_available"].notna()
                )
                supplement = supplement.loc[valid].copy(deep=False)
                duplicate_rows = int(
                    supplement.duplicated(["__symbol_id", "__event_time"], keep=False).sum()
                )
                if duplicate_rows:
                    raise RuntimeError(
                        f"supplement {dataset} date={trade_date} has {duplicate_rows} duplicate symbol_id/timestamp rows"
                    )
                values = [
                    column
                    for column in supplement.columns
                    if column
                    not in {
                        "trade_date",
                        "symbol_id",
                        "symbol",
                        "timestamp",
                        "available_time",
                        "date",
                        "__symbol_id",
                        "__event_time",
                        "__supplement_available",
                    }
                ]
                collisions = sorted(set(values) & occupied)
                if collisions:
                    raise RuntimeError(
                        f"supplement field collision for {dataset}: {collisions[:20]}"
                    )
                source = supplement[
                    ["__symbol_id", "__event_time", "__supplement_available", *values]
                ]
                before = len(work)
                work = work.merge(
                    source,
                    on=["__symbol_id", "__event_time"],
                    how="left",
                    sort=False,
                    copy=False,
                    validate="many_to_one",
                )
                if len(work) != before:
                    raise AssertionError("supplement join changed research row count")
                matched = work["__supplement_available"].notna()
                supplement_decision = (
                    work["__supplement_available"].dt.ceil("1min")
                    + pd.Timedelta(minutes=1)
                )
                admissible = matched & (supplement_decision <= work["__decision_time"])
                late = matched & ~admissible
                if values:
                    work.loc[~admissible, values] = np.nan
                audits.append(
                    {
                        "join_version": JOIN_VERSION,
                        "join_mode": "symbol_id_event_time",
                        "dataset": dataset,
                        "trade_date": trade_date,
                        "input_rows": before,
                        "source_rows": len(supplement),
                        "matched_rows": int(matched.sum()),
                        "admissible_rows": int(admissible.sum()),
                        "late_rows": int(late.sum()),
                        "duplicate_rows": duplicate_rows,
                        "reason": "complete",
                    }
                )
                work = work.drop(columns=["__supplement_available"])
                occupied.update(values)
        _atomic_audit(campaign, audits)
        return (
            work.drop(
                columns=["__symbol_id", "__event_time", "__decision_time"], errors="ignore"
            )
            .set_index(["datetime", "instrument"])
            .sort_index()
        )

    def factor_contract(group: pd.DataFrame, frame: pd.DataFrame) -> str:
        return campaign._hash(
            {
                "base_contract": base_factor_contract(group, frame),
                "supplement_join_version": JOIN_VERSION,
                "join_key": ["symbol_id", "event_time"],
                "availability_rule": "ceil(supplement_available_time,1min)+1min <= decision_time",
            }
        )

    campaign.R.NFFDataLoader = ResearchMetadataLoader
    campaign.FF.merge_supplements = merge_symbol_id
    campaign._factor_contract = factor_contract
