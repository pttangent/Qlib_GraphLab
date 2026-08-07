from __future__ import annotations

"""Training-only multi-track selection for the v2.8 pipeline.

Return alpha, risk/regime and cost/liquidity factors answer different economic
questions.  A factor with no return IC can still be a strong future-volatility
or liquidity gate.  This module prevents the primary-return screen from
silently deleting those factors before detailed diagnostics.
"""

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _track_defaults() -> list[dict[str, Any]]:
    return [
        {
            "name": "alpha",
            "label_families": ["return_vwap_to_vwap"],
            "horizons": [5, 15, 30, 60],
            "detailed_max_features": 80,
            "portfolio_max_features": 40,
            "detailed_max_per_family": 14,
            "portfolio_max_per_family": 8,
        },
        {
            "name": "risk_regime",
            "label_families": [
                "realized_volatility",
                "liquidity_deterioration",
                "jump_tail_event",
            ],
            "horizons": [5, 15, 30, 60],
            "detailed_max_features": 30,
            "portfolio_max_features": 0,
            "detailed_max_per_family": 8,
        },
        {
            "name": "cost_liquidity",
            "label_families": ["execution_cost_proxy"],
            "horizons": [5, 15, 30, 60],
            "detailed_max_features": 10,
            "portfolio_max_features": 0,
            "detailed_max_per_family": 4,
        },
    ]


def _track_summary(
    data: pd.DataFrame,
    track: Mapping[str, Any],
    universe: str,
) -> pd.DataFrame:
    families = {str(value) for value in track.get("label_families", ())}
    horizons = {int(value) for value in track.get("horizons", ())}
    work = data[
        data["label_family"].isin(families)
        & data["horizon_bars"].isin(horizons)
        & (data["universe"] == universe)
        & (data["rank_ic_method"] == "minute_mean_cs_rank_ic")
    ].copy()
    if work.empty:
        return pd.DataFrame()
    grouped = work.groupby(
        ["feature", "factor_family", "label_family", "horizon_bars"],
        sort=False,
    )
    summary = grouped.agg(
        mean_ic=("rank_ic_mean", "mean"),
        std_ic=("rank_ic_mean", "std"),
        valid_days=("rank_ic_mean", "count"),
        positive_day_ratio=("rank_ic_mean", lambda value: float((value > 0).mean())),
        coverage_mean=("coverage", "mean"),
        minute_positive_ratio=("rank_ic_positive_ratio", "mean"),
    ).reset_index()
    summary["direction"] = np.where(summary["mean_ic"] >= 0, 1.0, -1.0)
    summary["directional_day_ratio"] = np.where(
        summary["direction"] > 0,
        summary["positive_day_ratio"],
        1.0 - summary["positive_day_ratio"],
    )
    summary["directional_minute_ratio"] = np.where(
        summary["direction"] > 0,
        summary["minute_positive_ratio"],
        1.0 - summary["minute_positive_ratio"],
    )
    summary["icir"] = summary["mean_ic"].abs() / summary["std_ic"].replace(0, np.nan)
    summary["selection_score"] = (
        summary["mean_ic"].abs()
        * np.sqrt(summary["valid_days"].clip(lower=1))
        * summary["directional_day_ratio"].clip(lower=0.5)
        * summary["directional_minute_ratio"].clip(lower=0.5)
        * np.sqrt(summary["coverage_mean"].clip(lower=0))
    )
    summary["selection_track"] = str(track["name"])
    summary["direction_semantics"] = (
        "future_return_direction"
        if str(track["name"]) == "alpha"
        else "target_association_not_trade_direction"
    )
    # A factor can be measured against multiple labels/horizons within one
    # research track.  Keep the strongest training-only specification and
    # record which target selected it.
    return (
        summary.sort_values(
            ["selection_score", "valid_days", "feature"],
            ascending=[False, False, True],
        )
        .groupby("feature", sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def install(P: Any) -> None:
    def select_candidates_multitrack(config: dict[str, Any]) -> dict[str, Any]:
        P.bootstrap(config)
        root = P._pipeline_root(config) / "selection"
        root.mkdir(parents=True, exist_ok=True)
        contract = P._contract_hash(config)
        success = root / "_SUCCESS"
        meta_path = root / "meta.json"
        if success.exists() and meta_path.exists():
            meta = __import__("json").loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("contract_hash") == contract:
                return {"stage": "select", "status": "skipped", **meta}

        available_dates = [
            trade_date
            for trade_date in P._dates(config)
            if P._stage_success(config, "basic_screen", trade_date).exists()
        ]
        selection = config.get("selection", {})
        train_days = int(selection.get("train_days", 60))
        if len(available_dates) < train_days:
            raise RuntimeError(
                f"candidate selection requires {train_days} completed screen dates; "
                f"found {len(available_dates)}"
            )
        train_dates = available_dates[:train_days]
        frames = [
            pd.read_parquet(
                P._stage_root(config, "basic_screen", date)
                / "basic_factor_screen.parquet"
            )
            for date in train_dates
        ]
        data = pd.concat(frames, ignore_index=True)
        universe = str(
            selection.get("primary_universe", "final_trading_universe")
        )
        configured_tracks = selection.get("tracks")
        tracks = (
            list(configured_tracks)
            if isinstance(configured_tracks, list) and configured_tracks
            else _track_defaults()
        )
        minimum_days = int(
            selection.get("minimum_valid_days", max(10, train_days // 3))
        )
        minimum_coverage = float(selection.get("minimum_coverage", 0.70))
        minimum_stability = float(
            selection.get("minimum_directional_day_ratio", 0.55)
        )
        minimum_per_family = int(selection.get("minimum_per_family", 2))

        all_scores: list[pd.DataFrame] = []
        detailed_parts: list[pd.DataFrame] = []
        portfolio_parts: list[pd.DataFrame] = []
        track_meta: list[dict[str, Any]] = []
        for track in tracks:
            summary = _track_summary(data, track, universe)
            if summary.empty:
                track_meta.append(
                    {
                        "name": str(track.get("name")),
                        "status": "NO_SCREEN_ROWS",
                    }
                )
                continue
            eligible = summary[
                (summary["valid_days"] >= minimum_days)
                & (summary["coverage_mean"] >= minimum_coverage)
                & (
                    summary["directional_day_ratio"]
                    >= minimum_stability
                )
            ].copy()
            warning = ""
            if eligible.empty:
                eligible = summary.copy()
                warning = "FALLBACK_WEAK_NO_FACTOR_PASSED_THRESHOLDS"
            eligible["selection_warning"] = warning
            detailed_max = int(track.get("detailed_max_features", 0))
            portfolio_max = int(track.get("portfolio_max_features", 0))
            detailed = P._candidate_rows(
                eligible,
                max_features=detailed_max,
                min_per_family=minimum_per_family,
                max_per_family=int(
                    track.get("detailed_max_per_family", max(1, detailed_max))
                ),
            ) if detailed_max > 0 else eligible.iloc[0:0].copy()
            portfolio = P._candidate_rows(
                eligible,
                max_features=portfolio_max,
                min_per_family=minimum_per_family,
                max_per_family=int(
                    track.get("portfolio_max_per_family", max(1, portfolio_max))
                ),
            ) if portfolio_max > 0 else eligible.iloc[0:0].copy()
            if not detailed.empty:
                detailed["candidate_scope"] = "detailed"
                detailed_parts.append(detailed)
            if not portfolio.empty:
                if str(track.get("name")) != "alpha":
                    raise RuntimeError(
                        "only the alpha selection track may emit portfolio candidates"
                    )
                portfolio["candidate_scope"] = "portfolio"
                portfolio_parts.append(portfolio)
            all_scores.append(summary)
            track_meta.append(
                {
                    "name": str(track.get("name")),
                    "status": "complete",
                    "eligible_count": int(eligible["feature"].nunique()),
                    "detailed_count": int(detailed["feature"].nunique()),
                    "portfolio_count": int(portfolio["feature"].nunique()),
                    "label_families": list(track.get("label_families", ())),
                }
            )

        if not all_scores:
            raise RuntimeError("all configured selection tracks have no screen rows")
        score_frame = pd.concat(all_scores, ignore_index=True)
        detailed_frame = (
            pd.concat(detailed_parts, ignore_index=True)
            if detailed_parts
            else score_frame.iloc[0:0].copy()
        )
        # The same physical factor can be useful for both alpha and risk. Keep
        # one detailed copy with the strongest training score, while retaining
        # the winning track and target in the manifest.
        if not detailed_frame.empty:
            detailed_frame = (
                detailed_frame.sort_values("selection_score", ascending=False)
                .drop_duplicates("feature", keep="first")
            )
        portfolio_frame = (
            pd.concat(portfolio_parts, ignore_index=True)
            if portfolio_parts
            else score_frame.iloc[0:0].copy()
        )
        candidates = pd.concat(
            [detailed_frame, portfolio_frame], ignore_index=True
        )
        P._atomic_parquet(
            score_frame, root / "all_selection_scores.parquet", index=False
        )
        P._atomic_parquet(
            candidates, root / "candidates.parquet", index=False
        )
        freeze_date = train_dates[-1]
        meta = {
            "version": P.VERSION,
            "stage": "select",
            "status": "complete",
            "contract_hash": contract,
            "selection_dates": train_dates,
            "selection_start_date": train_dates[0],
            "selection_end_date": freeze_date,
            "portfolio_eligible_after": freeze_date,
            "direction_contract": (
                "alpha portfolio sign is sign(mean training-date rank IC); "
                "risk/cost signs are target associations only"
            ),
            "diagnostic_candidate_count": int(
                detailed_frame["feature"].nunique()
            ),
            "portfolio_candidate_count": int(
                portfolio_frame["feature"].nunique()
            ),
            "selection_tracks": track_meta,
            "thresholds": {
                "minimum_valid_days": minimum_days,
                "minimum_coverage": minimum_coverage,
                "minimum_directional_day_ratio": minimum_stability,
            },
        }
        P._atomic_json(meta_path, meta)
        success.write_text(pd.Timestamp.utcnow().isoformat(), encoding="utf-8")
        return meta

    P.select_candidates = select_candidates_multitrack
