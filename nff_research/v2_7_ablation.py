from __future__ import annotations

"""Train-only family ablation for the v2.7 Ridge baseline."""

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from nff_research import v2_5_full_campaign as BASE
from nff_research import v2_7_models as M


def _family(feature: str) -> str:
    try:
        return feature.split("__", 2)[1][:1].upper()
    except Exception:
        return "?"


def _fit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    candidates: list[str],
    config: Mapping[str, Any],
) -> tuple[float, float, int]:
    candidates = [column for column in candidates if column in train]
    contract = M._fit_feature_contract(
        train,
        candidates,
        minimum_coverage=float(config.get("minimum_feature_coverage", 0.70)),
        correlation_threshold=float(config.get("redundancy_correlation", 0.90)),
        max_features=int(config.get("max_features", 80)),
    )
    if not contract.features:
        return math.nan, math.nan, 0
    train_x = M._matrix(train, contract)
    validation_x = M._matrix(validation, contract)
    test_x = M._matrix(test, contract)
    train_y = M._rank_target(train)
    best = None
    for alpha in config.get("ridge_alphas", [0.001, 0.01, 0.1, 1.0]):
        model = M._fit_ridge(train_x, train_y, float(alpha))
        validation_prediction = M._predict_linear(model, validation_x)
        score = M._validation_score(validation, validation_prediction)
        if best is None or (np.isfinite(score) and score > best[0]):
            best = (score, model)
    if best is None:
        return math.nan, math.nan, len(contract.features)
    test_prediction = M._predict_linear(best[1], test_x)
    test_frame = M._prediction_frame(test, test_prediction, "ridge_ablation", -1)
    return float(best[0]), M._pooled_rank_ic(test_frame), len(contract.features)


def run_family_ablation(run_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    cache_root = run_root / "model_cache"
    paths = {
        path.parent.name.removeprefix("date="): path
        for path in cache_root.glob("date=*/dataset.parquet")
        if path.with_name("_SUCCESS").exists()
    }
    dates = sorted(paths)
    walk = config.get("walk_forward", {})
    folds = BASE._folds(
        dates,
        int(walk.get("train_days", 60)),
        int(walk.get("validation_days", 10)),
        int(walk.get("test_days", 10)),
        int(walk.get("step_days", 10)),
    )
    if not folds:
        return {"status": "MODEL_FAILED", "reason": "no temporal folds"}
    sample = pd.read_parquet(next(iter(paths.values())))
    candidates = [column for column in sample if column.startswith("full_factor__")]
    families = sorted({_family(feature) for feature in candidates if _family(feature) != "?"})
    metadata = ["label", "symbol", "decision_time"]
    rows: list[dict[str, Any]] = []
    for fold in folds:
        train = M._read_cache(paths, fold["train"], [*candidates, *metadata])
        validation = M._read_cache(paths, fold["validation"], [*candidates, *metadata])
        test = M._read_cache(paths, fold["test"], [*candidates, *metadata])
        for frame in (train, validation, test):
            frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
            frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
        train = train.dropna(subset=metadata)
        validation = validation.dropna(subset=metadata)
        test = test.dropna(subset=metadata)
        if len(train) < 100 or validation.empty or test.empty:
            continue
        baseline_validation, baseline_test, baseline_count = _fit_predict(
            train, validation, test, candidates, walk
        )
        rows.append(
            {
                "fold_id": fold["fold_id"],
                "removed_family": "NONE_BASELINE",
                "validation_rank_ic": baseline_validation,
                "test_rank_ic": baseline_test,
                "delta_test_rank_ic_vs_baseline": 0.0,
                "feature_count": baseline_count,
            }
        )
        for family in families:
            reduced = [feature for feature in candidates if _family(feature) != family]
            validation_ic, test_ic, feature_count = _fit_predict(
                train, validation, test, reduced, walk
            )
            rows.append(
                {
                    "fold_id": fold["fold_id"],
                    "removed_family": family,
                    "validation_rank_ic": validation_ic,
                    "test_rank_ic": test_ic,
                    "delta_test_rank_ic_vs_baseline": test_ic - baseline_test,
                    "feature_count": feature_count,
                }
            )
    frame = pd.DataFrame(rows)
    output = run_root / "family_ablation_v2_7.parquet"
    frame.to_parquet(output, index=False, compression="zstd")
    summary = (
        frame.loc[~frame["removed_family"].eq("NONE_BASELINE")]
        .groupby("removed_family", sort=False)
        .agg(
            folds=("fold_id", "nunique"),
            mean_delta_test_rank_ic=("delta_test_rank_ic_vs_baseline", "mean"),
            median_delta_test_rank_ic=("delta_test_rank_ic_vs_baseline", "median"),
            mean_test_rank_ic=("test_rank_ic", "mean"),
        )
        .reset_index()
        if not frame.empty
        else pd.DataFrame()
    )
    summary.to_parquet(run_root / "family_ablation_summary_v2_7.parquet", index=False, compression="zstd")
    result = {
        "status": "complete" if not frame.empty else "MODEL_FAILED",
        "rows": len(frame),
        "families": families,
        "summary": summary.to_dict("records"),
    }
    (run_root / "family_ablation_v2_7.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return result
