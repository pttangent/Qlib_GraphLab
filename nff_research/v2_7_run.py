from __future__ import annotations

"""Canonical v2.7 entrypoint.

Every detached date worker executes this file so that real-schema,
qcut-equivalent, universe, multi-model and family-ablation patches are installed
in both the parent scheduler and child processes.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Must be installed before importing v2_7_atomic_entry: that chain imports
# v2_6_full_defined_campaign, which parses prototypes at module-import time.
from nff_research import full_factor_engine as _FULL_FACTOR_ENGINE
from nff_research import v2_7_prototype_contract as _PROTOTYPE_CONTRACT

_PROTOTYPE_CONTRACT.install(_FULL_FACTOR_ENGINE)

from nff_research import v2_7_ablation as ABLATION
from nff_research import v2_7_atomic_entry as ENTRY
from nff_research import v2_7_models as MODELS


C = ENTRY.C
_ORIGINAL_RANK_FEATURES = MODELS._rank_features


def _rank_features_schema_safe(frame, features, directions):
    work = frame.copy(deep=False)
    missing = [feature for feature in features if feature not in work]
    if missing:
        work = work.copy()
        for feature in missing:
            work[feature] = np.nan
    return _ORIGINAL_RANK_FEATURES(work, features, directions)


def _fit_feature_contract_fast(
    train,
    candidates,
    *,
    minimum_coverage=0.70,
    correlation_threshold=0.90,
    max_features=80,
    min_n=30,
):
    available = [feature for feature in candidates if feature in train]
    if not available:
        return MODELS.FeatureContract([], {}, {}, {}, float(correlation_threshold))
    coverage = train[available].notna().mean()
    available = [feature for feature in available if float(coverage.get(feature, 0.0)) >= minimum_coverage]
    scores = MODELS._feature_ic_scores(train, available, min_n=min_n)
    ordered = sorted(
        [feature for feature in available if np.isfinite(scores.get(feature, math.nan))],
        key=lambda feature: (-abs(scores[feature]), feature),
    )
    # Features far below the eventual representative budget cannot be selected
    # after a 0.90 redundancy filter. Pre-truncating by train IC keeps the
    # matrix bounded while preserving the strongest candidates from every
    # family through deterministic ordering.
    ordered = ordered[: max(max_features * 4, max_features)]
    directions = {feature: (1.0 if scores[feature] >= 0 else -1.0) for feature in ordered}
    ranked = _rank_features_schema_safe(train, ordered, directions)
    if len(ranked) > 20_000:
        ranked = ranked.iloc[np.linspace(0, len(ranked) - 1, 20_000, dtype=int)]
    correlation = ranked.corr().abs().fillna(0.0)
    selected: list[str] = []
    for feature in ordered:
        if len(selected) >= max_features:
            break
        if selected and bool((correlation.loc[feature, selected] >= correlation_threshold).any()):
            continue
        selected.append(feature)
    medians = {
        feature: float(pd.to_numeric(train[feature], errors="coerce").median())
        for feature in selected
    }
    return MODELS.FeatureContract(
        features=selected,
        directions={feature: directions[feature] for feature in selected},
        ic_scores={feature: scores[feature] for feature in selected},
        medians=medians,
        correlation_threshold=float(correlation_threshold),
    )


MODELS._rank_features = _rank_features_schema_safe
MODELS._fit_feature_contract = _fit_feature_contract_fast
# Ablation imports the same module object, so it receives the same optimized,
# schema-safe train-only selector.


def _run_temporal_oos_complete(run_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    result = MODELS.run_temporal_oos_multi(run_root, config)
    if bool(config.get("walk_forward", {}).get("family_ablation", True)):
        try:
            result["family_ablation"] = ABLATION.run_family_ablation(run_root, config)
        except Exception as exc:
            result["family_ablation"] = {"status": "MODEL_FAILED", "error": repr(exc)}
    return result


C.BASE.run_temporal_oos = _run_temporal_oos_complete


def _install_worker_command() -> None:
    original = C.R.worker_command
    current = str(Path(__file__).resolve())
    known_scripts = {
        str(Path(C.R.__file__).resolve()),
        str(Path(C.V26.__file__).resolve()),
        str(Path(C.__file__).resolve()),
        str(Path(ENTRY.__file__).resolve()),
    }

    def worker_command(*args: Any, **kwargs: Any) -> list[str]:
        command = original(*args, **kwargs)
        replaced = False
        result: list[str] = []
        for value in command:
            if value in known_scripts or Path(value).name in {
                "v2_1_neutralized_runner.py",
                "v2_6_full_defined_campaign.py",
                "v2_7_atomic_campaign.py",
                "v2_7_atomic_entry.py",
            }:
                result.append(current)
                replaced = True
            else:
                result.append(value)
        if not replaced:
            raise RuntimeError(f"worker command has no replaceable research entrypoint: {command}")
        return result

    C.R.worker_command = worker_command


def _write_final_report(run_root: Path, model_result: dict[str, Any], config: dict[str, Any]) -> None:
    report = run_root / "reports" / "final_report_v2_7.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    contract_source = (
        C.FF.contract_source_manifest()
        if hasattr(C.FF, "contract_source_manifest")
        else {"source": "UNKNOWN"}
    )
    lines = [
        "# NFF v2.7 Real-Schema Atomic Campaign",
        "",
        f"- Status: `{model_result.get('status')}`",
        f"- OOS prediction rows: `{model_result.get('prediction_rows', 0)}`",
        f"- Models requested: `{model_result.get('models_requested', [])}`",
        f"- Models successful: `{model_result.get('models_successful', [])}`",
        f"- Primary model selected only on validation: `{model_result.get('primary_model_selected_on_validation')}`",
        f"- Prototype contract source: `{contract_source.get('source')}`",
        f"- Prototype contract SHA: `{contract_source.get('contract_sha256')}`",
        "",
        "## Contract",
        "",
        "The rebuilt `nvg_supplement` Parquet schema is audited per date. Factor direction, feature coverage, redundancy filtering, model hyperparameters and model selection are fitted without test data. Exact labels enter at t+1 and exit at t+h+1.",
        "",
        "```json",
        json.dumps(contract_source, indent=2, ensure_ascii=False, default=str),
        "```",
        "",
        "## Model validation scores",
        "",
        "```json",
        json.dumps(model_result.get("validation_model_scores", {}), indent=2, ensure_ascii=False, default=str),
        "```",
        "",
        "## Model failures",
        "",
        "```json",
        json.dumps(model_result.get("model_failures", []), indent=2, ensure_ascii=False, default=str),
        "```",
        "",
        "## Family ablation",
        "",
        "Each A-K/S family is removed before train-only feature selection and Ridge hyperparameter selection; the model is then retrained and tested on the same chronological fold.",
        "",
        "```json",
        json.dumps(model_result.get("family_ablation", {}), indent=2, ensure_ascii=False, default=str),
        "```",
        "",
        "## Account and capacity",
        "",
        "High expected-return predictions are long and low predictions are short. Orders are participation-limited; spread, impact and borrow remain estimated because a complete quote/order-book tape is unavailable.",
        "",
        "```json",
        json.dumps(model_result.get("account", {}), indent=2, ensure_ascii=False, default=str),
        "```",
        "",
        f"Capacity scenarios: `{model_result.get('capacity_rows', 0)}`.",
        "",
        "## Recorder",
        "",
        "```json",
        json.dumps(model_result.get("recorder", {}), indent=2, ensure_ascii=False, default=str),
        "```",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    compatibility = run_root / "reports" / "final_report_v2_5.md"
    compatibility.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")


C._install_worker_command = _install_worker_command
C.BASE._write_final_report = _write_final_report


def main() -> int:
    return C.main()


if __name__ == "__main__":
    raise SystemExit(main())
