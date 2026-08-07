from __future__ import annotations

"""Training-window-first orchestration for v2.8.

The first 60 chronological dates are materialized and raw-screened before the
remaining dates. Candidate selection can therefore freeze early instead of
waiting for all 138 materialization jobs.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


def _scoped(config: dict[str, Any], dates: list[str], suffix: str) -> dict[str, Any]:
    value = deepcopy(config)
    value.setdefault("pipeline", {})["_runtime_dates"] = list(dates)
    value["pipeline"]["_runtime_status_suffix"] = suffix
    return value


def install(P: Any) -> None:
    def main(argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument(
            "--stage",
            choices=[
                "bootstrap",
                "materialize",
                "basic_screen",
                "select",
                "detailed",
                "portfolio",
                "all",
            ],
            default="all",
        )
        parser.add_argument(
            "--worker-stage",
            choices=["materialize", "basic_screen", "detailed", "portfolio"],
        )
        parser.add_argument("--worker-date")
        args = parser.parse_args(argv)
        config_path = Path(args.config).resolve()
        config = P._load_config(config_path)
        P.bootstrap(config)
        P.write_architecture(config)

        if args.worker_stage:
            if not args.worker_date:
                raise ValueError("--worker-stage requires --worker-date")
            print(
                json.dumps(
                    P.run_worker(config, args.worker_stage, args.worker_date),
                    indent=2,
                    default=str,
                )
            )
            return 0

        all_dates = P._dates(config)
        train_days = int(config.get("selection", {}).get("train_days", 60))
        training_dates = all_dates[:train_days]
        remaining_dates = all_dates[train_days:]
        results: list[dict[str, Any]] = []

        def run_stage(stage: str, scoped_config: dict[str, Any] | None = None) -> bool:
            active = scoped_config or config
            if stage == "select":
                result = P.select_candidates(active)
            else:
                result = P._run_date_stage(config_path, active, stage)
            results.append(result)
            return result.get("status") in {"complete", "skipped"}

        if args.stage == "bootstrap":
            training = _scoped(config, training_dates, "training_window")
            ok = run_stage("materialize", training)
            ok = ok and run_stage("basic_screen", training)
            ok = ok and run_stage("select")
            print(json.dumps(results, indent=2, default=str))
            return 0 if ok else 2

        if args.stage == "all":
            training = _scoped(config, training_dates, "training_window")
            if not run_stage("materialize", training):
                print(json.dumps(results, indent=2, default=str))
                return 2
            if not run_stage("basic_screen", training):
                print(json.dumps(results, indent=2, default=str))
                return 2
            if not run_stage("select"):
                print(json.dumps(results, indent=2, default=str))
                return 2

            if remaining_dates:
                remaining = _scoped(config, remaining_dates, "post_selection")
                if not run_stage("materialize", remaining):
                    print(json.dumps(results, indent=2, default=str))
                    return 2
                if not run_stage("basic_screen", remaining):
                    print(json.dumps(results, indent=2, default=str))
                    return 2

            if not run_stage("detailed"):
                print(json.dumps(results, indent=2, default=str))
                return 2
            if not run_stage("portfolio"):
                print(json.dumps(results, indent=2, default=str))
                return 2
            print(json.dumps(results, indent=2, default=str))
            return 0

        ok = run_stage(args.stage)
        print(json.dumps(results, indent=2, default=str))
        return 0 if ok else 2

    P.main = main
