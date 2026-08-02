"""Inspect an NFF warehouse or run a standard Qlib DatasetH + LightGBM workflow.

Examples
--------
python examples/nff_qlib_pipeline.py inspect --warehouse-root D:\\...\\NFF_warehouse
python examples/nff_qlib_pipeline.py run --config examples/configs/nff_qlib_example.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

import qlib
from qlib.constant import REG_US
from qlib.contrib.data.nff import NFFDataHandlerLP, NFFWarehouseCatalog
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow import R


def _read_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rank_ic(prediction: pd.Series, label: pd.Series) -> pd.Series:
    aligned = pd.concat([prediction.rename("score"), label.rename("label")], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype="float64")
    return aligned.groupby(level="datetime").apply(
        lambda frame: frame["score"].corr(frame["label"], method="spearman") if len(frame) >= 2 else float("nan")
    )


def inspect(args: argparse.Namespace) -> None:
    catalog = NFFWarehouseCatalog(args.warehouse_root)
    print(json.dumps(catalog.describe(), indent=2, default=str))


def run(args: argparse.Namespace) -> None:
    config = _read_json(args.config)
    handler_config = dict(config["handler"])
    segments = dict(config["segments"])
    model_config = dict(config.get("model", {}))
    output_dir = Path(config.get("output_dir", "nff_qlib_output")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # The custom loader does not use Qlib's native feature provider, but Qlib's
    # Recorder still needs the global framework to be registered. Pointing the
    # provider URI at the existing NFF root creates no copied dataset.
    warehouse_root = Path(handler_config["warehouse_root"]).expanduser().resolve()
    qlib.init(provider_uri=str(warehouse_root), region=REG_US, skip_if_reg=True)

    handler = NFFDataHandlerLP(**handler_config)
    dataset = DatasetH(handler=handler, segments=segments)
    model = LGBModel(**model_config)

    experiment_name = str(config.get("experiment_name", "nff_qlib_screening"))
    with R.start(experiment_name=experiment_name):
        model.fit(dataset)
        prediction = model.predict(dataset, segment="test")
        label_frame = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
        label = label_frame.iloc[:, 0]
        ic = _rank_ic(prediction, label)

        prediction.to_frame("score").to_parquet(output_dir / "prediction.parquet")
        label.to_frame("label").to_parquet(output_dir / "label.parquet")
        ic.to_frame("rank_ic").to_parquet(output_dir / "daily_rank_ic.parquet")
        metrics = {
            "prediction_rows": int(prediction.notna().sum()),
            "label_rows": int(label.notna().sum()),
            "rank_ic_mean": None if ic.dropna().empty else float(ic.mean()),
            "rank_ic_std": None if ic.dropna().empty else float(ic.std()),
            "rank_ic_positive_ratio": None if ic.dropna().empty else float((ic > 0).mean()),
            "loader_report": handler.data_loader.last_load_report,
        }
        scalar_metrics = {key: value for key, value in metrics.items() if key != "loader_report" and value is not None}
        if scalar_metrics:
            R.log_metrics(**scalar_metrics)
        R.save_objects(**{"model.pkl": model})

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metrics, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="List NFF schemas, dates and columns")
    inspect_parser.add_argument("--warehouse-root", required=True)
    inspect_parser.set_defaults(func=inspect)

    run_parser = subparsers.add_parser("run", help="Train and evaluate a Qlib model from NFF")
    run_parser.add_argument("--config", required=True)
    run_parser.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
