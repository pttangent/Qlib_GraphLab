from __future__ import annotations

"""Train and evaluate the NFF-only P2 generic GRU baseline."""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlib.contrib.data.nff_episode import NFFStockDayEpisodeFactory, contract_hash
from qlib.contrib.model.nff_generic import (
    GenericGRUConfig,
    NFFGenericGRU,
    NormalizationState,
    RunningMoments,
    episodes_to_query_arrays,
    inverse_targets,
    masked_mse,
    prediction_metrics,
)


def _read_config(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Config must contain a YAML mapping")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), tmp)
    tmp.replace(path)


def _factory(config: Mapping[str, Any]) -> NFFStockDayEpisodeFactory:
    factory_config = dict(config.get("episode_factory") or {})
    if not factory_config:
        raise ValueError("Config requires episode_factory")
    return NFFStockDayEpisodeFactory(**factory_config)


def _manifest(config: Mapping[str, Any], explicit: Optional[Path] = None) -> pd.DataFrame:
    path = explicit or Path(config.get("output", {}).get("episode_run_root", "episode_runs/pilot")) / "aggregate" / "episode_manifest.parquet"
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Episode manifest not found: {path}. Run P1 build-manifest first.")
    frame = pd.read_parquet(path)
    if frame.empty:
        raise RuntimeError("Episode manifest is empty")
    return frame


def _split_dates(dates: Sequence[str], config: Mapping[str, Any]) -> Dict[str, list[str]]:
    ordered = sorted(set(str(value) for value in dates))
    split_config = dict(config.get("segments") or {})
    explicit = all(name in split_config for name in ("train", "valid", "test"))
    if explicit:
        result: Dict[str, list[str]] = {}
        for name in ("train", "valid", "test"):
            start, end = split_config[name]
            result[name] = [value for value in ordered if str(start) <= value <= str(end)]
    else:
        ratios = split_config.get("ratios", [0.70, 0.15, 0.15])
        if len(ratios) != 3 or not np.isclose(sum(float(value) for value in ratios), 1.0):
            raise ValueError("segments.ratios must contain three values summing to 1")
        n = len(ordered)
        train_end = max(1, int(np.floor(n * float(ratios[0]))))
        valid_end = max(train_end + 1, int(np.floor(n * (float(ratios[0]) + float(ratios[1])))))
        valid_end = min(valid_end, n - 1)
        result = {
            "train": ordered[:train_end],
            "valid": ordered[train_end:valid_end],
            "test": ordered[valid_end:],
        }
    if any(not result[name] for name in ("train", "valid", "test")):
        raise ValueError(f"Temporal split produced an empty segment: {result}")
    if max(result["train"]) >= min(result["valid"]) or max(result["valid"]) >= min(result["test"]):
        raise AssertionError("Segments are not strictly ordered in time")
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    return device


def _fit_normalization(
    factory: NFFStockDayEpisodeFactory,
    train_dates: Sequence[str],
    output_root: Path,
    force: bool,
) -> NormalizationState:
    final_path = output_root / "normalization.json"
    progress_path = output_root / "normalization_progress.json"
    if final_path.exists() and not force:
        return NormalizationState.from_dict(json.loads(final_path.read_text(encoding="utf-8")))

    feature_moments = None
    target_moments = None
    completed: list[str] = []
    feature_names: Tuple[str, ...] = ()
    target_names: Tuple[str, ...] = ()
    if progress_path.exists() and not force:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("episode_contract_hash") != factory.contract_hash:
            raise RuntimeError("Normalization progress belongs to a different episode contract")
        feature_moments = RunningMoments.from_state_dict(progress["feature_moments"])
        target_moments = RunningMoments.from_state_dict(progress["target_moments"])
        completed = list(progress.get("completed_dates", []))
        feature_names = tuple(progress["feature_names"])
        target_names = tuple(progress["target_names"])

    for trade_date in train_dates:
        if trade_date in completed:
            continue
        result = factory.load_date(trade_date)
        for episode in result.episodes:
            if feature_moments is None:
                feature_names = tuple(episode.feature_names)
                target_names = tuple(episode.target_names)
                feature_moments = RunningMoments(len(feature_names))
                target_moments = RunningMoments(len(target_names))
            if tuple(episode.feature_names) != feature_names or tuple(episode.target_names) != target_names:
                raise RuntimeError("Episode schema changed while fitting normalization")
            feature_moments.update(episode.query_x, episode.query_observed)
            target_moments.update(episode.targets, episode.target_observed)
        completed.append(trade_date)
        if feature_moments is not None and target_moments is not None:
            _atomic_json(
                progress_path,
                {
                    "episode_contract_hash": factory.contract_hash,
                    "completed_dates": completed,
                    "feature_names": feature_names,
                    "target_names": target_names,
                    "feature_moments": feature_moments.state_dict(),
                    "target_moments": target_moments.state_dict(),
                },
            )

    if feature_moments is None or target_moments is None:
        raise RuntimeError("No training episodes were available to fit normalization")
    normalization = NormalizationState(
        feature_mean=tuple(float(value) for value in feature_moments.mean),
        feature_std=tuple(float(value) for value in feature_moments.std()),
        target_mean=tuple(float(value) for value in target_moments.mean),
        target_std=tuple(float(value) for value in target_moments.std()),
        feature_names=feature_names,
        target_names=target_names,
    )
    _atomic_json(final_path, normalization.to_dict())
    return normalization


def _date_arrays(factory, trade_date, normalization):
    result = factory.load_date(trade_date)
    return episodes_to_query_arrays(result.episodes, normalization)


def _train_date(
    model,
    optimizer,
    arrays,
    *,
    device,
    batch_size,
    grad_clip,
    use_amp,
    scaler,
):
    x, y, y_mask, _ = arrays
    if len(x) == 0:
        return {"samples": 0, "loss": None}
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(y_mask))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model.train()
    losses = []
    for batch_x, batch_y, batch_mask in loader:
        batch_x = batch_x.to(device=device, dtype=torch.float32, non_blocking=True)
        batch_y = batch_y.to(device=device, dtype=torch.float32, non_blocking=True)
        batch_mask = batch_mask.to(device=device, dtype=torch.bool, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(batch_x)
            loss = masked_mse(prediction, batch_y, batch_mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return {"samples": int(len(x)), "loss": float(np.mean(losses)) if losses else None}


def _prediction_frame(prediction_standardized, target_standardized, target_mask, metadata, normalization):
    prediction = inverse_targets(prediction_standardized, normalization)
    target = inverse_targets(target_standardized, normalization)
    rows = []
    for target_index, target_name in enumerate(normalization.target_names):
        observed = target_mask[:, target_index].astype(bool)
        if not observed.any():
            continue
        part = metadata.loc[observed].copy()
        part["target_name"] = target_name
        part["prediction"] = prediction[observed, target_index]
        part["target"] = target[observed, target_index]
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _evaluate_dates(
    model,
    factory,
    dates,
    normalization,
    *,
    device,
    batch_size,
    output_dir=None,
):
    model.eval()
    frames = []
    date_metrics = []
    with torch.no_grad():
        for trade_date in dates:
            x, y, y_mask, metadata = _date_arrays(factory, trade_date, normalization)
            if len(x) == 0:
                continue
            dataset = TensorDataset(torch.from_numpy(x))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            predictions = []
            for (batch_x,) in loader:
                batch_x = batch_x.to(device=device, dtype=torch.float32, non_blocking=True)
                predictions.append(model(batch_x).cpu().numpy())
            frame = _prediction_frame(np.concatenate(predictions), y, y_mask, metadata, normalization)
            metrics = prediction_metrics(frame)
            metrics["trade_date"] = trade_date
            date_metrics.append(metrics)
            if output_dir is not None:
                _atomic_parquet(frame, output_dir / "predictions" / f"date={trade_date}.parquet")
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    metrics = prediction_metrics(combined)
    metrics["dates"] = len(date_metrics)
    if output_dir is not None:
        _atomic_parquet(pd.DataFrame(date_metrics), output_dir / "daily_metrics.parquet")
        _atomic_json(output_dir / "summary.json", metrics)
    return metrics, combined


def command_train(config, manifest_path, output_root, force_normalization):
    torch.set_num_threads(int(config.get("run", {}).get("torch_threads", 4)))
    seed = int(config.get("run", {}).get("seed", 20260804))
    _seed_everything(seed)
    manifest = _manifest(config, manifest_path)
    segments = _split_dates(manifest["trade_date"].astype(str).unique(), config)
    factory = _factory(config)
    output_root.mkdir(parents=True, exist_ok=True)
    normalization = _fit_normalization(factory, segments["train"], output_root, force_normalization)

    model_config = GenericGRUConfig(**dict(config.get("model") or {}))
    model = NFFGenericGRU(len(normalization.feature_names), len(normalization.target_names), model_config)
    training = dict(config.get("training") or {})
    device = _resolve_device(str(training.get("device", "auto")))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    epochs = int(training.get("epochs", 10))
    batch_size = int(training.get("batch_size", 256))
    grad_clip = float(training.get("grad_clip", 1.0))

    train_contract = {
        "version": "nff_generic_gru_p2_v1",
        "episode_contract_hash": factory.contract_hash,
        "normalization": normalization.to_dict(),
        "model": model.model_contract(),
        "segments": segments,
        "training": training,
        "seed": seed,
    }
    expected_hash = contract_hash(train_contract)
    _atomic_json(output_root / "train_contract.json", {"contract_hash": expected_hash, **train_contract})

    checkpoint_path = output_root / "checkpoint_last.pt"
    best_path = output_root / "checkpoint_best.pt"
    start_epoch = 0
    next_date_index = 0
    best_valid_loss = float("inf")
    history_rows = []
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("contract_hash") != expected_hash:
            raise RuntimeError("Existing checkpoint has a different training contract")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint.get("epoch", 0))
        next_date_index = int(checkpoint.get("next_date_index", 0))
        best_valid_loss = float(checkpoint.get("best_valid_loss", float("inf")))
        history_rows = list(checkpoint.get("history_rows", []))

    for epoch in range(start_epoch, epochs):
        train_dates = list(segments["train"])
        random.Random(seed + epoch).shuffle(train_dates)
        date_start = next_date_index if epoch == start_epoch else 0
        for date_index in range(date_start, len(train_dates)):
            trade_date = train_dates[date_index]
            started = time.perf_counter()
            metrics = _train_date(
                model,
                optimizer,
                _date_arrays(factory, trade_date, normalization),
                device=device,
                batch_size=batch_size,
                grad_clip=grad_clip,
                use_amp=use_amp,
                scaler=scaler,
            )
            row = {
                "epoch": epoch,
                "trade_date": trade_date,
                "segment": "train",
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                **metrics,
            }
            history_rows.append(row)
            print(json.dumps(row))
            _atomic_torch(
                checkpoint_path,
                {
                    "contract_hash": expected_hash,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "next_date_index": date_index + 1,
                    "best_valid_loss": best_valid_loss,
                    "history_rows": history_rows,
                },
            )

        valid_metrics, _ = _evaluate_dates(
            model,
            factory,
            segments["valid"],
            normalization,
            device=device,
            batch_size=batch_size,
        )
        valid_loss = float(valid_metrics["mse"]) if valid_metrics.get("mse") is not None else float("inf")
        validation_row = {"epoch": epoch, "segment": "valid", **valid_metrics}
        history_rows.append(validation_row)
        print(json.dumps(validation_row))
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            _atomic_torch(
                best_path,
                {
                    "contract_hash": expected_hash,
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "valid_metrics": valid_metrics,
                    "normalization": normalization.to_dict(),
                },
            )
        next_date_index = 0
        _atomic_torch(
            checkpoint_path,
            {
                "contract_hash": expected_hash,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "next_date_index": 0,
                "best_valid_loss": best_valid_loss,
                "history_rows": history_rows,
            },
        )
        _atomic_parquet(pd.DataFrame(history_rows), output_root / "training_history.parquet")

    if not best_path.exists():
        raise RuntimeError("Training completed without producing a best checkpoint")
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model"])
    test_metrics, test_predictions = _evaluate_dates(
        model,
        factory,
        segments["test"],
        normalization,
        device=device,
        batch_size=batch_size,
        output_dir=output_root / "test",
    )
    _atomic_parquet(test_predictions, output_root / "test" / "predictions_all.parquet")
    final = {
        "status": "complete",
        "contract_hash": expected_hash,
        "best_epoch": int(best["epoch"]),
        "best_valid_metrics": best["valid_metrics"],
        "test_metrics": test_metrics,
        "segments": segments,
        "device": str(device),
        "amp": use_amp,
    }
    _atomic_json(output_root / "summary.json", final)
    report = [
        "# NFF-only generic stock-day baseline",
        "",
        "- Model: shared GRU, no ticker embedding, no daily adapter.",
        "- Inputs: NFF features only.",
        "- GFF/GAL: not read.",
        f"- Best epoch: {final['best_epoch']}",
        f"- Test MSE: {test_metrics.get('mse')}",
        f"- Test MAE: {test_metrics.get('mae')}",
        f"- Test mean RankIC: {test_metrics.get('rank_ic_mean')}",
        f"- Test RankIC observations: {test_metrics.get('rank_ic_count')}",
        "",
        "The split is strictly ordered by complete trading dates. This is the P2 B0 baseline against which a later daily adapter must be compared.",
    ]
    (output_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, default=str))
    return 0


def command_evaluate(config, manifest_path, output_root):
    manifest = _manifest(config, manifest_path)
    segments = _split_dates(manifest["trade_date"].astype(str).unique(), config)
    factory = _factory(config)
    normalization = NormalizationState.from_dict(json.loads((output_root / "normalization.json").read_text(encoding="utf-8")))
    training = dict(config.get("training") or {})
    device = _resolve_device(str(training.get("device", "auto")))
    model = NFFGenericGRU(
        len(normalization.feature_names),
        len(normalization.target_names),
        GenericGRUConfig(**dict(config.get("model") or {})),
    ).to(device)
    checkpoint = torch.load(output_root / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    metrics, predictions = _evaluate_dates(
        model,
        factory,
        segments["test"],
        normalization,
        device=device,
        batch_size=int(training.get("batch_size", 256)),
        output_dir=output_root / "test",
    )
    _atomic_parquet(predictions, output_root / "test" / "predictions_all.parquet")
    print(json.dumps(metrics, indent=2, default=str))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output-root")
    parser.add_argument("--force-normalization", action="store_true")
    args = parser.parse_args()
    config = _read_config(Path(args.config).expanduser().resolve())
    output_root = Path(
        args.output_root or config.get("output", {}).get("model_run_root", "model_runs/nff_generic_pilot")
    ).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None
    if args.command == "train":
        return command_train(config, manifest_path, output_root, args.force_normalization)
    return command_evaluate(config, manifest_path, output_root)


if __name__ == "__main__":
    raise SystemExit(main())
