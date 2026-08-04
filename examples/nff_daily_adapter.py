from __future__ import annotations

"""Train and audit the NFF-only P3 daily context adapter."""

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlib.contrib.data.nff_episode import NFFStockDayEpisodeFactory, contract_hash
from qlib.contrib.model.nff_daily_adapter import (
    AdapterArrays,
    DailyAdapterConfig,
    NFFDailyContextAdapter,
    episodes_to_adapter_arrays,
    masked_episode_mse,
    paired_prediction_frame,
    previous_day_support,
    same_day_shuffled_support,
    update_previous_support_bank,
)
from qlib.contrib.model.nff_generic import GenericGRUConfig, NFFGenericGRU, NormalizationState, prediction_metrics


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
    path = explicit or Path(config["output"]["episode_run_root"]) / "aggregate" / "episode_manifest.parquet"
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Episode manifest not found: {path}. Run P1 first.")
    frame = pd.read_parquet(path)
    if frame.empty:
        raise RuntimeError("Episode manifest is empty")
    return frame


def _split_dates(dates: Sequence[str], config: Mapping[str, Any]) -> Dict[str, list[str]]:
    ordered = sorted(set(str(value) for value in dates))
    split_config = dict(config.get("segments") or {})
    if all(name in split_config for name in ("train", "valid", "test")):
        result = {
            name: [value for value in ordered if str(split_config[name][0]) <= value <= str(split_config[name][1])]
            for name in ("train", "valid", "test")
        }
    else:
        ratios = split_config.get("ratios", [0.70, 0.15, 0.15])
        if len(ratios) != 3 or not np.isclose(sum(float(value) for value in ratios), 1.0):
            raise ValueError("segments.ratios must contain three values summing to 1")
        n = len(ordered)
        train_end = max(1, int(np.floor(n * float(ratios[0]))))
        valid_end = max(train_end + 1, int(np.floor(n * (float(ratios[0]) + float(ratios[1])))))
        valid_end = min(valid_end, n - 1)
        result = {"train": ordered[:train_end], "valid": ordered[train_end:valid_end], "test": ordered[valid_end:]}
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
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _load_baseline(
    config: Mapping[str, Any],
    normalization: NormalizationState,
    device: torch.device,
) -> Tuple[NFFGenericGRU, Dict[str, Any], Path]:
    baseline_root = Path(config["output"]["baseline_model_root"]).expanduser().resolve()
    checkpoint_path = baseline_root / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"P2 checkpoint not found: {checkpoint_path}")
    base = NFFGenericGRU(
        len(normalization.feature_names),
        len(normalization.target_names),
        GenericGRUConfig(**dict(config.get("model") or {})),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base.load_state_dict(checkpoint["model"])
    return base, checkpoint, checkpoint_path


def _normalization(config: Mapping[str, Any]) -> NormalizationState:
    path = Path(config["output"]["baseline_model_root"]).expanduser().resolve() / "normalization.json"
    if not path.exists():
        raise FileNotFoundError(f"P2 normalization not found: {path}")
    return NormalizationState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _date_arrays(
    factory: NFFStockDayEpisodeFactory,
    trade_date: str,
    normalization: NormalizationState,
) -> AdapterArrays:
    result = factory.load_date(trade_date)
    return episodes_to_adapter_arrays(
        result.episodes,
        normalization,
        episode_contract_hash=factory.contract_hash,
    )


def _train_one_date(
    model: NFFDailyContextAdapter,
    optimizer: torch.optim.Optimizer,
    arrays: AdapterArrays,
    *,
    device: torch.device,
    episode_batch_size: int,
    support_length: int,
    grad_clip: float,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
) -> Dict[str, Any]:
    if arrays.episode_count == 0:
        return {"episodes": 0, "queries": 0, "loss": None}
    dataset = TensorDataset(
        torch.from_numpy(arrays.support_x),
        torch.from_numpy(arrays.query_x),
        torch.from_numpy(arrays.targets),
        torch.from_numpy(arrays.target_observed),
        torch.from_numpy(arrays.query_valid),
    )
    loader = DataLoader(dataset, batch_size=episode_batch_size, shuffle=True, drop_last=False)
    model.train()
    losses = []
    for support_x, query_x, target, target_observed, query_valid in loader:
        support_x = support_x[:, :support_length].to(device=device, dtype=torch.float32, non_blocking=True)
        query_x = query_x.to(device=device, dtype=torch.float32, non_blocking=True)
        target = target.to(device=device, dtype=torch.float32, non_blocking=True)
        target_observed = target_observed.to(device=device, dtype=torch.bool, non_blocking=True)
        query_valid = query_valid.to(device=device, dtype=torch.bool, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction, _ = model(support_x, query_x, query_valid=query_valid)
            loss = masked_episode_mse(prediction, target, target_observed, query_valid)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return {
        "episodes": arrays.episode_count,
        "queries": int(arrays.query_valid.sum()),
        "loss": float(np.mean(losses)) if losses else None,
    }


def _predict_arrays(
    model: NFFDailyContextAdapter,
    arrays: AdapterArrays,
    *,
    support_x: np.ndarray,
    support_length: int,
    device: torch.device,
    episode_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_count = model.base_model.target_count
    shape = (arrays.episode_count, arrays.query_slots, target_count)
    if arrays.episode_count == 0:
        return (
            np.empty(shape, dtype="float32"),
            np.empty(shape, dtype="float32"),
            np.empty((arrays.episode_count, arrays.query_slots, 1), dtype="float32"),
        )
    dataset = TensorDataset(
        torch.from_numpy(support_x),
        torch.from_numpy(arrays.query_x),
        torch.from_numpy(arrays.query_valid),
    )
    loader = DataLoader(dataset, batch_size=episode_batch_size, shuffle=False, drop_last=False)
    adapter_parts = []
    baseline_parts = []
    gate_parts = []
    model.eval()
    with torch.no_grad():
        for support_batch, query_batch, valid_batch in loader:
            support_batch = support_batch[:, :support_length].to(device=device, dtype=torch.float32, non_blocking=True)
            query_batch = query_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            valid_batch = valid_batch.to(device=device, dtype=torch.bool, non_blocking=True)
            adapter_prediction, gate = model(support_batch, query_batch, query_valid=valid_batch)
            baseline_prediction = model.baseline_forward(query_batch, query_valid=valid_batch)
            adapter_parts.append(adapter_prediction.cpu().numpy())
            baseline_parts.append(baseline_prediction.cpu().numpy())
            gate_parts.append(gate.cpu().numpy())
    return np.concatenate(adapter_parts), np.concatenate(baseline_parts), np.concatenate(gate_parts)


def _frame_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        empty = prediction_metrics(pd.DataFrame())
        return {"adapter": empty, "baseline": empty, "delta_mse": None, "delta_mae": None, "delta_rank_ic": None}
    adapter = prediction_metrics(frame.rename(columns={"prediction_adapter": "prediction"}))
    baseline = prediction_metrics(frame.rename(columns={"prediction_b0": "prediction"}))
    return {
        "adapter": adapter,
        "baseline": baseline,
        "delta_mse": None if adapter.get("mse") is None else float(adapter["mse"] - baseline["mse"]),
        "delta_mae": None if adapter.get("mae") is None else float(adapter["mae"] - baseline["mae"]),
        "delta_rank_ic": (
            None
            if adapter.get("rank_ic_mean") is None or baseline.get("rank_ic_mean") is None
            else float(adapter["rank_ic_mean"] - baseline["rank_ic_mean"])
        ),
    }


def _flatten_metrics(row: Mapping[str, Any]) -> Dict[str, Any]:
    item = {key: value for key, value in row.items() if key not in {"adapter", "baseline"}}
    for prefix in ("adapter", "baseline"):
        for key, value in row[prefix].items():
            item[f"{prefix}_{key}"] = value
    return item


def _group_metrics(frame: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    rows = []
    for keys, block in frame.groupby(list(groups), dropna=False, sort=True):
        values = keys if isinstance(keys, tuple) else (keys,)
        rows.append(_flatten_metrics({**dict(zip(groups, values)), **_frame_metrics(block)}))
    return pd.DataFrame(rows)


def _paired_date_bootstrap(daily: pd.DataFrame, seed: int, draws: int = 1000) -> Dict[str, Any]:
    if daily.empty or "delta_rank_ic" not in daily:
        return {"draws": 0}
    values = pd.to_numeric(daily["delta_rank_ic"], errors="coerce").dropna().to_numpy(dtype="float64")
    if values.size == 0:
        return {"draws": 0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return {
        "draws": draws,
        "dates": int(len(values)),
        "mean_delta_rank_ic": float(values.mean()),
        "ci_2_5": float(np.quantile(samples, 0.025)),
        "ci_97_5": float(np.quantile(samples, 0.975)),
        "probability_positive": float((samples > 0).mean()),
    }


def _evaluate_primary_and_ablations(
    model: NFFDailyContextAdapter,
    factory: NFFStockDayEpisodeFactory,
    dates: Sequence[str],
    normalization: NormalizationState,
    *,
    device: torch.device,
    episode_batch_size: int,
    output_root: Path,
    support_lengths: Sequence[int],
    support_modes: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    primary_frames = []
    ablation_rows = []
    gate_rows = []
    previous_bank: MutableMapping[str, np.ndarray] = {}
    default_length = max(int(value) for value in support_lengths)

    for trade_date in dates:
        arrays = _date_arrays(factory, trade_date, normalization)
        if arrays.episode_count == 0:
            continue
        same_support = arrays.support_x
        shuffled_support, shuffled_sources = same_day_shuffled_support(same_support, arrays.symbols)
        previous_support, previous_available = previous_day_support(same_support, arrays.symbols, previous_bank)
        support_by_mode = {
            "same_stock_same_day": same_support,
            "zero_support": np.zeros_like(same_support),
            "same_day_shuffled_stock": shuffled_support,
            "same_stock_previous_day": previous_support,
        }
        for mode in support_modes:
            if mode not in support_by_mode:
                raise ValueError(f"Unsupported support mode: {mode}")
            adapter_pred, baseline_pred, gate = _predict_arrays(
                model,
                arrays,
                support_x=support_by_mode[mode],
                support_length=default_length,
                device=device,
                episode_batch_size=episode_batch_size,
            )
            frame = paired_prediction_frame(
                adapter_pred,
                baseline_pred,
                arrays.targets,
                arrays.target_observed,
                arrays.query_valid,
                arrays.metadata,
                normalization,
                support_mode=mode,
                support_length=default_length,
                gate=gate,
            )
            if mode == "same_day_shuffled_stock" and not frame.empty:
                source_map = {index: shuffled_sources[index] for index in range(len(shuffled_sources))}
                frame["support_source_symbol"] = frame["episode_index"].astype(int).map(source_map)
            if mode == "same_stock_previous_day" and not frame.empty:
                availability_map = {index: bool(previous_available[index]) for index in range(len(previous_available))}
                frame["previous_support_available"] = frame["episode_index"].astype(int).map(availability_map)
            metrics = _frame_metrics(frame)
            ablation_rows.append(
                _flatten_metrics(
                    {
                        "trade_date": trade_date,
                        "support_mode": mode,
                        "support_length": default_length,
                        **metrics,
                    }
                )
            )
            if mode == "same_stock_same_day":
                primary_frames.append(frame)
                _atomic_parquet(frame, output_root / "test" / "predictions" / f"date={trade_date}.parquet")
                valid_gate = gate[arrays.query_valid]
                gate_rows.append(
                    {
                        "trade_date": trade_date,
                        "count": int(valid_gate.size),
                        "mean": float(valid_gate.mean()) if valid_gate.size else None,
                        "p10": float(np.quantile(valid_gate, 0.10)) if valid_gate.size else None,
                        "p50": float(np.quantile(valid_gate, 0.50)) if valid_gate.size else None,
                        "p90": float(np.quantile(valid_gate, 0.90)) if valid_gate.size else None,
                    }
                )

        for length in support_lengths:
            if int(length) == default_length:
                continue
            adapter_pred, baseline_pred, gate = _predict_arrays(
                model,
                arrays,
                support_x=same_support,
                support_length=int(length),
                device=device,
                episode_batch_size=episode_batch_size,
            )
            frame = paired_prediction_frame(
                adapter_pred,
                baseline_pred,
                arrays.targets,
                arrays.target_observed,
                arrays.query_valid,
                arrays.metadata,
                normalization,
                support_mode="same_stock_same_day",
                support_length=int(length),
                gate=gate,
            )
            ablation_rows.append(
                _flatten_metrics(
                    {
                        "trade_date": trade_date,
                        "support_mode": "same_stock_same_day",
                        "support_length": int(length),
                        **_frame_metrics(frame),
                    }
                )
            )
        update_previous_support_bank(previous_bank, same_support, arrays.symbols)

    primary = pd.concat(primary_frames, ignore_index=True) if primary_frames else pd.DataFrame()
    if primary.empty:
        raise RuntimeError("No P3 test predictions were produced")
    key_columns = ["sample_id", "target_name"]
    parity = {
        "rows": int(len(primary)),
        "unique_keys": int(primary[key_columns].drop_duplicates().shape[0]),
        "duplicate_keys": int(primary.duplicated(key_columns).sum()),
        "adapter_missing": int(primary["prediction_adapter"].isna().sum()),
        "baseline_missing": int(primary["prediction_b0"].isna().sum()),
        "sample_parity": bool(
            not primary.duplicated(key_columns).any()
            and primary["prediction_adapter"].notna().all()
            and primary["prediction_b0"].notna().all()
        ),
    }
    if not parity["sample_parity"]:
        raise AssertionError(f"P2/P3 sample parity failed: {parity}")

    daily = _group_metrics(primary, ["trade_date"])
    by_target = _group_metrics(primary, ["target_name"])
    by_query = _group_metrics(primary, ["query_index", "target_name"])
    by_overlap = _group_metrics(primary, ["support_query_overlap_minutes", "target_name"])
    ablation_frame = pd.DataFrame(ablation_rows)

    _atomic_parquet(primary, output_root / "test" / "predictions_b0_vs_adapter.parquet")
    _atomic_parquet(daily, output_root / "test" / "daily_metrics.parquet")
    _atomic_parquet(by_target, output_root / "test" / "target_metrics.parquet")
    _atomic_parquet(by_query, output_root / "test" / "query_time_metrics.parquet")
    _atomic_parquet(by_overlap, output_root / "test" / "support_overlap_metrics.parquet")
    _atomic_parquet(ablation_frame, output_root / "test" / "support_ablation_metrics.parquet")
    _atomic_parquet(pd.DataFrame(gate_rows), output_root / "test" / "adaptation_gate_distribution.parquet")
    _atomic_json(output_root / "test" / "sample_parity_audit.json", parity)
    bootstrap = _paired_date_bootstrap(daily, seed)
    _atomic_json(output_root / "test" / "paired_date_bootstrap.json", bootstrap)
    summary = {
        "primary": _frame_metrics(primary),
        "sample_parity": parity,
        "paired_date_bootstrap": bootstrap,
        "test_dates": len(dates),
    }
    _atomic_json(output_root / "test" / "summary.json", summary)
    return summary


def _train(
    config: Mapping[str, Any],
    manifest_path: Optional[Path],
    output_root: Path,
    *,
    joint: bool,
) -> int:
    run = dict(config.get("run") or {})
    seed = int(run.get("seed", 20260804))
    torch.set_num_threads(int(run.get("torch_threads", 4)))
    _seed_everything(seed)
    manifest = _manifest(config, manifest_path)
    segments = _split_dates(manifest["trade_date"].astype(str).unique(), config)
    factory = _factory(config)
    normalization = _normalization(config)
    training = dict(config.get("adapter_training") or {})
    device = _resolve_device(str(training.get("device", "auto")))
    base_model, baseline_checkpoint, baseline_path = _load_baseline(config, normalization, device)
    model = NFFDailyContextAdapter(
        base_model,
        DailyAdapterConfig(**dict(config.get("adapter_model") or {})),
        freeze_base=True,
    ).to(device)

    stage_name = "joint" if joint else "adapter"
    if joint:
        adapter_best = output_root / "checkpoint_adapter_best.pt"
        if not adapter_best.exists():
            raise FileNotFoundError("Run adapter-only training before train-joint")
        model.load_state_dict(torch.load(adapter_best, map_location=device)["model"])
        model.set_joint_trainable()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable P3 parameters")
    learning_rate = float(training.get("joint_learning_rate" if joint else "learning_rate", 1e-3))
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    epochs = int(training.get("joint_epochs" if joint else "epochs", 5))
    batch_size = int(training.get("episode_batch_size", 32))
    support_length = int(training.get("support_length", 30))
    grad_clip = float(training.get("grad_clip", 1.0))
    output_root.mkdir(parents=True, exist_ok=True)

    train_contract = {
        "version": f"nff_daily_context_{stage_name}_v1",
        "episode_contract_hash": factory.contract_hash,
        "baseline_checkpoint_contract": baseline_checkpoint.get("contract_hash"),
        "baseline_checkpoint_path": str(baseline_path),
        "normalization": normalization.to_dict(),
        "model": model.model_contract(),
        "segments": segments,
        "training": training,
        "stage": stage_name,
        "seed": seed,
    }
    expected_hash = contract_hash(train_contract)
    _atomic_json(output_root / f"{stage_name}_contract.json", {"contract_hash": expected_hash, **train_contract})
    checkpoint_path = output_root / f"checkpoint_{stage_name}_last.pt"
    best_path = output_root / f"checkpoint_{stage_name}_best.pt"
    start_epoch = 0
    next_date_index = 0
    best_valid_loss = float("inf")
    history_rows = []
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("contract_hash") != expected_hash:
            raise RuntimeError("Existing P3 checkpoint has a different contract")
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
            metrics = _train_one_date(
                model,
                optimizer,
                _date_arrays(factory, trade_date, normalization),
                device=device,
                episode_batch_size=batch_size,
                support_length=support_length,
                grad_clip=grad_clip,
                scaler=scaler,
                use_amp=use_amp,
            )
            row = {
                "stage": stage_name,
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

        valid_losses = []
        model.eval()
        for trade_date in segments["valid"]:
            arrays = _date_arrays(factory, trade_date, normalization)
            if arrays.episode_count == 0:
                continue
            prediction, _, _ = _predict_arrays(
                model,
                arrays,
                support_x=arrays.support_x,
                support_length=support_length,
                device=device,
                episode_batch_size=batch_size,
            )
            mask = arrays.target_observed & arrays.query_valid[:, :, None]
            if mask.any():
                valid_losses.append(float(np.square(prediction - arrays.targets)[mask].mean()))
        valid_loss = float(np.mean(valid_losses)) if valid_losses else float("inf")
        history_rows.append({"stage": stage_name, "epoch": epoch, "segment": "valid", "loss": valid_loss})
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            _atomic_torch(
                best_path,
                {
                    "contract_hash": expected_hash,
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "valid_loss": valid_loss,
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
        _atomic_parquet(pd.DataFrame(history_rows), output_root / f"training_history_{stage_name}.parquet")

    if not best_path.exists():
        raise RuntimeError(f"{stage_name} training did not produce a best checkpoint")
    print(json.dumps({"status": "complete", "stage": stage_name, "best_valid_loss": best_valid_loss}, indent=2))
    return 0


def _load_adapter_for_eval(config, output_root, device, normalization, *, joint=False):
    base_model, _, _ = _load_baseline(config, normalization, device)
    model = NFFDailyContextAdapter(
        base_model,
        DailyAdapterConfig(**dict(config.get("adapter_model") or {})),
        freeze_base=True,
    ).to(device)
    stage = "joint" if joint else "adapter"
    path = output_root / f"checkpoint_{stage}_best.pt"
    if not path.exists():
        raise FileNotFoundError(f"P3 checkpoint not found: {path}")
    model.load_state_dict(torch.load(path, map_location=device)["model"])
    return model


def _evaluate(config, manifest_path, output_root, *, joint=False):
    manifest = _manifest(config, manifest_path)
    segments = _split_dates(manifest["trade_date"].astype(str).unique(), config)
    factory = _factory(config)
    normalization = _normalization(config)
    training = dict(config.get("adapter_training") or {})
    device = _resolve_device(str(training.get("device", "auto")))
    model = _load_adapter_for_eval(config, output_root, device, normalization, joint=joint)
    ablation = dict(config.get("ablations") or {})
    summary = _evaluate_primary_and_ablations(
        model,
        factory,
        segments["test"],
        normalization,
        device=device,
        episode_batch_size=int(training.get("episode_batch_size", 32)),
        output_root=output_root,
        support_lengths=[int(value) for value in ablation.get("support_lengths", [5, 15, 30])],
        support_modes=[
            str(value)
            for value in ablation.get(
                "support_modes",
                ["same_stock_same_day", "same_day_shuffled_stock", "same_stock_previous_day", "zero_support"],
            )
        ],
        seed=int(config.get("run", {}).get("seed", 20260804)),
    )
    report = [
        "# NFF-only daily context adapter",
        "",
        "- Query backbone: frozen P2 shared GRU for the primary adapter-only experiment.",
        "- Support: fixed early-session NFF only; no GFF, GAL, ticker ID, or intraday labels.",
        f"- Paired sample parity: {summary['sample_parity']['sample_parity']}",
        f"- Delta MSE (adapter - B0): {summary['primary'].get('delta_mse')}",
        f"- Delta MAE (adapter - B0): {summary['primary'].get('delta_mae')}",
        f"- Delta mean RankIC: {summary['primary'].get('delta_rank_ic')}",
        f"- Date-bootstrap P(delta RankIC > 0): {summary['paired_date_bootstrap'].get('probability_positive')}",
        "",
        "Interpret same-stock support against shuffled-stock, previous-day and zero-support counterfactuals before treating any gain as stock-day personalization.",
    ]
    (output_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _atomic_json(output_root / "summary.json", {"status": "complete", **summary})
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _export_paired_baseline(config, manifest_path, output_root):
    manifest = _manifest(config, manifest_path)
    segments = _split_dates(manifest["trade_date"].astype(str).unique(), config)
    factory = _factory(config)
    normalization = _normalization(config)
    training = dict(config.get("adapter_training") or {})
    device = _resolve_device(str(training.get("device", "auto")))
    base_model, _, _ = _load_baseline(config, normalization, device)
    model = NFFDailyContextAdapter(
        base_model,
        DailyAdapterConfig(**dict(config.get("adapter_model") or {})),
        freeze_base=True,
    ).to(device)
    rows = []
    for trade_date in segments["test"]:
        arrays = _date_arrays(factory, trade_date, normalization)
        if arrays.episode_count == 0:
            continue
        _, baseline, _ = _predict_arrays(
            model,
            arrays,
            support_x=np.zeros_like(arrays.support_x),
            support_length=1,
            device=device,
            episode_batch_size=int(training.get("episode_batch_size", 32)),
        )
        frame = paired_prediction_frame(
            baseline,
            baseline,
            arrays.targets,
            arrays.target_observed,
            arrays.query_valid,
            arrays.metadata,
            normalization,
            support_mode="baseline_only",
            support_length=0,
        ).rename(columns={"prediction_b0": "prediction"})
        keep = [
            "sample_id",
            "episode_id",
            "symbol",
            "trade_date",
            "query_time",
            "target_name",
            "prediction",
            "target",
            "support_query_overlap_minutes",
        ]
        rows.append(frame[keep])
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if combined.empty:
        raise RuntimeError("No paired B0 predictions produced")
    if combined.duplicated(["sample_id", "target_name"]).any():
        raise AssertionError("Paired B0 keys are not unique")
    _atomic_parquet(combined, output_root / "paired_baseline_predictions.parquet")
    contract = {
        "episode_contract_hash": factory.contract_hash,
        "rows": int(len(combined)),
        "unique_samples": int(combined["sample_id"].nunique()),
        "test_dates": segments["test"],
    }
    _atomic_json(output_root / "paired_sample_contract.json", contract)
    print(json.dumps(contract, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["export-paired-baseline", "train", "train-joint", "evaluate", "evaluate-joint"],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config = _read_config(Path(args.config).expanduser().resolve())
    output_root = Path(args.output_root or config["output"]["adapter_run_root"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None
    if args.command == "export-paired-baseline":
        return _export_paired_baseline(config, manifest_path, output_root)
    if args.command == "train":
        return _train(config, manifest_path, output_root, joint=False)
    if args.command == "train-joint":
        return _train(config, manifest_path, output_root, joint=True)
    return _evaluate(config, manifest_path, output_root, joint=args.command == "evaluate-joint")


if __name__ == "__main__":
    raise SystemExit(main())
