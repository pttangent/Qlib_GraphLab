from __future__ import annotations

"""Train and evaluate NFF-only context decomposition variants B2-B5."""

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qlib.contrib.data.nff_context import ContextCacheConfig, NFFContextCacheStore, context_cache_contract
from qlib.contrib.data.nff_episode import NFFStockDayEpisodeFactory, contract_hash
from qlib.contrib.model.nff_context_decomposition import (
    ContextArrays,
    ContextDecompositionConfig,
    NFFContextDecompositionModel,
    episodes_to_context_arrays,
    masked_context_mse,
    shuffled_identity,
    shuffled_market_time,
    zero_identity,
    zero_market,
)
from qlib.contrib.model.nff_daily_adapter import (
    previous_day_support,
    same_day_shuffled_support,
    update_previous_support_bank,
)
from qlib.contrib.model.nff_generic import (
    GenericGRUConfig,
    NFFGenericGRU,
    NormalizationState,
    inverse_targets,
    prediction_metrics,
)

_VARIANTS = ("identity", "market", "combined", "full")


def _read_config(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Config must contain a YAML mapping")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(value), indent=2, sort_keys=True, default=str), encoding="utf-8")
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
    return NFFStockDayEpisodeFactory(**dict(config.get("episode_factory") or {}))


def _manifest(config: Mapping[str, Any], explicit: Optional[Path]) -> pd.DataFrame:
    path = explicit or Path(config.get("output", {}).get("episode_run_root", "episode_runs/pilot")) / "aggregate" / "episode_manifest.parquet"
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"P1 episode manifest not found: {path}")
    frame = pd.read_parquet(path)
    if frame.empty:
        raise RuntimeError("Episode manifest is empty")
    return frame


def _split_dates(dates: Sequence[str], config: Mapping[str, Any]) -> Dict[str, list[str]]:
    ordered = sorted(set(str(value) for value in dates))
    split_config = dict(config.get("segments") or {})
    if all(name in split_config for name in ("train", "valid", "test")):
        result = {
            name: [date for date in ordered if str(split_config[name][0]) <= date <= str(split_config[name][1])]
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
        result = {
            "train": ordered[:train_end],
            "valid": ordered[train_end:valid_end],
            "test": ordered[valid_end:],
        }
    if any(not result[name] for name in result):
        raise ValueError(f"Temporal split produced an empty segment: {result}")
    if max(result["train"]) >= min(result["valid"]) or max(result["valid"]) >= min(result["test"]):
        raise AssertionError("Segments are not strictly ordered")
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


def _generic_root(config: Mapping[str, Any]) -> Path:
    return Path(config.get("output", {}).get("generic_model_root", "model_runs/nff_generic_pilot")).expanduser().resolve()


def _context_root(config: Mapping[str, Any]) -> Path:
    return Path(config.get("output", {}).get("context_cache_root", "context_cache/pilot")).expanduser().resolve()


def _model_root(config: Mapping[str, Any], variant: str, explicit: Optional[Path]) -> Path:
    base = explicit or Path(config.get("output", {}).get("context_model_root", "model_runs/context_decomposition_v1"))
    return base.expanduser().resolve() / f"variant={variant}"


def _load_normalization(config: Mapping[str, Any]) -> NormalizationState:
    path = _generic_root(config) / "normalization.json"
    if not path.exists():
        raise FileNotFoundError(f"P2 normalization not found: {path}")
    return NormalizationState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _load_base(config: Mapping[str, Any], normalization: NormalizationState, device: torch.device) -> NFFGenericGRU:
    base = NFFGenericGRU(
        len(normalization.feature_names),
        len(normalization.target_names),
        GenericGRUConfig(**dict(config.get("model") or {})),
    ).to(device)
    checkpoint = torch.load(_generic_root(config) / "checkpoint_best.pt", map_location=device)
    base.load_state_dict(checkpoint["model"])
    base.eval()
    return base


def _store(config: Mapping[str, Any], factory, normalization) -> NFFContextCacheStore:
    cache_config = ContextCacheConfig(**dict(config.get("context_cache") or {}))
    contract = context_cache_contract(
        episode_contract_hash=factory.contract_hash,
        normalization=normalization,
        config=cache_config,
    )
    store = NFFContextCacheStore(_context_root(config), contract)
    store.initialize()
    return store


def _date_arrays(factory, store, identity_frame, trade_date, normalization, history_days) -> ContextArrays:
    result = factory.load_date(trade_date)
    return episodes_to_context_arrays(
        result.episodes,
        normalization,
        episode_contract_hash=factory.contract_hash,
        identity_history=identity_frame,
        market_context=store.read_market(trade_date),
        history_days=history_days,
    )


def _tensor_dataset(arrays: ContextArrays) -> TensorDataset:
    a = arrays.adapter
    return TensorDataset(
        torch.from_numpy(a.support_x),
        torch.from_numpy(a.query_x),
        torch.from_numpy(arrays.identity_x),
        torch.from_numpy(arrays.identity_valid),
        torch.from_numpy(arrays.market_x),
        torch.from_numpy(arrays.market_valid),
        torch.from_numpy(a.targets),
        torch.from_numpy(a.target_observed),
        torch.from_numpy(a.query_valid),
    )


def _move(batch, device):
    support, query, identity, identity_valid, market, market_valid, targets, target_observed, query_valid = batch
    return (
        support.to(device, torch.float32, non_blocking=True),
        query.to(device, torch.float32, non_blocking=True),
        identity.to(device, torch.float32, non_blocking=True),
        identity_valid.to(device, torch.bool, non_blocking=True),
        market.to(device, torch.float32, non_blocking=True),
        market_valid.to(device, torch.bool, non_blocking=True),
        targets.to(device, torch.float32, non_blocking=True),
        target_observed.to(device, torch.bool, non_blocking=True),
        query_valid.to(device, torch.bool, non_blocking=True),
    )


def _train_date(model, optimizer, arrays, *, device, batch_size, grad_clip, use_amp, scaler):
    if arrays.episode_count == 0:
        return {"episodes": 0, "queries": 0, "loss": None}
    loader = DataLoader(_tensor_dataset(arrays), batch_size=batch_size, shuffle=True, drop_last=False)
    model.train()
    losses = []
    for batch in loader:
        support, query, identity, identity_valid, market, market_valid, targets, target_observed, query_valid = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction, _ = model(
                support,
                query,
                identity,
                identity_valid,
                market,
                market_valid,
                query_valid=query_valid,
            )
            loss = masked_context_mse(prediction, targets, target_observed, query_valid)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return {
        "episodes": arrays.episode_count,
        "queries": int(arrays.adapter.query_valid.sum()),
        "loss": float(np.mean(losses)) if losses else None,
    }


def _prediction_frame(pred_std, base_std, gates, arrays, normalization):
    pred = inverse_targets(pred_std, normalization)
    base = inverse_targets(base_std, normalization)
    target = inverse_targets(arrays.adapter.targets, normalization)
    rows = []
    for meta in arrays.adapter.metadata.itertuples(index=False):
        episode_index, query_index = int(meta.episode_index), int(meta.query_index)
        for target_index, target_name in enumerate(normalization.target_names):
            if not arrays.adapter.target_observed[episode_index, query_index, target_index]:
                continue
            row = {
                "sample_id": meta.sample_id,
                "episode_id": meta.episode_id,
                "symbol": meta.symbol,
                "trade_date": meta.trade_date,
                "query_time": meta.query_time,
                "query_index": query_index,
                "support_query_overlap_minutes": meta.support_query_overlap_minutes,
                "target_name": target_name,
                "target": float(target[episode_index, query_index, target_index]),
                "prediction": float(pred[episode_index, query_index, target_index]),
                "prediction_b0": float(base[episode_index, query_index, target_index]),
            }
            for name, value in gates.items():
                row[f"gate_{name}"] = float(value[episode_index, query_index, 0])
            rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty and frame[["sample_id", "target_name"]].duplicated().any():
        raise AssertionError("Prediction frame contains duplicate paired keys")
    return frame


def _evaluate_arrays(model, arrays, normalization, *, device, batch_size):
    if arrays.episode_count == 0:
        return pd.DataFrame()
    loader = DataLoader(_tensor_dataset(arrays), batch_size=batch_size, shuffle=False)
    predictions: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    gate_parts: Dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            support, query, identity, identity_valid, market, market_valid, _, _, query_valid = _move(batch, device)
            prediction, gates = model(
                support,
                query,
                identity,
                identity_valid,
                market,
                market_valid,
                query_valid=query_valid,
            )
            baseline = model.baseline_forward(query, query_valid=query_valid)
            predictions.append(prediction.cpu().numpy())
            baselines.append(baseline.cpu().numpy())
            for name, value in gates.items():
                gate_parts.setdefault(name, []).append(value.cpu().numpy())
    gate_arrays = {name: np.concatenate(parts, axis=0) for name, parts in gate_parts.items()}
    return _prediction_frame(
        np.concatenate(predictions, axis=0),
        np.concatenate(baselines, axis=0),
        gate_arrays,
        arrays,
        normalization,
    )


def _paired_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {"rows": 0}
    adapted = prediction_metrics(frame[["trade_date", "query_time", "target_name", "prediction", "target"]])
    baseline_frame = frame[["trade_date", "query_time", "target_name", "prediction_b0", "target"]].rename(
        columns={"prediction_b0": "prediction"}
    )
    baseline = prediction_metrics(baseline_frame)
    return {
        "rows": int(len(frame)),
        "mse": adapted.get("mse"),
        "mae": adapted.get("mae"),
        "rank_ic_mean": adapted.get("rank_ic_mean"),
        "baseline_mse": baseline.get("mse"),
        "baseline_mae": baseline.get("mae"),
        "baseline_rank_ic_mean": baseline.get("rank_ic_mean"),
        "delta_mse": None if adapted.get("mse") is None else float(adapted["mse"] - baseline["mse"]),
        "delta_mae": None if adapted.get("mae") is None else float(adapted["mae"] - baseline["mae"]),
        "delta_rank_ic": None if adapted.get("rank_ic_mean") is None or baseline.get("rank_ic_mean") is None else float(adapted["rank_ic_mean"] - baseline["rank_ic_mean"]),
    }


def _bootstrap_by_date(frame: pd.DataFrame, *, seed: int, draws: int) -> Dict[str, Any]:
    if frame.empty:
        return {"draws": 0}
    daily = frame.assign(
        adapted_sq=np.square(frame["prediction"] - frame["target"]),
        baseline_sq=np.square(frame["prediction_b0"] - frame["target"]),
    ).groupby("trade_date", sort=True)[["adapted_sq", "baseline_sq"]].mean()
    values = (daily["adapted_sq"] - daily["baseline_sq"]).to_numpy(dtype="float64")
    rng = np.random.default_rng(seed)
    samples = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(draws)])
    return {
        "draws": int(draws),
        "dates": int(len(values)),
        "delta_mse_mean": float(values.mean()),
        "probability_adapter_better": float(np.mean(samples < 0.0)),
        "ci_2_5": float(np.quantile(samples, 0.025)),
        "ci_97_5": float(np.quantile(samples, 0.975)),
    }


def _clone_arrays(arrays: ContextArrays, *, support=None, identity=None, identity_valid=None, market=None, market_valid=None):
    from dataclasses import replace

    adapter = replace(arrays.adapter, support_x=arrays.adapter.support_x if support is None else support)
    return ContextArrays(
        adapter=adapter,
        identity_x=arrays.identity_x if identity is None else identity,
        identity_valid=arrays.identity_valid if identity_valid is None else identity_valid,
        market_x=arrays.market_x if market is None else market,
        market_valid=arrays.market_valid if market_valid is None else market_valid,
        identity_columns=arrays.identity_columns,
        market_columns=arrays.market_columns,
    )


def _counterfactual_frames(model, arrays, normalization, *, device, batch_size, previous_market=None, previous_support_bank=None):
    modes = {"same_context": arrays}
    if model.uses_identity:
        shuffled_x, shuffled_valid = shuffled_identity(arrays.identity_x, arrays.identity_valid)
        zero_x, zero_valid = zero_identity(arrays.identity_x, arrays.identity_valid)
        modes["shuffled_identity"] = _clone_arrays(arrays, identity=shuffled_x, identity_valid=shuffled_valid)
        modes["zero_identity"] = _clone_arrays(arrays, identity=zero_x, identity_valid=zero_valid)
    if model.uses_market:
        shuffled_x, shuffled_valid = shuffled_market_time(arrays.market_x, arrays.market_valid)
        zero_x, zero_valid = zero_market(arrays.market_x, arrays.market_valid)
        modes["shuffled_market_time"] = _clone_arrays(arrays, market=shuffled_x, market_valid=shuffled_valid)
        modes["zero_market"] = _clone_arrays(arrays, market=zero_x, market_valid=zero_valid)
        if previous_market is not None and previous_market.shape[1:] == arrays.market_x.shape[1:]:
            broadcast = np.repeat(previous_market, arrays.episode_count, axis=0)
            valid = np.isfinite(broadcast).any(axis=-1)
            modes["previous_day_market"] = _clone_arrays(
                arrays,
                market=np.where(np.isfinite(broadcast), broadcast, 0.0).astype("float32"),
                market_valid=valid,
            )
    if model.uses_daily:
        shuffled, _ = same_day_shuffled_support(arrays.adapter.support_x, arrays.adapter.symbols)
        modes["shuffled_daily_support"] = _clone_arrays(arrays, support=shuffled)
        modes["zero_daily_support"] = _clone_arrays(arrays, support=np.zeros_like(arrays.adapter.support_x))
        if previous_support_bank is not None:
            previous, available = previous_day_support(
                arrays.adapter.support_x,
                arrays.adapter.symbols,
                previous_support_bank,
            )
            if available.any():
                modes["previous_day_support"] = _clone_arrays(arrays, support=previous)
    frames = {}
    for mode, value in modes.items():
        frame = _evaluate_arrays(model, value, normalization, device=device, batch_size=batch_size)
        frame["context_mode"] = mode
        frames[mode] = frame
    return frames


def _build_model(config, normalization, variant, device, identity_width, market_width):
    return NFFContextDecompositionModel(
        _load_base(config, normalization, device),
        identity_width,
        market_width,
        ContextDecompositionConfig(**dict(config.get("context_model") or {})),
        variant=variant,
        freeze_base=True,
    ).to(device)


def command_train(config, variant, manifest_path, output_override):
    torch.set_num_threads(int(config.get("run", {}).get("torch_threads", 4)))
    seed = int(config.get("run", {}).get("seed", 20260804))
    _seed_everything(seed)
    manifest = _manifest(config, manifest_path)
    segments = _split_dates(manifest["trade_date"].astype(str).unique(), config)
    normalization = _load_normalization(config)
    factory = _factory(config)
    store = _store(config, factory, normalization)
    identity_frame = store.read_identity()
    history_days = int(config.get("context_cache", {}).get("history_days", 20))
    probe = _date_arrays(factory, store, identity_frame, segments["train"][0], normalization, history_days)
    output_root = _model_root(config, variant, output_override)
    output_root.mkdir(parents=True, exist_ok=True)
    training = dict(config.get("context_training") or {})
    device = _resolve_device(str(training.get("device", "auto")))
    model = _build_model(config, normalization, variant, device, len(probe.identity_columns), len(probe.market_columns))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    epochs = int(training.get("epochs", 10))
    batch_size = int(training.get("batch_size", 32))
    grad_clip = float(training.get("grad_clip", 1.0))
    contract = {
        "version": "nff_context_decomposition_p4_v1",
        "variant": variant,
        "episode_contract_hash": factory.contract_hash,
        "context_cache_hash": store.contract_hash,
        "normalization": normalization.to_dict(),
        "model": model.model_contract(),
        "segments": segments,
        "training": training,
        "seed": seed,
    }
    expected_hash = contract_hash(contract)
    _atomic_json(output_root / "train_contract.json", {"contract_hash": expected_hash, **contract})
    last_path = output_root / "checkpoint_last.pt"
    best_path = output_root / "checkpoint_best.pt"
    start_epoch = next_date_index = 0
    best_valid = float("inf")
    history_rows = []
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        if checkpoint.get("contract_hash") != expected_hash:
            raise RuntimeError("Existing checkpoint belongs to a different contract")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint.get("epoch", 0))
        next_date_index = int(checkpoint.get("next_date_index", 0))
        best_valid = float(checkpoint.get("best_valid", float("inf")))
        history_rows = list(checkpoint.get("history_rows", []))

    for epoch in range(start_epoch, epochs):
        dates = list(segments["train"])
        random.Random(seed + epoch).shuffle(dates)
        date_start = next_date_index if epoch == start_epoch else 0
        for date_index in range(date_start, len(dates)):
            trade_date = dates[date_index]
            started = time.perf_counter()
            metrics = _train_date(
                model,
                optimizer,
                _date_arrays(factory, store, identity_frame, trade_date, normalization, history_days),
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
                last_path,
                {
                    "contract_hash": expected_hash,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "next_date_index": date_index + 1,
                    "best_valid": best_valid,
                    "history_rows": history_rows,
                },
            )

        valid_frames = []
        for trade_date in segments["valid"]:
            arrays = _date_arrays(factory, store, identity_frame, trade_date, normalization, history_days)
            valid_frames.append(_evaluate_arrays(model, arrays, normalization, device=device, batch_size=batch_size))
        valid_frame = pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame()
        valid_metrics = _paired_metrics(valid_frame)
        valid_loss = float(valid_metrics.get("mse")) if valid_metrics.get("mse") is not None else float("inf")
        history_rows.append({"epoch": epoch, "segment": "valid", **valid_metrics})
        if valid_loss < best_valid:
            best_valid = valid_loss
            _atomic_torch(
                best_path,
                {
                    "contract_hash": expected_hash,
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "valid_metrics": valid_metrics,
                },
            )
        next_date_index = 0
        _atomic_torch(
            last_path,
            {
                "contract_hash": expected_hash,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "next_date_index": 0,
                "best_valid": best_valid,
                "history_rows": history_rows,
            },
        )
        _atomic_parquet(pd.DataFrame(history_rows), output_root / "training_history.parquet")

    return command_evaluate(config, variant, manifest_path, output_override)


def command_evaluate(config, variant, manifest_path, output_override):
    manifest = _manifest(config, manifest_path)
    segments = _split_dates(manifest["trade_date"].astype(str).unique(), config)
    normalization = _load_normalization(config)
    factory = _factory(config)
    store = _store(config, factory, normalization)
    identity_frame = store.read_identity()
    history_days = int(config.get("context_cache", {}).get("history_days", 20))
    probe = _date_arrays(factory, store, identity_frame, segments["test"][0], normalization, history_days)
    output_root = _model_root(config, variant, output_override)
    training = dict(config.get("context_training") or {})
    device = _resolve_device(str(training.get("device", "auto")))
    model = _build_model(config, normalization, variant, device, len(probe.identity_columns), len(probe.market_columns))
    checkpoint = torch.load(output_root / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    batch_size = int(training.get("batch_size", 32))
    prediction_frames = []
    ablation_rows = []
    previous_market = None
    previous_support_bank: Dict[str, np.ndarray] = {}
    for trade_date in segments["test"]:
        arrays = _date_arrays(factory, store, identity_frame, trade_date, normalization, history_days)
        modes = _counterfactual_frames(
            model,
            arrays,
            normalization,
            device=device,
            batch_size=batch_size,
            previous_market=previous_market,
            previous_support_bank=previous_support_bank,
        )
        for mode, frame in modes.items():
            metrics = _paired_metrics(frame)
            ablation_rows.append({"trade_date": trade_date, "context_mode": mode, **metrics})
            if mode == "same_context":
                prediction_frames.append(frame)
        if arrays.episode_count:
            previous_market = np.nanmean(
                np.where(arrays.market_valid[..., None], arrays.market_x, np.nan),
                axis=0,
                keepdims=True,
            ).astype("float32")
            update_previous_support_bank(previous_support_bank, arrays.adapter.support_x, arrays.adapter.symbols)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    test_root = output_root / "test"
    _atomic_parquet(predictions, test_root / "paired_predictions.parquet")
    _atomic_parquet(pd.DataFrame(ablation_rows), test_root / "context_ablation_metrics.parquet")
    metrics = _paired_metrics(predictions)
    bootstrap = _bootstrap_by_date(
        predictions,
        seed=int(config.get("run", {}).get("seed", 20260804)),
        draws=int(config.get("evaluation", {}).get("bootstrap_draws", 2000)),
    )
    _atomic_json(test_root / "paired_date_bootstrap.json", bootstrap)
    gate_columns = [column for column in predictions.columns if column.startswith("gate_")]
    gate_summary = predictions[gate_columns].describe().T.reset_index(names="gate") if gate_columns else pd.DataFrame()
    _atomic_parquet(gate_summary, test_root / "gate_distribution.parquet")
    query_metrics = []
    for (query_index, target_name), block in predictions.groupby(["query_index", "target_name"], sort=True):
        query_metrics.append({"query_index": int(query_index), "target_name": target_name, **_paired_metrics(block)})
    _atomic_parquet(pd.DataFrame(query_metrics), test_root / "query_time_metrics.parquet")
    summary = {
        "status": "complete",
        "variant": variant,
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "metrics": metrics,
        "bootstrap": bootstrap,
        "test_dates": segments["test"],
        "nff_only": True,
    }
    _atomic_json(output_root / "summary.json", summary)
    report = [
        f"# NFF context decomposition — {variant}",
        "",
        "- Inputs: NFF only; no GFF/GAL access.",
        f"- Test rows: {metrics.get('rows')}",
        f"- Delta MSE vs P2: {metrics.get('delta_mse')}",
        f"- Delta RankIC vs P2: {metrics.get('delta_rank_ic')}",
        f"- Date-bootstrap probability of lower MSE: {bootstrap.get('probability_adapter_better')}",
        "- Identity uses only dates strictly before the current episode.",
        "- Market context is exact leave-one-out at the current query time.",
    ]
    (output_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


def command_evaluate_all(config, output_override):
    rows = []
    roots = {
        "B0_generic": _generic_root(config),
        "B1_daily": Path(
            config.get("output", {}).get("daily_adapter_root", "model_runs/daily_adapter_pilot_v1")
        ).expanduser().resolve(),
    }
    for variant in _VARIANTS:
        roots[f"B{2 + _VARIANTS.index(variant)}_{variant}"] = _model_root(config, variant, output_override)
    for name, root in roots.items():
        path = root / "summary.json"
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        metrics = value.get("metrics") or value.get("test_metrics") or value.get("test", {})
        rows.append(
            {
                "model": name,
                "root": str(root),
                **{
                    key: metrics.get(key)
                    for key in ("mse", "mae", "rank_ic_mean", "delta_mse", "delta_rank_ic")
                },
            }
        )
    frame = pd.DataFrame(rows)
    root = (
        output_override
        or Path(config.get("output", {}).get("context_model_root", "model_runs/context_decomposition_v1"))
    ).expanduser().resolve()
    _atomic_parquet(frame, root / "model_comparison.parquet")
    print(frame.to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "evaluate", "evaluate-all"])
    parser.add_argument("--variant", choices=_VARIANTS)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config = _read_config(Path(args.config).expanduser().resolve())
    manifest = Path(args.manifest).expanduser().resolve() if args.manifest else None
    output = Path(args.output_root).expanduser().resolve() if args.output_root else None
    if args.command == "evaluate-all":
        return command_evaluate_all(config, output)
    if not args.variant:
        raise ValueError("--variant is required for train/evaluate")
    if args.command == "train":
        return command_train(config, args.variant, manifest, output)
    return command_evaluate(config, args.variant, manifest, output)


if __name__ == "__main__":
    raise SystemExit(main())
