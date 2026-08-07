from __future__ import annotations

"""Strict temporal walk-forward models for the NFF v2.7 campaign.

All feature direction, coverage filtering, redundancy filtering, imputation,
weights and hyperparameters are fitted on train/validation only. Test dates are
never used to select factors or models. The account simulator interprets model
output as expected future return: high predictions are long and low
predictions are short.
"""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nff_research import v2_5_full_campaign as BASE


EPS = 1e-8


@dataclass
class FeatureContract:
    features: list[str]
    directions: dict[str, float]
    ic_scores: dict[str, float]
    medians: dict[str, float]
    correlation_threshold: float


@dataclass
class LinearModel:
    model: str
    coefficients: np.ndarray
    intercept: float
    parameters: dict[str, Any]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temp, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temp, index=False, compression="zstd")
    os.replace(temp, path)


def _read_cache(paths: Mapping[str, Path], dates: Sequence[str], columns: Sequence[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for date in dates:
        path = paths.get(date)
        if path is None or not path.exists():
            continue
        available = set(pq.read_schema(path).names)
        selected = [column for column in columns if column in available]
        frame = pd.read_parquet(path, columns=selected)
        if not frame.empty:
            frame["trade_date"] = date
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, copy=False) if frames else pd.DataFrame(columns=list(columns))


def _rank_target(frame: pd.DataFrame) -> np.ndarray:
    ranked = frame.groupby("decision_time", sort=False)["label"].rank(method="average", pct=True)
    return ranked.sub(ranked.groupby(frame["decision_time"], sort=False).transform("mean")).to_numpy(dtype="float64")


def _rank_features(frame: pd.DataFrame, features: Sequence[str], directions: Mapping[str, float]) -> pd.DataFrame:
    values = frame[list(features)].apply(pd.to_numeric, errors="coerce")
    ranked = values.groupby(frame["decision_time"], sort=False).rank(method="average", pct=True)
    ranked = ranked.sub(ranked.groupby(frame["decision_time"], sort=False).transform("mean"))
    for feature in features:
        ranked[feature] *= float(directions.get(feature, 1.0))
    return ranked.astype("float32")


def _pooled_rank_ic(frame: pd.DataFrame, prediction: str = "prediction", min_n: int = 30) -> float:
    values: list[float] = []
    for _, block in frame.groupby("decision_time", sort=False):
        block = block[[prediction, "label"]].dropna()
        if len(block) < min_n:
            continue
        value = block[prediction].rank(method="average").corr(block["label"].rank(method="average"))
        if np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else math.nan


def _feature_ic_scores(frame: pd.DataFrame, features: Sequence[str], min_n: int = 30) -> dict[str, float]:
    label_rank = frame.groupby("decision_time", sort=False)["label"].rank(method="average", pct=True)
    label_rank -= label_rank.groupby(frame["decision_time"], sort=False).transform("mean")
    y = label_rank.to_numpy(dtype="float64")
    scores: dict[str, float] = {}
    for start in range(0, len(features), 32):
        chunk = list(features[start : start + 32])
        x = frame[chunk].apply(pd.to_numeric, errors="coerce")
        ranked = x.groupby(frame["decision_time"], sort=False).rank(method="average", pct=True)
        ranked -= ranked.groupby(frame["decision_time"], sort=False).transform("mean")
        for feature in chunk:
            values = ranked[feature].to_numpy(dtype="float64")
            valid = np.isfinite(values) & np.isfinite(y)
            if int(valid.sum()) < min_n:
                scores[feature] = math.nan
                continue
            xv = values[valid]
            yv = y[valid]
            denominator = math.sqrt(float(np.dot(xv, xv) * np.dot(yv, yv)))
            scores[feature] = float(np.dot(xv, yv) / denominator) if denominator > 0 else math.nan
    return scores


def _fit_feature_contract(
    train: pd.DataFrame,
    candidates: Sequence[str],
    *,
    minimum_coverage: float = 0.70,
    correlation_threshold: float = 0.90,
    max_features: int = 80,
    min_n: int = 30,
) -> FeatureContract:
    coverage = train[list(candidates)].notna().mean()
    available = [feature for feature in candidates if float(coverage.get(feature, 0.0)) >= minimum_coverage]
    scores = _feature_ic_scores(train, available, min_n=min_n)
    ranked_features = sorted(
        [feature for feature in available if np.isfinite(scores.get(feature, math.nan))],
        key=lambda feature: (-abs(scores[feature]), feature),
    )
    directions = {feature: (1.0 if scores[feature] >= 0 else -1.0) for feature in ranked_features}
    ranked = _rank_features(train, ranked_features, directions)
    sample = ranked
    if len(sample) > 100_000:
        sample = sample.iloc[np.linspace(0, len(sample) - 1, 100_000, dtype=int)]
    selected: list[str] = []
    for feature in ranked_features:
        if len(selected) >= max_features:
            break
        if selected:
            correlation = sample[selected].corrwith(sample[feature]).abs()
            if bool((correlation >= correlation_threshold).any()):
                continue
        selected.append(feature)
    medians = {
        feature: float(pd.to_numeric(train[feature], errors="coerce").median())
        for feature in selected
    }
    return FeatureContract(
        features=selected,
        directions={feature: directions[feature] for feature in selected},
        ic_scores={feature: scores[feature] for feature in selected},
        medians=medians,
        correlation_threshold=correlation_threshold,
    )


def _matrix(frame: pd.DataFrame, contract: FeatureContract) -> np.ndarray:
    ranked = _rank_features(frame, contract.features, contract.directions)
    return ranked.fillna(0.0).to_numpy(dtype="float64")


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> LinearModel:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return LinearModel("ridge", coefficients[1:], float(coefficients[0]), {"alpha": alpha})


def _soft_threshold(value: float, threshold: float) -> float:
    return math.copysign(max(abs(value) - threshold, 0.0), value)


def _fit_elastic_net(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    iterations: int = 80,
) -> LinearModel:
    intercept = float(np.mean(y))
    centered_y = y - intercept
    coefficients = np.zeros(x.shape[1], dtype="float64")
    column_norm = np.sum(x * x, axis=0) / max(len(x), 1) + alpha * (1.0 - l1_ratio)
    residual = centered_y.copy()
    for _ in range(iterations):
        previous = coefficients.copy()
        for column in range(x.shape[1]):
            residual += x[:, column] * coefficients[column]
            rho = float(np.dot(x[:, column], residual) / max(len(x), 1))
            coefficients[column] = _soft_threshold(rho, alpha * l1_ratio) / max(column_norm[column], EPS)
            residual -= x[:, column] * coefficients[column]
        if float(np.max(np.abs(coefficients - previous))) < 1e-7:
            break
    return LinearModel(
        "elastic_net",
        coefficients,
        intercept,
        {"alpha": alpha, "l1_ratio": l1_ratio, "iterations": iterations},
    )


def _predict_linear(model: LinearModel, x: np.ndarray) -> np.ndarray:
    return model.intercept + x @ model.coefficients


def _prediction_frame(source: pd.DataFrame, predictions: np.ndarray, model: str, fold_id: int) -> pd.DataFrame:
    result = source.copy()
    result["prediction"] = np.asarray(predictions, dtype="float64")
    result["model"] = model
    result["fold_id"] = fold_id
    return result


def _validation_score(source: pd.DataFrame, predictions: np.ndarray) -> float:
    return _pooled_rank_ic(_prediction_frame(source, predictions, "validation", -1))


def _fit_equal(contract: FeatureContract) -> dict[str, Any]:
    return {"model": "equal_weight", "weights": np.ones(len(contract.features)) / max(len(contract.features), 1)}


def _fit_ic_weight(contract: FeatureContract) -> dict[str, Any]:
    raw = np.array([abs(contract.ic_scores[feature]) for feature in contract.features], dtype="float64")
    weights = raw / raw.sum() if raw.sum() > 0 else np.ones(len(raw)) / max(len(raw), 1)
    return {"model": "ic_weight", "weights": weights}


def _predict_weighted(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    return x @ np.asarray(model["weights"], dtype="float64")


def _fit_lightgbm(train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray, validation_y: np.ndarray, n_jobs: int) -> tuple[Any, dict[str, Any]]:
    from lightgbm import LGBMRegressor

    candidates = [
        {"num_leaves": 15, "min_child_samples": 100},
        {"num_leaves": 31, "min_child_samples": 200},
    ]
    best_model = None
    best_score = math.inf
    best_parameters: dict[str, Any] = {}
    for parameters in candidates:
        model = LGBMRegressor(
            n_estimators=160,
            learning_rate=0.04,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,
            reg_lambda=0.10,
            random_state=20260806,
            verbosity=-1,
            n_jobs=max(1, n_jobs),
            **parameters,
        )
        model.fit(train_x, train_y)
        prediction = model.predict(validation_x)
        mse = float(np.mean((prediction - validation_y) ** 2))
        if mse < best_score:
            best_score = mse
            best_model = model
            best_parameters = {**parameters, "validation_mse": mse}
    return best_model, best_parameters


def _sequence_arrays(
    frame: pd.DataFrame,
    x: np.ndarray,
    sequence_length: int,
    row_filter: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = frame[["symbol", "decision_time"]].copy()
    work["__position"] = np.arange(len(work))
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    positions: list[int] = []
    labels = pd.to_numeric(frame["label"], errors="coerce").to_numpy(dtype="float64")
    for _, group in work.sort_values(["symbol", "decision_time"]).groupby("symbol", sort=False):
        ordered = group["__position"].to_numpy(dtype="int64")
        for index in range(sequence_length - 1, len(ordered)):
            position = int(ordered[index])
            if row_filter is not None and position not in row_filter:
                continue
            window = ordered[index - sequence_length + 1 : index + 1]
            if not np.isfinite(labels[position]):
                continue
            sequences.append(x[window])
            targets.append(labels[position])
            positions.append(position)
    if not sequences:
        return (
            np.empty((0, sequence_length, x.shape[1]), dtype="float32"),
            np.empty(0, dtype="float32"),
            np.empty(0, dtype="int64"),
        )
    return np.asarray(sequences, dtype="float32"), np.asarray(targets, dtype="float32"), np.asarray(positions, dtype="int64")


def _fit_predict_gru(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    contract: FeatureContract,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    sequence_length = int(config.get("sequence_length", 15))
    hidden_size = int(config.get("hidden_size", 32))
    epochs = int(config.get("epochs", 3))
    batch_size = int(config.get("batch_size", 1024))
    max_sequences = int(config.get("max_train_sequences", 200_000))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x = _matrix(train, contract)
    validation_x = _matrix(validation, contract)
    test_x = _matrix(test, contract)
    train_seq, train_y, _ = _sequence_arrays(train, train_x, sequence_length)
    validation_seq, _, validation_positions = _sequence_arrays(validation, validation_x, sequence_length)
    test_seq, _, test_positions = _sequence_arrays(test, test_x, sequence_length)
    if len(train_seq) > max_sequences:
        selected = np.linspace(0, len(train_seq) - 1, max_sequences, dtype=int)
        train_seq, train_y = train_seq[selected], train_y[selected]
    if len(train_seq) == 0 or len(validation_seq) == 0 or len(test_seq) == 0:
        raise RuntimeError("insufficient sequences for GRU")

    class Regressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(len(contract.features), hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, 1)

        def forward(self, values):
            state, _ = self.gru(values)
            return self.output(state[:, -1]).squeeze(-1)

    model = Regressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_seq), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    def predict(sequences: np.ndarray, positions: np.ndarray, length: int) -> np.ndarray:
        values = np.full(length, np.nan, dtype="float64")
        model.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch = torch.from_numpy(sequences[start : start + batch_size]).to(device)
                outputs.append(model(batch).cpu().numpy())
        values[positions] = np.concatenate(outputs)
        return values

    parameters = {
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "train_sequences": len(train_seq),
    }
    return (
        predict(validation_seq, validation_positions, len(validation)),
        predict(test_seq, test_positions, len(test)),
        parameters,
        model,
    )


def _save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        if hasattr(model, "state_dict"):
            torch.save(model.state_dict(), path.with_suffix(".pt"))
            return
    except Exception:
        pass
    if hasattr(model, "booster_"):
        model.booster_.save_model(str(path.with_suffix(".txt")))
        return
    with path.with_suffix(".pkl").open("wb") as handle:
        pickle.dump(model, handle)


def _fit_fold_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    candidates: Sequence[str],
    model_names: Sequence[str],
    config: Mapping[str, Any],
    fold_id: int,
    output_root: Path,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], FeatureContract]:
    contract = _fit_feature_contract(
        train,
        candidates,
        minimum_coverage=float(config.get("minimum_feature_coverage", 0.70)),
        correlation_threshold=float(config.get("redundancy_correlation", 0.90)),
        max_features=int(config.get("max_features", 80)),
    )
    if not contract.features:
        raise RuntimeError("no features survived train-only selection")
    fold_root = output_root / f"fold={fold_id:03d}"
    _atomic_json(
        fold_root / "feature_contract.json",
        {
            "features": contract.features,
            "directions": contract.directions,
            "ic_scores": contract.ic_scores,
            "correlation_threshold": contract.correlation_threshold,
        },
    )
    train_x = _matrix(train, contract)
    validation_x = _matrix(validation, contract)
    test_x = _matrix(test, contract)
    train_y = _rank_target(train)
    validation_y = _rank_target(validation)
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []

    def record(name: str, validation_prediction: np.ndarray, test_prediction: np.ndarray, parameters: Mapping[str, Any], model: Any) -> None:
        validation_frame = _prediction_frame(validation, validation_prediction, name, fold_id)
        test_frame = _prediction_frame(test, test_prediction, name, fold_id)
        validation_ic = _pooled_rank_ic(validation_frame)
        test_ic = _pooled_rank_ic(test_frame)
        predictions.append(test_frame)
        metrics.append(
            {
                "fold_id": fold_id,
                "model": name,
                "validation_rank_ic": validation_ic,
                "test_rank_ic": test_ic,
                "train_rows": len(train),
                "validation_rows": int(np.isfinite(validation_prediction).sum()),
                "test_rows": int(np.isfinite(test_prediction).sum()),
                "feature_count": len(contract.features),
                "parameters": json.dumps(parameters, sort_keys=True, default=str),
                "status": "SUCCESS",
            }
        )
        _save_model(model, fold_root / name / "model")
        _atomic_json(fold_root / name / "metrics.json", metrics[-1])

    for name in model_names:
        try:
            if name == "equal_weight":
                model = _fit_equal(contract)
                record(name, _predict_weighted(model, validation_x), _predict_weighted(model, test_x), {}, model)
            elif name == "ic_weight":
                model = _fit_ic_weight(contract)
                record(name, _predict_weighted(model, validation_x), _predict_weighted(model, test_x), {}, model)
            elif name == "ridge":
                best = None
                for alpha in config.get("ridge_alphas", [0.001, 0.01, 0.1, 1.0]):
                    model = _fit_ridge(train_x, train_y, float(alpha))
                    prediction = _predict_linear(model, validation_x)
                    score = _validation_score(validation, prediction)
                    if best is None or (np.isfinite(score) and score > best[0]):
                        best = (score, model, prediction)
                assert best is not None
                record(name, best[2], _predict_linear(best[1], test_x), best[1].parameters, best[1])
            elif name == "elastic_net":
                best = None
                for alpha in config.get("elastic_alphas", [0.0005, 0.001, 0.005]):
                    for l1_ratio in config.get("elastic_l1_ratios", [0.1, 0.5, 0.9]):
                        model = _fit_elastic_net(train_x, train_y, float(alpha), float(l1_ratio))
                        prediction = _predict_linear(model, validation_x)
                        score = _validation_score(validation, prediction)
                        if best is None or (np.isfinite(score) and score > best[0]):
                            best = (score, model, prediction)
                assert best is not None
                record(name, best[2], _predict_linear(best[1], test_x), best[1].parameters, best[1])
            elif name == "lightgbm":
                max_rows = int(config.get("max_tree_train_rows", 300_000))
                if len(train_x) > max_rows:
                    selected = np.linspace(0, len(train_x) - 1, max_rows, dtype=int)
                    fit_x, fit_y = train_x[selected], train_y[selected]
                else:
                    fit_x, fit_y = train_x, train_y
                model, parameters = _fit_lightgbm(
                    fit_x,
                    fit_y,
                    validation_x,
                    validation_y,
                    int(config.get("lightgbm_threads", 8)),
                )
                record(name, model.predict(validation_x), model.predict(test_x), parameters, model)
            elif name == "gru":
                validation_prediction, test_prediction, parameters, model = _fit_predict_gru(
                    train,
                    validation,
                    test,
                    contract,
                    config.get("gru", {}),
                )
                record(name, validation_prediction, test_prediction, parameters, model)
        except Exception as exc:
            metrics.append(
                {
                    "fold_id": fold_id,
                    "model": name,
                    "validation_rank_ic": math.nan,
                    "test_rank_ic": math.nan,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                    "feature_count": len(contract.features),
                    "parameters": "{}",
                    "status": "MODEL_UNAVAILABLE" if isinstance(exc, (ImportError, ModuleNotFoundError)) else "MODEL_FAILED",
                    "error": repr(exc),
                }
            )
            _atomic_json(fold_root / name / "failure.json", metrics[-1])
    return predictions, metrics, contract


def _weights(block: pd.DataFrame, quantile: float = 0.05) -> pd.Series:
    ranked = block["prediction"].rank(method="first", pct=True)
    long_index = block.index[ranked >= 1.0 - quantile]
    short_index = block.index[ranked <= quantile]
    if len(long_index) == 0 or len(short_index) == 0:
        return pd.Series(dtype="float64")
    weights = pd.Series(0.0, index=block.index, dtype="float64")
    weights.loc[long_index] = 0.5 / len(long_index)
    weights.loc[short_index] = -0.5 / len(short_index)
    return weights[weights != 0]


def account_replay(
    predictions: pd.DataFrame,
    output_root: Path,
    *,
    initial_equity: float,
    participation: float,
    one_way_bps: float,
    write_ledger: bool = True,
) -> dict[str, Any]:
    cash = float(initial_equity)
    positions: dict[str, float] = {}
    previous_prices: dict[str, float] = {}
    orders: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for timestamp, block in predictions.sort_values("decision_time").groupby("decision_time", sort=True):
        block = block.dropna(subset=["prediction", "vwap", "dollar_volume"])
        if len(block) < 40:
            continue
        if "hawkes_total_intensity" in block:
            block = block.loc[block["hawkes_total_intensity"].rank(pct=True).fillna(0) <= 0.80]
        target_weights = _weights(block)
        if target_weights.empty:
            continue
        target = {
            str(block.loc[index, "symbol"]): float(weight)
            for index, weight in target_weights.items()
        }
        prices = {
            str(row.symbol): float(row.vwap)
            for row in block.itertuples()
            if np.isfinite(row.vwap) and row.vwap > 0
        }
        equity_before = cash + sum(
            shares * prices.get(symbol, previous_prices.get(symbol, 0.0))
            for symbol, shares in positions.items()
        )
        current_values = {
            symbol: shares * prices.get(symbol, previous_prices.get(symbol, 0.0))
            for symbol, shares in positions.items()
        }
        timestamp_cost = 0.0
        for symbol in set(positions) | set(target):
            price = prices.get(symbol, previous_prices.get(symbol, np.nan))
            if not np.isfinite(price) or price <= 0:
                continue
            current_weight = current_values.get(symbol, 0.0) / max(equity_before, 1.0)
            submitted = (target.get(symbol, 0.0) - current_weight) * equity_before
            row = block.loc[block["symbol"].astype(str) == symbol]
            available = float(row["dollar_volume"].iloc[0]) if not row.empty else 0.0
            capacity = max(0.0, available * participation)
            filled = float(np.sign(submitted) * min(abs(submitted), capacity))
            shares = filled / price if price else 0.0
            if abs(shares) < 1e-10:
                continue
            impact_bps = 5.0 * math.sqrt(min(1.0, abs(filled) / max(available, 1.0)))
            execution_price = price * (
                1.0 + np.sign(shares) * (one_way_bps + impact_bps) / 10000.0
            )
            commission = abs(filled) * 0.2 / 10000.0
            spread = abs(filled) * one_way_bps / 10000.0
            impact = abs(filled) * impact_bps / 10000.0
            timestamp_cost += commission + spread + impact
            orders.append(
                {
                    "decision_time": timestamp,
                    "symbol": symbol,
                    "target_weight": target.get(symbol, 0.0),
                    "submitted_value": submitted,
                    "filled_value": filled,
                    "unfilled_value": submitted - filled,
                    "participation": abs(filled) / max(available, 1.0),
                }
            )
            fills.append(
                {
                    "decision_time": timestamp,
                    "symbol": symbol,
                    "shares": shares,
                    "fill_price": execution_price,
                    "notional": abs(filled),
                    "commission": commission,
                    "spread": spread,
                    "impact": impact,
                }
            )
            cash -= shares * execution_price + commission
            positions[symbol] = positions.get(symbol, 0.0) + shares
            if abs(positions[symbol]) < 1e-10:
                positions.pop(symbol, None)
        short_value = sum(
            abs(shares * prices.get(symbol, previous_prices.get(symbol, 0.0)))
            for symbol, shares in positions.items()
            if shares < 0
        )
        borrow = short_value * 50.0 / 10000.0 / 252.0 * 15.0 / 390.0
        cash -= borrow
        timestamp_cost += borrow
        equity = cash + sum(
            shares * prices.get(symbol, previous_prices.get(symbol, 0.0))
            for symbol, shares in positions.items()
        )
        snapshots.append(
            {
                "decision_time": timestamp,
                "cash": cash,
                "equity": equity,
                "equity_before": equity_before,
                "gross_exposure": sum(
                    abs(shares * prices.get(symbol, previous_prices.get(symbol, 0.0)))
                    for symbol, shares in positions.items()
                ) / max(equity, 1.0),
                "net_exposure": sum(
                    shares * prices.get(symbol, previous_prices.get(symbol, 0.0))
                    for symbol, shares in positions.items()
                ) / max(equity, 1.0),
                "borrow": borrow,
                "position_count": len(positions),
                "position_drift": equity - equity_before + timestamp_cost,
            }
        )
        previous_prices = prices
    orders_frame = pd.DataFrame(orders)
    fills_frame = pd.DataFrame(fills)
    snapshots_frame = pd.DataFrame(snapshots)
    if write_ledger:
        _atomic_parquet(orders_frame, output_root / "orders.parquet")
        _atomic_parquet(fills_frame, output_root / "fills.parquet")
        _atomic_parquet(snapshots_frame, output_root / "account_snapshots.parquet")
    if snapshots_frame.empty:
        return {"status": "NO_OOS_ROWS"}
    equity = (
        snapshots_frame.assign(decision_time=pd.to_datetime(snapshots_frame["decision_time"], utc=True))
        .set_index("decision_time")["equity"]
        .resample("1D")
        .last()
        .dropna()
    )
    returns = equity.pct_change().dropna()
    drawdown = equity / equity.cummax() - 1.0
    submitted = float(orders_frame["submitted_value"].abs().sum()) if not orders_frame.empty else 0.0
    filled = float(fills_frame["notional"].sum()) if not fills_frame.empty else 0.0
    return {
        "status": "SUCCESS",
        "initial_equity": initial_equity,
        "participation": participation,
        "one_way_bps": one_way_bps,
        "start_equity": float(equity.iloc[0]),
        "end_equity": float(equity.iloc[-1]),
        "net_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "sharpe": float(returns.mean() / returns.std() * math.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else math.nan,
        "max_drawdown": float(drawdown.min()),
        "orders": len(orders_frame),
        "fills": len(fills_frame),
        "fill_ratio": filled / submitted if submitted > 0 else math.nan,
        "unfilled_ratio": 1.0 - filled / submitted if submitted > 0 else math.nan,
        "turnover_notional": filled,
    }


def _recorder(
    run_root: Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    contracts: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    audit: dict[str, Any] = {"status": "FAILED", "recorders": []}
    try:
        import qlib
        from qlib.workflow import R as QlibR

        provider = run_root / str(config.get("qlib_recorder", {}).get("provider_subdir", "qlib_provider"))
        qlib.init(provider_uri=str(provider), region="us")
        experiment = str(config.get("qlib_recorder", {}).get("experiment_name", "nff_v2_7"))
        for row in metrics.loc[metrics["status"].eq("SUCCESS")].itertuples(index=False):
            recorder_name = f"{row.model}_fold_{int(row.fold_id):03d}"
            prediction = predictions.loc[
                predictions["model"].eq(row.model) & predictions["fold_id"].eq(row.fold_id)
            ]
            contract = next(
                value for value in contracts
                if value["fold_id"] == row.fold_id
            )
            with QlibR.start(experiment_name=experiment, recorder_name=recorder_name):
                QlibR.log_params(
                    research_version="2.7",
                    model=row.model,
                    fold_id=int(row.fold_id),
                    feature_count=int(row.feature_count),
                    primary_label="return_vwap_to_vwap__h30",
                )
                QlibR.log_metrics(
                    validation_rank_ic=float(row.validation_rank_ic),
                    test_rank_ic=float(row.test_rank_ic),
                )
                try:
                    QlibR.save_objects(
                        predictions=prediction,
                        labels=prediction[["decision_time", "symbol", "label"]],
                        selected_features=contract,
                    )
                except Exception:
                    pass
            audit["recorders"].append(recorder_name)
        audit.update({"status": "complete", "experiment_name": experiment})
    except Exception as exc:
        audit["error"] = repr(exc)
    _atomic_json(run_root / "recorder_audit_v2_7.json", audit)
    return audit


def run_temporal_oos_multi(run_root: Path, config: dict[str, Any]) -> dict[str, Any]:
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
    if not candidates:
        return {"status": "MODEL_FAILED", "reason": "no full_factor columns in model cache"}
    metadata = [
        "label",
        "symbol",
        "decision_time",
        "sector_code",
        "industry_code",
        "market_cap",
        "vwap",
        "dollar_volume",
        "trade_count",
        "hawkes_total_intensity",
    ]
    model_names = list(walk.get("models", ["equal_weight", "ic_weight", "ridge", "elastic_net", "lightgbm", "gru"]))
    output_root = run_root / "walk_forward_v2_7"
    all_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    feature_contracts: list[dict[str, Any]] = []
    for fold in folds:
        train = _read_cache(paths, fold["train"], [*candidates, *metadata])
        validation = _read_cache(paths, fold["validation"], [*candidates, *metadata])
        test = _read_cache(paths, fold["test"], [*candidates, *metadata])
        for frame in (train, validation, test):
            frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
            frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
        train = train.dropna(subset=["label", "decision_time", "symbol"])
        validation = validation.dropna(subset=["label", "decision_time", "symbol"])
        test = test.dropna(subset=["label", "decision_time", "symbol"])
        if len(train) < 100 or validation.empty or test.empty:
            continue
        predictions, metrics, contract = _fit_fold_models(
            train,
            validation,
            test,
            candidates,
            model_names,
            walk,
            int(fold["fold_id"]),
            output_root,
        )
        all_predictions.extend(predictions)
        metric_rows.extend(
            [
                {
                    **row,
                    "train_start": fold["train"][0],
                    "train_end": fold["train"][-1],
                    "validation_start": fold["validation"][0],
                    "validation_end": fold["validation"][-1],
                    "test_start": fold["test"][0],
                    "test_end": fold["test"][-1],
                }
                for row in metrics
            ]
        )
        feature_contracts.append(
            {
                "fold_id": int(fold["fold_id"]),
                "features": contract.features,
                "directions": contract.directions,
                "ic_scores": contract.ic_scores,
            }
        )
    predictions = pd.concat(all_predictions, ignore_index=True, copy=False) if all_predictions else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    _atomic_parquet(predictions, run_root / "predictions_oos_all_models.parquet")
    _atomic_parquet(metrics, run_root / "walk_forward_model_metrics.parquet")
    _atomic_json(run_root / "walk_forward_feature_contracts.json", feature_contracts)
    if predictions.empty or metrics.empty:
        return {"status": "MODEL_FAILED", "reason": "no successful OOS predictions", "metrics": metric_rows}

    successful = metrics.loc[metrics["status"].eq("SUCCESS")]
    validation_average = successful.groupby("model", sort=False)["validation_rank_ic"].mean().sort_values(ascending=False)
    primary_model = str(validation_average.index[0]) if not validation_average.empty else "ridge"
    account_summary: dict[str, Any] = {}
    for model in sorted(predictions["model"].unique()):
        model_predictions = predictions.loc[predictions["model"].eq(model)].copy()
        account_summary[model] = account_replay(
            model_predictions,
            run_root / "account" / f"model={model}" / "base",
            initial_equity=1_000_000.0,
            participation=0.05,
            one_way_bps=2.5,
            write_ledger=True,
        )
    capacity_rows = []
    primary_predictions = predictions.loc[predictions["model"].eq(primary_model)].copy()
    portfolio = config.get("portfolio_proxy", {})
    for account_size in portfolio.get("account_sizes", [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]):
        for participation in portfolio.get("participation_rates", [0.01, 0.02, 0.05, 0.10]):
            for cost in portfolio.get("costs_bps_one_way", [0, 1, 2.5, 5, 10]):
                result = account_replay(
                    primary_predictions,
                    run_root / "capacity" / f"model={primary_model}",
                    initial_equity=float(account_size),
                    participation=float(participation),
                    one_way_bps=float(cost),
                    write_ledger=False,
                )
                capacity_rows.append({"model": primary_model, **result})
    capacity = pd.DataFrame(capacity_rows)
    _atomic_parquet(capacity, run_root / "capacity_analysis_v2_7.parquet")
    _atomic_json(run_root / "account_metrics_all_models.json", account_summary)
    recorder = _recorder(run_root, metrics, predictions, feature_contracts, config)
    return {
        "status": "complete",
        "dates": len(dates),
        "folds": int(successful["fold_id"].nunique()),
        "prediction_rows": len(predictions),
        "models_requested": model_names,
        "models_successful": sorted(successful["model"].unique()),
        "model_failures": metrics.loc[~metrics["status"].eq("SUCCESS")].to_dict("records"),
        "primary_model_selected_on_validation": primary_model,
        "validation_model_scores": validation_average.to_dict(),
        "account": account_summary,
        "capacity_rows": len(capacity),
        "recorder": recorder,
    }
