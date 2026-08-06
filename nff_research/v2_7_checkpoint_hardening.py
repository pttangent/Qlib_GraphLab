from __future__ import annotations

"""Source-aware atomic checkpoints for the v2.7 daily research worker."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from nff_research import v2_7_formula_semantic_patch as SEMANTIC


_SOURCE_FINGERPRINT_CACHE: dict[tuple[str, str], str] = {}
_SAFE = re.compile(r"[^A-Za-z0-9_.=-]+")


def _safe(value: str) -> str:
    return _SAFE.sub("_", str(value)).strip("_")


def _files_under(root: Path, trade_date: str) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    if root.name == "nvg_supplement":
        for dataset in sorted(path for path in root.iterdir() if path.is_dir()):
            for schema in sorted(dataset.glob("schema=*")):
                partition = schema / f"date={trade_date}"
                if partition.exists():
                    files.extend(sorted(partition.rglob("*.parquet")))
        return files
    for schema in sorted(root.glob("schema=*")):
        partition = schema / f"date={trade_date}"
        if partition.exists():
            files.extend(sorted(partition.rglob("*.parquet")))
    return files


def _upstream_fingerprint(campaign: Any, trade_date: str) -> str:
    warehouse = Path(campaign.R.WAREHOUSE_ROOT)
    cache_key = (str(warehouse.resolve()), trade_date)
    cached = _SOURCE_FINGERPRINT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    roots = [
        warehouse / "features" / "minute_nvg",
        warehouse / "features" / "trade_nvg",
        warehouse / "features" / "hawkes_lite",
        warehouse / "canonical" / "bars_1m",
        warehouse / "canonical" / "trades_1m_core",
        warehouse / "canonical" / "trades_1m_sketch",
        warehouse / "canonical" / "trades_venue_1m",
        warehouse / "nvg_supplement",
    ]
    records: list[dict[str, Any]] = []
    for root in roots:
        for path in _files_under(root, trade_date):
            stat = path.stat()
            records.append(
                {
                    "path": path.relative_to(warehouse).as_posix(),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    fingerprint = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _SOURCE_FINGERPRINT_CACHE[cache_key] = fingerprint
    return fingerprint


def install(campaign: Any) -> None:
    # Install the final B/D/E direction-sensitive formulas in the same chain
    # used by every parent and detached date worker. Checkpoint contracts are
    # captured only after this patch, so old asymmetry-based blocks cannot be
    # reused as exact directional factors.
    SEMANTIC.install(campaign)
    base_factor_contract = campaign._factor_contract
    base_deciles = campaign._deciles_fast

    def factor_contract(group: pd.DataFrame, frame: pd.DataFrame) -> str:
        date = campaign.CTX.trade_date if campaign.CTX is not None else "unknown"
        return campaign._hash(
            {
                "base_contract": base_factor_contract(group, frame),
                "upstream_fingerprint": _upstream_fingerprint(campaign, date),
                "semantic_patch": "exact-direction-b-d-e-v2.7",
            }
        )

    def deciles_checkpointed(
        features: pd.DataFrame,
        labels: pd.DataFrame,
        label_masks: dict[str, pd.Series],
        controls: pd.DataFrame,
        residual_cache: dict[tuple[str, str], tuple[pd.DataFrame, pd.Series]],
        trade_date: str,
        min_n: int,
    ) -> pd.DataFrame:
        if campaign.CTX is None:
            return base_deciles(
                features,
                labels,
                label_masks,
                controls,
                residual_cache,
                trade_date,
                min_n,
            )
        context = campaign.CTX
        root = context.root / "deciles"
        upstream = _upstream_fingerprint(campaign, trade_date)
        masks = campaign.R.universe_masks(features, controls)
        factor_columns = [
            column for column in campaign.R.CORE_DECILE_FEATURES if column in features
        ]
        all_parts: list[pd.DataFrame] = []
        for universe in ("liquid_common_adv20_top1000", "common_structural"):
            for label_column in (
                column
                for column in labels
                if column.startswith("return_open_to_open__h")
                or column.startswith("return_vwap_to_vwap__h")
            ):
                family, horizon = label_column.rsplit("__h", 1)
                base_mask = (
                    masks[universe]
                    & labels[label_column].notna()
                    & label_masks[label_column].fillna(False)
                ).fillna(False)
                if int(base_mask.sum()) < min_n:
                    continue
                target_base = labels.loc[base_mask, label_column]
                adv = controls.loc[base_mask, "control__log_adv20"]
                price = features.loc[base_mask, "bars_1m__close"]
                trades = features.loc[base_mask].get(
                    "trades_1m_core__trade_count",
                    pd.Series(np.nan, index=target_base.index),
                )
                metadata_base = {
                    "trade_date": trade_date,
                    "universe": universe,
                    "label_family": family,
                    "horizon_bars": int(horizon),
                }
                for variant in ("raw", "winsorized", "neutralized"):
                    if variant == "neutralized":
                        cached = residual_cache.get((universe, label_column))
                        if cached is None:
                            continue
                        signal_source = cached[0].reindex(target_base.index)
                        target = cached[1].reindex(target_base.index)
                        available = [
                            column for column in factor_columns if column in signal_source
                        ]
                    else:
                        signal_source = features.loc[base_mask]
                        target = target_base
                        available = factor_columns
                    variant_root = (
                        root
                        / f"universe={_safe(universe)}"
                        / f"label={_safe(label_column)}"
                        / f"variant={_safe(variant)}"
                    )
                    block_size = max(1, int(context.factor_block_size))
                    for block_id, start in enumerate(range(0, len(available), block_size)):
                        selected = available[start : start + block_size]
                        if not selected:
                            continue
                        contract = campaign._hash(
                            {
                                "run_contract": context.contract_hash,
                                "upstream_fingerprint": upstream,
                                "index_hash": campaign._index_hash(target.index),
                                "universe": universe,
                                "label": label_column,
                                "variant": variant,
                                "factors": selected,
                                "min_n": min_n,
                                "decile_kernel": "pandas-qcut-edge-cache-v2.7",
                            }
                        )
                        path = variant_root / f"block={block_id:03d}.parquet"
                        manifest_path = variant_root / f"block={block_id:03d}.json"
                        if path.exists() and manifest_path.exists():
                            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                            if (
                                manifest.get("status") == "complete"
                                and manifest.get("contract_hash") == contract
                            ):
                                all_parts.append(pd.read_parquet(path))
                                campaign._event(
                                    "decile_block",
                                    "reused",
                                    universe=universe,
                                    label=label_column,
                                    variant=variant,
                                    block=block_id,
                                )
                                continue
                        signals = signal_source[selected]
                        if variant == "winsorized":
                            groups = signals.index.get_level_values("datetime")
                            grouped = signals.groupby(groups, sort=False, group_keys=False)
                            signals = signals.clip(
                                lower=grouped.transform("quantile", q=0.01),
                                upper=grouped.transform("quantile", q=0.99),
                            )
                        metadata = {**metadata_base, "variant": variant}
                        rows: list[dict[str, Any]] = []
                        with ThreadPoolExecutor(
                            max_workers=max(1, int(context.intra_workers))
                        ) as executor:
                            futures = [
                                executor.submit(
                                    campaign._decile_feature_rows,
                                    feature,
                                    signals[feature],
                                    target,
                                    adv,
                                    price,
                                    trades,
                                    metadata,
                                    min_n,
                                )
                                for feature in selected
                            ]
                            for future in futures:
                                rows.extend(future.result())
                        part = pd.DataFrame(rows)
                        campaign._atomic_parquet(part, path, index=False)
                        campaign._atomic_json(
                            manifest_path,
                            {
                                "status": "complete",
                                "contract_hash": contract,
                                "factors": selected,
                                "rows": len(part),
                            },
                        )
                        all_parts.append(part)
                        campaign._event(
                            "decile_block",
                            "complete",
                            universe=universe,
                            label=label_column,
                            variant=variant,
                            block=block_id,
                            factors=len(selected),
                        )
        if not all_parts:
            return pd.DataFrame()
        result = pd.concat(all_parts, ignore_index=True, copy=False)
        group_columns = [
            "trade_date",
            "universe",
            "feature",
            "bundle",
            "label_family",
            "horizon_bars",
            "variant",
            "rebalance",
            "decile",
        ]
        result = (
            result.groupby(group_columns, dropna=False)
            .agg(
                mean_label=("mean_label", "mean"),
                count=("count", "sum"),
                mean_log_adv20=("mean_log_adv20", "mean"),
                mean_price=("mean_price", "mean"),
                mean_trade_count=("mean_trade_count", "mean"),
            )
            .reset_index()
        )
        stage_path = context.root / "stages" / "decile_curves.parquet"
        campaign._atomic_parquet(result, stage_path, index=False)
        campaign._atomic_json(
            stage_path.with_suffix(".json"),
            {
                "status": "complete",
                "run_contract": context.contract_hash,
                "upstream_fingerprint": upstream,
                "rows": len(result),
            },
        )
        return result

    campaign._factor_contract = factor_contract
    campaign._deciles_fast = deciles_checkpointed
    # R.decile_curves was previously bound to the old function object. Rebind
    # the actual runner entrypoint so block-level resume is used in production.
    campaign.R.decile_curves = deciles_checkpointed
