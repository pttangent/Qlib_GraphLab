from __future__ import annotations

"""Block-checkpointed and mask-cached raw factor screening."""

import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd


def _mask_key(mask: pd.Series) -> bytes:
    values = mask.fillna(False).to_numpy(dtype=np.bool_, copy=False)
    return np.packbits(values, bitorder="little").tobytes()


def _file_contract(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.as_posix(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def install(P: Any) -> None:
    def basic_screen_blocked(config: dict[str, Any], trade_date: str) -> dict[str, Any]:
        P.bootstrap(config)
        root = P._stage_root(config, "basic_screen", trade_date)
        success = P._stage_success(config, "basic_screen", trade_date)
        contract = P._contract_hash(config)
        meta_path = root / "meta.json"
        if success.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("contract_hash") == contract:
                return {
                    "trade_date": trade_date,
                    "stage": "basic_screen",
                    "status": "skipped",
                }

        started = time.perf_counter()
        root.mkdir(parents=True, exist_ok=True)
        block_root = root / "blocks"
        block_root.mkdir(parents=True, exist_ok=True)
        support, labels, label_masks, _, universes = P._load_materialized(
            config, trade_date
        )
        label_columns = P._screen_label_columns(config, labels)
        universe_names = list(
            config.get("selection", {}).get(
                "screen_universes",
                ["all_pit_eligible", "final_trading_universe"],
            )
        )
        min_n = int(config["run"].get("min_cross_section_n", 30))
        inventory = P._factor_block_inventory(config, trade_date)
        if inventory.empty:
            raise RuntimeError(f"no factor blocks available for {trade_date}")
        materialize_meta_path = (
            P._stage_root(config, "materialize", trade_date) / "meta.json"
        )
        materialize_meta = (
            json.loads(materialize_meta_path.read_text(encoding="utf-8"))
            if materialize_meta_path.exists()
            else {}
        )
        upstream = materialize_meta.get("upstream_fingerprint")
        parts: list[pd.DataFrame] = []
        reused_blocks = 0
        computed_blocks = 0
        rank_cache_hits = 0
        rank_cache_misses = 0

        for block_number, (path_text, group) in enumerate(
            inventory.groupby("path", sort=False)
        ):
            path = Path(str(path_text))
            feature_columns = list(dict.fromkeys(str(value) for value in group["feature"]))
            block_id = hashlib.sha256(
                (path.as_posix() + "\n" + "\n".join(feature_columns)).encode("utf-8")
            ).hexdigest()[:16]
            output_path = block_root / f"block={block_number:03d}_{block_id}.parquet"
            block_meta_path = output_path.with_suffix(".json")
            block_contract = P._json_hash(
                {
                    "stage_contract": contract,
                    "upstream_fingerprint": upstream,
                    "factor_source": _file_contract(path),
                    "features": feature_columns,
                    "labels": label_columns,
                    "universes": universe_names,
                    "min_n": min_n,
                    "kernel": "rank-mask-cache-v2.8.1",
                }
            )
            if output_path.exists() and block_meta_path.exists():
                block_meta = json.loads(
                    block_meta_path.read_text(encoding="utf-8")
                )
                if (
                    block_meta.get("status") == "complete"
                    and block_meta.get("contract_hash") == block_contract
                ):
                    parts.append(pd.read_parquet(output_path))
                    reused_blocks += 1
                    continue

            block = pd.read_parquet(path, columns=feature_columns)
            block.index = support.index
            block_rows: list[dict[str, Any]] = []
            for universe in universe_names:
                if universe not in universes:
                    continue
                universe_mask = universes[universe].fillna(False)
                cache: dict[
                    bytes,
                    tuple[pd.DataFrame, pd.DataFrame, pd.Series],
                ] = {}
                for label_column in label_columns:
                    base_mask = (
                        universe_mask
                        & labels[label_column].notna()
                        & label_masks[label_column].fillna(False)
                    ).fillna(False)
                    if int(base_mask.sum()) < min_n:
                        continue
                    key = _mask_key(base_mask)
                    cached = cache.get(key)
                    if cached is None:
                        factor_frame = block.loc[base_mask]
                        ranked = P.C._rank_frame_by_datetime_average(factor_frame)
                        coverage = factor_frame.notna().mean()
                        cache[key] = (factor_frame, ranked, coverage)
                        rank_cache_misses += 1
                    else:
                        factor_frame, ranked, coverage = cached
                        rank_cache_hits += 1
                    label = labels.loc[base_mask, label_column]
                    minute_stats, pooled_stats = P.C._ranked_ic_stats_from_ranked_features(
                        ranked,
                        label,
                        feature_columns,
                        min_n,
                    )
                    family, horizon_text = label_column.rsplit("__h", 1)
                    for method, stats in (
                        ("minute_mean_cs_rank_ic", minute_stats),
                        ("pooled_cs_demeaned_pct_rank_ic", pooled_stats),
                    ):
                        for feature in feature_columns:
                            item = stats[feature]
                            block_rows.append(
                                {
                                    "trade_date": trade_date,
                                    "feature": feature,
                                    "factor_family": P._factor_family(feature),
                                    "universe": universe,
                                    "label_family": family,
                                    "horizon_bars": int(horizon_text),
                                    "rank_ic_method": method,
                                    "rank_ic_mean": item["rank_ic_mean"],
                                    "rank_ic_std": item["rank_ic_std"],
                                    "rank_ic_positive_ratio": item[
                                        "rank_ic_positive_ratio"
                                    ],
                                    "ic_minutes": item["ic_minutes"],
                                    "ic_count": item["ic_count"],
                                    "coverage": float(
                                        coverage.get(feature, np.nan)
                                    ),
                                }
                            )
                cache.clear()
            part = pd.DataFrame(block_rows)
            P._atomic_parquet(part, output_path, index=False)
            P._atomic_json(
                block_meta_path,
                {
                    "status": "complete",
                    "contract_hash": block_contract,
                    "features": feature_columns,
                    "rows": int(len(part)),
                },
            )
            parts.append(part)
            computed_blocks += 1
            del block
            gc.collect()

        result = (
            pd.concat(parts, ignore_index=True, copy=False)
            if parts
            else pd.DataFrame()
        )
        P._atomic_parquet(
            result,
            root / "basic_factor_screen.parquet",
            index=False,
        )
        meta = {
            "version": P.VERSION,
            "trade_date": trade_date,
            "stage": "basic_screen",
            "status": "complete",
            "contract_hash": contract,
            "rows": int(len(result)),
            "feature_count": int(result["feature"].nunique()) if not result.empty else 0,
            "factor_blocks": int(len(parts)),
            "computed_blocks": computed_blocks,
            "reused_blocks": reused_blocks,
            "rank_cache_hits": rank_cache_hits,
            "rank_cache_misses": rank_cache_misses,
            "elapsed_seconds": time.perf_counter() - started,
        }
        P._atomic_json(meta_path, meta)
        success.write_text(pd.Timestamp.utcnow().isoformat(), encoding="utf-8")
        return meta

    P.basic_screen_date = basic_screen_blocked
