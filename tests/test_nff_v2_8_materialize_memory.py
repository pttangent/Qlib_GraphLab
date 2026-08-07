from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from nff_research import v2_8_materialize_memory as M
from nff_research import v2_8_stage_contracts as CONTRACTS


def _fake_pipeline(tmp_path: Path):
    events = []
    context = SimpleNamespace(
        root=tmp_path / "atomic",
        factor_block_size=1,
        contract_hash="run-contract",
        trade_date="2026-01-02",
    )
    ff = SimpleNamespace(
        factor_name=lambda prototype_id, window: f"factor__{prototype_id.lower()}__w{window}",
        _DERIVE_CACHE={},
    )

    def atomic_parquet(frame: pd.DataFrame, path: Path, index: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=index)

    def atomic_json(path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, default=str), encoding="utf-8")

    values = {
        "A01": pd.Series([1.0, 2.0, 3.0, 4.0]),
        "A02": pd.Series([2.0, 4.0, 6.0, 8.0]),
    }
    c = SimpleNamespace(
        CTX=context,
        FF=ff,
        _factor_contract=lambda group, frame: "factor-contract",
        _derive_corrected=lambda frame, prototype_id, window: values.copy(),
        _atomic_parquet=atomic_parquet,
        _atomic_json=atomic_json,
        _event=lambda stage, state, **extra: events.append((stage, state, extra)),
    )
    specs = pd.DataFrame(
        [
            {
                "family": "A",
                "window": "10m",
                "prototype_id": "A01",
                "factor_id": "factor__a01__w10m",
            },
            {
                "family": "A",
                "window": "10m",
                "prototype_id": "A02",
                "factor_id": "factor__a02__w10m",
            },
        ]
    )
    p = SimpleNamespace(C=c, V26=SimpleNamespace(SPEC_REGISTRY=specs))
    return p, events


def test_streaming_factor_blocks_are_not_concatenated_back(tmp_path: Path) -> None:
    p, _ = _fake_pipeline(tmp_path)
    index = pd.MultiIndex.from_product(
        [["A"], pd.date_range("2026-01-02 14:30", periods=4, freq="min")],
        names=["instrument", "datetime"],
    )
    frame = pd.DataFrame({"base": np.arange(4, dtype="float32")}, index=index)
    result, runtime = M._streaming_derive_all(p, frame, [])

    assert result is frame
    assert list(result.columns) == ["base"]
    assert set(runtime) == {"factor__a01__w10m", "factor__a02__w10m"}
    manifest = (
        tmp_path
        / "atomic"
        / "factors"
        / "family=A"
        / "window=10m"
        / "manifest.json"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["materialization_mode"] == "stream_to_checkpoint_no_wide_concat"
    assert len(payload["blocks"]) == 2


def test_streaming_materialize_uses_preserved_non_strict_feature_builder() -> None:
    expected = object()
    strict = lambda frame: (_ for _ in ()).throw(
        AssertionError("strict physical gate must not run on streamed frame")
    )
    base = lambda frame: expected
    pipeline = SimpleNamespace(
        C=SimpleNamespace(
            _base_add_all_features=base,
            add_all_features=strict,
        )
    )

    assert M._streaming_feature_builder(pipeline)(pd.DataFrame()) is expected


def test_streaming_completion_audit_records_block_authority() -> None:
    audit = M._streaming_completion_audit(
        {"f1", "f2"},
        pd.DataFrame({"feature": ["f1", "f2"]}),
        wide_frame_columns=3,
    )

    assert audit["materialized_columns"] == 2
    assert audit["absent_columns"] == []
    assert audit["materialization_mode"] == "stream_factor_blocks"
    assert audit["wide_frame_columns"] == 3


def test_checkpoint_reuse_uses_parquet_metadata_not_dataframe_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    p, _ = _fake_pipeline(tmp_path)
    index = pd.MultiIndex.from_product(
        [["A"], pd.date_range("2026-01-02 14:30", periods=4, freq="min")],
        names=["instrument", "datetime"],
    )
    frame = pd.DataFrame({"base": np.arange(4, dtype="float32")}, index=index)
    M._streaming_derive_all(p, frame, [])

    p.C._derive_corrected = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("completed factor group must not be recomputed")
    )
    monkeypatch.setattr(
        M.pd,
        "read_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed factor blocks must not be materialized on reuse")
        ),
    )
    result, runtime = M._streaming_derive_all(p, frame, [])
    assert result is frame
    assert set(runtime) == {"factor__a01__w10m", "factor__a02__w10m"}


def test_peak_gate_trims_before_releasing_slot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        M.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=96 * 1024**3),
    )
    p = SimpleNamespace(
        _pipeline_root=lambda config: tmp_path,
        C=SimpleNamespace(_event=lambda *args, **kwargs: None),
    )
    config = {
        "pipeline": {
            "materialize_internal": {
                "peak_slots": 1,
                "peak_entry_min_available_gb": 48,
                "gate_poll_seconds": 0.01,
            }
        }
    }
    M._ACTIVE_LEASE = None
    lease = M._acquire_peak_lease(p, config, "2026-01-02")
    assert lease.path.exists()
    assert lease.slot == 0

    observed = {}

    def trim_while_owned():
        observed["lease_exists_during_trim"] = lease.path.exists()
        observed["active_during_trim"] = M._ACTIVE_LEASE is lease
        return {"rss_before_gb": 2.0, "rss_after_gb": 1.0}

    monkeypatch.setattr(M, "_memory_trim", trim_while_owned)
    M._release_peak_lease(p, "test")
    assert observed == {
        "lease_exists_during_trim": True,
        "active_during_trim": True,
    }
    assert not lease.path.exists()
    assert M._ACTIVE_LEASE is None


def test_operational_materialize_memory_settings_do_not_change_semantic_contract() -> None:
    p = SimpleNamespace(
        VERSION="2.8-pipelined-selection",
        C=SimpleNamespace(VERSION="2.7-real-schema-atomic"),
        V26=SimpleNamespace(FULL_FACTOR_NAMES=["f1", "f2"]),
    )
    base = {
        "run": {
            "start_date": "2026-01-02",
            "end_date": "2026-07-22",
            "horizons": [1, 5],
            "min_cross_section_n": 30,
        },
        "local_paths": {"warehouse_root": "D:/warehouse"},
        "study": {},
        "factor_resolution_policy": {},
        "atomic": {"factor_block_size": 8, "intra_date_workers": 4},
        "labels": {"exact_elapsed_minutes": True},
        "pipeline": {
            "controls_path": "D:/controls.parquet",
            "materialize_internal": {"peak_slots": 1},
            "stages": {
                "materialize": {
                    "max_workers": 4,
                    "estimated_worker_gb": 16,
                }
            },
        },
    }
    changed = json.loads(json.dumps(base))
    changed["pipeline"]["materialize_internal"]["peak_slots"] = 2
    changed["pipeline"]["stages"]["materialize"]["estimated_worker_gb"] = 30
    assert CONTRACTS.contract_payload(
        p,
        base,
        "materialize",
    ) == CONTRACTS.contract_payload(
        p,
        changed,
        "materialize",
    )
