from __future__ import annotations

import hashlib
import json

from nff_research import full_factor_engine as FF
from nff_research import v2_7_prototype_contract as CONTRACT
from nff_research import v2_7_run as RUN


def test_implementation_snapshot_contains_all_194_unique_prototypes() -> None:
    rows, source_sha = CONTRACT._implementation_rows(FF)
    ids = [row["prototype_id"] for row in rows]
    assert len(rows) == CONTRACT.EXPECTED_PROTOTYPES == 194
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids, key=lambda value: (value[0], int(value[1:])))
    assert all(row["family"] == row["prototype_id"][0] for row in rows)
    assert all(row["contract_source"] == "IMPLEMENTATION_SNAPSHOT" for row in rows)
    assert len(source_sha) == 64
    int(source_sha, 16)


def test_v27_import_uses_portable_contract_without_local_markdown() -> None:
    prototypes = RUN.C.V26.PROTOTYPES
    manifest = RUN.C.FF.contract_source_manifest()
    assert len(prototypes) == 194
    assert manifest["source"] in {"IMPLEMENTATION_SNAPSHOT", "AUTHORITATIVE_MARKDOWN"}
    assert len(manifest["contract_sha256"]) == 64
    int(manifest["contract_sha256"], 16)
    assert [row["prototype_id"] for row in prototypes] == manifest["prototype_ids"]


def test_implementation_contract_hash_is_deterministic() -> None:
    rows, source_sha = CONTRACT._implementation_rows(FF)
    payload = {
        "contract_version": CONTRACT.CONTRACT_VERSION,
        "source": "IMPLEMENTATION_SNAPSHOT",
        "implementation_sha256": source_sha,
        "prototype_ids": [row["prototype_id"] for row in rows],
    }
    first = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    second = FF.instruction_hash()
    manifest = FF.contract_source_manifest()
    if manifest["source"] == "IMPLEMENTATION_SNAPSHOT":
        assert first == second == manifest["contract_sha256"]
