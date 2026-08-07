from __future__ import annotations

"""Portable prototype-contract bootstrap for v2.7.

The original full-factor engine reads a user-local Markdown instruction at
module-import time.  v2.7 keeps that document authoritative whenever it is
available, but CI and detached machines must still be able to import and audit
the executable implementation.  The fallback below is generated
*deterministically* from the actual ``full_factor_engine.py`` implementation:
all quoted A01..Kxx prototype ids, their first source lines and the complete
implementation SHA are captured in the registry contract.

This fallback does not pretend to reproduce unavailable prose titles.  Its
source is explicitly reported as ``IMPLEMENTATION_SNAPSHOT``; a local or
environment-provided instruction remains ``AUTHORITATIVE_MARKDOWN``.
"""

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


EXPECTED_PROTOTYPES = 194
CONTRACT_VERSION = "v2.7-portable-prototype-contract-v1"
ENV_PATH = "NFF_FACTOR_INSTRUCTION_PATH"
_ID_PATTERN = re.compile(r"[\"']([A-K]\d{2})[\"']")


def _implementation_rows(engine: Any) -> tuple[list[dict[str, Any]], str]:
    source_path = Path(engine.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    first_line: dict[str, int] = {}
    for line_number, line in enumerate(source.splitlines(), start=1):
        for prototype_id in _ID_PATTERN.findall(line):
            first_line.setdefault(prototype_id, line_number)
    ids = sorted(first_line, key=lambda value: (value[0], int(value[1:])))
    if len(ids) != EXPECTED_PROTOTYPES:
        by_family = {
            family: sum(prototype_id.startswith(family) for prototype_id in ids)
            for family in "ABCDEFGHIJK"
        }
        raise RuntimeError(
            "implementation prototype snapshot drift: "
            f"expected={EXPECTED_PROTOTYPES}, actual={len(ids)}, by_family={by_family}"
        )
    rows = [
        {
            "prototype_id": prototype_id,
            "family": prototype_id[0],
            "title": f"Executable implementation {prototype_id}",
            "formula": (
                f"implementation_source_sha256={source_sha}; "
                f"prototype_id={prototype_id}; source=full_factor_engine.derive_prototype"
            ),
            "source_tokens": ["IMPLEMENTATION_SNAPSHOT", prototype_id],
            "instruction_line": first_line[prototype_id],
            "contract_source": "IMPLEMENTATION_SNAPSHOT",
            "contract_version": CONTRACT_VERSION,
        }
        for prototype_id in ids
    ]
    return rows, source_sha


def _requested_path(engine: Any, requested: Path | str | None = None) -> Path | None:
    if requested is not None:
        candidate = Path(requested).expanduser()
        return candidate if candidate.exists() else None
    environment = os.environ.get(ENV_PATH)
    if environment:
        candidate = Path(environment).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(
                f"{ENV_PATH} points to a missing factor instruction: {candidate}"
            )
        return candidate
    candidate = Path(engine.INSTRUCTION_PATH).expanduser()
    return candidate if candidate.exists() else None


def install(engine: Any) -> None:
    """Install portable parse/hash functions before v2.6 imports prototypes."""

    if getattr(engine, "_V27_PORTABLE_CONTRACT_INSTALLED", False):
        return
    original_parse = engine.parse_prototypes
    original_hash = engine.instruction_hash
    implementation_rows, implementation_sha = _implementation_rows(engine)
    implementation_payload = {
        "contract_version": CONTRACT_VERSION,
        "source": "IMPLEMENTATION_SNAPSHOT",
        "implementation_sha256": implementation_sha,
        "prototype_ids": [row["prototype_id"] for row in implementation_rows],
    }
    implementation_contract_sha = hashlib.sha256(
        json.dumps(
            implementation_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    def parse_prototypes(path: Path | str | None = None) -> list[dict[str, Any]]:
        resolved = _requested_path(engine, path)
        if resolved is None:
            engine.PROTOTYPE_CONTRACT_SOURCE = {
                **implementation_payload,
                "contract_sha256": implementation_contract_sha,
                "resolved_path": None,
            }
            return [dict(row) for row in implementation_rows]
        rows = original_parse(resolved)
        if len(rows) != EXPECTED_PROTOTYPES:
            raise RuntimeError(
                f"authoritative prototype count drift: expected={EXPECTED_PROTOTYPES}, actual={len(rows)}, path={resolved}"
            )
        document_sha = original_hash(resolved)
        for row in rows:
            row["contract_source"] = "AUTHORITATIVE_MARKDOWN"
            row["contract_version"] = CONTRACT_VERSION
        engine.PROTOTYPE_CONTRACT_SOURCE = {
            "contract_version": CONTRACT_VERSION,
            "source": "AUTHORITATIVE_MARKDOWN",
            "document_sha256": document_sha,
            "contract_sha256": document_sha,
            "resolved_path": str(resolved.resolve()),
            "prototype_ids": [row["prototype_id"] for row in rows],
        }
        return rows

    def instruction_hash(path: Path | str | None = None) -> str:
        resolved = _requested_path(engine, path)
        if resolved is None:
            return implementation_contract_sha
        return original_hash(resolved)

    def contract_source_manifest() -> dict[str, Any]:
        # Ensure source metadata exists even when instruction_hash is called
        # before parse_prototypes by a future consumer.
        if not getattr(engine, "PROTOTYPE_CONTRACT_SOURCE", None):
            parse_prototypes()
        return dict(engine.PROTOTYPE_CONTRACT_SOURCE)

    engine.parse_prototypes = parse_prototypes
    engine.instruction_hash = instruction_hash
    engine.contract_source_manifest = contract_source_manifest
    engine._V27_PORTABLE_CONTRACT_INSTALLED = True
