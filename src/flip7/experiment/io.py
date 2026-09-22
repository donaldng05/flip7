"""Filesystem, JSON serialization, and cryptographic hashing helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast


def write_json(path: Path, value: Mapping[str, object]) -> None:
    """Write formatted, sorted-key JSON, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest of a file in 1MB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_stage_summary(
    output_root: Path, stage: str, payload: Mapping[str, object]
) -> None:
    """Write a stage summary and update the aggregate summary.json file."""
    write_json(output_root / f"{stage}-summary.json", payload)
    summary_path = output_root / "summary.json"
    all_stages: dict[str, object] = {}
    if summary_path.is_file():
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            existing = cast(dict[str, object], raw)
            stages = existing.get("stages")
            if isinstance(stages, dict):
                all_stages.update(cast(dict[str, object], stages))
    all_stages[stage] = dict(payload)
    experiment_name = str(payload.get("experiment", "experiment"))
    write_json(
        summary_path,
        {"experiment": experiment_name, "stages": all_stages},
    )
