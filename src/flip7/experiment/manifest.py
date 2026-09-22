"""Artifact manifest validation and resume verification helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from flip7.evaluation.follow_up import validate_followup_manifest
from flip7.evaluation.phase7 import validate_artifact_manifest
from flip7.experiment.io import sha256_file


def is_run_complete(
    manifest_path: Path,
    checkpoint_path: Path,
    *,
    followup: bool = True,
) -> bool:
    """Verify that a previous run's manifest is valid and matches checkpoint hash."""
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return False
        manifest = cast(dict[str, object], raw)
        if followup:
            validate_followup_manifest(manifest)
        else:
            validate_artifact_manifest(manifest)
        expected_sha = manifest.get("checkpoint_sha256")
        return bool(
            expected_sha is None or expected_sha == sha256_file(checkpoint_path)
        )
    except Exception:
        return False


__all__ = [
    "is_run_complete",
    "validate_artifact_manifest",
    "validate_followup_manifest",
]
