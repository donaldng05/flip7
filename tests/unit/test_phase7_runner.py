"""Tests for Phase 7 experiment artifact validation."""

from pathlib import Path

import pytest
from scripts.run_phase7 import MANIFEST_PATH_KEYS, validate_artifact_manifest


def test_phase7_artifact_manifest_requires_all_run_files(tmp_path: Path) -> None:
    manifest: dict[str, object] = {}
    for key in MANIFEST_PATH_KEYS:
        path = tmp_path / f"{key}.json"
        path.write_text("{}\n", encoding="utf-8")
        manifest[key] = str(path)

    validate_artifact_manifest(manifest)

    missing_path = tmp_path / "missing.json"
    manifest["tournament"] = str(missing_path)
    with pytest.raises(ValueError, match="tournament"):
        validate_artifact_manifest(manifest)
