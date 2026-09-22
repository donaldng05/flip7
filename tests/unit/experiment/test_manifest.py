"""Unit tests for flip7.experiment.manifest."""

import json
from pathlib import Path

from flip7.experiment.io import sha256_file
from flip7.experiment.manifest import is_run_complete


def test_is_run_complete(tmp_path: Path) -> None:
    ckpt: Path = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"dummy model bytes")
    digest = sha256_file(ckpt)

    # Empty/missing files
    manifest_path: Path = tmp_path / "manifest.json"
    assert is_run_complete(manifest_path, ckpt) is False

    # Incomplete keys
    manifest_path.write_text(json.dumps({"checkpoint": str(ckpt)}), encoding="utf-8")
    assert is_run_complete(manifest_path, ckpt) is False

    # Valid followup manifest
    files: dict[str, str] = {}
    for key in (
        "checkpoint",
        "training_history",
        "population",
        "diversity",
        "evaluation",
        "tournament",
        "manifest",
    ):
        f: Path = tmp_path / f"{key}.json"
        f.write_text("{}", encoding="utf-8")
        files[key] = str(f)
    files["checkpoint"] = str(ckpt)
    files["checkpoint_sha256"] = digest

    manifest_path.write_text(json.dumps(files), encoding="utf-8")
    assert is_run_complete(manifest_path, ckpt, followup=True) is True

    # Checkpoint hash mismatch
    files["checkpoint_sha256"] = "wronghash"
    manifest_path.write_text(json.dumps(files), encoding="utf-8")
    assert is_run_complete(manifest_path, ckpt, followup=True) is False
