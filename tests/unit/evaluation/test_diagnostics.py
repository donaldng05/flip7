"""Tests for Phase 7 diagnostic validation and classification helpers."""

from pathlib import Path

import pytest

from flip7.evaluation.diagnostics import (
    classify_diagnostic,
    reference_reproduces,
    resolve_manifest_files,
    validate_seed_plan,
)


def test_validate_seed_plan_rejects_overlap_and_training_seeds() -> None:
    assert validate_seed_plan(
        [7, 17, 27], [21000, 22000, 23000], games=100, repeats=2, seed_stride=10000
    ) == ((21000, 22000, 23000), (31000, 32000, 33000))

    with pytest.raises(ValueError, match="overlap"):
        validate_seed_plan([7], [100, 150], games=100, repeats=1, seed_stride=1000)

    with pytest.raises(ValueError, match="training"):
        validate_seed_plan([105], [100], games=10, repeats=1, seed_stride=1000)


def test_resolve_manifest_files_validates_missing_files_and_seed(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    history = tmp_path / "training.json"
    checkpoint.write_bytes(b"checkpoint")
    history.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "checkpoint": str(checkpoint),
        "training_history": str(history),
        "seed": 7,
    }

    resolved = resolve_manifest_files(
        manifest,
        manifest_path,
        required_keys=("checkpoint", "training_history"),
        expected_seed=7,
    )
    assert resolved["checkpoint"] == checkpoint

    with pytest.raises(ValueError, match="does not match"):
        resolve_manifest_files(
            manifest,
            manifest_path,
            required_keys=("checkpoint",),
            expected_seed=17,
        )

    with pytest.raises(ValueError, match="does not exist"):
        resolve_manifest_files(
            {"checkpoint": str(tmp_path / "missing.pt")},
            manifest_path,
            required_keys=("checkpoint",),
        )


def test_diagnostic_classification_distinguishes_protocol_recipe_and_noise() -> None:
    good_reference = {"win_share": 0.654, "seat_spread": 0.027}
    assert reference_reproduces(0.654, 0.027, good_reference, tolerance=0.01)
    assert not reference_reproduces(0.60, 0.10, good_reference, tolerance=0.01)

    passing = {"win_share": 0.65, "seat_spread": 0.04}
    failing = {"win_share": 0.65, "seat_spread": 0.08}
    assert (
        classify_diagnostic(
            reference_valid=False,
            control_summary=failing,
            control_batch_summaries=[failing],
        )
        == "protocol-invalid"
    )
    assert (
        classify_diagnostic(
            reference_valid=True,
            control_summary=failing,
            control_batch_summaries=[failing, failing],
        )
        == "recipe-failure"
    )
    assert (
        classify_diagnostic(
            reference_valid=True,
            control_summary=passing,
            control_batch_summaries=[passing, failing],
        )
        == "evaluation-noise/inconclusive"
    )
    assert (
        classify_diagnostic(
            reference_valid=True,
            control_summary=passing,
            control_batch_summaries=[passing, passing],
        )
        == "protocol-valid"
    )
