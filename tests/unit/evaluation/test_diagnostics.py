"""Tests for Phase 7 diagnostic validation and classification helpers."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from flip7.evaluation.diagnostics import (
    aggregate_seed_summaries,
    bootstrap_rotated_uncertainty,
    classify_diagnostic,
    compare_seed_spreads,
    reference_reproduces,
    resolve_manifest_files,
    validate_seed_plan,
)
from flip7.evaluation.metrics import MatchupMetrics


def _summary(
    seats: tuple[float, float, float],
    *,
    games: int = 300,
    seat_margin: float = 0.01,
    spread_interval: tuple[float, float] | None = None,
) -> dict[str, object]:
    win_share = sum(seats) / len(seats)
    spread = max(seats) - min(seats)
    return {
        "games": games,
        "win_share": win_share,
        "approximate_95_ci": [win_share - seat_margin, win_share + seat_margin],
        "seat_spread": spread,
        "per_seat": {
            str(seat): {
                "win_share": value,
                "approximate_95_ci": [value - seat_margin, value + seat_margin],
            }
            for seat, value in enumerate(seats)
        },
        "uncertainty": {
            "method": "normal_summary_approximation",
            "seat_spread_interval": list(
                spread_interval
                or (max(0.0, spread - seat_margin), spread + seat_margin)
            ),
        },
    }


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


def test_reference_reproduction_uses_seed_level_uncertainty() -> None:
    historical = {
        7: _summary((0.60, 0.68, 0.68), seat_margin=0.05),
        17: _summary((0.70, 0.72, 0.62), seat_margin=0.05),
    }
    fresh = {
        7: _summary((0.61, 0.69, 0.68), seat_margin=0.02),
        17: _summary((0.69, 0.70, 0.63), seat_margin=0.02),
    }
    assert reference_reproduces(historical, fresh, tolerance=0.01)

    seed_level_win_share_failure = {
        7: _summary((0.65, 0.65, 0.65), seat_margin=0.01),
        17: _summary((0.40, 0.40, 0.40), seat_margin=0.01),
    }
    assert not reference_reproduces(
        historical, seed_level_win_share_failure, tolerance=0.01
    )

    pooled_pass_seed_fail = {
        7: _summary((0.55, 0.65, 0.75), seat_margin=0.01),
        17: _summary((0.55, 0.65, 0.75), seat_margin=0.01),
    }
    assert not all(
        float(cast(float, summary["seat_spread"])) <= 0.05
        for summary in pooled_pass_seed_fail.values()
    )


def test_seed_aggregation_and_relative_spread_classification() -> None:
    reference = {7: _summary((0.62, 0.66, 0.71), seat_margin=0.005)}
    better = {7: _summary((0.64, 0.66, 0.68), seat_margin=0.005)}
    worse = {7: _summary((0.57, 0.67, 0.73), seat_margin=0.005)}
    compatible = {7: _summary((0.61, 0.67, 0.72), seat_margin=0.02)}

    assert aggregate_seed_summaries([reference[7]])["seat_spread"] == pytest.approx(
        0.09
    )
    assert compare_seed_spreads(reference, better)[0]["relation"] == "better"
    assert compare_seed_spreads(reference, worse)[0]["relation"] == "worse"
    assert compare_seed_spreads(reference, compatible)[0]["relation"] == "compatible"


def test_bootstrap_is_stratified_by_learner_seat_and_reproducible() -> None:
    results = [
        MatchupMetrics(
            ("ppo", "a", "b"),
            3,
            win_share_samples={"ppo": [1.0, 0.0, 1.0]},
        ),
        MatchupMetrics(
            ("a", "ppo", "b"),
            3,
            win_share_samples={"ppo": [0.0, 1.0, 0.0]},
        ),
        MatchupMetrics(
            ("a", "b", "ppo"),
            3,
            win_share_samples={"ppo": [0.0, 0.0, 1.0]},
        ),
    ]
    first = bootstrap_rotated_uncertainty(results, seed=11, replicates=200)
    second = bootstrap_rotated_uncertainty(results, seed=11, replicates=200)

    assert first == second
    assert first["method"] == "stratified_game_bootstrap"
    assert set(cast(Mapping[str, object], first["seat_intervals"])) == {"0", "1", "2"}
    assert len(cast(Sequence[object], first["seat_spread_interval"])) == 2


def test_diagnostic_classification_distinguishes_reference_recipe_and_noise() -> None:
    passing = [{"seed": 7, "relation": "better"}]
    uncertain = [{"seed": 7, "relation": "compatible"}]
    failing = [{"seed": 7, "relation": "worse"}]
    assert (
        classify_diagnostic(
            reference_valid=False,
            reference_is_robust=False,
            control_seed_comparisons=failing,
        )
        == "reference-recalibration-required"
    )
    assert (
        classify_diagnostic(
            reference_valid=True,
            reference_is_robust=True,
            control_seed_comparisons=failing,
        )
        == "recipe-failure"
    )
    assert (
        classify_diagnostic(
            reference_valid=True,
            reference_is_robust=True,
            control_seed_comparisons=uncertain,
        )
        == "evaluation-noise/inconclusive"
    )
    assert (
        classify_diagnostic(
            reference_valid=True,
            reference_is_robust=True,
            control_seed_comparisons=passing,
        )
        == "protocol-valid"
    )
    assert (
        classify_diagnostic(
            reference_valid=True,
            reference_is_robust=False,
            control_seed_comparisons=failing,
        )
        == "reference-not-robust / candidate-comparison-only"
    )
