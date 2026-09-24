"""Unit tests for flip7.experiment.eval."""

from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import cast

import pytest

from flip7.agents import RandomLegalAgent
from flip7.envs import ObservationFamily
from flip7.experiment.eval import run_standard_evaluations


def _root() -> dict[str, object]:
    return {
        "evaluation": {
            "seed_bases": [10000, 11000, 12000],
            "heldout_seed_bases": [13000, 14000],
            "matchups": [["random", "threshold"], ["risk", "ev"], ["dp", "threshold"]],
            "heldout_matchups": [
                ["threshold_high", "risk_strict"],
                ["ev_penalized", "dp_reduced"],
            ],
        }
    }


def test_run_standard_evaluations_covers_all_seats(tmp_path: Path) -> None:
    baseline, heldout = run_standard_evaluations(
        _root(),
        tmp_path / "unused.pt",
        games=1,
        seed_offset=0,
        observation=ObservationFamily.BASIC,
        policy_factory=lambda: RandomLegalAgent(seed=11),
    )
    assert len(cast(list[object], baseline["matchups"])) == 9
    assert len(cast(list[object], heldout["matchups"])) == 6
    baseline_summary = cast(dict[str, object], baseline["summary"])
    assert baseline_summary["games"] == 9
    assert set(cast(dict[str, object], baseline_summary["per_seat"])) == {
        "0",
        "1",
        "2",
    }
    assert 0.0 <= float(cast(float, baseline_summary["win_share"])) <= 1.0


def test_run_standard_evaluations_is_deterministic(tmp_path: Path) -> None:
    first, _ = run_standard_evaluations(
        _root(),
        tmp_path / "unused.pt",
        games=1,
        seed_offset=7_0000,
        observation=ObservationFamily.BASIC,
        policy_factory=lambda: RandomLegalAgent(seed=11),
    )
    second, _ = run_standard_evaluations(
        _root(),
        tmp_path / "unused.pt",
        games=1,
        seed_offset=7_0000,
        observation=ObservationFamily.BASIC,
        policy_factory=lambda: RandomLegalAgent(seed=11),
    )
    assert first["summary"] == second["summary"]


def _root_with_workers(workers: int) -> dict[str, object]:
    root = _root()
    evaluation = cast(dict[str, object], root["evaluation"])
    evaluation["workers"] = workers
    return root


def test_explicit_workers_overrides_config(tmp_path: Path) -> None:
    """Explicit workers=1 stays serial even when config requests parallel."""
    baseline, _ = run_standard_evaluations(
        _root_with_workers(4),
        tmp_path / "missing.pt",
        games=1,
        seed_offset=0,
        observation=ObservationFamily.BASIC,
        policy_factory=lambda: RandomLegalAgent(seed=11),
        workers=1,
    )
    assert cast(dict[str, object], baseline["summary"])["games"] == 9


def test_default_workers_reads_config(tmp_path: Path) -> None:
    """Omitted workers falls back to evaluation.workers (here: parallel).

    The parallel path loads the checkpoint in worker initializers, so a
    missing file kills the pool instead of running serially.
    """
    with pytest.raises(BrokenProcessPool):
        run_standard_evaluations(
            _root_with_workers(2),
            tmp_path / "missing.pt",
            games=1,
            seed_offset=0,
            observation=ObservationFamily.BASIC,
            policy_factory=lambda: RandomLegalAgent(seed=11),
        )
