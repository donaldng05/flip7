"""Tests verifying worker-invariance and error contracts for parallel evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from flip7.agents import PPOAgent, RandomLegalAgent
from flip7.envs import ObservationFamily
from flip7.evaluation import (
    run_matchup,
    run_paired_rotated_evaluation,
    run_rotated_matchups,
    summarize_rotated_results,
)
from flip7.experiment import run_standard_evaluations
from flip7.training import PPOConfig, PPOTrainer, baseline_factories


def _create_checkpoint(target_dir: Path, seed: int = 3) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "policy.pt"
    PPOTrainer(
        PPOConfig(
            seed=seed,
            observation="basic",
            rollout_steps=4,
            updates=1,
            epochs=1,
            minibatch_size=2,
            hidden_size=8,
        ),
        opponent_names=("random",),
    ).train(path)
    return path


def test_run_matchup_baseline_worker_invariance() -> None:
    roster = baseline_factories()
    lineup = ("random", "threshold", "random")
    w1 = run_matchup(lineup, roster, games=4, seed=42, workers=1)
    w2 = run_matchup(lineup, roster, games=4, seed=42, workers=2)
    assert w1.as_dict() == w2.as_dict()
    assert w1.games == 4


def test_run_matchup_checkpoint_worker_invariance(tmp_path: Path) -> None:
    ckpt = _create_checkpoint(tmp_path / "matchup_agent")
    roster = baseline_factories() | {
        "ppo": lambda: PPOAgent.from_checkpoint(ckpt, deterministic=True),
    }
    lineup = ("random", "ppo", "random")
    w1 = run_matchup(
        lineup,
        roster,
        games=4,
        seed=101,
        observation=ObservationFamily.BASIC,
        workers=1,
        checkpoint_paths={"ppo": ckpt},
    )
    w2 = run_matchup(
        lineup,
        roster,
        games=4,
        seed=101,
        observation=ObservationFamily.BASIC,
        workers=2,
        checkpoint_paths={"ppo": ckpt},
    )
    assert w1.as_dict() == w2.as_dict()


def test_run_matchup_requires_checkpoint_paths_for_unknown_agents() -> None:
    roster = {
        "custom_ppo": lambda: RandomLegalAgent(seed=1),
    }
    with pytest.raises(ValueError, match=r"checkpoint_path.*required"):
        run_matchup(
            ("custom_ppo", "custom_ppo", "custom_ppo"),
            roster,
            games=2,
            seed=1,
            workers=2,
        )


def test_run_rotated_matchups_worker_invariance(tmp_path: Path) -> None:
    ckpt = _create_checkpoint(tmp_path / "rotated_agent")
    policy = PPOAgent.from_checkpoint(ckpt, deterministic=True)
    matchups = (("random", "threshold"), ("risk", "ev"))
    seed_bases = (1000, 2000)

    w1_results = run_rotated_matchups(
        lambda: policy,
        baseline_factories(),
        games=2,
        seed_bases=seed_bases,
        observation=ObservationFamily.BASIC,
        matchups=matchups,
        workers=1,
        checkpoint_path=ckpt,
    )
    w4_results = run_rotated_matchups(
        lambda: policy,
        baseline_factories(),
        games=2,
        seed_bases=seed_bases,
        observation=ObservationFamily.BASIC,
        matchups=matchups,
        workers=4,
        checkpoint_path=ckpt,
    )

    assert len(w1_results) == len(w4_results)
    assert [r.as_dict() for r in w1_results] == [r.as_dict() for r in w4_results]
    assert summarize_rotated_results(w1_results) == summarize_rotated_results(
        w4_results
    )


def test_run_rotated_matchups_requires_checkpoint_path_for_workers() -> None:
    with pytest.raises(ValueError, match="checkpoint_path is required"):
        run_rotated_matchups(
            lambda: RandomLegalAgent(seed=1),
            baseline_factories(),
            games=1,
            seed_bases=(100,),
            observation=ObservationFamily.BASIC,
            matchups=(("random", "threshold"),),
            workers=2,
            checkpoint_path=None,
        )


def test_run_paired_rotated_evaluation_worker_invariance(tmp_path: Path) -> None:
    ckpt1 = _create_checkpoint(tmp_path / "p1", seed=7)
    ckpt2 = _create_checkpoint(tmp_path / "p2", seed=17)
    first_policy = PPOAgent.from_checkpoint(ckpt1, deterministic=True)
    second_policy = PPOAgent.from_checkpoint(ckpt2, deterministic=True)
    matchups = (("random", "threshold"),)
    seed_bases = (5000,)

    w1 = run_paired_rotated_evaluation(
        lambda: first_policy,
        lambda: second_policy,
        baseline_factories(),
        games=2,
        seed_bases=seed_bases,
        observation=ObservationFamily.BASIC,
        matchups=matchups,
        workers=1,
        checkpoint_paths=(ckpt1, ckpt2),
    )
    w4 = run_paired_rotated_evaluation(
        lambda: first_policy,
        lambda: second_policy,
        baseline_factories(),
        games=2,
        seed_bases=seed_bases,
        observation=ObservationFamily.BASIC,
        matchups=matchups,
        workers=4,
        checkpoint_paths=(ckpt1, ckpt2),
    )

    assert w1.as_dict() == w4.as_dict()


def test_run_paired_rotated_evaluation_requires_checkpoint_paths_for_workers() -> None:
    with pytest.raises(ValueError, match=r"checkpoint_path.*required"):
        run_paired_rotated_evaluation(
            lambda: RandomLegalAgent(seed=1),
            lambda: RandomLegalAgent(seed=2),
            baseline_factories(),
            games=1,
            seed_bases=(100,),
            observation=ObservationFamily.BASIC,
            matchups=(("random", "threshold"),),
            workers=2,
            checkpoint_paths=None,
        )


def test_run_standard_evaluations_worker_invariance(tmp_path: Path) -> None:
    ckpt = _create_checkpoint(tmp_path / "standard", seed=21)
    root: dict[str, Any] = {
        "evaluation": {
            "games_per_seat": 2,
            "seed_bases": [100],
            "matchups": [["random", "threshold"]],
            "heldout_matchups": [["random", "risk"]],
            "heldout_seed_bases": [500],
        }
    }

    w1_base, w1_held = run_standard_evaluations(
        root,
        ckpt,
        games=2,
        seed_offset=1000,
        observation=ObservationFamily.BASIC,
        workers=1,
    )
    w2_base, w2_held = run_standard_evaluations(
        root,
        ckpt,
        games=2,
        seed_offset=1000,
        observation=ObservationFamily.BASIC,
        workers=2,
    )

    assert w1_base == w2_base
    assert w1_held == w2_held
