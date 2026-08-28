"""Focused tests for seat-balanced stability training."""

from pathlib import Path
from typing import cast

import numpy as np

from flip7.envs import ObservationFamily
from flip7.training import (
    DiversePolicyLeague,
    FollowUpLeagueConfig,
    StabilityLeaguePPOTrainer,
    balanced_seat_quotas,
)
from flip7.training.ppo import PPOConfig


def test_balanced_seat_quotas_are_deterministic_and_maximally_equal() -> None:
    assert balanced_seat_quotas(1026, 3) == (342, 342, 342)
    assert balanced_seat_quotas(1024, 3) == (342, 341, 341)


def test_stability_rollout_records_equal_seat_transitions(tmp_path: Path) -> None:
    trainer = StabilityLeaguePPOTrainer(
        PPOConfig(
            seed=31,
            observation="basic",
            learner_seat_mode="random",
            rollout_steps=6,
            updates=1,
            epochs=1,
            minibatch_size=3,
            hidden_size=8,
        ),
        league_config=FollowUpLeagueConfig(
            warmup_updates=1,
            archive_interval=1,
            max_population=2,
            baseline_names=("random",),
            state_bank_size=4,
            response_signature_games=0,
        ),
    )
    history = trainer.train(tmp_path / "checkpoint.pt")

    assert len(history) == 1
    quotas = cast(list[int], trainer.rollout_schedule[0]["seat_quotas"])
    assert quotas == [2, 2, 2]
    assert sum(quotas) == 6
    assert trainer.league.warmup_anchor is not None
    assert all(np.isfinite(value) for value in history[0].values())


def test_opponent_curriculum_is_zero_during_warmup() -> None:
    league = FollowUpLeagueConfig(
        warmup_updates=10,
        learned_opponent_probability_start=0.25,
        learned_opponent_probability_end=0.60,
        learned_opponent_ramp_updates=25,
        baseline_names=("random",),
        state_bank_size=2,
    )
    policy_league = DiversePolicyLeague(
        league,
        observation=ObservationFamily.BASIC,
        source_seed=7,
        player_count=3,
    )
    policy_league.set_training_update(10)
    assert policy_league.learned_probability == 0.0
    policy_league.set_training_update(35)
    assert policy_league.learned_probability == 0.60
