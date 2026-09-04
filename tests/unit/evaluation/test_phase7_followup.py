"""Focused tests for follow-up diversity and paired tournament primitives."""

from pathlib import Path

import numpy as np
import pytest

from flip7.agents import FixedThresholdAgent, PPOAgent, RandomLegalAgent
from flip7.envs import ObservationFamily
from flip7.evaluation import (
    TournamentParticipant,
    build_paired_schedule,
    build_state_bank,
    jensen_shannon_divergence,
    mean_jensen_shannon_divergence,
    policy_behavior,
    run_paired_round_robin,
)
from flip7.training import (
    DiversePolicyLeague,
    FollowUpLeagueConfig,
    PolicySnapshot,
    PPOConfig,
    PPOTrainer,
)


def _checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "policy.pt"
    PPOTrainer(
        PPOConfig(
            seed=3,
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


def test_state_bank_and_js_are_deterministic_and_nonnegative() -> None:
    first = build_state_bank(states=12, seed=101)
    second = build_state_bank(states=12, seed=101)

    assert first.as_dict() == second.as_dict()
    assert all(
        np.array_equal(left.observation, right.observation)
        and np.array_equal(left.action_mask, right.action_mask)
        for left, right in zip(first.entries, second.entries, strict=True)
    )
    assert jensen_shannon_divergence(
        np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)
    ) == pytest.approx(np.log(2.0))
    assert mean_jensen_shannon_divergence(
        np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
        np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32),
    ) == pytest.approx(np.log(2.0) / 2.0)


def test_behavior_analysis_supports_seat_conditioned_checkpoints(
    tmp_path: Path,
) -> None:
    config = PPOConfig(
        seed=5,
        observation="basic",
        rollout_steps=4,
        updates=1,
        epochs=1,
        minibatch_size=2,
        hidden_size=8,
        network="separate",
        critic_seat_conditioned=True,
    )
    trainer = PPOTrainer(config, opponent_names=("random",))
    checkpoint = tmp_path / "conditioned.pt"
    trainer.save_checkpoint(checkpoint, update=1)
    snapshot = PolicySnapshot(
        "conditioned",
        checkpoint,
        1,
        5,
        ObservationFamily.BASIC,
        trainer.observation_size,
        trainer.action_size,
        {},
    )

    behavior = policy_behavior(snapshot, build_state_bank(states=4, seed=9))

    assert behavior.probabilities.shape == (4, trainer.action_size)
    assert len(behavior.deterministic_actions) == 4


def test_followup_league_preserves_anchor_and_retains_all_archives(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    league = DiversePolicyLeague(
        FollowUpLeagueConfig(
            warmup_updates=1,
            archive_interval=1,
            max_population=3,
            retention_strategy="novelty",
            state_bank_size=8,
        ),
        observation=ObservationFamily.BASIC,
        source_seed=3,
        player_count=3,
    )
    for update in range(1, 7):
        league.register_snapshot(checkpoint, update=update, seed=3)

    assert len(league.archived_snapshots) == 6
    assert league.warmup_anchor is not None
    assert league.warmup_anchor.update == 1
    assert len(league.snapshots) == 3
    assert league.as_dict()["archived_snapshots"]


def test_novelty_retention_uses_per_state_distributions(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    league = DiversePolicyLeague(
        FollowUpLeagueConfig(
            warmup_updates=1,
            archive_interval=1,
            max_population=4,
            retention_strategy="novelty",
            state_bank_size=8,
        ),
        observation=ObservationFamily.BASIC,
        source_seed=3,
        player_count=3,
    )

    for update in range(1, 7):
        league.register_snapshot(checkpoint, update=update, seed=3)

    assert len(league.snapshots) == 4


def test_temporal_and_latest_retention_are_deterministic(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    temporal = DiversePolicyLeague(
        FollowUpLeagueConfig(
            warmup_updates=2,
            max_population=3,
            retention_strategy="temporal",
            state_bank_size=4,
        ),
        observation=ObservationFamily.BASIC,
        source_seed=3,
        player_count=3,
    )
    latest = DiversePolicyLeague(
        FollowUpLeagueConfig(
            warmup_updates=2,
            max_population=1,
            retention_strategy="latest",
            state_bank_size=4,
        ),
        observation=ObservationFamily.BASIC,
        source_seed=3,
        player_count=3,
    )
    for update in range(1, 7):
        temporal.register_snapshot(checkpoint, update=update, seed=3)
        latest.register_snapshot(checkpoint, update=update, seed=3)

    assert [snapshot.update for snapshot in temporal.snapshots] == [1, 3, 6]
    assert [snapshot.update for snapshot in latest.snapshots] == [6]


def test_paired_schedule_and_tournament_are_order_independent() -> None:
    participants = (
        TournamentParticipant(
            "random", lambda: RandomLegalAgent(seed=1), ObservationFamily.DECK_AWARE
        ),
        TournamentParticipant(
            "threshold",
            lambda: FixedThresholdAgent(threshold=15),
            ObservationFamily.DECK_AWARE,
        ),
        TournamentParticipant(
            "random2", lambda: RandomLegalAgent(seed=2), ObservationFamily.DECK_AWARE
        ),
    )
    schedule = build_paired_schedule(
        ("random2", "threshold", "random"), games=1, seed=77
    )
    first, first_schedule = run_paired_round_robin(
        participants, games=1, seed=77, schedule=schedule
    )
    second, second_schedule = run_paired_round_robin(
        tuple(reversed(participants)), games=1, seed=77
    )

    assert first_schedule == second_schedule
    assert first.as_dict() == second.as_dict()
    assert first.games == 6


def test_parallel_tournament_matches_serial_results(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    participants = (
        TournamentParticipant(
            "random", lambda: RandomLegalAgent(seed=101), ObservationFamily.DECK_AWARE
        ),
        TournamentParticipant(
            "threshold",
            lambda: FixedThresholdAgent(threshold=15),
            ObservationFamily.DECK_AWARE,
        ),
        TournamentParticipant(
            "final",
            lambda: PPOAgent.from_checkpoint(checkpoint, deterministic=True),
            ObservationFamily.BASIC,
        ),
    )
    serial, schedule = run_paired_round_robin(participants, games=1, seed=77)
    parallel, parallel_schedule = run_paired_round_robin(
        participants,
        games=1,
        seed=77,
        schedule=schedule,
        workers=2,
        checkpoint_paths={"final": checkpoint},
    )

    assert parallel_schedule == schedule
    assert parallel.as_dict() == serial.as_dict()
