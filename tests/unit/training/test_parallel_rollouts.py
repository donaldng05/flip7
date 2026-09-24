"""Unit tests for parallel PPO rollout collection."""

from __future__ import annotations

import random
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from flip7.training.league_followup import FollowUpLeagueConfig
from flip7.training.ppo import (
    EpisodeLineup,
    PPOConfig,
    PPOTrainer,
    Rollout,
    balanced_quotas,
    compute_gae,
)
from flip7.training.stability import (
    BalancedBaselineProvider,
    SeatBalancedPPOTrainer,
    StabilityLeaguePPOTrainer,
    balanced_seat_quotas,
    partition_seat_tasks,
)


def test_balanced_quotas() -> None:
    assert balanced_quotas(256, 4) == (64, 64, 64, 64)
    assert balanced_quotas(257, 4) == (65, 64, 64, 64)
    assert balanced_quotas(258, 4) == (65, 65, 64, 64)
    assert balanced_quotas(10, 3) == (4, 3, 3)

    with pytest.raises(ValueError, match="steps must be positive"):
        balanced_quotas(0, 4)
    with pytest.raises(ValueError, match="count must be positive"):
        balanced_quotas(256, 0)


def test_partition_seat_tasks() -> None:
    quotas = balanced_seat_quotas(256, 3)  # (86, 85, 85)
    tasks_3 = partition_seat_tasks(quotas, 3)
    assert len(tasks_3) == 3
    assert tasks_3 == [(0, 86), (1, 85), (2, 85)]

    tasks_4 = partition_seat_tasks(quotas, 4)
    assert len(tasks_4) == 4
    # Seat 0 gets split into 43 and 43, while seats 1 and 2 remain 85
    assert sum(steps for seat, steps in tasks_4 if seat == 0) == 86
    assert sum(steps for seat, steps in tasks_4 if seat == 1) == 85
    assert sum(steps for seat, steps in tasks_4 if seat == 2) == 85
    assert sum(steps for _, steps in tasks_4) == 256

    tasks_6 = partition_seat_tasks(quotas, 6)
    assert len(tasks_6) == 6
    assert sum(steps for _, steps in tasks_6) == 256

    with pytest.raises(ValueError, match="quotas must not be empty"):
        partition_seat_tasks((), 3)


def test_rollout_workers_validation() -> None:
    with pytest.raises(ValueError, match="rollout_workers must be at least 1"):
        PPOTrainer(PPOConfig(rollout_workers=0))
    with pytest.raises(ValueError, match="rollout_workers cannot exceed rollout_steps"):
        PPOTrainer(PPOConfig(rollout_steps=10, rollout_workers=20))


def test_rollout_workers_1_matches_serial_bit_identically() -> None:
    config_serial = PPOConfig(seed=123, rollout_steps=64, rollout_workers=1)
    trainer_serial = PPOTrainer(config_serial)
    rollout_serial = trainer_serial.collect_rollout()
    metrics_serial = trainer_serial.update(rollout_serial)

    config_default = PPOConfig(seed=123, rollout_steps=64)
    trainer_default = PPOTrainer(config_default)
    rollout_default = trainer_default.collect_rollout()
    metrics_default = trainer_default.update(rollout_default)

    np.testing.assert_array_equal(
        rollout_serial.observations, rollout_default.observations
    )
    np.testing.assert_array_equal(rollout_serial.masks, rollout_default.masks)
    np.testing.assert_array_equal(rollout_serial.actions, rollout_default.actions)
    np.testing.assert_array_equal(
        rollout_serial.old_log_probs, rollout_default.old_log_probs
    )
    np.testing.assert_array_equal(rollout_serial.rewards, rollout_default.rewards)
    np.testing.assert_array_equal(rollout_serial.dones, rollout_default.dones)
    np.testing.assert_array_equal(rollout_serial.values, rollout_default.values)
    assert rollout_serial.next_value == pytest.approx(rollout_default.next_value)

    for key in metrics_serial:
        assert metrics_serial[key] == pytest.approx(metrics_default[key])


@pytest.mark.parametrize("workers", [2, 4])
def test_parallel_rollout_multi_worker_shapes_and_gae(workers: int) -> None:
    config = PPOConfig(
        seed=42,
        rollout_steps=64,
        rollout_workers=workers,
        updates=2,
    )
    trainer = PPOTrainer(config)
    rollout = trainer.collect_rollout()

    assert rollout.observations.shape == (64, trainer.observation_size)
    assert rollout.masks.shape == (64, trainer.action_size)
    assert rollout.actions.shape == (64,)
    assert rollout.rewards.shape == (64,)
    assert rollout.dones.shape == (64,)
    assert rollout.values.shape == (64,)
    assert np.all(np.isfinite(rollout.observations))
    assert np.all(np.isfinite(rollout.values))
    assert np.all(np.isfinite(rollout.old_log_probs))

    assert rollout.segment_ends is not None
    assert rollout.segment_next_values is not None
    assert sum(1 for end in rollout.segment_ends if end) == workers

    advantages, returns = compute_gae(
        rollout.rewards,
        rollout.dones,
        rollout.values,
        rollout.next_value,
        config.gamma,
        config.gae_lambda,
        segment_ends=rollout.segment_ends,
        segment_next_values=rollout.segment_next_values,
    )
    assert advantages.shape == (64,)
    assert returns.shape == (64,)
    assert np.all(np.isfinite(advantages))
    assert np.all(np.isfinite(returns))

    metrics = trainer.update(rollout)
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])
    assert metrics["updates"] > 0


@pytest.mark.parametrize("workers", [3, 4])
def test_seat_balanced_parallel_rollout(workers: int) -> None:
    config = PPOConfig(
        seed=99,
        rollout_steps=60,
        player_count=3,
        rollout_workers=workers,
    )
    provider = BalancedBaselineProvider(("random", "risk"))
    trainer = SeatBalancedPPOTrainer(
        config,
        opponent_provider=lambda rng, count: provider.episode_lineup(rng, count),
        seat_opponent_provider=provider.episode_lineup_for_seat,
    )

    rollout = trainer.collect_rollout()

    assert rollout.observations.shape == (60, trainer.observation_size)
    assert rollout.seat_ids is not None
    expected_quotas = balanced_seat_quotas(60, 3)  # (20, 20, 20)
    assert trainer.last_rollout_seat_counts == expected_quotas
    assert len(trainer.last_rollout_reset_seeds) > 0

    # Count transitions for each seat
    for seat, quota in enumerate(expected_quotas):
        assert sum(1 for s in rollout.seat_ids if s == seat) == quota

    assert rollout.segment_ends is not None
    assert rollout.segment_next_values is not None
    # Number of segment ends equals number of partition tasks
    tasks = partition_seat_tasks(expected_quotas, workers)
    assert sum(1 for end in rollout.segment_ends if end) == len(tasks)

    metrics = trainer.update(rollout)
    assert np.isfinite(metrics["policy_loss"])


def _assert_rollouts_equal(first: object, second: object) -> None:
    """Assert two Rollout objects are bit-identical across reruns."""
    assert isinstance(first, Rollout)
    assert isinstance(second, Rollout)
    np.testing.assert_array_equal(first.observations, second.observations)
    np.testing.assert_array_equal(first.masks, second.masks)
    np.testing.assert_array_equal(first.actions, second.actions)
    np.testing.assert_array_equal(first.old_log_probs, second.old_log_probs)
    np.testing.assert_array_equal(first.rewards, second.rewards)
    np.testing.assert_array_equal(first.dones, second.dones)
    np.testing.assert_array_equal(first.values, second.values)
    assert first.next_value == second.next_value
    assert first.seat_ids is not None and second.seat_ids is not None
    np.testing.assert_array_equal(first.seat_ids, second.seat_ids)
    assert first.segment_ends is not None and second.segment_ends is not None
    np.testing.assert_array_equal(first.segment_ends, second.segment_ends)
    assert (
        first.segment_next_values is not None and second.segment_next_values is not None
    )
    np.testing.assert_array_equal(first.segment_next_values, second.segment_next_values)


def test_parallel_rollout_rerun_determinism() -> None:
    """Same seed + same worker count must reproduce bit-identical rollouts."""
    first = PPOTrainer(PPOConfig(seed=2024, rollout_steps=64, rollout_workers=4))
    second = PPOTrainer(PPOConfig(seed=2024, rollout_steps=64, rollout_workers=4))
    _assert_rollouts_equal(first.collect_rollout(), second.collect_rollout())


def test_seat_balanced_parallel_rerun_determinism() -> None:
    """Seat-balanced parallel rollouts must also reproduce bit-identically."""
    provider = BalancedBaselineProvider(("random", "risk"))

    def _make_trainer() -> SeatBalancedPPOTrainer:
        return SeatBalancedPPOTrainer(
            PPOConfig(seed=31337, rollout_steps=60, rollout_workers=4),
            opponent_provider=lambda rng, count: provider.episode_lineup(rng, count),
            seat_opponent_provider=provider.episode_lineup_for_seat,
        )

    _assert_rollouts_equal(
        _make_trainer().collect_rollout(), _make_trainer().collect_rollout()
    )


def test_league_parallel_rollout_merges_exposure_deterministically(
    tmp_path: Path,
) -> None:
    """League+parallel must pickle safely and merge worker exposure counts."""
    snapshot_path = tmp_path / "snap.pt"
    PPOTrainer(
        PPOConfig(
            seed=5,
            observation="basic",
            rollout_steps=8,
            updates=1,
            epochs=1,
            minibatch_size=4,
            hidden_size=8,
        ),
        opponent_names=("random",),
    ).train(snapshot_path)

    def _make_trainer() -> StabilityLeaguePPOTrainer:
        trainer = StabilityLeaguePPOTrainer(
            PPOConfig(
                seed=11,
                rollout_steps=30,
                rollout_workers=2,
                observation="basic",
                hidden_size=8,
            ),
            league_config=FollowUpLeagueConfig(state_bank_size=32),
        )
        trainer.league.register_snapshot(snapshot_path, update=10, seed=5)
        return trainer

    first, second = _make_trainer(), _make_trainer()
    before = sum(first.league.exposure.values())
    _assert_rollouts_equal(first.collect_rollout(), second.collect_rollout())

    assert first.league.exposure == second.league.exposure
    merged = sum(first.league.exposure.values()) - before
    # Every created env fills 2 opponent slots; a dropped merge would read 0.
    assert merged == 2 * len(first.last_rollout_reset_seeds) > 0
    for key in first.league.exposure:
        assert key.startswith(("baseline:", "snapshot:"))


def test_gae_chunk_boundaries_prevent_cross_chunk_leakage() -> None:
    """Segmented GAE must equal per-chunk GAE concatenated (no leakage)."""
    rewards = np.array(
        [
            0.1,
            -0.2,
            0.3,
            0.0,
            0.5,
            -0.1,
            0.2,
            0.4,
            -0.3,
            0.0,
            0.1,
            0.6,
            0.2,
            -0.4,
            0.3,
            0.1,
        ],
        dtype=np.float32,
    )
    dones = np.zeros(16, dtype=np.bool_)
    values = np.array(
        [
            0.5,
            0.4,
            0.6,
            0.3,
            0.2,
            0.7,
            0.1,
            0.5,
            0.4,
            0.3,
            0.2,
            0.6,
            0.5,
            0.4,
            0.3,
            0.2,
        ],
        dtype=np.float32,
    )
    bootstraps = [1.0, -0.5, 0.25, 0.0]
    segment_ends = np.zeros(16, dtype=np.bool_)
    segment_next_values = np.zeros(16, dtype=np.float32)
    for chunk in range(4):
        end = chunk * 4 + 3
        segment_ends[end] = True
        segment_next_values[end] = np.float32(bootstraps[chunk])

    joint_advantages, joint_returns = compute_gae(
        rewards,
        dones,
        values,
        bootstraps[-1],
        0.99,
        0.95,
        segment_ends=segment_ends,
        segment_next_values=segment_next_values,
    )
    chunk_advantages: list[np.ndarray] = []
    chunk_returns: list[np.ndarray] = []
    for chunk in range(4):
        sl = slice(chunk * 4, chunk * 4 + 4)
        adv, ret = compute_gae(
            rewards[sl], dones[sl], values[sl], bootstraps[chunk], 0.99, 0.95
        )
        chunk_advantages.append(adv)
        chunk_returns.append(ret)
    np.testing.assert_array_equal(joint_advantages, np.concatenate(chunk_advantages))
    np.testing.assert_array_equal(joint_returns, np.concatenate(chunk_returns))

    # Sensitivity check: without boundaries the result must differ.
    unsegmented_advantages, _ = compute_gae(
        rewards, dones, values, bootstraps[-1], 0.99, 0.95
    )
    assert not np.allclose(unsegmented_advantages, joint_advantages)


def test_persistent_worker_pool_lifecycle(tmp_path: Path) -> None:
    config = PPOConfig(
        seed=77,
        rollout_steps=32,
        updates=2,
        minibatch_size=16,
        rollout_workers=2,
    )
    trainer = PPOTrainer(config)
    assert trainer.worker_pool is None

    checkpoint = tmp_path / "checkpoint.pt"
    history = trainer.train(checkpoint)

    assert trainer.worker_pool is None
    assert len(history) == 2
    assert checkpoint.is_file()


def test_parallel_rollout_rejects_unpicklable_provider() -> None:
    def _local_provider(rng: random.Random, count: int) -> EpisodeLineup:
        # Nested scope makes this unpicklable while remaining callable.
        return BalancedBaselineProvider(("random",)).episode_lineup(rng, count)

    with pytest.raises(ValueError, match="must be picklable"):
        PPOTrainer(
            PPOConfig(seed=1, rollout_steps=16, rollout_workers=2),
            opponent_provider=_local_provider,
        )

    # Serial construction stays permissive: the check is parallel-gated.
    PPOTrainer(
        PPOConfig(seed=1, rollout_steps=16),
        opponent_provider=_local_provider,
    )


def test_seat_balanced_parallel_rejects_two_arg_provider() -> None:
    provider = BalancedBaselineProvider(("random", "threshold"))
    two_arg_seat_provider = partial(provider.episode_lineup, learner_id=0)
    trainer = SeatBalancedPPOTrainer(
        PPOConfig(seed=2, rollout_steps=6, rollout_workers=2),
        opponent_provider=provider.episode_lineup,
        seat_opponent_provider=two_arg_seat_provider,
    )
    with pytest.raises(ValueError, match="requested_learner_id"):
        trainer.collect_rollout()


def test_stability_league_train_archives_with_serial_signatures(
    tmp_path: Path,
) -> None:
    config = PPOConfig(
        seed=9,
        rollout_steps=6,
        updates=2,
        hidden_size=8,
        observation="basic",
        minibatch_size=4,
    )
    league_config = FollowUpLeagueConfig(
        warmup_updates=1,
        archive_interval=1,
        state_bank_size=8,
        baseline_names=("random", "threshold"),
        response_signature_games=1,
        learned_opponent_probability=0.0,
    )
    trainer = StabilityLeaguePPOTrainer(config, league_config=league_config)
    history = trainer.train(tmp_path / "ckpt.pt")
    assert len(history) == 2
    assert len(trainer.league.archived_snapshots) == 2
    assert len(trainer.league.response_signatures) == 2
