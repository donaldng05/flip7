"""Unit tests for parallel PPO rollout collection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flip7.training.ppo import (
    PPOConfig,
    PPOTrainer,
    balanced_quotas,
    compute_gae,
)
from flip7.training.stability import (
    BalancedBaselineProvider,
    SeatBalancedPPOTrainer,
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
