"""Tests for PPO rollout math, updates, and checkpoint restoration."""

from pathlib import Path

import numpy as np

from flip7.training import PPOConfig, PPOTrainer, compute_gae


def test_compute_gae_resets_at_terminal_transitions() -> None:
    advantages, returns = compute_gae(
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([True, False], dtype=np.bool_),
        np.array([0.0, 0.0], dtype=np.float32),
        0.0,
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert np.allclose(advantages, [1.0, 2.0])
    assert np.allclose(returns, [1.0, 2.0])


def _small_config() -> PPOConfig:
    return PPOConfig(
        seed=13,
        rollout_steps=8,
        updates=1,
        epochs=1,
        minibatch_size=4,
        hidden_size=16,
    )


def test_bounded_ppo_update_and_checkpoint_round_trip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    trainer = PPOTrainer(_small_config(), opponent_names=("random",))
    history = trainer.train(checkpoint)

    assert len(history) == 1
    assert all(np.isfinite(value) for value in history[0].values())
    assert checkpoint.exists()

    restored_trainer = PPOTrainer(_small_config(), opponent_names=("random",))
    assert restored_trainer.load_checkpoint(checkpoint) == 1
    assert restored_trainer.optimizer.state_dict()["state"]

    from flip7.agents import FixedThresholdAgent, PPOAgent, RandomLegalAgent
    from flip7.evaluation import run_matchup

    restored = PPOAgent.from_checkpoint(checkpoint)
    observation, info = trainer.new_env().reset(seed=3)
    action = restored(observation, info["action_mask"])
    assert info["action_mask"][action] == 1

    metrics = run_matchup(
        ("ppo", "random", "threshold"),
        {
            "ppo": lambda: PPOAgent.from_checkpoint(checkpoint),
            "random": lambda: RandomLegalAgent(seed=4),
            "threshold": lambda: FixedThresholdAgent(threshold=15),
        },
        games=1,
        seed=29,
    )
    assert metrics.games == 1
