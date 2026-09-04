"""Tests for PPO rollout math, updates, and checkpoint restoration."""

from pathlib import Path

import numpy as np
import pytest
import torch

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
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["config"]["learner_seat_mode"] == "fixed"

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


def test_fixed_learner_seat_mode_uses_configured_seat() -> None:
    trainer = PPOTrainer(
        PPOConfig(seed=3, learner_id=2, learner_seat_mode="fixed"),
        opponent_names=("random",),
    )

    assert trainer.new_env().learner_id == 2


def test_random_learner_seat_mode_is_reproducible_and_covers_all_seats() -> None:
    config = PPOConfig(seed=23, learner_seat_mode="random")
    first = PPOTrainer(config, opponent_names=("random",))
    second = PPOTrainer(config, opponent_names=("random",))

    first_ids = [first.new_env().learner_id for _ in range(32)]
    second_ids = [second.new_env().learner_id for _ in range(32)]

    assert first_ids == second_ids
    assert set(first_ids) == {0, 1, 2}


@pytest.mark.parametrize(
    ("observation", "size"),
    [("basic", 33), ("competitive", 96), ("deck_aware", 120)],
)
def test_ppo_supports_each_phase6_observation_family(
    observation: str, size: int
) -> None:
    trainer = PPOTrainer(
        PPOConfig(seed=5, observation=observation),
        opponent_names=("random",),
    )

    assert trainer.observation_size == size


def test_ppo_rejects_unknown_learner_seat_mode() -> None:
    with pytest.raises(ValueError, match="learner_seat_mode"):
        PPOTrainer(PPOConfig(learner_seat_mode="rotated"), opponent_names=("random",))


def test_basic_observation_can_train_against_deck_aware_baselines(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "basic.pt"
    trainer = PPOTrainer(
        PPOConfig(
            seed=29,
            observation="basic",
            rollout_steps=8,
            updates=1,
            epochs=1,
            minibatch_size=4,
            hidden_size=16,
        )
    )

    history = trainer.train(checkpoint)

    assert checkpoint.exists()
    assert len(history) == 1
    assert all(np.isfinite(value) for value in history[0].values())


def test_stable_ppo_recipe_has_conditioned_critic_and_schedules(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "stable.pt"
    config = PPOConfig(
        seed=31,
        rollout_steps=9,
        updates=2,
        epochs=1,
        minibatch_size=3,
        hidden_size=16,
        network="separate",
        critic_seat_conditioned=True,
        learning_rate=1e-4,
        learning_rate_end=5e-5,
        entropy_coefficient=0.03,
        entropy_coefficient_end=0.005,
        target_kl=0.02,
        value_clip_epsilon=0.2,
    )
    trainer = PPOTrainer(config, opponent_names=("random",))

    history = trainer.train(checkpoint)
    rollout = trainer.collect_rollout()

    assert rollout.seat_ids is not None
    assert set(rollout.seat_ids.tolist()) == {0}
    assert history[0]["learning_rate"] == pytest.approx(1e-4)
    assert history[-1]["learning_rate"] == pytest.approx(5e-5)
    assert history[0]["entropy_coefficient"] == pytest.approx(0.03)
    assert history[-1]["entropy_coefficient"] == pytest.approx(0.005)
    assert all(np.isfinite(value) for row in history for value in row.values())

    restored = PPOTrainer(config, opponent_names=("random",))
    assert restored.load_checkpoint(checkpoint) == 2


def test_stable_ppo_rejects_conditioned_critic_with_shared_network() -> None:
    with pytest.raises(ValueError, match="separate network"):
        PPOTrainer(
            PPOConfig(critic_seat_conditioned=True),
            opponent_names=("random",),
        )
