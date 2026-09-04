"""Focused MAPPO pilot tests."""

from pathlib import Path

import numpy as np
import torch

from flip7.envs import ObservationFamily
from flip7.training import (
    CentralizedCritic,
    MAPPOAgent,
    MAPPOConfig,
    MAPPOTrainer,
    SharedActor,
)


def test_centralized_critic_input_dimension_and_finite_smoke(tmp_path: Path) -> None:
    trainer = MAPPOTrainer(
        MAPPOConfig(
            seed=9,
            hidden_size=8,
            rollout_steps=8,
            updates=1,
            epochs=1,
            minibatch_size=4,
        )
    )
    assert trainer.observation_size == 33
    assert trainer.critic_input_size == 103
    assert isinstance(trainer.critic, CentralizedCritic)
    rollout = trainer.collect_rollout()
    metrics = trainer.update(rollout)
    assert rollout.observations.shape == (8, 33)
    assert rollout.global_observations.shape == (8, 103)
    assert all(np.isfinite(value) for value in metrics.values())

    checkpoint = tmp_path / "mappo.pt"
    trainer.save_checkpoint(checkpoint, 1)
    assert MAPPOAgent.from_checkpoint(checkpoint)(
        rollout.observations[0], rollout.masks[0]
    ) in np.flatnonzero(rollout.masks[0])


def test_shared_actor_respects_action_mask() -> None:
    actor = SharedActor(33, 5, 8)
    policy = MAPPOAgent(actor, deterministic=True)
    observation = np.zeros(33, dtype=np.float32)
    mask = np.array([0, 1, 0, 0, 0], dtype=np.int8)

    assert policy(observation, mask) == 1
    with torch.no_grad():
        assert torch.isfinite(actor(torch.zeros((1, 33)))).all()


def test_mappo_requires_basic_observation() -> None:
    try:
        MAPPOConfig(observation=ObservationFamily.COMPETITIVE.value)
    except ValueError as error:
        assert "basic" in str(error)
    else:
        raise AssertionError("competitive MAPPO input should be rejected")
