"""Tests for the masked PyTorch learned policy."""

import numpy as np
import pytest
import torch

from flip7.agents.learned import (
    ActorCritic,
    PPOAgent,
    SeparateActorCritic,
    masked_logits,
)


def test_masked_logits_make_illegal_actions_unselectable() -> None:
    logits = torch.tensor([[1.0, 10.0, 2.0]])
    mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)

    result = masked_logits(logits, mask)

    assert torch.isfinite(result[0, 0])
    assert torch.isfinite(result[0, 2])
    assert not torch.isfinite(result[0, 1])


def test_deterministic_policy_always_returns_a_legal_action() -> None:
    network = ActorCritic(4, 5, hidden_size=8)
    policy = PPOAgent(network, deterministic=True)
    observation = np.zeros(4, dtype=np.float32)
    mask = np.array([0, 0, 1, 0, 1], dtype=np.int8)

    assert policy(observation, mask) in {2, 4}


def test_seeded_stochastic_policy_is_reproducible() -> None:
    observation = np.ones(4, dtype=np.float32)
    mask = np.array([1, 0, 1, 0, 1], dtype=np.int8)
    first = PPOAgent(ActorCritic(4, 5, hidden_size=8), seed=17)
    second_network = ActorCritic(4, 5, hidden_size=8)
    second_network.load_state_dict(first.network.state_dict())
    second = PPOAgent(second_network, seed=17)

    assert [first(observation, mask) for _ in range(8)] == [
        second(observation, mask) for _ in range(8)
    ]


def test_separate_actor_critic_conditions_only_the_value_head() -> None:
    network = SeparateActorCritic(4, 5, hidden_size=8, critic_context_size=3)
    observations = torch.zeros((2, 4))
    contexts = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    logits, values = network(observations, contexts)

    assert logits.shape == (2, 5)
    assert values.shape == (2,)
    assert network.actor_logits(observations).shape == (2, 5)
    with pytest.raises(ValueError, match="critic context"):
        network(observations)


def test_separate_checkpoint_policy_inference_does_not_need_critic_context() -> None:
    network = SeparateActorCritic(4, 5, hidden_size=8, critic_context_size=3)
    policy = PPOAgent(network, deterministic=True)

    assert policy(np.zeros(4, dtype=np.float32), np.array([1, 0, 1, 0, 0])) in {
        0,
        2,
    }
