"""Tests for the masked PyTorch learned policy."""

import numpy as np
import torch

from flip7.agents.learned import ActorCritic, PPOAgent, masked_logits


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
