"""Tests for observation-only baseline agents."""

import random

import numpy as np
import pytest

from flip7.agents import (
    BustProbabilityAgent,
    ExpectedValueAgent,
    FixedThresholdAgent,
    RandomLegalAgent,
    RoundDPAgent,
)
from flip7.envs.encodings import HIT_INDEX, STAY_INDEX, TARGET_OFFSET


def _basic_observation(value: int) -> np.ndarray:
    observation = np.zeros(33, dtype=np.float32)
    for number in range(value):
        observation[number] = 1.0
    return observation


def _deck_observation(
    *, held: tuple[int, ...] = (), counts: tuple[int, ...] | None = None
) -> np.ndarray:
    observation = np.zeros(120, dtype=np.float32)
    observation[78] = 1.0  # ego is player 0
    for number in held:
        observation[number] = 1.0
    observation[96:109] = counts or tuple([0] * 13)
    return observation


def test_random_agent_samples_only_legal_actions_deterministically() -> None:
    mask = np.array([0, 1, 0, 1, 0], dtype=np.int8)
    agent = RandomLegalAgent(random.Random(7))
    choices = {agent(np.zeros(33, dtype=np.float32), mask) for _ in range(20)}
    assert choices <= {1, 3}


def test_fixed_threshold_hits_below_and_stays_at_threshold() -> None:
    mask = np.array([1, 1, 0, 0, 0], dtype=np.int8)
    agent = FixedThresholdAgent(threshold=3)
    assert agent(_basic_observation(2), mask) == HIT_INDEX
    assert agent(_basic_observation(3), mask) == STAY_INDEX


def test_deck_aware_agents_reject_non_deck_aware_observations() -> None:
    mask = np.array([1, 1, 0, 0, 0], dtype=np.int8)
    for agent in (BustProbabilityAgent(), ExpectedValueAgent(), RoundDPAgent()):
        with pytest.raises(ValueError, match="deck_aware"):
            agent(np.zeros(96, dtype=np.float32), mask)


def test_bust_probability_agent_stays_when_every_remaining_card_busts() -> None:
    observation = _deck_observation(
        held=(5,), counts=(0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0)
    )
    mask = np.array([1, 1, 0, 0, 0], dtype=np.int8)
    assert BustProbabilityAgent(risk_tolerance=0.5)(observation, mask) == STAY_INDEX


def test_expected_value_agent_hits_for_a_safe_high_value_card() -> None:
    observation = _deck_observation(
        held=(1,), counts=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2)
    )
    mask = np.array([1, 1, 0, 0, 0], dtype=np.int8)
    assert ExpectedValueAgent()(observation, mask) == HIT_INDEX


def test_dp_agent_returns_a_legal_action_on_a_small_fixture() -> None:
    observation = _deck_observation(
        held=(1,), counts=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    )
    mask = np.array([1, 1, 0, 0, 0], dtype=np.int8)
    assert RoundDPAgent(max_unique_numbers=2)(observation, mask) in {
        HIT_INDEX,
        STAY_INDEX,
    }


def test_all_agents_choose_a_legal_target() -> None:
    observation = _deck_observation(counts=tuple([1] * 13))
    observation[92] = 1.0  # Freeze pending
    mask = np.array([0, 0, 1, 1, 0], dtype=np.int8)
    for agent in (
        FixedThresholdAgent(),
        BustProbabilityAgent(),
        ExpectedValueAgent(),
        RoundDPAgent(),
    ):
        assert agent(observation, mask) in {TARGET_OFFSET + 0, TARGET_OFFSET + 1}
