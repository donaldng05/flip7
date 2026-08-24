"""Tests for the Gymnasium vs-opponents Flip 7 wrapper."""

import random

import numpy as np
import pytest

from flip7.core.cards import Card, NumberCard
from flip7.envs.encodings import STAY_INDEX, agent_name
from flip7.envs.vs_opponents import Flip7VsOpponentsEnv, RandomLegalPolicy


def _pile(*values: int) -> tuple[Card, ...]:
    return tuple(NumberCard(value) for value in values)


def test_random_legal_policy_samples_only_from_the_mask() -> None:
    policy = RandomLegalPolicy(random.Random(0))
    mask = np.array([0, 1, 0, 1, 0], dtype=np.int8)
    observation = np.zeros(4, dtype=np.float32)

    actions = {policy(observation, mask) for _ in range(20)}

    assert actions <= {1, 3}


def test_wrapper_rejects_incomplete_opponent_maps() -> None:
    with pytest.raises(ValueError, match="opponents must include"):
        Flip7VsOpponentsEnv(
            player_count=3,
            learner_id=0,
            opponents={"player_1": RandomLegalPolicy(random.Random(0))},
        )


def test_wrapper_rejects_invalid_learner_id() -> None:
    with pytest.raises(ValueError, match="learner_id must reference"):
        Flip7VsOpponentsEnv(player_count=3, learner_id=3)


def test_random_legal_policy_rejects_an_empty_mask() -> None:
    policy = RandomLegalPolicy(random.Random(0))
    with pytest.raises(RuntimeError, match="no legal actions remain"):
        policy(np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.int8))


def test_reset_advances_to_the_learner_seat() -> None:
    def always_stay(_obs: np.ndarray, _mask: np.ndarray) -> int:
        del _obs, _mask
        return STAY_INDEX

    env = Flip7VsOpponentsEnv(
        player_count=3,
        learner_id=0,
        opponents={
            "player_1": always_stay,
            "player_2": always_stay,
        },
    )
    observation, info = env.reset(
        seed=0, options={"draw_pile": _pile(4, 8, 6, 10, 11, 12)}
    )

    assert env.aec_env.agent_selection == "player_0"
    assert observation.shape == env.observation_space.shape
    assert int(info["action_mask"].sum()) >= 1


def test_wrapper_with_random_opponents_completes_a_seeded_game() -> None:
    env = Flip7VsOpponentsEnv(player_count=3, learner_id=0)
    observation, info = env.reset(seed=7)
    rng = random.Random(3)
    terminated = False
    steps = 0

    while not terminated:
        legal = np.flatnonzero(info["action_mask"])
        observation, _reward, terminated, truncated, info = env.step(
            int(rng.choice(legal))
        )
        assert truncated is False
        steps += 1
        assert steps < 5_000

    assert terminated is True
    assert env.aec_env.engine.state.is_game_terminal is True
    assert observation.shape == env.observation_space.shape


def test_wrapper_unique_win_reward_reaches_the_learner() -> None:
    env = Flip7VsOpponentsEnv(
        player_count=3,
        learner_id=0,
        opponents={
            agent_name(1): lambda _obs, _mask: STAY_INDEX,
            agent_name(2): lambda _obs, _mask: STAY_INDEX,
        },
    )
    _obs, info = env.reset(
        seed=0,
        options={
            "draw_pile": _pile(1, 2, 12),
            "starting_scores": (199, 0, 0),
        },
    )
    assert env.aec_env.agent_selection == "player_0"
    _obs, reward, terminated, _truncated, _info = env.step(STAY_INDEX)

    assert terminated is True
    assert reward == 1.0
    del info
