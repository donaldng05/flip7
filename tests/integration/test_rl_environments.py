"""Integration tests for complete games through the RL adapters."""

import random

import numpy as np

from flip7.envs import Flip7AECEnv, Flip7VsOpponentsEnv, ObservationFamily, RewardMode


def test_aec_and_gymnasium_seeded_games_both_reach_threshold() -> None:
    aec = Flip7AECEnv(
        player_count=4,
        observation=ObservationFamily.COMPETITIVE,
        reward=RewardMode.SPARSE_WIN,
    )
    aec.reset(seed=5)
    rng = random.Random(5)
    for _agent in aec.agent_iter(max_iter=20_000):
        _obs, _reward, terminated, truncated, info = aec.last()
        if terminated or truncated:
            aec.step(None)
            continue
        legal = np.flatnonzero(info["action_mask"])
        aec.step(int(rng.choice(legal)))

    gym_env = Flip7VsOpponentsEnv(
        player_count=4,
        learner_id=1,
        observation=ObservationFamily.DECK_AWARE,
    )
    _obs, info = gym_env.reset(seed=11)
    gym_rng = random.Random(11)
    terminated = False
    while not terminated:
        legal = np.flatnonzero(info["action_mask"])
        _obs, _reward, terminated, truncated, info = gym_env.step(
            int(gym_rng.choice(legal))
        )
        assert truncated is False

    assert aec.engine.state.is_game_terminal is True
    assert max(player.cumulative_score for player in aec.engine.state.players) >= 200
    assert gym_env.aec_env.engine.state.is_game_terminal is True
    assert (
        max(player.cumulative_score for player in gym_env.aec_env.engine.state.players)
        >= 200
    )
