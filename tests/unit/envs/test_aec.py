"""Tests for the PettingZoo Flip 7 AEC environment."""

import random

import numpy as np
import pytest

from flip7.core.actions import HitAction, InvalidActionError, StayAction, TargetAction
from flip7.core.cards import ActionCard, ActionCardName, Card, NumberCard
from flip7.core.engine import Flip7Engine
from flip7.core.state import InvalidPlayerCountError
from flip7.envs.aec import Flip7AECEnv
from flip7.envs.encodings import HIT_INDEX, STAY_INDEX, TARGET_OFFSET, encode_action
from flip7.envs.observations import ObservationFamily


def _pile(*faces: int | ActionCardName) -> tuple[Card, ...]:
    cards: list[Card] = []
    for face in faces:
        if isinstance(face, int):
            cards.append(NumberCard(face))
        else:
            cards.append(ActionCard(face))
    return tuple(cards)


def test_aec_env_rejects_out_of_scope_player_counts() -> None:
    with pytest.raises(InvalidPlayerCountError):
        Flip7AECEnv(player_count=2)


def test_live_agent_cannot_step_with_none() -> None:
    env = Flip7AECEnv(player_count=3)
    env.reset(seed=0, options={"draw_pile": _pile(4, 8, 6)})

    with pytest.raises(ValueError, match="None is only valid"):
        env.step(None)


def test_reset_selects_the_engine_decision_maker_and_matches_legal_mask() -> None:
    env = Flip7AECEnv(player_count=3)
    env.reset(seed=0, options={"draw_pile": _pile(4, 8, 6, 10)})
    engine = Flip7Engine(random.Random(0), 3, draw_pile=_pile(4, 8, 6, 10))

    observation, _reward, terminated, truncated, info = env.last()

    assert env.agent_selection == "player_1"
    assert terminated is False
    assert truncated is False
    assert observation is not None
    assert set(np.flatnonzero(info["action_mask"])) == {
        encode_action(action) for action in engine.legal_actions()
    }
    assert engine.legal_actions() == (
        HitAction(player_id=1),
        StayAction(player_id=1),
    )


def test_step_maps_hit_and_stay_the_same_way_as_the_engine() -> None:
    env = Flip7AECEnv(player_count=3)
    env.reset(seed=0, options={"draw_pile": _pile(4, 8, 6, 10)})
    engine = Flip7Engine(random.Random(0), 3, draw_pile=_pile(4, 8, 6, 10))

    env.step(HIT_INDEX)
    engine.apply(HitAction(player_id=1))

    assert env.engine.state == engine.state
    assert env.agent_selection == "player_2"


def test_seeded_resets_and_actions_reproduce() -> None:
    first = Flip7AECEnv(player_count=3)
    second = Flip7AECEnv(player_count=3)
    first.reset(seed=21)
    second.reset(seed=21)

    while not first.engine.state.is_game_terminal:
        observation, _reward, terminated, truncated, info = first.last()
        del observation
        if terminated or truncated:
            first.step(None)
            second.step(None)
            continue
        action = int(np.flatnonzero(info["action_mask"])[-1])
        first.step(action)
        second.step(action)

    assert first.engine.state == second.engine.state
    assert first.engine.state.is_game_terminal is True


def test_illegal_action_raises_and_does_not_advance_state() -> None:
    env = Flip7AECEnv(player_count=3)
    env.reset(seed=0, options={"draw_pile": _pile(4, 8, 6)})
    before = env.engine.state

    with pytest.raises(InvalidActionError, match="hit or stay"):
        env.step(TARGET_OFFSET)

    assert env.engine.state == before
    assert env.agent_selection == "player_1"


def test_freeze_target_mask_matches_engine_legal_actions() -> None:
    env = Flip7AECEnv(player_count=3)
    env.reset(seed=0, options={"draw_pile": _pile(4, 5, 6, ActionCardName.FREEZE)})
    env.step(HIT_INDEX)

    engine = Flip7Engine(
        random.Random(0), 3, draw_pile=_pile(4, 5, 6, ActionCardName.FREEZE)
    )
    engine.apply(HitAction(player_id=1))

    _obs, _reward, _term, _trunc, info = env.last()
    assert env.agent_selection == "player_1"
    assert set(np.flatnonzero(info["action_mask"])) == {
        encode_action(action) for action in engine.legal_actions()
    }
    assert all(isinstance(action, TargetAction) for action in engine.legal_actions())


def test_sparse_win_reward_is_assigned_to_the_unique_winner() -> None:
    env = Flip7AECEnv(player_count=3, reward="sparse_win")
    env.reset(
        seed=0,
        options={
            "draw_pile": _pile(1, 2, 12),
            "starting_scores": (199, 0, 0),
        },
    )
    env.step(STAY_INDEX)
    env.step(STAY_INDEX)
    env.step(STAY_INDEX)

    assert env.engine.state.is_game_terminal is True
    assert env.engine.state.winning_player_ids == (0,)
    assert env.rewards["player_0"] == 1.0
    assert env.rewards["player_1"] == 0.0
    assert env.rewards["player_2"] == 0.0


def test_sparse_win_splits_tied_winners() -> None:
    env = Flip7AECEnv(player_count=3)
    env.reset(
        seed=0,
        options={
            "draw_pile": _pile(10, 1, 10),
            "starting_scores": (190, 190, 0),
        },
    )
    env.step(STAY_INDEX)
    env.step(STAY_INDEX)
    env.step(STAY_INDEX)

    assert env.engine.state.winning_player_ids == (0, 1)
    assert env.rewards["player_0"] == 0.5
    assert env.rewards["player_1"] == 0.5
    assert env.rewards["player_2"] == 0.0


def test_aec_random_legal_game_completes() -> None:
    env = Flip7AECEnv(player_count=3, observation=ObservationFamily.DECK_AWARE)
    env.reset(seed=99)
    rng = random.Random(1)

    for _agent in env.agent_iter(max_iter=10_000):
        _obs, _reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            env.step(None)
            continue
        legal = np.flatnonzero(info["action_mask"])
        env.step(int(rng.choice(legal)))

    assert env.engine.state.is_game_terminal is True
    assert not env.agents
