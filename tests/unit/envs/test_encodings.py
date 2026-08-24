"""Tests for discrete Flip 7 action encodings."""

import pytest

from flip7.core.actions import HitAction, InvalidActionError, StayAction, TargetAction
from flip7.core.cards import ActionCardName
from flip7.core.state import GameState, PendingAction, PlayerState, RoundPhase
from flip7.envs.encodings import (
    HIT_INDEX,
    STAY_INDEX,
    TARGET_OFFSET,
    action_mask,
    action_space_size,
    agent_name,
    decode_action,
    encode_action,
    player_id_from_agent,
)


def test_agent_ids_round_trip_for_seated_players() -> None:
    assert agent_name(3) == "player_3"
    assert player_id_from_agent("player_3") == 3


def test_player_id_from_agent_rejects_malformed_ids() -> None:
    with pytest.raises(ValueError, match="invalid agent id"):
        player_id_from_agent("agent_0")
    with pytest.raises(ValueError, match="invalid agent id"):
        player_id_from_agent("player_01")


def test_action_space_size_covers_hit_stay_and_every_seat() -> None:
    assert action_space_size(3) == 5
    assert action_space_size(18) == 20


def test_encode_and_decode_hit_stay_and_target_actions() -> None:
    players = tuple(PlayerState(player_id=player_id) for player_id in range(3))
    turn_state = GameState(
        players=players,
        dealer_id=0,
        current_player_id=1,
        round_phase=RoundPhase.TURN,
    )
    target_state = GameState(
        players=players,
        dealer_id=0,
        current_player_id=1,
        round_phase=RoundPhase.CARD_RESOLUTION,
        pending_action=PendingAction(
            action_card=ActionCardName.FREEZE,
            resolving_player_id=1,
        ),
    )

    assert encode_action(HitAction(player_id=1)) == HIT_INDEX
    assert decode_action(HIT_INDEX, turn_state, 1) == HitAction(player_id=1)
    assert encode_action(StayAction(player_id=1)) == STAY_INDEX
    assert decode_action(STAY_INDEX, turn_state, 1) == StayAction(player_id=1)

    target = TargetAction(
        player_id=1,
        action_card=ActionCardName.FREEZE,
        target_player_id=2,
    )
    assert encode_action(target) == TARGET_OFFSET + 2
    assert decode_action(TARGET_OFFSET + 2, target_state, 1) == target


def test_decode_action_rejects_out_of_range_and_target_without_pending() -> None:
    state = GameState(
        players=tuple(PlayerState(player_id=player_id) for player_id in range(3)),
        dealer_id=0,
        current_player_id=0,
        round_phase=RoundPhase.TURN,
    )

    with pytest.raises(InvalidActionError, match="outside the discrete action space"):
        decode_action(5, state, 0)
    with pytest.raises(InvalidActionError, match="hit or stay"):
        decode_action(TARGET_OFFSET + 1, state, 0)


def test_action_mask_sets_bits_for_legal_core_actions() -> None:
    legal = (
        HitAction(player_id=1),
        StayAction(player_id=1),
    )
    mask = action_mask(legal, player_count=3)

    assert list(mask) == [1, 1, 0, 0, 0]
    assert mask.dtype == "int8"


def test_action_mask_encodes_legal_targets() -> None:
    legal = (
        TargetAction(
            player_id=0,
            action_card=ActionCardName.FREEZE,
            target_player_id=0,
        ),
        TargetAction(
            player_id=0,
            action_card=ActionCardName.FREEZE,
            target_player_id=2,
        ),
    )
    mask = action_mask(legal, player_count=3)

    assert list(mask) == [0, 0, 1, 0, 1]
