"""Tests for Flip 7 legal action models."""

import pytest

from flip7.core.actions import (
    ActionKind,
    HitAction,
    InvalidActionError,
    InvalidTargetError,
    StayAction,
    TargetAction,
    legal_targets,
    legal_turn_actions,
    validate_target_action,
    validate_turn_action,
)
from flip7.core.cards import ActionCardName, NumberCard
from flip7.core.state import (
    GameState,
    InvalidPhaseError,
    PlayerState,
    PlayerStatus,
    RoundPhase,
)


def turn_state(*players: PlayerState, current_player_id: int = 0) -> GameState:
    return GameState(
        players=players,
        dealer_id=0,
        current_player_id=current_player_id,
        round_phase=RoundPhase.TURN,
    )


def test_legal_turn_actions_allow_hit_before_cards_and_stay_after_cards() -> None:
    state_without_cards = turn_state(
        PlayerState(player_id=0),
        PlayerState(player_id=1),
        PlayerState(player_id=2),
    )

    assert legal_turn_actions(state_without_cards, 0) == (HitAction(player_id=0),)

    state_with_cards = turn_state(
        PlayerState(player_id=0, cards=(NumberCard(7),)),
        PlayerState(player_id=1),
        PlayerState(player_id=2),
    )

    assert legal_turn_actions(state_with_cards, 0) == (
        HitAction(player_id=0),
        StayAction(player_id=0),
    )


def test_validate_turn_action_rejects_wrong_phase_player_and_status() -> None:
    players = (
        PlayerState(player_id=0, cards=(NumberCard(7),)),
        PlayerState(player_id=1, cards=(NumberCard(8),)),
        PlayerState(player_id=2, status=PlayerStatus.BUSTED, cards=(NumberCard(9),)),
    )

    with pytest.raises(InvalidPhaseError, match="turn actions require turn phase"):
        validate_turn_action(
            GameState(players=players, dealer_id=0, current_player_id=0),
            HitAction(player_id=0),
        )

    state = turn_state(*players, current_player_id=0)

    with pytest.raises(InvalidActionError, match="current player"):
        validate_turn_action(state, HitAction(player_id=1))

    inactive_state = turn_state(*players, current_player_id=2)
    with pytest.raises(InvalidActionError, match="active player"):
        validate_turn_action(inactive_state, HitAction(player_id=2))


def test_validate_turn_action_rejects_stay_without_cards_and_terminal_rounds() -> None:
    players = (
        PlayerState(player_id=0),
        PlayerState(player_id=1),
        PlayerState(player_id=2),
    )
    state = turn_state(*players)

    with pytest.raises(InvalidActionError, match="at least one card"):
        validate_turn_action(state, StayAction(player_id=0))

    terminal_state = GameState(
        players=players,
        dealer_id=0,
        current_player_id=0,
        round_phase=RoundPhase.TURN,
        is_round_terminal=True,
    )

    with pytest.raises(InvalidActionError, match="terminal round"):
        validate_turn_action(terminal_state, HitAction(player_id=0))


def test_legal_targets_include_only_active_players_and_allow_self_targeting() -> None:
    state = turn_state(
        PlayerState(player_id=0, cards=(NumberCard(7),)),
        PlayerState(player_id=1, status=PlayerStatus.STAYED, cards=(NumberCard(8),)),
        PlayerState(player_id=2, status=PlayerStatus.BUSTED, cards=(NumberCard(9),)),
    )

    assert legal_targets(state, 0, ActionCardName.FREEZE) == (0,)


def test_validate_target_action_rejects_invalid_actor_target_and_action_card() -> None:
    state = turn_state(
        PlayerState(player_id=0, cards=(NumberCard(7),)),
        PlayerState(player_id=1, status=PlayerStatus.FROZEN, cards=(NumberCard(8),)),
        PlayerState(player_id=2, cards=(NumberCard(9),)),
    )

    valid_action = TargetAction(
        player_id=0,
        action_card=ActionCardName.FLIP_THREE,
        target_player_id=2,
    )

    assert valid_action.kind is ActionKind.TARGET
    assert validate_target_action(state, valid_action) == valid_action

    with pytest.raises(InvalidActionError, match="active player"):
        validate_target_action(
            state,
            TargetAction(
                player_id=1,
                action_card=ActionCardName.FREEZE,
                target_player_id=0,
            ),
        )

    with pytest.raises(InvalidTargetError, match="active target"):
        validate_target_action(
            state,
            TargetAction(
                player_id=0,
                action_card=ActionCardName.FREEZE,
                target_player_id=1,
            ),
        )

    with pytest.raises(InvalidTargetError, match="cannot keep a second Second Chance"):
        validate_target_action(
            state,
            TargetAction(
                player_id=0,
                action_card=ActionCardName.SECOND_CHANCE,
                target_player_id=0,
            ),
        )
