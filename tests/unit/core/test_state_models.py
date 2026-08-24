"""Tests for typed Flip 7 game-state models."""

import pytest

from flip7.core.cards import ActionCard, ActionCardName, NumberCard
from flip7.core.state import (
    GamePhase,
    GameState,
    InvalidPlayerCountError,
    InvalidPlayerError,
    PendingAction,
    PendingFlipThree,
    PlayerState,
    PlayerStatus,
    RoundPhase,
    create_initial_state,
)


def test_initial_state_validates_baseline_player_count_and_seating() -> None:
    state = create_initial_state(player_count=3, dealer_id=1)

    assert state.game_phase is GamePhase.IN_PROGRESS
    assert state.round_phase is RoundPhase.SETUP
    assert state.round_number == 1
    assert state.dealer_id == 1
    assert state.current_player_id is None
    assert state.is_round_terminal is False
    assert state.is_game_terminal is False
    assert state.flip7_player_id is None
    assert state.winning_player_ids == ()
    assert tuple(player.player_id for player in state.players) == (0, 1, 2)
    assert all(player.status is PlayerStatus.ACTIVE for player in state.players)
    assert all(player.cards == () for player in state.players)
    assert all(player.cumulative_score == 0 for player in state.players)


@pytest.mark.parametrize("player_count", [1, 2, 19])
def test_initial_state_rejects_out_of_scope_player_counts(player_count: int) -> None:
    with pytest.raises(
        InvalidPlayerCountError, match="baseline Flip 7 supports 3 to 18 players"
    ):
        create_initial_state(player_count=player_count)


def test_game_state_rejects_invalid_player_references() -> None:
    players = tuple(PlayerState(player_id=player_id) for player_id in range(3))

    with pytest.raises(InvalidPlayerError, match="dealer_id"):
        GameState(players=players, dealer_id=3)

    with pytest.raises(InvalidPlayerError, match="current_player_id"):
        GameState(players=players, dealer_id=0, current_player_id=-1)

    with pytest.raises(InvalidPlayerError, match="flip7_player_id"):
        GameState(players=players, dealer_id=0, flip7_player_id=5)


def test_player_state_is_immutable_and_tracks_round_cards_status_and_scores() -> None:
    player = PlayerState(
        player_id=2,
        status=PlayerStatus.FROZEN,
        cards=(NumberCard(7), ActionCard(ActionCardName.SECOND_CHANCE)),
        cumulative_score=42,
    )

    assert player.is_active is False
    assert player.has_cards is True

    with pytest.raises(AttributeError):
        player.status = PlayerStatus.ACTIVE  # type: ignore[reportAttributeAccessIssue]


def test_pending_action_and_flip_three_state_are_validated() -> None:
    players = tuple(PlayerState(player_id=player_id) for player_id in range(3))
    pending_action = PendingAction(
        action_card=ActionCardName.FREEZE,
        resolving_player_id=0,
        target_player_ids=(1,),
    )
    pending_flip_three = PendingFlipThree(
        resolving_player_id=0,
        target_player_id=1,
        cards_remaining=2,
        deferred_actions=(ActionCard(ActionCardName.FREEZE),),
    )

    state = GameState(
        players=players,
        dealer_id=0,
        pending_action=pending_action,
        pending_flip_three=pending_flip_three,
    )

    assert state.pending_action == pending_action
    assert state.pending_flip_three == pending_flip_three

    with pytest.raises(ValueError, match="cards_remaining must be between 0 and 3"):
        PendingFlipThree(
            resolving_player_id=0,
            target_player_id=1,
            cards_remaining=4,
        )
