"""Tests for Flip 7 environment reward functions."""

import pytest

from flip7.core.cards import NumberCard
from flip7.core.engine import EventKind, GameEvent, TransitionResult
from flip7.core.state import GamePhase, GameState, PlayerState, RoundPhase
from flip7.envs.rewards import RewardMode, rewards_from_result


def _result(
    *,
    winning_player_ids: tuple[int, ...] = (),
    score_changes: tuple[tuple[int, int], ...] = (),
    scores: tuple[int, ...] = (0, 0, 0),
    is_game_terminal: bool = False,
) -> TransitionResult:
    state = GameState(
        players=tuple(
            PlayerState(player_id=player_id, cumulative_score=scores[player_id])
            for player_id in range(3)
        ),
        dealer_id=0,
        current_player_id=None if is_game_terminal else 1,
        game_phase=GamePhase.COMPLETE if is_game_terminal else GamePhase.IN_PROGRESS,
        round_phase=RoundPhase.COMPLETE if is_game_terminal else RoundPhase.TURN,
        is_game_terminal=is_game_terminal,
        winning_player_ids=winning_player_ids,
    )
    events = (GameEvent(kind=EventKind.GAME_ENDED),) if is_game_terminal else ()
    return TransitionResult(
        state=state,
        events=events,
        drawn_cards=(NumberCard(4),) if not is_game_terminal else (),
        score_changes=score_changes,
        is_round_terminal=is_game_terminal,
        is_game_terminal=is_game_terminal,
    )


def test_sparse_win_is_zero_until_the_game_ends() -> None:
    result = _result(score_changes=((0, 12), (1, 8)))

    assert rewards_from_result(result, 3, RewardMode.SPARSE_WIN) == {
        0: 0.0,
        1: 0.0,
        2: 0.0,
    }


def test_sparse_win_splits_reward_among_tied_winners() -> None:
    result = _result(winning_player_ids=(0, 1), is_game_terminal=True)

    assert rewards_from_result(result, 3, RewardMode.SPARSE_WIN) == {
        0: 0.5,
        1: 0.5,
        2: 0.0,
    }


def test_round_score_adds_scaled_scores_and_terminal_win() -> None:
    result = _result(
        winning_player_ids=(0,),
        score_changes=((0, 20), (1, 10), (2, 0)),
        is_game_terminal=True,
    )

    rewards = rewards_from_result(result, 3, RewardMode.ROUND_SCORE)

    assert rewards[0] == pytest.approx(1.1)
    assert rewards[1] == pytest.approx(0.05)
    assert rewards[2] == 0.0


def test_potential_win_uses_public_score_differential_and_terminal_win() -> None:
    previous = _result(scores=(0, 0, 0)).state
    result = _result(
        scores=(20, 0, 0),
        winning_player_ids=(0,),
        is_game_terminal=True,
    )

    rewards = rewards_from_result(
        result,
        3,
        RewardMode.POTENTIAL_WIN,
        previous_state=previous,
        potential_discount=1.0,
    )

    assert rewards[0] == pytest.approx(1.1)
    assert rewards[1] == pytest.approx(-0.05)
    assert rewards[2] == pytest.approx(-0.05)


def test_potential_win_requires_the_previous_state() -> None:
    with pytest.raises(ValueError, match="previous game state"):
        rewards_from_result(_result(), 3, RewardMode.POTENTIAL_WIN)
